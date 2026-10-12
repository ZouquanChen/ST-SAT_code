import argparse
import math

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='ST-SAT federated training with spiking VGG only')
    parser.add_argument('-data_dir', default='./data', help='Dataset directory')
    parser.add_argument('-log_dir', type=str, required=False, default='./logs/', help='Log directory path')
    parser.add_argument('-simple_log', action='store_true', help='Write record.log directly in log_dir; preserve previous runs with numbered filenames')
    parser.add_argument('-result_dir', type=str, required=False, default='./saved/', help='Model directory path')
    parser.add_argument('-model', default='vgg9', choices=['vgg5', 'vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19'])
    parser.add_argument('-dataset', default='cifar10', choices=['cifar10'])
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('-T', type=int, default=4, help='time step for SNN neuron (default: 4), typically 4-16')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate (default: 0.001, typical range: 1e-2 to 1e-5)')
    parser.add_argument('-optimizer', type=str, default='adam', help='the optimizer (adam or sgd)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-05, help='weight decay for optimizer (L2 regularization, typical: 1e-5 to 1e-3)')
    parser.add_argument('-momentum', type=float, default=0.9, help='Parameter controlling the momentum SGD (typical: 0.9-0.99)')
    parser.add_argument('-gpu', type=int, default=0, help='GPU ID, -1 for CPU')
    parser.add_argument('-seed', '--init_seed', type=int, default=2025, help='Random seed')
    parser.add_argument('-desc', type=str, default='', help='description for log files')
    parser.add_argument('-partition', type=str, default='noniid-c-dir', help='the data partitioning strategy')
    parser.add_argument('-alpha', type=float, default=0.1, help='The parameter for the dirichlet distribution for data partitioning (smaller=more non-iid)')
    parser.add_argument('-np', '--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('-frac', type=float, default=0.2, help='Sample ratio [0., 1.] of parties for each local training round')
    parser.add_argument('-global_epochs', type=int, default=50, help='rounds of updating global model')
    parser.add_argument('-local_epochs', type=int, default=10, help='number of local training rounds')
    parser.add_argument('-lamda', type=float, default=0.1, help='Adaptive class distillation weight')
    parser.add_argument('-st_decay', type=float, default=0.25, help='ST-SAT: Initial leaky decay factor for membrane potential (0-1)')
    parser.add_argument('-st_learnable_decay', action='store_true', default=True, help='ST-SAT: Enable learnable decay factor per channel (PLIF)')
    parser.add_argument('-no_st_learnable_decay', action='store_false', dest='st_learnable_decay', help='ST-SAT: Disable learnable decay, use fixed decay value')
    parser.add_argument('-st_lambda_decay', type=float, default=0.01, help='ST-SAT: Regularization coefficient for PLIF decay factor')
    parser.add_argument('-st_v_th', type=float, default=1.0, help='ST-SAT: Threshold scale; firing occurs above 0.5 * st_v_th / exp(temporal) after spatial modulation')
    parser.add_argument('-st_temp', type=float, default=3.0, help='ST-SAT: Surrogate gradient temperature (fixed, higher=sharper)')
    parser.add_argument('-st_grad_scale', type=float, default=1.0, help='ST-SAT: Gradient scaling factor for backpropagation')
    parser.add_argument('-st_lambda_temporal', type=float, default=0.01, help='ST-SAT: Regularization coefficient for temporal factor')
    parser.add_argument('-st_lambda_nam', type=float, default=0.001, help='ST-SAT: Regularization coefficient for NAM weights')
    parser.add_argument('-st_survival_threshold', type=float, default=0.001, help='ST-SAT: Activity cutoff for survival statistics only; no pruning or masked aggregation')
    parser.add_argument('-st_lambda_feature', type=float, default=0.1, help='ST-SAT: Soft-feature distillation coefficient for penultimate layer')
    parser.add_argument('-st_exist_label_weight', type=float, default=0.3, help='ST-SAT: Weight for existing labels in adaptive distillation (lower for regularization)')
    parser.add_argument('-st_no_exist_label_weight', type=float, default=1.0, help='ST-SAT: Weight for non-existing labels in adaptive distillation')
    parser.add_argument('-st_kd_grad_weighting', '--st-kd-grad-weighting', choices=('none', 'cosine'), default='none',
                        help='ST-SAT: none keeps fixed KD coefficients; cosine applies detached CE-gradient cosine gates in [0, 1] to existing-class and soft-feature KD')
    parser.add_argument('-st_temporal_momentum', type=float, default=0.1, help='ST-SAT: Server temporal interpolation weight (0=keep old server value, 1=client weighted average); no persistent local parameters')
    parser.add_argument('-st_decay_momentum', type=float, default=0.1, help='ST-SAT: Server tau interpolation weight (0=keep old server value, 1=client weighted average); interpolation is in tau/logit space')
    parser.add_argument('-st_use_ltf', action='store_true', default=True, help='ST-SAT: Enable Learnable Temporal Factor (LTF)')
    parser.add_argument('-no_st_use_ltf', action='store_false', dest='st_use_ltf', help='ST-SAT: Disable Learnable Temporal Factor')
    parser.add_argument('-st_use_nam', action='store_true', default=True, help='ST-SAT: Enable Normalization-based Attention Module (NAM)')
    parser.add_argument('-no_st_use_nam', action='store_false', dest='st_use_nam', help='ST-SAT: Disable NAM, use standard BatchNorm')
    parser.add_argument('-strategy', default='st_sath', choices=['st_sath'], help='Compatibility name for ST-SAT; this is the only method')
    args = parser.parse_args(argv)
    if min(args.n_parties, args.batch_size, args.T, args.global_epochs, args.local_epochs) < 1:
        parser.error('Client count, batch size, time steps and epochs must be positive')
    if not 0 < args.frac <= 1 or int(args.n_parties * args.frac) < 1:
        parser.error('frac must select at least one client and must be <= 1')
    if args.alpha <= 0 or args.lr <= 0:
        parser.error('alpha and learning rate must be positive')
    if args.optimizer not in ('adam', 'sgd'):
        parser.error('optimizer must be adam or sgd')
    partitions = ['iid', 'noniid-c-dir', 'iid-diff-quantity'] + [f'noniid-cnum-{i}' for i in range(1, 10)]
    if args.partition not in partitions:
        parser.error('Unsupported partition: ' + args.partition)
    if not 0 < args.st_decay < 1:
        parser.error('st_decay must be between 0 and 1')
    if not all((0 <= m <= 1 for m in (args.st_temporal_momentum, args.st_decay_momentum))):
        parser.error('Aggregation momenta must be in [0, 1]')
    if any(not math.isfinite(value) or value < 0 for value in
           (args.lamda, args.st_no_exist_label_weight, args.st_exist_label_weight, args.st_lambda_feature)):
        parser.error('KD coefficients must be finite and nonnegative')
    args.snn = True
    return args
