import os
import time
import copy
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from utils import CLASS_NUM, INPUT_SIZE, Logger, load_data, get_dataloader, compute_accuracy, ensure_dir
from sampler import partition_data
from models.vgg_stlif import VGG_STLIF, get_all_lifact_params, get_lifact_temporal_info, compute_st_regularization
from models.kd_weighting import stsat_distillation_loss

def init_nets(n_parties, args):
    nets = {i: VGG_STLIF(args) for i in range(n_parties)}
    state = nets[0].state_dict()
    return (nets, [v.shape for v in state.values()], list(state))

def update_client_best_accuracy(client_best_accuracy, client_id, accuracy):
    accuracy = float(accuracy)
    if client_id not in client_best_accuracy or accuracy > client_best_accuracy[client_id]:
        client_best_accuracy[client_id] = accuracy
        return True
    return False

def summarize_client_best_accuracy(client_best_accuracy):
    valid = [value for value in client_best_accuracy.values() if value >= 0.0]
    return {'average_best_client_accuracy': float(np.mean(valid)) if valid else None, 'num_clients_evaluated': len(valid), 'client_best_accuracy': {str(client_id): None if value < 0.0 else float(value) for (client_id, value) in sorted(client_best_accuracy.items())}}

def run(args):
    args.num_classes = CLASS_NUM[args.dataset]
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.device.type != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    repeat_num = args.T
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger = Logger(args, args.desc)
    logger.info(str(args))
    logger.info('Loading data')
    (X_train, y_train, X_test, y_test) = load_data(args)
    logger.info('Partitioning data')
    net_dataidx_map = partition_data(args, y_train)
    if any((len(indices) == 0 for indices in net_dataidx_map.values())):
        raise ValueError('A client has no training samples; reduce clients or adjust partition settings')
    traindata_cls_counts = {}
    for (net_i, dataidx) in net_dataidx_map.items():
        (unq, unq_cnt) = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        traindata_cls_counts[net_i] = tmp
    logger.info(f'Finish partitioning, data statistics: {str(traindata_cls_counts)}')
    train_all_in_list = []
    test_all_in_list = []
    for party_id in range(args.n_parties):
        (_, _, train_ds_local, test_ds_local) = get_dataloader((X_train, X_test), (y_train, y_test), args.batch_size, 32, net_dataidx_map[party_id])
        train_all_in_list.append(train_ds_local)
        test_all_in_list.append(test_ds_local)
    train_ds_global = torch.utils.data.ConcatDataset(train_all_in_list)
    train_dl_global = torch.utils.data.DataLoader(dataset=train_ds_global, batch_size=args.batch_size, shuffle=True)
    test_ds_global = test_all_in_list[0]
    test_dl_global = torch.utils.data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)
    logger.info(f'length of train_dl_global: {len(train_ds_global)}')
    logger.info(f"Global test samples: {len(test_ds_global)}; policy={'single clean test set'}")
    logger.info('Inint models')
    global_acc_record = []
    best_global_param = None
    run_tag = f'alpha{args.alpha:g}_seed{args.init_seed}_np{args.n_parties}_frac{args.frac:g}'
    from models.st_lif import LIFAct, validate_st_snn_model
    from spikingjelly.activation_based import functional
    (nets, local_model_meta_data, layer_type) = init_nets(args.n_parties, args)
    (global_models, global_model_meta_data, global_layer_type) = init_nets(1, args)
    global_model = global_models[0]
    (input_channels, input_height, input_width) = INPUT_SIZE[args.dataset]
    st_check = validate_st_snn_model(nets[0], (args.T, 2, input_channels, input_height, input_width))
    logger.info(f"[ST-SAT Check] {st_check['num_lif_modules']} ST-LIF modules enabled; binary forward outputs: {st_check['binary_outputs']}")
    global_param = global_model.state_dict()
    for (_, net) in nets.items():
        net.load_state_dict(global_param)
    global_model.to(device)
    logger.info('=' * 80)
    logger.info('ST-SATH (Spatio-Temporal PLIF) Configuration with Enhanced Knowledge Distillation:')
    logger.info(f'[ST-SATH KD] mode={args.st_kd_grad_weighting}; existing-class and feature gates in [0, 1]; all trainable parameters')
    max_acc = -1.0
    best_epoch = 0
    client_best_acc = {i: -1.0 for i in range(args.n_parties)}
    for epoch in range(args.global_epochs):
        global_start_time = time.time()
        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(args.n_parties * args.frac)]
        global_param = global_model.state_dict()
        for idx in selected:
            nets[idx].load_state_dict(global_param)
        epoch_temporal_stats = []
        epoch_survival_rates = []
        global_model.eval()
        for (net_id, net) in nets.items():
            if net_id not in selected:
                continue
            net.to(device)
            net.train()
            train_ds_local = train_all_in_list[net_id]
            train_dl_local = torch.utils.data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            logger.info(f'Training network {net_id}. n_training: {len(train_ds_local)}')
            num_per_class = [traindata_cls_counts[net_id][c] if c in traindata_cls_counts[net_id].keys() else 0 for c in range(args.num_classes)]
            num_per_class = torch.tensor(num_per_class).float().to(device)
            prior = num_per_class / num_per_class.sum()
            no_exist_label = torch.where(prior == 0)[0]
            exist_label = torch.where(prior != 0)[0]
            no_exist_label = no_exist_label.clone().detach().int().to(device)
            exist_label = exist_label.clone().detach().int().to(device)
            (lifact_params, other_params) = get_all_lifact_params(net)
            kd_parameters = tuple(p for p in net.parameters() if p.requires_grad)
            if args.optimizer == 'adam':
                optimizer = optim.Adam([{'params': other_params, 'lr': args.lr}, {'params': lifact_params, 'lr': args.lr}], weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = optim.SGD([{'params': other_params, 'lr': args.lr}, {'params': lifact_params, 'lr': args.lr}], momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                raise NotImplementedError(f'Not support {args.optimizer}')
            criterion = nn.CrossEntropyLoss().to(device)
            neuron_spike_counts = {}
            neuron_total_samples = 0

            def make_activity_hook(module_id):

                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        with torch.no_grad():
                            spike_sum = output.sum(dim=(0, 1))
                            if module_id not in neuron_spike_counts:
                                neuron_spike_counts[module_id] = spike_sum.cpu().clone()
                            else:
                                neuron_spike_counts[module_id] += spike_sum.cpu()
                return hook
            hooks = []
            module_idx = 0
            for module in net.modules():
                if isinstance(module, LIFAct):
                    hooks.append(module.register_forward_hook(make_activity_hook(module_idx)))
                    module_idx += 1
            for l_epoch in range(args.local_epochs):
                net.train()
                local_start_time = time.time()
                epoch_loss_collector = []
                epoch_reg_loss_collector = []
                epoch_distill_loss_collector = []
                epoch_effective_kd_collector = []
                epoch_kd_weight_sum = torch.zeros(2, device=device)
                for (x, target) in train_dl_local:
                    (x, target) = (x.to(device), target.to(device))
                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()
                    x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
                    (out, local_features) = net.forward_with_features(x)
                    teacher_model = global_model
                    with torch.no_grad():
                        (teach_output, global_features) = teacher_model.forward_with_features(x)
                    teach_output = teach_output.detach()
                    global_features = global_features.detach() if global_features is not None else None
                    ce_loss = criterion(out, target)
                    reg_loss = compute_st_regularization(net, lambda_temporal=args.st_lambda_temporal, lambda_nam=args.st_lambda_nam, lambda_decay=args.st_lambda_decay)
                    kl = nn.KLDivLoss(reduction='none')
                    output_log_soft = nn.functional.log_softmax(out, dim=1)
                    teacher_soft = nn.functional.softmax(teach_output, dim=1)
                    kl_per_class = kl(output_log_soft, teacher_soft)
                    class_weights = torch.zeros(args.num_classes, device=device)
                    class_weights[exist_label] = args.st_exist_label_weight
                    class_weights[no_exist_label] = args.st_no_exist_label_weight
                    weighted_kl = kl_per_class * class_weights.unsqueeze(0)
                    l_ad = weighted_kl.sum(dim=1).mean()
                    if local_features is not None and global_features is not None:
                        l_feature = nn.functional.mse_loss(local_features, global_features)
                    else:
                        l_feature = torch.tensor(0.0, device=device)
                    if args.st_kd_grad_weighting == 'none':
                        # Preserve the original fixed-weight loss and addition order.
                        kd_loss = l_ad * args.lamda + l_feature * args.st_lambda_feature
                        kd_weights = {
                            'existing': ce_loss.new_tensor(float(args.lamda * args.st_exist_label_weight > 0 and exist_label.numel() > 0)),
                            'feature': ce_loss.new_tensor(float(args.st_lambda_feature > 0 and local_features is not None and global_features is not None)),
                        }
                        loss = ce_loss + reg_loss + l_ad * args.lamda + l_feature * args.st_lambda_feature
                    else:
                        kd_loss, kd_weights = stsat_distillation_loss(
                            ce_loss, kl_per_class, no_exist_label, exist_label, l_feature,
                            kd_parameters, args.lamda, args.st_no_exist_label_weight,
                            args.st_exist_label_weight, args.st_lambda_feature,
                            mode=args.st_kd_grad_weighting,
                        )
                        loss = ce_loss + reg_loss + kd_loss
                    loss.backward()
                    optimizer.step()
                    epoch_loss_collector.append(ce_loss.item())
                    epoch_reg_loss_collector.append(reg_loss.item())
                    epoch_distill_loss_collector.append(l_ad.item() + l_feature.item())
                    epoch_effective_kd_collector.append(kd_loss.item())
                    epoch_kd_weight_sum += torch.stack((kd_weights['existing'], kd_weights['feature']))
                    neuron_total_samples += x.shape[1]
                    functional.reset_net(net)
                    functional.reset_net(global_model)
                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
                epoch_reg = sum(epoch_reg_loss_collector) / len(epoch_reg_loss_collector)
                epoch_distill = sum(epoch_distill_loss_collector) / len(epoch_distill_loss_collector)
                mean_kd_weights = (epoch_kd_weight_sum / len(epoch_effective_kd_collector)).tolist()
                effective_kd = sum(epoch_effective_kd_collector) / len(epoch_effective_kd_collector)
                logger.info(f'[ST-SATH KD] client={net_id} local_epoch={l_epoch + 1} '
                            f'mode={args.st_kd_grad_weighting} '
                            f'existing_gate={mean_kd_weights[0]:.6f} feature_gate={mean_kd_weights[1]:.6f} '
                            f'effective_distill={effective_kd:.6f}')
                if (1 + l_epoch) % 5 == 0:
                    logger.info(f'Local Epoch [{1 + l_epoch}/{args.local_epochs}] - CE Loss: {epoch_loss:.4f}, Reg: {epoch_reg:.6f}, Distill: {epoch_distill:.6f}, Time: {time.time() - local_start_time:.2f}s')
            for hook in hooks:
                hook.remove()
            lifact_info = get_lifact_temporal_info(net)
            for info in lifact_info:
                epoch_temporal_stats.append({'client': net_id, 'layer': info['name'], 'temporal_weights': info['temporal_weights'], 'dynamic_thresholds': info['dynamic_thresholds']})
            total_neurons = 0
            alive_neurons = 0
            for (module_id, spike_counts) in neuron_spike_counts.items():
                avg_activity = spike_counts / (neuron_total_samples * args.T)
                mask = (avg_activity > args.st_survival_threshold).float()
                total_neurons += mask.numel()
                alive_neurons += int(mask.sum().item())
            survival_rate = alive_neurons / total_neurons if total_neurons > 0 else 0.0
            epoch_survival_rates.append(survival_rate)
            logger.info(f'Client {net_id} - Survival rate: {survival_rate:.4f} ({alive_neurons}/{total_neurons})')
            train_acc = compute_accuracy(net, train_dl_local, device, repeat_num, logger=logger, label=f'Client {net_id} / local train')
            test_acc = compute_accuracy(net, test_dl_global, device, repeat_num, logger=logger, label=f'Client {net_id} / global test')
            if update_client_best_accuracy(client_best_acc, net_id, test_acc):
                logger.info(f'Client {net_id} - NEW BEST! Test accuracy: {test_acc:.4f}')
            transfer_started = time.perf_counter()
            logger.info(f'[Transfer] Client {net_id}: moving model to CPU')
            net.to('cpu')
            logger.info(f'[Transfer] Client {net_id}: complete, time={time.perf_counter() - transfer_started:.2f}s')
            logger.info(f'Client {net_id} - Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}')
        if epoch_temporal_stats:
            all_weights = []
            for stat in epoch_temporal_stats:
                all_weights.append(stat['temporal_weights'])
            avg_weights = np.mean(all_weights, axis=0)
            avg_thresholds = args.st_v_th / avg_weights
            logger.info(f'[ST-SATH] Avg temporal weights across clients: {avg_weights}')
            logger.info(f'[ST-SATH] Corresponding thresholds: {avg_thresholds}')
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        aggregation_started = time.perf_counter()
        logger.info('[ST-SATH] Aggregation start (CPU weighted average + server load)')
        old_temporal_params = {}
        old_tau_params = {}
        for key in global_param:
            if 'temporal' in key:
                old_temporal_params[key] = global_param[key].clone()
            elif 'tau' in key:
                old_tau_params[key] = global_param[key].clone()
        aggregated_param = {}
        for idx in range(len(selected)):
            net_param = nets[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_param:
                    aggregated_param[key] = net_param[key].float() * fed_avg_freqs[idx]
            else:
                for key in net_param:
                    aggregated_param[key] = aggregated_param[key] + net_param[key].float() * fed_avg_freqs[idx]
        for key in global_param:
            if 'temporal' in key:
                old_val = old_temporal_params[key].cpu().float()
                new_val = aggregated_param[key].float()
                global_param[key] = (1 - args.st_temporal_momentum) * old_val + args.st_temporal_momentum * new_val
            elif 'tau' in key:
                old_val = old_tau_params[key].cpu().float()
                new_val = aggregated_param[key].float()
                global_param[key] = (1 - args.st_decay_momentum) * old_val + args.st_decay_momentum * new_val
            else:
                global_param[key] = aggregated_param[key]
        global_model.load_state_dict(global_param)
        logger.info(f'[ST-SATH] Decoupled aggregation complete (temporal_mom={args.st_temporal_momentum}, tau_mom={args.st_decay_momentum})')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        logger.info(f'[ST-SATH] Aggregation time: {time.perf_counter() - aggregation_started:.2f}s')
        eval_model = global_model
        train_acc = compute_accuracy(eval_model, train_dl_global, device, repeat_num, logger=logger, label='Server / global train')
        test_acc = compute_accuracy(eval_model, test_dl_global, device, repeat_num, logger=logger, label='Server / global test')
        global_acc_record.append(test_acc)
        if max_acc <= test_acc:
            max_acc = test_acc
            best_epoch = epoch + 1
            best_global_param = copy.deepcopy(eval_model.state_dict())
        avg_survival = np.mean(epoch_survival_rates) if epoch_survival_rates else 0
        logger.info(f'[ST-SATH] Epoch [{epoch + 1}/{args.global_epochs}] - Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}, Best: {max_acc:.4f}, Avg survival: {avg_survival:.4f}, Time: {time.time() - global_start_time:.2f}s')
    logger.info(f'Client best accuracies: {client_best_acc}')
    ensure_dir(args.result_dir)
    act = 'snn'
    best_state = best_global_param if best_global_param is not None else global_param
    checkpoint_path = os.path.join(args.result_dir, f'saved_{act}_{args.model}_{args.dataset}_T{args.T}_{args.strategy}_{args.partition}_{run_tag}_{args.global_epochs}GE_{args.local_epochs}LE.pt')
    torch.save(best_state, checkpoint_path)
    client_best_summary = summarize_client_best_accuracy(client_best_acc)
    best_acc_value = float(max_acc) if max_acc >= 0 else None
    best_epoch_value = int(best_epoch)
    metrics_path = checkpoint_path[:-3] + '_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as metrics_file:
        json.dump({'strategy': args.strategy, 'dataset': args.dataset, 'model': args.model, 'alpha': args.alpha, 'seed': args.init_seed, 'run_tag': run_tag, 'best_global_accuracy': best_acc_value, 'best_global_epoch': best_epoch_value, 'global_test_samples': len(test_ds_global), 'global_test_policy': 'single_clean', **client_best_summary}, metrics_file, indent=2, ensure_ascii=False)
    logger.info(f"[{args.strategy}] Average best client accuracy: {client_best_summary['average_best_client_accuracy']} ({client_best_summary['num_clients_evaluated']} clients)")
    logger.info(f'Best global checkpoint saved to: {checkpoint_path}')
    logger.info(f'Metrics saved to: {metrics_path}')
    global_acc_dynamics = pd.DataFrame({'epoch': list(range(args.global_epochs)), 'test_acc': global_acc_record})
    global_acc_dynamics.to_csv(os.path.join(args.result_dir, f'acc_{act}_{args.model}_{args.dataset}_T{args.T}_{args.strategy}_{args.partition}_{run_tag}_{args.global_epochs}GE_{args.local_epochs}LE_{args.n_parties}party_frac{args.frac}.csv'), index=False)
