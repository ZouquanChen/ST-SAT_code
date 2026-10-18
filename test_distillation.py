"""Exercise fixed-weight distillation through real SNN backward and training."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args


class FixedDistillationTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(17)

    def test_parser_rejects_negative_and_nonfinite_kd_coefficients(self):
        for flag in ('-lamda', '-st_no_exist_label_weight', '-st_exist_label_weight', '-st_lambda_feature'):
            for value in ('-1', 'nan', 'inf'):
                with self.subTest(flag=flag, value=value), patch('sys.stderr'), self.assertRaises(SystemExit):
                    parse_args([flag, value])

    def test_real_st_snn_repeated_backward_and_optimizer_step(self):
        from models.vgg_stlif import VGG_STLIF, compute_st_regularization
        from spikingjelly.activation_based import functional

        args = SimpleNamespace(model='vgg9', dataset='cifar10', snn=True, T=2, num_classes=10,
                               st_grad_scale=1., st_use_nam=True, st_use_ltf=True, st_learnable_decay=True)
        net = VGG_STLIF(args).train()
        teacher = VGG_STLIF(args).eval()
        optimizer = torch.optim.SGD(net.parameters(), lr=.01)
        class_weights = torch.tensor([.3, .3] + [1.] * 8)
        for _ in range(2):
            optimizer.zero_grad()
            x = torch.randn(2, 2, 3, 32, 32)
            out, features = net.forward_with_features(x)
            with torch.no_grad():
                teacher_out, teacher_features = teacher.forward_with_features(x)
            ce = F.cross_entropy(out, torch.tensor([0, 1]))
            kl = F.kl_div(out.log_softmax(1), teacher_out.softmax(1), reduction='none')
            l_ad = (kl * class_weights.unsqueeze(0)).sum(1).mean()
            l_feature = F.mse_loss(features, teacher_features)
            reg = compute_st_regularization(net, .01, .001, .01)
            loss = ce + reg + l_ad * .1 + l_feature * .1
            with patch('torch.autograd.grad', side_effect=AssertionError('Unexpected gradient probe')):
                loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            before = net.fc[-1].weight.detach().clone()
            optimizer.step()
            self.assertFalse(torch.equal(before, net.fc[-1].weight))
            functional.reset_net(net)
            functional.reset_net(teacher)

    def test_training_entry_saves_results_without_gradient_probes(self):
        import trainer

        dataset = torch.utils.data.TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
        images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        labels = np.array([0, 1])
        with tempfile.TemporaryDirectory(prefix='st_sat_fixed_kd_') as output:
            args = parse_args(['-gpu', '-1', '-np', '1', '-frac', '1', '-b', '2', '-T', '2',
                               '-global_epochs', '1', '-local_epochs', '2', '-result_dir', output])
            logger = Mock()
            with patch.multiple(trainer, Logger=Mock(return_value=logger),
                                load_data=Mock(return_value=(images, labels, images, labels)),
                                partition_data=Mock(return_value={0: [0, 1]}),
                                get_dataloader=Mock(return_value=(None, None, dataset, dataset)),
                                compute_accuracy=Mock(return_value=.5)), \
                 patch('torch.autograd.grad', side_effect=AssertionError('Unexpected gradient probe')):
                trainer.run(args)
            self.assertEqual(len(list(Path(output).glob('*.pt'))), 1)
            self.assertEqual(len(list(Path(output).glob('acc_*.csv'))), 1)
            metrics_file, = Path(output).glob('*_metrics.json')
            metrics = json.loads(metrics_file.read_text(encoding='utf-8'))
            self.assertEqual(metrics['best_global_accuracy'], .5)
            self.assertEqual(metrics['num_clients_evaluated'], 1)


if __name__ == '__main__':
    unittest.main()
