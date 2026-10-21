"""Verify the dynamic-B-only strategy, gradients and training outputs."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from models.kd_weighting import ce_aligned_kd_weights, stsat_distillation_loss


class DynamicBDistillationTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(17)

    def linear_losses(self):
        p = torch.nn.Parameter(torch.tensor([.8, .3], dtype=torch.float64))
        ce = p[0]
        kl = torch.stack((3 * p[0] + 4 * p[1], 4 * p[1])).unsqueeze(0)
        return p, ce, kl, p[1]

    def b_loss(self, ce, kl, feature, parameters, coefficients=(.1, 1., .3, .1)):
        return stsat_distillation_loss(
            ce, kl, torch.tensor([1]), torch.tensor([0]), feature, parameters,
            *coefficients)

    def test_dynamic_b_is_the_only_mode_and_default(self):
        self.assertEqual(parse_args([]).st_distill_mode, 'dynamic_b')
        self.assertEqual(parse_args(['-st_distill_mode', 'dynamic_b']).st_distill_mode, 'dynamic_b')
        self.assertEqual(parse_args(['--st-distill-mode', 'dynamic_b']).st_distill_mode, 'dynamic_b')
        for mode in ('fixed', 'invalid'):
            with self.subTest(mode=mode), patch('sys.stderr'), self.assertRaises(SystemExit):
                parse_args(['-st_distill_mode', mode])

    def test_feature_uses_composite_reference_without_ce_minimum_or_regularization(self):
        p, ce, kl, feature = self.linear_losses()
        kd, weights = self.b_loss(ce, kl, feature, [p])
        # CE=[1,0], existing=[3,4], missing=[0,4]; existing gate=.6.
        # The B reference gradient is [2.8,6.4], while CE/feature cosine is 0.
        feature_gate = 6.4 / math.sqrt(2.8 ** 2 + 6.4 ** 2)
        self.assertAlmostEqual(weights['existing'].item(), .6)
        self.assertAlmostEqual(weights['feature'].item(), feature_gate)
        self.assertTrue(all(not gate.requires_grad for gate in weights.values()))
        self.assertIsNone(p.grad)
        torch.testing.assert_close(kd, kl[:, 1].mean() + .6 * kl[:, 0].mean() + feature_gate * feature)
        (ce + kd).backward()
        torch.testing.assert_close(p.grad, p.new_tensor([2.8, 6.4 + feature_gate]))

    def test_nonlinear_backward_uses_detached_gates(self):
        p = torch.nn.Parameter(torch.tensor([[.1, -.2, .5], [-.4, .3, .2]], dtype=torch.float64))
        ce = F.cross_entropy(p, torch.tensor([0, 1]))
        teacher = p.new_tensor([[.5, .3, .2], [.2, .1, .7]])
        kl = F.kl_div(p.log_softmax(1), teacher, reduction='none')
        feature = (p[:, :2] - .2).square().mean()
        kd, gates = stsat_distillation_loss(
            ce, kl, torch.tensor([2]), torch.tensor([0, 1]), feature, [p], .1, 1., .3, .1)
        grad = lambda loss: torch.autograd.grad(loss, p, retain_graph=True)[0]
        expected = (grad(ce) + grad(kl[:, 2].mean())
                    + gates['existing'] * grad(kl[:, :2].sum(1).mean())
                    + gates['feature'] * grad(feature))
        self.assertTrue(all(not gate.requires_grad for gate in gates.values()))
        (ce + kd).backward()
        torch.testing.assert_close(p.grad, expected, atol=1e-12, rtol=1e-12)

    def test_positive_coefficients_are_only_enable_switches(self):
        p, ce, kl, feature = self.linear_losses()
        low, low_gates = self.b_loss(ce, kl, feature, [p])
        high, high_gates = self.b_loss(ce, kl, feature, [p], coefficients=(5., 7., 9., 11.))
        torch.testing.assert_close(low, high, rtol=0, atol=0)
        for name in low_gates:
            torch.testing.assert_close(low_gates[name], high_gates[name], rtol=0, atol=0)

    def test_disabled_or_empty_terms_skip_probes_and_missing_remains_unit_weight(self):
        p, ce, kl, feature = self.linear_losses()
        with patch('torch.autograd.grad', side_effect=AssertionError('Unexpected probe')):
            kd, gates = self.b_loss(ce, kl, feature, [p], coefficients=(.1, .2, 0., 0.))
            empty_kd, empty_gates = stsat_distillation_loss(
                ce, kl, torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), None,
                [p], .1, 1., .3, .1)
        torch.testing.assert_close(kd, kl[:, 1].mean(), rtol=0, atol=0)
        self.assertTrue(all(gate.item() == 0. for gate in gates.values()))
        self.assertEqual(empty_kd.item(), 0.)
        self.assertTrue(all(gate.item() == 0. for gate in empty_gates.values()))
        # Feature KD does not depend on lamda when both class KL terms are off.
        feature_only, gates = self.b_loss(ce, kl, p[0], [p], coefficients=(0., 1., .3, .1))
        torch.testing.assert_close(feature_only, p[0])
        self.assertEqual(gates['existing'].item(), 0.)
        self.assertEqual(gates['feature'].item(), 1.)

    def test_gates_cover_classifier_and_handle_conflict_zero_and_unused_gradients(self):
        shared = torch.nn.Parameter(torch.tensor(1.))
        classifier = torch.nn.Parameter(torch.tensor(1.))
        unused = torch.nn.Parameter(torch.tensor(1.))
        ce = shared + 10 * classifier
        classifier.grad = torch.tensor(7.)
        objectives = {'conflict': shared - classifier, 'zero': shared * 0, 'unused': unused}
        gates = ce_aligned_kd_weights(ce, objectives, [shared, classifier, unused])
        self.assertTrue(all(gate.item() == 0. for gate in gates.values()))
        self.assertEqual(classifier.grad.item(), 7.)
        self.assertIsNone(shared.grad)
        gates = ce_aligned_kd_weights(shared * 0, {'existing': shared}, [shared])
        self.assertEqual(gates['existing'].item(), 0.)

    def test_parser_rejects_negative_and_nonfinite_kd_enable_values(self):
        for flag in ('-lamda', '-st_no_exist_label_weight', '-st_exist_label_weight', '-st_lambda_feature'):
            for value in ('-1', 'nan', 'inf'):
                with self.subTest(flag=flag, value=value), patch('sys.stderr'), self.assertRaises(SystemExit):
                    parse_args([flag, value])

    def test_default_training_runs_dynamic_b_and_saves_results(self):
        import trainer

        dataset = torch.utils.data.TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
        images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        labels = np.array([0, 1])
        with tempfile.TemporaryDirectory(prefix='.test_dynamic_b_', dir=Path(__file__).resolve().parent) as output:
            args = parse_args(['-gpu', '-1', '-np', '1', '-frac', '1', '-b', '2', '-T', '2',
                               '-global_epochs', '1', '-local_epochs', '2', '-result_dir', output])
            logger = Mock()
            with patch.multiple(trainer, Logger=Mock(return_value=logger),
                                load_data=Mock(return_value=(images, labels, images, labels)),
                                partition_data=Mock(return_value={0: [0, 1]}),
                                get_dataloader=Mock(return_value=(None, None, dataset, dataset)),
                                compute_accuracy=Mock(return_value=.5)), \
                 patch('models.kd_weighting.stsat_distillation_loss', wraps=stsat_distillation_loss) as kd_call:
                trainer.run(args)
            self.assertEqual(kd_call.call_count, 2)
            messages = [call.args[0] for call in logger.info.call_args_list]
            self.assertTrue(any('mode=dynamic_b' in text and 'existing_gate=' in text
                                and 'feature_gate=' in text and 'effective_distill=' in text
                                for text in messages))
            checkpoint, = Path(output).glob('*.pt')
            self.assertIn('_dynamic_b', checkpoint.name)
            self.assertEqual(len(list(Path(output).glob('acc_*.csv'))), 1)
            metrics_path, = Path(output).glob('*_metrics.json')
            metrics = json.loads(metrics_path.read_text('utf-8'))
            self.assertEqual(metrics['distillation_mode'], 'dynamic_b')
            self.assertEqual(metrics['best_global_accuracy'], .5)
            self.assertEqual(metrics['num_clients_evaluated'], 1)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.assertTrue(all(torch.isfinite(value).all() for value in state.values()))

    def test_logs_use_dynamic_b_directory(self):
        from utils import Logger

        with tempfile.TemporaryDirectory(prefix='.test_dynamic_logs_', dir=Path(__file__).resolve().parent) as output:
            for simple in (False, True):
                args = parse_args(['-log_dir', output] + (['-simple_log'] if simple else []))
                log_path = Path(Logger(args).filename)
                self.assertIn('dynamic_b', log_path.parts)
                self.assertIn("st_distill_mode='dynamic_b'", log_path.read_text('utf-8'))


if __name__ == '__main__':
    unittest.main()
