"""Check bounded CE gates, unchanged CE gradients and real ST-SATH backward."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from models.kd_weighting import ce_aligned_kd_weights, stsat_distillation_loss
from config import parse_args


class KDWeightingTests(unittest.TestCase):
    def test_aligned_conflicting_orthogonal_and_partial_alignment(self):
        p = torch.nn.Parameter(torch.tensor([.2, -.3], dtype=torch.float64))
        ce = p[0]
        objectives = {'aligned': 2 * p[0], 'conflict': -p[0],
                      'orthogonal': p[1], 'partial': 3 * p[0] + 4 * p[1]}
        weights = ce_aligned_kd_weights(ce, objectives, [p])
        for name, expected in dict(aligned=1., conflict=0., orthogonal=0., partial=.6).items():
            self.assertAlmostEqual(weights[name].item(), expected)
            self.assertFalse(weights[name].requires_grad)
        self.assertIsNone(p.grad)
        (ce + sum(weights[name] * value for name, value in objectives.items())).backward()
        torch.testing.assert_close(p.grad, p.new_tensor([4.8, 2.4]))

    def test_zero_gradients_frozen_unused_parameters_and_existing_buffers(self):
        p = torch.nn.Parameter(torch.tensor([.2, -.3]))
        unused = torch.nn.Parameter(torch.ones(2))
        frozen = torch.ones(2)
        p.grad = torch.tensor([7., 8.])
        objectives = {'zero': (p * 0).sum(), 'unused_by_ce': unused.sum(),
                      'constant': p.new_zeros(())}
        weights = ce_aligned_kd_weights(p.sum(), objectives, [p, unused, frozen])
        self.assertTrue(all(value.item() == 0 for value in weights.values()))
        torch.testing.assert_close(p.grad, torch.tensor([7., 8.]))
        self.assertIsNone(unused.grad)
        weights = ce_aligned_kd_weights((p * 0).sum(), {'existing': p.sum()}, [p])
        self.assertEqual(weights['existing'].item(), 0.)

    def test_all_parameters_are_used_including_classifier(self):
        shared = torch.nn.Parameter(torch.tensor(1.))
        classifier = torch.nn.Parameter(torch.tensor(1.))
        ce = shared + 10 * classifier
        weights = ce_aligned_kd_weights(ce, {'existing': shared - classifier}, [shared, classifier])
        # Shared parameters alone align; the full model gradient conflicts.
        self.assertEqual(weights['existing'].item(), 0.)
        (ce + weights['existing'] * (shared - classifier)).backward()
        self.assertEqual(shared.grad.item(), 1.)
        self.assertEqual(classifier.grad.item(), 10.)

    def test_nonlinear_weights_are_detached_and_preserve_ce_coefficient(self):
        p = torch.nn.Parameter(torch.tensor([.8, -.3, .5], dtype=torch.float64))
        ce = p.square().sum()
        objectives = {'existing': (p - .1).square().sum(), 'feature': (p + .4).square().sum()}
        reference = torch.autograd.grad(ce, p, retain_graph=True)[0]
        grads = {name: torch.autograd.grad(loss, p, retain_graph=True)[0]
                 for name, loss in objectives.items()}
        weights = ce_aligned_kd_weights(ce, objectives, [p])
        self.assertTrue(all(0 < weight.item() < 1 for weight in weights.values()))
        for name in objectives:
            self.assertGreaterEqual(torch.dot(reference, weights[name] * grads[name]).item(), 0.)
        expected = reference + sum(weights[name] * grads[name] for name in objectives)
        (ce + sum(weights[name] * loss for name, loss in objectives.items())).backward()
        torch.testing.assert_close(p.grad, expected, atol=1e-12, rtol=1e-12)

    def make_losses(self):
        logits = torch.nn.Parameter(torch.tensor([[.1, -.2, .5], [-.4, .3, .2]], dtype=torch.float64))
        teacher = torch.tensor([[.5, .3, .2], [.2, .1, .7]], dtype=torch.float64)
        ce = F.cross_entropy(logits, torch.tensor([0, 1]))
        kl = F.kl_div(logits.log_softmax(1), teacher, reduction='none')
        feature = (logits[:, :2] - .2).square().mean()
        return logits, ce, kl, feature

    def test_disabled_mode_exactly_matches_original_loss_and_gradients(self):
        p, ce, kl, feature = self.make_losses()
        missing, existing = torch.tensor([2]), torch.tensor([0, 1])
        legacy = .1 * (kl * p.new_tensor([.3, .3, 1.])).sum(1).mean() + .2 * feature
        expected = torch.autograd.grad(ce + legacy, p, retain_graph=True)[0]
        with patch('torch.autograd.grad', side_effect=AssertionError('Unexpected gradient probe')):
            kd, weights = stsat_distillation_loss(ce, kl, missing, existing, feature, [p],
                                                  .1, 1., .3, .2, mode='none')
        torch.testing.assert_close(kd, legacy)
        self.assertTrue(all(weight.item() == 1. for weight in weights.values()))
        (ce + kd).backward()
        torch.testing.assert_close(p.grad, expected)

    def test_missing_kd_stays_fixed_and_two_gates_are_independent(self):
        p, ce, kl, feature = self.make_losses()
        ce_grad = torch.autograd.grad(ce, p, retain_graph=True)[0]
        missing_loss = .1 * kl[:, 2].mean()
        baseline_grad = torch.autograd.grad(ce + missing_loss, p, retain_graph=True)[0]
        parts = {'existing': .03 * kl[:, :2].sum(1).mean(), 'feature': .2 * feature}
        grads = {name: torch.autograd.grad(loss, p, retain_graph=True)[0]
                 for name, loss in parts.items()}
        kd, weights = stsat_distillation_loss(ce, kl, torch.tensor([2]), torch.tensor([0, 1]),
                                              feature, [p], .1, 1., .3, .2)
        for name, grad in grads.items():
            self.assertGreaterEqual((ce_grad * weights[name] * grad).sum().item(), -1e-12)
            self.assertGreaterEqual(weights[name].item(), 0.)
            self.assertLessEqual(weights[name].item(), 1.)
        torch.testing.assert_close(kd, missing_loss + sum(weights[name] * parts[name] for name in parts))
        (ce + kd).backward()
        torch.testing.assert_close(p.grad, baseline_grad + sum(weights[name] * grads[name] for name in grads))

    def test_disabled_empty_or_absent_terms_skip_probes(self):
        for empty_labels, absent_features in ((False, False), (True, True)):
            p, ce, kl, feature = self.make_losses()
            existing = torch.tensor([] if empty_labels else [0, 1], dtype=torch.long)
            feature = None if absent_features else feature
            with patch('torch.autograd.grad', side_effect=AssertionError('Unexpected gradient probe')):
                kd, weights = stsat_distillation_loss(
                    ce, kl, torch.tensor([2]), existing, feature, [p], .1, 1.,
                    1. if empty_labels else 0., 1. if absent_features else 0.)
            torch.testing.assert_close(kd, .1 * kl[:, 2].mean())
            self.assertTrue(all(value.item() == 0 for value in weights.values()))
        p, ce, kl, feature = self.make_losses()
        kd, _ = stsat_distillation_loss(ce, kl, torch.tensor([], dtype=torch.long),
                                       torch.tensor([0, 1, 2]), feature, [p], .1, 1., .3, .2)
        self.assertTrue(torch.isfinite(kd))

    def test_invalid_coefficients_and_nonfinite_gradients_fail_explicitly(self):
        p, ce, kl, feature = self.make_losses()
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                stsat_distillation_loss(ce, kl, torch.tensor([2]), torch.tensor([0, 1]),
                                         feature, [p], .1, 1., value, .2)
        with self.assertRaises(ValueError):
            ce_aligned_kd_weights(ce, {}, [p], mode='invalid')
        with self.assertRaisesRegex(RuntimeError, 'Non-finite existing'):
            ce_aligned_kd_weights(ce, {'existing': p.sum() * float('inf')}, [p])
        with self.assertRaisesRegex(RuntimeError, 'Non-finite CE'):
            ce_aligned_kd_weights(p.sum() * float('nan'), {'existing': p.sum()}, [p])

    def test_parser_defaults_to_fixed_weights_and_accepts_cosine(self):
        self.assertEqual(parse_args([]).st_kd_grad_weighting, 'none')
        self.assertEqual(parse_args(['-st_kd_grad_weighting', 'cosine']).st_kd_grad_weighting, 'cosine')
        self.assertEqual(parse_args(['--st-kd-grad-weighting', 'cosine']).st_kd_grad_weighting, 'cosine')
        self.assertEqual(parse_args(['-st_kd_grad_weighting', 'none']).st_kd_grad_weighting, 'none')
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            parse_args(['-st_kd_grad_weighting', 'invalid'])

    def test_parser_rejects_negative_and_nonfinite_kd_coefficients(self):
        for flag in ('-lamda', '-st_no_exist_label_weight', '-st_exist_label_weight', '-st_lambda_feature'):
            for value in ('-1', 'nan', 'inf'):
                with self.subTest(flag=flag, value=value), patch('sys.stderr'), self.assertRaises(SystemExit):
                    parse_args([flag, value])

    def test_real_st_snn_repeated_backward_and_optimizer_step(self):
        from models.vgg_stlif import VGG_STLIF, compute_st_regularization
        from spikingjelly.activation_based import functional
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(17)
        args = SimpleNamespace(model='vgg9', dataset='cifar10', snn=True, T=2, num_classes=10,
                               st_grad_scale=1., st_use_nam=True, st_use_ltf=True, st_learnable_decay=True)
        net = VGG_STLIF(args).train()
        teacher = VGG_STLIF(args).eval()
        optimizer = torch.optim.SGD(net.parameters(), lr=.01)
        parameters = tuple(net.parameters())
        for _ in range(2):
            optimizer.zero_grad()
            x = torch.randn(2, 2, 3, 32, 32)
            out, features = net.forward_with_features(x)
            with torch.no_grad():
                teacher_out, teacher_features = teacher.forward_with_features(x)
            ce = F.cross_entropy(out, torch.tensor([0, 1]))
            kl = F.kl_div(out.log_softmax(1), teacher_out.softmax(1), reduction='none')
            feature = F.mse_loss(features, teacher_features)
            kd, weights = stsat_distillation_loss(ce, kl, torch.arange(2, 10), torch.arange(2),
                                                  feature, parameters, .1, 1., .3, .1)
            loss = ce + kd + compute_st_regularization(net, .01, .001, .01)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(0 <= w.item() <= 1 for w in weights.values()))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None))
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            before = net.fc[-1].weight.detach().clone()
            optimizer.step()
            self.assertFalse(torch.equal(before, net.fc[-1].weight))
            functional.reset_net(net)
            functional.reset_net(teacher)

    def test_training_entry_runs_both_modes_and_logs_effective_kd(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        import trainer
        from unittest.mock import Mock

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        dataset = torch.utils.data.TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
        images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        labels = np.array([0, 1])
        for mode in ('none', 'cosine'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix='st_sat_kd_') as output:
                args = parse_args(['-gpu', '-1', '-np', '1', '-frac', '1', '-b', '2', '-T', '2',
                                   '-global_epochs', '1', '-local_epochs', '1', '-result_dir', output,
                                   '-st_kd_grad_weighting', mode])
                logger = Mock()
                with patch.multiple(trainer, Logger=Mock(return_value=logger),
                                    load_data=Mock(return_value=(images, labels, images, labels)),
                                    partition_data=Mock(return_value={0: [0, 1]}),
                                    get_dataloader=Mock(return_value=(None, None, dataset, dataset)),
                                    compute_accuracy=Mock(return_value=.5)), \
                     patch.object(trainer, 'stsat_distillation_loss', wraps=stsat_distillation_loss) as kd_call:
                    trainer.run(args)
                if mode == 'none':
                    kd_call.assert_not_called()
                else:
                    self.assertEqual(kd_call.call_count, 1)
                    self.assertEqual(kd_call.call_args.kwargs['mode'], 'cosine')
                messages = [call.args[0] for call in logger.info.call_args_list]
                self.assertTrue(any(f'mode={mode}' in text and 'existing_gate=' in text
                                    and 'feature_gate=' in text and 'effective_distill=' in text
                                    for text in messages))
                self.assertEqual(len(list(Path(output).glob('*.pt'))), 1)
                self.assertEqual(len(list(Path(output).glob('acc_*.csv'))), 1)


if __name__ == '__main__':
    unittest.main()
