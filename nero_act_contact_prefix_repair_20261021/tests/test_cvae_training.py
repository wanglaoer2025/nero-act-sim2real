import json
from pathlib import Path
import unittest

import numpy as np
import torch

from executor.model import BoundedActPolicy, physical_actions
from training.train import FullTaskPolicy

ROOT = Path(__file__).resolve().parents[1]


class CvaeTrainingTests(unittest.TestCase):
    def config(self, **updates):
        config = json.loads((ROOT/'model/run15/training_config.json').read_text())
        config.update(device='cpu', pretrained_backbone=False, dropout=0.,
                      training_observation_noise_normalized=[0.]*8,
                      pose_loss_weight=0., prefix_margin_loss_weight=0.,
                      cvae_training=True, kl_weight=.001)
        config.update(updates)
        return config

    def inputs(self):
        return (torch.zeros(1, 9), torch.zeros(1, 1, 3, 90, 160),
                torch.linspace(-.05, .05, 160).reshape(1, 20, 8),
                torch.zeros(1, 20, dtype=torch.bool))

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_supervised_training_reaches_posterior_and_backpropagates_kl(self):
        policy = FullTaskPolicy(self.config()).train()
        output = policy(*self.inputs())
        self.assertGreater(float(output['kl']), 0.)
        torch.testing.assert_close(output['loss'], output['l1']+.001*output['kl'])
        kl_gradients = torch.autograd.grad(output['kl'],
            [policy.model.encoder_action_proj.weight, policy.model.latent_proj.weight], retain_graph=True)
        for grad in kl_gradients:
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().max()), 0.)
        output['loss'].backward()
        for layer in [policy.model.encoder_action_proj, policy.model.latent_proj]:
            grad = layer.weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().max()), 0.)
        action = physical_actions(output['a_hat'], policy.loss_stats, bounded_gripper=True)
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue(((action[..., 7] >= 0.) & (action[..., 7] <= .095)).all())

    def test_validation_predictions_ignore_future_labels_and_match_runtime(self):
        policy = FullTaskPolicy(self.config()).eval()
        q, image, targets, padding = self.inputs()
        runtime = BoundedActPolicy(self.config()).eval()
        runtime.load_state_dict(policy.state_dict(), strict=True)
        with torch.no_grad():
            first = policy(q, image, targets, padding)
            second = policy(q, image, targets+10., padding)
            torch.testing.assert_close(first['a_hat'], second['a_hat'], rtol=0., atol=0.)
            torch.testing.assert_close(first['a_hat'], policy(q, image), rtol=0., atol=0.)
            torch.testing.assert_close(first['a_hat'], runtime(q, image), rtol=0., atol=0.)
            self.assertEqual(float(first['kl']), 0.)
            self.assertNotEqual(float(first['loss']), float(second['loss']))

    def test_padded_future_labels_do_not_influence_training_prediction(self):
        policy = FullTaskPolicy(self.config()).train()
        q, image, targets, padding = self.inputs()
        padding[:, 12:] = True
        changed = targets.clone()
        changed[:, 12:] = 100.
        torch.manual_seed(7)
        first = policy(q, image, targets, padding)
        torch.manual_seed(7)
        second = policy(q, image, changed, padding)
        torch.testing.assert_close(first['a_hat'], second['a_hat'], rtol=0., atol=0.)
        torch.testing.assert_close(first['loss'], second['loss'], rtol=0., atol=0.)

    def test_legacy_training_keeps_zero_kl_and_identical_prior_predictions(self):
        legacy = FullTaskPolicy(self.config(cvae_training=False)).train()
        cvae = FullTaskPolicy(self.config()).eval()
        cvae.load_state_dict(legacy.state_dict(), strict=True)
        q, image, targets, padding = self.inputs()
        with torch.no_grad():
            output = legacy(q, image, targets, padding)
            self.assertEqual(float(output['kl']), 0.)
            torch.testing.assert_close(output['loss'], output['l1'], rtol=0., atol=0.)
            torch.testing.assert_close(output['a_hat'], cvae(q, image), rtol=0., atol=0.)

    def test_cvae_training_rejects_missing_or_nonpositive_kl_regularization(self):
        for weight in [0., -.001, float('nan'), float('inf')]:
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                FullTaskPolicy(self.config(kl_weight=weight))

    def test_rebased_cvae_preserves_physical_posterior_conditioning(self):
        from training.rebase_initializer import paired_config, rebase_initializer
        config = self.config()
        old = {k: np.array(v, np.float32) for k, v in config['constraint_loss_stats'].items()}
        new = {k: v*1.5 if k.endswith('std') else v+.01 for k, v in old.items()}
        original = BoundedActPolicy(config).eval()
        rebased = BoundedActPolicy(paired_config(config, new)).eval()
        rebased.load_state_dict(rebase_initializer(original.state_dict(), old, new, config), strict=True)
        q, image, targets, padding = self.inputs()
        new_q = q.clone()
        old_t = {k: torch.as_tensor(v) for k, v in old.items()}
        new_t = {k: torch.as_tensor(v) for k, v in new.items()}
        new_q[:, :8] = (q[:, :8]*old_t['qpos_std']+old_t['qpos_mean']-new_t['qpos_mean'])/new_t['qpos_std']
        new_targets = (targets*old_t['action_std']+old_t['action_mean']-new_t['action_mean'])/new_t['action_std']
        with torch.no_grad():
            torch.manual_seed(13)
            before = original(q, image, targets, padding)
            torch.manual_seed(13)
            after = rebased(new_q, image, new_targets, padding)
        torch.testing.assert_close(before['kl'], after['kl'], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(physical_actions(before['a_hat'], old, bounded_gripper=True),
                                   physical_actions(after['a_hat'], new, bounded_gripper=True),
                                   rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
