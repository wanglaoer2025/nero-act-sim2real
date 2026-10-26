import importlib.util
import unittest

import torch


class ModelOutputTests(unittest.TestCase):
    def test_v2_head_can_represent_valid_recorded_training_prefixes(self):
        from executor.model import feedback_readout
        stats = dict(qpos_mean=torch.zeros(8), qpos_std=torch.ones(8),
                     action_mean=torch.zeros(8), action_std=torch.ones(8))
        state = torch.tensor([[.2]*7+[.0736]])
        # Training maxima are 28.8654mrad and 8.8863mm; include augmentation margin.
        required = torch.tensor([.0291]*7+[-.0090])
        span = torch.tensor([.0295]*7+[.0095])
        latent = state[:, None]+span*torch.atanh(required/span)
        prediction = latent.expand(1, 20, 8).clone().requires_grad_(True)
        result = feedback_readout(prediction, state, stats,
            torch.tensor([-2.]*7+[0.]), torch.tensor([2.]*7+[.095]), version='feedback_tanh_v2')
        torch.testing.assert_close(result[:, :10], (state[:, None]+required).expand(1, 10, 8))
        result.sum().backward()
        self.assertTrue(torch.all(prediction.grad > 0))
        extreme = feedback_readout(prediction.detach()*100, state, stats,
            torch.tensor([-2.]*7+[0.]), torch.tensor([2.]*7+[.095]), version='feedback_tanh_v2')
        self.assertLess(float(abs(extreme[:, :10, :7]-state[:, None, :7]).max()), .03)
        self.assertLess(float(abs(extreme[:, :10, 7]-state[:, None, 7]).max()), .01)

    def test_feedback_head_rejects_unpaired_external_statistics(self):
        from executor import model
        self.assertTrue(callable(getattr(model, 'verify_paired_readout_statistics', None)))
        stats = dict(qpos_mean=[0.]*8, action_mean=[0.]*8, qpos_std=[1.]*8, action_std=[1.]*8)
        config = dict(bounded_feedback_residual_output=True, constraint_loss_stats=stats)
        model.verify_paired_readout_statistics(config, stats)
        with self.assertRaises(ValueError):
            model.verify_paired_readout_statistics(config, dict(stats, action_mean=[.1]+[0.]*7))

    def test_smooth_feedback_readout_is_absolute_bounded_and_differentiable(self):
        from executor import model
        self.assertTrue(callable(getattr(model, 'feedback_readout', None)),
                        'shared trained residual readout missing')
        stats = {key: torch.tensor(value) for key, value in dict(
            qpos_mean=[.1]*7+[.04], qpos_std=[.5]*7+[.02],
            action_mean=[-.2]*7+[.05], action_std=[.3]*7+[.03]).items()}
        state = torch.tensor([[.2, .3, -.1, 1., .4, .1, -.4, .09]])
        qpos = (state-stats['qpos_mean'])/stats['qpos_std']
        prediction = ((state[:, None].repeat(1, 20, 1)-stats['action_mean'])/stats['action_std']).detach()
        prediction.requires_grad_(True)
        lower = torch.tensor([-2.]*7+[0.]);upper = torch.tensor([2.]*7+[.095])
        result = model.feedback_readout(prediction, qpos, stats, lower, upper)
        physical = result*stats['action_std']+stats['action_mean']
        torch.testing.assert_close(physical, state[:, None].expand_as(physical))
        result.sum().backward()
        self.assertTrue(torch.all(prediction.grad > .99))
        for sign in [-1., 1.]:
            result = model.feedback_readout(prediction.detach()+sign*100, qpos, stats, lower, upper)
            physical = result*stats['action_std']+stats['action_mean']
            delta = abs(physical-state[:, None])
            self.assertTrue(torch.all(delta[:, :10, :7] <= .025001))
            self.assertTrue(torch.all(delta[:, :10, 7] <= .008001))
            self.assertTrue(torch.all(delta[:, 10:, :7] <= .050001))
            self.assertTrue(torch.all(delta[:, 10:, 7] <= .016001))
            self.assertTrue(torch.all(physical <= upper+1e-6))
            self.assertTrue(torch.all(physical >= lower-1e-6))

    def test_shared_model_applies_the_same_head_during_training_and_inference(self):
        import json
        from pathlib import Path
        from unittest.mock import patch
        from executor.model import BoundedActPolicy
        from training.train import FullTaskPolicy
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root/'training/run6_snapshot_recovery/config.json').read_text())
        config.update(device='cpu', pretrained_backbone=False, bounded_feedback_residual_output=True,
                      feedback_readout_version='feedback_tanh_v1')
        state = torch.tensor(config['real_initial_state'])[None]
        stats = {key: torch.tensor(value) for key, value in config['constraint_loss_stats'].items()}
        qpos = (state-stats['qpos_mean'])/stats['qpos_std']
        with patch('executor.model.NeroActPolicy.forward', return_value=torch.full((1, 20, 8), 100.)):
            for version, arm, grip in [('feedback_tanh_v1', .025, .008), ('feedback_tanh_v2', .0295, .0095)]:
                for cls in [BoundedActPolicy, FullTaskPolicy]:
                    policy = cls(dict(config, feedback_readout_version=version))
                    for training in [False, True]:
                        policy.train(training)
                        prediction = policy(qpos, torch.zeros(1, 1, 3, 90, 160))
                        actual = prediction*stats['action_std']+stats['action_mean']
                        self.assertLessEqual(float(abs(actual[:, :10, :7]-state[:, None, :7]).max()), arm+1e-6)
                        self.assertLessEqual(float(abs(actual[:, :10, 7]-state[:, None, 7]).max()), grip+1e-6)

    def test_bounded_gripper_is_part_of_the_trained_model_output(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.model'), 'bounded ACT head is missing')
        from executor.model import bound_gripper, physical_actions
        values = torch.zeros(1, 3, 8)
        values[0, :, 0] = .25
        values[0, :, 7] = torch.tensor([2., 3., -3.])
        bounded = bound_gripper(values, .05, .02)
        stats = dict(action_mean=torch.tensor([0.]*7+[.05]),
                     action_std=torch.tensor([1.]*7+[.02]))
        actions = physical_actions(bounded, stats, bounded_gripper=True)
        self.assertAlmostEqual(float(actions[0, 0, 7]), .09, places=6)
        self.assertLessEqual(float(actions[0, 1, 7]), .095)
        self.assertGreaterEqual(float(actions[0, 2, 7]), 0.)
        self.assertAlmostEqual(float(actions[0, 1, 0]), .25)


if __name__ == '__main__':
    unittest.main()
