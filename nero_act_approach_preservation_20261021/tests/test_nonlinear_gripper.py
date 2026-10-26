import unittest

import numpy as np
import torch

from executor.model import BoundedActPolicy, physical_actions
from training.rebase_initializer import paired_config, rebase_initializer
from training.train import FullTaskPolicy
from tests import test_cvae_training


class NonlinearGripperTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def config(self, **updates):
        return test_cvae_training.CvaeTrainingTests().config(
            nonlinear_gripper_head=True, train_nonlinear_gripper_only=True,
            prior_gripper_loss_weight=.05, gripper_adapter_lr=.001, **updates)

    def inputs(self, config):
        stats = config['constraint_loss_stats']
        q = torch.zeros(1, 9)
        q[:, 7] = (.095-stats['qpos_mean'][7])/stats['qpos_std'][7]
        target = torch.zeros(1, 20, 8)
        target[..., 7] = (.093-stats['action_mean'][7])/stats['action_std'][7]
        return q, torch.zeros(1, 1, 3, 90, 160), target, torch.zeros(1, 20, dtype=torch.bool)

    def test_real_optimizer_changes_only_adapter_and_preserves_all_arm_outputs(self):
        config = self.config()
        policy = FullTaskPolicy(config)
        stats = config['constraint_loss_stats']
        with torch.no_grad():
            policy.model.action_head.weight[7].zero_()
            policy.model.action_head.bias[7] = (.098-stats['action_mean'][7])/stats['action_std'][7]
        q, image, target, padding = self.inputs(config)
        policy.eval()
        before_output = policy(q, image).detach()
        before = {key: value.clone() for key, value in policy.state_dict().items()}
        policy.train()
        policy.optimizer.zero_grad(set_to_none=True)
        output = policy(q, image, target, padding)
        output['loss'].backward()
        self.assertTrue(any(name.startswith('gripper_adapter.') and value.grad is not None
                            and value.grad.abs().max() > 0
                            for name, value in policy.named_parameters()))
        policy.optimizer.step()
        changed = [key for key, value in policy.state_dict().items()
                   if not torch.equal(value, before[key])]
        self.assertTrue(changed)
        self.assertTrue(all(key.startswith('gripper_adapter.') for key in changed), changed)
        policy.eval()
        torch.testing.assert_close(policy(q, image)[..., :7], before_output[..., :7], rtol=0., atol=0.)

    def test_adapter_training_and_validation_use_deployed_prior_without_future_labels(self):
        config = self.config()
        policy = FullTaskPolicy(config)
        q, image, target, padding = self.inputs(config)
        for train in [True, False]:
            policy.train(train)
            a = policy(q, image, target, padding)
            b = policy(q, image, target+10., padding)
            torch.testing.assert_close(a['a_hat'], b['a_hat'], rtol=0., atol=0.)
            self.assertEqual(float(a['kl']), 0.)
        runtime = BoundedActPolicy(config).eval()
        runtime.load_state_dict(policy.state_dict(), strict=True)
        torch.testing.assert_close(runtime(q, image), a['a_hat'], rtol=0., atol=0.)

    def test_initializer_upgrade_is_explicit_and_rebase_preserves_trained_adapter_output(self):
        config = self.config()
        old_config = dict(config, nonlinear_gripper_head=False, train_nonlinear_gripper_only=False)
        original = BoundedActPolicy(old_config).eval()
        stats = {key: np.array(value, np.float32)
                 for key, value in config['constraint_loss_stats'].items()}
        state = rebase_initializer(original.state_dict(), stats, stats, config)
        self.assertTrue(any(key.startswith('gripper_adapter.') for key in state))
        upgraded = BoundedActPolicy(config).eval()
        with self.assertRaises(RuntimeError):
            upgraded.load_state_dict(original.state_dict(), strict=True)
        upgraded.load_state_dict(state, strict=True)
        q, image, _, _ = self.inputs(config)
        torch.testing.assert_close(upgraded(q, image), original(q, image), rtol=0., atol=0.)
        with torch.no_grad():
            for name, parameter in upgraded.named_parameters():
                if name.startswith('gripper_adapter.'):
                    parameter.add_(.01)
        new = {key: value*1.1 if key.endswith('std') else value+.001 for key, value in stats.items()}
        rebased = BoundedActPolicy(paired_config(config, new)).eval()
        rebased.load_state_dict(rebase_initializer(upgraded.state_dict(), stats, new, config), strict=True)
        new_q = q.clone()
        new_q[:, :8] = (q[:, :8]*torch.tensor(stats['qpos_std'])+torch.tensor(stats['qpos_mean'])
                        -torch.tensor(new['qpos_mean']))/torch.tensor(new['qpos_std'])
        torch.testing.assert_close(physical_actions(upgraded(q, image), stats, bounded_gripper=True),
            physical_actions(rebased(new_q, image), new, bounded_gripper=True), rtol=1e-5, atol=1e-6)

    def test_incompatible_training_modes_and_invalid_adapter_rate_are_rejected(self):
        for update in [dict(train_gripper_head_only=True), dict(reset_action_head=True),
                       dict(nonlinear_gripper_head=False), dict(gripper_adapter_lr=0.),
                       dict(gripper_adapter_lr=float('nan')), dict(prior_gripper_loss_weight=0.)]:
            config = self.config()
            config.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                FullTaskPolicy(config)


if __name__ == '__main__':
    unittest.main()
