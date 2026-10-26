import unittest

import torch

from training.train import FullTaskPolicy


class GripperHeadTests(unittest.TestCase):
    def config(self, **updates):
        from tests.test_cvae_training import CvaeTrainingTests
        return CvaeTrainingTests().config(train_gripper_head_only=True,
            prior_gripper_loss_weight=.05, weight_decay=.2, **updates)

    def test_optimizer_updates_only_gripper_row_and_preserves_arm_predictions_and_buffers(self):
        torch.set_num_threads(2)
        policy = FullTaskPolicy(self.config())
        stats = policy.loss_stats
        with torch.no_grad():
            policy.model.action_head.weight[7].zero_()
            policy.model.action_head.bias[7] = (.098-stats['action_mean'][7])/stats['action_std'][7]
        q = torch.zeros(1, 9)
        q[:, 7] = (.095-stats['qpos_mean'][7])/stats['qpos_std'][7]
        image = torch.zeros(1, 1, 3, 90, 160)
        target = torch.zeros(1, 20, 8)
        target[..., 7] = (.0948-stats['action_mean'][7])/stats['action_std'][7]
        policy.eval()
        with torch.no_grad():
            before_prediction = policy(q, image).clone()
        before = {k: v.clone() for k, v in policy.state_dict().items()}
        policy.train()
        policy.optimizer.zero_grad(set_to_none=True)
        policy(q, image, target, torch.zeros(1, 20, dtype=torch.bool))['loss'].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), .1)
        policy.optimizer.step()
        policy.eval()
        with torch.no_grad():
            torch.testing.assert_close(policy(q, image)[..., :7], before_prediction[..., :7], rtol=0., atol=0.)
        after = policy.state_dict()
        for key in before:
            if key in ['model.action_head.weight', 'model.action_head.bias']:
                self.assertTrue(torch.equal(before[key][:7], after[key][:7]), key)
            else:
                self.assertTrue(torch.equal(before[key], after[key]), key)
        self.assertFalse(torch.equal(before['model.action_head.bias'][7], after['model.action_head.bias'][7]))

    def test_gripper_only_adaptation_cannot_reset_the_arm_head(self):
        with self.assertRaises(ValueError):
            FullTaskPolicy(self.config(reset_action_head=True))
