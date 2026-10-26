"""Catch a detached position path or an accidental change of the deployed actor."""
import unittest

import torch

from learning import load_for_learning, load_projected, prefix_objective


class PositionLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.policy, cls.stats, cls.groups = load_for_learning()
        cls.original, _ = load_projected()

    def inputs(self):
        current = torch.zeros((2, 15), device='cuda')
        current[:, 7] = (.094926-float(self.stats['qpos_mean'][7]))/float(self.stats['qpos_std'][7])
        current[:, 9:] = torch.tensor([[-2., .3, -.5, -1., -.7, -.5],
                                      [-1., -.3, -.4, 1., -.5, -.4]], device='cuda')
        return current, torch.zeros((2, 1, 3, 180, 320), device='cuda')

    def test_zero_update_keeps_all_twenty_deployed_actions(self):
        current, image = self.inputs()
        with torch.no_grad():
            old = self.original(current, image)
        # Compare the gradient-enabled computation that will actually be trained.
        new = self.policy(current, image)
        # Gradient-enabled/fused attention can differ at float32 rounding only.
        torch.testing.assert_close(new, old, atol=1e-6, rtol=1e-6)
        scale = torch.as_tensor(self.stats['action_std'], device='cuda')
        difference = ((new-old)*scale).abs()
        self.assertLess(difference[..., :7].max().item(), 1e-7)
        self.assertLess(difference[..., 7].max().item(), 1e-8)

    def test_deployed_arm_and_gripper_gradients_reach_position_projection(self):
        current, image = self.inputs()
        self.policy.zero_grad(set_to_none=True)
        output = self.policy(current, image)
        (output[:, :2, :7].square().mean()+output[:, :2, 7].square().mean()).backward()
        grad = self.policy.base.model.input_proj_robot_state.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad[:, 9:].abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for p in self.policy.base.model.backbones.parameters()))
        self.assertTrue(all(p.grad is None for p in self.policy.base.model.encoder.parameters()))

    def test_padded_labels_cannot_change_supervision(self):
        prediction = torch.zeros((1, 20, 8), device='cuda', requires_grad=True)
        target = torch.ones_like(prediction)
        padding = torch.zeros((1, 20), device='cuda', dtype=torch.bool)
        padding[:, 10:] = True
        before, _ = prefix_objective(prediction, target, padding, self.stats)
        target[:, 10:] = 10000.
        after, _ = prefix_objective(prediction, target, padding, self.stats)
        torch.testing.assert_close(before, after, atol=0., rtol=0.)


if __name__ == '__main__':
    unittest.main()
