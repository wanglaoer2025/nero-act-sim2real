import unittest

import torch

from executor.model import BoundedActPolicy
from training.train import FullTaskPolicy
from tests import test_cvae_training


class FixedSceneHeadsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def config(self, **updates):
        return test_cvae_training.CvaeTrainingTests().config(nonlinear_gripper_head=True,
            train_nonlinear_gripper_only=False, train_arm_output_and_gripper_only=True,
            train_gripper_head_only=False, freeze_backbone=True,
            prior_gripper_loss_weight=.05, arm_output_lr=.0001,
            gripper_adapter_lr=.001, **updates)

    def test_optimizer_changes_seven_arm_rows_and_adapter_but_no_other_tensor(self):
        policy = FullTaskPolicy(self.config()).train()
        before = {k: v.clone() for k, v in policy.state_dict().items()}
        output = policy(*test_cvae_training.CvaeTrainingTests().inputs())
        policy.optimizer.zero_grad(set_to_none=True)
        output['loss'].backward()
        self.assertGreater(float(policy.model.action_head.weight.grad[:7].abs().max()), 0.)
        policy.optimizer.step()
        changed = [k for k, v in policy.state_dict().items() if not torch.equal(v, before[k])]
        self.assertTrue(any(k.startswith('gripper_adapter.') for k in changed))
        self.assertIn('model.action_head.weight', changed)
        self.assertIn('model.action_head.bias', changed)
        self.assertTrue(all(k.startswith('gripper_adapter.') or k in
            ('model.action_head.weight', 'model.action_head.bias') for k in changed), changed)
        for key in ['model.action_head.weight', 'model.action_head.bias']:
            self.assertTrue(torch.equal(policy.state_dict()[key][7], before[key][7]))

    def test_train_and_eval_predictions_ignore_labels_and_use_runtime_zero_latent(self):
        config = self.config()
        policy = FullTaskPolicy(config)
        q, image, target, padding = test_cvae_training.CvaeTrainingTests().inputs()
        for train in [True, False]:
            policy.train(train)
            a = policy(q, image, target, padding)
            b = policy(q, image, target+10., padding)
            torch.testing.assert_close(a['a_hat'], b['a_hat'], atol=0., rtol=0.)
            self.assertEqual(float(a['kl']), 0.)
        runtime = BoundedActPolicy(config).eval()
        runtime.load_state_dict(policy.state_dict(), strict=True)
        torch.testing.assert_close(runtime(q, image), a['a_hat'], atol=0., rtol=0.)

    def test_conflicting_modes_and_nonpositive_learning_rates_are_rejected(self):
        for change in [dict(train_nonlinear_gripper_only=True), dict(train_gripper_head_only=True),
                       dict(reset_action_head=True), dict(nonlinear_gripper_head=False),
                       dict(freeze_backbone=False), dict(arm_output_lr=0.),
                       dict(arm_output_lr=float('nan')), dict(gripper_adapter_lr=0.),
                       dict(prior_gripper_loss_weight=0.)]:
            config = self.config()
            config.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                FullTaskPolicy(config)

    def test_cached_learning_probe_matches_complete_runtime_including_joint_anchor(self):
        from evidence.probe_fixed_heads import cached_readout
        policy=FullTaskPolicy(self.config()).eval()
        q,image,_,_=test_cvae_training.CvaeTrainingTests().inputs()
        q[:,:7]=.25
        captured=[]
        handle=policy.model.action_head.register_forward_pre_hook(
            lambda module,args:captured.append(args[0].detach().clone()))
        with torch.no_grad():expected=policy(q,image)
        # FullTaskPolicy returns the bounded tensor in inference mode.
        handle.remove()
        with torch.no_grad():_,actual=cached_readout(policy,captured[0],q)
        torch.testing.assert_close(actual,expected,rtol=0.,atol=0.)


if __name__ == '__main__':
    unittest.main()
