import copy
import unittest

import torch

from executor.model import physical_actions
from training.train import FullTaskPolicy


class GripperGradientTests(unittest.TestCase):
    def config(self, **updates):
        from tests.test_cvae_training import CvaeTrainingTests
        return CvaeTrainingTests().config(prior_gripper_loss_weight=.05, **updates)

    def test_verified_close_target_has_gradient_when_runtime_output_is_saturated_open(self):
        torch.set_num_threads(2)
        policy = FullTaskPolicy(self.config()).train()
        stats = policy.loss_stats
        with torch.no_grad():
            policy.model.action_head.weight[7].zero_()
            policy.model.action_head.bias[7] = (.098-stats['action_mean'][7])/stats['action_std'][7]
        q = torch.zeros(1, 9)
        q[:, 7] = (.095-stats['qpos_mean'][7])/stats['qpos_std'][7]
        image = torch.zeros(1, 1, 3, 90, 160)
        target = torch.zeros(1, 20, 8)
        target[..., 7] = (.0948-stats['action_mean'][7])/stats['action_std'][7]
        output = policy(q, image, target, torch.zeros(1, 20, dtype=torch.bool))
        actual = physical_actions(output['a_hat'], stats, bounded_gripper=True)
        torch.testing.assert_close(actual[..., 7], torch.full((1, 20), .095), rtol=0., atol=1e-8)
        output['loss'].backward()
        self.assertGreater(float(policy.model.action_head.bias.grad[7]), 0.)

    def test_auxiliary_validation_uses_zero_latent_and_padding_without_changing_predictions(self):
        torch.set_num_threads(2)
        policy = FullTaskPolicy(self.config()).eval()
        q, image = torch.zeros(1, 9), torch.zeros(1, 1, 3, 90, 160)
        target = torch.zeros(1, 20, 8)
        padding = torch.zeros(1, 20, dtype=torch.bool)
        padding[:, 10:] = True
        changed = target.clone()
        changed[:, 10:] = 100.
        with torch.no_grad():
            first = policy(q, image, target, padding)
            second = policy(q, image, changed, padding)
            torch.testing.assert_close(first['loss'], second['loss'], rtol=0., atol=0.)
            torch.testing.assert_close(first['a_hat'], second['a_hat'], rtol=0., atol=0.)
            torch.testing.assert_close(first['a_hat'], policy(q, image), rtol=0., atol=0.)
            self.assertIn('prior_gripper', first)
            self.assertGreater(float(first['prior_gripper']), 0.)
            self.assertEqual(float(first['kl']), 0.)

    def test_nonfinite_or_negative_auxiliary_weight_is_rejected(self):
        for weight in [-.1, float('nan'), float('inf')]:
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                config = self.config()
                config['prior_gripper_loss_weight'] = weight
                FullTaskPolicy(config)

    def test_verified_grasp_corrections_remain_partial_and_keep_source_splits(self):
        from tests.test_recovery_training import RecoveryTrainingTests
        from training.recoveries import validate_corrections
        manifest = RecoveryTrainingTests().rows()
        for row in manifest['episodes']:
            row['task'] = 'grasp_to_retreat_correction'
        try:
            paths = validate_corrections(manifest, [], check_files=False)
        except ValueError as exc:
            self.fail('Verified partial grasp correction rejected: '+str(exc))
        self.assertEqual(len(paths), 3)
        bad = copy.deepcopy(manifest)
        bad['episodes'][0]['full_task_verified'] = True
        with self.assertRaises(ValueError):
            validate_corrections(bad, [], check_files=False)

    def test_grasp_correction_cannot_replace_the_original_goal_rotation(self):
        from training.recoveries import validate_recovery_scene
        original = dict(ik_goal_rotation_link7=[[1,0,0],[0,1,0],[0,0,1]])
        snap = dict(qpos=[0.]*7+[.0475,-.0475])
        config = dict(original, continuation='grasp_to_retreat',
            joint_speed_limit_rad_s=.02, teacher_command_lead_limit_rad=.015,
            gripper_speed_limit_m_s=.004, teacher_gripper_feedback_lead_m=.008,
            dwell_seconds=dict(close=8,release=8),
            initial_robot_state=dict(qpos_rad=[0.]*7,gripper_total_width_m=.095))
        validate_recovery_scene(config, original, snap)
        config['ik_goal_rotation_link7'] = [[-1,0,0],[0,-1,0],[0,0,1]]
        with self.assertRaises(ValueError):
            validate_recovery_scene(config, original, snap)
