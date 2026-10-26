import unittest
import numpy as np
from full_core import FullGate
from async_runtime import ActionPrefix


def summary():
    return {'settings': {'joint_limits_rad': [[-3, 3]]*7},
            'train_qpos_min': [-.1]*7, 'train_qpos_max': [.1]*7}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.state = np.r_[np.zeros(7), .06]
        self.chunk = np.tile(self.state, (100, 1))

    def test_shadow_checkpoint_cannot_execute_even_if_pose_matches(self):
        gate = FullGate(summary(), False)
        self.assertIn('checkpoint', gate.startup_issues(self.state, self.chunk)[0])
        with self.assertRaisesRegex(ValueError, 'not enabled'):
            gate.point(self.state, self.state, self.state)

    def test_real_119degree_discontinuity_is_rejected_not_rebased(self):
        gate = FullGate(summary(), True)
        chunk = self.chunk.copy(); chunk[:, 2] = 2.185
        self.assertTrue(any('discontinuous' in issue for issue in gate.startup_issues(self.state, chunk)))
        with self.assertRaisesRegex(ValueError, 'discontinuous'):
            gate.point(chunk[0], self.state, self.state)

    def test_instant_gripper_closure_is_rejected(self):
        gate = FullGate(summary(), True)
        chunk = self.chunk.copy(); chunk[:, 7] = .00149
        self.assertTrue(any('gripper' in issue for issue in gate.startup_issues(self.state, chunk)))
        with self.assertRaisesRegex(ValueError, 'gripper'):
            gate.point(chunk[0], self.state, self.state)

    def test_safe_rate_limits_cover_arm_and_continuous_gripper(self):
        gate = FullGate(summary(), True)
        target = self.state.copy(); target[:7] = .02; target[7] = .065
        safe = gate.point(target, self.state, self.state)
        np.testing.assert_allclose(safe[:7], .0075)
        self.assertAlmostEqual(safe[7], .0605)

    def test_static_state_range_is_not_widened(self):
        gate = FullGate(summary(), True)
        state = self.state.copy(); state[2] = .5
        self.assertTrue(any('joint3' in issue for issue in gate.startup_issues(state)))

    def test_no_gripper_rate_clip_can_undo_range(self):
        gate = FullGate(summary(), True)
        state = self.state.copy(); state[7] = .1
        target = state.copy(); target[7] = .095
        with self.assertRaisesRegex(ValueError, 'gripper range'):
            gate.point(target, state, state)

    def test_frozen_or_late_prediction_cannot_be_reused(self):
        prefix = ActionPrefix(10, 15, .75)
        plan = {'observation_step': 0, 'chunk': self.chunk, 'details': {'observation_wall_s': 10.}}
        action, info = prefix.action(1, plan, now=11.)
        self.assertIsNone(action)
        self.assertEqual(info['hold_reason'], 'stale_wall_observation')

    def test_only_ten_actions_per_chunk_even_with_100_predictions(self):
        prefix = ActionPrefix(10, 15, .75)
        plan = {'observation_step': 0, 'chunk': self.chunk, 'details': {'observation_wall_s': 10.}}
        for i in range(10):
            self.assertIsNotNone(prefix.action(i, plan, now=10.+i*.05)[0])
        self.assertIsNone(prefix.action(10, plan, now=10.5)[0])

    def test_late_negative_gripper_is_found_in_complete_preview(self):
        gate = FullGate(summary(), True)
        chunk = self.chunk.copy(); chunk[99, 7] = -.004
        self.assertTrue(any('raw100' in issue for issue in gate.startup_issues(self.state, chunk)))

    def test_shadow_model_rejected_before_creating_ros_or_worker(self):
        import full_runtime
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        import hashlib, json
        with TemporaryDirectory() as directory:
            root = full_runtime.Path(directory)
            (root/'policy.ckpt').write_bytes(b'not loaded')
            (root/'dataset_stats.pkl').write_bytes(b'not loaded')
            digest = hashlib.sha256(b'not loaded').hexdigest()
            (root/'runtime_metadata.json').write_text(json.dumps({'checkpoint_sha256': digest, 'stats_sha256': digest,
                'hardware_deployment_allowed': False, 'experimental_air_trial_allowed': False}))
            (root/'safety_summary.json').write_text(json.dumps(summary()))
            args = SimpleNamespace(model_dir=root)
            with patch.object(full_runtime, 'PredictionProcess', side_effect=AssertionError('must not start')):
                with self.assertRaisesRegex(RuntimeError, 'permission'):
                    full_runtime.live(args, full_runtime.Path(directory), {})


if __name__ == '__main__':
    unittest.main()
