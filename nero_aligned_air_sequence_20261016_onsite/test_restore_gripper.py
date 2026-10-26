import unittest
import numpy as np

from restore_gripper import gripper_rows, gripper_result


class RestoreGripperTests(unittest.TestCase):
    def profile(self):
        return {'initial_state': [0.] * 7 + [.060118],
                'gripper_lower_m': .052, 'gripper_upper_m': .078,
                'arm_lower_rad': [-.05] * 7, 'arm_upper_rad': [.05] * 7}

    def test_recovery_holds_arm_and_slowly_reaches_exact_gripper_start(self):
        start = [.02] * 7 + [.06643]
        rows = gripper_rows(start, self.profile())
        self.assertTrue(all(r[:7] == start[:7] for r in rows))
        self.assertEqual(rows[-1][7], .060118)
        self.assertLessEqual(np.max(abs(np.diff([start[7]]+[r[7] for r in rows]))), .00025)

    def test_unexpected_position_and_large_width_change_are_rejected(self):
        for state in [[.1]*7+[.06643], [0.]*7+[.077], [0.]*7+[float('nan')]]:
            with self.subTest(state=state):
                with self.assertRaises(ValueError): gripper_rows(state, self.profile())

    def test_static_feedback_cannot_pass_recovery(self):
        start = [.02]*7+[.06643]
        self.assertFalse(gripper_result(start, .060118, [start]*10, True)['success'])

    def test_arm_drift_or_failed_action_is_not_accepted(self):
        start = [.02]*7+[.06643]; end = start[:7]+[.060118]
        self.assertTrue(gripper_result(start, .060118, [end]*10, True)['success'])
        self.assertFalse(gripper_result(start, .060118, [end]*10, False)['success'])
        end[0] += .002
        self.assertFalse(gripper_result(start, .060118, [end]*10, True)['success'])


if __name__ == '__main__': unittest.main()
