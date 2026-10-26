import copy
import math
import unittest

from run_act_response import projected_target, path_check, actual_check, response_result
from ros_support import ARM, retime_seconds


class ResponseTests(unittest.TestCase):
    def chunk(self):
        return [[1., -2., 3., .1, -.2, .3, -.4, .095] for _ in range(100)]

    def test_model_direction_is_preserved_and_bounded(self):
        target, info = projected_target([0.]*8, self.chunk())
        self.assertAlmostEqual(max(abs(v) for v in target), .005)
        self.assertAlmostEqual(target[0]/target[2], 1/3)
        self.assertLess(target[1], 0)
        self.assertEqual(info['raw_action_index'], 9)

    def test_different_model_prediction_changes_physical_target(self):
        first, _ = projected_target([0.]*8, self.chunk())
        chunk = self.chunk()
        chunk[9][:7] = [-v for v in chunk[9][:7]]
        second, _ = projected_target([0.]*8, chunk)
        for a, b in zip(first, second):
            self.assertAlmostEqual(a, -b)

    def test_hold_prediction_never_becomes_scripted_motion(self):
        with self.assertRaisesRegex(ValueError, 'no sufficiently observable'):
            projected_target([0.]*8, [[0.]*8]*100)

    def test_nonfinite_prediction_rejected(self):
        chunk = self.chunk(); chunk[2][4] = math.nan
        with self.assertRaises(ValueError):
            projected_target([0.]*8, chunk)

    def test_path_cannot_detour_outside_projection(self):
        start = [0.]*8; target, _ = projected_target(start, self.chunk())
        with self.assertRaisesRegex(ValueError, 'corridor'):
            path_check(ARM, [[0.]*7, [.02]*7, target], start, target)

    def test_planner_must_use_model_endpoint(self):
        target, _ = projected_target([0.]*8, self.chunk())
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            path_check(ARM, [[0.]*7, [.005]+[0.]*6], [0.]*8, target)

    def test_static_real_feedback_cannot_pass(self):
        start = [0.]*8; target, _ = projected_target(start, self.chunk())
        self.assertFalse(response_result(start, target, [start]*100, True)['success'])

    def test_negative_model_motion_can_pass(self):
        start = [0.]*8; chunk = self.chunk(); chunk[9][2] = -3.
        target, _ = projected_target(start, chunk)
        result = response_result(start, target, [target+[0.]]*20, True)
        self.assertTrue(result['success'])
        self.assertEqual(result['dominant_model_axis'], 'joint3')

    def test_action_failure_cannot_pass_good_feedback(self):
        target, _ = projected_target([0.]*8, self.chunk())
        self.assertFalse(response_result([0.]*8, target, [target+[0.]]*20, False)['success'])

    def test_unrequested_gripper_motion_aborts(self):
        with self.assertRaisesRegex(RuntimeError, 'gripper'):
            actual_check([0.]*7+[.002], [0.]*8, [.005]*7)

    def test_observation_window_exceeds_historical_startup(self):
        target, _ = projected_target([0.]*8, self.chunk())
        self.assertGreaterEqual(retime_seconds([[0.]*7, target])[-1], 4.)


if __name__ == '__main__':
    unittest.main()
