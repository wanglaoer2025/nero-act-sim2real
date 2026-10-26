import unittest
import numpy as np
from air_prepare import preparation_rows, preparation_result


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.start = [.1882, -.1265, .1000, 1.5708, -.0238, .2817, .1310, .00003]
        self.target = [.1848, -.1284, .1099, 1.5715, -.0227, .2778, .1355, .060118]
        self.limits = [[-3, 3]] * 7

    def test_closed_empty_gripper_opens_slowly_without_overshoot(self):
        rows = np.array(preparation_rows(self.start, self.target, self.limits))
        self.assertTrue(np.all(np.diff(rows[:, 7]) >= 0))
        self.assertLessEqual(np.max(np.abs(np.diff(rows, axis=0)[:, 7])) / .05, .005)
        self.assertLessEqual(np.max(np.abs(np.diff(rows, axis=0)[:, :7])) / .05, .005)
        np.testing.assert_allclose(rows[-1], self.target)

    def test_large_arm_transfer_and_gripper_closing_are_rejected(self):
        for axis, value in [(0, self.target[0]+.1), (7, -.001)]:
            target = self.target.copy(); target[axis] = value
            with self.assertRaises(ValueError):
                preparation_rows(self.start, target, self.limits)

    def test_static_or_failed_action_cannot_pass(self):
        self.assertFalse(preparation_result(self.start, self.target, [self.start]*20, True)['success'])
        self.assertFalse(preparation_result(self.start, self.target, [self.target]*20, False)['success'])

    def test_excursion_is_checked_even_when_endpoint_matches(self):
        bad = self.start.copy(); bad[0] += .01
        self.assertFalse(preparation_result(self.start, self.target, [bad]+[self.target]*20, True)['success'])
        self.assertTrue(preparation_result(self.start, self.target, [self.start]+[self.target]*20, True)['success'])


if __name__ == '__main__':
    unittest.main()
