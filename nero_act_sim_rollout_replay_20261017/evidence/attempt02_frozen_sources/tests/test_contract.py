import importlib.util
import unittest

import numpy as np


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.contract'), 'full-task prefix contract missing')
        from executor.contract import FullTaskContract
        self.guard = FullTaskContract(dict(chunk_size=20, state_dim=8, action_dim=8,
                                          task_scope='complete_pick_carry_place_retreat'))
        self.state = np.r_[np.zeros(7), .095]
        self.raw = np.broadcast_to(self.state, (20, 8)).copy()

    def call(self, raw=None, age=0.):
        return self.guard.validate_prefix(self.raw if raw is None else raw,
                                          dict(state=self.state, age_s=age), self.state)

    def test_wrong_shape_nonfinite_stale_or_discontinuous_prefix_is_rejected(self):
        for raw in [np.zeros((100, 8)), np.full((20, 8), np.nan)]:
            with self.assertRaises(ValueError):
                self.call(raw)
        for age in [.251, -.01, float('nan')]:
            with self.assertRaises(ValueError):
                self.call(age=age)
        self.raw[:, 0] = .04
        with self.assertRaises(ValueError):
            self.call()
        self.raw[:, 0] = 0
        self.raw[:, 7] = .08
        with self.assertRaises(ValueError):
            self.call()

    def test_slew_limit_preserves_provenance_without_rebasing_absolute_actions(self):
        self.state[:7] = .2
        self.raw[:, :7] = .205
        self.raw[:, 7] = .09
        sent = self.call()
        self.assertAlmostEqual(sent[0, 0], .202)
        self.assertAlmostEqual(sent[2, 0], .205)
        self.assertAlmostEqual(sent[0, 7], .0945)
        self.assertTrue(np.array_equal(self.guard.last_raw, self.raw[:10]))
        np.testing.assert_allclose(self.guard.last_delta, sent-self.raw[:10])

    def test_full_motion_is_not_capped_to_small_total_amplitude(self):
        for _ in range(20):
            raw = np.broadcast_to(self.state, (20, 8)).copy()
            raw[:, 0] += .01
            self.state = self.guard.validate_prefix(raw, dict(state=self.state, age_s=0), self.state)[-1]
        self.assertAlmostEqual(self.state[0], .2)


if __name__ == '__main__':
    unittest.main()
