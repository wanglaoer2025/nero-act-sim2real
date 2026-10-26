import unittest
import numpy as np
from cup_sweep import cylinder_bounds, check_cup_floor, rotate_vector


class CupSweepTests(unittest.TestCase):
    def test_historical_grasp_rotation_keeps_cup_vertical(self):
        q = [0, -np.sqrt(.5), np.sqrt(.5), 0]
        np.testing.assert_allclose(rotate_vector([0, -1, 0], q), [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(rotate_vector([.135, .0075, 0], q),
                                   [-.135, 0, -.0075], atol=1e-12)
    def test_vertical_contact_and_tilted_wall_extents(self):
        lower, upper = cylinder_bounds([0, 0, .0475], [0, 0, 1])
        np.testing.assert_allclose(lower, [-.0375, -.0375, 0], atol=1e-12)
        np.testing.assert_allclose(upper, [.0375, .0375, .095], atol=1e-12)
        lower, upper = cylinder_bounds([0, 0, .04], [1, 0, 0])
        self.assertAlmostEqual(lower[2], .0025)

    def test_mid_path_table_penetration_is_rejected(self):
        check_cup_floor([0, 0, .0475], [0, 0, 1])
        with self.assertRaises(ValueError):
            check_cup_floor([0, 0, .045], [0, 0, 1])
        with self.assertRaises(ValueError):
            check_cup_floor([0, 0, .04], [0, 1, 1])
        with self.assertRaises(ValueError):
            check_cup_floor([0, 0, float('nan')], [0, 0, 1])


if __name__ == '__main__':
    unittest.main()
