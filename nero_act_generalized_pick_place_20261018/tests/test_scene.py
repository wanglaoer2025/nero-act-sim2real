import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('simulation.scene'),
                             'complete-task scene builder has not been implemented')
        from simulation.scene import build_scene, tcp_in_link7
        self.build, self.tcp = build_scene, tcp_in_link7
        self.config = json.loads((ROOT/'scene/config.json').read_text())

    def test_model_preserves_limits_camera_and_selected_tcp(self):
        import mujoco
        model = self.build(self.config)
        self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                               'calibrated_nero_camera'), 0)
        np.testing.assert_allclose(model.jnt_range[6], [-1.5707963, 1.5707963])
        xyz, rotation = self.tcp(model)
        np.testing.assert_allclose(xyz, [.152, 0, -.0235], atol=1e-6)
        np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-8)
        table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'table')
        self.assertAlmostEqual(model.geom_pos[table, 2]+model.geom_size[table, 2], 0.)
        self.assertEqual(model.neq, 1)  # finger symmetry only; never weld the cup

    def test_same_source_and_destination_cannot_be_a_full_transfer(self):
        config = copy.deepcopy(self.config)
        config['place_xy_m'] = config['pick_xy_m']
        with self.assertRaisesRegex(ValueError, 'transfer'):
            self.build(config)

    def test_invalid_geometry_and_force_are_rejected(self):
        for key, value in [('table_top_z_m', float('nan')),
                           ('finger_force_limit_n', 0), ('finger_force_limit_n', 5)]:
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.assertRaises(ValueError):
                self.build(config)

    def test_reflected_camera_transform_is_rejected(self):
        config = copy.deepcopy(self.config)
        config['camera_calibration']['transform_base_from_raw_optical_m'][0][0] *= -1
        with self.assertRaises(ValueError):
            self.build(config)


if __name__ == '__main__':
    unittest.main()
