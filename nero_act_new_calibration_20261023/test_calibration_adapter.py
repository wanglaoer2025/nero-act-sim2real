import json
from pathlib import Path
import unittest

import numpy as np

from calibration_adapter import candidate_to_metres, verify_handeye_fit, apply_to_scene

ROOT = Path(__file__).resolve().parent


class CalibrationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.candidate = json.loads((ROOT/'inputs/candidate_camera_to_base.json').read_text())

    def test_camera_origin_is_converted_to_metres_without_rotating_the_optical_frame(self):
        transform = candidate_to_metres(self.candidate)
        np.testing.assert_allclose(transform[:3,3], [-.8802533802903315, -.0680567051197011, .38314256433174734])
        np.testing.assert_allclose(transform[:3,0], [.02255291879948775, .9997455961233941, .0003299780346987363])
        np.testing.assert_array_equal(transform[3], [0,0,0,1])

    def test_wrong_direction_units_and_invalid_rotation_are_rejected(self):
        for key, value in [('source_coordinate_system','nero_base'), ('translation_unit','meter')]:
            candidate = dict(self.candidate, **{key:value})
            with self.assertRaises(ValueError):
                candidate_to_metres(candidate)
        self.candidate['transform_base_from_camera'][0][0] = 2.
        with self.assertRaises(ValueError):
            candidate_to_metres(self.candidate)

    def test_saved_inlier_fit_is_reproduced_from_the_twenty_raw_pairs(self):
        samples = json.loads((ROOT/'inputs/handeye_samples.json').read_text())
        report = verify_handeye_fit(self.candidate, samples)
        self.assertEqual(report['inlier_count'], 15)
        self.assertAlmostEqual(report['translation_rmse_mm'], 4.5858453402331945, places=5)
        self.assertAlmostEqual(report['rotation_rmse_deg'], .6728617839143588, places=5)
        self.assertFalse(report['independent_task_space_validation_passed'])

    def test_apply_preserves_TCP_intrinsics_and_input_scene(self):
        old = json.loads((ROOT.parent/'nero_act_preservation_scope_20261023/current_scene.json').read_text())
        saved = json.loads(json.dumps(old))
        new = apply_to_scene(old, self.candidate)
        self.assertEqual(old, saved)
        self.assertEqual(new['tcp_link7_xyz_m'], old['tcp_link7_xyz_m'])
        self.assertEqual(new['camera_calibration']['intrinsic_matrix'], old['camera_calibration']['intrinsic_matrix'])
        self.assertEqual(new['camera_calibration']['image_rotation_degrees'], 180)
        self.assertFalse(new['camera_calibration']['installation_verified_on_real'])
        self.assertFalse(new['hardware_deployment_allowed'])


if __name__ == '__main__':
    unittest.main()
