import importlib.util
import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
from training.legacy_visual_data import training_profile


class CupObserverTests(unittest.TestCase):
    def setup_frame(self,case,query):
        path=ROOT/'cup_observer.py'
        self.assertTrue(path.is_file(),'Missing current-image visible-boundary cup estimator')
        spec=importlib.util.spec_from_file_location('_tested_cup_observer',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rows=json.loads((ROOT/'native_frame_audit/manifest.json').read_text())['records']
        row=next(r for r in rows if r['case']==case and r['query']==query)
        rgb=cv2.cvtColor(cv2.imread(row['image']),cv2.COLOR_BGR2RGB)
        return module.VisibleBoundaryCupEstimator(training_profile(row['scene_config']),rgb.shape[:2]),rgb,row

    def test_occluded_frame_cannot_keep_the_known_false_height_as_valid(self):
        e,rgb,row=self.setup_frame(9005,492)
        try:
            cup,_,_=e.measure(rgb)
        except ValueError as exc:
            self.assertIn('boundary',str(exc).lower())
            return
        self.assertLess(np.linalg.norm(cup-row['true_cup_bottom_for_diagnostic_m']),.002)

    def test_visible_cup_still_returns_a_current_image_position(self):
        e,rgb,row=self.setup_frame(9000,895)
        original=rgb.copy();cup,target,diagnostic=e.measure(rgb)
        np.testing.assert_array_equal(rgb,original)
        self.assertLess(np.linalg.norm(cup-row['true_cup_bottom_for_diagnostic_m']),.002)
        self.assertFalse(diagnostic['runtime_qualified'])
        self.assertFalse(diagnostic['truth_used_as_observer_input'])
        self.assertEqual(diagnostic['method'],'current_RGB_visible_boundary_cylinder')

    def test_full_occlusion_never_reuses_previous_position(self):
        e,rgb,_=self.setup_frame(9000,895)
        e.measure(rgb)
        with self.assertRaises(ValueError):e.measure(np.full_like(rgb,180))


if __name__=='__main__':unittest.main()
