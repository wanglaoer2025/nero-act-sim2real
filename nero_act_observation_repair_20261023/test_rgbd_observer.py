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


class RgbdObserverTests(unittest.TestCase):
    def frame(self,case,query):
        source=ROOT/'rgbd_observer.py'
        self.assertTrue(source.is_file(),'Missing explicit current RGB-D cup pose estimator')
        spec=importlib.util.spec_from_file_location('_tested_rgbd_observer',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rows=json.loads((ROOT/'rgbd_frame_audit/manifest.json').read_text())['records']
        row=next(x for x in rows if x['case']==case and x['query']==query)
        rgb=cv2.cvtColor(cv2.imread(row['image']),cv2.COLOR_BGR2RGB)
        with np.load(row['depth_path']) as f:depth=f['depth_m']
        return module.RgbdCupEstimator(training_profile(row['scene_config']),rgb.shape[:2]),rgb,depth,row

    def test_depth_disambiguates_the_occluded_false_height(self):
        e,rgb,depth,row=self.frame(9005,492)
        original=rgb.copy();cup,target,diagnostic=e.measure_depth(rgb,depth,np.zeros(depth.shape))
        np.testing.assert_array_equal(rgb,original)
        self.assertLess(np.linalg.norm(cup-row['true_cup_bottom_for_diagnostic_m']),.002)
        self.assertFalse(diagnostic['runtime_qualified'])
        self.assertFalse(diagnostic['truth_used_as_observer_input'])
        self.assertEqual(diagnostic['method'],'current_RGB_depth_hollow_cylinder')

    def test_visible_carried_cup_remains_measurable(self):
        e,rgb,depth,row=self.frame(9000,895)
        cup,_,_=e.measure_depth(rgb,depth,np.zeros(depth.shape))
        self.assertLess(np.linalg.norm(cup-row['true_cup_bottom_for_diagnostic_m']),.002)

    def test_missing_or_low_confidence_depth_never_falls_back_to_bbox_height(self):
        e,rgb,depth,_=self.frame(9005,492)
        with self.assertRaises(ValueError):e.measure_depth(rgb,np.full_like(depth,np.nan),np.zeros(depth.shape))
        with self.assertRaises(ValueError):e.measure_depth(rgb,depth,np.ones(depth.shape)*100)

    def test_sdk_xyz_vectors_are_not_rotated_twice(self):
        e,rgb,depth,_=self.frame(9005,492)
        self.assertTrue(hasattr(e,'measure_camera_xyz'),'Missing direct SDK optical XYZ entry')
        v,u=np.indices(depth.shape);raw=np.stack([1279-u,719-v,np.ones_like(u)],axis=-1)
        xyz=(raw@np.linalg.inv(e.K).T)*depth[:,:,None]
        p,t,_=e.measure_depth(rgb,depth,np.zeros(depth.shape))
        q,s,_=e.measure_camera_xyz(rgb,xyz,np.zeros(depth.shape))
        np.testing.assert_allclose(p,q,atol=1e-8)
        np.testing.assert_allclose(t,s,atol=1e-12)
        with self.assertRaises(ValueError):e.measure_camera_xyz(rgb,xyz[:2],np.zeros(depth.shape))

    def test_clipped_reflection_is_excluded_but_two_complete_yellow_objects_are_rejected(self):
        e,rgb,depth,_=self.frame(9005,492)
        reflected=rgb.copy();reflected[:30,:30]=[220,185,20]
        p,_,_=e.measure_depth(rgb,depth,np.zeros(depth.shape))
        q,_,_=e.measure_depth(reflected,depth,np.zeros(depth.shape))
        np.testing.assert_allclose(p,q,atol=1e-8)
        reflected[100:130,100:130]=[220,185,20]
        with self.assertRaises(ValueError):e.measure_depth(reflected,depth,np.zeros(depth.shape))


if __name__=='__main__':unittest.main()
