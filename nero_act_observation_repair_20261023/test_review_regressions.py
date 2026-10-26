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


class ReviewRegressions(unittest.TestCase):
    def module(self,stem):
        path=ROOT/(stem+'.py')
        self.assertTrue(path.is_file(),'Missing preserved-source review fix '+stem)
        spec=importlib.util.spec_from_file_location('_review_'+stem,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_one_sided_depth_outliers_cannot_pass_unsigned_quality_gate(self):
        m=self.module('rgbd_observer_v2')
        rows=json.loads((ROOT/'rgbd_frame_audit/manifest.json').read_text())['records']
        row=next(x for x in rows if x['case']==9005 and x['query']==492)
        rgb=cv2.cvtColor(cv2.imread(row['image']),cv2.COLOR_BGR2RGB)
        depth=np.load(row['depth_path'])['depth_m'].copy()
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        mask=(hsv[:,:,0]>=15)&(hsv[:,:,0]<=45)&(hsv[:,:,1]>85)&(hsv[:,:,2]>30)
        mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        rng=np.random.default_rng(42);indices=rng.choice(np.flatnonzero(mask),int(mask.sum()*.15),replace=False)
        depth.flat[indices]+=.005
        estimator=m.RgbdCupEstimator(training_profile(row['scene_config']),rgb.shape[:2])
        with self.assertRaisesRegex(ValueError,'RGB-D fit rejected'):
            estimator.measure_depth(rgb,depth,np.zeros(depth.shape))

    def test_axial_surface_motion_cannot_be_called_static_from_unchanged_roi(self):
        m=self.module('depth_quality_v2');mask=np.ones((8,8),bool);conf=np.zeros((8,8),np.float32)
        samples=[]
        for z in [.5,.51,.52]:
            xyz=np.zeros((8,8,3),np.float32);xyz[:,:,2]=z
            samples.append(dict(mask=mask,xyz=xyz,confidence=conf))
        result=m.sequence_quality(samples)
        self.assertFalse(result['static_repeatability_valid'])
        self.assertIsNone(result['per_pixel_depth_std_p95_mm'])

    def test_lateral_surface_motion_is_also_rejected_when_roi_stays_the_same(self):
        m=self.module('depth_quality_v2');mask=np.ones((8,8),bool);conf=np.zeros((8,8),np.float32)
        samples=[]
        for x in [0,.01,.02]:
            xyz=np.zeros((8,8,3),np.float32);xyz[:,:,2]=.5;xyz[:,:,0]=x
            samples.append(dict(mask=mask,xyz=xyz,confidence=conf))
        self.assertFalse(m.sequence_quality(samples)['static_repeatability_valid'])

    def test_small_static_depth_jitter_stays_a_repeatability_measure_only(self):
        m=self.module('depth_quality_v2');mask=np.ones((8,8),bool);conf=np.zeros((8,8),np.float32)
        samples=[]
        for z in [.5,.5002,.4998]:
            xyz=np.zeros((8,8,3),np.float32);xyz[:,:,2]=z
            samples.append(dict(mask=mask,xyz=xyz,confidence=conf))
        result=m.sequence_quality(samples)
        self.assertTrue(result['static_repeatability_valid'])
        self.assertLess(result['per_pixel_depth_std_p95_mm'],.3)
        self.assertFalse(result['absolute_accuracy_verified'])
        self.assertFalse(result['stationary_scene_verified'])


class RgbdV2ExistingBehaviorTests(__import__('test_rgbd_observer').RgbdObserverTests):
    def frame(self,case,query):
        _,rgb,depth,row=super().frame(case,query)
        spec=importlib.util.spec_from_file_location('_rgbd_v2_existing',ROOT/'rgbd_observer_v2.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.RgbdCupEstimator(training_profile(row['scene_config']),rgb.shape[:2]),rgb,depth,row


class DepthV2ExistingBehaviorTests(__import__('test_depth_quality').DepthQualityTests):
    def module(self):
        spec=importlib.util.spec_from_file_location('_depth_v2_existing',ROOT/'depth_quality_v2.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module


if __name__=='__main__':unittest.main()
