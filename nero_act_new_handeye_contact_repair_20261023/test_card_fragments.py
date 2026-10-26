import importlib.util
import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'nero_act_new_handeye_policy_repair_20261023'
sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
from training.legacy_visual_data import training_profile


class CardFragmentsTests(unittest.TestCase):
    def estimator(self):
        path=ROOT/'card_fragments.py'
        self.assertTrue(path.is_file(),'Missing current-frame card geometry consistency check')
        spec=importlib.util.spec_from_file_location('card_fragments',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        scene=json.loads((PREVIOUS/'results/9006/scene_configuration.json').read_text())
        return module.CardFragmentsEstimator(training_profile(scene),(180,320)),scene

    def image(self):
        bgr=cv2.imread(str(PREVIOUS/'visual_failure_diagnostic/failure_frame.png'))
        self.assertIsNotNone(bgr)
        return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)

    def test_two_visible_parts_of_the_same_card_support_a_current_position(self):
        estimator,scene=self.estimator()
        rgb=self.image();original=rgb.copy()
        cup,card,diagnostic=estimator.measure(rgb)
        np.testing.assert_array_equal(rgb,original)
        self.assertLess(np.linalg.norm(card[:2]-scene['place_xy_m']),.003)
        self.assertEqual(diagnostic['target_visible_fragments'],2)
        self.assertGreaterEqual(diagnostic['target_visible_corners'],3)
        self.assertFalse(diagnostic['runtime_qualified'])

    def test_full_occlusion_and_half_visible_card_are_rejected(self):
        estimator,_=self.estimator();rgb=self.image()
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        blue=(hsv[:,:,0]>=95)&(hsv[:,:,0]<=130)&(hsv[:,:,1]>90)&(hsv[:,:,2]>25)
        hidden=rgb.copy();hidden[blue]=[120,120,120]
        with self.assertRaises(ValueError):estimator.measure(hidden)
        hidden=rgb.copy();hidden[:,175:][blue[:,175:]]=[120,120,120]
        with self.assertRaises(ValueError):estimator.measure(hidden)

    def test_unrelated_blue_objects_are_not_merged_into_a_target(self):
        estimator,_=self.estimator();rgb=self.image()
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        blue=(hsv[:,:,0]>=95)&(hsv[:,:,0]<=130)&(hsv[:,:,1]>90)&(hsv[:,:,2]>25)
        rgb[blue]=[120,120,120]
        rgb[140:155,200:215]=[13,102,204]
        rgb[140:155,240:255]=[13,102,204]
        with self.assertRaises(ValueError):estimator.measure(rgb)

    def test_three_corner_patches_without_an_occluder_do_not_make_a_card(self):
        estimator,_=self.estimator();rgb=self.image()
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        blue=(hsv[:,:,0]>=95)&(hsv[:,:,0]<=130)&(hsv[:,:,1]>90)&(hsv[:,:,2]>25)
        y,x=np.nonzero(blue)
        points=np.asarray([estimator.ray_at_table([u,v])[:2] for u,v in zip(x,y)])
        low,high=points.min(0),points.max(0)
        corners=np.array([[low[0],low[1]],[low[0],high[1]],[high[0],low[1]],[high[0],high[1]]])
        keep=np.min(np.linalg.norm(points[:,None]-corners[None],axis=-1),axis=1)<.018
        rgb[y[~keep],x[~keep]]=[200,200,200]
        with self.assertRaises(ValueError):estimator.measure(rgb)


if __name__=='__main__':unittest.main()
