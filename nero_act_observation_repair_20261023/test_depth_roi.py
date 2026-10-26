import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent


class DepthRoiTests(unittest.TestCase):
    def module(self):
        spec=importlib.util.spec_from_file_location('_depth_roi_quality',ROOT/'depth_quality.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertTrue(hasattr(module,'cup_colour_roi'),'Missing current-frame clipped component filtering')
        return module

    def test_real_cup_remains_unique_when_border_reflection_is_present(self):
        m=self.module()
        rgb=cv2.cvtColor(cv2.imread(str(ROOT/'depth_capture_neural_light/frame_00.png')),cv2.COLOR_BGR2RGB)
        original=rgb.copy();mask,report=m.cup_colour_roi(rgb)
        np.testing.assert_array_equal(rgb,original)
        self.assertTrue(report['valid'])
        self.assertGreater(int(mask.sum()),10000)
        y,x=np.nonzero(mask)
        self.assertLess(x.max(),600)
        self.assertGreaterEqual(report['clipped_components_rejected'],1)

    def test_two_complete_yellow_objects_are_still_ambiguous(self):
        m=self.module();rgb=np.zeros((100,120,3),np.uint8)
        rgb[10:40,10:40]=[240,200,10];rgb[50:80,60:90]=[240,200,10]
        mask,report=m.cup_colour_roi(rgb)
        self.assertFalse(report['valid']);self.assertFalse(mask.any())


if __name__=='__main__':unittest.main()
