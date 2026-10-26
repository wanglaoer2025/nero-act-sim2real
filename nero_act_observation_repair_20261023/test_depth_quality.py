import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parent


class DepthQualityTests(unittest.TestCase):
    def module(self):
        path=ROOT/'depth_quality.py'
        self.assertTrue(path.is_file(),'Missing non-commanding depth quality analysis')
        spec=importlib.util.spec_from_file_location('_tested_depth_quality',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_invalid_depth_and_low_confidence_are_excluded(self):
        m=self.module();xyz=np.zeros((2,2,3),np.float32);xyz[:,:,2]=.5
        confidence=np.array([[10,80],[10,10]],np.float32)
        xyz[1,0,2]=np.nan;xyz[1,1,2]=-.5
        report=m.surface_quality(np.ones((2,2),bool),xyz,confidence)
        self.assertEqual(report['valid_points'],1)
        self.assertEqual(report['valid_fraction'],.25)
        self.assertEqual(report['median_surface_camera_xyz_m'],[0.,0.,.5])
        self.assertFalse(report['absolute_accuracy_verified'])
        self.assertNotIn('cup_bottom_base_m',report)

    def test_empty_roi_never_produces_a_position(self):
        m=self.module();report=m.surface_quality(np.zeros((3,3),bool),np.zeros((3,3,3)),np.zeros((3,3)))
        self.assertFalse(report['valid'])
        self.assertIsNone(report['median_surface_camera_xyz_m'])

    def test_rotation_reorders_pixels_without_rotating_xyz_vectors(self):
        m=self.module();rgb=np.arange(18,dtype=np.uint8).reshape(2,3,3)
        xyz=np.arange(18,dtype=np.float32).reshape(2,3,3);c=np.arange(6).reshape(2,3)
        a,b,d=m.upright_arrays(rgb,xyz,c)
        np.testing.assert_array_equal(a[0,0],rgb[-1,-1])
        np.testing.assert_array_equal(b[0,0],xyz[-1,-1])
        self.assertEqual(d[0,0],c[-1,-1])
        self.assertFalse(np.shares_memory(b,xyz))

    def test_motion_prevents_static_repeatability_claim(self):
        m=self.module();xyz=np.ones((12,12,3),np.float32);confidence=np.zeros((12,12),np.float32)
        first=np.zeros((12,12),bool);first[3:7,1:5]=True
        second=np.zeros_like(first);second[3:7,5:9]=True
        result=m.sequence_quality([dict(mask=first,xyz=xyz,confidence=confidence),dict(mask=second,xyz=xyz,confidence=confidence)])
        self.assertFalse(result['static_repeatability_valid'])
        self.assertIsNone(result['per_pixel_depth_std_p95_mm'])
        self.assertFalse(result['absolute_accuracy_verified'])

    def test_static_depth_repeatability_is_not_absolute_accuracy(self):
        m=self.module();mask=np.ones((6,6),bool);conf=np.zeros((6,6),np.float32)
        values=[]
        for z in [.5,.5002,.4998]:
            xyz=np.zeros((6,6,3),np.float32);xyz[:,:,2]=z
            values.append(dict(mask=mask,xyz=xyz,confidence=conf))
        report=m.sequence_quality(values)
        self.assertTrue(report['static_repeatability_valid'])
        self.assertGreater(report['per_pixel_depth_std_p95_mm'],.1)
        self.assertLess(report['per_pixel_depth_std_p95_mm'],.3)
        self.assertFalse(report['absolute_accuracy_verified'])


if __name__=='__main__':unittest.main()
