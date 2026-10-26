import importlib.util
from pathlib import Path
import unittest

import numpy as np


class RgbdGeometryTests(unittest.TestCase):
    def test_upright_depth_preserves_raw_optical_vectors_and_metric_units(self):
        path=Path(__file__).resolve().parent/'rgbd_geometry.py'
        self.assertTrue(path.is_file(),'Missing explicit native depth backprojection')
        spec=importlib.util.spec_from_file_location('_tested_rgbd_geometry',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        camera=dict(intrinsic_matrix=[[2,0,1],[0,2,1],[0,0,1]],resolution_wh=[3,3],
            image_rotation_degrees=180,transform_base_from_raw_optical_m=np.eye(4).tolist())
        depth=np.ones((3,3),np.float32)*2
        p=m.world_points_from_depth(depth,camera)
        np.testing.assert_allclose(p[1,1],[0,0,2])
        np.testing.assert_allclose(p[0,0],[1,1,2])
        np.testing.assert_allclose(p[2,2],[-1,-1,2])
        camera['transform_base_from_raw_optical_m'][0][3]=.3
        q=m.world_points_from_depth(depth,camera)
        np.testing.assert_allclose(q[1,1],[.3,0,2])
        with self.assertRaises(ValueError):m.world_points_from_depth(depth[:2],camera)


if __name__=='__main__':unittest.main()
