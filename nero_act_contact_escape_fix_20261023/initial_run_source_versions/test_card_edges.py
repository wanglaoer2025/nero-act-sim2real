import importlib.util
import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'nero_act_new_handeye_contact_repair_20261023'))
from test_card_fragments import CardFragmentsTests
from training.legacy_visual_data import training_profile


class CardEdgesTests(CardFragmentsTests):
    def estimator(self):
        spec=importlib.util.spec_from_file_location('_tested_card_edges',ROOT/'card_edges.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        scene=json.loads((ROOT/'results/9000/scene_configuration.json').read_text())
        return module.CardEdgesEstimator(training_profile(scene),(180,320)),scene

    def test_release_occlusion_with_three_continuous_edges_remains_measurable(self):
        estimator,scene=self.estimator()
        rgb=cv2.cvtColor(cv2.imread(str(ROOT/'release_visual_diagnostic/failure_frame.png')),cv2.COLOR_BGR2RGB)
        original=rgb.copy()
        _,target,diagnostic=estimator.measure(rgb)
        np.testing.assert_array_equal(rgb,original)
        self.assertLess(np.linalg.norm(target[:2]-scene['place_xy_m']),.003)
        self.assertLess(diagnostic['target_visible_corners'],3)
        self.assertGreaterEqual(diagnostic['target_supported_edges'],3)
        self.assertFalse(diagnostic['runtime_qualified'])


if __name__=='__main__':unittest.main()
