import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

class GeneralizationTests(unittest.TestCase):
    def test_scene_groups_cover_centimetres_without_train_validation_duplicates(self):
        self.assertIsNotNone(importlib.util.find_spec('simulation.generalization'))
        from simulation.generalization import episode_config
        base=json.loads((ROOT/'scene/center.json').read_text())
        scenes=[episode_config(base,i,'train' if i<36 else 'val') for i in range(48)]
        xy=np.array([s['pick_xy_m'] for s in scenes])-base['pick_xy_m']
        np.testing.assert_allclose(xy[:36].min(0),[-.03,-.03],atol=1e-12)
        np.testing.assert_allclose(xy[:36].max(0),[.03,.03],atol=1e-12)
        self.assertFalse({tuple(s['pick_xy_m']) for s in scenes[:36]} & {tuple(s['pick_xy_m']) for s in scenes[36:]})
        for i,s in enumerate(scenes):
            self.assertEqual(s['place_xy_m'],base['place_xy_m'])
            self.assertLessEqual(s['action_delay_steps'],2)
            self.assertTrue(.09<=s['initial_robot_state']['gripper_total_width_m']<=.095)
            self.assertLessEqual(np.max(np.abs(s['generalization']['initial_translation_m'])),.02)
            self.assertEqual(s,episode_config(base,i,'train' if i<36 else 'val'))
        with self.assertRaises(ValueError):episode_config(base,36,'train')

    def test_initial_ik_matches_declared_translation_and_has_no_unexpected_contact(self):
        self.assertIsNotNone(importlib.util.find_spec('simulation.generalization'))
        from simulation.generalization import episode_config,validate_initial_pose
        from simulation.scene import build_scene
        base=json.loads((ROOT/'scene/center.json').read_text())
        for i in [0,5,35,47]:
            scene=episode_config(base,i,'train' if i<36 else 'val')
            report=validate_initial_pose(build_scene(scene),scene,base)
            self.assertLess(report['position_error_m'],.0002)
            self.assertLess(report['rotation_error'],.002)
            self.assertFalse(report['unexpected_contact'])

