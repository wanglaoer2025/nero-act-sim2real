import importlib.util,json
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

class GeneralizedEvaluationTests(unittest.TestCase):
    def test_frozen_test_cases_cover9_cup4_start3_site_cases_without_training_seeds(self):
        self.assertIsNotNone(importlib.util.find_spec('simulation.evaluate_generalized'))
        from simulation.evaluate_generalized import test_cases
        base=json.loads((ROOT/'scene/center.json').read_text())
        site=dict(source_xy_m=(np.array(base['pick_xy_m'])+[0,-.024]).tolist(),
                  target_xy_m=base['place_xy_m'],actual_state=[*base['initial_robot_state']['qpos_rad'],.0809])
        cases=test_cases(base,site)
        self.assertEqual(len(cases),16)
        self.assertEqual(len({c['name'] for c in cases}),16)
        self.assertEqual(len({c['seed'] for c in cases}),16)
        self.assertGreater(min(c['seed'] for c in cases),180500)
        offsets={tuple(np.round(np.array(c['scene']['pick_xy_m'])-base['pick_xy_m'],5)) for c in cases[:9]}
        self.assertEqual(offsets,{(x,y) for x in [-.02,0,.02] for y in [-.02,0,.02]})
        self.assertTrue(all(c.get('requires_gripper_preparation') for c in cases[-3:]))
        self.assertTrue(all(c['scene']['initial_robot_state']['gripper_total_width_m']>=.09 for c in cases))
        self.assertEqual(len({c['servo_delay_steps'] for c in cases[-3:]}),3)
