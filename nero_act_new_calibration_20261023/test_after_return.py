import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


class AfterReturnTests(unittest.TestCase):
    def test_measured_start_is_preserved_and_preparation_is_explicitly_hypothetical(self):
        path = ROOT/'evaluate_after_return.py'
        self.assertTrue(path.is_file(), 'Missing current-scene offline diagnostic runner')
        spec = importlib.util.spec_from_file_location('after_return', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scene = json.loads((ROOT/'scene_relocalized_after_return.json').read_text())
        original = json.loads(json.dumps(scene))
        cases = module.build_cases(scene)
        self.assertEqual(scene, original)
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]['scene'], original)
        self.assertFalse(cases[0]['hypothetical_preparation'])
        self.assertAlmostEqual(cases[0]['scene']['initial_robot_state']['gripper_total_width_m'], .09942000359296799)
        self.assertTrue(cases[1]['hypothetical_preparation'])
        prepared = cases[1]['scene']
        self.assertEqual(prepared['initial_robot_state']['gripper_total_width_m'], .095)
        # Only gripper state and diagnostic provenance differ; no virtual arm/target alignment.
        prepared['initial_robot_state']['gripper_total_width_m'] = original['initial_robot_state']['gripper_total_width_m']
        self.assertEqual(prepared, original)


if __name__ == '__main__':
    unittest.main()
