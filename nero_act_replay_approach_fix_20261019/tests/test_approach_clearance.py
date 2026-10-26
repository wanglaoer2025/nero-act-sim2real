import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent/'fixtures'


class ApproachClearanceTests(unittest.TestCase):
    def test_safe_high_pose_is_accepted_and_malformed_reference_is_rejected(self):
        from executor.approach_clearance import approach_clearance
        scene = json.loads((ROOT/'model/frozen_run11/scene_config.json').read_text())
        first = np.load(FIXTURES/'last_real_approach.npz')['sent_targets'][0]
        rows = np.tile(first, (2, 1))
        manifest = dict(phases=[dict(name='approach', start=0, end=2)])
        result = approach_clearance(scene, manifest, rows)
        self.assertTrue(result['passed'])
        self.assertGreaterEqual(result['minimum_gap_m'], .005)
        rows[1, 0] = np.nan
        with self.assertRaises(ValueError): approach_clearance(scene, manifest, rows)

    def test_fresh_act_candidate_has_clearance_without_modifying_arm_or_gripper_targets(self):
        from executor.approach_clearance import require_approach_clearance
        from executor.scene_acceptance import execution_scene
        manifest = json.loads((ROOT/'replay/manifest.json').read_text())
        scene = execution_scene(ROOT, json.loads((ROOT/'model/frozen_run11/scene_config.json').read_text()), manifest)
        rows = np.load(ROOT/'replay/act_rollout.npz')['sent_targets']
        before = rows.copy()
        result = require_approach_clearance(scene, manifest, rows, {})
        self.assertTrue(result['passed'])
        self.assertGreater(result['minimum_gap_m'], .008)
        self.assertTrue(np.array_equal(rows, before))

    def test_nominally_collision_free_old_approach_is_rejected_before_camera_opens(self):
        from executor.replay import execute
        fixture = json.loads((FIXTURES/'last_real_approach.json').read_text())
        manifest, scene = fixture['manifest'], fixture['scene']
        rows = np.load(FIXTURES/'last_real_approach.npz')['sent_targets']
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = types.SimpleNamespace(model=ROOT/'model/frozen_run11', output=output)
            camera_calls = []
            def camera(*args):
                camera_calls.append(True)
                raise RuntimeError('UNSAFE_APPROACH_REACHED_CAMERA')
            with patch('executor.replay.execution_scene', return_value=scene), \
                    patch('executor.camera.SceneCamera', camera), \
                    patch('executor.replay.SceneObserver', return_value=None):
                report = execute(args, {}, manifest, rows)
            self.assertIn('clearance', report['error'].lower())
            self.assertEqual(camera_calls, [])
            self.assertFalse(report['execution_goal_submitted'])
            self.assertLess(report['approach_clearance']['minimum_gap_m'], .005)


if __name__ == '__main__':
    unittest.main()
