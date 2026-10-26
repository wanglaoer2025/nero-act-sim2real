import json
import sys
import tempfile
import types
import unittest
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np


class GraspSceneCheckTests(unittest.TestCase):
    def test_nonfinite_sensor_window_is_rejected_in_all_replay_stages(self):
        from executor.replay import observe
        camera = types.SimpleNamespace(capture=lambda: {})
        sensor = types.SimpleNamespace(measure=lambda _: dict(cup_xyz=np.array([np.nan, 0, 0]),
            target_xyz=np.array([.2, 0, 0]), camera_monotonic_s=time.monotonic()))
        session = types.SimpleNamespace(spin=lambda _: None)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertRaisesRegex(RuntimeError, 'finite'):
                observe(camera, sensor, pool, session)

    def execute_scene(self, cups, cards, *, real_observe=False):
        from executor.replay import execute, observe
        holder = {}
        class Session:
            def __init__(self, *args):
                holder['contact'] = args[-1]
                self.trace = []; self.health_trace = []
                self.node = types.SimpleNamespace(destroy_node=lambda: None)
                self.ros = types.SimpleNamespace(try_shutdown=lambda: None)
                self.geometry = types.SimpleNamespace(pose=lambda _: dict(tcp=np.array([0, 0, .055])))
            def inventory(self): return [0]*8
            def ready(self, _): pass
            def warm_services(self): pass
            def enable_monitoring(self): pass
            def spin(self, _): pass
            def configure_initial_scene(self): pass
            def clients(self): return []
            def actual(self): return [0]*8
            def run_phase(self, name, rows, *args):
                if name == 'close': raise RuntimeError('UNSAFE_CLOSE_ENTERED')
                return rows
            def cancel_all(self): pass
            def contact_permission(self, _): pass
            def allow_finger_contact(self, _): pass
        class Camera:
            def __init__(self, *args): pass
            def capture(self): return {}
            def close(self): pass
        module = types.ModuleType('task_replay')
        module.observer = lambda: types.SimpleNamespace(disconnect=lambda: None)
        initial = [dict(cup_xyz=np.zeros(3), target_xyz=np.array([.2, 0, 0]))]*8
        fresh = [dict(cup_xyz=np.array(c), target_xyz=np.array(t), frame=f'frame_{i}.png')
                 for i, (c, t) in enumerate(zip(cups, cards))]
        sequence = iter([initial[0]]+initial+fresh)
        def measure(_):
            return dict(next(sequence), camera_monotonic_s=time.monotonic())
        scene = dict(pick_xy_m=[0, 0], place_xy_m=[.2, 0])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); model = out/'model'; model.mkdir()
            (model/'scene_config.json').write_text(json.dumps(scene))
            geometry = out/'geometry.json'; geometry.write_text('{}')
            args = types.SimpleNamespace(model=model, geometry=geometry, output=out)
            with patch.dict(sys.modules, {'task_replay': module}), \
                    patch('executor.replay_session.replay_session_class', return_value=Session), \
                    patch('executor.camera.SceneCamera', Camera), \
                    patch('executor.replay.SceneObserver', return_value=types.SimpleNamespace(measure=measure)), \
                    patch('executor.replay.observe', side_effect=observe if real_observe else [initial, fresh]), \
                    patch('executor.replay.execution_scene', return_value=scene), \
                    patch('executor.replay.require_current_scene'), \
                    patch('executor.replay.require_approach_clearance'), \
                    patch('executor.replay.confirm'):
                report = execute(args, {}, dict(source_rollout_sha256='trace', phases=[
                    dict(name='approach', start=0, end=2), dict(name='close', start=2, end=4)]), np.zeros((4, 8)))
        return report, holder['contact']

    def test_real_unstable_observer_failure_keeps_the_collected_frames(self):
        report, contact = self.execute_scene([[.006, 0, 0]]*7+[[0, 0, 0]],
                                             [[.2, 0, 0]]*8, real_observe=True)
        self.assertFalse(contact.close_authorized)
        self.assertIn('not independently stable', report['error'])
        self.assertEqual(len(report['grasp_scene_observations']), 8)

    def test_bad_earlier_stable_frame_cannot_be_hidden_by_last_good_frame(self):
        report, contact = self.execute_scene([[.006, 0, 0]]*7+[[.004, 0, 0]], [[.2, 0, 0]]*8)
        self.assertFalse(contact.close_authorized)
        self.assertEqual(report['grasp_scene_check']['reason'], 'cup_source_shift')
        self.assertAlmostEqual(report['grasp_scene_check']['maximum_cup_offset_m'], .006)
        self.assertEqual(len(report['grasp_scene_observations']), 8)

    def test_card_movement_also_prevents_contact_authorization(self):
        report, contact = self.execute_scene([[0, 0, 0]]*8, [[.206, 0, 0]]*8)
        self.assertFalse(contact.close_authorized)
        self.assertEqual(report['grasp_scene_check']['reason'], 'card_target_shift')

    def test_nonfinite_measurement_cannot_authorize_close(self):
        report, contact = self.execute_scene([[float('nan'), 0, 0]]*8, [[.2, 0, 0]]*8)
        self.assertFalse(contact.close_authorized)
        self.assertEqual(report['grasp_scene_check']['reason'], 'invalid_scene_observation')

    def test_unchanged_source_reaches_existing_close_contact_gate(self):
        report, contact = self.execute_scene([[0, 0, 0]]*8, [[.2, 0, 0]]*8)
        self.assertTrue(contact.close_authorized)
        self.assertIn('UNSAFE_CLOSE_ENTERED', report['error'])
        self.assertTrue(report['grasp_scene_check']['passed'])
