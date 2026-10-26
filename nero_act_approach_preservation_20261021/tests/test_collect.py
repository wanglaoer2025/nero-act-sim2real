import importlib.util
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np


class CollectTests(unittest.TestCase):
    def test_initial_command_history_is_preserved_and_expert_projection_is_not_hidden(self):
        import json
        from simulation.collect import verify_action_contract
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'episode.hdf5'
            with h5py.File(path, 'w') as f:
                f['observations/qpos'] = np.zeros((20, 7))
                f['observations/gripper_state'] = np.full((20, 1), .074)
                f['action'] = np.c_[np.full((20, 7), .011), np.full(20, .065)]
            with self.assertRaises(ValueError):verify_action_contract(path)
            with h5py.File(path, 'a') as f:
                f.attrs['scene_config'] = json.dumps(dict(initial_simulation_snapshot={
                    'previous_target': [.01]*7+[.065]}))
            self.assertTrue(verify_action_contract(path)['passed'])

    def test_physically_successful_data_still_requires_prefix_contract(self):
        from simulation import collect
        self.assertTrue(callable(getattr(collect, 'verify_action_contract', None)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'episode.hdf5'
            with h5py.File(path, 'w') as f:
                f['observations/qpos'] = np.zeros((30, 7))
                f['observations/gripper_state'] = np.full((30, 1), .095)
                f['action'] = np.c_[np.full((30, 7), .04), np.full(30, .095)]
            with self.assertRaises(ValueError):collect.verify_action_contract(path)
            with h5py.File(path, 'a') as f:f['action'][:, :7] = .001
            self.assertTrue(collect.verify_action_contract(path)['passed'])

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('simulation.collect'), 'physical data export is missing')
        from simulation.collect import save_records, collect_episode
        self.save, self.collect = save_records, collect_episode

    def test_existing_output_is_rejected_without_initializing_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'episode.hdf5'
            path.write_bytes(b'user data')
            with self.assertRaises(FileExistsError):
                self.collect({}, path)
            self.assertEqual(path.read_bytes(), b'user data')

    def test_same_cycle_labels_measured_width_and_stage_are_preserved(self):
        q = np.zeros(16)
        q[7:9] = [.04, -.04]
        rows = [dict(qpos=q.copy(), qvel=np.zeros(15), action=np.r_[np.ones(7)*.1, .065],
                     timestamp=t, stage=stage, rgb=np.zeros((90, 160, 3), np.uint8),
                     cup_xyz=[-.5, 0, .0475], torque=np.zeros(7))
                for t, stage in [(0., 'close'), (.05, 'release')]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'episode.hdf5'
            self.save(path, rows, {'task': 'complete'}, {'success': True})
            with h5py.File(path, 'r') as f:
                self.assertEqual(f['observations/qpos'].shape, (2, 7))
                self.assertAlmostEqual(float(f['observations/gripper_state'][0, 0]), .08)
                self.assertAlmostEqual(float(f['action'][0, 7]), .065)
                self.assertEqual(f['simulation/stage'].asstr()[-1], 'release')
                self.assertTrue(f.attrs['sim'])
                self.assertFalse(f.attrs['hardware_deployment_allowed'])


if __name__ == '__main__':
    unittest.main()
