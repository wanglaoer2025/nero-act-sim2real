from collections import Counter
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from training import transition_sampling


class BalancedGripperTests(unittest.TestCase):
    def episode(self, path, *, release=True):
        with h5py.File(path, 'w') as stream:
            stages = ['approach']*400 + ['close']*60 + ['lift']*60
            stages += ['release' if release else 'retreat']*80
            stream.create_dataset('simulation/stage', data=stages, dtype=h5py.string_dtype())
            stream.create_dataset('observations/gripper_state', data=np.full((600, 1), .095))
            action = np.zeros((600, 8))
            action[:, 0] = np.arange(600)*.0001
            action[:, 7] = .094
            stream.create_dataset('action', data=action)

    def test_contact_onsets_get_equal_mass_while_retaining_background_and_each_source(self):
        function = getattr(transition_sampling, 'balanced_gripper_windows', None)
        self.assertTrue(callable(function), 'balanced gripper sampling is missing')
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/f'episode_{i}.hdf5' for i in range(2)]
            for path in paths:
                self.episode(path)
            samples = function(paths, 20, command_context=True)
            counts = Counter('close' if 400 <= start < 420 else
                             'release' if 520 <= start < 540 else 'background'
                             for _, start in samples)
            self.assertEqual(counts, {'close': 40, 'release': 40, 'background': 40})
            self.assertEqual(samples, function(paths, 20, command_context=True))
            self.assertEqual({episode for episode, _ in samples}, {0, 1})
            original = set(transition_sampling.critical_windows(paths, 20, command_context=True))
            self.assertTrue(set(samples) <= original)

    def test_missing_contact_role_refuses_a_misleading_balanced_run(self):
        function = getattr(transition_sampling, 'balanced_gripper_windows', None)
        self.assertTrue(callable(function), 'balanced gripper sampling is missing')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'episode_0.hdf5'
            self.episode(path, release=False)
            with self.assertRaises(ValueError):
                function([path], 20, command_context=True)

    def test_verified_direct_grasp_recovery_gets_its_own_offline_sampling_mass(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/f'episode_{index}.hdf5' for index in range(2)]
            for path in paths:
                self.episode(path)
            with h5py.File(paths[0], 'a') as stream:
                stream.attrs['task'] = 'grasp_to_retreat_correction'
            samples = transition_sampling.balanced_gripper_windows(
                paths, 20, command_context=True, focus_grasp_recovery=True)
            counts = Counter('actual_close' if episode == 0 and 400 <= start < 460 else
                             'close' if 400 <= start < 420 else
                             'release' if 520 <= start < 540 else 'background'
                             for episode, start in samples)
            self.assertEqual(counts, dict(actual_close=40, close=40, release=40, background=40))


if __name__ == '__main__':
    unittest.main()
