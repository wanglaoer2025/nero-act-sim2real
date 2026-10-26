"""Checks on the actual exported dataset/statistics and recorded-input prediction."""
import json
from pathlib import Path
import pickle
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent


class ArtifactTests(unittest.TestCase):
    def test_episode_split_has_no_leakage(self):
        split = json.loads((ROOT/'model/split.json').read_text())
        self.assertFalse(set(split['train']) & set(split['val']))
        self.assertEqual(len(split['train']), 8)
        self.assertEqual(len(split['val']), 2)

    def test_exported_stats_are_recomputed_from_actual_training_episodes(self):
        split = json.loads((ROOT/'model/split.json').read_text())
        states, actions = [], []
        for identifier in split['train']:
            with h5py.File(ROOT/f'training_data/episode_{identifier}.hdf5') as f:
                states.append(np.c_[f['observations/qpos'][:], f['observations/gripper_state'][:]])
                actions.append(f['action'][:])
        with (ROOT/'model/dataset_stats.pkl').open('rb') as f:
            stats = pickle.load(f)
        for name, values in [('qpos', np.concatenate(states)), ('action', np.concatenate(actions))]:
            np.testing.assert_allclose(stats[name+'_mean'], values.mean(0), atol=1e-8, rtol=0)
            np.testing.assert_allclose(stats[name+'_std'], np.maximum(values.std(0), .01), atol=1e-8, rtol=0)

    def test_new_prediction_has_no_old_119degree_or_closed_gripper_jump(self):
        result = json.loads((ROOT/'reference_preview/prediction.json').read_text())
        state = np.asarray(result['state']); prefix = np.asarray(result['raw_first10'])
        self.assertLess(float(np.max(abs(prefix[:, :7]-state[:7]))), .01)
        self.assertGreater(float(prefix[:, 7].min()), .052)
        self.assertLess(float(prefix[:, 7].max()), .078)


if __name__ == '__main__':
    unittest.main()
