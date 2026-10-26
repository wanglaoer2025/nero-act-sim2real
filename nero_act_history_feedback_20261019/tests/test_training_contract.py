import importlib.util
from pathlib import Path
import tempfile
import unittest
import pickle

import h5py
import numpy as np
import torch


class TrainingContractTests(unittest.TestCase):
    def test_task_budget_uses_training_demonstration_duration_and_excludes_validation(self):
        from training import train
        self.assertTrue(callable(getattr(train, 'training_reach_deadline', None)))
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, count in [(0, 3371), (1, 8000)]:
                path = Path(directory)/f'episode_{index}.hdf5'
                with h5py.File(path, 'w') as f:f.create_dataset('action', shape=(count, 8), dtype='f4')
                paths.append(path)
            self.assertEqual(train.training_reach_deadline(paths, [0]), 4300)
            partial = Path(directory)/'episode_2.hdf5'
            with h5py.File(partial, 'w') as f:
                f.create_dataset('action', shape=(7000, 8), dtype='f4');f.attrs['task'] = 'retreat_correction'
            self.assertEqual(train.training_reach_deadline(paths+[partial], [0, 2]), 4300)

    def test_finetuning_new_profile_requires_same_dataset_and_embedded_scene(self):
        from training import train
        self.assertTrue(callable(getattr(train, 'check_initializer_dataset', None)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);checkpoint = root/'model.ckpt'
            scene = {'pick_xy_m': [-.525, .015]}
            torch.save({'config': dict(dataset_dir=str(root), scene_config=scene)}, checkpoint)
            train.check_initializer_dataset(checkpoint, root, scene)
            with self.assertRaises(ValueError):
                train.check_initializer_dataset(checkpoint, root, {'pick_xy_m': [-.5, 0]})

    def test_prefix_margin_is_a_training_penalty_with_physical_units_and_gradients(self):
        from training import train
        self.assertTrue(callable(getattr(train, 'prefix_margin_penalty', None)))
        stats = dict(qpos_mean=np.zeros(8), qpos_std=np.ones(8),
                     action_mean=np.zeros(8), action_std=np.r_[np.full(7, .5), .01])
        state = torch.tensor([[0.]*7+[.075]])
        physical = np.tile(np.r_[np.full(7, .028), .065], (1, 20, 1))
        prediction = torch.tensor(physical/stats['action_std'], dtype=torch.float32, requires_grad=True)
        padding = torch.zeros(1, 20, dtype=torch.bool)
        penalty = train.prefix_margin_penalty(prediction, state, padding, stats)
        self.assertGreater(float(penalty.detach()), 0)
        penalty.backward()
        self.assertGreater(float(prediction.grad[0, 0, 0]), 0)
        self.assertLess(float(prediction.grad[0, 0, 7]), 0)
        self.assertEqual(float(prediction.grad[:, 10:].abs().sum()), 0)
        safe = torch.tensor(np.tile(np.r_[np.full(7, .01), .073], (1, 20, 1))/stats['action_std'])
        self.assertEqual(float(train.prefix_margin_penalty(safe, state, padding, stats)), 0)

    def test_supplemental_full_data_must_have_actual_full_task_physics_evidence(self):
        import json
        from training import train
        self.assertTrue(callable(getattr(train, 'supplemental_full_inputs', None)), 'full delayed-demo reader missing')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'episode_20.hdf5'
            item = dict(path=str(path), split='train', status='valid', full_task_verified=True)
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(dict(episodes=[item], rejected=[])))
            with h5py.File(path, 'w') as f:
                f.attrs.update(sim=True, teacher_success=True, full_task_verified=False,
                               task='retreat_correction')
            with self.assertRaises(ValueError):
                train.supplemental_full_inputs(manifest, list(range(12)))
            with h5py.File(path, 'a') as f:
                f.attrs.update(full_task_verified=True, task='complete_pick_carry_place_retreat')
            self.assertEqual(train.supplemental_full_inputs(manifest, list(range(12))), [item])
            with self.assertRaises(ValueError):
                train.supplemental_full_inputs(manifest, [20])

    def test_new_dataset_requires_its_own_scene_instead_of_historical_start(self):
        import json
        from training import train
        self.assertTrue(callable(getattr(train, 'dataset_scene', None)), 'dataset scene binding missing')
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            with self.assertRaises(FileNotFoundError):
                train.dataset_scene(data)
            expected = {'initial_robot_state': {'qpos_rad': [.1]*7}, 'pick_xy_m': [-.5, .01]}
            (data/'scene_config.json').write_text(json.dumps(expected))
            self.assertEqual(train.dataset_scene(data), expected)

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('training.train'), 'complete-task training adapter missing')
        from training.train import weighted_l1, complete_windows, training_statistics
        self.loss, self.windows, self.statistics = weighted_l1, complete_windows, training_statistics

    def test_all_twenty_steps_supervised_with_near_ten_weighted(self):
        prediction = torch.zeros(1, 20, 8, requires_grad=True)
        target = torch.zeros_like(prediction)
        target[:, 10:] = 1
        value = self.loss(prediction, target, torch.zeros(1, 20, dtype=torch.bool))
        self.assertAlmostEqual(float(value.detach()), 1/6, places=6)
        value.backward()
        self.assertGreater(float(prediction.grad[:, 10:].abs().sum()), 0)

    def test_training_only_statistics_and_terminal_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, value in enumerate([1., 100.]):
                path = Path(directory)/f'episode_{index}.hdf5'
                with h5py.File(path, 'w') as f:
                    f['observations/qpos'] = np.full((31, 7), value, np.float32)
                    f['observations/gripper_state'] = np.full((31, 1), .08, np.float32)
                    f['action'] = np.full((31, 8), value, np.float32)
                    f['simulation/stage'] = np.array([b'release']*20+[b'retreat']*11)
                paths.append(path)
            stats = self.statistics(paths, [0])
            self.assertAlmostEqual(float(stats['qpos_mean'][0]), 1.)
            windows = self.windows(paths[:1], 20, stride=10)
            self.assertIn((0, 30), windows)
            self.assertIn((0, 20), windows)

    def test_corrective_finetuning_does_not_import_validation_statistics(self):
        from training import train
        self.assertTrue(callable(getattr(train, 'reuse_training_statistics', None)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'policy.ckpt'
            torch.save(dict(split={'train': [0], 'val': [1]}), path)
            stats = dict(qpos_mean=np.ones(8), qpos_std=np.ones(8),
                         action_mean=np.ones(8), action_std=np.ones(8))
            with (path.parent/'dataset_stats.pkl').open('wb') as f:
                pickle.dump(stats, f)
            retained = train.reuse_training_statistics(path, [0, 12])
            self.assertAlmostEqual(float(retained['qpos_mean'][0]), 1.)
            with self.assertRaises(ValueError):
                train.reuse_training_statistics(path, [1, 12])

    def test_release_deadband_ramp_is_not_restarted_from_unchanged_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'episode_0.hdf5'
            actions=np.zeros((60,8),np.float32);actions[:,7]=.065+np.arange(60)*.00025
            with h5py.File(path,'w') as f:
                f['action']=actions
                f['observations/gripper_state']=np.full((60,1),.074)
                f['simulation/stage']=np.array([b'release']*60)
            windows=self.windows([path],20)
            for start in [0,10,20,30]:self.assertNotIn((0,start),windows)
            for start in [40,50]:self.assertIn((0,start),windows)

    def test_stationary_contact_and_open_dwell_do_not_teach_endless_wait(self):
        for stage, width, target, next_stage in [('close', .075, .065, 'lift'),
                                                 ('release', .095, .095, 'retreat')]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'episode_0.hdf5'
                actions = np.zeros((60, 8), np.float32)
                actions[:, 7] = target
                actions[40:, 0] = np.arange(1, 21)*.002
                with h5py.File(path, 'w') as f:
                    f['action'] = actions
                    f['observations/gripper_state'] = np.full((60, 1), width)
                    f['simulation/stage'] = np.array([stage.encode()]*40+[next_stage.encode()]*20)
                windows = self.windows([path], 20, stride=10)
                self.assertNotIn((0, 0), windows)
                self.assertNotIn((0, 30), windows)
                self.assertIn((0, 40), windows)


if __name__ == '__main__':
    unittest.main()
