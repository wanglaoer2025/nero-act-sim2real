import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from src.act.episode_io import save_episode
from src.act.nero_dataset import (
    NeroEpisodicDataset,
    compute_norm_stats,
    denormalize_actions,
    episode_effective_end,
    valid_episode_paths,
)


def make_episode(path: Path, length: int, offset: float):
    images = np.full((length, 12, 16, 3), 100, dtype=np.uint8)
    qpos = np.tile(np.arange(7, dtype=np.float32), (length, 1)) + offset
    gripper = np.linspace(0.09, 0.03, length, dtype=np.float32)[:, None]
    actions = np.concatenate([qpos, gripper], axis=1)
    timestamps = np.arange(length, dtype=np.float64) * 0.05
    save_episode(path, images, qpos, actions, timestamps, gripper_states=gripper,
                 metadata={"control_frequency_hz": 20.0})


class NeroDatasetTest(unittest.TestCase):
    def test_command_history_is_past_target_not_current_label_or_feedback(self):
        import h5py
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'episode_0.hdf5'
            make_episode(path,4,0.)
            with h5py.File(path,'a') as root:
                root.attrs['source_base_xyz_mm']=json.dumps([350,-80,47.5])
                root.attrs['destination_base_xyz_mm']=json.dumps([350,80,47.5])
                root['action'][:,7]=[0.,.095,0.,.095]
            stats=compute_norm_stats([path],coordinate_conditioning=True,previous_gripper_conditioning=True)
            self.assertEqual(stats['qpos_mean'].shape,(15,))
            def state(start):
                item=NeroEpisodicDataset([path],['camera_0'],stats,fixed_start_ts=start,
                    coordinate_conditioning=True,previous_gripper_conditioning=True)[0]
                return item[1].numpy()*stats['qpos_std']+stats['qpos_mean']
            self.assertAlmostEqual(float(state(0)[14]),.09,places=6)
            self.assertAlmostEqual(float(state(1)[14]),0.,places=6)
            before=state(2)
            self.assertAlmostEqual(float(before[14]),.095,places=6)
            with h5py.File(path,'a') as root:root['action'][2:,7]=.023
            np.testing.assert_array_equal(state(2),before)
            with self.assertRaises(ValueError):
                NeroEpisodicDataset([path],['camera_0'],stats,previous_gripper_conditioning=True)

    def test_return_home_tail_is_trimmed_after_gripper_reopens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_0.hdf5"
            make_episode(path, 10, 0.0)
            import h5py
            gripper = np.array([0.095, 0.095, 0, 0, 0, 0, 0.095, 0.095, 0.095, 0.095])
            with h5py.File(path, "a") as root:
                root["action"][:, 7] = gripper
                root["observations/gripper_state"][:, 0] = gripper
                self.assertEqual(episode_effective_end(root, 2), 9)
            stats = compute_norm_stats([path], trim_after_gripper_open_steps=2)
            sample = NeroEpisodicDataset(
                [path], ["camera_0"], stats, fixed_start_ts=5,
                trim_after_gripper_open_steps=2,
            )
            self.assertEqual(sample.max_action_len, 9)
            self.assertEqual(sample[0][3].tolist(), [False] * 4 + [True] * 5)

    def test_coordinate_conditioning_appends_source_and_destination_in_metres(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_0.hdf5"
            make_episode(path, 3, 0.0)
            import h5py
            with h5py.File(path, "a") as root:
                root.attrs["source_base_xyz_mm"] = json.dumps([-500, 20, 80])
                root.attrs["destination_base_xyz_mm"] = json.dumps([-550, -100, 0])
            stats = compute_norm_stats([path], coordinate_conditioning=True)
            sample = NeroEpisodicDataset(
                [path], ["camera_0"], stats, fixed_start_ts=0,
                coordinate_conditioning=True,
            )[0]

            self.assertEqual(stats["qpos_mean"].shape, (14,))
            self.assertEqual(sample[1].shape, (14,))
            restored = sample[1].numpy() * stats["qpos_std"] + stats["qpos_mean"]
            np.testing.assert_allclose(restored[8:], [-0.5, 0.02, 0.08, -0.55, -0.1, 0.0])

    def test_two_camera_frames_are_resized_and_stacked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_0.hdf5"
            length = 3
            qpos = np.zeros((length, 7), dtype=np.float32)
            gripper = np.zeros((length, 1), dtype=np.float32)
            actions = np.zeros((length, 8), dtype=np.float32)
            timestamps = np.arange(length, dtype=np.float64) * 0.05
            images = {
                "camera_0": np.zeros((length, 12, 16, 3), dtype=np.uint8),
                "camera_1": np.zeros((length, 9, 11, 3), dtype=np.uint8),
            }
            save_episode(path, images, qpos, actions, timestamps, gripper_states=gripper)
            stats = compute_norm_stats([path])

            sample = NeroEpisodicDataset(
                [path], ["camera_0", "camera_1"], stats,
                image_size=(10, 14), fixed_start_ts=0,
            )[0]

            self.assertEqual(sample[0].shape, (2, 3, 10, 14))

    def test_joint_lookahead_does_not_advance_gripper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_0.hdf5"
            length = 4
            images = np.zeros((length, 12, 16, 3), dtype=np.uint8)
            qpos = np.zeros((length, 7), dtype=np.float32)
            gripper = np.array([[0.095], [0.095], [0.0], [0.0]], dtype=np.float32)
            actions = np.concatenate([
                np.repeat(np.arange(length, dtype=np.float32)[:, None], 7, axis=1),
                gripper,
            ], axis=1)
            save_episode(path, images, qpos, actions, np.arange(length) * 0.05,
                         gripper_states=gripper)
            stats = compute_norm_stats([path])

            sample = NeroEpisodicDataset(
                [path], ["camera_0"], stats, fixed_start_ts=0,
                joint_lookahead_steps=2,
            )[0]
            restored = denormalize_actions(sample[2], stats).numpy()

            np.testing.assert_allclose(restored[:, 0], [2, 3, 3, 3], atol=1e-5)
            np.testing.assert_allclose(restored[:, 7], gripper[:, 0], atol=1e-5)

    def test_joint_targets_can_ignore_recording_time_action_lead(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_0.hdf5"
            length = 4
            make_episode(path, length, 0.0)
            import h5py
            with h5py.File(path, "a") as root:
                qpos = np.repeat(np.arange(length, dtype=np.float32)[:, None], 7, axis=1)
                root["observations/qpos"][:] = qpos
                root["action"][:, :7] = np.roll(qpos, -1, axis=0)
            stats = compute_norm_stats([path], joint_target_lead_steps=2)
            sample = NeroEpisodicDataset(
                [path], ["camera_0"], stats, fixed_start_ts=0,
                joint_target_lead_steps=2,
            )[0]
            restored = denormalize_actions(sample[2], stats).numpy()

            np.testing.assert_allclose(restored[:, 0], [2, 3, 3, 3], atol=1e-5)

    def test_manifest_variable_lengths_shapes_padding_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, rejected = root / "episode_1.hdf5", root / "episode_2.hdf5", root / "episode_3.hdf5"
            make_episode(first, 5, 0.0)
            make_episode(second, 8, 0.5)
            make_episode(rejected, 4, 1.0)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"episodes": [
                {"path": str(first), "status": "valid"},
                {"path": str(second), "status": "valid"},
                {"path": str(rejected), "status": "rejected"},
            ]}), encoding="utf-8")

            paths = valid_episode_paths(manifest)
            stats = compute_norm_stats(paths)
            dataset = NeroEpisodicDataset(paths, ["camera_0"], stats,
                                          image_size=(10, 14), fixed_start_ts=2)
            image, qpos, action, is_pad, start = dataset[0]

            self.assertEqual([path.name for path in paths], ["episode_1.hdf5", "episode_2.hdf5"])
            self.assertEqual(image.shape, (1, 3, 10, 14))
            self.assertEqual(qpos.shape, (8,))
            self.assertEqual(action.shape, (8, 8))
            self.assertEqual(is_pad.shape, (8,))
            self.assertEqual(start, 2)
            self.assertEqual(is_pad.tolist(), [False, False, False, True, True, True, True, True])
            self.assertEqual(stats["qpos_mean"].shape, (8,))
            self.assertEqual(stats["action_mean"].shape, (8,))
            restored = denormalize_actions(action[:3], stats).numpy()
            np.testing.assert_allclose(restored[:, :7], np.tile(np.arange(7), (3, 1)), atol=1e-5)

            batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
            self.assertEqual(batch[0].shape, (2, 1, 3, 10, 14))
            self.assertEqual(batch[1].shape, (2, 8))
            self.assertEqual(batch[2].shape, (2, 8, 8))
            self.assertEqual(batch[3].shape, (2, 8))

            shifted = NeroEpisodicDataset(paths, ["camera_0"], stats,
                                          alignment_offset=1, fixed_start_ts=1)[0]
            shifted_real = denormalize_actions(shifted[2][:1], stats).numpy()[0]
            np.testing.assert_allclose(shifted_real[:7], np.arange(7), atol=1e-5)
            self.assertAlmostEqual(shifted_real[-1], 0.06, places=5)

            fixed_samples = NeroEpisodicDataset(
                paths, ["camera_0"], stats, sample_starts=[(0, 1), (1, 6)]
            )
            self.assertEqual(len(fixed_samples), 2)
            self.assertEqual(fixed_samples[0][-1], 1)
            self.assertEqual(fixed_samples[1][-1], 6)


if __name__ == "__main__":
    unittest.main()
