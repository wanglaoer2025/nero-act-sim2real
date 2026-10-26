"""NERO HDF5 adapter for the official ACT episodic data flow."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


STATE_DIM = 8
ACTION_DIM = 8
TASK_COORDINATE_DIM = 6


def episode_trim_start(path: Path, default: int = 0,
                       by_parent: dict[str, int] | None = None) -> int:
    """Return a dataset-specific leading-frame trim without changing HDF5 files."""
    result = int(default)
    if result < 0:
        raise ValueError("trim_start_steps 不得为负")
    resolved = path.resolve()
    for parent, value in (by_parent or {}).items():
        if resolved.is_relative_to(Path(parent).resolve()):
            result = int(value)
            break
    if result < 0:
        raise ValueError("trim_start_steps_by_parent 不得包含负数")
    return result


def episode_effective_end(root: h5py.File,
                          keep_after_gripper_open_steps: int | None = None) -> int:
    """Return an exclusive end index, optionally trimming the return-home tail."""
    length = len(root["action"])
    if keep_after_gripper_open_steps is None:
        return length
    keep = int(keep_after_gripper_open_steps)
    if keep < 0:
        raise ValueError("trim_after_gripper_open_steps 不得为负")
    closed = root["action"][:, 7] < 0.0475
    close_steps = np.flatnonzero(closed)
    if not len(close_steps):
        raise ValueError("启用尾段裁剪的 episode 没有闭爪动作")
    reopen_steps = np.flatnonzero(~closed & (np.arange(length) > close_steps[0]))
    if not len(reopen_steps):
        raise ValueError("启用尾段裁剪的 episode 没有松爪动作")
    return min(length, int(reopen_steps[0]) + keep + 1)


def valid_episode_paths(manifest_path: Path, episode_ids: list[int] | None = None) -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wanted = None if episode_ids is None else set(episode_ids)
    paths = []
    for episode in manifest["episodes"]:
        path = Path(episode["path"])
        try:
            episode_id = int(path.stem.removeprefix("episode_"))
        except ValueError:
            continue
        if episode["status"] == "valid" and (wanted is None or episode_id in wanted):
            paths.append(path)
    if wanted is not None and {int(path.stem.removeprefix("episode_")) for path in paths} != wanted:
        raise ValueError("请求的 episode 不全在 manifest valid 集合中")
    if not paths:
        raise ValueError("manifest 中没有可用的 valid episode")
    return sorted(paths, key=lambda path: int(path.stem.removeprefix("episode_")))


def _task_coordinates(root: h5py.File) -> np.ndarray:
    """Read fixed source/destination base coordinates and convert mm to m."""
    values = []
    for name in ("source_base_xyz_mm", "destination_base_xyz_mm"):
        if name not in root.attrs:
            raise ValueError(f"episode 缺少任务坐标属性 {name}")
        raw = root.attrs[name]
        if isinstance(raw, (str, bytes)):
            raw = json.loads(raw)
        point = np.asarray(raw, dtype=np.float32)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"episode 属性 {name} 必须是有限的 3 维坐标")
        values.append(point)
    return np.concatenate(values) / 1000.0


def _read_state(root: h5py.File, coordinate_conditioning: bool = False) -> np.ndarray:
    qpos = root["observations/qpos"][:].astype(np.float32)
    gripper = root["observations/gripper_state"][:].astype(np.float32)
    state = np.concatenate([qpos, gripper], axis=1)
    if state.shape[1] != STATE_DIM:
        raise ValueError(f"完整 NERO state 必须为 {STATE_DIM} 维")
    if coordinate_conditioning:
        coordinates = np.broadcast_to(_task_coordinates(root), (len(state), TASK_COORDINATE_DIM))
        state = np.concatenate([state, coordinates], axis=1)
    return state


def compute_norm_stats(paths: list[Path], coordinate_conditioning: bool = False,
                       trim_start_steps: int = 0,
                       trim_start_steps_by_parent: dict[str, int] | None = None,
                       joint_target_lead_steps: int | None = None,
                       trim_after_gripper_open_steps: int | None = None
                       ) -> dict[str, np.ndarray]:
    """Compute train-only stats by concatenating variable-length episodes."""
    if joint_target_lead_steps is not None and joint_target_lead_steps < 0:
        raise ValueError("joint_target_lead_steps 不得为负")
    states, actions = [], []
    for path in paths:
        with h5py.File(path, "r") as root:
            first_step = episode_trim_start(
                path, trim_start_steps, trim_start_steps_by_parent
            )
            end_step = episode_effective_end(root, trim_after_gripper_open_steps)
            state = _read_state(root, coordinate_conditioning)[first_step:end_step]
            action = root["action"][:end_step].astype(np.float32)
            if joint_target_lead_steps is not None:
                indices = np.minimum(
                    np.arange(len(action)) + joint_target_lead_steps,
                    len(action) - 1,
                )
                action[:, :7] = root["observations/qpos"][:end_step][indices]
            action = action[first_step:]
            if not len(state):
                raise ValueError(f"{path} 裁剪后为空")
            states.append(state)
            if action.ndim != 2 or action.shape[1] != ACTION_DIM:
                raise ValueError(f"{path} action 必须为 (T,{ACTION_DIM})")
            actions.append(action)
    all_states = np.concatenate(states, axis=0)
    all_actions = np.concatenate(actions, axis=0)
    return {
        "qpos_mean": all_states.mean(axis=0).astype(np.float32),
        "qpos_std": np.maximum(all_states.std(axis=0), 1e-2).astype(np.float32),
        "action_mean": all_actions.mean(axis=0).astype(np.float32),
        "action_std": np.maximum(all_actions.std(axis=0), 1e-2).astype(np.float32),
    }


def denormalize_actions(actions: torch.Tensor | np.ndarray, stats: dict) -> torch.Tensor:
    values = torch.as_tensor(actions)
    mean = torch.as_tensor(stats["action_mean"], device=values.device, dtype=values.dtype)
    std = torch.as_tensor(stats["action_std"], device=values.device, dtype=values.dtype)
    return values * std + mean


def preprocess_images(images: np.ndarray, image_size=(90, 160)) -> torch.Tensor:
    """Convert RGB uint8 ``[N,H,W,3]`` images to ACT ``[N,3,h,w]`` tensors."""
    values = np.asarray(images)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("ACT 图像必须为 [N,H,W,3] RGB 数组")
    if values.dtype != np.uint8:
        raise ValueError("ACT 图像必须为 uint8")
    image = torch.from_numpy(np.ascontiguousarray(values)).permute(0, 3, 1, 2).float() / 255.0
    return F.interpolate(image, size=tuple(image_size), mode="bilinear", align_corners=False)


class NeroEpisodicDataset(Dataset):
    """Return one random observation and its padded future action sequence."""

    def __init__(self, paths: list[Path], camera_names: list[str], norm_stats: dict,
                 *, alignment_offset: int = 0, image_size=(90, 160), fixed_start_ts=None,
                 sample_starts: list[tuple[int, int]] | None = None,
                 coordinate_conditioning: bool = False, joint_lookahead_steps: int = 0,
                 joint_target_lead_steps: int | None = None,
                 trim_after_gripper_open_steps: int | None = None):
        self.paths = list(paths)
        self.camera_names = list(camera_names)
        self.stats = norm_stats
        self.alignment_offset = int(alignment_offset)
        self.image_size = tuple(image_size)
        self.fixed_start_ts = fixed_start_ts
        self.sample_starts = sample_starts
        self.coordinate_conditioning = bool(coordinate_conditioning)
        self.joint_lookahead_steps = int(joint_lookahead_steps)
        self.joint_target_lead_steps = (
            None if joint_target_lead_steps is None else int(joint_target_lead_steps)
        )
        self.trim_after_gripper_open_steps = trim_after_gripper_open_steps
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names 必须非空且不能重复")
        if self.joint_lookahead_steps < 0:
            raise ValueError("joint_lookahead_steps 不得为负")
        if self.joint_target_lead_steps is not None and (
            self.joint_target_lead_steps < 0 or self.joint_lookahead_steps
        ):
            raise ValueError("joint_target_lead_steps 必须非负且不能与 joint_lookahead_steps 同用")
        self.max_action_len = max(self._episode_length(path) for path in self.paths)

    def _episode_length(self, path: Path) -> int:
        with h5py.File(path, "r") as root:
            return episode_effective_end(root, self.trim_after_gripper_open_steps)

    def __len__(self):
        return len(self.sample_starts) if self.sample_starts is not None else len(self.paths)

    def __getitem__(self, index):
        if self.sample_starts is None:
            path = self.paths[index]
            fixed_start = self.fixed_start_ts
        else:
            path_index, fixed_start = self.sample_starts[index]
            path = self.paths[path_index]
        with h5py.File(path, "r") as root:
            length = episode_effective_end(root, self.trim_after_gripper_open_steps)
            start = int(fixed_start) if fixed_start is not None else int(np.random.randint(length))
            if not 0 <= start < length:
                raise IndexError(f"start_ts {start} 超出 episode 长度 {length}")
            state = np.concatenate([
                root["observations/qpos"][start].astype(np.float32),
                root["observations/gripper_state"][start].astype(np.float32),
            ])
            if self.coordinate_conditioning:
                state = np.concatenate([state, _task_coordinates(root)])
            images = [
                root[f"observations/images/{camera}"][start] for camera in self.camera_names
            ]
            action_start = min(max(start + self.alignment_offset, 0), length - 1)
            actions = root["action"][:length].astype(np.float32)
            future = actions[action_start:].copy()
            if self.joint_target_lead_steps is not None:
                joint_indices = np.minimum(
                    np.arange(action_start, length) + self.joint_target_lead_steps,
                    length - 1,
                )
                future[:, :7] = root["observations/qpos"][:length][joint_indices]
            elif self.joint_lookahead_steps:
                joint_indices = np.minimum(
                    np.arange(action_start, length) + self.joint_lookahead_steps,
                    length - 1,
                )
                future[:, :7] = actions[joint_indices, :7]

        padded = np.broadcast_to(self.stats["action_mean"], (self.max_action_len, ACTION_DIM)).copy()
        padded[:len(future)] = future
        is_pad = np.ones(self.max_action_len, dtype=bool)
        is_pad[:len(future)] = False
        image = torch.cat([
            preprocess_images(frame[None], self.image_size) for frame in images
        ])
        qpos = (torch.from_numpy(state) - torch.from_numpy(self.stats["qpos_mean"])) / torch.from_numpy(self.stats["qpos_std"])
        action = (torch.from_numpy(padded) - torch.from_numpy(self.stats["action_mean"])) / torch.from_numpy(self.stats["action_std"])
        return image, qpos.float(), action.float(), torch.from_numpy(is_pad), start
