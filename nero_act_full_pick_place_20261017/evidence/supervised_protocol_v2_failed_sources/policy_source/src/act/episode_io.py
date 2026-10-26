"""Validation and HDF5 I/O for NERO ACT demonstrations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from src.act.robot_wrapper import ARM_DOF, GRIPPER_STATE_DIM


ACTION_DIM = ARM_DOF + 1


def build_leader_actions(leader_qpos, gripper_targets, lead_steps=0):
    """Combine Leader joint targets and gripper targets into ACT actions.

    ``lead_steps`` is available for offline experiments, but recording uses
    zero so the raw same-cycle demonstration is preserved.
    """
    qpos_array = np.asarray(leader_qpos, dtype=np.float32)
    gripper_array = np.asarray(gripper_targets, dtype=np.float32)
    if qpos_array.ndim != 2 or qpos_array.shape[1] != ARM_DOF:
        raise ValueError(f"qpos shape 必须为 (T, {ARM_DOF})")
    if gripper_array.ndim != 2 or gripper_array.shape[1] != 1:
        raise ValueError("gripper_targets shape 必须为 (T, 1)")
    if len(qpos_array) != len(gripper_array) or not len(qpos_array):
        raise ValueError("qpos 与 gripper_targets 长度必须相等且非零")
    if isinstance(lead_steps, bool) or not isinstance(lead_steps, int) or lead_steps < 0:
        raise ValueError("lead_steps 必须是大于等于 0 的整数")
    if not np.isfinite(qpos_array).all() or not np.isfinite(gripper_array).all():
        raise ValueError("Leader action 输入包含 NaN 或 Inf")
    indices = np.minimum(np.arange(len(qpos_array)) + lead_steps, len(qpos_array) - 1)
    # Joint feedback is advanced to compensate control latency.  The gripper
    # target is an explicit event and must keep its original recording time.
    return np.concatenate([qpos_array[indices], gripper_array], axis=1)


def validate_episode(images, qpos, actions, timestamps, gripper_states=None):
    """Validate and return arrays for one ACT episode without truncation."""
    image_arrays = (
        {str(name): np.asarray(value) for name, value in images.items()}
        if isinstance(images, dict) else {"camera_0": np.asarray(images)}
    )
    qpos_array = np.asarray(qpos, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32)
    timestamp_array = np.asarray(timestamps, dtype=np.float64)

    if not image_arrays:
        raise ValueError("至少需要一个相机图像流")
    for name, image_array in image_arrays.items():
        if not name or "/" in name:
            raise ValueError("camera name 必须是非空单级名称")
        if image_array.ndim != 4 or image_array.shape[-1] != 3:
            raise ValueError(f"{name} images shape 必须为 (T, H, W, 3)")
        if image_array.dtype != np.uint8:
            raise ValueError(f"{name} images dtype 必须为 uint8")
    if qpos_array.ndim != 2 or qpos_array.shape[1] != ARM_DOF:
        raise ValueError(f"qpos shape 必须为 (T, {ARM_DOF})")
    if action_array.ndim != 2 or action_array.shape[1] != ACTION_DIM:
        raise ValueError(f"action shape 必须为 (T, {ACTION_DIM})")
    if timestamp_array.ndim != 1:
        raise ValueError("timestamps shape 必须为 (T,)")

    lengths = {
        **{f"images/{name}": len(value) for name, value in image_arrays.items()},
        "qpos": len(qpos_array),
        "actions": len(action_array),
        "timestamps": len(timestamp_array),
    }
    gripper_array = None
    if gripper_states is not None:
        gripper_array = np.asarray(gripper_states, dtype=np.float32)
        if gripper_array.ndim != 2 or gripper_array.shape[1] != GRIPPER_STATE_DIM:
            raise ValueError(
                f"gripper_states shape 必须为 (T, {GRIPPER_STATE_DIM})"
            )
        lengths["gripper_states"] = len(gripper_array)
    if not next(iter(lengths.values())) or len(set(lengths.values())) != 1:
        raise ValueError(f"episode 各时间维长度必须相等且非零：{lengths}")
    if not np.isfinite(qpos_array).all():
        raise ValueError("qpos 包含 NaN 或 Inf")
    if not np.isfinite(action_array).all():
        raise ValueError("action 包含 NaN 或 Inf")
    if not np.isfinite(timestamp_array).all():
        raise ValueError("timestamps 包含 NaN 或 Inf")
    if gripper_array is not None and not np.isfinite(gripper_array).all():
        raise ValueError("gripper_states 包含 NaN 或 Inf")
    if len(timestamp_array) > 1 and not np.all(np.diff(timestamp_array) > 0):
        raise ValueError("timestamps 必须严格单调递增")
    validated_images = image_arrays if isinstance(images, dict) else image_arrays["camera_0"]
    return validated_images, qpos_array, action_array, timestamp_array, gripper_array


def save_episode(
    output_path,
    images,
    qpos,
    actions,
    timestamps,
    *,
    camera_name="camera_0",
    gripper_states=None,
    metadata=None,
    overwrite=False,
):
    """Validate and save one ACT-compatible HDF5 episode."""
    arrays = validate_episode(images, qpos, actions, timestamps, gripper_states)
    validated_images, qpos_array, action_array, timestamp_array, gripper_array = arrays
    image_arrays = (
        validated_images if isinstance(validated_images, dict)
        else {camera_name: validated_images}
    )
    camera_names = list(image_arrays)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with h5py.File(path, mode) as root:
        root.attrs["sim"] = False
        root.attrs["camera_name"] = camera_names[0]
        root.attrs["camera_names"] = json.dumps(camera_names)
        root.attrs["arm_qpos_dim"] = ARM_DOF
        root.attrs["action_dim"] = ACTION_DIM
        root.attrs["gripper_state_dim"] = GRIPPER_STATE_DIM
        root.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        for key, value in (metadata or {}).items():
            root.attrs[key] = (
                value
                if isinstance(value, (str, bytes, bool, int, float, np.number))
                else json.dumps(value, ensure_ascii=False)
            )
        root.create_dataset("action", data=action_array)
        root.create_dataset("timestamp", data=timestamp_array)
        observations = root.create_group("observations")
        observations.create_dataset("qpos", data=qpos_array)
        if gripper_array is not None:
            observations.create_dataset("gripper_state", data=gripper_array)
        image_group = observations.create_group("images")
        for name, image_array in image_arrays.items():
            image_group.create_dataset(
                name, data=image_array, chunks=(1, *image_array.shape[1:]), compression="lzf"
            )
    return path


def load_episode(path, *, camera_name=None, camera_names=None):
    """Load one episode into memory and return arrays plus root attributes."""
    with h5py.File(path, "r") as root:
        if camera_name is not None and camera_names is not None:
            raise ValueError("camera_name 与 camera_names 不能同时使用")
        available = list(root["observations/images"].keys())
        selected = list(camera_names) if camera_names is not None else [
            camera_name or str(root.attrs.get("camera_name", available[0]))
        ]
        missing = [name for name in selected if name not in available]
        if missing:
            raise ValueError(f"episode 缺少相机：{missing}；可用：{available}")
        loaded_images = {name: root[f"observations/images/{name}"][:] for name in selected}
        result = {
            "images": loaded_images if camera_names is not None else loaded_images[selected[0]],
            "qpos": root["observations/qpos"][:],
            "actions": root["action"][:],
            "timestamps": root["timestamp"][:],
            "gripper_states": (
                root["observations/gripper_state"][:]
                if "gripper_state" in root["observations"]
                else None
            ),
            "metadata": dict(root.attrs),
            "camera_name": selected[0],
            "camera_names": selected,
        }
    validate_episode(
        result["images"],
        result["qpos"],
        result["actions"],
        result["timestamps"],
        result["gripper_states"],
    )
    return result
