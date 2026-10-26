"""Inspect and validate one NERO ACT HDF5 demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from src.act.episode_io import ACTION_DIM
from src.act.robot_wrapper import ARM_DOF, GRIPPER_STATE_DIM


def inspect_episode(path: Path) -> dict:
    """Validate an episode without loading the image dataset into memory."""
    required = ["action", "timestamp", "observations/qpos", "observations/images"]
    with h5py.File(path, "r") as root:
        missing = [name for name in required if name not in root]
        if missing:
            raise ValueError(f"缺少 HDF5 数据集/组：{missing}")
        camera_names = list(root["observations/images"].keys())
        if not camera_names:
            raise ValueError("episode 中没有相机图像")
        image_sets = {name: root[f"observations/images/{name}"] for name in camera_names}
        images = image_sets[camera_names[0]]
        qpos = root["observations/qpos"][:]
        actions = root["action"][:]
        timestamps = root["timestamp"][:]
        gripper = (
            root["observations/gripper_state"][:]
            if "gripper_state" in root["observations"]
            else None
        )
        if len(images) == 0:
            raise ValueError("episode 不能为空")
        for name, values in image_sets.items():
            if values.ndim != 4 or values.shape[-1] != 3 or values.dtype != np.uint8:
                raise ValueError(f"{name} 图像必须为 uint8 (T, H, W, 3)")
            if len(values) != len(images):
                raise ValueError(f"{name} 图像帧数与其他数据不一致")
        if qpos.dtype != np.float32 or actions.dtype != np.float32:
            raise ValueError("qpos 和 action dtype 必须为 float32")
        if timestamps.dtype != np.float64:
            raise ValueError("timestamp dtype 必须为 float64")
        if qpos.shape != (len(images), ARM_DOF):
            raise ValueError(f"qpos shape 必须为 ({len(images)}, {ARM_DOF})")
        if actions.shape != (len(images), ACTION_DIM):
            raise ValueError(f"action shape 必须为 ({len(images)}, {ACTION_DIM})")
        if timestamps.shape != (len(images),):
            raise ValueError(f"timestamp shape 必须为 ({len(images)},)")
        if gripper is not None and gripper.shape != (len(images), GRIPPER_STATE_DIM):
            raise ValueError(
                f"gripper_state shape 必须为 ({len(images)}, {GRIPPER_STATE_DIM})"
            )
        if gripper is not None and gripper.dtype != np.float32:
            raise ValueError("gripper_state dtype 必须为 float32")
        numeric = [qpos, actions, timestamps]
        if gripper is not None:
            numeric.append(gripper)
        if not all(np.isfinite(value).all() for value in numeric):
            raise ValueError("episode 包含 NaN 或 Inf")
        qpos_span = np.ptp(qpos, axis=0)
        action_arm_span = np.ptp(actions[:, :ARM_DOF], axis=0)
        if np.max(qpos_span) < 1e-6 and np.max(action_arm_span) > 1e-3:
            raise ValueError(
                "qpos 全程冻结但 Leader action 明显运动；"
                "该 episode 使用了 Leader 模式下的普通关节缓存，不能用于训练"
            )
        dt = np.diff(timestamps)
        if len(dt) and not np.all(dt > 0):
            raise ValueError("timestamp 不是严格单调递增")
        frequency = float(1.0 / np.mean(dt)) if len(dt) else 0.0
        requested = float(root.attrs.get("control_frequency_hz", 0.0))
        result = {
            "metadata": dict(root.attrs),
            "camera_name": camera_names[0],
            "camera_names": camera_names,
            "image_shape": images.shape,
            "image_shapes": {name: values.shape for name, values in image_sets.items()},
            "qpos_shape": qpos.shape,
            "action_shape": actions.shape,
            "timestamp_shape": timestamps.shape,
            "gripper_shape": None if gripper is None else gripper.shape,
            "qpos_min": qpos.min(axis=0),
            "qpos_max": qpos.max(axis=0),
            "action_min": actions.min(axis=0),
            "action_max": actions.max(axis=0),
            "qpos_max_step": (
                np.max(np.abs(np.diff(qpos, axis=0)), axis=0)
                if len(qpos) > 1 else np.zeros(ARM_DOF)
            ),
            "action_max_step": (
                np.max(np.abs(np.diff(actions, axis=0)), axis=0)
                if len(actions) > 1 else np.zeros(ACTION_DIM)
            ),
            "gripper_target_events": int(
                np.count_nonzero(np.abs(np.diff(actions[:, -1])) > 1e-5)
            ),
            "duration_s": float(timestamps[-1] - timestamps[0]),
            "actual_frequency_hz": frequency,
            "mean_dt_s": float(np.mean(dt)) if len(dt) else 0.0,
            "p95_dt_s": float(np.percentile(dt, 95)) if len(dt) else 0.0,
            "max_dt_s": float(np.max(dt)) if len(dt) else 0.0,
            "late_cycles": int(np.sum(dt > 1.5 / requested)) if requested > 0 else 0,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    try:
        result = inspect_episode(args.episode)
    except (OSError, KeyError, ValueError) as exc:
        print(f"episode 检查失败：{exc}")
        return 1

    metadata = result["metadata"]
    print(f"episode：{args.episode}")
    print(f"sim：{metadata.get('sim')}")
    print(f"task_name：{metadata.get('task_name', '')}")
    print(f"requested frequency：{metadata.get('control_frequency_hz', '')} Hz")
    print(f"cameras：{result['image_shapes']}")
    print(f"qpos shape：{result['qpos_shape']}")
    print(f"action shape：{result['action_shape']}")
    print(f"timestamp shape：{result['timestamp_shape']}")
    print(f"gripper shape：{result['gripper_shape']}")
    print(f"qpos min/max：{result['qpos_min'].tolist()} / {result['qpos_max'].tolist()}")
    print(
        f"action min/max：{result['action_min'].tolist()} / "
        f"{result['action_max'].tolist()}"
    )
    print(f"qpos max step：{result['qpos_max_step'].tolist()}")
    print(f"action max step：{result['action_max_step'].tolist()}")
    print(f"gripper target events：{result['gripper_target_events']}")
    print(f"duration：{result['duration_s']:.3f} s")
    print(f"actual frequency：{result['actual_frequency_hz']:.3f} Hz")
    print(
        "dt mean/P95/max："
        f"{result['mean_dt_s']:.6f} / {result['p95_dt_s']:.6f} / "
        f"{result['max_dt_s']:.6f} s"
    )
    print(f"late cycles：{result['late_cycles']}")
    print("NaN/Inf：none；timestamp：strictly increasing；lengths：consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
