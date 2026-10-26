"""Strict offline validation and alignment analysis for ACT episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.act.episode_io import ACTION_DIM
from src.act.robot_wrapper import ARM_DOF, GRIPPER_STATE_DIM


NERO_JOINT_LIMITS_RAD = np.asarray(
    [
        [-2.705261, 2.705261],
        [-1.745330, 1.745330],
        [-2.757621, 2.757621],
        [-1.012291, 2.146755],
        [-2.757621, 2.757621],
        [-0.733039, 0.959932],
        [-1.570797, 1.570797],
    ], dtype=np.float32,
)
DEFAULT_MAX_ARM_STEP_RAD = 0.08


def alignment_errors(qpos: np.ndarray, actions: np.ndarray, max_lag: int = 3) -> dict[int, float]:
    """Return MAE for command ``action[t]`` versus ``qpos[t + lag]``."""
    result = {}
    arm_actions = actions[:, :ARM_DOF]
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = arm_actions[-lag:], qpos[:lag]
        elif lag > 0:
            left, right = arm_actions[:-lag], qpos[lag:]
        else:
            left, right = arm_actions, qpos
        result[lag] = float(np.mean(np.abs(left - right))) if len(left) else float("nan")
    return result


def validate_episode_file(path: Path, *, max_arm_step_rad: float = DEFAULT_MAX_ARM_STEP_RAD,
                          joint_limit_margin_rad: float = 0.0) -> dict:
    """Validate one HDF5 file without loading the complete image stream."""
    reasons = []
    required = ("action", "timestamp", "observations/qpos", "observations/images")
    try:
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
            gripper = root["observations/gripper_state"][:] if "gripper_state" in root["observations"] else None
            metadata = dict(root.attrs)
            count = len(actions)

            for name, values in image_sets.items():
                if values.ndim != 4 or len(values) != count or values.shape[-1] != 3:
                    reasons.append(f"{name} image shape 不是 (T,H,W,3)")
                if values.dtype != np.uint8:
                    reasons.append(f"{name} image dtype 不是 uint8")
            if qpos.dtype != np.float32 or actions.dtype != np.float32:
                reasons.append("qpos/action dtype 必须是 float32")
            if timestamps.dtype != np.float64:
                reasons.append("timestamp dtype 必须是 float64")
            if qpos.shape != (count, ARM_DOF):
                reasons.append(f"qpos shape 应为 ({count},{ARM_DOF})")
            if actions.shape != (count, ACTION_DIM):
                reasons.append(f"action shape 应为 ({count},{ACTION_DIM})")
            if timestamps.shape != (count,):
                reasons.append(f"timestamp shape 应为 ({count},)")
            if gripper is None:
                reasons.append("缺少 gripper_state")
            elif gripper.shape != (count, GRIPPER_STATE_DIM):
                reasons.append(f"gripper_state shape 应为 ({count},1)")
            elif gripper.dtype != np.float32:
                reasons.append("gripper_state dtype 必须是 float32")
            if not count:
                reasons.append("episode 为空")
            numeric = (qpos, actions, timestamps) + (() if gripper is None else (gripper,))
            finite_numeric = all(np.isfinite(array).all() for array in numeric)
            if not finite_numeric:
                reasons.append("数值包含 NaN 或 Inf")
            if qpos.shape == (count, ARM_DOF) and np.any(np.abs(qpos) > 2 * np.pi):
                reasons.append("qpos 超出通用 ±2π 极值范围")
            if actions.shape == (count, ACTION_DIM) and np.any((actions[:, -1] < 0) | (actions[:, -1] > 0.1)):
                reasons.append("夹爪 action 超出 0..0.1 m")
            if gripper is not None and gripper.shape == (count, 1) and np.any((gripper < 0) | (gripper > 0.1)):
                reasons.append("gripper_state 超出 0..0.1 m")

            dt = np.diff(timestamps) if timestamps.ndim == 1 else np.asarray([])
            dt_valid = bool(dt.size and np.isfinite(dt).all() and np.all(dt > 0))
            if dt.size and not dt_valid:
                reasons.append("timestamp 不是严格单调递增")
            requested_hz = float(metadata.get("control_frequency_hz", 0.0))
            late_cycles = int(np.count_nonzero(dt > 1.5 / requested_hz)) if requested_hz > 0 else 0
            arm_step = (np.max(np.abs(np.diff(actions[:, :ARM_DOF], axis=0)), axis=0)
                        if count > 1 and actions.shape == (count, ACTION_DIM) else np.zeros(ARM_DOF))
            if np.max(arm_step) > max_arm_step_rad:
                reasons.append(f"action 单步变化 {np.max(arm_step):.4f} rad 超过 {max_arm_step_rad:.4f} rad")

            if qpos.shape == (count, ARM_DOF) and actions.shape == (count, ACTION_DIM):
                if np.max(np.ptp(qpos, axis=0)) < 1e-6 and np.max(np.ptp(actions[:, :ARM_DOF], axis=0)) > 1e-3:
                    reasons.append("qpos 全程冻结但 action 明显运动")
                lower = NERO_JOINT_LIMITS_RAD[:, 0] - joint_limit_margin_rad
                upper = NERO_JOINT_LIMITS_RAD[:, 1] + joint_limit_margin_rad
                outside = np.any((actions[:, :ARM_DOF] < lower) | (actions[:, :ARM_DOF] > upper), axis=0)
                if np.any(outside):
                    joints = ",".join(f"J{i + 1}" for i in np.flatnonzero(outside))
                    reasons.append(f"action 超出 NERO 标称关节限位：{joints}")
                abs_errors = np.abs(actions[:, :ARM_DOF] - qpos)
                lag_errors = alignment_errors(qpos, actions)
                finite_lags = {key: value for key, value in lag_errors.items() if np.isfinite(value)}
                best_lag = min(finite_lags, key=finite_lags.get) if finite_lags else None
            else:
                abs_errors, lag_errors, best_lag = np.empty((0, ARM_DOF)), {}, None

            if count and images.ndim == 4 and images.shape[-1] == 3:
                indices = np.unique(np.linspace(0, count - 1, min(count, 10), dtype=int))
                image_stds = {
                    name: float(np.mean([np.std(values[index]) for index in indices]))
                    for name, values in image_sets.items()
                }
                for name, value in image_stds.items():
                    if value < 1.0:
                        reasons.append(f"{name} 抽样图像近似空白")
            else:
                image_stds = {name: 0.0 for name in camera_names}

            return {
                "path": str(path), "status": "valid" if not reasons else "rejected", "reasons": reasons,
                "task": str(metadata.get("task_name", "")), "num_steps": count,
                "position_group": str(metadata.get("position_group", "")),
                "duration_s": float(timestamps[-1] - timestamps[0]) if count > 1 and dt_valid else 0.0,
                "requested_hz": requested_hz,
                "actual_hz": float(1.0 / np.mean(dt)) if dt_valid else 0.0,
                "dt_mean_s": float(np.mean(dt)) if dt_valid else 0.0,
                "dt_std_s": float(np.std(dt)) if dt_valid else 0.0,
                "dt_p50_s": float(np.percentile(dt, 50)) if dt_valid else 0.0,
                "dt_p95_s": float(np.percentile(dt, 95)) if dt_valid else 0.0,
                "dt_max_s": float(np.max(dt)) if dt_valid else 0.0,
                "late_cycles": late_cycles, "max_arm_step_rad": float(np.max(arm_step)),
                "tracking_mae_rad": float(np.mean(abs_errors)) if abs_errors.size else None,
                "tracking_p95_rad": float(np.percentile(abs_errors, 95)) if abs_errors.size else None,
                "tracking_max_rad": float(np.max(abs_errors)) if abs_errors.size else None,
                "lag_mae_rad": {str(key): value for key, value in lag_errors.items()},
                "best_lag_steps": best_lag,
                "best_lag_ms": best_lag / requested_hz * 1000.0 if best_lag is not None and requested_hz > 0 else None,
                "gripper_target_events": int(np.count_nonzero(np.abs(np.diff(actions[:, -1])) > 1e-5))
                if count > 1 and actions.shape == (count, ACTION_DIM) else 0,
                "camera_names": camera_names,
                "sampled_image_std": image_stds[camera_names[0]],
                "sampled_image_stds": image_stds,
            }
    except (OSError, KeyError, ValueError) as exc:
        return {"path": str(path), "status": "rejected", "reasons": [str(exc)]}


def write_manifest(results: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": 1, "episodes": results}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-arm-step-rad", type=float, default=DEFAULT_MAX_ARM_STEP_RAD)
    parser.add_argument("--joint-limit-margin-rad", type=float, default=0.0)
    args = parser.parse_args()
    if args.max_arm_step_rad <= 0 or args.joint_limit_margin_rad < 0:
        parser.error("阈值必须为正，关节限位 margin 不得为负")
    results = [validate_episode_file(path, max_arm_step_rad=args.max_arm_step_rad,
                                     joint_limit_margin_rad=args.joint_limit_margin_rad)
               for path in args.episodes]
    for result in results:
        reason = "；".join(result["reasons"]) or "通过离线结构/数值/连续性检查"
        print(f"{result['status'].upper():8} {result['path']}：{reason}")
        if "best_lag_steps" in result:
            print(f"  T={result['num_steps']} Hz={result['actual_hz']:.3f} step_max={result['max_arm_step_rad']:.4f} rad "
                  f"MAE={result['tracking_mae_rad']:.6f} rad best_lag={result['best_lag_steps']}")
    if args.manifest:
        write_manifest(results, args.manifest)
        print(f"manifest：{args.manifest}")
    return 1 if any(result["status"] != "valid" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
