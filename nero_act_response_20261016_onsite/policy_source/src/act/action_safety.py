"""Offline-safe projection for denormalized NERO ACT action chunks."""

from __future__ import annotations

import numpy as np

from src.act.validate_episode import DEFAULT_MAX_ARM_STEP_RAD, NERO_JOINT_LIMITS_RAD


def sanitize_action_chunk(actions, *, gripper_open_m: float = 0.095,
                          gripper_threshold_m: float = 0.0475,
                          max_joint_step_rad: float = DEFAULT_MAX_ARM_STEP_RAD):
    """Reject unsafe joints and map the demonstrated binary gripper target to 0/95 mm."""
    values = np.asarray(actions, dtype=np.float32).copy()
    if values.ndim < 2 or values.shape[-1] != 8 or not np.isfinite(values).all():
        raise ValueError("action chunk 必须为有限值 [...,T,8]")
    if not 0 < gripper_threshold_m < gripper_open_m <= 0.1 or max_joint_step_rad <= 0:
        raise ValueError("夹爪阈值/开口和关节步长阈值配置无效")
    lower, upper = NERO_JOINT_LIMITS_RAD[:, 0], NERO_JOINT_LIMITS_RAD[:, 1]
    if np.any((values[..., :7] < lower) | (values[..., :7] > upper)):
        raise ValueError("预测关节目标超出 NERO 标称限位")
    max_step = float(np.abs(np.diff(values[..., :7], axis=-2)).max()) if values.shape[-2] > 1 else 0.0
    if max_step > max_joint_step_rad:
        raise ValueError(f"预测关节单步变化 {max_step:.4f} rad 超过 {max_joint_step_rad:.4f} rad")
    values[..., 7] = np.where(values[..., 7] >= gripper_threshold_m, gripper_open_m, 0.0)
    return values, {"max_joint_step_rad": max_step, "gripper_mode": "binary"}
