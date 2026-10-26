"""Stateful safety filter for one denormalized NERO ACT target."""

from __future__ import annotations

import numpy as np


class SafetyFilter:
    def __init__(self, config: dict, train_action_min, train_action_max,
                 train_qpos_min=None, train_qpos_max=None):
        self.joint_limits = np.asarray(config["joint_limits_rad"], dtype=np.float32)
        self.train_min = np.asarray(train_action_min, dtype=np.float32)
        self.train_max = np.asarray(train_action_max, dtype=np.float32)
        if self.joint_limits.shape != (7, 2) or self.train_min.shape != (8,) or self.train_max.shape != (8,):
            raise ValueError("safety limits 必须为 (7,2)，train action range 必须为 (8,)")
        self.train_qpos_min = np.asarray(
            self.train_min[:7] if train_qpos_min is None else train_qpos_min, dtype=np.float32
        )
        self.train_qpos_max = np.asarray(
            self.train_max[:7] if train_qpos_max is None else train_qpos_max, dtype=np.float32
        )
        if self.train_qpos_min.shape != (7,) or self.train_qpos_max.shape != (7,):
            raise ValueError("train qpos range 必须为 (7,)")
        arrays = (self.joint_limits, self.train_min, self.train_max,
                  self.train_qpos_min, self.train_qpos_max)
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError("safety ranges 必须全部为有限值")
        if (np.any(self.joint_limits[:, 0] >= self.joint_limits[:, 1])
                or np.any(self.train_min > self.train_max)
                or np.any(self.train_qpos_min > self.train_qpos_max)):
            raise ValueError("safety range 的 min 必须小于等于 max")
        self.max_delta = float(config["max_delta_rad"])
        self.max_velocity = float(config["max_velocity_rad_s"])
        self.max_command_lead = float(config["max_command_lead_rad"])
        self.frequency = float(config["control_frequency_hz"])
        self.range_margin = float(config["train_range_margin"])
        self.severe_margin = float(config["severe_limit_margin_rad"])
        self.gripper_open = float(config["gripper_open_m"])
        self.gripper_close_threshold = float(config["gripper_close_threshold_m"])
        self.gripper_open_threshold = float(config["gripper_open_threshold_m"])
        if not all(value > 0 for value in (
            self.max_delta, self.max_velocity, self.max_command_lead, self.frequency
        )):
            raise ValueError("delta、velocity 和 control frequency 必须大于 0")
        if self.range_margin < 0 or self.severe_margin < 0:
            raise ValueError("safety margin 不得为负")
        if not 0 <= self.gripper_close_threshold < self.gripper_open_threshold <= self.gripper_open <= 0.1:
            raise ValueError("夹爪迟滞参数必须满足 0 <= close < open threshold <= open <= 0.1")
        self.gripper_is_open: bool | None = None

    @staticmethod
    def _state(values, label):
        state = np.asarray(values, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"{label}必须是有限 float32[8]")
        return state

    def validate_initial_state(self, state, margin: float) -> None:
        current = self._state(state, "初始 state")
        if margin < 0:
            raise ValueError("initial state margin 不得为负")
        lower, upper = self.train_qpos_min - margin, self.train_qpos_max + margin
        outside = np.flatnonzero((current[:7] < lower) | (current[:7] > upper))
        if len(outside):
            details = ", ".join(
                f"J{index + 1}={current[index]:.4f}, allowed=[{lower[index]:.4f},{upper[index]:.4f}]"
                for index in outside
            )
            raise ValueError(f"初始 qpos 超出 demonstration 分布允许范围：{details}")
        self.gripper_is_open = bool(current[7] >= (self.gripper_close_threshold + self.gripper_open_threshold) / 2)

    def filter(self, target, current_state, *, speed_scale: float = 1.0,
               command_reference=None):
        raw = self._state(target, "ACT target")
        current = self._state(current_state, "当前 state")
        reference = current if command_reference is None else self._state(command_reference, "上一安全目标")
        if not 0 < speed_scale <= 1:
            raise ValueError("speed_scale 必须在 (0,1] 内")
        lower, upper = self.joint_limits[:, 0], self.joint_limits[:, 1]
        if np.any((raw[:7] < lower - self.severe_margin) | (raw[:7] > upper + self.severe_margin)):
            raise ValueError("ACT target 严重超出关节限位")
        safe = raw.copy()
        before_limits = safe[:7].copy()
        safe[:7] = np.clip(safe[:7], lower, upper)
        limit_clamps = int(np.count_nonzero(before_limits != safe[:7]))
        range_anomalies = int(np.count_nonzero(
            (raw[:7] < self.train_min[:7] - self.range_margin)
            | (raw[:7] > self.train_max[:7] + self.range_margin)
        ))
        allowed_delta = min(self.max_delta, self.max_velocity / self.frequency) * speed_scale
        requested_delta = safe[:7] - reference[:7]
        clipped_delta = np.clip(requested_delta, -allowed_delta, allowed_delta)
        delta_clamps = int(np.count_nonzero(requested_delta != clipped_delta))
        safe[:7] = reference[:7] + clipped_delta
        before_lead = safe[:7].copy()
        safe[:7] = np.clip(
            safe[:7], current[:7] - self.max_command_lead, current[:7] + self.max_command_lead
        )
        command_lead_clamps = int(np.count_nonzero(before_lead != safe[:7]))

        if self.gripper_is_open is None:
            self.gripper_is_open = bool(current[7] >= (self.gripper_close_threshold + self.gripper_open_threshold) / 2)
        if raw[7] <= self.gripper_close_threshold:
            self.gripper_is_open = False
        elif raw[7] >= self.gripper_open_threshold:
            self.gripper_is_open = True
        safe[7] = self.gripper_open if self.gripper_is_open else 0.0
        return safe, {
            "joint_limit_clamps": limit_clamps,
            "delta_clamps": delta_clamps,
            "command_lead_clamps": command_lead_clamps,
            "range_anomalies": range_anomalies,
            "allowed_delta_rad": allowed_delta,
            "gripper_command_m": float(safe[7]),
        }
