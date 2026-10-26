"""Small target-aware adaptive MPC for ACT joint references."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


class AdaptiveMPC:
    """Track an ACT chunk while biasing its terminal TCP toward the task target."""

    def __init__(self, config: dict, joint_limits) -> None:
        self.horizon = int(config.get("horizon", 5))
        self.alpha = np.full(7, float(config.get("response_alpha", 0.35)))
        self.adaptation_rate = float(config.get("adaptation_rate", 0.1))
        self.alpha_min = float(config.get("response_alpha_min", 0.05))
        self.alpha_max = float(config.get("response_alpha_max", 1.0))
        self.min_excitation = float(config.get("min_excitation_rad", 0.002))
        self.act_weight = float(config.get("act_weight", 1.0))
        self.tcp_weight_near = float(config.get("tcp_weight_near", 4.0))
        self.tcp_weight_far = float(config.get("tcp_weight_far", 0.5))
        self.tcp_weight_distance_mm = float(config.get("tcp_weight_distance_mm", 150.0))
        self.activation_distance_mm = float(config.get("activation_distance_mm", 150.0))
        self.full_activation_distance_mm = float(
            config.get("full_activation_distance_mm", 50.0)
        )
        self.max_correction_rad = float(config.get("max_correction_rad", 0.005))
        self.smooth_weight = float(config.get("smooth_weight", 0.2))
        self.effort_weight = float(config.get("effort_weight", 0.05))
        self.jacobian_epsilon_rad = float(config.get("jacobian_epsilon_rad", 1e-3))
        self.jacobian_refresh_steps = int(config.get("jacobian_refresh_steps", 5))
        self.jacobian_recompute_rad = float(config.get("jacobian_recompute_rad", 0.08))
        self.max_target_step_rad = float(config.get("max_target_step_rad", 0.05))
        self.solve_budget_s = float(config.get("solve_budget_ms", 8.0)) / 1000.0
        self.joint_limits = np.asarray(joint_limits, dtype=np.float64)
        if self.joint_limits.shape != (7, 2) or not np.isfinite(self.joint_limits).all():
            raise ValueError("adaptive MPC joint_limits 必须是有限 (7,2)")
        positive = (
            self.horizon, self.adaptation_rate, self.alpha_min, self.alpha_max,
            self.min_excitation, self.act_weight, self.tcp_weight_distance_mm,
            self.activation_distance_mm, self.max_correction_rad,
            self.jacobian_epsilon_rad, self.jacobian_refresh_steps,
            self.jacobian_recompute_rad, self.max_target_step_rad, self.solve_budget_s,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("adaptive MPC 的 horizon、权重、步长和时间预算必须为正")
        if not 0 < self.alpha_min <= self.alpha_max <= 1 or not (
            self.alpha_min <= self.alpha[0] <= self.alpha_max
        ):
            raise ValueError("adaptive MPC response alpha 必须满足 0 < min <= initial <= max <= 1")
        if min(self.tcp_weight_near, self.tcp_weight_far,
               self.smooth_weight, self.effort_weight) < 0:
            raise ValueError("adaptive MPC 权重不得为负")
        if not 0 <= self.full_activation_distance_mm < self.activation_distance_mm:
            raise ValueError("adaptive MPC 完全介入距离必须小于启用距离且不得为负")
        self._last_state = self._last_command = None
        self._jacobian = self._jacobian_q = None
        self._plan_count = 0

    @staticmethod
    def _vector(values, size: int, label: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (size,) or not np.isfinite(result).all():
            raise ValueError(f"{label}必须是有限 float[{size}]")
        return result

    def observe(self, current_qpos) -> None:
        """Update the diagonal command-response model from one sent command."""
        current = self._vector(current_qpos, 7, "adaptive MPC qpos")
        if self._last_state is None:
            return
        requested = self._last_command - self._last_state
        observed = current - self._last_state
        mask = np.abs(requested) >= self.min_excitation
        ratio = np.divide(observed, requested, out=np.zeros(7), where=mask)
        mask &= np.isfinite(ratio) & (ratio > 0)
        if np.any(mask):
            ratio = np.clip(ratio, self.alpha_min, self.alpha_max)
            self.alpha[mask] = (
                (1.0 - self.adaptation_rate) * self.alpha[mask]
                + self.adaptation_rate * ratio[mask]
            )
        self._last_state = self._last_command = None

    def record_command(self, current_qpos, command_qpos) -> None:
        self._last_state = self._vector(current_qpos, 7, "adaptive MPC command state").copy()
        self._last_command = self._vector(command_qpos, 7, "adaptive MPC command").copy()

    def _tcp_jacobian(
        self, qpos: np.ndarray, tcp_xyz_m: np.ndarray,
        tcp_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        refresh = (
            self._jacobian is None
            or self._plan_count % self.jacobian_refresh_steps == 0
            or np.linalg.norm(qpos - self._jacobian_q, ord=np.inf)
            >= self.jacobian_recompute_rad
        )
        if refresh:
            jacobian = np.empty((3, 7), dtype=np.float64)
            for joint in range(7):
                shifted = qpos.copy()
                delta = self.jacobian_epsilon_rad
                if shifted[joint] + delta > self.joint_limits[joint, 1]:
                    delta = -delta
                shifted[joint] += delta
                shifted_tcp = self._vector(tcp_fn(shifted), 3, "adaptive MPC TCP")
                jacobian[:, joint] = (
                    shifted_tcp / 1000.0 - tcp_xyz_m
                ) / delta
            if not np.isfinite(jacobian).all():
                raise ValueError("adaptive MPC Jacobian 包含 NaN 或 Inf")
            self._jacobian, self._jacobian_q = jacobian, qpos.copy()
        self._plan_count += 1
        return self._jacobian

    def _prediction_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.horizon * 7, self.horizon * 7), dtype=np.float64)
        for step in range(self.horizon):
            for command in range(step + 1):
                coefficient = self.alpha * np.power(1.0 - self.alpha, step - command)
                rows = slice(step * 7, (step + 1) * 7)
                cols = slice(command * 7, (command + 1) * 7)
                matrix[rows, cols] = np.diag(coefficient)
        return matrix

    def plan(
        self, current_qpos, act_chunk, target_xyz_mm,
        tcp_fn: Callable[[np.ndarray], np.ndarray], *, previous_command=None,
    ) -> tuple[np.ndarray, dict]:
        """Return the first receding-horizon joint target and diagnostics."""
        started = time.perf_counter()
        current = self._vector(current_qpos, 7, "adaptive MPC current qpos")
        target_m = self._vector(target_xyz_mm, 3, "adaptive MPC target") / 1000.0
        references = np.asarray(act_chunk, dtype=np.float64)
        if references.ndim != 2 or references.shape[1] != 7 or not np.isfinite(references).all():
            raise ValueError("adaptive MPC ACT chunk 必须是有限 [T,7]")
        if not len(references):
            raise ValueError("adaptive MPC ACT chunk 不得为空")
        references = references[:self.horizon]
        if len(references) < self.horizon:
            references = np.vstack([
                references,
                np.repeat(references[-1][None], self.horizon - len(references), axis=0),
            ])
        previous = current if previous_command is None else self._vector(
            previous_command, 7, "adaptive MPC previous command"
        )
        tcp_m = self._vector(tcp_fn(current), 3, "adaptive MPC current TCP") / 1000.0
        distance_mm = float(np.linalg.norm(target_m - tcp_m) * 1000.0)
        blend = float(np.clip(
            (self.activation_distance_mm - distance_mm)
            / (self.activation_distance_mm - self.full_activation_distance_mm),
            0.0, 1.0,
        ))
        active = blend > 0.0
        if not active:
            latency = time.perf_counter() - started
            return references[0].astype(np.float32), {
                "active": False, "blend": 0.0, "correction_norm_rad": 0.0,
                "fallback": "", "solve_latency_s": latency,
                "tcp_distance_mm": distance_mm,
                "predicted_terminal_error_mm": distance_mm,
                "response_alpha": self.alpha.astype(float).tolist(),
                "target_xyz_mm": (target_m * 1000.0).astype(float).tolist(),
            }
        try:
            jacobian = self._tcp_jacobian(current, tcp_m, tcp_fn)
            prediction = self._prediction_matrix()
            reference_delta = (references - current).reshape(-1)
            distance_ratio = min(1.0, distance_mm / self.tcp_weight_distance_mm)
            tcp_weight = (
                self.tcp_weight_near
                + (self.tcp_weight_far - self.tcp_weight_near) * distance_ratio
            )
            smooth = np.zeros_like(prediction)
            smooth_target = np.zeros(self.horizon * 7, dtype=np.float64)
            smooth[:7, :7] = np.eye(7)
            smooth_target[:7] = previous - current
            for step in range(1, self.horizon):
                rows = slice(step * 7, (step + 1) * 7)
                smooth[rows, rows] = np.eye(7)
                smooth[rows, slice((step - 1) * 7, step * 7)] = -np.eye(7)
            tcp_prediction = np.kron(np.eye(self.horizon), jacobian) @ prediction
            system = np.vstack([
                np.sqrt(self.act_weight) * prediction,
                np.sqrt(tcp_weight) * tcp_prediction,
                np.sqrt(self.smooth_weight) * smooth,
                np.sqrt(self.effort_weight) * np.eye(self.horizon * 7),
            ])
            objective = np.concatenate([
                np.sqrt(self.act_weight) * reference_delta,
                np.sqrt(tcp_weight) * np.tile(target_m - tcp_m, self.horizon),
                np.sqrt(self.smooth_weight) * smooth_target,
                np.zeros(self.horizon * 7),
            ])
            command_delta = np.linalg.lstsq(system, objective, rcond=None)[0].reshape(
                self.horizon, 7
            )
            commands = current + command_delta
            for step in range(self.horizon):
                base = current if step == 0 else commands[step - 1]
                commands[step] = np.clip(
                    commands[step], base - self.max_target_step_rad,
                    base + self.max_target_step_rad,
                )
                commands[step] = np.clip(
                    commands[step], self.joint_limits[:, 0], self.joint_limits[:, 1]
                )
            corrections = np.clip(
                commands - references, -self.max_correction_rad, self.max_correction_rad
            )
            commands = np.clip(
                references + blend * corrections,
                self.joint_limits[:, 0], self.joint_limits[:, 1],
            )
            predicted = current.copy()
            for command in commands:
                predicted += self.alpha * (command - predicted)
            terminal_error_mm = float(np.linalg.norm(
                target_m - (tcp_m + jacobian @ (predicted - current))
            ) * 1000.0)
            result = commands[0]
            correction_norm = float(np.linalg.norm(result - references[0]))
            fallback = ""
        except (np.linalg.LinAlgError, ValueError) as exc:
            result = current.copy()
            terminal_error_mm = distance_mm
            correction_norm = 0.0
            fallback = str(exc)
        latency = time.perf_counter() - started
        if latency > self.solve_budget_s:
            result = current.copy()
            terminal_error_mm = distance_mm
            correction_norm = 0.0
            fallback = f"solve budget exceeded: {latency * 1000.0:.2f} ms"
        return result.astype(np.float32), {
            "active": active,
            "blend": blend,
            "correction_norm_rad": correction_norm,
            "fallback": fallback,
            "solve_latency_s": latency,
            "tcp_distance_mm": distance_mm,
            "predicted_terminal_error_mm": terminal_error_mm,
            "response_alpha": self.alpha.astype(float).tolist(),
            "target_xyz_mm": (target_m * 1000.0).astype(float).tolist(),
        }
