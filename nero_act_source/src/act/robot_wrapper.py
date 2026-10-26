"""Small, non-blocking NERO SDK adapter for joint-space ACT control."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from src.utils.arm import create_nero_arm, enable_arm


ARM_DOF = 7
GRIPPER_STATE_DIM = 1
DEFAULT_FEEDBACK_TIMEOUT_S = 0.15
DEFAULT_POLL_INTERVAL_S = 0.01
DEFAULT_CONNECT_SETTLE_S = 1.0
DEFAULT_GRIPPER_OPEN_M = 0.095
DEFAULT_GRIPPER_CLOSE_M = 0.0
DEFAULT_GRIPPER_FORCE_N = 1.0
DEFAULT_SPEED_SETTLE_S = 0.1
MAX_GRIPPER_WIDTH_M = 0.1
MAX_GRIPPER_FORCE_N = 3.0
EMERGENCY_STOP_STATUS = 0x01


class NeroRobotWrapper:
    """Expose NERO arm joints and gripper through an ACT-friendly interface.

    Arm positions are seven-element arrays in radians. Gripper state is a
    one-element array containing the measured opening width in metres. The
    gripper is initialized before ``robot.connect()`` as required by the SDK.
    """

    def __init__(
        self,
        robot=None,
        *,
        feedback_timeout_s: float = DEFAULT_FEEDBACK_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        connect_settle_s: float = DEFAULT_CONNECT_SETTLE_S,
        enable_timeout_s: float = 8.0,
        enable_settle_s: float = 0.3,
    ) -> None:
        if feedback_timeout_s <= 0 or poll_interval_s <= 0:
            raise ValueError("feedback_timeout_s 和 poll_interval_s 必须大于 0")
        if connect_settle_s < 0:
            raise ValueError("connect_settle_s 不得为负")
        self.robot = robot if robot is not None else create_nero_arm()
        self.gripper = self.robot.init_effector(
            self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER
        )
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.connect_settle_s = float(connect_settle_s)
        self.enable_timeout_s = float(enable_timeout_s)
        self.enable_settle_s = float(enable_settle_s)
        self.connected = False

    def connect(self) -> np.ndarray:
        """Connect, enable the arm, enter follower mode, and return initial qpos."""
        if self.connected:
            return self.get_arm_qpos()
        self.robot.connect()
        self.connected = True
        try:
            time.sleep(self.connect_settle_s)
            if self.robot.has_comm_error():
                raise RuntimeError("NERO CAN 通信异常")
            status = self.robot.get_arm_status()
            if status is not None and status.msg.arm_status == EMERGENCY_STOP_STATUS:
                raise RuntimeError(
                    "NERO 处于 EMERGENCY_STOP；请先释放实体急停并运行 "
                    "python -m src.arm.reset"
                )
            if not enable_arm(
                self.robot,
                timeout=self.enable_timeout_s,
                settle_duration=self.enable_settle_s,
            ):
                status = self.robot.get_arm_status()
                status_code = None if status is None else status.msg.arm_status
                joints = self.robot.get_joint_angles()
                raise RuntimeError(
                    "NERO 机械臂使能失败："
                    f"arm_status={status_code!r}, "
                    f"joint_feedback={'yes' if joints is not None else 'no'}"
                )
            self.robot.set_follower_mode()
            return self.get_arm_qpos()
        except Exception:
            self.disconnect()
            raise

    def get_arm_qpos(self, timeout_s: float | None = None) -> np.ndarray:
        """Return current arm qpos as ``float32[7]`` in radians."""
        self._require_connected()
        timeout = self.feedback_timeout_s if timeout_s is None else float(timeout_s)
        if timeout <= 0:
            raise ValueError("timeout_s 必须大于 0")
        deadline = time.monotonic() + timeout
        last_error: ValueError | None = None
        while time.monotonic() < deadline:
            feedback = self.robot.get_joint_angles()
            if feedback is not None:
                try:
                    return self._validate_arm_qpos(feedback.msg, "关节反馈")
                except ValueError as exc:
                    last_error = exc
            time.sleep(self.poll_interval_s)
        if last_error is not None:
            raise last_error
        raise TimeoutError(f"{timeout:.3f} s 内没有收到 NERO 关节角反馈")

    def send_arm_qpos(self, q_target: Sequence[float] | np.ndarray) -> None:
        """Send one non-blocking seven-joint target in radians via ``move_j``."""
        self._require_connected()
        target = self._validate_arm_qpos(q_target, "关节目标")
        self.robot.move_j(target.tolist())

    def set_speed_percent(
        self, speed_percent: int, settle_s: float = DEFAULT_SPEED_SETTLE_S
    ) -> None:
        """Set controller speed and wait for its mode-control frame to settle."""
        self._require_connected()
        if isinstance(speed_percent, bool) or not isinstance(speed_percent, int):
            raise ValueError("speed_percent 必须是整数")
        if not 1 <= speed_percent <= 100:
            raise ValueError("speed_percent 必须在 1..100 范围内")
        if settle_s < 0:
            raise ValueError("settle_s 不得为负")
        self.robot.set_speed_percent(speed_percent)
        time.sleep(settle_s)

    def enter_leader_mode(self) -> None:
        """Enter the SDK's zero-force drag mode for kinesthetic teaching."""
        self._require_connected()
        self.robot.set_leader_mode()

    def enter_follower_mode(self) -> None:
        """Return to the SDK's enabled controlled mode."""
        self._require_connected()
        self.robot.set_follower_mode()

    def get_leader_qpos(self, timeout_s: float | None = None) -> np.ndarray:
        """Return zero-force Leader joint feedback as ``float32[7]`` in radians."""
        self._require_connected()
        timeout = self.feedback_timeout_s if timeout_s is None else float(timeout_s)
        if timeout <= 0:
            raise ValueError("timeout_s 必须大于 0")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            feedback = self.robot.get_leader_joint_angles()
            if feedback is not None:
                return self._validate_arm_qpos(feedback.msg, "Leader 关节反馈")
            time.sleep(self.poll_interval_s)
        raise TimeoutError(f"{timeout:.3f} s 内没有收到 Leader 关节反馈")

    def get_gripper_state(self, timeout_s: float | None = None) -> np.ndarray:
        """Return measured gripper opening as ``float32[1]`` in metres."""
        self._require_connected()
        timeout = self.feedback_timeout_s if timeout_s is None else float(timeout_s)
        if timeout <= 0:
            raise ValueError("timeout_s 必须大于 0")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            feedback = self.gripper.get_gripper_status()
            if feedback is not None:
                if feedback.msg.mode != "width":
                    raise RuntimeError("夹爪当前不是宽度控制模式，不能作为 ACT 宽度状态")
                width = np.asarray([feedback.msg.value], dtype=np.float32)
                if np.isfinite(width).all():
                    return width
                raise ValueError("夹爪反馈包含 NaN 或 Inf")
            time.sleep(self.poll_interval_s)
        raise TimeoutError(f"{timeout:.3f} s 内没有收到夹爪状态反馈")

    def open_gripper(
        self,
        width_m: float = DEFAULT_GRIPPER_OPEN_M,
        force_n: float = DEFAULT_GRIPPER_FORCE_N,
    ) -> None:
        """Command the gripper to open; width is metres and force is newtons."""
        self._send_gripper(width_m, force_n)

    def close_gripper(
        self,
        width_m: float = DEFAULT_GRIPPER_CLOSE_M,
        force_n: float = DEFAULT_GRIPPER_FORCE_N,
    ) -> None:
        """Command the gripper to close; width is metres and force is newtons."""
        self._send_gripper(width_m, force_n)

    def emergency_stop(self) -> None:
        """Trigger the SDK's damped electronic emergency stop."""
        self._require_connected()
        self.robot.electronic_emergency_stop()

    def hold_current_position(self, settle_s: float = 0.2) -> np.ndarray:
        """Stop changing targets and command the measured joint pose once."""
        self._require_connected()
        if settle_s < 0:
            raise ValueError("settle_s 不得为负")
        current = self.get_arm_qpos()
        self.robot.set_follower_mode()
        self.robot.move_j(current.tolist())
        time.sleep(settle_s)
        return current

    def disconnect(self) -> None:
        """Release CAN resources without disabling joints or stopping motion."""
        try:
            self.robot.disconnect()
        finally:
            self.connected = False

    def _send_gripper(self, width_m: float, force_n: float) -> None:
        self._require_connected()
        width, force = float(width_m), float(force_n)
        if not np.isfinite([width, force]).all():
            raise ValueError("夹爪宽度和力必须是有限数值")
        if not 0.0 <= width <= MAX_GRIPPER_WIDTH_M:
            raise ValueError("夹爪宽度必须在 0..0.1 m 范围内")
        if not 0.0 < force <= MAX_GRIPPER_FORCE_N:
            raise ValueError("夹爪力必须在 (0, 3] N 范围内")
        self.gripper.move_gripper_m(value=width, force=force)

    @staticmethod
    def _validate_arm_qpos(values, label: str) -> np.ndarray:
        try:
            qpos = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}无法转换为 float32") from exc
        if qpos.shape != (ARM_DOF,):
            raise ValueError(f"{label}形状必须为 ({ARM_DOF},)，实际为 {qpos.shape}")
        if not np.isfinite(qpos).all():
            raise ValueError(f"{label}包含 NaN 或 Inf")
        return qpos

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("NERO 尚未连接，请先调用 connect()")
