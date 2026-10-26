import unittest
from unittest.mock import patch

import numpy as np

from src.act.robot_wrapper import NeroRobotWrapper
from src.act.test_joint_stream import parse_args, return_to_start


class Feedback:
    def __init__(self, msg):
        self.msg = msg


class ArmStatusMessage:
    arm_status = 0


class FakeGripper:
    def __init__(self):
        self.statuses = [
            Feedback(type("GripperStatus", (), {"value": 0.042, "mode": "width"})())
        ]
        self.commands = []

    def get_gripper_status(self):
        return self.statuses.pop(0) if self.statuses else None

    def move_gripper_m(self, *, value, force):
        self.commands.append((value, force))


class FakeOptions:
    class EFFECTOR:
        AGX_GRIPPER = "agx_gripper"


class FakeRobot:
    OPTIONS = FakeOptions

    def __init__(self):
        self.events = []
        self.gripper = FakeGripper()
        self.feedback = Feedback([0.1] * 7)
        self.leader_feedback = Feedback([0.2] * 7)
        self.targets = []
        self.status_code = 0
        self.enable_calls = 0

    def init_effector(self, name):
        self.events.append(("init_effector", name))
        return self.gripper

    def connect(self):
        self.events.append("connect")

    def enable(self):
        self.enable_calls += 1
        return True

    def has_comm_error(self):
        return False

    def get_arm_status(self):
        message = ArmStatusMessage()
        message.arm_status = self.status_code
        return Feedback(message)

    def set_follower_mode(self):
        self.events.append("follower")

    def set_leader_mode(self):
        self.events.append("leader")

    def get_joint_angles(self):
        return self.feedback

    def get_leader_joint_angles(self):
        return self.leader_feedback

    def move_j(self, target):
        self.targets.append(target)

    def set_speed_percent(self, value):
        self.events.append(("speed", value))

    def electronic_emergency_stop(self):
        self.events.append("emergency_stop")

    def disable(self):
        self.events.append("disable")

    def disconnect(self):
        self.events.append("disconnect")


def connected_wrapper(robot=None):
    robot = robot or FakeRobot()
    wrapper = NeroRobotWrapper(
        robot,
        feedback_timeout_s=0.01,
        poll_interval_s=0.001,
        connect_settle_s=0.0,
        enable_timeout_s=0.02,
        enable_settle_s=0.0,
    )
    wrapper.connect()
    return wrapper, robot


class NeroRobotWrapperTest(unittest.TestCase):
    def test_joint_stream_speed_percent_is_bounded(self):
        with patch("sys.argv", ["test_joint_stream", "--speed-percent", "10"]):
            self.assertEqual(parse_args().speed_percent, 10)
        with patch("sys.argv", ["test_joint_stream", "--speed-percent", "0"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_emergency_stop_is_reported_before_enable(self):
        robot = FakeRobot()
        robot.status_code = 0x01
        wrapper = NeroRobotWrapper(robot, connect_settle_s=0.0)

        with self.assertRaisesRegex(RuntimeError, "EMERGENCY_STOP.*src.arm.reset"):
            wrapper.connect()

        self.assertEqual(robot.enable_calls, 0)
        self.assertIn("disconnect", robot.events)

    def test_connect_waits_for_can_initialization_before_enable(self):
        robot = FakeRobot()
        wrapper = NeroRobotWrapper(
            robot,
            connect_settle_s=1.0,
            enable_settle_s=0.0,
        )

        with patch("src.act.robot_wrapper.time.sleep") as sleep:
            wrapper.connect()

        sleep.assert_called_once_with(1.0)
        self.assertTrue(wrapper.connected)

    def test_connect_accepts_stale_recoverable_planning_status(self):
        robot = FakeRobot()
        robot.status_code = 0x02

        wrapper, _ = connected_wrapper(robot)

        self.assertTrue(wrapper.connected)
        self.assertIn("follower", robot.events)

    def test_initializes_gripper_before_connect_and_reads_qpos(self):
        wrapper, robot = connected_wrapper()
        robot.feedback = Feedback(range(7))

        qpos = wrapper.get_arm_qpos()

        self.assertEqual(
            robot.events[:2],
            [("init_effector", "agx_gripper"), "connect"],
        )
        self.assertEqual(qpos.dtype, np.float32)
        self.assertEqual(qpos.shape, (7,))
        np.testing.assert_array_equal(qpos, np.arange(7, dtype=np.float32))

    def test_missing_feedback_times_out_clearly(self):
        wrapper, robot = connected_wrapper()
        robot.feedback = None

        with self.assertRaisesRegex(TimeoutError, "没有收到.*关节角反馈"):
            wrapper.get_arm_qpos(timeout_s=0.003)

    def test_bad_feedback_shape_is_rejected(self):
        wrapper, robot = connected_wrapper()
        robot.feedback = Feedback([0.0] * 6)

        with self.assertRaisesRegex(ValueError, "形状必须"):
            wrapper.get_arm_qpos(timeout_s=0.003)

    def test_send_qpos_validates_shape_and_finite_values(self):
        wrapper, robot = connected_wrapper()

        with self.assertRaisesRegex(ValueError, "形状必须"):
            wrapper.send_arm_qpos([0.0] * 6)
        with self.assertRaisesRegex(ValueError, "NaN"):
            wrapper.send_arm_qpos([0.0] * 6 + [float("nan")])

        wrapper.send_arm_qpos(np.arange(7, dtype=np.float64))
        self.assertEqual(robot.targets[-1], [float(value) for value in range(7)])

    def test_speed_setting_waits_for_controller_mode_frame(self):
        wrapper, robot = connected_wrapper()

        with patch("src.act.robot_wrapper.time.sleep") as sleep:
            wrapper.set_speed_percent(10)

        self.assertIn(("speed", 10), robot.events)
        sleep.assert_called_once_with(0.1)

    def test_gripper_state_and_commands(self):
        wrapper, robot = connected_wrapper()

        state = wrapper.get_gripper_state()
        wrapper.open_gripper()
        wrapper.close_gripper(width_m=0.01, force_n=0.5)

        self.assertEqual(state.dtype, np.float32)
        self.assertEqual(state.shape, (1,))
        self.assertAlmostEqual(float(state[0]), 0.042, places=5)
        self.assertEqual(robot.gripper.commands, [(0.095, 1.0), (0.01, 0.5)])

    def test_leader_mode_and_feedback(self):
        wrapper, robot = connected_wrapper()

        wrapper.enter_leader_mode()
        qpos = wrapper.get_leader_qpos()
        wrapper.enter_follower_mode()

        self.assertEqual(qpos.dtype, np.float32)
        np.testing.assert_array_equal(qpos, np.full(7, 0.2, dtype=np.float32))
        self.assertEqual(robot.events[-2:], ["leader", "follower"])

    def test_emergency_stop_and_disconnect_have_distinct_semantics(self):
        wrapper, robot = connected_wrapper()

        wrapper.emergency_stop()
        wrapper.disconnect()

        self.assertIn("emergency_stop", robot.events)
        self.assertIn("disconnect", robot.events)
        self.assertNotIn("disable", robot.events)
        self.assertFalse(wrapper.connected)

    def test_hold_current_position_sends_measured_pose_once(self):
        wrapper, robot = connected_wrapper()
        robot.feedback = Feedback([0.2] * 7)

        with patch("src.act.robot_wrapper.time.sleep") as sleep:
            held = wrapper.hold_current_position()

        np.testing.assert_array_equal(held, np.full(7, 0.2, dtype=np.float32))
        np.testing.assert_allclose(robot.targets, [[0.2] * 7])
        self.assertEqual(robot.events[-1], "follower")
        sleep.assert_called_once_with(0.2)

    def test_return_to_start_sends_one_target_then_only_polls(self):
        q0 = np.zeros(7, dtype=np.float32)

        class Wrapper:
            commands = []
            feedback = [q0 + 0.02, q0]

            def send_arm_qpos(self, target):
                self.commands.append(target.copy())

            def get_arm_qpos(self):
                return self.feedback.pop(0)

        wrapper = Wrapper()
        return_to_start(wrapper, q0, timeout_s=0.2)

        self.assertEqual(len(wrapper.commands), 1)
        np.testing.assert_array_equal(wrapper.commands[0], q0)


if __name__ == "__main__":
    unittest.main()
