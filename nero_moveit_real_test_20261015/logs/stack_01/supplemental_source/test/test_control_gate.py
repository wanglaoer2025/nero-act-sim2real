import math
import runpy
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "agx_arm_control_gate"
SCRIPT_GLOBALS = runpy.run_path(str(SCRIPT))
positions_are_synchronized = SCRIPT_GLOBALS["positions_are_synchronized"]
controller_is_synchronized = SCRIPT_GLOBALS["controller_is_synchronized"]
controlled_joint_names = SCRIPT_GLOBALS["controlled_joint_names"]


class ControlGateTest(unittest.TestCase):
    def test_requires_complete_finite_feedback_inside_tolerance(self):
        feedback = {"joint1": 0.1, "joint2": -0.2}

        self.assertTrue(positions_are_synchronized(
            {"joint1": 0.11, "joint2": -0.19},
            feedback,
            ["joint1", "joint2"],
            math.radians(3),
        ))
        self.assertFalse(positions_are_synchronized(
            {"joint1": 0.0, "joint2": 0.0},
            feedback,
            ["joint1", "joint2"],
            math.radians(3),
        ))
        self.assertFalse(positions_are_synchronized(
            {"joint1": 0.1}, feedback, ["joint1", "joint2"], 0.1
        ))

    def test_controller_groups_are_synchronized_and_forwarded_separately(self):
        feedback = {"joint1": 1.57, "joint2": -0.5, "gripper": 0.0}
        zeroed_arm = {"joint1": 0.0, "joint2": 0.0, "gripper": 0.0}

        self.assertFalse(controller_is_synchronized(
            "arm_controller/follow_joint_trajectory/_action/status",
            zeroed_arm,
            feedback,
            ["joint1", "joint2"],
            math.radians(3),
            0.005,
        ))
        self.assertTrue(controller_is_synchronized(
            "gripper_controller/follow_joint_trajectory/_action/status",
            zeroed_arm,
            feedback,
            ["joint1", "joint2"],
            math.radians(3),
            0.005,
        ))
        self.assertEqual(
            controlled_joint_names(
                ["arm_controller/follow_joint_trajectory/_action/status"],
                ["joint1", "joint2"],
            ),
            ["joint1", "joint2"],
        )
        self.assertEqual(
            controlled_joint_names(
                ["gripper_controller/follow_joint_trajectory/_action/status"],
                ["joint1", "joint2"],
            ),
            ["gripper"],
        )


if __name__ == "__main__":
    unittest.main()
