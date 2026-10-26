import unittest
import numpy as np
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path

from restore_start import return_target, return_path, return_result, plan_return
from ros_support import ARM

try:
    from moveit_msgs.srv import GetMotionPlan
    from moveit_msgs.msg import PlanningScene
    from trajectory_msgs.msg import JointTrajectoryPoint
    from builtin_interfaces.msg import Time
    HAS_ROS = True
except ImportError:
    HAS_ROS = False


class RestoreStartTests(unittest.TestCase):
    def profile(self):
        return {'initial_state': [0.] * 7 + [.06], 'initial_gripper_tolerance_m': .004}

    def test_target_is_exact_training_start_and_gripper_is_not_changed(self):
        state = [.01, -.02, .03, 0., 0., 0., 0., .060218]
        target = return_target(state, self.profile())
        self.assertEqual(target, [0.] * 7 + [.060218])

    def test_large_unknown_posture_is_not_automatically_repositioned(self):
        with self.assertRaisesRegex(ValueError, '0.05'):
            return_target([.1] + [0.] * 6 + [.06], self.profile())

    def test_gripper_mismatch_requires_explicit_separate_handling(self):
        with self.assertRaisesRegex(ValueError, 'gripper'):
            return_target([0.] * 7 + [.03], self.profile())

    def test_return_path_can_move_multiple_axes_in_either_direction(self):
        start = [.03, -.02] + [0.] * 5 + [.06]
        target = return_target(start, self.profile())
        rows = return_path(ARM, [start[:7], target[:7]], start, target)
        np.testing.assert_allclose(rows[-1], target[:7])

    def test_return_path_cannot_detour_outside_endpoint_corridor(self):
        start = [.03] + [0.] * 6 + [.06]
        with self.assertRaisesRegex(ValueError, 'corridor'):
            return_path(ARM, [start[:7], [.04] + [0.] * 6, [0.] * 7], start, [0.] * 7 + [.06])

    def test_mock_success_with_static_actual_feedback_cannot_pass(self):
        start = [.03] + [0.] * 6 + [.06]
        target = return_target(start, self.profile())
        self.assertFalse(return_result(start, target, [start] * 10, True)['success'])

    def test_real_return_must_reach_all_axes_and_hold_gripper(self):
        start = [.03, -.02] + [0.] * 5 + [.06]
        target = return_target(start, self.profile())
        self.assertTrue(return_result(start, target, [target] * 10, True)['success'])
        bad = target[:]; bad[7] += .002
        self.assertFalse(return_result(start, target, [bad] * 10, True)['success'])


@unittest.skipUnless(HAS_ROS, 'requires generated ROS messages')
class ReturnPlanningTests(unittest.TestCase):
    def fixture(self, output, collision=False):
        session = SimpleNamespace(output=Path(output),report={},requests=[],validated=[],published=[])
        session.scene = lambda: PlanningScene()
        session.spin = lambda seconds: None
        session.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time())),
            create_publisher=lambda *args: SimpleNamespace(publish=session.published.append))
        def call(name, kind, request):
            session.requests.append(request)
            response = GetMotionPlan.Response(); response.motion_plan_response.error_code.val = 1
            jt = response.motion_plan_response.trajectory.joint_trajectory
            jt.joint_names = ARM
            target = [c.position for c in request.motion_plan_request.goal_constraints[0].joint_constraints]
            jt.points = [JointTrajectoryPoint(positions=list(request.motion_plan_request.start_state.joint_state.position[:7])),
                         JointTrajectoryPoint(positions=target)]
            return response
        def validate(rows, group):
            session.validated.extend(rows)
            if collision: raise RuntimeError('collision rejected')
        session.call = call; session.validate_block = validate
        return session

    def test_plan_uses_exact_known_start_and_holds_actual_gripper(self):
        with TemporaryDirectory() as output:
            s = self.fixture(output); start = [.01] + [0.] * 6 + [.060218]; target = [0.] * 7 + [.060218]
            trajectory = plan_return(s,start,target,'arm')
            self.assertEqual(list(trajectory.joint_trajectory.joint_names),ARM)
            self.assertEqual(list(trajectory.joint_trajectory.points[-1].positions),target[:7])
            self.assertTrue(all(row[7] == start[7] for row in s.validated))
            self.assertFalse(s.report.get('execution_goal_submitted',False))
            self.assertEqual(s.published[0].trajectory[0],trajectory)

    def test_invalid_return_path_stops_before_display_or_execution(self):
        with TemporaryDirectory() as output:
            s = self.fixture(output,True)
            with self.assertRaisesRegex(RuntimeError,'collision rejected'):
                plan_return(s,[.01] + [0.] * 6 + [.06],[0.] * 7 + [.06],'arm')
            self.assertFalse(s.published)
            self.assertFalse(s.report.get('execution_goal_submitted',False))


if __name__ == '__main__':
    unittest.main()
