"""Real generated ROS message checks; fake services, no ROS graph or hardware."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from run_act_response import ResponseSession, projected_target
from ros_support import ARM, NAMES

try:
    from moveit_msgs.srv import GetMotionPlan, GetStateValidity
    from moveit_msgs.msg import RobotState, RobotTrajectory, PlanningScene
    from trajectory_msgs.msg import JointTrajectoryPoint
    from builtin_interfaces.msg import Time
    ROS_MESSAGES = True
except ImportError:
    ROS_MESSAGES = False


@unittest.skipUnless(ROS_MESSAGES, 'run under installed ROS Python environment')
class ROSPlanTests(unittest.TestCase):
    def fixture(self, directory, collision=False):
        output = Path(directory)
        output.joinpath('robot.urdf').write_text('<robot>'+''.join(
            f'<joint name="{name}" type="revolute"><limit lower="-3" upper="3"/></joint>'
            for name in ARM)+'</robot>')
        session = ResponseSession.__new__(ResponseSession)
        session.output = output; session.report = {}; session.scene_msg = PlanningScene()
        session.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time())),
            create_publisher=lambda *args: SimpleNamespace(publish=lambda message: None))
        session.spin = lambda seconds: None
        session.requests = []; session.validity_count = 0
        def call(service, kind, request):
            session.requests.append(request)
            if kind is GetMotionPlan:
                response = GetMotionPlan.Response()
                response.motion_plan_response.error_code.val = 1
                trajectory = response.motion_plan_response.trajectory.joint_trajectory
                trajectory.joint_names = ARM
                endpoint = [j.position for j in request.motion_plan_request.goal_constraints[0].joint_constraints]
                trajectory.points = [JointTrajectoryPoint(positions=[0.]*7), JointTrajectoryPoint(positions=endpoint)]
                return response
            session.validity_count += 1
            return GetStateValidity.Response(valid=not collision)
        session.call = call
        return session

    def test_generated_ros_plan_uses_act_endpoint_and_checks_path(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.fixture(directory)
            chunk = [[1., -2., 3., 0., 0., 0., 0., .095]]*100
            target, _ = projected_target([0.]*8, chunk)
            trajectory = session.plan_response([0.]*8, target, 'arm')
            self.assertEqual(list(trajectory.joint_trajectory.points[-1].positions), target)
            self.assertGreater(session.validity_count, 5)
            last = trajectory.joint_trajectory.points[-1].time_from_start
            self.assertGreaterEqual(last.sec+last.nanosec*1e-9, 4.)
            self.assertFalse(session.report.get('execution_goal_submitted', False))

    def test_collision_blocks_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.fixture(directory, collision=True)
            target, _ = projected_target([0.]*8, [[1., -2., 3., 0., 0., 0., 0., .095]]*100)
            with self.assertRaisesRegex(RuntimeError, 'validity'):
                session.plan_response([0.]*8, target, 'arm')
            self.assertFalse(session.report.get('plan_complete', False))


if __name__ == '__main__':
    unittest.main()
