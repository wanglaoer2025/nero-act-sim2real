from types import SimpleNamespace
import unittest
from full_runtime import FullSession

try:
    from control_msgs.action import FollowJointTrajectory
    HAS_ROS = True
except ImportError:
    HAS_ROS = False


@unittest.skipUnless(HAS_ROS, 'requires installed generated ROS messages')
class FullROSTests(unittest.TestCase):
    def test_both_arm_and_gripper_are_sent_before_wait(self):
        sent = []
        class Client:
            def send_goal_async(self, goal):
                sent.append(goal)
                return SimpleNamespace(accepted=True, get_result_async=lambda: 'result')
        session = FullSession.__new__(FullSession)
        def wait(handle, timeout):
            self.assertEqual(len(sent), 2)
            return handle
        session.wait = wait
        rows = [[float(i)*.001]*7+[.06-i*.0005] for i in range(10)]
        results = session.send_block([Client(), Client()], rows)
        self.assertEqual(len(results), 2)
        self.assertEqual(list(sent[0].trajectory.joint_names), [f'joint{i}' for i in range(1,8)])
        self.assertEqual(list(sent[1].trajectory.joint_names), ['gripper'])
        self.assertEqual(len(sent[0].trajectory.points), 10)
        self.assertAlmostEqual(sent[1].trajectory.points[-1].positions[0], .0555)
        last = sent[0].trajectory.points[-1].time_from_start
        self.assertAlmostEqual(last.sec+last.nanosec/1e9, .5)

    def test_rejected_gripper_is_not_reported_as_success(self):
        class Client:
            def __init__(self, accepted):
                self.accepted = accepted
            def send_goal_async(self, goal):
                return SimpleNamespace(accepted=self.accepted, get_result_async=lambda: 'result')
        session = FullSession.__new__(FullSession)
        session.wait = lambda handle, timeout: handle
        with self.assertRaisesRegex(RuntimeError, 'rejected'):
            session.send_block([Client(True), Client(False)], [[0.]*7+[.06]]*10)
        self.assertEqual(len(session.block_handles), 2)

    def test_pending_accepted_goal_is_included_in_cancel(self):
        cancelled = []
        handle = SimpleNamespace(accepted=True, cancel_goal_async=lambda: cancelled.append(True))
        session = FullSession.__new__(FullSession)
        session.block_handles = []; session.pending_goal_futures = [handle]; session.report = {}
        session.wait = lambda value, timeout: value
        session.cancel_all()
        self.assertEqual(cancelled, [True])


if __name__ == '__main__':
    unittest.main()
