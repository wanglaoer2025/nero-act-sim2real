import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import time
import unittest


spec = importlib.util.spec_from_file_location('air', Path(__file__).with_name('nero_moveit_real_test.py'))
air = importlib.util.module_from_spec(spec)
spec.loader.exec_module(air)


def bare():
    s = air.Session.__new__(air.Session)
    s.errors, s.latest, s.trace = [], {}, []
    s.last_stamp, s.last_advance = None, None
    return s


def message(names):
    return NS(name=names, position=[0.1] * len(names),
              header=NS(stamp=NS(sec=1, nanosec=0)))


def ready_session():
    s = bare()
    q = [0.1] * 7 + [0.06]
    s.report = {'moveit_parameters': {'use_sim_time': False, 'allow_trajectory_execution': True},
                'controllers': [{'name': 'arm_controller', 'state': 'active'}],
                'topic_publishers': dict.fromkeys(['/feedback/joint_states', '/control/joint_states',
                                                  '/control/gated_joint_states']),
                'direct_command_publishers': {'/control/move_j': 0}}
    s.latest['/control/joint_states'] = {'receipt': time.monotonic(), 'q': q}
    s.active_goals = {}
    s.spin = lambda _: None
    s.actual = lambda: q[:]
    s.driver = {'arm_type': 'nero', 'effector_type': 'agx_gripper', 'control_enabled': False,
                'speed_percent': 5, 'use_sim_time': False}
    s.parameters = lambda node, names: s.driver
    s.scene = lambda: NS(robot_state=NS(joint_state=NS(name=air.NAMES, position=q)))
    owners = {'/feedback/joint_states': 'agx_arm_ctrl_single_node',
              '/control/joint_states': 'joint_state_broadcaster',
              '/control/gated_joint_states': 'control_gate'}
    s.node = NS(get_publishers_info_by_topic=lambda t: [NS(node_name=owners[t], node_namespace='/')],
                count_publishers=lambda _: 0,
                get_subscriptions_info_by_topic=lambda t:
                    [NS(node_name='agx_arm_ctrl_single_node', node_namespace='/')]
                    if t == '/control/gated_joint_states' else [])
    return s, q


class TelemetryTests(unittest.TestCase):
    def test_gated_arm_only_is_valid(self):
        s = bare()
        s.feedback('/control/gated_joint_states', message(air.ARM))
        self.assertEqual([], s.errors)

    def test_gated_gripper_only_is_valid(self):
        s = bare()
        s.feedback('/control/gated_joint_states', message(['gripper']))
        self.assertEqual([], s.errors)

    def test_incomplete_actual_feedback_remains_invalid(self):
        s = bare()
        s.feedback('/feedback/joint_states', message(air.ARM))
        self.assertTrue(s.errors)

    def test_incomplete_gated_arm_remains_invalid(self):
        s = bare()
        s.feedback('/control/gated_joint_states', message(air.ARM[:-1]))
        self.assertTrue(s.errors)

    def test_duplicate_gated_joint_remains_invalid(self):
        s = bare()
        s.feedback('/control/gated_joint_states', message(air.ARM + ['joint1']))
        self.assertTrue(s.errors)


class ReadinessTests(unittest.TestCase):
    def test_idle_closed_gate_is_ready_with_synchronized_bridge(self):
        s, q = ready_session()
        s.ready(q)

    def test_initially_open_driver_is_rejected(self):
        s, q = ready_session()
        s.driver['control_enabled'] = True
        with self.assertRaises(RuntimeError):
            s.ready(q)

    def test_driver_subscribed_to_raw_topic_is_rejected(self):
        s, q = ready_session()
        s.node.get_subscriptions_info_by_topic = lambda t: [NS(node_name='agx_arm_ctrl_single_node', node_namespace='/')]
        with self.assertRaises(RuntimeError):
            s.ready(q)

    def test_mock_zero_start_is_rejected(self):
        s, q = ready_session()
        s.latest['/control/joint_states']['q'] = [0.0] * 8
        with self.assertRaises(RuntimeError):
            s.ready(q)

    def test_active_trajectory_is_rejected(self):
        s, q = ready_session()
        s.active_goals = {'arm': [2]}
        with self.assertRaises(RuntimeError):
            s.ready(q)


if __name__ == '__main__':
    unittest.main()
