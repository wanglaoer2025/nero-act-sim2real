import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import time
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('air', Path(__file__).with_name('ros_support.py'))
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


class InitialDiscoveryTests(unittest.TestCase):
    def initial_session(self, feedback_after):
        s = bare()
        clock = [0.0]
        def spin(seconds):
            clock[0] += seconds
            if feedback_after is not None and clock[0] >= feedback_after:
                s.trace = [{'position': [0.0] * 8}] * 10
        s.spin = spin
        def actual():
            if len(s.trace) < 10:
                raise RuntimeError('Missing or stale actual feedback')
            raise LookupError('Initial fresh feedback accepted')
        s.actual = actual
        return s, clock

    def test_initial_dds_discovery_can_take_two_seconds(self):
        s, clock = self.initial_session(2.0)
        with patch.object(air.time, 'monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(LookupError, 'fresh feedback accepted'):
                s.inventory()
        self.assertLessEqual(clock[0], 5.2)

    def test_missing_initial_feedback_still_times_out(self):
        s, clock = self.initial_session(None)
        with patch.object(air.time, 'monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(RuntimeError, 'Missing or stale actual feedback'):
                s.inventory()
        self.assertLessEqual(clock[0], 5.2)


class BridgeDiscoveryTests(unittest.TestCase):
    def delayed_gate(self, discover_after):
        s, q = ready_session()
        clock = [time.monotonic()]
        started = clock[0]
        get_publishers = s.node.get_publishers_info_by_topic
        def publishers(topic):
            if topic == '/control/gated_joint_states' and (
                    discover_after is None or clock[0] - started < discover_after):
                return []
            return get_publishers(topic)
        def spin(seconds):
            clock[0] += seconds
            s.latest['/control/joint_states']['receipt'] = clock[0]
        s.spin = spin
        s.node.get_publishers_info_by_topic = publishers
        return s, q, clock, started

    def test_delayed_gate_discovery_does_not_require_restart(self):
        s, q, clock, started = self.delayed_gate(.5)
        with patch.object(air.time, 'monotonic', side_effect=lambda: clock[0]):
            s.ready(q)
        self.assertGreaterEqual(clock[0] - started, .5)
        self.assertLessEqual(clock[0] - started, 5.2)

    def test_permanently_missing_gate_still_fails(self):
        s, q, clock, started = self.delayed_gate(None)
        with patch.object(air.time, 'monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(RuntimeError, 'sole gated publisher'):
                s.ready(q)
        self.assertLessEqual(clock[0] - started, 5.2)

    def test_duplicate_gate_is_rejected_without_waiting(self):
        s, q, clock, started = self.delayed_gate(0)
        original = s.node.get_publishers_info_by_topic
        s.node.get_publishers_info_by_topic = lambda topic: (
            original(topic) * 2 if topic == '/control/gated_joint_states' else original(topic))
        with patch.object(air.time, 'monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(RuntimeError, 'sole gated publisher'):
                s.ready(q)
        self.assertLess(clock[0] - started, .2)

    def test_stale_feedback_aborts_while_waiting_for_gate(self):
        s, q, clock, started = self.delayed_gate(None)
        def actual():
            if clock[0] - started > .2:
                raise RuntimeError('Missing or stale actual feedback')
            return q[:]
        s.actual = actual
        with patch.object(air.time, 'monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(RuntimeError, 'Missing or stale actual feedback'):
                s.ready(q)
        self.assertLess(clock[0] - started, .4)


if __name__ == '__main__':
    unittest.main()
