import unittest
from types import SimpleNamespace
import numpy as np
from src.act.response_probe import FreshReader, run_probe


class Clock:
    def __init__(self):
        self.t = 0.
    def now(self):
        return self.t
    def sleep(self, duration):
        self.t += max(duration, .00001)


class Robot:
    def __init__(self, clock, stationary=False, stale=False, delayed=False):
        self.clock = clock
        self.q = np.zeros(7)
        self.stationary, self.stale, self.delayed = stationary, stale, delayed
        self.pending = []
        self.commands = 0
        self.err_code = 0
    def get_joint_angles(self):
        if self.delayed:
            while self.pending and self.pending[0][0] <= self.clock.t:
                _, self.q = self.pending.pop(0)
        return SimpleNamespace(msg=self.q.copy(), timestamp=1. if self.stale else self.clock.t)
    def get_arm_status(self):
        return SimpleNamespace(timestamp=self.clock.t, msg=SimpleNamespace(
            ctrl_mode=1, arm_status=0, mode_feedback=1, teach_status=0,
            motion_status=0, trajectory_num=0, err_code=self.err_code))
    def send_arm_qpos(self, target):
        self.commands += 1
        if self.stationary:
            return
        if self.delayed:
            self.pending.append((self.clock.t + .8, np.asarray(target).copy()))
        else:
            self.q = np.asarray(target).copy()


class ProbeTests(unittest.TestCase):
    def run_case(self, robot, clock, rows):
        return run_probe(robot, FreshReader(robot, clock.now, clock.sleep), rows, 0,
                         np.array([[-1., 1.]] * 7), now=clock.now, sleep=clock.sleep)

    def test_stationary_fresh_feedback_fails_before_return(self):
        clock = Clock(); robot = Robot(clock, stationary=True); rows = []
        with self.assertRaisesRegex(RuntimeError, 'not demonstrated'):
            self.run_case(robot, clock, rows)
        self.assertEqual(robot.commands, 60)
        self.assertFalse(any(row['phase'] == 'return' for row in rows))

    def test_cached_feedback_blocks_before_motion(self):
        clock = Clock(); robot = Robot(clock, stale=True)
        with self.assertRaises(TimeoutError):
            self.run_case(robot, clock, [])
        self.assertEqual(robot.commands, 0)

    def test_delayed_response_passes_with_hold_window(self):
        clock = Clock(); robot = Robot(clock, delayed=True); rows = []
        self.assertTrue(self.run_case(robot, clock, rows)['passed'])
        self.assertGreaterEqual(clock.t, 6.)
        self.assertEqual(robot.commands, 120)
        self.assertLessEqual(max(abs(row['command'][0]) for row in rows if row.get('command')), .005)

    def test_controller_fault_blocks_before_motion(self):
        clock = Clock(); robot = Robot(clock); robot.err_code = 1
        with self.assertRaisesRegex(RuntimeError, 'controller fault'):
            self.run_case(robot, clock, [])
        self.assertEqual(robot.commands, 0)

    def test_stale_status_blocks_before_motion(self):
        clock = Clock(); robot = Robot(clock)
        getter = robot.get_arm_status
        def frozen():
            value = getter(); value.timestamp = 1.; return value
        robot.get_arm_status = frozen
        with self.assertRaises(TimeoutError):
            self.run_case(robot, clock, [])
        self.assertEqual(robot.commands, 0)

    def test_return_failure_cannot_pass(self):
        clock = Clock(); robot = Robot(clock)
        send = robot.send_arm_qpos
        def refuse_return(target):
            if robot.commands < 60:
                send(target)
            else:
                robot.commands += 1
        robot.send_arm_qpos = refuse_return
        with self.assertRaisesRegex(RuntimeError, 'acceptance failed'):
            self.run_case(robot, clock, [])


if __name__ == '__main__':
    unittest.main()
