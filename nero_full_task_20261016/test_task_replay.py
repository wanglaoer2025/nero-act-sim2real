import unittest
import numpy as np
from task_replay import validate_task, tracking_ok, physical_result, initial_delay, task_session_class, check_clearance, check_action_results
from types import SimpleNamespace
from concurrent.futures import Future
import tempfile
import hashlib
from pathlib import Path


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.start = np.zeros(8); self.start[7] = .06
        self.rows = np.repeat(self.start[None], 100, axis=0)
        self.rows[:, 0] = np.linspace(0, .1, 100)
        self.limits = np.array([[-2, 2]]*7)

    def test_accepts_full_amplitude_without_scaling(self):
        checked = validate_task(self.start, self.rows, self.start, self.limits)
        np.testing.assert_array_equal(checked, self.rows)
        self.assertEqual(checked[-1, 0], .1)

    def test_discontinuous_start_or_path_and_limit_violation_fail(self):
        for axis, row, value in [(0, 0, .1), (0, 20, .5), (7, 20, .1)]:
            bad = self.rows.copy(); bad[row, axis] = value
            with self.assertRaises(ValueError):
                validate_task(self.start, bad, self.start, self.limits)
        bad = self.start.copy(); bad[0] = .02
        with self.assertRaises(ValueError):
            validate_task(bad, self.rows, self.start, self.limits)

    def test_late_gripper_overflow_and_nonfinite_values_fail(self):
        for value in [.096, float('nan')]:
            bad = self.rows.copy(); bad[-1, 7] = value
            with self.assertRaises(ValueError):
                validate_task(self.start, bad, self.start, self.limits)

    def test_tracking_failure_stops_even_if_final_target_is_reachable(self):
        bad = self.start.copy(); bad[0] += .031
        self.assertFalse(tracking_ok(bad, self.start, self.rows))
        self.assertTrue(tracking_ok(self.start, self.start, self.rows))

    def test_stationary_feedback_and_failed_action_cannot_pass(self):
        for samples, action in [(np.repeat(self.start[None], 100, axis=0), True),
                                (self.rows, False)]:
            self.assertFalse(physical_result(self.start, self.rows, samples, action)['success'])

    def test_full_traversal_and_settled_endpoint_are_required(self):
        q = np.r_[self.rows, np.repeat(self.rows[-1][None], 5, axis=0)]
        self.assertTrue(physical_result(self.start, self.rows, q, True)['success'])
        bad = q.copy(); bad[-1, 0] += .02
        self.assertFalse(physical_result(self.start, self.rows, bad, True)['success'])

    def test_first_point_rate_is_checked_and_timing_can_preserve_original_positions(self):
        start = self.start.copy(); start[0] = .004; start[7] -= .0035
        with self.assertRaises(ValueError):
            validate_task(start, self.rows, self.start, self.limits)
        delay = initial_delay(start, self.rows[0])
        self.assertGreaterEqual(delay, .35)
        np.testing.assert_array_equal(validate_task(start, self.rows, self.start, self.limits, delay), self.rows)

    def test_false_or_expired_clearance_cannot_be_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'image'; image.write_bytes(b'frame')
            record = dict(cup_absent_in_new_camera_frame=True,operator_swept_volume_clear=True,
                          frame_capture_wall_s=100.,image=str(image),image_sha256=hashlib.sha256(b'frame').hexdigest())
            check_clearance(record, 150.)
            for changed in [dict(frame_capture_wall_s=0.),dict(cup_absent_in_new_camera_frame=False),dict(image_sha256='wrong')]:
                with self.assertRaises(RuntimeError):check_clearance({**record,**changed},150.)


class ActionTests(unittest.TestCase):
    def test_first_send_is_retained_if_second_send_raises(self):
        class Base:
            def send_block(self, clients, rows):
                goal = SimpleNamespace(trajectory=SimpleNamespace(header=SimpleNamespace(stamp=None),points=[]))
                self.pending_goal_futures = [c.send_goal_async(goal) for c in clients]
        cls = task_session_class(Base); session = cls.__new__(cls)
        session.common_ros_start = 'same start'; session.first_delay_s = .05
        future = Future()
        first = SimpleNamespace(send_goal_async=lambda goal: future)
        def fail(goal):raise RuntimeError('second send failed')
        with self.assertRaisesRegex(RuntimeError,'second send'):
            session.send_block([first,SimpleNamespace(send_goal_async=fail)], [])
        self.assertEqual(session.pending_goal_futures,[future])

    def test_pending_goal_acknowledgement_keeps_monitoring(self):
        class Base:pass
        cls = task_session_class(Base); session = cls.__new__(cls)
        future = Future(); calls = []
        def monitor():
            calls.append(True)
            if len(calls)==3:future.set_result('accepted')
        session.execution_monitor=monitor; session.awaiting_ack=False; session.cancelling=False
        self.assertEqual(session.wait(future,1), 'accepted')
        self.assertEqual(len(calls),3)

    def test_known_goal_cancel_is_sent_before_waiting_for_late_ack(self):
        class Base:pass
        cls=task_session_class(Base); session=cls.__new__(cls)
        events=[]; ack=Future(); ack.set_result(SimpleNamespace(goals_canceling=[True]))
        handle=SimpleNamespace(accepted=True,cancel_goal_async=lambda:(events.append('known_cancel') or ack))
        pending=Future(); late=SimpleNamespace(accepted=True,cancel_goal_async=lambda:(events.append('late_cancel') or ack))
        session.block_handles=[handle];session.pending_goal_futures=[pending];session.report={}
        session.close_control_gate=lambda:(events.append('gate_closed') or None)
        def spin(seconds):
            events.append('spin')
            if not pending.done():pending.set_result(late)
        session.spin=spin
        session.cancel_all()
        self.assertEqual(events[:2],['gate_closed','known_cancel'])
        self.assertIn('late_cancel',events)

    def test_gate_service_failure_does_not_skip_known_cancellation(self):
        class Base:pass
        session=task_session_class(Base).__new__(task_session_class(Base))
        events=[];ack=Future();ack.set_result(None)
        session.block_handles=[SimpleNamespace(accepted=True,cancel_goal_async=lambda:(events.append('cancelled') or ack))]
        session.pending_goal_futures=[];session.report={}
        def fail():raise RuntimeError('gate unavailable')
        session.close_control_gate=fail
        session.cancel_all()
        self.assertEqual(events,['cancelled'])
        self.assertIn('gate unavailable',session.report['cancel_gate_close_error'])

    def test_single_controller_failure_is_detected_while_other_is_pending(self):
        failed=Future();pending=Future()
        failed.set_result(SimpleNamespace(status=6,result=SimpleNamespace(error_code=-4)))
        with self.assertRaises(RuntimeError):check_action_results([pending,failed])

    def test_completed_but_late_ack_cannot_start_trajectory(self):
        class Base:pass
        session=task_session_class(Base).__new__(task_session_class(Base))
        session.execution_monitor=lambda:None;session.awaiting_ack=True;session.cancelling=False
        session.scheduled_monotonic=0.
        f=Future();f.set_result('accepted')
        with self.assertRaises(TimeoutError):session.wait(f,1.)


if __name__ == '__main__':
    unittest.main()
