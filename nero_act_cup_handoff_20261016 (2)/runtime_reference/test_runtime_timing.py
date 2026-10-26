import unittest
from unittest.mock import patch
from types import SimpleNamespace

import full_runtime
import async_runtime


class RuntimeTimingTests(unittest.TestCase):
    def test_next_observation_is_sampled_near_end_of_half_second_block(self):
        self.assertFalse(async_runtime.next_query_due(.3, False))
        self.assertFalse(async_runtime.next_query_due(.399, False))
        self.assertTrue(async_runtime.next_query_due(.4, False))
        self.assertFalse(async_runtime.next_query_due(.5, True))

    def validation(self, start, duration, error=None):
        clock = [start]
        def validate(rows, group):
            clock[0] += duration
            if error:
                raise error
        session = SimpleNamespace(validate_block=validate)
        plan = {'observation_step': 0, 'details': {'observation_wall_s': 10.0,
                'camera_received_monotonic_s': 10.005, 'inference_ms': 23.0,
                'response_received_monotonic_s': 10.03}}
        report = {}
        return session, plan, report, clock

    def test_validation_saves_separate_duration_and_observation_age(self):
        s, plan, report, clock = self.validation(10.12, .04)
        with patch.object(full_runtime.time, 'monotonic', side_effect=lambda: clock[0]):
            attempt = full_runtime.validate_fresh_block(s, [[0.] * 8] * 10, 'arm', plan, report)
        self.assertAlmostEqual(attempt['validation_duration_s'], .04)
        self.assertAlmostEqual(attempt['observation_age_after_validation_s'], .16)
        self.assertAlmostEqual(attempt['ready_prediction_wait_s'], .09)
        self.assertEqual(report['block_attempts'], [attempt])
        self.assertTrue(attempt['freshness_passed'])

    def test_stale_validation_still_fails_at_unchanged_quarter_second_limit(self):
        s, plan, report, clock = self.validation(10.20, .06)
        with patch.object(full_runtime.time, 'monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(RuntimeError, 'freshness budget'):
                full_runtime.validate_fresh_block(s, [[0.] * 8] * 10, 'arm', plan, report)
        attempt = report['block_attempts'][0]
        self.assertFalse(attempt['freshness_passed'])
        self.assertAlmostEqual(attempt['observation_age_after_validation_s'], .26)
        self.assertEqual(attempt['pre_send_age_limit_s'], .25)

    def test_collision_failure_keeps_validation_timing_and_fails_closed(self):
        s, plan, report, clock = self.validation(10.12, .02, RuntimeError('collision rejected'))
        with patch.object(full_runtime.time, 'monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(RuntimeError, 'collision rejected'):
                full_runtime.validate_fresh_block(s, [[0.] * 8] * 10, 'arm', plan, report)
        self.assertAlmostEqual(report['block_attempts'][0]['validation_duration_s'], .02)
        self.assertFalse(report['block_attempts'][0]['freshness_passed'])

    def test_failed_run_retains_block_rate_and_elapsed_duration(self):
        blocks = [{'sent_monotonic_s': t} for t in [10., 10.52, 11.04]]
        timing = full_runtime.timing_summary(blocks, 10., 11.7)
        self.assertAlmostEqual(timing['actual_block_hz'], 1 / .52)
        self.assertAlmostEqual(timing['execution_elapsed_s'], 1.7)
        self.assertEqual(timing['query_offset_within_block_s'], .4)
        self.assertFalse(timing['physical_timing_acceptance_completed'])


if __name__ == '__main__':
    unittest.main()
