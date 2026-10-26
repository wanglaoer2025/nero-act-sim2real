import unittest

from full_runtime import execution_limits, execution_pending


class StepBudgetTests(unittest.TestCase):
    def test_200_points_stop_after_twenty_completed_blocks_despite_overhead(self):
        seconds, steps = execution_limits(None, 200)
        completed = 0
        while execution_pending(seconds, steps, completed, completed / 10 * .56):
            completed += 10
        self.assertEqual(completed, 200)
        self.assertFalse(execution_pending(seconds, steps, completed, 11.2))

    def test_expired_watchdog_cannot_report_partial_190_points_as_success(self):
        seconds, steps = execution_limits(None, 200)
        with self.assertRaisesRegex(TimeoutError, 'step target'):
            execution_pending(seconds, steps, 190, 20.)

    def test_no_last_block_is_started_without_half_second_budget(self):
        with self.assertRaises(TimeoutError):
            execution_pending(20., 200, 190, 19.6)

    def test_legacy_seconds_and_default_are_retained(self):
        self.assertEqual(execution_limits(None, None), (18., None))
        self.assertEqual(execution_limits(10., None), (10., None))
        self.assertTrue(execution_pending(18., None, 330, 17.9))
        self.assertFalse(execution_pending(18., None, 330, 18.1))

    def test_invalid_or_conflicting_budgets_fail_before_hardware(self):
        for seconds, steps in [(10., 200), (None, 0), (None, 201),
                               (None, 410), (None, -10), (None, True),
                               (float('nan'), None), (float('inf'), None), (21., None)]:
            with self.subTest(seconds=seconds, steps=steps):
                with self.assertRaises(ValueError):
                    execution_limits(seconds, steps)


if __name__ == '__main__':
    unittest.main()
