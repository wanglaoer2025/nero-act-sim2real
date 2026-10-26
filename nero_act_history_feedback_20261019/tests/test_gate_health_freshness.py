import unittest

class GateHealthFreshnessTests(unittest.TestCase):
    def test_both_publish_and_receive_ages_obey_original150ms_without_queue_refresh(self):
        from executor.gate_health import require_gate_health
        status=dict(failed=False,driver_open=True,receipt_valid=True,received=10.,published_monotonic_s=9.99)
        require_gate_health(status,10.1)
        for change in [dict(received=9.8),dict(published_monotonic_s=9.8),dict(published_monotonic_s=11.),dict(failed=True),dict(receipt_valid=False)]:
            with self.subTest(change=change),self.assertRaises(RuntimeError):require_gate_health(dict(status,**change),10.1)
        with self.assertRaises(RuntimeError):require_gate_health(None,10.1)

    def test_fresh_local_callback_cannot_hide_a_stale_published_message(self):
        from executor.gate_health import require_gate_health
        with self.assertRaises(RuntimeError):require_gate_health(dict(failed=False,driver_open=True,receipt_valid=True,received=20.,published_monotonic_s=19.7),20.01)
