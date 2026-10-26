import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch
class AckTraceTests(unittest.TestCase):
    def test_received_ack_records_group_latency_and_deadline(self):
        from executor.transport import track_ack
        rows=[];future=Future()
        track_ack(future,['joint1'],10.,10.045,rows)
        with patch('executor.transport.time.monotonic',return_value=10.02):future.set_result(SimpleNamespace(accepted=True))
        self.assertTrue(rows[0]['accepted']);self.assertTrue(rows[0]['on_time']);self.assertAlmostEqual(rows[0]['latency_ms'],20.)
    def test_late_or_failed_ack_is_preserved_without_authorizing_a_goal(self):
        from executor.transport import track_ack
        rows=[];future=Future();track_ack(future,['gripper'],10.,10.045,rows)
        with patch('executor.transport.time.monotonic',return_value=10.06):future.set_result(SimpleNamespace(accepted=True))
        self.assertFalse(rows[0]['on_time'])
        future=Future();track_ack(future,['joint1'],20.,20.045,rows)
        future.set_exception(RuntimeError('transport failed'))
        self.assertIn('transport failed',rows[1]['error'])
