import importlib.util
import unittest


class GateTests(unittest.TestCase):
    def test_initial_close_must_be_acknowledged_before_any_enable(self):
        from executor.gate import GateLatch
        gate = GateLatch()
        self.assertIsNone(gate.request(True))
        closed = gate.request(False)
        self.assertIsNone(gate.request(True))
        gate.acknowledge(closed, True)
        self.assertIsNotNone(gate.request(True))

    def test_enable_requires_ack_and_late_enable_ack_cannot_reopen_closed_gate(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.gate'))
        from executor.gate import GateLatch
        gate = GateLatch()
        gate.acknowledge(gate.request(False), True)
        opened = gate.request(True)
        self.assertFalse(gate.open)
        closed = gate.request(False)
        gate.acknowledge(opened, True)
        self.assertFalse(gate.open)
        gate.acknowledge(closed, True)
        self.assertFalse(gate.open)
        opened = gate.request(True)
        gate.acknowledge(opened, True)
        self.assertTrue(gate.open)
        closed = gate.request(False)
        self.assertFalse(gate.open)
        gate.acknowledge(closed, False)
        self.assertFalse(gate.open)
        self.assertTrue(gate.failed)


if __name__ == '__main__':unittest.main()
