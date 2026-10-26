import importlib.util
import unittest


class VirtualBridgeTests(unittest.TestCase):
    def test_only_isolated_mock_hardware_can_accept_virtual_initialization(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.virtual_bridge'))
        from executor.virtual_bridge import require_virtual_isolation
        valid = dict(hardware_plugins=['mock_components/GenericSystem'], driver_nodes=0,
                     gate_nodes=0, gated_publishers=0, gated_subscribers=0, raw_other_subscribers=0)
        require_virtual_isolation(valid)
        for field, value in [('driver_nodes', 1), ('gate_nodes', 1), ('gated_publishers', 1),
                             ('raw_other_subscribers', 1), ('hardware_plugins', []),
                             ('hardware_plugins', ['real_hardware/Robot'])]:
            with self.assertRaises(PermissionError):require_virtual_isolation(dict(valid, **{field:value}))


if __name__ == '__main__':unittest.main()
