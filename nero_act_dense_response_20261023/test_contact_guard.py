import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'nero_act_transfer_resume_20261021'))
import checked_runtime
sys.path.insert(0, str(ROOT/'nero_act_sampling_retention_20261023'))
from contact_guard import PregraspSweptGuard
from predictive_guard import PredictivePhaseGuard
from executor.contact import ContactMonitor
from executor.ros_execution import RobotGeometry


class ContactGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = ROOT/'nero_act_sampling_retention_20261023/results/restored_current'
        cls.scene = json.loads((folder/'scene_configuration.json').read_text())
        with np.load(folder/'rollout.npz') as stream:
            cls.states = stream['states']
            cls.targets = stream['sent_targets']

    def test_saved_failure_is_rejected_before_physical_contact(self):
        # Old TCP-only checks admitted this target five points before contact.
        current, rows = self.states[1633], self.targets[1633:1634].copy()
        original = rows.copy()
        geometry = RobotGeometry(self.scene)
        tcp_guard = PredictivePhaseGuard(self.scene['pick_xy_m'], self.scene['place_xy_m'], .055)
        self.assertIsNone(tcp_guard.inspect(geometry.pose(current)['tcp'],
            np.array([geometry.pose(row)['tcp'] for row in rows]), rows[:,7], current[7], ContactMonitor()))
        with self.assertRaisesRegex(RuntimeError, 'Predicted pregrasp cup contact'):
            PregraspSweptGuard(self.scene).validate(current, rows, close_authorized=False)
        np.testing.assert_array_equal(rows, original)

    def test_clear_high_approach_is_still_admitted(self):
        self.assertGreater(PregraspSweptGuard(self.scene).validate(
            self.states[5], self.targets[5:7], close_authorized=False), 0)

    def test_nonfinite_target_is_never_admitted(self):
        rows = self.targets[5:7].copy()
        rows[0,0] = np.nan
        with self.assertRaises(ValueError):
            PregraspSweptGuard(self.scene).validate(self.states[5], rows, close_authorized=False)


if __name__ == '__main__':
    unittest.main()
