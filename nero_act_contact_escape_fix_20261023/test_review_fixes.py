import unittest
import numpy as np

from contact_escape import unsafe_attached_prefix
from action_provenance import action_provenance


class ReviewFixTests(unittest.TestCase):
    def test_first_floor_crossing_still_requires_upward_tcp(self):
        m=dict(supported=True,bilateral=True,unexpected_contact=False,unexpected_robot_contact=False)
        bad=unsafe_attached_prefix(-1.01e-6,np.array([-.91e-6]),.055,np.array([.055-1e-7]),
            measurement=m,airborne=False,simulation_only=True)
        self.assertEqual(bad.tolist(),[0])

    def test_action_provenance_discloses_retained_holds_without_global_false_claim(self):
        report=action_provenance()
        self.assertNotIn('policy_targets_rewritten',report)
        self.assertFalse(report['contact_escape_rewrites_targets'])
        self.assertTrue(report['existing_contact_arm_holds_retained'])
        self.assertTrue(report['existing_pregrasp_gripper_holds_retained'])


if __name__=='__main__':unittest.main()
