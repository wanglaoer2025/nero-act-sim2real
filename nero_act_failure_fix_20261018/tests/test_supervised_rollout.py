import unittest
import numpy as np


class SupervisedRolloutTests(unittest.TestCase):
    def test_simulated_contact_keeps_the_real_width_and_slip_limits(self):
        from simulation.supervised_rollout import PhysicsContact
        contact=PhysicsContact();contact.close_authorized=True
        m=dict(gripper_width=.095,bilateral=True)
        contact.update(m,.087,0.);contact.update(m,.087,1.)
        self.assertEqual(contact.state,'open','95mm cannot be an accepted held-cup width')
        m['gripper_width']=.074
        contact.update(m,.066,1.1);contact.update(m,.066,1.7)
        self.assertEqual(contact.state,'holding')
        with self.assertRaises(RuntimeError):
            contact.update(dict(m,gripper_width=.070),.062,1.8)

    def test_simulated_confirmation_cannot_accept_source_or_airborne_cup(self):
        from simulation.supervised_rollout import require_supported_target
        m=dict(cup_xyz=[-.52,-.16,.0475],supported=True,cup_upright=1.)
        require_supported_target(m,np.array([-.52,-.16]))
        for invalid in [dict(m,supported=False),dict(m,cup_xyz=[-.52,0,.0475])]:
            with self.assertRaises(RuntimeError):require_supported_target(invalid,np.array([-.52,-.16]))


if __name__=='__main__':unittest.main()
