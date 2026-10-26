import unittest
import numpy as np


class ReplaySafetyTests(unittest.TestCase):
    def test_contact_protection_never_closes_further_or_changes_arm(self):
        from executor.replay_session import protect_gripper
        rows=np.tile([0,0,0,1.5,0,0,0,.065],(20,1))
        result=protect_gripper(rows,.083)
        self.assertTrue(np.array_equal(result[:,:7],rows[:,:7]))
        self.assertTrue(np.allclose(result[:,7],.074))
        self.assertTrue(np.all(result[:,7]>=rows[:,7]))
        with self.assertRaises(ValueError):protect_gripper(rows,.1)

    def test_phase_boundary_rejects_reference_jump_even_within_tracking_tolerance(self):
        from executor.replay_session import require_phase_start
        previous=np.array([0,0,0,1.5,0,0,0,.095])
        first=previous.copy();first[0]=.01
        with self.assertRaises(RuntimeError):require_phase_start(previous,previous,first)
        first[0]=.002;require_phase_start(previous,previous,first)

    def test_measured_start_rounding_joins_without_exceeding_original_point_rate(self):
        from executor.replay_session import join_reference,require_phase_start
        from executor.replay_trace import validate_reference
        previous=np.array([0,0,0,1.5,0,0,0,.094875]);first=previous.copy();first[6]=-.002035
        connection=join_reference(previous,first)
        require_phase_start(previous,previous,connection[0])
        validate_reference(connection)
        self.assertTrue(np.array_equal(connection[-1],first))

    def test_contact_replacement_is_continuous_and_cannot_tighten_grip(self):
        from executor.replay_session import contact_remainder
        from executor.replay_trace import validate_reference
        rows=np.tile([0,0,0,1.5,0,0,0,.065],(80,1))
        rows[:,7]=np.maximum(.065,.083-np.arange(80)*.0005)
        # Contact at83mm,8mm command gap; protection stops at9mm,original gap limit12mm.
        initial=rows[16].copy()
        remainder=contact_remainder(rows,.85,initial,.083)
        validate_reference(remainder)
        self.assertLessEqual(abs(remainder[0,7]-initial[7]),.000501)
        self.assertGreaterEqual(remainder[:,7].min(),.074)
        self.assertTrue(np.all(remainder[:,7]>=rows[17:,7]))

    def test_confirmation_pause_keeps_the_same_target_with_active_controller_holds(self):
        from executor.replay_session import phase_hold_seconds
        for phase in ['approach','close','lift','carry_place','release']:
            self.assertEqual(phase_hold_seconds(phase),300.)
        self.assertEqual(phase_hold_seconds('retreat'),0.)

    def test_lift_cannot_start_without_real_measured_contact(self):
        from executor.replay import require_lift_ready
        from executor.contact import ContactMonitor
        contact=ContactMonitor()
        with self.assertRaises(RuntimeError):require_lift_ready(contact)
        contact.authorize('close','GRASP',{'aligned':True})
        with self.assertRaises(RuntimeError):require_lift_ready(contact)
        for stamp in [1,1.1,1.2,1.3,1.4,1.5]:
            contact.update(width=.083,force=.2,target=.075,stamp=stamp)
        require_lift_ready(contact)

    def test_carry_and_release_need_their_confirmations_and_support_region(self):
        from executor.replay import require_carry_ready,require_release_ready
        from executor.contact import ContactMonitor
        contact=ContactMonitor();scene={'pick_xy_m':[0,0],'place_xy_m':[.1,0]}
        contact.close_authorized=True;contact.state='holding';contact.contact_width=.083
        require_carry_ready(contact,np.array([0,0,.255]),scene)
        for pose in [[0,0,.055],[.1,0,.255]]:
            with self.assertRaises(RuntimeError):require_carry_ready(contact,np.array(pose),scene)
        with self.assertRaises(RuntimeError):require_release_ready(contact,np.array([.1,0,.055]),scene)
        contact.authorize('carry','NEXT',{'cup_lifted':True})
        require_release_ready(contact,np.array([.1,0,.055]),scene)
        with self.assertRaises(RuntimeError):require_release_ready(contact,np.array([.1,0,.255]),scene)
