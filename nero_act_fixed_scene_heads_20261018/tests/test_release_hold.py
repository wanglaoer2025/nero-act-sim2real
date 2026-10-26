import importlib.util
import unittest
import numpy as np
from types import SimpleNamespace


class ReleaseHoldTests(unittest.TestCase):
    def test_collision_stops_for_support_confirmation_without_accepting_any_motion(self):
        from executor import release_hold
        self.assertTrue(callable(getattr(release_hold,'can_request_support_confirmation',None)))
        ask=release_hold.can_request_support_confirmation
        self.assertTrue(ask([-.52,-.16,.065],[-.52,-.16]))
        self.assertFalse(ask([-.52,-.16,.20],[-.52,-.16]))
        self.assertFalse(ask([-.52,0,.065],[-.52,-.16]))
        self.assertFalse(ask([-.52,-.16,.02],[-.52,-.16]))
        self.assertFalse(ask([float('nan'),0,.065],[-.52,-.16]))

    def test_world_cup_remains_until_first_upward_act_point(self):
        from executor import release_hold
        self.assertTrue(callable(getattr(release_hold,'first_lift_index',None)))
        current=np.array([-.52,0,.055])
        self.assertEqual(release_hold.first_lift_index(current,[[-.52,0,.054],[-.52,0,.0545],[-.52,0,.056]]),2)
        self.assertIsNone(release_hold.first_lift_index(current,[[-.52,0,.054],[-.52,0,.055]]))
        self.assertEqual(release_hold.first_lift_index(current,[[-.52,0,.056]]),0)
    def test_pregrasp_hold_keeps_open_reference_without_changing_act_arm(self):
        from executor import release_hold
        self.assertTrue(callable(getattr(release_hold,'hold_open_before_grasp_depth',None)))
        previous=np.r_[np.zeros(7),.095];rows=np.tile(previous,(10,1))
        rows[:,0]=np.linspace(.001,.002,10);rows[:,7]=np.linspace(.0945,.09,10)
        result=release_hold.hold_open_before_grasp_depth(rows,previous)
        np.testing.assert_equal(result[:,:7],rows[:,:7]);np.testing.assert_equal(result[:,7],.095)
    def test_pose_hold_is_scoped_to_closing_and_supported_opening(self):
        from executor import release_hold
        self.assertTrue(callable(getattr(release_hold,'contact_hold_required',None)))
        state=SimpleNamespace(close_authorized=False,release_authorized=False,state='open')
        self.assertFalse(release_hold.contact_hold_required(state))
        state.close_authorized=True;self.assertTrue(release_hold.contact_hold_required(state))
        state.state='holding';self.assertFalse(release_hold.contact_hold_required(state))
        state.release_authorized=True;self.assertTrue(release_hold.contact_hold_required(state))
        state.state='released';self.assertFalse(release_hold.contact_hold_required(state))
    def test_only_past_act_arm_target_is_held_gripper_predictions_are_unchanged(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.release_hold'))
        from executor.release_hold import hold_act_arm_during_release
        history=np.r_[np.full(7,.1),.074]
        rows=np.tile(history,(10,1));rows[:,:7]+=.002;rows[:,7]=np.linspace(.0745,.079,10)
        original=rows.copy()
        result=hold_act_arm_during_release(rows,history)
        np.testing.assert_equal(result[:,:7],np.tile(history[:7],(10,1)))
        np.testing.assert_equal(result[:,7],original[:,7]);np.testing.assert_equal(rows,original)
        with self.assertRaises(ValueError):
            hold_act_arm_during_release(rows,np.full(8,np.nan))

    def test_measured_support_pose_drift_is_still_rejected(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.release_hold'))
        from executor.release_hold import require_support_pose
        require_support_pose([.001,0,0],[0,0,0])
        with self.assertRaises(RuntimeError):require_support_pose([.0031,0,0],[0,0,0])

    def test_feedback_reference_transfer_preserves_step_and_holds_gripper(self):
        from executor import release_hold
        self.assertTrue(callable(getattr(release_hold,'release_reference_transfer',None)))
        previous=np.r_[np.zeros(7),.066]
        measured=np.r_[[.0067,-.005,0,0,0,0,0],.0737]
        rows=release_hold.release_reference_transfer(previous,measured)
        self.assertEqual(len(rows),4)
        self.assertLessEqual(np.max(abs(np.diff(np.vstack([previous,rows]),axis=0)[:,:7])),.002)
        np.testing.assert_equal(rows[-1,:7],measured[:7]);np.testing.assert_equal(rows[:,7],.066)
        with self.assertRaises(RuntimeError):
            release_hold.release_reference_transfer(previous,np.r_[[.031]+[0.]*6,.0737])


if __name__=='__main__':unittest.main()
