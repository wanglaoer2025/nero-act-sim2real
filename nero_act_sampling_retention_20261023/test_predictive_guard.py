import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'nero_act_grouped_gripper_20261021'))
from executor.phases import PhaseGuard as OriginalPhaseGuard
from executor.contact import ContactMonitor
try:
    from predictive_guard import PredictivePhaseGuard
except ImportError:
    PredictivePhaseGuard=OriginalPhaseGuard


class PredictiveGuardTests(unittest.TestCase):
    def inspect(self,current,target):
        return PredictivePhaseGuard([0,0],[.2,0],.055).inspect(np.array(current),np.array([target]),np.array([.095]),.095,ContactMonitor())

    def test_predicts_crossing_below_grasp_floor_before_action_is_sent(self):
        with self.assertRaises(RuntimeError):self.inspect([.088,0,.046],[.088,0,.044])

    def test_off_axis_low_descent_is_rejected_before_the_cup_region(self):
        with self.assertRaises(RuntimeError):self.inspect([.04,0,.121],[.04,0,.119])

    def test_aligned_descent_and_off_axis_vertical_clearance_remain_possible(self):
        self.assertIsNone(self.inspect([.001,0,.13],[.001,0,.128]))
        self.assertIsNone(self.inspect([.04,0,.10],[.04,0,.102]))
        self.assertIsNone(self.inspect([.04,0,.25],[.04,0,.248]))

    def test_grasp_event_cannot_bypass_the_predicted_arm_floor(self):
        guard=PredictivePhaseGuard([0,0],[.2,0],.055)
        with self.assertRaises(RuntimeError):
            guard.inspect(np.array([.001,0,.046]),np.array([[.001,0,.044]]),np.array([.084]),.095,ContactMonitor())
        self.assertEqual(guard.inspect(np.array([.001,0,.055]),np.array([[.001,0,.055]]),
            np.array([.084]),.095,ContactMonitor()),'GRASP')


if __name__=='__main__':unittest.main()
