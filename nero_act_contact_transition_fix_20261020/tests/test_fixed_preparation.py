import copy
import json
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT.parent/'nero_act_fixed_scene_heads_20261018/model/run27_fixed_heads'

class FixedPreparationTests(unittest.TestCase):
    def test_ros_environment_preserves_paths_and_overrides_domain_without_duplicate_kwargs(self):
        from executor.fixed_trial_run import trial_environment
        result=trial_environment(dict(ROS_DOMAIN_ID='42',PYTHONPATH='/existing/ros'))
        self.assertEqual(result['ROS_DOMAIN_ID'],'42')
        self.assertEqual(result['PYTHONPATH'],'/existing/ros')
        self.assertEqual(result['OMP_NUM_THREADS'],'1')

    def test_stage_handoff_requires_acknowledged_close_and_resolved_cancellations(self):
        from executor.fixed_trial import require_clean_handoff
        good=dict(driver_gate_closed_for_abort=True,unresolved_goal_acknowledgements=0,
                  unresolved_cancellation_replies=0)
        require_clean_handoff(good)
        for changed in [dict(driver_gate_closed_for_abort=False),
                        dict(unresolved_goal_acknowledgements=1),
                        dict(unresolved_cancellation_replies=1),dict(cancel_errors=['failed'])]:
            with self.subTest(changed=changed),self.assertRaises(RuntimeError):
                require_clean_handoff(dict(good,**changed))

    def fixture(self):
        scene=json.loads((MODEL/'scene_config.json').read_text())
        start=np.r_[scene['initial_robot_state']['qpos_rad'],.080894]
        return scene,start

    def test_high_empty_open_holds_all_seven_axes_and_original_rates(self):
        from executor.fixed_trial import opening_rows
        scene,start=self.fixture();rows=opening_rows(start,scene)
        self.assertTrue(np.array_equal(rows[:,:7],np.broadcast_to(start[:7],rows[:,:7].shape)))
        self.assertEqual(rows[-1,7],.095)
        self.assertTrue(np.all(np.diff(np.r_[start[7],rows[:,7]])>=0))
        self.assertLessEqual(np.max(abs(np.diff(np.r_[start[7],rows[:,7]]))),.0005)

    def test_changed_arm_closed_gripper_nonfinite_and_low_start_denied(self):
        from executor.fixed_trial import opening_rows
        scene,start=self.fixture()
        for index,value in [(1,start[1]+.01),(7,.06),(7,.096),(7,float('nan'))]:
            changed=start.copy();changed[index]=value
            with self.subTest(index=index,value=value),self.assertRaises(ValueError):opening_rows(changed,scene)
        low=copy.deepcopy(scene);low['table_top_z_m']=1.
        with self.assertRaises(ValueError):opening_rows(start,low)

if __name__=='__main__':unittest.main()
