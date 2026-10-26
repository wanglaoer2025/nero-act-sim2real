import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parent


class DemoTests(unittest.TestCase):
    def setUp(self):
        path=ROOT/'entry.py'
        self.assertTrue(path.is_file(),'Missing independently bounded ACT transfer demo entry')
        spec=importlib.util.spec_from_file_location('_demo_entry',path)
        self.entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.entry)
        self.plan=json.loads((ROOT/'preparation/plan.json').read_text())
        self.rows=np.load(self.plan['targets_path'],allow_pickle=False)

    def test_bound_act_prefix_has_visible_motion_and_no_gripper_commands(self):
        self.entry.validate_plan(self.plan,self.rows)
        self.assertGreater(len(self.rows),250)
        self.assertGreater(max(self.plan['joint_excursion_deg']),20)
        self.assertEqual(self.plan['gripper_commands'],0)
        self.assertGreaterEqual(min(p[2] for p in self.plan['sdk_flange_poses']),.36)
        self.assertEqual(self.plan['scope'],'protected_ACT_simulation_joint_prefix_real_replay')

    def test_floor_speed_and_gripper_guards_reject_invalid_targets(self):
        for kind in ['floor','speed','gripper']:
            plan=copy.deepcopy(self.plan);rows=self.rows.copy()
            if kind=='floor':plan['sdk_flange_poses'][0][2]=.349
            if kind=='speed':plan['speed_percent']=6
            if kind=='gripper':rows[3,7]-=.003
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.entry.validate_plan(plan,rows)

    def test_original_act_arm_targets_cannot_be_replaced_by_teacher_or_adjusted_targets(self):
        rows=self.rows.copy();rows[5,0]+=.0001
        with self.assertRaisesRegex(ValueError,'ACT source'):self.entry.validate_plan(self.plan,rows)
        plan=copy.deepcopy(self.plan);plan['source_indices'][4]=plan['source_indices'][3]
        with self.assertRaises(ValueError):self.entry.validate_plan(plan,self.rows)

    def test_current_health_checks_remain_active(self):
        health=dict(state=self.plan['initial_state'],flange_pose=[0,0,.479,0,0,0],
            feedback=dict(arm_status=0,err_code=0,ctrl_mode=1,motion_status=0))
        self.entry.validate_health(health,self.plan,self.plan['initial_state'])
        for kind in ['floor','error','tracking']:
            value=copy.deepcopy(health)
            if kind=='floor':value['flange_pose'][2]=.349
            if kind=='error':value['feedback']['err_code']=1
            if kind=='tracking':value['state'][1]+=.02
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                self.entry.validate_health(value,self.plan,self.plan['initial_state'])


if __name__=='__main__':unittest.main()
