import copy
import ast
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np

import air_trial
from air_trial import validate_plan,validate_health


class AirTrialBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((Path(__file__).parent/'preparation_01/plan.json').read_text())
        self.rows=np.load(self.plan['targets_path'],allow_pickle=False)
        self.health=dict(state=self.plan['initial_state'],flange_pose=[0,0,.479,0,0,0],
            feedback=dict(arm_status=2,err_code=0,ctrl_mode=1,motion_status=0))

    def test_model_path_is_admitted_only_inside_the_high_arm_scope(self):
        validate_plan(self.plan,self.rows)
        for field,value in [('speed_percent',6),('gripper_commands',1),('enable_commands',1),('complete_task_not_released',False)]:
            bad=copy.deepcopy(self.plan);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_plan(bad,self.rows)

    def test_low_flange_path_is_rejected(self):
        bad=copy.deepcopy(self.plan);bad['sdk_flange_poses'][0][2]=.449
        with self.assertRaises(ValueError):validate_plan(bad,self.rows)

    def test_joint_jump_or_gripper_motion_is_rejected(self):
        for column,change in [(1,.01),(7,-.003)]:
            bad=self.rows.copy();bad[5,column]+=change
            with self.subTest(column=column),self.assertRaises(ValueError):validate_plan(self.plan,bad)

    def test_current_tracking_height_and_error_guards_remain_active(self):
        for kind in ['tracking','height','error','gripper']:
            bad=copy.deepcopy(self.health)
            if kind=='tracking':bad['state'][1]+=.02
            if kind=='height':bad['flange_pose'][2]=.449
            if kind=='error':bad['feedback']['err_code']=1
            if kind=='gripper':bad['state'][7]-=.003
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                validate_health(bad,self.plan,self.plan['initial_state'])

    def test_status2_alone_does_not_remove_other_protections(self):
        validate_health(self.health,self.plan,self.plan['initial_state'])

    def test_loaded_plan_and_targets_are_bound_to_their_own_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);target=root/'targets.npy';np.save(target,self.rows)
            plan=copy.deepcopy(self.plan);plan['targets_path']=str(target)
            plan['source_sha256'][str(target)]=hashlib.sha256(target.read_bytes()).hexdigest()
            path=root/'plan.json';raw=json.dumps(plan).encode();path.write_bytes(raw)
            loaded,rows,digest=air_trial.load_bound_plan(path)
            self.assertEqual(digest,hashlib.sha256(raw).hexdigest());np.testing.assert_array_equal(rows,self.rows)
            target.write_bytes(b'changed target bytes')
            with self.assertRaises(ValueError):air_trial.load_bound_plan(path)

    def test_failed_receiver_cannot_send_even_a_valid_next_target(self):
        tree=ast.parse(Path(air_trial.__file__).read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='receive')
        namespace=dict(np=np,time=time,rows=self.rows,plan=self.plan,digest='sealed',report={'events':[],'arm_attempts':0},
            dump=lambda *args:None,args=SimpleNamespace(output=Path('/unused')),Int32=lambda **kw:kw)
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual receive>','exec'),namespace)
        sdk=Mock();robot=SimpleNamespace(sdk=sdk,stop_sending=Mock())
        driver=SimpleNamespace(index=0,failure=RuntimeError('already stopped'),robot=robot,
            previous=np.array(self.plan['initial_state']),health=lambda _:self.health,ack=Mock())
        epoch=time.time();msg=SimpleNamespace(name=[f'joint{i}' for i in range(1,8)],position=self.rows[0,:7],
            header=SimpleNamespace(frame_id='sealed:0',stamp=SimpleNamespace(sec=int(epoch),nanosec=int((epoch%1)*1e9))))
        namespace['receive'](driver,msg)
        sdk.set_speed_percent.assert_not_called();sdk.move_j.assert_not_called()

    def test_expired_target_reports_exact_admission_cause_without_sending(self):
        tree=ast.parse(Path(air_trial.__file__).read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='receive')
        clock=iter([100.,100.251]);report={'events':[],'arm_attempts':0};sent=[]
        namespace=dict(np=np,time=SimpleNamespace(time=lambda:next(clock)),rows=self.rows,plan=self.plan,digest='sealed',
            report=report,dump=lambda *args:None,args=SimpleNamespace(output=Path('/unused')),Int32=lambda **kw:kw)
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual receive>','exec'),namespace)
        sdk=SimpleNamespace(move_j=lambda row:sent.append(row))
        robot=SimpleNamespace(sdk=sdk,stop_sending=lambda:None)
        driver=SimpleNamespace(index=1,failure=None,robot=robot,previous=self.rows[0],health=lambda _:self.health,ack=None)
        msg=SimpleNamespace(name=[f'joint{i}' for i in range(1,8)],position=self.rows[1,:7],
            header=SimpleNamespace(frame_id='sealed:1',stamp=SimpleNamespace(sec=100,nanosec=0)))
        namespace['receive'](driver,msg)
        self.assertEqual(sent,[])
        self.assertIn('stale_command',report['error'])
        self.assertGreater(report['events'][0]['admission']['command_age_s'],.25)
        self.assertEqual(report['events'][0]['send_health']['feedback']['motion_status'],0)


if __name__=='__main__':unittest.main()
