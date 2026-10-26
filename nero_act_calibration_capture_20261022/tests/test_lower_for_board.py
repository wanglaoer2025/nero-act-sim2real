"""The height entry must never accept an altered or larger reviewed motion."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest


def api():
    return importlib.import_module('lower_for_board')


def plan():
    fixture=json.loads((Path(__file__).resolve().parents[1]/'height_adjust_01/plan.json').read_text())
    # Current offline fixture; the immutable historical plan is never reauthorized.
    fixture['source_sha256']={source:hashlib.sha256(Path(source).read_bytes()).hexdigest()
                              for source in fixture['source_sha256']}
    return fixture


def test_historical_motion_plan_rejects_the_changed_capture_source():
    historical=json.loads((Path(__file__).resolve().parents[1]/'height_adjust_01/plan.json').read_text())
    with pytest.raises(ValueError,match='Reviewed planning source changed'):
        api().validate_plan(historical)


def test_reviewed_20mm_plan_is_arm_only():
    m=api(); p=plan()
    m.validate_plan(p)
    assert len(p['waypoints_rad']) == 16
    assert p['gripper_commands'] == 0


def test_shorter_framing_adjustment_retains_the_20mm_upper_bound():
    p=plan();p['drop_m']=.010
    p['waypoints_rad']=p['waypoints_rad'][:8]
    p['flange_poses_fk']=p['flange_poses_fk'][:8]
    p['waypoint_count']=8
    api().validate_plan(p)
    assert abs(p['initial_flange_fk'][2]-p['flange_poses_fk'][-1][2]-.010)<1e-5


@pytest.mark.parametrize('drop',[0.,-.01,.03,float('nan')])
def test_invalid_declared_framing_distance_rejected(drop):
    p=plan();p['drop_m']=drop
    with pytest.raises(ValueError):api().validate_plan(p)


@pytest.mark.parametrize('fault',['large_step','nan','wrong_speed','upward','sideways','rotation','gripper','source_changed'])
def test_invalid_motion_is_rejected_before_hardware(fault):
    m=api();p=copy.deepcopy(plan())
    if fault=='large_step':p['waypoints_rad'][0][0]+=.05
    elif fault=='nan':p['waypoints_rad'][0][0]=float('nan')
    elif fault=='wrong_speed':p['speed_percent']=5
    elif fault=='upward':p['flange_poses_fk'][-1][2]+=.04
    elif fault=='sideways':p['flange_poses_fk'][-1][0]+=.01
    elif fault=='rotation':p['flange_poses_fk'][-1][3]+=.05
    elif fault=='gripper':p['gripper_commands']=1
    else:p['source_sha256']['/home/user/zyh/scripts/visible_demo_start.py']='0'*64
    with pytest.raises(ValueError):m.validate_plan(p)


def test_feedback_drift_and_gripper_change_stop_before_next_target():
    m=api();p=plan();q=np.array(p['initial_state'])
    m.validate_feedback(q,p,q)
    bad=q.copy();bad[3]+=.02
    with pytest.raises(ValueError):m.validate_feedback(bad,p,q)
    bad=q.copy();bad[7]-=.003
    with pytest.raises(ValueError):m.validate_feedback(bad,p,q)


@pytest.mark.parametrize('boundary',['speed','arm'])
def test_entry_persists_send_intent_at_hardware_submission_boundary(tmp_path,monkeypatch,boundary):
    m=api();p=plan();plan_path=tmp_path/'plan.json'
    plan_path.write_text(json.dumps(p))
    onsite=tmp_path/'onsite.json'
    onsite.write_text(json.dumps(dict(plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        confirmed_epoch_s=time.time(),stop_access=True,whole_arm_and_new_mount_clearance=True,no_other_controller=True)))
    output=tmp_path/'execution'
    calls=[]
    class FakeSDK:
        def fk(self,q):
            for row,pose in zip(p['waypoints_rad'],p['flange_poses_fk']):
                if np.array_equal(q,row):return pose
            raise AssertionError('Unexpected unreviewed FK target')
        def set_speed_percent(self,speed):
            saved=json.loads((output/'execution.json').read_text())
            if boundary=='speed':assert saved.get('speed_setting_attempted') is True
            calls.append('speed')
            if boundary=='speed':raise RuntimeError('stop at speed boundary')
        def move_j(self,q):
            saved=json.loads((output/'execution.json').read_text())
            assert saved['events'][0]['send_attempted'] is True
            assert saved['events'][0]['send_returned'] is False
            calls.append('arm')
            raise RuntimeError('stop at arm boundary')
    class FakeRobot:
        def __init__(self,**_):
            self.sdk=FakeSDK();self.allow_no_solution=True;self.max_age_s=.15
        def connect(self):pass
        def disconnect(self):pass
        def stop_sending(self):pass
        def read_health(self,**_):
            h=copy.deepcopy(p['health_samples'][-1])
            h['feedback_timestamps']=[time.time()]*19
            return h
    monkeypatch.syspath_prepend('/home/user/zyh/src')
    import nero_pi05.robot
    monkeypatch.setattr(nero_pi05.robot,'NeroRobot',FakeRobot)
    read_text=Path.read_text
    def local_read(path,*a,**kw):
        return '907' if str(path)=='/sys/class/net/can0/statistics/tx_packets' else read_text(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',local_read)
    monkeypatch.setattr(sys,'argv',['lower_for_board','--plan',str(plan_path),'--output',str(output),
        '--execute','--onsite-confirmation',str(onsite)])
    with pytest.raises(RuntimeError,match='stop at '+boundary+' boundary'):m.main()
    assert calls==(['speed'] if boundary=='speed' else ['speed','arm'])
