import importlib
import copy
import hashlib
import json
import time
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def api():return importlib.import_module('automatic_sampling')


class StaticRobot:
    def __init__(self):self.clock=100.
    def read_health(self,**_):
        self.clock+=.05
        return dict(monotonic=self.clock,timestamp=self.clock,feedback_timestamps=[self.clock]*19,
            state=[0.]*7+[.095],flange_pose=[.1,.2,.4,0.,0.,0.],
            feedback=dict(arm_status=2,err_code=0,ctrl_mode=1,motion_status=0))


def vision():
    K=np.array([[600.,0.,320.],[0.,600.,180.],[0.,0.,1.]])
    h=.095/2;obj=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]])
    rv=Rotation.from_euler('xyz',[2.6,.2,.1]).as_rotvec()
    pixels=cv2.projectPoints(obj,rv,np.array([.01,-.02,.6]),K,None)[0]
    class Detector:
        def detectMarkers(self,_):return [pixels],np.array([[0]]),[]
    return K,Detector()


def test_stale_frame_is_skipped_without_relaxing_50ms(monkeypatch):
    m=api();robot=StaticRobot();K,detector=vision();payload={};reads=[]
    def image():
        stamp=robot.clock+(-.08 if not reads else .025);reads.append(stamp)
        return np.zeros((360,640,3),np.uint8),int(stamp*1e9)
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    m.acquire_pair(payload,robot,image,detector,K,role='fit')
    assert len(payload['frame_candidates'])==2
    assert payload['frame_candidates'][0]['max_pair_offset_ms']>100
    assert payload['frame_candidates'][1]['max_pair_offset_ms']<30
    assert payload['sample']['exposure_clock_verified'] is False
    assert payload['sample']['health_after']['feedback']['arm_status']==2
    assert payload['image_stamp_ns']==int(reads[-1]*1e9)


def test_persistently_stale_stream_cannot_produce_a_sample(monkeypatch):
    m=api();robot=StaticRobot();K,detector=vision();payload={}
    image=lambda:(np.zeros((360,640,3),np.uint8),int((robot.clock-.08)*1e9))
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    with pytest.raises(TimeoutError):m.acquire_pair(payload,robot,image,detector,K,role='fit')
    assert payload.get('sample') is None
    assert len(payload['frame_candidates'])<=30
    assert payload['image_stamp_ns']>0


def test_send_intent_is_durable_before_callback_and_return_separate(tmp_path):
    m=api();path=tmp_path/'events.jsonl'
    def hardware_boundary():
        event=json.loads(path.read_text().splitlines()[-1])
        assert event['kind']=='arm' and event['state']=='attempted'
        raise OSError('partial hardware submission')
    with pytest.raises(OSError):m.audited_call(path,'arm',{'target':[.001]*7},hardware_boundary)
    events=[json.loads(line) for line in path.read_text().splitlines()]
    assert events[0]['state']=='attempted'
    assert events[-1]['state']=='failed'
    assert not any(e['state']=='returned' for e in events)


def route():
    return json.loads((Path(__file__).resolve().parents[1]/'automatic_sampling_01/nominal_plan.json').read_text())


def test_full_route_has_distinct_15_fit_5_kept_out_and_100mm_coverage():
    p=route();api().validate_route(p)
    assert [s['role'] for s in p['stages']].count('fit')==15
    assert [s['role'] for s in p['stages']].count('pose_check')==5
    assert p['fit_position_span_m'][0]>.0999 and p['fit_position_span_m'][1]>.0999


@pytest.mark.parametrize('fault',['role','collision','step','speed','source','goal'])
def test_modified_route_rejected_before_hardware(fault):
    p=copy.deepcopy(route())
    if fault=='role':p['stages'][-1]['role']='fit'
    elif fault=='collision':p['nominal_route_passed']=False
    elif fault=='step':p['stages'][0]['waypoints_rad'][0][0]+=.03
    elif fault=='speed':p['speed_percent']=5
    elif fault=='goal':p['stages'][1]['goal_transform'][0][3]+=.02
    else:p['source_sha256'][next(iter(p['source_sha256']))]='0'*64
    with pytest.raises(ValueError):api().validate_route(p)


def test_camera_guard_uses_actual_marker_and_stops_on_displacement():
    m=api();p=route();root=Path(__file__).resolve().parents[1]
    diagnostic=json.loads((root/'timing_probe_01/report.json').read_text())
    image=cv2.imread(str(root/'timing_probe_01/optical_007.png'))
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners,ids,_=detector.detectMarkers(image)
    assert ids.ravel().tolist()==[0]
    pixels=np.asarray(corners[0]).reshape(4,2)
    m.guard_marker(p,p['initial_state'],pixels,np.array(diagnostic['K']))
    with pytest.raises(RuntimeError):m.guard_marker(p,p['initial_state'],pixels+[80.,0.],np.array(diagnostic['K']))


@pytest.mark.parametrize('camera_lost',[False,True])
def test_motion_stage_is_arm_only_and_missing_marker_sends_no_target(tmp_path,camera_lost):
    m=api();p=route();commands=[]
    class Robot:
        def __init__(self):
            self.state=np.array(p['initial_state']);self.allow_no_solution=True;self.max_age_s=.15;self.sdk=self
        def read_health(self,**_):
            T=m.flange_fk(self.state[:7]);pose=np.r_[T[:3,3],Rotation.from_matrix(T[:3,:3]).as_euler('xyz')]
            return dict(state=self.state.tolist(),flange_pose=pose.tolist(),feedback_timestamps=[time.time()]*19,
                feedback=dict(arm_status=2,err_code=0,ctrl_mode=1,motion_status=0))
        def set_speed_percent(self,value):assert value==1
        def move_j(self,q):commands.append(q);self.state[:7]=q
        def stop_sending(self):pass
    robot=Robot()
    class Camera:
        K=np.array([[263.7,0.,323.58],[0.,263.7,179.92],[0.,0.,1.]])
        def read(self):return np.zeros((360,640,3),np.uint8),int(time.time()*1e9)
        def show(self,*_):pass
        def corners(self,image):
            if camera_lost:return None
            CM=np.linalg.inv(p['provisional_camera_base'])@m.flange_fk(robot.state[:7])@p['provisional_mount_hypotheses'][0]
            h=.095/2;obj=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]])
            return cv2.projectPoints(obj,Rotation.from_matrix(CM[:3,:3]).as_rotvec(),CM[:3,3],self.K,None)[0].reshape(4,2)
    stage=copy.deepcopy(p['stages'][0]);stage['waypoints_rad']=stage['waypoints_rad'][:2]
    if camera_lost:
        with pytest.raises(RuntimeError):m.drive_stage(robot,Camera(),p,stage,tmp_path/'events.jsonl',time.monotonic()+10)
        assert commands==[]
    else:
        m.drive_stage(robot,Camera(),p,stage,tmp_path/'events.jsonl',time.monotonic()+10)
        assert len(commands)==2
        np.testing.assert_array_equal(robot.state[7],p['initial_state'][7])


def test_auto_entry_help_has_no_hardware_side_effects():
    script=Path(api().__file__)
    r=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True)
    assert r.returncode==0 and '--execute' in r.stdout and '--capture-current' in r.stdout


def test_truncated_audit_still_closes_devices_and_preserves_motion_possible(tmp_path):
    m=api();log=tmp_path/'control_events.jsonl'
    log.write_text('{"kind":"arm","state":"attempted"}\n{"kind":"arm","state":"ret')
    closed=[]
    class Camera:
        def close(self):closed.append('camera')
    class Robot:
        def stop_sending(self):closed.append('stop')
        def disconnect(self):closed.append('disconnect')
    report={}
    m.finish(tmp_path,report,Camera(),Robot(),log,lambda:1241)
    saved=json.loads((tmp_path/'manifest.json').read_text())
    assert closed==['stop','camera','disconnect']
    assert saved['running'] is False and saved['motion_sent_or_possible'] is True
    assert saved['arm_send_attempts']==1 and saved['audit_errors']


@pytest.mark.parametrize('delay_at',['camera','final_health'])
def test_deadline_crossing_before_sdk_boundary_sends_no_arm_target(tmp_path,monkeypatch,delay_at):
    m=api();p=route();clock=[0.];commands=[]
    class SDK:
        def set_speed_percent(self,value):pass
        def move_j(self,q):commands.append(q)
    class Robot:
        allow_no_solution=True;max_age_s=.15
        def __init__(self):self.sdk=SDK();self.calls=0
        def read_health(self,**_):
            self.calls+=1
            # prepare reads twice; the next read is ControlledStart's final pre-send health.
            if delay_at=='final_health' and self.calls==5:clock[0]=2.
            return dict(state=p['initial_state'],feedback_timestamps=[time.time()]*19,
                feedback=dict(arm_status=2,err_code=0,ctrl_mode=1,motion_status=0))
        def stop_sending(self):pass
    robot=Robot();original=robot.sdk
    monkeypatch.setattr(m.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    monkeypatch.setattr(m,'live_guard',lambda *_:None)
    def camera(*_):
        if delay_at=='camera':clock[0]=2.
    monkeypatch.setattr(m,'observe_motion',camera)
    stage=copy.deepcopy(p['stages'][0]);stage['waypoints_rad']=stage['waypoints_rad'][:1]
    with pytest.raises(TimeoutError):m.drive_stage(robot,None,p,stage,tmp_path/'events.jsonl',1.)
    assert commands==[] and robot.sdk is original


def test_plan_hash_matches_parsed_snapshot_when_path_changes(tmp_path,monkeypatch):
    m=api();path=tmp_path/'plan.json';original=json.dumps(route()).encode();path.write_bytes(original)
    validate=m.validate_route
    def validation_then_external_replacement(plan):
        validate(plan);path.write_text('{"different":"later replacement"}')
    monkeypatch.setattr(m,'validate_route',validation_then_external_replacement)
    parsed,digest=m.load_route(path)
    assert parsed['waypoint_count']==1641
    assert digest==hashlib.sha256(original).hexdigest()
    assert digest!=hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('bounds',[[[float('nan'),0],[0,1],[0,1]],[[1,0],[0,1],[0,1]],[[0,1],[0,1]]])
def test_invalid_workspace_cannot_disable_guard_before_hardware(bounds):
    p=route();p['flange_workspace_m']=bounds
    with pytest.raises(ValueError):api().validate_route(p)


def test_vision_stop_preserves_exact_native_frame_and_health(tmp_path):
    m=api();p=route();camera=m.Camera.__new__(m.Camera)
    camera.cv2=cv2;camera.output=tmp_path;camera.K=np.array([[263.7,0,323.6],[0,263.7,179.9],[0,0,1]])
    camera.detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    native=np.full((360,640,3),123,np.uint8);stamp=1792652500123456789
    camera.read=lambda:(native,stamp)
    camera.show=lambda *_:pytest.fail('Do not overwrite missing-marker diagnostic with a passed preview')
    health=dict(state=p['initial_state'],feedback=dict(arm_status=2,err_code=0,ctrl_mode=1,motion_status=0))
    with pytest.raises(RuntimeError,match='Exactly one complete ID0'):m.observe_motion(camera,p,health)
    np.testing.assert_array_equal(cv2.imread(str(tmp_path/'vision_stop_optical.png')),native)
    report=json.loads((tmp_path/'vision_stop.json').read_text())
    assert report['image_stamp_ns']==stamp and report['health']==health
    assert report['marker_ids'] is None and report['no_new_target_allowed'] is True


def test_real_stop_frame_keeps_valid_id0_when_background_has_id17():
    m=api();root=Path(__file__).resolve().parents[1]
    folder=root/'automatic_sampling_01/remainder_execution_01'
    failure=json.loads((folder/'vision_stop.json').read_text())
    frame=cv2.imread(str(folder/'vision_stop_optical.png'))
    camera=m.Camera.__new__(m.Camera)
    params=cv2.aruco.DetectorParameters();params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    camera.detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),params)
    pixels=camera.corners(frame)
    assert pixels is not None
    assert np.mean(pixels[:,0])>350 and np.mean(pixels[:,1])>250
    m.guard_marker(route(),failure['health']['state'],pixels,np.array(failure['K']))
