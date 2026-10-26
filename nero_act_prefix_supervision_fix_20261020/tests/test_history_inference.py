import copy
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from tests.test_visual_features import valid_inputs
from tests.test_visual_policy import config,statistics


def make_worker(variant='B'):
    from executor.visual_inference import VisualInferenceWorker
    f,g,p,cal=valid_inputs();c=config()
    c.update(calibration_sha256=cal['calibration_sha256'],history_runtime_profile='history_common_v1',
             numeric_history_profile='causal_numeric_gru5_v1' if variant=='B' else 'none',session_id='s')
    now=[10.02];index=[0];seen=[]
    def capture():
        frame=copy.copy(f);frame.frame_id=100+index[0];frame.exposure_monotonic_s=10+index[0]*.05
        frame.received_monotonic_s=frame.exposure_monotonic_s+.01
        now[0]=frame.exposure_monotonic_s+.02;index[0]+=1;return frame
    def measure(frame):
        geometry=copy.copy(g);geometry.frame_id=frame.frame_id;return geometry
    def feedback():return [dict(monotonic_s=now[0]-.02,position=p['state'])]
    def predict(rgb,state,previous,**kwargs):
        seen.append(kwargs);return np.tile(state,(20,1))
    policy=SimpleNamespace(config=c,stats=statistics(c),metadata={},predict=predict)
    worker=VisualInferenceWorker(policy,cal,SimpleNamespace(measure=measure),
        SimpleNamespace(pose=lambda _:dict(tcp=np.array([0,0,.1]))),capture=capture,feedback=feedback,
        clock=lambda:now[0],health=lambda:None,spin=lambda dt:now.__setitem__(0,now[0]+dt))
    worker.record_submission(dict(receipt_id='initial_ack',rows=[[0]*7+[.095]],sent_s=9.8,ack_s=9.81,
                                  accepted=True,kind='submitted_prefix',scope='s'))
    worker.prefix_completed('initial_ack',completed_s=9.82)
    return worker,seen,now,index


def test_A_B_share_warmup_and_submission_receipts():
    for variant in ['A','B']:
        worker,seen,now,index=make_worker(variant)
        packet=worker.predict(previous_gripper_target_m=.095,after_s=9.9)
        assert index[0]==5 and len(seen)==1
        batch=packet['history'];batch.verify()
        assert batch.frame_ids==(100,101,102,103,104)
        assert batch.receipt_ids==('initial_ack',)*5
        assert ('history' in seen[0])==(variant=='B')


def test_confirmation_discards_old_history_and_pending_prefix():
    worker,seen,now,index=make_worker()
    before=worker.predict(previous_gripper_target_m=.095,after_s=9.9)
    worker.confirmation_received(now_s=now[0])
    after=worker.predict(previous_gripper_target_m=.095,after_s=now[0])
    assert after['history'].epoch==before['history'].epoch+1
    assert index[0]==10 and len(seen)==2
    assert set(before['history'].frame_ids).isdisjoint(after['history'].frame_ids)


def test_packet_detects_mutated_past_frame():
    from executor.visual_inference import verify_prediction_packet
    worker,*_=make_worker();packet=worker.predict(previous_gripper_target_m=.095,after_s=9.9)
    verify_prediction_packet(packet)
    bad=packet['history'].observations.copy();bad[0,0]+=1
    packet['history']=replace(packet['history'],observations=bad)
    with pytest.raises(ValueError):verify_prediction_packet(packet)


def test_history_is_not_admitted_during_an_unfinished_prefix():
    worker,seen,now,index=make_worker()
    worker.record_submission(dict(receipt_id='new',rows=[[0]*7+[.094]],sent_s=9.9,ack_s=9.91,
                                  accepted=True,kind='submitted_prefix',scope='s'))
    with pytest.raises(ValueError):worker.predict(previous_gripper_target_m=.094,after_s=9.9)
    assert not seen and index[0]==0


def test_delayed_midprefix_exposure_is_excluded_without_losing_past():
    worker,seen,now,index=make_worker()
    before=worker.predict(previous_gripper_target_m=.095,after_s=9.9)
    worker.record_submission(dict(receipt_id='new',rows=[[0]*7+[.095]],sent_s=10.22,ack_s=10.23,
        accepted=True,kind='submitted_prefix',scope='s'))
    now[0]=10.32
    worker.prefix_completed('new',completed_s=10.32)
    after=worker.predict(previous_gripper_target_m=.095,after_s=10.22)
    assert len(seen)==2
    assert after['frame'].exposure_monotonic_s>10.32
    assert 105 not in after['history'].frame_ids and 106 not in after['history'].frame_ids
    assert set(before['history'].frame_ids)&set(after['history'].frame_ids)


def test_prediction_detaches_camera_and_geometry_before_inference():
    from executor.visual_inference import digest,verify_prediction_packet
    worker,seen,*_=make_worker()
    capture=worker.capture;measure=worker.observer.measure;shared={};model_input=[]
    def external_frame():
        frame=capture();shared['frame']=frame;return frame
    def external_geometry(frame):
        geometry=measure(frame);shared['geometry']=geometry;return geometry
    worker.capture=external_frame;worker.observer.measure=external_geometry
    def overwrite(rgb,state,previous,**kwargs):
        model_input.append(digest(rgb));shared['frame'].rgb[0,0,0]=255
        shared['geometry'].cup_bottom_base_m[0]=99
        return np.tile(state,(20,1))
    worker.policy.predict=overwrite
    packet=worker.predict(previous_gripper_target_m=.095,after_s=9.9)
    verify_prediction_packet(packet)
    assert packet['details']['rgb_sha256']==model_input[0]
    assert packet['geometry'].cup_bottom_base_m[0]!=99
    with pytest.raises(ValueError):packet['frame'].rgb.setflags(write=True)
