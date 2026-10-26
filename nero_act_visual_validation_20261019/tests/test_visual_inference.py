from types import SimpleNamespace
import hashlib
import copy

import numpy as np
import pytest

from tests.test_visual_features import valid_inputs
from tests.test_visual_policy import config,statistics


def api():
    import importlib.util
    assert importlib.util.find_spec('executor.visual_inference') is not None, 'Same-frame15D inference worker missing'
    import executor.visual_inference as m
    return m


def test_visual_worker_binds_same_rgb_geometry_feedback_and_previous_target():
    m=api();frame,geometry,paired,cal=valid_inputs();c=config();c['calibration_sha256']=cal['calibration_sha256']
    stats=statistics(c);received=[]
    def predict(rgb,state,previous,*,geometry_features_m):
        received.append((rgb.copy(),state.copy(),previous,geometry_features_m.copy()))
        raw=np.tile(state,(20,1));raw[:,7]=previous+np.arange(20)*.0001
        return raw
    policy=SimpleNamespace(config=c,stats=stats,metadata=dict(checkpoint_sha256='fixture'),predict=predict)
    observer=SimpleNamespace(measure=lambda _:geometry)
    robot=SimpleNamespace(pose=lambda _:dict(tcp=np.array([0.,0.,.1])))
    worker=m.VisualInferenceWorker(policy,cal,observer,robot,capture=lambda:frame,
        feedback=lambda:[dict(monotonic_s=paired['monotonic_s'],position=paired['state'])],clock=lambda:10.02)
    packet=worker.predict(previous_gripper_target_m=.09,after_s=9.9)
    assert packet['details']['policy_input_dim']==15
    assert packet['details']['previous_gripper_target_m']==.09
    np.testing.assert_allclose(received[0][3],[.1,.2,-.1,.3,.4,-.1])
    np.testing.assert_array_equal(received[0][0],frame.rgb)
    m.verify_prediction_packet(packet)
    from executor.visual_control import VisualControlCore
    from executor.visual_dispatch import VisualDispatcher
    from tests.test_visual_dispatch import Transport
    core=VisualControlCore(c,cal,np.r_[paired['state'][:7],.09]);clock=[10.02];port=Transport(clock)
    VisualDispatcher(core,port,clock=lambda:clock[0]).dispatch_prediction(packet)
    assert core.active_points==2 and len(port.sent)==1
    original_time=frame.exposure_monotonic_s
    frame.exposure_monotonic_s+=.01
    with pytest.raises(ValueError,match='binding|changed'):m.verify_prediction_packet(packet)
    frame.exposure_monotonic_s=original_time
    packet['raw'][0,0]+=.001
    with pytest.raises(ValueError,match='binding|changed'):m.verify_prediction_packet(packet)


def test_worker_cannot_reuse_preconfirmation_or_occluded_frame():
    m=api();frame,geometry,paired,cal=valid_inputs();c=config();c['calibration_sha256']=cal['calibration_sha256']
    policy=SimpleNamespace(config=c,stats=statistics(c),metadata={},predict=lambda *a,**k:pytest.fail('must not infer invalid frame'))
    for invalid in ['stale','occluded']:
        g=copy.copy(geometry)
        if invalid=='occluded':g.valid=False
        worker=m.VisualInferenceWorker(policy,cal,SimpleNamespace(measure=lambda _:g),None,
            capture=lambda:frame,feedback=lambda:[dict(monotonic_s=paired['monotonic_s'],position=paired['state'])],clock=lambda:10.02)
        with pytest.raises(ValueError):worker.predict(previous_gripper_target_m=.095,after_s=10.01 if invalid=='stale' else 9.9)


def test_nominal_training_calibration_cannot_be_silently_replaced_for_live_inference():
    m=api();*_,cal=valid_inputs();c=config();c['calibration_sha256']='different_old_nominal_profile'
    with pytest.raises(ValueError,match='calibration'):
        m.VisualInferenceWorker(SimpleNamespace(config=c),cal,None,None,capture=None,feedback=None)
