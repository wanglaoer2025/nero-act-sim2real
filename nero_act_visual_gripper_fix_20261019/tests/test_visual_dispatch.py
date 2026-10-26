from concurrent.futures import Future
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_visual_runtime import controller


def api():
    import importlib.util
    assert importlib.util.find_spec('executor.visual_dispatch') is not None, 'Visual live dispatch/ROS transport missing'
    import executor.visual_dispatch as m
    return m


class Transport:
    def __init__(self, clock): self.clock=clock; self.sent=[]; self.cleaned=[]; self.health_calls=0
    def health(self): self.health_calls+=1
    def validate(self, rows): pass
    def submit(self, rows, request_id):
        self.sent.append(rows.copy()); self.clock[0]+=.001
        future=Future(); future.set_result(True)
        return dict(sent_s=self.clock[0],ack_s=self.clock[0]+.001,accepted=True,results=[future])
    def complete(self, results, deadline): pass
    def latch_stop(self): self.cleaned.append('stop')
    def cancel_all(self): self.cleaned.append('cancel')
    def restore_readonly(self): self.cleaned.append('readonly')
    def spin(self, seconds): self.clock[0]+=seconds


def test_ros_adapter_sends_only_current_two_points_and_advances_after_ack():
    m=api(); core,raw,frame,geometry,paired=controller(); clock=[10.02]; transport=Transport(clock)
    runner=m.VisualDispatcher(core,transport,clock=lambda:clock[0])
    runner.dispatch(raw,frame,geometry,paired)
    assert transport.sent[0].shape==(2,8) and core.active_points==2
    np.testing.assert_array_equal(core.history,transport.sent[0][-1])


def test_camera_failure_or_late_collision_check_never_sends():
    m=api()
    for issue in ['occluded','slow_collision']:
        core,raw,frame,geometry,paired=controller(); clock=[10.02]; transport=Transport(clock)
        if issue=='occluded':geometry.valid=False
        else:transport.validate=lambda rows:clock.__setitem__(0,10.4)
        runner=m.VisualDispatcher(core,transport,clock=lambda:clock[0])
        with pytest.raises((ValueError,RuntimeError)):runner.dispatch(raw,frame,geometry,paired)
        assert not transport.sent and core.stopped


def test_cleanup_restores_readonly_even_if_cancel_throws():
    m=api();core,*_=controller();transport=Transport([10.02])
    def cancel(): transport.cleaned.append('cancel'); raise RuntimeError('cancel failed')
    transport.cancel_all=cancel
    runner=m.VisualDispatcher(core,transport)
    report=runner.close()
    assert transport.cleaned==['stop','cancel','readonly'] and core.stopped
    assert report['automatic_retry'] is False and len(report['cleanup_errors'])==1


def test_health_runs_while_inference_is_pending_and_failure_latches_stop():
    m=api();core,*_=controller();clock=[10.02];transport=Transport(clock)
    runner=m.VisualDispatcher(core,transport,clock=lambda:clock[0]);future=Future()
    def spin(seconds):
        clock[0]+=seconds
        if transport.health_calls==3:future.set_result('fresh_prediction')
    transport.spin=spin
    assert runner.await_prediction(future)=='fresh_prediction' and transport.health_calls>=3
    transport.health=lambda:(_ for _ in ()).throw(RuntimeError('health stale'))
    with pytest.raises(RuntimeError,match='health'):runner.await_prediction(Future())
    assert core.stopped


def test_unqualified_nominal_training_and_failed_rollout_deny_before_ros_construction():
    m=api()
    from tests.test_visual_features import valid_inputs
    *_,cal=valid_inputs()
    bad=dict(complete=True,finalized=True,passed=0,total=16,hardware_deployment_allowed=False)
    with pytest.raises(ValueError,match='16|qualified|nominal|calibration'):
        m.require_deployment_evidence(cal,bad,{},now_wall_s=10)


def test_ros_transport_waits_for_both_results_and_checks_failures():
    m=api();calls=[]
    futures=[Future(),Future()];futures[0].set_result(SimpleNamespace(result=SimpleNamespace(error_code=0),status=4))
    class Session:
        def guard(self): calls.append('health')
        def spin(self,t):
            if not futures[1].done():futures[1].set_result(SimpleNamespace(result=SimpleNamespace(error_code=-1),status=4))
    transport=m.RosVisualTransport(Session(),[],clock=lambda:10.)
    with pytest.raises(RuntimeError,match='result|controller'):transport.complete(futures,10.5)
    assert 'health' in calls


def test_already_completed_goals_still_require_current_health_and_deadline():
    m=api();calls=[];f=Future();f.set_result(SimpleNamespace(status=4,result=SimpleNamespace(error_code=0)))
    session=SimpleNamespace(guard=lambda:calls.append('health'))
    late=m.RosVisualTransport(session,[],clock=lambda:10.6)
    with pytest.raises(TimeoutError):late.complete([f,f],10.5)
    assert calls
    session.guard=lambda:(_ for _ in ()).throw(RuntimeError('stale health'))
    with pytest.raises(RuntimeError,match='health'):
        m.RosVisualTransport(session,[],clock=lambda:10.1).complete([f,f],10.5)
