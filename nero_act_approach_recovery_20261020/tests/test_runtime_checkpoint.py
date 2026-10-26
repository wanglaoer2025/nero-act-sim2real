"""A complete offline checkpoint must resume physics and supervisory state."""
import importlib.util
import importlib
import json
from pathlib import Path
import random

import mujoco
import numpy as np
import pytest
import torch


def api():
    assert importlib.util.find_spec('simulation.runtime_checkpoint'), 'Complete runtime checkpoint missing'
    return importlib.import_module('simulation.runtime_checkpoint')


def fixture():
    from simulation.scene import build_scene
    from simulation.nero_domain_randomization import NeroServo
    c=json.loads((Path(__file__).parents[1]/'scene/center.json').read_text())
    model=build_scene(c);data=mujoco.MjData(model)
    data.qpos[:7]=c['initial_robot_state']['qpos_rad'];data.qpos[7:9]=[.0475,-.0475]
    mujoco.mj_forward(model,data)
    servo=NeroServo(model,data,c);previous=np.r_[data.qpos[:7],.095]
    control=dict(contact={'state':'holding','first_contact_stamp':.5},
        phases={'retreat_confirmed':False},monitor={'max_lift':.01},
        attachment={'center':np.array([0,0,-.05]),'rotation':np.eye(3),'airborne':True},
        support_pose=None,active_points=7,global_physics_points=143,events=[],
        last_submission_sim_s=0.,confirmed_sim_s=0.,scenario_rng_state=np.random.default_rng(42).bit_generator.state,
        rejections=0,guard_diagnostic=None)
    observation=dict(exposures={'required':1,'rows':[]},history=None,receipts=[],history_epoch=0,
        pending={'image':np.zeros((90,160,3),np.uint8),'observed':previous.astype(np.float32)},
        last_observation_stamp=0.)
    return model,data,servo,previous,control,observation


def test_resume_preserves_next_prediction_and_guard():
    m=api();model,data,servo,p,c,o=fixture();bindings={'scope':'offline_simulation','source':'test'}
    np.random.seed(1);random.seed(2);torch.manual_seed(3)
    record=m.capture_runtime(model,data,servo,p,control_state=c,observation_state=o,bindings=bindings)
    expected=(np.random.rand(),random.random(),torch.rand(1))
    target=p.copy();target[0]+=.001
    servo.command(target[:7],target[7])
    for _ in range(25):servo.apply();mujoco.mj_step(model,data)
    continuous=data.qpos.copy()
    restored,rc,ro=m.restore_runtime(model,data,servo,json.loads(json.dumps(record)),expected_bindings=bindings)
    assert (np.random.rand(),random.random())==expected[:2]
    torch.testing.assert_close(torch.rand(1),expected[2],rtol=0,atol=0)
    np.testing.assert_array_equal(restored,p)
    np.testing.assert_array_equal(ro['pending']['image'],o['pending']['image'])
    servo.command(target[:7],target[7])
    for _ in range(25):servo.apply();mujoco.mj_step(model,data)
    np.testing.assert_allclose(data.qpos,continuous,atol=1e-7,rtol=0)
    assert rc['contact']==c['contact'] and rc['active_points']==7


def test_resume_preserves_disturbance_and_liftoff_latch():
    m=api();model,data,servo,p,c,o=fixture();bindings={'scope':'offline_simulation'}
    record=m.capture_runtime(model,data,servo,p,control_state=c,observation_state=o,bindings=bindings)
    c['attachment']['airborne']=False;c['global_physics_points']=0
    _,restored,_=m.restore_runtime(model,data,servo,record,expected_bindings=bindings)
    assert restored['attachment']['airborne'] is True
    assert restored['global_physics_points']==143


def test_incomplete_or_live_checkpoint_is_rejected():
    m=api();model,data,servo,p,c,o=fixture();bindings={'scope':'offline_simulation'}
    with pytest.raises(ValueError):m.capture_runtime(model,data,servo,p,control_state={},observation_state=o,bindings=bindings)
    with pytest.raises(ValueError):m.capture_runtime(model,data,servo,p,control_state=c,observation_state=o,bindings={'scope':'real'})
    record=m.capture_runtime(model,data,servo,p,control_state=c,observation_state=o,bindings=bindings)
    with pytest.raises(ValueError):m.restore_runtime(model,data,servo,record,expected_bindings={'scope':'offline_simulation','other':'source'})
    record['control']['active_points']=100
    with pytest.raises(ValueError):m.restore_runtime(model,data,servo,record,expected_bindings=bindings)


def test_capture_trace_comparison_rejects_changed_action_or_failure(tmp_path):
    assert importlib.util.find_spec('simulation.history_capture'), 'Failure capture CLI missing'
    from simulation.history_capture import verify_trace_reference
    for name in ['a','b']:
        folder=tmp_path/name;folder.mkdir()
        (folder/'result.json').write_text(json.dumps(dict(active_policy_points=2,error='guard_stop',success=False)))
        np.savez(folder/'rollout.npz',raw_chunks=np.zeros((1,20,8)),sent_targets=np.zeros((2,8)))
    assert verify_trace_reference(tmp_path/'a',tmp_path/'b')['matches_reference']
    np.savez(tmp_path/'b/rollout.npz',raw_chunks=np.ones((1,20,8)),sent_targets=np.zeros((2,8)))
    with pytest.raises(ValueError):verify_trace_reference(tmp_path/'a',tmp_path/'b')


@pytest.mark.parametrize('jitter',[False,True])
def test_rollout_checkpoint_resumes_the_same_next_prefix(monkeypatch,tmp_path,jitter):
    import inspect
    import simulation.supervised_rollout as runtime
    assert 'capture_control' in inspect.signature(runtime.rollout).parameters, 'Runtime capture integration missing'
    from tests.test_visual_policy import config
    c=config();c.pop('visual_observation_profile');c['policy_input_dim']=9
    scene=json.loads((Path(runtime.__file__).parents[1]/'scene/center.json').read_text())
    scenario=dict(seed=42,jitter=jitter,force=False,servo_delay_steps=0,query_stride=2,observation_age_s=.05)
    class Renderer:
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def update_scene(self,*a,**k):pass
        scene=type('Scene',(),{'flags':np.zeros(100,dtype=bool)})()
        def render(self):return np.zeros((90,160,3),np.uint8)
    monkeypatch.setattr(runtime.mujoco,'Renderer',Renderer)
    class Policy:
        device='cpu';bounded_gripper=False
        def __init__(self,limit):self.config=c;self.limit=limit;self.calls=0
        def __call__(self,qpos,image):
            self.calls+=1
            if self.calls>self.limit:raise RuntimeError('test_end')
            return qpos[:,:8,None].transpose(1,2).expand(-1,20,-1).clone()
    stats={key:np.zeros(8,np.float32) if key.endswith('mean') else np.ones(8,np.float32)
           for key in ['qpos_mean','qpos_std','action_mean','action_std']}
    runtime.rollout(scene,Policy(3),stats,scenario,tmp_path/'original',capture_control=True)
    saved=json.loads((tmp_path/'original/runtime_checkpoints.json').read_text())
    checkpoint=saved['records'][0] if not jitter else saved['named']['initial']
    runtime.rollout(scene,Policy(1 if not jitter else 2),stats,scenario,tmp_path/'resumed',resume_context=checkpoint,capture_control=True)
    a=np.load(tmp_path/'original/rollout.npz');b=np.load(tmp_path/'resumed/rollout.npz')
    offset=2 if not jitter else 0
    np.testing.assert_array_equal(a['raw_chunks'][offset:offset+len(b['raw_chunks'])],b['raw_chunks'])
    np.testing.assert_allclose(a['sent_targets'][offset*10:offset*10+len(b['sent_targets'])],b['sent_targets'],atol=1e-7,rtol=0)
