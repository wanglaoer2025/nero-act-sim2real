import numpy as np
import pytest


def pose(z):return dict(position=np.array([0.,0.,z]),rotation=np.eye(3))


def attachment():
    from simulation.supported_attachment import SupportedAttachment
    return SupportedAttachment(radius=.0375,half_height=.0475)


def test_attachment_uses_current_observed_cup_pose_instead_of_initial_source():
    a=attachment();p=pose(.1);cup=np.array([.008,.009,.0475])
    a.observe(p,cup,np.eye(3),supported=True)
    np.testing.assert_allclose(a.center,cup-p['position'])
    assert abs(a.bottom(p,0))<1e-12
    assert a.bottom(pose(.10003),0)>0
    assert a.bottom(pose(.09999),0)<-1e-6


def test_table_supported_compliance_updates_current_transform_without_lifting_targets():
    a=attachment();a.observe(pose(.1),np.array([0.,0.,.0475]),np.eye(3),supported=True)
    # Joint settling while table carries the cup must not move an imagined cup through the table.
    a.observe(pose(.09995),np.array([0.,0.,.0475]),np.eye(3),supported=True)
    assert abs(a.bottom(pose(.09995),0))<1e-12
    assert a.bottom(pose(.09997),0)>0


def test_first_airborne_transform_is_frozen_and_later_support_does_not_reset_slip_guard():
    a=attachment();a.observe(pose(.1),np.array([0.,0.,.0475]),np.eye(3),supported=True)
    a.observe(pose(.11),np.array([0.,0.,.0575]),np.eye(3),supported=False)
    center=a.center.copy();rotation=a.rotation.copy()
    a.observe(pose(.12),np.array([.01,0.,.0475]),np.eye(3),supported=True)
    np.testing.assert_array_equal(a.center,center);np.testing.assert_array_equal(a.rotation,rotation)
    assert a.airborne is True and a.bottom(pose(.099),0)<-1e-6


def test_invalid_measured_rotation_cannot_enter_attachment():
    a=attachment()
    with pytest.raises(ValueError):a.observe(pose(.1),np.zeros(3),np.zeros((3,3)),supported=True)


def test_rollout_latches_lift_and_landing_between_queries(monkeypatch,tmp_path):
    import json,torch
    from pathlib import Path
    import simulation.supervised_rollout as runtime
    from tests.test_visual_policy import config
    scene=json.loads((Path(runtime.__file__).resolve().parents[1]/'scene/center.json').read_text())
    c=config();c.pop('visual_observation_profile');c['policy_input_dim']=9;c['reach_deadline_steps']=3200
    initial=np.r_[scene['initial_robot_state']['qpos_rad'],scene['initial_robot_state']['gripper_total_width_m']]
    seen=[];base=runtime.SupportedAttachment
    class Tracked(base):
        def __init__(self,**kw):super().__init__(**kw);seen.append(self)
    monkeypatch.setattr(runtime,'SupportedAttachment',Tracked)
    class Contact:
        state='holding';close_authorized=True;carry_authorized=True;release_authorized=False
        contact_width=.074;first_contact_stamp=0.
        def update(self,*args):pass
    monkeypatch.setattr(runtime,'PhysicsContact',Contact)
    monkeypatch.setattr(runtime.PhaseGuard,'inspect',lambda *args:None)
    monkeypatch.setattr(runtime.PhaseGuard,'single_point_grasp_prefix',lambda *args:False)
    monkeypatch.setattr(runtime,'first_lift_index',lambda *args:0)
    original=runtime.observe;count=0
    def measured(*args,**kw):
        nonlocal count
        count+=1;m=original(*args,**kw)
        # Initial query sees table support; first physics step loses it;
        # landing happens before the next query can observe the cup.
        m['supported']=count!=2;m['unexpected_contact']=False
        return m
    monkeypatch.setattr(runtime,'observe',measured)
    class Renderer:
        def __init__(self,*args,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def update_scene(self,*args,**kw):pass
        scene=type('Scene',(),{'flags':np.zeros(100,dtype=bool)})()
        def render(self):return np.zeros((90,160,3),dtype=np.uint8)
    monkeypatch.setattr(runtime.mujoco,'Renderer',Renderer)
    class Policy:
        device='cpu';bounded_gripper=False;calls=0
        def __init__(self):self.config=c
        def __call__(self,*args):
            self.calls+=1
            if self.calls>1:raise RuntimeError('integration_test_end')
            return torch.tensor(np.tile(initial,(1,20,1)),dtype=torch.float32)
    stats={n:np.zeros(8,np.float32) if n.endswith('mean') else np.ones(8,np.float32)
           for n in ['qpos_mean','qpos_std','action_mean','action_std']}
    result=runtime.rollout(scene,Policy(),stats,dict(seed=42,jitter=False,force=False,
        servo_delay_steps=0,observation_age_s=.05,query_stride=2),tmp_path/'rollout')
    assert result['active_policy_points']==10 and 'integration_test_end' in result['error']
    assert seen[0].airborne is True, 'Lift observed inside advance must permanently freeze attachment'
