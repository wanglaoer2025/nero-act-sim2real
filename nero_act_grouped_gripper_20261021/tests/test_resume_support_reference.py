"""Recorded failed ACT outputs exercise real MuJoCo dynamics; only RGB is synthetic."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from tests.test_visual_policy import config


def test_teacher_release_resume_hands_off_measured_reference(monkeypatch,tmp_path):
    import simulation.supervised_rollout as runtime
    import simulation.visual_evaluation as visual
    fixture=json.loads((Path(__file__).parent/'fixtures/support_release_regression.json').read_text())
    class Renderer:
        scene=type('Scene',(),{'flags':np.zeros(100,dtype=bool)})()
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def update_scene(self,*a,**k):pass
        def render(self):return np.zeros((180,320,3),np.uint8)
    class Features:
        def __init__(self,*a,**k):pass
        def __call__(self,image,observed):return np.zeros(6,np.float32),{}
    monkeypatch.setattr(runtime.mujoco,'Renderer',Renderer);monkeypatch.setattr(visual,'ImageFeatures',Features)
    class Policy:
        device='cpu';bounded_gripper=False;calls=0
        config=dict(config(),numeric_history_profile='none',history_runtime_profile='history_common_v1')
        def __call__(self,q,image,**kwargs):
            if self.calls==len(fixture['raw_chunks']):raise RuntimeError('recorded_sequence_complete')
            output=torch.tensor(fixture['raw_chunks'][self.calls],dtype=torch.float32)[None];self.calls+=1
            return output
    stats={k:np.zeros(8,np.float32) if k.endswith('mean') else np.ones(8,np.float32) for k in ['qpos_mean','qpos_std','action_mean','action_std']}
    stats.update(geometry_mean=np.zeros(6,np.float32),geometry_std=np.ones(6,np.float32))
    checkpoint=fixture['checkpoint'];unsigned={k:v for k,v in checkpoint.items() if k!='sha256'}
    checkpoint['sha256']=hashlib.sha256(json.dumps(unsigned,sort_keys=True,allow_nan=False).encode()).hexdigest()
    result=runtime.rollout(fixture['scene'],Policy(),stats,dict(seed=42,jitter=False,force=False,
        servo_delay_steps=0,query_stride=2,observation_age_s=.05),tmp_path/'resume',
        runtime_profile='history_common_v1',resume_context=checkpoint,resume_policy_transition=True,
        expected_parent_source_sha256=checkpoint['bindings']['parent_source_sha256'])
    assert result['error']=='RuntimeError: recorded_sequence_complete',result['error']
    assert result['feedback_hold_points']>0
