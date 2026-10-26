import importlib
import pytest


def test_wait_100ms_checks_health_and_never_reuses_prediction():
    m=importlib.import_module('executor.visual_inference')
    assert hasattr(m,'await_fresh_observation'), 'Bounded fresh observation wait missing'
    now=[1.];health=[]
    def spin(dt):now[0]+=dt
    with pytest.raises(TimeoutError):m.await_fresh_observation(lambda:dict(exposure_s=.9),
        after_s=1.,deadline_s=1.1,clock=lambda:now[0],health=lambda:health.append(now[0]),spin=spin)
    assert now[0]<=1.1+1e-9 and len(health)>1
    now[0]=1.
    fresh=m.await_fresh_observation(lambda:dict(exposure_s=1.01),after_s=1.,deadline_s=1.1,
        clock=lambda:now[0],health=lambda:None,spin=spin)
    assert fresh['exposure_s']>1.


def test_wait_propagates_independent_health_failure():
    m=importlib.import_module('executor.visual_inference')
    assert hasattr(m,'await_fresh_observation')
    def bad():raise RuntimeError('feedback stale')
    with pytest.raises(RuntimeError,match='feedback stale'):
        m.await_fresh_observation(lambda:None,after_s=1.,deadline_s=1.1,clock=lambda:1.,health=bad,spin=lambda _:None)


def test_A_B_share_history_warmup_and_actual_delayed_exposure(monkeypatch,tmp_path):
    import json
    from pathlib import Path
    import numpy as np
    import torch
    import simulation.supervised_rollout as runtime
    import simulation.visual_evaluation as visual
    from tests.test_visual_policy import config as policy_config
    class Renderer:
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def update_scene(self,*a,**k):pass
        scene=type('Scene',(),{'flags':np.zeros(100,dtype=bool)})()
        def render(self):return np.zeros((180,320,3),np.uint8)
    monkeypatch.setattr(runtime.mujoco,'Renderer',Renderer)
    class Features:
        def __init__(self,*a,**k):pass
        def __call__(self,image,observed):return np.zeros(6,np.float32),{}
    monkeypatch.setattr(visual,'ImageFeatures',Features)
    scene=json.loads((Path(runtime.__file__).parents[1]/'scene/center.json').read_text())
    stats={n:np.zeros(8,np.float32) if n.endswith('mean') else np.ones(8,np.float32)
           for n in ['qpos_mean','qpos_std','action_mean','action_std']}
    stats.update(geometry_mean=np.zeros(6,np.float32),geometry_std=np.ones(6,np.float32))
    records=[]
    for variant in ['A','B']:
        class Policy:
            device='cpu';bounded_gripper=False;calls=0
            config=dict(policy_config(),numeric_history_profile='causal_numeric_gru5_v1' if variant=='B' else 'none',
                        history_runtime_profile='history_common_v1')
            def __call__(self,q,image,**kwargs):
                self.calls+=1
                if self.calls>1:raise RuntimeError('test_end')
                if variant=='B':
                    assert kwargs['history']['observations'].shape==(1,5,15)
                    torch.testing.assert_close(kwargs['history']['observations'][:,-1],q,atol=0,rtol=0)
                else:assert not kwargs
                return q[:,:8,None].transpose(1,2).expand(-1,20,-1).clone()
        result=runtime.rollout(scene,Policy(),stats,dict(seed=42,jitter=False,force=False,
            servo_delay_steps=0,query_stride=2,observation_age_s=.05,observation_delay_steps=1),
            tmp_path/variant,runtime_profile='history_common_v1',capture_control=True)
        assert result['active_policy_points']==2, result['error']
        saved=np.load(tmp_path/variant/'rollout.npz');obs=json.loads((tmp_path/variant/'visual_observations.json').read_text())
        assert obs[0]['consumed_simulation_s']-obs[0]['exposure_simulation_s']==pytest.approx(.05)
        assert saved['query_simulation_time_s'][0]>=.2
        records.append((saved['raw_chunks'],saved['query_simulation_time_s']))
        if variant=='B':
            import hashlib
            checkpoint=json.loads((tmp_path/variant/'runtime_checkpoints.json').read_text())['named']['initial']
            checkpoint['bindings']=dict(scope='offline_simulation',source_kind='teacher_interruption',parent_source_sha256='verified-parent')
            checkpoint['observation'].update(exposures=None,history=None,pending=None)
            unsigned={k:v for k,v in checkpoint.items() if k!='sha256'}
            checkpoint['sha256']=hashlib.sha256(json.dumps(unsigned,sort_keys=True,allow_nan=False).encode()).hexdigest()
            resumed=runtime.rollout(scene,Policy(),stats,dict(seed=42,jitter=False,force=False,
                servo_delay_steps=0,query_stride=2,observation_age_s=.05,observation_delay_steps=1),
                tmp_path/'teacher_resume',runtime_profile='history_common_v1',resume_context=checkpoint,
                resume_policy_transition=True,expected_parent_source_sha256='verified-parent')
            assert resumed['new_active_points']==2,resumed['error']
    for a,b in zip(*records):np.testing.assert_array_equal(a,b)
