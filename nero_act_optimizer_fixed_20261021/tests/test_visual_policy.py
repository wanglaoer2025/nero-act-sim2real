import copy

import numpy as np
import pytest
import torch

from tests.test_cvae_training import CvaeTrainingTests

PROFILE='visual_relative_base_v1'


def config():
    return CvaeTrainingTests().config(policy_input_dim=15,visual_observation_profile=PROFILE,
        hidden_dim=32,dim_feedforward=64,nheads=4,enc_layers=1,dec_layers=1,
        geometry_mean=[.1]*6,geometry_std=[.2]*6)


def statistics(c):
    return {**{k:np.array(v,np.float32) for k,v in c['constraint_loss_stats'].items()},
            'geometry_mean':np.array(c['geometry_mean'],np.float32),'geometry_std':np.array(c['geometry_std'],np.float32)}


def test_old_8_9_and_new_15_profiles_are_explicit():
    from executor.contract import FullTaskContract
    from executor.model import make_policy_observation
    c=config();stats=statistics(c);state=np.r_[np.zeros(7),.074]
    FullTaskContract(c)
    observed=make_policy_observation(state,stats,c,.065,geometry_features_m=np.array([.3]*6))
    assert observed.shape==(15,)
    np.testing.assert_allclose(observed[9:],[1.]*6,atol=1e-6)
    assert observed[8]==pytest.approx(-.9,abs=1e-6)
    for change in [dict(policy_input_dim=9),dict(visual_observation_profile='unknown'),dict(physical_state_dim=15)]:
        with pytest.raises(ValueError):FullTaskContract(dict(c,**change))
    with pytest.raises(ValueError):make_policy_observation(state,stats,c,.065)
    bad=dict(stats,geometry_std=np.zeros(6))
    with pytest.raises(ValueError):make_policy_observation(state,bad,c,.065,geometry_features_m=np.ones(6))


def test_init_9_to_15_preserves_same_image_prediction():
    from executor.model import BoundedActPolicy
    from training import command_context
    assert callable(getattr(command_context,'expand_visual_state_dict',None)), 'Explicit9→15 initializer expansion missing'
    torch.set_num_threads(1);torch.manual_seed(42)
    c=config();legacy=dict(c,policy_input_dim=9);legacy.pop('visual_observation_profile')
    old=BoundedActPolicy(legacy).eval();new=BoundedActPolicy(c).eval()
    expanded=command_context.expand_visual_state_dict(old.state_dict())
    new.load_state_dict(expanded,strict=True)
    q=torch.rand(1,9);image=torch.rand(1,1,3,90,160)
    with torch.no_grad():
        a=old(q,image);b=new(torch.cat([q,torch.ones(1,6)],1),image)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    with pytest.raises(ValueError):command_context.expand_visual_state_dict(expanded)


def test_readout_uses_only_first_eight_channels():
    from executor.model import BoundedActPolicy
    torch.set_num_threads(1);policy=BoundedActPolicy(config()).eval()
    with torch.no_grad():
        policy.model.action_head.weight.zero_();policy.model.action_head.bias.zero_()
        q=torch.zeros(2,15);q[1,9:]=100
        prediction=policy(q,torch.zeros(2,1,3,90,160))
    assert prediction.shape==(2,20,8)
    torch.testing.assert_close(prediction[0],prediction[1],atol=0,rtol=0)


def test_cvae_encoder_and_prior_accept_15():
    from training.train import FullTaskPolicy
    torch.set_num_threads(1);torch.manual_seed(42)
    policy=FullTaskPolicy(config()).train()
    q=torch.ones(1,15)*.2;image=torch.rand(1,1,3,90,160)
    target=torch.linspace(-.05,.05,160).reshape(1,20,8);padding=torch.zeros(1,20,dtype=torch.bool)
    out=policy(q,image,target,padding)
    assert out['a_hat'].shape==(1,20,8) and float(out['kl'].detach())>0
    out['loss'].backward()
    for layer in [policy.model.encoder_joint_proj,policy.model.input_proj_robot_state]:
        assert layer.weight.shape[1]==15 and float(layer.weight.grad[:,9:].abs().sum())>0
    assert float(policy.model.latent_proj.weight.grad.abs().sum())>0
    policy.eval()
    with torch.no_grad():
        a=policy(q,image,target,padding);b=policy(q,image,target+10,padding)
    torch.testing.assert_close(a['a_hat'],b['a_hat'],rtol=0,atol=0)
    assert float(a['kl'])==0


def test_frozen_worker_builds_the_same_15d_observation():
    from executor.model import BoundedActPolicy,make_policy_observation,physical_actions
    from executor.policy_worker import FrozenAct
    torch.set_num_threads(1);c=config();worker=object.__new__(FrozenAct)
    worker.config=c;worker.device='cpu';worker.stats=statistics(c);worker.model=BoundedActPolicy(c).eval()
    state=np.r_[np.zeros(7),.074];geometry=np.array([.3]*6);rgb=np.zeros((90,160,3),np.uint8)
    actual=worker.predict(rgb,state,.065,geometry_features_m=geometry)
    assert actual.shape==(20,8) and np.isfinite(actual).all()
    with pytest.raises(ValueError):worker.predict(rgb,state,.065)


def test_rebase_15d_initializer_keeps_physical_predictions():
    from executor.model import BoundedActPolicy,make_policy_observation,physical_actions
    from training.rebase_initializer import paired_config,rebase_initializer
    torch.set_num_threads(1);torch.manual_seed(42)
    c=config();old=statistics(c);new=copy.deepcopy(old)
    for name in ['qpos_std','action_std','geometry_std']:new[name]*=1.2
    for name in ['qpos_mean','action_mean','geometry_mean']:new[name]+=.002
    after_config=paired_config(c,new)
    a=BoundedActPolicy(c).eval();b=BoundedActPolicy(after_config).eval()
    b.load_state_dict(rebase_initializer(a.state_dict(),old,new,c))
    state=np.r_[np.zeros(7),.074];geometry=np.array([.3]*6);image=torch.rand(1,1,3,90,160)
    qa=make_policy_observation(state,old,c,.065,geometry_features_m=geometry)
    qb=make_policy_observation(state,new,after_config,.065,geometry_features_m=geometry)
    with torch.no_grad():
        pa=physical_actions(a(torch.from_numpy(qa)[None],image),old,bounded_gripper=True)
        pb=physical_actions(b(torch.from_numpy(qb)[None],image),new,bounded_gripper=True)
    torch.testing.assert_close(pa,pb,atol=2e-6,rtol=2e-5)
