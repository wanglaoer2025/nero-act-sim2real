"""History must affect the prior and posterior without changing physical anchors."""
import copy
import numpy as np
import pytest
import torch
from tests.test_visual_policy import config,statistics

PROFILE='causal_numeric_gru5_v1'


def fixtures():
    torch.set_num_threads(1);torch.manual_seed(42)
    c=dict(config(),hidden_dim=128,dim_feedforward=256,numeric_history_profile=PROFILE,history_lr=1e-4)
    q=torch.randn(2,15)*.1
    h=dict(observations=torch.randn(2,5,15)*.1,age_s=torch.tensor([[.4,.3,.2,.1,0]]*2),valid=torch.ones(2,5,dtype=torch.bool))
    h['observations'][:,-1]=q
    return c,q,torch.rand(2,1,3,90,160),h


def test_zero_history_projection_preserves_physical_policy():
    from executor.model import BoundedActPolicy,physical_actions
    c,q,image,h=fixtures()
    source=BoundedActPolicy(dict(c,numeric_history_profile='none')).eval()
    target=BoundedActPolicy(c).eval()
    assert hasattr(target,'history_encoder'), 'GRU history condition missing'
    state=target.state_dict();state.update(source.state_dict());target.load_state_dict(state,strict=True)
    with torch.no_grad():
        raw=physical_actions(target(q,image,history=h),statistics(c),bounded_gripper=True)
        old=physical_actions(source(q,image),statistics(c),bounded_gripper=True)
    assert raw.shape==(2,20,8)
    assert float((raw[...,:7]-old[...,:7]).abs().max())<=1e-6
    assert float((raw[...,7]-old[...,7]).abs().max())<=1e-7


def test_second_step_trains_gru():
    from executor.model import BoundedActPolicy
    c,q,image,h=fixtures();p=BoundedActPolicy(c).eval()
    assert hasattr(p,'history_encoder'), 'GRU history condition missing'
    optimizer=torch.optim.Adam(p.history_encoder.parameters(),lr=1e-4)
    target=torch.ones(2,20,8)*.1
    def gradient():
        p.zero_grad(set_to_none=True)
        loss=(p(q,image,history=h)-target).square().mean();loss.backward()
    gradient()
    assert float(p.history_encoder.projection.weight.grad.abs().sum())>0
    assert float(p.history_encoder.gru.weight_ih_l0.grad.abs().sum())==0
    optimizer.step();gradient()
    assert float(p.history_encoder.gru.weight_ih_l0.grad.abs().sum())>0


def test_prior_and_posterior_both_receive_history():
    from training.train import FullTaskPolicy
    c,q,image,h=fixtures();p=FullTaskPolicy(c)
    assert hasattr(p,'history_encoder'), 'GRU history condition missing'
    with torch.no_grad():p.history_encoder.projection.weight.fill_(.01)
    changed=copy.deepcopy(h);changed['observations'][:,:-1]+=.5
    targets=torch.ones(2,20,8)*.1;padding=torch.zeros(2,20,dtype=torch.bool)
    p.eval()
    with torch.no_grad():
        a=p(q,image,history=h);b=p(q,image,history=changed)
        v=p(q,image,targets,padding,history=h)
        v2=p(q,image,targets+1,padding,history=h)
    assert not torch.equal(a,b)
    torch.testing.assert_close(v['a_hat'],v2['a_hat'],atol=0,rtol=0)
    p.train();torch.manual_seed(42);x=p(q,image,targets,padding,history=h)
    torch.manual_seed(42);y=p(q,image,targets,padding,history=changed)
    assert x['kl']>0 and not torch.equal(x['kl'],y['kl'])
    x['loss'].backward()
    assert p.history_encoder.projection.weight.grad is not None


def test_legacy_and_history_profile_load_strictly():
    from executor.model import BoundedActPolicy
    c,q,image,h=fixtures();p=BoundedActPolicy(c)
    assert hasattr(p,'history_encoder'), 'GRU history condition missing'
    with pytest.raises(ValueError):p(q,image)
    with pytest.raises(ValueError):BoundedActPolicy(dict(c,numeric_history_profile='unknown'))
    bad=copy.deepcopy(h);bad['valid'][0,0]=False
    with pytest.raises(ValueError):p(q,image,history=bad)
    bad=copy.deepcopy(h);bad['observations'][0,-1,0]+=1
    with pytest.raises(ValueError):p(q,image,history=bad)
    old=BoundedActPolicy(dict(c,numeric_history_profile='none'))
    with pytest.raises(RuntimeError):old.load_state_dict(p.state_dict(),strict=True)
    with pytest.raises(ValueError):old(q,image,history=h)


def test_history_worker_warmup_and_prediction_require_real_history(tmp_path):
    import hashlib,json,pickle
    from executor.model import BoundedActPolicy,make_policy_observation
    from executor.policy_worker import FrozenAct
    from tests.test_history_contract import item
    from executor.observation_history import HistoryBuffer
    c,_,_,_=fixtures();stats=statistics(c);model=BoundedActPolicy(c)
    assert hasattr(model,'history_encoder')
    torch.save(dict(config=c,model=model.state_dict()),tmp_path/'policy.ckpt')
    (tmp_path/'dataset_stats.pkl').write_bytes(pickle.dumps(stats))
    metadata=dict(task_scope='complete_pick_carry_place_retreat',action_chunk_shape=[20,8],
        numeric_history_profile=PROFILE,numeric_history_shape=[5,15],
        checkpoint_sha256=hashlib.sha256((tmp_path/'policy.ckpt').read_bytes()).hexdigest(),
        stats_sha256=hashlib.sha256((tmp_path/'dataset_stats.pkl').read_bytes()).hexdigest())
    (tmp_path/'runtime_metadata.json').write_text(json.dumps(metadata))
    worker=FrozenAct(tmp_path)
    state=np.r_[np.zeros(7),.074].astype(np.float32);features=np.full(6,.3,np.float32)
    observed=make_policy_observation(state,stats,c,.065,geometry_features_m=features)
    buffer=HistoryBuffer();buffer.clear(epoch=1,after_s=9.9)
    for i in range(5):
        row=item(i);row['observation']=observed;buffer.append(row)
    raw=worker.predict(np.zeros((90,160,3),np.uint8),state,.065,geometry_features_m=features,history=buffer.snapshot(observed))
    assert raw.shape==(20,8) and np.isfinite(raw).all()
    with pytest.raises(ValueError):worker.predict(np.zeros((90,160,3),np.uint8),state,.065,geometry_features_m=features)
    assert not hasattr(worker,'history_buffer')  # synthetic warmup is not admitted camera history
