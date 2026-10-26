import copy
import importlib
import importlib.util
import json
import numpy as np
import pytest
import torch
from tests.test_history_policy import fixtures
from tests.test_visual_policy import statistics


def api():
    assert importlib.util.find_spec('training.history'), 'Paired history training interface missing'
    return importlib.import_module('training.history')


def output_fixture(tmp_path):
    (tmp_path/'checkpoints').mkdir();(tmp_path/'artifacts').mkdir()
    checkpoint=tmp_path/'checkpoints/policy_last.ckpt';checkpoint.write_bytes(b'complete-test-weights')
    (tmp_path/'checkpoints/dataset_stats.pkl').write_bytes(b'paired-stats')
    source=tmp_path/'input';source.write_bytes(b'frozen-input')
    from training.integrity import snapshot
    return checkpoint,snapshot([source])


def test_budget_999_cannot_finalize_1000(tmp_path):
    m=api();checkpoint,protected=output_fixture(tmp_path)
    config=dict(training_budget={'kind':'updates','limit':1000},num_epochs=0)
    with pytest.raises(ValueError):m.finalize_history_training(tmp_path,config,dict(updates=999,epochs=0),protected)
    assert not (tmp_path/'artifacts/training_integrity.json').exists()


def test_pilot_is_not_twenty_epochs_or_deployable(tmp_path):
    m=api();checkpoint,protected=output_fixture(tmp_path)
    config=dict(training_budget={'kind':'updates','limit':1000},num_epochs=0)
    proof=m.finalize_history_training(tmp_path,config,dict(updates=1000,epochs=0),protected)
    assert proof['actual_epochs']==0 and proof['actual_updates']==1000
    assert proof['hardware_deployment_allowed'] is False
    from training.integrity import require_finalized
    with pytest.raises(ValueError):require_finalized(checkpoint,config)
    assert require_finalized(checkpoint,config,allowed_budgets=('updates',)).is_file()


def test_same_batches_and_initial_physical_function():
    m=api()
    a=m.make_batch_order(77,batch_size=32,budget={'kind':'updates','limit':1000},seed=42)
    b=m.make_batch_order(77,batch_size=32,budget={'kind':'updates','limit':1000},seed=42)
    assert len(a)==1000;np.testing.assert_array_equal(a,b)
    c,q,image,h=fixtures();c["nonlinear_gripper_head"]=True
    from executor.model import BoundedActPolicy,make_policy_observation,physical_actions
    old=statistics(c);new={k:v.copy() for k,v in old.items()}
    for k in ['qpos_std','action_std','geometry_std']:new[k]*=1.2
    for k in ['qpos_mean','action_mean','geometry_mean']:new[k]+=.002
    original=BoundedActPolicy(dict(c,numeric_history_profile='none')).eval()
    source=dict(config=dict(c,numeric_history_profile='none'),model=original.state_dict())
    configs=[];policies=[]
    for variant in ['A','B']:
        cfg,weights=m.initialize_history(source,old,new,dict(device='cpu',pose_loss_weight=0.),variant=variant)
        p=m.build_history_policy(cfg).eval();p.load_state_dict(weights,strict=True)
        configs.append(cfg);policies.append(p)
    state=np.r_[np.zeros(7),.074];features=np.full(6,.3)
    qo=torch.from_numpy(make_policy_observation(state,old,c,.065,geometry_features_m=features))[None]
    qn=torch.from_numpy(make_policy_observation(state,new,configs[0],.065,geometry_features_m=features))[None]
    history=dict(observations=qn[:,None].expand(-1,5,-1).clone(),age_s=torch.tensor([[.4,.3,.2,.1,0]]),valid=torch.ones(1,5,dtype=torch.bool))
    with torch.no_grad():
        expected=physical_actions(original(qo,image[:1]),old,bounded_gripper=True)
        for p,variant in zip(policies,['A','B']):
            got=physical_actions(p(qn,image[:1],**({'history':history} if variant=='B' else {})),new,bounded_gripper=True)
            assert float((got[...,:7]-expected[...,:7]).abs().max())<=1e-6
            assert float((got[...,7]-expected[...,7]).abs().max())<=1e-7


def test_full_ACT_groups_have_gradients():
    m=api();c,q,image,h=fixtures();c.update(train_nonlinear_gripper_only=False,train_arm_output_and_gripper_only=False,nonlinear_gripper_head=True)
    p=m.build_history_policy(c).train()
    optimizer_rates={g['name']:g['lr'] for g in p.optimizer.param_groups}
    assert optimizer_rates=={'policy':1e-5,'visual':1e-5,'gripper':3e-4,'history':1e-4}
    target=torch.ones(2,20,8)*.1;padding=torch.zeros(2,20,dtype=torch.bool)
    p(q,image,target,padding,history=h)['loss'].backward()
    from training.visual import gradient_evidence
    proof=gradient_evidence(p)
    assert proof['posterior_grad_norm']>0 and not proof['frozen_backbone_has_grad']
    assert float(p.model.action_head.weight.grad[:7].abs().sum())>0
    assert any(v.grad is not None and v.grad.abs().sum()>0 for v in p.gripper_adapter.parameters())


def test_validation_and_test_data_do_not_select_pilot_checkpoint():
    m=api()
    assert m.selected_checkpoint_name({'kind':'updates','limit':1000})=='policy_last.ckpt'
    assert m.selected_checkpoint_name({'kind':'epochs','limit':20})=='policy_best_val.ckpt'


def test_protocol_frozen_before_optimizer_includes_history_runtime_and_assets(tmp_path):
    m=api()
    assert hasattr(m,'freeze_history_protocol'), 'Pre-optimization protocol seal missing'
    protocol=dict(parameters_frozen_before_optimization=True,resolved_cases=[dict(name=f'case_{i}',seed=i,
        query_stride=2,servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False,
        observation_delay_steps=1,scene={'physical_scene':i}) for i in range(16)])
    source=tmp_path/'source.json';source.write_text(json.dumps(protocol))
    frozen=m.freeze_history_protocol(source,dict(profile='history_common_v1',maximum_prefix=2),tmp_path/'sealed.json')
    assert frozen['resolved_cases']==protocol['resolved_cases']
    assert all(r['observation_delay_steps']==1 for r in frozen['resolved_cases'])
    assert frozen['history_runtime_contract']['profile']=='history_common_v1'
    assert frozen['origin_protocol_sha256']
    with pytest.raises(FileExistsError):m.freeze_history_protocol(source,{},tmp_path/'sealed.json')


def test_train_entry_rejects_missing_seal_before_creating_a_run(tmp_path,monkeypatch):
    m=api()
    assert hasattr(m,'train_history'), 'Paired training entry missing'
    monkeypatch.setattr(m,'ROOT',tmp_path/'unsealed')
    with pytest.raises(ValueError,match='seal|freeze'):
        m.train_history(tmp_path/'data',tmp_path/'initializer',dict(variant='A',stage='pilot'),tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_formal_export_binds_the_history_profile(tmp_path):
    import pickle
    m=api();_,protected=output_fixture(tmp_path)
    c,q,image,h=fixtures();c.update(num_epochs=20,training_budget={'kind':'epochs','limit':20},
                                  training_integrity_required=True)
    from executor.model import BoundedActPolicy
    checkpoint=tmp_path/'checkpoints/policy_best_val.ckpt'
    torch.save(dict(config=c,model=BoundedActPolicy(c).state_dict(),epoch=3),checkpoint)
    (tmp_path/'checkpoints/dataset_stats.pkl').write_bytes(pickle.dumps(statistics(c)))
    m.finalize_history_training(tmp_path,c,dict(updates=20,epochs=20),protected)
    (tmp_path/'artifacts/training_summary.json').write_text(json.dumps({'last_metrics':{'epoch':20}}))
    from training.export import export_model
    metadata=export_model(checkpoint,tmp_path/'export')
    assert metadata.get('numeric_history_profile')=='causal_numeric_gru5_v1'
    assert metadata['numeric_history_shape']==[5,15]
    assert metadata['training_budget']=={'kind':'epochs','limit':20}
    assert metadata['hardware_deployment_allowed'] is False
    schema=json.loads((tmp_path/'export/input_action_schema.json').read_text())
    assert schema['execution_prefix_points']==2
    assert schema['numeric_history_shape']==[5,15]


def test_twenty_epoch_counter_cannot_hide_missing_optimizer_steps(tmp_path):
    m=api();checkpoint,protected=output_fixture(tmp_path)
    checkpoint.rename(tmp_path/'checkpoints/policy_best_val.ckpt')
    config=dict(training_budget={'kind':'epochs','limit':20},num_epochs=20,declared_optimizer_steps=40)
    with pytest.raises(ValueError):m.finalize_history_training(tmp_path,config,dict(updates=39,epochs=20),protected)
