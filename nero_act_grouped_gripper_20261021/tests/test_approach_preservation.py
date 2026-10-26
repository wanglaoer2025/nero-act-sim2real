import copy
import json
import numpy as np
import pytest
import torch

from tests.test_restricted_history_heads import config


def test_preservation_penalizes_deployed_arm_and_gripper_changes_in_physical_units():
    from training.approach_preservation import prefix_preservation_loss
    prediction=torch.zeros(1,20,8,requires_grad=True)
    reference=torch.zeros_like(prediction,requires_grad=True)
    with torch.no_grad():
        prediction[0,0,0]=.0001
        prediction[0,1,7]=.0005
        prediction[0,2:,0]=10.
    loss=prefix_preservation_loss(prediction,reference,{'action_std':[1.]*8,'action_mean':[0.]*8},arm_unit_rad=.0001,gripper_unit_m=.0005)
    assert float(loss.detach())==pytest.approx(.125)
    loss.backward()
    assert prediction.grad[0,0,0]>0 and prediction.grad[0,1,7]>0
    assert prediction.grad[0,2:].abs().sum()==0
    assert reference.grad is None


def test_preservation_uses_the_same_final_gripper_clamp_as_runtime():
    from training.approach_preservation import prefix_preservation_loss
    prediction=torch.zeros(1,20,8,requires_grad=True);reference=torch.zeros_like(prediction)
    with torch.no_grad():prediction[...,7]=.0953;reference[...,7]=.095
    loss=prefix_preservation_loss(prediction,reference,{'action_std':[1.]*8,'action_mean':[0.]*8},arm_unit_rad=.0001,gripper_unit_m=.0005)
    assert float(loss.detach())==pytest.approx(0.,abs=1e-12)


def replay_file(tmp_path,c,q,image):
    from training.history import build_history_policy
    image=image[:,0].contiguous(memory_format=torch.channels_last).unsqueeze(1)
    p=build_history_policy(c).eval()
    with torch.no_grad():target=torch.cat([p(q[i:i+1],image[i:i+1]) for i in range(len(q))]).numpy()
    path=tmp_path/'replay.npz'
    np.savez_compressed(path,current=q.numpy(),image=image.numpy(),target=target)
    return p,path


def test_replay_retains_runtime_image_layout_after_disk_roundtrip(tmp_path):
    from training.approach_preservation import ApproachPreservation
    c,q,image,_=config();_,path=replay_file(tmp_path,c,q,image)
    settings=dict(profile='baseline_prefix_v1',weight=1.,arm_unit_rad=.0001,gripper_unit_m=.0005,batch_size=2,replay_path=str(path))
    keeper=ApproachPreservation(settings,c['constraint_loss_stats'],'cpu')
    selected=keeper.image[torch.tensor([1,0])]
    assert selected[:,0].is_contiguous(memory_format=torch.channels_last)


def test_preservation_matches_single_observation_runtime_queries(tmp_path):
    from training.approach_preservation import ApproachPreservation
    from training.history import build_history_policy
    c,q,image,_=config();p=build_history_policy(c).eval()
    image=image[:,0].contiguous(memory_format=torch.channels_last).unsqueeze(1)
    with torch.no_grad():target=torch.cat([p(q[i:i+1],image[i:i+1]) for i in range(len(q))]).numpy()
    path=tmp_path/'replay.npz';np.savez_compressed(path,current=q.numpy(),image=image.numpy(),target=target)
    keeper=ApproachPreservation(dict(profile='baseline_prefix_v1',weight=1.,arm_unit_rad=.0001,gripper_unit_m=.0005,batch_size=2,replay_path=str(path)),p.loss_stats,'cpu')
    assert float(keeper.loss(p,0).detach())==pytest.approx(0.,abs=1e-12)
    assert keeper.assess(p)['maximum_arm_prefix_change_rad']==0.


def test_replay_gradient_restores_baseline_without_updating_frozen_parameters(tmp_path):
    from training.approach_preservation import ApproachPreservation
    from training.history import frozen_output_head_state,require_frozen_output_head_state
    c,q,image,_=config();p,path=replay_file(tmp_path,c,q,image)
    settings=dict(profile='baseline_prefix_v1',weight=1.,arm_unit_rad=.0001,gripper_unit_m=.0005,batch_size=2,replay_path=str(path))
    keeper=ApproachPreservation(settings,p.loss_stats,'cpu')
    keys=set(p.state_dict());before=frozen_output_head_state(p)
    assert float(keeper.loss(p,0).detach())==pytest.approx(0.,abs=1e-10)
    with torch.no_grad():p.model.action_head.bias[:7].add_(.001)
    p.train();loss=keeper.loss(p,0)
    assert p.training and keeper.last_indices.tolist()==[0,1]
    assert float(loss.detach())>0
    loss.backward()
    assert p.model.action_head.bias.grad[:7].abs().sum()>0
    assert all(v.grad is None for n,v in p.named_parameters() if not v.requires_grad)
    for group in p.optimizer.param_groups:group['lr']=1e-7
    p.optimizer.step();require_frozen_output_head_state(p,before)
    assert set(p.state_dict())==keys
    assert float(keeper.loss(p,0).detach())<float(loss.detach())


@pytest.mark.parametrize('change',[{'weight':0.},{'weight':float('nan')},{'batch_size':0},{'arm_unit_rad':-1.},{'profile':'unknown'}])
def test_invalid_preservation_settings_fail_closed(tmp_path,change):
    from training.approach_preservation import ApproachPreservation
    c,q,image,_=config();_,path=replay_file(tmp_path,c,q,image)
    settings=dict(profile='baseline_prefix_v1',weight=1.,arm_unit_rad=.0001,gripper_unit_m=.0005,batch_size=2,replay_path=str(path));settings.update(change)
    with pytest.raises(ValueError):ApproachPreservation(settings,c['constraint_loss_stats'],'cpu')


def test_caller_cannot_disable_sealed_preservation(tmp_path,monkeypatch):
    import training.history as module
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'_require_seal',lambda path:{})
    (tmp_path/'config').mkdir()
    declared=dict(adaptation_scope='output_heads_v1',train_arm_output_and_gripper_only=True,arm_output_lr=.0001,gripper_adapter_lr=.0003,
        approach_preservation={'profile':'baseline_prefix_v1','weight':1.,'replay_path':'frozen.npz'})
    (tmp_path/'config/history_experiment.json').write_text(json.dumps(declared))
    caller=copy.deepcopy(declared);caller.update(variant='A',stage='pilot');caller.pop('approach_preservation')
    with pytest.raises(ValueError,match='preservation'):
        module.train_history(tmp_path/'data',tmp_path/'initializer',caller,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_seal_rejects_stale_replay_proof_and_changed_reference_source(tmp_path):
    import hashlib
    from training.approach_preservation import validate_approach_replay
    path=tmp_path/'replay.npz';np.savez_compressed(path,current=np.zeros((1,15)))
    baseline=tmp_path/'baseline.json';baseline.write_text(json.dumps(dict(optimizer_updates=0,result={'simulated_events':[dict(kind='GRASP',physics_step=8)]})))
    trace=tmp_path/'rollout.npz';np.savez_compressed(trace,query_steps=np.asarray([4,8]))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    proof=dict(complete=True,profile='baseline_prefix_v1',zero_update_reference=True,reference_optimizer_updates=0,samples=1,
        assessment_role='nominal_development_rehearsal_not_independent_acceptance',
        source_sha256={str(baseline):sha(baseline),str(trace):sha(trace)},replay_sha256=sha(path),
        reference_summary_path=str(baseline),reference_trace_path=str(trace),
        query_indices=[0],cutoff_before_GRASP_physics_step=8)
    write=lambda:path.with_suffix('.json').write_text(json.dumps(proof))
    write();assert str(baseline) in validate_approach_replay(path)
    proof['replay_sha256']='stale';write()
    with pytest.raises(ValueError):validate_approach_replay(path)
    proof['replay_sha256']=sha(path);proof['query_indices']=[2];write()
    with pytest.raises(ValueError):validate_approach_replay(path)
    proof['query_indices']=[0];write();baseline.write_text('{}')
    with pytest.raises(RuntimeError):validate_approach_replay(path)


def test_finalization_requires_and_binds_preservation_assessments(tmp_path):
    from tests.test_history_training import output_fixture
    from training.history import finalize_history_training
    from training.integrity import require_finalized
    checkpoint,protected=output_fixture(tmp_path)
    config=dict(training_budget={'kind':'updates','limit':1000},num_epochs=0,approach_preservation={'profile':'baseline_prefix_v1'})
    with pytest.raises(ValueError):finalize_history_training(tmp_path,config,dict(updates=1000,epochs=0),protected)
    assert not (tmp_path/'artifacts/training_integrity.json').exists()
    for name in ['approach_before.json','approach_after.json','frozen_parameters_verified.json']:
        (tmp_path/'artifacts'/name).write_text('{}')
    proof=finalize_history_training(tmp_path,config,dict(updates=1000,epochs=0),protected)
    report=tmp_path/'artifacts/approach_after.json'
    assert str(report.resolve()) in proof['output_sha256']
    require_finalized(checkpoint,config,allowed_budgets=('updates',))
    report.write_text('{"tampered":true}')
    with pytest.raises(RuntimeError):require_finalized(checkpoint,config,allowed_budgets=('updates',))
