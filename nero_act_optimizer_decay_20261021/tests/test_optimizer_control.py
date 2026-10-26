import copy
import json
import pytest
import torch
from tests.test_restricted_history_heads import config


def control(schedule='fixed'):
    return dict(schedule=schedule,initial_lr=1e-6,final_lr=1e-6 if schedule=='fixed' else 1e-7,updates=1000)


def test_learning_rate_endpoints_are_used_on_actual_first_and_last_updates():
    from training.optimizer_control import arm_rate
    assert arm_rate(control(),1)==1e-6 and arm_rate(control(),1000)==1e-6
    assert arm_rate(control('linear'),1)==1e-6
    assert arm_rate(control('linear'),1000)==pytest.approx(1e-7)
    assert arm_rate(control('linear'),500)==pytest.approx(5.504504504504505e-7)
    for step in [0,1001,True]:
        with pytest.raises(ValueError):arm_rate(control(),step)
    for change in [dict(updates=2000),dict(schedule='cosine'),dict(initial_lr=1e-4),dict(final_lr=0.)]:
        bad=control();bad.update(change)
        with pytest.raises(ValueError):arm_rate(bad,1)
    with pytest.raises(ValueError):arm_rate(None,1)


def test_schedule_updates_only_arm_group_and_checks_its_identity():
    from training.history import build_history_policy
    from training.optimizer_control import apply_arm_rate
    c,_,_,_=config();c.update(arm_output_lr=1e-6,optimizer_control=control('linear'),approach_preservation={'profile':'baseline_prefix_v1'})
    p=build_history_policy(c);gripper_lr=p.optimizer.param_groups[1]['lr']
    values=apply_arm_rate(p,1000)
    assert values['arm_lr']==pytest.approx(1e-7)
    assert p.optimizer.param_groups[1]['lr']==gripper_lr==.0003
    p.optimizer.param_groups.reverse()
    with pytest.raises(ValueError):apply_arm_rate(p,1)


def test_gradient_conflict_reports_opposition_and_undefined_zero_direction():
    from training.optimizer_control import gradient_relation
    assert gradient_relation(torch.tensor([1.,0.]),torch.tensor([-2.,0.]))==dict(first_norm=1.,second_norm=2.,cosine=-1.)
    result=gradient_relation(torch.tensor([1.,0.]),torch.zeros(2))
    assert result['second_norm']==0 and result['cosine'] is None
    with pytest.raises(ValueError):gradient_relation(torch.tensor([float('nan')]),torch.ones(1))


def test_update_probe_preserves_mode_gradients_and_frozen_parameters(tmp_path):
    from training.optimizer_control import prefix_probe
    from training.approach_preservation import ApproachPreservation
    from tests.test_approach_preservation import replay_file
    from training.history import frozen_output_head_state,require_frozen_output_head_state
    c,q,image,_=config();p,path=replay_file(tmp_path,c,q,image);p.train()
    keeper=ApproachPreservation(dict(profile='baseline_prefix_v1',weight=1.,arm_unit_rad=.0001,gripper_unit_m=.0005,batch_size=2,replay_path=str(path)),p.loss_stats,'cpu')
    before=frozen_output_head_state(p);grads={n:v.grad for n,v in p.named_parameters()}
    first=prefix_probe(p,keeper)
    with torch.no_grad():p.model.action_head.bias[:7].add_(.001)
    second=prefix_probe(p,keeper)
    assert first.shape==(2,2,8) and (first[...,:7]-second[...,:7]).abs().max()>0
    assert p.training and all(v.grad is grads[n] for n,v in p.named_parameters())
    require_frozen_output_head_state(p,before)


def test_caller_cannot_replace_sealed_optimizer_schedule(tmp_path,monkeypatch):
    import training.history as module
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'_require_seal',lambda path:{})
    (tmp_path/'config').mkdir();declared=dict(adaptation_scope='output_heads_v1',train_arm_output_and_gripper_only=True,
        arm_output_lr=1e-6,gripper_adapter_lr=.0003,optimizer_control=control())
    (tmp_path/'config/history_experiment.json').write_text(json.dumps(declared))
    caller=copy.deepcopy(declared);caller.update(variant='A',stage='pilot');caller['optimizer_control']=control('linear')
    with pytest.raises(ValueError,match='sealed optimizer'):
        module.train_history(tmp_path/'data',tmp_path/'init',caller,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_finalization_binds_complete_actual_learning_rate_trace(tmp_path):
    from tests.test_history_training import output_fixture
    from training.history import finalize_history_training
    from training.integrity import require_finalized
    from training.optimizer_control import arm_rate
    checkpoint,protected=output_fixture(tmp_path)
    c=dict(training_budget={'kind':'updates','limit':1000},num_epochs=0,optimizer_control=control('linear'),gripper_adapter_lr=.0003)
    with pytest.raises(ValueError):finalize_history_training(tmp_path,c,dict(updates=1000,epochs=0),protected)

    trace=tmp_path/'artifacts/optimizer_trace.jsonl'
    rows=[dict(update=i,arm_lr=arm_rate(control('linear'),i),gripper_lr=.0003) for i in range(1,1001)]
    trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    proof=finalize_history_training(tmp_path,c,dict(updates=1000,epochs=0),protected)
    assert str(trace.resolve()) in proof['output_sha256']
    rows[500]['arm_lr']=.01;trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(RuntimeError):require_finalized(checkpoint,c,allowed_budgets=('updates',))
    with pytest.raises(ValueError):finalize_history_training(tmp_path,c,dict(updates=1000,epochs=0),protected)


def test_optimizer_cannot_start_without_matched_zero_update_diagnostics(tmp_path):
    from training.optimizer_control import require_optimizer_diagnostics
    from training.integrity import snapshot
    source=tmp_path/'source';source.write_text('fixed-input')
    proof=dict(schema='optimizer_diagnostics_v1',complete=True,actual_optimizer_updates=0,model_states_unchanged=True,
        exact_teacher_conflict_groups_after_filter=0,excluded_ambiguous_windows=['episode_8:4'],gradient_records=[{}]*8,
        initial_replay_assessment=dict(maximum_arm_prefix_change_rad=0.,maximum_gripper_prefix_change_m=0.),source_sha256=snapshot([source]))
    path=tmp_path/'diagnostics.json';write=lambda:path.write_text(json.dumps(proof));write()
    assert str(source.resolve()) in require_optimizer_diagnostics(path,['episode_8:4'])
    with pytest.raises(ValueError):require_optimizer_diagnostics(path,[])
    proof['actual_optimizer_updates']=1;write()
    with pytest.raises(ValueError):require_optimizer_diagnostics(path,['episode_8:4'])
    proof['actual_optimizer_updates']=0;proof['exact_teacher_conflict_groups_after_filter']=1;write()
    with pytest.raises(ValueError):require_optimizer_diagnostics(path,['episode_8:4'])
    proof['exact_teacher_conflict_groups_after_filter']=0;write();source.write_text('changed')
    with pytest.raises(RuntimeError):require_optimizer_diagnostics(path,['episode_8:4'])
