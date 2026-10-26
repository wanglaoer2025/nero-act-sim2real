import copy
import pytest
import torch
from tests.test_history_policy import fixtures


def config():
    c,q,image,h=fixtures()
    c.update(variant='A',numeric_history_profile='none',adaptation_scope='output_heads_v1',
        train_arm_output_and_gripper_only=True,train_nonlinear_gripper_only=False,train_gripper_head_only=False,
        nonlinear_gripper_head=True,freeze_backbone=True,prior_gripper_loss_weight=.005,arm_output_lr=.0001,
        gripper_adapter_lr=.0003,training_budget={'kind':'updates','limit':1000},num_epochs=0,pose_loss_weight=0.)
    return c,q,image,h


def test_restricted_builder_changes_only_seven_output_rows_and_gripper_adapter():
    from training.history import build_history_policy,frozen_output_head_state,require_frozen_output_head_state
    c,q,image,h=config();p=build_history_policy(c).train();before=frozen_output_head_state(p)
    target=torch.ones(2,20,8)*.1;padding=torch.zeros(2,20,dtype=torch.bool)
    losses=p(q,image,target,padding);p.optimizer.zero_grad();losses['loss'].backward()
    from training.history import restricted_gradient_evidence
    e=restricted_gradient_evidence(p);assert e['arm_output_grad_norm']>0 and e['gripper_grad_norm']>0 and not e['frozen_base_has_grad']
    p.optimizer.step();require_frozen_output_head_state(p,before)
    with torch.no_grad():p.model.latent_proj.weight.add_(.1)
    with pytest.raises(RuntimeError):require_frozen_output_head_state(p,before)


def test_restricted_scope_rejects_history_and_formal_budgets():
    from training.history import build_history_policy
    c,q,image,h=config()
    for updates in [dict(variant='B'),dict(numeric_history_profile='causal_numeric_gru5_v1'),
                    dict(training_budget={'kind':'epochs','limit':20}),dict(train_arm_output_and_gripper_only=False),dict(cvae_training=False),dict(policy_input_dim=9)]:
        bad=copy.deepcopy(c);bad.update(updates)
        with pytest.raises(ValueError):build_history_policy(bad)


def test_restricted_scope_preserves_initial_runtime_predictions():
    from training.history import build_history_policy
    from executor.model import BoundedActPolicy
    c,q,image,h=config();torch.manual_seed(42);p=build_history_policy(c).eval()
    runtime=BoundedActPolicy(c).eval();runtime.load_state_dict(p.state_dict())
    with torch.no_grad():torch.testing.assert_close(p(q,image),runtime(q,image),atol=0,rtol=0)


@pytest.mark.parametrize('change',[{'adaptation_scope':'full'}, {'arm_output_lr':.1},
                                   {'train_arm_output_and_gripper_only':False}])
def test_caller_cannot_replace_the_sealed_restricted_profile(tmp_path,monkeypatch,change):
    import json
    import training.history as module
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'_require_seal',lambda path:{})
    (tmp_path/'config').mkdir();declared=dict(adaptation_scope='output_heads_v1',
        train_arm_output_and_gripper_only=True,arm_output_lr=.0001,gripper_adapter_lr=.0003)
    (tmp_path/'config/history_experiment.json').write_text(json.dumps(declared))
    caller=dict(declared,variant='A',stage='pilot');caller.update(change)
    with pytest.raises(ValueError,match='sealed'):
        module.train_history(tmp_path/'data',tmp_path/'initializer',caller,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_restricted_initializer_records_actual_trainable_scope():
    from training.history import initialize_history
    from executor.model import BoundedActPolicy
    from tests.test_visual_policy import statistics
    c,q,image,h=config();stats=statistics(c);model=BoundedActPolicy(c)
    source=dict(config=dict(c,trainable_parameter_scope='inherited_full_ACT'),model=model.state_dict())
    paired,_=initialize_history(source,stats,stats,c,variant='A')
    assert paired['training_revision']=='restricted_zero_latent_output_heads_v1'
    assert paired['trainable_parameter_scope']=='seven_arm_output_rows_and_nonlinear_gripper_only'
