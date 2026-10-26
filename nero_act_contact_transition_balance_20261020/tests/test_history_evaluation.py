from pathlib import Path
import pytest


def complete_result():
    return dict(success=True,error=None,max_lift_m=.19,maximum_horizontal_transfer_m=.15,
        stable_released_frames=100,unexpected_arm_cup_contact_frames=0,execution_contract_failures=0,
        premature_release_before_support=False,active_policy_points=3500,
        simulated_events=[{'kind':k} for k in ['GRASP','CARRY','RELEASE','RETREAT']],scenario={'name':'case_0'})


def test_candidate_selection_never_uses_independent_results():
    import simulation.visual_evaluation as m
    assert hasattr(m,'select_independent_candidate'), 'Predeclared history candidate selection missing'
    a=dict(variant='A',development_passed=16,development_total=16,finalized=True,zero_latent_val_loss=.2)
    b=dict(a,variant='B',zero_latent_val_loss=.1,development_passed=15)
    assert m.select_independent_candidate([a,b])=='A'
    assert m.select_independent_candidate([dict(a,development_passed=15),b]) is None
    b.update(development_passed=16,zero_latent_val_loss=.2-1e-9,independent_passed=16)
    a['independent_passed']=0
    assert m.select_independent_candidate([a,b])=='A'


def test_explicit_protocol_is_forwarded_and_hash_bound(tmp_path):
    import simulation.visual_suite as m
    assert hasattr(m,'case_command'), 'Explicit protocol child command missing'
    path=tmp_path/'history_protocol.json'
    command=m.case_command(tmp_path/'run',tmp_path/'out',3,protocol_path=path,development=True)
    assert command[command.index('--protocol')+1]==str(path)
    assert '--development-regression' in command and command[command.index('--case-index')+1]=='3'


def test_only_finalized_correct_case_children_can_aggregate():
    import simulation.visual_suite as m
    assert hasattr(m,'validate_child_summary'), 'Case identity/dependency validation missing'
    result=complete_result();summary=dict(complete=True,finalized=True,total=1,passed=1,results=[result])
    bindings={'weights':'same'}
    assert m.validate_child_summary(summary,bindings,bindings,'case_0')['passed']
    with pytest.raises(ValueError):m.validate_child_summary(summary,bindings,bindings,'different_case')
    with pytest.raises(ValueError):m.validate_child_summary(summary,bindings,{'weights':'different'},'case_0')
    result['stable_released_frames']=99
    with pytest.raises(ValueError):m.validate_child_summary(summary,bindings,bindings,'case_0')


def test_pilot_20mm_and_partial_events_cannot_pass_full():
    from simulation.visual_evaluation import full_task_passed
    complete=complete_result()
    assert full_task_passed(complete)
    assert not full_task_passed(dict(complete,resumed_partial_not_full=True))
    assert not full_task_passed(dict(complete,max_lift_m=.02))
    assert not full_task_passed(dict(complete,simulated_events=[{'kind':'GRASP'}]))
