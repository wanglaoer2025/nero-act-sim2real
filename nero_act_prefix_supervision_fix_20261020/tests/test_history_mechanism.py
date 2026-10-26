import copy
import importlib
import importlib.util
import pytest


def api():
    assert importlib.util.find_spec('simulation.history_mechanism'), 'History mechanism evaluator missing'
    return importlib.import_module('simulation.history_mechanism')


def lifted():
    return dict(mechanism_success=True,current_cup_lift_m=.0201,new_active_points=200,
                active_policy_points=1992,error=None,contact_state='holding',
                unexpected_arm_cup_contact_frames=0,execution_contract_failures=0,
                premature_release_before_support=False)


def gate_record(passed):
    return dict(complete=True,lift_probes=[passed]*3,remaining_roles={
        'pre_close':passed,'support_release':passed,'release_retreat':passed})


def test_20mm_is_mechanism_not_full_task():
    m=api();result=lifted()
    assert m.mechanism_passed(result,'hold_to_lift')
    from simulation.visual_evaluation import full_task_passed
    assert not full_task_passed(result)
    for update in [dict(new_active_points=401),dict(contact_state='open'),dict(active_policy_points=3901),dict(error='stop')]:
        assert not m.mechanism_passed(dict(result,**update),'hold_to_lift')


def test_all_heldout_roles_required_for_gate():
    m=api();a=gate_record(True);b=gate_record(True)
    assert m.pilot_gate(a,b)=='formal_pair'
    b['remaining_roles'].pop('release_retreat')
    assert m.pilot_gate(a,b)=='A_development_only'


def test_B_failure_never_launches_formal_pair():
    m=api()
    assert m.pilot_gate(gate_record(True),gate_record(False))=='A_development_only'
    assert m.pilot_gate(gate_record(False),gate_record(False))=='stop'


def test_policy_transition_requires_explicit_offline_scope_and_same_physics():
    import simulation.runtime_checkpoint as m
    assert hasattr(m,'resume_bindings'), 'Explicit offline policy transition missing'
    origin=dict(scope='offline_simulation',scene_sha256='scene',weights_sha256='old',runtime_profile='legacy_trace_v1')
    current=dict(origin,weights_sha256='new',runtime_profile='history_common_v1')
    assert m.resume_bindings(current,origin)==current
    assert m.resume_bindings(current,origin,allow_policy_transition=True)==origin
    with pytest.raises(ValueError):m.resume_bindings(dict(current,scene_sha256='different'),origin,allow_policy_transition=True)
    with pytest.raises(ValueError):m.resume_bindings(dict(current,runtime_profile='legacy_trace_v1'),origin,allow_policy_transition=True)
