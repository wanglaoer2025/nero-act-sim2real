from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_comparison import BoundedPolicy,approach_metrics


def test_budget_stops_diagnosis_without_touching_policy_config():
    class Inner:
        config={'reach_deadline_steps':3900}
        def __call__(self,x):return x
    inner=Inner();p=BoundedPolicy(inner,limit=2)
    assert p(1)==1 and p(2)==2
    with pytest.raises(RuntimeError,match='not full-task acceptance'):p(3)
    assert inner.config=={'reach_deadline_steps':3900}


def test_short_probe_is_not_a_full_task_even_when_target_reached():
    scene=dict(pick_xy_m=[0.,0.],table_top_z_m=0.,grasp_tcp_height_m=.05,lift_height_m=.2)
    measures=[dict(tcp_xyz=[0.,0.,.48]),dict(tcp_xyz=[0.,0.,.25])]
    result=dict(error='RuntimeError: Diagnostic300query cap; not full-task acceptance',
        execution_contract_failures=0,unexpected_arm_cup_contact_frames=0,simulated_events=[])
    r=approach_metrics(scene,result,measures)
    assert r['approach_reached'] and r['eligible_for_full_nominal']
    assert r['full_task_test'] is False and r['full_task_passed'] is False


def test_contact_error_or_stagnation_never_qualifies_for_full_task():
    scene=dict(pick_xy_m=[0.,0.],table_top_z_m=0.,grasp_tcp_height_m=.05,lift_height_m=.2)
    measures=[dict(tcp_xyz=[0.,0.,.48]),dict(tcp_xyz=[0.,0.,.48])]
    result=dict(error=None,execution_contract_failures=0,unexpected_arm_cup_contact_frames=0,simulated_events=[])
    assert not approach_metrics(scene,result,measures)['eligible_for_full_nominal']
    measures[-1]['tcp_xyz']=[0.,0.,.25];result['unexpected_arm_cup_contact_frames']=1
    assert not approach_metrics(scene,result,measures)['eligible_for_full_nominal']
    result['unexpected_arm_cup_contact_frames']=0;result['error']='ValueError: rejected'
    assert not approach_metrics(scene,result,measures)['eligible_for_full_nominal']
