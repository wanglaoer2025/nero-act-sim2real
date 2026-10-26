from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_increment import BoundedPolicy,approach_metrics


def test_query_budget_is_only_diagnostic_and_does_not_modify_contract():
    class Inner:
        config={'reach_deadline_steps':3900}
        def __call__(self,x):return x
    i=Inner();p=BoundedPolicy(i,limit=1)
    assert p(3)==3
    with pytest.raises(RuntimeError,match='not full-task acceptance'):p(4)
    assert i.config['reach_deadline_steps']==3900


def test_approach_or_diagnostic_cap_never_claims_full_success():
    scene=dict(pick_xy_m=[0.,0.],table_top_z_m=0.,grasp_tcp_height_m=.05,lift_height_m=.2)
    measures=[dict(tcp_xyz=[0.,0.,.48]),dict(tcp_xyz=[0.,0.,.25])]
    result=dict(error='RuntimeError: Diagnostic300query cap; not full-task acceptance',execution_contract_failures=0,unexpected_arm_cup_contact_frames=0,simulated_events=[])
    m=approach_metrics(scene,result,measures)
    assert m['eligible_for_full_nominal'] and not m['full_task_test'] and not m['full_task_passed']
    result['error']='ValueError: guard rejected'
    assert not approach_metrics(scene,result,measures)['eligible_for_full_nominal']
