"""Final immutable offline evidence audit; never issues hardware permission."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from executor.bundle import verify_model_bundle
from executor.evaluation_integrity import verify_frozen_artifacts
from executor.permission import CASES
from training.integrity import require_finalized


def main():
    model=ROOT/'model/run27_fixed_heads'
    metadata=verify_model_bundle(model)
    config=json.loads((model/'training_config.json').read_text())
    require_finalized(ROOT/'training/run27_fixed_heads/checkpoints/policy_best_val.ckpt',config)
    evaluation=json.loads((ROOT/'evaluation/run27_current_scene14/summary.json').read_text())
    verify_frozen_artifacts(model/'policy.ckpt',evaluation)
    assert evaluation['complete'] and evaluation['finalized']
    assert evaluation['total']==len(evaluation['results'])==14
    assert {r['scenario']['name'] for r in evaluation['results']}==CASES
    assert evaluation['passed']==sum(r['success'] for r in evaluation['results'])
    audit=json.loads((ROOT/'evidence/frozen_heads_audit.json').read_text());assert audit['passed']
    before=json.loads((ROOT.parent/'nero_act_gripper_contact_focus_20261018/evidence/run26_current_scene14/summary.json').read_text())
    previous={r['scenario']['name']:r for r in before['results']}
    rows=[]
    for r in evaluation['results']:
        name=r['scenario']['name']
        rows.append(dict(name=name,previous_cohort_success=previous[name]['success'],success=r['success'],
            physical_success=r['physical_success'],error=r['error'],
            max_lift_mm=r['max_lift_m']*1000,transfer_mm=r['maximum_horizontal_transfer_m']*1000,
            stable_released_frames=r['stable_released_frames'],active_points=r['active_policy_points'],
            last_tcp_z_mm=r['last_observation']['tcp_xyz'][2]*1000,
            events={e['kind']:e['active_points'] for e in r['simulated_events']},
            unexpected_contacts=r['unexpected_arm_cup_contact_frames'],
            contract_failures=r['execution_contract_failures']))
    report=dict(stage='offline_training_and_original14_evaluation_complete',
        epochs=20,selected_epoch=metadata['selected_epoch'],
        checkpoint_sha256=metadata['checkpoint_sha256'],
        previous_passed=before['passed'],passed=evaluation['passed'],total=14,
        same_seed_cohorts=False,previous_seed_offset=260018,current_seed_offset=270018,
        paired_single_case_diagnostic='evidence/paired_previous_policy/report.json',
        full_acceptance_passed=evaluation['passed']==14,original_acceptance_unchanged=True,
        frozen_parameter_scope_audit_passed=True,training_integrity_and_frozen_evaluation_verified=True,
        diagnostic_probe_global_1mm_passed=False,task_budget_points=config['reach_deadline_steps'],
        hardware_execution_ready=False,hardware_motion_sent=False,gripper_sent=False,enable_sent=False,
        cases=rows)
    (ROOT/'evidence/final_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
