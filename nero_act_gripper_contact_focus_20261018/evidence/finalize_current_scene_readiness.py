"""Audit the fixed-scene evaluation; this never creates an authorization receipt."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from executor.bundle import verify_model_bundle
from executor.evaluation_integrity import verify_frozen_artifacts
from executor.permission import CASES


def main():
    model = ROOT/'model/run26_current_fixed_scene'
    metadata = verify_model_bundle(model)
    summary = json.loads((ROOT/'evidence/run26_current_scene14/summary.json').read_text())
    verify_frozen_artifacts(model/'policy.ckpt', summary)
    assert summary['complete'] is True and summary['finalized'] is True
    assert len(summary['results']) == 14
    assert {r['scenario']['name'] for r in summary['results']} == CASES
    assert summary['passed'] == sum(r['success'] for r in summary['results'])
    cases = []
    for r in summary['results']:
        case = dict(name=r['scenario']['name'], success=r['success'], error=r['error'],
            lift_mm=r['max_lift_m']*1000, transfer_mm=r['maximum_horizontal_transfer_m']*1000,
            active_points=r['active_policy_points'],
            events={e['kind']: e['active_points'] for e in r['simulated_events']},
            last_tcp_z_mm=r['last_observation']['tcp_xyz'][2]*1000,
            stable_released_frames=r['stable_released_frames'])
        if not r['success']:
            case['observed_failure'] = ('policy_reclose_after_release' if r['error']
                and 'another closure' in r['error'] else 'declared_active_budget_exhausted')
        cases.append(case)
    profile = json.loads((ROOT/'evidence/current_ros_profile_readonly.json').read_text())
    prep = json.loads((ROOT/'evidence/high_open_preparation_plan/report.json').read_text())
    assert profile['read_completed'] and not profile['parameter_changes']
    assert profile['driver_parameters']['control_enabled']['bool_value'] is False
    assert not profile['control_topics']['/control/gated_joint_states']['publishers']
    assert prep['model_contact_checks_passed'] and not prep['motion_sent']
    report = dict(stage='fixed_scene_evaluation_failed_no_hardware_permission',
        passed=summary['passed'], total=14, complete=True, finalized=True,
        model_and_evaluation_source_hashes_verified=True,
        checkpoint_sha256=metadata['checkpoint_sha256'],
        scope='current_fixed_cup_card_and_start_only',
        cases=cases, high_open_preparation_plan_only=True,
        high_open_preparation_duration_s=prep['duration_s'],
        current_driver_control_disabled=True,
        execution_ready=False, authorization_receipt_created=False,
        hardware_motion_sent=False, gripper_command_sent=False, enable_sent=False,
        task_budget_points=3900, original_acceptance_and_protection_unchanged=True)
    assert summary['passed'] < 14
    (ROOT/'evidence/current_fixed_scene_readiness.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
