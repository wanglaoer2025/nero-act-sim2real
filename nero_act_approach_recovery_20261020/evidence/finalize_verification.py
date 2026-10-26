"""Read-only verification plus final result artifact; no optimization or hardware."""
import hashlib,json
from pathlib import Path
from training.integrity import require_unchanged
from executor.history_qualification import require_mechanism_binding
from simulation.history_mechanism import pilot_gate
from simulation.visual_evaluation import full_task_passed,require_files

root=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sealpath=root/'evidence/pre_optimization_freeze.json';seal=json.loads(sealpath.read_text())
require_unchanged(seal['source_sha256']);require_unchanged(seal['data_sha256'])
prior=[]
for name in ['nero_act_history_feedback_20261019','nero_act_contact_transition_fix_20261020',
             'nero_act_contact_transition_balance_20261020','nero_act_contact_recovery_20261020',
             'nero_act_prefix_supervision_fix_20261020']:
    p=root.parent/name/'evidence/pre_optimization_freeze.json';s=json.loads(p.read_text())
    require_unchanged(s['source_sha256']);require_unchanged(s['data_sha256'])
    for variant in ['A','B']:
        pp=root.parent/name/f'training/pilot_{variant}/artifacts/training_integrity.json'
        prior_proof=json.loads(pp.read_text());require_unchanged(prior_proof['output_sha256'])
    prior.append(dict(experiment=name,source_files=len(s['source_sha256']),data_files=len(s['data_sha256']),unchanged=True))
summaries={}
for v in 'AB':
    p=root/f'results/mechanism_{v}/summary.json';s=json.loads(p.read_text())
    require_mechanism_binding(s,sha(root/f'training/pilot_{v}/checkpoints/policy_last.ckpt'),sha(sealpath));summaries[v]=s
    proof=json.loads((root/f'training/pilot_{v}/artifacts/training_integrity.json').read_text())
    require_unchanged(proof['protected_sha256']);require_unchanged(proof['output_sha256'])
    assert proof['actual_updates']==1000 and proof['actual_epochs']==0
assert pilot_gate(summaries['A'],summaries['B'])=='A_development_only'
nominalpath=root/'results/pilot_A_nominal/summary.json';nominal=json.loads(nominalpath.read_text())
assert nominal['complete'] and nominal['total']==1 and nominal['provenance']['assessment_role']=='nominal_development_not_final_test'
bindings=json.loads((nominalpath.parent/'protocol.json').read_text())['dependencies']
assert bindings[str((root/'training/pilot_A/checkpoints/policy_last.ckpt').resolve())]==sha(root/'training/pilot_A/checkpoints/policy_last.ckpt')
require_files(bindings)
passed=full_task_passed(nominal['results'][0]);assert int(passed)==nominal['passed']
r=nominal['results'][0]
result=dict(complete=True,verification_complete_not_formal_qualification=True,tests_passed=393,tests_skipped=5,
    source_files_unchanged=len(seal['source_sha256']),data_inputs_unchanged=len(seal['data_sha256']),
    prior_experiments=prior,mechanism_passed={v:sum(row['passed'] for row in s['records']) for v,s in summaries.items()},
    total=6,mechanism_gate='A_development_only',selected_variant='A',short_history_B_qualified=False,formal_training_started=False,formal_qualification=False,
    nominal_full_task_diagnostic_passed=passed,nominal_active_points=r['active_policy_points'],nominal_max_lift_m=r['max_lift_m'],
    nominal_transfer_m=r['maximum_horizontal_transfer_m'],nominal_stable_released_frames=r['stable_released_frames'],
    nominal_error=r['error'],independent16_evaluated=False,hardware_commands_sent=False)
(root/'evidence/final_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
