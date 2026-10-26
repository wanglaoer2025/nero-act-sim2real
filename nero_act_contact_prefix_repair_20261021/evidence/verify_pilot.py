"""Fresh bounds, provenance and outcome checks for the single approved pilot."""
from pathlib import Path
import sys,json,pickle,re
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.integrity import require_unchanged,require_finalized,snapshot
from training.history import initialize_history
from training.optimizer_control import arm_rate,require_optimizer_diagnostics
from training.approach_preservation import validate_approach_replay
from training.contact_timebase import verify_uniform_episode


def main():
    seal=json.loads((ROOT/'evidence/pre_optimization_freeze.json').read_text())
    require_unchanged(seal['source_sha256']);require_unchanged(seal['data_sha256'])
    tested=json.loads((ROOT/'evidence/tested_sources.json').read_text());require_unchanged(tested['source_sha256'])
    assert tested['complete'] and tested['failed']==0
    settings=json.loads((ROOT/'config/history_experiment.json').read_text())
    require_optimizer_diagnostics(settings['optimizer_diagnostics_path'],settings['excluded_ambiguous_windows'])
    validate_approach_replay(settings['approach_preservation']['replay_path'])
    data=ROOT/'training_data/uniform_complete_v1';manifest=json.loads((data/'manifest.json').read_text())
    assert manifest['complete'] and len(manifest['episodes'])==68
    assert sum(r['split']=='train' for r in manifest['episodes'])==52
    assert sum(r['split']=='val' for r in manifest['episodes'])==16
    uniform=[verify_uniform_episode(r['path']) for r in manifest['episodes']]
    derivation=json.loads((data/'derivation_inputs.json').read_text());require_unchanged(derivation)
    physical=json.loads((ROOT/'training_data/uniform_physical_v1/manifest.json').read_text())
    require_unchanged(physical['source_sha256'])
    assert physical['complete'] and len(physical['episodes'])==48 and not physical['rejected']
    for folder in ['executor','simulation','calibration','vision','policy_source']:
        for path in (ROOT/folder).rglob('*.py'):
            assert path.read_bytes()==(ROOT.parent/'nero_act_optimizer_decay_20261021'/path.relative_to(ROOT)).read_bytes()
    run=ROOT/'training/pilot_A';checkpoint=run/'checkpoints/policy_last.ckpt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);config=payload['config']
    integrity=require_finalized(checkpoint,config,allowed_budgets=('updates',));proof=json.loads(integrity.read_text())
    assert proof['actual_updates']==1000 and proof['actual_epochs']==0
    assert config['gripper_supervision_profile']=='deployed_first2_v1'
    init=ROOT.parent/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'
    source=torch.load(init,map_location='cpu',weights_only=False)
    old=pickle.loads((init.parent/'dataset_stats.pkl').read_bytes());stats=pickle.loads((run/'checkpoints/dataset_stats.pkl').read_bytes())
    _,initial=initialize_history(source,old,stats,config,variant='A')
    changed=[k for k,v in payload['model'].items() if not torch.equal(v,initial[k])]
    expected={'model.action_head.weight','model.action_head.bias','gripper_adapter.0.weight','gripper_adapter.0.bias','gripper_adapter.2.weight','gripper_adapter.2.bias'}
    assert set(changed)==expected
    for key in ['model.action_head.weight','model.action_head.bias']:assert torch.equal(initial[key][7:],payload['model'][key][7:])
    frozen=json.loads((run/'artifacts/frozen_parameters_verified.json').read_text());assert frozen['before']==frozen['after']
    trace=[json.loads(v) for v in (run/'artifacts/optimizer_trace.jsonl').read_text().splitlines()]
    assert len(trace)==1000 and all(r['update']==i and r['arm_lr']==arm_rate(config['optimizer_control'],i)
        and r['gripper_lr']==.0003 for i,r in enumerate(trace,1))
    qualification=json.loads((ROOT/'results/qualification/summary.json').read_text());require_unchanged(qualification['source_sha256'])
    posthoc=json.loads((ROOT/'evidence/posthoc_gradients.json').read_text());require_unchanged(posthoc['source_sha256'])
    assert posthoc['complete'] and posthoc['actual_optimizer_updates']==0
    assert len(posthoc['gradient_records'])==8 and all(r['before']==r['after'] and r['optimizer_state_entries']==0
        and r['production_inverse_target_tail_independent'] for r in posthoc['model_states'])
    six=json.loads((ROOT/'results/mechanism_original/summary.json').read_text());assert six['complete'] and len(six['records'])==6
    for row in six['records']:
        folder=ROOT/'results/mechanism_original'/row['name'];result=json.loads((folder/'result.json').read_text())
        assert result==row['result'] and result['resumed_partial_not_full'] and result['teacher_free'] and not result['real_hardware_tested']
        with np.load(folder/'rollout.npz',allow_pickle=False) as f:assert len(f['states'])==result['frames']
    assert qualification['nominal_evaluated']==bool(qualification['exact_and_preservation_gate_passed'] and six['mechanism_and_recoveries_passed'])
    assert qualification['original_six_pass_count']==sum(r['passed'] for r in six['records'])
    assert not (ROOT/'training/pilot_B').exists() and not (ROOT/'training/formal_A').exists()
    old_proof=ROOT.parent/'nero_act_optimizer_control_20261021/evidence/final_verification.json'
    require_unchanged(json.loads(old_proof.read_text())['artifact_sha256'])
    sources=[Path(__file__),integrity,checkpoint,ROOT/'evidence/pre_optimization_freeze.json',ROOT/'evidence/tested_sources.json',
        data/'derivation_inputs.json',ROOT/'evidence/results.md',ROOT/'approved_scope.md',ROOT/'progress.md',old_proof,
        ROOT/'evidence/posthoc_gradients.json',ROOT/'evidence/final_source_review.json',ROOT/'evidence/pipeline_status.json']
    sources += list((run/'artifacts').glob('*.json'))+[run/'artifacts/optimizer_trace.jsonl']
    sources += [p for folder in ['results/qualification','results/mechanism_original','results/nominal_after']
        for p in (ROOT/folder).rglob('*') if p.is_file()]
    sources += [p for p in (ROOT/'evidence').glob('*.py')]
    for target in re.findall(r'\]\(([^)]+)\)',(ROOT/'evidence/results.md').read_text()):
        if not target.startswith(('http:','https:','#')):assert (ROOT/'evidence'/target).resolve().exists(),target
    report=dict(complete=True,verification_complete_not_task_success=True,actual_updates=1000,actual_formal_epochs=0,
        tests_passed=tested['tests_passed'],tests_skipped=tested['tests_skipped'],uniform_physical_episodes=len(uniform),
        changed_tensors=changed,frozen_tensor_digests=len(frozen['before']),core_runtime_matches_parent=True,
        qualification=qualification,hardware_commands_sent=False,artifact_sha256=snapshot(sources))
    require_unchanged(report['artifact_sha256'])
    (ROOT/'evidence/final_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['qualification','artifact_sha256']},indent=2))


if __name__=='__main__':main()
