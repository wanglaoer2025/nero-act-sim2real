"""Extra original-scene diagnostic only; cannot qualify formal training or hardware."""
import copy
import hashlib
import json
from pathlib import Path

from executor.history_qualification import require_mechanism_binding
from simulation.history_mechanism import load_pilot,pilot_gate
from simulation.supervised_rollout import rollout
from simulation.visual_evaluation import evaluation_dependencies,file_hashes,require_files,full_task_passed

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
records={v:json.loads((ROOT/f'results/mechanism_{v}/summary.json').read_text()) for v in 'AB'}
for v,s in records.items():
    require_mechanism_binding(s,sha(ROOT/f'training/pilot_{v}/checkpoints/policy_last.ckpt'),sha(ROOT/'evidence/pre_optimization_freeze.json'))
if pilot_gate(records['A'],records['B'])!='formal_pair':
    raise ValueError('This B diagnostic requires all six B mechanisms; no failed gate may be bypassed')
run=ROOT/'training/pilot_B';policy,stats=load_pilot(run)
case=dict(name='nominal_development',seed=191899,query_stride=2,servo_delay_steps=0,
    observation_age_s=.05,jitter=False,force=False,scene=copy.deepcopy(policy.config['scene_config']))
output=ROOT/'results/pilot_B_nominal_diagnostic';output.mkdir(exist_ok=False)
bindings=file_hashes(evaluation_dependencies(ROOT,run,protocol_path=ROOT/'evaluation/history_protocol.json',checkpoint_name='policy_last.ckpt')
    +[Path(__file__),ROOT/'evidence/pre_optimization_freeze.json']
    +[ROOT/f'results/mechanism_{v}/summary.json' for v in 'AB'])
(output/'bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
result=rollout(case['scene'],policy,stats,case,output/'nominal_development',runtime_profile='history_common_v1')
require_files(bindings)
result['full_task_passed']=full_task_passed(result)
summary=dict(complete=True,finalized=True,passed=int(result['full_task_passed']),total=1,results=[result],
    checkpoint_sha256=sha(run/'checkpoints/policy_last.ckpt'),source_seal_sha256=sha(ROOT/'evidence/pre_optimization_freeze.json'),
    assessment_role='pilot_nominal_diagnostic_not_formal',formal_qualification=False,hardware_deployment_allowed=False,
    real_hardware_tested=False,independent16_evaluated=False)
(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(passed=summary['passed'],active_points=result['active_policy_points'],error=result['error'],
    lift_m=result['max_lift_m'],transfer_m=result['maximum_horizontal_transfer_m'],stable_frames=result['stable_released_frames'],
    full_task_diagnostic_only=True)),flush=True)
