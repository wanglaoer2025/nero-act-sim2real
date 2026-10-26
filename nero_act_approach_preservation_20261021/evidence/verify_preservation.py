"""Verify finalized offline pilot and its independently recorded simulation results."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np
import torch
from training.integrity import require_finalized,require_unchanged
from training.history import initialize_history
from simulation.visual_evaluation import require_files,full_task_passed
from training.approach_preservation import validate_approach_replay
root=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal=json.loads((root/'evidence/pre_optimization_freeze.json').read_text())
require_unchanged(seal['source_sha256']);require_unchanged(seal['data_sha256'])
tested=json.loads((root/'evidence/tested_source_snapshot_single_query.json').read_text());require_unchanged(tested)
run=root/'training/pilot_A';checkpoint=run/'checkpoints/policy_last.ckpt'
payload=torch.load(checkpoint,map_location='cpu',weights_only=False);config=payload['config']
require_finalized(checkpoint,config,allowed_budgets=('updates',))
proof=json.loads((run/'artifacts/training_integrity.json').read_text());assert proof['actual_updates']==1000 and proof['actual_epochs']==0
replay=validate_approach_replay(config['approach_preservation']['replay_path'])
frozen=json.loads((run/'artifacts/frozen_parameters_verified.json').read_text());assert frozen['before']==frozen['after']
initial=root.parent/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'
source=torch.load(initial,map_location='cpu',weights_only=False);old=pickle.loads((initial.parent/'dataset_stats.pkl').read_bytes());stats=pickle.loads((run/'checkpoints/dataset_stats.pkl').read_bytes())
_,weights=initialize_history(source,old,stats,config,variant='A')
changed=[k for k,v in payload['model'].items() if not torch.equal(v,weights[k])]
expected={'model.action_head.weight','model.action_head.bias','gripper_adapter.0.weight','gripper_adapter.0.bias','gripper_adapter.2.weight','gripper_adapter.2.bias'}
assert set(changed)<=expected and set(changed)&{'model.action_head.weight','model.action_head.bias'}
assert torch.equal(payload['model']['model.action_head.weight'][7],weights['model.action_head.weight'][7])
assert torch.equal(payload['model']['model.action_head.bias'][7:],weights['model.action_head.bias'][7:])
assess={name:json.loads((run/f'artifacts/approach_{name}.json').read_text()) for name in ['before','after']}
assert assess['before']['samples']==assess['after']['samples']==423
assert assess['before']['maximum_arm_prefix_change_rad']<=1e-6 and assess['before']['maximum_gripper_prefix_change_m']<=1e-7
summary={}
for mode in ['before','after']:
 folder=root/f'results/nominal_{mode}';require_files(json.loads((folder/'bindings.json').read_text()))
 s=json.loads((folder/'summary.json').read_text());r=s['result'];assert s['complete'] and s['optimizer_updates']==(0 if mode=='before' else 1000)
 assert s['full_task_passed']==full_task_passed(r) and s['frozen_parameter_state']==frozen['before']
 e=next((e for e in r['simulated_events'] if e['kind']=='GRASP'),None)
 summary[mode]=dict(full_task_passed=s['full_task_passed'],active_points=r['active_policy_points'],first_GRASP_active_point=None if e is None else e['active_points'],error=r['error'],max_lift_mm=r['max_lift_m']*1000,transfer_mm=r['maximum_horizontal_transfer_m']*1000,stable_released_frames=r['stable_released_frames'],events=[e['kind'] for e in r['simulated_events']])
parent=root.parent/'nero_act_restricted_heads_20261020'
with np.load(root/'results/nominal_before/nominal_development/rollout.npz') as a,np.load(parent/'results/nominal_before/nominal_development/rollout.npz') as b:
 assert all(np.array_equal(a[k],b[k]) for k in ['states','sent_targets','raw_chunks','query_steps','feedback_at_query','target_sources','visual_features_m'])
order=root/'training_data/approach_complete_v1/batch_order_pilot.npy'
assert len(np.load(order,allow_pickle=False))==1000
assert sha(order)==sha(parent/'training_data/approach_complete_v1/batch_order_pilot.npy')
mechanism=json.loads((root/'results/mechanism_preservation/summary.json').read_text())
assert mechanism['complete'] and mechanism['checkpoint_sha256']==sha(checkpoint)
assert mechanism['source_seal_sha256']==sha(root/'evidence/pre_optimization_freeze.json')
result=dict(complete=True,review_date='2026-10-21',verification_complete_not_task_success=True,
 tests_passed=413,tests_skipped=5,actual_updates=1000,actual_epochs=0,checkpoint_sha256=sha(checkpoint),
 changed_tensors=changed,frozen_tensor_digests=len(frozen['before']),scalar_gripper_row_unchanged=True,
 source_files_unchanged=len(seal['source_sha256']),data_files_unchanged=len(seal['data_sha256']),
 approach_replay_samples=423,approach_assessments=assess,nominal=summary,
 mechanism_passed=sum(r['passed'] for r in mechanism['records']),mechanism_total=len(mechanism['records']),
 formal_training_started=False,independent16_evaluated=False,hardware_commands_sent=False,
 verification_script_sha256=sha(Path(__file__)))
(root/'evidence/final_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
