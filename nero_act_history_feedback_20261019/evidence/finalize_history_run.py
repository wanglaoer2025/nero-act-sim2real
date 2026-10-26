"""Result-only verification of the frozen, approved paired pilot. No optimization or hardware."""
import hashlib,json
from pathlib import Path
import numpy as np
from training.integrity import require_unchanged,require_finalized
from simulation.history_mechanism import pilot_gate,mechanism_passed
from executor.history_qualification import require_mechanism_binding
ROOT=Path.cwd()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
seal_path=ROOT/'evidence/pre_optimization_freeze.json';seal=read(seal_path)
require_unchanged(seal['source_sha256']);require_unchanged(seal['data_sha256'])
original=read('evidence/source_inventory.json');require_unchanged(original)
records={};artifacts={};telemetry=[];table=[]
for variant in ['A','B']:
 run=ROOT/f'training/pilot_{variant}';config=read(run/'config.json');summary_path=ROOT/f'results/mechanism_{variant}/summary.json'
 training_proof=require_finalized(run/'checkpoints/policy_last.ckpt',config,allowed_budgets=('updates',))
 trained=read(run/'artifacts/training_summary.json');assert trained['actual_updates']==1000 and trained['actual_epochs']==0
 summary=read(summary_path);require_mechanism_binding(summary,sha(run/'checkpoints/policy_last.ckpt'),sha(seal_path))
 assert len({r['name'] for r in summary['records']})==6
 for row in summary['records']:
  assert row['passed']==mechanism_passed(row['result'],row['role'])
  directory=summary_path.parent/row['name'];result=row['result'];visual=read(directory/'visual_observations.json')
  with np.load(directory/'rollout.npz') as trace:
   sources=trace['target_sources'];indices=np.flatnonzero(np.char.startswith(sources.astype(str),'ACT'))
   raw=trace['raw_chunks'];feedback=trace['feedback_at_query'];measurements=read(directory/'measurements.json')
   gaps=(feedback[:,7,None]-raw[:,:2,7]).tolist() if len(raw) else []
   telemetry.append(dict(variant=variant,case=row['name'],first2_raw=raw[0,:2].tolist() if len(raw) else [],
     first2_ACT_submitted=trace['sent_targets'][indices[:2]].tolist(),raw_first2_gripper_gap_m=gaps,
     actual_query_gripper_width_m=feedback[:,7].tolist() if len(feedback) else [],
     cup_height_trace_m=[m['cup_xyz'][2] for m in measurements],
     bilateral_contact_duration_s=sum(m['bilateral'] for m in measurements)*.05,
     exposure_to_consumption_s=[r['consumed_simulation_s']-r['exposure_simulation_s'] for r in visual],
     inference_time_s=trace['inference_time_s'].tolist(),capture_wall_time_s=None,
     timing_scope='nominal simulation exposure/consumption plus wall inference; no real camera timing',
     stop_reason=result['error'] or ('20mm mechanism reached' if row['passed'] and row['role']=='hold_to_lift' else
          'remaining flow complete' if row['passed'] else 'declared active-point budget exhausted'),
     trace_path=str((directory/'rollout.npz').resolve()),trace_sha256=sha(directory/'rollout.npz')))
  table.append(dict(variant=variant,name=row['name'],role=row['role'],passed=row['passed'],
    lift_mm=result.get('current_cup_lift_m',0)*1000,new_active_points=result['new_active_points'],error=result['error']))
 records[variant]=summary
 artifacts[f'{variant}_training_run']=str(run)
 artifacts[f'{variant}_mechanism_summary']=dict(path=str(summary_path),sha256=sha(summary_path))
 for p in [training_proof,run/'artifacts/training_summary.json',run/'checkpoints/policy_last.ckpt',summary_path]:artifacts[str(p)]=sha(p)
assert read('training/pilot_A/config.json')['batch_order_sha256']==read('training/pilot_B/config.json')['batch_order_sha256']
gate=pilot_gate(records['A'],records['B']);assert gate=='stop',gate
write('evidence/pilot_decision.json',dict(complete=True,gate=gate,source_seal_sha256=sha(seal_path),
 **{k:v for k,v in artifacts.items() if k in ['A_training_run','B_training_run','A_mechanism_summary','B_mechanism_summary']},
 reason='Neither variant passed all three lift probes and all three remaining-flow checks',
 formal_training_executed=False,independent16_evaluated=False,hardware_commands_sent=False))
write('evidence/pilot_mechanism_telemetry.json',dict(complete=True,records=telemetry,real_camera_latency_measured=False))
write('evidence/pilot_comparison.json',dict(complete=True,gate=gate,records=table,
 history_benefit_demonstrated=False,full_pick_place_qualified=False))
for variant in ['A','B']:
 report=read(f'results/saved_shadow_{variant}/report.json');assert report['motion_sent'] is False and report['gripper_sent'] is False
 assert report['checkpoint_sha256']==sha(ROOT/f'training/pilot_{variant}/checkpoints/policy_last.ckpt')
for name in ['nominal','heldout']:
 report=read(f'results/history_intervention_{name}/report.json');assert report['admitted_to_runtime'] is False
 assert report['checkpoint_sha256']==sha(ROOT/'training/pilot_B/checkpoints/policy_last.ckpt')
artifacts.update({str(p.resolve()):sha(p) for p in [ROOT/'evidence/pilot_decision.json',ROOT/'evidence/pilot_mechanism_telemetry.json',ROOT/'evidence/pilot_comparison.json',ROOT/'evidence/paired_pilot_training_verified.json']})
proof=read('evidence/post_review_verification.json');require_unchanged(proof['source_sha256']);assert read('evidence/cli_verification.json')['complete']
assert not (ROOT/'training/formal_A').exists() and not (ROOT/'training/formal_B').exists() and not (ROOT/'results/independent_selected').exists()
write('evidence/final_verification.json',dict(complete=True,source_and_data_unchanged=True,original_source_files_unchanged=len(original),
 source_files=len(seal['source_sha256']),sealed_data_files=len(seal['data_sha256']),tests_passed=proof['tests_passed'],tests_skipped=proof['tests_skipped'],cli_help_passed=10,
 actual_optimizer_updates=dict(A=1000,B=1000),actual_formal_epochs=dict(A=0,B=0),gate=gate,
 mechanism_passed={v:sum(r['passed'] for r in records[v]['records']) for v in ['A','B']},mechanism_total_per_variant=6,
 formal_training='not_executed_gate_not_satisfied',full_development='not_executed_gate_not_satisfied',independent16='not_consumed',
 saved_nominal_shadow='both passed input/output binding only',real_shadow='not_executed',measured_geometry=False,live_frontend_qualified=False,
 hardware_commands_sent=False,full_pick_place_qualified=False,history_benefit_demonstrated=False,
 source_seal_sha256=sha(seal_path),verification_script_sha256=sha(__file__),artifact_graph=artifacts))
print(json.dumps(dict(complete=True,gate=gate,results=table),indent=2))
