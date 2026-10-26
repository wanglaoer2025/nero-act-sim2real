"""Exact-state release/retreat supervision. Partial training data, no hardware access."""
import json
from pathlib import Path
import numpy as np
from simulation.collect import collect_episode,verify_replay

root=Path(__file__).resolve().parents[1];out=root/'training_data_current_source'
attempt=out/'supervised_release_recovery_retry_attempts.json';accepted=out/'accepted_snapshot_recoveries_v4.json'
if attempt.exists() or accepted.exists():raise FileExistsError('Retain previous attempts')
base=json.loads((out/'accepted_snapshot_recoveries_v3.json').read_text());prior=json.loads((out/'supervised_release_recovery_attempts.json').read_text());base['episodes']+=prior['episodes'];base['source_attempt_manifests'].append(str(out/'supervised_release_recovery_attempts.json'));base['excluded_source_candidates']+=prior['rejected'];entries=[];rejected=[]
requests=[(114,root/'evidence/run9_supervised_recovery_diagnostics/delay_500ms','train')]
for episode,origin,split in requests:
 c=json.loads((origin/'scene_configuration.json').read_text());result=json.loads((origin/'result.json').read_text())
 release=[e for e in result['simulated_events'] if e['kind']=='RELEASE']
 mode='descend_to_retreat' if episode==114 else 'release_to_retreat'
 if mode=='release_to_retreat' and len(release)!=1:raise ValueError('No unique simulated supported release for '+str(origin))
 record=json.loads((origin/'physical_snapshots.json').read_text());measure=json.loads((origin/'measurements.json').read_text())
 if mode=='release_to_retreat':
  choices=[i for i,s in enumerate(record['query_steps']) if s>=release[0]['physics_step']+120 and
    measure[s-1]['supported'] and measure[s-1]['bilateral'] and measure[s-1]['cup_upright']>.95]
 else:
  choices=[i for i,s in enumerate(record['query_steps']) if s>0 and .10<measure[s-1]['tcp_xyz'][2]<.13
    and measure[s-1]['gripper_width']>.09 and measure[s-1]['supported'] and not measure[s-1]['bilateral']
    and max(abs(np.asarray(record['snapshots'][i]['previous_target'])[:7]-np.asarray(record['snapshots'][i]['qpos'])[:7]))<.015]
 if not choices:raise ValueError('No supported exact release-context snapshot')
 index=choices[0];snapshot=record['snapshots'][index];q=np.asarray(snapshot['qpos'])
 c['initial_robot_state']=dict(qpos_rad=q[:7].tolist(),gripper_total_width_m=float(q[7]-q[8]))
 c.update(initial_simulation_snapshot=snapshot,continuation=mode,joint_speed_limit_rad_s=.02,
   teacher_command_lead_limit_rad=.015,gripper_speed_limit_m_s=.004,teacher_gripper_feedback_lead_m=.008,
   correction_source=dict(origin=str(origin),query_step=int(record['query_steps'][index]),split=split,
     role='partial_not_full',reason=('slow delayed descent' if mode=='descend_to_retreat' else 'ACT failed to open after actual support confirmation and feedback reference hold')))
 c['dwell_seconds']['release']=8;c['dwell_seconds']['close']=8;path=out/f'episode_{episode}.hdf5'
 try:
  result=collect_episode(c,path);replay=verify_replay(path)
  entries.append(dict(path=str(path),status='valid',task_role=result['task'],full_task_verified=False,
   correction_verified=True,split=split,frames=result['frames'],origin=c['correction_source'],replay=replay,
   prefix_contract_audit=result['prefix_contract_audit']))
  print(json.dumps(dict(episode=episode,success=True,frames=result['frames'],replay=replay)),flush=True)
 except Exception as exc:
  rejected.append(dict(episode=episode,error=str(exc)));print(json.dumps(rejected[-1]),flush=True)
 attempt.write_text(json.dumps(dict(episodes=entries,rejected=rejected),indent=2))
 if rejected:raise RuntimeError('Rejected candidate retained; no accepted manifest')
accepted.write_text(json.dumps(dict(episodes=base['episodes']+entries,rejected=[],
 source_attempt_manifests=base['source_attempt_manifests']+[str(attempt)],
 excluded_source_candidates=base['excluded_source_candidates']),indent=2))
