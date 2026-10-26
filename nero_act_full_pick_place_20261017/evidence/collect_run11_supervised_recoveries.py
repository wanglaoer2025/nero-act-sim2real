"""Exact failed-state supervision, never a runtime action fallback or full-task score."""
import json
from pathlib import Path
import numpy as np
from simulation.collect import collect_episode,verify_replay
from executor.ros_execution import RobotGeometry
root=Path(__file__).resolve().parents[1];out=root/'training_data_current_source';orig=root/'evaluation/run11_supervised_frozen'
attempt=out/'run11_supervised_recovery_attempts.json';accepted=out/'accepted_snapshot_recoveries_v5.json'
if attempt.exists() or accepted.exists():raise FileExistsError('Keep all earlier evidence')
base=json.loads((out/'accepted_snapshot_recoveries_v4.json').read_text());entries=[];rejected=[]
requests=[(115,'independent_1','release_to_retreat','train'),(116,'independent_5','release_to_retreat','train'),
 (117,'independent_6','grasp_to_retreat','train'),(118,'independent_3','descend_to_retreat','train'),
 (119,'independent_7','release_to_retreat','val'),(120,'cadence_11','descend_to_retreat','val'),
 (121,'independent_2','grasp_to_retreat','val'),(122,'delay_500ms','grasp_to_retreat','train')]
for episode,name,mode,split in requests:
 origin=orig/name;c=json.loads((origin/'scene_configuration.json').read_text());result=json.loads((origin/'result.json').read_text())
 record=json.loads((origin/'physical_snapshots.json').read_text());measure=json.loads((origin/'measurements.json').read_text());choices=[]
 if mode=='release_to_retreat' or episode==122:
  kind='RELEASE' if mode=='release_to_retreat' else 'GRASP';event=[e for e in result['simulated_events'] if e['kind']==kind];assert len(event)==1
  for i,s in enumerate(record['query_steps']):
   if s<event[0]['physics_step']+120:continue
   m=measure[s-1]
   if m['supported'] and m['cup_upright']>.95 and (m['bilateral'] if mode=='release_to_retreat' else m['gripper_width']>.09):choices.append(i)
 elif mode=='descend_to_retreat':
  for i,s in enumerate(record['query_steps']):
   if s<=0:continue
   m=measure[s-1];snap=record['snapshots'][i]
   if (m['supported'] and not m['bilateral'] and m['gripper_width']>.09 and .10<m['tcp_xyz'][2]<.13
       and max(abs(np.array(snap['previous_target'])[:7]-np.array(snap['qpos'])[:7]))<.015):choices.append(i)
 else:
  for i,s in enumerate(record['query_steps']):
   if s<11:continue
   m=measure[s-1];previous=record['snapshots'][i]['previous_target']
   if (all(x['bilateral'] for x in measure[s-11:s]) and m['supported'] and m['cup_upright']>.95
       and np.linalg.norm(np.array(m['cup_xyz'])[:2]-c['pick_xy_m'])<.01
       and .045<m['tcp_xyz'][2]<.075 and .003<=m['gripper_width']-previous[7]<=.012):choices.append(i)
 if not choices:raise ValueError('No valid exact physical context for '+name)
 index=choices[-1] if mode=='grasp_to_retreat' and episode!=122 else choices[0]
 snapshot=record['snapshots'][index];q=np.array(snapshot['qpos']);width=float(q[7]-q[8])
 c['initial_robot_state']=dict(qpos_rad=q[:7].tolist(),gripper_total_width_m=width)
 if mode=='grasp_to_retreat':
  c['ik_goal_rotation_link7']=RobotGeometry(c).pose(np.r_[q[:7],width])['rotation'].tolist()
 c.update(initial_simulation_snapshot=snapshot,continuation=mode,joint_speed_limit_rad_s=.02,
  teacher_command_lead_limit_rad=.015,gripper_speed_limit_m_s=.004,teacher_gripper_feedback_lead_m=.008,
  correction_source=dict(origin=str(origin),query_step=int(record['query_steps'][index]),split=split,role='partial_not_full',reason='run11 supervised execution failure context',orientation='retain measured grasp orientation for source-contact continuations' if mode=='grasp_to_retreat' else 'original task orientation'))
 c['dwell_seconds']['close']=8;c['dwell_seconds']['release']=8;path=out/f'episode_{episode}.hdf5'
 try:
  result=collect_episode(c,path);replay=verify_replay(path)
  entries.append(dict(path=str(path),status='valid',task_role=result['task'],full_task_verified=False,correction_verified=True,split=split,frames=result['frames'],origin=c['correction_source'],replay=replay,prefix_contract_audit=result['prefix_contract_audit']))
  print(json.dumps(dict(episode=episode,success=True,frames=result['frames'],replay=replay)),flush=True)
 except Exception as exc:
  rejected.append(dict(episode=episode,error=str(exc)));print(json.dumps(rejected[-1]),flush=True)
 attempt.write_text(json.dumps(dict(episodes=entries,rejected=rejected),indent=2))
 if rejected:raise RuntimeError('Rejected candidate retained; no accepted manifest')
accepted.write_text(json.dumps(dict(episodes=base['episodes']+entries,rejected=[],source_attempt_manifests=base['source_attempt_manifests']+[str(attempt)],excluded_source_candidates=base['excluded_source_candidates']),indent=2))
