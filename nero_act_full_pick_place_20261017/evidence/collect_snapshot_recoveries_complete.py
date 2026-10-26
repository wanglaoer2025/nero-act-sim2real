"""Curate passed partial recoveries, retaining links to every rejected source attempt."""
import json
from pathlib import Path
import numpy as np
from simulation.collect import collect_episode,verify_replay
root=Path(__file__).resolve().parents[1];out=root/'training_data_current_source';origins=root/'evidence/correction_origins'
manifest=out/'accepted_snapshot_recoveries.json'
if manifest.exists():raise FileExistsError(manifest)
old=json.loads((out/'snapshot_recoveries_manifest.json').read_text())
entries=list(old['episodes']);rejected=[]
requests=[
 (82,origins/'run4_current_source_independent_2',792,'grasp_to_retreat','train'),
 (83,origins/'run4_current_source_independent_2',814,'grasp_to_retreat','train'),
 (84,origins/'run4_current_source_independent_3',803,'grasp_to_retreat','val'),
 (86,origins/'run5_prefix_margin_independent_4',.18,'place_to_retreat','train'),
 (87,origins/'run5_prefix_margin_independent_4',.13,'place_to_retreat','train'),
 (88,origins/'run5_prefix_margin_independent_5',.15,'place_to_retreat','val')]
for episode,origin,requested,mode,split in requests:
 c=json.loads((origin/'scene_configuration.json').read_text());record=json.loads((origin/'physical_snapshots.json').read_text());steps=np.asarray(record['query_steps'])
 if isinstance(requested,float):
  measures=json.loads((origin/'measurements.json').read_text())
  choices=[i for i,s in enumerate(steps) if 0<s<len(measures) and measures[s-1]['bilateral']
   and .105<measures[s-1]['tcp_xyz'][2]<=requested and measures[s-1]['cup_xyz'][2]-.0475>.06
   and np.linalg.norm(np.asarray(measures[s-1]['cup_xyz'])[:2]-c['place_xy_m'])<.02]
  if not choices:raise ValueError('No held placement snapshot')
  index=choices[0]
 else:index=int(np.flatnonzero(steps<=requested)[-1])
 snap=record['snapshots'][index];q=np.asarray(snap['qpos'])
 c['initial_robot_state']=dict(qpos_rad=q[:7].tolist(),gripper_total_width_m=float(q[7]-q[8]))
 c.update(initial_simulation_snapshot=snap,continuation=mode,joint_speed_limit_rad_s=.02,
  teacher_command_lead_limit_rad=.015,gripper_speed_limit_m_s=.004,teacher_gripper_feedback_lead_m=.008,
  correction_source=dict(origin=str(origin),query_step=int(steps[index]),split=split,role='partial_not_full'))
 c['dwell_seconds']['close']=8;c['dwell_seconds']['release']=8
 path=out/f'episode_{episode}.hdf5'
 try:
  result=collect_episode(c,path);replay=verify_replay(path)
  entries.append(dict(path=str(path),status='valid',task_role=result['task'],full_task_verified=False,
   correction_verified=True,split=split,frames=result['frames'],origin=c['correction_source'],
   replay=replay,prefix_contract_audit=result['prefix_contract_audit']))
  print(json.dumps(dict(episode=episode,mode=mode,split=split,frames=result['frames'],success=True,replay=replay)),flush=True)
 except Exception as exc:
  rejected.append(dict(episode=episode,error=str(exc)));print(json.dumps(rejected[-1]),flush=True)
 manifest.write_text(json.dumps(dict(episodes=entries,rejected=rejected,
  source_attempt_manifest=str(out/'snapshot_recoveries_manifest.json'),excluded_source_candidates=old['rejected']+[
   dict(episode=50,reason='pilot released fingers not sufficiently open before retreat; excluded')]),indent=2))
 if rejected:raise RuntimeError('Recovery candidate failed; retained, no training')
