"""Explicitly partial recovery examples from exact development policy snapshots."""
import copy,json
from pathlib import Path
import numpy as np
from simulation.collect import collect_episode,verify_replay
root=Path(__file__).resolve().parents[1];out=root/'training_data_current_source'
origins=root/'evidence/correction_origins'
nominal=root/'evidence/run5_nominal_snapshot_diagnostic/nominal'
manifest=out/'snapshot_recoveries_manifest.json'
if manifest.exists():raise FileExistsError(manifest)
requests=[
 (70,nominal,1240,'carry_to_retreat','train'),(71,nominal,1180,'carry_to_retreat','train'),
 (72,origins/'run5_prefix_margin_independent_2',1287,'carry_to_retreat','val'),
 (73,origins/'run5_prefix_margin_independent_4',2800,'retreat_only','train'),
 (74,origins/'run5_prefix_margin_independent_4',3800,'retreat_only','train'),
 (75,origins/'run5_prefix_margin_independent_5',3600,'retreat_only','val'),
 (76,origins/'run4_current_source_independent_2',792,'grasp_to_retreat','train'),
 (77,origins/'run4_current_source_independent_2',814,'grasp_to_retreat','train'),
 (78,origins/'run4_current_source_independent_3',803,'grasp_to_retreat','val'),
 (79,origins/'run5_prefix_margin_independent_4',.18,'place_to_retreat','train'),
 (80,origins/'run5_prefix_margin_independent_4',.13,'place_to_retreat','train'),
 (81,origins/'run5_prefix_margin_independent_5',.15,'place_to_retreat','val')]
entries=[];rejected=[]
for episode,origin,requested,mode,split in requests:
 config=json.loads((origin/'scene_configuration.json').read_text())
 record=json.loads((origin/'physical_snapshots.json').read_text());steps=np.asarray(record['query_steps'])
 if isinstance(requested,float):
  measures=json.loads((origin/'measurements.json').read_text())
  candidates=[i for i,s in enumerate(steps) if 0<s<len(measures)
   and measures[s-1]['bilateral'] and .105<measures[s-1]['tcp_xyz'][2]<=requested
   and measures[s-1]['cup_xyz'][2]-.0475>.06
   and np.linalg.norm(np.asarray(measures[s-1]['cup_xyz'])[:2]-config['place_xy_m'])<.02]
  if not candidates:raise ValueError('No held pre-placement snapshot at requested height')
  index=candidates[0]
 else:index=int(np.flatnonzero(steps<=requested)[-1])
 snapshot=record['snapshots'][index];q=np.asarray(snapshot['qpos']);previous=np.asarray(snapshot['previous_target'])
 config['initial_robot_state']=dict(qpos_rad=q[:7].tolist(),gripper_total_width_m=float(q[7]-q[8]))
 config.update(initial_simulation_snapshot=snapshot,continuation=mode,joint_speed_limit_rad_s=.02,
  teacher_command_lead_limit_rad=.015,gripper_speed_limit_m_s=.004,teacher_gripper_feedback_lead_m=.008,
  correction_source=dict(origin=str(origin),query_step=int(steps[index]),split=split,role='partial_not_full'))
 config['dwell_seconds']['release']=8
 path=out/f'episode_{episode}.hdf5'
 try:
  result=collect_episode(config,path);replay=verify_replay(path)
  entries.append(dict(path=str(path),status='valid',task_role=result['task'],full_task_verified=False,
   correction_verified=True,split=split,frames=result['frames'],origin=config['correction_source'],
   replay=replay,prefix_contract_audit=result['prefix_contract_audit']))
  print(json.dumps(dict(episode=episode,mode=mode,split=split,frames=result['frames'],success=True,replay=replay)),flush=True)
 except Exception as exc:
  rejected.append(dict(episode=episode,error=str(exc)));print(json.dumps(rejected[-1]),flush=True)
 manifest.write_text(json.dumps(dict(episodes=entries,rejected=rejected),indent=2))
 if rejected:raise RuntimeError('Correction rejected; retained, no training')
