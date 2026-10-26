"""Additional exact-state delayed place/retreat partials; no hardware imports."""
import json
from pathlib import Path
import numpy as np
from simulation.collect import collect_episode,verify_replay

root=Path(__file__).resolve().parents[1];out=root/'training_data_current_source'
attempt=out/'run8_delayed_recovery_attempts.json';accepted=out/'accepted_snapshot_recoveries_v3.json'
if attempt.exists() or accepted.exists():raise FileExistsError('Retain earlier attempts')
base=json.loads((out/'accepted_snapshot_recoveries_v2.json').read_text());entries=[];rejected=[]
requests=[(106,'delay_500ms','retreat_only','train'),
          (107,'independent_0','retreat_only','val'),
          (108,'delay_100ms','carry_to_retreat','train')]
for episode,name,mode,split in requests:
    origin=root/'evaluation/run8_feedback_tanh_v2_frozen'/name
    c=json.loads((origin/'scene_configuration.json').read_text())
    record=json.loads((origin/'physical_snapshots.json').read_text());steps=record['query_steps']
    measurements=json.loads((origin/'measurements.json').read_text());choices=[]
    for i,s in enumerate(steps):
        if not 0<s<len(measurements):continue
        m=measurements[s-1];cup=np.asarray(m['cup_xyz']);tcp=np.asarray(m['tcp_xyz'])
        if mode=='retreat_only':
            valid=(m['supported'] and not m['bilateral'] and m['gripper_width']>.09
                and np.linalg.norm(cup[:2]-c['place_xy_m'])<.02 and .13<tcp[2]<.22)
        else:
            valid=(m['bilateral'] and m['cup_upright']>.98 and cup[2]>.10
                and tcp[2]>.16 and cup[1]<c['place_xy_m'][1]-.02)
        if valid:choices.append(i)
    if not choices:raise ValueError('No valid exact snapshot for '+name)
    index=choices[0] if split=='val' else choices[-1]
    snapshot=record['snapshots'][index];q=np.asarray(snapshot['qpos'])
    c['initial_robot_state']=dict(qpos_rad=q[:7].tolist(),gripper_total_width_m=float(q[7]-q[8]))
    c.update(initial_simulation_snapshot=snapshot,continuation=mode,joint_speed_limit_rad_s=.02,
        teacher_command_lead_limit_rad=.015,gripper_speed_limit_m_s=.004,teacher_gripper_feedback_lead_m=.008,
        correction_source=dict(origin=str(origin),query_step=int(steps[index]),split=split,
            role='partial_not_full',reason='delayed target overshoot or stalled retreat'))
    c['dwell_seconds']['close']=8;c['dwell_seconds']['release']=8
    path=out/f'episode_{episode}.hdf5'
    try:
        result=collect_episode(c,path);replay=verify_replay(path)
        entries.append(dict(path=str(path),status='valid',task_role=result['task'],full_task_verified=False,
            correction_verified=True,split=split,frames=result['frames'],origin=c['correction_source'],
            replay=replay,prefix_contract_audit=result['prefix_contract_audit']))
        print(json.dumps(dict(episode=episode,success=True,frames=result['frames'],replay=replay)),flush=True)
    except Exception as exc:
        rejected.append(dict(episode=episode,error=str(exc)));print(json.dumps(rejected[-1]),flush=True)
    attempt.write_text(json.dumps(dict(episodes=entries,rejected=rejected),indent=2))
    if rejected:raise RuntimeError('Rejected candidate retained; no accepted manifest published')
accepted.write_text(json.dumps(dict(episodes=base['episodes']+entries,rejected=[],
    source_attempt_manifests=base['source_attempt_manifests']+[str(attempt)],
    excluded_source_candidates=base['excluded_source_candidates']),indent=2))
