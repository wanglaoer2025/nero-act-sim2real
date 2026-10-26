"""Exact-state partial supervision for observed ACT target overshoot; simulation only."""
import json
from pathlib import Path
import numpy as np
from simulation.collect import collect_episode,verify_replay

root=Path(__file__).resolve().parents[1];out=root/'training_data_current_source'
evaluation=root/'evaluation/run8_feedback_tanh_v2_frozen'
attempt=out/'run8_target_recovery_attempts.json';accepted=out/'accepted_snapshot_recoveries_v2.json'
if attempt.exists() or accepted.exists():raise FileExistsError('Retain prior recovery attempts')
base=json.loads((out/'accepted_snapshot_recoveries.json').read_text())
entries=[];rejected=[]
requests=[(100,'nominal',1500,'train'),(101,'nominal',1600,'train'),
          (102,'independent_5',1507,'train'),(103,'independent_5',1606,'train'),
          (104,'independent_7',.21,'val'),(105,'independent_7',.15,'val')]
for episode,name,requested,split in requests:
    origin=evaluation/name;c=json.loads((origin/'scene_configuration.json').read_text())
    record=json.loads((origin/'physical_snapshots.json').read_text());steps=np.asarray(record['query_steps'])
    measurements=json.loads((origin/'measurements.json').read_text())
    if isinstance(requested,float):
        choices=[i for i,s in enumerate(steps) if 0<s<len(measurements)
            and measurements[s-1]['bilateral'] and measurements[s-1]['cup_upright']>.98
            and .10<measurements[s-1]['tcp_xyz'][2]<=requested
            and measurements[s-1]['cup_xyz'][1]<c['place_xy_m'][1]-.02
            and measurements[s-1]['cup_xyz'][2]>.10]
        if not choices:raise ValueError('No admissible held overshoot snapshot: '+name)
        index=choices[0]
    else:index=int(np.flatnonzero(steps==requested)[0])
    before=measurements[int(steps[index])-1]
    if not before['bilateral'] or before['cup_upright']<.98:raise ValueError('Cannot invent a held cup')
    snapshot=record['snapshots'][index];q=np.asarray(snapshot['qpos'])
    c['initial_robot_state']=dict(qpos_rad=q[:7].tolist(),gripper_total_width_m=float(q[7]-q[8]))
    c.update(initial_simulation_snapshot=snapshot,continuation='carry_to_retreat',
        joint_speed_limit_rad_s=.02,teacher_command_lead_limit_rad=.015,
        gripper_speed_limit_m_s=.004,teacher_gripper_feedback_lead_m=.008,
        correction_source=dict(origin=str(origin),query_step=int(steps[index]),split=split,
            role='partial_not_full',reason='observed held-cup overshoot beyond accepted blue region'))
    c['dwell_seconds']['close']=8;c['dwell_seconds']['release']=8
    path=out/f'episode_{episode}.hdf5'
    try:
        result=collect_episode(c,path);replay=verify_replay(path)
        entry=dict(path=str(path),status='valid',task_role=result['task'],full_task_verified=False,
            correction_verified=True,split=split,frames=result['frames'],origin=c['correction_source'],
            replay=replay,prefix_contract_audit=result['prefix_contract_audit'])
        entries.append(entry);print(json.dumps(dict(episode=episode,success=True,frames=result['frames'],replay=replay)),flush=True)
    except Exception as exc:
        rejected.append(dict(episode=episode,error=str(exc),origin=c['correction_source']))
        print(json.dumps(rejected[-1]),flush=True)
    attempt.write_text(json.dumps(dict(episodes=entries,rejected=rejected),indent=2))
    if rejected:raise RuntimeError('Failed correction retained; no training manifest published')
accepted.write_text(json.dumps(dict(episodes=base['episodes']+entries,rejected=[],
    source_attempt_manifests=[str(out/'snapshot_recoveries_manifest.json'),str(attempt)],
    excluded_source_candidates=base.get('excluded_source_candidates',[])),indent=2))
