"""Verified partial corrections supplement, and never replace, full demonstrations."""
import hashlib,json
from pathlib import Path
import h5py
import numpy as np


def validate_recovery_scene(config,original,snapshot):
    overrides={'continuation','initial_simulation_snapshot','correction_source','initial_robot_state',
        'ik_goal_rotation_link7','joint_speed_limit_rad_s','teacher_command_lead_limit_rad',
        'gripper_speed_limit_m_s','teacher_gripper_feedback_lead_m','dwell_seconds'}
    if {k:v for k,v in config.items() if k not in overrides}!={k:v for k,v in original.items() if k not in overrides}:
        raise ValueError('Recovery changed the original physical scene/model')
    expected=dict(joint_speed_limit_rad_s=.02,teacher_command_lead_limit_rad=.015,
        gripper_speed_limit_m_s=.004,teacher_gripper_feedback_lead_m=.008)
    expected['dwell_seconds']=dict(original.get('dwell_seconds',{}),close=8,release=8)
    expected['initial_robot_state']=dict(qpos_rad=snapshot['qpos'][:7],
        gripper_total_width_m=snapshot['qpos'][7]-snapshot['qpos'][8])
    if any(config.get(k)!=v for k,v in expected.items()):
        raise ValueError('Only the declared slower teacher and exact snapshot start overrides are allowed')
    if (config.get('continuation') in ('descend_to_retreat','grasp_to_retreat')
            and config.get('ik_goal_rotation_link7')!=original.get('ik_goal_rotation_link7')):
        raise ValueError('Descent correction must recover the original grasp orientation')


def validate_corrections(manifest,base_rows,*,check_files=True):
    rows=manifest['episodes'];ids={r['id'] for r in base_rows}
    paths={str(Path(r['path']).resolve()) for r in base_rows}
    groups={r['source_group']:r.get('split') for r in base_rows}
    origins={str(Path(r['origin']).resolve()):r.get('split') for r in base_rows if 'origin' in r}
    if not manifest.get('complete') or manifest.get('rejected') or not rows:
        raise ValueError('Only a complete audited correction batch can enter training')
    evaluation=Path(manifest['source_evaluation']).resolve()
    def require_hash(path,expected):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Correction provenance or physical dataset changed: '+str(path))
    if check_files:
        protocol=evaluation/'protocol.json';require_hash(protocol,manifest['source_protocol_sha256'])
        if json.loads(protocol.read_text())['checkpoint_sha256']!=manifest['source_checkpoint_sha256']:
            raise ValueError('Source ACT checkpoint does not match its frozen evaluation')
    accepted=[]
    for row in rows:
        index=row['id'];path=Path(row['path']);origin=Path(row['origin']).resolve();split=row['split'];group=row['source_group']
        resolved=str(path.resolve())
        if (not isinstance(index,int) or index<1000 or index in ids or resolved in paths
                or path.name!=f'episode_{index}.hdf5' or origin.parent!=evaluation
                or split not in ('train','val') or not group
                or group in {r['source_group'] for r in base_rows}
                or groups.get(group,split)!=split or origins.get(str(origin),split)!=split
                or row.get('status')!='valid' or row.get('task') not in ('lift_to_retreat_correction','descend_to_retreat_correction','grasp_to_retreat_correction')
                or row.get('teacher_success') is not True or row.get('full_task_verified') is not False
                or row.get('correction_verified') is not True or not row.get('replay',{}).get('passed')
                or not row.get('prefix_contract_audit',{}).get('passed')):
            raise ValueError('Separate verified partial roles, IDs and independent grouped splits required')
        ids.add(index);paths.add(resolved);groups[group]=split;origins[str(origin)]=split
        if check_files:
            require_hash(path,row['sha256']);require_hash(origin/'physical_snapshots.json',row['snapshot_sha256'])
            require_hash(origin/'scene_configuration.json',row['scene_sha256'])
            record=json.loads((origin/'physical_snapshots.json').read_text())
            step=row['query_step']
            if step not in record['query_steps']:raise ValueError('Unknown source query step')
            snapshot=record['snapshots'][record['query_steps'].index(step)]
            with h5py.File(path,'r') as f:
                if (f.attrs.get('task')!=row['task'] or f.attrs.get('full_task_verified',True)
                        or not all(f.attrs.get(k,False) for k in ('sim','teacher_success','same_cycle_observation_before_action'))):
                    raise ValueError('Correction HDF role differs from manifest')
                config=json.loads(f.attrs['scene_config']);source=config['correction_source']
                validate_recovery_scene(config,json.loads((origin/'scene_configuration.json').read_text()),snapshot)
                if (config.get('continuation')!=row['task'].removesuffix('_correction') or config['initial_simulation_snapshot']!=snapshot
                        or any(source.get(k)!=v for k,v in dict(origin=str(origin),query_step=step,split=split,
                            source_group=group,role='partial_not_full').items())
                        or not np.array_equal(f['simulation/qpos'][0],snapshot['qpos'])
                        or not np.array_equal(f['simulation/qvel'][0],snapshot['qvel'])):
                    raise ValueError('Correction must restore the exact original failed physical state and role')
        accepted.append(path)
    return accepted
