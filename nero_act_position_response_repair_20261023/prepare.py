"""Physical pregrasp counterfactuals, explicitly partial and hardware disconnected."""
import copy
import json
import math
from pathlib import Path
import sys

import h5py
import mujoco
import numpy as np

from objective import pregrasp_goal

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'nero_act_transfer_resume_20261021'))
import checked_runtime
from simulation.scene import build_scene, tcp_in_link7
from simulation.teacher import cartesian_path, expert_arm_target, expert_gripper_target, observe
from simulation.nero_domain_randomization import NeroServo
from simulation.collect import save_records, verify_replay, verify_action_contract
from training.legacy_visual_data import derive_episode, training_profile, sha


def definitions():
    current = ROOT.parent/'nero_act_position_adaptation_20261022/data/visual/episode_1000.hdf5'
    old = json.loads((ROOT.parent/'nero_act_grouped_gripper_20261021/training_data/uniform_complete_v1/manifest.json').read_text())
    reference = Path(next(r['path'] for r in old['episodes'] if r['id']==14))
    poses = []
    for label, path, starts in [('current', current, [0, 710, 800, 880]),
                                ('reference', reference, None)]:
        with h5py.File(path) as stream:
            scene = json.loads(stream.attrs['scene_config'])
            if starts is None:
                descend = np.flatnonzero(stream['simulation/stage'].asstr()[:] == 'descend')
                starts = [int(descend[0]), int(descend[len(descend)//2])]
            for t in starts:
                poses.append(dict(key=f'{label}_{t}', scene=scene,
                    q=stream['simulation/qpos'][t, :7].tolist(), source_path=str(path), source_sha256=sha(path)))
    failure = ROOT.parent/'nero_act_position_recovery_20261022/results/current_development'
    saved = json.loads((failure/'physical_snapshots.json').read_text())['snapshots'][-1]
    scene = json.loads((failure/'scene_configuration.json').read_text())
    poses.append(dict(key='actual_low_failure', scene=scene, q=saved['qpos'][:7],
        source_path=str(failure/'physical_snapshots.json'), source_sha256=sha(failure/'physical_snapshots.json')))
    return poses


def require_no_pregrasp_contact(model, data):
    cup = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cup')
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'table')
    for contact in data.contact[:data.ncon]:
        geoms = {int(contact.geom1), int(contact.geom2)}
        bodies = {int(model.geom_bodyid[g]) for g in geoms}
        if cup in bodies and table not in geoms:
            raise ValueError('Counterfactual touches cup before grasp permission')
    if observe(model, data, {})['unexpected_contact']:
        raise ValueError('Counterfactual robot/environment contact')


def collect_prefix(scene, path):
    model = build_scene(scene); data = mujoco.MjData(model)
    data.qpos[:7] = scene['initial_robot_state']['qpos_rad']
    data.qpos[7:9] = [.0475, -.0475]; data.qvel[:] = 0
    mujoco.mj_forward(model, data); require_no_pregrasp_contact(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_tcp')
    cup = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cup')
    initial_cup = data.xpos[cup].copy(); bottom = np.r_[initial_cup[:2], scene['table_top_z_m']]
    goal, role = pregrasp_goal(data.site_xpos[site], bottom)
    _, local = tcp_in_link7(model)
    rotation = (data.site_xmat[site].reshape(3,3).copy() if role == 'clearance'
                else np.array(scene['ik_goal_rotation_link7'])@local)
    if data.site_xpos[site,2] < .10 and role == 'descend':
        raise ValueError('Aligned grasp inputs belong to existing closing supervision')
    waypoints = cartesian_path(model, data.qpos[:7].copy(), goal, rotation)
    servo = NeroServo(model, data, scene); previous = np.r_[data.qpos[:7], .095]
    desired = []
    for target in waypoints:
        start = previous[:7] if not desired else desired[-1]
        count = max(1, math.ceil(float(np.max(abs(target-start)))/.0015))
        desired.extend(start+(target-start)*(i+1)/count for i in range(count))
        if len(desired)>=20:break
    desired += [desired[-1]]*max(0,20-len(desired))
    rows = []
    for target in desired[:20]:
        arm = expert_arm_target(target, previous[:7], data.qpos[:7], scene)
        grip = expert_gripper_target(.095, previous[7], scene, measured_width=data.qpos[7]-data.qpos[8])
        action = np.r_[arm, grip].astype(np.float32)
        rows.append(dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), action=action,
            timestamp=float(data.time), stage='approach' if role!='descend' else 'descend',
            rgb=np.zeros((90,160,3),np.uint8), cup_xyz=data.xpos[cup].copy(), torque=np.zeros(7)))
        servo.command(arm, grip)
        for _ in range(25):
            servo.apply(); mujoco.mj_step(model, data); require_no_pregrasp_contact(model, data)
        mujoco.mj_forward(model, data)
        if np.linalg.norm(data.xpos[cup,:2]-initial_cup[:2])>.0005:
            raise ValueError('Pregrasp prefix moved the cup')
        previous=action.astype(float)
    result=dict(success=True, task='pregrasp_position_response_correction', full_task_verified=False,
                frames=20, role=role, hardware_deployment_allowed=False,
                limitation='Collision-free pregrasp prefix only; no grasp/lift/place success claim')
    save_records(path, rows, scene, result, preserve_action_precision=True)
    path.with_suffix('.result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    output=ROOT/'data'; output.mkdir(exist_ok=False)
    poses=definitions(); (output/'pose_protocol.json').write_text(json.dumps(poses,indent=2)+'\n')
    train=[(x,y) for x in [-.03,0.,.03] for y in [-.03,0.,.03]]
    val=[(x,y) for x in [-.015,.015] for y in [-.015,.015]]
    rows=[]; rejected=[]; index=3000
    for pose in poses:
        for split, offsets in [('train',train),('val',val)]:
            for dx,dy in offsets:
                index+=1; scene=copy.deepcopy(pose['scene'])
                scene.update(initial_robot_state=dict(qpos_rad=pose['q'],gripper_total_width_m=.095),
                    action_delay_steps=0,joint_speed_limit_rad_s=.03,
                    teacher_command_lead_limit_rad=.016,gripper_speed_limit_m_s=.004,
                    teacher_gripper_feedback_lead_m=.008)
                scene.pop('arm_disturbance',None); scene.pop('initial_simulation_snapshot',None)
                scene['pick_xy_m']=(np.array(scene['pick_xy_m'])+[dx,dy]).tolist()
                base='reference' if pose['key'].startswith('reference') else 'current'
                scene['generalization']=dict(index=index,split=split,
                    source_group=f'position_prefix_{base}_{dx:+.3f}_{dy:+.3f}',
                    pose_key=pose['key'],offset_xy_m=[dx,dy],
                    parent_pose_sha256=pose['source_sha256'],synthetic_stationary_initialization=True)
                physical=output/'physical'/f'episode_{index}.hdf5';physical.parent.mkdir(exist_ok=True)
                try:
                    result=collect_prefix(scene,physical)
                    replay=verify_replay(physical);contract=verify_action_contract(physical)
                    visual=output/'visual'/physical.name
                    report=derive_episode(physical,visual,expected_sha=sha(physical),profile=training_profile(scene))
                    rows.append(dict(report,id=index,status='valid',split=split,
                        source_group=scene['generalization']['source_group'],pose_key=pose['key'],
                        offset_xy_m=[dx,dy],role=result['role'],full_task_verified=False,
                        physical_path=str(physical.resolve()),physical_sha256=sha(physical),
                        replay=replay,prefix_contract_audit=contract))
                except (RuntimeError,ValueError) as exc:
                    rejected.append(dict(id=index,pose_key=pose['key'],split=split,offset=[dx,dy],error=str(exc)))
                manifest=dict(complete=False,episodes=rows,rejected=rejected,
                    teacher_prefixes_are_not_full_task_success=True,hardware_deployment_allowed=False)
                (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(json.dumps(dict(pose=pose['key'],split=split,accepted=len(rows),rejected=len(rejected))),flush=True)
    if not any(r['pose_key']=='actual_low_failure' and r['split']=='train' and r['role']=='clearance' for r in rows):
        raise ValueError('The actual low off-target failure has no accepted correction labels')
    if not all(any(r['split']==split for r in rows) for split in ['train','val']):
        raise ValueError('Missing disjoint position splits')
    manifest['complete']=True
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
