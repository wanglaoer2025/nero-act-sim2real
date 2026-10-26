"""Offline ACT trajectory export for a separately bounded, noncontact arm trial."""
import copy
import json
from pathlib import Path
import select
import subprocess
import sys
import time

import mujoco
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
POLICY=ROOT.parent/'nero_act_paired_response_20261023'
sys.path.insert(0,str(POLICY))
from experiment import load_latest,require_unchanged,snapshot
from executor.ros_execution import checked_interpolation
from simulation.scene import build_scene
from simulation.supervised_rollout import rollout
from pyAgxArm.utiles.mdh_kinematics import get_mdh,fk_from_mdh


def main():
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    preflight=json.loads((ROOT/'preflight_01/report.json').read_text())
    if preflight.get('error') or preflight['can_tx_after']!=preflight['can_tx_before']:
        raise ValueError('Clean receive-only preflight required')
    state=np.array(preflight['final_SDK_health']['state'])
    old=ROOT.parent/'nero_motion_request_20261022/existing_calibration_transfer_01'
    scene=json.loads((old/'current_scene_full/scene_configuration.json').read_text())
    output=ROOT/'preparation_01';output.mkdir(exist_ok=False)
    with (output/'perception.log').open('w') as log:
        worker=subprocess.Popen(['/home/user/nero/bin/python','-u',str(old/'perception_worker_table.py'),
            str(old/'current_scene_full/scene_configuration.json'),str(output)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        try:
            def receive():
                if not select.select([worker.stdout],[],[],45)[0]:raise TimeoutError('Scene observer timeout')
                value=json.loads(worker.stdout.readline())
                if value.get('type')=='error':raise ValueError(value['error'])
                return value
            if receive()!={'type':'ready'}:raise RuntimeError('Observer handshake failed')
            worker.stdin.write(json.dumps(dict(index=0,camera_frame=preflight['camera']['frame'],
                camera_received_monotonic_s=preflight['camera']['received_monotonic']))+'\n');worker.stdin.flush()
            observation=receive()
        finally:
            if worker.poll() is None:
                worker.stdin.write('{"stop":true}\n');worker.stdin.flush()
                try:worker.wait(timeout=3)
                except subprocess.TimeoutExpired:worker.terminate();worker.wait(timeout=3)
    (output/'scene_observation.json').write_text(json.dumps(observation,indent=2)+'\n')
    scene['pick_xy_m']=observation['cup'][:2];scene['place_xy_m']=observation['target'][:2]
    scene['visual_marker_xy_m']=scene['place_xy_m']
    scene['initial_robot_state']=dict(qpos_rad=state[:7].tolist(),gripper_total_width_m=float(state[7]))
    checkpoint=POLICY/'training/pilot_A/policy_last.ckpt'
    proof=json.loads((checkpoint.parent/'integrity.json').read_text())
    require_unchanged(proof['source_sha256']);require_unchanged(proof['checkpoint_sha256'])
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['profile']!='paired_position_response_v1' or payload['actual_updates']!=2000:
        raise ValueError('Wrong ACT source')
    policy,stats,_,_=load_latest();policy.load_state_dict(payload['model'],strict=True);policy.requires_grad_(False).eval()
    queries=[0]
    def limit(_module,_args):
        queries[0]+=1
        if queries[0]>200:raise RuntimeError('Bounded_noncontact_export_end_after400_ACT_points')
    hook=policy.register_forward_pre_hook(limit)
    try:
        result=rollout(copy.deepcopy(scene),policy,stats,dict(name='noncontact_prefix_export_only',seed=191899,
            query_stride=2,servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False),
            output/'ACT_source',runtime_profile='history_common_v1')
    finally:hook.remove()
    data=np.load(output/'ACT_source/rollout.npz',allow_pickle=False)
    mdh=get_mdh('nero');fk=lambda q:np.asarray(fk_from_mdh(mdh,np.asarray(q).tolist()))
    if np.linalg.norm(fk(state[:7])[:3]-preflight['final_SDK_health']['flange_pose'][:3])>.003:
        raise ValueError('SDK/model initial flange mismatch')
    rows=[];poses=[];stop='400_point_budget';previous=state.copy()
    for target,source in zip(data['sent_targets'],data['target_sources']):
        if not str(source).startswith('ACT'):continue
        if np.max(abs(target[:7]-previous[:7]))<2e-5:continue
        pose=fk(target[:7])
        if pose[2]<.45:stop='flange_height_floor450mm';break
        if abs(target[7]-state[7])>.001:stop='approach_gripper_prediction_changed';break
        if np.max(abs(target[:7]-previous[:7]))>.002001:stop='ACT_rate_discontinuity';break
        rows.append(np.r_[target[:7],state[7]]);poses.append(pose);previous=rows[-1]
        if len(rows)==400:break
    rows=np.array(rows);poses=np.array(poses)
    if rows.ndim!=2 or len(rows)<20 or np.max(abs(rows[:,:7]-state[:7]))<.01:
        raise ValueError('No meaningful bounded high ACT prefix; do not send token motion')
    conservative=copy.deepcopy(scene)
    conservative['table_top_z_m']=scene['table_top_z_m']+.05
    conservative['cup']['height_m']=.20;conservative['cup']['outer_diameter_m']=.25
    model=build_scene(conservative);physical=mujoco.MjData(model)
    checks=checked_interpolation(state,rows)
    for q in np.vstack([state,checks]):
        if fk(q[:7])[2]<.45:raise ValueError('Interpolated SDK flange below450mm')
        physical.qpos[:7]=q[:7];physical.qpos[7:9]=[q[7]/2,-q[7]/2];mujoco.mj_forward(model,physical)
        for contact in physical.contact[:physical.ncon]:
            if contact.dist>=0:continue
            names={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) for g in [contact.geom1,contact.geom2]}
            if names in ({'world','base_link'},{'world','cup'}):continue
            raise ValueError('Inflated complete-arm/cup/table collision: '+','.join(sorted(names)))
    np.save(output/'arm_targets.npy',rows,allow_pickle=False)
    hashes=snapshot([Path(__file__),ROOT/'preflight.py',ROOT/'preflight_01/report.json',
        Path(preflight['camera']['frame']),output/'scene_observation.json',checkpoint,checkpoint.parent/'integrity.json',
        output/'ACT_source/rollout.npz',output/'ACT_source/scene_configuration.json',output/'arm_targets.npy',
        Path(sys.modules[rollout.__module__].__file__),Path(sys.modules[fk_from_mdh.__module__].__file__)])
    plan=dict(scope='ACT-generated high noncontact arm-only trial; not full pick/place or qualified online visual ACT',
        initial_state=state.tolist(),targets_path=str(output/'arm_targets.npy'),target_count=len(rows),nominal_duration_s=len(rows)*.05,
        sdk_flange_poses=poses.tolist(),sdk_flange_floor_m=.45,speed_percent=5,period_s=.05,
        joint_step_limit_rad=.002,latest_feedback_lead_limit_rad=.012,tracking_error_limit_rad=.012,
        gripper_commands=0,enable_commands=0,reset_commands=0,follower_transitions=0,
        endpoint_rule=stop,joint_excursion_deg=np.rad2deg(np.max(abs(rows[:,:7]-state[:7]),axis=0)).tolist(),
        table_inflation_m=.05,cup_collision_envelope_diameter_m=.25,cup_collision_envelope_height_m=.20,
        geometry_checks=len(checks)+1,collision_passed=True,ACT_model_qualification_passed=False,
        complete_task_not_released=True,independent_TCP_calibration_claimed=False,source_sha256=hashes,
        hardware_commands_sent=False,offline_result_error=result.get('error'))
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({k:plan[k] for k in ['target_count','nominal_duration_s','joint_excursion_deg','geometry_checks','collision_passed','endpoint_rule']}),flush=True)


if __name__=='__main__':main()
