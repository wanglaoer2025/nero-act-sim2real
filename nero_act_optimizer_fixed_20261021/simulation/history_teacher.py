"""Offline teacher interruptions and exact-state continuations, never runtime ACT."""
import copy
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from executor.phases import PhaseGuard
from simulation.collect import save_records,verify_replay,verify_action_contract
from simulation.nero_domain_randomization import NeroServo,disturbance_torque
from simulation.outcome import OutcomeMonitor
from simulation.runtime_checkpoint import capture_runtime,restore_runtime
from simulation.scene import build_scene
from simulation.supervised_rollout import PhysicsContact,require_supported_target
from simulation.supported_attachment import SupportedAttachment
from simulation.teacher import observe,run_teacher
from training.history_recoveries import ROLES,sha


def role_for_stage(stage):
    return {'close':'pre_close','lift':'hold_to_lift','release':'support_release','retreat':'release_retreat'}.get(stage)


def update_teacher_attachment(attachment,flange,cup_center,cup_rotation,*,supported,contact):
    if contact.state=='holding' or attachment.center is not None:
        attachment.observe(flange,cup_center,cup_rotation,supported=supported)


def capture_parent_contexts(path:Path,output:Path)->dict:
    """Replay the verified teacher, checking every source state, with simulated permissions."""
    path,output=Path(path),Path(output);output.mkdir(parents=True,exist_ok=False)
    with h5py.File(path,'r') as f:
        scene=json.loads(f.attrs['scene_config']);states=f['simulation/qpos'][:];vel=f['simulation/qvel'][:]
        actions=f['action'][:];stages=f['simulation/stage'].asstr()[:];torques=f['simulation/external_arm_torque_nm'][:]
    model=build_scene(scene);data=mujoco.MjData(model);data.qpos[:]=states[0];data.qvel[:]=vel[0]
    mujoco.mj_forward(model,data);servo=NeroServo(model,data,scene)
    previous=np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]]
    contact=PhysicsContact();phases=PhaseGuard(scene['pick_xy_m'],scene['place_xy_m'],scene['table_top_z_m']+scene['grasp_tcp_height_m'])
    monitor=OutcomeMonitor(scene['pick_xy_m'],scene['place_xy_m'],scene['table_top_z_m'])
    attachment=SupportedAttachment(radius=scene['cup']['outer_diameter_m']/2,half_height=scene['cup']['height_m']/2)
    rng=np.random.default_rng(42);events=[];captured={};last_stage=None;confirmed=float('-inf');support_pose=None
    flange_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'link7')
    cup_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'cup')
    def track_attachment(measurement):
        update_teacher_attachment(attachment,dict(position=data.xpos[flange_id],rotation=data.xmat[flange_id].reshape(3,3)),
            data.xpos[cup_id],data.xmat[cup_id].reshape(3,3),supported=measurement['supported'],contact=contact)
    for i,(state,action,stage,torque) in enumerate(zip(states,actions,stages,torques)):
        if not np.allclose(data.qpos,state,atol=1e-7,rtol=0):raise ValueError('Teacher context replay diverged from source')
        measurement=observe(model,data,scene);contact.update(measurement,previous[7],float(data.time))
        track_attachment(measurement)
        role=role_for_stage(stage)
        if stage!=last_stage:
            event=None
            if stage=='close':
                if (np.linalg.norm(np.asarray(measurement['cup_xyz'])[:2]-scene['pick_xy_m'])>.005
                        or abs(measurement['tcp_xyz'][2]-scene['table_top_z_m']-scene['grasp_tcp_height_m'])>.002):
                    raise ValueError('Teacher interruption not in confirmed grasp region')
                contact.close_authorized=True;event='GRASP';phases.lowest_before_hold=measurement['tcp_xyz'][2]
                support_pose=np.asarray(measurement['tcp_xyz']).copy()
            elif stage=='carry':
                if contact.state!='holding' or measurement['cup_xyz'][2]-scene['cup']['height_m']/2-scene['table_top_z_m']<=.06:
                    raise ValueError('Teacher carry interruption lacks stable actual lift')
                contact.carry_authorized=True;event='CARRY'
            elif stage=='release':
                require_supported_target(measurement,scene['place_xy_m']);contact.release_authorized=True;event='RELEASE'
                support_pose=np.asarray(measurement['tcp_xyz']).copy()
            elif stage=='retreat':
                require_supported_target(measurement,scene['place_xy_m'])
                if contact.state!='released':raise ValueError('Teacher retreat lacks actual release')
                phases.retreat_confirmed=True;event='RETREAT'
            if event:
                confirmed=float(data.time);events.append(dict(kind=event,physics_step=i,simulated_permission_only=True,real_receipt=False))
        if role and role not in captured:
            control=dict(contact=vars(contact),phases=vars(phases),monitor=vars(monitor),attachment=vars(attachment),
                support_pose=support_pose,active_points=i,global_physics_points=i,events=events,last_submission_sim_s=float(data.time)-.05,
                confirmed_sim_s=confirmed,scenario_rng_state=rng.bit_generator.state,rejections=0,guard_diagnostic=None)
            observation=dict(exposures=None,history=None,receipts=[],history_epoch=len(events),pending=None,last_observation_stamp=float(data.time))
            record=capture_runtime(model,data,servo,previous,control_state=control,observation_state=observation,
                bindings=dict(scope='offline_simulation',source_kind='teacher_interruption',parent_source_sha256=sha(path),role=role))
            destination=output/(role+'.json');destination.write_text(json.dumps(record,allow_nan=False))
            captured[role]=destination
            if len(captured)==4:break
        servo.command(action[:7],action[7]);data.qfrc_applied[:7]=torque
        for _ in range(25):
            servo.apply();mujoco.mj_step(model,data)
            m=observe(model,data,scene)
            track_attachment(m)
            if m['unexpected_contact']:raise RuntimeError('Unexpected contact in original teacher replay')
        mujoco.mj_forward(model,data);m=observe(model,data,scene);monitor.update(m)
        previous=action.astype(float);last_stage=stage
    if set(captured)!=set(ROLES):raise ValueError('Teacher replay did not reach all four roles')
    return captured


def collect_partial(scene:dict,context:Path,role:str,output:Path,*,parent_group:str,split:str)->dict:
    """Keep exact physics and supervision; hold for fresh context, then teach remaining flow."""
    context,output=Path(context),Path(output)
    checkpoint=json.loads(context.read_text());config=copy.deepcopy(scene)
    model=build_scene(config);data=mujoco.MjData(model);servo=NeroServo(model,data,config)
    previous,control,observation=restore_runtime(model,data,servo,checkpoint,expected_bindings=checkpoint['bindings'])
    contact=PhysicsContact();contact.__dict__.update(control['contact'])
    monitor=OutcomeMonitor(config['pick_xy_m'],config['place_xy_m'],config['table_top_z_m']);monitor.__dict__.update(control['monitor'])
    rows=[]
    def row(m,d,action,stage,index):
        cup=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'cup')
        rows.append(dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),action=action.copy(),timestamp=float(d.time),stage=stage,
            rgb=np.zeros((90,160,3),np.uint8),cup_xyz=d.xpos[cup].copy(),torque=disturbance_torque(config,index)))
    offset=control['global_physics_points']
    for tick in range(5):
        row(model,data,previous,'history_context',offset+tick)
        servo.command(previous[:7],previous[7]);data.qfrc_applied[:7]=disturbance_torque(config,offset+tick)
        for _ in range(25):
            servo.apply();mujoco.mj_step(model,data)
            if observe(model,data,config)['unexpected_contact']:raise RuntimeError('Unsafe recovery context warmup')
        mujoco.mj_forward(model,data);m=observe(model,data,config);monitor.update(m);contact.update(m,previous[7],float(data.time))
    from simulation.snapshot import capture
    resumed_physics=capture(model,data,servo,previous)
    config.update(continuation=({**ROLES,"approach":"descend_to_retreat"})[role],initial_simulation_snapshot=resumed_physics,
        initial_robot_state=dict(qpos_rad=data.qpos[:7].tolist(),gripper_total_width_m=float(data.qpos[7]-data.qpos[8])),
        runtime_physics_offset=offset+5,joint_speed_limit_rad_s=.02,teacher_command_lead_limit_rad=.015,
        gripper_speed_limit_m_s=.004,teacher_gripper_feedback_lead_m=.008,
        dwell_seconds=dict(config['dwell_seconds'],close=8,release=8))
    control['monitor']=vars(monitor);control['contact']=vars(contact)
    # Exact solver state after the five context holds is the teacher start.
    def record(m,d,action,stage):row(m,d,action,stage,offset+len(rows))
    result=run_teacher(config,record,runtime_control_state=control)
    result.pop('trace')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.result.json').write_text(json.dumps(result,indent=2))
    if not result['success']:raise RuntimeError('Partial teacher failed: '+str(result['error']))
    # The complete file starts before warmup, not at the later teacher start.
    saved_config=dict(config,initial_simulation_snapshot=checkpoint['physics'],runtime_physics_offset=offset,
        initial_robot_state=dict(qpos_rad=checkpoint['physics']['qpos'][:7],
            gripper_total_width_m=float(checkpoint['physics']['qpos'][7]-checkpoint['physics']['qpos'][8])))
    save_records(output,rows,saved_config,result,preserve_action_precision=True)
    with h5py.File(output,'r+') as f:
        f['supervision_valid']=np.arange(len(rows))>=5
        f.attrs.update(source_group=parent_group,split=split,recovery_role=role,
            label_source='offline_verified_teacher',context_sha256=sha(context),full_task_verified=False)
    replay=verify_replay(output);audit=verify_action_contract(output)
    return dict(path=str(output.resolve()),sha256=sha(output),teacher_success=True,remaining_flow_verified=True,
                full_task_verified=False,replay=replay,prefix_contract_audit=audit,frames=len(rows))
