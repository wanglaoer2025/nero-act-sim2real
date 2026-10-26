"""ACT physics with supervision pauses and the shared zero-motion RELEASE hold.

Permissions here use simulation truth only; they are never real onsite receipts.
"""
import copy
import json
import time

import mujoco
import numpy as np
import torch

from executor.contract import FullTaskContract
from executor.phases import PhaseGuard
from executor.release_hold import hold_act_arm_during_release,release_reference_transfer,require_support_pose,contact_hold_required,hold_open_before_grasp_depth,first_lift_index,can_request_support_confirmation
from executor.ros_execution import RobotGeometry
from executor.model import physical_actions
from simulation.scene import build_scene
from simulation.teacher import observe
from simulation.snapshot import capture
from simulation.outcome import OutcomeMonitor
from simulation.nero_domain_randomization import NeroServo,disturbance_torque
from executor.evaluation_integrity import SUPERVISED_PROTOCOL
from src.act.nero_dataset import preprocess_images

PROTOCOL=SUPERVISED_PROTOCOL


class PhysicsContact:
    """True bilateral contact plus the unchanged width gap; not SDK force calibration."""
    def __init__(self):
        self.state='open';self.close_authorized=self.carry_authorized=self.release_authorized=False
        self.first_contact_stamp=self.contact_width=None

    def update(self,m,target,stamp):
        width=m['gripper_width'];valid=m['bilateral'] and .06<=width<=.09 and .003<=width-target<=.012
        if self.state=='holding' and not self.release_authorized:
            if not valid or width<self.contact_width-.003:
                raise RuntimeError('Physical held contact/slip contract failed')
        if self.close_authorized and valid and not self.release_authorized:
            if self.first_contact_stamp is None:self.first_contact_stamp=stamp
            if stamp-self.first_contact_stamp>=.5 and self.state!='holding':
                self.state='holding';self.contact_width=width
        elif not self.release_authorized:self.first_contact_stamp=None
        if self.release_authorized and width>.09 and not m['bilateral']:self.state='released'


def require_supported_target(m,goal):
    if (not m['supported'] or m['cup_upright']<=.95
            or np.linalg.norm(np.asarray(m['cup_xyz'])[:2]-goal)>=.02):
        raise RuntimeError('Simulated onsite confirmation would refuse unsupported/off-target cup')


def rollout(config,policy,stats,scenario,output):
    config=copy.deepcopy(config);rng=np.random.default_rng(scenario['seed'])
    if scenario['jitter']:
        for key in ['pick_xy_m','place_xy_m']:
            config[key]=(np.asarray(config[key])+rng.uniform(-.002,.002,2)).tolist()
        config['initial_robot_state']['qpos_rad']=(np.asarray(config['initial_robot_state']['qpos_rad'])+
            rng.uniform(-.002,.002,7)).tolist()
    config['action_delay_steps']=scenario['servo_delay_steps']
    if scenario['force']:
        config['arm_disturbance']=dict(seed=scenario['seed'],max_torque_nm=.15,
            period_steps=47,duration_steps=3,start_step=90)
    model=build_scene(config);data=mujoco.MjData(model)
    data.qpos[:7]=config['initial_robot_state']['qpos_rad'];width=config['initial_robot_state']['gripper_total_width_m']
    data.qpos[7:9]=[width/2,-width/2];mujoco.mj_forward(model,data)
    servo=NeroServo(model,data,config);geometry=RobotGeometry(config);guard=FullTaskContract(policy.config)
    contact=PhysicsContact();phases=PhaseGuard(config['pick_xy_m'],config['place_xy_m'],config['table_top_z_m']+config['grasp_tcp_height_m'])
    monitor=OutcomeMonitor(config['pick_xy_m'],config['place_xy_m'],config['table_top_z_m'])
    previous=np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]]
    states=[];targets=[];sources=[];measures=[];queries=[];query_steps=[];query_states=[];snapshots=[];events=[]
    active=0;rejections=0;failure=None;support_pose=None;held_center=held_rotation=None
    reach=policy.config.get('reach_deadline_steps',3200);started=time.monotonic();freeze_step=None
    if type(reach) is not int or not 3200<=reach<=6000:raise ValueError('Invalid declared task budget')
    def actual():return np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]]
    def advance(target,source):
        nonlocal previous
        states.append(actual());targets.append(np.asarray(target).copy());sources.append(source)
        data.qfrc_applied[:7]=disturbance_torque(config,len(states)-1)
        servo.command(target[:7],target[7]);bad=False
        for _ in range(25):
            servo.apply();mujoco.mj_step(model,data)
            if any(w.number for w in data.warning) or not np.isfinite(data.qpos).all():
                raise RuntimeError('Unstable supervised physical rollout')
            bad|=observe(model,data,config)['unexpected_contact']
        mujoco.mj_forward(model,data);m=observe(model,data,config);m['unexpected_contact']|=bad
        monitor.update(m);measures.append(m);previous=np.asarray(target).copy()
        contact.update(m,previous[7],float(data.time))
        if support_pose is not None and contact_hold_required(contact):
            require_support_pose(geometry.pose(actual())['tcp'],support_pose)
        if bad:raise RuntimeError('Unexpected robot/environment/cup contact')
        return m
    def pause(count,kind):
        for _ in range(count):advance(previous,kind)
    def prepare_posture_hold():
        nonlocal support_pose
        support_pose=geometry.pose(actual())['tcp']
        transfer=release_reference_transfer(previous,actual())
        for offset in range(0,len(transfer),10):
            part=transfer[offset:offset+10];padded=np.pad(part,((0,20-len(part)),(0,0)),mode='edge')
            checked=guard.validate_prefix(padded,dict(state=actual(),age_s=.05),previous)[:len(part)]
            for row in checked:advance(row,'feedback_safety_hold')
        pause(40,'reference_settle')
    def bottom(state):
        p=geometry.pose(state);center=p['position']+p['rotation']@held_center
        axis=p['rotation']@held_rotation[:,2]
        return float(center[2]-.0475*abs(axis[2])-.0375*np.sqrt(max(0,1-axis[2]**2))-config['table_top_z_m'])
    try:
        with mujoco.Renderer(model,height=90,width=160) as renderer:
            while active<reach:
                m=observe(model,data,config);contact.update(m,previous[7],float(data.time))
                state=actual();renderer.update_scene(data,camera='calibrated_nero_camera')
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                image=renderer.render().copy();observed=state.astype(np.float32)
                if scenario['jitter']:observed[:7]+=rng.uniform(-.0002,.0002,7)
                with torch.inference_mode():
                    tensor=policy(torch.from_numpy((observed-stats['qpos_mean'])/stats['qpos_std'])[None].to(policy.device),
                        preprocess_images(image[None],(90,160))[None].to(policy.device))
                raw=physical_actions(tensor[0],stats,bounded_gripper=policy.bounded_gripper).cpu().numpy()
                queries.append(raw);query_steps.append(len(states));query_states.append(state)
                snapshots.append(capture(model,data,servo,previous))
                try:rows=guard.validate_prefix(raw,dict(state=state,age_s=scenario['observation_age_s']),previous)
                except ValueError:
                    rejections+=1;raise
                arm_hold=support_pose is not None and contact_hold_required(contact)
                if arm_hold:rows=hold_act_arm_during_release(rows,previous)
                current=geometry.pose(state)['tcp'];poses=np.array([geometry.pose(row)['tcp'] for row in rows])
                fine_grasp_query=phases.single_point_grasp_prefix(current,poses,contact)
                event=phases.inspect(current,poses,rows[:,7],previous[7],contact)
                pregrasp_hold=(not contact.close_authorized and
                    abs(current[2]-(config['table_top_z_m']+config['grasp_tcp_height_m']))>.002)
                if pregrasp_hold:
                    rows=hold_open_before_grasp_depth(rows,previous)
                    if event=='GRASP':event=None
                if event=='WAIT_CONTACT':
                    for _ in range(80):
                        advance(previous,'contact_wait')
                        if contact.state=='holding':break
                    if contact.state!='holding':raise RuntimeError('No stable contact within4s')
                    continue
                if contact.state=='holding' and not contact.release_authorized:
                    if held_center is None:
                        first_lift=first_lift_index(current,poses)
                        if first_lift==0:
                            p=geometry.pose(state);center=np.r_[config['pick_xy_m'],config['table_top_z_m']+.0475]
                            held_center=p['rotation'].T@(center-p['position']);held_rotation=p['rotation'].T
                        elif first_lift is not None:rows=rows[:first_lift]
                    unsafe=np.flatnonzero(np.array([bottom(row) for row in rows])<-1e-6) if held_center is not None else []
                    if len(unsafe):
                        if unsafe[0]>0:rows=rows[:unsafe[0]]
                        elif can_request_support_confirmation(current,config['place_xy_m']):
                            event='RELEASE'
                        else:raise RuntimeError('Attached-cup model would cross the table')
                if event is not None:
                    events.append(dict(kind=event,physics_step=len(states),active_points=active,
                        simulated_permission_only=True,real_receipt=False))
                    pause(80,'operator_pause')
                    m=observe(model,data,config)
                    if event=='GRASP':
                        if np.linalg.norm(np.asarray(m['cup_xyz'])[:2]-config['pick_xy_m'])>.005:
                            raise RuntimeError('Source cup moved during simulated GRASP pause')
                        contact.close_authorized=True
                        prepare_posture_hold()
                    elif event=='CARRY':
                        if not m['bilateral'] or m['cup_xyz'][2]-.0475-config['table_top_z_m']<=.06:
                            raise RuntimeError('Simulated lift confirmation failed')
                        contact.carry_authorized=True
                    elif event=='RELEASE':
                        require_supported_target(m,config['place_xy_m']);contact.release_authorized=True
                        prepare_posture_hold();require_supported_target(observe(model,data,config),config['place_xy_m'])
                    elif event=='RETREAT':
                        require_supported_target(m,config['place_xy_m']);phases.retreat_confirmed=True
                    else:raise RuntimeError('Unknown supervision event')
                    continue
                if fine_grasp_query:rows=rows[:1]
                for row in rows:
                    source='ACT_gripper_feedback_arm_hold' if arm_hold else ('ACT_arm_pregrasp_open_hold' if pregrasp_hold else 'ACT')
                    advance(row,source);active+=1
                    if active>=reach:break
                if scenario['query_stride']>10 and active<reach:
                    advance(previous,'cadence_idle');active+=1
                if phases.retreat_confirmed and monitor.carried and monitor.stable_released>=1:
                    freeze_step=len(states)-1;pause(100,'post_completion_hold');break
    except Exception as exc:failure=f'{type(exc).__name__}: {exc}'
    result=dict(monitor.result(),scenario=scenario,frames=len(states),active_policy_points=active,
        queries=len(queries),error=failure,teacher_free=True,real_hardware_tested=False,
        supervision_failures=int(failure is not None and rejections==0),
        supervised_contact_protocol=PROTOCOL,simulated_events=events,post_completion_freeze_step=freeze_step,
        feedback_hold_points=sources.count('feedback_safety_hold'),runtime_s=time.monotonic()-started)
    result['contact_arm_hold_points']=sources.count('ACT_gripper_feedback_arm_hold')
    result['execution_contract_failures']+=rejections
    result['success']&=failure is None and phases.retreat_confirmed and contact.state=='released'
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'rollout.npz',states=np.asarray(states),sent_targets=np.asarray(targets),
        raw_chunks=np.asarray(queries),query_steps=query_steps,feedback_at_query=np.asarray(query_states),
        target_sources=np.asarray(sources))
    for name,value in [('measurements',measures),('scene_configuration',config),('result',result),
            ('physical_snapshots',dict(query_steps=query_steps,snapshots=snapshots))]:
        (output/(name+'.json')).write_text(json.dumps(value,indent=2 if name=='result' else None))
    return result
