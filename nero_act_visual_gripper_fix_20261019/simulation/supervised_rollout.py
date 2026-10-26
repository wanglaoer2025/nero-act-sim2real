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
from executor.model import physical_actions,make_policy_observation
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
    active=0;rejections=0;failure=None;support_pose=None;held_center=held_rotation=None;guard_diagnostic=None
    visual=policy.config.get('visual_observation_profile')=='visual_relative_base_v1'
    reach=min(policy.config.get('reach_deadline_steps',3200),3900) if visual else policy.config.get('reach_deadline_steps',3200)
    started=time.monotonic();freeze_step=None
    visual_inputs=[];visual_records=[];query_times=[];inference_times=[]
    visual_observer=None
    exposures=None;last_submission_sim_s=float('-inf');confirmed_sim_s=float('-inf')
    if visual:
        from simulation.visual_evaluation import ImageFeatures,ExposureHistory
        visual_observer=ImageFeatures(config,(180,320),camera_translation_m=scenario.get('observation_camera_translation_m'))
        exposures=ExposureHistory(scenario.get('observation_delay_steps',0))
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
        if exposures is not None:
            exposures.append(dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),stamp=float(data.time),previous=previous.copy()))
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
        with mujoco.Renderer(model,height=180 if visual else 90,width=320 if visual else 160) as renderer:
            while active<reach:
                m=observe(model,data,config);contact.update(m,previous[7],float(data.time))
                state=actual();render_data=data;observation_stamp=float(data.time);context_width=previous[7]
                if visual and scenario.get('observation_delay_steps',0):
                    try:exposure=exposures.take()
                    except ValueError:
                        pause(2,'camera_delay_warmup_hold');continue
                    render_data=mujoco.MjData(model)
                    render_data.qpos[:]=exposure['qpos'];render_data.qvel[:]=exposure['qvel'];render_data.time=exposure['stamp']
                    mujoco.mj_forward(model,render_data)
                    observation_stamp=exposure['stamp'];context_width=exposure['previous'][7]
                if visual and not scenario.get('observation_delay_steps',0) and observation_stamp<=confirmed_sim_s:
                    pause(1,'post_confirmation_fresh_camera_wait');continue
                if visual and (observation_stamp<=last_submission_sim_s or observation_stamp<=confirmed_sim_s):
                    raise RuntimeError('Delayed exposure predates last submission/confirmation; no old prediction reused')
                renderer.update_scene(render_data,camera='calibrated_nero_camera')
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                image=renderer.render().copy()
                observed=(np.r_[render_data.qpos[:7],render_data.qpos[7]-render_data.qpos[8]] if visual else state).astype(np.float32)
                if scenario['jitter']:observed[:7]+=rng.uniform(-.0002,.0002,7)
                features=None
                if visual:
                    features,record=visual_observer(image,observed)
                    record.update(exposure_simulation_s=observation_stamp,consumed_simulation_s=float(data.time),
                        paired_feedback_state=observed.tolist(),causal_previous_gripper_target_m=float(context_width))
                    visual_inputs.append(features);visual_records.append(record);query_times.append(observation_stamp)
                    if len(visual_inputs)%20==1:
                        import cv2
                        images=output.parent/(output.name+'_images');images.mkdir(exist_ok=True)
                        cv2.imwrite(str(images/f'query_{len(visual_inputs)-1:05d}.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                inference_start=time.monotonic()
                with torch.inference_mode():
                    tensor=policy(torch.from_numpy(make_policy_observation(observed,stats,policy.config,context_width,geometry_features_m=features))[None].to(policy.device),
                        preprocess_images(image[None],policy.config.get('image_size',[90,160]))[None].to(policy.device))
                raw=physical_actions(tensor[0],stats,bounded_gripper=policy.bounded_gripper).cpu().numpy()
                inference_times.append(time.monotonic()-inference_start)
                queries.append(raw);query_steps.append(len(states));query_states.append(state)
                snapshots.append(capture(model,data,servo,previous))
                try:rows=guard.validate_prefix(raw,dict(state=state,age_s=float(data.time)-observation_stamp if visual else scenario['observation_age_s']),previous)
                except ValueError:
                    rejections+=1;raise
                if visual:rows=rows[:2]
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
                        else:
                            cup_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'cup')
                            guard_diagnostic=dict(step=len(states),current_actual=state.tolist(),previous=previous.tolist(),
                                raw_prefix=raw[:10].tolist(),checked_prefix=rows.tolist(),
                                current_tcp=current.tolist(),proposed_tcp=poses.tolist(),
                                predicted_cup_bottom_m=[bottom(row) for row in rows],
                                held_center=held_center.tolist(),held_rotation=held_rotation.tolist(),
                                actual_cup_center=data.xpos[cup_id].tolist(),actual_cup_rotation=data.xmat[cup_id].reshape(3,3).tolist(),
                                measured_physics=m,contact_width=contact.contact_width,
                                actual_attached_model_bottom_m=bottom(state))
                            raise RuntimeError('Attached-cup model would cross the table')
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
                    if visual:
                        confirmed_sim_s=float(data.time);exposures.reset()
                    continue
                if fine_grasp_query:rows=rows[:1]
                if visual:last_submission_sim_s=float(data.time)
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
    if visual:
        result.update(observation_schema='visual_relative_base_v1',maximum_ACT_prefix_points=2,
            geometry_truth_policy_input=False,nominal_geometry_only=True,
            inference_latency_p95_ms=float(np.percentile(inference_times,95)*1000) if inference_times else None)
    result['guard_diagnostic']=guard_diagnostic
    result['execution_contract_failures']+=rejections
    result['success']&=failure is None and phases.retreat_confirmed and contact.state=='released'
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'rollout.npz',states=np.asarray(states),sent_targets=np.asarray(targets),
        raw_chunks=np.asarray(queries),query_steps=query_steps,feedback_at_query=np.asarray(query_states),
        target_sources=np.asarray(sources),visual_features_m=np.asarray(visual_inputs),
        query_simulation_time_s=query_times,inference_time_s=inference_times)
    if visual:(output/'visual_observations.json').write_text(json.dumps(visual_records))
    for name,value in [('measurements',measures),('scene_configuration',config),('result',result),
            ('physical_snapshots',dict(query_steps=query_steps,snapshots=snapshots))]:
        (output/(name+'.json')).write_text(json.dumps(value,indent=2 if name=='result' else None))
    return result
