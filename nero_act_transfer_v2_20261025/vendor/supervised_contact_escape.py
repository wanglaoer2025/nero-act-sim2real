"""ACT physics with supervision pauses and the shared zero-motion RELEASE hold.

Permissions here use simulation truth only; they are never real onsite receipts.
"""
import copy
from collections import deque
import hashlib
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
from simulation.supported_attachment import SupportedAttachment
from contact_escape import unsafe_attached_rows
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


def rollout(config,policy,stats,scenario,output,*,resume_context=None,capture_control=False,runtime_profile='legacy_trace_v1',
            resume_policy_transition=False,expected_parent_source_sha256=None,mechanism_role=None):
    if runtime_profile not in ('legacy_trace_v1','history_common_v1'):raise ValueError('Unknown runtime profile')
    common_history=runtime_profile=='history_common_v1'
    if mechanism_role not in (None,'hold_to_lift') or (mechanism_role is not None and (resume_context is None or not common_history)):
        raise ValueError('Lift mechanism is only a bounded offline resumed history experiment')
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
    active=0;rejections=0;failure=None;support_pose=None;guard_diagnostic=None
    attachment=SupportedAttachment(radius=config['cup']['outer_diameter_m']/2,half_height=config['cup']['height_m']/2)
    cup_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'cup')
    flange_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'link7')
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
    history_manager=None
    if common_history:
        from simulation.history_observation import SimulationHistory
        if not visual:raise ValueError('Common history requires visual15D observation')
        history_manager=SimulationHistory(policy.config,stats,scenario,rng,previous)
        history_manager.capture_eligible(data)
    if type(reach) is not int or not 3200<=reach<=6000:raise ValueError('Invalid declared task budget')
    physical_offset=0;pending_observation=None
    runtime_records=deque(maxlen=2);named_contexts={}
    from simulation.runtime_checkpoint import capture_runtime,restore_runtime,resume_bindings
    runtime_bindings=None
    if capture_control or resume_context is not None:
        def hashed(value):
            return hashlib.sha256(json.dumps(value,sort_keys=True,default=lambda a:np.asarray(a).tolist()).encode()).hexdigest()
        weight_digest=hashlib.sha256()
        if hasattr(policy,'state_dict'):
            for key,value in policy.state_dict().items():
                weight_digest.update(key.encode());weight_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        runtime_bindings=dict(scope='offline_simulation',scene_sha256=hashed(config),policy_config_sha256=hashed(policy.config),
            stats_sha256=hashed(stats),scenario_sha256=hashed(scenario),weights_sha256=weight_digest.hexdigest(),runtime_profile=runtime_profile)
    if resume_context is not None:
        expected=resume_bindings(runtime_bindings,resume_context['bindings'],allow_policy_transition=resume_policy_transition,
                                 expected_parent_source_sha256=expected_parent_source_sha256)
        previous,restored,observation=restore_runtime(model,data,servo,resume_context,expected_bindings=expected)
        for obj,key in [(contact,'contact'),(phases,'phases'),(monitor,'monitor'),(attachment,'attachment')]:
            obj.__dict__.clear();obj.__dict__.update(restored[key])
        support_pose=restored['support_pose'];active=restored['active_points'];physical_offset=restored['global_physics_points']
        events=restored['events'];rejections=restored['rejections'];guard_diagnostic=restored['guard_diagnostic']
        last_submission_sim_s=restored['last_submission_sim_s'];confirmed_sim_s=restored['confirmed_sim_s']
        rng.bit_generator.state=restored['scenario_rng_state'];pending_observation=observation['pending']
        if common_history:
            if resume_policy_transition:
                history_manager=SimulationHistory(policy.config,stats,scenario,rng,previous)
                history_manager.epoch=observation['history_epoch']+1
                history_manager.buffer.clear(epoch=history_manager.epoch,after_s=float(data.time))
                for key in ['sent_s','ack_s']:history_manager.receipts[0][key]=float(data.time)-.05
                pending_observation=None
            else:
                if observation['history'] is None:raise ValueError('Full history runtime state required')
                history_manager.restore(observation['history'])
        if exposures is not None and not resume_policy_transition:
            if observation['exposures']['required']!=exposures.required:raise ValueError('Exposure delay checkpoint mismatch')
            exposures.rows=deque(observation['exposures']['rows'],maxlen=exposures.required)
    initial_active=active;active_limit=min(reach,initial_active+400) if mechanism_role is not None else reach
    mechanism_reached=False
    def control_checkpoint(pending):
        return capture_runtime(model,data,servo,previous,control_state=dict(contact=vars(contact),phases=vars(phases),
            monitor=vars(monitor),attachment=vars(attachment),support_pose=support_pose,active_points=active,
            global_physics_points=physical_offset+len(states),events=events,last_submission_sim_s=last_submission_sim_s,
            confirmed_sim_s=confirmed_sim_s,scenario_rng_state=rng.bit_generator.state,rejections=rejections,
            guard_diagnostic=guard_diagnostic),observation_state=dict(
            exposures=None if exposures is None else dict(required=exposures.required,rows=list(exposures.rows)),
            history=None if history_manager is None else history_manager.export(),
            receipts=[] if history_manager is None else history_manager.receipts,
            history_epoch=0 if history_manager is None else history_manager.epoch,pending=pending,
            last_observation_stamp=pending['observation_stamp']),bindings=runtime_bindings)
    def actual():return np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]]
    def update_attachment(measurement):
        if attachment.center is not None and not attachment.airborne:
            # These body poses and support contacts come from the same physics
            # forward pass, including during prefix execution and operator holds.
            attachment.observe(dict(position=data.xpos[flange_id],rotation=data.xmat[flange_id].reshape(3,3)),
                data.xpos[cup_id],data.xmat[cup_id].reshape(3,3),supported=measurement['supported'])
    def advance(target,source,*,prefix_part=False,admit=True):
        nonlocal previous
        states.append(actual());targets.append(np.asarray(target).copy());sources.append(source)
        data.qfrc_applied[:7]=disturbance_torque(config,physical_offset+len(states)-1)
        servo.command(target[:7],target[7]);bad=False
        if common_history and not prefix_part:history_manager.record_prefix(np.asarray(target)[None],float(data.time))
        for _ in range(25):
            servo.apply();mujoco.mj_step(model,data)
            if any(w.number for w in data.warning) or not np.isfinite(data.qpos).all():
                raise RuntimeError('Unstable supervised physical rollout')
            measured=observe(model,data,config);bad|=measured['unexpected_contact']
            update_attachment(measured)
        mujoco.mj_forward(model,data);m=observe(model,data,config);m['unexpected_contact']|=bad
        update_attachment(m)
        monitor.update(m);measures.append(m);previous=np.asarray(target).copy()
        if exposures is not None:
            exposures.append(dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),stamp=float(data.time),previous=previous.copy()))
        contact.update(m,previous[7],float(data.time))
        if support_pose is not None and contact_hold_required(contact):
            require_support_pose(geometry.pose(actual())['tcp'],support_pose)
        if bad:raise RuntimeError('Unexpected robot/environment/cup contact')
        if common_history and admit:
            history_manager.capture_eligible(data)
            history_manager.sample(model,data,renderer,visual_observer)
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
        return attachment.bottom(geometry.pose(state),config['table_top_z_m'])
    try:
        image_hw=policy.config.get('geometry_image_size',[180,320]) if visual else [90,160]
        model.vis.global_.offheight=image_hw[0];model.vis.global_.offwidth=image_hw[1]
        with mujoco.Renderer(model,height=image_hw[0],width=image_hw[1]) as renderer:
            # Teacher interruptions already carry permission, so the ordinary
            # GRASP/RELEASE event cannot perform its measured-reference handoff.
            if (resume_context is not None and resume_policy_transition
                    and resume_context['bindings'].get('source_kind')=='teacher_interruption'
                    and contact_hold_required(contact)):
                prepare_posture_hold()
            fresh_wait_started=None
            while active<active_limit:
                m=observe(model,data,config);contact.update(m,previous[7],float(data.time))
                state=actual()
                if pending_observation is not None:
                    image=pending_observation['image'];observed=pending_observation['observed']
                    features=pending_observation['features'];observation_stamp=pending_observation['observation_stamp']
                    context_width=pending_observation['context_width'];pending_observation=None
                elif common_history:
                    context=history_manager.sample(model,data,renderer,visual_observer)
                    if len(history_manager.buffer.items)<5:
                        pause(1,'history_warmup_hold');continue
                    if context is None or context['observation_stamp']<=max(last_submission_sim_s,confirmed_sim_s):
                        if fresh_wait_started is None:fresh_wait_started=float(data.time)
                        if float(data.time)-fresh_wait_started+.05>.100000001:
                            raise RuntimeError('Fresh exposure unavailable within100ms; no old prediction reused')
                        pause(1,'post_submission_fresh_camera_wait');continue
                    fresh_wait_started=None
                    image=context['image'];observed=context['observed'];features=context['features']
                    observation_stamp=context['observation_stamp'];context_width=context['context_width']
                    visual_inputs.append(features);visual_records.append(copy.deepcopy(context['record']));query_times.append(observation_stamp)
                else:
                    render_data=data;observation_stamp=float(data.time);context_width=previous[7]
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
                if capture_control:
                    saved=control_checkpoint(dict(image=image,observed=observed,features=features,
                        observation_stamp=observation_stamp,context_width=context_width))
                    runtime_records.append(saved)
                    if 'initial' not in named_contexts:named_contexts['initial']=saved
                    if contact.close_authorized and not contact.release_authorized:
                        key='hold_to_lift' if contact.state=='holding' else 'pre_close'
                        if key not in named_contexts:named_contexts[key]=saved
                inference_start=time.monotonic()
                policy_kwargs={}
                if common_history and policy.config.get('numeric_history_profile','none')!='none':
                    batch=history_manager.buffer.snapshot(make_policy_observation(observed,stats,policy.config,context_width,geometry_features_m=features))
                    policy_kwargs['history']={key:torch.from_numpy(np.array(value,copy=True))[None].to(policy.device)
                        for key,value in [('observations',batch.observations),('age_s',batch.age_s),('valid',batch.valid)]}
                with torch.inference_mode():
                    tensor=policy(torch.from_numpy(make_policy_observation(observed,stats,policy.config,context_width,geometry_features_m=features))[None].to(policy.device),
                        preprocess_images(image[None],policy.config.get('image_size',[90,160]))[None].to(policy.device),**policy_kwargs)
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
                    if attachment.center is None:
                        first_lift=first_lift_index(current,poses)
                        if first_lift==0:
                            attachment.observe(geometry.pose(state),data.xpos[cup_id],
                                data.xmat[cup_id].reshape(3,3),supported=m['supported'])
                        elif first_lift is not None:rows=rows[:first_lift]
                    elif not attachment.airborne:
                        attachment.observe(geometry.pose(state),data.xpos[cup_id],
                            data.xmat[cup_id].reshape(3,3),supported=m['supported'])
                    unsafe=unsafe_attached_rows(state,rows,bottom,geometry,measurement=m,airborne=attachment.airborne) if attachment.center is not None else []
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
                                held_center=attachment.center.tolist(),held_rotation=attachment.rotation.tolist(),
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
                        if common_history:history_manager.confirmation_received(confirmed_sim_s)
                    continue
                if fine_grasp_query:rows=rows[:1]
                rows=rows[:active_limit-active]
                if visual:last_submission_sim_s=float(data.time)
                if common_history:history_manager.record_prefix(rows,float(data.time))
                for row_index,row in enumerate(rows):
                    source='ACT_gripper_feedback_arm_hold' if arm_hold else ('ACT_arm_pregrasp_open_hold' if pregrasp_hold else 'ACT')
                    advance(row,source,prefix_part=common_history,admit=row_index==len(rows)-1);active+=1
                    if active>=active_limit:break
                if mechanism_role is not None:
                    current_measurement=observe(model,data,config)
                    if (current_measurement['cup_xyz'][2]-config['cup']['height_m']/2-config['table_top_z_m']>=.02
                            and current_measurement['bilateral'] and contact.state=='holding'):
                        mechanism_reached=True;break
                if scenario['query_stride']>10 and active<active_limit:
                    advance(previous,'cadence_idle');active+=1
                if phases.retreat_confirmed and monitor.carried and monitor.stable_released>=1:
                    freeze_step=len(states)-1;pause(100,'post_completion_hold');break
    except Exception as exc:failure=f'{type(exc).__name__}: {exc}'
    result=dict(monitor.result(),scenario=scenario,frames=len(states),active_policy_points=active,
        queries=len(queries),error=failure,teacher_free=True,real_hardware_tested=False,
        full_task_from_fresh_start=(resume_context is None and not config.get('continuation')
                                   and not config.get('initial_simulation_snapshot')),
        supervision_failures=int(failure is not None and rejections==0),
        supervised_contact_protocol=PROTOCOL,simulated_events=events,post_completion_freeze_step=freeze_step,
        feedback_hold_points=sources.count('feedback_safety_hold'),runtime_s=time.monotonic()-started)
    result['contact_arm_hold_points']=sources.count('ACT_gripper_feedback_arm_hold')
    if resume_context is not None:
        last=monitor.last or observe(model,data,config)
        result.update(resumed_partial_not_full=True,resume_policy_transition=resume_policy_transition,
            new_active_points=active-initial_active,contact_state=contact.state,
            mechanism_success=bool(mechanism_reached and failure is None),
            current_cup_lift_m=float(last['cup_xyz'][2]-config['cup']['height_m']/2-config['table_top_z_m']))
    if visual:
        result.update(observation_schema='visual_relative_base_v1',maximum_ACT_prefix_points=2,
            geometry_truth_policy_input=False,nominal_geometry_only=True,
            attachment_safety_source='current_physics_pose_until_first_liftoff_then_frozen',
            attachment_airborne_transform_frozen=attachment.airborne,
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
    if capture_control:
        (output/'runtime_checkpoints.json').write_text(json.dumps(dict(records=list(runtime_records),named=named_contexts,
            resumed_from_physics_point=physical_offset,runtime_profile=runtime_profile),allow_nan=False))
    return result
