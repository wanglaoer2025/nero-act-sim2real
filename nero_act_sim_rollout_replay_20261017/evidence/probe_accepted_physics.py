"""Saved ACT references through contact physics, not a robot or new policy inference."""
import json
from pathlib import Path
import mujoco,numpy as np
from executor.bundle import verify_model_bundle
from executor.replay_trace import verify_replay
from executor.replay_session import protect_gripper,join_reference
from executor.release_hold import release_reference_transfer,require_support_pose,contact_hold_required
from simulation.scene import build_scene
from simulation.teacher import observe
from simulation.outcome import OutcomeMonitor
from simulation.supervised_rollout import PhysicsContact,require_supported_target
from simulation.nero_domain_randomization import NeroServo
ROOT=Path(__file__).resolve().parents[1];output=ROOT/'evidence/accepted_offset_replay_physics.json'
if output.exists():raise FileExistsError(output)
manifest,source=verify_replay(ROOT,verify_model_bundle(ROOT/'model/frozen_run11'))
from executor.scene_acceptance import execution_scene,scene_hash
config=execution_scene(ROOT,json.loads((ROOT/'model/frozen_run11/scene_config.json').read_text()),manifest);execution_hash=scene_hash(config);config['action_delay_steps']=0
model=build_scene(config);data=mujoco.MjData(model)
data.qpos[:7]=config['initial_robot_state']['qpos_rad'];w=config['initial_robot_state']['gripper_total_width_m']
data.qpos[7:9]=[w/2,-w/2];mujoco.mj_forward(model,data)
servo=NeroServo(model,data,config);contact=PhysicsContact();monitor=OutcomeMonitor(config['pick_xy_m'],config['place_xy_m'],config['table_top_z_m'])
def actual():return np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]]
previous=actual();floor=None;events=[];failure=None;count=0
try:
    for phase in manifest['phases']:
        name=phase['name'];rows=protect_gripper(source[phase['start']:phase['end']],floor)
        if name=='close':contact.close_authorized=True
        elif name=='lift':
            if contact.state!='holding':raise RuntimeError('Physics grasp did not establish contact')
        elif name=='carry_place':
            m=observe(model,data,config)
            if not m['bilateral'] or m['cup_xyz'][2]-.0475<.06:raise RuntimeError('Physics cup was not lifted')
            contact.carry_authorized=True
        elif name=='release':
            require_supported_target(observe(model,data,config),config['place_xy_m']);contact.release_authorized=True
            measured=actual();transfer=release_reference_transfer(previous,measured);rows[:,:7]=measured[:7];rows=np.r_[transfer,rows]
        elif name=='retreat':
            if contact.state!='released':raise RuntimeError('Physics cup was not released')
            require_supported_target(observe(model,data,config),config['place_xy_m'])
            rows=np.r_[join_reference(previous,rows[0]),rows]
        events.append(dict(phase=name,point=count,contact=contact.state))
        for row in rows:
            if floor is not None:row=protect_gripper([row],floor)[0]
            servo.command(row[:7],row[7])
            for _ in range(25):servo.apply();mujoco.mj_step(model,data)
            mujoco.mj_forward(model,data);m=observe(model,data,config);monitor.update(m);contact.update(m,row[7],data.time)
            if contact.state=='holding' and floor is None:floor=contact.contact_width
            if m['unexpected_contact']:raise RuntimeError('Unexpected physics collision')
            previous=row.copy();count+=1
        print('PHYSICS_PHASE '+name+' '+contact.state,flush=True)
except BaseException as exc:failure=f'{type(exc).__name__}: {exc}'
result=dict(monitor.result(),error=failure,events=events,frames=count,measured_contact_width_m=floor,
            mode='offline_physics_ACT_reference_replay',hardware_connected=False,real_receipt=False)
result.update(execution_scene_sha256=execution_hash,trace_sha256=manifest['source_rollout_sha256'])
result['success']=bool(result['success'] and failure is None and contact.state=='released')
output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if not result['success']:raise SystemExit(1)
