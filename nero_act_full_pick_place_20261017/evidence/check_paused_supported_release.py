"""Physics-only fresh ACT inference after a supervision pause; never hardware authorization."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from executor.policy_worker import FrozenAct
from executor.contract import FullTaskContract
from executor.ros_execution import RobotGeometry
from executor.release_hold import hold_act_arm_during_release,require_support_pose
from simulation.scene import build_scene
from simulation.snapshot import restore
from simulation.teacher import observe
from simulation.nero_domain_randomization import NeroServo

parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--act-arm-hold',action='store_true')
parser.add_argument('--feedback-reference-hold',action='store_true')
args=parser.parse_args();args.output.mkdir(exist_ok=False)
config=json.loads((args.case/'scene_configuration.json').read_text())
record=json.loads((args.case/'physical_snapshots.json').read_text())
measurements=json.loads((args.case/'measurements.json').read_text());trace=np.load(args.case/'rollout.npz')
held=[m['gripper_width'] for m in measurements if m['bilateral'] and m['cup_xyz'][2]>.12]
if not held:raise ValueError('No actual held cup in recorded physical rollout')
contact_width=float(np.median(held[:20]));selected=None
for index,step in enumerate(record['query_steps']):
    if not 0<step<len(measurements):continue
    before=measurements[step-1]
    if (before['bilateral'] and before['tcp_xyz'][2]<.085
            and np.linalg.norm(np.asarray(before['cup_xyz'])[:2]-config['place_xy_m'])<.02
            and trace['raw_chunks'][index,:10,7].max()>contact_width-.003):
        selected=index;break
if selected is None:raise ValueError('No admissible RELEASE request in this recorded case')
snapshot=record['snapshots'][selected];model=build_scene(config);data=mujoco.MjData(model)
servo=NeroServo(model,data,config);restore(model,data,servo,snapshot)
previous=np.asarray(snapshot['previous_target']);geometry=RobotGeometry(config)
report=dict(mode='simulation_pause_and_release_guard_diagnostic',real_hardware_tested=False,
    counts_toward_frozen14=False,source_query_step=record['query_steps'][selected],
    pause_seconds=4,success=False,queries=[],measured_trace=[])
report['act_arm_hold']=args.act_arm_hold
def advance(target):
    servo.command(target[:7],target[7])
    for _ in range(25):servo.apply();mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
try:
    for _ in range(80):advance(previous)
    actual=np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]]
    before=observe(model,data,config);report['after_pause']=before
    if (not before['supported'] or before['cup_upright']<.95
            or np.linalg.norm(np.asarray(before['cup_xyz'])[:2]-config['place_xy_m'])>=.02):
        raise RuntimeError('Onsite supported-target RELEASE could not be confirmed after pause')
    support_pose=geometry.pose(actual)['tcp'];policy=FrozenAct(args.model,device='cpu')
    report.update(reference_tcp=support_pose.tolist(),previous_ACT_target=previous.tolist(),
                  after_pause_state=actual.tolist(),held_goal_tcp=geometry.pose(previous)['tcp'].tolist())
    report['feedback_reference_hold']=args.feedback_reference_hold
    if args.feedback_reference_hold:
        start=previous.copy();hold=actual.copy();hold[7]=previous[7]
        count=max(1,int(np.ceil(np.max(abs(hold[:7]-start[:7]))/.002)))
        report['reference_transfer_points']=[]
        for index in range(1,count+1):
            target=start+(hold-start)*index/count;advance(target);previous=target.copy()
            measured=observe(model,data,config);report['reference_transfer_points'].append(measured)
            require_support_pose(measured['tcp_xyz'],support_pose)
        for _ in range(40):
            advance(previous);require_support_pose(observe(model,data,config)['tcp_xyz'],support_pose)
        if not observe(model,data,config)['supported']:
            raise RuntimeError('Cup lost support during zero-motion reference transfer')
    guard=FullTaskContract(policy.config)
    with mujoco.Renderer(model,height=90,width=160) as renderer:
        for query in range(40):
            state=np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]]
            if state[7]>.09:
                report['success']=True;report['fingers_clear_width_m']=float(state[7]);break
            renderer.update_scene(data,camera='calibrated_nero_camera')
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            raw=policy.predict(renderer.render().copy(),state)
            rows=guard.validate_prefix(raw,dict(state=state,age_s=.05),previous)
            if args.act_arm_hold:
                rows=hold_act_arm_during_release(rows,previous)
            poses=np.array([geometry.pose(row)['tcp'] for row in rows])
            shift=np.linalg.norm(poses-support_pose,axis=1)
            report['queries'].append(dict(query=query,width_m=float(state[7]),
                max_tcp_shift_m=float(max(shift)),raw_gripper=raw[:10,7].tolist()))
            if not args.act_arm_hold and max(shift)>.003:
                raise RuntimeError('Real runtime3mm support-pose guard would refuse this fresh ACT block')
            for row in rows:
                advance(row);previous=row.copy()
                measured=observe(model,data,config)
                report['measured_trace'].append(measured)
                if args.act_arm_hold and data.qpos[7]-data.qpos[8]<=.09:
                    require_support_pose(geometry.pose(np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]])['tcp'],support_pose)
                if observe(model,data,config)['unexpected_contact']:
                    raise RuntimeError('Unexpected physical contact during paused-release diagnostic')
    if not report['success']:raise RuntimeError('No measured fingers-clear state within diagnostic budget')
except BaseException as exc:
    report['error']=f'{type(exc).__name__}: {exc}'
    report['final_observation']=observe(model,data,config)
(args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if not report['success']:raise SystemExit(1)
