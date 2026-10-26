"""Offline exact-state teacher recovery. Never imports hardware interfaces."""
import copy
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

PARENT = Path(__file__).resolve().parents[1] / 'nero_act_grouped_gripper_20261021'
sys.path.append(str(PARENT))
from executor.ros_execution import RobotGeometry
from simulation.nero_domain_randomization import NeroServo, disturbance_torque
from simulation.runtime_checkpoint import _decode, _digest
from simulation.scene import build_scene
from simulation.snapshot import restore
from simulation.supervised_rollout import PhysicsContact
from simulation.supported_attachment import SupportedAttachment
from simulation.teacher import cartesian_path, expert_arm_target, observe


def collect_lift_prefix(scene, checkpoint, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (checkpoint.get('schema') != 'complete_offline_runtime_v1'
            or checkpoint.get('sha256') != _digest({k: v for k, v in checkpoint.items() if k != 'sha256'})
            or checkpoint['bindings'].get('scope') != 'offline_simulation'):
        raise ValueError('Original complete offline failure checkpoint required')
    config = copy.deepcopy(scene)
    config.update(joint_speed_limit_rad_s=.02, teacher_command_lead_limit_rad=.015)
    model = build_scene(config)
    data = mujoco.MjData(model)
    servo = NeroServo(model, data, config)
    previous = restore(model, data, servo, checkpoint['physics'])
    control = _decode(checkpoint['control'])
    contact = PhysicsContact()
    contact.__dict__.update(control['contact'])
    if contact.state != 'holding' or not observe(model, data, config)['bilateral']:
        raise ValueError('Recovery must start at the actual held-cup failure')
    geometry = RobotGeometry(config)
    attachment = SupportedAttachment(radius=config['cup']['outer_diameter_m']/2,
                                     half_height=config['cup']['height_m']/2)
    attachment.__dict__.update(control['attachment'])
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_tcp')
    cup = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cup')
    flange = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link7')
    start = data.site_xpos[site].copy()
    path = cartesian_path(model, data.qpos[:7].copy(), start+[0, 0, .024],
                          data.site_xmat[site].reshape(3, 3).copy())
    rows = {k: [] for k in ['qpos', 'qvel', 'time_s', 'action', 'holding',
                            'predicted_cup_bottom_m', 'target_tcp_dz_m', 'torque']}
    def tick(desired):
        nonlocal previous
        if len(rows['action']) >= 400:
            raise RuntimeError('Original400-point lift-mechanism budget exceeded')
        m = observe(model, data, config)
        attachment.observe(dict(position=data.xpos[flange], rotation=data.xmat[flange].reshape(3, 3)),
                           data.xpos[cup], data.xmat[cup].reshape(3, 3), supported=m['supported'])
        target = expert_arm_target(desired, previous[:7], data.qpos[:7], config)
        action = np.r_[target, previous[7]]
        proposed = geometry.pose(action)
        bottom = attachment.bottom(proposed, config['table_top_z_m'])
        if bottom < -1e-6:
            raise RuntimeError('Teacher target rejected by unchanged attached-cup table guard')
        torque = disturbance_torque(config, control['global_physics_points']+len(rows['action']))
        for key, value in dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), time_s=float(data.time),
                action=action.copy(), holding=contact.state == 'holding', predicted_cup_bottom_m=bottom,
                target_tcp_dz_m=proposed['tcp'][2]-data.site_xpos[site, 2], torque=torque).items():
            rows[key].append(value)
        data.qfrc_applied[:7] = torque
        servo.command(action[:7], action[7])
        for _ in range(25):
            servo.apply()
            mujoco.mj_step(model, data)
            if observe(model, data, config)['unexpected_contact']:
                raise RuntimeError('Unexpected contact in exact-state recovery')
        mujoco.mj_forward(model, data)
        contact.update(observe(model, data, config), action[7], float(data.time))
        if contact.state != 'holding':
            raise RuntimeError('Exact-state recovery lost the held cup')
        previous = action.copy()
    for desired in path:
        initial = previous[:7].copy()
        count = max(1, math.ceil(float(abs(desired-initial).max())/.001))
        for i in range(count):
            tick(initial+(desired-initial)*(i+1)/count)
    for _ in range(20):
        tick(path[-1])
    final = observe(model, data, config)
    lift = final['cup_xyz'][2]-config['cup']['height_m']/2-config['table_top_z_m']
    if lift < .02 or not all(rows['holding']):
        raise RuntimeError('Physical recovery did not retain and lift the cup20mm')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{k: np.asarray(v) for k, v in rows.items()})
    report = dict(verified=True, frames=len(rows['action']), final_cup_lift_m=float(lift),
                  minimum_predicted_cup_bottom_m=float(min(rows['predicted_cup_bottom_m'])),
                  unexpected_contact_frames=0, checkpoint_sha256=checkpoint['sha256'],
                  source_group='nominal_diagnostic_train', split='train',
                  partial_lift_teacher_not_full_demonstration=True, hardware_commands_sent=False)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
