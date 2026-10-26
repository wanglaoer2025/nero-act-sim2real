"""Complete physical expert for simulation data only. Never imports ROS or SDK."""
import math
import numpy as np
import mujoco

from simulation.scene import build_scene, tcp_in_link7
from simulation.nero_domain_randomization import NeroServo, disturbance_torque
from simulation.outcome import OutcomeMonitor
from simulation.snapshot import restore

CONTINUATION_ROLES = dict(retreat_only='retreat_correction',
    grasp_to_retreat='grasp_to_retreat_correction', carry_to_retreat='carry_to_retreat_correction',
    place_to_retreat='place_to_retreat_correction')


def task_waypoints(config):
    pick, place = config['pick_xy_m'], config['place_xy_m']
    z = config['table_top_z_m']+config['grasp_tcp_height_m']
    high = z+config['lift_height_m']
    opened = float(config.get('gripper_open_m', .095))
    if not config['cup']['outer_diameter_m']+.005 <= opened <= .095:
        raise ValueError('Gripper must have sufficient finite opening around the cup')
    points = [('approach', [*pick, high], opened), ('descend', [*pick, z], opened),
            ('close', [*pick, z], .065), ('lift', [*pick, high], .065),
            ('carry', [*place, high], .065), ('place', [*place, z], .065),
            ('release', [*place, z], opened), ('retreat', [*place, high], opened)]
    if config.get('continuation') is not None:
        mode = config['continuation']
        if mode not in CONTINUATION_ROLES:
            raise ValueError('Unknown explicitly labeled simulation continuation')
        return points[dict(retreat_only=7, grasp_to_retreat=2, carry_to_retreat=4, place_to_retreat=5)[mode]:]
    return points


def inverse_kinematics(model, point, seed, rotation):
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_tcp')
    point, rotation = np.asarray(point, float), np.asarray(rotation, float)
    rng = np.random.default_rng(4)
    for attempt in range(20):
        q = seed.copy() if attempt == 0 else rng.uniform(model.jnt_range[:7, 0]*.7,
                                                       model.jnt_range[:7, 1]*.7)
        for _ in range(300):
            data.qpos[:7] = q
            mujoco.mj_forward(model, data)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (rotation@data.site_xmat[site].reshape(3, 3).T).ravel())
            if quat[0] < 0:
                quat *= -1
            pos_error, rot_error = point-data.site_xpos[site], 2*quat[1:]
            if np.linalg.norm(pos_error) < .0001 and np.linalg.norm(rot_error) < .001:
                return q
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, site)
            jac = np.vstack([jp[:, :7], .3*jr[:, :7]])
            error = np.r_[pos_error, .3*rot_error]
            delta = jac.T@np.linalg.solve(jac@jac.T+.002**2*np.eye(6), error)
            q = np.clip(q+np.clip(delta, -.1, .1), model.jnt_range[:7, 0]+.01,
                        model.jnt_range[:7, 1]-.01)
    raise RuntimeError(f'No converged IK for {point.tolist()}')


def cartesian_path(model, start_q, point, rotation):
    data = mujoco.MjData(model)
    data.qpos[:7] = start_q
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_tcp')
    initial = data.site_xpos[site].copy()
    count = max(1, math.ceil(float(np.linalg.norm(np.asarray(point)-initial))/.005))
    seed, path = start_q.copy(), []
    for alpha in np.linspace(1/count, 1, count):
        seed = inverse_kinematics(model, initial+(np.asarray(point)-initial)*alpha, seed, rotation)
        path.append(seed.copy())
    return path


def observe(model, data, config, *, contract_valid=True):
    cup = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cup')
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_tcp')
    fingers = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
               for name in ['gripper_link1', 'gripper_link2']}
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'table')
    contacts, supported, unexpected, robot_contact = set(), False, False, False
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    for contact in data.contact[:data.ncon]:
        geoms = {int(contact.geom1), int(contact.geom2)}
        bodies = {int(model.geom_bodyid[g]) for g in geoms}
        if cup in bodies:
            contacts.update(bodies & fingers)
            supported |= table in geoms
            unexpected |= bool(bodies-set([0, cup, *fingers]))
        elif bodies != {0, base} and contact.dist < -1e-6:
            robot_contact = True
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, cup, velocity, 0)
    return dict(cup_xyz=data.xpos[cup].tolist(), tcp_xyz=data.site_xpos[site].tolist(),
                cup_upright=float(data.xmat[cup].reshape(3, 3)[2, 2]),
                cup_speed=float(np.linalg.norm(velocity)), bilateral=len(contacts) == 2,
                supported=bool(supported), unexpected_contact=bool(unexpected or robot_contact),
                unexpected_robot_contact=bool(robot_contact),
                gripper_width=float(data.qpos[7]-data.qpos[8]), contract_valid=bool(contract_valid))


def expert_arm_target(desired, previous, measured, config):
    increment = config['joint_speed_limit_rad_s']/config['control_hz']
    lead = config.get('teacher_command_lead_limit_rad', .03)
    if not math.isfinite(lead) or not 0 < lead <= .03:
        raise ValueError('Expert lead must remain within the original30mrad protection')
    lower = np.maximum(previous-increment, measured-lead)
    upper = np.minimum(previous+increment, measured+lead)
    if np.any(lower > upper):
        raise RuntimeError('Expert feedback cannot satisfy rate/lead intersection')
    return np.clip(desired, lower, upper).astype(np.float32)


def expert_gripper_target(desired, previous, config, measured_width=None):
    step = config.get('gripper_speed_limit_m_s', .01)/config['control_hz']
    if not math.isfinite(step) or not 0 < step <= .0005:
        raise ValueError('Expert gripper rate cannot exceed0.5mm per point')
    if 'teacher_gripper_feedback_lead_m' in config:
        lead = config['teacher_gripper_feedback_lead_m']
        if measured_width is None or not math.isfinite(lead) or not 0 < lead <= .01:
            raise ValueError('Measured width and valid expert feedback lead required')
        desired = max(desired, measured_width-lead)
    return float(np.float32(np.clip(desired, previous-step, previous+step)))


def check_stage_start(stage, measurement):
    if stage == 'lift' and not measurement['bilateral']:
        raise RuntimeError('No bilateral physical contact after closing; refuse lift')
    if stage == 'retreat' and not measurement['gripper_width'] > .09:
        raise RuntimeError('Actual gripper is not open/clear before retreat')


def run_teacher(config, record=None):
    points = task_waypoints(config)
    snapshot = config.get('initial_simulation_snapshot')
    if config.get('continuation') in ['grasp_to_retreat', 'carry_to_retreat', 'place_to_retreat'] and snapshot is None:
        raise ValueError('New contact correction requires an exact physical snapshot')
    model = build_scene(config)
    data = mujoco.MjData(model)
    data.qpos[:7] = config['initial_robot_state']['qpos_rad']
    width = config['initial_robot_state']['gripper_total_width_m']
    data.qpos[7:9] = [width/2, -width/2]
    partial = config.get('continuation') is not None
    if config.get('continuation') == 'retreat_only' and snapshot is None:
        xyz = np.asarray(config.get('correction_initial_cup_xyz_m', [
            *config['place_xy_m'], config['table_top_z_m']+.0485]), float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError('Invalid initial correction cup pose')
        data.qpos[9:12] = xyz
    mujoco.mj_forward(model, data)
    servo = NeroServo(model, data, config)
    if snapshot is not None:
        previous = restore(model, data, servo, snapshot)
        if (not np.array_equal(data.qpos[:7], config['initial_robot_state']['qpos_rad'])
                or abs(data.qpos[7]-data.qpos[8]-width) > 1e-12):
            raise ValueError('Configured correction start disagrees with exact physical snapshot')
    else:
        previous = np.r_[data.qpos[:7], data.qpos[7]-data.qpos[8]]
    monitor = OutcomeMonitor(config['pick_xy_m'], config['place_xy_m'], config['table_top_z_m'])
    _, local_rotation = tcp_in_link7(model)
    rotation = np.asarray(config['ik_goal_rotation_link7'])@local_rotation
    seed = data.qpos[:7].copy()
    targets = []
    for stage, point, grip in points:
        if stage not in ('close', 'release'):
            path = cartesian_path(model, seed, point, rotation)
            seed = path[-1].copy()
        else:
            path = [seed.copy()]
        targets.append((stage, path, grip))
    trace, summaries = [], []
    increment = config['joint_speed_limit_rad_s']/config['control_hz']
    physics_steps = round(1/config['control_hz']/model.opt.timestep)

    def tick(desired, width, stage):
        nonlocal previous
        target = expert_arm_target(desired, previous[:7], data.qpos[:7], config)
        grip = expert_gripper_target(width, previous[7], config, measured_width=data.qpos[7]-data.qpos[8])
        action = np.r_[target, grip].astype(np.float32)
        before = data.qpos.copy()
        if record is not None:
            record(model, data, action, stage)
        data.qfrc_applied[:7] = disturbance_torque(config, len(trace))
        servo.command(target, grip)
        bad = False
        for _ in range(physics_steps):
            servo.apply()
            mujoco.mj_step(model, data)
            if any(w.number for w in data.warning) or not np.isfinite(data.qpos).all():
                raise RuntimeError('Unstable or nonfinite physics')
            bad |= observe(model, data, config)['unexpected_contact']
        mujoco.mj_forward(model, data)
        measurement = observe(model, data, config)
        measurement['unexpected_contact'] |= bad
        monitor.update(measurement)
        trace.append(dict(t=float(data.time), stage=stage, action=action.tolist(),
                          measured_qpos=before.tolist(), **measurement))
        previous = action.astype(float)
        if bad:
            raise RuntimeError(f'Unexpected robot/cup contact in {stage}')

    error = None
    try:
        for stage, path, grip in targets:
            check_stage_start(stage, observe(model, data, config))
            for target in path:
                start = previous[:7].copy()
                count = max(1, math.ceil(float(np.max(abs(target-start)))/increment))
                for index in range(count):
                    tick(start+(target-start)*(index+1)/count, grip, stage)
            dwell = config['dwell_seconds'].get(stage, config['dwell_seconds']['moving'])
            for _ in range(round(config['control_hz']*dwell)):
                tick(target, grip, stage)
            summaries.append(dict(stage=stage, **observe(model, data, config)))
        for _ in range(110):
            tick(target, grip, 'settle')
    except RuntimeError as exc:
        error = str(exc)
    result = dict(monitor.result(), stages=summaries, frames=len(trace), trace=trace,
                  error=error, real_hardware_tested=False, task='complete_pick_carry_place_retreat')
    result['success'] &= error is None
    if partial:
        result.update(task=CONTINUATION_ROLES[config['continuation']], full_task_verified=False,
                      success=bool(error is None and monitor.stable_released >= 100
                                   and monitor.unexpected == 0 and not monitor.premature_release))
    else:
        result['full_task_verified'] = bool(result['success'])
    return result
