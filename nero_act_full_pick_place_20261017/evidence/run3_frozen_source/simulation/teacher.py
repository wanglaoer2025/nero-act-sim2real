"""Complete physical expert for simulation data only. Never imports ROS or SDK."""
import math
import numpy as np
import mujoco

from simulation.scene import build_scene, tcp_in_link7
from simulation.nero_domain_randomization import NeroServo, disturbance_torque
from simulation.outcome import OutcomeMonitor


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
    if config.get('continuation') == 'retreat_only':
        return points[-1:]
    if config.get('continuation') is not None:
        raise ValueError('Unknown explicitly labeled simulation continuation')
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


def run_teacher(config, record=None):
    points = task_waypoints(config)
    model = build_scene(config)
    data = mujoco.MjData(model)
    data.qpos[:7] = config['initial_robot_state']['qpos_rad']
    width = config['initial_robot_state']['gripper_total_width_m']
    data.qpos[7:9] = [width/2, -width/2]
    partial = config.get('continuation') == 'retreat_only'
    if partial:
        xyz = np.asarray(config.get('correction_initial_cup_xyz_m', [
            *config['place_xy_m'], config['table_top_z_m']+.0485]), float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError('Invalid initial correction cup pose')
        data.qpos[9:12] = xyz
    mujoco.mj_forward(model, data)
    servo = NeroServo(model, data, config)
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
    previous = np.r_[data.qpos[:7], data.qpos[7]-data.qpos[8]]
    trace, summaries = [], []
    increment = config['joint_speed_limit_rad_s']/config['control_hz']
    physics_steps = round(1/config['control_hz']/model.opt.timestep)

    def tick(desired, width, stage):
        nonlocal previous
        lower = np.maximum(previous[:7]-increment, data.qpos[:7]-.03)
        upper = np.minimum(previous[:7]+increment, data.qpos[:7]+.03)
        if np.any(lower > upper):
            raise RuntimeError('Expert feedback cannot satisfy rate/lead intersection')
        target = np.clip(desired, lower, upper).astype(np.float32)
        grip = float(np.float32(np.clip(width, previous[7]-.0005, previous[7]+.0005)))
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
            if stage == 'lift' and not observe(model, data, config)['bilateral']:
                raise RuntimeError('No bilateral physical contact after closing; refuse lift')
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
        result.update(task='retreat_correction', full_task_verified=False,
                      success=bool(error is None and monitor.stable_released >= 100
                                   and monitor.unexpected == 0))
    else:
        result['full_task_verified'] = bool(result['success'])
    return result
