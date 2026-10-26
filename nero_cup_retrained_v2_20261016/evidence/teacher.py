"""Contact-driven Nero baseline; no attachment constraints or cup pose resets during rollout."""
import argparse
import json
import time
import threading
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from build_nero_model import ROOT, build
from nero_domain_randomization import NeroServo, disturbance_torque
from nero_scene_camera import add_calibrated_camera


def scene(config):
    tree = ET.parse(ROOT / 'assets/nero_mujoco/nero_gripper.xml')
    root = tree.getroot()
    contact = ET.SubElement(root, 'contact')
    # Convex hulls of the fitted wrist housing overlap connected components.
    # These local assembly pairs are excluded explicitly; other self contacts remain enabled.
    for first, second in [('base_link','link1'),('link5','link7'),('link5','flange_link')]:
        ET.SubElement(contact,'exclude',body1=first,body2=second)
    world = root.find('worldbody')
    ET.SubElement(world,'light',pos='0.3 -0.3 1.5',dir='0 0 -1',diffuse='0.8 0.8 0.8')
    ET.SubElement(world,'light',pos='0.2 0.8 1.0',dir='0 -0.5 -1',diffuse='0.5 0.5 0.5')
    z = config['table_top_z_m']
    table_center=np.asarray(config.get('table_center_xy_m',[.4,0]),dtype=float)
    table_half_size=np.asarray(config.get('table_half_size_xy_m',[.5,.5]),dtype=float)
    if (table_center.shape!=(2,) or table_half_size.shape!=(2,)
            or not np.isfinite(np.r_[table_center,table_half_size,z]).all() or (table_half_size<=0).any()):
        raise ValueError('Invalid table centre/half size/height')
    ET.SubElement(world, 'geom', name='table', type='box', pos=f'{table_center[0]} {table_center[1]} {z-0.025}',
                  size=f'{table_half_size[0]} {table_half_size[1]} 0.025', rgba='0.65 0.55 0.4 1')
    add_calibrated_camera(world,config)
    if config.get('show_target_marker',False):
        ET.SubElement(world,'geom',name='target_marker',type='box',
                      pos=f"{config.get('visual_marker_xy_m',config['place_xy_m'])[0]} {config.get('visual_marker_xy_m',config['place_xy_m'])[1]} {z+.0005}",
                      size='.05 .05 .0005',rgba='.05 .4 .8 1',contype='0',conaffinity='0')
    cup = config['cup']
    radius, height, mass, wall = cup['outer_diameter_m']/2, cup['height_m'], cup['mass_kg'], cup['wall_thickness_m']
    if not (0 < 2*wall < min(height, radius) and mass > 0 and 0 < 2*radius < 0.10):
        raise ValueError('Cup requires positive physical values and clearance within 0.10 m gripper opening')
    x, y = config['pick_xy_m']
    body = ET.SubElement(world, 'body', name='cup', pos=f'{x} {y} {z+height/2+0.001}')
    ET.SubElement(body, 'freejoint', name='cup_free')
    # Hollow cup assembled from a bottom disk and 32 tangential wall segments.
    count = 32
    areas = [np.pi*radius**2*wall, count*2*radius*np.sin(np.pi/count)*wall*height]
    bottom_mass = mass*areas[0]/sum(areas)
    segment_mass = (mass-bottom_mass)/count
    ET.SubElement(body, 'geom', name='cup_bottom', type='cylinder', size=f'{radius-wall} {wall/2}',
                  pos=f'0 0 {-height/2+wall/2}', mass=str(bottom_mass), rgba='0.2 0.6 0.85 1')
    for i in range(count):
        angle = 2*np.pi*i/count
        ET.SubElement(body, 'geom', name=f'cup_wall_{i}', type='box',
                      pos=f'{(radius-wall/2)*np.cos(angle)} {(radius-wall/2)*np.sin(angle)} 0',
                      euler=f'0 0 {angle}', size=f'{wall/2} {radius*np.sin(np.pi/count)} {height/2}',
                      mass=str(segment_mass), rgba='0.2 0.6 0.85 1')
    root.find('compiler').set('inertiafromgeom', 'auto')
    for geom in root.iter('geom'):
        geom.set('friction', ' '.join(map(str, config['contact_friction'])))
        geom.set('condim', '4')
    appearance=config.get('appearance')
    if appearance:
        def rgba(rgb):return ' '.join(map(str,[*rgb,1]))
        for geom in root.iter('geom'):
            name=geom.get('name','')
            key='table_rgb' if name=='table' else ('cup_rgb' if name.startswith('cup_') else 'arm_rgb')
            if name!='target_marker':geom.set('rgba',rgba(appearance[key]))
        for light in world.findall('light'):
            diffuse=np.fromstring(light.get('diffuse'),sep=' ')*appearance['light_scale']
            light.set('diffuse',' '.join(map(str,np.clip(diffuse,0,1))))
        asset=root.find('asset')
        ET.SubElement(asset,'texture',name='random_background',type='skybox',builtin='gradient',
                      rgb1=' '.join(map(str,appearance['background_rgb'])),
                      rgb2=' '.join(map(str,np.array(appearance['background_rgb'])*.5)),width='128',height='128')
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))



def ik(model, point, seed, goal_rotation=None):
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_tcp')
    goal_rotation = np.diag([1., -1., -1.]) if goal_rotation is None else np.asarray(goal_rotation,dtype=float)
    if (goal_rotation.shape!=(3,3) or not np.isfinite(goal_rotation).all()
            or not np.allclose(goal_rotation.T@goal_rotation,np.eye(3),atol=1e-6)
            or np.linalg.det(goal_rotation)<.999):raise ValueError('IK orientation must be a proper rotation')
    best = None
    rng = np.random.default_rng(4)
    for attempt in range(20):
        q = seed.copy() if attempt == 0 else rng.uniform(model.jnt_range[:7,0]*0.7, model.jnt_range[:7,1]*0.7)
        for _ in range(300):
            data.qpos[:7] = q
            mujoco.mj_forward(model, data)
            current = data.site_xmat[site].reshape(3,3)
            # Cross-product orientation error is ambiguous at 180 degrees;
            # use the rotation quaternion for a shortest-axis error.
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (goal_rotation @ current.T).reshape(9))
            if quat[0] < 0:
                quat *= -1
            rot_error = 2*quat[1:]
            pos_error = np.asarray(point)-data.site_xpos[site]
            error = np.r_[pos_error, 0.3*rot_error]
            if np.linalg.norm(pos_error) < 0.001 and np.linalg.norm(rot_error) < 0.01:
                return q
            jp, jr = np.zeros((3,model.nv)), np.zeros((3,model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, site)
            jac = np.vstack([jp[:,:7], 0.3*jr[:,:7]])
            delta = jac.T @ np.linalg.solve(jac@jac.T + 0.002**2*np.eye(6), error)
            q = np.clip(q + np.clip(delta, -0.10, 0.10), model.jnt_range[:7,0]+0.01, model.jnt_range[:7,1]-0.01)
        score = np.linalg.norm(error)
        if best is None or score < best[0]:
            best = (score, pos_error.tolist())
    raise RuntimeError(f'IK failed for {point}: residual {best}')


def run(config, viewer=False, output_name='baseline', planner_kind='ik', observation_callback=None):
    model = scene(config)
    data = mujoco.MjData(model)
    cup_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cup')
    cup_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cup_free')
    cup_geoms = set(np.flatnonzero(model.geom_bodyid == cup_id))
    finger_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ['gripper_link1','gripper_link2']]
    finger_geoms = [{int(g) for g in np.flatnonzero(model.geom_bodyid == b)} for b in finger_bodies]
    height = config['cup']['height_m']
    z = config['table_top_z_m'] + height*0.75
    pick, place = config['pick_xy_m'], config['place_xy_m']
    lift = config['lift_height_m']
    points = [('approach', [*pick,z+lift],.095),('descend', [*pick,z],.095),('close', [*pick,z],.065),('lift', [*pick,z+lift],.065)]
    recovery = config.get('recovery_initial_state')
    initial_robot=config.get('initial_robot_state')
    if initial_robot is not None:
        if recovery is not None:raise ValueError('Specify a formal robot start or a recovery state, not both')
        arm=np.asarray(initial_robot['qpos_rad'],dtype=float)
        width=float(initial_robot['gripper_total_width_m'])
        if arm.shape!=(7,) or not np.isfinite(arm).all() or not np.isfinite(width) or not 0<=width<=.1:
            raise ValueError('Invalid measured formal robot start')
        q=model.qpos0.copy();q[:7]=arm;q[7:9]=[width/2,-width/2]
        recovery={'qpos':q.tolist(),'qvel':np.zeros(model.nv).tolist()}
    if recovery is not None:
        if planner_kind != 'ik':
            raise ValueError('Recovery teacher currently supports IK only')
        q = np.asarray(recovery['qpos'], dtype=float)
        v = np.asarray(recovery['qvel'], dtype=float)
        if q.shape != (model.nq,) or v.shape != (model.nv,) or not np.isfinite(np.r_[q,v]).all():
            raise ValueError('Recovery initial state must match the full physical model')
        if np.any(q[:7] < model.jnt_range[:7,0]) or np.any(q[:7] > model.jnt_range[:7,1]):
            raise ValueError('Recovery initial arm state is outside original joint limits')
        data.qpos[:] = q; data.qvel[:] = v
        mujoco.mj_forward(model, data)
        if data.xmat[cup_id].reshape(3,3)[2,2] < .99:
            raise ValueError('This recovery teacher handles upright cups only')
        if np.linalg.norm(data.xpos[cup_id,:2]-pick) > 1e-6:
            raise ValueError('Recovery source must be relocalized to the actual current cup centre')
        tcp = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_tcp')
        current_point = data.site_xpos[tcp].copy()
        clear_point = current_point.copy();clear_point[2] = max(current_point[2],z+lift)
        strategy=(config.get('initial_approach_strategy','open_then_pregrasp') if initial_robot is not None
                  else config.get('recovery_strategy','open_clear_approach'))
        if strategy=='continue_descent':
            width=float(data.qpos[7]-data.qpos[8])
            if width<config['cup']['outer_diameter_m']+.005:
                raise ValueError('Descent continuation requires an already open gripper')
            if np.linalg.norm(current_point[:2]-pick)>.03 or not z+.005<=current_point[2]<=z+lift+.02:
                raise ValueError('Descent continuation requires a nearby tool above grasp height')
            # The student is already approaching an upright, ungrasped cup.
            # Continue that motion; adding a lift-clear detour would teach an
            # opposing action at otherwise nearly identical approach states.
            points=points[1:]
        elif strategy=='open_clear_approach':
            points = [('recover_open',current_point.tolist(),.095),
                      ('recover_clear',clear_point.tolist(),.095)] + points
        elif strategy=='open_then_pregrasp' and initial_robot is not None:
            # Do not demand a downward TCP orientation at the original high
            # horizontal-tool point; it can be unreachable. Open at measured
            # joints, then interpolate toward the pregrasp IK pose.
            points = [('recover_open',current_point.tolist(),.095)] + points
        else:
            raise ValueError(f'Unknown recovery strategy: {strategy}')
    targets = []
    seed = data.qpos[:7].copy() if recovery is not None else np.array(config.get('ik_seed',np.zeros(7)),dtype=float)
    for name, point, width in points:
        if name != 'recover_open':seed = ik(model, point, seed, config.get('ik_goal_rotation'))
        targets.append((name,seed.copy(),width))
    initial_offset=np.asarray(config.get('initial_joint_offset_rad',np.zeros(7)),dtype=float)
    if initial_offset.shape!=(7,) or not np.isfinite(initial_offset).all():
        raise ValueError('Initial joint offset must be a finite seven-dimensional vector')
    if recovery is not None and np.any(initial_offset):
        raise ValueError('An explicit full initial state cannot also have a joint offset')
    if recovery is None:
        data.qpos[:7] = np.clip(targets[0][1]+initial_offset,model.jnt_range[:7,0],model.jnt_range[:7,1])
        data.qpos[7:9] = [0.0475,-0.0475]
    mujoco.mj_forward(model, data)
    kp, kd = np.array(config['arm_kp']), np.array(config['arm_kd'])
    # Integrate the simulation-only servo damping implicitly. Explicit derivative
    # torques are unstable for the small URDF wrist/finger inertias at 2 ms.
    model.dof_damping[:7] = kd
    model.dof_damping[7:9] = 4
    steps = round(1/config['control_hz']/model.opt.timestep)
    trace, stages = [], []
    max_lift, bilateral_frames = 0., 0
    unexpected_contact_frames = 0
    initial_z = float(data.xpos[cup_id,2])
    handle = None
    viewer_threads = []
    if viewer:
        from mujoco import viewer as mj_viewer
        previous_threads = set(threading.enumerate())
        handle = mj_viewer.launch_passive(model, data)
        # On Linux launch_passive owns a daemon render thread; close() only
        # requests its exit. Join that newly created thread before Python's
        # GLFW atexit handler tears down the still-active graphics context.
        viewer_threads = [thread for thread in threading.enumerate()
                          if thread not in previous_threads
                          and '_launch_internal' in thread.name]
        handle.cam.lookat[:] = [0.3,0,0.2]
        handle.cam.distance = 1.3
        handle.cam.azimuth = 135
        handle.cam.elevation = -25
    current = data.qpos[:7].copy()
    servo=NeroServo(model,data,config)
    issued_target=np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]]
    planner = None
    if planner_kind == 'moveit':
        from nero_moveit_planner import NeroMoveItPlanner
        planner = NeroMoveItPlanner(model,data,config)
    def tick(target, width, stage):
        nonlocal max_lift, bilateral_frames, unexpected_contact_frames, issued_target
        wall_start = time.monotonic()
        width=float(np.clip(width,issued_target[7]-.0005,issued_target[7]+.0005))
        # External torques create recoverable deviations through physics only.
        # The same per-control-step schedule is also used by export replay/evaluation.
        data.qfrc_applied[:7]=disturbance_torque(config,len(trace)) if stage in ('approach','descend') else np.zeros(7)
        if 'teacher_command_lead_limit_rad' in config:
            # Match the deployment target slew and feedback lead before recording.
            increment=config['joint_speed_limit_rad_s']/config['control_hz']
            target=issued_target[:7]+np.clip(np.asarray(target)-issued_target[:7],-increment,increment)
            lead=config['teacher_command_lead_limit_rad']
            target=np.clip(target,data.qpos[:7]-lead,data.qpos[:7]+lead)
        # Record the observation BEFORE executing its same-cycle expert action.
        if observation_callback is not None:
            # HDF5/ACT targets are float32. Execute those exact values as well;
            # rounding only on export can diverge under contact dynamics.
            target=np.asarray(target,dtype=np.float32).astype(float)
            width=float(np.float32(width))
            observation_callback(model,data,target.copy(),float(width),stage)
        issued_target=np.r_[target,width]
        servo.command(target,width)
        unexpected_contact = False
        for _ in range(steps):
            servo.apply()
            mujoco.mj_step(model,data)
            if any(w.number for w in data.warning):
                raise RuntimeError('MuJoCo raised a simulation warning; rollout rejected')
            for contact in data.contact[:data.ncon]:
                bodies={int(model.geom_bodyid[contact.geom1]),int(model.geom_bodyid[contact.geom2])}
                if cup_id in bodies and bodies-set([0,cup_id,*finger_bodies]):unexpected_contact=True
        contacts = [False,False]
        for contact in data.contact[:data.ncon]:
            pair = {int(contact.geom1),int(contact.geom2)}
            if pair & cup_geoms:
                for i,geoms in enumerate(finger_geoms):
                    contacts[i] |= bool(pair & geoms)
                other = pair - cup_geoms
                if any(model.geom_bodyid[g] not in (0, *finger_bodies, cup_id) for g in other):
                    unexpected_contact = True
        unexpected_contact_frames += int(unexpected_contact)
        if all(contacts):
            bilateral_frames += 1
        max_lift = max(max_lift, float(data.xpos[cup_id,2]-initial_z))
        trace.append({'t': float(data.time),'stage':stage,'q':data.qpos[:7].tolist(),
                      'qpos':data.qpos.tolist(),
                      'q_target':target.tolist(),'width_m':float(data.qpos[7]-data.qpos[8]),
                      'cup_xyz':data.xpos[cup_id].tolist(),'bilateral_contact':all(contacts)})
        if not np.isfinite(data.qpos).all():
            raise RuntimeError('Nonfinite simulation state')
        if handle:
            handle.sync()
            time.sleep(max(0,1/config['control_hz']-(time.monotonic()-wall_start)))
    try:
        for index,(name, target, width) in enumerate(targets):
            if planner:
                if name == 'lift':
                    if not trace[-1]['bilateral_contact']:
                        raise RuntimeError('No bilateral contact; refuse to attach planning cup')
                    planner.attach(True)
                if name == 'retreat':
                    planner.attach(False)
                if name in ('close','release'):
                    target=current.copy()
                else:
                    times,path = planner.plan(points[index][1],name)
                    for t in np.arange(.02,times[-1]+.02,.02):
                        command=np.array([np.interp(min(t,times[-1]),times,path[:,j]) for j in range(7)])
                        tick(command,width,name)
                    target=path[-1].copy()
            else:
                ticks = max(1,int(np.ceil(np.max(np.abs(target-current))/config['joint_speed_limit_rad_s']*config['control_hz'])))
                for i in range(ticks):
                    alpha = (i+1)/ticks
                    tick(current+(target-current)*alpha,width,name)
            current = target.copy()
            dwell=config.get('dwell_seconds',{})
            seconds=dwell.get(name,2 if name in ('close','release') else dwell.get('moving',.6))
            for _ in range(int(config['control_hz']*seconds)):
                tick(current,width,name)
            stages.append({'stage':name,'cup_xyz':data.xpos[cup_id].tolist(),
                           'max_joint_tracking_error_rad':float(np.max(np.abs(data.qpos[:7]-target)))})
        before_settle = data.xpos[cup_id].copy()
        for _ in range(round(config['control_hz']*config.get('final_settle_seconds',2))):
            tick(current,.065,'hold')
    finally:
        if planner:
            planner.close()
        if handle:
            handle.close()
            for thread in viewer_threads:
                thread.join(timeout=10)
                if thread.is_alive():
                    raise RuntimeError('MuJoCo viewer did not finish shutting down')
    xy_error=float(np.linalg.norm(data.xpos[cup_id,:2]-pick))
    upright=float(data.xmat[cup_id].reshape(3,3)[2,2])
    speed=float(np.linalg.norm(data.qvel[model.jnt_dofadr[cup_joint]:model.jnt_dofadr[cup_joint]+6]))
    final_contacts=set()
    for co in data.contact[:data.ncon]:
        bodies={int(model.geom_bodyid[g]) for g in [co.geom1,co.geom2]}
        if cup_id in bodies:final_contacts.update(bodies&set(finger_bodies))
    final_lift=float(data.xpos[cup_id,2]-initial_z)
    result={'task':'current_cup_pickup_hold','config':config,'stages':stages,'max_lift_m':max_lift,
      'final_lift_m':final_lift,'bilateral_contact_frames':bilateral_frames,'held_at_finish':len(final_contacts)==2,
      'final_xy_error_m':xy_error,'final_upright_cosine':upright,'final_speed_norm':speed,
      'unexpected_arm_cup_contact_frames':unexpected_contact_frames,'constraint_attachment_used':False,
      'real_transfer_verified':False,'success':bool(final_lift>.06 and bilateral_frames>10 and len(final_contacts)==2 and unexpected_contact_frames==0 and xy_error<.02 and upright>.95 and speed<.05)}
    output=ROOT/'experiments/nero'/output_name;output.mkdir(parents=True,exist_ok=True)
    (output/'result.json').write_text(json.dumps(result,indent=2));(output/'trace.json').write_text(json.dumps(trace))
    print(json.dumps({k:v for k,v in result.items() if k not in ('config','stages')}),flush=True)
    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--config',type=str,default=str(ROOT/'configs/nero/cup_pick_place.json'))
    parser.add_argument('--output-name',default='baseline')
    parser.add_argument('--planner',choices=['ik','moveit'],default='ik')
    args = parser.parse_args()
    run(json.loads(open(args.config).read()),args.viewer,args.output_name,args.planner)
