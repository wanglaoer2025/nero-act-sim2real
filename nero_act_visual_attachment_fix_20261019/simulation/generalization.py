"""Deterministic centimetre-scale scenes; simulator only, no hardware imports."""
import copy
import mujoco
import numpy as np
from simulation.scene import build_scene
from simulation.teacher import inverse_kinematics,observe


def rotation_xyz(angles):
    x,y,z=angles;cx,cy,cz=np.cos(angles);sx,sy,sz=np.sin(angles)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])


def pose(model,q,width):
    data=mujoco.MjData(model);data.qpos[:7]=q;data.qpos[7:9]=[width/2,-width/2]
    mujoco.mj_forward(model,data)
    site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_tcp')
    return data,data.site_xpos[site].copy(),data.site_xmat[site].reshape(3,3).copy()


def episode_config(center,index,split,attempt=0):
    if (not isinstance(index,int) or not 0<=index<48 or split!=('train' if index<36 else 'val')
            or not isinstance(attempt,int) or not 0<=attempt<3):
        raise ValueError('Unique48-scene profile with explicit36/12 splits and at most3 candidates required')
    scene=copy.deepcopy(center);seed=180000+index*10+attempt
    rng=np.random.default_rng(seed)
    if index<36:
        grid=np.linspace(-.03,.03,6);offset=[grid[index%6],grid[index//6]]
        if index==14:offset=[0.,0.]
    else:
        j=index-36;offset=[[-.027,-.009,.009,.027][j%4],[-.024,0.,.024][j//4]]
    scene['pick_xy_m']=(np.array(center['pick_xy_m'])+offset).tolist()
    translation=rng.uniform(-.02,.02,3);angles=rng.uniform(-np.deg2rad(3),np.deg2rad(3),3)
    if index==0:translation[:]=0;angles[:]=0
    model=build_scene(scene);start=center['initial_robot_state']
    _,xyz,rotation=pose(model,start['qpos_rad'],start['gripper_total_width_m'])
    desired=rotation_xyz(angles)@rotation
    q=inverse_kinematics(model,xyz+translation,np.array(start['qpos_rad']),desired)
    width=float(rng.uniform(.09,.095)) if index else float(start['gripper_total_width_m'])
    scene['initial_robot_state']=dict(qpos_rad=q.tolist(),gripper_total_width_m=width)
    scene.update(action_delay_steps=index%3,joint_speed_limit_rad_s=.03,
        teacher_command_lead_limit_rad=.016,gripper_speed_limit_m_s=.008,
        teacher_gripper_feedback_lead_m=.008)
    scene['dwell_seconds'].update(close=6,release=6)
    scene['appearance'].update(light_scale=float(rng.uniform(.82,1.06)),
        cup_rgb=[float(rng.uniform(.88,.99)),float(rng.uniform(.70,.88)),float(rng.uniform(.01,.09))],
        table_rgb=[float(rng.uniform(.82,.96))]*3,background_rgb=[float(rng.uniform(.20,.35))]*3)
    scene['generalization']=dict(index=index,split=split,source_group=f'scene_{index}',seed=seed,
        attempt=attempt,source_offset_m=list(map(float,offset)),initial_translation_m=translation.tolist(),
        initial_rotation_rad=angles.tolist(),training_envelope_only=True)
    validate_initial_pose(build_scene(scene),scene,center)
    return scene


def validate_initial_pose(model,scene,center):
    initial=scene['initial_robot_state'];base=center['initial_robot_state'];g=scene['generalization']
    q=np.array(initial['qpos_rad']);origin=np.array(base['qpos_rad'])
    if (q.shape!=(7,) or not np.isfinite(q).all() or max(abs(q-origin))>.20
            or np.any(q<=model.jnt_range[:7,0]) or np.any(q>=model.jnt_range[:7,1])
            or not .09<=initial['gripper_total_width_m']<=.095):
        raise ValueError('Initial IK branch/joints/opening outside the declared high-pose profile')
    _,xyz0,r0=pose(model,origin,base['gripper_total_width_m'])
    data,xyz,r=pose(model,q,initial['gripper_total_width_m'])
    error=float(np.linalg.norm(xyz-xyz0-np.array(g['initial_translation_m'])))
    rotation_error=float(np.max(abs(r-rotation_xyz(g['initial_rotation_rad'])@r0)))
    bad=observe(model,data,scene)['unexpected_contact']
    if error>.0002 or rotation_error>.002 or bad or xyz[2]<.30:
        raise ValueError('Initial physical pose fails translation/rotation/contact/high-clearance check')
    return dict(position_error_m=error,rotation_error=rotation_error,unexpected_contact=bool(bad),q=q.tolist())
