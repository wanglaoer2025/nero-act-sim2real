"""Validate an exported ACT prefix against separate inflated collision obstacles."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'nero_act_paired_response_20261023'))
from experiment import snapshot
from executor.ros_execution import checked_interpolation
from simulation import scene as scene_source
from pyAgxArm.utiles.mdh_kinematics import get_mdh,fk_from_mdh


def main():
    output=ROOT/'preparation_01';preflight=json.loads((ROOT/'preflight_01/report.json').read_text())
    state=np.array(preflight['final_SDK_health']['state'])
    scene=json.loads((output/'ACT_source/scene_configuration.json').read_text())
    if not np.array_equal(state[:7],scene['initial_robot_state']['qpos_rad']):raise ValueError('ACT source start differs')
    result=json.loads((output/'ACT_source/result.json').read_text())
    data=np.load(output/'ACT_source/rollout.npz',allow_pickle=False)
    mdh=get_mdh('nero');fk=lambda q:np.asarray(fk_from_mdh(mdh,np.asarray(q).tolist()))
    if np.linalg.norm(fk(state[:7])[:3]-preflight['final_SDK_health']['flange_pose'][:3])>.003:
        raise ValueError('SDK/model initial flange mismatch')
    rows=[];poses=[];stop='400_point_budget';previous=state.copy()
    for target,source in zip(data['sent_targets'],data['target_sources']):
        if not str(source).startswith('ACT'):continue
        if np.max(abs(target[:7]-previous[:7]))<2e-5:continue
        pose=fk(target[:7])
        if pose[2]<.45:stop='flange_height_floor450mm';break
        if abs(target[7]-state[7])>.001:stop='approach_gripper_prediction_changed';break
        if np.max(abs(target[:7]-previous[:7]))>.002001:stop='ACT_rate_discontinuity';break
        rows.append(np.r_[target[:7],state[7]]);poses.append(pose);previous=rows[-1]
        if len(rows)==400:break
    rows=np.array(rows);poses=np.array(poses)
    if rows.ndim!=2 or len(rows)<20 or np.max(abs(rows[:,:7]-state[:7]))<.01:
        raise ValueError('No meaningful bounded high ACT prefix; no token motion')
    xml=ET.parse(scene_source.ROOT/'scene_template.xml').getroot();assets=[]
    for mesh in xml.findall('asset/mesh'):
        path=(scene_source.ROOT/mesh.get('file')).resolve();mesh.set('file',str(path));assets.append(path)
    world=xml.find('worldbody');table=world.find("geom[@name='table']")
    table.set('pos',' '.join(map(str,[*scene['table_center_xy_m'],scene['table_top_z_m']+.05-.025])))
    table.set('size',' '.join(map(str,[*scene['table_half_size_xy_m'],.025])))
    world.find("body[@name='cup']").set('pos',' '.join(map(str,[*scene['pick_xy_m'],scene['table_top_z_m']+.0485])))
    # Keep original cup dimensions; this additional static obstacle is a conservative safety envelope.
    ET.SubElement(world,'geom',name='trial_cup_envelope',type='cylinder',size='.125 .125',
        pos=' '.join(map(str,[*scene['pick_xy_m'],scene['table_top_z_m']+.125])),rgba='1 0 0 .2')
    collision_path=output/'collision_model.xml';collision_path.write_text(ET.tostring(xml,encoding='unicode'))
    model=mujoco.MjModel.from_xml_path(str(collision_path));physical=mujoco.MjData(model)
    checks=checked_interpolation(state,rows)
    for q in np.vstack([state,checks]):
        if fk(q[:7])[2]<.45:raise ValueError('Interpolated flange below450mm')
        physical.qpos[:7]=q[:7];physical.qpos[7:9]=[q[7]/2,-q[7]/2];mujoco.mj_forward(model,physical)
        for contact in physical.contact[:physical.ncon]:
            if contact.dist>=0:continue
            names={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) for g in [contact.geom1,contact.geom2]}
            if names in ({'world','base_link'},{'world','cup'}):continue
            raise ValueError('Inflated full-arm collision: '+','.join(sorted(names)))
    np.save(output/'arm_targets.npy',rows,allow_pickle=False)
    checkpoint=ROOT.parent/'nero_act_paired_response_20261023/training/pilot_A/policy_last.ckpt'
    files=[Path(__file__),ROOT/'prepare_air_prefix.py',ROOT/'preflight.py',ROOT/'preflight_01/report.json',
        Path(preflight['camera']['frame']),output/'scene_observation.json',checkpoint,checkpoint.parent/'integrity.json',
        output/'ACT_source/rollout.npz',output/'ACT_source/scene_configuration.json',output/'arm_targets.npy',collision_path,
        Path(scene_source.__file__),scene_source.ROOT/'scene_template.xml',Path(sys.modules[fk_from_mdh.__module__].__file__),*assets]
    hashes=snapshot(files)
    plan=dict(scope='ACT-generated high noncontact arm-only trial; not full pick/place or qualified online visual ACT',
        initial_state=state.tolist(),targets_path=str(output/'arm_targets.npy'),target_count=len(rows),nominal_duration_s=len(rows)*.05,
        sdk_flange_poses=poses.tolist(),sdk_flange_floor_m=.45,speed_percent=5,period_s=.05,joint_step_limit_rad=.002,
        latest_feedback_lead_limit_rad=.012,tracking_error_limit_rad=.012,gripper_commands=0,enable_commands=0,
        reset_commands=0,follower_transitions=0,endpoint_rule=stop,
        joint_excursion_deg=np.rad2deg(np.max(abs(rows[:,:7]-state[:7]),axis=0)).tolist(),
        table_inflation_m=.05,cup_collision_envelope_diameter_m=.25,cup_collision_envelope_height_m=.25,
        collision_model_path=str(collision_path),geometry_checks=len(checks)+1,collision_passed=True,
        ACT_model_qualification_passed=False,complete_task_not_released=True,independent_TCP_calibration_claimed=False,
        source_sha256=hashes,hardware_commands_sent=False,offline_result_error=result.get('error'))
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({k:plan[k] for k in ['target_count','nominal_duration_s','joint_excursion_deg','geometry_checks','collision_passed','endpoint_rule']}))


if __name__=='__main__':main()
