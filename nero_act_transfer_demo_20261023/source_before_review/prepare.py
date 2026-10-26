"""Export original ACT arm targets and densely validate a separate noncontact demo."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'nero_act_contact_escape_fix_20261023'
sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
from executor.ros_execution import checked_interpolation
from simulation import scene as scene_source
from pyAgxArm.utiles.mdh_kinematics import get_mdh,fk_from_mdh


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output=ROOT/'preparation';output.mkdir(exist_ok=False)
    preflight=json.loads((ROOT/'preflight_01/report.json').read_text())
    if preflight.get('error') or preflight['can_tx_before']!=preflight['can_tx_after']:
        raise ValueError('Clean unchanged-CAN current preflight required')
    actual=preflight['final_SDK_health'];state=np.asarray(actual['state'],float)
    if actual['feedback']['motion_status']!=0 or preflight['SDK_ROS_arm_max_difference_rad']>.0005:
        raise ValueError('SDK/ROS must agree at stationary start')
    summary=json.loads((BASE/'results/summary.json').read_text())
    for p,h in summary['source_sha256'].items():
        if sha(p)!=h:raise ValueError('Frozen ACT simulation source changed: '+p)
    parent=next(x for x in summary['results'] if x['id']==9000)
    if parent['full_task_passed'] is not True:raise ValueError('Expected the original normal ACT center recording')
    protocol=json.loads((BASE/'evaluation_protocol.json').read_text());scene=next(x['scene'] for x in protocol['cases'] if x['id']==9000)
    trace=BASE/'results/9000/rollout.npz';proof=ROOT.parent/'nero_act_observation_repair_20261023/frame_audit/manifest.json'
    capture=json.loads(proof.read_text())
    if sha(trace)!=capture['source_sha256'][str(trace)]:raise ValueError('Recorded normal ACT trace changed')
    with np.load(trace,allow_pickle=False) as saved:
        targets=saved['sent_targets'].copy();sources=saved['target_sources'].copy()
    mdh=get_mdh('nero');fk=lambda q:np.asarray(fk_from_mdh(mdh,np.asarray(q).tolist()))
    if (np.max(abs(targets[0,:7]-state[:7]))>.0005
            or np.linalg.norm(fk(state[:7])[:3]-actual['flange_pose'][:3])>.003):
        raise ValueError('Real high start differs from original ACT start or SDK FK; no rebase')
    indices=[0];poses=[fk(targets[0,:7])];previous=targets[0,:7]
    for i in range(1,len(targets)):
        if not str(sources[i]).startswith('ACT'):continue
        if np.max(abs(targets[i,:7]-previous))<2e-5:continue
        pose=fk(targets[i,:7])
        if pose[2]<.36:break
        if np.max(abs(targets[i,:7]-previous))>.002001:raise ValueError('Original ACT prefix exceeds original2mrad rate')
        indices.append(i);poses.append(pose);previous=targets[i,:7]
        if len(indices)==400:break
    rows=np.c_[targets[indices,:7],np.full(len(indices),state[7])];poses=np.asarray(poses)
    if len(rows)<250 or np.max(abs(rows[:,:7]-state[:7]))<.35:
        raise ValueError('No meaningful noncontact ACT prefix')
    xml=ET.parse(scene_source.ROOT/'scene_template.xml').getroot();assets=[]
    for mesh in xml.findall('asset/mesh'):
        path=(scene_source.ROOT/mesh.get('file')).resolve();mesh.set('file',str(path));assets.append(path)
    world=xml.find('worldbody');table=world.find("geom[@name='table']")
    table.set('pos',' '.join(map(str,[*scene['table_center_xy_m'],scene['table_top_z_m']+.05-.025])))
    table.set('size',' '.join(map(str,[*scene['table_half_size_xy_m'],.025])))
    world.find("body[@name='cup']").set('pos',' '.join(map(str,[*scene['pick_xy_m'],scene['table_top_z_m']+.0485])))
    ET.SubElement(world,'geom',name='demo_cup_envelope',type='cylinder',size='.125 .125',
        pos=' '.join(map(str,[*scene['pick_xy_m'],scene['table_top_z_m']+.125])),rgba='1 0 0 .2')
    collision=output/'collision_model.xml';collision.write_text(ET.tostring(xml,encoding='unicode'))
    model=mujoco.MjModel.from_xml_path(str(collision));data=mujoco.MjData(model)
    checks=checked_interpolation(state,rows)
    for q in np.vstack([state,checks]):
        if fk(q[:7])[2]<.35:raise ValueError('Swept SDK flange below new350mm noncontact scope')
        data.qpos[:7]=q[:7];data.qpos[7:9]=[q[7]/2,-q[7]/2];mujoco.mj_forward(model,data)
        for contact in data.contact[:data.ncon]:
            if contact.dist>=0:continue
            names={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) for g in [contact.geom1,contact.geom2]}
            if names in ({'world','base_link'},{'world','cup'}):continue
            raise ValueError('Inflated full-arm collision: '+','.join(sorted(names)))
    target_path=output/'arm_targets.npy';np.save(target_path,rows,allow_pickle=False)
    paths=[Path(__file__),ROOT/'entry.py',ROOT/'runtime_entry.py',ROOT/'preflight.py',ROOT/'preflight_01/report.json',Path(preflight['camera']['native_frame']),trace,BASE/'results/summary.json',BASE/'evaluation_protocol.json',proof,collision,target_path,Path(scene_source.__file__),scene_source.ROOT/'scene_template.xml',Path(sys.modules[fk_from_mdh.__module__].__file__),*assets]
    hashes=dict(summary['source_sha256']);hashes.update({str(p.resolve()):sha(p) for p in paths})
    plan=dict(scope='protected_ACT_simulation_joint_prefix_real_replay',initial_state=state.tolist(),
        targets_path=str(target_path),target_count=len(rows),source_trace_path=str(trace),source_indices=indices,
        nominal_duration_s=len(rows)*.05,sdk_flange_poses=poses.tolist(),sdk_flange_floor_m=.35,speed_percent=5,period_s=.05,
        joint_step_limit_rad=.002,latest_feedback_lead_limit_rad=.012,tracking_error_limit_rad=.012,
        gripper_commands=0,enable_commands=0,reset_commands=0,follower_transitions=0,
        endpoint_rule='stop_before_first_ACT_target_below360mm_flange_keep10mm_guard_margin',joint_excursion_deg=np.rad2deg(np.max(abs(rows[:,:7]-state[:7]),axis=0)).tolist(),
        endpoint_flange_delta_m=(poses[-1,:3]-np.asarray(actual['flange_pose'])[:3]).tolist(),
        table_inflation_m=.05,cup_collision_envelope_diameter_m=.25,cup_collision_envelope_height_m=.25,
        collision_model_path=str(collision),geometry_checks=len(checks)+1,collision_passed=True,
        ACT_model_qualification_passed=False,complete_task_not_released=True,independent_TCP_calibration_claimed=False,
        source_sha256=hashes,hardware_commands_sent=False,online_visual_policy=False,
        arm_targets_rebased=False,arm_targets_exact_source=True,real_gripper_held_m=float(state[7]),simulation_gripper_m=.095)
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    im=cv2.imread(preflight['camera']['native_frame']);camera=scene['camera_calibration'];t=np.asarray(camera['transform_base_from_raw_optical_m']);k=np.asarray(camera['intrinsic_matrix']);size=np.asarray(camera['resolution_wh'])
    optical=(poses[:,:3]-t[:3,3])@t[:3,:3];pixels=optical@k.T;pixels=(size-1-pixels[:,:2]/pixels[:,2,None]).round().astype(int)
    cv2.polylines(im,[pixels.reshape(-1,1,2)],False,(20,220,20),3)
    for label,point in [('START',pixels[0]),('STOP at flange350mm',pixels[-1])]:
        cv2.circle(im,tuple(point),7,(20,20,255),-1);cv2.putText(im,label,tuple(point+[10,-10]),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,255),2)
    cv2.putText(im,'ACT joint-prefix replay / gripper held / no cup contact',(25,680),cv2.FONT_HERSHEY_SIMPLEX,.6,(10,10,10),2)
    if not cv2.imwrite(str(output/'route.png'),im):raise RuntimeError('Cannot save route')
    print(json.dumps({key:plan[key] for key in ['target_count','nominal_duration_s','joint_excursion_deg','endpoint_flange_delta_m','geometry_checks','collision_passed']}),flush=True)


if __name__=='__main__':main()
