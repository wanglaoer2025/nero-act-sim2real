"""Offline complete-route check; tentative camera/mount hypotheses stay unverified."""
from pathlib import Path
import sys,json,copy,hashlib,time,xml.etree.ElementTree as ET
import numpy as np,cv2,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),'/home/user/zyh/src']
from capture import dump,flange_transform
from nero_pi05.semantics import JOINT_LIMITS_RAD
from pyAgxArm.utiles.mdh_kinematics import get_mdh,fk_from_mdh

result=json.loads((ROOT/'auto_plan_probe_01/result.json').read_text())
q0=np.array(result['initial_state'])[:7];gripper=result['initial_state'][7]
mdh=get_mdh('nero');fk=lambda q:flange_transform(fk_from_mdh(mdh,np.asarray(q).tolist()))
BC=np.array(result['provisional_camera_base']);mounts=[np.array(v) for v in result['provisional_mount_hypotheses']]
K=np.array(json.loads((ROOT/'timing_probe_01/report.json').read_text())['K'])
h=.095/2;obj=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]])
models=[]
for index,mount in enumerate(mounts):
 root=ET.parse(ROOT/'auto_plan_probe_01/model.xml').getroot()
 body=ET.SubElement(root.find('worldbody'),'body',name='sampling_board')
 ET.SubElement(body,'freejoint',name='sampling_board_free')
 ET.SubElement(body,'geom',name='sampling_board_box',type='box',size='.065 .065 .013',mass='.02')
 # Operator confirmed object is held between fingers, not bolted to the flange.
 # The existing gripper geometry already represents the proximal hardware.
 # This provisional connector starts at the retained grasp-center reference.
 tcp=np.r_[np.asarray([.152,0.,-.0235]),1.]
 endpoint=(np.linalg.inv(mount)@tcp)[:3]
 ET.SubElement(body,'geom',name='sampling_bracket_envelope',type='capsule',fromto=' '.join(map(str,[0,0,0,*endpoint])),size='.025',mass='.02')
 contact=root.find('contact')
 if contact is None:contact=ET.SubElement(root,'contact')
 for link in ['link7','flange_link','gripper_base','gripper_link1','gripper_link2']:
  ET.SubElement(contact,'exclude',body1='sampling_board',body2=link)
 model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
 joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'sampling_board_free')
 models.append((model,mujoco.MjData(model),int(model.jnt_qposadr[joint])))

def check(q,margin=4):
 BF=fk(q)
 if not (-.36<=BF[0,3]<=-.14 and -.20<=BF[1,3]<=.02 and .30<=BF[2,3]<=.43):return 'flange_workspace'
 for mount,(model,data,address) in zip(mounts,models):
  BM=BF@mount;CM=np.linalg.inv(BC)@BM;pts=obj@CM[:3,:3].T+CM[:3,3]
  px=pts@K.T;px=px[:,:2]/px[:,2,None]
  if np.any(pts[:,2]<=.15) or np.any(px[:,0]<margin) or np.any(px[:,0]>639-margin) or np.any(px[:,1]<margin) or np.any(px[:,1]>359-margin):return 'predicted_fov'
  data.qpos[:7]=q;data.qpos[7:9]=[gripper/2,-gripper/2]
  data.qpos[address:address+3]=BM[:3,3]
  data.qpos[address+3:address+7]=Rotation.from_matrix(BM[:3,:3]).as_quat(scalar_first=True)
  mujoco.mj_forward(model,data)
  for c in data.contact[:data.ncon]:
   if c.dist>=0:continue
   names={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) for g in [c.geom1,c.geom2]}
   if names in ({'world','base_link'},{'world','cup'}):continue
   return 'contact:'+','.join(sorted(names))
 return None

def edge(a,b):
 n=max(1,int(np.ceil(np.max(abs(b-a))/.0005)))
 for q in np.linspace(a,b,n+1):
  bad=check(q)
  if bad:return bad,n
 return None,n

pool=result['pool'];endpoint_bad={}
for i,c in enumerate(pool):
 reason=check(np.array(c['q']),margin=20)
 if reason:endpoint_bad[i]=reason
valid=[i for i in range(len(pool)) if i not in endpoint_bad]
positions=np.array([np.array(c['goal_transform'])[:3,3] for c in pool])
if len(valid)<20:
 from collections import Counter
 report=dict(nominal_route_passed=False,valid_endpoints=len(valid),endpoint_rejections=endpoint_bad,
     rejection_counts=dict(Counter(endpoint_bad.values())),actual_motion_sent=False,geometry_and_camera_verified=False)
 dump(ROOT/'automatic_sampling_01/nominal_plan.json',report)
 print(json.dumps(report['rejection_counts']))
 raise SystemExit(2)
fit_ids=[]
for axis in [0,1]:
 for i in [min(valid,key=lambda i:positions[i,axis]),max(valid,key=lambda i:positions[i,axis])]:
  if i not in fit_ids:fit_ids.append(i)
def distance(i,j):
 a=np.array(pool[i]['goal_transform']);b=np.array(pool[j]['goal_transform'])
 return np.linalg.norm(a[:3,3]-b[:3,3])+.1*Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude()
while len(fit_ids)<15:
 remaining=[i for i in valid if i not in fit_ids]
 if not remaining:break
 fit_ids.append(max(remaining,key=lambda i:min(distance(i,j) for j in fit_ids)))
check_ids=[i for i in valid if i not in fit_ids][:5]
T0=fk(q0);Tprep=T0.copy();Tprep[2,3]=.35
def residual(q):
 T=fk(q)
 return np.r_[T[:3,3]-Tprep[:3,3],.25*Rotation.from_matrix(Tprep[:3,:3].T@T[:3,:3]).as_rotvec(),1e-5*(q-q0)]
fit=least_squares(residual,q0,bounds=(JOINT_LIMITS_RAD[:,0]+.01,JOINT_LIMITS_RAD[:,1]-.01),max_nfev=200,ftol=1e-12,xtol=1e-12,gtol=1e-12)
qprep=fit.x
assert np.linalg.norm(fk(qprep)[:3,3]-Tprep[:3,3])<.0001
initial_bad,initial_checks=edge(q0,qprep)
stages=[];errors=[];total_checks=initial_checks;previous=q0
if initial_bad:errors.append(dict(stage='prepare',reason=initial_bad))
else:
 stages.append(dict(role='prepare',candidate_id=None,goal_q_rad=qprep.tolist()))
 previous=qprep
 for role,ids in [('fit',fit_ids),('pose_check',check_ids)]:
  remaining=set(ids)
  while remaining:
   attempts=[];chosen=None
   for i in sorted(remaining,key=lambda i:np.max(abs(np.array(pool[i]['q'])-previous))):
    bad,n=edge(previous,np.array(pool[i]['q']));total_checks+=n
    attempts.append(dict(candidate=i,reason=bad))
    if bad is None:chosen=i;break
   if chosen is None:
    errors.append(dict(stage=role,remaining=sorted(remaining),attempts=attempts));break
   c=pool[chosen];stages.append(dict(role=role,candidate_id=chosen,goal_q_rad=c['q'],goal_transform=c['goal_transform'],view_hypotheses=c['view_hypotheses']))
   previous=np.array(c['q']);remaining.remove(chosen)
  if remaining:break

previous=q0.copy();all_poses=[];total_rows=0
for index,stage in enumerate(stages):
 target=np.array(stage['goal_q_rad']);count=max(1,int(np.ceil(np.max(abs(target-previous))/.006)))
 stage['waypoints_rad']=np.linspace(previous,target,count+1)[1:].tolist()
 stage['index']=index;total_rows+=count
 all_poses.extend(fk(q) for q in stage['waypoints_rad']);previous=target
fit_span=np.ptp(positions[fit_ids],axis=0) if fit_ids else np.zeros(3)
output=dict(scope='scoped noncontact hand-eye sampling; NOT ACT or verified TCP',
 initial_state=result['initial_state'],stages=stages,checks=total_checks,waypoint_count=total_rows,
 speed_percent=1,max_step_rad=.006,max_latest_feedback_lead_rad=.012,fit_samples_required=15,pose_check_samples_required=5,
 flange_workspace_m=[[-.36,-.14],[-.20,.02],[.30,.43]],
 fit_position_span_m=fit_span.tolist(),coverage_target_met=bool(np.count_nonzero(fit_span>=.0999)>=2),
 endpoint_rejections=endpoint_bad,route_errors=errors,nominal_route_passed=not errors and len(stages)==21,
 board_box_full_size_m=[.13,.13,.026],bracket_capsule_radius_m=.025,
 mounting_source='Operator: held between fingers; connector starts at nominal grasp reference, not flange',
 grasp_reference_measured=False,
 fixed_mount_collision_exclusions=['link7','flange_link','gripper_base','gripper_link1','gripper_link2'],
 geometry_model_uses_both_planar_pose_hypotheses=True,geometry_and_camera_verified=False,
 predicted_camera_fov_only=True,requires_onsite_whole_sweep_confirmation=True,
 calibration_verified=False,independent_absolute_validation=False,actual_motion_sent=False,hardware_deployment_allowed=False,
 model_source=str(ROOT/'auto_plan_probe_01/model.xml'),provisional_camera_base=result['provisional_camera_base'],
 provisional_mount_hypotheses=result['provisional_mount_hypotheses'],
 source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
  ROOT/'auto_plan_probe_01/result.json',ROOT/'auto_plan_probe_01/model.xml',Path(__file__),
  ROOT/'auto_plan_probe_01/probe.py',ROOT/'timing_probe_01/report.json',
  ROOT/'marker_measured_95mm/marker_profile.json',
  Path('/home/user/wq/nero_act_visual_feedback_20261019/scene/config.json'),
  Path('/home/user/zyh/src/nero_pi05/semantics.py'),
  Path(sys.modules[fk_from_mdh.__module__].__file__),
  *[Path(mesh.attrib['file']) for mesh in ET.parse(ROOT/'auto_plan_probe_01/model.xml').findall('.//asset/mesh')]]})
if all_poses:
 xyz=np.array(all_poses)[:,:3,3];output['flange_swept_min_m']=xyz.min(axis=0).tolist();output['flange_swept_max_m']=xyz.max(axis=0).tolist()
dump(ROOT/'automatic_sampling_01/nominal_plan.json',output)
print(json.dumps({k:output[k] for k in ['nominal_route_passed','checks','waypoint_count','fit_position_span_m','flange_swept_min_m','flange_swept_max_m','route_errors']}))
print(json.dumps({'valid_endpoints':len(valid),'endpoint_rejections':endpoint_bad}))
