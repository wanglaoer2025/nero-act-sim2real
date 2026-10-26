"""Offline feasibility probe only. No device connections or executable permission."""
from pathlib import Path
import sys,json,itertools,hashlib,time
import numpy as np
import cv2,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),'/home/user/zyh/src','/home/user/wq/nero_act_visual_feedback_20261019']
from capture import dump,flange_transform
from nero_pi05.semantics import JOINT_LIMITS_RAD
from pyAgxArm.utiles.mdh_kinematics import get_mdh,fk_from_mdh
from simulation.scene import build_scene

mdh=get_mdh('nero');fk=lambda q:flange_transform(fk_from_mdh(mdh,np.asarray(q).tolist()))
health=json.loads((ROOT/'framing_complete/verification.json').read_text())['final_health']
q0=np.array(health['state'])[:7];T0=fk(q0)
config_path=Path('/home/user/wq/nero_act_visual_feedback_20261019/scene/config.json')
config=json.loads(config_path.read_text());BC=np.array(config['camera_calibration']['transform_base_from_raw_optical_m'])
probe=json.loads((ROOT/'timing_probe_01/report.json').read_text())
diagnostic=json.loads((ROOT/'timing_probe_01/diagnosis.json').read_text())
BF_observed=flange_transform(probe['frames'][-1]['health_after']['flange_pose'])
mounts=[np.linalg.inv(BF_observed)@BC@np.array(c['transform']) for c in diagnostic['candidates']]
K=np.array(probe['K']);h=.095/2
obj=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]])
model=build_scene(config);data=mujoco.MjData(model)
lower=np.maximum(JOINT_LIMITS_RAD[:,0]+.01,q0-.8)
upper=np.minimum(JOINT_LIMITS_RAD[:,1]-.01,q0+.8)

def image_quality(T):
    values=[]
    for mount in mounts:
        CM=np.linalg.inv(BC)@T@mount
        pts=obj@CM[:3,:3].T+CM[:3,3]
        if np.any(pts[:,2]<=.15):return None
        px=(pts@K.T);px=px[:,:2]/px[:,2,None]
        if not (np.all(px[:,0]>20)&np.all(px[:,0]<619)&np.all(px[:,1]>20)&np.all(px[:,1]<339)):return None
        f=cv2.solvePnPGeneric(obj,px,K,None,flags=cv2.SOLVEPNP_IPPE_SQUARE)
        errors=np.sort(np.asarray(f[3]).ravel())*np.sqrt(2)
        if len(errors)<2 or errors[1]<1.20:return None
        values.append(dict(native_corners=px.tolist(),alternative_rmse_px=float(errors[1])))
    return values

def contacts(q):
    data.qpos[:7]=q;data.qpos[7:9]=[health['state'][7]/2,-health['state'][7]/2]
    mujoco.mj_forward(model,data);bad=[]
    for c in data.contact[:data.ncon]:
        names={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) for g in [c.geom1,c.geom2]}
        if names in ({'world','base_link'},{'world','cup'}):continue
        if c.dist<0:bad.append(sorted(names))
    return bad

rng=np.random.default_rng(221019)
grid=list(itertools.product([-.30,-.25,-.20],[-.1414,-.0914,-.0414],[.33,.35,.37],[-28,-20,20,28],[-28,-20,20,28],[-20,0,20]))
rng.shuffle(grid);pool=[];stats=dict(proposed=0,fov_or_ambiguity_rejected=0,ik_rejected=0,collision_rejected=0)
started=time.monotonic()
for x,y,z,rx,ry,rz in grid:
    stats['proposed']+=1
    T=np.eye(4);T[:3,3]=[x,y,z];T[:3,:3]=T0[:3,:3]@Rotation.from_euler('xyz',[rx,ry,rz],degrees=True).as_matrix()
    views=image_quality(T)
    if views is None:stats['fov_or_ambiguity_rejected']+=1;continue
    def residual(q):
        A=fk(q)
        return np.r_[A[:3,3]-T[:3,3],.25*Rotation.from_matrix(T[:3,:3].T@A[:3,:3]).as_rotvec(),1e-5*(q-q0)]
    fit=least_squares(residual,q0,bounds=(lower,upper),xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=100)
    actual=fk(fit.x)
    if np.linalg.norm(actual[:3,3]-T[:3,3])>.0001 or Rotation.from_matrix(actual[:3,:3].T@T[:3,:3]).magnitude()>.001:
        stats['ik_rejected']+=1;continue
    bad=contacts(fit.x)
    if bad:stats['collision_rejected']+=1;continue
    pool.append(dict(q=fit.x.tolist(),goal_transform=T.tolist(),relative_euler_deg=[rx,ry,rz],view_hypotheses=views))
    if len(pool)>=70:break

report=dict(scope='offline feasibility; not an executable or qualified calibration',initial_state=health['state'],
    provisional_camera_base=BC.tolist(),provisional_mount_hypotheses=[v.tolist() for v in mounts],
    camera_and_mount_geometry_verified=False,source_ambiguity_preserved=True,pool=pool,statistics=stats,
    elapsed_s=time.monotonic()-started,actual_motion_sent=False,hardware_deployment_allowed=False)
if len(pool)>=20:
    # Keep both100mm position extrema in fit; select the remainder with separated rotations.
    fit_ids=[]
    positions=np.array([np.array(c['goal_transform'])[:3,3] for c in pool])
    for axis in [0,1]:
        for idx in [int(np.argmin(positions[:,axis])),int(np.argmax(positions[:,axis]))]:
            if idx not in fit_ids:fit_ids.append(idx)
    while len(fit_ids)<15:
        remaining=[i for i in range(len(pool)) if i not in fit_ids]
        def distance(i,j):
            a=np.array(pool[i]['goal_transform']);b=np.array(pool[j]['goal_transform'])
            return np.linalg.norm(a[:3,3]-b[:3,3])+ .1*Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude()
        fit_ids.append(max(remaining,key=lambda i:min(distance(i,j) for j in fit_ids)))
    checks=[i for i in range(len(pool)) if i not in fit_ids][:5]
    report['fit_candidate_ids']=fit_ids;report['pose_check_candidate_ids']=checks
    report['fit_position_span_m']=np.ptp(positions[fit_ids],axis=0).tolist()
    report['coverage_target_met']=bool(np.count_nonzero(np.ptp(positions[fit_ids],axis=0)>=.0999)>=2)
dump(ROOT/'auto_plan_probe_01/result.json',report)
print(json.dumps({k:report[k] for k in ['statistics','elapsed_s','actual_motion_sent']}))
print(json.dumps({'pool':len(pool),'fit_position_span_m':report.get('fit_position_span_m'),'coverage_target_met':report.get('coverage_target_met',False)}))
