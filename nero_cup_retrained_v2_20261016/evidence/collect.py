import os,sys,json,copy
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco,h5py
sys.path.insert(0,'/home/wq/robot_learning/src');sys.path.insert(0,'/home/wq/sim2real/nero_act_source-offline-ready')
from teacher import run
from nero_scene_camera import observation_camera
from src.act.episode_io import save_episode
root=Path(__file__).resolve().parent;out=root/'data';out.mkdir(exist_ok=True)
base=json.loads((root/'scene.json').read_text());entries=json.loads((out/'manifest.json').read_text())['episodes'] if (out/'manifest.json').exists() else [];failures=[]
for i in range(8):
 if (out/f'episode_{i}.hdf5').exists():continue
 rng=np.random.default_rng(51000+i);c=copy.deepcopy(base)
 if i:
  c['pick_xy_m']=(np.array(c['pick_xy_m'])+rng.uniform(-.003,.003,2)).tolist();c['place_xy_m']=c['pick_xy_m'].copy();c['initial_robot_state']['qpos_rad']=(np.array(c['initial_robot_state']['qpos_rad'])+rng.uniform(-.002,.002,7)).tolist()
 if i>=2:c['arm_disturbance']={'seed':52000+i,'max_torque_nm':.4,'period_steps':45,'duration_steps':5,'start_step':90}
 c['appearance']['light_scale']=.92+.025*i;records=[];renderer=None;rd=None;image=None
 def capture(m,d,target,width,stage):
  global renderer,rd,image
  if renderer is None:renderer=mujoco.Renderer(m,height=90,width=160);rd=mujoco.MjData(m)
  if len(records)%10==0:
   rd.qpos[:]=d.qpos;rd.qvel[:]=d.qvel;mujoco.mj_forward(m,rd);renderer.update_scene(rd,camera=observation_camera(c));renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False;image=renderer.render().copy()
  cup=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'cup');records.append((d.qpos.copy(),d.qvel.copy(),np.r_[target,width],float(d.time),stage,image.copy(),d.xpos[cup].copy(),d.qfrc_applied[:7].copy()))
 try:
  result=run(c,output_name=f'cup_retrain_v2_collection/episode_{i}',observation_callback=capture)
  if not result['success']:raise RuntimeError('Teacher failed '+json.dumps(result))
  q=np.array([r[0] for r in records]);a=np.array([r[2] for r in records],np.float32);path=out/f'episode_{i}.hdf5'
  assert np.max(abs(np.diff(a[:,7])))<=.000501
  assert np.max(abs(np.diff(a[:,:7],axis=0)))<=.002001
  save_episode(path,np.array([r[5] for r in records]),q[:,:7],a,np.array([r[3] for r in records]),gripper_states=(q[:,7]-q[:,8])[:,None],metadata={'sim':True,'scene_config':c,'control_hz':20,'task':'pickup_hold_short_wrist_branch','camera_refresh_hz':2,'camera_zero_order_hold_between_queries':True,'physical_disturbance_corrections':i>=2,'hardware_deployment_allowed':False})
  with h5py.File(path,'a') as f:
   sim=f.create_group('simulation');sim['qpos']=q;sim['qvel']=np.array([r[1] for r in records]);sim['stage']=np.array([r[4].encode() for r in records]);sim['external_arm_torque_nm']=np.array([r[7] for r in records]);f['observations']['object_xyz']=np.array([r[6] for r in records])
  entries.append({'path':str(path),'status':'valid','frames':len(a),'teacher_success':True,'seed':51000+i,'final_lift_m':result['final_lift_m']});print('COLLECTED',i,len(a),flush=True)
 except Exception as e:failures.append({'episode':i,'error':repr(e)});print('FAILED',repr(e),flush=True)
 finally:
  if renderer is not None:renderer.close()
 (out/'manifest.json').write_text(json.dumps({'episodes':entries,'rejected':failures},indent=2))
 if failures:raise RuntimeError('collection failed; retaining evidence')
(out/'split.json').write_text(json.dumps({'train':list(range(6)),'val':[6,7]},indent=2))
