import os,sys,json,hashlib,pickle,time,argparse
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco,torch,h5py
from PIL import Image
sys.path.insert(0,'/home/wq/robot_learning/src');sys.path.insert(0,'/home/wq/sim2real/nero_act_source-offline-ready')
from teacher import scene
from nero_scene_camera import observation_camera
from nero_domain_randomization import NeroServo,disturbance_torque
from src.act.nero_policy import NeroActPolicy
from src.act.nero_dataset import preprocess_images
p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,default=p/'checkpoints/policy_best_val.ckpt');parser.add_argument('--output',type=Path,default=p/'evaluation');parser.add_argument('--dev',action='store_true');args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1);checkpoint=args.checkpoint;bundle=torch.load(checkpoint,map_location='cpu',weights_only=False);config=dict(bundle['config'],device='cpu',pretrained_backbone=False)
assert config['chunk_size']==20 and config['loss_prefix_steps']==20 and config['state_dim']==8
policy=NeroActPolicy(config);policy.load_state_dict(bundle['model'],strict=True);policy.eval()
with (checkpoint.parent/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
real=np.array(Image.open('/home/wq/sim2real/nero_act_cup_handoff_20261016/scene/current_capture/camera_0_upright_rgb.png').convert('RGB'))
scenarios=[('recorded_start',0,10,0,False),('heldout6',6,10,0,False),('heldout7',7,10,0,False),('real_appearance',0,10,0,True),('observed_cadence',6,11,1,False),('half_second_delay',0,11,10,False)]+[(f'independent_{seed}',6+seed%2,11,seed%3,False) for seed in range(61000,61008)]+[(f'force_{seed}',0,11,1,False) for seed in range(62000,62002)]
if args.dev:scenarios=scenarios[:1]
protocol={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'stats_sha256':hashlib.sha256((checkpoint.parent/'dataset_stats.pkl').read_bytes()).hexdigest(),'epoch':bundle['epoch'],'teacher_free':True,'runtime_geometry_closure_guard':False,'control_hz':20,'chunk_size':20,'initial_qpos_or_object_reset_during_rollout':False,'thresholds':{'lift_m':.06,'bilateral_frames':10,'final_xy_error_m':.02,'upright_cosine':.95,'final_speed':.05},'scenarios':scenarios,'independent_cup_xy_jitter_m':.002,'independent_initial_joint_jitter_rad':.002,'feedback_noise_rad':.0002,'real_hardware_tested':False,'timing_protocol':'separate reach deadline and full100-point post-completion hold; original truncated-window results retained','validation_timing_fix_after_primary_trials':True,'completion_monitor':{'lift_m':.06,'stable_grasp_seconds':1,'post_completion_frozen_goal_seconds':5,'goal_observation':'simulator cup pose/contact for evaluation termination only; not policy inputs; real monitoring not validated'}}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[]
for name,episode,stride,delay,use_real in scenarios:
 with h5py.File(p/f'data/episode_{episode}.hdf5','r') as f:c=json.loads(f.attrs['scene_config']);initial=f['simulation/qpos'][0];nv=f['simulation/qvel'][0];horizon=len(f['action'])+200
 rng=np.random.default_rng(int(name.split('_')[-1]) if name.startswith(('independent_','force_')) else 60000+episode)
 if name.startswith('independent_'):c['pick_xy_m']=(np.array(c['pick_xy_m'])+rng.uniform(-.002,.002,2)).tolist();c['place_xy_m']=c['pick_xy_m'].copy()
 c['action_delay_steps']=delay;m=scene(c);d=mujoco.MjData(m);d.qpos[:]=initial;d.qvel[:]=nv
 if name.startswith('independent_'):
  free=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'cup_free');qadr=m.jnt_qposadr[free];d.qpos[qadr:qadr+2]=c['pick_xy_m'];d.qpos[:7]+=rng.uniform(-.002,.002,7)
 if name.startswith('force_'):c['arm_disturbance']={'seed':int(name.split('_')[-1]),'max_torque_nm':.4,'period_steps':47,'duration_steps':4,'start_step':160}
 mujoco.mj_forward(m,d);servo=NeroServo(m,d,c);rd=mujoco.MjData(m);rd.qpos[:]=d.qpos;rd.qvel[:]=d.qvel;mujoco.mj_forward(m,rd)
 cup=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'cup');tcp=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_tcp');fingers={mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in ['gripper_link1','gripper_link2']};adr=m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'cup_free')]
 initial_z=float(rd.xpos[cup,2]);maxlift=0.;bilateral=0;unexpected=0;trace=[];raws=[];queries=[];query_steps=[];last=np.r_[d.qpos[:7],d.qpos[7]-d.qpos[8]];chunk=None;block_violations=0;first_gap=None;failure=None;contacts=set();completion_step=None;stable_frames=0;hold_modes=[];cup_upright_history=[];start=time.monotonic()
 try:
  with mujoco.Renderer(m,height=90,width=160) as renderer:
   for step in range(horizon+100):
    if completion_step is None and step>=horizon:break
    state=np.r_[d.qpos[:7],d.qpos[7]-d.qpos[8]]
    if completion_step is None and step%stride==0:
     rd.qpos[:]=d.qpos;rd.qvel[:]=d.qvel;mujoco.mj_forward(m,rd);renderer.update_scene(rd,camera=observation_camera(c));renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
     frame=real if use_real else renderer.render().copy();observed=state.astype(np.float32);observed[:7]+=rng.uniform(-.0002,.0002,7) if stride==11 else 0
     with torch.inference_mode():chunk=policy(torch.from_numpy((observed-stats['qpos_mean'])/stats['qpos_std'])[None],preprocess_images(frame[None],config['image_size'])[None])[0].numpy()*stats['action_std']+stats['action_mean']
     assert chunk.shape==(20,8) and np.isfinite(chunk).all()
     queries.append(chunk.copy());query_steps.append(step)
     if step==0:first_gap=float(np.max(abs(chunk[0,:7]-state[:7])))
     block_violations+=int(np.max(abs(chunk[:10,:7]-state[:7]))>.03 or np.max(abs(chunk[:10,7]-state[7]))>.01)
    index=step%stride;raw=chunk[index].copy() if completion_step is None and index<10 else last.copy()
    target=raw.copy();increment=c['joint_speed_limit_rad_s']/20
    target[:7]=np.clip(target[:7],last[:7]-increment,last[:7]+increment);target[:7]=np.clip(target[:7],state[:7]-.03,state[:7]+.03);target[:7]=np.clip(target[:7],m.jnt_range[:7,0],m.jnt_range[:7,1]);target[7]=np.clip(target[7],max(0,last[7]-.0005),min(.095,last[7]+.0005));servo.command(target[:7],target[7]);last=target.copy()
    d.qfrc_applied[:7]=disturbance_torque(c,step) if name.startswith('force_') and step<700 else 0
    bad=False
    for _ in range(25):
     servo.apply();mujoco.mj_step(m,d)
     if any(w.number for w in d.warning):raise RuntimeError('simulation warning')
     for co in d.contact[:d.ncon]:
      bodies={int(m.geom_bodyid[co.geom1]),int(m.geom_bodyid[co.geom2])}
      if cup in bodies and bodies-set([0,cup,*fingers]):bad=True
    rd.qpos[:]=d.qpos;rd.qvel[:]=d.qvel;mujoco.mj_forward(m,rd);contacts=set()
    for co in rd.contact[:rd.ncon]:
     bodies={int(m.geom_bodyid[co.geom1]),int(m.geom_bodyid[co.geom2])}
     if cup in bodies:contacts.update(bodies&fingers)
    bilateral+=int(len(contacts)==2);unexpected+=int(bad);maxlift=max(maxlift,float(rd.xpos[cup,2]-initial_z));trace.append(np.r_[d.qpos[:7],d.qpos[7]-d.qpos[8],target,rd.xpos[cup],rd.site_xpos[tcp],len(contacts)==2,bad]);raws.append(raw);hold_modes.append(completion_step is not None);cup_upright_history.append(float(rd.xmat[cup].reshape(3,3)[2,2]))
    stable=(rd.xpos[cup,2]-initial_z>.06 and len(contacts)==2 and unexpected==0 and rd.xmat[cup].reshape(3,3)[2,2]>.95 and np.linalg.norm(d.qvel[adr:adr+6])<.05 and np.linalg.norm(rd.xpos[cup,:2]-c['pick_xy_m'])<.02)
    stable_frames=stable_frames+1 if stable else 0
    if completion_step is None and stable_frames>=20:completion_step=step
    if completion_step is not None and step-completion_step>=100:break
 except Exception as e:failure=repr(e)
 xy=float(np.linalg.norm(rd.xpos[cup,:2]-c['place_xy_m']));upright=float(rd.xmat[cup].reshape(3,3)[2,2]);speed=float(np.linalg.norm(d.qvel[adr:adr+6]));final_lift=float(rd.xpos[cup,2]-initial_z)
 post=np.array(trace)[np.array(hold_modes,dtype=bool)] if trace else np.empty((0,24))
 hold_verified=bool(len(post)>=100 and np.all(post[:,18]-initial_z>.06) and np.all(post[:,22]==1) and np.all(np.array(cup_upright_history)[np.array(hold_modes,dtype=bool)]>.95))
 success=hold_verified and not failure and completion_step is not None and len(trace)-1-completion_step>=100 and final_lift>.06 and bilateral>10 and unexpected==0 and xy<.02 and upright>.95 and speed<.05 and len(contacts)==2
 row={'scenario':name,'teacher_free':True,'success':bool(success),'max_lift_m':maxlift,'final_lift_m':final_lift,'bilateral_frames':bilateral,'held_at_finish':len(contacts)==2,'unexpected_arm_cup_contact_frames':unexpected,'final_xy_error_m':xy,'upright_cosine':upright,'final_speed':speed,'first_arm_gap_rad':first_gap,'existing_strict_block_contract_violations':block_violations,'failure':failure,'runtime_s':time.monotonic()-start,'completion_step':completion_step,'goal_reach_deadline_steps':horizon,'continuous_post_hold_verified':hold_verified,'post_completion_hold_points':sum(hold_modes),'real_hardware_tested':False}
 np.savez_compressed(out/f'{name}.npz',states_targets_cup_tcp_contacts=np.array(trace),raw_actions=np.array(raws),raw_chunks=np.array(queries),query_steps=query_steps,completion_hold_mode=hold_modes,cup_upright_history=cup_upright_history);(out/f'{name}.json').write_text(json.dumps(row,indent=2));results.append(row);(out/'summary.json').write_text(json.dumps({**protocol,'results':results,'passed':sum(r['success'] for r in results),'total':len(results),'complete':len(results)==len(scenarios),'hardware_deployment_allowed':False},indent=2));print(json.dumps(row),flush=True)
 if failure:raise RuntimeError(failure)
