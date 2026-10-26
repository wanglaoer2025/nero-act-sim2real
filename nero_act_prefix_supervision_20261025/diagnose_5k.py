import os
os.environ['MUJOCO_GL']='egl'
import sys,json,hashlib,collections,time
from pathlib import Path
import numpy as np,torch,h5py,cv2,mujoco
from torch.utils.data import default_collate
ROOT=Path('/home/user/wq/nero_act_transfer_v2_20261025')
sys.path[:0]=[str(ROOT),str(ROOT/'vendor')]
from policy import ACTPolicy
from dataset import V2Dataset
from contract import make_observation
from executor.model import make_policy_observation,physical_actions
from executor.contract import FullTaskContract
from executor.release_hold import hold_open_before_grasp_depth
from src.act.nero_dataset import preprocess_images
from simulation.scene import build_scene
torch.set_num_threads(2)
OUT=Path('/home/user/wq/nero_act_prefix_supervision_20261025/diagnosis_5k');OUT.mkdir(exist_ok=False)
checkpoint=ROOT/'training/formal/policy_5000.ckpt'
cp=torch.load(checkpoint,map_location='cpu',weights_only=False);stats=cp['stats']
policy=ACTPolicy(stats,pretrained=False).cuda().eval();policy.load_state_dict(cp['model']);policy.config=cp['config']
case=ROOT/'training/formal/dev_5000/centre_0'
trace=np.load(case/'rollout.npz');scene=json.loads((case/'scene_configuration.json').read_text())
snap=json.loads((case/'physical_snapshots.json').read_text())['snapshots'][0]
records=json.loads((case/'visual_observations.json').read_text())
measures=json.loads((case/'measurements.json').read_text())
model=build_scene(scene);data=mujoco.MjData(model)
data.qpos[:]=snap['qpos'];data.qvel[:]=snap['qvel'];data.time=trace['query_simulation_time_s'][0];mujoco.mj_forward(model,data)
model.vis.global_.offheight=720;model.vis.global_.offwidth=1280
with mujoco.Renderer(model,height=720,width=1280) as renderer:
 renderer.update_scene(data,camera='calibrated_nero_camera');renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
 rgb=renderer.render().copy()
cv2.imwrite(str(OUT/'initial_rgb.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
state=np.array(records[0]['paired_feedback_state']);features=trace['visual_features_m'][0]
previous=records[0]['causal_previous_gripper_target_m']
obs=make_policy_observation(state,stats,cp['config'],previous,geometry_features_m=features)
reference_obs=make_observation(state,features,previous,stats)
image=preprocess_images(rgb[None],[180,320])[None].cuda()
def predict(observation,im):
 with torch.inference_mode():
  out=policy(torch.as_tensor(observation)[None].to(policy.device),im)
 return physical_actions(out[0],stats,bounded_gripper=True).cpu().numpy()
prediction=predict(obs,image)
replay_error=float(abs(prediction-trace['raw_chunks'][0]).max())
assert replay_error<1e-5,('Failed to replay actual policy input',replay_error)
cvimage=torch.from_numpy((cv2.resize(rgb,(320,180),interpolation=cv2.INTER_LINEAR).astype('float32')/255).transpose(2,0,1))[None,None].cuda()
pixel_error=float((image-cvimage).abs().max())
ablations={}
for name,o,im in [('zero_RGB',obs,image*0),('CV2_RGB',obs,cvimage),
 ('cup_X_plus20mm',make_observation(state,features+np.array([.02,0,0,0,0,0]),previous,stats),image),
 ('cup_Y_plus20mm',make_observation(state,features+np.array([0,.02,0,0,0,0]),previous,stats),image)]:
 pred=predict(o,im);ablations[name]=dict(max_prefix_arm_delta_rad=float(abs(pred[:2,:7]-prediction[:2,:7]).max()))
site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_tcp')
def tcp(q):
 data.qpos[:7]=q;mujoco.mj_forward(model,data);return data.site_xpos[site].copy()
goal=np.r_[scene['pick_xy_m'],scene['table_top_z_m']+scene['grasp_tcp_height_m']+scene['lift_height_m']]
guard=FullTaskContract(cp['config'])
sent=guard.validate_prefix(prediction,dict(state=trace['feedback_at_query'][0],age_s=0.),trace['sent_targets'][3])[:2]
start=tcp(state[:7])
movement=dict(initial_tcp_m=start.tolist(),approach_goal_m=goal.tolist(),initial_distance_m=float(np.linalg.norm(start-goal)),
 raw_prefix_tcp_m=[tcp(q[:7]).tolist() for q in prediction[:2]],sent_prefix_tcp_m=[tcp(q[:7]).tolist() for q in sent],
 raw_prefix_distance_m=[float(np.linalg.norm(tcp(q[:7])-goal)) for q in prediction[:2]],
 sent_prefix_distance_m=[float(np.linalg.norm(tcp(q[:7])-goal)) for q in sent])
errors=[]
for i,step in enumerate(trace['query_steps']):
 before=trace['sent_targets'][step-1]
 checked=guard.validate_prefix(trace['raw_chunks'][i],dict(state=trace['feedback_at_query'][i],age_s=0.),before)[:2]
 errors.append(float(abs(checked[:,:7]-trace['sent_targets'][step:step+2,:7]).max()))
assert max(errors)<1e-12
tcp_trace=np.array([x['tcp_xyz'] for x in measures]);steps=trace['query_steps']
sent_prefix=np.stack([trace['sent_targets'][i:i+2] for i in steps])
tracking=abs(trace['states'][1:,:7]-trace['sent_targets'][:-1,:7])
execution=dict(max_replayed_arm_submission_error_rad=max(errors),queries=len(errors),
 max_tcp_displacement_m=float(np.linalg.norm(tcp_trace-tcp_trace[0],axis=1).max()),
 tcp_height_range_m=float(np.ptp(tcp_trace[:,2])),final_tcp_m=tcp_trace[-1].tolist(),
 last500_raw_sent_arm_max_error_rad=float(abs(trace['raw_chunks'][-500:,:2,:7]-sent_prefix[-500:,:,:7]).max()),
 last1000_tracking_mean_abs_rad=float(tracking[-1000:].mean()),
 source_counts={str(k):int(v) for k,v in zip(*np.unique(trace['target_sources'],return_counts=True))})
manifest=json.loads((ROOT/'data/v2_contour_01/manifest.json').read_text())
representatives=[];splits=[]
for split,count in [('train',12),('dev',6)]:
 for i in range(count):
  row=next(r for r in sorted(manifest['episodes'],key=lambda r:r['path']) if r['position_group']==f'{split}_{i:02d}')
  representatives.append(row['path']);splits.append(split)
dataset=V2Dataset(representatives,stats,stride=80)
aggregate=collections.defaultdict(lambda:collections.Counter());samples=[]
for offset in range(0,len(dataset),32):
 batch=default_collate([dataset[i] for i in range(offset,min(offset+32,len(dataset)))])
 tensors={k:v.cuda() for k,v in batch.items() if isinstance(v,torch.Tensor)}
 with torch.inference_mode():
  prior=policy(tensors['observation'],tensors['image'])
  post=policy(tensors['observation'],tensors['image'],tensors['target'],tensors['padding'])
  current=tensors['observation'][:,:7]*policy.qpos_std[:7]+policy.qpos_mean[:7]
  identity=((current-policy.action_mean[:7])/policy.action_std[:7])[:,None,:].expand(-1,20,-1)
  valid=~tensors['padding'];target=tensors['target'][:,:,:7]
  prior_error=(prior[:,:,:7]-target).abs();post_error=(post['prediction'][:,:,:7]-target).abs()
  hold_error=(identity-target).abs()
  prefix_physical_error=prior_error[:,:2]*policy.action_std[:7]
  prefix_teacher_change=hold_error[:,:2]*policy.action_std[:7]
 for j in range(len(batch['stage'])):
  k=offset+j;valid20=valid[j];valid2=valid20[:2]
  sums=dict(prior_error=float(prior_error[j][valid20].sum()),post_error=float(post_error[j][valid20].sum()),
   hold_error=float(hold_error[j][valid20].sum()),count=int(valid20.sum())*7,
   prefix_error_rad=float(prefix_physical_error[j][valid2].sum()),
   prefix_teacher_change_rad=float(prefix_teacher_change[j][valid2].sum()),prefix_count=int(valid2.sum())*7,samples=1)
  for group in ['all',splits[dataset.samples[k][0]],batch['stage'][j]]:aggregate[group].update(sums)
 if offset%160==0:print(json.dumps(dict(samples=min(offset+32,len(dataset)),total=len(dataset))),flush=True)
summary={}
for group,a in aggregate.items():
 summary[group]=dict(samples=a['samples'],deployment_arm20_normalized_MAE=a['prior_error']/a['count'],
  posterior_arm20_normalized_MAE=a['post_error']/a['count'],copy_state_arm20_normalized_MAE=a['hold_error']/a['count'],
  deployment_prefix2_MAE_rad=a['prefix_error_rad']/a['prefix_count'],
  teacher_prefix2_motion_mean_abs_rad=a['prefix_teacher_change_rad']/a['prefix_count'])
result=dict(checkpoint=str(checkpoint),updates=cp['updates'],scope='frozen5Kdiagnosis only; representative18episodes stride80; posterior has labels and is not deployment',
 actual_initial_prediction_replay_max_error=replay_error,normalization_max_abs_difference=float(abs(obs-reference_obs).max()),
 preprocess_max_pixel_difference=pixel_error,preprocess_and_counterfactual_sensitivity=ablations,
 initial_movement=movement,execution=execution,representative_data_comparison=summary,
 source_initialization=cp['config']['action_initializer'],optimizer_updates=0,training_process_changed=False,
 hardware_commands_sent=False)
(OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
