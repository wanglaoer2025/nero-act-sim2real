"""Read-only causal audits of frozen run15, never imports a hardware driver."""
import collections,json,pickle
from pathlib import Path
import h5py,numpy as np,torch
from executor.model import BoundedActPolicy,make_policy_observation,physical_actions,FEEDBACK_BUDGETS
from src.act.nero_dataset import preprocess_images
from training.train import complete_windows,weighted_l1,FullTaskPolicy
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'nero_act_contact_recovery_20261018'
payload=torch.load(OLD/'model/run15_first_lift_recovery/policy.ckpt',map_location='cpu',weights_only=False)
config=dict(payload['config'],device='cuda',pretrained_backbone=False)
with (OLD/'model/run15_first_lift_recovery/dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
torch.set_num_threads(2);torch.manual_seed(42)
fresh=BoundedActPolicy(dict(config,device="cpu"))
keys=[k for k in payload['model'] if 'backbones' in k]
random_equal=[k for k in keys if torch.equal(payload['model'][k],fresh.state_dict()[k])]
learnable=[k for k,v in fresh.named_parameters() if 'backbones' in k]
report=dict(backbone_tensors=len(keys),equal_fresh_seed42=len(random_equal),
 backbone_parameters_equal_fresh_seed42=sum(torch.equal(payload['model'][k],fresh.state_dict()[k]) for k in learnable),
 backbone_parameter_tensors=len(learnable))
del fresh
frozen=FullTaskPolicy(dict(config,device='cpu',freeze_backbone=False))
report['backbone_trainable_when_config_false']=sum(p.numel() for n,p in frozen.named_parameters() if 'backbones' in n and p.requires_grad)
del frozen
policy=BoundedActPolicy(config).cuda().eval();policy.load_state_dict(payload['model'])
manifest=json.loads((OLD/'training/run15_first_lift_recovery/manifest.json').read_text())
rows=[r for r in manifest['episodes'] if r['split']=='val'];paths=[Path(r['path']) for r in rows]
windows=list(dict.fromkeys(complete_windows(paths,20,command_context=True)));records=[]
for i,path in enumerate(paths):
 with h5py.File(path) as f:
  stages=f['simulation/stage'].asstr()[:];actions=f['action'][:];states=np.c_[f['observations/qpos'][:],f['observations/gripper_state'][:]]
  choices=[t for j,t in windows if j==i]
  scene=json.loads(f.attrs['scene_config'])
  for stage in dict.fromkeys(stages):
   indices=[t for t in choices if stages[t]==stage]
   for index in np.unique(np.linspace(0,len(indices)-1,min(4,len(indices))).astype(int)) if indices else []:
    t=indices[index];end=min(len(actions),t+20);target=actions[t:end];pad=20-len(target)
    target=np.pad(target,((0,pad),(0,0)),mode='edge')
    history=actions[t-1,7] if t else scene.get('initial_simulation_snapshot',{}).get('previous_target',states[t])[7]
    records.append(dict(stage=stage,q=states[t],obs=make_policy_observation(states[t],stats,config,history),
     rgb=f['observations/images/camera_0'][t],target=target,pad=np.arange(20)>=end-t))
metrics=collections.defaultdict(list);visual=[]
for start in range(0,len(records),32):
 batch=records[start:start+32];q=torch.tensor(np.stack([r['obs'] for r in batch]),device='cuda')
 image=preprocess_images(np.stack([r['rgb'] for r in batch]),(90,160))[:,None].cuda()
 with torch.inference_mode():
  pred=physical_actions(policy(q,image),stats,bounded_gripper=True).cpu().numpy()
  noimage=physical_actions(policy(q,torch.zeros_like(image)),stats,bounded_gripper=True).cpu().numpy()
 for r,p,blank in zip(batch,pred,noimage):
  target=r['target'];delta=target[:10]-r['q'];pd=p[:10]-r['q'];std=stats['action_std'];weights=np.r_[np.full(10,5),np.ones(10)]
  valid=~r['pad'];weights*=valid
  loss=float(np.sum(np.abs((p-target)/std)*weights[:,None])/(weights.sum()*8))
  hold=float(np.sum(np.abs((r['q'][None]-target)/std)*weights[:,None])/(weights.sum()*8))
  direction=float(np.sum(delta[:,:7]*pd[:,:7])/(np.linalg.norm(delta[:,:7])*np.linalg.norm(pd[:,:7])+1e-12))
  reach=np.any(np.abs(delta)>np.array([.0295]*7+[.0095]),axis=1)
  metrics[r['stage']].append([loss,hold,direction,float(np.mean(reach)),float(np.max(np.abs(p-blank)))])
report['validation_windows']=len(records)
report['stages']={k:dict(zip(['model_l1','hold_l1','joint_direction_cosine','unreachable_first10_fraction','max_blank_image_effect'],np.mean(v,axis=0).tolist()),windows=len(v)) for k,v in metrics.items()}
with (ROOT/'evidence/run15_training_diagnosis.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
