"""Throwaway cached-feature capacity probe, never a deployable checkpoint."""
from pathlib import Path
import sys,json,pickle,copy,hashlib
import torch,numpy as np
from torch.utils.data import default_collate
ROOT=Path('/home/user/wq/nero_act_grouped_gripper_20261021');sys.path.insert(0,str(ROOT))
from training.history import initialize_history,build_history_policy,_batch_to_device
from training.history_dataset import HistoryReferenceDataset
from training.pose_objective import fk_spec_from_scene
from training.integrity import snapshot,require_unchanged
from executor.model import BoundedActPolicy,feedback_preimage,feedback_readout,physical_actions

def main():
 torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);assert torch.cuda.is_available()
 data=(ROOT/'training_data/uniform_complete_v1').resolve();manifest=json.loads((data/'manifest.json').read_text())
 init=ROOT.parent/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'
 payload=torch.load(init,map_location='cpu',weights_only=False);stats=pickle.loads((data/'dataset_stats.pkl').read_bytes());old=pickle.loads((init.parent/'dataset_stats.pkl').read_bytes())
 cfg=json.loads((ROOT/'config/history_experiment.json').read_text());scene=json.loads((data/'scene_config.json').read_text());cfg.update(scene_config=scene,pose_kinematics_spec=fk_spec_from_scene(scene),training_budget=dict(kind='updates',limit=1000),num_epochs=0,variant='A',stage='pilot')
 config,initial=initialize_history(payload,old,stats,cfg,variant='A');policy=build_history_policy(config).cuda().eval();policy.load_state_dict(initial,strict=True)
 refs={};all_stats={};selections={};hook_values=[]
 handle=policy.model.action_head.register_forward_pre_hook(lambda m,a:hook_values.append(a[0].detach()))
 for split,count in [('train',256),('val',64)]:
  ds=HistoryReferenceDataset([Path(r['path']) for r in manifest['episodes'] if r['split']==split],stats,config,weighted=False)
  unique={}
  for i,(fi,j) in enumerate(ds.samples):
   start=int(ds.cache[fi]['starts'][j]);group=int(ds.cache[fi]['gripper_groups'][start]);unique.setdefault((ds.paths[fi].stem,start),(i,group))
  indices=[]
  for group in range(4):
   ids=[i for i,g in unique.values() if g==group];indices += [ids[j] for j in np.linspace(0,len(ids)-1,min(count,len(ids)),dtype=int)]
  if split=='train':
   for sid in [('episode_2017',5),('episode_2018',5),('episode_2018',10),('episode_2017',1674),('episode_2017',1689)]:indices.append(unique[sid][0])
  indices=list(dict.fromkeys(indices));selections[split]=[ds.sample_ids[i] for i in indices];collected={k:[] for k in ['hidden','current','base','target','physical','labels','groups']}
  for left in range(0,len(indices),32):
   batch=default_collate([ds[i] for i in indices[left:left+32]]);values,kwargs=_batch_to_device(batch,'cuda',with_history=False);q,image,actions,pad=values;hook_values.clear()
   with torch.no_grad():
    output=BoundedActPolicy.forward(policy,q,image,return_unbounded=True);h=hook_values[-1][:,:2].contiguous()
    base=torch.nn.functional.linear(h,policy.model.action_head.weight[7:8],policy.model.action_head.bias[7:8])[...,0]
    paired={k:getattr(policy,'feedback_'+k) for k in ['qpos_mean','qpos_std','action_mean','action_std']}
    target=feedback_preimage(actions,q[:,:8],paired,policy.feedback_lower,policy.feedback_upper,version=policy.feedback_version)[:,:2,7]-base
    physical=physical_actions(output['a_hat'],stats,bounded_gripper=True)[:,:2,7]
    labels=actions[:,:2,7]*float(stats['action_std'][7])+float(stats['action_mean'][7])
   for k,v in dict(hidden=h,current=q,base=base,target=target,physical=physical,labels=labels,groups=kwargs['gripper_groups']).items():collected[k].append(v.detach())
  refs[split]={k:torch.cat(v) for k,v in collected.items()};print(json.dumps(dict(cached=split,count=len(indices))),flush=True)
 handle.remove();torch.save(dict(refs=refs,selections=selections,stats=stats,config=config),'/tmp/nero_head_features.pt');return
 results=[]
 def physical(q,raw):
  fake=raw.new_zeros(len(q),20,8);fake[:,:2,7]=raw
  out=feedback_readout(fake,q[:,:8],paired,policy.feedback_lower,policy.feedback_upper,version=policy.feedback_version)
  return physical_actions(out,stats,bounded_gripper=True)[:,:2,7]
 for variant in ['existing_hidden','hidden_plus_current15']:
  torch.manual_seed(42)
  head=copy.deepcopy(policy.gripper_adapter)
  if variant=='hidden_plus_current15':
   old_linear=head[0];layer=torch.nn.Linear(old_linear.in_features+15,old_linear.out_features,device='cuda')
   with torch.no_grad():layer.weight.zero_();layer.weight[:,:old_linear.in_features].copy_(old_linear.weight);layer.bias.copy_(old_linear.bias)
   head[0]=layer
  def features(r):return r['hidden'] if variant=='existing_hidden' else torch.cat([r['hidden'],r['current'][:,None].expand(-1,2,-1)],dim=-1)
  train=refs['train'];val=refs['val'];x=features(train);xv=features(val)
  with torch.no_grad():assert torch.equal(head(x[0:1])[...,0],policy.gripper_adapter(train['hidden'][0:1])[...,0])
  optimizer=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=0.)
  for step in range(1000):
   i=torch.randint(len(x),(128,),device='cuda');loss=(head(x[i])[...,0]-train['target'][i]).square().mean();optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),.1);optimizer.step()
  summary=dict(variant=variant,temporary_head_updates=1000,train_size=len(x),val_size=len(xv),data_used='stratified existing training views only; validation never optimized',critical=[])
  with torch.no_grad():
   for split,r,features_r in [('train',train,x),('val',val,xv)]:
    prediction=physical(r['current'],r['base']+head(features_r)[...,0]);errors=(prediction-r['labels']).abs().max(1).values*1000
    summary[split]=dict(mean_mm=float(errors.mean()),max_mm=float(errors.max()),per_group={str(g):float(errors[r['groups']==g].mean()) for g in range(4)})
    if split=='train':
     for j,sid in enumerate(selections[split]):
      if sid[:2] in [('episode_2017',5),('episode_2018',5),('episode_2018',10),('episode_2017',1674),('episode_2017',1689)]:
       state=r['current'][j,:8]*paired['qpos_std']+paired['qpos_mean'];previous=state[7]+r['current'][j,8]*.01
       summary['critical'].append(dict(sample=sid,error_mm=float(errors[j]),prediction_mm=(prediction[j]*1000).cpu().tolist(),teacher_mm=(r['labels'][j]*1000).cpu().tolist(),next_minus_previous_mm=float((prediction[j,1]-previous)*1000)))
  results.append(summary);print(json.dumps(summary),flush=True)
 sources=snapshot([Path(__file__),init,init.parent/'dataset_stats.pkl',data/'manifest.json',data/'dataset_stats.pkl',ROOT/'config/history_experiment.json'])
 proof=dict(complete=True,results=results,source_sha256=sources,selections=selections,persisted_checkpoints=0,temporary_head_updates=2000,scope='cached-feature single control; no rollout, not full training, not architecture accepted',hardware_commands_sent=False)
 Path('/tmp/nero_head_capacity_probe.json').write_text(json.dumps(proof,indent=2)+'\n')
if __name__=='__main__':main()
