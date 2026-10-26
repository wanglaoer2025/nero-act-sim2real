"""Throwaway counterfactual optimizer probe; no checkpoint or dataset is changed."""
from pathlib import Path
import sys,json,pickle,hashlib,copy
import numpy as np
import torch
from torch.utils.data import default_collate
ROOT=Path('/home/user/wq/nero_act_grouped_gripper_20261021');sys.path.insert(0,str(ROOT))
from training.history import initialize_history,build_history_policy,_batch_to_device
from training.history_dataset import HistoryReferenceDataset
from training.approach_preservation import ApproachPreservation
from training.pose_objective import fk_spec_from_scene
from training.optimizer_control import gradient_relation
from executor.model import physical_actions
from training.integrity import snapshot,require_unchanged

def digest(policy):
 h=hashlib.sha256()
 for k,v in policy.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def main():
 torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
 assert torch.cuda.is_available()
 data=(ROOT/'training_data/uniform_complete_v1').resolve();manifest=json.loads((data/'manifest.json').read_text())
 init=ROOT.parent/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'
 source=torch.load(init,map_location='cpu',weights_only=False);old=pickle.loads((init.parent/'dataset_stats.pkl').read_bytes());stats=pickle.loads((data/'dataset_stats.pkl').read_bytes())
 settings=json.loads((ROOT/'config/history_experiment.json').read_text());scene=json.loads((data/'scene_config.json').read_text())
 settings.update(training_budget=dict(kind='updates',limit=1000),num_epochs=0,variant='A',stage='pilot',scene_config=scene,pose_kinematics_spec=fk_spec_from_scene(scene))
 config,initial=initialize_history(source,old,stats,settings,variant='A')
 dataset=HistoryReferenceDataset([Path(r['path']) for r in manifest['episodes'] if r['split']=='train'],stats,config)
 order=np.load(data/'batch_order_pilot.npy',allow_pickle=False)
 ids=[('episode_2018',5,0),('episode_2017',5,0),('episode_0',10,0),('episode_2017',1674,0),('episode_2018',10,0)]
 for name in ['episode_2002','episode_2006','episode_2010','episode_2017']:
  rows=[i for i,(n,t,v) in enumerate(dataset.sample_ids) if n==name and v==0 and int(dataset[i]['gripper_group'])==3] if False else [i for i,(fi,j) in enumerate(dataset.samples) if dataset.paths[fi].stem==name and dataset.cache[fi]['views'][j]==0 and dataset.cache[fi]['gripper_groups'][dataset.cache[fi]['starts'][j]]==3]
  unique={dataset.sample_ids[i]:i for i in rows};rows=list(unique.values())
  if rows:
   ids += [dataset.sample_ids[rows[min(len(rows)-1,j)]] for j in [0,len(rows)//2]]
 ids=list(dict.fromkeys(ids));samples=[dataset[next(i for i,x in enumerate(dataset.sample_ids) if x==sid)] for sid in ids]
 q=torch.stack([s['current'] for s in samples]).cuda();images=torch.stack([s['image'] for s in samples]).cuda();target=torch.stack([s['actions'][:2,7] for s in samples]).cuda()*float(stats['action_std'][7])+float(stats['action_mean'][7])
 inputs=[Path(__file__),init,init.parent/'dataset_stats.pkl',data/'manifest.json',data/'dataset_stats.pkl',data/'batch_order_pilot.npy',ROOT/'config/history_experiment.json',ROOT/'training/pilot_A/checkpoints/policy_last.ckpt']
 inputs+=list((ROOT/'training').glob('*.py'))+list((ROOT/'executor').glob('*.py'));hashes=snapshot(inputs)
 records=[];states=[]
 latest=torch.load(ROOT/'training/pilot_A/checkpoints/policy_last.ckpt',map_location='cpu',weights_only=False)
 for name,weights in [('B252',initial),('grouped1000',latest['model'])]:
  policy=build_history_policy(config).cuda();policy.load_state_dict(weights,strict=True);policy.eval();before_hash=digest(policy)
  keeper=ApproachPreservation(config['approach_preservation'],stats,'cuda');params=[p for p in policy.parameters() if p.requires_grad];grip=list(policy.gripper_adapter.parameters())
  saved=[p.detach().clone() for p in params];group_ids={id(p) for p in grip}
  def evaluate():
   return torch.cat([physical_actions(policy(q[i:i+1],images[i:i+1]),stats,bounded_gripper=True)[0,:2,7][None] for i in range(len(samples))])
  base=evaluate();probe_gradients=[]
  for i in range(len(samples)):
   g=torch.autograd.grad((base[i]-target[i]).abs().mean(),grip,retain_graph=True)
   probe_gradients.append(torch.cat([v.flatten() for v in g]).detach())
  before=(base-target).abs().max(1).values.detach()
  for ordinal in [0,24,99,499,149,249,749,999,74,349,649,899]:
   torch.manual_seed(42+ordinal);policy.train()
   values,kwargs=_batch_to_device(default_collate([dataset[int(i)] for i in order[ordinal] if i>=0]),'cuda',with_history=False)
   losses=policy(*values,**kwargs);keep=keeper.loss(policy,ordinal)
   policy.optimizer.zero_grad(set_to_none=True);(losses['loss']+keeper.weight*keep).backward()
   raw=[p.grad.detach().clone() for p in params]
   raw_grip=torch.cat([p.grad.flatten() for p in grip]);raw_cos=[gradient_relation(raw_grip,g)['cosine'] for g in probe_gradients]
   for mode in ['global_clip_adam','separate_clip_adam']:
    with torch.no_grad():
     for p,value,g in zip(params,saved,raw):p.copy_(value);p.grad=g.clone()
    policy.optimizer.state.clear()
    if mode=='global_clip_adam':norm=float(torch.nn.utils.clip_grad_norm_(params,.1))
    else:
     arm=[p for p in params if id(p) not in group_ids];torch.nn.utils.clip_grad_norm_(arm,.1);norm=float(torch.nn.utils.clip_grad_norm_(grip,.1))
    policy.optimizer.step();policy.eval()
    delta=torch.cat([(p-v).flatten() for p,v in zip(params,saved) if id(p) in group_ids])
    # Align flattening with gripper_adapter.parameters() independently of module traversal.
    old_by_id={id(p):v for p,v in zip(params,saved)};delta=torch.cat([(p-old_by_id[id(p)]).flatten() for p in grip])
    actual_cos=[gradient_relation(-delta,g)['cosine'] for g in probe_gradients]
    with torch.no_grad():after=(evaluate()-target).abs().max(1).values
    records.append(dict(policy=name,batch=ordinal,mode=mode,raw_cosines=raw_cos,actual_descent_cosines=actual_cos,before_error_mm=(before*1000).cpu().tolist(),after_error_mm=(after*1000).cpu().tolist(),gradient_norm_before_clip=norm))
    with torch.no_grad():
     for p,value in zip(params,saved):p.copy_(value)
    policy.optimizer.state.clear()
   print(json.dumps(dict(policy=name,batch=ordinal,raw=raw_cos[:4],adam=records[-2]['actual_descent_cosines'][:4])),flush=True)
  after_hash=digest(policy);assert before_hash==after_hash and not policy.optimizer.state
  states.append(dict(policy=name,before=before_hash,after=after_hash));del policy,keeper;torch.cuda.empty_cache()
 require_unchanged(hashes)
 proof=dict(complete=True,probe_samples=ids,records=records,source_sha256=hashes,model_states=states,persisted_training_updates=0,counterfactual_optimizer_steps=len(records),optimizer_moments='fresh temporary moments, not reconstructed past1000 history',hardware_commands_sent=False)
 Path('/tmp/nero_actual_update_probe.json').write_text(json.dumps(proof,indent=2)+'\n')
 print(json.dumps(dict(complete=True,records=len(records),probes=len(ids))),flush=True)
if __name__=='__main__':main()
