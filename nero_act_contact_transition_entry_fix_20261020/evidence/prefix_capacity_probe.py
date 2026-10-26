"""Discarded-weight training-only capacity diagnostic. Never exports or controls hardware."""
import json,pickle
from pathlib import Path
import numpy as np,torch
from training.train import FullTaskPolicy
from training.history_dataset import HistoryReferenceDataset
from executor.model import feedback_preimage,physical_actions
from evidence.probe_fixed_heads import cached_readout
ROOT=Path('/home/user/wq/nero_act_contact_transition_balance_20261020');OUT=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.manual_seed(42)
run=ROOT/'training/pilot_B';payload=torch.load(run/'checkpoints/policy_last.ckpt',map_location='cpu',weights_only=False)
c=dict(payload['config'],device='cpu',pretrained_backbone=False,train_arm_output_and_gripper_only=True,train_nonlinear_gripper_only=False,train_gripper_head_only=False,arm_output_lr=.0001,gripper_adapter_lr=.0003)
stats=pickle.loads((run/'checkpoints/dataset_stats.pkl').read_bytes());p=FullTaskPolicy(c).eval();p.load_state_dict(payload['model'])
manifest=json.loads((ROOT/'training_data/history_recoveries_v3/manifest.json').read_text());rows=[];features=[];observed=[];targets=[];indices=[]
captured=[];handle=p.model.action_head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
for row in manifest['episodes']:
 if row['split']!='train':continue
 path=ROOT/f'training_data/history_complete_v1/episode_{row["id"]}.hdf5'
 dataset=HistoryReferenceDataset([path],stats,c,weighted=False);chosen=[];starts=set()
 for i,(_,j) in enumerate(dataset.samples):
  start=int(dataset.cache[0]['starts'][j])
  if start not in starts:chosen.append(i);starts.add(start)
  if len(chosen)==5:break
 for i in chosen:
  sample=dataset[i];q=sample['current'][None];image=sample['image'][None];history={k:v[None] for k,v in sample['history'].items()}
  with torch.no_grad():p(q,image,history=history)
  features.append(captured.pop());observed.append(sample['current']);targets.append(sample['actions']);rows.append(dict(episode=row['id'],start=int(dataset.cache[0]['starts'][dataset.samples[i][1]])))
handle.remove();feature=torch.cat(features);q=torch.stack(observed);target=torch.stack(targets)
paired={k:getattr(p,'feedback_'+k) for k in ['qpos_mean','qpos_std','action_mean','action_std']}
inverse=feedback_preimage(target,q[:,:8],paired,p.feedback_lower,p.feedback_upper,version=p.feedback_version).detach()
before={k:v.clone() for k,v in p.state_dict().items()}
def metrics():
 with torch.no_grad():
  raw,bounded=cached_readout(p,feature,q);actual=physical_actions(bounded,stats,bounded_gripper=True);desired=physical_actions(target,stats,bounded_gripper=True)
  return [dict(**row,first_width_m=actual[i,0,7].item(),desired_first_width_m=desired[i,0,7].item(),first2_gripper_mae_mm=float((actual[i,:2,7]-desired[i,:2,7]).abs().mean()*1000),first2_arm_mae_rad=float((actual[i,:2,:7]-desired[i,:2,:7]).abs().mean())) for i,row in enumerate(rows)]
initial=metrics()
for step in range(1000):
 raw,bounded=cached_readout(p,feature,q);loss=(raw[:,:2]-inverse[:,:2]).abs().mean()
 p.optimizer.zero_grad(set_to_none=True);loss.backward();p.optimizer.step()
final=metrics();changed=[k for k,v in p.state_dict().items() if not torch.equal(v,before[k])]
assert all(k in ['model.action_head.weight','model.action_head.bias'] or k.startswith('gripper_adapter.') for k in changed)
report=dict(complete=True,diagnostic_only=True,weights_saved=False,optimizer_updates=1000,training_examples=len(rows),training_only=True,zero_latent_cached_decoder=True,initial=initial,final=final,changed_tensors=changed,hardware_commands_sent=False)
(OUT/'prefix_capacity_probe.json').write_text(json.dumps(report,indent=2)+'\n')
for name,records in [('initial',initial),('final',final)]:print(name,[(r['episode'],round(r['first2_gripper_mae_mm'],5)) for r in records if r['episode'] in [2016,2017] and r['start']==5],flush=True)
