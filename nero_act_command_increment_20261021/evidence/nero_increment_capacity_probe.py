"""Throwaway same-feature comparison: learn increments from causal previous command."""
from pathlib import Path
import json,torch,copy

def main():
 torch.set_num_threads(2);cache=torch.load('/tmp/nero_head_features.pt',map_location='cuda',weights_only=False)
 refs,selections,stats=cache['refs'],cache['selections'],cache['stats'];results=[]
 def previous(r):
  physical=r['current'][:,:8]*torch.as_tensor(stats['qpos_std'],device='cuda')+torch.as_tensor(stats['qpos_mean'],device='cuda')
  return physical[:,7:8]+r['current'][:,8:9]*.01,physical[:,7:8]
 for mode in ['increment_hidden_current','increment_hidden_current_query']:
  torch.manual_seed(42)
  def features(r):
   values=[r['hidden'],r['current'][:,None].expand(-1,2,-1)]
   if mode.endswith('_query'):values.append(torch.eye(20,device='cuda')[:2][None].expand(len(r['current']),-1,-1))
   return torch.cat(values,dim=-1)
  train,val=refs['train'],refs['val'];x,xv=features(train),features(val);prev,state=previous(train)
  target=(train['labels']-prev)/.001
  assert target.abs().max()<1.00001
  head=torch.nn.Sequential(torch.nn.Linear(x.shape[-1],64),torch.nn.SiLU(),torch.nn.Linear(64,1)).cuda();torch.nn.init.zeros_(head[2].weight);torch.nn.init.zeros_(head[2].bias)
  with torch.no_grad():assert torch.equal(torch.tanh(head(x))[...,0],torch.zeros_like(target))
  opt=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=0.)
  for step in range(1000):
   i=torch.randint(len(x),(128,),device='cuda');pred=head(x[i])[...,0].tanh();loss=(pred-target[i]).square().mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),.1);opt.step()
  summary=dict(mode=mode,temporary_head_updates=1000,critical=[])
  with torch.no_grad():
   for split,r,f in [('train',train,x),('val',val,xv)]:
    p,s=previous(r);pred=(p+.001*head(f)[...,0].tanh()).clamp(0,.095);pred=torch.maximum(torch.minimum(pred,s+.0095),s-.0095)
    err=(pred-r['labels']).abs().max(1).values*1000
    summary[split]=dict(mean_mm=float(err.mean()),max_mm=float(err.max()),per_group={str(g):float(err[r['groups']==g].mean()) for g in range(4)})
    if split=='train':
     for j,sid in enumerate(selections[split]):
      if sid[:2] in [('episode_2017',5),('episode_2018',5),('episode_2018',10),('episode_2017',1674),('episode_2017',1689)]:summary['critical'].append(dict(sample=sid,error_mm=float(err[j]),next_minus_previous_mm=float((pred[j,1]-p[j,0])*1000),prediction_mm=(pred[j]*1000).cpu().tolist(),teacher_mm=(r['labels'][j]*1000).cpu().tolist()))
  results.append(summary);print(json.dumps(summary),flush=True)
  torch.save(dict(state=head.state_dict(),input_dim=x.shape[-1],prototype_only=True,mode=mode),f'/tmp/nero_{mode}_prototype.pt')
 Path('/tmp/nero_increment_capacity_probe.json').write_text(json.dumps(dict(complete=True,results=results,temporary_head_updates=2000,deployable=False,scope='cached-feature prototype, no full-loop qualification',hardware_commands_sent=False),indent=2)+'\n')
if __name__=='__main__':main()
