"""Development-only snapshot rollouts of previously failed cases, not independent acceptance."""
import json
from pathlib import Path
import torch
from executor.policy_worker import FrozenAct
from simulation.evaluate import rollout,scenarios,evaluation_scene
root=Path(__file__).resolve().parents[1]
out=root/'evidence/correction_origins';out.mkdir(exist_ok=False)
requests=[('run5_prefix_margin',4000,'independent_2'),('run5_prefix_margin',4000,'independent_4'),
          ('run5_prefix_margin',4000,'independent_5'),('run4_current_source',3000,'independent_2'),
          ('run4_current_source',3000,'independent_3')]
policies={}
provenance=[]
for model,offset,name in requests:
 if model not in policies:
  p=FrozenAct(root/'model'/('candidate_'+model),device='cuda');p.model.config=p.config;p.model.device=torch.device('cuda');policies[model]=p
 p=policies[model];torch.set_num_threads(2)
 case=next(c for c in scenarios(offset) if c['name']==name)
 destination=out/(model+'_'+name)
 r=rollout(evaluation_scene(p.config),p.model,p.stats,case,destination)
 provenance.append(dict(checkpoint_sha256=p.metadata['checkpoint_sha256'],model=model,case=case,output=str(destination),role='development_snapshot_source_not_independent_acceptance'))
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
 print(json.dumps(dict(model=model,case=name,frames=r['frames'],success=r['success'],error=r['error'])),flush=True)
