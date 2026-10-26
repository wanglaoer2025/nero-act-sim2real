import json,pickle
from pathlib import Path
import torch,numpy as np
from training.history import initialize_history,build_history_policy
from training.approach_preservation import ApproachPreservation
root=Path('.').resolve();torch.set_num_threads(2);torch.manual_seed(42)
source_path=root.parent/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'
payload=torch.load(source_path,map_location='cpu',weights_only=False)
old=pickle.loads((source_path.parent/'dataset_stats.pkl').read_bytes());stats=pickle.loads((root/'training_data/approach_complete_v1/dataset_stats.pkl').read_bytes())
settings=json.loads((root/'config/history_experiment.json').read_text());settings.update(device='cpu',pretrained_backbone=False,variant='A',stage='pilot',num_epochs=0,training_budget={'kind':'updates','limit':1000},pose_loss_weight=0.)
c,w=initialize_history(payload,old,stats,settings,variant='A');p=build_history_policy(c).to('cpu').eval();p.load_state_dict(w)
keeper=ApproachPreservation(settings['approach_preservation'],stats,'cpu')
report=keeper.assess(p);report['first_queries']=[]
with torch.no_grad():
 for i in [0,50,200,380,422]:
  error=(p(keeper.current[i:i+1],keeper.image[i:i+1])-keeper.target[i:i+1])*torch.tensor(stats['action_std'],device='cpu')
  report['first_queries'].append(dict(index=i,arm_max=float(error[:,:2,:7].abs().max()),grip_max=float(error[:,:2,7].abs().max()),arm_delta=error[0,0,:7].cpu().tolist()))
(root/'evidence/initial_replay_cpu.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

errors=[]
with torch.no_grad():
 for i in range(len(keeper.current)):
  e=(p(keeper.current[i:i+1],keeper.image[i:i+1])-keeper.target[i:i+1])*torch.tensor(stats['action_std'])
  errors.append([float(e[:,:2,:7].abs().max()),float(e[:,:2,7].abs().max())])
report['per_query_errors']=errors
report['worst_indices']=np.argsort(np.asarray(errors)[:,0])[-12:].tolist()
(root/'evidence/initial_replay_cpu.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(maximum=report['maximum_arm_prefix_change_rad'],worst=[(i,errors[i]) for i in report['worst_indices']])),flush=True)
