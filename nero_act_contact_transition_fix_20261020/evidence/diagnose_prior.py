import json,pickle,hashlib
from pathlib import Path
import numpy as np
import torch
from executor.model import BoundedActPolicy,physical_actions
from training.history_dataset import HistoryReferenceDataset
ROOT=Path('/home/user/wq/nero_act_history_feedback_20261019');OUT=Path('/home/user/wq/nero_act_contact_transition_fix_20261020/evidence')
torch.set_num_threads(2);torch.manual_seed(42)
rows=[]
for variant in ['A','B']:
 run=ROOT/f'training/pilot_{variant}';payload=torch.load(run/'checkpoints/policy_last.ckpt',map_location='cpu',weights_only=False)
 config=dict(payload['config'],device='cpu',pretrained_backbone=False);stats=pickle.loads((run/'checkpoints/dataset_stats.pkl').read_bytes())
 policy=BoundedActPolicy(config).eval();policy.load_state_dict(payload['model'])
 for number in [2016,2017,2013,2014]:
  dataset=HistoryReferenceDataset([ROOT/f'training_data/history_complete_v1/episode_{number}.hdf5'],stats,config,weighted=False)
  sample=dataset[0];q=sample['current'][None];image=sample['image'][None];actions=sample['actions'][None];pad=sample['padding'][None]
  kwargs={'history':{k:v[None] for k,v in sample['history'].items()}} if variant=='B' else {}
  with torch.no_grad():
   prior=policy(q,image,return_unbounded=True,**kwargs)
   target=physical_actions(actions,stats,bounded_gripper=True)[0];deployed=physical_actions(prior['a_hat'],stats,bounded_gripper=True)[0]
   unbounded=physical_actions(prior['unbounded_a_hat'],stats,bounded_gripper=False)[0]
   posterior=[]
   for seed in range(8):
    torch.manual_seed(seed);value=policy(q,image,actions,pad,**kwargs)
    posterior.append(physical_actions(value['a_hat'],stats,bounded_gripper=True)[0].numpy())
  posterior=np.array(posterior);label=target.numpy();pred=deployed.numpy()
  records=dict(variant=variant,episode=number,prior_prefix_grip_mae_mm=float(abs(pred[:2,7]-label[:2,7]).mean()*1000),
   posterior_prefix_grip_mae_mm=float(abs(posterior[:,:2,7]-label[None,:2,7]).mean()*1000),
   prior_prefix_arm_mae_rad=float(abs(pred[:2,:7]-label[:2,:7]).mean()),posterior_prefix_arm_mae_rad=float(abs(posterior[:,:2,:7]-label[None,:2,:7]).mean()),
   prior_first2_width_m=pred[:2,7].tolist(),unbounded_first2_width_m=unbounded[:2,7].tolist(),
   posterior_first2_width_mean_m=posterior[:,:2,7].mean(0).tolist(),teacher_first2_width_m=label[:2,7].tolist())
  rows.append(records);print(json.dumps(records),flush=True)
(OUT/'prior_posterior_probe.json').write_text(json.dumps(dict(complete=True,diagnostic_only=True,uses_future_labels_only_for_posterior_comparison=True,records=rows),indent=2)+'\n')
