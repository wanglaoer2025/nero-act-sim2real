import json
from pathlib import Path
import h5py,numpy as np,torch
from executor.model import BoundedActPolicy,make_policy_observation,physical_actions
from src.act.nero_dataset import preprocess_images
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
old=torch.load(ROOT/'model/run14/policy.ckpt',map_location='cpu',weights_only=False)
new=torch.load(ROOT/'training/run15_first_lift_recovery/artifacts/rebased_initializer.ckpt',map_location='cpu',weights_only=False)
policies=[];stats=[]
for payload in [old,new]:
 c=dict(payload['config'],device='cpu',pretrained_backbone=False)
 p=BoundedActPolicy(c).eval();p.load_state_dict(payload['model']);policies.append(p)
 stats.append({k:np.array(v,np.float32) for k,v in c['constraint_loss_stats'].items()})
with h5py.File(ROOT/'training_data/recoveries_v1/episode_1001.hdf5') as f:
 states=np.c_[f['observations/qpos'][:],f['observations/gripper_state'][:]]
 images=f['observations/images/camera_0'][:];actions=f['action'][:]
 previous=json.loads(f.attrs['scene_config'])['initial_simulation_snapshot']['previous_target'][7]
 errors=[]
 with torch.inference_mode():
  for index in [0,20,500,1500]:
   history=previous if index==0 else actions[index-1,7];rgb=preprocess_images(images[index][None],(90,160))[None]
   outputs=[]
   for payload,policy,stat in zip([old,new],policies,stats):
    state=torch.from_numpy(make_policy_observation(states[index],stat,payload['config'],history))[None]
    outputs.append(physical_actions(policy(state,rgb),stat,bounded_gripper=True))
   torch.testing.assert_close(outputs[0],outputs[1],atol=2e-6,rtol=2e-5)
   errors.append(float((outputs[0]-outputs[1]).abs().max()))
report=dict(passed=True,real_frozen_run14_weights=True,observations=4,maximum_physical_prediction_error=max(errors),
 initialization_only=True,trained_model_tested=False)
with (ROOT/'evidence/run15_actual_initializer_rebase_audit.json').open('x') as f:json.dump(report,f,indent=2)
print(report)
