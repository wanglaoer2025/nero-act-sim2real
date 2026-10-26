import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import sys,json
import torch,numpy as np,cv2
ROOT=Path('/home/user/wq/nero_act_transfer_v2_20261025');sys.path[:0]=[str(ROOT),str(ROOT/'vendor')]
from policy import ACTPolicy
import supervised_contact_escape as runner
torch.set_num_threads(2)
cp=torch.load(ROOT/'training/formal/policy_2000.ckpt',map_location='cpu',weights_only=False)
p=ACTPolicy(cp['stats'],pretrained=False).cuda().eval();p.load_state_dict(cp['model']);p.config=cp['config']
case=ROOT/'training/formal/dev_2000/centre_0'
scene=json.loads((case/'scene_configuration.json').read_text());scenario=json.loads((case/'result.json').read_text())['scenario']
out=ROOT/'evidence/regression_2k_diagnosis/resize_control';out.mkdir(exist_ok=False)
class BudgetPolicy:
 def __init__(self,inner):self.inner=inner;self.queries=0
 def __getattr__(self,key):return getattr(self.inner,key)
 def __call__(self,*args,**kwargs):
  if self.queries>=300:raise RuntimeError('Diagnostic300query cap; not full-task acceptance')
  self.queries+=1;return self.inner(*args,**kwargs)
def cv2_images(images,image_size):
 array=np.stack([cv2.resize(im,(image_size[1],image_size[0]),interpolation=cv2.INTER_LINEAR) for im in images])
 return torch.from_numpy(array.astype('float32').transpose(0,3,1,2)/255.)
original=runner.preprocess_images;reports=[]
for name,resize in [('runtime_Torch',original),('training_CV2',cv2_images)]:
 runner.preprocess_images=resize
 result=runner.rollout(scene,BudgetPolicy(p),cp['stats'],scenario,out/name,runtime_profile='history_common_v1')
 trace=np.load(out/name/'rollout.npz');measure=json.loads((out/name/'measurements.json').read_text());tcp=np.array([r['tcp_xyz'] for r in measure])
 goal=np.r_[scene['pick_xy_m'],scene['table_top_z_m']+scene['grasp_tcp_height_m']+scene['lift_height_m']]
 if name=='runtime_Torch':assert np.max(abs(trace['raw_chunks'][0]-np.load(case/'rollout.npz')['raw_chunks'][0]))<1e-5
 record=dict(condition=name,queries=len(trace['raw_chunks']),active_points=result['active_policy_points'],
  endpoint_tcp_m=tcp[-1].tolist(),tcp_height_range_m=float(np.ptp(tcp[:,2])),
  endpoint_distance_to_approach_m=float(np.linalg.norm(tcp[-1]-goal)),simulated_events=result['simulated_events'],
  deliberate_diagnostic_cap=result['error'],full_task_test=False,optimizer_updates=0)
 reports.append(record);print(json.dumps(record),flush=True)
(out/'comparison.json').write_text(json.dumps(dict(records=reports,scope='matched600point diagnosis; only preprocessing changes, both model/physics/initial state fixed',training_process_changed=False,hardware_commands_sent=False),indent=2)+'\n')
