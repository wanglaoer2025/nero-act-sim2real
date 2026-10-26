import json
from pathlib import Path
import h5py,mujoco,numpy as np
import torch
torch.cuda.init()
from simulation.scene import build_scene
from simulation.runtime_checkpoint import restore_runtime
from simulation.nero_domain_randomization import NeroServo,disturbance_torque
from executor.ros_execution import RobotGeometry
from simulation.teacher import observe
ROOT=Path('/home/user/wq/nero_act_history_feedback_20261019');manifest=json.loads((ROOT/'training_data/history_recoveries_v3/manifest.json').read_text())
row=next(r for r in manifest['episodes'] if r['id']==2014);parent=next(r for r in manifest['base_episodes'] if r['id']==row['parent_episode_id'])
with h5py.File(parent['path'],'r') as f:scene=json.loads(f.attrs['scene_config'])
context=json.loads(Path(row['context_path']).read_text());records=[]
for reference in ['previous','measured']:
 model=build_scene(scene);data=mujoco.MjData(model);servo=NeroServo(model,data,scene)
 previous,control,_=restore_runtime(model,data,servo,context,expected_bindings=context['bindings'])
 geometry=RobotGeometry(scene);pose=np.asarray(control['support_pose']);target=previous.copy()
 if reference=='measured':target[:7]=data.qpos[:7]
 drifts=[]
 commands=np.load(ROOT/'results/mechanism_B/heldout_support_release/rollout.npz')['sent_targets']
 for i in range(len(commands)):
  target[7]=commands[i,7]
  data.qfrc_applied[:7]=disturbance_torque(scene,control['global_physics_points']+i);servo.command(target[:7],target[7])
  for _ in range(25):servo.apply();mujoco.mj_step(model,data)
  mujoco.mj_forward(model,data);actual=np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]]
  drifts.append(float(np.linalg.norm(geometry.pose(actual)['tcp']-pose)))
 records.append(dict(reference=reference,max_drift_mm=max(drifts)*1000,first_3mm_violation=next((i+1 for i,x in enumerate(drifts) if x>.003),None),trace_m=drifts))
 print(records[-1],flush=True)
Path('/home/user/wq/nero_act_contact_transition_fix_20261020/evidence/support_release_command_probe.json').write_text(json.dumps(dict(complete=True,no_ACT_predictions=True,records=records),indent=2)+'\n')
