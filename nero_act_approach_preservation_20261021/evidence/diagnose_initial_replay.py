import json,pickle
from pathlib import Path
import torch,numpy as np
from training.history import initialize_history,build_history_policy
from training.approach_preservation import ApproachPreservation
root=Path('.').resolve();torch.set_num_threads(2);torch.manual_seed(42)
source_path=root.parent/'nero_act_visual_gripper_fix_20261019/training/gripper_force_building_v1/checkpoints/policy_best_val.ckpt'
payload=torch.load(source_path,map_location='cpu',weights_only=False)
old=pickle.loads((source_path.parent/'dataset_stats.pkl').read_bytes());stats=pickle.loads((root/'training_data/approach_complete_v1/dataset_stats.pkl').read_bytes())
settings=json.loads((root/'config/history_experiment.json').read_text());settings.update(device='cuda',pretrained_backbone=False,variant='A',stage='pilot',num_epochs=0,training_budget={'kind':'updates','limit':1000},pose_loss_weight=0.)
c,w=initialize_history(payload,old,stats,settings,variant='A');p=build_history_policy(c).cuda().eval();p.load_state_dict(w)
keeper=ApproachPreservation(settings['approach_preservation'],stats,'cuda')
report=keeper.assess(p);report['first_queries']=[]
with torch.no_grad():
 for i in [0,50,200,380,422]:
  error=(p(keeper.current[i:i+1],keeper.image[i:i+1])-keeper.target[i:i+1])*torch.tensor(stats['action_std'],device='cuda')
  report['first_queries'].append(dict(index=i,arm_max=float(error[:,:2,:7].abs().max()),grip_max=float(error[:,:2,7].abs().max()),arm_delta=error[0,0,:7].cpu().tolist()))
(root/'evidence/initial_replay_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
import mujoco
from simulation.scene import build_scene
from src.act.nero_dataset import preprocess_images
source=root.parent/'nero_act_restricted_heads_20261020/results/nominal_before/nominal_development'
scene=json.loads((source/'scene_configuration.json').read_text());snapshots=json.loads((source/'physical_snapshots.json').read_text())['snapshots']
model=build_scene(scene);data=mujoco.MjData(model);pixels=[]
with mujoco.Renderer(model,height=180,width=320) as renderer:
 for i in [0,50,200,380,422]:
  saved=snapshots[i];data.qpos[:]=saved['qpos'];data.qvel[:]=saved['qvel'];mujoco.mj_forward(model,data)
  renderer.update_scene(data,camera='calibrated_nero_camera');renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
  image=preprocess_images(renderer.render().copy()[None],[180,320])[None].cuda()
  with torch.no_grad():err=(p(keeper.current[i:i+1],image)-keeper.target[i:i+1])*torch.tensor(stats['action_std'],device='cuda')
  pixels.append(dict(index=i,max_pixel_difference=float((image-keeper.image[i:i+1]).abs().max()),host_render_arm_max=float(err[:,:2,:7].abs().max())))
report['host_render_comparison']=pixels
(root/'evidence/initial_replay_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(pixels),flush=True)
