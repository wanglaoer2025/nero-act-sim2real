"""Zero-update probes of saved failing20K query contexts, re-rendered poses."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json
from pathlib import Path
import numpy as np
import torch,mujoco
from experiment import BASE,ArmPolicy
from support import save_json,digest_file
from executor.model import make_policy_observation,physical_actions
from src.act.nero_dataset import preprocess_images
from simulation.scene import build_scene

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parent
cp=torch.load(BASE/'training/formal/policy_20000.ckpt',map_location='cpu',weights_only=False)
delta=json.loads((ROOT/'delta_stats.json').read_text())
a=ArmPolicy(cp['stats'],delta,variant='A',pretrained=False).cuda().eval();a.load_source(cp)
b=ArmPolicy(cp['stats'],delta,variant='B',pretrained=False).cuda().eval();b.load_source(cp)
case=BASE/'training/formal/dev_20000/centre_0'
trace=np.load(case/'rollout.npz');scene=json.loads((case/'scene_configuration.json').read_text())
snapshots=json.loads((case/'physical_snapshots.json').read_text())['snapshots']
observations=json.loads((case/'visual_observations.json').read_text())
model=build_scene(scene);data=mujoco.MjData(model)
model.vis.global_.offheight=720;model.vis.global_.offwidth=1280
records=[]
with mujoco.Renderer(model,height=720,width=1280) as renderer:
    for i in [0,len(snapshots)//2,len(snapshots)-1]:
        snap=snapshots[i];data.qpos[:]=snap['qpos'];data.qvel[:]=snap['qvel']
        data.time=trace['query_simulation_time_s'][i];mujoco.mj_forward(model,data)
        renderer.update_scene(data,camera='calibrated_nero_camera');renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        rgb=renderer.render().copy();record=observations[i]
        obs=make_policy_observation(np.array(record['paired_feedback_state']),cp['stats'],cp['config'],
            record['causal_previous_gripper_target_m'],geometry_features_m=trace['visual_features_m'][i])
        tensor=torch.from_numpy(obs)[None].cuda();image=preprocess_images(rgb[None],[180,320])[None].cuda()
        with torch.inference_mode():
            source=a.source_prediction(tensor,image);pa=a(tensor,image);pb=b(tensor,image)
        errors=[float(((p[:,:,:7]-source[:,:,:7])*a.action_std[:7]).abs().max()) for p in [pa,pb]]
        grip=max(float((p[:,:,7]-source[:,:,7]).abs().max()) for p in [pa,pb])
        assert max(errors)<=1e-6 and grip==0.
        physical=physical_actions(source[0],cp['stats'],bounded_gripper=True).cpu().numpy()
        records.append(dict(query=i,maximum_physical_equivalence_error_rad=max(errors),gripper_error=grip,
            original_raw_replay_max_error=float(abs(physical-trace['raw_chunks'][i]).max()),
            exposure_paired_state_vs_physics_snapshot_error_rad=float(abs(np.array(record['paired_feedback_state'])[:7]-np.array(snap['qpos'])[:7]).max())))
save_json(ROOT/'preflight/actual_inputs.json',dict(complete=True,optimizer_updates=0,records=records,
    image_source='rerendered_saved_query_physics_snapshot_with_saved_visual_estimate_and_paired_feedback',
    scene_sha256=digest_file(case/'scene_configuration.json'),geometry_truth_policy_input=False,hardware_commands_sent=False))
print(json.dumps(records),flush=True)
