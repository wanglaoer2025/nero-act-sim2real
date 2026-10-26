"""Reconstruct matched-exposure ACT queries from the frozen successful center."""
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import torch
from common import ROOT,CHECKPOINT,CHECKPOINT_SHA,load_initial,historical
from executor.model import make_policy_observation,physical_actions
from simulation.scene import build_scene
from simulation.snapshot import restore
from simulation.nero_domain_randomization import NeroServo
from src.act.nero_dataset import preprocess_images


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    output=ROOT/'preservation';output.mkdir(exist_ok=False)
    source=ROOT.parent/'nero_act_contact_escape_fix_20261023/results/9000'
    scene=json.loads((source/'scene_configuration.json').read_text())
    result=json.loads((source/'result.json').read_text())
    cutoff=next(e['physics_step'] for e in result['simulated_events'] if e['kind']=='GRASP')
    records=json.loads((source/'visual_observations.json').read_text())
    snapshots=json.loads((source/'physical_snapshots.json').read_text())['snapshots']
    with np.load(source/'rollout.npz',allow_pickle=False) as f:
        indices=np.flatnonzero(f['query_steps']<cutoff)
        indices=indices[np.unique(np.linspace(0,len(indices)-1,min(96,len(indices))).round().astype(int))]
        reference=f['raw_chunks'][indices].copy();features=f['visual_features_m'][indices].copy()
    policy,stats,_=load_initial()
    model=build_scene(scene);data=mujoco.MjData(model);servo=NeroServo(model,data,scene)
    current=[];images=[];targets=[];error=0.
    hashes=historical.snapshot([*source.glob('*.json'),source/'rollout.npz',CHECKPOINT,
        Path(__file__),ROOT/'common.py',ROOT/'preservation.py'])
    with mujoco.Renderer(model,height=180,width=320) as renderer,torch.no_grad():
        for slot,index in enumerate(indices):
            restore(model,data,servo,snapshots[int(index)])
            renderer.update_scene(data,camera='calibrated_nero_camera')
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            rgb=renderer.render().copy();record=records[int(index)]
            observed=make_policy_observation(record['paired_feedback_state'],stats,policy.config,
                record['causal_previous_gripper_target_m'],geometry_features_m=features[slot])
            image=preprocess_images(rgb[None],policy.config['image_size'])[None]
            prediction=policy(torch.from_numpy(observed)[None].cuda(),image.cuda())
            physical=physical_actions(prediction,stats,bounded_gripper=True)[0].cpu().numpy()
            error=max(error,float(np.max(abs(physical-reference[slot]))))
            current.append(observed);images.append(image[0].numpy());targets.append(prediction[0].cpu().numpy())
    if error>1e-6:raise ValueError('Matched baseline inference did not reproduce original trajectory: '+str(error))
    path=output/'current_camera.npz'
    np.savez_compressed(path,current=np.asarray(current),image=np.asarray(images),target=np.asarray(targets),query_indices=indices)
    historical.require_unchanged(hashes)
    proof=dict(complete=True,samples=len(indices),initializer_sha256=CHECKPOINT_SHA,
        original_query_reproduction_max_abs_rad_m=error,source_sha256=hashes,
        replay_sha256=historical.snapshot([path]),matched_exposure_and_previous_command=True,
        pre_GRASP_only=True,training_only=True,hardware_deployment_allowed=False)
    (output/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if 'sha256' not in k}),flush=True)


if __name__=='__main__':main()
