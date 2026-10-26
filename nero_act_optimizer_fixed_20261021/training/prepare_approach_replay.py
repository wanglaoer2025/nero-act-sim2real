"""Re-render frozen baseline approach observations for offline action rehearsal."""
import hashlib
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np

from executor.model import make_policy_observation
from simulation.scene import build_scene
from simulation.visual_evaluation import require_files
from src.act.nero_dataset import preprocess_images

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'nero_act_restricted_heads_20261020'


def main():
    settings=json.loads((ROOT/'config/history_experiment.json').read_text())
    output=Path(settings['approach_preservation']['replay_path'])
    if output.exists() or output.with_suffix('.json').exists():raise FileExistsError(output)
    reference=PARENT/'results/nominal_before'
    bindings=json.loads((reference/'bindings.json').read_text());require_files(bindings)
    summary=json.loads((reference/'summary.json').read_text())
    if summary['optimizer_updates']!=0:raise ValueError('Zero-update baseline required')
    grasp=next(e for e in summary['result']['simulated_events'] if e['kind']=='GRASP')
    source=reference/'nominal_development'
    snapshots=json.loads((source/'physical_snapshots.json').read_text())
    records=json.loads((source/'visual_observations.json').read_text())
    stats=pickle.loads((ROOT/'training_data/approach_complete_v1/dataset_stats.pkl').read_bytes())
    config=json.loads((PARENT/'training/pilot_A/config.json').read_text())
    scene=json.loads((source/'scene_configuration.json').read_text())
    model=build_scene(scene);data=mujoco.MjData(model)
    with np.load(source/'rollout.npz',allow_pickle=False) as trace:
        steps=trace['query_steps'];features=trace['visual_features_m'];raw=trace['raw_chunks']
        selected=np.flatnonzero(steps<int(grasp['physics_step']))
        if len(steps)!=len(records) or steps.tolist()!=snapshots['query_steps'] or not len(selected):
            raise ValueError('Complete aligned approach query records required')
        current=[];images=[];targets=[]
        with mujoco.Renderer(model,height=180,width=320) as renderer:
            for i in selected:
                saved=snapshots['snapshots'][int(i)];record=records[int(i)]
                data.qpos[:]=saved['qpos'];data.qvel[:]=saved['qvel']
                mujoco.mj_forward(model,data)
                measured=np.asarray(record['paired_feedback_state'],np.float32)
                if np.max(np.abs(data.qpos[:7]-measured[:7]))>1e-6:
                    raise ValueError('Baseline exposure and physical snapshot disagree')
                renderer.update_scene(data,camera='calibrated_nero_camera')
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                image=renderer.render().copy()
                current.append(make_policy_observation(measured,stats,config,
                    record['causal_previous_gripper_target_m'],geometry_features_m=features[i]))
                images.append(preprocess_images(image[None],[180,320]).numpy())
                targets.append((raw[i]-stats['action_mean'])/stats['action_std'])
    inputs=[source/name for name in ['rollout.npz','physical_snapshots.json','visual_observations.json','scene_configuration.json']]
    inputs += [reference/'summary.json',reference/'bindings.json',PARENT/'training/pilot_A/config.json',
               ROOT/'training_data/approach_complete_v1/dataset_stats.pkl',Path(__file__)]
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source_hashes=dict(bindings,**{str(p.resolve()):sha(p) for p in inputs})
    require_files(bindings);require_files(source_hashes)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,current=np.asarray(current,np.float32),image=np.asarray(images,np.float32),target=np.asarray(targets,np.float32))
    proof=dict(complete=True,profile='baseline_prefix_v1',samples=len(current),query_indices=selected.tolist(),
        zero_update_reference=True,reference_optimizer_updates=0,cutoff_before_GRASP_physics_step=grasp['physics_step'],
        reference_summary_path=str((reference/'summary.json').resolve()),reference_trace_path=str((source/'rollout.npz').resolve()),
        source_sha256=source_hashes,replay_sha256=sha(output),
        assessment_role='nominal_development_rehearsal_not_independent_acceptance',hardware_commands_sent=False)
    output.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(complete=True,samples=len(current),replay_sha256=sha(output))),flush=True)


if __name__=='__main__':main()
