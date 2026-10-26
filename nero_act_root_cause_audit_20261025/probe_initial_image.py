"""Read-only first-query probe: recorded runtime state, two RGB sources, frozen CPU ACT."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import mujoco
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent
loader_root=WORKSPACE/'nero_act_real_visual_bridge_20261024'
sys.path.insert(0,str(loader_root))
spec=importlib.util.spec_from_file_location('_root_cause_cpu_loader',loader_root/'experiment.py')
loader=importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    policy,stats=loader.load_baseline()
    before={k:v.clone() for k,v in policy.state_dict().items()}
    from executor.model import make_policy_observation,physical_actions
    from src.act.nero_dataset import preprocess_images
    from simulation.scene import build_scene
    old=WORKSPACE/'nero_act_contact_escape_fix_20261023/results/9000'
    current=WORKSPACE/'nero_act_full_acceptance_20261024/run_01/9000'
    archive=WORKSPACE/'nero_act_new_handeye_policy_repair_20261023/data/visual/episode_9000.hdf5'
    scene=json.loads((old/'scene_configuration.json').read_text())
    snapshot=json.loads((old/'physical_snapshots.json').read_text())['snapshots'][0]
    current_snapshot=json.loads((current/'physical_snapshots.json').read_text())['snapshots'][0]
    a=np.load(old/'rollout.npz');b=np.load(current/'rollout.npz')
    observed=a['feedback_at_query'][0];features=a['visual_features_m'][0]
    previous=json.loads((old/'visual_observations.json').read_text())[0]['causal_previous_gripper_target_m']
    numeric=make_policy_observation(observed,stats,policy.config,previous,geometry_features_m=features)
    model=build_scene(scene);data=mujoco.MjData(model)
    data.qpos[:]=snapshot['qpos'];data.qvel[:]=snapshot['qvel']
    mujoco.mj_forward(model,data)
    with mujoco.Renderer(model,height=180,width=320) as renderer:
        renderer.update_scene(data,camera='calibrated_nero_camera')
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        rendered=renderer.render().copy()
    with h5py.File(archive) as f:
        archived=f['observations/images/camera_0'][0]
        teacher_qpos=f['observations/qpos'][0]
        teacher_features=f['observations/visual_relative_m'][0]
    predictions={}
    for name,rgb in [('mesa_recorded_runtime_state',rendered),('archived_teacher_frame0',archived)]:
        with torch.inference_mode():
            normalized=policy(torch.from_numpy(numeric)[None],preprocess_images(rgb[None],policy.config['image_size'])[None])
            predictions[name]=physical_actions(normalized[0],stats,bounded_gripper=policy.bounded_gripper).numpy()
    def compare(value,reference):
        delta=np.abs(value-reference)
        return dict(arm_all20_max_rad=float(delta[:,:7].max()),arm_prefix2_max_rad=float(delta[:2,:7].max()),gripper_all20_max_m=float(delta[:,7].max()))
    difference=np.abs(rendered.astype(np.int16)-archived.astype(np.int16))
    result=dict(scope='read_only_two_RGB_probe_not_archived_runtime_image_recovery',
        same_numeric_input_for_both_CPU_probes=True,
        old_and_current_first_snapshot_qpos_equal=np.array_equal(snapshot['qpos'],current_snapshot['qpos']),
        archived_teacher_joint_state_max_difference_rad=float(np.abs(teacher_qpos-observed[:7]).max()),
        archived_teacher_position_features_exactly_match=np.array_equal(teacher_features,features),
        RGB_sources_equal=bool(np.array_equal(rendered,archived)),
        RGB_mean_absolute_channel_difference=float(difference.mean()),
        RGB_different_pixel_fraction=float(np.any(difference>0,axis=2).mean()),
        comparisons=dict(
            CPU_Mesa_to_recorded_current_CPU=compare(predictions['mesa_recorded_runtime_state'],b['raw_chunks'][0]),
            CPU_archived_teacher_RGB_to_historical_GPU_runtime=compare(predictions['archived_teacher_frame0'],a['raw_chunks'][0]),
            CPU_Mesa_to_historical_GPU_runtime=compare(predictions['mesa_recorded_runtime_state'],a['raw_chunks'][0]),
            CPU_different_RGB_same_numeric=compare(predictions['mesa_recorded_runtime_state'],predictions['archived_teacher_frame0'])),
        limitations=['Archived teacher frame0 is not proven to be the historical runtime RGB; cannot identify sole cause of3/7 to1/7.',
                    'CPU versus historical GPU cannot be fully isolated at the runtime first state without the original runtime RGB.'],
        parameter_or_buffer_changes=[k for k,v in policy.state_dict().items() if not torch.equal(v,before[k])],
        optimizer_updates=0,hardware_commands_sent=False,
        source_sha256={str(p):digest(p) for p in [Path(__file__),loader_root/'experiment.py',loader_root/'bridge.py',loader.CHECKPOINT,
            archive,old/'rollout.npz',old/'scene_configuration.json',old/'physical_snapshots.json',current/'rollout.npz']})
    (ROOT/'initial_image_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
