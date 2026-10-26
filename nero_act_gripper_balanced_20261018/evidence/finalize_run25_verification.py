"""Read-only audit of completed offline evaluations; no training or execution."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from executor.evaluation_integrity import verify_frozen_artifacts
from simulation.scene import build_scene, tcp_in_link7
from training.pose_objective import SerialArmKinematics, fk_spec_from_scene

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    outcomes = {}
    for name, case in [('nominal', 'cup_+0_+0'), ('validation42', 'validation42_grasp')]:
        directory = ROOT / f'evaluation/run25_{name}'
        protocol = json.loads((directory / 'summary.json').read_text())
        verify_frozen_artifacts(ROOT / 'model/run25_balanced_gripper/policy.ckpt', protocol)
        result = json.loads((directory / case / 'result.json').read_text())
        outcomes[name] = {key: result[key] for key in [
            'success', 'physical_success', 'max_lift_m', 'maximum_horizontal_transfer_m',
            'stable_released_frames', 'unexpected_arm_cup_contact_frames',
            'execution_contract_failures', 'constraint_attachment_used', 'teacher_free',
            'real_hardware_tested', 'error', 'simulated_events']}
    directory = ROOT / 'evaluation/run25_nominal/cup_+0_+0'
    scene = json.loads((directory / 'scene_configuration.json').read_text())
    snapshots = json.loads((directory / 'physical_snapshots.json').read_text())
    rollout = np.load(directory / 'rollout.npz')
    joints = torch.tensor([snapshot['qpos'][:7] for snapshot in snapshots['snapshots']],
                          dtype=torch.float64)
    with torch.no_grad():
        position, rotation = SerialArmKinematics(fk_spec_from_scene(scene))(
            joints, return_rotation=True)
    position, rotation = position.numpy(), rotation.numpy()
    model = build_scene(scene)
    target_rotation = np.array(scene['ik_goal_rotation_link7']) @ tcp_in_link7(model)[1]
    angle = np.degrees(np.arccos(np.clip(
        (np.einsum('ij,nij->n', target_rotation, rotation) - 1) / 2, -1, 1)))
    index = int(np.argmin(abs(position[:, 2] - .055)))
    # Independently check the reported FK against the actual MuJoCo snapshot.
    data = mujoco.MjData(model)
    data.qpos[:] = snapshots['snapshots'][index]['qpos']
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_tcp')
    fk_error = float(np.max(abs(position[index] - data.site_xpos[site])))
    assert fk_error < 1e-10
    near = abs(position[:, 2] - .055) <= .002
    widths = rollout['raw_chunks'][:, :10, 7]
    diagnosis = dict(
        closest_grasp_query=dict(query=index, step=int(rollout['query_steps'][index]),
            tcp_m=position[index].tolist(), rotation_error_deg=float(angle[index]),
            source_xy_error_m=float(np.linalg.norm(position[index, :2] - scene['pick_xy_m'])),
            bounded_width_first10_mm=(widths[index] * 1000).tolist()),
        queries_in_unchanged_grasp_depth=int(near.sum()),
        bounded_width_min_in_depth_mm=float(widths[near].min() * 1000),
        bounded_width_max_in_depth_mm=float(widths[near].max() * 1000),
        fk_independent_mujoco_error_m=fk_error,
        recorded_chunks_are_bounded_physical_predictions=True)
    report = dict(
        stage='offline_verification_complete_original_scene_not_fixed',
        frozen_artifacts_and_sources_verified=True,
        frozen_adapter_audit=json.loads((ROOT / 'evidence/frozen_adapter_audit.json').read_text()),
        prior_gripper_fixed_input_report='evidence/fixed_grasp_readout_comparison.json',
        outcomes=outcomes, nominal_diagnosis=diagnosis,
        independent16_executed=False, hardware_connected=False,
        hardware_deployment_allowed=False,
        next_fix_requires_architectural_discussion=False)
    (ROOT / 'evidence/final_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(outcomes=outcomes, nominal_diagnosis=diagnosis), indent=2))


if __name__ == '__main__':
    main()
