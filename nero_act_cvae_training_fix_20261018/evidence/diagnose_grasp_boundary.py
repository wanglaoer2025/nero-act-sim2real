"""Offline diagnosis at the measured ACT grasp-height snapshot; no training/export."""
import copy
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
import torch

from executor.model import BoundedActPolicy, make_policy_observation, physical_actions
from simulation.nero_domain_randomization import NeroServo
from simulation.scene import build_scene, tcp_in_link7
from simulation.snapshot import restore
from simulation.teacher import run_teacher
from src.act.nero_dataset import preprocess_images

ROOT = Path(__file__).resolve().parents[1]


def main():
    origin = ROOT/'evaluation/run21_nominal/cup_+0_+0'
    scene = json.loads((origin/'scene_configuration.json').read_text())
    record = json.loads((origin/'physical_snapshots.json').read_text())
    model = build_scene(scene)
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_tcp')
    heights = []
    for snapshot in record['snapshots']:
        data.qpos[:] = snapshot['qpos']
        mujoco.mj_forward(model, data)
        heights.append(float(data.site_xpos[site, 2]))
    index = int(np.argmin(abs(np.array(heights)-.055)))
    snapshot = record['snapshots'][index]
    servo = NeroServo(model, data, scene)
    previous = restore(model, data, servo, snapshot)
    target_rotation = np.array(scene['ik_goal_rotation_link7'])@tcp_in_link7(model)[1]
    rotation = data.site_xmat[site].reshape(3, 3)
    angle = np.rad2deg(np.arccos(np.clip((np.trace(target_rotation.T@rotation)-1)/2, -1, 1)))
    result = dict(query=index, step=record['query_steps'][index], height_m=heights[index],
                  xy_error_m=float(np.linalg.norm(data.site_xpos[site, :2]-scene['pick_xy_m'])),
                  rotation_error_deg=float(angle), hardware_connected=False,
                  teacher_is_diagnostic_only=True)
    correction = copy.deepcopy(scene)
    correction.update(continuation='grasp_to_retreat', initial_simulation_snapshot=snapshot,
        initial_robot_state=dict(qpos_rad=snapshot['qpos'][:7],
            gripper_total_width_m=snapshot['qpos'][7]-snapshot['qpos'][8]),
        joint_speed_limit_rad_s=.02, teacher_command_lead_limit_rad=.015,
        gripper_speed_limit_m_s=.004, teacher_gripper_feedback_lead_m=.008)
    correction['dwell_seconds'].update(close=8, release=8)
    teacher = run_teacher(correction)
    trace = teacher.pop('trace')
    result['direct_grasp_teacher_result'] = teacher
    if teacher['success']:
        directory = ROOT/'model/run21_cvae_posterior'
        payload = torch.load(directory/'policy.ckpt', map_location='cpu', weights_only=False)
        config = dict(payload['config'], device='cuda', pretrained_backbone=False)
        with (directory/'dataset_stats.pkl').open('rb') as stream:
            stats = pickle.load(stream)
        torch.set_num_threads(2)
        policy = BoundedActPolicy(config).cuda().eval()
        policy.load_state_dict(payload['model'], strict=True)
        with mujoco.Renderer(model, height=90, width=160) as renderer:
            renderer.update_scene(data, camera='calibrated_nero_camera')
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            image = renderer.render().copy()
        state = np.r_[data.qpos[:7], data.qpos[7]-data.qpos[8]]
        q = torch.from_numpy(make_policy_observation(state, stats, config, previous[7]))[None].cuda()
        image = preprocess_images(image[None], (90, 160))[None].cuda()
        captured = []
        hook = policy.model.action_head.register_forward_hook(lambda module, args, output: captured.append(output))
        prediction = policy(q, image)
        hook.remove()
        physical = physical_actions(prediction, stats, bounded_gripper=True)
        target = prediction.new_tensor([row['action'][7] for row in trace[:20]])[None]
        loss = ((physical[..., 7]-target)/stats['action_std'][7]).abs().mean()
        grad = torch.autograd.grad(loss, captured[0], retain_graph=True)[0][..., 7]
        latent = captured[0][..., 7]*stats['action_std'][7]+stats['action_mean'][7]
        raw_loss = ((latent-target)/stats['action_std'][7]).abs().mean()
        raw_grad = torch.autograd.grad(raw_loss, captured[0])[0][..., 7]
        result['actual_policy_input_gradient'] = dict(
            raw_width_range_m=[float(v) for v in [latent.min().detach(), latent.max().detach()]],
            bounded_width_range_m=[float(v) for v in [physical[..., 7].min().detach(), physical[..., 7].max().detach()]],
            verified_teacher_target_range_m=[float(v) for v in [target.min(), target.max()]],
            bounded_loss_gripper_gradient_max=float(grad.abs().max()),
            zero_gripper_gradient_points=int((grad == 0).sum()),
            preactivation_loss_gripper_gradient_max=float(raw_grad.abs().max()))
    (ROOT/'evidence/grasp_boundary_diagnosis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
