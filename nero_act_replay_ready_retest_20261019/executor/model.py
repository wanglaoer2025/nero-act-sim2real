"""Shared trained ACT output head for simulation, shadow and eventual execution."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'policy_source'))
from src.act.nero_policy import NeroActPolicy
from executor.contract import FullTaskContract

FEEDBACK_BUDGETS = {'feedback_tanh_v1': (.025, .008),
                    'feedback_tanh_v2': (.0295, .0095)}


def bound_gripper(normalized, mean, std):
    if not 0 < std or not torch.isfinite(torch.tensor([mean, std])).all():
        raise ValueError('Finite paired gripper statistics required')
    grip = ((normalized[..., 7:8]*std+mean).clamp(0., .095)-mean)/std
    return torch.cat([normalized[..., :7], grip], dim=-1)


def physical_actions(normalized, stats, *, bounded_gripper):
    mean = torch.as_tensor(stats['action_mean'], dtype=normalized.dtype, device=normalized.device)
    std = torch.as_tensor(stats['action_std'], dtype=normalized.dtype, device=normalized.device)
    actions = normalized*std+mean
    if bounded_gripper:
        # Repeat the physical-domain activation after affine roundoff only.
        actions = torch.cat([actions[..., :7], actions[..., 7:8].clamp(0., .095)], dim=-1)
    return actions


def feedback_readout(prediction, qpos, stats, lower, upper, *, version='feedback_tanh_v1'):
    """Trained absolute-target activation, anchored to the SAME network input."""
    state = qpos*stats['qpos_std']+stats['qpos_mean']
    absolute = prediction*stats['action_std']+stats['action_mean']
    delta = absolute-state[:, None]
    arm_budget, width_budget = FEEDBACK_BUDGETS[version]
    prefix = prediction.new_tensor([arm_budget]*7+[width_budget]).expand(10, -1)
    budget = torch.cat([prefix, prefix*2])[None]
    positive = torch.minimum((upper-state)[:, None].clamp_min(0), budget)
    negative = torch.minimum((state-lower)[:, None].clamp_min(0), budget)
    span = torch.where(delta >= 0, positive, negative)
    output = state[:, None]+span*torch.tanh(delta/span.clamp_min(torch.finfo(delta.dtype).eps))
    return (output-stats['action_mean'])/stats['action_std']


def verify_paired_readout_statistics(config, stats):
    if config.get('bounded_feedback_residual_output'):
        for key in ['qpos_mean', 'qpos_std', 'action_mean', 'action_std']:
            if not torch.equal(torch.as_tensor(config['constraint_loss_stats'][key], dtype=torch.float64),
                               torch.as_tensor(stats[key], dtype=torch.float64)):
                raise ValueError('Feedback readout and external statistics differ: '+key)


class BoundedActPolicy(NeroActPolicy):
    def __init__(self, config):
        super().__init__(config)
        self.bounded_gripper = bool(config.get('bounded_gripper_output', False))
        self.gripper_mean = float(config.get('gripper_action_mean', 0.))
        self.gripper_std = float(config.get('gripper_action_std', 1.))
        self.feedback_bounded = bool(config.get('bounded_feedback_residual_output', False))
        self.feedback_version = config.get('feedback_readout_version')
        if self.feedback_bounded:
            if (self.feedback_version not in FEEDBACK_BUDGETS
                    or config.get('action_representation') != 'absolute_joint_position'):
                raise ValueError('Explicit trained feedback readout version and absolute targets required')
            contract = FullTaskContract(config)
            for key in ['qpos_mean', 'qpos_std', 'action_mean', 'action_std']:
                value = torch.as_tensor(config['constraint_loss_stats'][key], dtype=torch.float32)
                if value.shape != (8,) or not torch.isfinite(value).all() or (key.endswith('std') and not (value > 0).all()):
                    raise ValueError('Finite paired eight-dimensional readout statistics required')
                self.register_buffer('feedback_'+key, value, persistent=False)
            self.register_buffer('feedback_lower', torch.tensor([*contract.lower, 0.], dtype=torch.float32), persistent=False)
            self.register_buffer('feedback_upper', torch.tensor([*contract.upper, .095], dtype=torch.float32), persistent=False)

    def forward(self, qpos, image):
        prediction = super().forward(qpos, image)
        if self.feedback_bounded:
            stats = {key: getattr(self, 'feedback_'+key) for key in [
                'qpos_mean', 'qpos_std', 'action_mean', 'action_std']}
            prediction = feedback_readout(prediction, qpos, stats, self.feedback_lower, self.feedback_upper,
                                          version=self.feedback_version)
        elif self.bounded_gripper:
            prediction = bound_gripper(prediction, self.gripper_mean, self.gripper_std)
        return prediction
