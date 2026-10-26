"""Shared trained ACT output head for simulation, shadow and eventual execution."""
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'policy_source'))
from src.act.nero_policy import NeroActPolicy
from executor.contract import FullTaskContract,GRIPPER_CONTEXT,GRIPPER_CONTEXT_SCALE_M

FEEDBACK_BUDGETS = {'feedback_tanh_v1': (.025, .008),
                    'feedback_tanh_v2': (.0295, .0095)}


def make_policy_observation(state,stats,config,previous_gripper_target_m=None):
    state=np.asarray(state,np.float32)
    mean,std=[np.asarray(stats[k],np.float32) for k in ['qpos_mean','qpos_std']]
    if (state.shape!=(8,) or mean.shape!=(8,) or std.shape!=(8,)
            or not np.isfinite(np.r_[state,mean,std]).all() or np.any(std<=0)):
        raise ValueError('Finite measured8D state and paired physical statistics required')
    normalized=(state-mean)/std
    context=config.get('gripper_command_context')
    if not context:
        if config.get('state_dim',8)!=8 or config.get('policy_input_dim',8)!=8:raise ValueError('Missing command-context profile')
        return normalized
    if (context!=GRIPPER_CONTEXT or config.get('state_dim')!=8 or config.get('policy_input_dim')!=9
            or config.get('physical_state_dim')!=8 or previous_gripper_target_m is None
            or np.ndim(previous_gripper_target_m)!=0):
        raise ValueError('Explicit causal previous-gripper context required')
    previous=float(previous_gripper_target_m)
    if not np.isfinite(previous) or not 0<=previous<=.0950001:
        raise ValueError('Previous gripper target is not a valid physical command')
    return np.r_[normalized,(previous-float(state[7]))/GRIPPER_CONTEXT_SCALE_M].astype(np.float32)


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
        self.gripper_context=bool(config.get('gripper_command_context'))
        if self.gripper_context:
            FullTaskContract(config)
            if not config.get('bounded_feedback_residual_output') or not config.get('bounded_gripper_output'):
                raise ValueError('Command context requires the explicitly bounded full-task policy')
            if config.get('coordinate_conditioning') or config.get('task_displacement_features'):
                raise ValueError('Command context does not include task-coordinate conditioning')
        base_config=dict(config,state_dim=9,joint_residual_anchor=False) if self.gripper_context else config
        super().__init__(base_config)
        if self.gripper_context and config.get('joint_residual_anchor',False):
            scale,offset=[torch.tensor(config[k],dtype=torch.float32) for k in ['joint_anchor_scale','joint_anchor_offset']]
            if (scale.shape!=(7,) or offset.shape!=(7,) or not torch.isfinite(torch.cat([scale,offset])).all()
                    or not (scale>0).all()):raise ValueError('Paired seven-joint anchors required')
            self.joint_residual_anchor=True
            self.register_buffer('joint_anchor_scale',scale,persistent=False)
            self.register_buffer('joint_anchor_offset',offset,persistent=False)
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
            prediction = feedback_readout(prediction, qpos[:,:8], stats, self.feedback_lower, self.feedback_upper,
                                          version=self.feedback_version)
        elif self.bounded_gripper:
            prediction = bound_gripper(prediction, self.gripper_mean, self.gripper_std)
        return prediction
