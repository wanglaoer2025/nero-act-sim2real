"""Shared trained ACT output head for simulation, shadow and eventual execution."""
from pathlib import Path
import math
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'policy_source'))
from src.act.nero_policy import NeroActPolicy
from executor.contract import FullTaskContract,GRIPPER_CONTEXT,GRIPPER_CONTEXT_SCALE_M,VISUAL_PROFILE

FEEDBACK_BUDGETS = {'feedback_tanh_v1': (.025, .008),
                    'feedback_tanh_v2': (.0295, .0095)}


def make_gripper_adapter(hidden_dim):
    # Fixed construction and a zero residual preserve the pretrained policy.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        head = torch.nn.Sequential(torch.nn.Linear(hidden_dim, 64),
                                   torch.nn.SiLU(), torch.nn.Linear(64, 1))
        torch.nn.init.zeros_(head[2].weight)
        torch.nn.init.zeros_(head[2].bias)
    return head


def make_policy_observation(state,stats,config,previous_gripper_target_m=None,*,geometry_features_m=None):
    state=np.asarray(state,np.float32)
    mean,std=[np.asarray(stats[k],np.float32) for k in ['qpos_mean','qpos_std']]
    if (state.shape!=(8,) or mean.shape!=(8,) or std.shape!=(8,)
            or not np.isfinite(np.r_[state,mean,std]).all() or np.any(std<=0)):
        raise ValueError('Finite measured8D state and paired physical statistics required')
    normalized=(state-mean)/std
    context=config.get('gripper_command_context')
    visual=config.get('visual_observation_profile')
    if visual not in (None,VISUAL_PROFILE):raise ValueError('Unknown visual observation profile')
    if not context:
        if visual or config.get('state_dim',8)!=8 or config.get('policy_input_dim',8)!=8:raise ValueError('Missing command-context profile')
        return normalized
    if (context!=GRIPPER_CONTEXT or config.get('state_dim')!=8 or config.get('policy_input_dim')!=(15 if visual else 9)
            or config.get('physical_state_dim')!=8 or previous_gripper_target_m is None
            or np.ndim(previous_gripper_target_m)!=0):
        raise ValueError('Explicit causal previous-gripper context required')
    previous=float(previous_gripper_target_m)
    if not np.isfinite(previous) or not 0<=previous<=.0950001:
        raise ValueError('Previous gripper target is not a valid physical command')
    observed=np.r_[normalized,(previous-float(state[7]))/GRIPPER_CONTEXT_SCALE_M]
    if visual:
        try:
            features,mean,std=[np.asarray(x,np.float32) for x in [geometry_features_m,stats['geometry_mean'],stats['geometry_std']]]
        except (KeyError,TypeError,ValueError) as exc:raise ValueError('Explicit paired metric visual features required') from exc
        if any(x.shape!=(6,) or not np.isfinite(x).all() for x in [features,mean,std]) or np.any(std<=0):
            raise ValueError('Explicit finite6D visual features and paired positive statistics required')
        observed=np.r_[observed,(features-mean)/std]
    elif geometry_features_m is not None:
        raise ValueError('Legacy policy cannot silently ignore visual conditions')
    return observed.astype(np.float32)


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


def feedback_preimage(target, qpos, stats, lower, upper, *, version='feedback_tanh_v1'):
    """Train pre-activation targets in the inverse of the deployed readout."""
    state=qpos*stats['qpos_std']+stats['qpos_mean']
    absolute=target*stats['action_std']+stats['action_mean']
    delta=absolute-state[:,None]
    arm_budget,width_budget=FEEDBACK_BUDGETS[version]
    prefix=target.new_tensor([arm_budget]*7+[width_budget]).expand(10,-1)
    budget=torch.cat([prefix,prefix*2])[None]
    positive=torch.minimum((upper-state)[:,None].clamp_min(0),budget)
    negative=torch.minimum((state-lower)[:,None].clamp_min(0),budget)
    span=torch.where(delta>=0,positive,negative)
    epsilon=torch.finfo(target.dtype).eps
    ratio=(delta/span.clamp_min(epsilon)).clamp(-1+epsilon,1-epsilon)
    raw=state[:,None]+span*torch.atanh(ratio)
    return (raw-stats['action_mean'])/stats['action_std']


def verify_paired_readout_statistics(config, stats):
    if config.get('visual_observation_profile'):
        for key in ['geometry_mean','geometry_std']:
            try:a,b=[torch.as_tensor(v,dtype=torch.float32) for v in [config[key],stats[key]]]
            except (KeyError,TypeError,ValueError) as exc:raise ValueError('Paired visual normalization required') from exc
            if (a.shape!=(6,) or b.shape!=(6,) or not torch.isfinite(a).all()
                    or not torch.equal(a,b) or (key.endswith('std') and not (a>0).all())):
                raise ValueError('Visual normalization differs from paired configuration')
    if config.get('bounded_feedback_residual_output'):
        for key in ['qpos_mean', 'qpos_std', 'action_mean', 'action_std']:
            if not torch.equal(torch.as_tensor(config['constraint_loss_stats'][key], dtype=torch.float64),
                               torch.as_tensor(stats[key], dtype=torch.float64)):
                raise ValueError('Feedback readout and external statistics differ: '+key)


class BoundedActPolicy(NeroActPolicy):
    def __init__(self, config):
        self.gripper_context=bool(config.get('gripper_command_context'))
        self.visual_conditioning=config.get('visual_observation_profile')==VISUAL_PROFILE
        if self.gripper_context:
            FullTaskContract(config)
            if not config.get('bounded_feedback_residual_output') or not config.get('bounded_gripper_output'):
                raise ValueError('Command context requires the explicitly bounded full-task policy')
            if config.get('coordinate_conditioning') or config.get('task_displacement_features'):
                raise ValueError('Command context does not include task-coordinate conditioning')
        base_config=dict(config,state_dim=15 if self.visual_conditioning else 9,joint_residual_anchor=False) if self.gripper_context else config
        super().__init__(base_config)
        if config.get('nonlinear_gripper_head', False):
            if int(config['action_dim']) != 8:
                raise ValueError('Nonlinear gripper branch requires eight physical actions')
            rate = float(config.get('gripper_adapter_lr', config['lr']))
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError('Finite positive gripper adapter learning rate required')
            self.gripper_adapter = make_gripper_adapter(int(config['hidden_dim'])).to(
                self.model.action_head.weight.device)
            self.model.action_head.register_forward_hook(self._adapt_gripper)
            self.optimizer.add_param_group(dict(params=list(self.gripper_adapter.parameters()),
                                                lr=rate, weight_decay=0.))
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

    def _adapt_gripper(self, module, inputs, output):
        residual = self.gripper_adapter(inputs[0])
        return torch.cat([output[..., :7], output[..., 7:8] + residual], dim=-1)

    def forward(self, qpos, image, actions=None, is_pad=None, *, return_unbounded=False, history=None):
        posterior = (super().forward(qpos, image, actions, is_pad,history=history)
                     if actions is not None else None)
        prediction = posterior['a_hat'] if posterior is not None else super().forward(qpos, image,history=history)
        unbounded = prediction
        if self.feedback_bounded:
            stats = {key: getattr(self, 'feedback_'+key) for key in [
                'qpos_mean', 'qpos_std', 'action_mean', 'action_std']}
            prediction = feedback_readout(prediction, qpos[:,:8], stats, self.feedback_lower, self.feedback_upper,
                                          version=self.feedback_version)
        elif self.bounded_gripper:
            prediction = bound_gripper(prediction, self.gripper_mean, self.gripper_std)
        return (dict(a_hat=prediction, kl=posterior['kl'] if posterior is not None else prediction.sum()*0,
                     unbounded_a_hat=unbounded)
                if posterior is not None or return_unbounded else prediction)
