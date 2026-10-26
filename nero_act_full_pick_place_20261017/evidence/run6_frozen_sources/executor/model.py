"""Shared trained ACT output head for simulation, shadow and eventual execution."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'policy_source'))
from src.act.nero_policy import NeroActPolicy


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


class BoundedActPolicy(NeroActPolicy):
    def __init__(self, config):
        super().__init__(config)
        self.bounded_gripper = bool(config.get('bounded_gripper_output', False))
        self.gripper_mean = float(config.get('gripper_action_mean', 0.))
        self.gripper_std = float(config.get('gripper_action_std', 1.))

    def forward(self, qpos, image):
        prediction = super().forward(qpos, image)
        if self.bounded_gripper:
            prediction = bound_gripper(prediction, self.gripper_mean, self.gripper_std)
        return prediction
