"""Offline-only supervision and loss scaling; no runtime motion generation."""
import math

import numpy as np
import torch


def pregrasp_goal(tcp, cup_bottom):
    tcp, cup = np.asarray(tcp, float), np.asarray(cup_bottom, float)
    if tcp.shape != (3,) or cup.shape != (3,) or not np.isfinite(np.r_[tcp, cup]).all():
        raise ValueError('Finite current TCP and cup bottom in the same frame required')
    error = float(np.linalg.norm(tcp[:2]-cup[:2]))
    if error > .012 and tcp[2] < cup[2]+.20:
        return np.r_[tcp[:2], cup[2]+.255], 'clearance'
    if error > .006:
        return np.r_[cup[:2], min(max(tcp[2], cup[2]+.205), cup[2]+.30)], 'align'
    return np.r_[cup[:2], cup[2]+.055], 'descend'


def preservation_scale(task_loss, preservation_loss, shared_parameters, *, maximum=.1, ratio=.5):
    if not math.isfinite(maximum) or not math.isfinite(ratio) or maximum <= 0 or not 0 < ratio <= 1:
        raise ValueError('Finite positive maximum and gradient fraction required')
    parameters = list(shared_parameters)
    if not parameters:
        raise ValueError('Shared position parameters are required')
    norms = []
    for loss in [task_loss, preservation_loss]:
        if loss.ndim or not torch.isfinite(loss):
            raise ValueError('Finite scalar objectives required')
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        squared = sum(float(g.detach().square().sum()) for g in gradients if g is not None)
        norms.append(math.sqrt(squared))
    if not all(math.isfinite(n) for n in norms):
        raise ValueError('Nonfinite objective gradient')
    return min(maximum, ratio*norms[0]/max(norms[1], 1e-12)), dict(task_gradient_norm=norms[0],
        preservation_gradient_norm=norms[1])


def executed_objective(prediction, target, padding, stats, pose_loss):
    if (prediction.shape != target.shape or prediction.ndim != 3
            or prediction.shape[1:] != (20, 8) or padding.shape != prediction.shape[:2]
            or padding.dtype != torch.bool or not (~padding[:, :2]).any()
            or not torch.isfinite(prediction).all() or not torch.isfinite(target[~padding]).all()):
        raise ValueError('Finite matched20x8 actions and nonempty executable prefix required')
    scale = torch.as_tensor(stats['action_std'], device=prediction.device, dtype=prediction.dtype)
    valid = ~padding[:, :2]
    error = (prediction[:, :2]-target[:, :2])*scale
    arm = torch.nn.functional.smooth_l1_loss(error[..., :7]/.0005,
        torch.zeros_like(error[..., :7]), reduction='none')
    arm = (arm*valid[..., None]).sum()/(7*valid.sum()).clamp_min(1)
    grip = ((error[..., 7]/.0002).square()*valid).sum()/valid.sum().clamp_min(1)
    executed_padding = padding.clone(); executed_padding[:, 2:] = True
    position, rotation = pose_loss(prediction, target, executed_padding)
    return arm+grip+100*position+rotation, dict(arm=arm, gripper=grip,
        position_100um=100*position, rotation_degree=rotation)
