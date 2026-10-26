"""Bounded offline seven-row output tuning; the ACT interface is unchanged."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
INCREMENT = ROOT.parent / 'nero_act_command_increment_20261021'
sys.path.append(str(INCREMENT))


def configure_arm_tuning(policy):
    policy.requires_grad_(False)
    head = policy.base.model.action_head
    if head.weight.shape[0] != 8 or head.bias.shape != (8,):
        raise ValueError('The existing eight-axis ACT output layer is required')
    parameters = [head.weight, head.bias]
    for parameter in parameters:
        parameter.requires_grad_(True)
        mask = torch.ones_like(parameter)
        mask[7] = 0
        parameter.register_hook(lambda gradient, mask=mask: gradient*mask)
    return parameters


def masked_arm_loss(prediction, target, valid):
    if (prediction.ndim != 3 or prediction.shape[-2:] != (2, 7)
            or target.shape != prediction.shape or valid.shape != prediction.shape[:2]
            or valid.dtype != torch.bool or not valid.any()
            or not torch.isfinite(prediction).all() or not torch.isfinite(target[valid]).all()):
        raise ValueError('Finite paired first-two arm targets with nonempty supervision required')
    return ((prediction[valid]-target[valid])/.0001).square().mean()


def arm_prefix(policy, hidden, current):
    from executor.model import feedback_readout, physical_actions
    if hidden.shape[1] == 2:
        hidden = torch.cat([hidden, hidden[:, -1:].expand(-1, 18, -1)], dim=1)
    if hidden.ndim != 3 or hidden.shape[1] != 20 or current.shape != (len(hidden), 15):
        raise ValueError('Matched ACT query features and current15D required')
    base = policy.base
    prediction = base.absolute_prediction(base.model.action_head(hidden), current)
    statistics = {k: getattr(base, 'feedback_'+k) for k in
                  ['qpos_mean', 'qpos_std', 'action_mean', 'action_std']}
    prediction = feedback_readout(prediction, current[:, :8], statistics,
                                  base.feedback_lower, base.feedback_upper,
                                  version=base.feedback_version)
    return physical_actions(prediction, statistics, bounded_gripper=True)[:, :2, :7]


def load_tuned():
    import json
    from pipeline import load_policy
    from training.integrity import require_unchanged
    run = ROOT / 'training/pilot_A'
    proof = json.loads((run/'integrity.json').read_text())
    if (proof.get('actual_updates') != 1000 or proof.get('complete') is not True
            or proof.get('hardware_deployment_allowed') is not False):
        raise ValueError('Completed source-bound offline trial required')
    require_unchanged(proof['source_sha256'])
    require_unchanged(proof['checkpoint_sha256'])
    payload = torch.load(run/'policy_last.ckpt', map_location='cpu', weights_only=False)
    if payload.get('profile') != 'increment_plus_arm_rows_v1' or payload.get('actual_updates') != 1000:
        raise ValueError('Wrong restricted-tuning checkpoint profile')
    policy, stats = load_policy()
    with torch.no_grad():
        head = policy.base.model.action_head
        head.weight[:7].copy_(payload['arm_weight'].to(head.weight))
        head.bias[:7].copy_(payload['arm_bias'].to(head.bias))
    return policy.eval(), stats
