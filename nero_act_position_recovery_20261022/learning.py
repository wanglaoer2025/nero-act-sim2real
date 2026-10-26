"""Gradient-enabled training of the unchanged deployed ACT computation."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'nero_act_transfer_resume_20261021'))
from checked_runtime import load_projected as load_original_projected
from model import FrozenIncrementPolicy


def load_projected():
    import json
    from pipeline import require_checkpoint
    first = ROOT.parent/'nero_act_position_adaptation_20261022/training/pilot_A'
    checkpoint = first/'policy_last.ckpt'
    proof = json.loads((first/'integrity.json').read_text())
    require_checkpoint(proof, checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('profile') != 'position_projection_transformer_pilot_v1':
        raise ValueError('Exact fixed-last first pilot initializer required')
    original, stats = load_original_projected()
    original.load_state_dict(payload['model'], strict=True)
    return original, stats


class PositionLearningPolicy(FrozenIncrementPolicy):
    def _capture(self, module, args):
        if self._encoding:
            self._hidden = args[0]

    def encode(self, current, image):
        self.head.previous(current)
        if self._encoding:
            raise RuntimeError('Concurrent ACT feature capture is unsupported')
        self._encoding = True
        self._hidden = None
        self.base.eval()
        try:
            prediction = self.base(current, image)
            if self._hidden is None:
                raise RuntimeError('ACT query features were not captured')
            return prediction, self._hidden
        finally:
            self._encoding = False
            self._hidden = None


def load_for_learning():
    original, stats = load_projected()
    original._capture_handle.remove()
    policy = PositionLearningPolicy(original.base, original.head, stats, original.config).cuda().eval()
    policy.requires_grad_(False)
    scopes = {
        'position_projection': (policy.base.model.input_proj_robot_state, 1e-4),
        'action_decoder': (policy.base.model.transformer, 1e-5),
        'arm_output': (policy.base.model.action_head, 1e-4),
        'gripper_increment': (policy.head, 3e-5),
    }
    groups = []
    for name, (module, rate) in scopes.items():
        module.requires_grad_(True)
        groups.append(dict(params=list(module.parameters()), lr=rate, name=name))
    # Inference never consumes the CVAE posterior; keep it and visual encoding frozen.
    return policy, stats, groups


def prefix_objective(prediction, target, padding, stats):
    scale = torch.as_tensor(stats['action_std'], device=prediction.device, dtype=prediction.dtype)
    if prediction.shape != target.shape or prediction.shape[1:] != (20, 8):
        raise ValueError('Paired 20x8 prediction/target required')
    if padding.shape != prediction.shape[:2] or padding.dtype != torch.bool:
        raise ValueError('Paired boolean padding required')
    valid = ~padding
    if not valid[:, :2].any() or not torch.isfinite(target[valid]).all():
        raise ValueError('Finite nonempty executable-prefix targets required')
    error = (prediction-target)*scale
    arm = torch.nn.functional.smooth_l1_loss(error[..., :7]/.002,
                                            torch.zeros_like(error[..., :7]), reduction='none')
    weights = valid.to(error.dtype)
    weights[:, 2:] *= .1
    arm_loss = (arm*weights[..., None]).sum()/(7*weights.sum()).clamp_min(1.)
    grip = error[:, :2, 7]/.0002
    grip_loss = (grip.square()*valid[:, :2]).sum()/valid[:, :2].sum().clamp_min(1)
    return arm_loss+grip_loss, dict(arm=arm_loss, gripper=grip_loss)
