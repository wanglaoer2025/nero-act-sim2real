"""Small visual sidecar and real8-only auxiliary task; no runtime input filling."""
import torch
from torch import nn
from torch.nn import functional as F


SCALES = [.03] * 7 + [.095]


class VisualResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(128, 16, 1)
        self.up = nn.Conv2d(16, 128, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != 128:
            raise ValueError('Projected visual features must have128 channels')
        return features + self.up(F.silu(self.down(features)))


class AuxiliaryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(136, 128), nn.SiLU(), nn.Linear(128, 160))

    def forward(self, features, state8):
        if (features.ndim != 4 or features.shape[1] != 128
                or state8.shape != (len(features), 8)):
            raise ValueError('Auxiliary head needs paired image features and real8D state')
        pooled = features.mean((-2, -1))
        return self.layers(torch.cat([pooled, state8], dim=1)).reshape(-1, 20, 8)


def physical_state(normalized8, stats):
    mean, std = [normalized8.new_tensor(stats[k]) for k in ('qpos_mean', 'qpos_std')]
    if (normalized8.ndim != 2 or normalized8.shape[1] != 8
            or mean.shape != (8,) or std.shape != (8,)
            or not torch.isfinite(normalized8).all()
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
            or not (std > 0).all()):
        raise ValueError('Paired finite8D state statistics required')
    return normalized8 * std + mean


def decoded_auxiliary(prediction, measured8):
    return measured8[:, None] + prediction * prediction.new_tensor(SCALES)


def auxiliary_loss(prediction, measured8, targets, padding):
    if (prediction.shape != targets.shape or prediction.shape != (len(measured8), 20, 8)
            or measured8.shape != (len(prediction), 8)
            or padding.shape != (len(prediction), 20) or padding.dtype != torch.bool
            or not (~padding).any()):
        raise ValueError('Paired20x8 targets and nonempty valid padding mask required')
    if (not torch.isfinite(prediction).all() or not torch.isfinite(measured8).all()
            or not torch.isfinite(targets[~padding]).all()):
        raise ValueError('Finite valid auxiliary arrays required')
    delta = torch.where(padding[..., None], 0., targets - measured8[:, None])
    scaled = delta / prediction.new_tensor(SCALES)
    error = F.smooth_l1_loss(prediction, scaled, reduction='none')
    error = error[:, :, :7].mean(-1) + error[:, :, 7]
    terms, weights = [], []
    for part, weight in ((slice(0, 2), .75), (slice(2, 20), .25)):
        valid = ~padding[:, part]
        if valid.any():
            terms.append(error[:, part][valid].mean() * weight)
            weights.append(weight)
    return sum(terms) / sum(weights)


def projected_features(policy, images):
    if (images.ndim != 5 or images.shape[1:3] != (1, 3)
            or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1):
        raise ValueError('One RGB camera per observation required')
    base = getattr(policy, 'base', policy)
    normalized = (images - base.image_mean) / base.image_std
    features, _ = base.model.backbones[0](normalized[:, 0])
    return base.model.input_proj(features[0])


def attach_visual_adapter(policy, adapter):
    base = getattr(policy, 'base', policy)
    return base.model.input_proj.register_forward_hook(lambda module, inputs, output: adapter(output))
