"""Thin NERO adapter around the official ACT DETR/CVAE model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from third_party.act.detr.main import build_ACT_model_and_optimizer


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()


class NeroActPolicy(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.chunk_size = int(config["chunk_size"])
        self.kl_weight = float(config["kl_weight"])
        self.gripper_bce_weight = float(config.get("gripper_bce_weight", 0.0))
        self.gripper_logit_scale = float(config.get("gripper_logit_scale", 1.0))
        args = {
            "state_dim": int(config["state_dim"]),
            "action_dim": int(config["action_dim"]),
            "camera_names": list(config["camera_names"]),
            "num_queries": self.chunk_size,
            "hidden_dim": int(config["hidden_dim"]),
            "dim_feedforward": int(config["dim_feedforward"]),
            "nheads": int(config["nheads"]),
            "enc_layers": int(config["enc_layers"]),
            "dec_layers": int(config["dec_layers"]),
            "dropout": float(config["dropout"]),
            "lr": float(config["lr"]),
            "lr_backbone": float(config["lr_backbone"]),
            "weight_decay": float(config["weight_decay"]),
            "backbone": config["backbone"],
            "pretrained_backbone": bool(config["pretrained_backbone"]),
            "device": config["device"],
        }
        self.model, self.optimizer = build_ACT_model_and_optimizer(args)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward(self, qpos, image, actions=None, is_pad=None):
        image = (image - self.image_mean) / self.image_std
        if actions is None:
            prediction, _, _ = self.model(qpos, image, None)
            return prediction
        actions = actions[:, :self.chunk_size]
        is_pad = is_pad[:, :self.chunk_size]
        prediction, _, (mu, logvar) = self.model(qpos, image, None, actions, is_pad)
        valid = (~is_pad).unsqueeze(-1).expand_as(prediction)
        l1 = F.l1_loss(prediction, actions, reduction="none")[valid].mean()
        kl = kl_divergence(mu, logvar)
        valid_steps = ~is_pad
        gripper_bce = F.binary_cross_entropy_with_logits(
            prediction[..., -1][valid_steps] * self.gripper_logit_scale,
            (actions[..., -1][valid_steps] > 0).to(prediction.dtype),
        )
        return {"loss": l1 + self.kl_weight * kl + self.gripper_bce_weight * gripper_bce,
                "l1": l1, "kl": kl, "gripper_bce": gripper_bce,
                "a_hat": prediction, "mu": mu, "logvar": logvar}
