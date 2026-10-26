"""Thin NERO adapter around the official ACT DETR/CVAE model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from third_party.act.detr.main import build_ACT_model_and_optimizer
from src.act.nero_kinematics import SerialArmKinematics


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()


class NeroActPolicy(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.chunk_size = int(config["chunk_size"])
        self.kl_weight = float(config["kl_weight"])
        self.gripper_bce_weight = float(config.get("gripper_bce_weight", 0.0))
        self.gripper_logit_scale = float(config.get("gripper_logit_scale", 1.0))
        history = bool(config.get('previous_gripper_conditioning', False))
        visual = config.get('visual_observation_profile') == 'visual_relative_base_v1'
        if visual and (int(config['state_dim']) != 15 or history
                       or config.get('coordinate_conditioning') or config.get('task_displacement_features')):
            raise ValueError('Visual-relative15D profile is separate from legacy absolute-coordinate features')
        if history and (int(config['state_dim']) != 15 or not config.get('coordinate_conditioning', False)):
            raise ValueError('Previous-gripper conditioning requires explicit 15-D coordinate-conditioned state')
        if int(config['state_dim']) == 15 and not (history or visual):
            raise ValueError('15-D state requires previous_gripper_conditioning')
        self.task_displacement_features=bool(config.get('task_displacement_features',False))
        if self.task_displacement_features and (not config.get('coordinate_conditioning',False) or int(config['state_dim']) not in (14,15)):
            raise ValueError('Derived task displacements require explicit task-coordinate state')
        args = {
            "state_dim": int(config["state_dim"])+(6 if self.task_displacement_features else 0),
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
        from src.act.history_encoder import NumericHistoryEncoder,PROFILE
        self.numeric_history_profile=config.get('numeric_history_profile','none')
        if self.numeric_history_profile not in ('none',PROFILE):
            raise ValueError('Unknown numeric history profile')
        if self.numeric_history_profile==PROFILE:
            if not visual or int(config['hidden_dim'])!=128:
                raise ValueError('History profile requires visual15D and hidden128')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                self.history_encoder=NumericHistoryEncoder().to(next(self.model.parameters()).device)
            self.optimizer.add_param_group(dict(params=self.history_encoder.parameters(),lr=float(config.get('history_lr',1e-4))))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        if self.task_displacement_features:
            mean=torch.tensor(config['kinematic_state_mean'],dtype=torch.float32)
            std=torch.tensor(config['kinematic_state_std'],dtype=torch.float32)
            size=int(config['state_dim'])
            if mean.shape!=(size,) or std.shape!=(size,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std>0).all():
                raise ValueError('Derived geometry requires paired observation statistics')
            self.kinematic_scale_m=float(config['kinematic_feature_scale_m'])
            if not 0<self.kinematic_scale_m<10:raise ValueError('Invalid geometry feature scale')
            self.register_buffer('kinematic_mean',mean,persistent=False)
            self.register_buffer('kinematic_std',std,persistent=False)
            self.feedback_kinematics=SerialArmKinematics(config['feedback_kinematics_specification'])
        self.joint_residual_anchor = bool(config.get('joint_residual_anchor',False))
        if self.joint_residual_anchor:
            if int(config['state_dim']) not in (8,14,15) or int(config['action_dim'])!=8:
                raise ValueError('Joint anchoring requires 8-D feedback, 14-D task state or explicit 15-D command-history state, with 8-D actions')
            if config.get('action_representation','absolute_joint_position')!='absolute_joint_position':
                raise ValueError('Anchored policy outputs absolute joint targets, not feedback deltas')
            scale=torch.tensor(config['joint_anchor_scale'],dtype=torch.float32)
            offset=torch.tensor(config['joint_anchor_offset'],dtype=torch.float32)
            if scale.shape!=(7,) or offset.shape!=(7,) or not torch.isfinite(scale).all() or not torch.isfinite(offset).all() or not (scale>0).all():
                raise ValueError('Joint anchor scale/offset require finite seven-dimensional statistics')
            # Derived constants live in the checkpoint config; legacy state_dict keys stay compatible.
            self.register_buffer('joint_anchor_scale',scale,persistent=False)
            self.register_buffer('joint_anchor_offset',offset,persistent=False)

    def absolute_prediction(self,prediction,qpos):
        if not self.joint_residual_anchor:return prediction
        anchor=qpos[:,:7]*self.joint_anchor_scale+self.joint_anchor_offset
        return torch.cat([prediction[:,:,:7]+anchor[:,None,:],prediction[:,:,7:]],dim=-1)

    def forward(self, qpos, image, actions=None, is_pad=None, *, history=None):
        history_embedding=None
        if self.numeric_history_profile!='none':
            if (not isinstance(history,dict) or set(history)!={'observations','age_s','valid'}
                    or not torch.equal(history['observations'][:,-1],qpos)):
                raise ValueError('Explicit history whose last item equals current15D is required')
            history_embedding=self.history_encoder(**history)
        elif history is not None:
            raise ValueError('Single-frame profile cannot silently ignore numeric history')
        image = (image - self.image_mean) / self.image_std
        model_state=self.encoder_state(qpos)
        if actions is None:
            prediction, _, _ = self.model(model_state, image, None, history_embedding=history_embedding)
            return self.absolute_prediction(prediction,qpos)
        actions = actions[:, :self.chunk_size]
        is_pad = is_pad[:, :self.chunk_size]
        prediction, _, (mu, logvar) = self.model(model_state, image, None, actions, is_pad,history_embedding=history_embedding)
        prediction=self.absolute_prediction(prediction,qpos)
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

    def encoder_state(self,qpos):
        if not self.task_displacement_features:return qpos
        # Derive after sensor-noise augmentation, using only observed joints and
        # supplied static task centres. Never use simulator object poses.
        physical=qpos*self.kinematic_std+self.kinematic_mean
        tcp=self.feedback_kinematics(physical[:,:7])
        relative=torch.cat([physical[:,8:11]-tcp,physical[:,11:14]-tcp],dim=1)/self.kinematic_scale_m
        return torch.cat([qpos,relative],dim=1)
