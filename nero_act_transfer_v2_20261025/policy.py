"""Fresh ACT/CVAE with a differentiable causal gripper-increment readout."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VENDOR=Path(__file__).resolve().parent/'vendor'
if str(VENDOR) not in sys.path:
    sys.path.insert(0,str(VENDOR))
from third_party.act.detr.main import get_args_parser
from third_party.act.detr.models.detr_vae import build


class ACTPolicy(nn.Module):
    def __init__(self,stats,*,pretrained=True,seed=42,kl_weight=10.):
        super().__init__()
        torch.manual_seed(seed)
        args=get_args_parser().parse_args([])
        for k,v in dict(state_dim=15,action_dim=8,num_queries=20,hidden_dim=128,
                        dim_feedforward=512,nheads=8,enc_layers=2,dec_layers=2,
                        dropout=.1,lr_backbone=1e-5,pretrained_backbone=pretrained,
                        camera_names=['camera_0']).items():
            setattr(args,k,v)
        self.model=build(args)
        # Upstream exposes every decoder layer and DETRVAE selects index0.
        # Return only the FINAL layer so both declared layers actually train.
        self.model.transformer.decoder.return_intermediate=False
        self.kl_weight=float(kl_weight)
        self.config=dict(schema='nero_act_transfer_v2_15d',hidden_dim=128,dim_feedforward=512,
                         enc_layers=2,dec_layers=2,latent_dim=32,chunk_size=20,execute_points=2,
                         pretrained_backbone=bool(pretrained),seed=seed,kl_weight=self.kl_weight,
                         action_initializer='fresh_no_previous_ACT_checkpoint',
                         hardware_deployment_allowed=False)
        self.config.update(state_dim=8,physical_state_dim=8,policy_input_dim=15,action_dim=8,
                           task_scope='complete_pick_carry_place_retreat',reach_deadline_steps=3900,
                           gripper_command_context='previous_gripper_target_delta_v1',
                           visual_observation_profile='visual_relative_base_v1',
                           image_size=[180,320],geometry_image_size=[720,1280],numeric_history_profile='none')
        self.bounded_gripper=True
        for name in ['qpos_mean','qpos_std','action_mean','action_std']:
            values=torch.as_tensor(stats[name],dtype=torch.float32)
            if values.shape!=(8,) or not torch.isfinite(values).all() or ('std' in name and (values<=0).any()):
                raise ValueError('Finite paired physical8 normalization required')
            self.register_buffer(name,values)
        for name,p in self.model.backbones.named_parameters():
            p.requires_grad_('body.layer4.' in name)
        self.gripper_head=nn.Sequential(nn.Linear(128+15,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.gripper_head[-1].weight)
        nn.init.zeros_(self.gripper_head[-1].bias)
        self.register_buffer('image_mean',torch.tensor([.485,.456,.406]).view(1,1,3,1,1))
        self.register_buffer('image_std',torch.tensor([.229,.224,.225]).view(1,1,3,1,1))
        self._query_features=None
        self.model.action_head.register_forward_pre_hook(self._capture)

    def _capture(self,module,args):
        self._query_features=args[0]  # Deliberately keep autograd connection.

    def _previous(self,obs):
        if obs.ndim!=2 or obs.shape[1]!=15 or not torch.isfinite(obs).all():
            raise ValueError('Finite15D observation required')
        width=obs[:,7:8]*self.qpos_std[7]+self.qpos_mean[7]
        previous=width+obs[:,8:9]*.01
        if (width<0).any() or (width>.1).any() or (previous<0).any() or (previous>.0950001).any():
            raise ValueError('Measured width/causal previous command outside physical limits')
        return width,previous

    def forward(self,obs,image,actions=None,padding=None):
        width,previous=self._previous(obs)
        if image.ndim!=5 or image.shape[:3]!=(len(obs),1,3) or not torch.isfinite(image).all():
            raise ValueError('One finite RGB image per observation required')
        training=actions is not None
        safe_actions=None
        if training:
            if (actions.shape!=(len(obs),20,8) or padding is None
                    or padding.shape!=(len(obs),20) or padding.dtype!=torch.bool
                    or not (~padding).any() or not (~padding[:,:2]).any()
                    or not torch.isfinite(actions[~padding]).all()):
                raise ValueError('Matched finite20x8 targets and nonempty boolean mask required')
            safe_actions=actions.masked_fill(padding[:,:,None],0.)
            desired_physical=safe_actions[:,:2,7]*self.action_std[7]+self.action_mean[7]
            desired=(desired_physical-previous)/.001
            if (desired[~padding[:,:2]].abs()>1.0001).any():
                raise ValueError('Gripper prefix increment exceeds declared1mm span')
        elif padding is not None:
            raise ValueError('Padding without training actions is invalid')
        self._query_features=None
        raw,_,latent=self.model(obs,(image-self.image_mean)/self.image_std,None,safe_actions,padding)
        hidden=self._query_features
        if hidden is None or hidden.shape!=(len(obs),20,128):
            raise RuntimeError('ACT query feature contract changed')
        increments=self.gripper_head(torch.cat([hidden,obs[:,None].expand(-1,20,-1)],dim=-1))[...,0].tanh()
        proposed=previous+.001*increments
        lower=(width-.0095).clamp_min(0.)
        upper=(width+.0095).clamp_max(.095)
        grip=proposed.clamp(min=lower,max=upper)
        prediction=torch.cat([raw[:,:,:7],((grip-self.action_mean[7])/self.action_std[7])[:,:,None]],dim=-1)
        if not training:
            return prediction
        arm=F.l1_loss(raw[:,:,:7],safe_actions[:,:,:7],reduction='none')[~padding].mean()
        grip_loss=F.mse_loss(increments[:,:2][~padding[:,:2]],desired[~padding[:,:2]])
        mu,logvar=latent
        kl=(-.5*(1+logvar-mu.square()-logvar.exp())).sum(1).mean()
        return dict(loss=arm+grip_loss+self.kl_weight*kl,arm=arm,gripper=grip_loss,kl=kl,
                    prediction=prediction,mu=mu,logvar=logvar)

    def optimizer(self):
        vision=[p for p in self.model.backbones.parameters() if p.requires_grad]
        vision_ids={id(p) for p in vision}
        other=[p for p in self.parameters() if p.requires_grad and id(p) not in vision_ids]
        return torch.optim.AdamW([{'params':other,'lr':1e-4},{'params':vision,'lr':1e-5}],weight_decay=1e-4)

    @property
    def device(self):
        return next(self.parameters()).device
