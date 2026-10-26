"""Frozen ACT features, physically equivalent absolute/increment7 readouts."""
from pathlib import Path
import sys
import torch
from torch import nn

BASE=Path(__file__).resolve().parents[1]/'nero_act_transfer_v2_20261025'
sys.path.insert(0,str(BASE))
from policy import ACTPolicy


def encode_delta(target,current,mean,scale):
    return (target-current[:,None]-mean)/scale


def decode_delta(value,current,mean,scale):
    return current[:,None]+mean+scale*value


def weighted_loss(prediction,target,padding):
    first=~padding[:,:2]
    if not first.any():raise ValueError('Valid first2 labels required')
    prefix=(prediction[:,:2][first]-target[:,:2][first]).abs().mean()
    later=~padding[:,2:]
    if not later.any():return prefix
    return .5*prefix+.5*(prediction[:,2:][later]-target[:,2:][later]).abs().mean()


class ArmPolicy(ACTPolicy):
    def __init__(self,stats,delta_stats,*,variant,**kwargs):
        if variant not in ('A','B'):raise ValueError('Invalid variant')
        dm=torch.as_tensor(delta_stats['mean'],dtype=torch.float32)
        ds=torch.as_tensor(delta_stats['std'],dtype=torch.float32)
        if dm.shape!=(7,) or ds.shape!=(7,) or not torch.isfinite(dm).all() or not torch.isfinite(ds).all() or (ds<=0).any():
            raise ValueError('Finite7 delta mean and positive scale required')
        super().__init__(stats,**kwargs)
        for p in self.parameters():p.requires_grad_(False)
        self.variant=variant
        self.register_buffer('delta_mean',dm)
        self.register_buffer('delta_std',ds)
        self.arm_readout=nn.Linear(143,7)
        self.train(False)

    def train(self,mode=True):
        super().train(False)
        if hasattr(self,'arm_readout'):self.arm_readout.train(mode)
        return self

    def load_source(self,checkpoint):
        expected={'delta_mean','delta_std','arm_readout.weight','arm_readout.bias'}
        result=self.load_state_dict(checkpoint['model'],strict=False)
        if set(result.missing_keys)!=expected or result.unexpected_keys:
            raise ValueError('Source ACT state does not match the declared architecture')
        with torch.no_grad():
            w=self.model.action_head.weight[:7];b=self.model.action_head.bias[:7]
            self.arm_readout.weight.zero_()
            if self.variant=='A':
                self.arm_readout.weight[:,:128].copy_(w)
                self.arm_readout.bias.copy_(b)
            else:
                self.arm_readout.weight[:,:128].copy_(w*(self.action_std[:7]/self.delta_std)[:,None])
                self.arm_readout.weight[:,128:135].copy_(torch.diag(-self.qpos_std[:7]/self.delta_std))
                self.arm_readout.bias.copy_((self.action_std[:7]*b+self.action_mean[:7]-self.qpos_mean[:7]-self.delta_mean)/self.delta_std)
        self.config.update(arm_coordinate=self.variant,arm_core_frozen=True,
                           arm_delta_reference='current_paired_feedback_no_cumsum',hardware_deployment_allowed=False)

    def source_prediction(self,obs,image):
        with torch.no_grad():return ACTPolicy.forward(self,obs,image)

    def forward(self,obs,image,actions=None,padding=None):
        original=self.source_prediction(obs,image)
        hidden=self._query_features.detach()
        value=self.arm_readout(torch.cat([hidden,obs[:,None].expand(-1,20,-1)],dim=-1))
        current=obs[:,:7]*self.qpos_std[:7]+self.qpos_mean[:7]
        arm=value if self.variant=='A' else (decode_delta(value,current,self.delta_mean,self.delta_std)-self.action_mean[:7])/self.action_std[:7]
        prediction=torch.cat([arm,original[:,:,7:8]],dim=-1)
        if actions is None:
            if padding is not None:raise ValueError('Padding without actions')
            return prediction
        if (actions.shape!=(len(obs),20,8) or padding is None or padding.shape!=(len(obs),20)
                or padding.dtype!=torch.bool or not torch.isfinite(actions[~padding]).all()):
            raise ValueError('Matched finite20x8 submitted labels and mask required')
        target=actions[:,:,:7]
        if self.variant=='B':
            physical=target*self.action_std[:7]+self.action_mean[:7]
            target=encode_delta(physical,current,self.delta_mean,self.delta_std)
        arm_loss=weighted_loss(value,target,padding)
        return dict(loss=arm_loss,arm=arm_loss,prediction=prediction)

    def optimizer(self):
        return torch.optim.AdamW(self.arm_readout.parameters(),lr=1e-4,weight_decay=1e-4)
