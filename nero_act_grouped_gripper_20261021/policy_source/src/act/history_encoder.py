"""Numeric-only causal five-observation condition; current RGB is encoded once."""
import torch
from torch import nn

PROFILE='causal_numeric_gru5_v1'


class NumericHistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(16,128,num_layers=1,batch_first=True)
        self.projection=nn.Linear(128,128)
        nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)

    def forward(self,observations:torch.Tensor,age_s:torch.Tensor,valid:torch.Tensor)->torch.Tensor:
        if (observations.ndim!=3 or observations.shape[1:]!=(5,15)
                or age_s.shape!=observations.shape[:2] or valid.shape!=age_s.shape
                or valid.dtype!=torch.bool or not bool(valid.all())
                or not bool(torch.isfinite(observations).all()) or not bool(torch.isfinite(age_s).all())
                or not bool((age_s[:,-1]==0).all()) or not bool((age_s[:,0]<=.40000001).all())
                or not bool((age_s[:,:-1]>age_s[:,1:]).all())
                or not bool((age_s[:,:-1]-age_s[:,1:]<=.15000001).all())):
            raise ValueError('Five finite ordered valid15D observations and causal ages required')
        values=torch.cat([observations,age_s[...,None]/.1],dim=-1)
        _,hidden=self.gru(values)
        return self.projection(hidden[-1])
