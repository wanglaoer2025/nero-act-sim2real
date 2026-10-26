"""One-variable offline comparison; sealed v2 remains read-only."""
from pathlib import Path
import sys
import torch

BASE=Path(__file__).resolve().parents[1]/'nero_act_transfer_v2_20261025'
sys.path.insert(0,str(BASE))
from policy import ACTPolicy


def prefix_arm_loss(prediction,target,padding):
    if (prediction.shape!=target.shape or prediction.ndim!=3 or prediction.shape[1:]!=(20,7)
            or padding.shape!=prediction.shape[:2] or padding.dtype!=torch.bool
            or not (~padding[:,:2]).any()):
        raise ValueError('Matched20x7 arm labels and valid prefix required')
    first=~padding[:,:2]
    prefix=(prediction[:,:2][first]-target[:,:2][first]).abs().mean()
    remaining=~padding[:,2:]
    if not remaining.any():
        return prefix
    suffix=(prediction[:,2:][remaining]-target[:,2:][remaining]).abs().mean()
    return .5*prefix+.5*suffix


class PrefixPolicy(ACTPolicy):
    def __init__(self,stats,*,variant,**kwargs):
        if variant not in ('A','B'):
            raise ValueError('variant must be A or B')
        super().__init__(stats,**kwargs)
        self.variant=variant

    def forward(self,obs,image,actions=None,padding=None):
        result=super().forward(obs,image,actions,padding)
        if actions is not None and self.variant=='B':
            arm=prefix_arm_loss(result['prediction'][:,:,:7],actions[:,:,:7],padding)
            result.update(arm=arm,loss=arm+result['gripper']+self.kl_weight*result['kl'])
        return result
