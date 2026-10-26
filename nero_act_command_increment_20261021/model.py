"""ACT-conditioned command increments; offline development only, unchanged physical action interface."""
import torch
from torch import nn

PROFILE='causal_gripper_increment_v1'


class IncrementHead(nn.Module):
    def __init__(self,hidden_dim,stats):
        super().__init__();self.hidden_dim=hidden_dim;self.scale_m=.001;self.feedback_span_m=.0095
        for name in ['qpos_mean','qpos_std']:
            value=torch.as_tensor(stats[name],dtype=torch.float32)
            if value.shape!=(8,) or not torch.isfinite(value).all() or (name.endswith('std') and not (value>0).all()):
                raise ValueError('Paired finite measured-state statistics required')
            self.register_buffer(name,value)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.network=nn.Sequential(nn.Linear(hidden_dim+15,64),nn.SiLU(),nn.Linear(64,1))
            nn.init.zeros_(self.network[2].weight);nn.init.zeros_(self.network[2].bias)

    def previous(self,current):
        if current.ndim!=2 or current.shape[1]!=15 or not torch.isfinite(current).all():
            raise ValueError('Finite current15D causal observation required')
        width=current[:,7:8]*self.qpos_std[7]+self.qpos_mean[7]
        previous=width+current[:,8:9]*.01
        if torch.any((previous < -1e-7)|(previous > .0950001)):
            raise ValueError('Previous command outside physical range')
        lower=(width-self.feedback_span_m).clamp_min(0.)
        upper=(width+self.feedback_span_m).clamp_max(.095)
        if torch.any(lower>upper):raise ValueError('Measured width has no feasible physical output')
        return previous.clamp(0.,.095),lower,upper

    def normalized_delta(self,hidden,current):
        self.previous(current)
        if (hidden.ndim!=3 or hidden.shape[0]!=len(current) or not 1<=hidden.shape[1]<=20
                or hidden.shape[2]!=self.hidden_dim or not torch.isfinite(hidden).all()):
            raise ValueError('Matched finite ACT query features required')
        features=torch.cat([hidden,current[:,None].expand(-1,hidden.shape[1],-1)],dim=-1)
        return self.network(features)[...,0].tanh()

    def forward(self,hidden,current):
        previous,lower,upper=self.previous(current)
        proposed=previous+self.scale_m*self.normalized_delta(hidden,current)
        return proposed.clamp(min=lower,max=upper)


def increment_loss(head,hidden,current,target,padding):
    prediction=head.normalized_delta(hidden,current);previous,_,_=head.previous(current)
    if target.shape!=prediction.shape or padding.shape!=prediction.shape or padding.dtype!=torch.bool:
        raise ValueError('Paired first-prefix targets and boolean padding required')
    valid=~padding
    if not valid.any() or not torch.isfinite(target[valid]).all() or torch.any((target[valid]<0)|(target[valid]>.0950001)):
        raise ValueError('Finite valid physical prefix targets required')
    desired=torch.where(valid,(target-previous)/head.scale_m,torch.zeros_like(target))
    if torch.any(desired[valid].abs()>1.0001):raise ValueError('Teacher increment exceeds declared output span')
    return (prediction[valid]-desired[valid]).square().mean()


class FrozenIncrementPolicy(nn.Module):
    def __init__(self,base,head,stats,config):
        super().__init__();self.base=base;self.head=head;self.base.requires_grad_(False);self.base.eval()
        self.config=dict(config,increment_gripper_profile=PROFILE,increment_gripper_scale_m=.001,
                         increment_maximum_executable_points=2,hardware_deployment_allowed=False)
        self.device=next(base.parameters()).device;self.bounded_gripper=True
        for name in ['action_mean','action_std']:
            value=torch.as_tensor(stats[name],dtype=torch.float32,device=self.device)
            if value.shape!=(8,) or not torch.isfinite(value).all() or (name.endswith('std') and not (value>0).all()):
                raise ValueError('Paired finite physical-action statistics required')
            self.register_buffer(name,value)
        self._hidden=None;self._encoding=False
        self._capture_handle=self.base.model.action_head.register_forward_pre_hook(self._capture)

    def _capture(self,module,args):
        if self._encoding:self._hidden=args[0].detach()

    def encode(self,current,image):
        self.head.previous(current)
        if self._encoding:raise RuntimeError('Concurrent ACT feature capture is unsupported')
        self._encoding=True;self._hidden=None;self.base.eval()
        try:
            with torch.no_grad():prediction=self.base(current,image)
            if self._hidden is None:raise RuntimeError('ACT query features were not captured')
            return prediction,self._hidden
        finally:self._encoding=False;self._hidden=None

    def forward(self,current,image,*,history=None):
        if history is not None:raise ValueError('This profile consumes current RGB and15D only')
        baseline,hidden=self.encode(current,image)
        physical=self.head(hidden,current)
        normalized=(physical-self.action_mean[7])/self.action_std[7]
        return torch.cat([baseline[...,:7],normalized[...,None]],dim=-1)
