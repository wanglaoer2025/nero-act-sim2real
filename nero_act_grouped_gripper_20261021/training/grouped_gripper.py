"""Training-only gripper groups and exact executed-prefix loss contributions."""
import numpy as np
import torch
from itertools import combinations
from executor.model import feedback_preimage

GROUP_NAMES=('closing','holding','opening','release')


def minimum_norm_weights(vectors):
    vectors=np.asarray(vectors,np.float64)
    if vectors.ndim!=2 or not vectors.shape[0] or not vectors.shape[1] or not np.isfinite(vectors).all():
        raise ValueError('Finite nonempty group gradient vectors required')
    gram=vectors@vectors.T;best=None
    # Four groups have only15 nonempty faces; solve each exactly and keep a feasible convex combination.
    for size in range(1,len(vectors)+1):
        for active in combinations(range(len(vectors)),size):
            block=gram[np.ix_(active,active)]
            system=np.block([[2*block,np.ones((size,1))],[np.ones((1,size)),np.zeros((1,1))]])
            answer=np.linalg.lstsq(system,np.r_[np.zeros(size),1.],rcond=None)[0][:size]
            if answer.min()<-1e-9 or abs(answer.sum()-1)>1e-8:continue
            weight=np.zeros(len(vectors));weight[list(active)]=np.maximum(answer,0);weight/=weight.sum()
            score=float(weight@gram@weight)
            if best is None or score<best[0]-1e-14:best=(score,weight)
    if best is None:raise RuntimeError('No feasible group loss combination')
    return best[1]


def classify_gripper_groups(stages,margins):
    stages=np.asarray(stages);margins=np.asarray(margins)
    known=['close','lift','carry','place','approach','descend','retreat','settle','release','history_context']
    if stages.ndim!=1 or margins.shape!=stages.shape or not np.isfinite(margins).all() or not np.isin(stages,known).all():
        raise ValueError('Known offline teacher stages and finite matching current margins required')
    result=np.full(len(stages),-1,np.int64)
    result[(stages=='close')&(margins<.003)]=0
    result[((stages=='close')&(margins>=.003))|np.isin(stages,['lift','carry','place'])]=1
    result[np.isin(stages,['approach','descend','retreat','settle'])]=2
    result[stages=='release']=3
    return result


def gripper_contributions(policy,prior,qpos,target,padding):
    """One scalar per sample; sum is the original first2 gripper part of the loss."""
    if policy.gripper_supervision_steps!=2 or policy.margin_weight!=0:
        raise ValueError('Grouped diagnostics require the existing first2 restricted loss without margin penalty')
    prediction=prior['a_hat'];valid=(~padding[:,:20]).to(prediction.dtype)
    weights=valid.clone();weights[:,:10]*=5
    first=valid[:,:2];error=(prediction[:,:2,7]-target[:,:2,7]).abs()
    contribution=(error*first*5).sum(1)/(weights.sum()*8).clamp_min(1)
    inverse=target
    if policy.prior_target_profile=='feedback_preimage_v1':
        stats={k:getattr(policy,'feedback_'+k) for k in ['qpos_mean','qpos_std','action_mean','action_std']}
        inverse=feedback_preimage(target,qpos[:,:8],stats,policy.feedback_lower,policy.feedback_upper,version=policy.feedback_version).detach()
    contribution=contribution+policy.prior_gripper_weight*((prior['unbounded_a_hat'][:,:2,7]-inverse[:,:2,7]).abs()*first).sum(1)/first.sum().clamp_min(1)
    if policy.deployed_prefix_weight:
        scale=torch.as_tensor(policy.loss_stats['action_std'][7],dtype=prediction.dtype,device=prediction.device)
        contribution=contribution+policy.deployed_prefix_weight*(error*scale/.0005*first).sum(1)/(first.sum()*8).clamp_min(1)
    return contribution
