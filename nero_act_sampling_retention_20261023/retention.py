"""Retain position-intervention labels instead of merely loading their paths."""
import numpy as np

STAGES=('approach','descend','close','lift','carry','place','release','retreat')


def make_batches(pools,*,steps,seed):
    keys=['new_correction','retained_align','retained_clearance']+[f'{kind}:{stage}' for stage in STAGES for kind in ['new','old']]
    if type(steps) is not int or steps<=0 or any(not pools.get(key) for key in keys):
        raise ValueError('Every original correction role and complete-task stage is required')
    groups=[set(pools[key]) for key in keys]
    if any(groups[i]&groups[j] for i in range(len(groups)) for j in range(i)):
        raise ValueError('Supervision roles must not overlap')
    rng=np.random.default_rng(seed)
    return np.array([np.r_[rng.choice(pools['new_correction'],4),rng.choice(pools['retained_align'],2),
        rng.choice(pools['retained_clearance'],2),
        *[np.r_[rng.choice(pools['new:'+stage],1),rng.choice(pools['old:'+stage],2)] for stage in STAGES]]
        for _ in range(steps)],dtype=np.int64)
