"""Keep complete-task supervision present while learning off-trajectory corrections."""
import numpy as np

STAGES=('approach','descend','close','lift','carry','place','release','retreat')


def make_batches(pools,*,steps,seed):
    keys=['correction']+[f'{kind}:{stage}' for stage in STAGES for kind in ['new','old']]
    if type(steps) is not int or steps<=0 or any(not pools.get(key) for key in keys):
        raise ValueError('Nonempty corrections and new/old samples from every task stage required')
    values=[set(pools[key]) for key in keys]
    if any(any(type(v) not in (int,np.int64) or v<0 for v in group) for group in values):
        raise ValueError('Valid nonnegative dataset indices required')
    if any(values[i]&values[j] for i in range(len(values)) for j in range(i)):
        raise ValueError('A sample cannot belong to two supervision roles')
    rng=np.random.default_rng(seed)
    return np.array([np.r_[rng.choice(pools['correction'],8),
        *[np.r_[rng.choice(pools['new:'+stage],1),rng.choice(pools['old:'+stage],2)] for stage in STAGES]]
        for _ in range(steps)],dtype=np.int64)
