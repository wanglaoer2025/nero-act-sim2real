"""Every batch retains all original stages plus exact post-grasp recovery labels."""
import numpy as np

STAGES=('approach','descend','close','lift','carry','place','release','retreat')


def contact_batches(pools):
    keys=[f'{kind}:{stage}' for kind in ['center','other'] for stage in STAGES]
    keys+=['contact:close','contact:lift']
    if set(pools)!=set(keys) or any(not len(pools[k]) for k in keys):
        raise ValueError('All original-stage and contact-recovery pools required')
    flat=[int(x) for key in keys for x in pools[key]]
    if len(set(flat))!=len(flat) or min(flat)<0:
        raise ValueError('Each training sample belongs to one pool')
    rng=np.random.default_rng(231024)
    return np.array([[int(x) for key in keys for x in rng.choice(pools[key],
        8 if key.startswith('contact:') else 1)] for _ in range(1000)],dtype=np.int64)
