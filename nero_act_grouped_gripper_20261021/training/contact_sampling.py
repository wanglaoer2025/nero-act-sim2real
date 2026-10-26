"""Keep force-building supervision without repeated timer-based contact dwells.

Stage labels select offline samples only; physical actions and runtime contact
thresholds are unchanged. All other complete-task window roles stay intact.
"""
from collections import Counter

import h5py
import numpy as np

from training.transition_sampling import critical_windows

PROFILE='visual_complete_close_context_v1'
SELECTION_PROFILE='visual_force_building_dedup_v1'


def contact_boundary_pool(dataset):
    """Training recovery observations on both sides of the unchanged 3mm contact threshold."""
    from training.history_data import causal_previous
    selected={}
    for i,path in enumerate(dataset.paths):
        with h5py.File(path,'r') as f:
            if 'recovery_role' not in f.attrs:continue
            if f.attrs['split']!='train':raise ValueError('Boundary pool must use training recoveries only')
            stages=f['simulation/stage'].asstr()[:];widths=f['observations/gripper_state'][:,0]
            _,previous=causal_previous(f['action'][:],f['action_time_monotonic_s'][:],
                f['timestamp'][:],float(f.attrs['initial_previous_gripper_target_m']))
            gap=widths-previous
        selected[i]={int(t) for t in dataset.cache[i]['starts']
            if stages[t] in ('close','lift') and .0025-1e-8<=gap[t]<=.0045+1e-8}
    pool=[i for i,(file,j) in enumerate(dataset.samples) if file in selected and int(dataset.cache[file]['starts'][j]) in selected[file]]
    if not pool:raise ValueError('No valid training contact-boundary samples')
    return np.asarray(pool,np.int64)


def visual_sampling_windows(paths,chunk_size):
    if chunk_size!=20:raise ValueError('Visual ACT requires20-step action supervision')
    counts=Counter(critical_windows(paths,chunk_size,command_context=True))
    for index,path in enumerate(paths):
        with h5py.File(path,'r') as f:
            stages=f['simulation/stage'].asstr()[:]
            actions=f['action'][:]
            widths=f['observations/gripper_state'][:,0]
        for sample in list(counts):
            if sample[0]==index and stages[sample[1]]=='close':del counts[sample]
        for start in np.flatnonzero(stages=='close'):
            first=start==0 or stages[start-1]!='close'
            last=start==len(stages)-1 or stages[start+1]!='close'
            changing=(start>0 and (abs(actions[start,7]-actions[start-1,7])>=.00005
                or abs(widths[start]-widths[start-1])>=.00005
                or np.max(abs(actions[start,:7]-actions[start-1,:7]))>=.0001))
            # Keep the final settled representative: its chunk proceeds to lift.
            # Teacher dwell time is deliberately absent from the runtime input.
            if first or last or changing:counts[index,int(start)]=3
    return [sample for sample in sorted(counts) for _ in range(counts[sample])]
