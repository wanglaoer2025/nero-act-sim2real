"""Train on the critical20Hz inputs used by fine contact queries at execution."""
from collections import Counter
import h5py
import numpy as np
from training.train import complete_windows,window_is_trainable


def critical_inputs(stages,*,dense_descend=False):
    boundaries=np.flatnonzero(np.r_[True,stages[1:]!=stages[:-1]])
    for first in boundaries:
        stage=stages[first]
        if dense_descend and stage=='descend':
            for start in range(first,len(stages)):
                if stages[start]!=stage:break
                yield int(start),str(stage),int(start-first)
            continue
        if stage not in ('close','lift','release'):continue
        for start in range(first,min(first+20,len(stages))):
            if stages[start]!=stage:break
            yield int(start),str(stage),int(start-first)


def critical_windows(paths,chunk_size,*,stride=10,command_context=False,**kwargs):
    counts=Counter(complete_windows(paths,chunk_size,stride=stride,command_context=command_context,**kwargs))
    for index,path in enumerate(paths):
        with h5py.File(path,'r') as f:
            stages=f['simulation/stage'].asstr()[:];actions=f['action'][:]
            widths=f['observations/gripper_state'][:,0]
            dense_descend=f.attrs.get('task')=='descend_to_retreat_correction'
        for start,stage,offset in critical_inputs(stages,dense_descend=dense_descend):
            if not window_is_trainable(stages,actions,widths,start,command_context=command_context):continue
            repeat=(6 if offset<5 else 2) if stage=='lift' else 1
            counts[index,start]=max(counts[index,start],repeat)
    return [sample for sample in sorted(counts) for _ in range(counts[sample])]


def critical_validation_windows(paths,samples_per_episode,chunk_size,*args,**kwargs):
    samples=list(dict.fromkeys(critical_windows(paths,chunk_size,command_context=kwargs.get('command_context',False))))
    records=[dict(episode_id=int(paths[i].stem.removeprefix('episode_')),start_ts=t) for i,t in samples]
    return samples,records
