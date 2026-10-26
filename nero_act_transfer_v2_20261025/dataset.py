"""The same causal15D observation and submitted-action labels used at inference."""
from pathlib import Path
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from contract import make_observation
from data_audit import audit_episode


def _snapshot(p,dependencies=()):
    return tuple((str(name),Path(name).stat().st_size,Path(name).stat().st_mtime_ns)
                 for name in [p,*dependencies])


def fit_statistics(train_paths):
    states=[];actions=[];geometry=[]
    for p in train_paths:
        report=audit_episode(p)
        if not report['main15_ready']:
            raise ValueError(f'Unqualified training episode: {p}: {report["errors"]}')
        with h5py.File(p) as f:
            states.append(np.c_[f['observations/qpos'][:],f['observations/gripper_state'][:]])
            actions.append(f['action'][:]);geometry.append(f['observations/visual_relative_m'][:])
    if not states:raise ValueError('Nonempty qualified train data required')
    result={'source_paths':[str(p) for p in train_paths]}
    for name,values in [('qpos',states),('action',actions),('geometry',geometry)]:
        array=np.concatenate(values)
        result[name+'_mean']=array.mean(0).tolist()
        result[name+'_std']=np.maximum(array.std(0),.01).tolist()
    return result


class V2Dataset(Dataset):
    def __init__(self,paths,stats,*,stride=4):
        if stride<1:raise ValueError('Positive query stride required')
        self.paths=[Path(p) for p in paths]
        self.stats=stats
        self.dependencies={}
        for p in self.paths:
            with h5py.File(p) as f:
                self.dependencies[str(p)]=[f.attrs['original_episode_path']] if 'original_episode_path' in f.attrs else []
        self.snapshots={str(p):_snapshot(p,self.dependencies[str(p)]) for p in self.paths}
        self.samples=[];self.stages=[]
        self.stage_pools={}
        for i,p in enumerate(self.paths):
            report=audit_episode(p)
            if not report['main15_ready']:
                raise ValueError(f'Unqualified data: {p}: {report["errors"]}')
            with h5py.File(p) as f:
                stages=f['simulation/stage'].asstr()[:] if 'simulation/stage' in f else ['real']*len(f['action'])
                for t in range(0,len(f['action']),stride):
                    self.samples.append((i,t));self.stages.append(str(stages[t]))

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        i,t=self.samples[index];p=self.paths[i]
        if _snapshot(p,self.dependencies[str(p)])!=self.snapshots[str(p)]:raise ValueError('Dataset source changed')
        with h5py.File(p) as f:
            state=np.r_[f['observations/qpos'][t],f['observations/gripper_state'][t]]
            exposed=float(f['observations/camera_monotonic_s'][t])
            times=f['action_time_monotonic_s'][:]
            previous_index=int(np.searchsorted(times,exposed,side='left'))-1
            previous=float(f['action'][previous_index,7]) if previous_index>=0 else float(f.attrs['initial_previous_gripper_target_m'])
            obs=make_observation(state,f['observations/visual_relative_m'][t],previous,self.stats)
            cache='observations/policy_images/camera_0'
            image=f[cache if cache in f else 'observations/images/camera_0'][t]
            image=cv2.resize(image,(320,180),interpolation=cv2.INTER_LINEAR).astype('float32')/255.
            count=min(20,len(f['action'])-t)
            target=np.zeros((20,8),np.float32);padding=np.ones(20,bool)
            mean,std=[np.asarray(self.stats[k],np.float32) for k in ['action_mean','action_std']]
            target[:count]=(f['action'][t:t+count]-mean)/std;padding[:count]=False
            return dict(observation=torch.from_numpy(obs),image=torch.from_numpy(image.transpose(2,0,1)[None]),
                        target=torch.from_numpy(target),padding=torch.from_numpy(padding),stage=self.stages[index])


def balanced_indices(dataset,step,seed=42,batch_size=32):
    rng=np.random.default_rng(seed+step)
    groups=dataset.stage_pools
    if not groups:
        for i,stage in enumerate(dataset.stages):
            name={'approach':'approach','descend':'approach','close':'grasp_lift','lift':'grasp_lift',
                  'carry':'carry','place':'place_release','release':'place_release',
                  'retreat':'retreat','settle':'retreat'}.get(stage)
            if name is not None:groups.setdefault(name,[]).append(i)
    if len(groups)!=5:raise ValueError('All five task-stage pools required')
    uniform=rng.integers(0,len(dataset),batch_size//2).tolist()
    keys=sorted(groups)
    staged=[int(rng.choice(groups[keys[(step+j)%5]])) for j in range(batch_size-len(uniform))]
    return uniform+staged
