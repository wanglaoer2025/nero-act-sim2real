"""The same current samples, labels and history references for both A and B."""
import hashlib
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from executor.model import make_policy_observation
from training.history_data import causal_previous


class HistoryReferenceDataset(Dataset):
    def __init__(self,paths,stats,config,*,weighted=True):
        self.paths=[Path(p) for p in paths];self.stats=stats;self.config=config
        excluded=config.get('excluded_ambiguous_windows',[])
        if (not isinstance(excluded,list) or len(set(excluded))!=len(excluded)
                or any(not isinstance(s,str) or not re.fullmatch(r'episode_\d+:\d+',s) for s in excluded)
                or (excluded and config.get('numeric_history_profile','none')!='none')):
            raise ValueError('Declared unique single-frame ambiguous window exclusions required')
        excluded=set(excluded)
        self.samples=[];self.sample_ids=[];self.cache=[]
        for file_index,path in enumerate(self.paths):
            with h5py.File(path,'r') as f:
                states=np.c_[f['observations/qpos'][:],f['observations/gripper_state'][:]]
                times=f['timestamp'][:];actions=f['action'][:]
                _,previous=causal_previous(actions,f['action_time_monotonic_s'][:],times,float(f.attrs['initial_previous_gripper_target_m']))
                features=f['observations/visual_relative_m'][:];valid=f['observations/visual_valid'][:].astype(bool)
                observed=np.full((len(times),15),np.nan,np.float32)
                for i in np.flatnonzero(valid):
                    observed[i]=make_policy_observation(states[i],stats,config,float(previous[i]),geometry_features_m=features[i])
                refs=f['history/indices'][:];starts=f['history/start_ts'][:];views=f['history/view_id'][:]
                weights=f['history/weights'][:]
                image_hashes=f['observations/rgb_sha256'][:] if 'observations/rgb_sha256' in f else None
                for j,(indices,start,view,weight) in enumerate(zip(refs,starts,views,weights)):
                    ages=times[start]-times[indices]
                    if (indices.shape!=(5,) or indices[-1]!=start or not np.all(np.diff(indices)>0)
                            or not valid[indices].all() or ages[0]>.400000001
                            or np.any(-np.diff(ages)>.150000001) or not np.isfinite(observed[indices]).all()):
                        raise ValueError('Dataset contains invalid causal history reference')
                    if f'{path.stem}:{int(start)}' in excluded:continue
                    for _ in range(int(weight) if weighted else 1):
                        self.samples.append((file_index,j));self.sample_ids.append((path.stem,int(start),int(view)))
                self.cache.append(dict(times=times,actions=actions,observed=observed,refs=refs,starts=starts,
                                       views=views,image_hashes=image_hashes))
        if not self.samples:raise ValueError('No common A/B history samples')

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        file_index,j=self.samples[index];c=self.cache[file_index]
        indices=c['refs'][j];start=int(c['starts'][j])
        with h5py.File(self.paths[file_index],'r') as f:rgb=f['observations/images/camera_0'][start]
        if c['image_hashes'] is not None and hashlib.sha256(rgb.tobytes()).hexdigest()!=c['image_hashes'][start].decode():
            raise ValueError('Current training RGB changed')
        from src.act.nero_dataset import preprocess_images
        image=preprocess_images(rgb[None],self.config.get('image_size',[180,320]))
        count=min(20,len(c['actions'])-start);action=np.zeros((20,8),np.float32)
        action[:count]=(c['actions'][start:start+count]-self.stats['action_mean'])/self.stats['action_std']
        padding=np.arange(20)>=count
        return dict(image=image,current=torch.from_numpy(c['observed'][start].copy()),
            history=dict(observations=torch.from_numpy(c['observed'][indices].copy()),
                age_s=torch.from_numpy((c['times'][start]-c['times'][indices]).astype(np.float32)),
                valid=torch.ones(5,dtype=torch.bool)),actions=torch.from_numpy(action),
            padding=torch.from_numpy(padding),sample_id=self.sample_ids[index])
