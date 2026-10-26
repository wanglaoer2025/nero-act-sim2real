"""15D observations from recorded visual estimates and causal submitted targets."""
import h5py
import numpy as np
import torch

from executor.model import make_policy_observation
from executor.contract import GRIPPER_CONTEXT,VISUAL_PROFILE
from executor.visual_observation import geometry_statistics
from training.command_context import NeroEpisodicDataset


def audit_visual_episode(path):
    with h5py.File(path,'r') as f:
        if (f.attrs.get('visual_observation_profile') != VISUAL_PROFILE
                or f.attrs.get('action_semantics') != 'submitted_absolute_target'
                or not all(f.attrs.get(k) for k in ['source_group','calibration_sha256','observation_code_sha256'])):
            raise ValueError('Explicit visual schema, actual submitted labels and provenance required')
        n = len(f['action'])
        required = {'action':(n,8),'action_time_monotonic_s':(n,),
            'observations/qpos':(n,7),'observations/gripper_state':(n,1),
            'observations/visual_relative_m':(n,6),'observations/visual_valid':(n,),
            'observations/frame_id':(n,),'observations/visual_error_bound_m':(n,),
            'observations/camera_monotonic_s':(n,),'observations/feedback_monotonic_s':(n,)}
        if not n or any(k not in f or f[k].shape != shape for k,shape in required.items()):
            raise ValueError('Complete paired visual episode arrays required')
        q,action = f['observations/qpos'][:],f['action'][:]
        if not np.isfinite(np.r_[q.ravel(),action.ravel()]).all():
            raise ValueError('Nonfinite physical state/targets')
        if not f.attrs.get('sim',False) and np.allclose(q,action[:,:7],atol=1e-9,rtol=0):
            raise ValueError('Real same-frame state labels provide zero-lead supervision; actual targets required')
        command,exposure,feedback = [f[k][:] for k in [
            'action_time_monotonic_s','observations/camera_monotonic_s','observations/feedback_monotonic_s']]
        frames = f['observations/frame_id'][:]
        if (not np.isfinite(np.r_[command,exposure,feedback]).all()
                or np.any(np.diff(command) <= 0) or np.any(np.diff(exposure) <= 0)
                or np.any(np.diff(frames) <= 0) or np.any(abs(exposure-feedback) > .05)):
            raise ValueError('Frames and submitted command times must be ordered and exposures paired within50ms')
        if np.any(command<exposure) or np.any(command-exposure>.25):
            raise ValueError('Future action labels must follow their paired observation within250ms')
        previous = float(f.attrs.get('initial_previous_gripper_target_m',float('nan')))
        if not np.isfinite(previous) or not 0 <= previous <= .095:
            raise ValueError('Explicit initial submitted gripper target required')
        return dict(source_group=str(f.attrs['source_group']),split=str(f.attrs.get('split','')),
                    frames=n,calibration_sha256=str(f.attrs['calibration_sha256']),
                    observation_code_sha256=str(f.attrs['observation_code_sha256']))


def validate_visual_groups(train_paths,val_paths):
    train=[audit_visual_episode(p) for p in train_paths]
    val=[audit_visual_episode(p) for p in val_paths]
    if (any(r['split']!='train' for r in train) or any(r['split']!='val' for r in val)
            or {r['source_group'] for r in train}&{r['source_group'] for r in val}
            or {str(p.resolve()) for p in train_paths}&{str(p.resolve()) for p in val_paths}):
        raise ValueError('Independent train/validation scene groups required; retry leakage found')
    return dict(train_groups=sorted({r['source_group'] for r in train}),val_groups=sorted({r['source_group'] for r in val}))


def visual_statistics(paths):
    states,actions,features,groups = [],[],[],[]
    for path in paths:
        audit = audit_visual_episode(path)
        if audit['split']!='train':raise ValueError('Statistics can use train groups only')
        with h5py.File(path,'r') as f:
            mask=f['observations/visual_valid'][:].astype(bool)
            values=f['observations/visual_relative_m'][:]
            bound=f['observations/visual_error_bound_m'][:]
            mask &= np.isfinite(values).all(axis=1)&np.isfinite(bound)&(bound>=0)
            state=np.c_[f['observations/qpos'][:],f['observations/gripper_state'][:]]
            states.extend(state[mask]);actions.extend(f['action'][:]);features.extend(values[mask])
            groups.extend([audit['source_group']]*int(mask.sum()))
    if not features:raise ValueError('No valid train visual observations')
    stats=geometry_statistics(features,groups,set(groups))
    for name,values in [('qpos',states),('action',actions)]:
        x=np.asarray(values,np.float32)
        stats[name+'_mean']=x.mean(0)
        stats[name+'_std']=np.maximum(x.std(0),1e-2)
    return stats


class VisualReferenceDataset(NeroEpisodicDataset):
    def __init__(self,paths,camera_names,norm_stats,*,policy_config,**kwargs):
        if (policy_config.get('visual_observation_profile') != VISUAL_PROFILE
                or policy_config.get('policy_input_dim') != 15
                or kwargs.get('coordinate_conditioning') or kwargs.get('previous_gripper_conditioning')
                or kwargs.get('joint_target_lead_steps') is not None
                or any(kwargs.get(k,0) for k in ['alignment_offset','joint_lookahead_steps'])):
            raise ValueError('Explicit visual15D schema with already aligned submitted action labels required')
        self.policy_config = dict(policy_config)
        for p in paths:audit_visual_episode(p)
        super().__init__(paths,camera_names,norm_stats,**kwargs)

    def __getitem__(self,index):
        image,_,actions,padding,start=super().__getitem__(index)
        path=self.paths[index] if self.sample_starts is None else self.paths[self.sample_starts[index][0]]
        with h5py.File(path,'r') as f:
            bound=float(f['observations/visual_error_bound_m'][start])
            if not f['observations/visual_valid'][start] or not np.isfinite(bound) or bound<0:
                raise ValueError('Invalid or occluded visual observation; truth fallback forbidden')
            features=f['observations/visual_relative_m'][start]
            state=np.r_[f['observations/qpos'][start],f['observations/gripper_state'][start]]
            exposure=float(f['observations/camera_monotonic_s'][start])
            command_index=int(np.searchsorted(f['action_time_monotonic_s'][:],exposure,side='right'))-1
            previous=(float(f['action'][command_index,7]) if command_index>=0
                      else float(f.attrs['initial_previous_gripper_target_m']))
        observed=make_policy_observation(state,self.stats,self.policy_config,previous,geometry_features_m=features)
        if len(actions)<20:
            n=20-len(actions)
            actions=torch.cat([actions,torch.zeros(n,8,dtype=actions.dtype)])
            padding=torch.cat([padding,torch.ones(n,dtype=torch.bool)])
        return image,torch.from_numpy(observed),actions[:20],padding[:20],start
