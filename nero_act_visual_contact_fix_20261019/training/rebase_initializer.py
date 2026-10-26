"""Change paired normalization without changing the initializer's physical policy."""
import copy
import numpy as np
import torch


def paired_config(config,stats):
    result=copy.deepcopy(config)
    for key in ['qpos_mean','qpos_std','action_mean','action_std']:
        x=np.asarray(stats[key])
        if x.shape!=(8,) or not np.isfinite(x).all() or (key.endswith('std') and np.any(x<=0)):
            raise ValueError('Finite positive paired8D normalization required')
    result.update(constraint_loss_stats={k:np.asarray(stats[k]).tolist() for k in ['qpos_mean','qpos_std','action_mean','action_std']},
        joint_anchor_scale=(stats['qpos_std'][:7]/stats['action_std'][:7]).tolist(),
        joint_anchor_offset=((stats['qpos_mean'][:7]-stats['action_mean'][:7])/stats['action_std'][:7]).tolist(),
        gripper_action_mean=float(stats['action_mean'][7]),gripper_action_std=float(stats['action_std'][7]),
        training_observation_noise_normalized=(np.r_[np.full(7,.0002),.0001]/stats['qpos_std']).tolist())
    if config.get('visual_observation_profile'):
        for key in ['geometry_mean','geometry_std']:
            value=np.asarray(stats[key])
            if value.shape!=(6,) or not np.isfinite(value).all() or (key.endswith('std') and np.any(value<=0)):
                raise ValueError('Paired finite6D visual statistics required')
            result[key]=value.tolist()
        result['training_observation_noise_normalized']=[0.]*8
    return result


def rebase_initializer(state,old_stats,new_stats,config):
    if not config.get('gripper_command_context') or not config.get('joint_residual_anchor'):
        raise ValueError('Explicit9D context and seven-joint absolute residual anchor required')
    paired_config(config,old_stats);paired_config(config,new_stats)
    result={k:v.clone() for k,v in state.items()}
    for prefix in ['model.input_proj_robot_state','model.encoder_joint_proj']:
        w=result[prefix+'.weight'];b=result[prefix+'.bias']
        visual=bool(config.get('visual_observation_profile'))
        if w.shape[1]!=(15 if visual else 9):raise ValueError('Expected matched9/15-column context projection')
        scale=torch.as_tensor(new_stats['qpos_std']/old_stats['qpos_std'],dtype=w.dtype,device=w.device)
        shift=torch.as_tensor((new_stats['qpos_mean']-old_stats['qpos_mean'])/old_stats['qpos_std'],dtype=w.dtype,device=w.device)
        old=w[:,:8].clone();result[prefix+'.bias']=b+old@shift
        w[:,:8]=old*scale[None,:]
        if visual:
            scale=torch.as_tensor(new_stats['geometry_std']/old_stats['geometry_std'],dtype=w.dtype,device=w.device)
            shift=torch.as_tensor((new_stats['geometry_mean']-old_stats['geometry_mean'])/old_stats['geometry_std'],dtype=w.dtype,device=w.device)
            old=w[:,9:].clone();result[prefix+'.bias']=result[prefix+'.bias']+old@shift
            w[:,9:]=old*scale[None,:]
    if config.get('cvae_training'):
        prefix='model.encoder_action_proj'
        w=result[prefix+'.weight'];b=result[prefix+'.bias']
        scale=torch.as_tensor(new_stats['action_std']/old_stats['action_std'],dtype=w.dtype,device=w.device)
        shift=torch.as_tensor((new_stats['action_mean']-old_stats['action_mean'])/old_stats['action_std'],dtype=w.dtype,device=w.device)
        result[prefix+'.bias']=b+w@shift
        result[prefix+'.weight']=w*scale[None,:]
    prefix='model.action_head';w=result[prefix+'.weight'];b=result[prefix+'.bias']
    ratio=torch.as_tensor(old_stats['action_std']/new_stats['action_std'],dtype=w.dtype,device=w.device)
    result[prefix+'.weight']=w*ratio[:,None]
    b=b*ratio
    # Joint means are already represented by the absolute residual anchor.
    b[7]+=(float(old_stats['action_mean'][7])-float(new_stats['action_mean'][7]))/float(new_stats['action_std'][7])
    result[prefix+'.bias']=b
    adapter_keys = [key for key in result if key.startswith('gripper_adapter.')]
    if adapter_keys and not config.get('nonlinear_gripper_head'):
        raise ValueError('Cannot remove a trained nonlinear gripper branch')
    if config.get('nonlinear_gripper_head'):
        if adapter_keys:
            # Hidden features stay physically paired; residual output changes units.
            for key in ['gripper_adapter.2.weight', 'gripper_adapter.2.bias']:
                result[key] *= ratio[7]
        else:
            from executor.model import make_gripper_adapter
            for key, value in make_gripper_adapter(int(config['hidden_dim'])).state_dict().items():
                result['gripper_adapter.'+key] = value
    return result
