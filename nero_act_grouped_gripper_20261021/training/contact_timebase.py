"""Uniform physical gripper timing and explicit normalization transfer for approach replay."""
import copy
import json
from pathlib import Path
import h5py
import numpy as np
from executor.model import make_policy_observation


def unified_scene(scene):
    if scene.get('control_hz')!=20:raise ValueError('Uniform physical data requires20Hz without timestamp rewriting')
    result=copy.deepcopy(scene);result['gripper_speed_limit_m_s']=.004
    result['teacher_gripper_feedback_lead_m']=.008
    result['dwell_seconds'].update(close=8,release=8)
    return result


def verify_uniform_episode(path):
    with h5py.File(Path(path),'r') as f:
        scene=json.loads(f.attrs['scene_config']);times=f['timestamp'][:];actions=f['action'][:]
        if (scene.get('control_hz')!=20 or scene.get('gripper_speed_limit_m_s')!=.004
                or not f.attrs.get('teacher_success') or not f.attrs.get('same_cycle_observation_before_action')):
            raise ValueError('Physical episode lacks the common20Hz/4mm-per-second contract')
    if (actions.ndim!=2 or actions.shape[1]!=8 or len(times)<2 or len(times)!=len(actions)
            or not np.isfinite(actions).all() or not np.isfinite(times).all()
            or not np.allclose(np.diff(times),.05,rtol=0,atol=1e-8)
            or np.max(abs(np.diff(actions[:,7])))>.00020002):
        raise ValueError('Actual physical timestamps or gripper targets violate the uniform contract')
    return dict(complete=True,frames=len(times),control_hz=20,gripper_speed_m_s=.004,
        maximum_gripper_step_m=float(np.max(abs(np.diff(actions[:,7])))))


def reencode_reference(current,target,old_stats,new_stats,config):
    current=np.asarray(current);target=np.asarray(target)
    if (current.ndim!=2 or current.shape[1]!=15 or target.shape!=(len(current),20,8)
            or not np.isfinite(current).all() or not np.isfinite(target).all()):
        raise ValueError('Finite paired15D/20x8 replay arrays required')
    states=current[:,:8]*old_stats['qpos_std']+old_stats['qpos_mean']
    features=current[:,9:]*old_stats['geometry_std']+old_stats['geometry_mean']
    previous=states[:,7]+current[:,8]*.01
    encoded=np.array([make_policy_observation(s,new_stats,config,p,geometry_features_m=g)
        for s,p,g in zip(states,previous,features)],np.float32)
    actions=(target*old_stats['action_std']+old_stats['action_mean']-new_stats['action_mean'])/new_stats['action_std']
    return encoded,actions.astype(np.float32)
