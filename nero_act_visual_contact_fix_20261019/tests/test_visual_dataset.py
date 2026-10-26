import importlib
import importlib.util
import json

import h5py
import numpy as np
import pytest

from tests.test_visual_policy import config,statistics


def api():
    assert importlib.util.find_spec('training.visual_dataset') is not None, 'Causal15D visual dataset interface missing'
    return importlib.import_module('training.visual_dataset')


def episode(path,*,group='train0',split='train',sim=True):
    n=25;times=10+np.arange(n)*.05
    actions=np.zeros((n,8),np.float32);actions[:,0]=np.arange(n)*.001;actions[:,7]=.070+np.arange(n)*.0002
    with h5py.File(path,'w') as f:
        f.attrs.update(sim=sim,source_group=group,split=split,visual_observation_profile='visual_relative_base_v1',
                       action_semantics='submitted_absolute_target',calibration_sha256='test-only',observation_code_sha256='test-only',
                       initial_previous_gripper_target_m=.065)
        f['action']=actions;f['action_time_monotonic_s']=times+.02
        f['observations/qpos']=np.zeros((n,7),np.float32);f['observations/gripper_state']=np.full((n,1),.074,np.float32)
        f['observations/images/camera_0']=np.zeros((n,90,160,3),np.uint8)
        f['observations/visual_relative_m']=np.tile(np.array([.3]*6,np.float32),(n,1))
        f['observations/visual_valid']=np.ones(n,bool);f['observations/frame_id']=np.arange(n)
        f['observations/visual_error_bound_m']=np.full(n,.002)
        f['observations/camera_monotonic_s']=times;f['observations/feedback_monotonic_s']=times+.004
        f['simulation/cup_truth_base_m']=np.zeros((n,3))
    return path


def test_sim_and_real_use_identical_observation_function(tmp_path):
    m=api();path=episode(tmp_path/'episode_0.hdf5');c=config();stats=statistics(c)
    data=m.VisualReferenceDataset([path],['camera_0'],stats,policy_config=c,sample_starts=[(0,0),(0,2)])
    from executor.model import make_policy_observation
    expected=make_policy_observation(np.r_[np.zeros(7),.074],stats,c,.065,geometry_features_m=np.array([.3]*6))
    image,observed,actions,padding,start=data[0]
    np.testing.assert_array_equal(observed.numpy(),expected)
    assert image.shape==(1,3,90,160) and actions.shape==(20,8) and padding.shape==(20,)
    assert start==0


def test_no_truth_fallback_on_occlusion(tmp_path):
    m=api();path=episode(tmp_path/'episode_0.hdf5')
    with h5py.File(path,'r+') as f:f['observations/visual_valid'][0]=False
    data=m.VisualReferenceDataset([path],['camera_0'],statistics(config()),policy_config=config(),sample_starts=[(0,0)])
    with pytest.raises(ValueError,match='invalid|occlu'):data[0]


def test_previous_command_is_last_submitted(tmp_path):
    m=api();path=episode(tmp_path/'episode_0.hdf5');c=config();stats=statistics(c)
    # A delayed command at index1 was not submitted before image2.
    with h5py.File(path,'r+') as f:f['action_time_monotonic_s'][1:]=10.2+np.arange(24)*.05
    data=m.VisualReferenceDataset([path],['camera_0'],stats,policy_config=c,sample_starts=[(0,2)])
    assert float(data[0][1][8])==pytest.approx(-.4,abs=1e-6)
    with h5py.File(path,'r+') as f:f['action'][2:,7]=.09
    assert float(data[0][1][8])==pytest.approx(-.4,abs=1e-6)


def test_real_same_frame_state_is_not_action_label(tmp_path):
    m=api();path=episode(tmp_path/'episode_0.hdf5',sim=False)
    with h5py.File(path,'r+') as f:
        f['observations/qpos'][:]=f['action'][:,:7]
    with pytest.raises(ValueError,match='same.frame|zero.lead'):m.audit_visual_episode(path)


def test_scene_groups_and_retries_cannot_leak(tmp_path):
    m=api();a=episode(tmp_path/'episode_0.hdf5');b=episode(tmp_path/'episode_1.hdf5',group='train0',split='val')
    with pytest.raises(ValueError,match='group|leak'):m.validate_visual_groups([a],[b])
    with pytest.raises(ValueError,match='train'):m.visual_statistics([b])
    stats=m.visual_statistics([a])
    np.testing.assert_allclose(stats['geometry_mean'],[.3]*6,atol=1e-7)
    assert np.all(np.asarray(stats['geometry_std'])>0)


def test_zero_joint_lead_cannot_replace_submitted_labels_with_qpos(tmp_path):
    m=api();path=episode(tmp_path/'episode_0.hdf5');c=config()
    with pytest.raises(ValueError,match='lead|aligned'):
        m.VisualReferenceDataset([path],['camera_0'],statistics(c),policy_config=c,
                                 sample_starts=[(0,2)],joint_target_lead_steps=0)


def test_past_action_labels_cannot_be_used_as_future_supervision(tmp_path):
    m=api();path=episode(tmp_path/'episode_0.hdf5')
    with h5py.File(path,'r+') as f:f['action_time_monotonic_s'][:]-=10
    with pytest.raises(ValueError,match='future|label|observation'):m.audit_visual_episode(path)
