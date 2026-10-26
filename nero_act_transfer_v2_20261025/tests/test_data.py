import h5py
import json
import numpy as np
import pytest


def episode(path,*,full=True,kind='simulation_submitted_targets'):
    with h5py.File(path,'w') as f:
        f.attrs.update(sim=True,synthetic=True,full_task_verified=full,teacher_success=full,
                       action_semantics=kind,same_cycle_observation_before_action=True,
                       initial_previous_gripper_target_m=.095,
                       initial_previous_acknowledged=True,initial_previous_time_monotonic_s=-.001,
                       initial_previous_source='verified_test_initial_submission',
                       observation_schema='nero_act_transfer_v2_15d')
        n=24
        f['action']=np.c_[np.zeros((n,7)),np.full(n,.095)]
        f['timestamp']=np.arange(n)*.05
        f['action_time_monotonic_s']=np.arange(n)*.05+.001
        f['action_acknowledged']=np.ones(n,bool)
        f['observations/qpos']=np.zeros((n,7))
        f['observations/gripper_state']=np.full((n,1),.095)
        f['observations/visual_relative_m']=np.zeros((n,6))
        f['observations/visual_valid']=np.ones(n,bool)
        f['observations/frame_id']=np.arange(n)
        f['observations/camera_monotonic_s']=np.arange(n)*.05
        f['observations/feedback_monotonic_s']=np.arange(n)*.05
        f['observations/images/camera_0']=np.full((n,32,64,3),100,np.uint8)
        stages=['approach','descend','close','lift','carry','place','release','retreat']*3
        f['simulation/stage']=np.array(stages,dtype='S8')


def test_proxy_and_partial_demonstrations_cannot_qualify_main_policy(tmp_path):
    from data_audit import audit_episode
    good=tmp_path/'good.hdf5';episode(good)
    assert audit_episode(good)['main15_ready']
    proxy=tmp_path/'proxy.hdf5';episode(proxy,kind='future_measured_qpos_plus_recorded_gripper_intent_v1')
    assert not audit_episode(proxy)['main15_ready']
    partial=tmp_path/'partial.hdf5';episode(partial,full=False)
    assert not audit_episode(partial)['main15_ready']


def test_future_previous_context_or_invalid_visual_frame_is_not_qualified(tmp_path):
    from data_audit import audit_episode
    p=tmp_path/'bad.hdf5';episode(p)
    with h5py.File(p,'r+') as f:f['action_time_monotonic_s'][1]=-.1
    assert not audit_episode(p)['main15_ready']
    p2=tmp_path/'occluded.hdf5';episode(p2)
    with h5py.File(p2,'r+') as f:f['observations/visual_valid'][12]=False
    assert not audit_episode(p2)['main15_ready']


def test_position_groups_have_exact_counts_and_five_mm_exclusion():
    from protocol import plan_positions,validate_positions
    p=plan_positions()
    assert {k:len(v) for k,v in p['positions_mm'].items()}=={'train':12,'dev':6,'test':20}
    assert p['episode_counts']=={'train':120,'dev':30,'test':40}
    assert validate_positions(p)
    p['positions_mm']['test'][0]=p['positions_mm']['train'][0]
    with pytest.raises(ValueError):validate_positions(p)


def test_vds_missing_source_is_an_error_not_black_images(tmp_path):
    from data_audit import audit_episode
    p=tmp_path/'missing.hdf5';episode(p)
    with h5py.File(p,'r+') as f:
        del f['observations/images/camera_0']
        layout=h5py.VirtualLayout((24,32,64,3),dtype='uint8')
        layout[:]=h5py.VirtualSource(str(tmp_path/'nonexistent.hdf5'),'images',shape=(24,32,64,3))
        f.create_virtual_dataset('observations/images/camera_0',layout)
    assert not audit_episode(p)['main15_ready']
