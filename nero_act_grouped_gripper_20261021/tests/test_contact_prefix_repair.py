import copy
import json
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
import torch
from tests.test_restricted_history_heads import config


def gradients(policy,q,image,target,padding,parameters):
    torch.manual_seed(42)
    loss=policy(q,image,target,padding)['loss']
    return torch.cat([v.flatten() for v in torch.autograd.grad(loss,parameters)])


def test_future_gripper_labels_cannot_cancel_executed_prefix_gradient():
    from training.history import build_history_policy
    c,q,image,_=config();c.update(gripper_supervision_profile='deployed_first2_v1',deployed_prefix_loss_weight=.01)
    p=build_history_policy(c).train();padding=torch.zeros(2,20,dtype=torch.bool)
    first=torch.zeros(2,20,8);second=first.clone();first[:,2:,7]=-10.;second[:,2:,7]=10.
    params=list(p.gripper_adapter.parameters())
    one=gradients(p,q,image,first,padding,params);two=gradients(p,q,image,second,padding,params)
    assert one.norm()>0
    torch.testing.assert_close(one,two,rtol=0,atol=0)


def test_seven_axis_tail_supervision_keeps_its_original_gradient():
    from training.history import build_history_policy
    c,q,image,_=config();legacy=build_history_policy(c).train()
    repaired=build_history_policy(dict(c,gripper_supervision_profile='deployed_first2_v1')).train();repaired.load_state_dict(legacy.state_dict())
    padding=torch.zeros(2,20,dtype=torch.bool);target=torch.ones(2,20,8)*.1
    one=gradients(legacy,q,image,target,padding,list(legacy.model.action_head.parameters()))
    two=gradients(repaired,q,image,target,padding,list(repaired.model.action_head.parameters()))
    torch.testing.assert_close(one,two,rtol=0,atol=0)
    changed=target.clone();changed[:,2:,:7]=-10.
    assert not torch.allclose(two,gradients(repaired,q,image,changed,padding,list(repaired.model.action_head.parameters())))


def test_padded_prefix_does_not_train_gripper_from_valid_future_points():
    from training.history import build_history_policy
    c,q,image,_=config();c.update(gripper_supervision_profile='deployed_first2_v1',deployed_prefix_loss_weight=.01)
    p=build_history_policy(c).train();padding=torch.zeros(2,20,dtype=torch.bool);padding[:,:2]=True
    target=torch.ones(2,20,8)*.1
    assert gradients(p,q,image,target,padding,list(p.gripper_adapter.parameters())).abs().sum()==0


def test_unknown_gripper_supervision_is_rejected():
    from training.history import build_history_policy
    c,_,_,_=config()
    with pytest.raises(ValueError):build_history_policy(dict(c,gripper_supervision_profile='unknown'))


def test_caller_cannot_replace_sealed_prefix_profile(tmp_path,monkeypatch):
    import training.history as module
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'_require_seal',lambda path:{})
    (tmp_path/'config').mkdir();declared=dict(gripper_supervision_profile='deployed_first2_v1')
    (tmp_path/'config/history_experiment.json').write_text(json.dumps(declared))
    with pytest.raises(ValueError,match='sealed'):
        module.train_history(tmp_path/'data',tmp_path/'initializer',dict(variant='A',stage='pilot'),tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_uniform_scene_preserves_geometry_and_uses_slow_physical_commands():
    from training.contact_timebase import unified_scene
    from simulation.teacher import expert_gripper_target
    old=dict(control_hz=20,pick_xy_m=[.1,.2],dwell_seconds=dict(close=6,release=6,moving=.2),generalization=dict(seed=42))
    saved=copy.deepcopy(old);scene=unified_scene(old)
    assert old==saved and scene['pick_xy_m']==old['pick_xy_m'] and scene['generalization']==old['generalization']
    assert scene['gripper_speed_limit_m_s']==.004 and scene['dwell_seconds']['close']==scene['dwell_seconds']['release']==8
    assert expert_gripper_target(.065,.073,scene,.075)==pytest.approx(.0728,abs=1e-8)
    with pytest.raises(ValueError):unified_scene(dict(old,control_hz=10))


def test_timebase_audit_reads_actual_command_differences(tmp_path):
    from training.contact_timebase import verify_uniform_episode
    path=tmp_path/'episode.hdf5'
    with h5py.File(path,'w') as f:
        f.attrs.update(scene_config=json.dumps(dict(control_hz=20,gripper_speed_limit_m_s=.004)),same_cycle_observation_before_action=True,teacher_success=True)
        f['timestamp']=np.arange(4)*.05;f['action']=np.c_[np.zeros((4,7)),[.095,.0948,.0946,.0944]]
    assert verify_uniform_episode(path)['complete']
    with h5py.File(path,'r+') as f:f['action'][2,7]=.09
    with pytest.raises(ValueError):verify_uniform_episode(path)


def test_boundary_pool_keeps_post_start_contact_crossing_and_rejects_validation(tmp_path):
    from training.contact_sampling import contact_boundary_pool
    path=tmp_path/'episode.hdf5';starts=np.arange(5,15)
    with h5py.File(path,'w') as f:
        f.attrs.update(recovery_role='pre_close',split='train',initial_previous_gripper_target_m=.075)
        f['timestamp']=np.arange(15)*.05
        f['action_time_monotonic_s']=np.arange(15)*.05+.001
        f['simulation/stage']=np.array([b'close']*15)
        f['observations/gripper_state']=np.full((15,1),.075)
        actions=np.zeros((15,8));actions[:,7]=.075
        actions[starts-1,7]=.075-np.array([.002,.0024,.0027,.0029,.0031,.0035,.0042,.0045,.0048,.008]);f['action']=actions
    ds=SimpleNamespace(paths=[path],cache=[dict(starts=starts)],samples=[(0,j) for j in range(10)])
    assert contact_boundary_pool(ds).tolist()==list(range(2,8))
    with h5py.File(path,'r+') as f:f.attrs['split']='val'
    with pytest.raises(ValueError,match='training'):contact_boundary_pool(ds)


def test_quarter_schedule_reserves_four_slots_for_boundary_inputs():
    from training.history import make_batch_order
    a=make_batch_order(100,batch_size=32,budget=dict(kind='updates',limit=1000),seed=42,critical_indices=np.arange(10),boundary_indices=np.arange(20,30))
    assert np.isin(a[:,:4],np.arange(10)).all() and np.isin(a[:,4:8],np.arange(20,30)).all()
    for invalid in [[],[-1],[100],[1.5]]:
        with pytest.raises(ValueError):make_batch_order(100,batch_size=32,budget=dict(kind='updates',limit=1000),seed=42,critical_indices=np.arange(10),boundary_indices=invalid)


def test_reference_reencoding_preserves_physical_targets_and_previous_command():
    from training.contact_timebase import reencode_reference
    from tests.test_visual_policy import statistics
    from executor.model import make_policy_observation
    c,_,_,_=config();old=statistics(c);new={k:v.copy() for k,v in old.items()}
    for key in ['qpos_std','action_std','geometry_std']:new[key]*=1.1
    for key in ['qpos_mean','action_mean','geometry_mean']:new[key]+=.001
    state=np.r_[np.zeros(7),.074];features=np.full(6,.2);previous=.070
    current=np.array([make_policy_observation(state,old,c,previous,geometry_features_m=features)])
    target=np.zeros((1,20,8),np.float32);before=current.copy()
    q,a=reencode_reference(current,target,old,new,c)
    np.testing.assert_array_equal(current,before)
    np.testing.assert_allclose(q[:,:8]*new['qpos_std']+new['qpos_mean'],current[:,:8]*old['qpos_std']+old['qpos_mean'],atol=1e-7)
    assert float((q[0,7]*new['qpos_std'][7]+new['qpos_mean'][7])+q[0,8]*.01)==pytest.approx(previous,abs=1e-8)
    np.testing.assert_allclose(a*new['action_std']+new['action_mean'],target*old['action_std']+old['action_mean'],atol=1e-7)


def test_batch_preparation_binds_actual_boundary_indices(tmp_path):
    import pickle
    from training.history import prepare_batch_orders
    from tests.test_visual_policy import statistics
    c,_,_,_=config();c['transition_sampling_profile']='contact_boundary_halves_v1';stats=statistics(c)
    path=tmp_path/'episode_2000.hdf5';n=20;starts=np.arange(4,n)
    with h5py.File(path,'w') as f:
        f.attrs.update(split='train',recovery_role='pre_close',initial_previous_gripper_target_m=.075)
        f['observations/qpos']=np.zeros((n,7),np.float32);f['observations/gripper_state']=np.full((n,1),.075,np.float32)
        f['timestamp']=np.arange(n)*.05;f['action_time_monotonic_s']=np.arange(n)*.05+.001
        f['observations/visual_relative_m']=np.full((n,6),.2,np.float32);f['observations/visual_valid']=np.ones(n,bool)
        f['observations/images/camera_0']=np.zeros((n,12,16,3),np.uint8)
        actions=np.zeros((n,8));actions[:,7]=.075-np.linspace(.001,.006,n);f['action']=actions
        f['simulation/stage']=np.array([b'close']*n)
        f['history/indices']=np.array([np.arange(t-4,t+1) for t in starts]);f['history/start_ts']=starts
        f['history/view_id']=np.zeros(len(starts),int);f['history/weights']=np.ones(len(starts),int)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(episodes=[dict(path=str(path),split='train')])))
    (tmp_path/'dataset_stats.pkl').write_bytes(pickle.dumps(stats))
    proof=prepare_batch_orders(tmp_path,c)['pilot'];order=np.load(proof['path'])
    assert proof['boundary_slots_per_batch']==4 and proof['critical_slots_per_batch']==4
    assert np.isin(order[:,:4],proof['critical_pool_indices']).all()
    assert np.isin(order[:,4:8],proof['boundary_pool_indices']).all()
