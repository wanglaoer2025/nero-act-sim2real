import copy
import json
import numpy as np
import pytest
import torch
from tests.test_restricted_history_heads import config

SETTINGS=dict(profile='four_phase_static_v1',weights=[.05,.5688575789669676,.05,.3311424210330323])


def test_group_weights_keep_each_present_group_mass_when_samples_repeat():
    from training.group_loss import group_sample_weights
    for groups in [torch.tensor([0,1,3]),torch.tensor([0,0,0,1,3])]:
        weight=group_sample_weights(groups,SETTINGS)
        assert weight.sum().item()==pytest.approx(len(groups),abs=1e-6)
        for g in [0,1,3]:
            assert (weight[groups==g].sum()/len(groups)).item()==pytest.approx(SETTINGS['weights'][g]/.95,abs=1e-6)
    for groups in [torch.tensor([0.,1.]),torch.tensor([-1]),torch.tensor([4])]:
        with pytest.raises(ValueError):group_sample_weights(groups,SETTINGS)


def test_grouped_training_preserves_all_arm_gradients_and_runtime_outputs():
    from training.history import build_history_policy
    c,q,image,_=config();c.update(gripper_supervision_profile='deployed_first2_v1',prior_gripper_target_profile='feedback_preimage_v1',deployed_prefix_loss_weight=.01)
    old=build_history_policy(c);new=build_history_policy(dict(c,gripper_group_loss=SETTINGS));new.load_state_dict(old.state_dict())
    old.eval();new.eval()
    with torch.no_grad():torch.testing.assert_close(old(q,image),new(q,image),rtol=0,atol=0)
    for p in [old,new]:p.train()
    target=torch.ones(2,20,8)*.1;padding=torch.zeros(2,20,dtype=torch.bool)
    results=[]
    for p,kwargs in [(old,{}),(new,dict(gripper_groups=torch.tensor([0,3])))]:
        torch.manual_seed(42);loss=p(q,image,target,padding,**kwargs)['loss']
        arm=list(p.model.action_head.parameters());grip=list(p.gripper_adapter.parameters())
        values=torch.autograd.grad(loss,arm+grip);results.append([torch.cat([g.flatten() for g in values[:2]]),torch.cat([g.flatten() for g in values[2:]])])
    torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)
    assert not torch.allclose(results[0][1],results[1][1],rtol=1e-5,atol=1e-9)
    with pytest.raises(ValueError):new(q,image,target,padding)
    with pytest.raises(ValueError):new(q,image,gripper_groups=torch.tensor([0,3]))
    with pytest.raises(ValueError):old(q,image,target,padding,gripper_groups=torch.tensor([0,3]))


def test_group_settings_reject_unknown_profiles_or_dropped_roles():
    from training.history import build_history_policy
    c,_,_,_=config();c['gripper_supervision_profile']='deployed_first2_v1'
    for bad in [dict(profile='unknown',weights=[.25]*4),dict(profile='four_phase_static_v1',weights=[0,0,0,1]),
        dict(profile='four_phase_static_v1',weights=[.1]*4),dict(profile='four_phase_static_v1',weights=[.25]*3)]:
        with pytest.raises(ValueError):build_history_policy(dict(c,gripper_group_loss=bad))


def test_caller_cannot_replace_sealed_group_loss(tmp_path,monkeypatch):
    import training.history as module
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'_require_seal',lambda path:{})
    (tmp_path/'config').mkdir();(tmp_path/'config/history_experiment.json').write_text(json.dumps(dict(gripper_group_loss=SETTINGS)))
    with pytest.raises(ValueError,match='sealed'):
        module.train_history(tmp_path/'data',tmp_path/'init',dict(variant='A',stage='pilot'),tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_dataset_groups_reach_only_training_kwargs_and_leave_observations_identical(tmp_path):
    import h5py
    from training.history_dataset import HistoryReferenceDataset
    from training.history import _batch_to_device
    from tests.test_visual_policy import statistics
    from torch.utils.data import default_collate
    c,_,_,_=config();stats=statistics(c);path=tmp_path/'episode_2000.hdf5';n=20;starts=np.arange(4,n)
    with h5py.File(path,'w') as f:
        f.attrs.update(split='train',initial_previous_gripper_target_m=.075)
        f['observations/qpos']=np.zeros((n,7),np.float32);f['observations/gripper_state']=np.full((n,1),.075,np.float32)
        f['timestamp']=np.arange(n)*.05;f['action_time_monotonic_s']=np.arange(n)*.05+.001
        f['observations/visual_relative_m']=np.full((n,6),.2,np.float32);f['observations/visual_valid']=np.ones(n,bool)
        f['observations/images/camera_0']=np.zeros((n,12,16,3),np.uint8)
        f['action']=np.c_[np.zeros((n,7)),.075-np.arange(n)*.0002]
        f['simulation/stage']=np.array([b'close']*10+[b'release']*10)
        f['history/indices']=np.array([np.arange(t-4,t+1) for t in starts]);f['history/start_ts']=starts
        f['history/view_id']=np.zeros(len(starts),int);f['history/weights']=np.ones(len(starts),int)
    old=HistoryReferenceDataset([path],stats,c)
    new=HistoryReferenceDataset([path],stats,dict(c,gripper_group_loss=SETTINGS))
    assert old.sample_ids==new.sample_ids
    assert 'gripper_group' not in old[0] and new[0]['gripper_group'].item()==0
    assert new[7]['gripper_group'].item()==3
    for key in ['current','actions','image']:torch.testing.assert_close(old[0][key],new[0][key],rtol=0,atol=0)
    assert new[0]['current'].shape==(15,)
    _,kwargs=_batch_to_device(default_collate([new[0],new[7]]),'cpu',with_history=False)
    assert list(kwargs)==['gripper_groups'] and kwargs['gripper_groups'].tolist()==[0,3]
