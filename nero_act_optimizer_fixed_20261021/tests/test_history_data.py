"""Parent retries cannot leak splits; sparse RGB cannot masquerade as history."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest


def api(name):
    assert importlib.util.find_spec(name), 'History data interface missing: '+name
    return importlib.import_module(name)


def manifest():
    base=[dict(id=i,source_group=f'scene_{i}',split='train' if i<36 else 'val') for i in range(48)]
    rows=[]
    for role in ['pre_close','hold_to_lift','support_release','release_retreat']:
        for parent in [0,12,24,36]:
            rows.append(dict(id=2000+len(rows),parent_episode_id=parent,source_group=f'scene_{parent}',
                split='train' if parent<36 else 'val',role=role,source_kind='teacher_interruption',
                status='valid',full_task_verified=False,teacher_success=True,remaining_flow_verified=True,
                replay={'passed':True},prefix_contract_audit={'passed':True},
                context_sha256='context',sha256='file',path=f'episode_{2000+len(rows)}.hdf5'))
    for role in ['pre_close','hold_to_lift']:
        rows.append(dict(rows[0],id=2000+len(rows),parent_episode_id='nominal_development',source_group='nominal_diagnostic_train',
            role=role,source_kind='ACT_failure',path=f'episode_{2000+len(rows)}.hdf5'))
    return dict(schema='history_recovery_v1',complete=True,episodes=rows,base_episodes=base,rejected=[]),dict(train=list(range(36)),val=list(range(36,48)))


def test_parent_recovery_cannot_cross_split():
    m=api('training.history_recoveries');records,split=manifest()
    report=m.validate_history_manifest(records,split,check_files=False)
    assert len(report['train_groups'])==37 and len(report['val_groups'])==12
    assert set(report['train_groups']).isdisjoint(report['val_groups'])
    bad=copy.deepcopy(records);bad['episodes'][3]['split']='train';bad['episodes'][3]['source_group']='renamed_retry'
    with pytest.raises(ValueError):m.validate_history_manifest(bad,split,check_files=False)
    bad=copy.deepcopy(records);bad['episodes'][0]['full_task_verified']=True
    with pytest.raises(ValueError):m.validate_history_manifest(bad,split,check_files=False)


def test_failed_actions_are_context_not_labels():
    m=api('training.history_recoveries');records,split=manifest()
    records['episodes'][0]['label_source']='failed_ACT_actions'
    with pytest.raises(ValueError):m.validate_history_manifest(records,split,check_files=False)


def test_sparse_images_are_rerendered_not_repeated(monkeypatch,tmp_path):
    m=api('training.history_data')
    from training.legacy_visual_data import training_profile
    scene=json.loads((Path(m.__file__).parents[1]/'scene/center.json').read_text())
    scene['generalization']=dict(index=0,source_group='scene_0',split='train')
    n=12;source=tmp_path/'source.hdf5';target=tmp_path/'episode_0.hdf5'
    with h5py.File(source,'w') as f:
        f.attrs.update(scene_config=json.dumps(scene),sim=True,full_task_verified=True,teacher_success=True)
        f['timestamp']=np.arange(n)*.05
        q=np.zeros((n,16));q[:,:7]=scene['initial_robot_state']['qpos_rad'];q[:,7:9]=[.0475,-.0475]
        q[:,9:]=[0,0,.05,1,0,0,0]
        f['simulation/qpos']=q;f['simulation/qvel']=np.zeros((n,15));f['simulation/stage']=np.array([b'close']*n)
        action=np.c_[q[:,:7],.095-np.arange(n)*.0001].astype(np.float32)
        f['action']=action;f['observations/qpos']=q[:,:7].astype(np.float32)
        f['observations/gripper_state']=np.full((n,1),.095,np.float32)
        f['observations/images/camera_0']=np.zeros((n,90,160,3),np.uint8)
    class Renderer:
        scene=type('Scene',(),{'flags':np.zeros(100,dtype=bool)})()
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def update_scene(self,data,**k):self.index=int(round(data.time/.05))
        def render(self):return np.full((180,320,3),self.index,np.uint8)
    class Estimator:
        def __init__(self,*a,**k):pass
        def measure(self,rgb):return np.array([float(rgb[0,0,0])*.001,0,0]),np.array([.1,0,0]),{}
    monkeypatch.setattr(m.mujoco,'Renderer',Renderer);monkeypatch.setattr(m,'NominalImageEstimator',Estimator)
    report=m.derive_history_episode(source,target,expected_sha=m.sha(source),profile=training_profile(scene))
    assert report['valid_samples']>0
    with h5py.File(target) as f:
        np.testing.assert_array_equal(f['action'][:],action)
        for indices in f['history/indices'][:]:
            assert len(set(indices))==5 and np.all(indices<=indices[-1])
            assert len({f['observations/images/camera_0'][i].tobytes() for i in indices})==5
            assert f['timestamp'][indices[-1]]-f['timestamp'][indices[0]]<=.40000001
    dataset_module=api('training.history_dataset')
    stats={k:np.zeros(d,np.float32) if k.endswith('mean') else np.ones(d,np.float32)
           for d,names in [(8,['qpos_mean','qpos_std','action_mean','action_std']),(6,['geometry_mean','geometry_std'])] for k in names}
    from executor.contract import GRIPPER_CONTEXT
    c=dict(gripper_command_context=GRIPPER_CONTEXT,state_dim=8,physical_state_dim=8,policy_input_dim=15,
           visual_observation_profile='visual_relative_base_v1',image_size=[180,320])
    a=dataset_module.HistoryReferenceDataset([target],stats,c,weighted=False)
    b=dataset_module.HistoryReferenceDataset([target],stats,c,weighted=False)
    assert a.sample_ids==b.sample_ids and len(a)>0
    sample=a[0];same=b[0]
    assert sample['history']['observations'].shape==(5,15)
    np.testing.assert_array_equal(sample['current'],sample['history']['observations'][-1])
    np.testing.assert_array_equal(sample['actions'],same['actions'])
    np.testing.assert_array_equal(sample['actions'][0],action[sample['sample_id'][1]])


def test_statistics_use_only_37_training_groups(tmp_path):
    m=api('training.history_data')
    with pytest.raises(ValueError):m.history_statistics([],expected_train_groups=set())
    paths=[];groups={f'train_{i}' for i in range(37)}
    for i in range(38):
        path=tmp_path/f'{i}.hdf5'
        with h5py.File(path,'w') as f:
            f.attrs.update(source_group=f'train_{i}' if i<37 else 'val',split='train' if i<37 else 'val')
            f['observations/visual_valid']=np.ones(2,bool);f['observations/qpos']=np.full((2,7),0 if i<37 else 999,np.float32)
            f['observations/gripper_state']=np.full((2,1),.074,np.float32)
            f['observations/visual_relative_m']=np.full((2,6),.2 if i<37 else 999,np.float32)
            f['action']=np.c_[np.zeros((2,7)),[.065,.065]].astype(np.float32)
        paths.append(path)
    stats=m.history_statistics(paths[:37],expected_train_groups=groups)
    np.testing.assert_allclose(stats['qpos_mean'],[0]*7+[.074],atol=1e-7)
    np.testing.assert_allclose(stats['geometry_mean'],[.2]*6,atol=1e-7)
    with pytest.raises(ValueError):m.history_statistics(paths,expected_train_groups=groups)


def test_preparation_cannot_accept_incomplete_recoveries(tmp_path):
    m=api('training.history_data')
    assert hasattr(m,'prepare_history_data'), 'Complete combined history preparation entry missing'
    source=tmp_path/'base';recoveries=tmp_path/'recoveries';source.mkdir();recoveries.mkdir()
    records,split=manifest();records['complete']=False
    (recoveries/'manifest.json').write_text(json.dumps(records))
    (source/'split.json').write_text(json.dumps(split))
    with pytest.raises(ValueError):m.prepare_history_data(source,recoveries,tmp_path/'output')
    assert not (tmp_path/'output').exists()
