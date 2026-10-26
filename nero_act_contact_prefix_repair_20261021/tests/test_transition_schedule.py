from types import SimpleNamespace
import h5py,numpy as np,pytest


def test_fixed_budget_reserves_quarter_for_training_transition_pool():
    from training.history import make_batch_order
    pool=np.arange(20)
    schedule=make_batch_order(1000,batch_size=32,budget={'kind':'updates','limit':1000},seed=42,critical_indices=pool)
    assert schedule.shape==(1000,32)
    assert np.isin(schedule[:,:8],pool).all()
    assert np.bincount(schedule[:,:8].ravel(),minlength=20).min()>300
    identical=make_batch_order(1000,batch_size=32,budget={'kind':'updates','limit':1000},seed=42,critical_indices=pool)
    np.testing.assert_array_equal(schedule,identical)


def test_pool_selects_first_five_actual_starts_from_training_recoveries(tmp_path):
    from training.history import transition_pool
    paths=[]
    for index,role in enumerate([None,'pre_close']):
        p=tmp_path/f'{index}.hdf5';paths.append(p)
        with h5py.File(p,'w') as f:
            f.attrs['split']='train'
            if role:f.attrs['recovery_role']=role
    dataset=SimpleNamespace(paths=paths,cache=[dict(starts=np.arange(10)),dict(starts=np.arange(5,15))],
        samples=[(i,j) for i in range(2) for j in range(10)])
    np.testing.assert_array_equal(transition_pool(dataset),np.arange(10,15))
    with h5py.File(paths[1],'r+') as f:f.attrs['split']='val'
    with pytest.raises(ValueError,match='training'):transition_pool(dataset)


def test_malformed_or_foreign_pool_is_rejected():
    from training.history import make_batch_order
    for pool in [[],[-1],[1000],[1.5]]:
        with pytest.raises(ValueError):make_batch_order(1000,batch_size=32,budget={'kind':'updates','limit':1000},seed=42,critical_indices=pool)


def test_balanced_contact_run_cannot_expand_into_formal_training(tmp_path,monkeypatch):
    import training.history as module
    monkeypatch.setattr(module,'_require_seal',lambda data:{})
    monkeypatch.setattr(module,'ROOT',tmp_path)
    (tmp_path/'config').mkdir()
    (tmp_path/'config/history_experiment.json').write_text('{"transition_sampling_profile":"recovery_first5_quarter_v1"}')
    with pytest.raises(ValueError,match='pilot-only'):
        module.train_history(tmp_path/'data',tmp_path/'initializer',dict(variant='B',stage='formal',
            transition_sampling_profile='recovery_first5_quarter_v1'),tmp_path/'output')
    assert not (tmp_path/'output').exists()
