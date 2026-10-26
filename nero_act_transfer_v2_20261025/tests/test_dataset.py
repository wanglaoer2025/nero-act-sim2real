import numpy as np
import pytest
from test_data import episode


def test_last_window_masks_missing_future_instead_of_repeating_target(tmp_path):
    from dataset import V2Dataset,fit_statistics
    p=tmp_path/'one.hdf5';episode(p)
    stats=fit_statistics([p]);ds=V2Dataset([p],stats,stride=1)
    row=ds[len(ds)-1]
    assert row['observation'].shape==(15,) and row['image'].shape==(1,3,180,320)
    assert row['padding'].tolist()==[False]+[True]*19
    assert np.isfinite(row['target'].numpy()).all()
    assert row['target'][1:].count_nonzero()==0


def test_dataset_rejects_changed_source_and_future_feedback_proxy(tmp_path):
    import h5py
    from dataset import V2Dataset,fit_statistics
    p=tmp_path/'one.hdf5';episode(p)
    stats=fit_statistics([p]);ds=V2Dataset([p],stats)
    with h5py.File(p,'r+') as f:f['action'][0,7]=.09
    with pytest.raises(ValueError,match='changed'):ds[0]
    proxy=tmp_path/'proxy.hdf5';episode(proxy,kind='future_measured_qpos')
    with pytest.raises(ValueError):V2Dataset([proxy],stats)


def test_statistics_are_from_explicit_train_paths_only(tmp_path):
    from dataset import fit_statistics
    import h5py
    p=tmp_path/'train.hdf5';episode(p)
    dev=tmp_path/'dev.hdf5';episode(dev)
    with h5py.File(dev,'r+') as f:f['observations/qpos'][:]=100
    stats=fit_statistics([p])
    assert max(abs(np.asarray(stats['qpos_mean'])[:7]))==0
    assert stats['source_paths']==[str(p)]
