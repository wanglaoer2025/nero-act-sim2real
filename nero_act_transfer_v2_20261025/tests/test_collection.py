import pytest
from test_data import episode


def test_failed_physical_teacher_cannot_publish_qualified_episode(tmp_path):
    from collect_sim import acceptance_record
    p=tmp_path/'one.hdf5';episode(p)
    with pytest.raises(ValueError):
        acceptance_record(p,dict(success=False,full_task_verified=False,error='cup slipped'))
    record=acceptance_record(p,dict(success=True,full_task_verified=True,error=None))
    assert record['full_task_verified'] and record['sha256']


def test_reobservation_preserves_labels_and_pixels_and_tracks_underlying_changes(tmp_path):
    import h5py,numpy as np
    from rebuild_data import publish_reobserved
    from dataset import V2Dataset,fit_statistics
    source=tmp_path/'original.h5';episode(source)
    target=tmp_path/'new.h5'
    geometry=np.full((24,6),.02)
    publish_reobserved(source,target,geometry)
    with h5py.File(target) as f, h5py.File(source) as original:
        assert f['observations/images/camera_0'].is_virtual
        np.testing.assert_equal(f['action'][:],original['action'][:])
        np.testing.assert_equal(f['observations/images/camera_0'][0],original['observations/images/camera_0'][0])
        np.testing.assert_equal(f['observations/visual_relative_m'][:],geometry)
    stats=fit_statistics([target]);ds=V2Dataset([target],stats)
    with h5py.File(source,'r+') as f:f['observations/images/camera_0'][0]=101
    with pytest.raises(ValueError,match='changed'):ds[0]
