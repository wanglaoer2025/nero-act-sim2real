import hashlib
import h5py
import numpy as np
import pytest
from tests.test_restricted_history_heads import config


def test_excluding_an_ambiguous_wait_keeps_later_labels_and_original_file(tmp_path):
    from training.history_dataset import HistoryReferenceDataset
    from tests.test_visual_policy import statistics
    c,_,_,_=config();stats=statistics(c);path=tmp_path/'episode_8.hdf5';n=6
    image=np.zeros((n,12,16,3),np.uint8);action=np.zeros((n,8),np.float32);action[:,7]=.095;action[5,0]=.01
    with h5py.File(path,'w') as f:
        f.attrs['initial_previous_gripper_target_m']=.095
        f['observations/qpos']=np.zeros((n,7),np.float32);f['observations/gripper_state']=np.full((n,1),.095,np.float32)
        f['timestamp']=np.arange(n)*.05;f['action_time_monotonic_s']=np.arange(n)*.05-.001;f['action']=action
        f['observations/visual_relative_m']=np.full((n,6),.2,np.float32);f['observations/visual_valid']=np.ones(n,bool)
        f['observations/images/camera_0']=image
        f['observations/rgb_sha256']=np.asarray([hashlib.sha256(v.tobytes()).hexdigest().encode() for v in image])
        f['history/indices']=np.asarray([[0,1,2,3,4],[1,2,3,4,5]])
        f['history/start_ts']=np.asarray([4,5]);f['history/view_id']=np.asarray([0,0]);f['history/weights']=np.asarray([2,3])
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    all_samples=HistoryReferenceDataset([path],stats,c)
    selected=HistoryReferenceDataset([path],stats,dict(c,excluded_ambiguous_windows=['episode_8:4']))
    assert len(all_samples)==5 and len(selected)==3
    assert selected.sample_ids==[('episode_8',5,0)]*3
    np.testing.assert_array_equal(selected[0]['actions'].numpy(),all_samples[-1]['actions'].numpy())
    actual=selected[0]['actions'][0].numpy()*stats['action_std']+stats['action_mean']
    assert actual[0]==pytest.approx(.01,abs=1e-7)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    with pytest.raises(ValueError):HistoryReferenceDataset([path],stats,dict(c,excluded_ambiguous_windows=['invalid']))
