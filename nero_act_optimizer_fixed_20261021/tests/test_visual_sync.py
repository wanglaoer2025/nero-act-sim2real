import importlib
import importlib.util

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('vision.synchronization') is not None, 'Same-frame feedback pairing missing'
    return importlib.import_module('vision.synchronization')


def frame(number=1, stamp=10.):
    m=api()
    return m.CameraFrame(frame_id=number,rgb=np.zeros((360,640,3),np.uint8),
        exposure_monotonic_s=stamp,received_monotonic_s=stamp+.01,
        camera_metadata_hash='camera',clock_verified=True)


def test_pair_rejects_51ms_skew_repeated_future_frame():
    m=api();f=frame()
    bad=[dict(monotonic_s=10.051,position=[0.]*7+[.095])]
    with pytest.raises(ValueError,match='50|skew'):m.pair_feedback(f,bad)
    good=[dict(monotonic_s=9.97,position=[.1]*7+[.095]),dict(monotonic_s=10.004,position=[.2]*7+[.095])]
    value=m.pair_feedback(f,good)
    assert value['state'] == [.2]*7+[.095] and value['skew_s']==pytest.approx(.004)
    sync=m.FrameSynchronizer();sync.pair(f,good,now_s=10.02)
    with pytest.raises(ValueError,match='repeat|order'):sync.pair(f,good,now_s=10.03)
    with pytest.raises(ValueError,match='future'):m.FrameSynchronizer().pair(frame(2,11),good,now_s=10.1)


def test_pair_rejects_unknown_clock_malformed_state_or_stale_data():
    m=api();f=frame();f.clock_verified=False
    with pytest.raises(ValueError,match='clock'):m.pair_feedback(f,[dict(monotonic_s=10.,position=[0.]*8)])
    with pytest.raises(ValueError):m.pair_feedback(frame(),[dict(monotonic_s=10.,position=[0.]*9)])
    with pytest.raises(ValueError,match='stale'):m.FrameSynchronizer().pair(frame(),[dict(monotonic_s=10.,position=[0.]*8)],now_s=10.3)
