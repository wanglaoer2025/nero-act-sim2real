import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_visual_policy import config
from tests.test_visual_features import valid_inputs


def api():
    assert importlib.util.find_spec('executor.visual_control') is not None, 'Fresh visual short-prefix control core missing'
    return importlib.import_module('executor.visual_control')


def controller():
    m=api();frame,geometry,paired,cal=valid_inputs()
    core=m.VisualControlCore(config(),cal,np.array(paired['state']))
    raw=np.tile(paired['state'],(20,1));raw[:,0]=np.arange(1,21)*.001
    return core,raw,frame,geometry,paired


def test_send_two_advances_only_two_history_targets():
    core,raw,frame,geometry,paired=controller()
    packet=core.prepare(raw,frame,geometry,paired,now_s=10.02)
    assert packet['rows'].shape==(2,8) and core.active_points==0
    core.commit(packet['request_id'],packet['rows'],sent_s=10.021,ack_s=10.025,accepted=True)
    np.testing.assert_array_equal(core.history,packet['rows'][1])
    assert core.active_points==2 and core.history[0] != raw[9,0]


def test_grasp_confirmation_discards_previous_chunk():
    core,raw,frame,geometry,paired=controller()
    packet=core.prepare(raw,frame,geometry,paired,now_s=10.02)
    core.confirmation_received(now_s=10.03)
    with pytest.raises(ValueError):core.commit(packet['request_id'],packet['rows'],sent_s=10.04,ack_s=10.045,accepted=True)
    fresh,raw,frame,geometry,paired=controller();fresh.confirmation_received(now_s=10.03)
    with pytest.raises(ValueError,match='confirmation|fresh'):fresh.prepare(raw,frame,geometry,paired,now_s=10.04)


def test_camera_stall_or_occlusion_latches_stop():
    for invalid in ['stale','occluded']:
        core,raw,frame,geometry,paired=controller()
        if invalid=='occluded':geometry.valid=False
        with pytest.raises(ValueError):core.prepare(raw,frame,geometry,paired,now_s=10.3 if invalid=='stale' else 10.02)
        assert core.stopped
        geometry.valid=True
        with pytest.raises(ValueError):core.prepare(raw,frame,geometry,paired,now_s=10.02)
        assert core.active_points==0


def test_invalid_feedback_100_06mm_stops_before_close():
    core,raw,frame,geometry,paired=controller();paired['state'][7]=.10006
    with pytest.raises(ValueError):core.prepare(raw,frame,geometry,paired,now_s=10.02)
    assert core.stopped and core.pending is None


def test_late_ack_or_changed_rows_cannot_advance_history():
    for problem in ['late','changed']:
        core,raw,frame,geometry,paired=controller();before=core.history.copy()
        packet=core.prepare(raw,frame,geometry,paired,now_s=10.02)
        rows=packet['rows'].copy()
        if problem=='changed':rows[0,0]+=.1
        with pytest.raises(ValueError):core.commit(packet['request_id'],rows,sent_s=10.021,ack_s=10.1 if problem=='late' else 10.025,accepted=True)
        assert core.stopped and core.active_points==0
        np.testing.assert_array_equal(core.history,before)


def test_old_frame_and_unacknowledged_prediction_cannot_be_reused():
    core,raw,frame,geometry,paired=controller();packet=core.prepare(raw,frame,geometry,paired,now_s=10.02)
    core.commit(packet['request_id'],packet['rows'],sent_s=10.021,ack_s=10.025,accepted=True)
    with pytest.raises(ValueError):core.prepare(raw,frame,geometry,paired,now_s=10.03)
    assert core.stopped


def test_new_frame_id_cannot_hide_exposure_before_previous_submission():
    core,raw,frame,geometry,paired=controller()
    packet=core.prepare(raw,frame,geometry,paired,now_s=10.02)
    core.commit(packet['request_id'],packet['rows'],sent_s=10.021,ack_s=10.025,accepted=True)
    frame.frame_id=2;frame.exposure_monotonic_s=10.01;frame.received_monotonic_s=10.03
    geometry.frame_id=2;paired.update(frame_id=2,monotonic_s=10.014,skew_s=.004)
    with pytest.raises(ValueError,match='submission|fresh'):
        core.prepare(raw,frame,geometry,paired,now_s=10.04)
    assert core.stopped and core.active_points==2
