"""Changing a receipt, timestamp or immutable past value must stop inference."""
import copy
import importlib
import importlib.util

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('executor.observation_history'), 'Causal history contract missing'
    return importlib.import_module('executor.observation_history')


def item(i, *, epoch=1):
    return dict(observation=np.full(15, i, np.float32), frame_id=i,
                exposure_s=10+i*.1, feedback_s=10+i*.1, admitted_s=10+i*.1+.01,
                receipt_id=str(i), receipt_ack_s=10+i*.1-.01, epoch=epoch,
                calibration_sha256='cal', camera_sha256='camera', session_id='s',
                admission_valid=True, prefix_complete=True)


def history():
    m=api();h=m.HistoryBuffer();h.clear(epoch=1, after_s=9.9)
    for i in range(5):h.append(item(i))
    return h


def test_past_item_uses_its_own_receipt():
    m=api()
    receipts=[dict(receipt_id=str(i),rows=[[0]*7+[.09-i*.001]],sent_s=9+i*.1,
                   ack_s=9.01+i*.1, accepted=True,kind='submitted_prefix',scope='s') for i in range(8)]
    assert m.lookup_receipt(receipts,9.35)['receipt_id']=='3'
    assert m.lookup_receipt(receipts,9.41)['receipt_id']=='3'  # ACK strictly before exposure
    with pytest.raises(ValueError):m.lookup_receipt(receipts,8.)
    receipts[-1]['accepted']=False
    assert m.lookup_receipt(receipts,10.)['receipt_id']=='6'


def test_history_order_span_and_confirmation():
    h=history();b=h.snapshot(np.full(15,4,np.float32))
    assert b.observations.shape==(5,15) and b.valid.all()
    np.testing.assert_allclose(b.age_s,[.4,.3,.2,.1,0],atol=1e-7)
    assert b.receipt_ids[0]!=b.receipt_ids[-1]
    assert np.array_equal(b.observations[-1],np.full(15,4))
    h.clear(epoch=2,after_s=10.5)
    with pytest.raises(ValueError):h.snapshot(np.full(15,4))
    with pytest.raises(ValueError):h.append(item(4))


def test_snapshot_cannot_be_mutated():
    h=history();b=h.snapshot(np.full(15,4))
    with pytest.raises(ValueError):b.observations[0,0]=123
    with pytest.raises(ValueError):b.observations.setflags(write=True)
    with pytest.raises(ValueError):b.age_s.setflags(write=True)
    b.verify()
    h.append(item(5))
    assert b.frame_ids==(0,1,2,3,4)
    b.verify()


@pytest.mark.parametrize('change',[
    {'exposure_s':10.151}, {'frame_id':0}, {'epoch':2}, {'camera_sha256':'other'},
    {'calibration_sha256':'other'}, {'session_id':'other'}, {'admission_valid':False},
    {'prefix_complete':False}, {'receipt_ack_s':10.1}, {'feedback_s':11},
    {'admitted_s':10.0}, {'observation':np.zeros(14)},
])
def test_invalid_admission_stops_history(change):
    m=api();h=m.HistoryBuffer();h.clear(epoch=1,after_s=9.9);h.append(item(0))
    bad=item(1);bad.update(change)
    with pytest.raises(ValueError):h.append(bad)
    with pytest.raises(ValueError):h.append(item(1))  # cannot quietly recover stopped stream


def test_current_mismatch_and_modified_admitted_item_are_rejected():
    h=history()
    with pytest.raises(ValueError):h.snapshot(np.zeros(15))
    m=api();h=m.HistoryBuffer();h.clear(epoch=1,after_s=9.9)
    a=item(0);h.append(a);a['observation'][0]=999
    for i in range(1,5):h.append(item(i))
    assert h.snapshot(np.full(15,4)).observations[0,0]==0
