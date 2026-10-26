from dataclasses import replace
import numpy as np
import pytest
from tests.test_history_inference import make_worker
from tests.test_visual_dispatch import Transport


def components():
    import inspect
    from executor.visual_control import VisualControlCore
    assert "initial_receipt" in inspect.signature(VisualControlCore).parameters, "History dispatch receipt binding missing"
    from executor.visual_dispatch import VisualDispatcher
    worker,seen,now,index=make_worker()
    core=VisualControlCore(worker.policy.config,worker.calibration,np.r_[np.zeros(7),.095],
                           initial_receipt=worker.receipts[0])
    port=Transport(now)
    port.complete=lambda results,deadline:now.__setitem__(0,now[0]+.002)
    dispatcher=VisualDispatcher(core,port,clock=lambda:now[0],inference_worker=worker)
    return worker,core,port,dispatcher,now


def test_history_dispatch_records_actual_ack_and_completion():
    worker,core,port,dispatcher,now=components()
    packet=worker.predict(previous_gripper_target_m=.095,after_s=9.9)
    dispatcher.dispatch_prediction(packet)
    assert core.prefix_complete and worker.pending_receipt is None
    assert len(core.receipts)==len(worker.receipts)==2
    assert core.receipts[-1]['rows']==worker.receipts[-1]['rows']
    assert core.receipts[-1]['ack_s']>=core.receipts[-1]['sent_s']
    dispatcher.confirmation_received(now_s=now[0])
    assert worker.history_buffer.epoch==core.history_epoch==1


def test_collision_time_mutation_is_rechecked_before_submission():
    worker,core,port,dispatcher,now=components()
    packet=worker.predict(previous_gripper_target_m=.095,after_s=9.9)
    def corrupt(rows):
        values=packet['history'].observations.copy();values[0,0]+=1
        packet['history']=replace(packet['history'],observations=values)
    port.validate=corrupt
    with pytest.raises(ValueError):dispatcher.dispatch_prediction(packet)
    assert core.stopped and not port.sent


def test_history_runtime_cannot_bypass_packet_with_raw_dispatch():
    worker,core,port,dispatcher,now=components()
    packet=worker.predict(previous_gripper_target_m=.095,after_s=9.9)
    with pytest.raises(ValueError):dispatcher.dispatch(packet['raw'],packet['frame'],packet['geometry'],packet['paired_feedback'])
    assert not port.sent
