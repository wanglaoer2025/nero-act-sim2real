import threading
import numpy as np
import pytest


def prediction():
    from runtime import PendingPrediction
    return PendingPrediction(frame_id=1, exposure_s=1., predicted_s=1.02,
                             feedback_s=1.02, targets=np.zeros((20, 8)))


def test_stale_prediction_sends_nothing_and_stop_latches():
    from runtime import PrefixRunner
    sent = []
    runner = PrefixRunner(sent.append, clock=lambda: 1.3)
    with pytest.raises(RuntimeError, match='stale'): runner.submit(prediction())
    assert sent == [] and runner.stopped
    runner.clock = lambda: 1.04
    with pytest.raises(RuntimeError): runner.submit(prediction())
    assert sent == []


def test_prefix_only_advances_actual_two_points_and_rechecks_each_send():
    from runtime import PrefixRunner
    times = iter([1.04,1.04,1.041,1.06,1.06,1.061])
    sent = []
    runner = PrefixRunner(sent.append, clock=lambda: next(times))
    assert runner.submit(prediction()) == 2
    assert len(sent) == 2 and runner.submitted_points == 2
    assert runner.history.before(1.07).shape == (8,)


def test_slow_sender_cannot_emit_a_second_expired_point():
    from runtime import PrefixRunner
    times = iter([1.04,1.04,1.041,1.4])
    sent = []
    runner = PrefixRunner(sent.append, clock=lambda: next(times))
    with pytest.raises(RuntimeError, match='stale'): runner.submit(prediction())
    assert len(sent) == 1 and runner.submitted_points == 1 and runner.stopped


def test_async_logging_does_not_block_control_and_overflow_is_explicit():
    from runtime import AsyncTrace
    gate = threading.Event()
    entered = threading.Event()
    values = []
    def writer(value):
        entered.set()
        gate.wait(2.)
        values.append(value)
    trace = AsyncTrace(writer, capacity=1)
    try:
        trace.append({'n': 1})
        assert entered.wait(1.)
        trace.append({'n': 2})
        with pytest.raises(RuntimeError, match='full'): trace.append({'n': 3})
    finally:
        gate.set()
        trace.close()
    assert values == [{'n': 1}, {'n': 2}]


def test_failed_log_writer_prevents_further_send():
    from runtime import AsyncTrace, PrefixRunner
    failed = threading.Event()
    def writer(_):
        failed.set()
        raise OSError('disk failure')
    trace = AsyncTrace(writer)
    trace.append({'n': 1})
    assert failed.wait(1.)
    with pytest.raises(RuntimeError): trace.close()
    sent = []
    runner = PrefixRunner(sent.append, clock=lambda: 1.04, trace=trace)
    with pytest.raises(RuntimeError): runner.submit(prediction())
    assert sent == [] and runner.stopped


def test_log_admission_delay_is_rechecked_before_actual_send():
    from runtime import PrefixRunner
    now=[1.04];sent=[]
    class SlowTrace:
        def check(self):pass
        def append(self,value):now[0]=1.4
    runner=PrefixRunner(sent.append,clock=lambda:now[0],trace=SlowTrace())
    with pytest.raises(RuntimeError,match='stale'):runner.submit(prediction())
    assert sent==[] and runner.stopped


def test_ack_delay_stops_after_recording_the_one_actual_submission():
    from runtime import PrefixRunner
    times=iter([1.04,1.04,1.10]);sent=[]
    runner=PrefixRunner(sent.append,clock=lambda:next(times))
    with pytest.raises(RuntimeError,match='ACK'):runner.submit(prediction())
    assert len(sent)==1 and runner.submitted_points==1 and runner.stopped
