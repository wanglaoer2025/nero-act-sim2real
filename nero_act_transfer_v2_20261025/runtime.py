"""Offline timing core: injected clocks/senders only; no robot connection API."""
from dataclasses import dataclass
from queue import Queue, Full, Empty
import copy
import math
import threading
import time
import numpy as np

from contract import CommandHistory, physical_state


@dataclass(frozen=True)
class PendingPrediction:
    frame_id: int
    exposure_s: float
    predicted_s: float
    feedback_s: float
    targets: np.ndarray


class AsyncTrace:
    def __init__(self, writer, capacity=64):
        if capacity < 1:
            raise ValueError('Positive bounded trace capacity required')
        self.queue = Queue(maxsize=capacity)
        self.writer = writer
        self.error = None
        self.closed = False
        self._closing=threading.Event()
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def _work(self):
        while True:
            try:value = self.queue.get(timeout=.05)
            except Empty:
                if self._closing.is_set():return
                continue
            try:
                if value is None:
                    return
                self.writer(value)
            except Exception as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Trace writer failed') from self.error
        if self.closed:
            raise RuntimeError('Trace already closed')

    def append(self, value):
        self.check()
        try:
            self.queue.put_nowait(copy.deepcopy(value))
        except Full as exc:
            raise RuntimeError('Trace queue full; no silent data loss') from exc

    def close(self,timeout_s=5.):
        if not math.isfinite(timeout_s) or timeout_s<=0:raise ValueError('Positive shutdown bound required')
        if not self.closed:
            self.closed = True
            self._closing.set()
            # Drain queued records before exiting; never block admitting a
            # sentinel into the same full queue whose writer may have stalled.
            self.thread.join(timeout=timeout_s)
            if self.thread.is_alive():
                raise RuntimeError('Trace writer did not shut down')
        if self.error is not None:
            raise RuntimeError('Trace writer failed') from self.error


class PrefixRunner:
    """Protocol harness, NOT a replacement for collision/health admission."""
    def __init__(self, sender, *, clock=time.monotonic, trace=None):
        self.sender, self.clock, self.trace = sender, clock, trace
        self.history = CommandHistory()
        self.stopped = False
        self.submitted_points = 0
        self.last_frame = -1

    def submit(self, prediction):
        if self.stopped:
            raise RuntimeError('Stop latched')
        try:
            p = prediction
            targets = np.array(p.targets, dtype=float, copy=True)
            if targets.shape != (20, 8) or not np.isfinite(targets).all():
                raise RuntimeError('Invalid20x8 prediction')
            if p.frame_id <= self.last_frame:
                raise RuntimeError('Repeated/out-of-order frame')
            if not all(math.isfinite(x) for x in [p.exposure_s, p.predicted_s, p.feedback_s]):
                raise RuntimeError('Nonfinite timestamp')
            if p.predicted_s < p.exposure_s:
                raise RuntimeError('Prediction precedes exposure')
            for target in targets[:2]:
                physical_state(target,command=True)
            for index, target in enumerate(targets[:2]):
                now = self.clock()
                self._fresh(p,now)
                if self.trace is not None:
                    self.trace.check()
                    self.trace.append(dict(frame_id=p.frame_id, index=index,
                                           stamp=now,kind='submission_intent',target=target.tolist()))
                send_time=self.clock()
                self._fresh(p,send_time)
                # The injected sender must raise on transmission/ACK failure.
                if self.sender(target.copy()) is False:
                    raise RuntimeError('Sender rejected command')
                ack_time=self.clock()
                self.history.submit(target, ack_time)
                self.submitted_points += 1
                if ack_time-send_time>.045 or ack_time<send_time:
                    raise RuntimeError('ACK delay exceeds45ms')
            self.last_frame = p.frame_id
            return 2
        except Exception:
            self.stopped = True
            raise

    @staticmethod
    def _fresh(p,now):
        if (not math.isfinite(now) or now<max(p.exposure_s,p.predicted_s,p.feedback_s)
                or now-p.exposure_s>.25 or now-p.feedback_s>.15 or now-p.predicted_s>.25):
            raise RuntimeError('stale/future prediction or feedback')
