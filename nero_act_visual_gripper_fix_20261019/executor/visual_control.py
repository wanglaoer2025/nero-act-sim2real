"""Pure short-prefix protocol; no ROS, SDK, IK, simulation rollout or hardware.

This validates prediction/submission state. A future ROS adapter still owns actual
transport, goal completion, contact supervision and receive-only restoration.
"""
import numpy as np

from executor.contract import FullTaskContract,VISUAL_PROFILE
from executor.visual_observation import require_visual_observation


def select_prefix(details, phase):
    # Two50ms points bound the first version to at most100ms of open-loop motion.
    return 2


class VisualControlCore:
    def __init__(self, config, calibration, initial_target):
        if config.get('visual_observation_profile') != VISUAL_PROFILE:
            raise ValueError('Explicit visual-relative15D protocol required')
        self.guard = FullTaskContract(config)
        self.calibration = calibration
        self._history = np.asarray(initial_target,float).copy()
        if self._history.shape != (8,) or not np.isfinite(self._history).all() or not 0 <= self._history[7] <= .095:
            raise ValueError('Finite actual submitted8D starting reference required')
        self.stopped = False
        self.pending = None
        self.active_points = 0
        self.sequence = 0
        self.last_frame_id = -1
        self.last_exposure = float('-inf')
        self.confirmed_at = float('-inf')
        self.last_submission_s = float('-inf')

    @property
    def history(self):
        return self._history.copy()

    def stop(self):
        self.stopped = True
        self.pending = None

    def confirmation_received(self, *, now_s):
        if self.stopped or not np.isfinite(now_s) or now_s < self.confirmed_at:
            self.stop()
            raise ValueError('Stopped protocol cannot restart or accept stale confirmation')
        self.pending = None
        self.confirmed_at = now_s

    def prepare(self, raw, frame, geometry, paired_feedback, *, now_s, phase='approach'):
        try:
            if self.stopped or self.pending is not None:
                raise ValueError('Stopped protocol or unacknowledged prediction cannot dispatch')
            require_visual_observation(frame,geometry,paired_feedback,now_s=now_s,calibration=self.calibration)
            if frame.exposure_monotonic_s <= self.confirmed_at:
                raise ValueError('Confirmation requires fresh post-confirmation image and prediction')
            if frame.exposure_monotonic_s <= self.last_submission_s or paired_feedback['monotonic_s'] <= self.last_submission_s:
                raise ValueError('New image and feedback must be fresh after the previous submission')
            if frame.frame_id <= self.last_frame_id or frame.exposure_monotonic_s <= self.last_exposure:
                raise ValueError('Repeated/out-of-order prediction frame')
            rows = self.guard.validate_prefix(raw,dict(state=paired_feedback['state'],
                age_s=now_s-min(frame.exposure_monotonic_s,paired_feedback['monotonic_s'])),self._history)
            rows = rows[:select_prefix(dict(quality=geometry.quality),phase)].copy()
            if self.active_points+len(rows) > 3900:
                raise ValueError('Original3900 active-point budget exceeded')
            self.sequence += 1
            self.pending = dict(request_id=self.sequence,rows=rows.copy(),prepared_s=now_s,
                                frame_id=frame.frame_id,exposure_s=frame.exposure_monotonic_s)
            self.last_frame_id,self.last_exposure = frame.frame_id,frame.exposure_monotonic_s
            return dict(request_id=self.sequence,rows=rows,raw=np.asarray(raw).copy(),frame_id=frame.frame_id)
        except (ValueError,TypeError,KeyError):
            self.stop()
            raise

    def commit(self, request_id, rows, *, sent_s, ack_s, accepted):
        try:
            pending = self.pending
            if (self.stopped or pending is None or pending['request_id'] != request_id
                    or accepted is not True or not np.array_equal(np.asarray(rows),pending['rows'])
                    or not np.isfinite([sent_s,ack_s]).all()
                    or not pending['prepared_s'] <= sent_s <= ack_s
                    or ack_s-sent_s > .045 or ack_s-pending['exposure_s'] > .25):
                raise ValueError('Unbound, changed, cancelled or late-ACK submission')
            self._history = pending['rows'][-1].copy()
            self.last_submission_s = sent_s
            self.active_points += len(pending['rows'])
            self.pending = None
        except (ValueError,TypeError,KeyError):
            self.stop()
            raise
