"""Pair fresh camera exposures with physical feedback, without robot imports."""
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraFrame:
    frame_id: int
    rgb: np.ndarray
    exposure_monotonic_s: float
    received_monotonic_s: float
    camera_metadata_hash: str
    clock_verified: bool = False


def pair_feedback(frame, feedback, *, max_skew_s=.05):
    if not frame.clock_verified:
        raise ValueError('Camera exposure clock mapping is not verified')
    if (not feedback or not np.isfinite([frame.exposure_monotonic_s,frame.received_monotonic_s,max_skew_s]).all()
            or not 0 <= max_skew_s <= .05 or frame.exposure_monotonic_s > frame.received_monotonic_s):
        raise ValueError('Finite exposure and valid feedback pairing required')
    rows = []
    for row in feedback:
        stamp, state = float(row['monotonic_s']), np.asarray(row['position'], float)
        if state.shape != (8,) or not np.isfinite(np.r_[stamp,state]).all() or not 0 <= state[7] <= .1:
            raise ValueError('Physical8D state with valid gripper feedback required')
        rows.append((abs(stamp-frame.exposure_monotonic_s),stamp,state))
    skew, stamp, state = min(rows, key=lambda row:row[0])
    if skew > max_skew_s:
        raise ValueError('Camera/feedback skew exceeds50ms')
    return dict(state=state.tolist(),monotonic_s=stamp,skew_s=skew,frame_id=frame.frame_id)


class FrameSynchronizer:
    def __init__(self):
        self.last_frame_id = -1
        self.last_exposure = float('-inf')

    def pair(self, frame, feedback, *, now_s):
        if not np.isfinite(now_s):
            raise ValueError('Finite monotonic now required')
        if frame.frame_id <= self.last_frame_id or frame.exposure_monotonic_s <= self.last_exposure:
            raise ValueError('Repeated or out-of-order image')
        if frame.exposure_monotonic_s > now_s or frame.received_monotonic_s > now_s:
            raise ValueError('Camera frame is future-dated')
        if now_s-frame.exposure_monotonic_s > .25:
            raise ValueError('Camera frame is stale')
        result = pair_feedback(frame,feedback)
        if result['monotonic_s'] > now_s or now_s-result['monotonic_s'] > .15:
            raise ValueError('Feedback is future-dated or stale')
        self.last_frame_id,self.last_exposure = frame.frame_id,frame.exposure_monotonic_s
        return result
