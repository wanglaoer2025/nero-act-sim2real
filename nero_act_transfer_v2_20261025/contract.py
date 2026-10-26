"""Physical units and strictly causal command context, with no hardware imports."""
from bisect import bisect_left
import math
import numpy as np

SCHEMA = 'nero_act_transfer_v2_15d'


def physical_state(value, *, command=False):
    value = np.asarray(value, dtype=np.float64)
    upper=.0950001 if command else .1
    if value.shape != (8,) or not np.isfinite(value).all() or not 0 <= value[7] <= upper:
        raise ValueError('Finite physical8; feedback0..100mm, commands0..95mm required')
    return value


class CommandHistory:
    def __init__(self):
        self.times, self.targets = [], []
        self._last_event = -math.inf

    def submit(self, target8, stamp, ack=True):
        target = physical_state(target8,command=True)
        stamp = float(stamp)
        if not math.isfinite(stamp) or stamp <= self._last_event:
            raise ValueError('Submission events must have strictly increasing finite times')
        self._last_event = stamp
        if ack:
            self.times.append(stamp)
            self.targets.append(target.copy())

    def before(self, exposure):
        if not math.isfinite(exposure):
            raise ValueError('Finite exposure time required')
        index = bisect_left(self.times, exposure) - 1
        if index < 0:
            raise ValueError('No acknowledged command strictly before exposure')
        return self.targets[index].copy()


def make_observation(state8, geometry6, previous, stats):
    state = physical_state(state8)
    geometry = np.asarray(geometry6, dtype=float)
    if geometry.shape != (6,) or not np.isfinite(geometry).all():
        raise ValueError('Six finite current metric geometry features required')
    if not np.isscalar(previous) or not np.isfinite(previous) or not 0 <= previous <= .0950001:
        raise ValueError('Valid previous submitted gripper target required')
    arrays = [np.asarray(stats[k], dtype=float) for k in
              ['qpos_mean', 'qpos_std', 'geometry_mean', 'geometry_std']]
    for a, n in zip(arrays, [8, 8, 6, 6]):
        if a.shape != (n,) or not np.isfinite(a).all():
            raise ValueError('Paired finite physical and geometry statistics required')
    qm, qs, gm, gs = arrays
    if (qs <= 0).any() or (gs <= 0).any():
        raise ValueError('Positive normalization scales required')
    return np.r_[(state-qm)/qs, (previous-state[7])/.01, (geometry-gm)/gs].astype(np.float32)
