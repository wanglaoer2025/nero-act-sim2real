"""Causal numeric history shared by training, simulation and read-only inference."""
from collections import deque
from dataclasses import dataclass
import hashlib
import json

import numpy as np

PROFILE = 'causal_numeric_gru5_v1'
K = 5


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _readonly(value, dtype):
    a = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def lookup_receipt(receipts: list[dict], exposure_s: float) -> dict:
    if not np.isfinite(exposure_s):
        raise ValueError('Finite exposure required')
    eligible = []
    ids = set()
    for receipt in receipts:
        if not receipt.get('accepted'):
            continue
        rows = np.asarray(receipt['rows'], float)
        sent, ack = receipt['sent_s'], receipt['ack_s']
        if (rows.ndim != 2 or rows.shape[1] != 8 or not 1 <= len(rows) <= 2
                or not np.isfinite(np.r_[rows.ravel(), sent, ack]).all()
                or np.any((rows[:, 7] < 0) | (rows[:, 7] > .0950001))
                or sent > ack or receipt['receipt_id'] in ids
                or receipt['kind'] != 'submitted_prefix' or not receipt['scope']):
            raise ValueError('Actual accepted prefix receipt required')
        ids.add(receipt['receipt_id'])
        if ack < exposure_s:
            eligible.append(receipt)
    if not eligible:
        raise ValueError('No accepted command before exposure')
    chosen = max(eligible, key=lambda r: r['ack_s'])
    # Detach from caller-owned receipt arrays and dictionaries.
    return json.loads(json.dumps(chosen, allow_nan=False))


def make_history_item(packet: dict, receipts: list[dict], stats: dict, config: dict, *, epoch: int) -> dict:
    from executor.model import make_policy_observation
    frame, paired, details = packet['frame'], packet['paired_feedback'], packet['details']
    receipt = lookup_receipt(receipts, frame.exposure_monotonic_s)
    if receipt['scope'] != details['session_id']:
        raise ValueError('Receipt belongs to another session')
    state = np.asarray(paired['state'], np.float32)
    previous = float(receipt['rows'][-1][7])
    observed = make_policy_observation(state, stats, config, previous,
                                      geometry_features_m=details['geometry_features_m'])
    if not np.array_equal(observed, np.asarray(details['policy_observation'], np.float32)):
        raise ValueError('Observation command differs from exposure receipt')
    return dict(observation=observed, frame_id=frame.frame_id,
                exposure_s=frame.exposure_monotonic_s, feedback_s=paired['monotonic_s'],
                admitted_s=details['admitted_s'], receipt_id=receipt['receipt_id'],
                receipt_ack_s=receipt['ack_s'], epoch=epoch,
                calibration_sha256=packet['geometry'].calibration_sha256,
                camera_sha256=frame.camera_metadata_hash, session_id=details['session_id'],
                admission_valid=details['admission_valid'], prefix_complete=details['prefix_complete'])


@dataclass(frozen=True)
class HistoryBatch:
    observations: np.ndarray
    age_s: np.ndarray
    valid: np.ndarray
    frame_ids: tuple
    receipt_ids: tuple
    epoch: int
    context_sha256: str
    items_json: str
    profile: str = PROFILE

    def verify(self):
        items = json.loads(self.items_json)
        expected = _batch(items, self.epoch)
        if (self.profile != PROFILE or self.context_sha256 != expected.context_sha256
                or self.frame_ids != expected.frame_ids or self.receipt_ids != expected.receipt_ids
                or not np.array_equal(self.observations, expected.observations)
                or not np.array_equal(self.age_s, expected.age_s)
                or not np.array_equal(self.valid, expected.valid)):
            raise ValueError('History binding changed')


def _batch(items, epoch):
    latest = items[-1]['exposure_s']
    observations = _readonly([x['observation'] for x in items], np.float32)
    ages = _readonly([latest-x['exposure_s'] for x in items], np.float32)
    serialized = json.dumps(items, sort_keys=True, allow_nan=False)
    return HistoryBatch(observations, ages, _readonly([True]*K, bool),
                        tuple(x['frame_id'] for x in items), tuple(x['receipt_id'] for x in items),
                        epoch, _digest(dict(epoch=epoch, items=items, profile=PROFILE)), serialized)


class HistoryBuffer:
    def __init__(self):
        self.items = deque(maxlen=K)
        self.epoch = None
        self.after_s = None
        self.stopped = False

    def clear(self, *, epoch: int, after_s: float):
        if self.stopped:
            raise ValueError('Stopped history cannot reset permission')
        if not isinstance(epoch, int) or not np.isfinite(after_s) or (self.epoch is not None and epoch <= self.epoch):
            raise ValueError('New confirmation epoch and finite boundary required')
        self.items.clear()
        self.epoch, self.after_s = epoch, after_s

    def append(self, item: dict):
        if self.stopped:
            raise ValueError('History is stopped')
        try:
            self._append(item)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            self.stopped = True
            raise ValueError('Invalid history admission: '+str(exc)) from exc

    def _append(self, item):
        x = dict(item, observation=np.asarray(item['observation'], np.float32).tolist())
        t, feedback, now = (float(x[k]) for k in ['exposure_s', 'feedback_s', 'admitted_s'])
        if (self.epoch is None or x['epoch'] != self.epoch or t <= self.after_s
                or np.shape(x['observation']) != (15,)
                or not np.isfinite(np.r_[x['observation'], t, feedback, now, x['receipt_ack_s']]).all()
                or not x['admission_valid'] or not x['prefix_complete']
                or abs(t-feedback) > .05+1e-9 or max(t, feedback) > now
                or now-t > .25+1e-9 or now-feedback > .15+1e-9
                or x['receipt_ack_s'] >= t or not x['receipt_id']
                or not isinstance(x['frame_id'], int)
                or any(not x[k] for k in ['calibration_sha256', 'camera_sha256', 'session_id'])):
            raise ValueError('Invalid shape, timing, context or permission')
        if self.items:
            prev = self.items[-1]
            gap = t-prev['exposure_s']
            if (x['frame_id'] <= prev['frame_id'] or gap <= 0 or gap > .15+1e-9
                    or any(x[k] != prev[k] for k in ['calibration_sha256', 'camera_sha256', 'session_id'])):
                raise ValueError('History order, gap or binding changed')
        prospective = (list(self.items)+[x])[-K:]
        if t-prospective[0]['exposure_s'] > .4+1e-9:
            raise ValueError('History span exceeds400ms')
        self.items.append(json.loads(json.dumps(x, allow_nan=False)))

    def snapshot(self, current: np.ndarray) -> HistoryBatch:
        if self.stopped or len(self.items) != K:
            raise ValueError('Five admitted history frames required')
        if not np.array_equal(np.asarray(current, np.float32), self.items[-1]['observation']):
            raise ValueError('Current observation differs from newest history')
        return _batch(list(self.items), self.epoch)
