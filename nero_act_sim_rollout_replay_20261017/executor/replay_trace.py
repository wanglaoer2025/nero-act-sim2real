"""Frozen ACT simulation execution references; no model inference or robot connection."""
import hashlib
import json
import numpy as np
from executor.contract import FullTaskContract

MODE = 'ACT_simulation_trajectory_replay_not_live_visual_policy'


def validate_reference(rows):
    rows = np.asarray(rows, float)
    guard = FullTaskContract(dict(chunk_size=20,state_dim=8,action_dim=8,
                                  task_scope='complete_pick_carry_place_retreat'))
    if rows.ndim != 2 or rows.shape[1:] != (8,) or len(rows)<2 or not np.isfinite(rows).all():
        raise ValueError('Finite replay reference[N,8] required')
    delta = abs(np.diff(rows,axis=0))
    if (np.any(rows[:,:7]<guard.lower) or np.any(rows[:,:7]>guard.upper)
            or np.any(rows[:,7]<0) or np.any(rows[:,7]>.095)
            or np.any(delta[:,:7]>.002001) or np.any(delta[:,7]>.000501)):
        raise ValueError('ACT replay exceeds original physical or per-point limits')
    return rows


def reference_at(initial,rows,elapsed):
    initial,rows=np.asarray(initial,float),np.asarray(rows,float)
    if elapsed<=0:return initial.copy()
    position=elapsed/.05
    i=int(position)
    if i>=len(rows):return rows[-1].copy()
    before=initial if i==0 else rows[i-1]
    return before+(rows[i]-before)*(position-i)


def verify_replay(root,metadata,receipt=None):
    root=root.resolve();path=root/'replay/manifest.json'
    manifest=json.loads(path.read_text())
    if manifest.get('mode')!=MODE or manifest.get('reference_period_s')!=.05:
        raise ValueError('Unknown replay contract')
    for key in ['checkpoint_sha256','stats_sha256','scene_sha256','training_config_sha256']:
        if not metadata.get(key) or metadata[key]!=manifest.get(key):
            raise ValueError('Replay/model artifact mismatch: '+key)
    trace=root/'replay/act_rollout.npz';result=root/'replay/simulation_result.json'
    if (hashlib.sha256(trace.read_bytes()).hexdigest()!=manifest['source_rollout_sha256']
            or hashlib.sha256(result.read_bytes()).hexdigest()!=manifest['simulation_result_sha256']):
        raise ValueError('ACT source trace or physical simulation result changed')
    physical=json.loads(result.read_text())
    if not (physical.get('success') is True and physical.get('physical_success') is True
            and physical.get('teacher_free') is True and physical.get('error') is None
            and physical.get('unexpected_arm_cup_contact_frames')==0
            and physical.get('constraint_attachment_used') is False):
        raise ValueError('Replay needs a successful teacher-free physical ACT rollout')
    with np.load(trace,allow_pickle=False) as data:
        rows=validate_reference(data['sent_targets'])
        if (len(rows)!=manifest['reference_points']
                or hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()!=manifest['reference_sha256']):
            raise ValueError('ACT reference array checksum mismatch')
    phases=manifest['phases']
    if ([p['name'] for p in phases]!=['approach','close','lift','carry_place','release','retreat']
            or phases[0]['start']!=0 or phases[-1]['end']!=len(rows)
            or any(p['end']<=p['start'] for p in phases)
            or any(a['end']!=b['start'] for a,b in zip(phases,phases[1:]))):
        raise ValueError('Complete nonoverlapping ACT replay phases required')
    if receipt is not None and not (receipt.get('execution_mode')==MODE
            and receipt.get('trace_sha256')==manifest['source_rollout_sha256']
            and receipt.get('manifest_sha256')==hashlib.sha256(path.read_bytes()).hexdigest()):
        raise PermissionError('Current ACT_REPLAY receipt must bind this exact replay trace and phases')
    return manifest,rows
