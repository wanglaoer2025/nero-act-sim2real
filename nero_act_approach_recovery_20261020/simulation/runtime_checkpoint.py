"""Versioned complete offline checkpoints; never restore real motion permission."""
import base64
import hashlib
import json
import random

import numpy as np
import torch

from simulation.snapshot import capture, restore

CONTROL_KEYS = {'contact','phases','monitor','attachment','support_pose','active_points',
                'global_physics_points','events','last_submission_sim_s','confirmed_sim_s',
                'scenario_rng_state','rejections','guard_diagnostic'}
OBSERVATION_KEYS = {'exposures','history','receipts','history_epoch','pending','last_observation_stamp'}


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('Object arrays cannot enter runtime checkpoint')
        return {'__array__':base64.b64encode(np.ascontiguousarray(value).tobytes()).decode(),
                'dtype':value.dtype.str,'shape':list(value.shape)}
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, tuple):
        return {'__tuple__':[_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k,str) for k in value):
            raise ValueError('String state keys required')
        return {k:_encode(v) for k,v in value.items()}
    if isinstance(value,float) and not np.isfinite(value):
        if np.isnan(value):raise ValueError('NaN runtime state')
        return {'__boundary__': '-infinity' if value<0 else 'infinity'}
    if value is None or isinstance(value,(str,bool,int,float)):
        return value
    raise ValueError('Unsupported runtime state type: '+type(value).__name__)


def _decode(value):
    if isinstance(value,list):return [_decode(v) for v in value]
    if not isinstance(value,dict):return value
    if '__array__' in value:
        dtype=np.dtype(value['dtype'])
        if dtype.hasobject:raise ValueError('Unsafe array dtype')
        return np.frombuffer(base64.b64decode(value['__array__'],validate=True),dtype=dtype).reshape(value['shape']).copy()
    if '__tuple__' in value:return tuple(_decode(v) for v in value['__tuple__'])
    if '__boundary__' in value:
        if value['__boundary__'] not in ('infinity','-infinity'):raise ValueError('Invalid time boundary')
        return float('inf') if value['__boundary__']=='infinity' else float('-inf')
    return {k:_decode(v) for k,v in value.items()}


def _digest(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _require_states(control,observation,bindings):
    if (bindings.get('scope')!='offline_simulation' or not CONTROL_KEYS<=control.keys()
            or not OBSERVATION_KEYS<=observation.keys()
            or type(control['active_points']) is not int or control['active_points']<0
            or type(control['global_physics_points']) is not int or control['global_physics_points']<0):
        raise ValueError('Complete offline simulation supervisor state required')


def resume_bindings(current,origin,*,allow_policy_transition=False,expected_parent_source_sha256=None):
    if not allow_policy_transition:return current
    if (current.get('scope')!='offline_simulation' or origin.get('scope')!='offline_simulation'
            or current.get('runtime_profile')!='history_common_v1'):
        raise ValueError('Policy transitions are explicitly offline and use the common history runtime')
    if origin.get('scene_sha256') is not None:
        if origin['scene_sha256']!=current.get('scene_sha256'):raise ValueError('Policy transition changed source physics scene')
    elif (origin.get('source_kind')!='teacher_interruption' or not expected_parent_source_sha256
            or expected_parent_source_sha256!=origin.get('parent_source_sha256')):
        raise ValueError('Teacher transition must bind the verified exact parent physical source')
    return origin


def capture_runtime(model,data,servo,previous,*,control_state:dict,observation_state:dict,bindings:dict)->dict:
    _require_states(control_state,observation_state,bindings)
    record=dict(schema='complete_offline_runtime_v1',bindings=_encode(bindings),
        physics=capture(model,data,servo,previous),control=_encode(control_state),
        observation=_encode(observation_state),
        random_state=_encode(dict(python=random.getstate(),numpy=np.random.get_state(),
                                 torch_cpu=torch.random.get_rng_state().numpy())))
    # Evaluation has no stochastic CUDA forward, but preserve initialized CUDA RNGs.
    if torch.cuda.is_initialized():
        record['cuda_random_state']=_encode([v.cpu().numpy() for v in torch.cuda.get_rng_state_all()])
    record['sha256']=_digest(record)
    return record


def restore_runtime(model,data,servo,record:dict,*,expected_bindings:dict)->tuple[np.ndarray,dict,dict]:
    unsigned={k:v for k,v in record.items() if k!='sha256'}
    if (record.get('schema')!='complete_offline_runtime_v1' or _digest(unsigned)!=record.get('sha256')
            or record.get('bindings')!=_encode(expected_bindings)):
        raise ValueError('Incomplete, changed or differently bound offline runtime checkpoint')
    control,observation=_decode(record['control']),_decode(record['observation'])
    _require_states(control,observation,expected_bindings)
    previous=restore(model,data,servo,record['physics'])
    rng=_decode(record['random_state'])
    random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    torch.random.set_rng_state(torch.from_numpy(rng['torch_cpu']))
    if 'cuda_random_state' in record:
        if not torch.cuda.is_initialized():raise ValueError('CUDA checkpoint requires matching initialized CUDA runtime')
        torch.cuda.set_rng_state_all([torch.from_numpy(x) for x in _decode(record['cuda_random_state'])])
    return previous,control,observation
