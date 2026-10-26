"""Source-bound cached data proof and train-only causal increment statistics."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save_json(path,value):
    path=Path(path);temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def snapshot(path):
    s=Path(path).stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def reuse_data_proof(proof_path,binding_path,manifest_path):
    binding=json.loads(Path(binding_path).read_text())
    for path in [proof_path,manifest_path]:
        if digest_file(path)!=binding.get(str(Path(path).resolve())):raise RuntimeError('Prior binding mismatch:'+str(path))
    proof=json.loads(Path(proof_path).read_text());manifest=json.loads(Path(manifest_path).read_text())
    if proof.get('complete') is not True:raise RuntimeError('Prior data proof incomplete')
    expected={str(Path(item['path']).resolve()):item['sha256'] for row in manifest['episodes'] for item in [row,*row.get('source_dependencies',[])]}
    if expected!={name:row['sha256'] for name,row in proof['files'].items()}:raise RuntimeError('Prior data proof scope mismatch')
    verify_snapshots(proof)
    return dict(complete=True,files=proof['files'],verification_kind='prior_full_SHA_plus_unchanged_metadata_continuity',
                prior_proof_path=str(proof_path),prior_proof_sha256=digest_file(proof_path),
                prior_binding_sha256=digest_file(binding_path),manifest_sha256=digest_file(manifest_path),optimizer_updates=0)


def verify_snapshots(proof):
    for name,row in proof['files'].items():
        if snapshot(name)!=row['snapshot']:raise RuntimeError('Data snapshot changed:'+name)


def fit_delta_statistics(paths):
    offsets=[]
    for path in paths:
        with h5py.File(path) as f:
            q=f['observations/qpos'][:].astype(np.float64);a=f['action'][:,:7].astype(np.float64)
            offsets.extend([a-q,a[1:]-q[:-1]])
    if not offsets:raise ValueError('Training episodes required')
    values=np.concatenate(offsets)
    if not np.isfinite(values).all():raise ValueError('Nonfinite training increments')
    raw=values.std(0)
    return dict(mean=values.mean(0).tolist(),std=np.maximum(raw,.0005).tolist(),raw_std=raw.tolist(),
                floor_rad=.0005,samples=len(values),source_paths=[str(p) for p in paths],prefix_points=2,
                reference='one_current_feedback_per_query_not_future_state')


def object_digest(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()


def core_state(policy):
    return {k:v for k,v in policy.state_dict().items() if not k.startswith(('arm_readout.','delta_'))}


def verify_bindings(binding):
    for name,digest in binding.items():
        if digest_file(name)!=digest:raise RuntimeError('Source binding changed:'+name)
