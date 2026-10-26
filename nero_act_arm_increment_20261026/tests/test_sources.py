from pathlib import Path
import sys
import json
import hashlib
import pytest
import numpy as np
import h5py

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from support import reuse_data_proof,fit_delta_statistics


def test_reuse_requires_original_proof_hash_manifest_and_metadata(tmp_path):
    f=tmp_path/'episode';f.write_bytes(b'original')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    s=f.stat();snapshot=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'episodes':[{'path':str(f),'sha256':digest(f)}]}))
    proof=tmp_path/'proof.json';proof.write_text(json.dumps({'complete':True,'files':{str(f):{'sha256':digest(f),'snapshot':snapshot}}}))
    binding=tmp_path/'binding.json';binding.write_text(json.dumps({str(proof):digest(proof),str(manifest):digest(manifest)}))
    assert reuse_data_proof(proof,binding,manifest)['complete']
    f.write_bytes(b'modified')
    with pytest.raises(RuntimeError,match='snapshot'):reuse_data_proof(proof,binding,manifest)
    proof.write_text('{}')
    with pytest.raises(RuntimeError,match='binding'):reuse_data_proof(proof,binding,manifest)


def test_delta_stats_use_two_targets_from_same_current_not_future_feedback(tmp_path):
    p=tmp_path/'train.h5'
    with h5py.File(p,'w') as f:
        q=np.array([[0.]*7,[.01]*7,[.02]*7])
        f['observations/qpos']=q;f['action']=np.c_[q+.001,np.zeros(3)]
    s=fit_delta_statistics([p])
    # t0/t1/t2 first targets +.001; t0/t1 second targets +.011.
    assert s['samples']==5
    np.testing.assert_allclose(s['mean'],[.005]*7,atol=1e-8)
    assert min(s['std'])>=.0005
