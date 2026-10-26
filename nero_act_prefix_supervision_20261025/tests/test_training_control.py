from pathlib import Path
import copy
import sys
import random
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import seed_matched,restore_training,claim_variant,configure_runtime,object_digest,verify_manifest_data,verify_data_snapshots


def test_restores_model_and_optimizer_moments_without_reinitialization():
    model=torch.nn.Linear(3,2)
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    model(torch.ones(1,3)).sum().backward();opt.step()
    cp={'model':model.state_dict(),'optimizer':opt.state_dict()}
    restored=torch.nn.Linear(3,2)
    opt2=torch.optim.AdamW(restored.parameters(),lr=.4)
    restore_training(restored,opt2,cp)
    for a,b in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert opt2.param_groups[0]['lr']==.003
    for s1,s2 in zip(opt.state.values(),opt2.state.values()):
        for k in s1:torch.testing.assert_close(s1[k],s2[k],rtol=0,atol=0)


def test_matched_rng_covers_python_numpy_torch():
    seed_matched();a=(random.random(),np.random.random(),torch.rand(3))
    seed_matched();b=(random.random(),np.random.random(),torch.rand(3))
    assert a[:2]==b[:2]
    torch.testing.assert_close(a[2],b[2],rtol=0,atol=0)


def test_experiment_output_is_claimed_once(tmp_path):
    path=claim_variant(tmp_path,'A')
    assert path.is_dir()
    with pytest.raises(FileExistsError):claim_variant(tmp_path,'A')
    with pytest.raises(ValueError):claim_variant(tmp_path,'../other')


def test_runtime_does_not_require_unsupported_cuda_cumsum_determinism():
    configure_runtime()
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_training_A_cannot_mutate_source_optimizer_or_B_start():
    source=torch.nn.Linear(3,2);opt=torch.optim.AdamW(source.parameters())
    source(torch.ones(1,3)).sum().backward();opt.step()
    cp=copy.deepcopy({'model':source.state_dict(),'optimizer':opt.state_dict()})
    before=object_digest(cp['optimizer'])
    a=torch.nn.Linear(3,2);oa=torch.optim.AdamW(a.parameters())
    restore_training(a,oa,cp)
    a(torch.ones(1,3)).sum().backward();oa.step()
    assert object_digest(cp['optimizer'])==before
    b=torch.nn.Linear(3,2);ob=torch.optim.AdamW(b.parameters())
    restore_training(b,ob,cp)
    assert object_digest(ob.state_dict())==before


def test_episode_and_dependency_bytes_are_verified_before_snapshot(tmp_path):
    import hashlib
    p=tmp_path/'episode.h5';dep=tmp_path/'pixels.h5'
    p.write_bytes(b'episode');dep.write_bytes(b'RGB')
    digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    manifest={'episodes':[dict(path=str(p),sha256=digest(p),source_dependencies=[dict(path=str(dep),sha256=digest(dep))])]}
    proof=verify_manifest_data(manifest)
    assert proof['complete'] and len(proof['files'])==2
    verify_data_snapshots(proof)
    dep.write_bytes(b'BAD')
    with pytest.raises(RuntimeError,match='source changed'):verify_data_snapshots(proof)
    with pytest.raises(RuntimeError,match='hash mismatch'):verify_manifest_data(manifest)
    dep.write_bytes(b'RGB');p.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='hash mismatch'):verify_manifest_data(manifest)
