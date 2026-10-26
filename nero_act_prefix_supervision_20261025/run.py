"""Exactly two1000-update offline continuations of the immutable5K model."""
import hashlib
import copy
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import default_collate
from experiment import BASE,PrefixPolicy
from dataset import V2Dataset,balanced_indices

ROOT=Path(__file__).resolve().parent
CHECKPOINT=BASE/'training/formal/policy_5000.ckpt'
MANIFEST=BASE/'data/v2_contour_01/manifest.json'
UPDATES=1000
SEED=20261025


def seed_matched():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)


def configure_runtime():
    torch.set_num_threads(2)
    # ACT's mask cumsum lacks a strict CUDA certificate; do not reject it.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def restore_training(model,optimizer,checkpoint):
    model.load_state_dict(checkpoint['model'],strict=True)
    # AdamW's CPU step tensors otherwise alias and mutate the shared source.
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))


def claim_variant(root,variant):
    if variant not in ('A','B'):raise ValueError('Invalid variant')
    path=Path(root)/variant;path.mkdir(parents=True,exist_ok=False)
    return path


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def object_digest(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for item in v:visit(item)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()


def save_json(path,value):
    path=Path(path);tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def file_snapshot(path):
    s=Path(path).stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def verify_manifest_data(manifest,progress_path=None):
    expected={}
    for row in manifest['episodes']:
        for item in [row,*row.get('source_dependencies',[])]:
            name=str(Path(item['path']).resolve());digest=item['sha256']
            if name in expected and expected[name]!=digest:raise RuntimeError('Conflicting source hashes')
            expected[name]=digest
    records={};cache={};started=time.monotonic();hashed_bytes=0
    for name,digest in expected.items():
        before=file_snapshot(name);key=tuple(before)
        if key not in cache:
            cache[key]=digest_file(name);hashed_bytes+=before[2]
        if file_snapshot(name)!=before:raise RuntimeError('Data source changed during hash:'+name)
        if cache[key]!=digest:raise RuntimeError('Data hash mismatch:'+name)
        records[name]=dict(sha256=digest,snapshot=before)
        if progress_path and (len(records)%10==0 or len(records)==len(expected)):
            save_json(progress_path,dict(verified=len(records),total=len(expected),
                hashed_bytes=hashed_bytes,elapsed_s=time.monotonic()-started,optimizer_updates=0))
    return dict(complete=True,episodes=len(manifest['episodes']),files=records,
        hashed_bytes=hashed_bytes,elapsed_s=time.monotonic()-started,optimizer_updates=0)


def verify_data_snapshots(proof):
    for name,row in proof['files'].items():
        if file_snapshot(name)!=row['snapshot']:raise RuntimeError('Data source changed:'+name)


def verify_bindings(binding):
    for name,digest in binding.items():
        if digest_file(name)!=digest:raise RuntimeError('Sealed source changed:'+name)


def source_bindings():
    existing=json.loads((BASE/'evidence/implementation_manifest.json').read_text())['sha256']
    paths=[CHECKPOINT,MANIFEST,BASE/'evidence/implementation_manifest.json',
           BASE/'evidence/main_dataset_completion_verified.json',
           ROOT/'data_verification.json',
           *[ROOT/n for n in ['experiment.py','run.py','metrics.py','evaluate_comparison.py']],
           *[BASE/name for name in existing]]
    binding={str(p):digest_file(p) for p in paths}
    for name,digest in existing.items():
        if binding[str(BASE/name)]!=digest:raise RuntimeError('Existing v2 seal mismatch:'+name)
    return binding


def train_variant(variant,checkpoint,dataset,binding):
    out=claim_variant(ROOT/'training',variant)
    policy=PrefixPolicy(checkpoint['stats'],variant=variant,pretrained=False).cuda().train()
    optimizer=policy.optimizer();restore_training(policy,optimizer,checkpoint)
    policy.config=dict(checkpoint['config'])
    model_digest=object_digest(policy.state_dict());opt_digest=object_digest(optimizer.state_dict())
    if model_digest!=object_digest(checkpoint['model']) or opt_digest!=object_digest(checkpoint['optimizer']):
        raise RuntimeError('Model or optimizer initial state mismatch')
    seed_matched()
    initial=dict(variant=variant,model_sha256=model_digest,optimizer_sha256=opt_digest,rng_seed=SEED,
        source_updates=5000,additional_update_limit=UPDATES,restored_optimizer=True,
        source_checkpoint_sha256=binding[str(CHECKPOINT)],
        strict_bitwise_GPU_determinism_claimed=False,
        objective='original_masked_arm20' if variant=='A' else 'half_valid_prefix2_half_valid_suffix18',
        hardware_commands_sent=False)
    save_json(out/'initialization.json',initial)
    batches=hashlib.sha256();started=time.monotonic()
    for update in range(1,UPDATES+1):
        rows=balanced_indices(dataset,5000+update)
        batches.update(np.asarray(rows,dtype='<i8').tobytes())
        batch=default_collate([dataset[i] for i in rows])
        t={k:v.cuda() for k,v in batch.items() if isinstance(v,torch.Tensor)}
        optimizer.zero_grad(set_to_none=True)
        losses=policy(t['observation'],t['image'],t['target'],t['padding'])
        if not torch.isfinite(losses['loss']):raise RuntimeError('Nonfinite loss')
        if update==1:
            first=dict(prediction_sha256=object_digest(losses['prediction']),
                gripper=float(losses['gripper'].detach()),kl=float(losses['kl'].detach()))
            save_json(out/'first_forward.json',first)
            if variant=='B':
                reference=json.loads((ROOT/'training/A/first_forward.json').read_text())
                if first!=reference:raise RuntimeError('First stochastic forward differs between matched arms')
        losses['loss'].backward()
        norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if update==1 or update%100==0:
            verify_bindings(binding)
            verify_data_snapshots(json.loads((ROOT/'data_verification.json').read_text()))
            for name,snapshot in dataset.snapshots.items():
                from dataset import _snapshot
                if _snapshot(name,dataset.dependencies[name])!=snapshot:raise RuntimeError('Dataset source changed')
            status=dict(variant=variant,additional_updates=update,limit=UPDATES,total_updates=5000+update,
                losses={k:float(losses[k].detach()) for k in ['loss','arm','gripper','kl']},
                preclip_gradient_norm=float(norm),elapsed_s=time.monotonic()-started,
                batch_sequence_sha256=batches.hexdigest(),hardware_commands_sent=False)
            save_json(out/'status.json',status);print(json.dumps(status),flush=True)
    policy.eval()
    torch.save(dict(model=policy.state_dict(),optimizer=optimizer.state_dict(),config=policy.config,
        stats=checkpoint['stats'],updates=6000,additional_updates=1000,variant=variant,
        source_checkpoint_sha256=binding[str(CHECKPOINT)],hardware_deployment_allowed=False),out/'policy_6000.ckpt')
    save_json(out/'completion.json',dict(complete=True,variant=variant,additional_updates=1000,
        batch_sequence_sha256=batches.hexdigest(),checkpoint_sha256=digest_file(out/'policy_6000.ckpt'),
        full_task_passed=False,hardware_commands_sent=False))
    del policy,optimizer;torch.cuda.empty_cache()


def main():
    configure_runtime()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG before CUDA initialization')
    if not torch.cuda.is_available():raise RuntimeError('CUDA training required')
    (ROOT/'training').mkdir(exist_ok=False)
    save_json(ROOT/'execution.json',dict(stage='data_byte_verification',pid=os.getpid(),hardware_commands_sent=False))
    manifest=json.loads(MANIFEST.read_text())
    if len(manifest['episodes'])!=150:raise RuntimeError('Exact150 data manifest required')
    proof=verify_manifest_data(manifest,ROOT/'data_verification_progress.json')
    save_json(ROOT/'data_verification.json',proof)
    binding=source_bindings();save_json(ROOT/'source_binding.json',binding)
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    if checkpoint['updates']!=5000 or checkpoint.get('hardware_deployment_allowed') is not False:
        raise RuntimeError('Wrong source checkpoint')
    paths=[r['path'] for r in manifest['episodes'] if r['split']=='train']
    if len(paths)!=120 or paths!=checkpoint['stats']['source_paths']:
        raise RuntimeError('Exact120 train trajectories and source normalization required')
    dataset=V2Dataset(paths,checkpoint['stats'])
    save_json(ROOT/'data_snapshots.json',dataset.snapshots)
    save_json(ROOT/'execution.json',dict(stage='training',pid=os.getpid(),seed=SEED,
        total_additional_update_budget=2000,hardware_commands_sent=False))
    for variant in ['A','B']:train_variant(variant,checkpoint,dataset,binding)
    completions=[json.loads((ROOT/'training'/v/'completion.json').read_text()) for v in ['A','B']]
    initial=[json.loads((ROOT/'training'/v/'initialization.json').read_text()) for v in ['A','B']]
    for key in ['model_sha256','optimizer_sha256','rng_seed']:
        if initial[0][key]!=initial[1][key]:raise RuntimeError('Unmatched initialization:'+key)
    if completions[0]['batch_sequence_sha256']!=completions[1]['batch_sequence_sha256']:
        raise RuntimeError('Unmatched batch sequence')
    verify_bindings(binding)
    save_json(ROOT/'training_completion.json',dict(complete=True,additional_updates=2000,
        matched_initialization=True,matched_optimizer=True,matched_batches=True,variants=completions,
        full_task_passed=False,hardware_commands_sent=False))
    save_json(ROOT/'execution.json',dict(stage='evaluating',pid=os.getpid(),hardware_commands_sent=False))
    from evaluate_comparison import evaluate
    evaluate(binding)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save_json(ROOT/'failure.json',dict(error=type(exc).__name__+': '+str(exc),hardware_commands_sent=False))
        save_json(ROOT/'execution.json',dict(stage='failed',pid=os.getpid(),error=str(exc),hardware_commands_sent=False))
        raise
