"""Bounded A/B readout adaptation, exactly1000updates per coordinate."""
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import default_collate
from experiment import BASE,ArmPolicy
from dataset import V2Dataset,balanced_indices
from support import save_json,digest_file,object_digest,core_state,verify_bindings,verify_snapshots

ROOT=Path(__file__).resolve().parent
CHECKPOINT=BASE/'training/formal/policy_20000.ckpt'
MANIFEST=BASE/'data/v2_contour_01/manifest.json'
SEED=20261026
LIMIT=1000


def configure_runtime():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def seed_matched():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)


def claim_training(root):
    out=Path(root)/'training';out.mkdir(exist_ok=False)
    return out


def bindings():
    pre=json.loads((ROOT/'preflight/completion.json').read_text())
    for name,digest in pre['source_files_sha256'].items():
        if digest_file(ROOT/name)!=digest:raise RuntimeError('Zero-update source changed:'+name)
    if not pre['complete'] or pre['optimizer_updates']!=0 or pre['maximum_physical_arm_equivalence_error_rad']>1e-6:
        raise RuntimeError('Zero-update equivalence failed')
    actual=json.loads((ROOT/'preflight/actual_inputs.json').read_text())
    if not actual['complete'] or actual['optimizer_updates']!=0 or len(actual['records'])!=3:
        raise RuntimeError('Saved failing-context equivalence missing')
    if digest_file(CHECKPOINT)!=pre['source_checkpoint_sha256'] or digest_file(MANIFEST)!=pre['manifest_sha256']:
        raise RuntimeError('20Ksource/data changed')
    old=json.loads((BASE/'evidence/implementation_manifest.json').read_text())['sha256']
    paths=[CHECKPOINT,MANIFEST,BASE/'evidence/implementation_manifest.json',ROOT/'preflight/completion.json',ROOT/'preflight/actual_inputs.json',ROOT/'actual_inputs.py',
           *[ROOT/n for n in ['experiment.py','support.py','prepare.py','metrics.py','run.py','evaluate_increment.py','delta_stats.json','data_proof.json']],
           *[BASE/name for name in old]]
    result={str(p):digest_file(p) for p in paths}
    for name,digest in old.items():
        if result[str(BASE/name)]!=digest:raise RuntimeError('v2source changed:'+name)
    return result


def train_variant(variant,cp,ds,dataset,binding):
    out=ROOT/'training'/variant;out.mkdir(exist_ok=False)
    p=ArmPolicy(cp['stats'],ds,variant=variant,pretrained=False).cuda();p.load_source(cp);p.train()
    initial_core=object_digest(core_state(p))
    if initial_core!=object_digest(cp['model']):raise RuntimeError('Core initialization mismatch')
    opt=p.optimizer()
    if opt.state:raise RuntimeError('Old optimizer state was incorrectly reused')
    seed_matched()
    save_json(out/'initialization.json',dict(variant=variant,source_checkpoint_sha256=binding[str(CHECKPOINT)],
        frozen_core_sha256=initial_core,readout_shape=list(p.arm_readout.weight.shape),
        old_optimizer_restored=False,new_optimizer='AdamW_lr1e-4_wd1e-4',seed=SEED,
        physical_initial_equivalence_verified=True,strict_GPU_bitwise_determinism_claimed=False))
    import hashlib
    batches=hashlib.sha256();started=time.monotonic()
    for update in range(1,LIMIT+1):
        rows=balanced_indices(dataset,20000+update,seed=42)
        batches.update(np.asarray(rows,dtype='<i8').tobytes())
        batch=default_collate([dataset[i] for i in rows]);t={k:v.cuda() for k,v in batch.items() if isinstance(v,torch.Tensor)}
        opt.zero_grad(set_to_none=True)
        losses=p(t['observation'],t['image'],t['target'],t['padding'])
        if not torch.isfinite(losses['loss']):raise RuntimeError('Nonfinite loss')
        losses['loss'].backward()
        norm=torch.nn.utils.clip_grad_norm_(p.arm_readout.parameters(),1.,error_if_nonfinite=True);opt.step()
        if update==1 or update%100==0:
            verify_bindings(binding);verify_snapshots(json.loads((ROOT/'data_proof.json').read_text()))
            if object_digest(core_state(p))!=initial_core:raise RuntimeError('Frozen core mutated')
            status=dict(variant=variant,additional_updates=update,limit=LIMIT,readout_loss=float(losses['loss'].detach()),
                preclip_gradient_norm=float(norm),elapsed_s=time.monotonic()-started,
                batch_sequence_sha256=batches.hexdigest(),frozen_core_unchanged=True,hardware_commands_sent=False)
            save_json(out/'status.json',status);print(json.dumps(status),flush=True)
    torch.save(dict(model=p.state_dict(),optimizer=opt.state_dict(),config=p.config,stats=cp['stats'],delta_stats=ds,arm_variant=variant,
        source_updates=20000,additional_updates=LIMIT,hardware_deployment_allowed=False),out/'policy.ckpt')
    record=dict(complete=True,variant=variant,additional_updates=LIMIT,frozen_core_unchanged=True,
        batch_sequence_sha256=batches.hexdigest(),checkpoint_sha256=digest_file(out/'policy.ckpt'),hardware_commands_sent=False)
    save_json(out/'completion.json',record)
    del p,opt;torch.cuda.empty_cache()
    return record


def main():
    configure_runtime()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8') or not torch.cuda.is_available():
        raise RuntimeError('CUDA and configured CUBLAS required')
    claim_training(ROOT)
    binding=bindings();save_json(ROOT/'source_binding.json',binding)
    verify_snapshots(json.loads((ROOT/'data_proof.json').read_text()))
    cp=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    ds=json.loads((ROOT/'delta_stats.json').read_text());manifest=json.loads(MANIFEST.read_text())
    paths=[r['path'] for r in manifest['episodes'] if r['split']=='train']
    if paths!=cp['stats']['source_paths'] or paths!=ds['source_paths'] or len(paths)!=120:raise RuntimeError('Train data/stats mismatch')
    dataset=V2Dataset(paths,cp['stats'])
    save_json(ROOT/'execution.json',dict(stage='training',pid=os.getpid(),budget=2000,hardware_commands_sent=False))
    records=[train_variant(v,cp,ds,dataset,binding) for v in ['A','B']]
    if records[0]['batch_sequence_sha256']!=records[1]['batch_sequence_sha256']:raise RuntimeError('Different batches')
    save_json(ROOT/'training_completion.json',dict(complete=True,additional_updates=2000,
        physical_init_equivalent=True,frozen_core_unchanged=True,matched_batches=True,variants=records,hardware_commands_sent=False))
    save_json(ROOT/'execution.json',dict(stage='evaluating',pid=os.getpid(),hardware_commands_sent=False))
    from evaluate_increment import evaluate_comparison
    evaluate_comparison(binding)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save_json(ROOT/'execution.json',dict(stage='failed',error=type(exc).__name__+': '+str(exc),hardware_commands_sent=False))
        raise
