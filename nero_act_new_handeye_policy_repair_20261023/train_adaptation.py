"""Exactly1000 offline ACT updates on complete new-camera trajectories."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
INITIALIZER = ROOT.parent/'nero_act_preservation_scope_20261023/training/pilot_A/policy_last.ckpt'
INITIALIZER_SHA = '9dd36d386903e3a65198d049a9244388e819024357e9dc12d5d05a1ae575c9a3'
PROFILE = 'new_handeye_complete_recovery_v1'
sys.path.insert(0,str(ROOT.parent/'nero_act_full_flow_repair_20261023'))
import train_repair as historical
from training.visual_dataset import VisualReferenceDataset,audit_visual_episode,validate_visual_groups
from training.approach_preservation import ApproachPreservation,validate_approach_replay
from training.pose_objective import CartesianPoseLoss,fk_spec_from_scene

_spec = importlib.util.spec_from_file_location('_new_camera_protocol',ROOT/'protocol.py')
protocol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(protocol)
_spec = importlib.util.spec_from_file_location('_all_group_preservation',ROOT.parent/'nero_act_preservation_scope_20261023/group_scope.py')
_scope = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scope)


def load_initial():
    proof = json.loads((INITIALIZER.parent/'integrity.json').read_text())
    historical.require_unchanged(proof['source_sha256'])
    historical.require_unchanged(proof['checkpoint_sha256'])
    if historical.snapshot([INITIALIZER])[str(INITIALIZER.resolve())]!=INITIALIZER_SHA:
        raise ValueError('Exact declared latest fixed-last initializer required')
    policy,stats,groups = historical.load_initializer()
    payload = torch.load(INITIALIZER,map_location='cpu',weights_only=False)
    if payload['config']!=policy.config or any(not np.array_equal(payload['stats'][k],stats[k]) for k in stats):
        raise ValueError('Keep paired original architecture and normalization')
    policy.load_state_dict(payload['model'],strict=True)
    return policy,stats,groups


def prepare_dataset(policy,stats):
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    if not manifest['complete'] or manifest['failed_ACT_actions_used_as_labels'] is not False:
        raise ValueError('Completed audited physical labels required')
    rows = manifest['episodes']
    train = [r for r in rows if r['split']=='train']
    val = [r for r in rows if r['split']=='val']
    split_audit = validate_visual_groups([Path(r['path']) for r in train],[Path(r['path']) for r in val])
    historical.require_unchanged(manifest['source_sha256'])
    historical.require_unchanged({r[k]:r[h] for r in rows for k,h in [('path','sha256'),('physical_path','physical_sha256')]})
    samples = []
    pools = {f'{kind}:{stage}':[] for kind in ['center','other'] for stage in protocol.STAGES}
    for i,row in enumerate(train):
        audit_visual_episode(row['path'])
        with h5py.File(row['path']) as f:
            valid = np.flatnonzero(f['observations/visual_valid'][:])
            stages = f['simulation/stage'].asstr()[:]
        for t in valid:
            if stages[t] not in protocol.STAGES:
                continue
            index = len(samples)
            samples.append((i,int(t)))
            pools[f'{"center" if row["role"]=="center" else "other"}:{stages[t]}'].append(index)
    paths = [Path(r['path']) for r in train]
    dataset = VisualReferenceDataset(paths,['camera_0'],stats,policy_config=policy.config,
        image_size=tuple(policy.config['image_size']),sample_starts=samples)
    batches = protocol.balanced_batches(pools)
    assert batches.shape==(1000,32)
    audit = dict(train_groups=split_audit['train_groups'],val_groups=split_audit['val_groups'],
        validation_rows_in_training=0,training_episodes=len(train),validation_episodes=len(val),
        samples=len(samples),pool_sizes={k:len(v) for k,v in pools.items()},updates=1000,batch_size=32,
        all_stage_center_fraction=.5,statistics_recomputed=False)
    (ROOT/'sampling_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return dataset,batches,rows


def load_candidate():
    path = ROOT/'training/pilot_A/policy_last.ckpt'
    proof = json.loads((path.parent/'integrity.json').read_text())
    historical.require_unchanged(proof['source_sha256'])
    historical.require_unchanged(proof['checkpoint_sha256'])
    payload = torch.load(path,map_location='cpu',weights_only=False)
    if proof['actual_updates']!=1000 or payload['profile']!=PROFILE or payload['actual_updates']!=1000:
        raise ValueError('Completed declared1000-update candidate required')
    policy,stats,_ = load_initial()
    if payload['config']!=policy.config or any(not np.array_equal(payload['stats'][k],stats[k]) for k in stats):
        raise ValueError('Unchanged model interface/statistics required')
    policy.load_state_dict(payload['model'],strict=True)
    return policy.requires_grad_(False).eval(),stats


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    output = ROOT/'training/pilot_A'
    output.mkdir(parents=True,exist_ok=False)
    policy,stats,groups = load_initial()
    dataset,batches,rows = prepare_dataset(policy,stats)
    np.save(output/'batch_order.npy',batches,allow_pickle=False)
    source = ROOT.parent/'nero_act_new_calibration_20261023'
    scene = json.loads((source/'scene_relocalized_after_return.json').read_text())
    files = list(ROOT.glob('*.py'))+[ROOT/'plan.md',ROOT/'data/manifest.json',ROOT/'data/protocol.json',
        ROOT/'sampling_audit.json',output/'batch_order.npy',INITIALIZER,INITIALIZER.parent/'integrity.json',
        source/'scene_relocalized_after_return.json']
    files += [Path(row[key]) for row in rows for key in ['path','physical_path']]
    files += [Path(m.__file__) for m in tuple(sys.modules.values()) if getattr(m,'__file__',None)
        and str(m.__file__).startswith(str(ROOT.parent)+'/') and str(m.__file__).endswith('.py')]
    hashes = historical.snapshot(files)
    hashes.update(validate_approach_replay(policy.config['approach_preservation']['replay_path']))
    keeper = ApproachPreservation(dict(policy.config['approach_preservation'],batch_size=4),stats,'cuda')
    pose_loss = CartesianPoseLoss(fk_spec_from_scene(scene),stats).cuda()
    before = {n:p.detach().cpu().clone() for n,p in policy.state_dict().items()}
    allowed = [n for n,p in policy.named_parameters() if p.requires_grad]
    assert sum(len(g['params']) for g in groups)==len(allowed)
    optimizer = torch.optim.AdamW(groups,weight_decay=0.)
    rates = [g['lr'] for g in groups]
    loader = DataLoader(dataset,batch_sampler=batches.tolist(),num_workers=2,
        multiprocessing_context='spawn',pin_memory=True)
    gradient_audits = []
    started = time.monotonic()
    for updates,(image,current,target,padding,_) in enumerate(loader,1):
        optimizer.zero_grad(set_to_none=True)
        prediction = policy(current.cuda(),image.cuda())
        task,parts = historical.executed_objective(prediction,target.cuda(),padding.cuda(),stats,pose_loss)
        preservation = keeper.loss(policy,updates-1)
        coefficient,audit = _scope.preservation_scale(task,preservation,groups)
        loss = task+coefficient*preservation
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite new-camera objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        for group,rate in zip(optimizer.param_groups,rates):
            group['lr'] = rate*(1-.9*(updates-1)/999)
        optimizer.step()
        if updates==1 or updates%100==0:
            status = dict(updates=updates,limit=1000,task=float(task.detach()),preservation=float(preservation.detach()),
                preservation_coefficient=coefficient,terms={k:float(v.detach()) for k,v in parts.items()},
                elapsed_s=time.monotonic()-started)
            gradient_audits.append(dict(update=updates,coefficient=coefficient,**audit))
            (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            (output/'gradient_scope_audit.json').write_text(json.dumps(gradient_audits,indent=2)+'\n')
            print(json.dumps(status),flush=True)
    assert updates==1000
    changed = [n for n,p in policy.state_dict().items() if not torch.equal(p.detach().cpu(),before[n])]
    if not set(changed)<=set(allowed):
        raise ValueError('Frozen visual/CVAE or runtime buffers changed')
    historical.require_unchanged(hashes)
    checkpoint = output/'policy_last.ckpt'
    torch.save(dict(profile=PROFILE,model=policy.state_dict(),stats=stats,config=policy.config,
        actual_updates=1000,optimizer=optimizer.state_dict(),hardware_deployment_allowed=False),checkpoint)
    policy.requires_grad_(False).eval()
    preservation = keeper.assess(policy)
    (output/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    proof = dict(complete=True,actual_updates=1000,trainable_parameters=allowed,changed_tensors=changed,
        frozen_parameters_unchanged=True,source_sha256=hashes,checkpoint_sha256=historical.snapshot([checkpoint]),
        initializer_sha256=INITIALIZER_SHA,architecture_stats_and_runtime_unchanged=True,
        optimizer_initialization='Fresh AdamW for newly declared camera/data adaptation',
        full_task_passed=False,hardware_commands_sent=False,hardware_deployment_allowed=False)
    (output/'integrity.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(training_complete=True,updates=1000,changed_tensors=len(changed),preservation=preservation)),flush=True)


if __name__ == '__main__':
    main()
