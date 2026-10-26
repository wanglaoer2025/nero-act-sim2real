"""One fixed1000-update experiment; full rollouts require the position-response gate."""
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from learning import ROOT,SOURCE,load_for_learning
from objective import executed_objective,preservation_scale
from response import evaluate_response
from pipeline import PARENT,snapshot,require_unchanged
from training.visual_dataset import VisualReferenceDataset,validate_visual_groups
from training.transition_sampling import critical_windows
from training.approach_preservation import ApproachPreservation,validate_approach_replay
from training.pose_objective import CartesianPoseLoss,fk_spec_from_scene

STAGES=('approach','descend','close','lift','carry','place','release','retreat')


def prepare_data(policy,stats):
    manifests=[ROOT/'data/manifest.json',SOURCE/'data/manifest.json',PARENT/'training_data/uniform_complete_v1/manifest.json']
    datasets=[json.loads(p.read_text()) for p in manifests]
    if not all(m['complete'] for m in datasets):raise ValueError('Complete audited data required')
    prefix=datasets[0];rows=[r for m in datasets for r in m['episodes'] if r['split']=='train']
    val=[r for r in prefix['episodes'] if r['split']=='val']
    require_unchanged({r['path']:r['sha256'] for r in rows+val})
    paths=[Path(r['path']) for r in rows];validate_visual_groups(paths,[Path(r['path']) for r in val])
    candidates=sorted(set(critical_windows(paths,20,command_context=True)))
    samples=[];pools={s:[] for s in STAGES};prefix_pool=[]
    cache=[]
    for path in paths:
        with h5py.File(path) as f:
            cache.append((f['observations/visual_valid'][:],f['simulation/stage'].asstr()[:]))
    for i,t in candidates:
        valid,stages=cache[i]
        if not valid[t]:continue
        stage=stages[t]
        if rows[i]['id']<3000 and stage not in STAGES:continue
        index=len(samples);samples.append((i,t))
        if rows[i]['id']>=3000:prefix_pool.append(index)
        else:pools[stage].append(index)
    if not prefix_pool or any(not p for p in pools.values()):raise ValueError('All task stages and prefix interventions required')
    rng=np.random.default_rng(42)
    batches=np.array([np.r_[rng.choice(prefix_pool,16),*[rng.choice(p,2) for p in pools.values()]] for _ in range(1000)])
    dataset=VisualReferenceDataset(paths,['camera_0'],stats,policy_config=policy.config,
        image_size=tuple(policy.config['image_size']),sample_starts=samples)
    with h5py.File(paths[0]) as f:scene=json.loads(f.attrs['scene_config'])
    protected=manifests+paths+[Path(r['path']) for r in val]
    protected += [Path(r['physical_path']) for r in prefix['episodes']]
    return dataset,batches,prefix,scene,protected


def main():
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    output=ROOT/'training/pilot_A';output.mkdir(parents=True,exist_ok=False)
    policy,stats,groups=load_for_learning()
    dataset,batches,manifest,scene,files=prepare_data(policy,stats)
    np.save(output/'batch_order.npy',batches,allow_pickle=False)
    files+=list(ROOT.glob('*.py'))+[ROOT/'plan.json',ROOT/'data/pose_protocol.json',ROOT/'data/scene_protocol.json',output/'batch_order.npy',
        SOURCE/'training/pilot_A/policy_last.ckpt',SOURCE/'training/pilot_A/integrity.json']
    files += [Path(m.__file__) for m in tuple(sys.modules.values()) if getattr(m,'__file__',None)
        and str(m.__file__).startswith(str(ROOT.parent)+'/') and str(m.__file__).endswith('.py')]
    hashes=snapshot(files)
    hashes.update(validate_approach_replay(policy.config['approach_preservation']['replay_path']))
    keeper=ApproachPreservation(dict(policy.config['approach_preservation'],batch_size=4),stats,'cuda')
    pose_loss=CartesianPoseLoss(fk_spec_from_scene(scene),stats).cuda()
    before=evaluate_response(policy,stats,manifest)
    (output/'response_before.json').write_text(json.dumps(before,indent=2)+'\n')
    print(json.dumps(dict(stage='response_before',passed=before['passed'],direction=before['direction_pass_fraction'],error_m=before['maximum_prefix_position_error_m'])),flush=True)
    frozen={n:p.detach().cpu().clone() for n,p in policy.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW(groups,weight_decay=0.)
    rates=[g['lr'] for g in groups];started=time.monotonic()
    loader=DataLoader(dataset,batch_sampler=batches.tolist(),num_workers=2,multiprocessing_context='spawn',pin_memory=True)
    updates=0
    for updates,(image,current,target,padding,_) in enumerate(loader,1):
        optimizer.zero_grad(set_to_none=True)
        prediction=policy(current.cuda(),image.cuda())
        task,parts=executed_objective(prediction,target.cuda(),padding.cuda(),stats,pose_loss)
        preservation=keeper.loss(policy,updates-1)
        weight,norms=preservation_scale(task,preservation,policy.base.model.input_proj_robot_state.parameters())
        loss=task+weight*preservation
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate*(1-.9*(updates-1)/999)
        optimizer.step()
        if updates==1 or updates%100==0:
            record=dict(updates=updates,limit=1000,task=float(task.detach()),preservation=float(preservation.detach()),
                preservation_weight=weight,weighted_preservation_gradient_fraction=weight*norms['preservation_gradient_norm']/max(norms['task_gradient_norm'],1e-12),
                terms={k:float(v.detach()) for k,v in parts.items()},elapsed_s=time.monotonic()-started)
            (output/'status.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    if updates!=1000:raise ValueError('Exactly1000 updates required')
    if any(not torch.equal(p.detach().cpu(),frozen[n]) for n,p in policy.named_parameters() if n in frozen):
        raise ValueError('Frozen visual/CVAE parameters changed')
    require_unchanged(hashes)
    checkpoint=output/'policy_last.ckpt'
    torch.save(dict(profile='position_response_loss_repair_v1',model=policy.state_dict(),stats=stats,
        config=policy.config,actual_updates=1000,hardware_deployment_allowed=False),checkpoint)
    policy.requires_grad_(False).eval()
    after=evaluate_response(policy,stats,manifest,checkpoint_sha256=snapshot([checkpoint])[str(checkpoint.resolve())])
    (output/'response_gate.json').write_text(json.dumps(after,indent=2)+'\n')
    proof=dict(complete=True,actual_updates=1000,formal_epochs=0,initializer_new_updates=2000,
        cumulative_new_optimizer_updates=3000,source_sha256=hashes,
        checkpoint_sha256=snapshot([checkpoint,output/'response_gate.json']),
        frozen_parameters_unchanged=True,position_response_gate_passed=after['passed'],
        full_task_passed=False,hardware_commands_sent=False,hardware_deployment_allowed=False)
    (output/'integrity.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(stage='training_complete',updates=1000,response_passed=after['passed'],
        direction=after['direction_pass_fraction'],error_m=after['maximum_prefix_position_error_m'])),flush=True)


if __name__=='__main__':main()
