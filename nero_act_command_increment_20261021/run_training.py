"""Single fixed1000-update increment-head development trial; no hardware entry point."""
from pathlib import Path
import json,time
import torch,numpy as np
from torch.utils.data import default_collate
from pipeline import ROOT,PARENT,DATA,INITIALIZER,build_policy,prefix_targets,validate_order,validated_replay,state_hash,snapshot,require_unchanged
from training.history_dataset import HistoryReferenceDataset
from training.approach_preservation import ApproachPreservation
from model import increment_loss,PROFILE


def encode_runtime_features(policy,current,images):
    features=[]
    for i in range(len(current)):
        # Preserve the outer image stride as well as pixels; CUDA normalization uses it.
        _,hidden=policy.encode(current[i:i+1],images[i:i+1])
        features.append(hidden)
    return torch.cat(features)


def cache_dataset(policy,dataset,stats):
    unique={};indices=[]
    for i,(fi,j) in enumerate(dataset.samples):
        key=(fi,int(dataset.cache[fi]['starts'][j]))
        if key not in unique:unique[key]=len(indices);indices.append(i)
    mapping=torch.tensor([unique[(fi,int(dataset.cache[fi]['starts'][j]))] for fi,j in dataset.samples],dtype=torch.long)
    values={key:[] for key in ['hidden','current','target','padding','group']};max_delta=0.
    for left in range(0,len(indices),32):
        batch=default_collate([dataset[i] for i in indices[left:left+32]])
        q,image=batch['current'].cuda(),batch['image'].cuda()
        hidden=encode_runtime_features(policy,q,image)
        target,delta,valid=prefix_targets(q,batch['actions'].cuda(),batch['padding'].cuda(),stats)
        max_delta=max(max_delta,float(delta[valid].abs().max()))
        for key,value in dict(hidden=hidden[:,:2],current=q,target=target,padding=~valid,group=batch['gripper_group']).items():values[key].append(value.detach().cpu())
        if left%1024==0:print(json.dumps(dict(stage='caching_runtime_features',done=min(left+32,len(indices)),total=len(indices))),flush=True)
    cache={key:torch.cat(value) for key,value in values.items()};cache['mapping']=mapping
    cache['sample_ids']=[dataset.sample_ids[i] for i in indices];cache['maximum_teacher_delta_m']=max_delta*.001
    # Compare the actual deployed single-observation path, not merely two batch caches.
    differences=[]
    for j in sorted(set([0,len(indices)//2,len(indices)-1])):
        sample=dataset[indices[j]];_,h=policy.encode(sample['current'][None].cuda(),sample['image'][None].cuda())
        difference=float((h[0,:2].cpu()-cache['hidden'][j]).abs().max());differences.append(difference)
        if difference>1e-5:raise ValueError('Cached ACT features differ from runtime single-input features')
    cache['runtime_feature_differences']=differences
    return cache


def assess(head,cache):
    maxima=[];mse=0.;count=0
    with torch.no_grad():
        for left in range(0,len(cache['current']),512):
            h,q,target,padding=[cache[k][left:left+512].cuda() for k in ['hidden','current','target','padding']]
            pred=head(h,q);err=(pred-target).abs();valid=~padding
            maxima.append(float(err[valid].max()));mse+=float(err[valid].square().sum());count+=int(valid.sum())
    return dict(maximum_prefix_error_mm=max(maxima)*1000,rms_prefix_error_mm=(mse/count)**.5*1000,unique_states=len(cache['current']))


def main():
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    output=ROOT/'training/pilot_A'
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('Host CUDA required')
    tests=json.loads((ROOT/'evidence/tests_verified.json').read_text());require_unchanged(tests['source_sha256'])
    if not tests['complete'] or tests['failed']:raise ValueError('Green source-bound suite required')
    policy,stats=build_policy();frozen=state_hash(policy.base)
    manifest=json.loads((DATA/'manifest.json').read_text())
    if len(manifest['episodes'])!=68:raise ValueError('Original68 files required')
    source_files=[p for p in ROOT.glob('*.py')]+list((ROOT/'tests').glob('*.py'))+[ROOT/'design.md',ROOT/'implementation_plan.md',ROOT/'approved_scope.md']
    source_files += [p for folder in ['training','executor','simulation','calibration','vision','policy_source'] for p in (PARENT/folder).rglob('*.py')]
    source_files += [PARENT/'config/history_experiment.json',PARENT/'evidence/qualify_pilot.py',INITIALIZER,INITIALIZER.parent/'dataset_stats.pkl',DATA/'manifest.json',DATA/'dataset_stats.pkl',DATA/'scene_config.json',DATA/'batch_order_pilot.npy',DATA/'batch_order_manifest.json']
    replay_sources=validated_replay(policy.config['approach_preservation']['replay_path'])
    source_hashes=dict(snapshot(source_files),**replay_sources);data_hashes=snapshot([r['path'] for r in manifest['episodes']])
    if any(data_hashes[str(Path(r['path']).resolve())]!=r['sha256'] for r in manifest['episodes']):raise ValueError('Original data changed')
    caches={}
    for split in ['train','val']:
        dataset=HistoryReferenceDataset([Path(r['path']) for r in manifest['episodes'] if r['split']==split],stats,policy.config,weighted=split=='train')
        if split=='train':order=validate_order(dataset.sample_ids,json.loads((DATA/'batch_order_manifest.json').read_text())['pilot'])
        caches[split]=cache_dataset(policy,dataset,stats);print(json.dumps(dict(stage='features_ready',split=split,states=len(caches[split]['current']),samples=len(dataset),maximum_delta_m=caches[split]['maximum_teacher_delta_m'])),flush=True)
    require_unchanged(source_hashes);require_unchanged(data_hashes)
    output.mkdir();torch.save(caches,output/'features.pt')
    seal=dict(complete=True,source_sha256=source_hashes,data_sha256=data_hashes,cache_sha256=snapshot([output/'features.pt']),tests=tests,frozen_base_sha256=frozen,optimizer_updates_before_seal=0,hardware_commands_sent=False)
    (output/'seal.json').write_text(json.dumps(seal,indent=2)+'\n')
    train={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in caches['train'].items()}
    optimizer=torch.optim.AdamW(policy.head.parameters(),lr=.0003,weight_decay=0.)
    torch.manual_seed(42);started=time.monotonic();trace=[]
    for step,row in enumerate(order,1):
        indices=train['mapping'][torch.as_tensor(row[row>=0],device='cuda')]
        h,q,target,padding=[train[k][indices] for k in ['hidden','current','target','padding']]
        optimizer.zero_grad(set_to_none=True);loss=increment_loss(policy.head,h,q,target,padding)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite increment loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.head.parameters(),.1)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite head gradients')
        optimizer.step();trace.append(dict(update=step,loss=float(loss.detach()),gradient_norm=float(norm),lr=.0003))
        if step==1 or step%25==0:
            status=dict(stage='training',actual_updates=step,budget=1000,head_loss=float(loss.detach()),hardware_deployment_allowed=False)
            (output/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
    policy.eval()
    if state_hash(policy.base)!=frozen or any(p.grad is not None for p in policy.base.parameters()):raise RuntimeError('Frozen ACT changed')
    metrics={split:assess(policy.head,cache) for split,cache in caches.items()}
    keeper=ApproachPreservation(policy.config['approach_preservation'],stats,'cuda');replay=keeper.assess(policy)
    # Only arm preservation is a gate here; gripper semantics intentionally changed.
    if replay['samples']!=423 or replay['rms_arm_prefix_change_rad']>.0001 or replay['maximum_arm_prefix_change_rad']>.0003:raise RuntimeError('Arm preservation failed')
    checkpoint=output/'policy_last.ckpt'
    torch.save(dict(profile=PROFILE,head=policy.head.state_dict(),config=policy.config,actual_updates=1000,formal_epochs=0,stats=stats),checkpoint)
    torch.save(optimizer.state_dict(),output/'optimizer_last.pt')
    (output/'optimizer_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    require_unchanged(source_hashes);require_unchanged(data_hashes);require_unchanged(seal['cache_sha256'])
    proof=dict(complete=True,actual_updates=1000,formal_epochs=0,source_sha256=source_hashes,checkpoint_sha256=snapshot([checkpoint]),frozen_base_before=frozen,frozen_base_after=state_hash(policy.base),hardware_deployment_allowed=False)
    (output/'integrity.json').write_text(json.dumps(proof,indent=2)+'\n')
    summary=dict(complete=True,actual_updates=1000,formal_epochs=0,elapsed_training_s=time.monotonic()-started,metrics=metrics,arm_reference=replay,head_only=True,hardware_deployment_allowed=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(output/'status.json').write_text(json.dumps(dict(stage='training_complete_evaluation_pending',**summary),indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
