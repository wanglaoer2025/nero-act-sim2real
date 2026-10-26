"""Paired single-frame/history ACT with explicit pilot and formal budgets."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from training.integrity import snapshot,require_unchanged
from training.rebase_initializer import paired_config,rebase_initializer
from training.train import FullTaskPolicy

ROOT=Path(__file__).resolve().parents[1]
PROFILE='causal_numeric_gru5_v1'


def validate_budget(budget):
    if (not isinstance(budget,dict) or set(budget)!={'kind','limit'}
            or type(budget['limit']) is not int
            or (budget['kind'],budget['limit']) not in [('updates',1000),('epochs',20)]):
        raise ValueError('Declared1000-update pilot or20-epoch formal budget required')
    return budget


def selected_checkpoint_name(budget):
    validate_budget(budget)
    return 'policy_last.ckpt' if budget['kind']=='updates' else 'policy_best_val.ckpt'


def make_batch_order(size,*,batch_size,budget,seed,critical_indices=None):
    validate_budget(budget)
    if type(size) is not int or size<=0 or batch_size!=32 or seed!=42:
        raise ValueError('Common nonempty samples, batch32 and seed42 required')
    rng=np.random.default_rng(seed);rows=[]
    cycles=budget['limit'] if budget['kind']=='epochs' else (budget['limit']+(size+31)//32-1)//((size+31)//32)
    for _ in range(cycles):
        indices=rng.permutation(size)
        for start in range(0,size,32):
            part=indices[start:start+32];rows.append(np.pad(part,(0,32-len(part)),constant_values=-1))
    result=np.asarray(rows[:budget['limit']] if budget['kind']=='updates' else rows,dtype=np.int64)
    if critical_indices is not None:
        pool=np.asarray(critical_indices)
        if pool.ndim!=1 or not len(pool) or pool.dtype.kind not in 'iu' or np.any((pool<0)|(pool>=size)):
            raise ValueError('Nonempty training sample indices required')
        result[:,:8]=np.random.default_rng(seed).choice(pool,size=(len(result),8),replace=True)
    return result


def transition_pool(dataset):
    """Offline-only selection: five earliest valid starts per training recovery."""
    import h5py
    first={}
    for index,path in enumerate(dataset.paths):
        with h5py.File(path,'r') as f:
            if 'recovery_role' not in f.attrs:continue
            if f.attrs['split']!='train':raise ValueError('Transition pool must use training recoveries only')
        first[index]=set(sorted(set(map(int,dataset.cache[index]['starts'])))[:5])
    pool=[i for i,(file,j) in enumerate(dataset.samples) if file in first and int(dataset.cache[file]['starts'][j]) in first[file]]
    if not pool:raise ValueError('No valid training transition samples')
    return np.asarray(pool,dtype=np.int64)


def initialize_history(source,old_stats,new_stats,overrides,*,variant):
    if variant not in ('A','B') or source['config'].get('policy_input_dim')!=15:
        raise ValueError('Matched15→15 initializer and explicit A/B variant required')
    config=copy.deepcopy(source['config'])
    config.update(state_dim=8,physical_state_dim=8,policy_input_dim=15,cvae_training=True,
        nonlinear_gripper_head=True,train_nonlinear_gripper_only=False,train_arm_output_and_gripper_only=False,
        train_gripper_head_only=False,reset_action_head=False,freeze_backbone=True,
        lr=1e-5,lr_backbone=1e-5,gripper_adapter_lr=3e-4,history_lr=1e-4,
        kl_weight=.001,prior_gripper_loss_weight=.005,pose_loss_weight=.05,prefix_margin_loss_weight=0.,
        training_observation_noise_normalized=[0.]*8,numeric_history_profile=PROFILE if variant=='B' else 'none',
        history_runtime_profile='history_common_v1',training_revision='history_feedback_preimage_v1',
        prior_gripper_target_profile='feedback_preimage_v1',
        training_integrity_required=True,hardware_deployment_allowed=False,
        checkpoint_selection='fixed_last_update' if overrides.get('training_budget',{}).get('kind')=='updates' else 'minimum_zero_latent_validation_total_loss')
    config.update(overrides);config=paired_config(config,new_stats)
    rebased=rebase_initializer(source['model'],old_stats,new_stats,config)
    if variant=='B':
        # Model initialization must not alter the shared sampling/noise random streams.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42);target=FullTaskPolicy(dict(config,device='cpu',pretrained_backbone=False))
        extra={k:v for k,v in target.state_dict().items() if k.startswith('history_encoder.')}
        if set(target.state_dict())-set(rebased)!=set(extra) or len(extra)!=6:
            raise ValueError('Only the exact new GRU/projection keys may be initialized')
        rebased.update(extra)
    return config,rebased


def build_history_policy(config):
    if (config.get('policy_input_dim')!=15 or config.get('cvae_training') is not True
            or any(config.get(k) for k in ['train_nonlinear_gripper_only','train_arm_output_and_gripper_only','train_gripper_head_only'])):
        raise ValueError('Joint ACT/CVAE/arm/gripper training required')
    policy=FullTaskPolicy(dict(config,freeze_backbone=True))
    groups={key:[] for key in ['policy','visual','gripper','history']}
    for name,p in policy.named_parameters():
        if name.startswith('model.backbones.'):
            p.requires_grad_('.layer4.' in name);group='visual'
        elif name.startswith('gripper_adapter.'):group='gripper'
        elif name.startswith('history_encoder.'):group='history'
        else:group='policy'
        if p.requires_grad:groups[group].append(p)
    if not all(groups[k] for k in ['policy','visual','gripper']):raise ValueError('Missing required trainable parameter group')
    rates=dict(policy=1e-5,visual=1e-5,gripper=3e-4,history=1e-4)
    policy.optimizer=torch.optim.AdamW([dict(name=k,params=params,lr=rates[k],
        weight_decay=0. if k=='gripper' else float(config.get('weight_decay',0.))) for k,params in groups.items() if params])
    return policy


def finalize_history_training(output,config,counters,protected):
    budget=validate_budget(config['training_budget']);output=Path(output)
    if (not protected or counters.get(budget['kind'])!=budget['limit']
            or (config.get('declared_optimizer_steps') is not None and counters.get('updates')!=config['declared_optimizer_steps'])
            or (budget['kind']=='updates' and config.get('num_epochs')!=0)
            or (budget['kind']=='epochs' and config.get('num_epochs')!=20)):
        raise ValueError('Every declared update/epoch must finish under the correct budget')
    require_unchanged(protected)
    checkpoints=list((output/'checkpoints').glob('policy*.ckpt'))
    if not checkpoints or not (output/'checkpoints'/selected_checkpoint_name(budget)).is_file():
        raise ValueError('Declared selected checkpoint missing')
    proof=dict(schema='history_training_integrity_v1',finalized=True,declared_budget=budget,
        actual_updates=counters['updates'],actual_epochs=counters['epochs'],protected_sha256=protected,
        output_sha256=snapshot(checkpoints+[output/'checkpoints/dataset_stats.pkl']),hardware_deployment_allowed=False)
    (output/'artifacts/training_integrity.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


def freeze_history_protocol(source:Path,runtime_contract:dict,output:Path)->dict:
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError(output)
    original=json.loads(source.read_text());cases=original.get('resolved_cases',[])
    if (original.get('parameters_frozen_before_optimization') is not True or len(cases)!=16
            or len({r.get('name') for r in cases})!=16
            or any(not isinstance(r.get('scene'),dict) or not r['scene'] or type(r.get('seed')) is not int
                   or 'query_stride' not in r or r.get('observation_delay_steps') not in (0,1) for r in cases)):
        raise ValueError('All16 original numeric scenes and actual delay settings must already be frozen')
    result=copy.deepcopy(original)
    result.update(schema='history_common_numeric16_v2',origin_protocol_path=str(source.resolve()),
        origin_protocol_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        history_runtime_contract=copy.deepcopy(runtime_contract),
        history_runtime_contract_sha256=hashlib.sha256(json.dumps(runtime_contract,sort_keys=True).encode()).hexdigest())
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def _require_seal(data_dir):
    seal=ROOT/'evidence/pre_optimization_freeze.json'
    if not seal.is_file():raise ValueError('Pre-optimization source/data seal required before training')
    record=json.loads(seal.read_text())
    if (record.get('schema')!='history_pre_optimization_v1' or record.get('complete') is not True
            or record.get('data_dir')!=str(Path(data_dir).resolve())
            or not record.get('source_sha256') or not record.get('data_sha256')):
        raise ValueError('Incomplete or mismatched pre-optimization freeze')
    require_unchanged(record['source_sha256']);require_unchanged(record['data_sha256'])
    return record


def prepare_batch_orders(data_dir:Path,config:dict)->dict:
    """Called at Task9 sealing, not by either optimizer; both variants read these files."""
    from training.history_dataset import HistoryReferenceDataset
    import pickle
    data_dir=Path(data_dir);manifest=json.loads((data_dir/'manifest.json').read_text())
    stats=pickle.loads((data_dir/'dataset_stats.pkl').read_bytes())
    paths=[Path(r['path']) for r in manifest['episodes'] if r['split']=='train']
    dataset=HistoryReferenceDataset(paths,stats,config,weighted=True)
    ids_sha=hashlib.sha256(json.dumps(dataset.sample_ids,separators=(',',':')).encode()).hexdigest()
    records={}
    profile=config.get('transition_sampling_profile','none')
    if profile not in ('none','recovery_first5_quarter_v1'):raise ValueError('Unknown transition sampling profile')
    pool=transition_pool(dataset) if profile!='none' else None
    for stage,budget in [('pilot',dict(kind='updates',limit=1000)),('formal',dict(kind='epochs',limit=20))]:
        order=make_batch_order(len(dataset),batch_size=32,budget=budget,seed=42,critical_indices=pool if stage=='pilot' else None)
        path=data_dir/f'batch_order_{stage}.npy'
        if path.exists():raise FileExistsError(path)
        np.save(path,order,allow_pickle=False)
        records[stage]=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            batch_count=len(order),sample_ids_sha256=ids_sha,weighted_training_samples=len(dataset),budget=budget)
        if stage=='pilot' and pool is not None:
            records[stage].update(transition_sampling_profile=profile,critical_pool_indices=pool.tolist(),
                critical_slots_per_batch=8,selection_uses_training_data_only=True)
    (data_dir/'batch_order_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    return records


def _batch_to_device(batch,device,*,with_history):
    result=[batch[k].to(device) for k in ['current','image','actions','padding']]
    kwargs={'history':{k:v.to(device) for k,v in batch['history'].items()}} if with_history else {}
    return result,kwargs


def history_epoch(policy,loader,device,*,training,counters,output=None,epoch=0):
    from training.visual import gradient_evidence
    policy.train(training);totals={};seen=0
    for batch in loader:
        values,kwargs=_batch_to_device(batch,device,with_history=policy.numeric_history_profile!='none')
        with torch.set_grad_enabled(training):
            losses=policy(*values,**kwargs)
            if not torch.isfinite(losses['loss']):raise RuntimeError('Nonfinite history ACT loss')
            if training:
                policy.optimizer.zero_grad(set_to_none=True);losses['loss'].backward()
                step=counters['updates']+1
                if step in (1,2):
                    proof=gradient_evidence(policy)
                    if hasattr(policy,'history_encoder'):
                        norm=lambda p:float(p.grad.detach().norm()) if p.grad is not None else 0.
                        proof.update(history_projection_grad_norm=norm(policy.history_encoder.projection.weight),
                                     history_GRU_grad_norm=norm(policy.history_encoder.gru.weight_ih_l0))
                        if proof['history_projection_grad_norm']<=0 or (step==2 and proof['history_GRU_grad_norm']<=0):
                            raise RuntimeError('History branch lacks required two-step gradients')
                    if output is not None:(output/f'artifacts/gradient_step_{step}.json').write_text(json.dumps(proof,indent=2))
                norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),.1)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite history ACT gradients')
                policy.optimizer.step();counters['updates']=step
        size=len(values[0]);seen+=size
        for key,value in losses.items():
            if key!='a_hat':totals[key]=totals.get(key,0.)+float(value.detach())*size
        if training and output is not None and (counters['updates']==1 or counters['updates']%25==0):
            status=dict(stage='optimizing',epoch=epoch,actual_updates=counters['updates'],
                training_budget=policy.training_budget,loss=float(losses['loss'].detach()),hardware_deployment_allowed=False)
            (output/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
    if not seen:raise ValueError('Empty history training/validation loader')
    return {k:v/seen for k,v in totals.items()}


def train_history(data_dir:Path,initializer:Path,config:dict,output:Path)->dict:
    import pickle
    import time
    from torch.utils.data import DataLoader
    from training.history_dataset import HistoryReferenceDataset
    from training.train import save_checkpoint
    from training.pose_objective import fk_spec_from_scene
    from executor.model import BoundedActPolicy,make_policy_observation,physical_actions
    from training.integrity import require_finalized
    data_dir,initializer,output=[Path(p).resolve() for p in [data_dir,initializer,output]]
    sealed=_require_seal(data_dir)
    if output.exists():raise FileExistsError(output)
    settings=dict(config);variant=settings.pop('variant');stage=settings.pop('stage')
    if variant not in ('A','B') or stage not in ('pilot','formal'):raise ValueError('Explicit paired stage/variant required')
    if stage=='formal' and settings.get('transition_sampling_profile','none')!='none':
        raise ValueError('Balanced contact repair is pilot-only; formal budget requires a separately declared experiment')
    gate_inputs=[]
    if stage=='formal':
        from executor.history_qualification import require_formal_gate
        gate_inputs=require_formal_gate(ROOT)
    budget=dict(kind='updates',limit=1000) if stage=='pilot' else dict(kind='epochs',limit=20)
    validate_budget(budget)
    if (settings.get('seed')!=42 or settings.get('batch_size')!=32 or settings.get('image_size')!=[180,320]
            or settings.get('device')!='cuda'):
        raise ValueError('Approved seed42/batch32/320×180 CUDA experiment required')
    if not torch.cuda.is_available():raise RuntimeError('Host CUDA training runtime required')
    if hashlib.sha256(initializer.read_bytes()).hexdigest()!='b252dd578b4bf5626566ec5b60a47be0bdcfee30e9eab990cde05ac122bdea71':
        raise ValueError('The common frozen15D initialization differs')
    manifest=json.loads((data_dir/'manifest.json').read_text());accepted=json.loads((data_dir/'data_acceptance.json').read_text())
    if (not manifest.get('complete') or not accepted.get('complete') or len(manifest.get('episodes',[]))!=66
            or accepted.get('training_groups')!=37 or accepted.get('validation_groups')!=12
            or accepted.get('manifest_sha256')!=hashlib.sha256((data_dir/'manifest.json').read_bytes()).hexdigest()):
        raise ValueError('Complete verified48+18 common history data required')
    stats=pickle.loads((data_dir/'dataset_stats.pkl').read_bytes())
    source=torch.load(initializer,map_location='cpu',weights_only=False);require_finalized(initializer,source['config'])
    old_stats=pickle.loads((initializer.parent/'dataset_stats.pkl').read_bytes())
    scene=json.loads((data_dir/'scene_config.json').read_text())
    settings.update(training_budget=budget,num_epochs=0 if stage=='pilot' else 20,
        scene_config=scene,pose_kinematics_spec=fk_spec_from_scene(scene),calibration_sha256=manifest['calibration_sha256'],
        dataset_dir=str(data_dir),variant=variant,stage=stage)
    paired,weights=initialize_history(source,old_stats,stats,settings,variant=variant)
    datasets={key:HistoryReferenceDataset([Path(r['path']) for r in manifest['episodes'] if r['split']==key],stats,paired,weighted=key=='train')
              for key in ['train','val']}
    schedule=json.loads((data_dir/'batch_order_manifest.json').read_text())[stage]
    ids_sha=hashlib.sha256(json.dumps(datasets['train'].sample_ids,separators=(',',':')).encode()).hexdigest()
    if ids_sha!=schedule['sample_ids_sha256']:raise ValueError('A/B current samples or order changed')
    order_path=Path(schedule['path']);order=np.load(order_path,allow_pickle=False)
    profile=paired.get('transition_sampling_profile','none')
    if profile not in ('none','recovery_first5_quarter_v1'):raise ValueError('Unknown transition sampling profile')
    pool=transition_pool(datasets['train']) if profile!='none' else None
    if pool is not None and (schedule.get('transition_sampling_profile')!=profile
            or schedule.get('critical_pool_indices')!=pool.tolist()):raise ValueError('Frozen transition pool changed')
    if (hashlib.sha256(order_path.read_bytes()).hexdigest()!=schedule['sha256']
            or not np.array_equal(order,make_batch_order(len(datasets['train']),batch_size=32,budget=budget,seed=42,critical_indices=pool))):
        raise ValueError('Frozen common batch schedule changed')
    paired.update(declared_optimizer_steps=len(order),batch_order_sha256=schedule['sha256'],sample_ids_sha256=ids_sha,
        checkpoint_dir=str(output/'checkpoints'),artifact_dir=str(output/'artifacts'))
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    device=torch.device('cuda');policy=build_history_policy(paired).to(device);policy.load_state_dict(weights,strict=True)
    policy.training_budget=budget
    # Verify actual current training inputs under their old/new paired normalization.
    old_policy=BoundedActPolicy(dict(source['config'],device='cuda',pretrained_backbone=False)).cuda().eval()
    old_policy.load_state_dict(source['model'],strict=True);policy.eval();maximum=np.zeros(2)
    for index in range(min(4,len(datasets['train']))):
        sample=datasets['train'][index];current=sample['current'].numpy()
        measured=current[:8]*stats['qpos_std']+stats['qpos_mean']
        features=current[9:]*stats['geometry_std']+stats['geometry_mean']
        previous=float(measured[7]+current[8]*.01)
        before=make_policy_observation(measured,old_stats,source['config'],previous,geometry_features_m=features)
        image=sample['image'][None].to(device);kwargs={}
        if variant=='B':kwargs['history']={k:v[None].to(device) for k,v in sample['history'].items()}
        with torch.inference_mode():
            a=physical_actions(old_policy(torch.from_numpy(before)[None].to(device),image),old_stats,bounded_gripper=True)
            b=physical_actions(policy(sample['current'][None].to(device),image,**kwargs),stats,bounded_gripper=True)
        maximum=np.maximum(maximum,[float((a[...,:7]-b[...,:7]).abs().max()),float((a[...,7]-b[...,7]).abs().max())])
    del old_policy
    if maximum[0]>1e-6 or maximum[1]>1e-7:raise ValueError('Initial physical policy was not preserved: '+str(maximum))
    output.mkdir(parents=True);(output/'checkpoints').mkdir();(output/'artifacts').mkdir()
    (output/'config.json').write_text(json.dumps(paired,indent=2)+'\n')
    (output/'checkpoints/dataset_stats.pkl').write_bytes((data_dir/'dataset_stats.pkl').read_bytes())
    (output/'artifacts/initial_equivalence.json').write_text(json.dumps(dict(maximum_joint_rad=maximum[0],maximum_gripper_m=maximum[1]),indent=2))
    protected=dict(sealed['source_sha256'],**sealed['data_sha256'])
    protected.update(snapshot([ROOT/'evidence/pre_optimization_freeze.json',initializer,initializer.parent/'dataset_stats.pkl',output/'config.json',output/'checkpoints/dataset_stats.pkl']))
    protected.update(snapshot(gate_inputs))
    (output/'artifacts/training_inputs_before.json').write_text(json.dumps(protected,indent=2))
    generator=torch.Generator().manual_seed(42)
    loader_kwargs=dict(num_workers=int(paired.get('data_workers',2)),pin_memory=True,generator=generator)
    val_loader=DataLoader(datasets['val'],batch_size=32,shuffle=False,**loader_kwargs)
    counters=dict(updates=0,epochs=0);metrics=[];best=float('inf');started=time.monotonic()
    per_epoch=(len(datasets['train'])+31)//32
    ranges=[(0,len(order))] if stage=='pilot' else [(i*per_epoch,(i+1)*per_epoch) for i in range(20)]
    split=json.loads((data_dir/'split.json').read_text())
    for iteration,(left,right) in enumerate(ranges,1):
        epoch=0 if stage=='pilot' else iteration
        loader=DataLoader(datasets['train'],batch_sampler=[row[row>=0].tolist() for row in order[left:right]],**loader_kwargs)
        train=history_epoch(policy,loader,device,training=True,counters=counters,output=output,epoch=epoch)
        with torch.random.fork_rng(devices=[device.index or 0]):
            torch.manual_seed(42)
            val=history_epoch(policy,val_loader,device,training=False,counters=counters)
        if stage=='formal':counters['epochs']=epoch
        row=dict(epoch=epoch,actual_updates=counters['updates'],train=train,val=val,elapsed_s=time.monotonic()-started)
        if stage=='formal':save_checkpoint(output/f'checkpoints/policy_epoch_{epoch:02d}.ckpt',policy,paired,split,epoch,row)
        metrics.append(row);save_checkpoint(output/'checkpoints/policy_last.ckpt',policy,paired,split,epoch,row)
        if stage=='formal' and val['loss']<best:
            best=val['loss'];save_checkpoint(output/'checkpoints/policy_best_val.ckpt',policy,paired,split,epoch,row)
        (output/'artifacts/metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(row),flush=True)
    if counters['updates']!=len(order):raise ValueError('Declared batch schedule was not completed')
    finalize_history_training(output,paired,counters,protected)
    summary=dict(complete=True,training_budget=budget,actual_updates=counters['updates'],actual_epochs=counters['epochs'],
        selected_checkpoint=selected_checkpoint_name(budget),last_metrics=metrics[-1],
        best_zero_latent_val_loss=metrics[-1]['val']['loss'] if stage=='pilot' else best,
        elapsed_s=time.monotonic()-started,hardware_deployment_allowed=False)
    (output/'artifacts/training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'status.json').write_text(json.dumps(dict(stage='training_complete_evaluation_pending',**summary),indent=2)+'\n')
    return summary


def main()->int:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['data-dir','initializer','config','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--variant',choices=['A','B'],required=True);parser.add_argument('--stage',choices=['pilot','formal'],required=True)
    args=parser.parse_args();settings=json.loads(args.config.read_text());settings.update(variant=args.variant,stage=args.stage)
    try:train_history(args.data_dir,args.initializer,settings,args.output)
    except BaseException as exc:
        if args.output.exists():(args.output/'failure.json').write_text(json.dumps(dict(error=str(exc),hardware_deployment_allowed=False),indent=2))
        raise
    return 0


if __name__=='__main__':raise SystemExit(main())
