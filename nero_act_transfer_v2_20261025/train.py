"""Fixed-budget offline ACT training. Default is preflight, never hardware."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import torch
import numpy as np
import h5py
from torch.utils.data import default_collate
from data_audit import audit_episode
from protocol import validate_positions,validate_final_positions

ROOT=Path(__file__).resolve().parent


def parse_arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['check','pilot','formal'],default='check')
    p.add_argument('--pilot-proof',type=Path,default=ROOT/'training/pilot/completion.json')
    return p.parse_args(argv)


def spatial_errors(row,base,protocol):
    split=row.get('split');group=row.get('position_group')
    expected={f'{split}_{i:02d}':xy for i,xy in enumerate(protocol['positions_mm'].get(split,[]))}
    if split not in ['train','dev'] or group not in expected:return ['invalid_spatial_split_group']
    try:
        with h5py.File(row['path']) as f:
            scene=json.loads(f.attrs['scene_config'])
            if f.attrs.get('split')!=split or f.attrs.get('position_group')!=group:
                return ['HDF_spatial_metadata_mismatch']
            actual=np.asarray(scene['pick_xy_m'],float)
            target=np.asarray(base['pick_xy_m'])+np.asarray(expected[group])/1000
            if actual.shape!=(2,) or not np.allclose(actual,target,atol=1e-9,rtol=0):
                return ['actual_cup_position_does_not_match_group']
    except (OSError,KeyError,ValueError):return ['spatial_scene_metadata_unreadable']
    return []


def preflight(manifest_path,*,mode='check'):
    blockers=[];rows=[];mechanism_only=False
    try:
        validate_final_positions(json.loads((ROOT/'final_position_protocol.json').read_text()),
            json.loads((ROOT/'position_protocol.json').read_text()),
            json.loads((ROOT/'historical_development_cases.json').read_text()))
    except (OSError,KeyError,ValueError):blockers.append('final_position_exclusion_not_sealed')
    manifest_path=Path(manifest_path)
    if not manifest_path.exists():blockers.append('qualified_data_manifest_missing')
    else:
        try:manifest=json.loads(manifest_path.read_text())
        except (OSError,json.JSONDecodeError):
            return dict(allowed=False,blockers=['qualified_manifest_unreadable'],optimizer_updates=0,
                        hardware_commands_sent=False)
        mechanism_only=(mode=='pilot' and manifest.get('schema')=='nero_act_transfer_v2_mechanism_pilot_data'
                        and manifest.get('pilot_only') is True)
        if not manifest.get('complete') or (not mechanism_only and manifest.get('schema')!='nero_act_transfer_v2_qualified_data'):
            blockers.append('manifest_incomplete_or_wrong_schema')
        rows=manifest.get('episodes',[])
        required_train=90 if mechanism_only else 120
        if sum(r['split']=='train' for r in rows)!=required_train or sum(r['split']=='dev' for r in rows)!=30:
            blockers.append(f'complete_{required_train}train_30dev_required')
        protocol=json.loads((ROOT/'position_protocol.json').read_text())
        validate_positions(protocol)
        with h5py.File(manifest['source_scene']) as f:base=json.loads(f.attrs['scene_config'])
        for name,filename in [('observer_sha256','observer.py'),('position_protocol_sha256','position_protocol.json')]:
            if manifest.get(name)!=hashlib.sha256((ROOT/filename).read_bytes()).hexdigest():
                blockers.append('source_binding_changed:'+filename)
        seen=set();groups={'train':{},'dev':{}}
        for row in rows:
            path=Path(row['path']).resolve()
            if str(path) in seen:blockers.append('duplicate_episode:'+str(path))
            seen.add(str(path))
            for dependency in row.get('source_dependencies',[]):
                source=Path(dependency['path'])
                if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest()!=dependency['sha256']:
                    blockers.append('underlying_source_changed:'+str(source))
            split=row.get('split');group=row.get('position_group')
            blockers.extend(spatial_errors(row,base,protocol))
            if split not in groups or not group or not group.startswith(split+'_'):
                blockers.append('invalid_spatial_split_group')
            else:groups[split][group]=groups[split].get(group,0)+1
            if (not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']
                    or not audit_episode(path)['main15_ready']):
                blockers.append('unqualified_or_changed_episode:'+str(path))
        if (len(groups['train'])!=(9 if mechanism_only else 12) or any(n!=10 for n in groups['train'].values())
                or len(groups['dev'])!=6 or any(n!=5 for n in groups['dev'].values())):
            blockers.append('spatial_group_counts_do_not_match_protocol')
    if not torch.cuda.is_available():blockers.append('CUDA_not_visible_in_this_execution_context')
    if not (ROOT/'evaluate.py').is_file():blockers.append('full_development_rollout_not_integrated')
    binding=ROOT/'evidence/implementation_manifest.json'
    if not binding.exists():blockers.append('implementation_not_sealed')
    else:
        for name,digest in json.loads(binding.read_text())['sha256'].items():
            if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                blockers.append('implementation_changed:'+name)
    return dict(allowed=not blockers,blockers=blockers,episodes=len(rows),optimizer_updates=0,
                hardware_commands_sent=False,formal_budget=20000,pilot_budget=200,
                admission_scope='mechanism_pilot_only' if mechanism_only else 'full_main_data',
                full_main_data_ready=not mechanism_only and not blockers)


def deployment_error(policy,dataset):
    arm_total=grip_total=0.;arm_count=grip_count=0
    policy.eval()
    with torch.inference_mode():
        for offset in range(0,len(dataset),32):
            batch=default_collate([dataset[i] for i in range(offset,min(offset+32,len(dataset)))])
            pred=policy(batch['observation'].to(policy.device),batch['image'].to(policy.device))
            target=batch['target'].to(policy.device);valid=~batch['padding'].to(policy.device)
            error=(pred-target).abs()
            arm_total+=float(error[:,:,:7][valid].sum());arm_count+=int(valid.sum())*7
            grip_total+=float(error[:,:2,7][valid[:,:2]].sum());grip_count+=int(valid[:,:2].sum())
    return arm_total/max(1,arm_count)+grip_total/max(1,grip_count)


def select_candidate(records):
    eligible=[r for r in records if r['adaptation_gate_passed']]
    return max(eligible or records,key=lambda r:(r['passed'],r['phase_progress'],-r['deployment_error']))


def requery_probes(policy,dataset):
    """Frozen-feedback mechanism probes, not a complete physics evaluation."""
    records=[]
    for stage in ['close','carry','release']:
        candidates=[i for i,s in enumerate(dataset.stages) if s==stage]
        if not candidates:raise ValueError('Mechanism probe stage missing:'+stage)
        for i in [candidates[len(candidates)//2]]:
            row=dataset[i];obs=row['observation'][None].to(policy.device);image=row['image'][None].to(policy.device)
            width=obs[:,7:8]*policy.qpos_std[7]+policy.qpos_mean[7]
            previous=width+obs[:,8:9]*.01;start=float(previous[0,0])
            with torch.inference_mode():
                for _ in range(10):
                    pred=policy(obs,image)
                    previous=pred[:,1,7:8]*policy.action_std[7]+policy.action_mean[7]
                    obs=obs.clone();obs[:,8:9]=(previous-width)/.01
            delta=float(previous[0,0])-start
            passed=delta<-.0001 if stage=='close' else delta>.0001 if stage=='release' else abs(delta)<.001
            records.append(dict(stage=stage,net_target_change_m=delta,passed=bool(passed)))
    return dict(passed=all(r['passed'] for r in records),records=records,
                scope='ten_requeries_with_frozen_feedback_not_physics')


def main():
    args=parse_arguments()
    report=preflight(args.manifest,mode=args.mode)
    if args.mode=='formal':
        from pilot_revalidate import admission_errors
        errors=admission_errors(args.pilot_proof,args.manifest,ROOT/'evidence/implementation_manifest.json',root=ROOT)
        if errors:
            report['allowed']=False;report['blockers'].extend(errors)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.mode=='check' or not report['allowed']:
        print(json.dumps(report));return 0 if args.mode=='check' else 2
    from dataset import V2Dataset,fit_statistics,balanced_indices
    from policy import ACTPolicy
    from evaluate import evaluate_development
    manifest=json.loads(args.manifest.read_text())
    train_paths=[r['path'] for r in manifest['episodes'] if r['split']=='train']
    stats=fit_statistics(train_paths)
    dataset=V2Dataset(train_paths,stats)
    dev_dataset=V2Dataset([r['path'] for r in manifest['episodes'] if r['split']=='dev'],stats,stride=40)
    policy=ACTPolicy(stats).cuda().train()
    optimizer=policy.optimizer()
    limit=200 if args.mode=='pilot' else 20000
    started=time.monotonic()
    code_binding_sha=hashlib.sha256((ROOT/'evidence/implementation_manifest.json').read_bytes()).hexdigest()
    data_binding_sha=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    checkpoint_records=[]
    for step in range(1,limit+1):
        rows=balanced_indices(dataset,step)
        batch=default_collate([dataset[i] for i in rows])
        tensors={k:v.cuda() for k,v in batch.items() if isinstance(v,torch.Tensor)}
        optimizer.zero_grad(set_to_none=True)
        losses=policy(tensors['observation'],tensors['image'],tensors['target'],tensors['padding'])
        if not torch.isfinite(losses['loss']):raise ValueError('Nonfinite training loss')
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step==1 or step%100==0:
            for name,digest in json.loads((ROOT/'evidence/implementation_manifest.json').read_text())['sha256'].items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('Training implementation changed mid-run:'+name)
            status=dict(updates=step,limit=limit,elapsed_s=time.monotonic()-started,
                        losses={k:float(losses[k].detach()) for k in ['loss','arm','gripper','kl']},
                        formal=args.mode=='formal',admission_scope=report['admission_scope'],hardware_commands_sent=False)
            (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            print(json.dumps(status),flush=True)
        if step in [2000,5000,10000,20000] and args.mode=='formal':
            path=args.output/f'policy_{step}.ckpt'
            torch.save(dict(model=policy.state_dict(),config=policy.config,stats=stats,updates=step,
                            mode=args.mode,optimizer=optimizer.state_dict(),hardware_deployment_allowed=False),path)
            policy.eval()
            result=evaluate_development(policy,stats,manifest,args.output/f'dev_{step}')
            result['deployment_error']=deployment_error(policy,dev_dataset)
            checkpoint_records.append(dict(updates=step,path=str(path),passed=result['passed'],
                phase_progress=result['phase_progress'],adaptation_gate_passed=result['adaptation_gate_passed'],
                deployment_error=result['deployment_error']))
            (args.output/f'dev_{step}.json').write_text(json.dumps(result,indent=2)+'\n')
            policy.train()
    policy.eval()
    mechanism=requery_probes(policy,dataset)
    (args.output/'requery_probes.json').write_text(json.dumps(mechanism,indent=2)+'\n')
    if args.mode=='pilot':
        torch.save(dict(model=policy.state_dict(),config=policy.config,stats=stats,updates=limit,
                        mode='pilot',hardware_deployment_allowed=False),args.output/'pilot.ckpt')
    if checkpoint_records:
        selected=select_candidate(checkpoint_records)
        (args.output/'candidate_selection.json').write_text(json.dumps(dict(candidates=checkpoint_records,
            selected=selected,independent_test_consumed=False,hardware_deployment_allowed=False),indent=2)+'\n')
    (args.output/'completion.json').write_text(json.dumps(dict(complete=True,updates=limit,
        formal=args.mode=='formal',mechanism_passed=mechanism['passed'],full_task_passed=False,
        admission_scope=report['admission_scope'],full_main_data_admission=report['full_main_data_ready'],
        data_manifest_sha256=data_binding_sha,implementation_binding_sha256=code_binding_sha,
        hardware_deployment_allowed=False),indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
