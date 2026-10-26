"""Parent-lineage validation for verified partial history recoveries."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import h5py

ROLES={'pre_close':'grasp_to_retreat','hold_to_lift':'lift_to_retreat',
       'support_release':'release_to_retreat','release_retreat':'retreat_only'}
ADDITIONAL_ROLES={'approach':'descend_to_retreat'}
ADDITIONAL_EXPECTED={2018:'pre_close',2019:'approach'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_history_manifest(manifest:dict,split:dict,*,check_files=True,additional_failure_ids=())->dict:
    base=manifest['base_episodes'];rows=manifest['episodes']
    extra=set(additional_failure_ids)
    if (tuple(additional_failure_ids) not in ((),(2018,),(2018,2019))
            or manifest.get('additional_failure_ids',[])!=list(additional_failure_ids)):
        raise ValueError('Additional failure contexts must be explicitly declared before optimization')
    parents={r['id']:r for r in base}
    if (manifest.get('schema')!='history_recovery_v1' or manifest.get('complete') is not True
            or manifest.get('rejected') or len(base)!=48 or len(parents)!=48 or len(rows)!=18+len(extra)
            or len(split['train'])!=36 or len(split['val'])!=12
            or set(split['train'])&set(split['val']) or set(parents)!=set(split['train']+split['val'])):
        raise ValueError('Complete18 recoveries and original independent36/12 split required')
    groups={r['source_group']:r['split'] for r in base}
    if len(groups)!=48 or any(r['split']!=('train' if r['id'] in split['train'] else 'val') for r in base):
        raise ValueError('Original independent parent groups changed')
    groups['nominal_diagnostic_train']='train'
    ids=set(parents);counts=Counter();role_parents=set();nominal_roles=[]
    for r in rows:
        parent=r['parent_episode_id'];role=r['role']
        additional=r['id'] in extra
        if additional and (parent!='nominal_development' or role!=ADDITIONAL_EXPECTED[r['id']] or r.get('source_kind')!='ACT_failure'
                or r.get('context_sha256') in {old.get('context_sha256') for old in rows if old['id']!=r['id']}):
            raise ValueError('Additional context must be a new actual nominal training failure')
        if parent=='nominal_development':
            expected_group,expected_split='nominal_diagnostic_train','train'
            if not additional:nominal_roles.append(role)
        elif parent in parents:
            expected_group,expected_split=parents[parent]['source_group'],parents[parent]['split']
            counts[role,expected_split]+=1
        else:raise ValueError('Unknown recovery parent')
        if (r['id'] in ids or type(r['id']) is not int or r['id']<2000
                or r['source_group']!=expected_group or r['split']!=expected_split
                or role not in (dict(ROLES,**ADDITIONAL_ROLES) if additional else ROLES) or ((parent,role) in role_parents and not additional)
                or r.get('source_kind') not in ('ACT_failure','teacher_interruption')
                or r.get('status')!='valid' or r.get('full_task_verified') is not False
                or r.get('teacher_success') is not True or r.get('remaining_flow_verified') is not True
                or not r.get('replay',{}).get('passed') or not r.get('prefix_contract_audit',{}).get('passed')
                or r.get('label_source','offline_verified_teacher')!='offline_verified_teacher'
                or not r.get('context_sha256') or not r.get('sha256')):
            raise ValueError('Verified partial teacher labels must inherit parent group/split')
        ids.add(r['id'])
        if not additional:role_parents.add((parent,role))
        if check_files:
            path=Path(r['path']);context=Path(r['context_path'])
            if sha(path)!=r['sha256'] or sha(context)!=r['context_sha256']:
                raise ValueError('Recovery data/context changed')
            from simulation.runtime_checkpoint import CONTROL_KEYS,OBSERVATION_KEYS
            checkpoint=json.loads(context.read_text())
            if (checkpoint.get('schema')!='complete_offline_runtime_v1'
                    or checkpoint.get('bindings',{}).get('scope')!='offline_simulation'
                    or not CONTROL_KEYS<=checkpoint['control'].keys()
                    or not OBSERVATION_KEYS<=checkpoint['observation'].keys()):
                raise ValueError('Complete offline recovery context required')
            if role in ('support_release','release_retreat') and (checkpoint['control']['attachment'].get('airborne') is not True
                    or checkpoint['control']['monitor'].get('had_lifted') is not True):
                raise ValueError('Post-transfer context cannot erase the original liftoff latch/monitor')
            with h5py.File(path,'r') as f:
                if (f.attrs.get('full_task_verified',True) or not f.attrs.get('teacher_success')
                        or f.attrs.get('source_group')!=expected_group or f.attrs.get('split')!=expected_split
                        or f.attrs.get('recovery_role')!=role or f.attrs.get('label_source')!='offline_verified_teacher'):
                    raise ValueError('Recovery HDF role/split differs from manifest')
    if (not extra<=ids or sorted(nominal_roles)!=['hold_to_lift','pre_close']
            or any(counts[role,s]!=(3 if s=='train' else 1) for role in ROLES for s in ['train','val'])):
        raise ValueError('Four roles each require3train/1val plus2 nominal diagnostic contexts')
    return {key+'_groups':sorted(k for k,v in groups.items() if v==key) for key in ['train','val']}


def prepare_recoveries(base_data:Path,captures:Path,output:Path)->dict:
    import copy
    from simulation.history_teacher import capture_parent_contexts,collect_partial
    from training.generalized import validate_manifest
    base_data,captures,output=map(Path,[base_data,captures,output])
    source=captures/'runtime_checkpoints.json'
    if not source.is_file():raise ValueError('Actual nominal failure contexts are required first')
    original=json.loads(source.read_text())
    if not {'pre_close','hold_to_lift'}<=original.get('named',{}).keys():
        raise ValueError('Both actual nominal diagnostic contexts required')
    if any('cuda_random_state' in v for v in original['named'].values()):
        import torch
        if not torch.cuda.is_available():raise ValueError('Diagnostic contexts require the matching host CUDA RNG runtime')
        torch.cuda.init()
    base=json.loads((base_data/'manifest.json').read_text());split=json.loads((base_data/'split.json').read_text())
    validate_manifest(base,split)
    output.mkdir(parents=True,exist_ok=False)
    selected=[split['train'][0],split['train'][12],split['train'][24],split['val'][0]]
    (output/'generation_protocol.json').write_text(json.dumps(dict(parent_ids=selected,roles=list(ROLES),
        nominal_context_source=str(source.resolve()),nominal_context_source_sha256=sha(source),
        fallback='teacher_interruption_only_where_no_ACT_failure_context_is_available',
        training_groups=37,validation_groups=12,hardware_commands_sent=False),indent=2)+'\n')
    rows=[];rejected=[]
    result=dict(schema='history_recovery_v1',complete=False,episodes=rows,base_episodes=base['episodes'],rejected=rejected)
    by_id={r['id']:r for r in base['episodes']}
    def generate(scene,parent,group,partition,role,context,kind):
        index=2000+len(rows)
        path=output/f'episode_{index}.hdf5'
        report=collect_partial(scene,context,role,path,parent_group=group,split=partition)
        rows.append(dict(report,id=index,parent_episode_id=parent,source_group=group,split=partition,role=role,
            source_kind=kind,status='valid',label_source='offline_verified_teacher',context_path=str(context.resolve()),
            context_sha256=sha(context)))
        print(json.dumps(dict(completed=len(rows),total=18,id=index,role=role,source_kind=kind,frames=report['frames'])),flush=True)
        (output/'generation_status.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        for parent in selected:
            row=by_id[parent]
            with h5py.File(row['path'],'r') as f:scene=json.loads(f.attrs['scene_config'])
            contexts=capture_parent_contexts(row['path'],output/'contexts'/f'parent_{parent}')
            for role in ROLES:generate(scene,parent,row['source_group'],row['split'],role,contexts[role],'teacher_interruption')
        nominal_scene=json.loads((captures/'scene_configuration.json').read_text())
        nominal_scene['generalization']=dict(index='nominal_development',source_group='nominal_diagnostic_train',split='train')
        for role in ['pre_close','hold_to_lift']:
            context=output/'contexts'/('nominal_'+role+'.json')
            context.write_text(json.dumps(original['named'][role],allow_nan=False))
            generate(copy.deepcopy(nominal_scene),'nominal_development','nominal_diagnostic_train','train',role,context,'ACT_failure')
        result['complete']=True;result['group_validation']=validate_history_manifest(result,split)
    except BaseException as exc:
        result['complete']=False;rejected.append(dict(error=type(exc).__name__+': '+str(exc),completed=len(rows)))
        (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');raise
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main()->int:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-data',type=Path,required=True);parser.add_argument('--captures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare_recoveries(args.base_data,args.captures,args.output);return 0


if __name__=='__main__':raise SystemExit(main())
