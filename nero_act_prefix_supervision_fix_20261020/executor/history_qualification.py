"""Bind pilot and development decisions to actual finalized checkpoints, offline only."""
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _artifact(entry):
    path=Path(entry['path']).resolve()
    if not path.is_file() or sha(path)!=entry['sha256']:raise ValueError('Qualification artifact changed')
    return path,json.loads(path.read_text())


def _training(root,run,variant,kind):
    import torch
    from training.integrity import require_finalized
    run=Path(run).resolve();root=Path(root).resolve()
    if not run.is_relative_to(root/'training'):raise ValueError('Qualification must use this experiment training runs')
    checkpoint=run/'checkpoints'/('policy_last.ckpt' if kind=='updates' else 'policy_best_val.ckpt')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);config=payload['config']
    if (config.get('variant')!=variant or config.get('training_budget')!={'kind':kind,'limit':1000 if kind=='updates' else 20}
            or config.get('history_runtime_profile')!='history_common_v1'):
        raise ValueError('Different variant or budget in qualification graph')
    proof=require_finalized(checkpoint,config,allowed_budgets=(kind,))
    return payload,checkpoint,[proof,checkpoint,run/'config.json']


def require_mechanism_binding(summary,checkpoint_sha256,source_seal_sha256):
    if (summary.get('complete') is not True or summary.get('checkpoint_sha256')!=checkpoint_sha256
            or summary.get('source_seal_sha256')!=source_seal_sha256 or len(summary.get('records',[]))!=6):
        raise ValueError('Mechanism result must bind this completed pilot and unchanged source seal')


def require_formal_gate(root)->list[Path]:
    from simulation.history_mechanism import pilot_gate
    root=Path(root);path=root/'evidence/pilot_decision.json'
    if not path.is_file():raise ValueError('Completed pilot mechanism decision required before formal training')
    decision=json.loads(path.read_text());seal=root/'evidence/pre_optimization_freeze.json'
    if decision.get('complete') is not True or decision.get('gate')!='formal_pair' or decision.get('source_seal_sha256')!=sha(seal):
        raise ValueError('B must pass all pilot mechanisms before formal paired training')
    inputs=[path];records={}
    for variant in ['A','B']:
        _,checkpoint,files=_training(root,decision[f'{variant}_training_run'],variant,'updates')
        result_path,summary=_artifact(decision[f'{variant}_mechanism_summary'])
        require_mechanism_binding(summary,sha(checkpoint),sha(seal))
        inputs+=files+[result_path];records[variant]=summary
    if pilot_gate(records['A'],records['B'])!='formal_pair':raise ValueError('Actual B mechanism evidence did not pass')
    return inputs


def _nominal(path,checkpoint,*,require_pass):
    from simulation.visual_evaluation import full_task_passed
    path=Path(path)
    if not path.is_file():raise ValueError('Successful nominal prerequisite required before development')
    summary=json.loads(path.read_text());protocol_path=path.parent/'protocol.json'
    if (summary.get('complete') is not True or summary.get('finalized') is not True or summary.get('total')!=1
            or len(summary.get('results',[]))!=1 or summary.get('provenance',{}).get('assessment_role')!='nominal_development_not_final_test'):
        raise ValueError('Finalized nominal prerequisite required')
    protocol=json.loads(protocol_path.read_text())
    if protocol['dependencies'].get(str(checkpoint.resolve()))!=sha(checkpoint):
        raise ValueError('Nominal prerequisite used another checkpoint')
    passed=full_task_passed(summary['results'][0])
    if require_pass and not passed:raise ValueError('Nominal prerequisite did not pass')
    return passed,[path,protocol_path]


def require_development_gate(root,training_run,*,pilot=False,nominal_summary=None):
    root=Path(root)
    if pilot:
        from simulation.history_mechanism import pilot_gate
        path=root/'evidence/pilot_decision.json';seal=root/'evidence/pre_optimization_freeze.json'
        if not path.is_file():raise ValueError('Pilot mechanism decision prerequisite required')
        decision=json.loads(path.read_text())
        if (decision.get('complete') is not True or decision.get('gate')!='A_development_only'
                or decision.get('source_seal_sha256')!=sha(seal)
                or Path(decision['A_training_run']).resolve()!=Path(training_run).resolve()):
            raise ValueError('Pilot decision does not qualify this A development run')
        inputs=[path];records={}
        for variant in ['A','B']:
            _,checkpoint,files=_training(root,decision[f'{variant}_training_run'],variant,'updates')
            result_path,summary=_artifact(decision[f'{variant}_mechanism_summary'])
            require_mechanism_binding(summary,sha(checkpoint),sha(seal))
            inputs+=files+[result_path];records[variant]=summary
        if pilot_gate(records['A'],records['B'])!='A_development_only':raise ValueError('Pilot mechanism decision disagrees')
        return inputs
    run=Path(training_run);config=json.loads((run/'config.json').read_text()) if (run/'config.json').is_file() else {}
    nominal=Path(nominal_summary or root/f'results/nominal_{config.get("variant", "unknown")}/summary.json')
    checkpoint=run/'checkpoints/policy_best_val.ckpt'
    _,inputs=_nominal(nominal,checkpoint,require_pass=True)
    return inputs


def development_count(path,summary,checkpoint,expected_names):
    from simulation.visual_evaluation import full_task_passed
    role=summary.get('assessment_role',summary.get('provenance',{}).get('assessment_role'))
    if (summary.get('complete') is not True or summary.get('finalized') is not True
            or summary.get('total')!=16 or role!='paired_baseline_development_not_independent'):
        raise ValueError('All16 finalized original development cases required')
    path=Path(path);inputs=[]
    if 'records' in summary:
        records=summary['records'];names=[r['name'] for r in records];results=[r['result'] for r in records]
        if summary.get('checkpoint_sha256')!=sha(checkpoint):raise ValueError('Development used another checkpoint')
        bindings_path=path.parent/'bindings.json'
    else:
        results=summary.get('results',[]);names=[r.get('scenario',{}).get('name') for r in results]
        bindings_path=path.parent/'protocol.json'
    bindings=json.loads(bindings_path.read_text());bindings=bindings.get('dependencies',bindings);inputs.append(bindings_path)
    if bindings.get(str(checkpoint.resolve()))!=sha(checkpoint):raise ValueError('Development checkpoint binding changed')
    if len(names)!=16 or len(set(names))!=16 or set(names)!=set(expected_names):
        raise ValueError('All16 unique original development case identities required')
    actual=sum(full_task_passed(r) for r in results)
    if actual!=summary.get('passed'):raise ValueError('Development summary disagrees with physical results')
    return actual,inputs


def require_independent_candidate(root,training_run)->list[Path]:
    from simulation.visual_evaluation import select_independent_candidate,full_task_passed
    root=Path(root);selection=root/'evidence/candidate_selection.json'
    if not selection.is_file():raise ValueError('Qualified development candidate required before independent evaluation')
    document=json.loads(selection.read_text())
    if document.get('complete') is not True or {r.get('variant') for r in document.get('records',[])}!={'A','B'}:
        raise ValueError('Both formal candidate development records required')
    inputs=[selection];records=[];runs={}
    for entry in document['records']:
        variant=entry['variant'];payload,checkpoint,files=_training(root,entry['training_run'],variant,'epochs')
        runs[variant]=Path(entry['training_run']).resolve();inputs+=files
        nominal_path,nominal=_artifact(entry['nominal_summary']);inputs.append(nominal_path)
        nominal_passed,nominal_files=_nominal(nominal_path,checkpoint,require_pass=False);inputs+=nominal_files
        row=dict(variant=variant,finalized=True,development_passed=0,development_total=16,
                 zero_latent_val_loss=payload['metrics']['val']['loss'])
        if nominal_passed:
            development_path,development=_artifact(entry['development_summary']);inputs.append(development_path)
            baseline=json.loads((root/'evidence/baseline_development_protocol.json').read_text())
            expected=[r['name'] for r in baseline['resolved_cases']] if 'resolved_cases' in baseline else [r['id'] for r in baseline['cases']]
            row['development_passed'],files=development_count(development_path,development,checkpoint,expected)
            inputs+=files
        records.append(row)
    selected=select_independent_candidate(records)
    if (selected is None or selected!=document.get('selected_variant')
            or runs[selected]!=Path(training_run).resolve()):
        raise ValueError('Independent candidate must be the predeclared eligible validation-loss selection')
    return inputs
