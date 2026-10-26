"""One-variable cup-position ablation; ALL truth-fed results are diagnostic-only."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'nero_act_contact_escape_fix_20261023'
sys.path.insert(0,str(BASE))
import evaluate_escape as baseline
from simulation import history_observation,visual_evaluation
from oracle_observation import DiagnosticHistory,MODE,diagnostic_record
from evaluation_binding import actual_sources,bind_snapshot

evaluate_dense=baseline.evaluate_dense
ORIGINAL_HISTORY=history_observation.SimulationHistory


def main_evaluation():
    parent=json.loads((BASE/'results/summary.json').read_text())
    if not parent['complete']:
        raise ValueError('Completed frozen RGB baseline required')
    baseline.historical.require_unchanged(parent['source_sha256'])
    policy,stats=baseline.load_candidate()
    protocol=json.loads((ROOT/'diagnostic_protocol.json').read_text())
    rows=json.loads((BASE/'evaluation_protocol.json').read_text())['cases']
    output=ROOT/'results';output.mkdir(exist_ok=False)
    protected=baseline.historical.snapshot([*ROOT.glob('*.py'),ROOT/'diagnostic_protocol.json',
        BASE/'results/summary.json',BASE/'evaluation_protocol.json',
        Path('/home/user/wq/docs/superpowers/specs/2026-10-23-nero-act-observation-repair-design.md'),
        Path('/home/user/wq/docs/superpowers/plans/2026-10-23-nero-act-observation-repair.md')])
    records=[]
    summary=dict(complete=False,mode=MODE,results=records,source_sha256=protected,
        full_task_passed=False,migration_acceptance=False,hardware_commands_sent=False,
        hardware_deployment_allowed=False,training_updates=0,geometry_truth_policy_input=True,
        actual_95mm_preparation_performed=False,normal_baseline_reproduced=False)
    def run(row,name,oracle):
        history_observation.SimulationHistory=DiagnosticHistory if oracle else ORIGINAL_HISTORY
        torch.manual_seed(42);np.random.seed(42)
        scenario=dict(name=f'contact_escape_{row["id"]}',**protocol['scenario'])
        if oracle:scenario['diagnostic_observation_mode']=MODE
        return baseline.runtime.rollout(row['scene'],policy,stats,scenario,output/name,
            runtime_profile='history_common_v1')
    row=next(x for x in rows if x['id']==9005)
    result=run(row,'normal_9005',False)
    fields=['states','sent_targets','raw_chunks','query_steps','feedback_at_query','visual_features_m']
    with np.load(BASE/'results/9005/rollout.npz',allow_pickle=False) as previous, np.load(output/'normal_9005/rollout.npz',allow_pickle=False) as current:
        identical={key:np.array_equal(previous[key],current[key]) for key in fields}
    if not all(identical.values()) or result['error']!=next(x for x in parent['results'] if x['id']==9005)['result']['error']:
        raise ValueError('Normal replay must exactly reproduce the bound baseline before ablation: '+str(identical))
    summary.update(normal_baseline_reproduced=True,normal_arrays_identical=identical)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(normal_reproduced=True,id=9005,points=result['active_policy_points'],error=result['error'])),flush=True)
    for case_id in protocol['case_ids']:
        row=next(x for x in rows if x['id']==case_id)
        result=run(row,f'oracle_{case_id}',True)
        observed=evaluate_dense.evaluation.full_task_passed(result)
        record=diagnostic_record(case_id,result,full_flow_observed=observed)
        record['normal_baseline_passed']=next(x for x in parent['results'] if x['id']==case_id)['full_task_passed']
        records.append(record)
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(dict(id=case_id,diagnostic_full_flow_observed=observed,migration_acceptance=False,
            points=result['active_policy_points'],events=result['simulated_events'],
            lift_m=result['max_lift_m'],error=result['error'])),flush=True)
    baseline.historical.require_unchanged(protected)
    baseline.historical.require_unchanged(parent['source_sha256'])
    summary.update(complete=True,diagnostic_full_flow_count=sum(x['diagnostic_full_flow_observed'] for x in records))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(2)
    visual_evaluation.NominalImageEstimator=baseline.CardEdgesEstimator
    evaluate_dense.runtime=baseline.runtime
    original=baseline.historical.snapshot
    dependencies=actual_sources(sys.modules[__name__])
    root=Path(sys.modules['simulation'].__file__).resolve().parent
    dependencies += [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in
        {'.py','.xml','.stl','.obj','.msh','.urdf','.json','.png'}]
    dependencies += [*ROOT.glob('*.py'),Path(evaluate_dense.evaluation.__file__)]
    baseline.historical.snapshot=bind_snapshot(original,dependencies)
    evaluate_dense.ROOT=ROOT;evaluate_dense.evaluation.main=main_evaluation
    try:
        evaluate_dense.main()
    finally:
        history_observation.SimulationHistory=ORIGINAL_HISTORY
        baseline.historical.snapshot=original
