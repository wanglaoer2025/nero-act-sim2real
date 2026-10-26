"""Teacher-free full ACT: center/development and two new independent positions."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'nero_act_new_handeye_policy_repair_20261023'
sys.path.insert(0,str(ROOT.parent/'nero_act_dense_response_20261023'))
import evaluate_dense
from train_contact import load_candidate,historical
from card_fragments import CardFragmentsEstimator
from simulation import visual_evaluation
sys.path.insert(0,str(PREVIOUS))
from evaluation_binding import actual_sources,bind_snapshot


def evaluate():
    policy,stats=load_candidate()
    original=json.loads((PREVIOUS/'data/protocol.json').read_text())
    protocol=json.loads((ROOT/'data/protocol.json').read_text())
    cases=[dict(r,assessment_role='center' if r['id']==9000 else 'development')
        for r in original if r['id'] in {9000,9005,9006}]
    cases += [dict(r,assessment_role='independent') for r in protocol if r['split']=='val']
    assert [r['id'] for r in cases]==[9000,9005,9006,9200,9201]
    output=ROOT/'results'
    output.mkdir(exist_ok=False)
    protected=historical.snapshot([*ROOT.glob('*.py'),ROOT/'plan.md',ROOT/'data/protocol.json',
        ROOT/'data/manifest.json',PREVIOUS/'data/protocol.json',
        ROOT/'training/pilot_A/policy_last.ckpt',ROOT/'training/pilot_A/integrity.json'])
    records=[]
    summary=dict(complete=False,results=records,full_task_passed=False,
        teacher_action_fallback=False,geometry_truth_policy_input=False,
        all_start_from_hypothetical_completed_95mm_preparation=True,
        actual_preparation_performed=False,source_sha256=protected,
        hardware_commands_sent=False,hardware_deployment_allowed=False)
    for row in cases:
        torch.manual_seed(42);np.random.seed(42)
        scenario=dict(name=f'new_camera_contact_{row["id"]}',seed=191899,query_stride=2,
            servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False)
        result=evaluate_dense.runtime.rollout(row['scene'],policy,stats,scenario,
            output/str(row['id']),runtime_profile='history_common_v1')
        passed=evaluate_dense.evaluation.full_task_passed(result)
        records.append(dict(id=row['id'],assessment_role=row['assessment_role'],
            offset_xy_m=row['offset_xy_m'],full_task_passed=passed,result=result))
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(dict(id=row['id'],role=row['assessment_role'],passed=passed,
            points=result['active_policy_points'],events=result['simulated_events'],
            lift_m=result['max_lift_m'],error=result['error'])),flush=True)
    historical.require_unchanged(protected)
    summary.update(complete=True,full_task_passed=all(r['full_task_passed'] for r in records),
        independent_passed=sum(r['full_task_passed'] for r in records if r['assessment_role']=='independent'),
        independent_total=2)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(2)
    # Train and evaluate with the exact same RGB-only observer; never alter rendered RGB.
    visual_evaluation.NominalImageEstimator=CardFragmentsEstimator
    original=historical.snapshot
    dependencies=actual_sources(sys.modules[__name__])
    dependencies.extend([*ROOT.glob('*.py'),ROOT/'contact_guard.py',Path(evaluate_dense.evaluation.__file__)])
    historical.snapshot=bind_snapshot(original,dependencies)
    evaluate_dense.ROOT=ROOT
    evaluate_dense.evaluation.main=evaluate
    try:
        evaluate_dense.main()
    finally:
        historical.snapshot=original
