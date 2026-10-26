"""Full teacher-free ACT rollouts: center and two independent new-camera positions."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'nero_act_dense_response_20261023'))
import evaluate_dense
sys.path.insert(0,str(ROOT))
from train_adaptation import load_candidate,historical


def evaluate():
    policy,stats = load_candidate()
    protocol = json.loads((ROOT/'data/protocol.json').read_text())
    cases = [row for row in protocol if row['role']=='center' or row['split']=='val']
    assert len(cases)==3 and sum(r['split']=='val' for r in cases)==2
    output = ROOT/'results'
    output.mkdir(exist_ok=False)
    protected = historical.snapshot([Path(__file__),ROOT/'train_adaptation.py',ROOT/'protocol.py',
        ROOT/'data/protocol.json',ROOT/'data/manifest.json',ROOT/'training/pilot_A/policy_last.ckpt',
        ROOT/'training/pilot_A/integrity.json'])
    records = []
    for row in cases:
        torch.manual_seed(42)
        np.random.seed(42)
        scenario = dict(name=f'new_camera_{row["id"]}',seed=191899,query_stride=2,
            servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False)
        result = evaluate_dense.runtime.rollout(row['scene'],policy,stats,scenario,
            output/str(row['id']),runtime_profile='history_common_v1')
        passed = evaluate_dense.evaluation.full_task_passed(result)
        records.append(dict(id=row['id'],split=row['split'],offset_xy_m=row['offset_xy_m'],
            full_task_passed=passed,result=result))
        summary = dict(complete=False,results=records,full_task_passed=False,
            teacher_action_fallback=False,geometry_truth_policy_input=False,
            all_start_from_hypothetical_completed_95mm_preparation=True,
            actual_preparation_performed=False,source_sha256=protected,
            hardware_commands_sent=False,hardware_deployment_allowed=False)
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(dict(id=row['id'],split=row['split'],passed=passed,
            points=result['active_policy_points'],events=result['simulated_events'],
            lift_m=result['max_lift_m'],error=result['error'])),flush=True)
    historical.require_unchanged(protected)
    summary.update(complete=True,full_task_passed=all(r['full_task_passed'] for r in records))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    torch.set_num_threads(2)
    evaluate_dense.ROOT = ROOT
    evaluate_dense.evaluation.main = evaluate
    evaluate_dense.main()
