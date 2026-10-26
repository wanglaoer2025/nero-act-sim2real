"""Unchanged frozen ACT targets with narrowly tested offline contact escape rejection."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'nero_act_new_handeye_policy_repair_20261023'
sys.path.insert(0,str(ROOT.parent/'nero_act_dense_response_20261023'))
import evaluate_dense
sys.path.insert(0,str(PREVIOUS))
from train_adaptation import load_candidate,historical
from evaluation_binding import actual_sources,bind_snapshot
sys.path.insert(0,str(ROOT.parent/'nero_act_new_handeye_contact_repair_20261023'))
from card_edges import CardEdgesEstimator
from action_provenance import action_provenance
from simulation import visual_evaluation

_spec=importlib.util.spec_from_file_location('_offline_contact_escape_runtime',ROOT/'supervised_contact_escape.py')
runtime=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=runtime
_spec.loader.exec_module(runtime)


def evaluate():
    policy,stats=load_candidate()
    protocol=json.loads((ROOT/'evaluation_protocol.json').read_text())
    output=ROOT/'results'
    output.mkdir(exist_ok=False)
    protected=historical.snapshot([*ROOT.glob('*.py'),ROOT/'plan.md',ROOT/'evaluation_protocol.json',
        ROOT/'runtime_origin.json',ROOT/'runtime_patch.diff',
        PREVIOUS/'training/pilot_A/policy_last.ckpt',PREVIOUS/'training/pilot_A/integrity.json'])
    assert protected[str((PREVIOUS/'training/pilot_A/policy_last.ckpt').resolve())]==protocol['checkpoint_sha256']
    records=[]
    summary=dict(complete=False,results=records,full_task_passed=False,
        teacher_action_fallback=False,**action_provenance(),geometry_truth_policy_input=False,
        all_start_from_hypothetical_completed_95mm_preparation=True,actual_preparation_performed=False,
        attached_cup_micro_contact_escape_simulation_only=True,source_sha256=protected,
        additional_training_updates=0,hardware_commands_sent=False,hardware_deployment_allowed=False)
    for row in protocol['cases']:
        torch.manual_seed(42);np.random.seed(42)
        scenario=dict(name=f'contact_escape_{row["id"]}',seed=191899,query_stride=2,
            servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False)
        result=runtime.rollout(row['scene'],policy,stats,scenario,output/str(row['id']),
            runtime_profile='history_common_v1')
        passed=evaluate_dense.evaluation.full_task_passed(result)
        records.append(dict(id=row['id'],assessment_role=row['assessment_role'],
            offset_xy_m=row['offset_xy_m'],full_task_passed=passed,result=result))
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(dict(id=row['id'],role=row['assessment_role'],passed=passed,
            points=result['active_policy_points'],events=result['simulated_events'],
            lift_m=result['max_lift_m'],error=result['error'])),flush=True)
    historical.require_unchanged(protected)
    summary.update(complete=True,full_task_passed=all(r['full_task_passed'] for r in records),
        new_scene_passed=sum(r['full_task_passed'] for r in records if r['assessment_role']=='new_untrained_scene'),new_scene_total=2)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(2)
    visual_evaluation.NominalImageEstimator=CardEdgesEstimator
    evaluate_dense.runtime=runtime
    original=historical.snapshot
    dependencies=actual_sources(sys.modules[__name__])
    # The alternate runtime lives outside the original simulation package.
    simulation_root=Path(sys.modules['simulation'].__file__).resolve().parent
    dependencies += [p for p in simulation_root.rglob('*') if p.is_file() and
        p.suffix.lower() in {'.py','.xml','.stl','.obj','.msh','.urdf','.json','.png'}]
    dependencies += [*ROOT.glob('*.py'),Path(evaluate_dense.evaluation.__file__)]
    historical.snapshot=bind_snapshot(original,dependencies)
    evaluate_dense.ROOT=ROOT
    evaluate_dense.evaluation.main=evaluate
    try:
        evaluate_dense.main()
    finally:
        historical.snapshot=original
