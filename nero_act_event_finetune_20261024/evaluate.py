"""Same frozen full-task runtime for baseline and fixed-last event-data candidate."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from common import ROOT,CHECKPOINT,CHECKPOINT_SHA,load_initial,historical

SOURCE=ROOT.parent/'nero_act_contact_escape_fix_20261023'
sys.path.insert(0,str(SOURCE))
spec=importlib.util.spec_from_file_location('_frozen_event_evaluation',SOURCE/'evaluate_escape.py')
sealed=importlib.util.module_from_spec(spec);sys.modules[spec.name]=sealed;spec.loader.exec_module(sealed)
evaluate_dense=sealed.evaluate_dense


def evaluate():
    protocol=json.loads((ROOT/'evaluation_protocol.json').read_text())
    candidate=ROOT/'training/pilot_A/policy_last.ckpt'
    proof=json.loads((candidate.parent/'integrity.json').read_text())
    if not proof['complete'] or proof['actual_updates']!=1000:raise ValueError('Fixed final1000-update model required')
    historical.require_unchanged(proof['source_sha256']);historical.require_unchanged(proof['checkpoint_sha256'])
    policy,stats,_=load_initial()
    candidate_payload=torch.load(candidate,map_location='cpu',weights_only=False)
    if candidate_payload['config']!=policy.config or any(not np.array_equal(candidate_payload['stats'][k],stats[k]) for k in stats):
        raise ValueError('Evaluation must keep architecture and statistics')
    output=ROOT/'results';output.mkdir(exist_ok=False)
    hashes=historical.snapshot([*ROOT.glob('*.py'),ROOT/'evaluation_protocol.json',candidate,candidate.parent/'integrity.json',CHECKPOINT])
    records=[];summary=dict(complete=False,records=records,source_sha256=hashes,
        fixed_last_checkpoint_sha256=proof['checkpoint_sha256'][str(candidate)],additional_training_updates=0,
        online_visual_simulation=True,truth_used_as_policy_input=False,
        attached_cup_micro_contact_escape_simulation_only=True,
        existing_contact_arm_and_pregrasp_gripper_holds_retained=True,
        teacher_generated_runtime_actions=False,hardware_commands_sent=False,hardware_deployment_allowed=False)
    for variant in ['baseline','event_candidate']:
        (output/variant).mkdir()
        payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False) if variant=='baseline' else candidate_payload
        policy.load_state_dict(payload['model'],strict=True);policy.requires_grad_(False).eval()
        for row in protocol['cases']:
            if variant=='baseline' and row['role']=='new_final_only_position':continue
            torch.manual_seed(42);np.random.seed(42)
            scenario=dict(name=f'event_finetune_{variant}_{row["id"]}',seed=191899,query_stride=2,
                servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False)
            result=sealed.runtime.rollout(row['scene'],policy,stats,scenario,output/variant/str(row['id']),
                runtime_profile='history_common_v1')
            passed=evaluate_dense.evaluation.full_task_passed(result)
            record=dict(variant=variant,id=row['id'],role=row['role'],offset_xy_m=row['offset_xy_m'],
                full_task_passed=passed,result=result)
            records.append(record)
            (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps(dict(variant=variant,id=row['id'],passed=passed,points=result['active_policy_points'],
                events=result['simulated_events'],lift_m=result['max_lift_m'],error=result['error'])),flush=True)
    historical.require_unchanged(hashes)
    summary['comparison']={variant:dict(passed=sum(r['full_task_passed'] for r in records if r['variant']==variant and r['role']=='development_regression'),
        total=3) for variant in ['baseline','event_candidate']}
    new=[r for r in records if r['role']=='new_final_only_position']
    summary.update(complete=True,new_position_passed=sum(r['full_task_passed'] for r in new),new_position_total=len(new),
        all_candidate_flows_passed=all(r['full_task_passed'] for r in records if r['variant']=='event_candidate'))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(comparison=summary['comparison'],new_position_passed=summary['new_position_passed'],new_position_total=len(new))),flush=True)


def main():
    torch.set_num_threads(2)
    sealed.visual_evaluation.NominalImageEstimator=sealed.CardEdgesEstimator
    evaluate_dense.runtime=sealed.runtime
    dependencies=sealed.actual_sources(sys.modules[__name__])
    # Include the alternative runtime's imported contact exception and template assets.
    dependencies += [*SOURCE.glob('*.py')]
    original=historical.snapshot
    historical.snapshot=sealed.bind_snapshot(original,dependencies)
    evaluate_dense.ROOT=ROOT;evaluate_dense.evaluation.main=evaluate
    try:evaluate_dense.main()
    finally:historical.snapshot=original


if __name__=='__main__':main()
