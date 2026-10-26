"""Full current flow, followed by unchanged original qualification when it completes."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from train_repair import ROOT,load_initializer,require_unchanged,snapshot
from simulation.supervised_rollout import rollout
from simulation.visual_evaluation import full_task_passed


def load_candidate():
    folder=ROOT/'training/pilot_A';checkpoint=folder/'policy_last.ckpt'
    proof=json.loads((folder/'integrity.json').read_text())
    if proof.get('complete') is not True or proof.get('actual_updates')!=1000 or proof.get('hardware_deployment_allowed') is not False:
        raise ValueError('Completed fixed1000-update offline candidate required')
    require_unchanged(proof['source_sha256']);require_unchanged(proof['checkpoint_sha256'])
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['profile']!='full_flow_recovery_v1' or payload['actual_updates']!=1000:raise ValueError('Wrong candidate')
    policy,stats,_=load_initializer()
    if payload['config']!=policy.config or any(not np.array_equal(payload['stats'][k],stats[k]) for k in stats):
        raise ValueError('Unchanged ACT configuration/statistics required')
    policy.load_state_dict(payload['model'],strict=True)
    return policy.requires_grad_(False).eval(),stats


def main():
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    policy,stats=load_candidate();scene=json.loads((ROOT/'current_scene.json').read_text())
    output=ROOT/'results';output.mkdir(exist_ok=True)
    protected=snapshot([Path(__file__),ROOT/'current_scene.json',ROOT/'training/pilot_A/policy_last.ckpt',
        ROOT/'training/pilot_A/integrity.json'])
    scenario=dict(name='restored_current_full_flow',seed=191899,query_stride=2,
        servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False)
    baseline,baseline_stats,_=load_initializer();baseline.requires_grad_(False).eval()
    baseline_result=rollout(copy.deepcopy(scene),baseline,baseline_stats,scenario,output/'baseline_same_scene',runtime_profile='history_common_v1')
    del baseline
    print(json.dumps(dict(stage='same_scene_baseline',success=baseline_result['success'],error=baseline_result['error'],
        lift_m=baseline_result['max_lift_m'])),flush=True)
    torch.manual_seed(42);np.random.seed(42)
    result=rollout(copy.deepcopy(scene),policy,stats,scenario,output/'restored_current',runtime_profile='history_common_v1')
    passed=full_task_passed(result);require_unchanged(protected)
    summary=dict(current_full_flow_passed=passed,current_result={k:result.get(k) for k in ['success','error',
        'active_policy_points','queries','max_lift_m','maximum_horizontal_transfer_m','stable_released_frames',
        'simulated_events','unexpected_arm_cup_contact_frames','execution_contract_failures','runtime_s']},
        source_sha256=protected,nominal_scene_only=True,calibration_verified=False,
        hardware_commands_sent=False,hardware_deployment_allowed=False,original_qualification_run=False)
    summary['baseline_same_scene']={k:baseline_result.get(k) for k in summary['current_result']}
    (output/'full_flow_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['current_result']),flush=True)
    if not passed:return
    path=ROOT.parent/'nero_act_command_increment_20261021/run_qualification.py'
    spec=importlib.util.spec_from_file_location('unchanged_repair_qualification',path)
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    evaluator.ROOT=ROOT;evaluator.load_policy=load_candidate
    for name in ['results/mechanism_original','results/nominal_after']:(ROOT/name).mkdir(exist_ok=True)
    evaluator.main();require_unchanged(protected)
    summary['original_qualification_run']=True
    original=json.loads((ROOT/'results/qualification/summary.json').read_text())
    summary['original_six_pass_count']=original['original_six_pass_count']
    summary['original_exact_and_preservation_gate_passed']=original['exact_and_preservation_gate_passed']
    (output/'full_flow_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
