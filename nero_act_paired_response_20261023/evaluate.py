"""Keep original response and complete-task gates for the paired-response candidate."""
import importlib.util
import json

import numpy as np
import torch

from experiment import ROOT,PROFILE,load_latest,require_unchanged,snapshot


def load_candidate():
    folder=ROOT/'training/pilot_A';checkpoint=folder/'policy_last.ckpt'
    proof=json.loads((folder/'integrity.json').read_text())
    if (proof.get('complete') is not True or proof.get('actual_updates')!=2000
            or proof.get('formal_epochs')!=0 or proof.get('hardware_deployment_allowed') is not False
            or not proof.get('source_sha256') or str(checkpoint.resolve()) not in proof.get('checkpoint_sha256',{})):
        raise ValueError('Finalized offline2000-update paired-response proof required')
    require_unchanged(proof['source_sha256']);require_unchanged(proof['checkpoint_sha256'])
    gate=json.loads((folder/'response_gate.json').read_text())
    if not gate['passed'] or gate['checkpoint_sha256']!=snapshot([checkpoint])[str(checkpoint.resolve())]:
        raise ValueError('Original position-response gate must pass for this exact checkpoint')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload.get('profile')!=PROFILE or payload.get('actual_updates')!=2000:
        raise ValueError('Wrong paired-response checkpoint or training budget')
    policy,stats,_,_=load_latest()
    if payload['config']!=policy.config or any(not np.array_equal(payload['stats'][k],stats[k]) for k in stats):
        raise ValueError('Unchanged paired ACT configuration/statistics required')
    policy.load_state_dict(payload['model'],strict=True)
    return policy.requires_grad_(False).eval(),stats


def main():
    scope=ROOT/'scope.json'
    if scope.exists() and not json.loads(scope.read_text())['candidate_trained']:
        result=dict(full_task_evaluated=False,full_task_passed=False,reason='Training-only learnability probe failed',hardware_commands_sent=False)
    else:
        gate=json.loads((ROOT/'training/pilot_A/response_gate.json').read_text())
        if gate['passed']:
            path=ROOT.parent/'nero_act_position_recovery_20261022/evaluate.py'
            spec=importlib.util.spec_from_file_location('unchanged_complete_assessment',path)
            evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
            evaluator.ROOT=ROOT;evaluator.load_candidate=load_candidate
            evaluator.snapshot=lambda paths:snapshot([ROOT/'experiment.py' if p==ROOT/'learning.py' else p for p in paths])
            evaluator.main()
            summary_path=ROOT/'results/qualification/summary.json';summary=json.loads(summary_path.read_text())
            summary.update(actual_updates=2000,initializer_lineage_updates=6000,cumulative_lineage_updates=8000,
                diagnostic_updates=200,inherited_fixed_budget_metadata=1000)
            summary_path.write_text(json.dumps(summary,indent=2)+'\n');return
        result=dict(full_task_evaluated=False,full_task_passed=False,reason='Original position-response gate failed',hardware_commands_sent=False)
    (ROOT/'full_evaluation_scope.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
