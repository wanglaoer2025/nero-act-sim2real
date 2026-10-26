"""Run unchanged complete-task assessment only after actual position response passes."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from learning import ROOT,SOURCE,load_for_learning
from pipeline import require_checkpoint,snapshot


def load_candidate():
    path=ROOT/'training/pilot_A/policy_last.ckpt'
    proof=json.loads((path.parent/'integrity.json').read_text())
    require_checkpoint(proof,path)
    gate=json.loads((path.parent/'response_gate.json').read_text())
    if not gate['passed'] or gate['checkpoint_sha256']!=snapshot([path])[str(path.resolve())]:
        raise ValueError('Position response gate did not pass for this exact checkpoint')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('profile')!='position_response_loss_repair_v1' or payload.get('actual_updates')!=1000:
        raise ValueError('Wrong checkpoint profile or optimizer budget')
    policy,stats,_=load_for_learning()
    if payload['config']!=policy.config or any(not np.array_equal(payload['stats'][k],stats[k]) for k in stats):
        raise ValueError('Checkpoint/config/statistics mismatch')
    policy.load_state_dict(payload['model'],strict=True)
    return policy.requires_grad_(False).eval(),stats


def main():
    gate=json.loads((ROOT/'training/pilot_A/response_gate.json').read_text())
    if not gate['passed']:
        result=dict(full_task_evaluated=False,full_task_passed=False,
            reason='Pregrasp position response gate failed; no promotion or hardware execution',
            hardware_commands_sent=False)
        (ROOT/'full_evaluation_scope.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result));return
    spec=importlib.util.spec_from_file_location('unchanged_full_assessment',SOURCE/'evaluate.py')
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    evaluator.ROOT=ROOT;evaluator.load_candidate=load_candidate
    evaluator.main()


if __name__=='__main__':main()
