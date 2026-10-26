"""Full closed-loop development and unchanged original qualification; no hardware."""
import copy
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from learning import ROOT, load_for_learning
from pipeline import require_checkpoint, snapshot, require_unchanged
from simulation.supervised_rollout import rollout
from simulation.visual_evaluation import full_task_passed


def load_candidate():
    folder = ROOT/'training/pilot_A'
    checkpoint = folder/'policy_last.ckpt'
    proof = json.loads((folder/'integrity.json').read_text())
    require_checkpoint(proof, checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('profile') != 'position_projection_transformer_pilot_v1' or payload.get('actual_updates') != 1000:
        raise ValueError('Wrong pilot checkpoint or optimizer budget')
    policy, stats, _ = load_for_learning()
    if payload['config'] != policy.config or any(not np.array_equal(payload['stats'][k], stats[k]) for k in stats):
        raise ValueError('Paired unchanged inference config/statistics required')
    policy.load_state_dict(payload['model'], strict=True)
    return policy.requires_grad_(False).eval(), stats


def main():
    torch.set_num_threads(2); torch.manual_seed(42); np.random.seed(42)
    policy, stats = load_candidate()
    sources = ROOT.parent/'nero_motion_request_20261022/existing_calibration_transfer_01'
    current = json.loads((sources/'current_scene_full/scene_configuration.json').read_text())
    reference = json.loads((sources/'reference_layout_control/scene_configuration.json').read_text())
    protocol = json.loads((ROOT/'data/scene_protocol.json').read_text())
    cases = [('current_development', current), ('reference_layout', reference)]
    for config in protocol:
        if config['generalization']['split'] == 'val':
            # Keep evaluation dynamics/budgets identical to the original stopped current-scene run.
            scene = copy.deepcopy(current)
            for key in ['pick_xy_m', 'place_xy_m', 'visual_marker_xy_m']:
                scene[key] = config[key]
            cases.append((config['generalization']['source_group'], scene))
    outputs = ROOT/'results'; outputs.mkdir(exist_ok=True)
    hashes = snapshot([Path(__file__), ROOT/'learning.py', ROOT/'training/pilot_A/policy_last.ckpt',
                       ROOT/'training/pilot_A/integrity.json', ROOT/'data/scene_protocol.json',
                       sources/'current_scene_full/scene_configuration.json',
                       sources/'reference_layout_control/scene_configuration.json'])
    records = []
    for index, (name, scene) in enumerate(cases):
        started = time.monotonic()
        scenario = dict(name=name, seed=191899+index, query_stride=2, servo_delay_steps=0,
                        observation_age_s=.05, jitter=False, force=False, scene=copy.deepcopy(scene))
        result = rollout(copy.deepcopy(scene), policy, stats, scenario, outputs/name,
                         runtime_profile='history_common_v1')
        record = dict(name=name, full_task_passed=full_task_passed(result), result=result,
                      elapsed_s=time.monotonic()-started)
        records.append(record)
        (outputs/'full_task_summary.json').write_text(json.dumps(dict(complete=len(records)==len(cases),
            records=records, source_sha256=hashes, hardware_commands_sent=False,
            hardware_deployment_allowed=False), indent=2)+'\n')
        print(json.dumps(dict(name=name, passed=record['full_task_passed'],
            error=result.get('error'), points=result.get('new_active_points'),
            lift_m=result.get('max_lift_m'), elapsed_s=record['elapsed_s'])), flush=True)
    require_unchanged(hashes)
    # Reuse the six original snapshot cases and all original input/preservation thresholds.
    path = ROOT.parent/'nero_act_command_increment_20261021/run_qualification.py'
    spec = importlib.util.spec_from_file_location('unchanged_original_qualification', path)
    evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    evaluator.ROOT = ROOT
    evaluator.load_policy = load_candidate
    for directory in ['results/mechanism_original', 'results/nominal_after']:
        (ROOT/directory).mkdir(exist_ok=True)
    evaluator.main()
    require_unchanged(hashes)


if __name__ == '__main__':
    main()
