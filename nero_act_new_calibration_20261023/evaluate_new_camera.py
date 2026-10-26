"""Frozen ACT policy, new hand-eye extrinsic, fixed-world offline sensitivity test."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
POLICY_ROOT = ROOT.parent/'nero_act_preservation_scope_20261023'
sys.path.insert(0, str(ROOT.parent/'nero_act_dense_response_20261023'))
import evaluate_dense


def current_scene_only():
    evaluation = evaluate_dense.evaluation
    evaluation.ROOT = POLICY_ROOT
    policy, stats = evaluation.load_candidate()
    scene = json.loads((ROOT/'scene_new_camera.json').read_text())
    scenario = dict(name='new_handeye_fixed_world_sensitivity', seed=191899, query_stride=2,
        servo_delay_steps=0, observation_age_s=.05, jitter=False, force=False)
    hashes = evaluation.snapshot([Path(__file__),ROOT/'scene_new_camera.json',
        ROOT/'inputs/candidate_camera_to_base.json',POLICY_ROOT/'training/pilot_A/policy_last.ckpt'])
    output = ROOT/'results'
    output.mkdir(exist_ok=True)
    result = evaluate_dense.runtime.rollout(scene,policy,stats,scenario,output/'fixed_world',
                                           runtime_profile='history_common_v1')
    evaluation.require_unchanged(hashes)
    summary = dict(complete=True,full_task_passed=evaluation.full_task_passed(result),result=result,
        changed_variable='camera extrinsic only; same arm/world/cup/target/intrinsics/TCP/weights/stats',
        calibration_candidate_sha256=hashes[str((ROOT/'inputs/candidate_camera_to_base.json').resolve())],
        actual_current_scene_relocalized=False,live_camera_opened=False,optimizer_updates=0,
        actual_hardware_pose_verified=False,calibration_independent_geometry_verified=False,
        hardware_commands_sent=False,hardware_deployment_allowed=False,source_sha256=hashes)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ['success','error','active_policy_points','simulated_events','max_lift_m']}),flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    evaluate_dense.ROOT = ROOT
    evaluate_dense.evaluation.main = current_scene_only
    evaluate_dense.main()
