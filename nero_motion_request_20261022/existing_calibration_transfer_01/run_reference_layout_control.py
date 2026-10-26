"""Offline control: same live start and ACT, changing only source/target layout."""
import copy
import json
import time

from run_current_scene import ACT, HERE, load_projected, rollout, full_task_passed, snapshot, require_unchanged, torch, np


torch.set_num_threads(2)
torch.manual_seed(42)
np.random.seed(42)
source = HERE/'current_scene_full/scene_configuration.json'
scene = json.loads(source.read_text())
reference = json.loads((ACT/'results/unqualified_nominal_full_development/scene_configuration.json').read_text())
for key in ['pick_xy_m', 'place_xy_m', 'visual_marker_xy_m']:
    scene[key] = reference[key]
policy, stats = load_projected()
hashes = snapshot([source, ACT/'training/projected_A/policy_last.ckpt', ACT/'checked_runtime.py'])
start = time.monotonic()
count = [0]


def progress(module, args, output):
    count[0] += 1
    if count[0] % 400 == 0:
        print(json.dumps(dict(control_queries=count[0], runtime_s=round(time.monotonic()-start, 2))), flush=True)


hook = policy.register_forward_hook(progress)
try:
    scenario = dict(name='training_layout_same_live_start_control', seed=191899,
                    query_stride=2, servo_delay_steps=0, observation_age_s=.05,
                    jitter=False, force=False, scene=copy.deepcopy(scene))
    result = rollout(scene, policy, stats, scenario, HERE/'reference_layout_control',
                     runtime_profile='history_common_v1')
finally:
    hook.remove()
require_unchanged(hashes)
report = dict(mode='counterfactual_training_layout_not_current_task',
              full_task_passed=full_task_passed(result), result=result,
              source_sha256=hashes, same_live_start=True, same_model=True,
              same_retained_calibration=True, only_task_layout_changed=True,
              hardware_commands_sent=False, hardware_deployment_allowed=False)
(HERE/'reference_layout_control_summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['result', 'source_sha256']}, indent=2), flush=True)
