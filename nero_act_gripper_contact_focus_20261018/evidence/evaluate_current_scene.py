"""Offline frozen ACT on measured current geometry; never grants execution permission."""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import torch

from executor.bundle import verify_model_bundle
from executor.evaluation_integrity import evaluation_sources, verify_frozen_artifacts, SUPERVISED_PROTOCOL
from executor.model import BoundedActPolicy, verify_paired_readout_statistics
from simulation.evaluate import scenarios, run_frozen_cases
from simulation.supervised_rollout import rollout

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nominal-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    directory = ROOT / 'model/run26_contact_focus'
    metadata = verify_model_bundle(directory)
    payload = torch.load(directory / 'policy.ckpt', map_location='cpu', weights_only=False)
    config = dict(payload['config'], device='cuda', pretrained_backbone=False)
    geometry_path = ROOT / 'evidence/current_fixed_scene_geometry_no_viewer/report.json'
    geometry = json.loads(geometry_path.read_text())
    assert geometry['success'] and geometry['geometry_fit_passed']
    feedback_path = ROOT / 'evidence/run26_live_readonly_shadow_no_viewer/predictions.json'
    feedback = json.loads(feedback_path.read_text())[-1]['actual_feedback']['state']
    scene = copy.deepcopy(config['scene_config'])
    observed = geometry['observations'][-1]
    scene['pick_xy_m'] = observed['cup_xyz'][:2]
    scene['place_xy_m'] = observed['target_xyz'][:2]
    scene['visual_marker_xy_m'] = scene['place_xy_m']
    scene['initial_robot_state'] = dict(qpos_rad=feedback[:7], gripper_total_width_m=.095)
    scene['current_scene_trial_note'] = 'Same frozen ACT; current measured arm/cup/card; high-open95mm is planned, not actual80.894mm'
    with (directory / 'dataset_stats.pkl').open('rb') as stream:
        stats = pickle.load(stream)
    verify_paired_readout_statistics(config, stats)
    torch.set_num_threads(2)
    policy = BoundedActPolicy(config).cuda().eval()
    policy.load_state_dict(payload['model'], strict=True)
    policy.config, policy.device = config, torch.device('cuda')
    cases = scenarios()[:1] if args.nominal_only else scenarios()
    cases = [dict(case, scene=copy.deepcopy(scene)) for case in cases]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = dict(task_scope=config['task_scope'],
        **{key: metadata[key] for key in ['checkpoint_sha256', 'stats_sha256', 'scene_sha256', 'training_config_sha256']},
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in evaluation_sources()},
        scenarios=cases, execution_scene=scene,
        geometry_source=str(geometry_path), geometry_source_sha256=digest(geometry_path),
        feedback_source=str(feedback_path), feedback_source_sha256=digest(feedback_path),
        assessment_role='current_geometry_diagnostic_not_legacy_execution_scene_acceptance',
        trained_scene_unchanged=True, planned_high_open_not_yet_performed=True,
        teacher_free=True, supervised_contact_protocol=SUPERVISED_PROTOCOL,
        real_hardware_tested=False, hardware_deployment_allowed=False)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    run_frozen_cases(args.output, protocol,
        lambda case: rollout(case['scene'], policy, stats, case, args.output / case['name']),
        lambda: verify_frozen_artifacts(directory / 'policy.ckpt', protocol))


if __name__ == '__main__':
    main()
