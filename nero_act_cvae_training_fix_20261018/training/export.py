"""Freeze a completed full-task training result; hardware permission stays disabled."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import pickle

import torch
from executor.model import verify_paired_readout_statistics

ROOT = Path(__file__).resolve().parents[1]


def export_model(checkpoint, destination):
    checkpoint, destination = Path(checkpoint), Path(destination)
    summary_path = checkpoint.parent.parent/'artifacts/training_summary.json'
    bundle = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if bundle.get('initialization_only'):
        raise ValueError('An initializer is not a trained execution model')
    config = bundle['config']
    integrity=None
    if config.get('training_integrity_required') or config.get('training_revision')=='position_generalization_causal_context_v1':
        from training.integrity import require_finalized
        integrity=require_finalized(checkpoint,config)
    with (checkpoint.parent/'dataset_stats.pkl').open('rb') as stream:
        verify_paired_readout_statistics(config, pickle.load(stream))
    if (not summary_path.exists()
            or json.loads(summary_path.read_text())['last_metrics']['epoch'] != config['num_epochs']):
        raise ValueError('Training has not completed its declared epochs')
    if (config.get('task_scope') != 'complete_pick_carry_place_retreat'
            or config.get('chunk_size') != 20 or not config.get('bounded_gripper_output')):
        raise ValueError('Only the trained bounded full-task ACT can be exported')
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint, destination/'policy.ckpt')
    shutil.copy2(checkpoint.parent/'dataset_stats.pkl', destination/'dataset_stats.pkl')
    if integrity is not None:shutil.copy2(integrity,destination/'training_integrity.json')
    (destination/'training_config.json').write_text(json.dumps(config, indent=2))
    source_files = list((ROOT/'policy_source').rglob('*.py')) + [
        ROOT/'executor/model.py', ROOT/'executor/policy_worker.py',ROOT/'executor/shadow.py', ROOT/'executor/bundle.py', ROOT/'scene/config.json']
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_files}
    scene = config.get('scene_config', json.loads((ROOT/'scene/config.json').read_text()))
    (destination/'scene_config.json').write_text(json.dumps(scene, indent=2))
    metadata = dict(task_scope=config['task_scope'], action_chunk_shape=[20, 8],
                    selected_epoch=bundle['epoch'], declared_epochs=config['num_epochs'],
                    checkpoint_sha256=hashlib.sha256((destination/'policy.ckpt').read_bytes()).hexdigest(),
                    stats_sha256=hashlib.sha256((destination/'dataset_stats.pkl').read_bytes()).hexdigest(),
                    training_config_sha256=hashlib.sha256((destination/'training_config.json').read_bytes()).hexdigest(),
                    scene_sha256=hashlib.sha256((destination/'scene_config.json').read_bytes()).hexdigest(),
                    source_sha256=source_hashes, hardware_deployment_allowed=False,
                    real_full_task_verified=False, simulation_evaluation_completed=False,
                    simulation_evaluation_passed=False, source_training_summary=str(summary_path),
                    initial_state=scene['initial_robot_state'], tcp_link7_xyz_m=scene['tcp_link7_xyz_m'],
                    tcp_independently_measured=False, table_top_z_m=scene['table_top_z_m'])
    if integrity is not None:
        metadata['training_integrity_sha256']=hashlib.sha256(integrity.read_bytes()).hexdigest()
    (destination/'runtime_metadata.json').write_text(json.dumps(metadata, indent=2))
    schema = dict(task='complete_pick_carry_place_retreat',
                  physical_state_dim=8,policy_input_dim=config.get('policy_input_dim',config['state_dim']),
                  gripper_command_context=config.get('gripper_command_context'),
                  context_definition='(previous submitted gripper target - actual opening)/0.01m' if config.get('gripper_command_context') else None,
                  state_order=[f'joint{i}_rad' for i in range(1, 8)]+['gripper_total_width_m'],
                  action_order=[f'absolute_joint{i}_target_rad' for i in range(1, 8)]+['gripper_total_width_target_m'],
                  action_chunk_shape=[20, 8], execution_prefix_points=10, nominal_point_hz=20,
                  maximum_arm_step_rad=.002, maximum_gripper_step_m=.0005,
                  camera='camera_0 RGB upright rotated180 once, then90x160',
                  gripper_head='bounded0..95mm inside both training and inference',
                  feedback_readout=config.get('feedback_readout_version', 'legacy_unbounded_arm'),
                  continuity_by_construction=bool(config.get('bounded_feedback_residual_output')),
                  task_success_requires='actual lift, nonzero transfer, supported release, retreat and5s stability')
    (destination/'input_action_schema.json').write_text(json.dumps(schema, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export_model(args.checkpoint, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
