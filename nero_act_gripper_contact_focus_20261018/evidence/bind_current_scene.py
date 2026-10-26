"""Bind a deployment scene without optimization; preserve original training provenance."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import torch

from executor.bundle import verify_model_bundle
from executor.policy_worker import FrozenAct
from training.export import export_model
from training.integrity import require_finalized, require_unchanged, snapshot

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'model/run26_contact_focus'
    metadata = verify_model_bundle(source)
    payload = torch.load(source / 'policy.ckpt', map_location='cpu', weights_only=False)
    original_config = copy.deepcopy(payload['config'])
    training_checkpoint = Path(original_config['checkpoint_dir']) / 'policy_best_val.ckpt'
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(training_checkpoint) == metadata['checkpoint_sha256']
    original_proof_path = require_finalized(training_checkpoint, original_config)
    original_proof = json.loads(original_proof_path.read_text())
    prototype_path = ROOT / 'evidence/run26_current_geometry_nominal/summary.json'
    prototype = json.loads(prototype_path.read_text())
    assert prototype['finalized'] and prototype['passed'] == prototype['total'] == 1
    assert prototype['checkpoint_sha256'] == metadata['checkpoint_sha256']
    scene = prototype['execution_scene']
    destination = ROOT / 'deployment/run26_current_fixed_scene'
    destination.mkdir(parents=True, exist_ok=False)
    artifacts, checkpoints = destination / 'artifacts', destination / 'checkpoints'
    artifacts.mkdir(); checkpoints.mkdir()
    shutil.copy2(original_proof_path, artifacts / 'source_training_integrity.json')
    config = copy.deepcopy(original_config)
    config.update(source_training_scene_config=original_config['scene_config'], scene_config=scene,
        deployment_scene_binding=dict(mode='posttraining_static_scene_binding_no_optimization',
            source_checkpoint_sha256=metadata['checkpoint_sha256'],
            original_training_config_sha256=metadata['training_config_sha256'],
            original_training_integrity_sha256=digest(original_proof_path),
            prototype_sha256=digest(prototype_path),
            prepared_open_width_m=.095, actual_open_preparation_not_yet_performed=True))
    payload = dict(payload, config=config, posttraining_scene_binding=True,
        additional_optimization_steps=0, source_checkpoint_sha256=metadata['checkpoint_sha256'])
    checkpoint = checkpoints / 'policy_scene_bound.ckpt'
    torch.save(payload, checkpoint)
    shutil.copy2(source / 'dataset_stats.pkl', checkpoints / 'dataset_stats.pkl')
    summary_source = Path(original_config['artifact_dir']) / 'training_summary.json'
    summary = json.loads(summary_source.read_text())
    summary.update(deployment_derivation='scene binding only; optimization metrics copied from verified original run26',
        source_training_summary=str(summary_source), additional_optimization_steps=0)
    summary_path = artifacts / 'training_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    config_path = destination / 'scene_binding_config.json'
    config_path.write_text(json.dumps(config, indent=2))
    protected = dict(original_proof['protected_sha256'])
    protected.update(snapshot([Path(__file__), training_checkpoint, original_proof_path,
        prototype_path, Path(prototype['geometry_source']), Path(prototype['feedback_source']),
        config_path, artifacts / 'source_training_integrity.json']))
    proof = dict(finalized=True, declared_epochs=original_proof['declared_epochs'],
        actual_epochs=original_proof['actual_epochs'], epochs_describe_original_training=True,
        additional_optimization_steps=0, protected_sha256=protected,
        output_sha256=snapshot([checkpoint, checkpoints / 'dataset_stats.pkl', summary_path]),
        derivation='Same complete trained tensors/statistics, current deployment scene binding',
        source_training_proof_sha256=digest(original_proof_path))
    require_unchanged(protected)
    (artifacts / 'training_integrity.json').write_text(json.dumps(proof, indent=2))
    exported = ROOT / 'model/run26_current_fixed_scene'
    export_model(checkpoint, exported)
    new_payload = torch.load(exported / 'policy.ckpt', map_location='cpu', weights_only=False)
    assert new_payload['model'].keys() == payload['model'].keys()
    assert all(torch.equal(payload['model'][key], new_payload['model'][key]) for key in payload['model'])
    assert digest(source / 'dataset_stats.pkl') == digest(exported / 'dataset_stats.pkl')
    # Same actual camera/feedback inputs must retain bit-identical predictions.
    import cv2
    old_policy, new_policy = FrozenAct(source), FrozenAct(exported)
    records = json.loads(Path(prototype['feedback_source']).read_text())
    maximum = 0.
    for record in records:
        details = record['prediction']['details']
        rgb = cv2.cvtColor(cv2.imread(details['camera_frame']), cv2.COLOR_BGR2RGB)
        state, previous = details['feedback_state'], details['previous_gripper_target_m']
        before, after = old_policy.predict(rgb, state, previous), new_policy.predict(rgb, state, previous)
        maximum = max(maximum, float(abs(before-after).max()))
    assert maximum == 0.
    report = dict(passed=True, original_training_preserved=True,
        all_trained_tensors_bit_identical=True, normalization_file_identical=True,
        actual_shadow_prediction_max_difference=maximum, additional_optimization_steps=0,
        source_model=str(source), deployment_model=str(exported),
        original_checkpoint_sha256=metadata['checkpoint_sha256'],
        deployment_checkpoint_sha256=digest(exported / 'policy.ckpt'),
        original_training_scene_retained_in_config=True, deployment_scene=scene,
        execution_ready=False, hardware_motion_sent=False)
    (ROOT / 'evidence/current_scene_binding_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'deployment_scene'}, indent=2))


if __name__ == '__main__':
    main()
