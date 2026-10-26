"""Artifact validation without loading model, ROS or SDK namespaces."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_model_bundle(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'runtime_metadata.json').read_text())
    if metadata.get('training_integrity_sha256'):
        if hashlib.sha256((directory/'training_integrity.json').read_bytes()).hexdigest()!=metadata['training_integrity_sha256']:
            raise ValueError('Finalized training integrity changed')
    if (metadata.get('task_scope') != 'complete_pick_carry_place_retreat'
            or metadata.get('action_chunk_shape') != [20, 8]):
        raise ValueError('Wrong model task or prediction window')
    for name, key in [('policy.ckpt', 'checkpoint_sha256'), ('dataset_stats.pkl', 'stats_sha256')]:
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != metadata[key]:
            raise ValueError(f'Frozen model checksum mismatch: {name}')
    if ('scene_sha256' in metadata and hashlib.sha256((directory/'scene_config.json').read_bytes()).hexdigest()
            != metadata['scene_sha256']):
        raise ValueError('Frozen scene checksum mismatch')
    if ('training_config_sha256' in metadata and hashlib.sha256((directory/'training_config.json').read_bytes()).hexdigest()
            != metadata['training_config_sha256']):
        raise ValueError('Frozen training configuration checksum mismatch')
    for name, expected in metadata.get('source_sha256', {}).items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Inference source changed: {name}')
    return metadata

