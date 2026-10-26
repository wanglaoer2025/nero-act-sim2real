"""Strict public entry; historical sealed training sources remain immutable."""
import hashlib
import json
from pathlib import Path

from arm_tuning import ROOT
from pipeline import require_unchanged


def require_bound_scene(scene, checkpoint):
    digest = hashlib.sha256(json.dumps(scene, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if checkpoint.get('bindings', {}).get('scene_sha256') != digest:
        raise ValueError('Recovery scene does not match the original checkpoint scene binding')


def require_checkpoint_binding(proof, checkpoint):
    path = str(Path(checkpoint).resolve())
    sources, checkpoints = proof.get('source_sha256'), proof.get('checkpoint_sha256')
    if (not isinstance(sources, dict) or not sources or not isinstance(checkpoints, dict)
            or path not in checkpoints):
        raise ValueError('Nonempty source provenance and exact checkpoint-path binding required')
    require_unchanged(sources)
    require_unchanged(checkpoints)


def collect_lift_prefix(scene, checkpoint, output):
    from lift_recovery import collect_lift_prefix as historical_teacher
    require_bound_scene(scene, checkpoint)
    return historical_teacher(scene, checkpoint, output)


def load_projected():
    from arm_projection import load_projected as historical_loader
    for folder in ['pilot_A', 'projected_A']:
        path = ROOT/'training'/folder
        proof = json.loads((path/'integrity.json').read_text())
        require_checkpoint_binding(proof, path/'policy_last.ckpt')
    return historical_loader()
