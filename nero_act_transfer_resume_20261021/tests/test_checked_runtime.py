import copy
import hashlib
import json
from pathlib import Path

import pytest

from checked_runtime import require_bound_scene, require_checkpoint_binding


ROOT = Path(__file__).resolve().parents[1]


def test_exact_state_recovery_rejects_a_changed_scene():
    source = ROOT/'results/exact_lift_failure'
    scene = json.loads((source/'scene_configuration.json').read_text())
    context = json.loads((source/'runtime_checkpoints.json').read_text())['records'][-1]
    require_bound_scene(scene, context)
    changed = copy.deepcopy(scene)
    changed['cup']['mass_kg'] *= 2
    with pytest.raises(ValueError, match='scene'):
        require_bound_scene(changed, context)


@pytest.mark.parametrize('kind', ['empty', 'wrong_checkpoint', 'empty_sources'])
def test_unbound_checkpoint_proofs_are_rejected_before_loading(tmp_path, kind):
    checkpoint = tmp_path/'policy.ckpt'
    checkpoint.write_bytes(b'not a pickle: rejection must precede deserialization')
    other = tmp_path/'other.ckpt'
    other.write_bytes(b'different checkpoint')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    proof = dict(source_sha256={str(other): digest(other)}, checkpoint_sha256={str(checkpoint): digest(checkpoint)})
    if kind == 'empty':
        proof['source_sha256'] = {}; proof['checkpoint_sha256'] = {}
    elif kind == 'wrong_checkpoint':
        proof['checkpoint_sha256'] = {str(other): digest(other)}
    else:
        proof['source_sha256'] = {}
    with pytest.raises(ValueError):
        require_checkpoint_binding(proof, checkpoint)


def test_loader_rejects_empty_maps_before_any_torch_deserialization(tmp_path, monkeypatch):
    import checked_runtime
    import torch
    run = tmp_path/'training/pilot_A'
    run.mkdir(parents=True)
    (run/'integrity.json').write_text(json.dumps(dict(source_sha256={}, checkpoint_sha256={})))
    monkeypatch.setattr(checked_runtime, 'ROOT', tmp_path)
    def forbidden_load(*args, **kwargs):
        raise AssertionError('Malformed proof reached checkpoint deserialization')
    monkeypatch.setattr(torch, 'load', forbidden_load)
    with pytest.raises(ValueError):
        checked_runtime.load_projected()
