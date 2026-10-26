"""Optimization provenance must finalize before a generalized checkpoint can export."""
import hashlib,json
from pathlib import Path


def snapshot(paths):
    return {str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def require_unchanged(hashes):
    for name,expected in hashes.items():
        path=Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Training input/source changed: '+name)


def require_finalized(checkpoint,config):
    path=Path(checkpoint).parent.parent/'artifacts/training_integrity.json'
    if not path.exists():raise ValueError('Generalized training integrity was not finalized')
    proof=json.loads(path.read_text())
    if (proof.get('finalized') is not True or proof.get('declared_epochs')!=config['num_epochs']
            or proof.get('actual_epochs')!=config['num_epochs'] or not proof.get('protected_sha256')
            or not proof.get('output_sha256')):
        raise ValueError('Incomplete or mismatched generalized training integrity')
    require_unchanged(proof['protected_sha256']);require_unchanged(proof['output_sha256'])
    if proof['output_sha256'].get(str(Path(checkpoint).resolve()))!=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
        raise ValueError('Checkpoint not bound to finalized training run')
    return path
