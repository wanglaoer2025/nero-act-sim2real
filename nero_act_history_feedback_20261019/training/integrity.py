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


def require_finalized(checkpoint,config,*,allowed_budgets=('epochs',)):
    path=Path(checkpoint).parent.parent/'artifacts/training_integrity.json'
    if not path.exists():raise ValueError('Generalized training integrity was not finalized')
    proof=json.loads(path.read_text())
    if proof.get('schema')=='history_training_integrity_v1':
        budget=config.get('training_budget',{})
        kind=budget.get('kind');limit=budget.get('limit')
        if (kind not in allowed_budgets or (kind,limit) not in [('updates',1000),('epochs',20)]
                or proof.get('finalized') is not True or proof.get('declared_budget')!=budget
                or proof.get('actual_'+str(kind))!=limit
                or (config.get('declared_optimizer_steps') is not None and proof.get('actual_updates')!=config['declared_optimizer_steps'])
                or not proof.get('protected_sha256') or not proof.get('output_sha256')
                or (kind=='updates' and config.get('num_epochs')!=0)
                or (kind=='epochs' and config.get('num_epochs')!=20)):
            raise ValueError('Incomplete history budget or pilot used outside its explicit offline purpose')
        require_unchanged(proof['protected_sha256']);require_unchanged(proof['output_sha256'])
        if proof['output_sha256'].get(str(Path(checkpoint).resolve()))!=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
            raise ValueError('History checkpoint not bound to finalized training run')
        return path
    if (proof.get('finalized') is not True or proof.get('declared_epochs')!=config['num_epochs']
            or proof.get('actual_epochs')!=config['num_epochs'] or not proof.get('protected_sha256')
            or not proof.get('output_sha256')):
        raise ValueError('Incomplete or mismatched generalized training integrity')
    require_unchanged(proof['protected_sha256']);require_unchanged(proof['output_sha256'])
    if proof['output_sha256'].get(str(Path(checkpoint).resolve()))!=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
        raise ValueError('Checkpoint not bound to finalized training run')
    return path
