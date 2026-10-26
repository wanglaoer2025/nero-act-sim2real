"""Offline group weights; no group labels or new inputs enter deployed inference."""
import math
import json
from pathlib import Path
import torch
from training.integrity import snapshot,require_unchanged


def validate_group_loss(settings):
    if (not isinstance(settings,dict) or set(settings)!={'profile','weights'}
            or settings['profile']!='four_phase_static_v1' or not isinstance(settings['weights'],list)
            or len(settings['weights'])!=4 or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<.05-1e-12 for v in settings['weights'])
            or abs(sum(settings['weights'])-1)>1e-8):
        raise ValueError('Declared four-group weights must retain each role and sum to one')


def group_sample_weights(groups,settings):
    validate_group_loss(settings)
    if groups.ndim!=1 or not len(groups) or groups.dtype!=torch.long or torch.any((groups<0)|(groups>3)):
        raise ValueError('Offline integer group labels0..3 required')
    counts=torch.bincount(groups,minlength=4);declared=torch.tensor(settings['weights'],device=groups.device,dtype=torch.float32)
    mass=declared[counts>0].sum()
    return len(groups)*declared[groups]/(counts[groups]*mass)


def require_group_diagnostics(path,settings):
    validate_group_loss(settings);path=Path(path);proof=json.loads(path.read_text());grouped=proof.get('grouped_loss',{})
    if (proof.get('actual_optimizer_updates')!=0 or grouped.get('settings')!=settings
            or grouped.get('actual_loss_verified') is not True or not proof.get('source_sha256')):
        raise ValueError('Matched zero-update production group-loss diagnostics required')
    records=grouped.get('physical_records',[])
    expected={(policy,ordinal) for policy in ['zero_update_baseline','failed_preservation_1000']
        for ordinal in [0,24,99,499,149,249,749,999,74,349,649,899]}
    if (grouped.get('physical_directions_revalidated') is not True or len(records)!=24
            or {(r.get('policy'),r.get('batch')) for r in records}!=expected):
        raise ValueError('Production physical directions must be recomputed for all24 records')
    for record in records:
        cosines=record.get('cosines',[]);ratio=record.get('noncollapse_ratio',float('nan'))
        if (record.get('passed') is not True or len(cosines)!=4 or not math.isfinite(ratio) or ratio<1e-3
                or any(c is None or not math.isfinite(c) or c<.01 for c in cosines[:2])
                or any(c is not None and (not math.isfinite(c) or c < -1e-5) for c in cosines[2:])):
            raise ValueError('Production physical direction/noncollapse gate failed')
    require_unchanged(proof['source_sha256'])
    calibration_path=Path(grouped['calibration_path']).resolve()
    if str(calibration_path) not in proof['source_sha256']:raise ValueError('Group calibration must be bound')
    calibration=json.loads(calibration_path.read_text())
    case=next((c for c in calibration.get('cases',[]) if c.get('case')=='all24'),{})
    if (not calibration.get('complete') or calibration.get('actual_optimizer_updates')!=0
            or case.get('weights')!=settings['weights'] or case.get('full_original_gate_passed') is not True
            or len(case.get('records',[]))!=24 or not all(r.get('passed') is True for r in case['records'])
            or not calibration.get('source_sha256')):
        raise ValueError('The unchanged24-record calibration gate must pass before training')
    require_unchanged(calibration['source_sha256'])
    return dict(proof['source_sha256'],**calibration['source_sha256'],**snapshot([path,calibration_path]))
