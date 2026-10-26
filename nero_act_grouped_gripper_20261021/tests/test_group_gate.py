import json
import pytest
from tests.test_group_loss_integration import SETTINGS


def physical_records():
    return [dict(policy=policy,batch=ordinal,passed=True,cosines=[.1,.2,None,.05],noncollapse_ratio=.2)
        for policy in ['zero_update_baseline','failed_preservation_1000']
        for ordinal in [0,24,99,499,149,249,749,999,74,349,649,899]]


def test_group_gate_requires_exact_calibration_and_unchanged_evidence(tmp_path):
    from training.group_loss import require_group_diagnostics
    from training.integrity import snapshot
    source=tmp_path/'source.py';source.write_text('unchanged')
    calibration=tmp_path/'calibration.json'
    calibration.write_text(json.dumps(dict(complete=True,actual_optimizer_updates=0,source_sha256=snapshot([source]),
        cases=[dict(case='all24',weights=SETTINGS['weights'],full_original_gate_passed=True,records=[dict(passed=True)]*24)])))
    proof=dict(actual_optimizer_updates=0,source_sha256=snapshot([source,calibration]),
        grouped_loss=dict(settings=SETTINGS,calibration_path=str(calibration),actual_loss_verified=True,
            physical_directions_revalidated=True,physical_records=physical_records()))
    path=tmp_path/'diagnostics.json';path.write_text(json.dumps(proof))
    binding=require_group_diagnostics(path,SETTINGS)
    assert str(path.resolve()) in binding and str(calibration.resolve()) in binding
    for field,value in [('actual_loss_verified',False),('settings',dict(SETTINGS,weights=[.25]*4))]:
        changed=json.loads(json.dumps(proof));changed['grouped_loss'][field]=value;path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):require_group_diagnostics(path,SETTINGS)
    path.write_text(json.dumps(proof));source.write_text('changed')
    with pytest.raises(RuntimeError):require_group_diagnostics(path,SETTINGS)


def test_group_gate_rejects_missing_or_failed_production_physical_directions(tmp_path):
    from training.group_loss import require_group_diagnostics
    from training.integrity import snapshot
    source=tmp_path/'source.py';source.write_text('bound implementation')
    calibration=tmp_path/'calibration.json'
    calibration.write_text(json.dumps(dict(complete=True,actual_optimizer_updates=0,source_sha256=snapshot([source]),
        cases=[dict(case='all24',weights=SETTINGS['weights'],full_original_gate_passed=True,records=[dict(passed=True)]*24)])))
    grouped=dict(settings=SETTINGS,calibration_path=str(calibration),actual_loss_verified=True,
        physical_directions_revalidated=True,physical_records=physical_records())
    proof=dict(actual_optimizer_updates=0,source_sha256=snapshot([source,calibration]),grouped_loss=grouped)
    path=tmp_path/'diagnostics.json'
    for mutation in ['missing','false','truncated','duplicate','reversed_closing','reversed_release','collapsed']:
        changed=json.loads(json.dumps(proof));g=changed['grouped_loss']
        if mutation=='missing':g.pop('physical_records')
        elif mutation=='false':g['physical_directions_revalidated']=False
        elif mutation=='truncated':g['physical_records'].pop()
        elif mutation=='duplicate':g['physical_records'][0]=g['physical_records'][1]
        elif mutation=='reversed_closing':g['physical_records'][0]['cosines'][0]=-.1
        elif mutation=='reversed_release':g['physical_records'][0]['cosines'][3]=-.1
        elif mutation=='collapsed':g['physical_records'][0]['noncollapse_ratio']=1e-5
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):require_group_diagnostics(path,SETTINGS)
