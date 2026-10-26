import hashlib
import importlib
import itertools
import json

import pytest


def api():
    assert importlib.util.find_spec('pilot_revalidate'), 'Full-data zero-update revalidation is missing'
    return importlib.import_module('pilot_revalidate')


def fixture(tmp_path):
    module = api()
    root = tmp_path
    old = {'sha256': {}}
    for name in module.CORE_FILES:
        (root / name).write_text('unchanged-training-core')
        old['sha256'][name] = module.sha(root / name)
    binding = root / 'current.json'; binding.write_text(json.dumps(old))
    original_binding = root / 'original.json'; original_binding.write_text(json.dumps(old))
    source = root / 'limited.json'
    source.write_text(json.dumps(dict(complete=True, updates=200, formal=False, mechanism_passed=True,
        full_main_data_admission=False, admission_scope='mechanism_pilot_only',
        implementation_binding_sha256=module.sha(original_binding))))
    checkpoint = root / 'pilot.ckpt'; checkpoint.write_bytes(b'immutable original checkpoint')
    (root / 'observer.py').write_text('current-observer')
    equivalent = root / 'equivalence.json'; equivalent.write_text(json.dumps(
        {'compatible': True, 'new_observer_sha256': module.sha(root / 'observer.py')}))
    data = root / 'data.json'; data.write_text('full data is independently checked by train.preflight')
    records = [dict(position_group=g, stage=s, queries=10, passed=True,
                    net_target_change_m=(-.001 if s == 'close' else .001 if s == 'release' else 0.))
               for g, s in itertools.product(module.GROUPS, ['close', 'carry', 'release'])]
    cases = [g + '/' + s for g, s in itertools.product(module.GROUPS, module.STAGES)]
    report = dict(schema=module.SCHEMA, complete=True, optimizer_updates=0, mechanism_passed=True,
        full_main_data_admission=True, data_manifest_sha256=module.sha(data),
        implementation_binding_sha256=module.sha(binding), records=records,
        gradient_batches=[dict(cases=cases, norms={p: 1. for p in module.GRADIENT_PREFIXES})])
    for key, path in [('source_pilot_completion', source), ('source_checkpoint', checkpoint),
                      ('source_implementation', original_binding), ('observer_equivalence', equivalent)]:
        report[key] = {'path': str(path), 'sha256': module.sha(path)}
    proof = root / 'proof.json'; proof.write_text(json.dumps(report))
    return module, root, proof, data, binding, report


def test_complete_revalidation_admits_but_original_limited_pilot_alone_does_not(tmp_path):
    m, root, proof, data, binding, report = fixture(tmp_path)
    assert not m.admission_errors(proof, data, binding, root=root)
    assert m.admission_errors(root / 'limited.json', data, binding, root=root)


@pytest.mark.parametrize('change', ['missing_group', 'failed_direction', 'extra_updates', 'missing_gradient'])
def test_revalidation_refuses_incomplete_or_failed_checks(tmp_path, change):
    m, root, proof, data, binding, report = fixture(tmp_path)
    if change == 'missing_group': report['records'] = report['records'][:-1]
    elif change == 'failed_direction': report['records'][0]['net_target_change_m'] = .001
    elif change == 'extra_updates': report['optimizer_updates'] = 1
    else: report['gradient_batches'][0]['norms'][m.GRADIENT_PREFIXES[0]] = 0.
    proof.write_text(json.dumps(report))
    assert m.admission_errors(proof, data, binding, root=root)


@pytest.mark.parametrize('change', ['checkpoint', 'training_core', 'data', 'code_binding', 'observer'])
def test_revalidation_refuses_changed_bound_artifacts(tmp_path, change):
    m, root, proof, data, binding, report = fixture(tmp_path)
    path = {'checkpoint': root / 'pilot.ckpt', 'training_core': root / m.CORE_FILES[0],
            'data': data, 'code_binding': binding, 'observer': root / 'observer.py'}[change]
    path.write_text('changed')
    assert m.admission_errors(proof, data, binding, root=root)


def test_formal_entry_accepts_only_separately_revalidated_proof(tmp_path, monkeypatch):
    import train
    m, root, proof, data, binding, report = fixture(tmp_path)
    (root / 'evidence').mkdir()
    (root / 'evidence/implementation_manifest.json').write_bytes(binding.read_bytes())
    monkeypatch.setattr(train, 'ROOT', root)
    # Isolate numerical-proof wiring; full150-file preflight has its own real-file test.
    monkeypatch.setattr(train, 'preflight', lambda *a, **k:
        dict(allowed=True, blockers=[], admission_scope='full_main_data', full_main_data_ready=True))
    output = root / 'formal'
    monkeypatch.setattr(train, 'parse_arguments', lambda: type('Args', (),
        dict(manifest=data, output=output, mode='formal', pilot_proof=proof))())
    # Deliberately invalid training manifest stops BEFORE constructing a model.
    with pytest.raises(json.JSONDecodeError):
        train.main()
    assert json.loads((output / 'preflight.json').read_text())['allowed']
