import importlib.util
import importlib
import json
import pytest


def test_source_seal_refuses_unverified_suite_before_any_artifact(tmp_path,monkeypatch):
    assert importlib.util.find_spec('training.history_seal'), 'Pre-optimization source seal entry missing'
    m=importlib.import_module('training.history_seal');monkeypatch.setattr(m,'ROOT',tmp_path)
    evidence=tmp_path/'evidence';evidence.mkdir()
    verification=evidence/'tests.json';verification.write_text(json.dumps(dict(complete=False,failed=1)))
    with pytest.raises(ValueError,match='suite|test'):
        m.seal_history_experiment(tmp_path/'data',tmp_path/'initializer',verification)
    assert not (evidence/'pre_optimization_freeze.json').exists()
    assert not (tmp_path/'evaluation/history_protocol.json').exists()
