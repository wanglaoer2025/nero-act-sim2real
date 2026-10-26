import importlib.util
from pathlib import Path


def test_acceptance_loader_does_not_shadow_historical_experiment_module():
    path = Path(__file__).with_name('run.py')
    spec = importlib.util.spec_from_file_location('_tested_acceptance_entry', path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    assert entry.load_baseline.__module__ != 'experiment'
    assert entry.sealed.runtime.__file__.endswith('supervised_contact_escape.py')
