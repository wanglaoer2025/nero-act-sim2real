import importlib.util
from pathlib import Path

import h5py
import pytest

from test_adapter import source, implementation as adapter_implementation


def test_independent_preflight_detects_changed_converted_targets(source, tmp_path, monkeypatch):
    folder = Path(__file__).parents[1]
    path = folder / 'preflight.py'
    assert path.exists(), 'Independent conversion preflight has not been implemented'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('real_data_preflight', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    api = adapter_implementation()
    output = tmp_path / 'adapted.hdf5'
    row = api.convert_episode(source, output)
    assert module.verify_conversion(row)['arrays_exact']
    with h5py.File(output, 'r+') as f:
        f['action'][0, 0] += .001
    with pytest.raises(ValueError, match='target|conversion|array'):
        module.verify_conversion(row)
