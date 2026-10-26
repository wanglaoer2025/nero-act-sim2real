import importlib.util
import json
from pathlib import Path

import pytest


def implementation(monkeypatch):
    folder = Path(__file__).parents[1]
    path = folder / 'prepare.py'
    assert path.exists(), 'Data preparation has not been implemented'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('real_data_prepare', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def selection(tmp_path):
    for i in range(2):
        (tmp_path / f'episode_{i}.hdf5').write_bytes(b'path-only selection fixture')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'episodes': [
        {'path': 'episode_0.hdf5', 'status': 'valid'},
        {'path': 'episode_1.hdf5', 'status': 'valid'}]}))
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'train': [0], 'val': [1]}))
    return tmp_path, manifest, split


def test_preserves_existing_episode_split_without_merging_ids(monkeypatch, selection):
    api = implementation(monkeypatch)
    root, manifest, split = selection
    result = api.assign_splits(root, manifest, split)
    assert result == {str(root / 'episode_0.hdf5'): 'train', str(root / 'episode_1.hdf5'): 'val'}


def test_same_episode_cannot_appear_on_both_sides(monkeypatch, selection):
    api = implementation(monkeypatch)
    root, manifest, split = selection
    split.write_text(json.dumps({'train': [0], 'val': [0, 1]}))
    with pytest.raises(ValueError, match='overlap|Duplicate'):
        api.assign_splits(root, manifest, split)


def test_duplicate_source_alias_cannot_count_as_another_episode(monkeypatch, selection):
    api = implementation(monkeypatch)
    root, manifest, split = selection
    (root / 'episode_1.hdf5').unlink()
    (root / 'episode_1.hdf5').symlink_to(root / 'episode_0.hdf5')
    with pytest.raises(ValueError, match='Duplicate|alias'):
        api.assign_splits(root, manifest, split)


def test_missing_file_or_split_is_rejected(monkeypatch, selection):
    api = implementation(monkeypatch)
    root, manifest, split = selection
    (root / 'episode_1.hdf5').unlink()
    with pytest.raises(FileNotFoundError):
        api.assign_splits(root, manifest, split)
    split.unlink()
    with pytest.raises(FileNotFoundError):
        api.assign_splits(root, manifest, split)
