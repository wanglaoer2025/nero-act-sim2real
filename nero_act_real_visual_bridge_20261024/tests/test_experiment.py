import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def implementation(monkeypatch):
    folder = Path(__file__).parents[1]
    path = folder / 'experiment.py'
    assert path.exists(), 'Bounded A/B experiment is not implemented'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('real_bridge_experiment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_event_windows_keep_close_and_release_without_crossing_tail(monkeypatch):
    api = implementation(monkeypatch)
    actions = np.zeros((100, 8), np.float32)
    actions[:, 7] = .095; actions[20:70, 7] = 0
    assert api.event_query_starts(actions) == [0, 16, 19, 20, 22, 40, 68, 70, 72, 80]


def test_same_real_source_cannot_be_train_and_validation(monkeypatch, tmp_path):
    api = implementation(monkeypatch)
    rows = [{'source_path': str(tmp_path/'episode.hdf5'), 'split': split} for split in ('train', 'val')]
    with pytest.raises(ValueError, match='source|split'):
        api.validate_real_partition(rows)


def test_stage_rehearsal_queries_are_disjoint_from_prediction_checks(monkeypatch):
    api = implementation(monkeypatch)
    stages = np.array(['approach']*12 + ['close']*12)
    train, check = api.stage_partition(stages)
    assert not set(train) & set(check)
    assert len(train) == 6 and len(check) == 6
    assert all(stages[slot] in ('approach', 'close') for slot in train + check)


def test_stage_sampling_excludes_invalid_visual_queries(monkeypatch):
    api = implementation(monkeypatch)
    stages = np.array(['approach']*12 + ['close']*12)
    valid = np.ones(24, bool); valid[:6] = False
    train, check = api.stage_partition(stages, valid=valid)
    assert train == [6, 7, 8, 12, 14, 16]
    assert check == [9, 10, 11, 19, 21, 23]


@pytest.mark.parametrize('loss,accuracy,drift,expected', [(.90,.8,.0005,True),(.99,.8,.0005,False),(.90,.79,.0005,False),(.90,.8,.002,False)])
def test_gate_requires_gain_classification_and_all20_preservation(monkeypatch, loss, accuracy, drift, expected):
    api = implementation(monkeypatch)
    a = dict(loss=1., gripper_intent_accuracy=.8)
    b = dict(loss=loss, gripper_intent_accuracy=accuracy)
    result = api.acceptance(a,b,dict(maximum_arm_change_rad=drift, maximum_gripper_change_m=.0001))
    assert result['mechanism_gate_passed'] == expected
    assert result['policy_adopted'] is False


def test_full_increment_wrapper_checkpoint_is_loaded_without_dropping_head(monkeypatch):
    api = implementation(monkeypatch)
    policy, _ = api.load_baseline()
    saved = torch.load(api.CHECKPOINT, map_location='cpu', weights_only=False)['model']
    assert hasattr(policy, 'base') and hasattr(policy, 'head')
    assert set(policy.state_dict()) == set(saved)
    assert all(torch.equal(value, saved[name]) for name, value in policy.state_dict().items())
    assert not any(p.requires_grad for p in policy.parameters())
