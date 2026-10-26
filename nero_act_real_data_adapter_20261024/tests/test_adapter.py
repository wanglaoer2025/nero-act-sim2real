import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


def implementation():
    path = Path(__file__).parents[1] / 'adapter.py'
    assert path.exists(), 'The offline adapter has not been implemented'
    spec = importlib.util.spec_from_file_location('real_data_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'source' / 'episode_0.hdf5'
    path.parent.mkdir()
    q = np.arange(32, dtype=np.float32)[:, None] * .01 + np.arange(7, dtype=np.float32)[None] * .001
    intent = np.full(32, .095, np.float32)
    intent[4:20] = 0
    action = np.c_[q[np.minimum(np.arange(32) + 2, 31)], intent]
    with h5py.File(path, 'w') as f:
        f.attrs.update(sim=False, demonstration_mode='yolo_teacher_policy',
                       action_semantics='future_measured_qpos_plus_gripper_target',
                       action_lead_steps=2, image_format='RGB')
        f['action'] = action
        f['timestamp'] = 100 + np.arange(32) * .05
        f['observations/qpos'] = q
        f['observations/gripper_state'] = np.full((32, 1), .08, np.float32)
        images = np.broadcast_to(np.arange(32, dtype=np.uint8)[:, None, None, None], (32, 4, 6, 3))
        f['observations/images/camera_0'] = images
        f['observations/images/camera_1'] = images + 1
    return path


def unit_stats():
    return {name: np.full(8, value, np.float32) for name, value in
            [('qpos_mean', 0), ('qpos_std', 1), ('action_mean', 0), ('action_std', 1)]}


def test_conversion_uses_raw_feedback_once_and_preserves_gripper_event(source, tmp_path):
    api = implementation()
    output = tmp_path / 'adapted.hdf5'
    row = api.convert_episode(source, output, lead_steps=2, split='train')
    with h5py.File(output, 'r') as f, h5py.File(source, 'r') as original:
        assert len(f['action']) == 30
        np.testing.assert_array_equal(f['action'][:, :7], original['observations/qpos'][2:])
        assert f['action'][4, 7] == 0 and f['action'][3, 7] > .09
        assert f['joint_target_feedback_timestamp'][0] == pytest.approx(100.1)
        assert f['gripper_intent_timestamp'][4] == pytest.approx(100.2)
        assert f['observations/images/camera_0'].shape == (30, 4, 6, 3)
        assert f['observations/images/camera_0'].is_virtual
        np.testing.assert_array_equal(f['observations/images/camera_1'][5], original['observations/images/camera_1'][5])
        assert f.attrs['action_semantics'] != 'submitted_absolute_target'
    assert row['full_chunk_queries'] == 11


def test_tail_is_masked_without_repeating_the_final_target(source, tmp_path):
    api = implementation()
    output = tmp_path / 'adapted.hdf5'
    api.convert_episode(source, output)
    image, state, actions, padding, start = api.read_window(output, 28, unit_stats(), image_size=(4, 6))
    assert tuple(image.shape) == (1, 3, 4, 6) and tuple(state.shape) == (8,)
    assert tuple(actions.shape) == (20, 8) and start == 28
    assert padding.tolist() == [False, False] + [True] * 18
    np.testing.assert_allclose(actions[0, 0].item(), .30, atol=1e-7)
    assert not actions[2:].any()


def test_missing_vds_source_is_rejected_instead_of_becoming_zero_images(source, tmp_path):
    api = implementation()
    output = tmp_path / 'adapted.hdf5'
    api.convert_episode(source, output)
    source.unlink()
    with pytest.raises((FileNotFoundError, ValueError), match='source|Source'):
        api.read_window(output, 0, unit_stats())


@pytest.mark.parametrize('corruption', ['nan', 'time', 'label', 'length'])
def test_invalid_sources_are_rejected_before_any_output(source, tmp_path, corruption):
    api = implementation()
    with h5py.File(source, 'r+') as f:
        if corruption == 'nan':
            f['observations/qpos'][0, 0] = np.nan
        elif corruption == 'time':
            f['timestamp'][1] = f['timestamp'][0]
        elif corruption == 'label':
            f['action'][0, 0] = 123
        else:
            del f['observations/gripper_state']
            f['observations/gripper_state'] = np.zeros((31, 1), np.float32)
    output = tmp_path / 'adapted.hdf5'
    with pytest.raises(ValueError):
        api.convert_episode(source, output)
    assert not output.exists()


def test_existing_outputs_and_source_changes_are_rejected(source, tmp_path):
    api = implementation()
    output = tmp_path / 'adapted.hdf5'
    api.convert_episode(source, output)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        api.convert_episode(source, output)
    assert output.read_bytes() == before
    with h5py.File(source, 'r+') as f:
        f['observations/qpos'][0, 0] = .4
    with pytest.raises(ValueError, match='source|Source'):
        api.read_window(output, 0, unit_stats())


def test_normalization_uses_only_explicit_train_files(source, tmp_path):
    api = implementation()
    train = tmp_path / 'train.hdf5'
    val = tmp_path / 'val.hdf5'
    api.convert_episode(source, train, split='train')
    api.convert_episode(source, val, split='val')
    stats = api.fit_statistics([train])
    assert stats['qpos_mean'][0] == pytest.approx(.145, abs=1e-6)
    assert stats['action_mean'][0] == pytest.approx(.165, abs=1e-6)
    with pytest.raises(ValueError, match='train'):
        api.fit_statistics([val])


def test_legacy_view_does_not_pass_as_current_visual15(source, tmp_path):
    api = implementation()
    output = tmp_path / 'adapted.hdf5'
    api.convert_episode(source, output)
    with pytest.raises(ValueError, match='15|visual|submitted'):
        api.FutureStateDataset([output], unit_stats(), policy_input_dim=15)
