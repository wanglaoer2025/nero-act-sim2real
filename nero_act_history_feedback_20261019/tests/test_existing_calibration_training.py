import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def scene():
    return json.loads((ROOT.parent/'nero_act_replay_context_retest_20261019/model/current_run27/scene_config.json').read_text())


def test_existing_calibration_is_offline_assumption_not_verified():
    from training.legacy_visual_data import training_profile
    p = training_profile(scene())
    assert p['verified'] is False and p['hardware_deployment_allowed'] is False
    assert p['tcp_link7_xyz_m'] == [.152, 0, -.0235]
    changed = scene(); changed['tcp_link7_xyz_m'][0] += .001
    assert training_profile(changed)['sha256'] != p['sha256']


@pytest.mark.parametrize('xyz', [[-.54, -.02, 0.], [-.56, .02, .2], [-.52, -.16, .1]])
def test_image_estimator_tracks_lift_and_xy_without_truth_input(xyz):
    from training.legacy_visual_data import NominalImageEstimator, training_profile
    observer = NominalImageEstimator(training_profile(scene()), (180, 320))
    rgb = np.full((180, 320, 3), 190, np.uint8)
    hull = observer.cup_hull(np.array(xyz))
    cv2.fillConvexPoly(rgb, np.rint(hull).astype(np.int32), (220, 180, 15))
    target = observer.project(np.array([[-.52, -.16, 0.]]))[0]
    cv2.rectangle(rgb, tuple(np.rint(target-[4, 3]).astype(int)), tuple(np.rint(target+[4, 3]).astype(int)), (15, 65, 160), -1)
    # Only RGB goes in; no initial position, phase, measured state or simulator truth.
    cup, card, diagnostics = observer.measure(rgb)
    np.testing.assert_allclose(cup, xyz, atol=.013)
    np.testing.assert_allclose(card, [-.52, -.16, 0.], atol=.01)
    assert diagnostics['runtime_qualified'] is False
    with pytest.raises(ValueError, match='visible|unique'):
        observer.measure(np.zeros_like(rgb))


def test_incomplete_formal_training_cannot_finalize(tmp_path):
    from training.visual import finalize_training
    with pytest.raises(ValueError, match='complete|epoch'):
        finalize_training(tmp_path, dict(num_epochs=20), [dict(epoch=1)], {})


def test_new_statistics_and_normalization_are_bound():
    from training.visual import initialize_visual
    import torch
    from tests.test_visual_policy import config, statistics
    from executor.model import BoundedActPolicy
    c = config(); c.pop('visual_observation_profile'); c['policy_input_dim'] = 9
    torch.set_num_threads(1)
    old = BoundedActPolicy(c)
    stats = statistics(config())
    stats['geometry_mean'] = np.arange(6, dtype=np.float32)
    cfg, weights = initialize_visual(dict(config=c, model=old.state_dict()), stats, dict(image_size=[90, 160]))
    assert cfg['policy_input_dim'] == 15 and cfg['geometry_mean'] == list(range(6))
    assert weights['model.input_proj_robot_state.weight'].shape[1] == 15
    assert torch.count_nonzero(weights['model.input_proj_robot_state.weight'][:, 9:]) == 0
    assert cfg['hardware_deployment_allowed'] is False


def test_sampling_uses_only_fresh_valid_observations(tmp_path):
    from training.visual import visual_windows
    from tests.test_visual_dataset import episode
    import h5py
    p = episode(tmp_path/'episode_0.hdf5')
    with h5py.File(p, 'r+') as f:
        f['simulation/stage'] = np.array(['descend']*25, dtype='S8')
        f['observations/visual_valid'][:] = False
        f['observations/visual_valid'][10] = True
    assert visual_windows([p]) == [(0, 10)]


def test_formal_config_rejects_old_heads_only_or_wrong_input():
    from training.visual import validate_formal_config
    c = dict(seed=42, batch_size=32, num_epochs=20, image_size=[180, 320])
    validate_formal_config(c)
    for change in [dict(num_epochs=0), dict(batch_size=0), dict(train_arm_output_and_gripper_only=True)]:
        with pytest.raises(ValueError): validate_formal_config(dict(c, **change))


def test_first_training_step_records_real_optimizer_and_status(tmp_path):
    import torch
    from training.visual import build_visual_policy, run_epoch
    from tests.test_visual_policy import config
    c = config(); c.update(nonlinear_gripper_head=True, prior_gripper_loss_weight=.005, num_epochs=20)
    policy = build_visual_policy(c)
    (tmp_path/'artifacts').mkdir()
    batch = (torch.rand(2, 1, 3, 90, 160), torch.rand(2, 15),
             torch.rand(2, 20, 8), torch.zeros(2, 20, dtype=torch.bool), torch.zeros(2))
    before = policy.model.input_proj_robot_state.weight.detach().clone()
    metrics = run_epoch(policy, [batch], torch.device('cpu'), training=True, output=tmp_path, epoch=1)
    assert np.isfinite(metrics['loss'])
    assert not torch.equal(before, policy.model.input_proj_robot_state.weight)
    assert json.loads((tmp_path/'status.json').read_text())['total_epochs'] == 20


def test_card_center_not_shifted_by_cup_occluding_middle():
    from training.legacy_visual_data import NominalImageEstimator, training_profile
    observer = NominalImageEstimator(training_profile(scene()), (180, 320))
    target = np.array([-.52, -.16, 0.])
    rgb = np.full((180, 320, 3), 180, np.uint8)
    corners = target+np.array([[-.05, -.05, 0], [-.05, .05, 0], [.05, .05, 0], [.05, -.05, 0]])
    poly = np.rint(observer.project(corners)).astype(np.int32)
    cv2.fillConvexPoly(rgb, poly, (15, 65, 160))
    hull = observer.cup_hull(target+[-.01, 0, .002])
    cv2.fillConvexPoly(rgb, np.rint(hull).astype(np.int32), (220, 180, 15))
    _, card, _ = observer.measure(rgb)
    np.testing.assert_allclose(card, target, atol=.004)


def test_changed_training_input_never_writes_successful_summary(tmp_path):
    from training.visual import complete_training
    from training.integrity import snapshot
    (tmp_path/'artifacts').mkdir()
    source = tmp_path/'protected.py'; source.write_text('before')
    hashes = snapshot([source]); source.write_text('changed_during_training')
    with pytest.raises(RuntimeError, match='changed'):
        complete_training(tmp_path, dict(num_epochs=1), [dict(epoch=1)], hashes, dict(complete=True))
    assert not (tmp_path/'artifacts/training_summary.json').exists()
    assert not (tmp_path/'status.json').exists()
