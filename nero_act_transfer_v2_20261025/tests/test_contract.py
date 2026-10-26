import numpy as np
import pytest


def test_previous_command_is_strictly_before_exposure_and_acknowledged():
    from contract import CommandHistory
    h = CommandHistory()
    a = np.r_[np.zeros(7), .095]
    b = np.r_[np.zeros(7), .09]
    h.submit(a, 1.)
    h.submit(b, 2., ack=False)
    np.testing.assert_array_equal(h.before(2.5), a)
    h.submit(b, 3.)
    np.testing.assert_array_equal(h.before(3.), a)
    np.testing.assert_array_equal(h.before(3.1), b)
    with pytest.raises(ValueError): h.before(.5)
    with pytest.raises(ValueError): h.submit(a, 2.9)


def test_observation_has_one_causal_context_and_six_metric_features():
    from contract import make_observation
    stats = dict(qpos_mean=np.zeros(8), qpos_std=np.ones(8),
                 geometry_mean=np.zeros(6), geometry_std=np.full(6, .1))
    state = np.r_[np.zeros(7), .08]
    obs = make_observation(state, np.arange(6)*.01, .081, stats)
    assert obs.shape == (15,) and obs.dtype == np.float32
    np.testing.assert_allclose(obs[8], .1, atol=1e-6)
    np.testing.assert_allclose(obs[9:], np.arange(6)*.1, atol=1e-6)
    with pytest.raises(ValueError): make_observation(state, np.zeros(6), .1, stats)


def test_measured_state_cannot_contain_a_gripper_outside_physical_range():
    from contract import make_observation
    stats = dict(qpos_mean=np.zeros(8), qpos_std=np.ones(8),
                 geometry_mean=np.zeros(6), geometry_std=np.ones(6))
    with pytest.raises(ValueError):
        make_observation(np.r_[np.zeros(7), .10006], np.zeros(6), .095, stats)


def test_measured_soft_limit_overshoot_is_preserved_but_not_a_command():
    from contract import make_observation,CommandHistory
    stats=dict(qpos_mean=np.zeros(8),qpos_std=np.ones(8),geometry_mean=np.zeros(6),geometry_std=np.ones(6))
    obs=make_observation(np.r_[np.zeros(7),.09518],np.zeros(6),.095,stats)
    np.testing.assert_allclose(obs[7],.09518)
    np.testing.assert_allclose(obs[8],-.018,atol=1e-6)
    with pytest.raises(ValueError):CommandHistory().submit(np.r_[np.zeros(7),.09518],1.)
