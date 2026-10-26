import pytest
import torch

from arm_projection import project_rows


def test_projection_contracts_only_the_trained_update_until_both_old_limits_hold():
    original = torch.zeros(7, 4)
    trained = original.clone()
    trained[0, 0] = 1.
    def assess(rows):
        errors = rows*.004
        return dict(samples=423, maximum_arm_prefix_change_rad=float(errors.abs().max()),
                    rms_arm_prefix_change_rad=float(errors.square().mean().sqrt()))
    chosen, report = project_rows(original, trained, assess)
    assert report['factor'] == .0625
    assert chosen[0, 0] == .0625
    assert torch.count_nonzero(chosen) == 1
    assert torch.count_nonzero(original) == 0
    assert report['assessment']['maximum_arm_prefix_change_rad'] <= .0003
    assert report['assessment']['rms_arm_prefix_change_rad'] <= .0001


def test_projection_rejects_an_already_invalid_reference():
    with pytest.raises(ValueError):
        project_rows(torch.zeros(7, 4), torch.ones(7, 4),
                     lambda _: dict(samples=423, maximum_arm_prefix_change_rad=.001,
                                    rms_arm_prefix_change_rad=.001))
