"""Protect valid-prefix supervision and the already learned gripper/backbone."""
import pytest
import torch
from torch import nn

from arm_tuning import configure_arm_tuning, masked_arm_loss


def test_padded_future_points_do_not_enter_arm_loss():
    prediction = torch.zeros(1, 2, 7, requires_grad=True)
    target = torch.full((1, 2, 7), .0002)
    target[:, 1] = 1e6
    loss = masked_arm_loss(prediction, target, torch.tensor([[True, False]]))
    assert float(loss) == pytest.approx(4.)
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:, 1]) == 0


def test_empty_arm_supervision_is_rejected():
    with pytest.raises(ValueError):
        masked_arm_loss(torch.zeros(1, 2, 7), torch.zeros(1, 2, 7), torch.zeros(1, 2, dtype=torch.bool))


def test_only_seven_arm_rows_can_change_during_optimization():
    policy = nn.Module()
    policy.base = nn.Module()
    policy.base.model = nn.Module()
    policy.base.model.action_head = nn.Linear(3, 8)
    policy.base.encoder = nn.Linear(3, 3)
    policy.head = nn.Linear(3, 1)
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    parameters = configure_arm_tuning(policy)
    optimizer = torch.optim.AdamW(parameters, lr=.01, weight_decay=0.)
    for _ in range(3):
        optimizer.zero_grad()
        # Even an accidental8-axis objective cannot train the gripper row.
        policy.base.model.action_head(torch.ones(2, 3)).square().mean().backward()
        optimizer.step()
    after = policy.state_dict()
    for key in before:
        if 'action_head' not in key:
            assert torch.equal(before[key], after[key])
        else:
            assert torch.equal(before[key][7], after[key][7])
    assert not torch.equal(before['base.model.action_head.weight'][:7], after['base.model.action_head.weight'][:7])
