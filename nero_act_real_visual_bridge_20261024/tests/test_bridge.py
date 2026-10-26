import importlib.util
from pathlib import Path

import pytest
import torch


def implementation():
    path = Path(__file__).parents[1] / 'bridge.py'
    assert path.exists(), 'Visual bridge is not implemented'
    spec = importlib.util.spec_from_file_location('real_visual_bridge', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('wrapped', [False, True])
def test_zero_adapter_is_exact_identity_and_hook_is_removable(wrapped):
    api = implementation()
    layer = torch.nn.Conv2d(5, 128, 1)
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.model = torch.nn.Module(); self.model.input_proj = layer
    policy = Policy()
    if wrapped:
        wrapper = torch.nn.Module(); wrapper.base = policy; policy = wrapper
    adapter = api.VisualResidual()
    x = torch.randn(2, 5, 3, 4)
    before = layer(x).detach()
    state_keys = set(policy.state_dict())
    handle = api.attach_visual_adapter(policy, adapter)
    assert torch.equal(layer(x), before)
    assert set(policy.state_dict()) == state_keys
    with torch.no_grad(): adapter.up.bias.fill_(.2)
    assert not torch.equal(layer(x), before)
    handle.remove()
    assert torch.equal(layer(x), before)


def test_auxiliary_head_accepts_only_real8_and_predicts20x8():
    api = implementation()
    head = api.AuxiliaryHead()
    features, state = torch.randn(3, 128, 4, 6), torch.randn(3, 8)
    assert tuple(head(features, state).shape) == (3, 20, 8)
    with pytest.raises(ValueError, match='8'):
        head(features, torch.randn(3, 15))


def test_padding_is_excluded_and_physical_delta_target_is_correct():
    api = implementation()
    state = torch.zeros(1, 8); state[:, 0] = .1; state[:, 7] = .08
    target = state[:, None].repeat(1, 20, 1); target[:, :, 0] += .03
    target[:, :, 7] = 0
    prediction = torch.zeros(1, 20, 8)
    prediction[:, :, 0] = 1; prediction[:, :, 7] = -.08/.095
    mask = torch.zeros(1, 20, dtype=torch.bool); mask[:, 2:] = True
    target[:, 2:] = 999
    loss = api.auxiliary_loss(prediction, state, target, mask)
    assert loss.item() == pytest.approx(0, abs=1e-12)
    mask[:] = True
    with pytest.raises(ValueError, match='valid|padding'):
        api.auxiliary_loss(prediction, state, target, mask)


def test_new_modules_can_learn_without_changing_frozen_projection():
    api = implementation()
    frozen = torch.nn.Conv2d(3, 128, 1).requires_grad_(False)
    before = {n: p.clone() for n, p in frozen.state_dict().items()}
    adapter, head = api.VisualResidual(), api.AuxiliaryHead()
    optimizer = torch.optim.AdamW([*adapter.parameters(), *head.parameters()], lr=.01)
    features = frozen(torch.randn(2, 3, 3, 4)).detach()
    old = adapter(features).detach().clone()
    output = head(adapter(features), torch.randn(2, 8))
    output.square().mean().backward(); optimizer.step()
    assert not torch.equal(adapter(features), old)
    assert all(torch.equal(value, before[name]) for name, value in frozen.state_dict().items())


def test_real_state_denormalization_uses_paired8_statistics():
    api = implementation()
    stats = dict(qpos_mean=[.2]*8, qpos_std=[.1]*8)
    restored = api.physical_state(torch.ones(2, 8), stats)
    assert torch.allclose(restored, torch.full((2, 8), .3))
    with pytest.raises(ValueError):
        api.physical_state(torch.ones(2, 15), stats)
