"""Behavior tests for causal learned identity potentials (not paper metrics)."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from transvision.models.event_track_v2x.learned_identity import (
    CausalIdentityInput, RecoverableIdentityModel, detached_log_potentials, identity_training_loss,
)


def sample(n=2, m=2, h=3):
    return CausalIdentityInput(torch.randn(n, 203), torch.randn(m, 203), torch.randn(h, 203),
        torch.full((n,), 900_000), torch.full((m,), 900_000), torch.full((h,), 800_000),
        torch.full((n,), 950_000), torch.full((m,), 950_000), torch.full((h,), 850_000),
        torch.arange(h) % 2, 1_000_000)


@pytest.fixture
def model():
    torch.manual_seed(3)
    return RecoverableIdentityModel(hidden=16, heads=4, dropout=0).eval()


@pytest.mark.parametrize('n,m,h', [(0, 0, 0), (0, 3, 2), (3, 0, 0), (2, 2, 0), (2, 2, 3)])
def test_finite_shapes_and_empty(model, n, m, h):
    output = model(sample(n, m, h))
    assert output.pair.shape == (n, m)
    assert output.temporal.shape == (n + m, h)
    factors = detached_log_potentials(output)
    assert all(np.isfinite(x).all() for x in factors)
    loss = identity_training_loss(output, torch.zeros(n, m), torch.zeros(n + m, h))
    assert torch.isfinite(loss['total'])
    loss['total'].backward()


def test_history_really_changes_current_association(model):
    x = sample()
    first = model(x)
    changed = model(replace(x, history=x.history.flip(1)))
    assert not torch.allclose(first.pair, changed.pair)
    assert not torch.allclose(first.context_left, changed.context_left)


def test_permutation_equivariance_and_history_order(model):
    x = sample()
    y = model(x)
    reverse = torch.tensor([1, 0])
    swapped = replace(x, left=x.left[reverse], left_information_us=x.left_information_us[reverse],
                      left_arrival_us=x.left_arrival_us[reverse])
    assert torch.allclose(model(swapped).pair, y.pair[reverse], atol=1e-6)
    order = torch.tensor([2, 0, 1])
    reordered = replace(x, history=x.history[order], history_information_us=x.history_information_us[order],
        history_arrival_us=x.history_arrival_us[order], history_source=x.history_source[order])
    assert torch.allclose(model(reordered).pair, y.pair, atol=1e-6)


@pytest.mark.parametrize('field', ['left', 'right', 'history'])
def test_future_source_or_arrival_rejected(model, field):
    x = sample()
    for suffix in ('_information_us', '_arrival_us'):
        original = getattr(x, field + suffix)
        with pytest.raises(ValueError):
            model(replace(x, **{field + suffix: torch.full_like(original, 1_000_001)}))


def test_nonfinite_and_invalid_metadata_rejected(model):
    x = sample()
    with pytest.raises(ValueError):
        model(replace(x, left=x.left * float('nan')))
    with pytest.raises(ValueError):
        model(replace(x, history_source=torch.full((3,), 2)))
    with pytest.raises(ValueError):
        model(replace(x, decision_us=True))
    with pytest.raises(ValueError):
        model(replace(x, left_arrival_us=x.left_arrival_us.float()))


def test_trainable_temporal_and_cross_paths(model):
    model.train()
    x = sample()
    output = model(x)
    targets = torch.eye(2)
    temporal = torch.zeros(4, 3)
    temporal[0, 0] = temporal[1, 1] = temporal[2, 0] = 1
    loss = identity_training_loss(output, targets, temporal)
    loss['total'].backward()
    for name in ('encoder.0.weight', 'temporal_attention.in_proj_weight', 'temporal_scorer.0.weight',
                 'pair_scorer.0.weight', 'source_embedding.weight', 'time_embedding.0.weight'):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name


def test_training_targets_are_separate_and_exclusive(model):
    x = sample()
    output = model(x)
    with pytest.raises(ValueError):
        identity_training_loss(output, torch.ones(2, 2), torch.zeros(4, 3))
    with pytest.raises(ValueError):
        identity_training_loss(output, torch.eye(2), torch.ones(4, 3))
    # Inference does not accept labels/GT as extra fields or keyword arguments.
    with pytest.raises(TypeError):
        model(x, gt_identity=torch.ones(2))


def test_deterministic_eval_and_no_input_mutation(model):
    x = sample()
    before = x.history.clone()
    first, second = model(x), model(x)
    assert torch.equal(first.pair, second.pair)
    assert torch.equal(x.history, before)


def test_boolean_targets_supported(model):
    output = model(sample())
    targets = torch.eye(2, dtype=torch.bool)
    temporal = torch.zeros(4, 3, dtype=torch.bool)
    temporal[0, 0] = True
    boolean = identity_training_loss(output, targets, temporal)
    floating = identity_training_loss(output, targets.float(), temporal.float())
    assert torch.equal(boolean['total'], floating['total'])


def test_finite_inputs_cannot_silently_emit_nonfinite_output(model):
    x = sample()
    with pytest.raises(ValueError, match='nonfinite'):
        model(replace(x, left=torch.full_like(x.left, torch.finfo(x.left.dtype).max)))
