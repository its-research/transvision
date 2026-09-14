from dataclasses import replace

import numpy as np
import pytest
import torch

from transvision.models.event_track_v2x.learned_identity import CausalIdentityInput, RecoverableIdentityModel, detached_log_potentials
from transvision.models.event_track_v2x.recoverable_identity import LearnedHypothesisSession
from transvision.models.event_track_v2x.hypothesis_bank import LogAssociationFactors, assignment_log_weight


def inputs():
    torch.manual_seed(2)
    return CausalIdentityInput(torch.randn(2, 203), torch.randn(2, 203), torch.randn(3, 203),
        torch.tensor([800_000, 800_000]), torch.tensor([800_000, 800_000]), torch.tensor([700_000]*3),
        torch.tensor([900_000]*2), torch.tensor([900_000]*2), torch.tensor([750_000]*3),
        torch.tensor([0, 1, 0]), 1_000_000)


def make():
    x = inputs()
    model = RecoverableIdentityModel(hidden=16, heads=4, dropout=0).eval()
    session = LearnedHypothesisSession(model, x, left_node_ids=('v0', 'v1'), right_node_ids=('r0', 'r1'))
    return x, model, session


def update(session, x, evidence='frame2', budget=10):
    return session.rescore(evidence, x, left_node_ids=('v0', 'v1'), right_node_ids=('r0', 'r1'), expansion_budget=budget)


def test_learned_rescore_matches_new_absolute_distribution_not_product():
    x, model, session = make()
    first = session.start(expansion_budget=10)
    modified = replace(x, history=x.history.flip(1), decision_us=1_100_000)
    second = update(session, modified)
    expected = LogAssociationFactors(*detached_log_potentials(model(modified)))
    for branch in second.bank.active:
        assert branch.log_weight == pytest.approx(assignment_log_weight(expected, branch.choices), abs=1e-10)
    assert first.bank.commit == second.bank.previous_commit
    assert session.start(expansion_budget=999) == first
    assert update(session, modified) == second  # immutable duplicate, not a second update
    assert not second.calibrated_real_posterior and not second.complete_3d_tracker


def test_future_conflicting_duplicate_and_node_reorder_fail_before_commit():
    x, _, session = make()
    session.start(expansion_budget=10)
    y = replace(x, decision_us=1_100_000)
    snapshot = update(session, y)
    with pytest.raises(ValueError):
        update(session, replace(y, history=y.history+1))
    with pytest.raises(ValueError):
        update(session, replace(y, history_arrival_us=torch.tensor([2_000_000]*3)), 'future')
    with pytest.raises(ValueError):
        session.rescore('reorder', y, left_node_ids=('v1', 'v0'), right_node_ids=('r0', 'r1'), expansion_budget=1)
    assert update(session, y) == snapshot


def test_explicit_window_token_and_frozen_model_guards():
    x, model, session = make()
    session.start(expansion_budget=2)
    with pytest.raises(ValueError):
        update(session, replace(x, decision_us=4_000_000))
    with pytest.raises(ValueError):
        LearnedHypothesisSession(model, x, left_node_ids=('v0', 'v1'), right_node_ids=('r0', 'r1'), max_history_tokens=1)
    with torch.no_grad():
        next(model.parameters()).add_(0.1)
    with pytest.raises(ValueError):
        update(session, replace(x, decision_us=1_100_000))


def test_two_fresh_sessions_replay_identically():
    x, model, first = make()
    second = LearnedHypothesisSession(model, x, left_node_ids=('v0', 'v1'), right_node_ids=('r0', 'r1'))
    assert first.audit(first.start(expansion_budget=2)) == second.audit(second.start(expansion_budget=2))
    y = replace(x, decision_us=1_100_000)
    assert first.audit(update(first, y, budget=4)) == second.audit(update(second, y, budget=4))


@pytest.mark.parametrize('child', ['temporal_attention', 'encoder', 'pair_scorer'])
def test_child_train_mode_rejected_before_mutation(child):
    x, model, session = make()
    first = session.start(expansion_budget=2)
    getattr(model, child).train()
    assert not model.training
    with pytest.raises(ValueError, match='model changed'):
        update(session, replace(x, decision_us=1_100_000))
    assert session.bank.commits == (first.bank,)
    with pytest.raises(ValueError, match='eval-mode'):
        LearnedHypothesisSession(model, x, left_node_ids=('v0', 'v1'), right_node_ids=('r0', 'r1'))


@pytest.mark.parametrize('limit', [{'max_current_tokens': 3}, {'max_cross_pairs': 3}, {'max_temporal_pairs': 11}])
def test_neural_budget_rejected_before_forward(limit, monkeypatch):
    x, model, _ = make()
    def forbidden(*args, **kwargs):
        pytest.fail('network must not run after exceeding neural budget')
    monkeypatch.setattr(model, 'forward', forbidden)
    with pytest.raises(ValueError, match='pair/token budget'):
        LearnedHypothesisSession(model, x, left_node_ids=('v0', 'v1'), right_node_ids=('r0', 'r1'), **limit)
