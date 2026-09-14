"""Independent full-assignment sums, sparse-frontier and BP limitations."""
from dataclasses import replace
import itertools
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.hypothesis_bank import LogAssociationFactors
from transvision.models.event_track_v2x.jpda_marginals import (
    JPDALimits, exact_jpda, lbp_jpda, _excluding_lse,
)


def oracle(factors):
    n, m = factors.shape
    events, logs = [], []
    for event in itertools.product(range(-1, m), repeat=n):
        used = [j for j in event if j >= 0]
        if len(set(used)) != len(used) or any(j >= 0 and not factors.allowed[i][j] for i, j in enumerate(event)):
            continue
        weight = sum(factors.log_left_unmatched[i] if j < 0 else factors.log_pair[i][j] for i, j in enumerate(event))
        weight += sum(factors.log_right_unmatched[j] for j in range(m) if j not in used)
        logs.append(weight)
        events.append(event)
    maximum = max(logs)
    weights = np.exp(np.asarray(logs) - maximum)
    total = math.fsum(weights)
    weights /= total
    pair, left, right = np.zeros((n, m)), np.zeros(n), np.zeros(m)
    for event, weight in zip(events, weights):
        for i, j in enumerate(event):
            if j < 0:
                left[i] += weight
            else:
                pair[i, j] += weight
        for j in range(m):
            if j not in event:
                right[j] += weight
    return pair, left, right, maximum + math.log(total)


@pytest.mark.parametrize('shape', itertools.product(range(5), repeat=2))
def test_exact_matches_independent_exhaustive_joint_posterior(shape):
    n, m = shape
    rng = np.random.default_rng(n * 11 + m)
    for _ in range(5):
        factors = LogAssociationFactors(rng.normal(0, 3, (n, m)), rng.normal(0, 2, n),
                                       rng.normal(0, 2, m), rng.random((n, m)) > .25)
        expected = oracle(factors)
        actual = exact_jpda(factors)
        assert actual.exact_model_marginals
        assert actual.factor_sha256 == factors.digest()
        np.testing.assert_allclose(np.asarray(actual.pair).reshape(n, m), expected[0], rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(actual.left_unmatched, expected[1], atol=1e-12)
        np.testing.assert_allclose(actual.right_unmatched, expected[2], atol=1e-12)
        assert actual.log_partition == pytest.approx(expected[3], abs=1e-11)


@pytest.mark.parametrize('solver', [exact_jpda, lbp_jpda])
def test_unmatched_factors_and_gauge_invariance(solver):
    original = LogAssociationFactors.from_positive([[8.]], [2.], [2.])
    changed = LogAssociationFactors([[math.log(8.) + 700]], [math.log(2.) + 200], [math.log(2.) + 500])
    a, b = solver(original), solver(changed)
    assert a.pair[0][0] == pytest.approx(2 / 3)
    np.testing.assert_allclose(a.pair, b.pair, atol=1e-12)
    assert a.left_unmatched[0] == pytest.approx(1 / 3)
    assert a.right_unmatched[0] == pytest.approx(1 / 3)
    if solver is exact_jpda:
        assert b.log_partition - a.log_partition == pytest.approx(700)


@pytest.mark.parametrize('solver', [exact_jpda, lbp_jpda])
def test_tiny_unmatched_mass_is_not_subtracted_from_one(solver):
    factors = LogAssociationFactors([[500.]], [0.], [0.])
    result = solver(factors)
    assert result.left_unmatched[0] > 0
    assert result.right_unmatched[0] > 0
    assert math.log(result.left_unmatched[0]) == pytest.approx(-500)
    assert math.log(result.right_unmatched[0]) == pytest.approx(-500)


def test_sparse_connected_chain_does_not_enumerate_full_assignments():
    n = 400
    allowed = np.eye(n, dtype=bool) | np.eye(n, k=1, dtype=bool)
    factors = LogAssociationFactors(np.zeros((n, n)), np.zeros(n), np.zeros(n), allowed)
    result = exact_jpda(factors, limits=JPDALimits(max_dp_states=1000, max_transitions=6000))
    assert result.components == 1
    assert result.peak_layer_states <= 2
    assert result.stored_dp_states < 1000
    assert result.transitions < 6000
    assert result.marginal_consistency_error < 1e-9


@pytest.mark.parametrize('field,value', [('max_dp_states', 1), ('max_transitions', 1), ('max_factor_cells', 2)])
def test_exact_resource_exhaustion_never_silently_truncates(field, value):
    factors = LogAssociationFactors(np.zeros((4, 4)), np.zeros(4), np.zeros(4))
    before = factors.digest()
    with pytest.raises(ValueError, match='budget|capacity'):
        exact_jpda(factors, limits=replace(JPDALimits(), **{field: value}))
    assert factors.digest() == before
    assert exact_jpda(factors).exact_model_marginals


@pytest.mark.parametrize('solver', [exact_jpda, lbp_jpda])
def test_isolates_and_forbidden_huge_scores_have_no_mass(solver):
    f = LogAssociationFactors([[900., 0.], [0., 900.]], [0., 0.], [0., 0.], [[False, False], [False, False]])
    result = solver(f)
    assert result.components == 4
    assert result.pair == ((0., 0.), (0., 0.))
    assert result.left_unmatched == result.right_unmatched == (1., 1.)


def test_excluding_logsum_is_stable_without_cancellation():
    values = np.array([[1000., -1000., -math.inf], [0., 0., 0.]])
    result = _excluding_lse(values, 1)
    assert result[0, 0] == pytest.approx(0.)
    assert result[0, 1] == pytest.approx(1000.)
    np.testing.assert_allclose(result[1], math.log(3))
    np.testing.assert_allclose(_excluding_lse(values.T, 0), result.T)
    assert _excluding_lse(np.array([[1000.]]), 1)[0, 0] == 0.


def test_lbp_is_exact_on_acyclic_bipartite_support():
    allowed = [[True, False, False], [True, True, False], [False, True, True]]
    f = LogAssociationFactors([[1., 0., 0.], [2., 3., 0.], [0., 2., 1.]], [1., 0., 1.], [0., 1., 1.], allowed)
    a, b = exact_jpda(f), lbp_jpda(f)
    np.testing.assert_allclose(b.pair, a.pair, atol=1e-10)
    np.testing.assert_allclose(b.left_unmatched, a.left_unmatched, atol=1e-10)
    np.testing.assert_allclose(b.right_unmatched, a.right_unmatched, atol=1e-10)


def test_lbp_convergence_is_not_exact_posterior_or_error_certificate():
    f = LogAssociationFactors(np.zeros((2, 2)), [0., 0.], [0., 0.])
    exact, approximate = exact_jpda(f), lbp_jpda(f)
    assert exact.pair[0][0] == pytest.approx(2 / 7)
    assert abs(approximate.pair[0][0] - exact.pair[0][0]) > .005
    assert approximate.log_message_residual <= 1e-10
    assert approximate.marginal_consistency_error <= 1e-10
    assert not approximate.exact_model_marginals
    assert approximate.log_partition is None
    assert approximate.posterior_error_bound is None


@pytest.mark.parametrize('limit', [JPDALimits(max_iterations=1), JPDALimits(max_message_updates=1)])
def test_lbp_work_budget_has_no_unconverged_result(limit):
    f = LogAssociationFactors(np.ones((2, 2)), [0., 0.], [0., 0.])
    with pytest.raises(ValueError, match='budget'):
        lbp_jpda(f, limits=limit)


@pytest.mark.parametrize('field,value', [('max_dp_states', True), ('max_iterations', 0),
                                      ('tolerance', float('nan')), ('tolerance', 1.), ('tolerance', True)])
def test_configuration_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        replace(JPDALimits(), **{field: value})
