"""Exhaustive independent assignments validate ranking and residual regions."""
from dataclasses import replace
import itertools
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.hypothesis_bank import LogAssociationFactors
from tools.event_track_v2x.ranked_partial_assignment import (
    RankedAssignmentLimits, k_best_partial_assignments,
)


def oracle(factors):
    n, m = factors.shape
    result = {}
    for choices in itertools.product(range(-1, m), repeat=n):
        used = [j for j in choices if j >= 0]
        if len(used) != len(set(used)) or any(j >= 0 and not factors.allowed[i][j] for i, j in enumerate(choices)):
            continue
        result[choices] = math.fsum([
            *(factors.log_left_unmatched[i] if j < 0 else factors.log_pair[i][j] for i, j in enumerate(choices)),
            *(v for j, v in enumerate(factors.log_right_unmatched) if j not in used),
        ])
    return result


def logsum(values):
    values = list(values)
    if not values:
        return -math.inf
    maximum = max(values)
    return maximum+math.log(math.fsum(math.exp(v-maximum) for v in values))


def assert_cover(result, expected):
    found = {h.choices: h.log_weight for h in result.hypotheses}
    assert len(found) == len(result.hypotheses)
    assert all(found[c] == pytest.approx(expected[c], abs=1e-11) for c in found)
    for choices in expected:
        assert int(choices in found)+sum(r.contains(choices) for r in result.frontier) == 1
    for region in result.frontier:
        actual = logsum(v for c, v in expected.items() if region.contains(c))
        assert actual <= region.log_mass_upper+1e-11
    residual = logsum(v for c, v in expected.items() if c not in found)
    assert residual <= result.residual_log_mass_upper+1e-11
    eta = 0. if residual == -math.inf else math.exp(residual-logsum(expected.values()))
    assert eta <= result.omitted_mass_upper+1e-11
    ordered = sorted(expected.values(), reverse=True)
    assert [h.log_weight for h in result.hypotheses] == pytest.approx(ordered[:len(found)], abs=1e-10)
    assert result.interval_arithmetic_certified is False


@pytest.mark.parametrize('shape', itertools.product(range(5), repeat=2))
def test_full_enumeration_matches_independent_partial_assignment_oracle(shape):
    n, m = shape
    rng = np.random.default_rng(17*n+m)
    factors = LogAssociationFactors(rng.normal(size=(n, m))*4, rng.normal(size=n),
                                   rng.normal(size=m), rng.random((n, m)) > .25)
    expected = oracle(factors)
    result = k_best_partial_assignments(factors, len(expected)+1)
    assert_cover(result, expected)
    assert result.factor_sha256 == factors.digest()
    assert result.support_exhausted and not result.requested_k_reached
    assert result.omitted_mass_upper == 0.


@pytest.mark.parametrize('solves,frontier', itertools.product([0, 1, 2, 5, 20], [1, 2, 8]))
def test_budget_stop_preserves_ranked_prefix_and_complete_disjoint_cover(solves, frontier):
    factors = LogAssociationFactors([[5., 4., 2.], [4., 2., 1.], [2., 1., 3.]], [0.]*3, [0.]*3)
    limits = RankedAssignmentLimits(max_solves=solves, max_frontier=frontier)
    result = k_best_partial_assignments(factors, 10, limits=limits)
    assert_cover(result, oracle(factors))
    assert result.assignment_solves <= solves and result.peak_frontier <= frontier
    if len(result.hypotheses) < 10 and not result.support_exhausted:
        assert result.termination in ('solve-budget', 'frontier-budget')
        assert not result.requested_k_reached


def test_ties_have_seven_distinct_physical_matchings_not_dummy_permutations():
    factors = LogAssociationFactors(np.zeros((2, 2)), [0., 0.], [0., 0.])
    a = k_best_partial_assignments(factors, 8)
    b = k_best_partial_assignments(factors, 8)
    assert a == b and len(a.hypotheses) == 7
    assert_cover(a, oracle(factors))
    assert a.retained_log_mass == pytest.approx(math.log(7))


def test_requested_k_is_not_full_posterior_or_exhaustion():
    factors = LogAssociationFactors.from_positive([[8.]], [2.], [2.])
    result = k_best_partial_assignments(factors, 1)
    assert result.requested_k_reached and not result.support_exhausted
    assert result.hypotheses[0].choices == (0,)
    assert result.hypotheses[0].log_weight == pytest.approx(math.log(8))
    assert result.omitted_mass_upper == pytest.approx(1/3, abs=1e-12)
    changed = LogAssociationFactors.from_positive([[8.]], [3.], [3.])
    assert k_best_partial_assignments(changed, 1).hypotheses[0].choices == (-1,)


def test_row_column_gauges_preserve_ranking_and_mass_fraction():
    f = LogAssociationFactors([[4., 1.], [2., 3.]], [0., -1.], [-2., 0.])
    a, b = [200., -300.], [700., -500.]
    g = LogAssociationFactors([[f.log_pair[i][j]+a[i]+b[j] for j in range(2)] for i in range(2)],
        [f.log_left_unmatched[i]+a[i] for i in range(2)],
        [f.log_right_unmatched[j]+b[j] for j in range(2)])
    left, right = k_best_partial_assignments(f, 4), k_best_partial_assignments(g, 4)
    assert [h.choices for h in left.hypotheses] == [h.choices for h in right.hypotheses]
    assert [y.log_weight-x.log_weight for x, y in zip(left.hypotheses, right.hypotheses)] == pytest.approx([sum(a+b)]*4)
    assert left.omitted_mass_upper == pytest.approx(right.omitted_mass_upper, abs=1e-10)


def test_gated_huge_pair_potentials_are_never_candidates():
    f = LogAssociationFactors([[1000., 0.], [0., 1000.]], [0., 0.], [0., 0.],
                             [[False, False], [False, False]])
    result = k_best_partial_assignments(f, 2)
    assert [h.choices for h in result.hypotheses] == [(-1, -1)]
    assert result.support_exhausted and result.omitted_mass_upper == 0.


def test_tiny_positive_residual_does_not_turn_into_false_zero_certificate():
    f = LogAssociationFactors([[1000.]], [0.], [0.])
    result = k_best_partial_assignments(f, 1)
    assert result.residual_log_mass_upper >= 0.
    assert 0. < result.omitted_mass_upper < 1e-300
    assert not result.support_exhausted


def test_dense_realistic_size_does_not_call_an_exhaustive_oracle(monkeypatch):
    import transvision.models.event_track_v2x.hypothesis_bank as bank
    monkeypatch.setattr(bank, 'enumerate_assignments', lambda *a, **kw: pytest.fail('oracle called'))
    n = 32
    f = LogAssociationFactors(np.eye(n)*20., [-10.]*n, [-10.]*n)
    result = k_best_partial_assignments(f, 4, limits=RankedAssignmentLimits(max_solves=150))
    assert result.requested_k_reached
    assert result.hypotheses[0].choices == tuple(range(n))
    assert result.assignment_solves <= 150 and result.peak_matrix_cells == 2*n*n


@pytest.mark.parametrize('name,value', [('max_solves', -1), ('max_solves', True),
    ('max_frontier', 0), ('max_matrix_cells', 0), ('max_returned', 0)])
def test_invalid_resource_limits(name, value):
    with pytest.raises(ValueError):
        replace(RankedAssignmentLimits(), **{name: value})


@pytest.mark.parametrize('k', [0, -1, True, 1.5, 4097])
def test_invalid_requested_count(k):
    with pytest.raises(ValueError):
        k_best_partial_assignments(LogAssociationFactors([[0.]], [0.], [0.]), k)


def test_matrix_capacity_fails_before_solver_or_allocation(monkeypatch):
    import tools.event_track_v2x.ranked_partial_assignment as module
    f = LogAssociationFactors(np.zeros((4, 4)), [0.]*4, [0.]*4)
    monkeypatch.setattr(module.np, 'full', lambda *a, **kw: pytest.fail('matrix allocated'))
    with pytest.raises(ValueError, match='capacity'):
        module.k_best_partial_assignments(f, 1, limits=RankedAssignmentLimits(max_matrix_cells=8))


def test_zero_solve_root_bound_counts_both_unmatched_families():
    f = LogAssociationFactors.from_positive([[2., 3.], [4., 5.]], [7., 11.], [13., 17.])
    result = k_best_partial_assignments(f, 2, limits=RankedAssignmentLimits(max_solves=0))
    expected = 13*17*(7+2/13+3/17)*(11+4/13+5/17)
    assert result.residual_log_mass_upper == pytest.approx(math.log(expected), abs=1e-11)
    assert result.hypotheses == () and result.omitted_mass_upper == 1.
    assert_cover(result, oracle(f))


def test_increasing_k_preserves_prefix_without_changing_any_score():
    f = LogAssociationFactors([[3., 1.], [4., 2.]], [0., 0.], [0., 0.])
    complete = k_best_partial_assignments(f, 8)
    for k in range(1, 8):
        part = k_best_partial_assignments(f, k)
        assert part.hypotheses == complete.hypotheses[:k]
        assert_cover(part, oracle(f))


def test_unexpected_solver_failure_is_not_treated_as_zero_mass(monkeypatch):
    import tools.event_track_v2x.ranked_partial_assignment as module
    def failure(*args, **kwargs):
        raise ValueError('numerical backend failure')
    monkeypatch.setattr(module, 'linear_sum_assignment', failure)
    with pytest.raises(ValueError, match='numerical backend failure'):
        module.k_best_partial_assignments(LogAssociationFactors([[0.]], [0.], [0.]), 1)
