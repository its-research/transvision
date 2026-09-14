"""Independent exhaustive-action verification of conditional Hamming decoding."""
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import itertools
import json
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.hypothesis_bank import HypothesisBank, LogAssociationFactors, logsumexp
from transvision.models.event_track_v2x.identity_decision import decode_conditional_identity


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def signed(snapshot, **changes):
    changed = replace(snapshot, **changes)
    payload = asdict(changed)
    payload.pop("commit")
    return replace(changed, commit=digest(payload))


def oracle_weights(factors):
    """Cartesian actions and weights, without the production matcher/oracle."""
    n, m = factors.shape
    result = {}
    for action in itertools.product(range(-1, m), repeat=n):
        used = [column for column in action if column >= 0]
        if len(set(used)) != len(used) or any(column >= 0 and not factors.allowed[row][column]
                                            for row, column in enumerate(action)):
            continue
        log_weight = sum(factors.log_left_unmatched[row] if column < 0 else factors.log_pair[row][column]
                         for row, column in enumerate(action))
        log_weight += sum(value for column, value in enumerate(factors.log_right_unmatched) if column not in used)
        result[action] = math.exp(log_weight)
    return result


def risk(action, distribution):
    return (sum(probability * sum(x != y for x, y in zip(action, hypothesis)) / len(action)
                for hypothesis, probability in distribution.items()) if action else 0.)


@pytest.mark.parametrize("seed", range(24))
def test_random_conditional_bayes_action_matches_all_legal_actions_and_regret(seed):
    rng = np.random.default_rng(seed)
    n, m = seed % 4, (seed // 4) % 4
    factors = LogAssociationFactors(rng.normal(size=(n, m)), rng.normal(size=n), rng.normal(size=m),
                                    rng.uniform(size=(n, m)) > .2)
    weights = oracle_weights(factors)
    bank = HypothesisBank(factors, active_limit=3)
    snapshot = bank.advance(decision_us=1, expansion_budget=2 + seed % 7)
    decision = decode_conditional_identity(factors, snapshot)
    if not snapshot.active:
        assert decision.status == "unresolved_no_active_hypotheses"
        assert decision.choices is None and decision.model_truncation_regret_upper_estimate is None
        return
    retained = {leaf.choices: weights[leaf.choices] for leaf in snapshot.active}
    q = {key: value / sum(retained.values()) for key, value in retained.items()}
    rho = {key: value / sum(weights.values()) for key, value in weights.items()}
    best_conditional = min(risk(action, q) for action in weights)
    best_full = min(risk(action, rho) for action in weights)
    assert decision.choices in weights
    assert decision.conditional_expected_loss == pytest.approx(best_conditional, abs=1e-12)
    assert risk(decision.choices, q) == pytest.approx(best_conditional, abs=1e-12)
    actual_regret = risk(decision.choices, rho) - best_full
    assert actual_regret <= decision.model_truncation_regret_upper_estimate + 1e-12
    assert all(sum(row) == pytest.approx(1.) for row in decision.conditional_row_marginals)


def test_bayes_hamming_action_can_be_absent_from_retained_posterior_support():
    factors = LogAssociationFactors(np.zeros((3, 3)), np.zeros(3), np.zeros(3), np.eye(3, dtype=bool))
    complete = HypothesisBank(factors, active_limit=8).advance(decision_us=1, expansion_budget=20)
    retained_choices = {(0, 1, -1), (0, -1, 2), (-1, 1, 2)}
    active = tuple(leaf for leaf in complete.active if leaf.choices in retained_choices)
    frontier = tuple(sorted(leaf.choices for leaf in complete.active if leaf.choices not in retained_choices))
    log_retained = logsumexp(leaf.log_weight for leaf in active)
    eta = -math.expm1(log_retained - complete.log_partition_upper)
    # A valid synthetic retained-set receipt; full inference has eight legal
    # leaves. This deliberately tests decoder action scope, not bank search order.
    snapshot = signed(complete, active=active, active_limit=3, frontier_count=len(frontier),
                      frontier_prefix_slots=sum(map(len, frontier)), frontier_sha256=digest(frontier),
                      log_retained_weight=log_retained, eta_upper=eta)
    decision = decode_conditional_identity(factors, snapshot)
    assert decision.choices == (0, 1, 2)
    assert not decision.action_in_retained_set
    assert decision.conditional_expected_loss == pytest.approx(1. / 3)
    # Every MAP hypothesis is retained and has Hamming risk 4/9, greater than
    # the absent-support Bayes action. Whole-hypothesis 0-1 loss is different:
    # retained MAP has 2/3 risk; the absent action has risk one.
    assert all(risk(action, {key: 1. / 3 for key in retained_choices}) == pytest.approx(4. / 9)
               for action in retained_choices)
    assert 2. / 3 < 1.


def test_allowed_class_gates_and_private_unmatched_dummies_remain_legal():
    allowed = np.array([[True, False], [False, True], [False, False]])
    factors = LogAssociationFactors(np.zeros((3, 2)), np.full(3, 5.), np.zeros(2), allowed)
    snapshot = HypothesisBank(factors, active_limit=4).advance(decision_us=0, expansion_budget=20)
    decision = decode_conditional_identity(factors, snapshot)
    assert decision.choices == (-1, -1, -1)
    assert decision.unmatched_left == (0, 1, 2) and decision.unmatched_right == (0, 1)
    assert decision.pairs == ()
    # Separate dummy columns permit several unmatched rows simultaneously.
    assert all(row[-1] > .9 for row in decision.conditional_row_marginals[:2])


@pytest.mark.parametrize("n,m", [(0, 0), (0, 3), (3, 0)])
def test_empty_sides_have_declared_loss_and_actions(n, m):
    factors = LogAssociationFactors(np.zeros((n, m)), np.zeros(n), np.zeros(m))
    snapshot = HypothesisBank(factors).advance(decision_us=0, expansion_budget=10)
    decision = decode_conditional_identity(factors, snapshot)
    assert decision.choices == tuple(-1 for _ in range(n))
    assert decision.conditional_expected_loss == 0.
    assert decision.loss_range == (1. if n else 0.)
    assert decision.model_truncation_regret_upper_estimate == 0.


def test_no_active_set_does_not_fabricate_probability_or_action():
    factors = LogAssociationFactors([[0.]], [0.], [0.])
    snapshot = HypothesisBank(factors).advance(decision_us=0, expansion_budget=0)
    decision = decode_conditional_identity(factors, snapshot)
    assert decision.choices is None and decision.conditional_expected_loss is None
    assert decision.conditional_row_marginals == ()
    assert decision.model_truncation_regret_upper_estimate is None


def test_snapshot_factor_hash_and_active_integrity_are_checked():
    factors = LogAssociationFactors(np.zeros((2, 2)), np.zeros(2), np.zeros(2))
    snapshot = HypothesisBank(factors, active_limit=7).advance(decision_us=0, expansion_budget=10)
    different = LogAssociationFactors(np.ones((2, 2)), np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="digest"):
        decode_conditional_identity(different, snapshot)
    with pytest.raises(ValueError, match="payload hash"):
        decode_conditional_identity(factors, replace(snapshot, eta_upper=.5))
    with pytest.raises(ValueError, match="distinct"):
        decode_conditional_identity(factors, signed(snapshot, active=snapshot.active + (snapshot.active[0],)))
    illegal = replace(snapshot.active[0], choices=(0, 0))
    with pytest.raises(ValueError, match="one-to-one"):
        decode_conditional_identity(factors, signed(snapshot, active=(illegal,)))
    wrong_weight = replace(snapshot.active[0], log_weight=1.)
    with pytest.raises(ValueError, match="binding"):
        decode_conditional_identity(factors, signed(snapshot, active=(wrong_weight,)))
    with pytest.raises(ValueError, match="mass estimate"):
        decode_conditional_identity(factors, signed(snapshot, eta_upper=.5))


def test_large_common_log_offset_normalizes_without_losing_log_count():
    # Every row choice has zero log contribution; unused right baseline is a
    # common 1e20 offset. Direct exp(log_w - log_Z) would assign probability one
    # to both choices because float64 cannot retain log(2) beside 1e20.
    factors = LogAssociationFactors([[1e20]], [0.], [1e20])
    snapshot = HypothesisBank(factors, active_limit=2).advance(decision_us=0, expansion_budget=1)
    decision = decode_conditional_identity(factors, snapshot)
    assert decision.conditional_row_marginals == ((.5, .5),)
    assert decision.conditional_expected_loss == .5
    assert decision.model_omitted_mass_upper_estimate is None
    assert decision.model_truncation_regret_upper_estimate is None
    assert decision.bank_reported_omitted_mass_estimate == 0.


def test_original_factor_differences_survive_lost_absolute_leaf_weight_information():
    factors = LogAssociationFactors([[1e20]], [1.], [1e20])
    snapshot = HypothesisBank(factors, active_limit=2).advance(decision_us=0, expansion_budget=1)
    assert snapshot.active[0].log_weight == snapshot.active[1].log_weight == 1e20
    decision = decode_conditional_identity(factors, snapshot)
    assert decision.choices == (-1,)
    assert decision.conditional_row_marginals[0] == pytest.approx((1. / (1. + math.e), math.e / (1. + math.e)))
    assert decision.conditional_expected_loss == pytest.approx(1. / (1. + math.e))
    assert decision.model_omitted_mass_upper_estimate is None
    assert decision.model_truncation_regret_upper_estimate is None
    assert "magnitude-1000" in decision.mass_estimate_unavailable_reason


def test_unresolved_outside_numeric_domain_withholds_both_mass_and_regret():
    factors = LogAssociationFactors([[1001.]], [0.], [0.])
    snapshot = HypothesisBank(factors).advance(decision_us=0, expansion_budget=0)
    decision = decode_conditional_identity(factors, snapshot)
    assert decision.status == "unresolved_no_active_hypotheses"
    assert decision.choices is None
    assert decision.bank_reported_omitted_mass_estimate == 1.
    assert decision.model_omitted_mass_upper_estimate is None
    assert decision.model_truncation_regret_upper_estimate is None


def test_decoding_is_readonly_immutable_and_deterministic():
    factors = LogAssociationFactors([[2.]], [0.], [0.])
    bank = HypothesisBank(factors, active_limit=2)
    snapshot = bank.advance(decision_us=0, expansion_budget=1)
    before = asdict(snapshot)
    first = decode_conditional_identity(factors, snapshot)
    second = decode_conditional_identity(factors, snapshot)
    assert first == second and asdict(snapshot) == before
    assert bank.commits == (snapshot,)
    assert not first.true_posterior_or_tracking_metric_bound
    with pytest.raises(FrozenInstanceError):
        first.choices = (-1,)
