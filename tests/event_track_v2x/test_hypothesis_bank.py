"""Independent small-graph checks for bounded recoverable assignment inference."""
from dataclasses import FrozenInstanceError, asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from transvision.models.event_track_v2x import hypothesis_bank as core
from transvision.models.event_track_v2x.hypothesis_bank import (
    EvidenceUpdate, HypothesisBank, LogAssociationFactors, ambiguity_components,
    assignment_log_weight, component_factors, enumerate_assignments, logsumexp,
    posterior_omitted_mass, regret_upper_bound, row_relaxation_log_upper,
)


def independent_weights(factors):
    """Cartesian-product oracle independent of recursive inference and weight code."""
    n, m = factors.shape
    result = {}
    for choices in itertools.product(range(-1, m), repeat=n):
        used = [column for column in choices if column >= 0]
        if len(set(used)) != len(used):
            continue
        if any(column >= 0 and not factors.allowed[row][column] for row, column in enumerate(choices)):
            continue
        log_weight = sum(factors.log_left_unmatched[row] if column < 0 else factors.log_pair[row][column]
                         for row, column in enumerate(choices))
        log_weight += sum(value for column, value in enumerate(factors.log_right_unmatched) if column not in used)
        result[choices] = math.exp(log_weight)
    return result


@pytest.mark.parametrize("pair,left,right", [
    ([[0.]], [1.], [1.]), ([[-1.]], [1.], [1.]), ([[math.nan]], [1.], [1.]),
    ([[math.inf]], [1.], [1.]), ([[1.]], [0.], [1.]), ([[1.]], [1.], [-1.]),
    ([[1., 2.]], [1., 1.], [1.]),
])
def test_positive_potential_contract(pair, left, right):
    with pytest.raises(ValueError):
        LogAssociationFactors.from_positive(pair, left, right)


@pytest.mark.parametrize("n,m", [(0, 0), (0, 3), (3, 0), (1, 1), (2, 3), (3, 2)])
def test_empty_and_rectangular_exact_oracle(n, m):
    factors = LogAssociationFactors(np.zeros((n, m)), np.zeros(n), np.zeros(m))
    brute = independent_weights(factors)
    oracle = enumerate_assignments(factors)
    assert {item.choices: math.exp(item.log_weight) for item in oracle} == brute
    bank = HypothesisBank(factors, active_limit=len(brute))
    snapshot = bank.advance(decision_us=0, expansion_budget=1000)
    assert len(snapshot.active) == len(brute)
    assert snapshot.eta_upper == 0
    assert snapshot.frontier_count == 0
    assert snapshot.log_retained_weight == pytest.approx(math.log(len(brute)))


def test_input_and_committed_output_are_deeply_immutable():
    pair = np.array([[3., 1.], [1., 3.]])
    allowed = np.ones((2, 2), dtype=bool)
    factors = LogAssociationFactors.from_positive(pair, [1., 1.], [1., 1.], allowed=allowed)
    digest = factors.digest()
    pair[:] = 123
    allowed[:] = False
    assert factors.digest() == digest
    with pytest.raises(FrozenInstanceError):
        factors.log_pair = ((0., 0.), (0., 0.))
    bank = HypothesisBank(factors, active_limit=1)
    first = bank.advance(decision_us=1, expansion_budget=2)
    original = asdict(first)
    bank.advance(decision_us=2, expansion_budget=5)
    assert asdict(bank.commits[0]) == original
    with pytest.raises(FrozenInstanceError):
        first.active[0].choices = (-1, -1)
    with pytest.raises(TypeError):
        factors.log_pair[0][0] = 0


@pytest.mark.parametrize("seed", range(16))
def test_random_mass_bounds_prefix_partition_and_monotonic_refinement(seed):
    rng = np.random.default_rng(seed)
    n, m = seed % 4, (seed // 4) % 4
    factors = LogAssociationFactors(rng.normal(size=(n, m)), rng.normal(size=n), rng.normal(size=m),
                                    rng.uniform(size=(n, m)) > .25)
    brute = independent_weights(factors)
    z = sum(brute.values())
    oracle = enumerate_assignments(factors)
    assert math.exp(logsumexp(item.log_weight for item in oracle)) == pytest.approx(z)
    assert math.exp(row_relaxation_log_upper(factors)) >= z - 1e-10
    bank = HypothesisBank(factors, active_limit=2)
    last_upper = math.inf
    for decision in range(10):
        snapshot = bank.advance(decision_us=decision, expansion_budget=1)
        retained = sum(brute[item.choices] for item in snapshot.active)
        eta = 1. - retained / z
        assert snapshot.eta_upper >= eta - 2e-13
        assert snapshot.log_partition_upper >= math.log(z) - 2e-13
        assert snapshot.log_partition_upper <= last_upper + 1e-11
        assert snapshot.expansions <= 1
        last_upper = snapshot.log_partition_upper
        # Exactly one active leaf or frontier prefix represents each hypothesis.
        active = {item.choices for item in snapshot.active}
        for choices in brute:
            coverage = int(choices in active) + sum(choices[:len(prefix)] == prefix
                                                  for prefix in bank.frontier_constraints)
            assert coverage == 1
        for prefix in bank.frontier_constraints:
            actual_mass = sum(weight for choices, weight in brute.items() if choices[:len(prefix)] == prefix)
            assert math.exp(row_relaxation_log_upper(factors, prefix)) >= actual_mass - 1e-10


def test_ambiguity_components_preserve_partition_including_isolates():
    factors = LogAssociationFactors.from_positive(np.full((4, 4), 2.), np.full(4, .3), np.full(4, .4),
                                                 allowed=np.array([[1, 1, 0, 0], [1, 1, 0, 0],
                                                                   [0, 0, 1, 0], [0, 0, 0, 0]], bool))
    components = ambiguity_components(factors)
    assert [(item.left, item.right) for item in components] == [((0, 1), (0, 1)), ((2,), (2,)), ((3,), ()), ((), (3,))]
    full_log_z = logsumexp(item.log_weight for item in enumerate_assignments(factors))
    component_log_z = sum(logsumexp(item.log_weight for item in enumerate_assignments(component_factors(factors, component)))
                          for component in components)
    assert full_log_z == pytest.approx(component_log_z)


def reversal_fixture():
    initial = LogAssociationFactors.from_positive([[1000., 1.], [1., 1000.]], [.01, .01], [.01, .01])
    evidence = EvidenceUpdate("late-disambiguating-observation", 20,
                              LogAssociationFactors.from_positive([[1., 1e8], [1e8, 1.]], [1., 1.], [1., 1.]))
    return initial, evidence


def test_reconstruct_never_enumerated_assignment_without_oracle(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("bounded inference must not call exhaustive enumeration")
    monkeypatch.setattr(core, "enumerate_assignments", forbidden)
    initial, evidence = reversal_fixture()
    bank = HypothesisBank(initial, active_limit=1)
    first = bank.advance(decision_us=10, expansion_budget=2)
    assert first.active[0].choices == (0, 1)
    assert (1,) in bank.frontier_constraints  # Swapped identity remains an unexpanded constraint.
    assert (1, 0) not in bank._discovered
    updated = bank.advance(decision_us=20, expansion_budget=1, evidence=evidence)
    assert updated.active[0].choices == (1, 0)
    assert updated.recovery_events[0].previously_enumerated is False
    assert updated.expansions == 1 and updated.total_expansions == 3
    assert updated.active[0].first_discovered_us == 20
    assert updated.active[0].evidence_history == (evidence.evidence_id,)
    # Same initial top-1 and subsequent expansion allowance cannot help a method
    # that has permanently erased all prefixes except its retained identity.
    permanently_retained = first.active[0].choices
    assert permanently_retained != updated.active[0].choices
    combined = initial.add(evidence.factors)
    assert assignment_log_weight(combined, updated.active[0].choices) > assignment_log_weight(combined, permanently_retained)
    assert bank.commits[0] == first  # Past outputs are never rewritten.


def test_recovery_can_be_delayed_until_compute_budget_available():
    initial, evidence = reversal_fixture()
    bank = HypothesisBank(initial, active_limit=1)
    bank.advance(decision_us=10, expansion_budget=2)
    delayed = bank.advance(decision_us=20, expansion_budget=0, evidence=evidence)
    assert delayed.recovery_events == ()
    resumed = bank.advance(decision_us=21, expansion_budget=1)
    assert resumed.recovery_events[0].choices == (1, 0)


def test_future_conflicting_duplicate_and_same_payload_idempotence():
    initial, evidence = reversal_fixture()
    bank = HypothesisBank(initial, active_limit=1, information_us=10)
    with pytest.raises(ValueError, match="causal"):
        bank.advance(decision_us=9, expansion_budget=1)
    first = bank.advance(decision_us=10, expansion_budget=2)
    with pytest.raises(ValueError, match="future"):
        bank.advance(decision_us=19, expansion_budget=1, evidence=evidence)
    assert len(bank.commits) == 1 and bank.commits[0] == first
    second = bank.advance(decision_us=20, expansion_budget=1, evidence=evidence)
    assert bank.advance(decision_us=25, expansion_budget=100, evidence=evidence) is second
    assert len(bank.commits) == 2
    conflicting = EvidenceUpdate(evidence.evidence_id, 21, evidence.factors)
    with pytest.raises(ValueError, match="conflicting"):
        bank.advance(decision_us=25, expansion_budget=1, evidence=conflicting)
    assert len(bank.commits) == 2


def test_evidence_shape_or_gate_change_fails_before_mutation():
    initial, _ = reversal_fixture()
    bank = HypothesisBank(initial)
    original = bank.advance(decision_us=0, expansion_budget=0)
    mismatched = LogAssociationFactors(np.zeros((2, 2)), [0., 0.], [0., 0.],
                                       np.array([[True, False], [True, True]]))
    with pytest.raises(ValueError, match="support"):
        bank.advance(decision_us=1, expansion_budget=1, evidence=EvidenceUpdate("different", 1, mismatched))
    assert bank.commits == (original,)


def test_commit_hash_recomputes_and_replays_byte_identically():
    initial, evidence = reversal_fixture()
    transcripts = []
    for _ in range(2):
        bank = HypothesisBank(initial, active_limit=1)
        bank.advance(decision_us=10, expansion_budget=2)
        bank.advance(decision_us=20, expansion_budget=1, evidence=evidence)
        for snapshot in bank.commits:
            encoded = asdict(snapshot)
            commit = encoded.pop("commit")
            assert hashlib.sha256(json.dumps(encoded, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() == commit
        assert bank.commits[1].previous_commit == bank.commits[0].commit
        transcripts.append(json.dumps([asdict(snapshot) for snapshot in bank.commits], sort_keys=True, allow_nan=False))
    assert transcripts[0] == transcripts[1]


@pytest.mark.parametrize("scale", [-1000., 1000.])
def test_log_space_extreme_positive_potentials_are_stable(scale):
    factors = LogAssociationFactors([[scale, -scale], [-scale, scale]], [scale, -scale], [-scale, scale])
    bank = HypothesisBank(factors, active_limit=2)
    snapshot = bank.advance(decision_us=0, expansion_budget=4)
    assert math.isfinite(snapshot.log_partition_upper)
    assert 0 <= snapshot.eta_upper <= 1
    exact_log_z = logsumexp(item.log_weight for item in enumerate_assignments(factors))
    assert snapshot.log_partition_upper >= exact_log_z - 1e-10


def test_oracle_fails_instead_of_silently_truncating():
    factors, _ = reversal_fixture()
    with pytest.raises(ValueError, match="exceeded"):
        enumerate_assignments(factors, max_hypotheses=2)


@pytest.mark.parametrize("kwargs,reason", [
    ({"max_frontier_nodes": 2}, "max_frontier_nodes"),
    ({"max_discovered_leaves": 1}, "max_discovered_leaves"),
])
def test_resource_limits_keep_unresolved_support_without_silent_pruning(kwargs, reason):
    factors, _ = reversal_fixture()
    bank = HypothesisBank(factors, active_limit=1, **kwargs)
    snapshot = bank.advance(decision_us=0, expansion_budget=1000)
    assert snapshot.resource_limited and reason in snapshot.limit_reasons
    assert snapshot.frontier_count <= kwargs.get("max_frontier_nodes", 4096)
    assert snapshot.discovered_assignments <= kwargs.get("max_discovered_leaves", 4096)
    brute = independent_weights(factors)
    for choices in brute:
        assert sum(choices == item.choices for item in snapshot.active) + sum(
            choices[:len(prefix)] == prefix for prefix in bank.frontier_constraints) == 1
    retained = sum(brute[item.choices] for item in snapshot.active)
    assert snapshot.eta_upper >= 1. - retained / sum(brute.values()) - 1e-12


def test_evidence_and_commit_history_caps_fail_closed_without_changing_state():
    factors, evidence = reversal_fixture()
    bank = HypothesisBank(factors, active_limit=1, max_evidence_updates=1)
    first = bank.advance(decision_us=20, expansion_budget=2, evidence=evidence)
    with pytest.raises(ValueError, match="max_evidence_updates"):
        bank.advance(decision_us=21, expansion_budget=1,
                     evidence=EvidenceUpdate("second", 21, evidence.factors))
    assert bank.commits == (first,)
    capped = HypothesisBank(factors, active_limit=1, max_commits=1)
    first = capped.advance(decision_us=20, expansion_budget=1, evidence=evidence)
    assert capped.advance(decision_us=21, expansion_budget=1, evidence=evidence) is first
    with pytest.raises(ValueError, match="max_commits"):
        capped.advance(decision_us=21, expansion_budget=1)
    assert capped.commits == (first,)


@pytest.mark.parametrize("bad", [True, -1, 1.5, "2"])
def test_budget_and_timestamps_are_strict_integers(bad):
    factors, _ = reversal_fixture()
    with pytest.raises(ValueError):
        HypothesisBank(factors).advance(decision_us=0, expansion_budget=bad)
    with pytest.raises(ValueError):
        HypothesisBank(factors).advance(decision_us=bad, expansion_budget=0)


def test_illegal_prefixes_and_nonfinite_factors_rejected():
    factors, _ = reversal_fixture()
    for choices in [(0, 0), (2, -1), (True, -1), (-2,), (0, 1, -1)]:
        with pytest.raises(ValueError):
            row_relaxation_log_upper(factors, choices)
    with pytest.raises(ValueError):
        LogAssociationFactors([[math.nan]], [0.], [0.])
    with pytest.raises(ValueError):
        LogAssociationFactors([[0.]], [0.], [0.], [[1]])
    with pytest.raises(ValueError):
        LogAssociationFactors([[1e308]], [1e308], [1e308])


def test_bayes_omitted_mass_formula_and_likelihood_ratio_bound():
    rng = np.random.default_rng(13)
    for _ in range(100):
        eta = rng.uniform(.001, .999)
        u, v = np.exp(rng.uniform(-10, 10, size=2))
        actual = eta * v / ((1 - eta) * u + eta * v)
        assert posterior_omitted_mass(eta, log_likelihood_ratio=math.log(v / u)) == pytest.approx(actual)
        kappa = v / u * rng.uniform(1, 10)
        upper = posterior_omitted_mass(eta, log_likelihood_ratio=math.log(kappa))
        assert actual <= upper + 1e-14
    amplified = posterior_omitted_mass(.001, log_likelihood_ratio=math.log(1e6))
    assert amplified == pytest.approx(.999001997004992)
    false_bound = posterior_omitted_mass(.001, log_likelihood_ratio=math.log(2))
    assert amplified > false_bound  # Invalid assumed kappa is not a guarantee.
    assert posterior_omitted_mass(0., log_likelihood_ratio=1000.) == 0.
    assert posterior_omitted_mass(1., log_likelihood_ratio=-1000.) == 1.


def test_random_bayes_regret_with_explicit_conditional_and_model_tv_errors():
    rng = np.random.default_rng(42)
    for _ in range(200):
        model = rng.dirichlet(np.ones(7))
        retained = rng.choice(7, size=3, replace=False)
        eta = 1 - sum(model[retained])
        conditional = np.zeros(7)
        conditional[retained] = model[retained] / (1 - eta)
        approximate = .95 * conditional + .05 * rng.dirichlet(np.ones(7))
        truth = .9 * model + .1 * rng.dirichlet(np.ones(7))
        loss = rng.uniform(size=(5, 7))
        epsilon = .5 * sum(abs(approximate - conditional))
        zeta = .5 * sum(abs(truth - model))
        a_approximate = np.argmin(loss @ approximate)
        a_truth = np.argmin(loss @ truth)
        regret = (loss @ truth)[a_approximate] - (loss @ truth)[a_truth]
        assert regret <= regret_upper_bound(eta, epsilon=epsilon, zeta=zeta) + 1e-12
        a_conditional = np.argmin(loss @ conditional)
        model_regret = (loss @ model)[a_conditional] - np.min(loss @ model)
        assert model_regret <= eta + 1e-12


def test_shared_action_space_and_calibration_are_necessary_not_automatically_true():
    # eta=0, but deleting the optimal action violates the common-action-space premise.
    losses = np.array([[0.], [1.]])
    restricted_regret = losses[1, 0] - losses[0, 0]
    assert restricted_regret > regret_upper_bound(0.)
    # Model strongly favors identity 0; reality strongly favors identity 1.
    model = np.array([.999, .001])
    truth = model[::-1]
    actual_regret = truth[1] - truth[0]
    assert actual_regret > regret_upper_bound(.001)
    assert actual_regret <= regret_upper_bound(.001, zeta=.5 * sum(abs(model - truth)))


def test_cli_verifier_create_once_and_explicit_scope(tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root / "tools/event_track_v2x/verify_recoverable_hypotheses.py"),
               "--output", str(tmp_path / "proof.json"), "--trials", "8"]
    first = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    original_bytes = (tmp_path / "proof.json").read_bytes()
    report = json.loads(original_bytes)
    assert report["status"] == "verified"
    assert len(report["partitions"]) == 8
    assert report["scope"]["complete_tracker"] is False
    assert report["scope"]["real_dataset_validation"] is False
    assert report["scope"]["gpu_used"] is False
    assert report["recovery"][0]["replay_byte_equal"] is True
    for relative, expected in report["source_hashes"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    repeat = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert repeat.returncode != 0 and "create-once" in repeat.stderr
    assert (tmp_path / "proof.json").read_bytes() == original_bytes
