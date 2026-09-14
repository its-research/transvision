"""Model-risk split allocation invariants; no tracking-metric claims."""
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json

import numpy as np
import pytest

from transvision.models.event_track_v2x.hypothesis_bank import EvidenceUpdate, HypothesisBank, LogAssociationFactors
from transvision.models.event_track_v2x.risk_budget import RiskBudgetAllocator


def factors():
    return LogAssociationFactors.from_positive([[1000., 1.], [1., 1000.]], [.01, .01], [.01, .01])


def bank(**kwargs):
    return HypothesisBank(factors(), active_limit=1, **kwargs)


def test_highest_risk_then_stable_component_id_and_global_budget():
    allocator = RiskBudgetAllocator({"z": bank(), "a": bank()}, loss_ranges={"z": 2., "a": 1.})
    result = allocator.allocate("first", decision_us=0, split_budget=4)
    assert [step.component_id for step in result.trace[:2]] == ["z", "z"]
    # z now has low residual mass. a, still with no active leaf, gets work next.
    assert result.trace[2].component_id == "a"
    assert result.consumed_splits == 4
    assert sum(component.consumed_splits for component in result.components) == 4
    tie = RiskBudgetAllocator({"z": bank(), "a": bank()})
    tied = tie.allocate("first", decision_us=0, split_budget=1)
    assert tied.trace[0].component_id == "a"


def test_empty_active_starts_and_fully_empty_components_complete():
    empty = HypothesisBank(LogAssociationFactors(np.empty((0, 0)), [], []))
    allocator = RiskBudgetAllocator({"empty": empty, "unexpanded": bank()})
    initial = allocator.allocate("initial", decision_us=0, split_budget=0)
    results = {item.component_id: item for item in initial.components}
    assert results["empty"].status == "complete_support_retained"
    assert results["empty"].truncation_risk_upper == 0
    assert results["unexpanded"].bank.active == ()
    assert results["unexpanded"].truncation_risk_upper == 1
    next_step = allocator.allocate("next", decision_us=1, split_budget=2)
    assert next_step.consumed_splits == 2
    assert allocator.component_snapshot("unexpanded").active


def test_resource_exhausted_component_is_probed_once_not_busy_looped():
    allocator = RiskBudgetAllocator({"blocked": bank(max_frontier_nodes=1), "work": bank()},
                                    loss_ranges={"blocked": 100., "work": 1.})
    result = allocator.allocate("step", decision_us=0, split_budget=100)
    blocked = [item for item in result.trace if item.component_id == "blocked"]
    assert len(blocked) == 1 and blocked[0].consumed_splits == 0
    components = {item.component_id: item for item in result.components}
    assert components["blocked"].status == "resource_limited"
    assert components["work"].status == "active_capacity_limited"
    assert components["blocked"].bank.eta_upper == 1
    assert result.consumed_splits < 100


def test_all_component_updates_are_atomic_and_future_updates_fail():
    allocator = RiskBudgetAllocator({"a": bank(), "z": bank()})
    initial = allocator.allocate("initial", decision_us=10, split_budget=2)
    before = {key: allocator.component_snapshot(key) for key in ("a", "z")}
    delta = LogAssociationFactors(np.zeros((2, 2)), np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="future"):
        allocator.allocate("bad", decision_us=20, split_budget=4,
                           evidence={"a": EvidenceUpdate("valid", 20, delta),
                                     "z": EvidenceUpdate("future", 21, delta)})
    assert allocator.commits == (initial,)
    assert {key: allocator.component_snapshot(key) for key in before} == before
    good = allocator.allocate("good", decision_us=20, split_budget=1,
                              evidence={"a": EvidenceUpdate("valid", 20, delta)})
    assert good.previous_commit == initial.commit


def test_request_idempotence_conflicting_retries_and_unknown_components():
    allocator = RiskBudgetAllocator({"a": bank()})
    first = allocator.allocate("one", decision_us=0, split_budget=2)
    second = allocator.allocate("two", decision_us=1, split_budget=1)
    assert allocator.allocate("one", decision_us=0, split_budget=2) is first
    assert allocator.component_snapshot("a") == second.components[0].bank
    assert len(allocator.commits) == 2
    with pytest.raises(ValueError, match="conflicting"):
        allocator.allocate("one", decision_us=0, split_budget=3)
    with pytest.raises(ValueError, match="unknown"):
        allocator.allocate("three", decision_us=2, split_budget=1,
                           evidence={"unknown": EvidenceUpdate("x", 2, factors())})


def test_per_component_support_and_evidence_idempotency_preserved():
    allocator = RiskBudgetAllocator({"a": bank()})
    first = allocator.allocate("one", decision_us=10, split_budget=2)
    delta = LogAssociationFactors.from_positive([[1., 1e8], [1e8, 1.]], [1., 1.], [1., 1.])
    evidence = EvidenceUpdate("reversal", 20, delta)
    second = allocator.allocate("two", decision_us=20, split_budget=1, evidence={"a": evidence})
    assert second.components[0].bank.active[0].choices == (1, 0)
    third = allocator.allocate("three", decision_us=21, split_budget=0, evidence={"a": evidence})
    assert third.components[0].bank.evidence_ids == ("reversal",)
    assert allocator.commits[0] is first
    gated = LogAssociationFactors(np.zeros((2, 2)), np.zeros(2), np.zeros(2),
                                  np.array([[True, False], [True, True]]))
    with pytest.raises(ValueError, match="support"):
        allocator.allocate("bad", decision_us=22, split_budget=1,
                           evidence={"a": EvidenceUpdate("changed", 22, gated)})


def test_owned_copies_do_not_mutate_caller_banks():
    original = bank()
    allocator = RiskBudgetAllocator({"a": original})
    allocator.allocate("one", decision_us=0, split_budget=2)
    assert original.commits == () and original.frontier_constraints == ((),)
    original.advance(decision_us=1, expansion_budget=10)
    assert allocator.component_snapshot("a").total_expansions == 2


def test_zero_loss_empty_allocator_and_request_cap():
    allocator = RiskBudgetAllocator({"a": bank()}, loss_ranges={"a": 0.}, max_requests=1)
    result = allocator.allocate("one", decision_us=0, split_budget=100)
    assert result.consumed_splits == 0 and result.trace == ()
    assert result.components[0].status == "zero_loss_range"
    with pytest.raises(ValueError, match="max_requests"):
        allocator.allocate("two", decision_us=1, split_budget=0)
    empty = RiskBudgetAllocator({}).allocate("empty", decision_us=0, split_budget=10)
    assert empty.components == () and empty.consumed_splits == 0


def test_replay_and_hash_chain_are_exact():
    transcripts = []
    for _ in range(2):
        allocator = RiskBudgetAllocator({"z": bank(), "a": bank()})
        first = allocator.allocate("one", decision_us=0, split_budget=3)
        second = allocator.allocate("two", decision_us=1, split_budget=2)
        assert second.previous_commit == first.commit
        for item in allocator.commits:
            payload = asdict(item)
            commit = payload.pop("commit")
            assert hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() == commit
        transcripts.append([asdict(item) for item in allocator.commits])
    assert transcripts[0] == transcripts[1]


@pytest.mark.parametrize("invalid", [-1., float("nan"), float("inf"), True])
def test_invalid_risk_ranges_rejected(invalid):
    with pytest.raises(ValueError):
        RiskBudgetAllocator({"a": bank()}, loss_ranges={"a": invalid})


def test_child_history_cap_aborts_entire_staged_allocation():
    allocator = RiskBudgetAllocator({"a": bank(), "z": bank(max_commits=1)})
    first = allocator.allocate("one", decision_us=0, split_budget=0)
    with pytest.raises(ValueError, match="max_commits"):
        allocator.allocate("two", decision_us=1, split_budget=0)
    assert allocator.commits == (first,)
    assert allocator.component_snapshot("a") == first.components[0].bank


def test_readonly_component_ancestry_returns_complete_exclusive_inclusive_slice():
    allocator = RiskBudgetAllocator({"a": bank()})
    assert allocator.component_ancestry("a", "0" * 64) == ()
    initial = allocator.allocate("one", decision_us=0, split_budget=1).components[0].bank
    final = allocator.allocate("two", decision_us=1, split_budget=2).components[0].bank
    original_commits = allocator.commits
    original_tip = allocator.component_snapshot("a")
    path = allocator.component_ancestry("a", initial.commit)
    assert len(path) == 3  # Decision-only commit plus two internal split commits.
    assert path[0].previous_commit == initial.commit
    assert path[-1] is original_tip and path[-1].commit == final.commit
    assert all(right.previous_commit == left.commit for left, right in zip(path, path[1:]))
    assert allocator.component_ancestry("a", initial.commit, path[-2].commit) == path[:-1]
    assert allocator.component_ancestry("a", final.commit, final.commit) == ()
    assert allocator.component_ancestry("a", "0" * 64)[0].previous_commit == "0" * 64
    with pytest.raises(FrozenInstanceError):
        path[0].eta_upper = 0.
    assert allocator.commits is original_commits
    assert allocator.component_snapshot("a") is original_tip


def test_ancestry_unknown_hash_fork_and_reversed_ranges_fail_closed():
    allocator = RiskBudgetAllocator({"a": bank(), "b": bank()})
    first = allocator.allocate("one", decision_us=0, split_budget=2).components[0].bank
    last = allocator.allocate("two", decision_us=1, split_budget=1).components[0].bank
    other_bank = bank()
    foreign = other_bank.advance(decision_us=123, expansion_budget=2)
    for key, after, through in (("missing", first.commit, None),
                                ("a", "", None), ("a", first.commit, ""),
                                ("a", "f" * 64, None), ("a", foreign.commit, None),
                                ("a", first.commit, foreign.commit), ("a", last.commit, first.commit)):
        with pytest.raises(ValueError):
            allocator.component_ancestry(key, after, through)
    assert allocator.component_snapshot("a") == last


@pytest.mark.parametrize("corruption", ["origin", "parent", "payload"])
def test_ancestry_validates_owned_chain_before_returning_any_snapshots(corruption):
    allocator = RiskBudgetAllocator({"a": bank()})
    allocator.allocate("one", decision_us=0, split_budget=2)
    owned = allocator._banks["a"]
    original = owned.commits[1]
    if corruption == "origin":
        damaged = replace(original, original_factors_sha256="f" * 64)
    elif corruption == "parent":
        damaged = replace(original, previous_commit="f" * 64)
    else:
        damaged = replace(original, eta_upper=.1234)
    # Exercise verification against in-memory corruption, not a supported writer.
    owned._commits = (owned.commits[0], damaged, *owned.commits[2:])
    with pytest.raises(ValueError, match="ancestry"):
        allocator.component_ancestry("a", "0" * 64)
