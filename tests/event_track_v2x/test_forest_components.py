from dataclasses import asdict, replace
from itertools import product
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.forest_components import (
    ComponentForestBank, join_actions, split_forest, validate_component_snapshot,
)
from transvision.models.event_track_v2x.identity_forest import (
    ForestFactors, IdentityNode, RecoverableForestBank, decode_identity_roots, digest,
)
from test_identity_forest import oracle, risk, factors as dense_factors


def scene(seed=0, size=6):
    rng = np.random.default_rng(seed)
    nodes = tuple(IdentityNode(str(i), (i//2) % 2, i, i, str(i//3)) for i in range(size))
    rows = tuple(tuple((p, float(rng.normal())) for p in [-1] + list(range(i % 2, i, 2))) for i in range(size))
    return ForestFactors(nodes, rows)


@pytest.mark.parametrize('seed', range(24))
def test_component_partition_mass_and_risk_match_independent_joint_enumeration(seed):
    factors = scene(seed)
    components = split_forest(factors)
    assert [c.indices for c in components] == [(0, 2, 4), (1, 3, 5)]
    joint = oracle(factors)
    local = [oracle(c.factors) for c in components]
    lifted = {}
    for actions in product(*local):
        action = join_actions(factors, components, actions)
        lifted[action] = math.prod(o[a][1] for o, a in zip(local, actions))
    assert set(lifted) == set(joint)
    assert all(lifted[a] == pytest.approx(w, rel=1e-13) for a, (_, w) in joint.items())
    bank = ComponentForestBank(active_limit=1+seed % 3, max_component_nodes=3)
    snapshot = bank.advance(factors=factors, decision_us=10, expansion_budget=seed,
                            message_id='initial')
    validate_component_snapshot(factors, snapshot)
    z = math.fsum(w for _, w in joint.values())
    kept = math.prod(math.fsum(o[a][1] for a in c.posterior.active)
                     for o, c in zip(local, snapshot.components))
    assert math.log(z) <= snapshot.log_partition_upper+1e-12
    assert 1-kept/z <= snapshot.product_omitted_mass_upper+1e-12
    actions, bound = [], 0.
    for c in snapshot.components:
        decoded = decode_identity_roots(c.component.factors, c.posterior, expansion_budget=1000)
        actions.append(decoded['parents'] or (-1,)*len(c.component.indices))
        bound += len(c.component.indices)/len(factors.nodes) * (
            decoded['model_regret_upper_estimate'] if decoded['parents'] is not None else 1.)
    action = join_actions(factors, components, actions)
    distribution = [(r, w/z) for r, w in joint.values()]
    regret = risk(factors.roots(action), distribution, factors) - min(
        risk(r, distribution, factors) for r, _ in joint.values())
    assert regret <= bound + 1e-12
    assert snapshot.expansions <= seed


def test_refinement_chunks_preserve_monolithic_commit_without_intermediate_history():
    f = scene(size=5)
    normal = RecoverableForestBank(f, active_limit=2)
    expected = normal.advance(decision_us=10, expansion_budget=8)
    staged = RecoverableForestBank(f, active_limit=2)
    context = staged._begin_update(f, 10)
    total, limited = 0, False
    for _ in range(8):
        count, stopped = staged._refine(1)
        total += count
        limited |= stopped
        assert staged.commits == ()
    actual = staged._finish_update(context, 10, total, limited)
    assert actual == expected and len(staged.commits) == 1


def test_append_keeps_existing_component_frontier_and_identity_lineage():
    f = scene(size=4)
    extended = scene(size=6)
    bank = ComponentForestBank(active_limit=2)
    old = bank.advance(factors=f, decision_us=10, expansion_budget=10, message_id='first')
    new = bank.advance(factors=extended, decision_us=11, expansion_budget=10, message_id='next')
    for a, b in zip(old.components, new.components):
        assert a.component.component_id == b.component.component_id
        assert b.predecessors == ((a.component.component_id, a.posterior.commit),)
        assert b.posterior.previous_commit == a.posterior.commit
        assert not b.merge_restart
    validate_component_snapshot(extended, new)


def test_bridge_rebuilds_full_support_not_only_previous_retained_cross_product():
    f = scene(size=4)
    bank = ComponentForestBank(active_limit=1)
    before = bank.advance(factors=f, decision_us=10, expansion_budget=3, message_id='before')
    old_hash = digest(asdict(before))
    nodes = f.nodes + (IdentityNode('bridge', 0, 11, 11, 'bridge'),)
    # A low-weight bridge edge still connects the support components.
    merged = ForestFactors(nodes, f.rows + (((-1, 0.), (2, -900.), (3, 2.)),))
    after = bank.advance(factors=merged, decision_us=11, expansion_budget=1, message_id='bridge')
    assert len(after.components) == 1 and after.components[0].merge_restart
    assert len(after.components[0].predecessors) == 2
    c = after.components[0]
    for action in oracle(merged):
        assert sum(action[:len(p)] == p for p in c.posterior.frontier+c.posterior.active) == 1
    assert digest(asdict(before)) == old_hash
    assert after.expansions == 1 and after.product_omitted_mass_upper == 1.
    validate_component_snapshot(merged, after)


def test_shared_storage_caps_do_not_drop_unexpanded_support():
    f = scene(size=6)
    bank = ComponentForestBank(active_limit=1, max_total_frontier=2, max_total_discovered=1)
    result = bank.advance(factors=f, decision_us=10, expansion_budget=1000, message_id='bounded')
    assert result.frontier_count <= 2 and result.discovered_count <= 1
    assert any(c.posterior.resource_limited for c in result.components)
    assert len(result.allocation_trace) < 1000
    validate_component_snapshot(f, result)


def test_rejected_large_merge_is_atomic_and_does_not_cut_candidate_edges():
    f = scene(size=4)
    bank = ComponentForestBank(max_component_nodes=2)
    first = bank.advance(factors=f, decision_us=10, expansion_budget=10, message_id='first')
    merged = ForestFactors(f.nodes+(IdentityNode('bridge', 1, 11, 11, 'next'),),
                           f.rows+(((-1, 0.), (2, 0.), (3, 0.)),))
    with pytest.raises(ValueError, match='capacity'):
        bank.advance(factors=merged, decision_us=11, expansion_budget=10, message_id='bad')
    assert bank.commits == (first,) and bank.factors == f
    assert len(bank.banks) == 2 and 'bad' not in bank.messages


def test_future_changed_support_and_conflicting_retries_are_atomic():
    f = scene(size=4)
    bank = ComponentForestBank()
    first = bank.advance(factors=f, decision_us=10, expansion_budget=10, message_id='first')
    assert bank.advance(factors=f, decision_us=10, expansion_budget=10, message_id='first') is first
    for candidate, decision, budget, name in [
        (f, 11, 10, 'first'), (f, 2, 10, 'future'),
        (ForestFactors(f.nodes, f.rows[:2]+(((-1, 0.),),)+f.rows[3:]), 11, 10, 'cut'),
    ]:
        with pytest.raises(ValueError):
            bank.advance(factors=candidate, decision_us=decision, expansion_budget=budget, message_id=name)
        assert bank.commits == (first,) and bank.factors == f


def test_empty_components_and_snapshot_tampering():
    f = ForestFactors((), ())
    bank = ComponentForestBank()
    snapshot = bank.advance(factors=f, decision_us=0, expansion_budget=10, message_id='empty')
    assert snapshot.expansions == 0 and snapshot.product_omitted_mass_upper == 0.
    assert join_actions(f, (), ()) == ()
    validate_component_snapshot(f, snapshot)
    bad = replace(snapshot, log_partition_upper=1.)
    payload = asdict(bad)
    payload.pop('commit')
    bad = replace(bad, commit=digest(payload))
    with pytest.raises(ValueError, match='mass aggregation'):
        validate_component_snapshot(f, bad)


def test_risk_allocation_spends_one_commit_per_component_per_decision():
    f = scene(size=6)
    bank = ComponentForestBank(active_limit=1, max_commits=2)
    one = bank.advance(factors=f, decision_us=10, expansion_budget=100, message_id='one')
    assert all(len(b.commits) == 1 for b in bank.banks.values())
    two = bank.advance(factors=f, decision_us=11, expansion_budget=100, message_id='two')
    assert all(len(b.commits) == 2 for b in bank.banks.values())
    assert two.previous_commit == one.commit
    with pytest.raises(ValueError, match='capacity'):
        bank.advance(factors=f, decision_us=12, expansion_budget=100, message_id='three')


def test_transaction_forks_do_not_copy_immutable_history_or_share_mutable_frontiers():
    f = scene(size=4)
    bank = ComponentForestBank(active_limit=1)
    first = bank.advance(factors=f, decision_us=10, expansion_budget=1, message_id='first')
    original = {k: (set(b.frontier), set(b.active), set(b.discovered), b.commits) for k, b in bank.banks.items()}
    fork = bank.fork()
    assert fork.commits is bank.commits
    fork.advance(factors=f, decision_us=11, expansion_budget=100, message_id='next')
    assert bank.commits == (first,) and 'next' not in bank.messages
    for key, b in bank.banks.items():
        assert (b.frontier, b.active, b.discovered, b.commits) == original[key]
        assert fork.banks[key].commits[0] is b.commits[0]
        assert fork.banks[key].frontier is not b.frontier
        assert fork.banks[key].discovered is not b.discovered


def test_frontier_pressure_coarsens_to_reconstructable_region_without_cutting_support():
    f = dense_factors(size=3)
    bank = ComponentForestBank(active_limit=1, max_frontier=2)
    first = bank.advance(factors=f, decision_us=10, expansion_budget=100, message_id='first')
    old_bank = next(iter(bank.banks.values()))
    assert len(old_bank.frontier | old_bank.active) > old_bank.max_frontier
    extended = dense_factors(size=4)
    after = bank.advance(factors=extended, decision_us=11, expansion_budget=0, message_id='append')
    c = after.components[0]
    assert c.restart_reason == 'append_frontier_pressure' and not c.merge_restart
    assert c.posterior.frontier == ((),) and c.posterior.eta_upper == 1.
    assert c.predecessors == ((first.components[0].component.component_id, first.components[0].posterior.commit),)
    for action in oracle(extended):
        assert sum(action[:len(p)] == p for p in c.posterior.active+c.posterior.frontier) == 1
    validate_component_snapshot(extended, after)
