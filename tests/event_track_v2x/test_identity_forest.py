"""Small Cartesian reference independent of the production prefix inference."""
from dataclasses import asdict, replace
from itertools import product
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.identity_forest import (
    ForestFactors, IdentityNode, RecoverableForestBank, decode_identity_roots,
    digest, validate_snapshot,
)


def factors(seed=0, size=5):
    rng = np.random.default_rng(seed)
    nodes = tuple(IdentityNode(str(i), i % 2, i // 2, i, str(i // 3)) for i in range(size))
    rows = tuple(tuple((p, float(rng.normal())) for p in range(-1, i)) for i in range(size))
    return ForestFactors(nodes, rows)


def oracle(f):
    result = {}
    for action in product(*[[p for p, _ in row] for row in f.rows]):
        roots, slots, valid = [], set(), True
        for i, p in enumerate(action):
            root = i if p == -1 else roots[p]
            roots.append(root)
            n = f.nodes[i]
            slot = (root, n.source_id, n.frame_id)
            if n.source_id != -1 and slot in slots:
                valid = False
            slots.add(slot)
        if valid:
            result[action] = (tuple(roots), math.exp(sum(dict(row)[p] for row, p in zip(f.rows, action))))
    return result


def risk(action_roots, distribution, f):
    indices = [i for i, node in enumerate(f.nodes) if node.source_id != -1]
    return sum(prob * sum(action_roots[i] != roots[i] for i in indices) / len(indices)
               for roots, prob in distribution) if indices else 0.


@pytest.mark.parametrize('seed', range(24))
def test_exact_cover_mass_and_regret_against_independent_enumeration(seed):
    f = factors(seed, seed % 6)
    truth = oracle(f)
    bank = RecoverableForestBank(f, active_limit=1 + seed % 4)
    snapshot = bank.advance(decision_us=100, expansion_budget=seed % 15)
    validate_snapshot(f, snapshot)
    for action in truth:
        assert sum(action[:len(p)] == p for p in snapshot.frontier + snapshot.active) == 1
    partition = sum(weight for _, weight in truth.values())
    kept = sum(truth[p][1] for p in snapshot.active)
    assert math.log(partition) <= snapshot.log_partition_upper + 1e-12
    assert 1. - kept / partition <= snapshot.eta_upper + 1e-12
    answer = decode_identity_roots(f, snapshot, expansion_budget=10000)
    if not snapshot.active:
        assert answer['status'] == 'unresolved'
        assert answer['model_regret_upper_estimate'] is None
        return
    q = [(truth[p][0], truth[p][1] / kept) for p in snapshot.active]
    pi = [(roots, weight / partition) for roots, weight in truth.values()]
    action = answer['roots']
    assert risk(action, q, f) == pytest.approx(min(risk(r, q, f) for r, _ in truth.values()), abs=1e-12)
    assert answer['action_search_gap'] == pytest.approx(0., abs=1e-12)
    regret = risk(action, pi, f) - min(risk(r, pi, f) for r, _ in truth.values())
    assert regret <= answer['model_regret_upper_estimate'] + 1e-12
    # Tiny decoder budgets still retain a valid upper bound on action regret.
    bounded = decode_identity_roots(f, snapshot, expansion_budget=1, max_frontier=1)
    assert risk(bounded['roots'], pi, f) - min(risk(r, pi, f) for r, _ in truth.values()) <= (
        bounded['model_regret_upper_estimate'] + 1e-12)


def test_global_collision_across_cross_source_and_temporal_edges():
    f = factors(size=3)
    # Nodes 0 and 2 share source/frame. Linking through the other source does
    # not allow two same-frame observations to acquire the same identity.
    with pytest.raises(ValueError, match='collision'):
        f.roots((-1, 0, 1))
    assert f.roots((-1, 0, -1)) == (0, 0, 2)


def test_new_evidence_recovers_a_never_enumerated_history():
    f = ForestFactors(factors(size=3).nodes,
                      (((-1, 0.),), ((-1, 8.), (0, -8.)), ((-1, 0.), (1, -1.))))
    bank = RecoverableForestBank(f, active_limit=1)
    first = bank.advance(decision_us=10, expansion_budget=3, message_id='first')
    target = (-1, 0, -1)
    assert target not in bank.discovered
    new = ForestFactors(f.nodes, (f.rows[0], ((-1, -8.), (0, 8.)), f.rows[2]))
    second = bank.advance(decision_us=11, expansion_budget=2, factors=new, message_id='late')
    assert target in second.active and target in second.never_enumerated_recoveries
    assert second.previous_commit == first.commit
    assert bank.advance(decision_us=10, expansion_budget=999, factors=f, message_id='first') is first
    assert bank.commits == (first, second)
    validate_snapshot(new, second)


def test_append_preserves_old_history_support_and_committed_output():
    f = factors(size=3)
    initial = ForestFactors(f.nodes[:2], f.rows[:2])
    bank = RecoverableForestBank(initial, active_limit=2)
    first = bank.advance(decision_us=1, expansion_budget=10)
    first_digest = digest(asdict(first))
    second = bank.advance(decision_us=2, expansion_budget=10, factors=f)
    validate_snapshot(f, second)
    assert not second.never_enumerated_recoveries
    assert digest(asdict(first)) == first_digest
    assert all(sum(p[:len(prefix)] == prefix for prefix in second.frontier + second.active) == 1
               for p in oracle(f))


@pytest.mark.parametrize('limit', [{'max_frontier': 1}, {'max_discovered': 1}])
def test_exhausted_storage_preserves_an_unexpanded_region(limit):
    f = factors(size=4)
    bank = RecoverableForestBank(f, active_limit=1, **limit)
    s = bank.advance(decision_us=10, expansion_budget=1000)
    assert s.resource_limited
    assert len(s.frontier) <= bank.max_frontier
    assert len(bank.discovered) <= bank.max_discovered
    validate_snapshot(f, s)


def test_rejected_updates_do_not_mutate_bank():
    f = factors(size=2)
    bank = RecoverableForestBank(f, max_commits=2)
    first = bank.advance(decision_us=10, expansion_budget=1, message_id='first')
    different_support = ForestFactors(f.nodes, (f.rows[0], ((-1, 0.),)))
    changed = ForestFactors(f.nodes, (f.rows[0], ((-1, 1.), (0, 2.))))
    future = ForestFactors(f.nodes + (IdentityNode('late', 1, 11, 20, 'later'),),
                           f.rows + (((-1, 0.),),))
    for args in ({'factors': different_support}, {'factors': future},
                 {'factors': changed, 'message_id': 'first'}, {'decision_us': 9}):
        with pytest.raises(ValueError):
            bank.advance(**dict(decision_us=10, expansion_budget=2, **{}) | args)
        assert bank.commits == (first,) and bank.factors == f
    second = bank.advance(decision_us=11, expansion_budget=1)
    with pytest.raises(ValueError, match='storage'):
        bank.advance(decision_us=12, expansion_budget=1)
    assert bank.commits == (first, second)


def test_append_resource_rejection_precedes_mutation():
    f = factors(size=2)
    bank = RecoverableForestBank(f, active_limit=1, max_frontier=2)
    s = bank.advance(decision_us=3, expansion_budget=3)
    # One retained and one omitted leaf need two frontier slots on append.
    bank.max_frontier = 1
    extended = factors(size=3)
    with pytest.raises(ValueError, match='append frontier'):
        bank.advance(decision_us=3, expansion_budget=2, factors=extended)
    assert bank.commits == (s,) and bank.factors == f


def test_snapshot_cover_and_mass_tampering_fail_closed():
    f = factors(size=4)
    s = RecoverableForestBank(f, active_limit=2).advance(decision_us=10, expansion_budget=100)
    for updates in ({'eta_upper': .123}, {'active': s.active * 2},
                    {'frontier': s.frontier[1:]}, {'frontier': s.frontier + ((),)},
                    {'active_log_weights': tuple(0. for _ in s.active)}):
        changed = replace(s, **updates)
        payload = asdict(changed)
        payload.pop('commit')
        changed = replace(changed, commit=digest(payload))
        with pytest.raises(ValueError):
            decode_identity_roots(f, changed)


def test_anchor_identity_and_empty_scene():
    anchor = IdentityNode('carry', -1, 0, 0, 'carry')
    node = IdentityNode('car', 0, 1, 1, 'frame')
    f = ForestFactors((anchor, node), (((-1, 0.),), ((-1, -10.), (0, 10.))))
    s = RecoverableForestBank(f, active_limit=2).advance(decision_us=1, expansion_budget=10)
    d = decode_identity_roots(f, s)
    assert d['roots'] == (0, 0)
    assert d['conditional_expected_loss'] == pytest.approx(1. / (1. + math.exp(20)), abs=1e-12)
    empty = ForestFactors((), ())
    s = RecoverableForestBank(empty).advance(decision_us=0, expansion_budget=0)
    assert decode_identity_roots(empty, s)['model_regret_upper_estimate'] == 0.


def test_factor_inputs_are_finite_immutable_and_causally_ordered():
    f = factors(size=2)
    for rows in ((((0, 0.),), f.rows[1]), (f.rows[0], ((-1, math.nan),)),
                 (f.rows[0], ((-1, 1001.),)), (f.rows[0], ((-1, 0.), (-1, 1.)))):
        with pytest.raises(ValueError):
            ForestFactors(f.nodes, rows)
    with pytest.raises(ValueError, match='arrival order'):
        ForestFactors(f.nodes[::-1], f.rows)
    with pytest.raises(ValueError):
        IdentityNode('future', 0, 10, 9, 'f')
