"""History pruning must survive rescore, component merge and legal decoding."""
import itertools
import math
from dataclasses import asdict

import numpy as np
import pytest

from transvision.models.event_track_v2x.forest_tracking import (
    PaperForestTrackingConfig as ForestTrackingConfig, RawIdentityDetection,
)
from transvision.models.event_track_v2x.identity_forest import ForestFactors, IdentityNode
from transvision.models.event_track_v2x.exclusive_completion_tracking import (
    ExclusiveCompletionTracker, PersistentExclusiveCompletionConfig,
)
from transvision.models.event_track_v2x.recovery_off_tracking import (
    HistoryRestriction, RestrictedFactors, RecoveryOffConfig, RecoveryOffTracker,
)


def obs(name, *, source=0, time=1_000_000, frame=None, arrival=None, index=0):
    features = np.zeros(203)
    features[138], features[200], features[201] = 1, .8, .8
    return RawIdentityDetection('0003', IdentityNode(name, source, time,
        time + 10 if arrival is None else arrival, frame or name), index, time,
        [0., 0., 1., 4., 2., 1.5, 0., 0., 0.], np.eye(9)*.2, .8, features, 'a'*64)


def tracker(path, *, disabled=True, budget=100, width=1):
    config_type = RecoveryOffConfig if disabled else PersistentExclusiveCompletionConfig
    cls = RecoveryOffTracker if disabled else ExclusiveCompletionTracker
    return cls(path, sequence_id='0003', config=config_type(state=ForestTrackingConfig(
        active_limit=width, expansion_budget=budget, decision_mode='all-legal-hamming', max_model_regret=1.)))


def step(t, raw=(), rows=(), *, time=1_100_000, event='first', **kwargs):
    return t.step(raw, rows, frame_id=event, event_id=event,
                  reference_us=time, decision_us=time, **kwargs)


def global_roots(t, result):
    roots = [-1]*t.n
    for s in result.audit['components']:
        k = t.kernels[s['component']]
        members = t.store.members(s['component'])
        for i, g in enumerate(members):
            roots[g] = members[k._prefix(k.ancestor(s['output_handle'], i+1)).root]
    return tuple(roots)


@pytest.mark.parametrize('clauses', [
    [((0, 1), ())], [((1, 0), ((0, 0),))], [((0, 1), ((0,),))],
    [((0, 1), ((0, 2),))], [((0, 1), ((1, 0),))],
    [((0,), ((0,),)), ((0,), ((0,),))],
])
def test_invalid_restrictions_are_rejected(clauses):
    with pytest.raises(ValueError):
        HistoryRestriction(clauses)


def test_whole_class_correlation_is_preserved_not_cartesian_per_node_roots():
    r = HistoryRestriction([((0, 1, 2), ((0, 0, 2), (0, 1, 1)))])
    assert r.allows((0,)) and r.allows((0, 0)) and r.allows((0, 1))
    assert r.allows((0, 0, 2)) and r.allows((0, 1, 1))
    assert not r.allows((0, 0, 1)) and not r.allows((0, 1, 2))


def test_restricted_decoder_children_match_independent_full_enumeration():
    nodes = tuple(obs(str(i), source=i%2).node for i in range(5))
    rows = tuple(tuple((j, 0.) for j in range(-1, i)) for i in range(5))
    factors = ForestFactors(nodes, rows)
    r = HistoryRestriction([((0, 2), ((0, 0),)), ((1, 3), ((1, 3),))])
    view = RestrictedFactors(factors, r)
    expected = set()
    for p in itertools.product(*(tuple(x for x, _ in row) for row in rows)):
        roots = factors.roots(p)
        if roots[0] == 0 and roots[2] == 0 and roots[1] == 1 and roots[3] == 3:
            expected.add(p)
    found, todo = set(), [()]
    while todo:
        p = todo.pop()
        if len(p) == len(nodes): found.add(p)
        else: todo.extend(view.children(p))
    assert found == expected and len(found) > 1
    assert (-1, -1, 0, -1, 0) in found and (-1, -1, 0, -1, 2) in found
    with pytest.raises(ValueError, match='recover'):
        view.roots((-1, -1, -1))


def test_rescore_and_repeated_compute_cannot_restore_pruned_identity(tmp_path):
    raw = (obs('a'), obs('b', source=1))
    rows = (((-1, 0.),), ((-1, 0.), (0, -8.)))
    on, off = tracker(tmp_path/'on.db', disabled=False), tracker(tmp_path/'off.db')
    try:
        initial_on, initial_off = step(on, raw, rows), step(off, raw, rows)
        assert initial_on.prediction_json == initial_off.prediction_json
        assert global_roots(off, initial_off) == (0, 1)
        saved = initial_off.prediction_json
        update = [(1, ((-1, -15.), (0, 15.)))]
        next_on = step(on, time=1_200_000, event='rescore', rescored_rows=update)
        next_off = step(off, time=1_200_000, event='rescore', rescored_rows=update)
        assert global_roots(on, next_on) == (0, 0)
        assert global_roots(off, next_off) == (0, 1)
        assert next_on.audit['factor_rows_sha256'] == next_off.audit['factor_rows_sha256']
        for i in range(3):
            result = step(off, time=1_300_000+i, event='compute-'+str(i))
            assert global_roots(off, result) == (0, 1)
            assert all(not s['recovery_events'] for s in result.audit['components'])
        assert step(off, raw, rows).prediction_json == saved
        assert next_off.audit['model_regret_upper'] == 1.
        assert next_off.audit['original_model_risk_certified'] is False
    finally:
        on.close(); off.close()


@pytest.mark.parametrize('reopen', [False, True])
def test_bridge_late_arrival_and_restart_do_not_erase_predecessor_pruning(tmp_path, reopen):
    p = tmp_path/'merge.db'; t = tracker(p)
    raw = (obs('a'), obs('separate', source=1), obs('c', time=1_010_000))
    rows = (((-1, 0.),), ((-1, 0.),), ((-1, 0.), (0, -10.)))
    first = step(t, raw, rows)
    assert global_roots(t, first) == (0, 1, 2)
    if reopen:
        file_sha = t.close()
        t = RecoveryOffTracker.open(p, expected_database_sha256=file_sha,
                                    expected_prediction_sha256=first.prediction['commit_sha256'])
    try:
        bridge = obs('bridge', source=1, time=900_000, arrival=1_200_000)
        second = step(t, [bridge], [((-1, -3.), (1, 1.), (2, 1.))],
                      time=1_300_000, event='merge', rescored_rows=[(2, ((-1, -20.), (0, 20.)))])
        assert len(second.audit['components']) == 1
        assert second.audit['components'][0]['merge_restart']
        assert global_roots(t, second)[:3] == (0, 1, 2)
        assert len(second.audit['components'][0]['historical_support']['clauses']) == 2
        assert step(t, raw, rows).prediction_json == first.prediction_json
        final = step(t, time=1_400_000, event='compute')
        assert global_roots(t, final)[:3] == (0, 1, 2)
    finally:
        t.close()


def test_materializing_old_archived_prefix_cannot_bypass_restriction(tmp_path):
    t = tracker(tmp_path/'materialize.db')
    try:
        step(t, [obs('a'), obs('b', source=1)], [((-1, 0.),), ((-1, 0.), (0, -8.))])
        k = next(iter(t.kernels.values()))
        old_rejected = k._child(k._child(0, -1), 0)
        step(t, [obs('later', time=1_200_000)], [((-1, 0.), (0, 1.))], time=1_300_000, event='next')
        k = next(iter(t.kernels.values()))
        prefix = k._child(0, -1)
        with pytest.raises(ValueError, match='pruned'):
            k._child(prefix, 0)
        with pytest.raises(ValueError, match='archived'):
            k._child(old_rejected, -1)
        with pytest.raises(ValueError, match='archived'):
            k._choices(old_rejected)
    finally:
        t.close()


def test_failed_merge_is_transactional_and_duplicate_event_stays_immutable(tmp_path, monkeypatch):
    t = tracker(tmp_path/'rollback.db')
    try:
        first = step(t, [obs('a'), obs('b', source=1)], [((-1, 0.),), ((-1, 0.),)])
        before = tuple(t.db.iterdump())
        original = t.store.save_kernel
        def fail(*a, **kw): raise RuntimeError('injected save failure')
        monkeypatch.setattr(t.store, 'save_kernel', fail)
        bridge = [obs('bridge', time=1_200_000)]
        rows = [((-1, 0.), (0, 1.), (1, 1.))]
        with pytest.raises(RuntimeError, match='injected'):
            step(t, bridge, rows, time=1_300_000, event='merge')
        assert tuple(t.db.iterdump()) == before
        monkeypatch.setattr(t.store, 'save_kernel', original)
        second = step(t, bridge, rows, time=1_300_000, event='merge')
        assert second.prediction['previous_commit_sha256'] == first.prediction['commit_sha256']
    finally:
        t.close()


def test_configuration_changes_only_the_explicit_ablation_policy():
    state = ForestTrackingConfig(decision_mode='all-legal-hamming')
    old = asdict(PersistentExclusiveCompletionConfig(state=state))
    new = asdict(RecoveryOffConfig(state=state))
    assert new.pop('recovery_off_version') == 1
    assert old == new


def test_still_explicit_alternative_can_change_rank_after_rescore(tmp_path):
    t = tracker(tmp_path/'retained.db', width=2)
    try:
        first = step(t, [obs('a'), obs('b', source=1)],
                     [((-1, 0.),), ((-1, 0.), (0, -8.))])
        assert len(first.audit['components'][0]['active']) == 2
        second = step(t, time=1_200_000, event='rescore',
                      rescored_rows=[(1, ((-1, -15.), (0, 15.)))])
        assert global_roots(t, second) == (0, 0)
        assert not second.audit['components'][0]['recovery_events']
    finally:
        t.close()


@pytest.mark.parametrize('budget', [0, 1])
def test_budget_exhaustion_cannot_enable_recovery(tmp_path, budget):
    t = tracker(tmp_path/'bounded.db', budget=budget)
    try:
        first = step(t, [obs('a'), obs('b', source=1)],
                     [((-1, 0.),), ((-1, 0.), (0, -8.))])
        before = global_roots(t, first)
        second = step(t, time=1_200_000, event='rescore',
                      rescored_rows=[(1, ((-1, -15.), (0, 15.)))])
        assert global_roots(t, second) == before
        assert second.audit['search_steps'] <= budget
    finally:
        t.close()


@pytest.mark.parametrize('seed', [1337, 2027, 3407])
def test_merged_class_mass_matches_independent_parent_path_enumeration(tmp_path, seed):
    rng = np.random.default_rng(seed)
    t = tracker(tmp_path/'mass.db', width=2)
    raw = tuple(obs(str(i), source=i%2, time=1_000_000+i) for i in range(3))
    rows = [((-1, 0.),), ((-1, .2), (0, -.3)), ((-1, 0.),)]
    try:
        first = step(t, raw, rows)
        # Capture actual retained/output classes from the prior committed bank,
        # then enumerate raw parent assignments without the candidate restriction.
        allowed = []
        for s in first.audit['components']:
            members = t.store.members(s['component']); k = t.kernels[s['component']]
            roots = {tuple(members[k._prefix(k.ancestor(h, i+1)).root] for i in range(k.n))
                     for h in {a['handle'] for a in s['active']} | {s['output_handle']}}
            allowed.append((members, roots))
        additions = (obs('3', source=1, time=1_200_000), obs('4', time=1_200_001))
        new_rows = tuple(tuple((p, float(rng.normal())) for p in candidates)
                         for candidates in [(-1, 0, 2), (-1, 1, 3)])
        result = step(t, additions, new_rows, time=1_300_000, event='merge')
        factors = ForestFactors(tuple(o.node for o in raw+additions), tuple(rows)+new_rows)
        masses = {}
        for path in itertools.product(*(tuple(p for p, _ in row) for row in factors.rows)):
            roots = factors.roots(path)
            if any(tuple(roots[i] for i in members) not in variants for members, variants in allowed):
                continue
            w = math.exp(math.fsum(dict(row)[p] for row, p in zip(factors.rows, path)))
            masses[roots] = masses.get(roots, 0.) + w
        assert len(result.audit['components']) == 1
        s = result.audit['components'][0]; k = t.kernels[s['component']]
        for leaf in s['active']:
            roots = tuple(k._prefix(k.ancestor(leaf['handle'], i+1)).root for i in range(k.n))
            assert math.exp(leaf['log_weight']) == pytest.approx(masses[roots], rel=1e-12, abs=1e-12)
        assert global_roots(t, result) in masses
        assert sum(masses.values()) <= math.exp(s['log_partition_upper'])+1e-10
        assert len(t.shared_cache) <= t.config.max_total_cache_entries
    finally:
        t.close()
