from dataclasses import asdict
import hashlib
import json
import math
import sqlite3

import pytest

from transvision.models.event_track_v2x.detection_cache_v2 import canonical
from transvision.models.event_track_v2x.forest_components import split_forest
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig, replay_forest_states
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from transvision.models.event_track_v2x.persistent_component_store import PersistentComponentStore
from transvision.models.event_track_v2x.persistent_forest import PersistentForestConfig, PersistentForestTracker
from test_forest_tracking import observation
from test_persistent_forest import brute


@pytest.fixture
def store(tmp_path):
    owner = PersistentForestTracker(tmp_path/'shared.sqlite', sequence_id='0003')
    result = PersistentComponentStore(owner)
    owner.db.execute('BEGIN IMMEDIATE')
    result.initialize()
    owner.db.execute('COMMIT')
    yield result
    owner.close()


def insert(store, raw, rows):
    start = store.db.execute('SELECT count(*) FROM observations').fetchone()[0]
    for i, (observation, row) in enumerate(zip(raw, rows), start):
        payload = canonical(asdict(observation))
        store.db.execute('INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?)',
            (i, observation.node.node_id, observation.node.source_id, observation.node.frame_id,
             observation.detection_index, observation.state_us, observation.node.arrival_us, observation.score,
             payload, hashlib.sha256(payload).hexdigest()))
        store.db.executemany('INSERT INTO potentials VALUES(?,?,?)', ((i, p, w) for p, w in row))
    return start


def kernel_config(active_limit=100):
    return PersistentForestConfig(state=ForestTrackingConfig(active_limit=active_limit, expansion_budget=100, max_model_regret=1.))


def finish(store, kernel, context, time=1_100_000):
    kernel._refine(time)
    active, weights, retained, upper, eta = kernel._mass()
    chosen, decision = kernel._decode(active, weights, retained, eta, tuple(range(kernel.n)), context['fallback'])
    result = kernel._predict(chosen, time)
    store.save_kernel(kernel, chosen=chosen, reference_us=time, decision_us=time)
    return result, upper, eta


def test_shared_raw_views_factorize_exactly_and_kernels_preserve_state_replay(store):
    raw = tuple(observation(str(i), source=i % 2, frame=str(i), state_us=1_000_000+i) for i in range(4))
    rows = (((-1, 0.),), ((-1, -1.),), ((-1, -2.), (0, 2.)), ((-1, -3.), (1, 3.)))
    store.db.execute('BEGIN IMMEDIATE')
    start = insert(store, raw, rows)
    changes = store.append_partition(start, decision_us=1_100_000)
    factors = ForestFactors(tuple(o.node for o in raw), rows)
    assert {store.members(c.component) for c in changes} == {c.indices for c in split_forest(factors)}
    partitions = []
    for change in changes:
        kernel = store.create_kernel(change.component, kernel_config())
        context = store.activate_kernel(kernel, decision_us=1_100_000)
        result, upper, eta = finish(store, kernel, context)
        component_raw = tuple(raw[i] for i in store.members(change.component))
        component_factors = ForestFactors(tuple(o.node for o in component_raw), tuple(kernel._row(i) for i in range(kernel.n)))
        _, _, partition = brute(component_factors)
        partitions.append(partition)
        assert math.exp(upper) == pytest.approx(partition) and eta == 0
        expected, _ = replay_forest_states('0003', component_raw, component_factors, kernel.parents(kernel.meta['output']),
                                          1_100_000, kernel.config.state)
        assert result == [{k: t[k] for k in ('track_id', 'class_label', 'mean', 'covariance', 'score')} for t in expected]
        with pytest.raises(sqlite3.OperationalError):
            kernel.db.execute('DELETE FROM observations')
        with pytest.raises(ValueError, match='cannot control'):
            kernel.db.execute('COMMIT')
    assert math.prod(partitions) == pytest.approx(brute(factors)[2])
    assert store.db.execute('SELECT count(*) FROM observations').fetchone()[0] == 4
    store.db.execute('COMMIT')


def test_bridge_preserves_unexpanded_support_prior_identity_and_single_raw_copy(store):
    initial = (observation('a', source=0, frame='a'), observation('b', source=1, frame='b'),
               observation('c', source=1, frame='c', state_us=1_010_000))
    first_rows = (((-1, 0.),), ((-1, 0.),), ((-1, -6.), (0, 0.)))
    config = kernel_config(active_limit=1)
    store.db.execute('BEGIN IMMEDIATE')
    insert(store, initial, first_rows)
    first = store.append_partition(0, decision_us=1_100_000)
    old_ids = set()
    for change in first:
        k = store.create_kernel(change.component, config)
        context = store.activate_kernel(k, decision_us=1_100_000)
        predictions, _, _ = finish(store, k, context)
        old_ids.update(p['track_id'] for p in predictions)
    store.db.execute('COMMIT')
    old_catalog = store.live()
    before_tables = set(r[0] for r in store.db.execute("SELECT name FROM sqlite_master WHERE type='table'"))
    store.db.execute('BEGIN IMMEDIATE')
    bridge = observation('bridge', source=0, frame='bridge', state_us=1_200_000)
    last = insert(store, [bridge], [((-1, -2.), (1, 0.), (2, 1.))])
    changes = store.append_partition(last, decision_us=1_300_000)
    assert len(changes) == 1 and changes[0].merge_restart
    new = changes[0]
    assert new.predecessors == old_catalog
    fallback = store.merged_fallback(new.component, new.predecessors)
    assert fallback == (-1, -1, 0, -1)
    kernel = store.create_kernel(new.component, config)
    context = store.activate_kernel(kernel, decision_us=1_300_000, fallback_parents=fallback)
    assert kernel.parents(context['fallback']) == fallback
    assert old_ids <= {p['track_id'] for p in kernel._predict(context['fallback'], 1_300_000)}
    # Fresh merged frontier covers EVERYTHING, including unretained births of c.
    assert kernel.frontier == {0} and not kernel.active
    assert kernel._row(2) == ((-1, -6.), (0, 0.))
    assert store.db.execute('SELECT count(*) FROM observations').fetchone()[0] == 4
    assert store.db.execute('SELECT count(*) FROM component_members').fetchone()[0] == 7
    assert before_tables <= set(r[0] for r in store.db.execute("SELECT name FROM sqlite_master WHERE type='table'"))
    store.db.execute('COMMIT')


def test_transaction_rolls_back_component_ddl_prefixes_maps_and_shared_observations(store):
    before = tuple(store.db.iterdump())
    store.db.execute('BEGIN IMMEDIATE')
    insert(store, [observation('a')], [((-1, 0.),)])
    change, = store.append_partition(0, decision_us=1_100_000)
    kernel = store.create_kernel(change.component, kernel_config())
    context = store.activate_kernel(kernel, decision_us=1_100_000)
    finish(store, kernel, context)
    store.db.execute('ROLLBACK')
    assert tuple(store.db.iterdump()) == before


def test_weak_candidate_edge_still_merges_components_and_resource_caps_do_not_prune(store):
    store.db.execute('BEGIN IMMEDIATE')
    insert(store, [observation('a'), observation('b', source=1)], [((-1, 0.),), ((-1, 0.), (0, -999.))])
    changes = store.append_partition(0, decision_us=1_100_000)
    assert len(changes) == 1 and store.members(changes[0].component) == (0, 1)
    store.db.execute('COMMIT')
    with pytest.raises(ValueError, match='shared transaction'):
        store.append_partition(2, decision_us=1_100_000)


def test_shared_one_step_refinement_cannot_exceed_caps(store):
    store.db.execute('BEGIN IMMEDIATE')
    insert(store, [observation('a'), observation('b', source=1)], [((-1, 0.),), ((-1, -1.), (0, 1.))])
    change, = store.append_partition(0, decision_us=1_100_000)
    kernel = store.create_kernel(change.component, kernel_config())
    context = store.activate_kernel(kernel, decision_us=1_100_000)
    assert kernel._refine(1_100_000, budget=1)[0] == 1
    count = kernel.prefix_count
    assert kernel._refine(1_100_000, budget=100, prefix_cap=count) == (0, True)
    assert kernel.prefix_count == count
    kernel._refine(1_100_000)
    active, weights, retained, upper, eta = kernel._mass()
    chosen, decision = kernel._decode(active, weights, retained, eta, (0, 1), context['fallback'], force_fallback=True)
    assert chosen == context['fallback'] and decision['conditional_risk'] > 0
    store.db.execute('ROLLBACK')


def test_equivalent_parent_paths_sum_to_one_root_class_and_rescore_exactly(store):
    store.db.execute('BEGIN IMMEDIATE')
    raw = tuple(observation(str(i), source=0, frame=str(i), state_us=1_000_000+i) for i in range(3))
    rows = (((-1, 0.),), ((-1, 0.), (0, 0.)), ((-1, 0.), (0, 0.), (1, 0.)))
    insert(store, raw, rows)
    change, = store.append_partition(0, decision_us=1_100_000)
    kernel = store.create_kernel(change.component, kernel_config())
    context = store.activate_kernel(kernel, decision_us=1_100_000)
    kernel.seed_complete_action(context['fallback'], 1_100_000)
    kernel._refine(1_100_000)
    active, weights, retained, upper, eta = kernel._mass()
    assert len(active) == 5  # Six parent forests, but only five root partitions.
    assert sorted(math.exp(w) for w in weights) == pytest.approx([1, 1, 1, 1, 2])
    assert math.exp(upper) == pytest.approx(6) and eta == 0
    all_same = next(h for h in active if kernel.parents(h) == (-1, 0, 0))
    store.save_kernel(kernel, chosen=all_same, reference_us=1_100_000, decision_us=1_100_000)
    store.db.execute('UPDATE potentials SET w=? WHERE i=2 AND p=0', (math.log(3),))
    store.db.execute('UPDATE potentials SET w=? WHERE i=2 AND p=1', (math.log(5),))
    store.activate_kernel(kernel, decision_us=1_200_000, rescore=True)
    assert math.exp(kernel._weight(all_same)) == pytest.approx(8)
    assert kernel.parents(all_same) == (-1, 0, 0)
    store.db.execute('ROLLBACK')
