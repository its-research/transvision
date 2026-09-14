from dataclasses import replace
import itertools
import math

import numpy as np
import pytest

from tools.event_track_v2x.probe_frontier_completion import compare
from transvision.models.event_track_v2x.completion_component_tracking import (
    CompletionComponentTracker, PersistentCompletionConfig,
)
from transvision.models.event_track_v2x.covered_proposal_capacity import (
    covered_proposal_operation, proposal_frontier_reserve,
)
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.frontier_completion import frontier_completion_operation
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from transvision.models.event_track_v2x.learned_component_allocation import AllocationTeacherTracker
from test_forest_tracking import observation
from test_persistent_component_tracking import step
from test_persistent_forest import brute
from test_learned_component_allocation import scene, config


def inputs(seed):
    raw = (observation('a', source=0, index=0), observation('b', source=0, index=1),
           observation('c', source=1, index=0), observation('d', source=1, index=1))
    rng = np.random.default_rng(seed)
    rows = (((-1, 0.),), ((-1, 0.),),
            tuple((p, float(rng.normal())) for p in (-1, 0, 1)),
            tuple((p, float(rng.normal())) for p in (-1, 0, 1)))
    return raw, rows


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('history_proposal', [False, True])
def test_full_frontier_covered_seed_needs_no_extra_slot_preserves_mass_and_history(tmp_path, seed, history_proposal):
    cfg = PersistentCompletionConfig(state=ForestTrackingConfig(
        active_limit=2, expansion_budget=0, max_frontier=1), max_total_frontier=1)
    tracker = CompletionComponentTracker(tmp_path/'state.sqlite', sequence_id='0003', config=cfg)
    raw, rows = inputs(seed)
    first = step(tracker, raw, rows)
    kernel = next(iter(tracker.kernels.values()))
    counts = dict(prefix_count=tracker._storage_counts()[0], frontier_count=1)
    assert kernel.frontier == {0} and proposal_frontier_reserve(kernel) == 0
    assert frontier_completion_operation(kernel, remaining=4, config=cfg, **counts) is None
    proposals = [kernel.ancestor(next(iter(kernel.active)), 3)] if history_proposal else None
    operation = covered_proposal_operation(kernel, proposals=proposals, remaining=4, config=cfg, **counts)
    assert operation is not None and operation['frontier_reserve'] == 0
    before = kernel._mass()
    tracker.db.execute('SAVEPOINT coverage_admission')
    try:
        done, limited = tracker._execute_allocation_work(kernel, operation, 1_100_000, **counts)
        assert not limited and done == operation['requested_steps'] == (1 if history_proposal else 4)
        assert len(kernel.frontier) <= 1
        after = kernel._mass()
        assert after[2] >= before[2]-1e-12 and after[3] == pytest.approx(before[3])
        assert after[4] <= before[4]+1e-12
        probabilities, roots, z = brute(ForestFactors(tuple(o.node for o in raw), rows))
        explicit = {tuple(kernel._prefix(kernel.ancestor(h, i+1)).root for i in range(kernel.n))
                    for h in kernel.active}
        omitted = 1-math.fsum(p for h, p in probabilities.items() if roots[h] in explicit)
        assert omitted <= after[4]+1e-12 and math.log(z) <= after[3]+1e-12
    finally:
        tracker.db.execute('ROLLBACK TO coverage_admission')
        tracker.db.execute('RELEASE coverage_admission')
        tracker._restore_runtime()
    assert step(tracker, raw, rows).prediction_json == first.prediction_json
    tracker.close()


def test_bound_on_frontier_growth_for_every_small_legal_class_insertion(tmp_path):
    tracker = CompletionComponentTracker(tmp_path/'all.sqlite', sequence_id='0003',
        config=PersistentCompletionConfig(state=ForestTrackingConfig(active_limit=2, expansion_budget=0)))
    raw, rows = inputs(4)
    step(tracker, raw, rows)
    kernel = next(iter(tracker.kernels.values()))
    tracker.db.execute('SAVEPOINT insertions')
    checks = set()
    try:
        leaves = [0]
        for _ in range(kernel.n):
            leaves = [kernel._child(h, p) for h in leaves for p in kernel._choices(h)]
        for active in itertools.combinations(leaves, 2):
            for covered_count in (0, 1, 2):
                for proposed in leaves:
                    kernel.active = set(active)
                    kernel.frontier = set(leaves)-set(active)
                    kernel.frontier.update(active[:covered_count])
                    reserve = proposal_frontier_reserve(kernel)
                    before = len(kernel.frontier)
                    kernel.seed_complete_action(proposed, 1_100_000)
                    assert len(kernel.frontier)-before <= reserve
                    assert kernel.active | kernel.frontier == set(leaves)
                    checks.add(reserve)
        assert checks == {0, 1}
    finally:
        tracker.db.execute('ROLLBACK TO insertions')
        tracker.db.execute('RELEASE insertions')
        tracker._restore_runtime()
    tracker.close()


@pytest.mark.parametrize('override', [dict(remaining=0), dict(remaining=3),
    dict(prefix_count=1_000_000), dict(frontier_count=2)])
def test_zero_extra_slot_does_not_bypass_compute_or_other_storage_limits(tmp_path, override):
    cfg = PersistentCompletionConfig(state=ForestTrackingConfig(expansion_budget=0, max_frontier=1),
                                     max_total_frontier=1)
    tracker = CompletionComponentTracker(tmp_path/'limits.sqlite', sequence_id='0003', config=cfg)
    step(tracker, *inputs(0))
    kernel = next(iter(tracker.kernels.values()))
    args = dict(remaining=4, prefix_count=tracker._storage_counts()[0], frontier_count=1, config=cfg)
    assert covered_proposal_operation(kernel, **(args | override)) is None
    with pytest.raises(ValueError, match='nonnegative integer'):
        covered_proposal_operation(kernel, **(args | dict(remaining=True)))
    assert covered_proposal_operation(kernel, **args, excluded=kernel.frontier) is None
    tracker.close()


def test_coverage_counterfactual_preserves_original_database_and_rolls_back(tmp_path):
    path = tmp_path/'source.sqlite'
    tracker = AllocationTeacherTracker(path, sequence_id='0003', config=config(1))
    raw, rows = scene()
    first = step(tracker, raw, rows)
    digest = tracker.close()
    result = compare(path, digest, first.prediction['commit_sha256'], tmp_path/'comparison',
                     check_exact_dump=True, check_coverage_admission=True)
    assert result['coverage_admission_compared'] and not result['new_event_committed']
    assert sha_file(path) == digest
    for candidate in result['candidates']:
        measured = candidate['coverage_result']
        if measured:
            assert measured['frontier_after'] <= measured['local_frontier_cap']
            assert measured['global_frontier_after'] <= measured['global_frontier_cap']
