import math

import pytest

from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.persistent_class_bound_beam import _RankingWork
from transvision.models.event_track_v2x.persistent_sparse_slot_bound_beam import _SparseSlotEnvelope
from transvision.models.event_track_v2x.persistent_reachable_slot_bound_beam import (
    PersistentReachableSlotBoundBeamConfig, PersistentReachableSlotBoundBeamTracker,
    _ReachableSlotEnvelope,
)
import test_persistent_slot_bound_beam as dense
import test_persistent_sparse_slot_bound_beam as sparse
from test_forest_tracking import observation
from test_persistent_component_tracking import step


def tracker(path, width=2, **limits):
    return PersistentReachableSlotBoundBeamTracker(path, sequence_id='0003',
        config=PersistentReachableSlotBoundBeamConfig(
            state=ForestTrackingConfig(active_limit=width), **limits))


@pytest.fixture
def reachable(monkeypatch):
    monkeypatch.setattr(dense, 'tracker', tracker)
    monkeypatch.setattr(dense, 'PersistentSlotBoundBeamTracker', PersistentReachableSlotBoundBeamTracker)
    monkeypatch.setattr(dense, '_SlotEnvelope', _ReachableSlotEnvelope)
    monkeypatch.setattr(sparse, 'tracker', tracker)


@pytest.mark.parametrize('seed', range(6))
def test_every_prefix_against_independent_full_parent_oracle(tmp_path, reachable, seed):
    dense.test_every_prefix_bounds_independent_exhaustive_identity_classes(tmp_path, seed)


@pytest.mark.parametrize('seed', range(4))
def test_single_source_suffix_exact_maximum(tmp_path, reachable, seed):
    dense.test_one_remaining_source_frame_bound_is_exact_maximum_not_partition(tmp_path, seed)


@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('width', [1, 2, 20])
def test_joint_ranking_rescore_state_and_risk_oracle(tmp_path, monkeypatch, reachable, seed, width):
    dense.test_join_rescore_state_and_risk_against_existing_full_parent_oracle(tmp_path, monkeypatch, width, seed)


@pytest.mark.parametrize('check', [
    'test_new_evidence_can_select_prior_combination_below_the_old_product_top_k',
    'test_joint_search_is_lazy_when_old_best_combination_is_already_decisive',
    'test_reopen_preserves_irreversible_previous_event_pruning',
    'test_late_state_time_on_joint_merge_replays_raw_observations_and_preserves_old_output',
])
def test_irreversibility_causality_idempotency(tmp_path, monkeypatch, reachable, check):
    dense.test_no_old_preselection_recovery_or_history_rewrite(tmp_path, monkeypatch, check)


@pytest.mark.parametrize('check', [
    'test_no_available_old_roots_and_extreme_log_weights_stay_finite',
    'test_sparse_unknown_parent_can_reach_a_known_root_without_a_direct_edge',
    'test_truncated_mass_still_sums_multiple_classes',
])
def test_mass_and_indirect_known_root(tmp_path, reachable, check):
    getattr(dense, check)(tmp_path)


def test_dual_not_primal_with_suboptimal_solver(tmp_path, monkeypatch, reachable):
    dense.test_dual_cover_remains_an_upper_bound_with_a_suboptimal_solver_assignment(tmp_path, monkeypatch)


@pytest.mark.parametrize('limits', [dict(max_assignment_matrix_cells=1), dict(max_assignment_solves=1),
    dict(max_ranking_operations=80), dict(max_reachability_terms=1)])
def test_entire_event_rolls_back_on_work_or_memory_caps(tmp_path, reachable, limits):
    sparse.test_nontrivial_sparse_block_caps_roll_back_event(tmp_path, limits)


@pytest.mark.parametrize('value', [0, -1, True, 2.5])
def test_reachability_cap_requires_positive_integer(value):
    with pytest.raises(ValueError):
        PersistentReachableSlotBoundBeamConfig(max_reachability_terms=value)


def test_shared_future_root_has_capacity_not_independent_unknown_options(tmp_path):
    t = tracker(tmp_path/'future.db', 64)
    raw = [observation('a', frame='v', index=0), observation('b', frame='v', index=1),
           observation('c', source=1, frame='r', index=0), observation('d', source=1, frame='r', index=1)]
    # b's edge to a is impossible (same source/frame), so b can only be born.
    # Both c and d can reach b, but only one may use it in source/frame r.
    rows = [((-1, 0.),), ((-1, 0.), (0, 10.)),
            ((-1, -10.), (1, 10.)), ((-1, -10.), (1, 10.))]
    result = step(t, raw, rows)
    kernel = t.kernels[result.audit['components'][0]['component']]
    handle = kernel.db.execute('SELECT h FROM prefixes WHERE depth=1').fetchone()[0]
    new = _ReachableSlotEnvelope(kernel, 1, _RankingWork(t.config)).prefix_upper(handle)
    old = _SparseSlotEnvelope(kernel, 1, _RankingWork(t.config)).prefix_upper(handle)
    assert new == pytest.approx(0., abs=1e-9)
    assert old == pytest.approx(20., abs=1e-9)
    assert [kernel._row(i) for i in range(4)] == rows
    assert result.audit['positive_parent_support_unchanged']
    assert not result.audit['ranking_bound_used_for_partition_mass']
    t.close()


def test_parent_with_multiple_reachable_routes_is_not_counted_twice(tmp_path):
    t = tracker(tmp_path/'aliases.db', 256)
    raw = [observation(str(i), frame=str(i), index=i) for i in range(4)]
    rows = [((-1, 0.),), ((-1, -10.), (0, 0.)),
            ((-1, -10.), (0, 0.), (1, 0.)), ((-1, -10.), (2, math.log(3.)))]
    result = step(t, raw, rows)
    kernel = t.kernels[result.audit['components'][0]['component']]
    handle = kernel.db.execute('SELECT h FROM prefixes WHERE depth=1').fetchone()[0]
    # Row 2 contributes 2 via its two distinct parents; row 3 contributes 3,
    # not 6, even though its single parent has two paths to root 0.
    bound = _ReachableSlotEnvelope(kernel, 1, _RankingWork(t.config)).prefix_upper(handle)
    assert bound == pytest.approx(math.log(6.), abs=1e-9)
    t.close()
