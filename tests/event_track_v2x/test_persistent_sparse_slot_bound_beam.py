from itertools import product
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.persistent_class_bound_beam import _RankingWork
from transvision.models.event_track_v2x.persistent_slot_bound_beam import _SlotEnvelope
from transvision.models.event_track_v2x.persistent_sparse_slot_bound_beam import (
    PersistentSparseSlotBoundBeamConfig,PersistentSparseSlotBoundBeamTracker,_SparseSlotEnvelope,
)
import test_persistent_slot_bound_beam as dense
from test_forest_tracking import observation
from test_persistent_component_tracking import step


def tracker(path,width=2,**limits):
    return PersistentSparseSlotBoundBeamTracker(path,sequence_id='0003',config=PersistentSparseSlotBoundBeamConfig(
        state=ForestTrackingConfig(active_limit=width),**limits))


@pytest.fixture
def sparse(monkeypatch):
    monkeypatch.setattr(dense,'tracker',tracker)
    monkeypatch.setattr(dense,'PersistentSlotBoundBeamTracker',PersistentSparseSlotBoundBeamTracker)
    monkeypatch.setattr(dense,'_SlotEnvelope',_SparseSlotEnvelope)


@pytest.mark.parametrize('seed',range(6))
def test_full_parent_oracle_for_every_mixed_slot_prefix(tmp_path,sparse,seed):
    dense.test_every_prefix_bounds_independent_exhaustive_identity_classes(tmp_path,seed)


@pytest.mark.parametrize('seed',range(4))
def test_single_slot_exact_maximum_with_independent_oracle(tmp_path,sparse,seed):
    dense.test_one_remaining_source_frame_bound_is_exact_maximum_not_partition(tmp_path,seed)


@pytest.mark.parametrize('seed',range(3))
@pytest.mark.parametrize('width',[1,2,20])
def test_joint_rescore_state_and_risk_oracle(tmp_path,monkeypatch,sparse,seed,width):
    dense.test_join_rescore_state_and_risk_against_existing_full_parent_oracle(tmp_path,monkeypatch,width,seed)


@pytest.mark.parametrize('check',[
    'test_new_evidence_can_select_prior_combination_below_the_old_product_top_k',
    'test_joint_search_is_lazy_when_old_best_combination_is_already_decisive',
    'test_reopen_preserves_irreversible_previous_event_pruning',
    'test_late_state_time_on_joint_merge_replays_raw_observations_and_preserves_old_output',
])
def test_persistence_causality_and_no_recovery(tmp_path,monkeypatch,sparse,check):
    dense.test_no_old_preselection_recovery_or_history_rewrite(tmp_path,monkeypatch,check)


@pytest.mark.parametrize('check',[
    'test_no_available_old_roots_and_extreme_log_weights_stay_finite',
    'test_sparse_unknown_parent_can_reach_a_known_root_without_a_direct_edge',
    'test_truncated_mass_still_sums_multiple_classes',
])
def test_mass_and_unknown_root_regressions(tmp_path,sparse,check):
    getattr(dense,check)(tmp_path)


def test_suboptimal_primal_remains_covered(tmp_path,monkeypatch,sparse):
    dense.test_dual_cover_remains_an_upper_bound_with_a_suboptimal_solver_assignment(tmp_path,monkeypatch)


@pytest.mark.parametrize('seed',range(10))
def test_sparse_matrix_bound_against_all_legal_assignments(tmp_path,seed):
    rng=np.random.default_rng(seed);t=tracker(tmp_path/'matrix.db')
    result=step(t,[observation('a')],[((-1,0.),)])
    kernel=t.kernels[result.audit['components'][0]['component']]
    details=[(float(rng.normal()),{r:float(rng.normal()) for r in range(3) if rng.random()<.4},
              float(rng.normal()) if rng.random()<.3 else -math.inf) for _ in range(5)]
    occupied={seed%3} if seed%2 else set();roots=set(range(3));available=sorted(roots-occupied)
    weights=[]
    for choices in product([-1,*available],repeat=5):
        picked=[r for r in choices if r>=0]
        if len(picked)!=len(set(picked)): continue
        values=[max(b,u) if r<0 else float(np.logaddexp(k.get(r,-math.inf),u))
                for (b,k,u),r in zip(details,choices)]
        weights.append(math.fsum(values))
    bound,_=_SparseSlotEnvelope(kernel,0,_RankingWork(t.config))._assignment(details,roots,occupied)
    reference,_=_SlotEnvelope(kernel,0,_RankingWork(t.config))._assignment(details,roots,occupied)
    assert max(weights)<=bound+1e-12
    assert bound==pytest.approx(reference,abs=1e-10)
    t.close()


def test_disconnected_singletons_allocate_no_dense_matrix(tmp_path):
    t=tracker(tmp_path/'singletons.db');result=step(t,[observation('a')],[((-1,0.),)])
    kernel=t.kernels[result.audit['components'][0]['component']]
    work=_RankingWork(t.config);e=_SparseSlotEnvelope(kernel,0,work)
    upper,_=e._assignment([(0.,{r:2.},-math.inf) for r in range(40)],set(range(40)),set())
    assert upper==pytest.approx(80.) and work.counts['assignment_solves']==0
    assert work.counts['assignment_matrix_cells']==0 and work.sparse_blocks_peak==40
    t.close()


@pytest.mark.parametrize('limits',[dict(max_assignment_matrix_cells=1),dict(max_assignment_solves=1),
                                  dict(max_ranking_operations=80)])
def test_nontrivial_sparse_block_caps_roll_back_event(tmp_path,limits):
    t=tracker(tmp_path/'cap.db',1,**limits)
    raw=[observation('a'),observation('b',index=1)]
    step(t,raw,[((-1,0.),),((-1,0.),)])
    before=tuple(t.db.iterdump())
    new=[observation(str(i),source=1,frame='new',index=i,state_us=1_200_000) for i in range(4)]
    with pytest.raises(ValueError,match='capacity'):
        step(t,new,[((-1,0.),(i//2,0.)) for i in range(4)],time=1_300_000,event='new')
    assert tuple(t.db.iterdump())==before and t.n==2
    t.close()
