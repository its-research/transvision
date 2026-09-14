from collections import defaultdict
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from transvision.models.event_track_v2x.persistent_class_bound_beam import _ClassEnvelope,_RankingWork
from transvision.models.event_track_v2x.persistent_slot_bound_beam import (
    PersistentSlotBoundBeamConfig,PersistentSlotBoundBeamTracker,_SlotEnvelope,
)
import test_persistent_joint_beam as joint
from test_forest_tracking import observation
from test_persistent_component_tracking import step
from test_persistent_forest import brute


def tracker(path,width=2,**limits):
    return PersistentSlotBoundBeamTracker(path,sequence_id='0003',config=PersistentSlotBoundBeamConfig(
        state=ForestTrackingConfig(active_limit=width),**limits))


@pytest.mark.parametrize('seed',range(6))
def test_every_prefix_bounds_independent_exhaustive_identity_classes(tmp_path,seed):
    rng=np.random.default_rng(seed)
    slots=((0,'a'),(1,'b'),(0,'a'),(1,'b'),(0,'c'),(1,'c'))
    raw=tuple(observation(str(i),source=s,frame=f,index=i) for i,(s,f) in enumerate(slots))
    rows=tuple(tuple((p,float(rng.uniform(-2,2))) for p in range(-1,i)) for i in range(6))
    t=tracker(tmp_path/'all.db',256);result=step(t,raw,rows)
    summary,=result.audit['components'];kernel=t.kernels[summary['component']]
    work=_RankingWork(t.config);envelope=_SlotEnvelope(kernel,0,work)
    old=_ClassEnvelope(kernel,0,_RankingWork(t.config))
    probabilities,labels,z=brute(ForestFactors(tuple(o.node for o in raw),rows))
    weights=defaultdict(float)
    for h,p in probabilities.items(): weights[labels[h]]+=p*z
    checked=set()
    for handle,depth in kernel.db.execute('SELECT h,depth FROM prefixes ORDER BY h'):
        roots=tuple(kernel._prefix(kernel.ancestor(handle,i+1)).root for i in range(depth))
        bound=envelope.prefix_upper(handle)
        maximum=max(w for r,w in weights.items() if r[:depth]==roots)
        assert math.log(maximum)<=bound+1e-10
        assert bound<=old.prefix_upper(handle)+1e-10
        checked.add(roots)
    # The oracle, not the search output, determines the expected prefix domain.
    assert checked=={r[:d] for r in weights for d in range(7)}
    assert result.audit['ranking_operations']==sum(result.audit['ranking_work_counts'].values())
    assert result.audit['source_frame_exclusion_in_ranking_bound']
    assert not result.audit['ranking_bound_used_for_partition_mass']
    t.close()


@pytest.mark.parametrize('seed',range(4))
def test_one_remaining_source_frame_bound_is_exact_maximum_not_partition(tmp_path,seed):
    rng=np.random.default_rng(seed)
    slots=((0,'old0'),(1,'old1'),(1,'new'),(1,'new'),(1,'new'))
    raw=tuple(observation(str(i),source=s,frame=f,index=i) for i,(s,f) in enumerate(slots))
    rows=tuple(tuple((p,float(rng.normal())) for p in range(-1,i)) for i in range(5))
    t=tracker(tmp_path/'slot.db',256);result=step(t,raw,rows)
    kernel=t.kernels[result.audit['components'][0]['component']]
    probabilities,labels,z=brute(ForestFactors(tuple(o.node for o in raw),rows))
    weights=defaultdict(float)
    for h,p in probabilities.items(): weights[labels[h]]+=p*z
    envelope=_SlotEnvelope(kernel,2,_RankingWork(t.config))
    for handle,depth in kernel.db.execute('SELECT h,depth FROM prefixes WHERE depth>=2'):
        roots=tuple(kernel._prefix(kernel.ancestor(handle,i+1)).root for i in range(depth))
        actual=[w for r,w in weights.items() if r[:depth]==roots]
        assert envelope.prefix_upper(handle)==pytest.approx(math.log(max(actual)),abs=1e-9)
    assert result.audit['source_frame_assignment_is_not_full_posterior_inference'] is True
    t.close()


@pytest.mark.parametrize('width',[1,2,20])
@pytest.mark.parametrize('seed',range(3))
def test_join_rescore_state_and_risk_against_existing_full_parent_oracle(tmp_path,monkeypatch,width,seed):
    monkeypatch.setattr(joint,'tracker',tracker)
    joint.test_three_way_join_rank_weights_risk_and_state_against_full_parent_oracle(tmp_path,width,seed)


@pytest.mark.parametrize('check',[
    'test_new_evidence_can_select_prior_combination_below_the_old_product_top_k',
    'test_joint_search_is_lazy_when_old_best_combination_is_already_decisive',
    'test_reopen_preserves_irreversible_previous_event_pruning',
    'test_late_state_time_on_joint_merge_replays_raw_observations_and_preserves_old_output',
])
def test_no_old_preselection_recovery_or_history_rewrite(tmp_path,monkeypatch,check):
    monkeypatch.setattr(joint,'tracker',tracker)
    monkeypatch.setattr(joint,'PersistentJointBeamTracker',PersistentSlotBoundBeamTracker)
    getattr(joint,check)(tmp_path)


@pytest.mark.parametrize('limits',[dict(max_assignment_matrix_cells=1),dict(max_assignment_solves=1)])
def test_assignment_limits_rollback_entire_event(tmp_path,limits):
    t=tracker(tmp_path/'caps.db',1,**limits)
    first=step(t,[observation('a')],[((-1,0.),)])
    before=tuple(t.db.iterdump())
    with pytest.raises(ValueError,match='slot assignment .*capacity'):
        step(t,[observation('b',frame='b',state_us=1_200_000),
                observation('c',frame='c',state_us=1_200_000)],
             [((-1,0.),(0,0.)),((-1,0.),(0,0.))],time=1_300_000,event='later')
    assert tuple(t.db.iterdump())==before and t.n==1
    assert step(t,[observation('a')],[((-1,0.),)]).prediction_json==first.prediction_json
    t.close()


@pytest.mark.parametrize('limits',[dict(max_assignment_matrix_cells=0),dict(max_assignment_solves=True)])
def test_assignment_caps_require_positive_integers(limits):
    with pytest.raises(ValueError): PersistentSlotBoundBeamConfig(**limits)


def test_no_available_old_roots_and_extreme_log_weights_stay_finite(tmp_path):
    raw=[observation('a',frame='same',index=0),observation('b',frame='same',index=1)]
    rows=[((-1,1000.),),((-1,-1000.),(0,1000.))]
    t=tracker(tmp_path/'finite.db',1);result=step(t,raw,rows)
    summary,=result.audit['components']
    assert summary['active'][0]['log_weight']==pytest.approx(0.)
    assert result.audit['ranking_work_counts']['same_slot_parent_edges_excluded']>0
    # All legal support is retained, so the inherited mass path is exact here.
    assert summary['full_model_support_still_in_beam'] and summary['log_partition_upper']==pytest.approx(0.)
    assert t.kernels[summary['component']]._upper(0)>=1999.
    t.close()


def test_sparse_unknown_parent_can_reach_a_known_root_without_a_direct_edge(tmp_path):
    raw=[observation('a',frame='old'),observation('b',source=1,frame='new'),
         observation('c',frame='later')]
    rows=[((-1,0.),),((-1,-4.),(0,0.)),((-1,-4.),(1,math.log(2.)))]
    t=tracker(tmp_path/'sparse.db',32);result=step(t,raw,rows)
    kernel=t.kernels[result.audit['components'][0]['component']]
    h=kernel.db.execute('SELECT h FROM prefixes WHERE depth=1').fetchone()[0]
    assert _SlotEnvelope(kernel,1,_RankingWork(t.config)).prefix_upper(h)>=math.log(2.)
    t.close()


def test_truncated_mass_still_sums_multiple_classes(tmp_path):
    t=tracker(tmp_path/'mass.db',1)
    result=step(t,[observation('a'),observation('b',source=1)],
                [((-1,0.),),((-1,0.),(0,0.))])
    summary,=result.audit['components']
    assert summary['log_retained']==pytest.approx(0.)
    assert summary['log_partition_upper']==pytest.approx(math.log(2.))
    assert summary['eta_upper']==pytest.approx(.5)
    t.close()


def test_dual_cover_remains_an_upper_bound_with_a_suboptimal_solver_assignment(tmp_path,monkeypatch):
    import transvision.models.event_track_v2x.persistent_slot_bound_beam as module
    t=tracker(tmp_path/'dual.db',1)
    result=step(t,[observation('a')],[((-1,0.),)])
    kernel=t.kernels[result.audit['components'][0]['component']]
    envelope=_SlotEnvelope(kernel,0,_RankingWork(t.config))
    # Wrong but legal solution: row 0 private, row 1 root. Its score is 2,
    # whereas row 0 root and row 1 private has score 3.
    monkeypatch.setattr(module,'linear_sum_assignment',lambda costs:(np.array([0,1]),np.array([1,0])))
    bound,_=envelope._assignment([(0.,{0:3.},-math.inf),(0.,{0:2.},-math.inf)],{0},set())
    assert bound>=3. and envelope.work.assignment_dual_gap_max>=1.
    t.close()
