import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.persistent_class_bound_beam import (
    PersistentClassBoundBeamConfig, PersistentClassBoundBeamTracker, _ClassEnvelope, _RankingWork,
)
from transvision.models.event_track_v2x.persistent_joint_beam import PersistentJointBeamConfig, PersistentJointBeamTracker
from tools.event_track_v2x.ranked_class_bound import SingleClassBoundModel
import test_persistent_joint_beam as joint
from test_forest_tracking import observation
from test_persistent_component_tracking import step


def tracker(path, width=2, **limits):
    return PersistentClassBoundBeamTracker(path, sequence_id='0003', config=PersistentClassBoundBeamConfig(
        state=ForestTrackingConfig(active_limit=width), **limits))


@pytest.mark.parametrize('width',[1,2,20])
@pytest.mark.parametrize('seed',range(3))
def test_same_independent_full_parent_oracle_for_join_rescore_risk_and_state(tmp_path,monkeypatch,width,seed):
    monkeypatch.setattr(joint,'tracker',tracker)
    joint.test_three_way_join_rank_weights_risk_and_state_against_full_parent_oracle(tmp_path,width,seed)


@pytest.mark.parametrize('check',[
    'test_new_evidence_can_select_prior_combination_below_the_old_product_top_k',
    'test_joint_search_is_lazy_when_old_best_combination_is_already_decisive',
    'test_reopen_preserves_irreversible_previous_event_pruning',
    'test_late_state_time_on_joint_merge_replays_raw_observations_and_preserves_old_output',
])
def test_existing_semantic_guards_apply_to_the_new_schema(tmp_path,monkeypatch,check):
    monkeypatch.setattr(joint,'tracker',tracker)
    monkeypatch.setattr(joint,'PersistentJointBeamTracker',PersistentClassBoundBeamTracker)
    getattr(joint,check)(tmp_path)


@pytest.mark.parametrize('limits',[dict(max_cartesian_states=1),dict(max_batch_frontier=1),dict(max_merge_prefix_steps=1)])
def test_existing_caps_still_fail_transactionally(tmp_path,monkeypatch,limits):
    monkeypatch.setattr(joint,'tracker',tracker)
    joint.test_joint_capacity_failure_rolls_back_old_weights_maps_and_new_observations(tmp_path,limits)


@pytest.mark.parametrize('limits',[dict(max_ranking_operations=7),dict(max_ranking_catalog_terms=1)])
def test_new_bound_work_caps_rollback_without_rewriting_old_ids(tmp_path,limits):
    t=tracker(tmp_path/'work.db',1,**limits)
    first=step(t,[observation('a')],[((-1,0.),)])
    before=tuple(t.db.iterdump())
    with pytest.raises(ValueError,match='ranking .*capacity'):
        step(t,[observation('b',frame='later',state_us=1_200_000)],
             [((-1,0.),(0,0.))],time=1_300_000,event='later')
    assert tuple(t.db.iterdump())==before and t.n==1 and not hasattr(t,'_ranking_work')
    assert step(t,[observation('a')],[((-1,0.),)]).prediction_json==first.prediction_json
    t.close()


def test_mass_bound_and_risk_do_not_use_the_single_class_upper(tmp_path):
    t=tracker(tmp_path/'mass.db',1)
    result=step(t,[observation('a'),observation('b',source=1)],
                [((-1,0.),),((-1,0.),(0,0.))])
    summary,=result.audit['components'];kernel=t.kernels[summary['component']]
    # Two equiprobable root classes: one retained weight is 1, total weight 2.
    assert summary['log_retained']==pytest.approx(0.)
    assert summary['log_partition_upper']==pytest.approx(math.log(2))
    assert summary['eta_upper']==pytest.approx(.5)
    assert not result.audit['ranking_bound_used_for_partition_mass']
    assert not result.audit['recovery_enabled']
    assert result.audit['ranking_operations']==sum(result.audit['ranking_work_counts'].values())
    assert summary['log_partition_upper']==kernel._upper(0)
    t.close()


def test_live_bound_matches_independently_tested_diagnostic_for_all_stored_prefixes(tmp_path):
    rng=np.random.default_rng(44)
    raw=tuple(observation(str(i),source=i%2,frame=str(i)) for i in range(5))
    rows=tuple(tuple((p,float(rng.uniform(-2,2))) for p in range(-1,i)) for i in range(5))
    t=tracker(tmp_path/'bound.db',64);result=step(t,raw,rows)
    summary,=result.audit['components'];kernel=t.kernels[summary['component']]
    work=_RankingWork(t.config);envelope=_ClassEnvelope(kernel,0,work)
    diagnostic=SingleClassBoundModel(rows)
    for handle,depth in kernel.db.execute('SELECT h,depth FROM prefixes ORDER BY h'):
        roots=tuple(kernel._prefix(kernel.ancestor(handle,i+1)).root for i in range(depth))
        assert envelope.prefix_upper(handle)==pytest.approx(
            diagnostic.for_prefix(roots).log_single_class_upper,abs=1e-10)
    assert work.counts['root_ancestor_queries']>0
    assert work.root_cache_peak<=5 and work.catalog_peak==sum(map(len,rows))
    t.close()


def test_tighter_ranking_finishes_with_same_small_frontier_without_shrinking_mass(tmp_path):
    state=ForestTrackingConfig(active_limit=1)
    old=PersistentJointBeamTracker(tmp_path/'old.db',sequence_id='0003',
        config=PersistentJointBeamConfig(state=state,max_batch_frontier=32))
    new=tracker(tmp_path/'new.db',1,max_batch_frontier=32)
    for t in (old,new): step(t,[observation('a')],[((-1,0.),)])
    raw=tuple(observation('b'+str(i),frame='new'+str(i),state_us=1_200_000) for i in range(20))
    rows=tuple(((-1,math.log(.9)),(0,0.)) for _ in raw)
    with pytest.raises(ValueError,match='frontier capacity'):
        step(old,raw,rows,time=1_300_000,event='batch')
    assert old.n==1
    result=step(new,raw,rows,time=1_300_000,event='batch')
    summary,=result.audit['components']
    assert summary['active'][0]['log_weight']==pytest.approx(0.)
    assert summary['pruning'][0]['frontier_peak']<=32
    assert summary['log_partition_upper']==pytest.approx(20*math.log(1.9))
    assert summary['eta_upper']>.99
    assert result.audit['ranking_operations']>0 and not result.audit['formal_numeric_certificate']
    old.close();new.close()


def test_predecessor_group_bound_rejects_cut_support(tmp_path):
    t=tracker(tmp_path/'groups.db',2)
    step(t,[observation('a'),observation('b',source=1)],[((-1,0.),),((-1,0.),(0,0.))])
    kernel=next(iter(t.kernels.values()))
    with pytest.raises(ValueError,match='old support edge'):
        _ClassEnvelope(kernel,2,_RankingWork(t.config)).predecessor_suffix([1,2])
    t.close()


@pytest.mark.parametrize('limits',[dict(max_ranking_operations=0),dict(max_ranking_operations=True),
                                   dict(max_ranking_catalog_terms=1.5)])
def test_ranking_limits_are_positive_integer_configuration(limits):
    with pytest.raises(ValueError): PersistentClassBoundBeamConfig(**limits)
