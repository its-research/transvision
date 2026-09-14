from dataclasses import replace
import math

import numpy as np
import pytest

from tools.event_track_v2x.probe_frontier_completion import compare
from transvision.models.event_track_v2x.completion_component_tracking import (
    PersistentCompletionConfig,CompletionComponentTracker,CompletionTeacherTracker,CompletionLearnedTracker,
)
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.frontier_completion import frontier_completion_operation
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from transvision.models.event_track_v2x.learned_component_allocation import AllocationTeacherTracker
from test_forest_tracking import observation
from test_learned_component_allocation import policy,scene,config
from test_persistent_component_tracking import step,joint_action
from test_persistent_forest import brute


@pytest.mark.parametrize('cls',[CompletionComponentTracker,CompletionTeacherTracker,CompletionLearnedTracker])
@pytest.mark.parametrize('budget',[1,8,24])
@pytest.mark.parametrize('seed',range(3))
def test_completion_retains_full_legal_support_mass_regret_and_immutable_history(tmp_path,cls,budget,seed):
    cfg=PersistentCompletionConfig(state=ForestTrackingConfig(active_limit=2,expansion_budget=budget))
    options={'allocation_policy':policy(seed)} if cls is CompletionLearnedTracker else {}
    t=cls(tmp_path/'tracker.sqlite',sequence_id='0003',config=cfg,**options)
    raw=(observation('a',source=0,index=0),observation('b',source=0,index=1),
         observation('c',source=1,index=0),observation('d',source=1,index=1))
    rng=np.random.default_rng(seed)
    rows=(((-1,0.),),((-1,0.),),tuple((p,float(rng.normal())) for p in (-1,0,1)),
          tuple((p,float(rng.normal())) for p in (-1,0,1)))
    first=step(t,raw,rows)
    result=step(t,time=1_200_000,event='more-compute')
    factors=ForestFactors(tuple(o.node for o in raw),rows)
    probabilities,roots,z=brute(factors)
    chosen=joint_action(t,result)
    def risk(h):
        return math.fsum(p*sum(a!=b for a,b in zip(roots[h],roots[g]))/len(raw) for g,p in probabilities.items())
    assert risk(chosen)-min(risk(h) for h in probabilities)<=result.audit['model_regret_upper']+1e-12
    assert math.log(z)<=result.audit['log_partition_upper']+1e-12
    assert result.audit['search_steps']<=budget
    assert result.audit['frontier_completion_enabled']
    assert all(r['charged_search_steps']>0 for r in result.audit['allocation_trace']
               if r['work_kind']=='frontier_completion_proposal')
    for summary in result.audit['components']:
        k=t.kernels[summary['component']]; members=t.store.members(summary['component'])
        explicit={tuple(members[k._prefix(k.ancestor(a['handle'],i+1)).root] for i in range(k.n))
                  for a in summary['active']}
        retained=math.fsum(p for h,p in probabilities.items() if tuple(roots[h][i] for i in members) in explicit)
        assert 1-retained<=summary['eta_upper']+1e-12
        prefixes=[k.parents(h['handle']) for h in summary['frontier']]
        for h in probabilities:
            local_roots=tuple(roots[h][i] for i in members)
            representative=tuple(-1 if local_roots[i]==members[i] else min(p for p,_ in k._row(i)
                if p>=0 and local_roots[p]==local_roots[i]) for i in range(k.n))
            covered=sum(representative[:len(prefix)]==prefix for prefix in prefixes)
            assert covered<=1 and (local_roots in explicit or covered==1)
    assert step(t,raw,rows).prediction_json==first.prediction_json
    digest=t.close()
    reopened=cls.open(tmp_path/'tracker.sqlite',expected_prediction_sha256=result.prediction['commit_sha256'],
                      expected_database_sha256=digest,**options)
    assert step(reopened,raw,rows).prediction_json==first.prediction_json
    reopened.close()


def test_completion_operation_requires_room_for_all_charged_suffix_steps(tmp_path):
    t=CompletionComponentTracker(tmp_path/'capacity.sqlite',sequence_id='0003',
        config=PersistentCompletionConfig(state=ForestTrackingConfig(expansion_budget=0)))
    raw,rows=scene();step(t,raw,rows)
    k=next(k for k in t.kernels.values() if k.n>1)
    counts=dict(prefix_count=t._storage_counts()[0],frontier_count=t._storage_counts()[1])
    def operation(**overrides):
        return frontier_completion_operation(k,**(dict(remaining=100,config=t.config,**counts)|overrides))
    found=operation()
    assert found is not None and found['requested_steps']==k.n-k._prefix(found['base']).depth
    assert operation(remaining=found['requested_steps']-1) is None
    assert operation(prefix_count=t.config.max_total_prefix_nodes) is None
    assert operation(frontier_count=t.config.max_total_frontier) is None
    assert operation(excluded=k.frontier) is None
    with pytest.raises(ValueError,match='nonnegative integer'):
        operation(remaining=True)
    with pytest.raises(ValueError,match='one to eight'):
        replace(t.config,frontier_completions_per_component=9)
    t.close()


def test_real_state_probe_rolls_back_all_candidates_and_does_not_commit_event(tmp_path):
    path=tmp_path/'source.sqlite'
    t=AllocationTeacherTracker(path,sequence_id='0003',config=config(1))
    raw,rows=scene(); result=step(t,raw,rows)
    digest=t.close()
    report=compare(path,digest,result.prediction['commit_sha256'],tmp_path/'comparison',check_exact_dump=True)
    assert report['candidates'] and report['committed_events']==1
    assert not report['new_event_committed'] and report['new_observations']==0
    assert sha_file(path)==digest and not report['paper_eligible']


@pytest.mark.parametrize('budget', [0, 4, 20])
@pytest.mark.parametrize('seed', range(3))
def test_fixed_factor_complete_class_insertion_preserves_upper_and_grows_lower(tmp_path, budget, seed):
    """Check the insertion lemma separately from future data or factor updates."""
    t = CompletionComponentTracker(tmp_path/'insertion.sqlite', sequence_id='0003',
        config=PersistentCompletionConfig(state=ForestTrackingConfig(active_limit=1, expansion_budget=budget)))
    raw = (observation('a', source=0, index=0), observation('b', source=0, index=1),
           observation('c', source=1, index=0), observation('d', source=1, index=1))
    rng = np.random.default_rng(seed)
    rows = (((-1, 0.),), ((-1, 0.),),
            tuple((p, float(rng.normal())) for p in (-1, 0, 1)),
            tuple((p, float(rng.normal())) for p in (-1, 0, 1)))
    first = step(t, raw, rows)
    assert len(t.kernels) == 1
    k = next(iter(t.kernels.values()))
    # Isolated insertion experiments must not leave an uncommitted search state.
    t.db.execute('SAVEPOINT insertion_lemma')
    try:
        handles = [0]
        for _ in range(k.n):
            handles = [k._child(h, p) for h in handles for p in k._choices(h)]
        rng.shuffle(handles)
        for handle in handles:
            before = k._mass()
            k.seed_complete_action(handle, 1_100_000)
            after = k._mass()
            assert after[2] >= before[2]-1e-12
            assert after[3] == pytest.approx(before[3], rel=0, abs=1e-12)
            assert after[4] <= before[4]+1e-12
    finally:
        t.db.execute('ROLLBACK TO insertion_lemma')
        t.db.execute('RELEASE insertion_lemma')
        t._restore_runtime()
    assert step(t, raw, rows).prediction_json == first.prediction_json
    t.close()
