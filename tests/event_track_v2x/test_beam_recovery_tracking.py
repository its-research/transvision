import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.beam_recovery_tracking import BeamRecoveryConfig,BeamRecoveryTracker
from transvision.models.event_track_v2x.detection_cache_v2 import canonical
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig,replay_forest_states
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from transvision.models.event_track_v2x.persistent_beam_tracking import PersistentBeamConfig,PersistentBeamTracker
from test_forest_tracking import observation
from test_persistent_beam_tracking import evidence_scene
from test_persistent_component_tracking import step,joint_action
from test_persistent_forest import brute


def make(path,**kwargs):
    return BeamRecoveryTracker(path,sequence_id='0003',config=BeamRecoveryConfig(**kwargs))


@pytest.mark.parametrize('budget',[0,1,256])
def test_disabled_recovery_is_original_beam_with_no_candidate_or_state_change(tmp_path,budget):
    state=ForestTrackingConfig(active_limit=2)
    beam=PersistentBeamTracker(tmp_path/'beam.db',sequence_id='0003',config=PersistentBeamConfig(state=state))
    disabled=make(tmp_path/'disabled.db',state=state,enable_recovery=False,recovery_budget=budget)
    raw,rows,new,new_rows=evidence_scene()
    events=[(raw,rows,1_100_000),(new,new_rows,1_300_000)]+[((),(),1_400_000+i) for i in range(3)]
    for i,(obs,factors,time) in enumerate(events):
        a,b=[step(t,obs,factors,time=time,event=str(i)) for t in (beam,disabled)]
        assert a.prediction_json==b.prediction_json
        for key in ('factor_rows_sha256','candidate_evaluations','beam_expansions','state_updates','components'):
            assert a.audit[key]==b.audit[key]
        assert b.audit['recovery_search_steps']==0 and not b.audit['recovery_enabled']
    sha=disabled.close()
    disabled=BeamRecoveryTracker.open(tmp_path/'disabled.db',expected_database_sha256=sha,
        expected_prediction_sha256=b.prediction['commit_sha256'])
    assert not disabled.config.enable_recovery
    assert step(disabled,raw,rows,time=1_100_000,event='0').prediction_json==step(beam,raw,rows,time=1_100_000,event='0').prediction_json
    disabled.close();beam.close()


@pytest.mark.parametrize('seed',range(3))
@pytest.mark.parametrize('budget',[0,2,100])
def test_full_raw_mass_cover_absolute_weights_regret_and_states_against_brute(tmp_path,seed,budget):
    t=make(tmp_path/'recover.db',state=ForestTrackingConfig(active_limit=2),recovery_budget=budget)
    rng=np.random.default_rng(seed)
    raw=tuple(observation(str(i),source=i%2,frame=str(i),state_us=1_000_000+i) for i in range(5))
    support=((-1,),(-1,),(-1,0),(-1,0,2),(-1,1))
    rows=tuple(tuple((p,float(rng.normal())) for p in options) for options in support)
    result=step(t,raw,rows);factors=ForestFactors(tuple(o.node for o in raw),rows)
    probability,labels,z=brute(factors);action=joint_action(t,result);scope=result.audit['decision_indices']
    def risk(a):return sum(p*sum(labels[a][i]!=labels[h][i] for i in scope)/len(scope) for h,p in probability.items())
    assert risk(action)-min(map(risk,probability))<=result.audit['model_regret_upper']+1e-12
    assert math.log(z)<=result.audit['log_partition_upper']+1e-12
    assert result.audit['recovery_search_steps']<=budget
    assert result.audit['recovery_extra_state_updates']>=0
    for summary in result.audit['components']:
        kernel=t.kernels[summary['component']]
        local=ForestFactors(tuple(raw[i].node for i in t.store.members(summary['component'])),tuple(kernel._row(i) for i in range(kernel.n)))
        probs,roots,partition=brute(local)
        def signature(h):return tuple(kernel._prefix(kernel.ancestor(h,i+1)).root for i in range(kernel._prefix(h).depth))
        active={signature(item['handle']) for item in summary['active']}
        frontier=[signature(h) for h in kernel.frontier]
        assert all(not (a[:len(b)]==b or b[:len(a)]==a) for i,a in enumerate(frontier) for b in frontier[i+1:])
        assert all(g in active or any(g[:len(p)]==p for p in frontier) for g in roots.values())
        assert sum(p for h,p in probs.items() if roots[h] not in active)<=summary['eta_upper']+1e-12
        for item in summary['active']:
            mass=partition*sum(p for h,p in probs.items() if roots[h]==signature(item['handle']))
            assert math.exp(item['log_weight'])==pytest.approx(mass)
        assert summary['log_retained']+1e-12>=summary['backbone_retained_log_mass']
    expected,_=replay_forest_states('0003',raw,factors,action,1_100_000,t.config.state)
    fields=('track_id','class_label','mean','covariance','score')
    expected=sorted([{k:p[k] for k in fields} for p in expected],key=lambda p:p['track_id'])
    assert canonical(result.prediction['predictions'])==canonical(expected)
    t.close()


def test_future_evidence_can_recover_old_branch_without_rewriting_history(tmp_path):
    full=make(tmp_path/'full.db',state=ForestTrackingConfig(active_limit=1),recovery_budget=3)
    off=make(tmp_path/'off.db',state=ForestTrackingConfig(active_limit=1),enable_recovery=False,recovery_budget=3)
    raw,rows,new,new_rows=evidence_scene()
    first=[step(t,raw,rows) for t in (full,off)]
    result=[step(t,new,new_rows,time=1_300_000,event='new') for t in (full,off)]
    recovered=[]
    for i in range(30):
        recovered.extend(e for c in result[0].audit['components'] for e in c['recovery_events'])
        if joint_action(full,result[0])[2]==1:break
        result=[step(t,time=1_400_000+i,event='compute'+str(i)) for t in (full,off)]
    assert joint_action(full,result[0])[2]==1 and joint_action(off,result[1])[2]==0
    assert recovered and all(e['scope']=='actual_output_class_outside_previous_active_and_output' for e in recovered)
    assert step(full,raw,rows).prediction_json==first[0].prediction_json
    sha=full.close();full=BeamRecoveryTracker.open(tmp_path/'full.db',expected_database_sha256=sha,
        expected_prediction_sha256=result[0].prediction['commit_sha256'])
    step(full,time=1_500_000,event='reopened')
    assert step(full,raw,rows).prediction_json==first[0].prediction_json
    full.close();off.close()


def test_recovery_after_merge_and_rescore_has_complete_cover(tmp_path):
    t=make(tmp_path/'merge.db',state=ForestTrackingConfig(active_limit=1),recovery_budget=100)
    raw=(observation('a'),observation('b',source=1,frame='b'),
         observation('c',source=1,frame='c',state_us=1_010_000))
    rows=(((-1,0.),),((-1,0.),),((-1,-10.),(0,0.)))
    old=step(t,raw,rows);assert joint_action(t,old)[2]==0
    bridge=observation('bridge',source=0,frame='bridge',state_us=1_200_000)
    result=step(t,[bridge],[((-1,-4.),(1,1.),(2,1.))],time=1_300_000,event='merge',
        rescored_rows=[(2,((-1,30.),(0,-30.)))])
    assert joint_action(t,result)[2]==-1
    assert result.audit['components'][0]['recovery_events']
    assert result.audit['components'][0]['recovery_cover_restarted_from_root']
    assert step(t,raw,rows).prediction_json==old.prediction_json
    t.close()


def test_failed_recovery_rolls_back_backbone_and_raw_event_then_retry_succeeds(tmp_path,monkeypatch):
    t=make(tmp_path/'rollback.db',recovery_budget=2)
    raw,rows,new,new_rows=evidence_scene();step(t,raw,rows)
    before=tuple(t.db.iterdump());execute=t._execute_recovery
    def fail(*args):
        execute(*args);raise RuntimeError('injected recovery failure')
    monkeypatch.setattr(t,'_execute_recovery',fail)
    with pytest.raises(RuntimeError,match='injected'):
        step(t,new,new_rows,time=1_300_000,event='new')
    assert tuple(t.db.iterdump())==before
    monkeypatch.setattr(t,'_execute_recovery',execute)
    accepted=step(t,new,new_rows,time=1_300_000,event='new')
    assert accepted.audit['recovery_enabled']
    t.close()


def test_future_input_and_missing_cover_are_rejected(tmp_path):
    t=make(tmp_path/'future.db')
    with pytest.raises(ValueError):
        step(t,[observation('future',arrival_us=2_000_000)],[((-1,0.),)],time=1_100_000)
    raw,rows,_,_=evidence_scene();step(t,raw,rows)
    for kernel in t.kernels.values():kernel.db.execute("DELETE FROM meta WHERE k='recovery_frontier'")
    t.db.commit()
    with pytest.raises(ValueError,match='missing its full-support'):
        step(t,time=1_300_000,event='missing')
    t.close()


@pytest.mark.parametrize('values',[dict(enable_recovery=1),dict(recovery_budget=True),dict(recovery_budget=-1),
    dict(recovery_budget=100001),dict(recovery_completions_per_component=9),dict(max_beam_expansions=0)])
def test_invalid_configuration_rejected(values):
    with pytest.raises(ValueError):BeamRecoveryConfig(**values)
