import pytest

from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig,replay_forest_states
from transvision.models.event_track_v2x.identity_forest import ForestFactors
from transvision.models.event_track_v2x.persistent_beam_tracking import PersistentBeamConfig,PersistentBeamTracker
from transvision.models.event_track_v2x.detection_cache_v2 import canonical
from test_forest_tracking import observation
from test_persistent_component_tracking import step,joint_action


def history(path,cap=18):
    t=PersistentBeamTracker(path,sequence_id='0003',
        config=PersistentBeamConfig(state=ForestTrackingConfig(active_limit=1,max_replay_operations=cap)))
    raw=[];rows=[]
    for i in range(6):
        incoming=[observation(f'{side}-{i}',x=side*30+i*.02,source=side,frame=str(i),
                    state_us=1_000_000-i*1000,arrival_us=1_100_000+i*10_000) for side in range(2)]
        factors=[((-1,0.),) if i==0 else ((-1,-100.),(2*(i-1)+side,0.)) for side in range(2)]
        step(t,incoming,factors,time=1_100_000+i*10_000,event=str(i))
        raw.extend(incoming);rows.extend(factors)
    incoming=[observation('bridge',frame='bridge',state_us=1_200_000,arrival_us=1_300_000)]
    factors=[((-1,-100.),(10,0.),(11,-10.))]
    return t,raw,rows,incoming,factors


def test_retained_predecessor_cache_avoids_replaying_all_late_prefixes_under_same_cap(tmp_path,monkeypatch):
    t,raw,rows,incoming,factors=history(tmp_path/'reuse.db')
    result=step(t,incoming,factors,time=1_300_000,event='join')
    assert result.audit['state_updates']==1
    assert result.audit['merge_state_cache_copies']==12
    assert result.audit['merge_state_cache_queries']==result.audit['merge_prefix_steps']==12
    raw.extend(incoming);rows.extend(factors)
    expected,_=replay_forest_states('0003',raw,ForestFactors(tuple(o.node for o in raw),tuple(rows)),
                                  joint_action(t,result),1_300_000,t.config.state)
    fields=('track_id','class_label','mean','covariance','score')
    expected=sorted([{k:p[k] for k in fields} for p in expected],key=lambda p:p['track_id'])
    assert canonical(result.prediction['predictions'])==canonical(expected)
    assert not result.audit['recovery_enabled']
    t.close()
    old,_,_,incoming,factors=history(tmp_path/'uncached.db')
    monkeypatch.setattr(old,'_transfer_state',lambda *a:None)
    before=tuple(old.db.iterdump())
    with pytest.raises(ValueError,match='branch-state work cap'):
        step(old,incoming,factors,time=1_300_000,event='join')
    assert tuple(old.db.iterdump())==before
    old.close()


def test_copied_cache_and_old_replay_have_identical_predictions_and_model_decisions(tmp_path,monkeypatch):
    results=[]
    for mode in ('reuse','replay'):
        t,_,_,incoming,factors=history(tmp_path/(mode+'.db'),cap=1000)
        if mode=='replay':monkeypatch.setattr(t,'_transfer_state',lambda *a:None)
        result=step(t,incoming,factors,time=1_300_000,event='join');results.append(result)
        t.close()
    assert results[0].prediction_json==results[1].prediction_json
    for key in ('factor_rows_sha256','model_regret_upper','log_partition_upper'):
        assert results[0].audit[key]==results[1].audit[key]
    for left,right in zip(results[0].audit['components'],results[1].audit['components']):
        for key in ('active','decision','branches','output_sha256','log_retained','eta_upper'):
            assert left[key]==right[key]
    assert results[0].audit['state_updates']<results[1].audit['state_updates']


def test_partial_cache_transfer_is_part_of_the_same_rollback_transaction(tmp_path,monkeypatch):
    t,_,_,incoming,factors=history(tmp_path/'rollback.db');before=tuple(t.db.iterdump())
    transfer=t._transfer_state
    def fail(*args):
        transfer(*args)
        raise RuntimeError('injected after cache copy')
    monkeypatch.setattr(t,'_transfer_state',fail)
    with pytest.raises(RuntimeError,match='injected after cache copy'):
        step(t,incoming,factors,time=1_300_000,event='join')
    assert tuple(t.db.iterdump())==before
    monkeypatch.setattr(t,'_transfer_state',transfer)
    accepted=step(t,incoming,factors,time=1_300_000,event='join')
    sha=t.close();t=PersistentBeamTracker.open(tmp_path/'rollback.db',expected_database_sha256=sha,
        expected_prediction_sha256=accepted.prediction['commit_sha256'])
    assert step(t,incoming,factors,time=1_300_000,event='join').prediction_json==accepted.prediction_json
    t.close()


def test_missing_old_cache_remains_exact_lazy_replay_not_a_fabricated_state(tmp_path):
    t,_,_,incoming,factors=history(tmp_path/'missing.db',cap=1000)
    for component in t.store.live():t.db.execute(f'DELETE FROM pc{component}_states')
    t.db.commit()
    result=step(t,incoming,factors,time=1_300_000,event='join')
    assert result.audit['merge_state_cache_copies']==0 and result.audit['state_updates']>12
    assert result.prediction['predictions']
    t.close()
