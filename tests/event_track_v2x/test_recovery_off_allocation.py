"""Actual restricted teacher, fit, checkpoint and learned solver fixtures."""
import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from test_recovery_off_tracking import obs, step, global_roots
from test_paper_pipeline import native_frame
from transvision.models.event_track_v2x.allocation_policy import FrozenPriorityPolicy, FEATURES
from transvision.models.event_track_v2x.allocation_training import (
    backend_sources, training_binding, validate_backend_binding, fit_priority, load_priority,
)
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.exclusive_completion_tracking import PersistentExclusiveCompletionConfig
from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery
from transvision.models.event_track_v2x.forest_potentials import LearnedForestScorer
from transvision.models.event_track_v2x.forest_tracking import PaperForestTrackingConfig
from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
from transvision.models.event_track_v2x.paper_native_cache import NativePaperCache, write_native_cache
from transvision.models.event_track_v2x.paper_protocol import PaperProtocol
from transvision.models.event_track_v2x import recovery_off_paper_runtime as runtime
from transvision.models.event_track_v2x.recovery_off_allocation import (
    RecoveryOffTeacher, RecoveryOffLearned, decision_bound, SCOPE,
)
from transvision.models.event_track_v2x.recovery_off_tracking import RecoveryOffConfig, RecoveryOffTracker
from tools.event_track_v2x.train_paper_priority import export


def policy():
    return FrozenPriorityPolicy([np.zeros((2,len(FEATURES))), np.zeros(2), np.zeros((1,2)), np.zeros(1)])


def config(budget=10, width=1):
    return RecoveryOffConfig(state=PaperForestTrackingConfig(active_limit=width,
        expansion_budget=budget, decision_mode='all-legal-hamming', max_model_regret=1.))


@pytest.mark.parametrize('cls', [RecoveryOffTeacher, RecoveryOffLearned])
def test_training_or_learned_order_never_recovers_pruned_roots(tmp_path, cls):
    kwargs = {'allocation_policy':policy()} if cls is RecoveryOffLearned else {}
    tracker = cls(tmp_path/'actual.db', sequence_id='0003', config=config(), **kwargs)
    try:
        first = step(tracker, [obs('a'),obs('b',source=1)],
            [((-1,0.),),((-1,0.),(0,-8.))])
        assert global_roots(tracker, first)==(0,1)
        second = step(tracker,time=1_200_000,event='rescore',
            rescored_rows=[(1,((-1,-15.),(0,15.)))])
        assert global_roots(tracker,second)==(0,1)
        assert second.audit['original_model_risk_certified'] is False
        assert second.audit['model_regret_upper']==1.
        assert second.audit['priority_decision_support_scope']==SCOPE
        for event in (first, second):
            for trace in event.audit['allocation_trace']:
                if cls is RecoveryOffLearned:
                    record = trace['priority_selection']
                    candidates=record['candidates']
                    assert record['priority_is_bound'] is False
                    assert all(r['priority_score']==0. for r in candidates)
                    assert trace['component']==min(r['component'] for r in candidates)
                else:
                    record=trace['allocation_training']
                    assert record['labels_are_model_not_true_risk']
                    for row in record['candidates']:
                        assert row['model_bound_scope']==SCOPE
                        expected=row['features'][0]*(row['model_bound_before']-row['model_bound_after'])/max(1,row['charged_steps'])
                        assert row['target']==pytest.approx(expected,abs=1e-12)
        saved=tuple(tracker.db.iterdump())
        k=next(iter(tracker.kernels.values()))
        fallback=second.audit['components'][0]['output_handle']
        result=decision_bound(k,k._mass(),tuple(range(k.n)),fallback)
        assert result['risk_scope']==SCOPE and not result['original_model_risk_certified']
        assert tuple(tracker.db.iterdump())==saved
        assert step(tracker,[obs('a'),obs('b',source=1)],
            [((-1,0.),),((-1,0.),(0,-8.))]).prediction_json==first.prediction_json
    finally:
        tracker.close()


def test_teacher_probe_rolls_back_and_matches_bound_predictions(tmp_path):
    teacher=RecoveryOffTeacher(tmp_path/'teacher.db',sequence_id='0003',config=config(3,2))
    bound=RecoveryOffTracker(tmp_path/'bound.db',sequence_id='0003',config=config(3,2))
    raw=[obs('a'),obs('b',source=1),obs('c',time=1_010_000),obs('d',source=1,time=1_010_000)]
    rows=[((-1,0.),),((-1,0.),(0,.2)),((-1,0.),),((-1,0.),(2,.3))]
    try:
        pairs=[(step(teacher,raw,rows),step(bound,raw,rows)),
               (step(teacher,time=1_200_000,event='empty'),step(bound,time=1_200_000,event='empty'))]
        assert sum(a.audit['teacher_probe_executions'] for a,_ in pairs)>0
        for a,b in pairs:
            assert a.prediction_json==b.prediction_json
            assert a.audit['factor_rows_sha256']==b.audit['factor_rows_sha256']
            assert a.audit['restricted_support_model_regret_upper']==b.audit['restricted_support_model_regret_upper']
            assert [c['active'] for c in a.audit['components']]==[c['active'] for c in b.audit['components']]
            assert [c['frontier'] for c in a.audit['components']]==[c['frontier'] for c in b.audit['components']]
    finally:
        teacher.close(); bound.close()


def test_teacher_work_cap_failure_rolls_back_entire_event(tmp_path, monkeypatch):
    teacher=RecoveryOffTeacher(tmp_path/'cap.db',sequence_id='0003',config=config(3,2))
    try:
        before=tuple(teacher.db.iterdump())
        monkeypatch.setattr(teacher,'MAX_TEACHER_STEPS',0)
        with pytest.raises(ValueError,match='teacher work capacity'):
            step(teacher,[obs('a'),obs('b',source=1)],
                 [((-1,0.),),((-1,0.),(0,.2))])
        assert tuple(teacher.db.iterdump())==before
    finally:
        teacher.close()


def test_learned_actual_priority_scores_choose_component_and_preserve_budget(tmp_path):
    weights=np.zeros((2,len(FEATURES))); weights[0,7]=1.  # causal log-node count
    learned_policy=FrozenPriorityPolicy([weights,np.zeros(2),np.array([[1.,0.]]),np.zeros(1)])
    tracker=RecoveryOffLearned(tmp_path/'order.db',sequence_id='0003',config=config(3,2),allocation_policy=learned_policy)
    try:
        result=step(tracker,[obs('a'),obs('b',source=1),obs('c',time=1_010_000),
            obs('d',source=1,time=1_010_000),obs('e',time=1_020_000)],
            [((-1,0.),),((-1,0.),(0,.2)),((-1,0.),),((-1,0.),(2,.3)),((-1,0.),(2,.4))])
        options=[r for r in result.audit['allocation_trace'] if len(r['priority_selection']['candidates'])>1]
        assert options
        assert any(len({row['priority_score'] for row in event['priority_selection']['candidates']})>1 for event in options)
        for event in result.audit['allocation_trace']:
            candidates=event['priority_selection']['candidates']
            predicted=learned_policy.scores([r['features'] for r in candidates])
            assert [r['priority_score'] for r in candidates]==list(predicted)
            chosen=min(candidates,key=lambda r:(-r['priority_score'],r['component']))
            assert event['component']==chosen['component']
        assert result.audit['search_steps']<=result.audit['search_budget']==3
        db_sha=tracker.close()
        tracker=RecoveryOffLearned.open(tmp_path/'order.db',expected_database_sha256=db_sha,
            expected_prediction_sha256=result.prediction['commit_sha256'],allocation_policy=learned_policy)
        repeat=step(tracker,time=1_300_000,event='empty')
        assert repeat.audit['priority_decision_support_scope']==SCOPE
        assert not repeat.audit['recovery_enabled']
    finally:
        tracker.close()


def test_source_config_binding_rejects_unrestricted_policy_and_changed_limits():
    c=config()
    binding=training_binding(c,'a'*64,{'fixture':True})
    assert binding['configuration']['recovery_off_version']==1
    assert binding['model_progress_support_scope']==SCOPE
    assert not binding['unrestricted_priority_checkpoint_transfer']
    for module in ('recovery_off_allocation','recovery_off_tracking','recovery_off_paper_runtime'):
        assert 'transvision/models/event_track_v2x/'+module+'.py' in binding['backend_implementation_sha256']
    validate_backend_binding(binding)
    incorrect=copy.deepcopy(binding)
    incorrect['backend_implementation_sha256'].pop('transvision/models/event_track_v2x/recovery_off_allocation.py')
    with pytest.raises(ValueError,match='source binding'):
        validate_backend_binding(incorrect)
    for key in ('model_progress_support_scope','unrestricted_priority_checkpoint_transfer'):
        incorrect=copy.deepcopy(binding); incorrect.pop(key)
        with pytest.raises(ValueError,match='conditional model-progress'):
            validate_backend_binding(incorrect)
    assert training_binding(PersistentExclusiveCompletionConfig(state=c.state),'a'*64,{'fixture':True})!=binding
    with pytest.raises(ValueError,match='recovery-off'):
        backend_sources(dict(asdict(c),recovery_off_version=True))


def test_real_teacher_export_fit_load_and_learned_replay(tmp_path):
    frames=[native_frame(sequence_id=seq,side=side,agent_mask=i+1,dataset_split='train')
        for seq in ('0003','0007') for i,side in enumerate(('vehicle-side','infrastructure-side'))]
    root=tmp_path/'cache'
    digest=write_native_cache(root,frames,split='train',producer={'fit_split':'train'},fixture=True)
    cache=NativePaperCache(root,digest)
    events=[]
    for seq in ('0003','0007'):
        deliveries=[]
        for (scene,side,fid),(entry,metadata) in cache.index.items():
            if scene!=seq: continue
            entry,metadata=json.loads(entry),json.loads(metadata)
            arrival=max(metadata['box_reference_timestamp_us'],metadata['source_image_timestamp_us'])+100_000
            deliveries.append(asdict(CacheDelivery(seq,side,fid,arrival,entry['frame_sha256'])))
        reference=max(d['arrival_us'] for d in deliveries)
        events.append(dict(sequence_id=seq,frame_id='first',reference_us=reference,decision_us=reference,event_id='first',deliveries=deliveries))
    torch.manual_seed(1337)
    scorer=LearnedForestScorer(RecoverableIdentityModel(hidden=8,heads=2,dropout=0.).eval().requires_grad_(False))
    protocol=PaperProtocol('v2v4real','train')
    model_binding=dict(candidate_protocol=protocol.candidates,dataset=protocol.dataset,fit_split='train',frozen_cache_identity={'fixture':'synthetic-train-only'})
    configuration=runtime.default_configuration(allocation='teacher')
    configuration['state']['expansion_budget']=1
    teacher=tmp_path/'teacher'
    runtime.replay(cache,events,teacher,protocol=protocol,configuration=configuration,scorer=scorer,model_binding=model_binding,fixture=True)
    data=tmp_path/'data'; manifest=export(teacher,sha_file(teacher/'receipt.json'),data)
    assert sum(r['groups'] for r in manifest['shards'])>0
    assert not manifest['full_official_train_trace']
    fitted=fit_priority(data,sha_file(data/'manifest.json'),tmp_path/'fit',epochs=1,hidden=4,require_full_train=False)
    p,checkpoint=load_priority(tmp_path/'fit/1337',fitted['seeds'][0]['checkpoint_sha256'],binding=manifest['binding'],require_full_train=False)
    assert checkpoint['binding']['configuration']['recovery_off_version']==1
    wrong=copy.deepcopy(manifest['binding']); wrong['configuration']['state']['expansion_budget']+=1
    with pytest.raises(ValueError,match='binding differs'):
        load_priority(tmp_path/'fit/1337',fitted['seeds'][0]['checkpoint_sha256'],binding=wrong,require_full_train=False)
    wrong=training_binding(PersistentExclusiveCompletionConfig(state=PaperForestTrackingConfig(**configuration['state']),
        **{k:v for k,v in configuration['limits'].items() if k!='recovery_off_version'}),scorer.signature,model_binding['frozen_cache_identity'])
    with pytest.raises(ValueError,match='binding differs'):
        load_priority(tmp_path/'fit/1337',fitted['seeds'][0]['checkpoint_sha256'],binding=wrong,require_full_train=False)
    result=runtime.replay(cache,events,tmp_path/'learned',protocol=protocol,configuration=dict(configuration,allocation='learned'),scorer=scorer,model_binding=model_binding,allocation_policy=p,fixture=True)
    assert result['completed_events']==2 and not result['paper_results_verified']
    rows=[json.loads(line) for line in (tmp_path/'learned/audit.jsonl').read_bytes().splitlines()]
    assert all(row['learned_allocation_policy'] and not row['recovery_enabled'] for row in rows)
    assert all(row['priority_decision_support_scope']==SCOPE for row in rows)
