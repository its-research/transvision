import json
from dataclasses import replace

import pytest

from tools.event_track_v2x.collect_allocation_training import select_teacher_schedule
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows
from tools.event_track_v2x.train_forest_identity import FitConfig,fit_dataset
from transvision.models.event_track_v2x.allocation_training import (
    allocation_sources,export_training,fit_priority,load_priority,training_binding,
)
from transvision.models.event_track_v2x.completion_component_tracking import PersistentCompletionConfig,CompletionLearnedTracker
from transvision.models.event_track_v2x.covered_completion_tracking import (
    PersistentCoveredCompletionConfig, CoveredCompletionLearned,
)
from transvision.models.event_track_v2x.beam_recovery_tracking import BeamRecoveryConfig
from transvision.models.event_track_v2x.beam_recovery_allocation import LearnedBeamRecoveryTracker
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
from transvision.models.event_track_v2x.persistent_component_tracking import PersistentComponentConfig
from test_forest_training_data import prepared_rows
from test_run_persistent_forest_v2 import replay_inputs
from test_detection_cache_v2 import sources


@pytest.mark.parametrize('config_type,tracker_type', [
    (PersistentCompletionConfig, CompletionLearnedTracker),
    (PersistentCoveredCompletionConfig, CoveredCompletionLearned),
    (BeamRecoveryConfig, LearnedBeamRecoveryTracker),
])
def test_completion_two_head_pipeline_uses_same_frozen_factors_and_strict_binding(prepared_rows,tmp_path,config_type,tracker_type):
    data,_,cache,rows=prepared_rows
    fit_dir=tmp_path/'identity'
    fitted=fit_dataset(data,sha_file(data/'manifest.json'),fit_dir,
        config=FitConfig(epochs=1,batch_size=4,hidden=8,heads=2,dropout=0.),require_full_train=False)
    cp=fitted['seeds'][0]
    config=config_type()
    scorer,_=load_identity_checkpoint((fit_dir/cp['checkpoint_manifest']).parent,cp['checkpoint_sha256'],config=config.state)
    binding=training_binding(config,scorer.signature,frozen_cache_identity(cache))
    teacher,data_dir,priority=(tmp_path/p for p in ('teacher','priority-data','priority-fit'))
    result=replay_rows(cache,rows,teacher,config,learned_scorer=scorer,allocation_teacher=True,
        plan=dict(source_sha256=allocation_sources(),allocation_training_binding=binding,full_official_train_verified=False))
    assert result['frontier_completion_enabled'] and result['allocation_teacher']
    assert result['coverage_aware_proposal_admission'] == (config_type in (PersistentCoveredCompletionConfig,BeamRecoveryConfig))
    exported=export_training(teacher,sha_file(teacher/'receipt.json'),data_dir)
    assert not exported['full_official_train_trace']
    with pytest.raises(ValueError,match='fixture is not full train'):
        fit_priority(data_dir,sha_file(data_dir/'manifest.json'),tmp_path/'cannot-claim-full-train')
    priority_fit=fit_priority(data_dir,sha_file(data_dir/'manifest.json'),priority,epochs=1,hidden=4,require_full_train=False)
    head=priority_fit['seeds'][0]
    policy,_=load_priority(priority/'1337',head['checkpoint_sha256'],binding=binding,require_full_train=False)
    old_binding=training_binding(PersistentComponentConfig(),scorer.signature,frozen_cache_identity(cache))
    with pytest.raises(ValueError,match='tracking binding'):
        load_priority(priority/'1337',head['checkpoint_sha256'],binding=old_binding,require_full_train=False)
    if config_type is PersistentCoveredCompletionConfig:
        conservative_binding=training_binding(PersistentCompletionConfig(),scorer.signature,frozen_cache_identity(cache))
        with pytest.raises(ValueError,match='tracking binding'):
            load_priority(priority/'1337',head['checkpoint_sha256'],binding=conservative_binding,require_full_train=False)
    learned=tmp_path/'learned-completion'
    result=replay_rows(cache,rows,learned,config,learned_scorer=scorer,allocation_policy=policy)
    assert result['frontier_completion_enabled'] and result['learned_allocation_enabled'] and not result['paper_eligible']
    ordinary=tmp_path/'ordinary'
    replay_rows(cache,rows,ordinary,PersistentComponentConfig(),learned_scorer=scorer)
    def factors(path):
        return [json.loads(line)['tracking']['factor_rows_sha256'] for line in (path/'tracking.jsonl').read_bytes().splitlines()]
    assert factors(teacher)==factors(learned)==factors(ordinary)
    for head in result['sequence_heads'].values():
        t=tracker_type.open(learned/head['database'],allocation_policy=policy,
            expected_prediction_sha256=head['prediction_sha256'],expected_database_sha256=head['database_sha256'])
        t.close()


def test_development_schedule_preserves_whole_selected_sequence_without_full_train_claim():
    rows=[dict(sequence_id=s,frame=i) for s in ('0000','0001') for i in range(3)]
    selected,full=select_teacher_schedule(rows)
    assert selected is rows and full
    selected,full=select_teacher_schedule(rows,'0000')
    assert selected==rows[:3] and not full
    for unknown in ('test','0002',0):
        with pytest.raises(ValueError,match='verified train schedule'):
            select_teacher_schedule(rows,unknown)


def test_coverage_teacher_variant_requires_explicit_completion_mode():
    from tools.event_track_v2x.collect_allocation_training import teacher_config_type
    assert teacher_config_type() is PersistentComponentConfig
    assert teacher_config_type(frontier_completion=True) is PersistentCompletionConfig
    assert teacher_config_type(frontier_completion=True,coverage_aware_admission=True) is PersistentCoveredCompletionConfig
    with pytest.raises(ValueError,match='requires frontier completion'):
        teacher_config_type(coverage_aware_admission=True)
    with pytest.raises(ValueError,match='boolean'):
        teacher_config_type(frontier_completion=1)


@pytest.mark.parametrize('config_type',[PersistentCompletionConfig,PersistentCoveredCompletionConfig,BeamRecoveryConfig])
def test_completion_teacher_rejects_validation_before_output(replay_inputs,tmp_path,config_type):
    cache,rows=replay_inputs
    with pytest.raises(ValueError,match='actual sealed train'):
        replay_rows(cache,rows,tmp_path/'forbidden-teacher',config_type(),allocation_teacher=True)
    assert not (tmp_path/'forbidden-teacher').exists()


def test_beam_teacher_flags_are_exclusive():
    from tools.event_track_v2x.collect_allocation_training import teacher_config_type
    assert teacher_config_type(beam_recovery=True) is BeamRecoveryConfig
    for options in (dict(frontier_completion=True),dict(coverage_aware_admission=True)):
        with pytest.raises(ValueError,match='exclusive'):
            teacher_config_type(beam_recovery=True,**options)


def test_beam_recovery_cache_duplicate_rejects_changed_policy(replay_inputs,tmp_path):
    from transvision.models.event_track_v2x.persistent_cache_stream import PersistentForestCacheStream
    from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
    from test_persistent_cache_stream import delivery,step
    from test_learned_component_allocation import policy
    cache,_=replay_inputs
    t=LearnedBeamRecoveryTracker(tmp_path/'bound.db',sequence_id='0003',
        config=BeamRecoveryConfig(),allocation_policy=policy())
    stream=PersistentForestCacheStream(cache,t,GeometryForestScorer(),origin_us=1_000_000)
    d=delivery(stream);first=step(stream,[d])
    assert step(stream,[d])==first
    t.allocation_policy=policy(1)
    with pytest.raises(ValueError,match='policy binding'):step(stream,[d])
    t.close()


def test_beam_teacher_and_learned_cache_share_arrived_pose_scope(prepared_rows,tmp_path):
    from transvision.models.event_track_v2x.recovery_task_scope import SCOPE_MODE
    from test_recovery_task_scope import make_poses
    from test_learned_component_allocation import policy
    _,_,cache,rows=prepared_rows
    _,poses=make_poses(cache,tmp_path/'navigation')
    config=BeamRecoveryConfig(recovery_allocation_scope=SCOPE_MODE,recovery_budget=4)
    runs=[]
    for name,options in (('teacher',dict(allocation_teacher=True)),('learned',dict(allocation_policy=policy()))):
        out=tmp_path/name
        result=replay_rows(cache,rows,out,config,ego_poses=poses,**options)
        assert result['recovery_allocation_scope']==SCOPE_MODE
        assert result['ego_pose_table_sha256']==poses.manifest_sha256
        runs.append([json.loads(line)['tracking'] for line in (out/'tracking.jsonl').read_bytes().splitlines()])
    for a,b in zip(*runs,strict=True):
        assert a['factor_rows_sha256']==b['factor_rows_sha256']
        assert a['recovery_task_scope']==b['recovery_task_scope']
        assert a['priority_label_weight_scope']==SCOPE_MODE
        assert not a['recovery_scope_changes_decision_loss']
    with pytest.raises(ValueError,match='pose table'):
        replay_rows(cache,rows,tmp_path/'missing-pose',config,allocation_teacher=True)
    with pytest.raises(ValueError,match='backend exclusively'):
        replay_rows(cache,rows,tmp_path/'disabled',replace(config,enable_recovery=False),
            ego_poses=poses,allocation_teacher=True)
