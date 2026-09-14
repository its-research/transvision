from dataclasses import asdict
import json

import pytest

from tools.event_track_v2x.summarize_allocation_teacher import summarize, _captured_lines
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows
from transvision.models.event_track_v2x.completion_component_tracking import PersistentCompletionConfig
from transvision.models.event_track_v2x.beam_recovery_tracking import BeamRecoveryConfig
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from test_forest_training_data import prepared_rows


def replay_fixture(tmp_path, prepared_rows, config=None):
    _, _, cache, rows = prepared_rows
    replay = tmp_path/'teacher'
    config = config or PersistentCompletionConfig()
    replay_rows(cache, rows, replay, config, allocation_teacher=True,
        plan=dict(class_scope=['car'], configuration=asdict(config), scheduled_frames=len(rows)))
    return replay


def test_complete_teacher_summary_checks_every_target_and_full_coverage(tmp_path, prepared_rows):
    replay = replay_fixture(tmp_path, prepared_rows)
    report = summarize(replay, tmp_path/'report', receipt_sha256=sha_file(replay/'receipt.json'))
    assert not report['partial_snapshot'] and report['captured_events'] == report['scheduled_events']
    counts = report['counts']
    assert counts['target_rows'] == counts['positive_targets']+counts['negative_targets']+counts['near_zero_targets']
    assert report['snapshot']['captured_prefix_sha256'] == sha_file(replay/'tracking.jsonl')
    assert not report['producer_liveness_checked'] and not report['real_tracking_evaluation']
    assert not report['full_official_train_completion_verified'] and not report['paper_eligible']


def test_beam_teacher_summary_counts_only_extra_recovery_work(tmp_path,prepared_rows):
    replay=replay_fixture(tmp_path,prepared_rows,BeamRecoveryConfig(recovery_budget=7))
    report=summarize(replay,tmp_path/'summary',receipt_sha256=sha_file(replay/'receipt.json'))
    original=[json.loads(line)['tracking'] for line in (replay/'tracking.jsonl').read_bytes().splitlines()]
    rows=sum(len(s['allocation_training']['candidates']) for a in original for s in a['recovery_allocation_trace'])
    assert report['counts']['target_rows']==rows>0
    for event,audit in zip(report['events'],original,strict=True):
        assert event['search_scope']=='extra_beam_recovery'
        assert event['search_steps']==audit['recovery_search_steps']<=7
    original[0]['allocation_trace_field']='allocation_trace'
    (replay/'tracking.jsonl').write_text(''.join(json.dumps(dict(tracking=a))+'\n' for a in original))
    with pytest.raises(ValueError,match='trace/configuration'):
        summarize(replay,tmp_path/'bad',partial_snapshot=True)


def test_partial_snapshot_records_unfinished_tail_without_claiming_completion(tmp_path, prepared_rows):
    replay = replay_fixture(tmp_path, prepared_rows)
    path = replay/'tracking.jsonl'
    original = path.read_bytes()
    path.write_bytes(original+b'{"unfinished":')
    report = summarize(replay, tmp_path/'partial', partial_snapshot=True)
    assert report['partial_snapshot'] and report['receipt_sha256'] is None
    assert report['snapshot']['incomplete_tail_bytes'] == len(b'{"unfinished":')
    assert report['snapshot']['captured_prefix_bytes'] == len(original)
    with pytest.raises(ValueError, match='coverage or identity'):
        summarize(replay, tmp_path/'cannot-complete', receipt_sha256=sha_file(replay/'receipt.json'))
    assert not (tmp_path/'cannot-complete').exists()


def test_rewriting_captured_prefix_is_rejected(tmp_path):
    path = tmp_path/'trace'
    path.write_bytes(b'{"x":1}\n')
    state = {}
    iterator = _captured_lines(path, state)
    assert next(iterator) == {'x': 1}
    path.write_bytes(b'{"x":2}\n')
    with pytest.raises(ValueError, match='prefix changed'):
        list(iterator)


def test_duplicate_event_or_changed_label_is_not_a_valid_progress_summary(tmp_path, prepared_rows):
    replay = replay_fixture(tmp_path, prepared_rows)
    path = replay/'tracking.jsonl'
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b''.join(lines+[lines[0]]))
    with pytest.raises(ValueError, match='duplicate'):
        summarize(replay, tmp_path/'duplicate', partial_snapshot=True)
    parsed = [json.loads(line) for line in lines]
    changed = False
    for record in parsed:
        for step in record['tracking']['allocation_trace']:
            if step['allocation_training']['candidates']:
                step['allocation_training']['candidates'][0]['target'] = .731
                changed = True
                break
        if changed:
            break
    assert changed
    path.write_text(''.join(json.dumps(record)+'\n' for record in parsed))
    with pytest.raises(ValueError, match='target arithmetic'):
        summarize(replay, tmp_path/'changed-label', partial_snapshot=True)
