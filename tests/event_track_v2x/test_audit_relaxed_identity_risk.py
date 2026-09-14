import pytest

from tools.event_track_v2x.audit_relaxed_identity_risk import run
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.learned_component_allocation import AllocationTeacherTracker
from transvision.models.event_track_v2x.completion_component_tracking import CompletionTeacherTracker, PersistentCompletionConfig
from test_learned_component_allocation import config, scene
from test_persistent_component_tracking import step


@pytest.mark.parametrize('completion', [False, True])
@pytest.mark.parametrize('capacity', [2, 1024])
def test_risk_diagnostic_copies_source_and_reports_all_components_including_capacity_rejections(tmp_path, completion, capacity):
    cls = CompletionTeacherTracker if completion else AllocationTeacherTracker
    cfg = PersistentCompletionConfig() if completion else config(4)
    path = tmp_path/'source.sqlite'
    tracker = cls(path, sequence_id='0003', config=cfg)
    raw, rows = scene()
    first = step(tracker, raw, rows)
    digest = tracker.close()
    result = run(path, digest, first.prediction['commit_sha256'], tmp_path/'audit', maximum_nodes=capacity)
    assert len(result['components']) == len(first.audit['components'])
    assert result['committed_events'] == 1 and sha_file(path) == digest
    assert not result['action_selection_changed'] and not result['new_inputs_or_gt_read']
    assert not result['tracking_validation'] and not result['paper_eligible']
    for component in result['components']:
        if component['nodes'] > capacity:
            assert not component['computed'] and component['model_regret_upper'] == 1.
        else:
            assert component['computed'] and sum(a['is_stored_output'] for a in component['actions']) == 1
    with pytest.raises(ValueError, match='closed sealed'):
        run(path, '0'*64, first.prediction['commit_sha256'], tmp_path/'wrong-digest')
    assert not (tmp_path/'wrong-digest').exists()
