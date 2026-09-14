import json

import pytest

from tools.event_track_v2x.profile_allocation_teacher_event import profile
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.learned_component_allocation import AllocationTeacherTracker
from test_learned_component_allocation import scene, config
from test_persistent_component_tracking import step


@pytest.fixture
def sealed_teacher(tmp_path):
    path=tmp_path/'source.sqlite'
    tracker=AllocationTeacherTracker(path,sequence_id='0003',config=config())
    raw,rows=scene()
    result=step(tracker,raw,rows)
    return path,tracker.close(),result.prediction['commit_sha256']


def test_copy_profile_preserves_source_and_compares_identical_no_input_events(sealed_teacher,tmp_path):
    source,source_hash,head=sealed_teacher
    a,b=tmp_path/'cached',tmp_path/'uncached'
    for output,enabled in ((a,True),(b,False)):
        report=profile(source,source_hash,head,output,use_probe_cache=enabled)
        assert sha_file(source)==source_hash
        assert report['status']=='complete' and report['original_event_count']==1
        assert report['new_observations']==0 and report['observation_count']==6
        assert not report['scheduled_replay'] and not report['latency_benchmark'] and not report['paper_eligible']
        assert (output/'event.pstats').is_file()
    assert (a/'prediction.json').read_bytes()==(b/'prediction.json').read_bytes()
    aa,bb=(json.loads((p/'audit.json').read_bytes()) for p in (a,b))
    assert aa['allocation_trace']==bb['allocation_trace']
    assert aa['factor_rows_sha256']==bb['factor_rows_sha256']


def test_profile_rejects_unsealed_or_wrong_source_before_output(sealed_teacher,tmp_path):
    source,source_hash,head=sealed_teacher
    with pytest.raises(ValueError,match='identity'):
        profile(source,'0'*64,head,tmp_path/'wrong')
    with pytest.raises(ValueError,match='output head'):
        profile(source,source_hash,'0'*64,tmp_path/'wrong-head')
    assert not (tmp_path/'wrong').exists() and not (tmp_path/'wrong-head').exists()
    journal=source.with_name(source.name+'-journal')
    journal.touch()
    with pytest.raises(ValueError,match='close/checkpoint'):
        profile(source,source_hash,head,tmp_path/'not-sealed')
    assert not (tmp_path/'not-sealed').exists()
