from dataclasses import replace
import json

import pytest

from tools.event_track_v2x.build_detection_cache_v2 import build_cache
from transvision.models.event_track_v2x.detection_cache_v2 import DetectionCacheV2, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import (
    CacheDelivery, CausalForestCacheStream, VerifiedForestCache,
)
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
from transvision.models.event_track_v2x.forest_tracking import CausalForestTracker, ForestTrackingConfig
from test_detection_cache_v2 import sources


@pytest.fixture
def stream(sources):
    roots, calibration, inputs, output = sources
    sha = build_cache(roots, calibration, sha_file(calibration), inputs, output)
    cache = VerifiedForestCache(output, sha)
    tracker = CausalForestTracker(sequence_id='0003', start_us=1_000_000,
        scorer=GeometryForestScorer(), config=ForestTrackingConfig(max_nodes=16, max_model_regret=1.))
    return CausalForestCacheStream(cache, tracker, origin_us=1_000_000)


def delivery(stream, side='vehicle-side', arrival=1_100_000, sequence='0003', frame='000120'):
    entry = json.loads(stream.cache.index[(sequence, side, frame)][0])
    return CacheDelivery(sequence, side, frame, arrival, entry['frame_sha256'])


def step(stream, deliveries, number=0, reference=None):
    reference = 1_100_000 + number * 100_000 if reference is None else reference
    return stream.step(deliveries, frame_id=str(number), reference_us=reference,
                       decision_us=reference, event_id='event-' + str(number))


def test_v2_stream_first_receipt_is_preserved_and_duplicate_does_not_reload(stream, monkeypatch):
    d = delivery(stream)
    first = step(stream, [d])
    old_observations = stream.tracker.observations
    monkeypatch.setattr(stream.cache, 'load_arrived', lambda *a: pytest.fail('duplicate reloaded'))
    repeated = step(stream, [replace(d, arrival_us=1_200_000)], 1)
    assert stream.tracker.observations == old_observations
    assert len(stream.seen) == 1
    assert repeated.ingestion_audit['new_observations'] == 0
    assert repeated.ingestion_audit['duplicate_deliveries'][0]['first_arrival_us'] == 1_100_000
    assert first.prediction['commit_sha256'] == repeated.prediction['previous_commit_sha256']
    assert step(stream, [d]) is first
    copy = first.prediction
    copy['predictions'].clear()
    assert first.prediction['predictions']


def test_two_sources_are_passed_once_and_in_batch_arrival_order(stream):
    left = delivery(stream, arrival=1_110_000)
    right = delivery(stream, 'infrastructure-side', arrival=1_100_000)
    result = step(stream, [left, right, replace(left, arrival_us=1_120_000)], reference=1_120_000)
    assert result.ingestion_audit['new_observations'] == 4
    assert len(result.ingestion_audit['duplicate_deliveries']) == 1
    assert {o.node.source_id for o in stream.tracker.observations} == {0, 1}
    assert [o.node.arrival_us for o in stream.tracker.observations] == [1_100_000]*2 + [1_110_000]*2
    assert all(t['class_label'] == 'car' for t in result.prediction['predictions'])


@pytest.mark.parametrize('change', [
    {'arrival_us': 1_200_000}, {'arrival_us': 1_099_999},
    {'frame_sha256': 'f'*64}, {'frame_id': 'missing'},
])
def test_bad_receipt_is_rejected_before_payload_read(stream, monkeypatch, change):
    d = replace(delivery(stream), **change)
    monkeypatch.setattr(DetectionCacheV2, 'load', lambda *a: pytest.fail('unavailable frame read'))
    with pytest.raises(ValueError):
        step(stream, [d])
    assert stream.seen == {} and stream.tracker.observations == ()


def test_cross_sequence_is_rejected_even_inside_verified_cache(stream, monkeypatch):
    d = delivery(stream, sequence='0007', frame='000121', arrival=1_100_001)
    monkeypatch.setattr(DetectionCacheV2, 'load', lambda *a: pytest.fail('cross-sequence payload read'))
    with pytest.raises(ValueError, match='cross-sequence'):
        step(stream, [d], reference=1_100_001)


def test_model_failure_does_not_commit_receipt_and_retry_is_identical(stream, monkeypatch):
    d = delivery(stream)
    original = stream.tracker.step
    def fail(*args, **kwargs):
        raise RuntimeError('temporary inference failure')
    monkeypatch.setattr(stream.tracker, 'step', fail)
    with pytest.raises(RuntimeError):
        step(stream, [d])
    assert stream.seen == stream.events == {}
    assert stream.last_decision_us == -1
    monkeypatch.setattr(stream.tracker, 'step', original)
    result = step(stream, [d])
    fresh_tracker = CausalForestTracker(sequence_id='0003', start_us=1_000_000,
        scorer=GeometryForestScorer(), config=stream.tracker.config)
    fresh = CausalForestCacheStream(stream.cache, fresh_tracker, origin_us=1_000_000)
    assert result == step(fresh, [d])


def test_payload_changed_after_initial_audit_is_rejected_without_commit(stream):
    d = delivery(stream)
    entry, _ = stream.cache.describe(d)
    path = stream.cache.root / entry['arrays']['path']
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='payload identity'):
        step(stream, [d])
    assert stream.seen == {} and stream.tracker.commits == ()


def test_storage_budget_precedes_payload_read(stream, monkeypatch):
    bounded = CausalForestCacheStream(stream.cache, stream.tracker, origin_us=1_000_000, max_receipts=1)
    monkeypatch.setattr(DetectionCacheV2, 'load', lambda *a: pytest.fail('storage overflow read'))
    with pytest.raises(ValueError, match='receipt storage'):
        step(bounded, [delivery(stream), delivery(stream, 'infrastructure-side')])
    assert bounded.seen == {}


def test_withheld_new_frame_and_revised_first_arrival_rejected(stream):
    d = delivery(stream, arrival=1_150_000)
    step(stream, [d], reference=1_150_000)
    with pytest.raises(ValueError, match='revise the first'):
        step(stream, [replace(d, arrival_us=1_120_000)], 1)
    with pytest.raises(ValueError, match='withheld'):
        step(stream, [delivery(stream, 'infrastructure-side')], 1)


def test_conflicting_event_and_changed_selection_rejected(stream):
    d = delivery(stream)
    step(stream, [d])
    with pytest.raises(ValueError, match='conflicting duplicate'):
        step(stream, [])
    stream.maximum_detections = 1
    with pytest.raises(ValueError, match='configuration'):
        step(stream, [], 1)


def test_cache_rejects_mixed_feature_checkpoint_even_with_rehashed_cohort(stream):
    # A cohort with internally valid frame hashes can still mix incompatible
    # embeddings. Patch the already verified inventory read for this unit only.
    from unittest.mock import patch
    real_loads = json.loads
    count = [0]
    def changed_loads(payload, *args, **kwargs):
        value = real_loads(payload, *args, **kwargs)
        if isinstance(value, dict) and value.get('kind') == 'detection_cache_v2':
            count[0] += 1
            if value['side'] == 'infrastructure-side':
                value['feature_checkpoint_sha256'] = 'e'*64
        return value
    entries = [json.loads(v[0]) for v in stream.cache.index.values()]
    with patch('transvision.models.event_track_v2x.forest_cache_stream.load_manifest',
               return_value=(json.loads(stream.cache.manifest_json), entries)), patch(
               'transvision.models.event_track_v2x.forest_cache_stream.json.loads', side_effect=changed_loads):
        with pytest.raises(ValueError, match='feature space'):
            VerifiedForestCache(stream.cache.root, stream.cache.manifest_sha256)
    assert count[0] == 4
