from dataclasses import replace
import json

import numpy as np
import pytest

from transvision.models.event_track_v2x.detection_cache_v2 import DetectionCacheV2, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
from transvision.models.event_track_v2x.forest_row_context import build_row_contexts
from transvision.models.event_track_v2x.forest_supervision import AnnotationIdentity, AnnotationIdentityIndex
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig, RawIdentityDetection, cache_detections
from transvision.models.event_track_v2x.identity_forest import IdentityNode
from transvision.models.event_track_v2x.forest_training_data import (
    FrameSupervision, TrainingShard, exact_cooperative_links, frame_from_converted,
    labels_for_frame, prepare_training_rows,
)
from transvision.models.event_track_v2x.persistent_cache_stream import RowContextScorer
from transvision.models.event_track_v2x.persistent_forest import PersistentForestTracker
from test_detection_cache_v2 import _build, _frame, _sources


@pytest.fixture
def prepared_rows(tmp_path):
    source = _sources(tmp_path, split='train')
    cache = VerifiedForestCache(source[-1], _build(source))
    frames, annotations, links = {}, [], []
    for (scene, side, frame_id), (entry, meta) in cache.index.items():
        frame = DetectionCacheV2.load(cache.root, json.loads(entry))
        source_id = 0 if side == 'vehicle-side' else 1
        ann = tuple(AnnotationIdentity(scene, source_id, frame_id, str(i), side+'-'+str(i)) for i in range(frame.count))
        annotations.extend(ann)
        frames[(side, frame_id)] = FrameSupervision(scene, source_id, frame_id,
            frame.metadata['box_reference_timestamp_us'], frame.states[:, :7], ann)
    schedule = []
    for sequence in json.loads(cache.manifest_json)['sequences']:
        pair = [next(f for f in frames.values() if f.sequence_id == sequence and f.source_id == side) for side in (0, 1)]
        links.extend((a.reference, b.reference) for a, b in zip(pair[0].annotations, pair[1].annotations))
        schedule.append(dict(sequence_id=sequence, vehicle_frame=pair[0].frame_id,
            infrastructure_frame=pair[1].frame_id, box_reference_timestamp_us=pair[0].timestamp_us))
    index = AnnotationIdentityIndex(annotations, links)
    output = tmp_path/'train-rows'
    manifest = prepare_training_rows(cache, schedule, frames, index, output, provenance={'fixture_only': True})
    return output, manifest, cache, schedule


def test_shards_preserve_prediction_context_and_exclude_gt_payload(prepared_rows, tmp_path):
    output, manifest, cache, rows = prepared_rows
    assert manifest['split'] == 'train' and manifest['scheduled_frames'] == 2
    assert manifest['ingested_source_frames'] == 4 and sum(r['nodes'] for r in manifest['shards']) == 6
    assert manifest['gt_boxes_or_ids_in_shards'] is False
    for record in manifest['shards']:
        shard = TrainingShard(output/record['path'], record, 8)
        assert not any(k.startswith('gt_') or k == 'target_id' for k in shard.arrays)
        contexts = [shard.example(i)[0] for i in range(record['nodes'])]
        raw = tuple(context.observations[-1] for context in contexts)
        t = PersistentForestTracker(tmp_path/(record['sequence_id']+'.sqlite'), sequence_id=record['sequence_id'])
        _, indices = RowContextScorer(t, GeometryForestScorer()).rows(raw, contexts[0].decision_us)
        assert indices == tuple(c.indices for c in contexts)
        for i, context in enumerate(contexts):
            for index, observation in zip(context.indices, context.observations):
                assert observation == raw[index]
        t.close()


def test_full_raw_match_not_recomputed_after_selection():
    frame = _frame(2)
    raw = cache_detections(frame, arrival_us=1_100_000, decision_us=1_100_000, origin_us=1_000_000)
    annotation = AnnotationIdentity('0003', 0, '000123', '1', 'token')
    gt = FrameSupervision('0003', 0, '000123', 1_000_000, frame.states[:1, :7], (annotation,))
    index = AnnotationIdentityIndex([annotation], [])
    assert labels_for_frame(frame, gt, index, [raw[0]])[0].status == 'matched'
    assert labels_for_frame(frame, gt, index, [raw[1]])[0].status == 'unmatched_prediction'
    with pytest.raises(ValueError, match='alignment'):
        labels_for_frame(frame, replace(gt, timestamp_us=2_000_000), index, raw)


def test_source_gt_conversion_does_not_read_velocity_or_future_targets():
    info = dict(scene_token='0003', token='frame', timestamp=1_000_000, gt_boxes=np.ones((1, 7)),
                gt_names=['car'], gt_inds=[1], anno_tokens=['token'], gt_velocity=[[np.nan, np.nan]],
                next_info={'future': 'must not enter'})
    frame = frame_from_converted(info, 0)
    assert frame.annotations[0].track_id == '000001'
    assert not hasattr(frame, 'gt_velocity') and not hasattr(frame, 'next_info')
    empty = frame_from_converted(dict(info, gt_boxes=[], gt_names=[], gt_inds=[], anno_tokens=[]), 0)
    assert empty.boxes.shape == (0, 7) and empty.annotations == ()
    with pytest.raises(ValueError, match='track IDs'):
        frame_from_converted(dict(info, gt_inds=[1.5]), 0)


def test_exact_cooperative_tokens_missing_side_and_repeat_validation():
    a = AnnotationIdentity('0003', 0, 'left', '000001', 'a')
    b = AnnotationIdentity('0003', 1, 'right', '000004', 'b')
    left = FrameSupervision('0003', 0, 'left', 1, np.ones((1, 7)), (a,))
    right = FrameSupervision('0003', 1, 'right', 1, np.ones((1, 7)), (b,))
    label = dict(veh_frame_id='left', inf_frame_id='right', veh_track_id='000001', inf_track_id='000004', veh_token='a', inf_token='b')
    assert exact_cooperative_links([label], left, right) == ((a.reference, b.reference),)
    assert exact_cooperative_links([dict(label, inf_track_id='-1', inf_token='-1')], left, right) == ()
    for bad in ([label, label], [dict(label, inf_track_id='-1')], [dict(label, veh_token='wrong')]):
        with pytest.raises(ValueError):
            exact_cooperative_links(bad, left, right)


def test_shard_mutation_rejected(prepared_rows):
    output, manifest, _, _ = prepared_rows
    record = manifest['shards'][0]
    path = output/record['path']
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='identity changed'):
        TrainingShard(path, record, 8)


def test_shard_reuses_only_immutable_prediction_observations(prepared_rows, monkeypatch):
    from transvision.models.event_track_v2x import forest_tracking
    output, manifest, _, _ = prepared_rows
    record = manifest['shards'][0]
    shard = TrainingShard(output/record['path'], record, 8)
    calls = []
    original = forest_tracking._state

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(forest_tracking, '_state', counted)
    first = [shard.example(i) for i in range(record['nodes'])]
    assert len(calls) == record['nodes']
    for row in reversed(range(record['nodes'])):
        context, target = shard.example(row)
        assert target == first[row][1]
        assert context.indices == first[row][0].indices
        assert all(raw is first[i][0].observations[-1] for i, raw in zip(context.indices, context.observations))
    assert len(calls) == record['nodes']
    # Reconstruct directly from persisted arrays as an independent reference;
    # reuse must not change any raw field or introduce target/GT information.
    a = shard.arrays
    for i in range(record['nodes']):
        direct = RawIdentityDetection(record['sequence_id'],
            IdentityNode(str(a['node_id'][i]), int(a['source'][i]), int(a['information_us'][i]),
                         int(a['arrival_us'][i]), str(a['frame_id'][i])),
            int(a['detection_index'][i]), int(a['state_us'][i]), a['mean'][i], a['covariance'][i],
            float(a['score'][i]), a['features'][i], str(a['cache_sha256'][i]))
        assert direct == first[i][0].observations[-1]
    for value in (*shard.arrays.values(), shard.valid_indices):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(TypeError):
        shard.arrays['mean'] = np.zeros_like(a['mean'])
    another = TrainingShard(output/record['path'], record, 8)
    assert another.example(0)[0].observations[-1] is not first[0][0].observations[-1]
