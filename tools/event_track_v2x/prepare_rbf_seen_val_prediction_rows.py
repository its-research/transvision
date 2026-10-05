"""Target-free original row contexts for a fully admitted matching val cache.

This preparation preserves every original cooperative event, including events
with no newly available selected detections. It uses the declared paired 100ms
snapshot contract and does not manufacture a measured network arrival trace.
Its output is a new prediction-only schema; it never creates dummy supervision
to pass the train-only TrainingShard interface.
"""
import argparse
import ast
from bisect import bisect_left, bisect_right, insort
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np

R = Path('/Volumes/Data/test/recover-before-fuse')
V2 = R / 'artifacts/rbf-nested-seen-val-matching-V2-full-admission-v1-20261004'
CK = R / 'artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004'
NUMERIC = R / 'receipts/rbf-final-refit-three-seed-all-row-full-independent-numeric-acceptance-20261004.json'
ORIGINAL = R / 'artifacts/rbf-original-joint-identity-source-byte-readback-20261001/source'
DEST = R / 'artifacts/rbf-matching-seen-val-target-free-original-row-contexts-v1-20261004'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def new(path, value):
    with path.open('xb') as stream:
        stream.write(canonical(value))


def original_packages():
    # No import of the dirty checkout or train-only data/label preparation code.
    sys.path.insert(0, str(ORIGINAL))
    for name in ('transvision', 'transvision.models', 'transvision.models.event_track_v2x'):
        module = types.ModuleType(name)
        module.__path__ = [str(ORIGINAL.joinpath(*name.split('.')))]
        sys.modules[name] = module
    # The original geometric module also imports a legacy Torch association
    # tracker. Load only the three unchanged NumPy function definitions needed
    # by row inputs, without fabricating a Torch module or rewriting the source.
    from transvision.models.event_track_v2x.prediction_features import wrap_angle, transform_state_covariance
    from transvision.models.event_track_v2x.fusion import covariance_intersection
    path = ORIGINAL / 'transvision/models/event_track_v2x/tracking_v2.py'
    names = ('propagate', 'physical_world', 'ci')
    definitions = [node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == set(names)
    module = types.ModuleType('transvision.models.event_track_v2x.tracking_v2')
    module.__file__ = str(path)
    module.__dict__.update(np=np, wrap_angle=wrap_angle,
        transform_state_covariance=transform_state_covariance, covariance_intersection=covariance_intersection)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), module.__dict__)
    sys.modules[module.__name__] = module
    return dict(kind='prediction_input_pure_NumPy_dependency_extraction_v1',
        original_file=str(path), original_file_sha256=sha(path),
        selected_unchanged_function_AST_sha256={node.name: hashlib.sha256(
            ast.dump(node, include_attributes=False).encode()).hexdigest() for node in definitions},
        unused_legacy_association_tracker_not_loaded=True, fake_Torch_module_used=False,
        model_NN_forward_executed=False)


def serialize(path, observations, contexts, sequence, events, parent_limit):
    n, width = len(observations), parent_limit + 1
    indices = np.full((n, width), -1, dtype=np.int64)
    for index, context in enumerate(contexts):
        assert context.indices[-1] == index and len(context.indices) <= width
        indices[index, :len(context.indices)] = context.indices
    assert len(contexts) == n
    values = dict(
        features=np.asarray([o.features for o in observations], dtype=np.float64).reshape(n, 203),
        mean=np.asarray([o.mean for o in observations], dtype=np.float64).reshape(n, 9),
        covariance=np.asarray([o.covariance for o in observations], dtype=np.float64).reshape(n, 9, 9),
        score=np.asarray([o.score for o in observations], dtype=np.float64),
        source=np.asarray([o.node.source_id for o in observations], dtype=np.int64),
        node_id=np.asarray([o.node.node_id for o in observations], dtype='U64'),
        frame_id=np.asarray([o.node.frame_id for o in observations], dtype='U128'),
        cache_sha256=np.asarray([o.source_cache_sha256 for o in observations], dtype='U64'),
        information_us=np.asarray([o.node.information_us for o in observations], dtype=np.int64),
        arrival_us=np.asarray([o.node.arrival_us for o in observations], dtype=np.int64),
        state_us=np.asarray([o.state_us for o in observations], dtype=np.int64),
        detection_index=np.asarray([o.detection_index for o in observations], dtype=np.int64),
        contexts=indices, lengths=np.asarray([len(c.indices) for c in contexts], dtype=np.int64),
        decision_us=np.asarray([c.decision_us for c in contexts], dtype=np.int64))
    assert all(not a.dtype.hasobject and len(a) == n for a in values.values())
    assert all(len(o.node.node_id) <= 64 and len(o.node.frame_id) <= 128 for o in observations)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **values)
    event_path = path.with_suffix('.events.json')
    new(event_path, events)
    counts = Counter(('car', 'bicycle', 'pedestrian')[o.class_index] for o in observations)
    return dict(sequence_id=sequence, path=path.name, sha256=sha(path), bytes=path.stat().st_size,
        nodes=n, events_path=event_path.name, events_sha256=sha(event_path),
        original_events=len(events), empty_arrival_events=sum(e['new_count'] == 0 for e in events),
        candidate_class_counts=dict(counts), has_supervision_fields=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(1337, 2027, 3407))
    args = parser.parse_args()
    admission_path = V2 / f'seed{args.seed}/independent-acceptance.json'
    admission = json.loads(admission_path.read_bytes())
    assert admission['seed'] == args.seed and admission['full_raw_to_V2_numeric_admission'] is True
    assert admission['all_payloads_independently_rehashed'] is True and admission['GT_free'] is True
    assert admission['atol'] == admission['rtol'] == 1e-8 and admission['original_schedule_events'] == 3316
    assert admission['schedule_kind'] == 'scheduled_pair_snapshot_at_reference_plus_100ms'
    numeric_index = json.loads(NUMERIC.read_bytes())
    assert numeric_index['kind'] == 'rbf_new_nested_selected_all_class_final_refit_three_seed_independent_full_numeric_index_v1'
    numeric_entry = next(entry for entry in numeric_index['seeds'] if entry['seed'] == args.seed)
    assert numeric_entry['numeric_failures'] == 0
    assert sha(numeric_entry['numeric_completion']) == numeric_entry['numeric_completion_sha256']
    assert sha(numeric_entry['training_byte_proof']) == numeric_entry['training_byte_proof_sha256']
    byte_admission = json.loads((CK / f'seed{args.seed}/acceptance.json').read_bytes())
    checkpoint_path = CK / f'seed{args.seed}/checkpoint'
    assert sha(checkpoint_path) == byte_admission['artifacts']['checkpoint']['sha256']
    checkpoint = json.loads(checkpoint_path.read_bytes())
    assert checkpoint['seed'] == args.seed and checkpoint['row_protocol']['candidate_protocol'] == 'rbf-all-class-top64-v1'
    assert numeric_entry['training_task_id'] == byte_admission['task_id']
    for relative, digest in checkpoint['source_sha256'].items():
        assert sha(ORIGINAL / relative) == digest
    CPU_dependency_extraction = original_packages()
    from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache, CacheDelivery
    from transvision.models.event_track_v2x.forest_tracking import cache_detections, PaperForestTrackingConfig
    from transvision.models.event_track_v2x.forest_row_context import build_row_contexts
    protocol = checkpoint['row_protocol']
    config = PaperForestTrackingConfig(candidate_protocol=protocol['candidate_protocol'],
        parent_limit=protocol['parent_limit'], max_parent_gap_us=protocol['max_parent_gap_us'],
        gate_distance_m=protocol['gate_distance_m'], process_noise=protocol['process_noise'])
    cache = VerifiedForestCache(admission['cache'], admission['cache_manifest_sha256'])
    manifest = json.loads(cache.manifest_json)
    assert manifest['split'] == 'val' and len(manifest['sequences']) == 21
    expected = checkpoint['frozen_cache_identity']
    for _, metadata_bytes in cache.index.values():
        metadata = json.loads(metadata_bytes)
        assert all(metadata[key] == expected[key] for key in ('calibration_sha256', 'feature_method', 'feature_checkpoint_sha256'))
        assert metadata['detector_checkpoint_sha256'] == expected['detector_checkpoint_sha256'][metadata['side']]
    schedule_path = R / 'artifacts/spd-mht-k4-input-recovery-20260929/schedule.json'
    assert sha(schedule_path) == admission['original_schedule_sha256']
    schedule = json.loads(schedule_path.read_bytes())
    assert schedule['contains_ground_truth'] is False and schedule['contains_system_error_offset'] is False
    rows = schedule['frames']
    assert len(rows) == 3316 and {r['sequence_id'] for r in rows} == set(manifest['sequences'])
    output = DEST / f'seed{args.seed}'
    output.mkdir(parents=True, exist_ok=False)
    new(output / 'input-gate.json', dict(V2_independent_acceptance=str(admission_path),
        V2_independent_acceptance_sha256=sha(admission_path), new_NN_numeric_acceptance_sha256=sha(NUMERIC),
        checkpoint_sha256=sha(checkpoint_path), original_source_sha256=checkpoint['source_sha256'],
        CPU_dependency_extraction=CPU_dependency_extraction,
        source_sha256=sha(__file__), created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    shards, completed_events, completed_rows, last = [], 0, 0, 0
    started = time.monotonic()
    for sequence in manifest['sequences']:
        scene_rows = [r for r in rows if r['sequence_id'] == sequence]
        assert scene_rows
        observations, contexts, times, seen, events = [], [], [], set(), []
        origin = min(json.loads(meta)['box_reference_timestamp_us'] for key, (_, meta) in cache.index.items() if key[0] == sequence)
        previous = -1
        for row in scene_rows:
            reference = row['box_reference_timestamp_us']
            assert type(reference) is int and reference > previous
            previous, decision = reference, reference + 100000
            deliveries, unavailable = [], []
            for side, field in (('vehicle-side', 'vehicle_frame'), ('infrastructure-side', 'infrastructure_frame')):
                entry, meta = [json.loads(v) for v in cache.index[(sequence, side, row[field])]]
                assert (side, row[field]) not in seen
                seen.add((side, row[field]))
                if side == 'vehicle-side':
                    assert meta['box_reference_timestamp_us'] == reference
                if max(meta['box_reference_timestamp_us'], meta['source_image_timestamp_us']) > decision:
                    unavailable.append(dict(side=side, frame_id=row[field], frame_sha256=entry['frame_sha256']))
                    continue
                deliveries.append(CacheDelivery(sequence, side, row[field], decision, entry['frame_sha256']))
            new_observations = []
            source_receipts = []
            for delivery in sorted(deliveries, key=lambda d: (d.arrival_us, d.side, d.frame_id)):
                frame = cache.load_arrived(delivery, decision)
                selected = cache_detections(frame, arrival_us=delivery.arrival_us, decision_us=decision,
                    origin_us=origin, minimum_raw_score=protocol['minimum_raw_score'],
                    maximum_detections=protocol['maximum_detections'], candidate_protocol=config.candidate_protocol)
                new_observations.extend(selected)
                source_receipts.append(dict(side=delivery.side, frame_id=delivery.frame_id,
                    frame_sha256=delivery.frame_sha256, declared_arrival_us=decision,
                    selected_detection_indices=[o.detection_index for o in selected]))

            def older(lo, hi):
                start, stop = bisect_left(times, (lo, -1)), bisect_right(times, (hi, len(observations)))
                for _, index in times[start:stop]:
                    yield index, observations[index]

            current = build_row_contexts(new_observations, old_count=len(observations), older_candidates=older,
                config=config, decision_us=decision, sequence_id=sequence)
            events.append(dict(original_schedule_row=row, decision_us=decision, origin_us=origin,
                old_count=len(observations), new_count=len(new_observations),
                source_receipts=source_receipts, unavailable_sources=unavailable))
            for observation in new_observations:
                insort(times, (observation.state_us, len(observations)))
                observations.append(observation)
            contexts.extend(current)
            completed_events += 1
            completed_rows += len(new_observations)
            now = time.monotonic()
            if now - last > 20 or completed_events == 3316:
                print(json.dumps(dict(stage='target_free_original_seen_val_row_preparation', seed=args.seed,
                    completed_events=completed_events, total_events=3316, completed_rows=completed_rows,
                    ETA_seconds=(now-started)*(3316-completed_events)/completed_events,
                    ETA_scope='remaining original schedule row-context construction only')), flush=True)
                last = now
        shards.append(serialize(output / f'sequence-{len(shards):04d}.npz', observations, contexts,
                                sequence, events, config.parent_limit))
    assert completed_events == 3316 and len(shards) == 21 and sum(s['nodes'] for s in shards) == completed_rows
    envelope = dict(kind='rbf_prediction_only_original_arrival_row_contexts_v1', split='val', seed=args.seed,
        sequences=manifest['sequences'], shards=shards, row_protocol=protocol,
        frozen_cache_identity=expected, cache_manifest_sha256=cache.manifest_sha256,
        original_schedule_sha256=sha(schedule_path), original_events=completed_events, rows=completed_rows,
        no_GT_or_dummy_supervision_fields=True, candidate_protocol='rbf-all-class-top64-v1',
        original_source_contract=checkpoint['source_sha256'], source_sha256=sha(__file__),
        CPU_dependency_extraction=CPU_dependency_extraction,
        all_empty_arrival_events_preserved=True, actual_measured_network_history=False,
        independent_complete_feature_history_row_admission=False,
        full_recoverable_forest_replay_complete=False, paper_performance_complete=False)
    new(output / 'manifest.json', envelope)
    print(json.dumps(dict(seed=args.seed, target_free_original_val_row_candidate_complete=True,
        rows=completed_rows, original_events=completed_events, manifest_sha256=sha(output/'manifest.json'),
        independent_input_admission_still_required=True, full_online_RBF_accepted=False)), flush=True)


if __name__ == '__main__':
    main()
