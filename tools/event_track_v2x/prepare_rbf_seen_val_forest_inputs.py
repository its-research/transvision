"""Bind admitted seen-val caches and NN rows to the forest event interface.

This creates local transport manifests only. It neither rebuilds caches nor
starts replay, and cannot grant forest, resource, or paper acceptance.
"""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import sys
import time

from rbf_nested_seen_val_v2_common import R, new, register, sha

SIDES = (('vehicle-side', 'vehicle_frame'),
         ('infrastructure-side', 'infrastructure_frame'))
BASE = 'rbf-seen-val-forest-input-bridge-v1-20261005'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_file(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts,
            'unsafe manifest path')
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()) and path.is_file(),
            'missing or escaping manifest file')
    require(not any(p.is_symlink() for p in (path, *path.parents)
                    if p != root and p.is_relative_to(root)), 'symlink in input')
    return path


def convert_events(schedule, metadata, recorded):
    """Use the complete schedule, including events selecting zero queries."""
    require(schedule['contains_ground_truth'] is False and
            schedule['contains_system_error_offset'] is False, 'GT-bearing schedule')
    rows = schedule['frames']
    require(rows and len(rows) == sum(map(len, recorded.values())), 'schedule coverage')
    sequences = {r['sequence_id'] for r in rows}
    require(sequences == set(recorded), 'sequence coverage')
    origins = {s: min(m['box_reference_timestamp_us'] for (seq, _, _), (_, m)
                      in metadata.items() if seq == s) for s in sequences}
    events, previous, seen, offsets = [], {}, set(), Counter()
    eligible, unavailable = Counter(), Counter()
    for row in rows:
        require(set(row) == {'sequence_id', 'vehicle_frame', 'infrastructure_frame',
                            'box_reference_timestamp_us'}, 'unexpected schedule fields')
        sequence = row['sequence_id']
        reference = row['box_reference_timestamp_us']
        require(type(reference) is int and reference > previous.get(sequence, -1),
                'nonmonotonic schedule')
        previous[sequence] = reference
        decision = reference + 100000
        original = recorded[sequence][offsets[sequence]]
        offsets[sequence] += 1
        require(original['original_schedule_row'] == row and
                original['decision_us'] == decision and
                original['origin_us'] == origins[sequence], 'row event differs from schedule')
        deliveries, expected_unavailable = [], []
        for side, field in SIDES:
            key = (sequence, side, row[field])
            require(key not in seen, 'source frame repeated')
            seen.add(key)
            digest, meta = metadata[key]
            require((meta['sequence_id'], meta['side'], meta['frame_id']) == key,
                    'metadata frame identity mismatch')
            if side == 'vehicle-side':
                require(meta['box_reference_timestamp_us'] == reference, 'vehicle clock')
            available = max(meta['box_reference_timestamp_us'],
                            meta['source_image_timestamp_us']) <= decision
            identity = dict(side=side, frame_id=row[field], frame_sha256=digest)
            if available:
                eligible[side] += 1
                deliveries.append(dict(sequence_id=sequence, arrival_us=decision, **identity))
            else:
                unavailable[side] += 1
                expected_unavailable.append(identity)
        deliveries.sort(key=lambda d: (d['arrival_us'], d['side'], d['frame_id']))
        received = original['source_receipts']
        require(len(received) == len(deliveries), 'arrived source count mismatch')
        for delivery, receipt in zip(deliveries, received):
            require(all(receipt[k] == delivery[k] for k in ('side', 'frame_id', 'frame_sha256'))
                    and receipt['declared_arrival_us'] == delivery['arrival_us'],
                    'arrived source identity or clock mismatch')
            indices = receipt['selected_detection_indices']
            require(len(indices) <= 64 and len(set(indices)) == len(indices) and
                    all(type(i) is int and 0 <= i < 900 for i in indices), 'candidate indices')
        require(original['unavailable_sources'] == expected_unavailable, 'future source mismatch')
        require(original['new_count'] == sum(len(v['selected_detection_indices']) for v in received),
                'new observation count mismatch')
        events.append(dict(sequence_id=sequence, frame_id=row['vehicle_frame'],
            reference_us=reference, decision_us=decision,
            event_id=f"{sequence}:{row['vehicle_frame']}", deliveries=deliveries))
    return dict(events=events, origin_us_by_sequence=origins,
        information_time_census=dict(eligible=eligible, unavailable=unavailable),
        zero_selected_query_events=sum(e['new_count'] == 0 for v in recorded.values() for e in v),
        zero_delivery_events=sum(not e['deliveries'] for e in events))


def build(seed, destination):
    ledger = json.loads((R/'receipts/20260928-execution-ledger.json').read_bytes())
    proofs = []

    def admitted(path):
        digest = sha(path)
        require(any(e.get('receipt') == str(path) and e.get('receipt_sha256') == digest
                    for e in ledger['entries']), f'unregistered or changed proof: {path.name}')
        proofs.append(dict(path=str(path), sha256=digest))
        return json.loads(path.read_bytes())

    cache_root = R/f'artifacts/rbf-nested-seen-val-matching-V2-full-admission-v1-20261004/seed{seed}'
    row_root = R/f'artifacts/rbf-matching-seen-val-target-free-original-row-contexts-v1-20261004/seed{seed}'
    nn_root = R/f'artifacts/rbf-matching-seen-val-joint-identity-full-independent-output-admission-v1-20261004/seed{seed}'
    ck_root = R/f'artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004/seed{seed}'
    cache_proof = admitted(cache_root/'independent-acceptance.json')
    row_proof = admitted(row_root/'independent-acceptance.json')
    byte_proof = admitted(nn_root/'independent-byte-coverage.json')
    numeric = admitted(nn_root/'full-numeric/completion.json')
    training = admitted(ck_root/'acceptance.json')
    require(all(x['seed'] == seed for x in (cache_proof, row_proof, byte_proof, numeric, training)), 'seed binding')
    for field in ('full_raw_to_V2_numeric_admission', 'all_payloads_independently_rehashed',
                  'raw_scores_classes_appearance_exactly_preserved', 'no_query_class_or_topk_selection', 'GT_free'):
        require(cache_proof[field] is True, field)
    require(row_proof['full_original_seen_val_features_and_contexts_independently_accepted'] is True,
            'row feature/context proof')
    require(byte_proof['all_cloud_output_bytes_independently_read'] is True and
            byte_proof['all_original_seen_val_input_rows_exactly_once'] is True, 'NN byte/coverage proof')
    require(numeric['full_independent_numeric_pass'] is True and numeric['numeric_failed_rows'] == 0
            and numeric['atol'] == numeric['rtol'] == 1e-4, 'NN numeric proof')
    require(numeric['prediction_byte_proof_sha256'] == sha(nn_root/'independent-byte-coverage.json')
            and numeric['row_input_admission_sha256'] == byte_proof['row_input_admission_sha256']
            == sha(row_root/'independent-acceptance.json'), 'NN proof linkage')
    cache = cache_root/'cache'
    cm = json.loads((cache/'manifest.json').read_bytes())
    rm = json.loads((row_root/'manifest.json').read_bytes())
    ck = json.loads((ck_root/'checkpoint').read_bytes())
    require(training['all_six_registered_artifact_bytes_independently_read'] is True and
            training['artifacts']['checkpoint']['sha256'] == sha(ck_root/'checkpoint'), 'final checkpoint proof')
    require(sha(cache/'manifest.json') == cache_proof['cache_manifest_sha256']
            == row_proof['cache_manifest_sha256'] == rm['cache_manifest_sha256'], 'cache binding')
    require(sha(row_root/'manifest.json') == row_proof['row_manifest_sha256']
            == byte_proof['row_manifest_sha256'], 'row manifest binding')
    require(row_proof['V2_independent_admission_sha256'] == sha(cache_root/'independent-acceptance.json'), 'V2 proof linkage')
    require(row_proof['checkpoint_sha256'] == sha(ck_root/'checkpoint') and
            ck['weights']['sha256'] == byte_proof['weights_sha256'] == numeric['weights_sha256']
            and ck['row_protocol'] == rm['row_protocol'] and
            ck['frozen_cache_identity'] == rm['frozen_cache_identity'], 'checkpoint/NN binding')
    require(cm['split'] == rm['split'] == 'val' and cm['gt_in_cache'] is False and
            cm['frame_count'] == len(cm['frames']) == 7189 and
            len(cm['sequences']) == len(set(cm['sequences'])) == 21 and
            cm['sequences'] == rm['sequences'] and rm['all_empty_arrival_events_preserved'] is True and
            rm['no_GT_or_dummy_supervision_fields'] is True and
            rm['candidate_protocol'] == 'rbf-all-class-top64-v1', 'seen-val scope')
    require(ck['data_split'] == 'train' and ck['validation_or_test_selection'] is False
            and ck['selection'] == 'frozen_nested_selected_epoch_full_train_refit', 'training boundary')
    schedule_path = R/'artifacts/spd-mht-k4-input-recovery-20260929/schedule.json'
    require(sha(schedule_path) == cache_proof['original_schedule_sha256']
            == rm['original_schedule_sha256'] == row_proof['original_schedule_sha256'], 'schedule binding')
    metadata, inventory, paths = {}, [], set()
    started = time.monotonic()
    for i, frame in enumerate(cm['frames']):
        for role in ('arrays', 'metadata'):
            record = frame[role]
            require(record['path'] not in paths, 'duplicate cache payload')
            paths.add(record['path'])
            path = safe_file(cache, record['path'])
            require(path.stat().st_size == record['bytes'], 'cache payload size')
            inventory.append(dict(role=role, **record))
        path = safe_file(cache, frame['metadata']['path'])
        require(sha(path) == frame['metadata']['sha256'], 'cache metadata bytes')
        meta = json.loads(path.read_bytes())
        require(meta['arrays_sha256'] == frame['arrays']['sha256'] and
                meta['dataset_split'] == 'val' and meta['calibration_fit_split'] == 'train', 'frame binding')
        identity = ck['frozen_cache_identity']
        require(all(meta[k] == identity[k] for k in ('calibration_sha256', 'feature_method', 'feature_checkpoint_sha256'))
                and meta['detector_checkpoint_sha256'] == identity['detector_checkpoint_sha256'][meta['side']], 'detector/embedding binding')
        key = (meta['sequence_id'], meta['side'], meta['frame_id'])
        require(key not in metadata, 'duplicate frame identity')
        metadata[key] = (frame['frame_sha256'], meta)
        if (i + 1) % 1000 == 0:
            print(json.dumps(dict(stage='seen_val_forest_metadata_binding', seed=seed,
                completed_frames=i+1, total_frames=len(cm['frames']),
                ETA_seconds=(time.monotonic()-started)*(len(cm['frames'])-i-1)/(i+1),
                ETA_scope='metadata binding only')), flush=True)
    recorded = {}
    for shard in rm['shards']:
        path = safe_file(row_root, shard['events_path'])
        require(sha(path) == shard['events_sha256'], 'row event bytes')
        require(shard['sequence_id'] not in recorded, 'duplicate event shard')
        recorded[shard['sequence_id']] = json.loads(path.read_bytes())
    converted = convert_events(json.loads(schedule_path.read_bytes()), metadata, recorded)
    require(len(converted['events']) == rm['original_events'] == 3316, 'full event count')
    require(sum(e['new_count'] for v in recorded.values() for e in v)
            == rm['rows'] == byte_proof['rows'] == numeric['rows'], 'full candidate count')
    envelope = dict(kind='rbf_matching_seen_val_complete_forest_events_v1', seed=seed, split='val',
        cache_manifest_sha256=sha(cache/'manifest.json'), original_schedule_sha256=sha(schedule_path),
        arrival_policy=rm['row_protocol']['arrival_policy'], measured_network_arrival_history_verified=False,
        events=converted['events'], origin_us_by_sequence=converted['origin_us_by_sequence'])
    new(destination/'events.json', envelope)
    new(destination/'cache-inventory.json', dict(cache_root=str(cache), cache_manifest_sha256=sha(cache/'manifest.json'),
        files=inventory, arrays_rehashed_in_this_bridge=False,
        arrays_require_byte_verification_during_transport=True))
    receipt = dict(kind='rbf_seen_val_forest_input_bridge_v1', seed=seed,
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        command=sys.argv, source_sha256=sha(__file__), source_freeze_sha256=sha(Path(__file__).with_name('source-freeze.json')),
        prerequisite_receipts=proofs, checkpoint=dict(path=str(ck_root/'checkpoint'), sha256=sha(ck_root/'checkpoint'),
        model_sha256=ck['model_sha256']), rows=rm['rows'], events=3316, sequences=21,
        cache_frames=len(metadata), cache_bytes=sum(f['bytes'] for f in inventory),
        events_sha256=sha(destination/'events.json'), inventory_sha256=sha(destination/'cache-inventory.json'),
        information_time_census=converted['information_time_census'],
        zero_selected_query_events=converted['zero_selected_query_events'], zero_delivery_events=converted['zero_delivery_events'],
        forward_artifacts=[v for k, v in byte_proof['artifacts'].items() if k.startswith('predictions-rank')],
        full_original_schedule_to_forest_event_binding_passed=True,
        scope='SPD seen-val exploratory scheduled snapshots only',
        measured_network_arrival_history_verified=False, GPU_task_created=False,
        full_forest_independently_accepted=False, same_resource_accepted=False, paper_performance_complete=False,
        remaining_gates=['full train forest interface admission', 'cache transport and independent byte readback',
                         'frozen seen-val forest producer and independent full output admission',
                         'learned priority and same-resource comparisons', 'independent exploratory evaluation'])
    new(destination/'input-binding.json', receipt)
    register(destination/'input-binding.json', receipt['kind'])
    print(json.dumps({k: receipt[k] for k in ('seed', 'rows', 'events', 'sequences', 'cache_bytes',
        'full_original_schedule_to_forest_event_binding_passed')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(1337, 2027, 3407))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    freeze = json.loads((root/'source-freeze.json').read_bytes())
    require(freeze['kind'] == BASE, 'source freeze kind')
    for name, digest in freeze['sources'].items():
        require(sha(root/name) == digest, 'source bytes differ')
    destination = R/'artifacts'/BASE/f'seed{args.seed}'
    destination.mkdir(parents=True, exist_ok=False)
    try:
        build(args.seed, destination)
    except BaseException as error:
        failure = destination/'failure.json'
        new(failure, dict(seed=args.seed, exception_type=type(error).__name__, message=str(error),
                         partials_preserved=True, no_automatic_retry=True, accepted=False))
        register(failure, 'rbf_seen_val_forest_input_bridge_failure_v1')
        raise


if __name__ == '__main__':
    main()
