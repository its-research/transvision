"""Independent schedule/transport linkage review, without rerunning inference."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path

from rbf_nested_seen_val_v2_common import R, new, register, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.parent == R/'receipts' and not args.output.exists()
    entries = json.loads((R/'receipts/20260928-execution-ledger.json').read_bytes())['entries']

    def registered(path):
        digest = sha(path)
        assert any(e.get('receipt') == str(path) and e.get('receipt_sha256') == digest for e in entries)
        return dict(path=str(path), sha256=digest)

    schedule_path = R/'artifacts/spd-mht-k4-input-recovery-20260929/schedule.json'
    schedule = json.loads(schedule_path.read_bytes())
    assert schedule['contains_ground_truth'] is False and schedule['contains_system_error_offset'] is False
    outcomes = []
    for seed in (1337, 2027, 3407):
        root = R/f'artifacts/rbf-seen-val-forest-input-bridge-v1-20261005/seed{seed}'
        transport = R/f'artifacts/rbf-seen-val-forest-cache-transport-v1-20261005/seed{seed}'
        bp, tp = root/'input-binding.json', transport/'local-transport-readback.json'
        records = [registered(bp), registered(tp)]
        b, t = json.loads(bp.read_bytes()), json.loads(tp.read_bytes())
        assert b['seed'] == t['seed'] == seed and t['input_binding_sha256'] == sha(bp)
        for spec in b['prerequisite_receipts']:
            assert registered(Path(spec['path'])) == spec
        assert b['full_original_schedule_to_forest_event_binding_passed'] is True
        assert t['all_member_bytes_independently_read'] is True
        assert t['uploaded'] is False and t['cache_rebuilt'] is False and t['NN_forward_repeated'] is False
        assert sha(root/'events.json') == b['events_sha256'] == t['events_sha256']
        assert sha(root/'cache-inventory.json') == b['inventory_sha256']
        ev = json.loads((root/'events.json').read_bytes())
        inventory = json.loads((root/'cache-inventory.json').read_bytes())
        cache = Path(inventory['cache_root'])
        manifest = json.loads((cache/'manifest.json').read_bytes())
        assert sha(cache/'manifest.json') == t['cache_manifest_sha256'] == ev['cache_manifest_sha256']
        assert ev['original_schedule_sha256'] == sha(schedule_path)
        archive = Path(t['archive']['path'])
        assert archive == transport/'cache.tar' and archive.stat().st_size == t['archive']['bytes']
        assert sha(archive) == t['archive']['sha256']
        assert t['members'] == len(inventory['files'])+1 == 14379
        assert t['payload_bytes'] == b['cache_bytes']+(cache/'manifest.json').stat().st_size
        metadata, origins = {}, {}
        for frame in manifest['frames']:
            mp = cache/frame['metadata']['path']
            assert sha(mp) == frame['metadata']['sha256']
            m = json.loads(mp.read_bytes())
            key = (m['sequence_id'], m['side'], m['frame_id'])
            assert key not in metadata
            metadata[key] = (frame['frame_sha256'], m)
            origins[key[0]] = min(origins.get(key[0], m['box_reference_timestamp_us']), m['box_reference_timestamp_us'])
        assert origins == ev['origin_us_by_sequence']
        assert len(ev['events']) == len(schedule['frames']) == 3316
        seen, counts, missing = set(), Counter(), Counter()
        for row, event in zip(schedule['frames'], ev['events']):
            assert set(event) == {'sequence_id', 'frame_id', 'reference_us', 'decision_us', 'event_id', 'deliveries'}
            seq, ref = row['sequence_id'], row['box_reference_timestamp_us']
            assert event['sequence_id'] == seq and event['frame_id'] == row['vehicle_frame']
            assert event['reference_us'] == ref and event['decision_us'] == ref+100000
            assert (seq,event['event_id']) not in seen
            seen.add((seq,event['event_id']))
            expected = {}
            for side, field in [('infrastructure-side','infrastructure_frame'),('vehicle-side','vehicle_frame')]:
                key = (seq, side, row[field])
                digest, meta = metadata[key]
                if meta['box_reference_timestamp_us'] <= ref+100000 and meta['source_image_timestamp_us'] <= ref+100000:
                    expected[key] = dict(sequence_id=seq, side=side, frame_id=row[field], arrival_us=ref+100000, frame_sha256=digest)
                    counts[side] += 1
                else:
                    missing[side] += 1
            assert len(event['deliveries']) == len(expected)
            assert event['deliveries'] == [expected[key] for key in sorted(expected)]
        assert set(origins) == set(manifest['sequences']) and len(origins) == 21
        assert dict(counts) == b['information_time_census']['eligible']
        assert dict(missing) == b['information_time_census']['unavailable']
        outcomes.append(dict(seed=seed, events=3316, sequences=21, rows=b['rows'],
            cache_frames=len(metadata), archive=t['archive'], receipts=records,
            eligible_source_frames=dict(counts), unavailable_source_frames=dict(missing),
            full_schedule_transport_linkage_review_passed=True))
    result = dict(kind='rbf_seen_val_three_seed_forest_input_transport_linkage_review_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sha(__file__), common_sha256=sha(Path(__file__).with_name('rbf_nested_seen_val_v2_common.py')),
        seeds=outcomes, original_schedule_sha256=sha(schedule_path), total_rows=sum(x['rows'] for x in outcomes),
        all_three_local_forest_input_packages_bound=True,
        independent_review_imports_bridge_or_packager=False, neural_inference_repeated=False,
        scope='SPD seen-val exploratory scheduled-snapshot inputs and local transport only',
        measured_network_arrival_history_verified=False, GPU_task_created=False, uploaded=False,
        full_forest_independently_accepted=False, same_resource_accepted=False, paper_performance_complete=False)
    new(args.output,result)
    register(args.output,result['kind'])
    print(json.dumps(dict(receipt=str(args.output),sha256=sha(args.output),total_rows=result['total_rows'],
        all_three_local_forest_input_packages_bound=True)),flush=True)


if __name__ == '__main__':
    main()
