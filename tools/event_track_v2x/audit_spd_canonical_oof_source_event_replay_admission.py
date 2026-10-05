#!/usr/bin/env python3
"""Audit full-source replay dependencies without changing deadlines or running a tracker.

All frames remain inventoried. A source arriving after the last vehicle deadline
is explicitly pending, never fabricated as an evaluated vehicle event.
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, output):
    index_path = root / 'receipts/spd-canonical-oof-fivefold-heldout-car-association-final-independent-acceptance-index-20261001.json'
    index = json.loads(index_path.read_text())
    assert index['all_five_folds_independently_accepted']
    folds = []
    for binding in index['jobs']:
        fold = binding['fold_id']
        receipt = root / binding['receipt']
        assert sha(receipt) == binding['receipt_sha256']
        admitted = json.loads(receipt.read_text())
        assert admitted['all_logits_match_independent_NumPy']
        data = root / f'artifacts/spd-canonical-oof-fold{fold}-full-heldout-car-association-input-v1-20261001'
        manifest = json.loads((data / 'manifest.json').read_text())
        for name, key in [('source-events.json', 'source_event_index_sha256'), ('examples.json', 'example_index_sha256')]:
            assert sha(data / name) == manifest[key]
        sources = json.loads((data / 'source-events.json').read_text())
        rows = json.loads((data / 'examples.json').read_text())
        assert len(sources) == binding['source_frames']
        assert len(rows) == binding['full_vehicle_reference_frames']
        by_sequence = {}
        for row in rows:
            by_sequence.setdefault(row['sequence_id'], []).append(row)
        deadlines = {}
        for seq, seq_rows in by_sequence.items():
            values = sorted(row['decision_time_us'] for row in seq_rows)
            assert len(values) == len(set(values)), 'duplicate vehicle decision needs explicit resolution'
            deadlines[seq] = values
        identities, delivery_inventory, pending = set(), [], []
        for source in sources:
            identity = (source['sequence_id'], source['side'], source['frame_id'])
            assert identity not in identities
            identities.add(identity)
            assert identity[0] in manifest['held_out_sequence_ids']
            assert identity[0] not in manifest['excluded_fit_sequence_ids']
            arrival = max(source['source_image_timestamp_us'], source['box_reference_timestamp_us'])
            values = deadlines[identity[0]]
            position = bisect.bisect_left(values, arrival)
            entry = dict(sequence_id=identity[0], side=identity[1], frame_id=identity[2],
                         source_complete_at_us=arrival,
                         arrays_sha256=source['arrays_sha256'], metadata_sha256=source['metadata_sha256'])
            if position == len(values):
                pending.append(entry)
            else:
                delivery_inventory.append(dict(**entry, first_eligible_vehicle_deadline_us=values[position]))
        # Independent accounting: each source must be assigned exactly once to
        # its earliest eligible deadline, or explicitly retained after the horizon.
        assert len(delivery_inventory) + len(pending) == len(sources)
        for entry in delivery_inventory:
            values = deadlines[entry['sequence_id']]
            selected = entry['first_eligible_vehicle_deadline_us']
            assert selected >= entry['source_complete_at_us']
            assert not any(entry['source_complete_at_us'] <= prior < selected for prior in values)
        folds.append(dict(fold_id=fold, association_receipt_sha256=sha(receipt),
                          source_inventory_sha256=sha(data / 'source-events.json'),
                          vehicle_references=len(rows), source_frames=len(sources),
                          first_eligible_deadline_inventory=delivery_inventory,
                          after_last_vehicle_deadline_preserved=pending,
                          source_complete_frame_accounting_passed=True))
        print(json.dumps(dict(stage='source_event_replay_admission_audit', fold_id=fold,
                              source_frames=len(sources), eligible=len(delivery_inventory),
                              pending_after_horizon=len(pending), ETA='completed')), flush=True)
    result = dict(kind='canonical_oof_source_event_replay_preparation_only',
                  association_index_sha256=sha(index_path), source_sha256=sha(Path(__file__)), folds=folds,
                  GT_read=False, arrays_payload_read=False, source_complete_C0_inventory_only=True,
                  paired_frame_links_used_to_drop_source_events=False, deadlines_extended=False,
                  tracking_consumer_executed=False, same_root_agent_mask_consumer_executed=False,
                  network_packet_mapping_executed=False, full_tracking_acceptance=False,
                  paper_eligible=False, next_dependency='canonical role-aware full-source tracking consumer',
                  ETA='unknown until consumer implementation and task-level replay progress')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
