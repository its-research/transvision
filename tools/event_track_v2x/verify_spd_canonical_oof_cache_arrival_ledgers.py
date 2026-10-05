#!/usr/bin/env python3
"""Independent full arrival-ledger audit; no production consumer imports."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

R = Path('/Volumes/Data/test/recover-before-fuse')
B = R / 'artifacts/spd-canonical-oof-full-source-cache-arrival-consumer-v1-20261001'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
canonical = lambda d: json.dumps(d, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def main():
    graph_path = R / 'artifacts/spd-canonical-oof-fivefold-content-numeric-index-and-same-root-mask-20261001/acceptance-index.json'
    graph = json.loads(graph_path.read_text())
    summary = json.loads((B / 'completion.json').read_text())
    results = []
    for fold in range(5):
        started = time.monotonic()
        binding = next(x for x in graph['folds'] if x['fold_id'] == fold)
        cache = Path(binding['content_acceptance_path']).parent / 'cache'
        manifest_path = cache / 'manifest.json'
        assert sha(manifest_path) == binding['cache_manifest_sha256']
        manifest = json.loads(manifest_path.read_text())
        entries = {}
        for e in manifest['frames']:
            p = cache / e['metadata']['path']; assert sha(p) == e['metadata']['sha256']
            meta = json.loads(p.read_text()); key = (meta['sequence_id'], meta['side'], meta['frame_id'])
            assert key not in entries; entries[key] = (e, meta)
        original = next(x for x in summary['folds'] if x['fold_id'] == fold)
        lp = B / f'fold-{fold}-arrival-ledger.jsonl'; assert sha(lp) == original['arrival_ledger_sha256']
        seen = set(); prefixes = ['0' * 64, '0' * 64]; counts = [0, 0]; selected_counts = [0, 0]
        with lp.open() as stream:
            for i, line in enumerate(stream):
                record = json.loads(line); row = record['both']; key = (row['sequence_id'], row['side'], row['frame_id'])
                assert key in entries and key not in seen; seen.add(key)
                entry, meta = entries[key]; arrival = max(meta['source_image_timestamp_us'], meta['box_reference_timestamp_us'])
                arrays_path = cache / entry['arrays']['path']; assert sha(arrays_path) == entry['arrays']['sha256']
                with np.load(arrays_path, allow_pickle=False) as z:
                    scores, classes = z['raw_scores'], z['class_indices']
                ranked = sorted(range(len(scores)), key=lambda j: (-float(scores[j]), j))
                candidates = [j for j in ranked if scores[j] >= .05][:64]
                expected = [j for j in candidates if classes[j] == 0]
                for position, name in enumerate(['both', 'vehicle_only']):
                    row = record[name]; include = position == 0 or key[1] == 'vehicle-side'
                    assert (row['sequence_id'], row['side'], row['frame_id']) == key
                    assert row['arrival_us'] == row['decision_us'] == row['source_complete_at_us'] == arrival
                    assert row['frame_sha256'] == entry['frame_sha256']
                    assert row['included_by_same_root_mask'] is include
                    assert row['selected_car_queries'] == (expected if include else [])
                    assert row['previous_prefix_sha256'] == prefixes[position]
                    unsigned = {k: v for k, v in row.items() if k != 'prefix_sha256'}
                    assert hashlib.sha256(canonical(unsigned)).hexdigest() == row['prefix_sha256']
                    prefixes[position] = row['prefix_sha256']; counts[position] += int(include)
                    selected_counts[position] += len(row['selected_car_queries'])
                if (i + 1) % 500 == 0:
                    print(json.dumps(dict(stage='independent_full_arrival_ledger_readback', fold_id=fold, completed=i + 1,
                        total=len(entries), eta_seconds=(time.monotonic() - started) / (i + 1) * (len(entries) - i - 1))), flush=True)
        assert seen == set(entries)
        for position, name in enumerate(['both', 'vehicle_only']):
            assert original[name]['prefix_sha256'] == prefixes[position]
            assert original[name]['payload_frames_consumed'] == counts[position]
            assert original[name]['selected_car_queries'] == selected_counts[position]
        results.append(dict(fold_id=fold, source_frames=len(seen), vehicle_payload_frames=counts[1],
                            all_registered_frame_bytes_and_query_selections_independently_recomputed=True,
                            both_prefix_sha256=prefixes[0], vehicle_only_prefix_sha256=prefixes[1],
                            arrival_ledger_sha256=sha(lp)))
    receipt = dict(folds=results, completion_sha256=sha(B / 'completion.json'), cache_graph_sha256=sha(graph_path),
                   verifier_sha256=sha(Path(__file__)), all_source_frames_and_mask_prefixes_verified=True,
                   production_consumer_imported=False, tracking_executed=False, paper_eligible=False)
    with (B / 'independent-ingestion-acceptance.json').open('x') as f:
        json.dump(receipt, f, indent=2); f.write('\n')
    print(json.dumps(dict(stage='full_source_ingestion_independent_acceptance', completed=True)), flush=True)


if __name__ == '__main__': main()
