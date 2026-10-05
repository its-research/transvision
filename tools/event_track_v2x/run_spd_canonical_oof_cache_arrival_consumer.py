#!/usr/bin/env python3
"""Full real-cache ingestion/mask experiment, separate from tracking metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(root, source, output):
    for row in json.loads((source / 'source-inventory.json').read_text()):
        assert sha(source / row['path']) == row['sha256']
    runtime = source / 'runtime'
    for name, rel in [('transvision', 'transvision'), ('transvision.models', 'transvision/models'),
                      ('transvision.models.event_track_v2x', 'transvision/models/event_track_v2x')]:
        module = types.ModuleType(name)
        module.__path__ = [str(runtime / rel)]
        sys.modules[name] = module
    from transvision.models.event_track_v2x.canonical_oof_cache_arrivals import CanonicalOOFCacheArrivals
    from transvision.models.event_track_v2x.canonical_oof_predicted_features import load_fold_calibration
    graph_path = root / 'artifacts/spd-canonical-oof-fivefold-content-numeric-index-and-same-root-mask-20261001/acceptance-index.json'
    graph = json.loads(graph_path.read_text())
    preparation_path = root / 'receipts/spd-canonical-oof-source-event-replay-admission-audit-20261001.json'
    preparation = json.loads(preparation_path.read_text())
    assert not output.exists()
    output.mkdir()
    results = []
    for fold in range(5):
        started = time.monotonic()
        binding = next(x for x in graph['folds'] if x['fold_id'] == fold)
        cp = Path(binding['content_acceptance_path'])
        assert sha(cp) == binding['content_acceptance_sha256']
        np_path = Path(binding['numeric_acceptance_path'])
        assert sha(np_path) == binding['numeric_acceptance_sha256']
        cache = cp.parent / 'cache'
        raw_input = root / f'artifacts/spd-canonical-oof-fold{fold}-full-heldout-car-association-input-v1-20261001'
        m = json.loads((raw_input / 'manifest.json').read_text())
        cal = m['calibration']
        calibration = load_fold_calibration(Path(cal['path']), cal['sha256'], fold_id=fold)
        kwargs = dict(calibration_sha256=cal['sha256'], fold_id=fold, role='held_out')
        both = CanonicalOOFCacheArrivals(cache, binding['cache_manifest_sha256'], calibration, **kwargs)
        vehicle = CanonicalOOFCacheArrivals(cache, binding['cache_manifest_sha256'], calibration, agent_mask=1, **kwargs)
        mask_path = Path(binding['mask_path'])
        assert sha(mask_path) == binding['mask_sha256']
        mask = json.loads(mask_path.read_text())
        assert mask['cache_manifest_sha256'] == binding['cache_manifest_sha256']
        masked_frames = {entry['frame_sha256'] for entry in mask['frames']}
        assert masked_frames == {entry['frame_sha256'] for key, entry in both.entries.items() if key[1] == 'vehicle-side'}
        identities = lambda rows: {(r['sequence_id'], r['side'], r['frame_id']) for r in rows}
        sources = json.loads((raw_input / 'source-events.json').read_text())
        assert identities(sources) == set(both.entries)
        ordered = sorted(both.entries, key=lambda k: (k[0], max(both.metadata[k]['source_image_timestamp_us'], both.metadata[k]['box_reference_timestamp_us']), k[1], k[2]))
        log = output / f'fold-{fold}-arrival-ledger.jsonl'
        with log.open('x') as stream:
            for i, key in enumerate(ordered):
                meta = both.metadata[key]
                arrival = max(meta['source_image_timestamp_us'], meta['box_reference_timestamp_us'])
                frame_sha = both.entries[key]['frame_sha256']
                previous = both.prefix_sha256
                try:
                    both.consume(key, arrival_us=arrival - 1, decision_us=arrival, frame_sha256=frame_sha)
                except ValueError:
                    pass
                else:
                    raise AssertionError('incomplete source admitted')
                assert previous == both.prefix_sha256
                actual, payload = both.consume(key, arrival_us=arrival, decision_us=arrival, frame_sha256=frame_sha)
                masked, masked_payload = vehicle.consume(key, arrival_us=arrival, decision_us=arrival, frame_sha256=frame_sha)
                candidates = sorted((int(j) for j in np.flatnonzero(payload.raw_scores >= .05)), key=lambda j: (-float(payload.raw_scores[j]), j))[:64]
                expected = [j for j in candidates if payload.class_indices[j] == 0]
                assert actual['selected_car_queries'] == expected
                if key[1] == 'vehicle-side':
                    assert masked['selected_car_queries'] == expected and masked_payload.digest() == payload.digest()
                else:
                    assert masked_payload is None and not masked['selected_car_queries']
                prefix = both.prefix_sha256
                duplicate, repeated = both.consume(key, arrival_us=arrival + 1, decision_us=arrival + 1, frame_sha256=frame_sha)
                assert duplicate == actual and repeated is None and both.prefix_sha256 == prefix
                stream.write(json.dumps(dict(both=actual, vehicle_only=masked), sort_keys=True) + '\n')
                if (i + 1) % 200 == 0 or i + 1 == len(ordered):
                    elapsed = time.monotonic() - started
                    print(json.dumps(dict(stage='full_source_cache_ingestion_and_mask', fold_id=fold,
                                          completed=i + 1, total=len(ordered), eta_seconds=elapsed / (i + 1) * (len(ordered) - i - 1))), flush=True)
        tail = next(x for x in preparation['folds'] if x['fold_id'] == fold)['after_last_vehicle_deadline_preserved']
        assert identities(tail) <= set(both.receipts)
        result = dict(fold_id=fold, both=both.completion(), vehicle_only=vehicle.completion(),
                      mask_sha256=sha(mask_path), content_acceptance_sha256=sha(cp),
                      numeric_acceptance_sha256=sha(np_path), arrival_ledger_sha256=sha(log),
                      independently_checked_all_selected_query_identities=True,
                      future_source_rejections_checked=len(ordered), duplicate_prefix_checks=len(ordered),
                      post_vehicle_horizon_frames_consumed=len(tail),
                      tracking_or_metrics_executed=False, paper_eligible=False)
        (output / f'fold-{fold}-ingestion-result.json').write_text(json.dumps(result, indent=2) + '\n')
        results.append(result)
    result = dict(folds=results, cache_graph_sha256=sha(graph_path), source_inventory_sha256=sha(source / 'source-inventory.json'),
                  preparation_sha256=sha(preparation_path), all_five_source_ingestion_experiments_completed=True,
                  tracking_executed=False, paper_eligible=False)
    (output / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.source, a.output)
