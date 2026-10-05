#!/usr/bin/env python3
"""Read sealed V2 through the new adapter and compare every admitted raw fixture."""
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


def main(root, source, output):
    for row in json.loads((source / 'source-inventory.json').read_text()):
        assert sha(source / row['path']) == row['sha256']
    for name, rel in [('transvision', 'transvision'), ('transvision.models', 'transvision/models'),
                      ('transvision.models.event_track_v2x', 'transvision/models/event_track_v2x')]:
        module = types.ModuleType(name); module.__path__ = [str(source / 'runtime' / rel)]; sys.modules[name] = module
    from transvision.models.event_track_v2x.canonical_oof_cache_arrivals import CanonicalOOFCacheArrivals
    from transvision.models.event_track_v2x.canonical_oof_predicted_features import load_fold_calibration
    from transvision.models.event_track_v2x.canonical_oof_v2_features import frame_features
    from transvision.models.event_track_v2x.detection_cache_v2 import DetectionCacheV2
    graph_path = root / 'artifacts/spd-canonical-oof-fivefold-content-numeric-index-and-same-root-mask-20261001/acceptance-index.json'
    graph = json.loads(graph_path.read_text())
    assert not output.exists(); output.mkdir(); results = []
    for fold in range(5):
        started = time.monotonic()
        b = next(x for x in graph['folds'] if x['fold_id'] == fold)
        cache = Path(b['content_acceptance_path']).parent / 'cache'
        inputs = root / f'artifacts/spd-canonical-oof-fold{fold}-full-heldout-car-association-input-v1-20261001'
        m = json.loads((inputs / 'manifest.json').read_text())
        proof_path = inputs / 'independent-full-input-acceptance.json'
        proof = json.loads(proof_path.read_text())
        assert proof['all_feature_bytes_encoder_values_gate_distances_and_deadlines_verified']
        assert sha(inputs / 'manifest.json') == proof['manifest_sha256']
        cal = m['calibration']; calibration = load_fold_calibration(Path(cal['path']), cal['sha256'], fold_id=fold)
        admission = CanonicalOOFCacheArrivals(cache, b['cache_manifest_sha256'], calibration,
                    calibration_sha256=cal['sha256'], fold_id=fold, role='held_out')
        assert sha(inputs / 'examples.json') == m['example_index_sha256']
        rows = json.loads((inputs / 'examples.json').read_text()); maximum = 0.; failures = []; compared = 0
        for i, row in enumerate(rows):
            p = inputs / row['features']['path']; assert sha(p) == row['features']['sha256']
            with np.load(p, allow_pickle=False) as z:
                expected = {k: z[k] for k in z.files}
            seq = row['sequence_id']; keys = [(seq, 'vehicle-side', row['vehicle_frame_id']),
                                              (seq, 'infrastructure-side', row['infrastructure_frame_id'])]
            reference = None
            if row['available'][0]:
                reference = DetectionCacheV2.load(cache, admission.entries[keys[0]])
            for side, name in enumerate(['left', 'right']):
                if reference is None or not row['available'][side]:
                    assert expected[name].shape == (0, 203)
                    continue
                frame = reference if side == 0 else DetectionCacheV2.load(cache, admission.entries[keys[1]])
                ids, actual = frame_features(frame, reference, calibration,
                    calibration_sha256=cal['sha256'], fold_id=fold, role='held_out',
                    decision_us=row['decision_time_us'], origin_us=row['origin_us'])
                assert np.array_equal(ids, expected[name + '_query_indices'])
                assert actual.shape == expected[name].shape
                error = float(np.max(np.abs(actual.astype(float) - expected[name]))) if actual.size else 0.
                maximum = max(maximum, error); compared += len(ids)
                if not np.allclose(actual, expected[name], atol=1e-7, rtol=1e-7):
                    if len(failures) < 8: failures.append(dict(vehicle_frame_id=row['vehicle_frame_id'], side=name, max_absolute_error=error))
            if (i + 1) % 200 == 0 or i + 1 == len(rows):
                print(json.dumps(dict(stage='canonical_V2_encoder_full_integration', fold_id=fold,
                                      completed=i + 1, total=len(rows), eta_seconds=(time.monotonic() - started) / (i + 1) * (len(rows) - i - 1))), flush=True)
        result = dict(fold_id=fold, cache_manifest_sha256=b['cache_manifest_sha256'],
                      raw_input_acceptance_sha256=sha(proof_path), vehicle_references=len(rows),
                      compared_car_query_features=compared, max_absolute_error=maximum,
                      failures=failures, all_full_V2_features_match_accepted_raw_input=not failures,
                      absolute_tolerance=1e-7, relative_tolerance=1e-7, GT_read=False,
                      tracking_executed=False, paper_eligible=False)
        (output / f'fold-{fold}-result.json').write_text(json.dumps(result, indent=2) + '\n'); results.append(result)
    result = dict(folds=results, all_folds_match=all(x['all_full_V2_features_match_accepted_raw_input'] for x in results),
                  source_inventory_sha256=sha(source / 'source-inventory.json'), cache_graph_sha256=sha(graph_path),
                  GPU_inference_executed=False, tracking_executed=False, paper_eligible=False)
    (output / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); main(a.root, a.source, a.output)
