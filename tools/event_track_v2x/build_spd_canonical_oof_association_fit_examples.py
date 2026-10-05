#!/usr/bin/env python3
"""Build complementary-fit, prediction-only features and separate offline labels.

Clean-link training availability is explicit. This does not supply measured
network traces, hard-negative scenario certification, OOF selection or metrics.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def build(spec_path, output):
    freeze = Path(__file__).parent
    for r in json.loads((freeze / 'source-inventory.json').read_text()):
        assert sha(freeze / r['path']) == r['sha256'], 'frozen source changed'
    sys.path.insert(0, str(freeze / 'runtime'))
    from transvision.models.event_track_v2x.canonical_oof_predicted_features import load_fold_calibration, predicted_features
    from transvision.models.event_track_v2x.canonical_oof_prediction_gate import world_predictions, geometry_gate
    from transvision.models.event_track_v2x.canonical_oof_association_targets import association_targets
    from transvision.models.event_track_v2x.prediction_features import raw_state, match_predictions, regular_path
    spec = json.loads(spec_path.read_text())
    root = Path(spec['root']); fold = spec['fold_id']
    fitter = root / 'source-freezes/spd-canonical-calibration-runtime-v5-config-role-input-gate-closure-20261001'
    for r in json.loads((fitter / 'source-freeze-receipt.json').read_text())['inventory']:
        assert sha(fitter / r['path']) == r['sha256']
    sys.path.insert(0, str(fitter / 'tools/event_track_v2x'))
    from spd_canonical_oof_calibration_supervision import load_fit_supervision
    accept = Path(spec['mapping_acceptance']['path'])
    assert sha(accept) == spec['mapping_acceptance']['sha256']
    proof = json.loads(accept.read_text()); assert proof['fold_id'] == fold
    mapping = Path(spec['mapping']); manifest = mapping.parent / 'manifest.json'
    assert sha(manifest) == proof['producer_manifest_sha256'] and sha(mapping) == proof['mapping_sha256']
    package = root / f'artifacts/spd-official-oof-fivefold-20260930/fold-{fold}-package/package-manifest.json'
    assert sha(package) == proof['package_manifest_sha256']
    pkg = json.loads(package.read_text())
    cal = load_fold_calibration(Path(spec['calibration']['path']), spec['calibration']['sha256'], fold_id=fold)
    assert cal['fit_sequences'] == sorted(pkg['fit_sequence_ids'])
    gt, gt_proof = load_fit_supervision(root / f'artifacts/spd-official-oof-fivefold-20260930/remote-conversion/fold-{fold}-converted', package,
                                      root / f'artifacts/spd-single-fit-overlay-materializer-readback-20261001/job-fold-{fold}/fold-{fold}')
    index = {}; origins = {}; raw_manifests = []
    for side in ('vehicle-side', 'infrastructure-side'):
        for shard in (0, 1):
            folder = Path(spec['raw_root']) / f'{side}-shard-{shard}-cache'
            mp = folder / 'raw-cache-manifest.json'; value = json.loads(mp.read_text())
            raw_manifests.append(dict(path=str(mp), sha256=sha(mp), frames=len(value['frames'])))
            for row in value['frames']:
                meta_path = regular_path(folder, row['metadata']['path'])
                assert sha(meta_path) == row['metadata']['sha256']
                meta = json.loads(meta_path.read_text()); key = (side, meta['sequence_id'], meta['frame_id'])
                assert key not in index and meta['side'] == side and meta['sequence_id'] in cal['fit_sequences']
                index[key] = (folder, row, meta)
                if side == 'vehicle-side':
                    origins[meta['sequence_id']] = min(origins.get(meta['sequence_id'], meta['box_reference_timestamp_us']), meta['box_reference_timestamp_us'])
    assert set(index) == set(gt), 'raw/fit GT frame coverage differs'
    records = [json.loads(line) for line in mapping.read_text().splitlines()]
    assert len(records) == proof['pairs']
    assert not output.exists(), 'preserve existing output'
    output.mkdir(); (output / 'features').mkdir(); (output / 'offline-targets').mkdir()
    counts = dict(pairs=0, available_bilateral_pairs=0, vehicle_unavailable_pairs=0, positive_gate_pairs=0, negative_gate_pairs=0,
                  unknown_gate_pairs=0, left_assignment_rows=0, right_assignment_rows=0)
    started = time.monotonic(); example_rows = []
    for record in records:
        seq = record['sequence_id']; frames = (record['vehicle_frame_id'], record['infrastructure_frame_id'])
        values = []
        for side, frame in zip(('vehicle-side', 'infrastructure-side'), frames):
            folder, row, meta = index[side, seq, frame]
            ap = regular_path(folder, row['arrays']['path'])
            assert sha(ap) == row['arrays']['sha256']
            with np.load(ap, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            values.append((arrays, meta, gt[side, seq, frame], row))
        decision = values[0][1]['box_reference_timestamp_us'] + 100000
        availability = [all(v[1][k] <= decision for k in ('source_image_timestamp_us', 'box_reference_timestamp_us')) for v in values]
        selected = []; features = []; matches = []; world = []
        for arrays, meta, truth, _ in values:
            # Without an arrived vehicle reference image, do not encode either
            # side against that unadmitted reference; retain an empty example.
            available = availability[0] and all(meta[k] <= decision for k in ('source_image_timestamp_us', 'box_reference_timestamp_us'))
            if available:
                sel, feat = predicted_features(arrays, meta, values[0][1], cal, role='fit', decision_time_us=decision, origin_us=origins[seq])
                physical = world_predictions(arrays, meta, sel, cal, decision_time_us=decision)
                match = match_predictions(raw_state(arrays)[sel], arrays['scores'][sel], arrays['class_indices'][sel], truth['state'], truth['classes'])
            else:
                sel = np.empty(0, np.int64); feat = np.empty((0, 203), np.float32); match = np.empty(0, np.int64)
                physical = (np.empty((0, 9)), np.empty((0, 9, 9)), np.empty(0, np.int64))
            selected.append(sel); features.append(feat); matches.append(match); world.append(physical)
        gate, d2 = geometry_gate(*world, probability=.99)
        targets = association_targets(*matches, values[0][2]['tokens'], values[1][2]['tokens'], record['source_identity_bindings'], gate)
        name = seq + '-' + frames[0] + '-' + frames[1] + '.npz'
        fp = output / 'features' / name; tp = output / 'offline-targets' / name
        np.savez_compressed(fp, left=features[0], right=features[1], left_query_indices=selected[0], right_query_indices=selected[1],
                            geometry_gate=gate, innovation_distance_squared=d2,
                            left_classes=world[0][2], right_classes=world[1][2])
        np.savez_compressed(tp, **targets, left_matches=matches[0], right_matches=matches[1])
        supervised = targets['supervised_pair_mask']; positive = int((targets['targets'].astype(bool) & supervised).sum())
        counts['pairs'] += 1; counts['available_bilateral_pairs'] += int(all(availability))
        counts['vehicle_unavailable_pairs'] += int(not availability[0])
        counts['positive_gate_pairs'] += positive; counts['negative_gate_pairs'] += int(supervised.sum()) - positive
        counts['unknown_gate_pairs'] += int((gate & ~supervised).sum())
        counts['left_assignment_rows'] += int(targets['left_assignment_mask'].sum())
        counts['right_assignment_rows'] += int(targets['right_assignment_mask'].sum())
        example_rows.append(dict(sequence_id=seq, vehicle_frame_id=frames[0], infrastructure_frame_id=frames[1],
                                 decision_time_us=decision, origin_us=origins[seq], available=availability,
                                 features=dict(path=str(fp.relative_to(output)), sha256=sha(fp), bytes=fp.stat().st_size),
                                 offline_targets=dict(path=str(tp.relative_to(output)), sha256=sha(tp), bytes=tp.stat().st_size),
                                 raw_inputs=[dict(arrays_sha256=v[3]['arrays']['sha256'], metadata_sha256=v[3]['metadata']['sha256']) for v in values]))
        if counts['pairs'] % 100 == 0 or counts['pairs'] == len(records):
            elapsed = time.monotonic() - started; done = counts['pairs']; eta = elapsed / done * (len(records) - done)
            print(json.dumps(dict(kind='rbf_experiment_progress_v1', stage='canonical_association_fit_examples', fold_id=fold,
                                  completed=done, total=len(records), eta_seconds=eta, elapsed_seconds=elapsed)), flush=True)
    write(output / 'examples.json', example_rows)
    write(output / 'manifest.json', dict(kind='canonical_complementary_fit_prediction_only_association_examples_candidate_v1', fold_id=fold,
          fit_sequence_ids=cal['fit_sequences'], excluded_held_out_sequence_ids=cal['canonical_oof_binding']['held_out_sequence_ids'],
          package_manifest_sha256=sha(package), calibration=spec['calibration'], mapping_sha256=sha(mapping),
          mapping_acceptance=spec['mapping_acceptance'], raw_manifests=raw_manifests, converted_GT_supervision_readback=gt_proof,
          example_index_sha256=sha(output / 'examples.json'), counts=counts,
          feature_dimension=203, candidate_policy='raw-score>=0.05/all-class-top64', gate_probability=.99,
          gate_innovation_bound='2*(P+R)', gate_dimensions=3, process_noise_per_second=.1,
          availability='clean-link/source-complete-at-vehicle-box-time-plus-100ms', network_trace_used=False,
          source_inventory_sha256=sha(freeze / 'source-inventory.json'), specification_sha256=sha(spec_path),
          created_at_utc=datetime.now(timezone.utc).isoformat(), held_out_GT_used=False, val_test_payloads_read=False,
          independent_example_readback_passed=False, hard_negative_scenario_coverage_certified=False,
          GPU_training_completed=False, paper_eligible=False, ETA='complete_build_pending_independent_acceptance'))
    print(json.dumps(dict(fold_id=fold, counts=counts, manifest_sha256=sha(output / 'manifest.json'))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--specification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); build(a.specification, a.output)
