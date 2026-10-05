#!/usr/bin/env python3
"""Create a separate train-only native vehicle existence calibration candidate.

Existing prediction manifests and role partitions remain byte-identical. This
entry consumes newly prepared vehicle GT and an explicit detector source audit,
and never accepts official_test for fitting. No cloud action is performed.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.experiment_progress import ExperimentProgress
from transvision.models.event_track_v2x.v2v4real_ground_truth import load_vehicle_ground_truth
from transvision.models.event_track_v2x.v2v4real_vehicle_calibration import fit_vehicle_existence, DETECTOR_SEMANTIC_SOURCE_PINS


def pinned(path, digest):
    path = Path(path).absolute()
    if (not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None
            or any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file()
            or sha_file(path) != digest):
        raise ValueError('ordinary unchanged SHA-256 bound input required')
    return path


def fit(prediction_root, prediction_manifest_sha256, gt_root, gt_manifest_sha256,
        partition_path, partition_sha256, detector_semantics_path, detector_semantics_sha256, output):
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('fresh calibration output required')
    root = Path(prediction_root).absolute()
    if root in output.parents or Path(gt_root).absolute() in output.parents:
        raise ValueError('calibration output must be outside input projections')
    if not root.is_dir() or {p.name for p in root.iterdir()} != {'manifest.json', 'predictions.jsonl'}:
        raise ValueError('exact prediction archive inventory required')
    manifest_path = pinned(root/'manifest.json', prediction_manifest_sha256)
    manifest = json.loads(manifest_path.read_bytes())
    # Reject test before GT label bytes are opened.
    if manifest.get('split') != 'train' or manifest.get('official_test_read') is not False:
        raise ValueError('calibration accepts train-only predictions')
    gt_manifest_path = pinned(Path(gt_root)/'manifest.json', gt_manifest_sha256)
    if json.loads(gt_manifest_path.read_bytes()).get('split') != 'train':
        raise ValueError('calibration accepts train-only GT; official test is evaluation-only')
    rows_path = pinned(root/'predictions.jsonl', manifest.get('rows_sha256'))
    raw = rows_path.read_bytes()
    if len(raw.splitlines()) != manifest.get('rows'):
        raise ValueError('prediction stream row count differs')
    rows = tuple(json.loads(line) for line in raw.splitlines())
    gt_manifest, frames = load_vehicle_ground_truth(gt_root, expected_manifest_sha256=gt_manifest_sha256)
    partition_path = pinned(partition_path, partition_sha256)
    semantics_path = pinned(detector_semantics_path, detector_semantics_sha256)
    partition, semantics = json.loads(partition_path.read_bytes()), json.loads(semantics_path.read_bytes())
    if set(semantics.get('source_paths', {})) != set(DETECTOR_SEMANTIC_SOURCE_PINS):
        raise ValueError('three explicit original detector source paths required')
    source_paths = {k: pinned(semantics['source_paths'][k], digest)
                    for k, digest in DETECTOR_SEMANTIC_SOURCE_PINS.items()}
    inputs = {manifest_path: prediction_manifest_sha256, rows_path: manifest['rows_sha256'],
              gt_manifest_path: gt_manifest_sha256, partition_path: partition_sha256,
              semantics_path: detector_semantics_sha256}
    inputs.update({path: DETECTOR_SEMANTIC_SOURCE_PINS[k] for k, path in source_paths.items()})
    sources = {str(path): sha_file(path) for path in (
        Path(__file__), ROOT/'transvision/models/event_track_v2x/v2v4real_vehicle_calibration.py',
        ROOT/'transvision/models/event_track_v2x/paper_calibration.py',
        ROOT/'transvision/models/event_track_v2x/prediction_features.py',
        ROOT/'transvision/models/event_track_v2x/paper_evaluation_policy.py')}
    progress = ExperimentProgress('vehicle_train_only_calibration_matching', len(rows),
        eta_scope='matching only; optimizer and independent acceptance excluded')
    model, receipt = fit_vehicle_existence(manifest, rows, gt_manifest, frames, partition,
        prediction_sha256=prediction_manifest_sha256, gt_manifest_sha256=gt_manifest_sha256,
        partition_sha256=partition_sha256, detector_semantics=semantics,
        detector_semantics_sha256=detector_semantics_sha256,
        progress=lambda v: progress.update(v['completed_rows']))
    if any(sha_file(path) != digest for path, digest in {**inputs, **sources}.items()):
        raise ValueError('calibration inputs changed during fitting')
    load_vehicle_ground_truth(gt_root, expected_manifest_sha256=gt_manifest_sha256)
    output.mkdir(parents=True)
    with (output/'calibration.json').open('xb') as stream:
        stream.write(canonical(model))
    receipt.update(calibration_sha256=sha_file(output/'calibration.json'),
                   input_sha256={str(path): digest for path, digest in inputs.items()},
                   source_sha256=sources)
    with (output/'receipt.json').open('xb') as stream:
        stream.write(canonical(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prediction-root', 'prediction-manifest-sha256', 'gt-root', 'gt-manifest-sha256',
                 'partition', 'partition-sha256', 'detector-semantics', 'detector-semantics-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    a = parser.parse_args()
    print(json.dumps(fit(a.prediction_root, a.prediction_manifest_sha256, a.gt_root, a.gt_manifest_sha256,
        a.partition, a.partition_sha256, a.detector_semantics, a.detector_semantics_sha256, a.output), sort_keys=True))


if __name__ == '__main__':
    main()
