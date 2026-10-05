#!/usr/bin/env python3
"""Independent dataset-specific evaluation of sealed paper replays, including per-sequence
metrics.

Runs only in the evaluator environment. No tracker, cache, model, or training module is imported; GT remains outside all prediction artifacts.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / 'transvision/models/event_track_v2x/tracking_evaluation_v2.py'
PROGRESS = ROOT / 'transvision/models/event_track_v2x/experiment_progress.py'
CLASS_POLICY = ROOT / 'transvision/models/event_track_v2x/paper_evaluation_policy.py'


def class_policy():
    spec = importlib.util.spec_from_file_location('rbf_independent_class_policy', CLASS_POLICY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def progress_class():
    # Load only the standalone stdlib logger, without importing model packages.
    spec = importlib.util.spec_from_file_location('rbf_evaluation_progress', PROGRESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ExperimentProgress


def adapter():
    spec = importlib.util.spec_from_file_location('rbf_independent_evaluator', ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_vector(metrics, class_name='car'):
    n = metrics['nuscenes']['label_metrics']
    t = metrics['trackeval'][class_name]['summary']
    return {
        **{k: t[k]
           for k in ('HOTA', 'AssA', 'DetA', 'IDF1')},
        **{out: n[key][class_name]
           for out, key in (('AMOTA', 'amota'), ('AMOTP', 'amotp'), ('FP', 'fp'), ('FN', 'fn'), ('IDS', 'ids'), ('Frag', 'frag'))}
    }


def evaluate(gt_manifest, gt_sha256, replay, receipt_sha256, output):
    module = adapter()
    policy = class_policy()
    gt_manifest = Path(gt_manifest)
    replay = Path(replay)
    output = Path(output)
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('create-once evaluation output required')
    if module.sha(gt_manifest) != gt_sha256 or module.sha(replay / 'receipt.json') != receipt_sha256:
        raise ValueError('GT/replay receipt changed')
    m = json.loads(gt_manifest.read_bytes())
    receipt = json.loads((replay / 'receipt.json').read_bytes())
    if m['kind'] != 'rbf_paper_evaluation_gt_v1' or type(m['fixture']) is not bool:
        raise ValueError('explicit independent GT manifest required')
    if receipt['status'] != 'software_replay_completed' or receipt['fixture'] != m['fixture']:
        raise ValueError('incomplete replay or mixed fixture/real evidence')
    inputs = {gt_manifest: gt_sha256, replay / 'receipt.json': receipt_sha256, ADAPTER: module.sha(ADAPTER), PROGRESS: module.sha(PROGRESS), Path(__file__): module.sha(__file__)}
    inputs[CLASS_POLICY] = module.sha(CLASS_POLICY)
    for name, digest in receipt['files'].items():
        if Path(name).name != name:
            raise ValueError('unsafe replay artifact path')
        inputs[replay / name] = digest
    relative = Path(m['ground_truth']['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('unsafe GT path')
    gt_path = gt_manifest.parent / relative
    inputs[gt_path] = m['ground_truth']['sha256']
    if any(module.sha(path) != sha for path, sha in inputs.items()):
        raise ValueError('evaluation input changed')
    plan = json.loads((replay / 'plan.json').read_bytes())
    if plan['protocol'] != m['protocol']:
        raise ValueError('GT/replay protocol differs')
    allowed = {'spd': {'train', 'val'}, 'v2v4real': {'train', 'official_test'}}
    protocol = m['protocol']
    selected_class = policy.require_evaluation_binding(protocol, plan.get('model_binding'))
    vehicle = selected_class == 'vehicle'
    if vehicle and (m.get('evaluation_protocol') != policy.VEHICLE_PROTOCOL
                    or m.get('native_label_source') != policy.NATIVE_VEHICLE_SELECTION):
        raise ValueError('explicit native vehicle GT class provenance required')
    if protocol['dataset'] not in allowed or protocol['split'] not in allowed[protocol['dataset']]:
        raise ValueError('forbidden dataset split')
    gt = [json.loads(line) for line in gt_path.read_bytes().splitlines()]
    predictions = [json.loads(line) for line in (replay / 'predictions.jsonl').read_bytes().splitlines()]
    if len(gt) != m['frames'] or len(gt) != receipt['completed_events'] or not gt:
        raise ValueError('missing evaluation frame')
    counts = module.validate_predictions(predictions, gt)
    for frame in gt:
        ids = [b['track_id'] for b in frame['objects']]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate GT identity')
        for b in frame['objects']:
            if vehicle and (b.get('class_label') != 'vehicle' or not policy.native_vehicle(b.get('raw_class'))):
                raise ValueError('vehicle GT must preserve native obj_type and explicit vehicle class')
            mean = np.asarray(b['mean'], float)
            if mean.shape != (9, ) or not np.isfinite(mean).all() or np.any(mean[3:6] <= 0):
                raise ValueError('invalid GT physical box')
    roi = m['roi']

    def within(box, frame):
        if box['class_label'] != selected_class:
            return False
        xy = np.asarray(box['mean'][:3]) - np.asarray(frame['ego_translation_world'])
        if roi['kind'] == 'strict_radial_xy':
            if not np.isfinite(roi['radius_m']) or roi['radius_m'] <= 0:
                raise ValueError('invalid radial ROI')
            return np.linalg.norm(xy[:2]) < roi['radius_m']
        if roi['kind'] == 'ego_xy_rectangle':
            rotation = np.asarray(frame['world_to_ego_row_rotation'])
            if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3)) or not np.isclose(np.linalg.det(rotation), 1):
                raise ValueError('explicit proper ego rotation required')
            xy = xy @ rotation
            bounds = np.asarray(roi['bounds_xy'], float)
            if bounds.shape != (4, ) or not np.isfinite(bounds).all() or not (bounds[0] < bounds[2] and bounds[1] < bounds[3]):
                raise ValueError('invalid rectangular ROI')
            return bounds[0] <= xy[0] <= bounds[2] and bounds[1] <= xy[1] <= bounds[3]
        raise ValueError('unknown explicit evaluation ROI')

    filtered_gt, filtered_p = copy.deepcopy(gt), copy.deepcopy(predictions)
    if vehicle:
        # Validate original immutable commits above. Only a separately bound
        # working copy receives this explicit detector-channel interpretation.
        mapping = plan['model_binding']['prediction_class_mapping']
        for p in filtered_p:
            for b in p['predictions']:
                if b['class_label'] not in mapping:
                    raise ValueError('prediction class outside bound native single-class detector')
                b['class_label'] = mapping[b['class_label']]
    for g, p in zip(filtered_gt, filtered_p):
        g['objects'] = [b for b in g['objects'] if within(b, g)]
        p['predictions'] = [b for b in p['predictions'] if within(b, g)]
    if vehicle:
        # nuScenes TrackingBox accepts only its registered class names. This
        # disclosed backend alias changes no geometry, thresholds, or IDs and
        # is not the native V2V4Real IoU-based AMOTP metric.
        for g, p in zip(filtered_gt, filtered_p):
            for b in (*g['objects'], *p['predictions']):
                b['class_label'] = 'car'
    # ROI was applied above to both sides. Do not apply the legacy SPD 50m ROI
    # again to a V2V4Real rectangular ROI. Metric engines remain unchanged.
    module._roi = lambda boxes, ego, name: [b for b in boxes if b['class_label'] == name]
    runtime = module.runtime_evidence()
    progress = progress_class()
    aggregate_progress = progress('independent_evaluation_aggregate', 1)
    metrics = module.compute_metrics(filtered_gt, filtered_p, classes=('car', ))
    aggregate_progress.update(1, force=True)
    per_sequence = {}
    sequence_ids = sorted({g['sequence_id'] for g in gt})
    sequence_progress = progress('independent_evaluation_sequences', len(sequence_ids))
    for completed, sid in enumerate(sequence_ids, 1):
        indices = [i for i, g in enumerate(gt) if g['sequence_id'] == sid]
        result = module.compute_metrics([filtered_gt[i] for i in indices], [filtered_p[i] for i in indices], classes=('car', ))
        per_sequence[sid] = dict(metrics=metric_vector(result), frames=len(indices), cluster=m['sequence_clusters'][sid])
        sequence_progress.update(completed, force=True)
    if any(module.sha(path) != sha for path, sha in inputs.items()):
        raise ValueError('inputs changed during evaluation')
    output.mkdir()
    report = dict(
        kind='rbf_paper_independent_evaluation_v1',
        status='evaluated',
        protocol=protocol,
        roi=roi,
        fixture=m['fixture'],
        metrics=metric_vector(metrics),
        per_sequence=per_sequence,
        coverage=counts,
        aggregate='official_metric_sequence_aggregation_not_arithmetic_mean',
        amotp_definition='nuScenes_mean_center_distance_m_lower_is_better_not_native_AB3DMOT_AMOTP',
        native_protocol_reproduction=False,
        paper_results_verified=False,
        runtime=runtime,
        input_sha256={str(path): sha
                      for path, sha in inputs.items()})
    if vehicle:
        report.update(evaluation_class_binding=policy.vehicle_binding(),
            metric_backend_class_alias={'vehicle': 'car'},
            native_v2v4real_tracking_metrics=False,
            prediction_commit_class_labels_unchanged=True,
            coverage_class_semantics={'car': 'vehicle'},
            auxiliary_metrics_only=True)
        metrics = dict(metrics, evaluation_class='vehicle', metric_backend_class_alias={'vehicle': 'car'},
                       native_v2v4real_tracking_metrics=False)
    module.write_json(output / 'metrics.json', metrics)
    module.write_json(output / 'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('gt-manifest', 'gt-sha256', 'replay', 'receipt-sha256', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    print(json.dumps(evaluate(a.gt_manifest, a.gt_sha256, a.replay, a.receipt_sha256, a.output), sort_keys=True))


if __name__ == '__main__':
    main()
