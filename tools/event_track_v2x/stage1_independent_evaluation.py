#!/usr/bin/env python3
"""Independent native-frame SPD Stage 1 metric input and geometry construction.

Does not import a predictor, the production adapter or its ROI/IoU helpers.
Official metric engines execute unchanged in the frozen original environment.
Polygon clipping uses local coordinates and no GEOS/shared polygon kernel.
"""
import argparse
from collections import defaultdict
import datetime
import hashlib
from importlib import metadata
import inspect
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

VERSIONS = {'motmetrics': '1.4.0', 'numpy': '1.26.4', 'nuscenes-devkit': '1.2.0',
            'pandas': '2.3.3', 'pyquaternion': '0.9.9', 'scipy': '1.15.3',
            'shapely': '2.0.7', 'trackeval': '1.0.0'}
TREES = {'motmetrics': 'd4a797e97023cb7147a98484030b454283f00e7e56a0bcd3ec737b1ee229c7e8',
         'nuscenes': '49a8f50dcaae5cee5f025eafceab216a4e77d53771befb05a9624e7955af9d5a',
         'trackeval': '90a4c6beb47735e74f1b785b55456f8e746daa938671dc9d3430f75cb96cbbf1'}
GT_SHA = '94908e0010003b42a5ed2c35ccc894005fe8d40cadfe57653871bbd95e0a9a76'
GT_MANIFEST_SHA = '94675ac8585d893195b7e801b754f62ca4ee3524cf852e05e53121fe2cc4084a'
FLOAT_ATOL = 1e-12
CLASSES = {'car', 'bicycle', 'pedestrian'}


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()


def ordinary(path):
    p = Path(path).absolute()
    if not p.is_file() or any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('ordinary frozen evidence required')
    return p


def read(path, digest=None):
    path = ordinary(path)
    if digest is not None and sha(path) != digest: raise ValueError('frozen input SHA differs')
    return json.loads(path.read_bytes())


def polygon(mean, origin):
    """Counterclockwise physical length/width rectangle, common local origin."""
    x, y, length, width, yaw = mean[0], mean[1], mean[3], mean[4], mean[6]
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x - origin[0] + c * u - s * v, y - origin[1] + s * u + c * v)
            for u, v in [(-length / 2, -width / 2), (length / 2, -width / 2),
                         (length / 2, width / 2), (-length / 2, width / 2)]]


def cross(a, b): return a[0] * b[1] - a[1] * b[0]
def minus(a, b): return (a[0] - b[0], a[1] - b[1])


def intersection_area(subject, clip):
    """Sutherland-Hodgman clipping of convex counterclockwise polygons."""
    output = subject
    for a, b in zip(clip, clip[1:] + clip[:1]):
        current, output = output, []
        if not current: break
        edge = minus(b, a); previous = current[-1]
        prev_side = cross(edge, minus(previous, a))
        for point in current:
            side = cross(edge, minus(point, a))
            if (side >= 0) != (prev_side >= 0):
                fraction = prev_side / (prev_side - side)
                output.append((previous[0] + fraction * (point[0] - previous[0]),
                               previous[1] + fraction * (point[1] - previous[1])))
            if side >= 0: output.append(point)
            previous, prev_side = point, side
    if len(output) < 3: return 0.
    return abs(math.fsum(cross(a, b) for a, b in zip(output, output[1:] + output[:1]))) / 2


def bev_iou(left, right):
    origin = (left[0], left[1]); a, b = polygon(left, origin), polygon(right, origin)
    if any(max(x[k] for x in a) <= min(x[k] for x in b) or max(x[k] for x in b) <= min(x[k] for x in a) for k in (0, 1)):
        return 0.
    inter = intersection_area(a, b)
    area_a, area_b = left[3] * left[4], right[3] * right[4]
    if inter < 0 or inter > min(area_a, area_b) + 1e-10: raise ValueError('invalid convex intersection area')
    return min(1., max(0., inter / (area_a + area_b - inter)))


def car_rows(boxes, ego):
    seen = set(); result = []
    for box in boxes:
        tid = box['track_id']; name = box['class_label']; mean = box['mean']
        if not isinstance(tid, str) or not tid or tid != tid.strip() or tid in seen:
            raise ValueError('invalid or duplicate original frame identity')
        seen.add(tid)
        if name not in CLASSES or len(mean) != 9 or any(not math.isfinite(x) for x in mean) or min(mean[3:6]) <= 0:
            raise ValueError('invalid original class/world box')
        if name == 'car' and math.hypot(mean[0] - ego[0], mean[1] - ego[1]) < 50.:
            result.append(dict(track_id=tid, mean=mean, score=box.get('score', -1.)))
    return result


def load_frames(gt_path, pred_path, report):
    gt, pred, counts, tips, previous_times = [], [], defaultdict(int), {}, {}
    with ordinary(gt_path).open() as g, ordinary(pred_path).open() as p:
        from itertools import zip_longest
        for gline, pline in zip_longest(g, p):
            if gline is None or pline is None: raise ValueError('complete frames including empty outputs required')
            gf, pf = json.loads(gline), json.loads(pline)
            sid, fid, timestamp = (gf[k] for k in ('sequence_id', 'frame_id', 'box_reference_timestamp_us'))
            if (pf['sequence_id'], pf['frame_id'], pf['box_reference_timestamp_us']) != (sid, fid, timestamp):
                raise ValueError('native frame/time alignment differs')
            if type(timestamp) is not int or timestamp <= previous_times.get(sid, -1) or type(pf['decision_timestamp_us']) is not int or pf['decision_timestamp_us'] != timestamp + 100000:
                raise ValueError('native timing/causal commitment differs')
            if pf['coordinate_frame'] != 'world' or pf['state_layout'] != 'gravity_xyz_length_width_height_yaw_vxy':
                raise ValueError('physical world box convention required')
            if pf['previous_commit_sha256'] != tips.get(sid, '0' * 64): raise ValueError('prediction commit prefix differs')
            committed = dict(pf); digest = committed.pop('commit_sha256')
            if hashlib.sha256(canonical(committed)).hexdigest() != digest: raise ValueError('prediction canonical commit differs')
            tips[sid], previous_times[sid] = digest, timestamp
            for box in pf['predictions']:
                score = box['score']
                if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('invalid original prediction score')
            ego = gf['ego_translation_world']
            gt.append(dict(sid=sid, fid=fid, time=timestamp, boxes=car_rows(gf['objects'], ego)))
            pred.append(dict(sid=sid, fid=fid, time=timestamp, boxes=car_rows(pf['predictions'], ego)))
            counts[sid] += 1
    if len(gt) != 3316 or len(counts) != 21 or tips != report['coverage']['final_sequence_commit_sha256']:
        raise ValueError('complete 3316-frame/21-sequence original commits required')
    return gt, pred, dict(counts)


def runtime():
    import trackeval, nuscenes, motmetrics
    observed = {k: metadata.version(k) for k in VERSIONS}
    if observed != VERSIONS: raise ValueError('original metric library versions required')
    hashes = {}
    for name, module in [('trackeval', trackeval), ('nuscenes', nuscenes), ('motmetrics', motmetrics)]:
        root = Path(inspect.getfile(module)).parent
        inventory = [dict(path=str(p.relative_to(root)), sha256=sha(p)) for p in sorted(root.rglob('*.py'))]
        hashes[name] = hashlib.sha256(canonical(inventory)).hexdigest()
    if hashes != TREES: raise ValueError('unchanged complete author metric source trees required')
    return dict(versions=observed, author_tree_sha256=hashes)


def trackeval_metrics(gt, pred):
    from trackeval.metrics import HOTA, Identity
    groups = defaultdict(list)
    for g, p in zip(gt, pred): groups[g['sid']].append((g, p))
    hota = HOTA(); identity = Identity({'THRESHOLD': .5, 'PRINT_CONFIG': False}); results = {}
    for sid, frames in groups.items():
        gmap = {tid: i for i, tid in enumerate(sorted({b['track_id'] for g, _ in frames for b in g['boxes']}))}
        pmap = {tid: i for i, tid in enumerate(sorted({b['track_id'] for _, p in frames for b in p['boxes']}))}
        data = dict(num_timesteps=len(frames), num_gt_ids=len(gmap), num_tracker_ids=len(pmap),
                    num_gt_dets=sum(len(g['boxes']) for g, _ in frames), num_tracker_dets=sum(len(p['boxes']) for _, p in frames),
                    gt_ids=[], tracker_ids=[], similarity_scores=[])
        for g, p in frames:
            data['gt_ids'].append(np.array([gmap[b['track_id']] for b in g['boxes']], dtype=int))
            data['tracker_ids'].append(np.array([pmap[b['track_id']] for b in p['boxes']], dtype=int))
            similarities = np.empty((len(g['boxes']), len(p['boxes'])))
            for i, a in enumerate(g['boxes']):
                for j, b in enumerate(p['boxes']): similarities[i, j] = bev_iou(a['mean'], b['mean'])
            data['similarity_scores'].append(similarities)
        results[sid] = dict(HOTA=hota.eval_sequence(data), Identity=identity.eval_sequence(data), frames=len(frames),
                            gt_objects=data['num_gt_dets'], predicted_objects=data['num_tracker_dets'])
    h = hota.combine_sequences({s: v['HOTA'] for s, v in results.items()})
    i = identity.combine_sequences({s: v['Identity'] for s, v in results.items()})
    return dict(summary={**{k: float(np.mean(h[k])) for k in ('HOTA', 'AssA', 'DetA')}, **i}, HOTA_curves=h, sequences=results)


def nuscenes_metrics(gt, pred):
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingConfig
    from nuscenes.eval.tracking.evaluate import TrackingEval
    cfg = config_factory('tracking_nips_2019').serialize(); cfg['tracking_names'] = ['car']
    for key in ('class_range', 'pretty_tracking_names', 'tracking_colors'): cfg[key] = {'car': cfg[key]['car']}
    ground, predicted, scores = defaultdict(dict), defaultdict(dict), defaultdict(list)
    for g, p in zip(gt, pred):
        sid, timestamp = g['sid'], g['time']; ground[sid][timestamp], predicted[sid][timestamp] = [], []
        for frame, destination, is_gt in [(g, ground[sid][timestamp], True), (p, predicted[sid][timestamp], False)]:
            for b in frame['boxes']:
                m = b['mean']; tid = sid + '/' + b['track_id']
                destination.append(TrackingBox(sample_token=sid + '/' + g['fid'], translation=tuple(m[:3]),
                    size=(m[4], m[3], m[5]), rotation=(math.cos(m[6] / 2), 0., 0., math.sin(m[6] / 2)),
                    velocity=tuple(m[7:9]), tracking_id=tid, tracking_name='car', tracking_score=-1. if is_gt else float(b['score'])))
                if not is_gt: scores[(sid, tid)].append(b['score'])
    means = {key: float(np.mean(value)) for key, value in scores.items()}
    for sid, times in predicted.items():
        for boxes in times.values():
            for b in boxes: b.tracking_score = means[(sid, b.tracking_id)]
    engine = TrackingEval.__new__(TrackingEval); engine.cfg = TrackingConfig.deserialize(cfg)
    engine.tracks_gt, engine.tracks_pred = ground, predicted
    engine.verbose, engine.output_dir, engine.render_classes = False, None, []
    summary, curves = engine.evaluate(); summary = summary.serialize()
    excluded = {}
    for name in ('tid', 'lgd'):
        excluded[name] = summary.pop(name, None); excluded['label_' + name] = summary['label_metrics'].pop(name, None)
    return summary, curves.serialize(), excluded


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)): return [clean(v) for v in x]
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, (float, np.floating)): return float(x) if math.isfinite(x) else None
    return x


def compare(actual, expected, path=()):
    """No tolerance for counts/structure; frozen 1e-12 for real-valued metrics."""
    if path == ('nuscenes', 'eval_time'): return 0.
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected): raise ValueError('metric structure differs at ' + '/'.join(path))
        return max((compare(actual[k], v, path + (k,)) for k, v in expected.items()), default=0.)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected): raise ValueError('metric array coverage differs at ' + '/'.join(path))
        return max((compare(a, b, path + (str(i),)) for i, (a, b) in enumerate(zip(actual, expected))), default=0.)
    if expected is None or isinstance(expected, (str, bool)):
        if actual != expected: raise ValueError('metric value differs at ' + '/'.join(path))
        return 0.
    count_fields = {'HOTA_TP', 'HOTA_FN', 'HOTA_FP', 'IDTP', 'IDFN', 'IDFP', 'frames', 'gt_objects', 'predicted_objects', 'fp', 'fn', 'ids', 'frag', 'tp', 'gt', 'mt', 'ml'}
    error = abs(actual - expected)
    if (any(k in count_fields for k in path) and error != 0) or error > FLOAT_ATOL:
        raise ValueError('independent metric differs at ' + '/'.join(path) + ': ' + str(error))
    return float(error)


def evaluate(args):
    evidence = read(args.evidence); pred = ordinary(args.predictions); ground = ordinary(args.ground_truth)
    report = read(args.report, evidence['report_sha256']); expected = read(args.metrics, evidence['metrics_sha256'])
    if sha(pred) != evidence['predictions_sha256'] or sha(ground) != GT_SHA or sha(args.ground_truth_manifest) != GT_MANIFEST_SHA:
        raise ValueError('original complete predictions/evaluator-only GT bytes required')
    if report['predictions_sha256'] != evidence['predictions_sha256'] or report['ground_truth_sha256'] != GT_SHA or report['status'] != 'completed' or report['paper_eligible'] is not False:
        raise ValueError('original seen-val car evaluation binding differs')
    source = sha(__file__); before = runtime(); started = time.monotonic()
    gt, predictions, counts = load_frames(ground, pred, report)
    n, curves, excluded = nuscenes_metrics(gt, predictions)
    actual = clean(dict(nuscenes=n, nuscenes_curves=curves, nuscenes_inapplicable_2hz_time_metrics_not_valid_seconds=excluded,
                        trackeval={'car': trackeval_metrics(gt, predictions)}))
    # Retain actual metrics even when independent comparison fails.
    result = dict(kind='rbf_stage1_independent_native_frame_official_metric_engine_recheck_v1', run_id=evidence['run_id'],
                  checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), runtime=before, source_sha256=source,
                  input_evidence=evidence, full_native_frames=3316, sequence_frame_counts=counts,
                  sealed_adapter_imported=False, independent_GEOS_free_convex_clipping=True, models_executed=False,
                  actual_metric_engines_executed=True, actual_metrics=actual, strict_replay_pass=evidence['strict_replay_pass'],
                  stage1_qualified=False, paper_metrics_complete=False)
    try:
        result['max_absolute_metric_difference'] = compare(actual, expected)
        if runtime() != before or sha(__file__) != source or sha(pred) != evidence['predictions_sha256'] or sha(ground) != GT_SHA:
            raise ValueError('input/author runtime/source changed during metric computation')
        result['all_original_full_sequence_and_cohort_metrics_independently_reproduced'] = True
    except BaseException as error:
        result.update(all_original_full_sequence_and_cohort_metrics_independently_reproduced=False,
                      error_type=type(error).__name__, error=str(error))
    result['seconds'] = time.monotonic() - started
    print('RBF_INDEPENDENT_METRICS_JSON ' + json.dumps(clean(result), sort_keys=True, allow_nan=False), flush=True)
    if result['all_original_full_sequence_and_cohort_metrics_independently_reproduced'] is not True: return 2
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence', 'ground-truth', 'ground-truth-manifest', 'predictions', 'report', 'metrics'):
        p.add_argument('--' + name, required=True)
    raise SystemExit(evaluate(p.parse_args()))
