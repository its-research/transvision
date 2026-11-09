#!/usr/bin/env python3
"""Complete SPD emitted-score reliability; a diagnostic, not identity posterior calibration.

Geometry/frame commitment checks copied from the independently qualified native
Stage1 evaluator. No predictor, metric engine or production geometry is imported.
Real-data execution belongs on the admitted L40 CPU host.
"""
import argparse
from collections import defaultdict
import datetime
import hashlib
import gzip
import json
import math
from pathlib import Path
import time

CLASSES = {'car', 'bicycle', 'pedestrian'}
IOU_THRESHOLD = 0.5
BIN_COUNT = 10

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

def assignment_labels(similarities, prediction_count):
    """Maximize valid match cardinality, then total IoU, without using scores.

    Rectangular Hungarian algorithm. Each GT has an available unmatched dummy
    column. The cardinality bonus exceeds every possible total-IoU improvement.
    Input ordering, not confidence, breaks exact assignment ties.
    """
    n = len(similarities)
    if type(prediction_count) is not int or prediction_count < 0:
        raise ValueError('invalid prediction count')
    if any(len(row) != prediction_count for row in similarities):
        raise ValueError('nonrectangular matching input')
    if any(not math.isfinite(x) or not 0 <= x <= 1 for row in similarities for x in row):
        raise ValueError('invalid IoU')
    labels = [None] * prediction_count
    if not n or not prediction_count:
        return labels
    bonus = min(n, prediction_count) + 1
    costs = [[-(bonus + x) if x >= IOU_THRESHOLD else 0. for x in row] + [0.] * n
             for row in similarities]
    m = prediction_count + n
    u, v, p, way = [0.] * (n + 1), [0.] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0, minimum, used = 0, [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minimum[j]:
                    minimum[j], way[j] = cur, j0
                if minimum[j] < delta:
                    delta, j1 = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if not j0:
                break
    for j in range(1, prediction_count + 1):
        if p[j] and similarities[p[j] - 1][j - 1] >= IOU_THRESHOLD:
            labels[j - 1] = p[j] - 1
    return labels


class ScoreAccumulator:
    def __init__(self):
        self.n = self.tp = self.gt = self.frames = self.infinite_nll = 0
        self.score_terms, self.brier_terms, self.finite_nll_terms = [], [], []
        self.bins = [[] for _ in range(BIN_COUNT)]

    def add(self, score, label):
        if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('score outside unit interval')
        if type(label) is not int or label not in (0, 1):
            raise ValueError('binary match label required')
        self.n += 1
        self.tp += label
        self.score_terms.append(score)
        self.brier_terms.append((score - label) ** 2)
        if (label == 1 and score == 0) or (label == 0 and score == 1):
            self.infinite_nll += 1
        else:
            self.finite_nll_terms.append(-math.log(score) if label else -math.log1p(-score))
        self.bins[sum(score >= i / BIN_COUNT for i in range(1, BIN_COUNT))].append((score, label))

    def result(self):
        bins = []
        errors = []
        for i, values in enumerate(self.bins):
            count = len(values)
            confidence = math.fsum(x[0] for x in values) / count if count else None
            frequency = sum(x[1] for x in values) / count if count else None
            bins.append(dict(index=i, lower=i / BIN_COUNT, upper=(i + 1) / BIN_COUNT,
                             upper_inclusive=i == BIN_COUNT - 1, count=count,
                             mean_emitted_score=confidence, TP_frequency=frequency))
            if count:
                errors.append(count * abs(confidence - frequency))
        return dict(frames=self.frames, emitted_predictions=self.n, GT_objects=self.gt,
                    TP=self.tp, FP=self.n - self.tp, FN=self.gt - self.tp,
                    recall=self.tp / self.gt if self.gt else None,
                    mean_emitted_score=math.fsum(self.score_terms) / self.n if self.n else None,
                    Brier=math.fsum(self.brier_terms) / self.n if self.n else None,
                    ECE=math.fsum(errors) / self.n if self.n else None,
                    NLL=None if not self.n or self.infinite_nll else math.fsum(self.finite_nll_terms) / self.n,
                    NLL_status='empty' if not self.n else 'positive_infinity' if self.infinite_nll else 'finite',
                    infinite_NLL_observations=self.infinite_nll,
                    finite_observation_NLL_mean=math.fsum(self.finite_nll_terms) / len(self.finite_nll_terms)
                        if self.finite_nll_terms else None,
                    clipping_applied=False, reliability_bins=bins)


def verify_population(result, sequences, native_metrics, pre_roi_car_predictions):
    """Bind post-ROI populations to the already-qualified full native metrics.

    report.coverage.car_predictions is a pre-ROI class census, not this population.
    Exact per-sequence frame/GT/prediction counts preserve the original ROI scope.
    """
    expected = native_metrics['trackeval']['car']['sequences']
    if set(sequences) != set(expected):
        raise ValueError('original complete ROI sequence set differs')
    for sid, actual in sequences.items():
        reference = expected[sid]
        for actual_key, reference_key in [('frames', 'frames'), ('GT_objects', 'gt_objects'),
                                          ('emitted_predictions', 'predicted_objects')]:
            if actual[actual_key] != reference[reference_key]:
                raise ValueError('original ROI population differs: ' + sid + '.' + actual_key)
    gt_count = sum(x['gt_objects'] for x in expected.values())
    pred_count = sum(x['predicted_objects'] for x in expected.values())
    if result['GT_objects'] != gt_count or result['emitted_predictions'] != pred_count:
        raise ValueError('original complete aggregate ROI counts differ')
    if pre_roi_car_predictions < pred_count:
        raise ValueError('post-ROI count exceeds original pre-ROI car census')


def evaluate_run(ground, predictions, report, rows_output, native_metrics):
    gt, pred, counts = load_frames(ground, predictions, report)
    all_scores, sequences = ScoreAccumulator(), defaultdict(ScoreAccumulator)
    with gzip.open(rows_output, 'xt', compresslevel=1) as output:
        for g, p in zip(gt, pred):
            matches = assignment_labels([[bev_iou(a['mean'], b['mean']) for b in p['boxes']]
                                         for a in g['boxes']], len(p['boxes']))
            for accumulator in (all_scores, sequences[g['sid']]):
                accumulator.frames += 1
                accumulator.gt += len(g['boxes'])
            for i, (box, match) in enumerate(zip(p['boxes'], matches)):
                label = int(match is not None)
                for accumulator in (all_scores, sequences[g['sid']]):
                    accumulator.add(box['score'], label)
                output.write(json.dumps(dict(sequence_id=g['sid'], frame_id=g['fid'],
                    box_reference_timestamp_us=g['time'], predicted_track_id=box['track_id'],
                    emitted_score=box['score'], TP_label=label,
                    matched_GT_track_id=g['boxes'][match]['track_id'] if label else None,
                    matched_BEV_IoU=bev_iou(g['boxes'][match]['mean'], box['mean']) if label else None),
                    sort_keys=True, allow_nan=False) + '\n')
    result = all_scores.result()
    sequence_results = {k: v.result() for k, v in sorted(sequences.items())}
    verify_population(result, sequence_results, native_metrics, report['coverage']['car_predictions'])
    return dict(aggregate=result, per_sequence=sequence_results,
                complete_sequence_frame_counts=counts, label_rows_sha256=sha(rows_output))


def run(plan_path, input_root, output_root):
    plan = read(plan_path)
    root, output = ordinary(plan_path).parent, Path(output_root)
    output.mkdir(exist_ok=False, parents=True)
    expected_source = plan['sources']['stage1_output_score_reliability.py']['sha256']
    if sha(__file__) != expected_source:
        raise ValueError('frozen evaluator source differs')
    data = Path(input_root)
    for item in plan['inputs']:
        p = ordinary(data / item['member'])
        if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise ValueError('frozen input bytes differ: ' + item['member'])
    results = {}
    started = time.monotonic()
    for spec in plan['runs']:
        report = read(data / spec['report_member'])
        if report['status'] != 'completed' or report['paper_eligible'] is not False:
            raise ValueError('original exploratory report required')
        if report['predictions_sha256'] != spec['predictions_sha256'] or report['ground_truth_sha256'] != plan['GT_sha256']:
            raise ValueError('report input binding differs')
        r = evaluate_run(data / plan['GT_member'], data / spec['prediction_member'], report,
                         output / (spec['run_id'] + '-labels.jsonl.gz'), read(data / spec['native_metrics_member']))
        r['strict_replay_pass'] = spec['strict_replay_pass']
        r['accepted_three_seed_statistics_eligible'] = spec['strict_replay_pass']
        results[spec['run_id']] = r
        print('RBF_SCORE_RUN_COMPLETE ' + spec['run_id'], flush=True)
    # Refuse mutation across the computation, including the evaluator.
    if sha(__file__) != expected_source:
        raise ValueError('evaluator changed during execution')
    for item in plan['inputs']:
        if sha(data / item['member']) != item['sha256']:
            raise ValueError('input changed during execution')
    receipt = dict(kind='complete_SPD_stage1_emitted_score_TP_reliability_diagnostic_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        plan_sha256=sha(plan_path), evaluator_sha256=expected_source, runs=results,
        semantics=plan['semantics'], seconds=time.monotonic() - started,
        complete_configs=len(results), full_K_complete=False, full_E_complete=False,
        identity_posterior_calibration=False, physical_identity_risk=False,
        joint_TV_error_proof=False, quality_bound_certificate=False,
        calibration_fitting=False, resource_selection=False, paper_accepted=False)
    with (output / 'score-reliability-receipt.json').open('x') as f:
        json.dump(receipt, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--input-root', required=True)
    p.add_argument('--output-root', required=True)
    args = p.parse_args()
    run(args.plan, args.input_root, args.output_root)
