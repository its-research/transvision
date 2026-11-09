#!/usr/bin/env python3
"""Audit every complete emitted-score diagnostic label against frozen SPD inputs.

Real data runs on L40 CPU. Geometry/commit validation uses the qualified native
reader; assignment optimality uses a separate alternating-cycle checker. Scalar
statistics are recomputed here without the producer's score accumulator.
"""
import argparse
from collections import defaultdict
import datetime
import gzip
import json
import math
from pathlib import Path

from stage1_output_score_reliability import bev_iou, load_frames, read, sha
from stage1_score_matching_audit import audit_assignment


def scalar_statistics(observations, frames, gt):
    n = len(observations)
    tp = sum(y for _, y in observations)
    infinite = sum((s == 0 and y == 1) or (s == 1 and y == 0) for s, y in observations)
    finite_terms = [-math.log(s) if y else -math.log1p(-s) for s, y in observations
                    if not ((s == 0 and y == 1) or (s == 1 and y == 0))]
    bins = []
    errors = []
    for i in range(10):
        values = [(s, y) for s, y in observations if (i / 10 <= s < (i + 1) / 10) or (i == 9 and s == 1)]
        count = len(values)
        score = math.fsum(s for s, _ in values) / count if count else None
        freq = sum(y for _, y in values) / count if count else None
        bins.append(dict(index=i, lower=i / 10, upper=(i + 1) / 10, upper_inclusive=i == 9,
                         count=count, mean_emitted_score=score, TP_frequency=freq))
        if count:
            errors.append(count * abs(score - freq))
    if sum(x['count'] for x in bins) != n:
        raise ValueError('bin partition does not cover every observation')
    return dict(frames=frames, emitted_predictions=n, GT_objects=gt, TP=tp, FP=n - tp, FN=gt - tp,
                recall=tp / gt if gt else None,
                mean_emitted_score=math.fsum(s for s, _ in observations) / n if n else None,
                Brier=math.fsum((s - y) ** 2 for s, y in observations) / n if n else None,
                ECE=math.fsum(errors) / n if n else None,
                NLL=None if not n or infinite else math.fsum(finite_terms) / n,
                NLL_status='empty' if not n else 'positive_infinity' if infinite else 'finite',
                infinite_NLL_observations=infinite,
                finite_observation_NLL_mean=math.fsum(finite_terms) / len(finite_terms) if finite_terms else None,
                clipping_applied=False, reliability_bins=bins)


def compare(actual, expected, tolerance, path='root'):
    if type(actual) is dict:
        if type(expected) is not dict or set(actual) != set(expected):
            raise ValueError('receipt keys differ at ' + path)
        return max([compare(actual[k], expected[k], tolerance, path + '.' + k) for k in actual] + [0.])
    if type(actual) is list:
        if type(expected) is not list or len(actual) != len(expected):
            raise ValueError('receipt list differs at ' + path)
        return max([compare(a, b, tolerance, path + '[' + str(i) + ']')
                    for i, (a, b) in enumerate(zip(actual, expected))] + [0.])
    if type(actual) is float:
        if type(expected) not in (float, int) or isinstance(expected, bool) or not math.isfinite(expected):
            raise ValueError('receipt scalar type differs at ' + path)
        error = abs(actual - expected)
        if error > tolerance:
            raise ValueError('receipt scalar differs at ' + path)
        return error
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError('receipt exact value differs at ' + path)
    return 0.


def audit_run(spec, ground, predictions, report, label_path, expected, tolerance, native_metrics=None):
    if sha(label_path) != expected['label_rows_sha256']:
        raise ValueError('producer complete label byte SHA differs')
    gt, pred, counts = load_frames(ground, predictions, report)
    all_observations, per_sequence = [], defaultdict(list)
    gt_counts, frame_counts = defaultdict(int), defaultdict(int)
    audited_frames = 0
    with gzip.open(label_path, 'rt') as rows:
        for g, p in zip(gt, pred):
            matrix = [[bev_iou(a['mean'], b['mean']) for b in p['boxes']] for a in g['boxes']]
            ground_ids = {b['track_id']: i for i, b in enumerate(g['boxes'])}
            labels = []
            for j, box in enumerate(p['boxes']):
                line = next(rows, None)
                if line is None:
                    raise ValueError('incomplete label coverage')
                value = json.loads(line)
                required = dict(sequence_id=g['sid'], frame_id=g['fid'], box_reference_timestamp_us=g['time'],
                                predicted_track_id=box['track_id'], emitted_score=box['score'])
                if any(value.get(k) != v or type(value.get(k)) is not type(v) for k, v in required.items()):
                    raise ValueError('label does not bind exact original frame, identity and score')
                y = value['TP_label']
                if type(y) is not int or y not in (0, 1):
                    raise ValueError('nonbinary label')
                if y:
                    i = ground_ids.get(value['matched_GT_track_id'])
                    if i is None or matrix[i][j] < .5 or type(value['matched_BEV_IoU']) is not float:
                        raise ValueError('invalid claimed GT match')
                    if not math.isfinite(value['matched_BEV_IoU']) or abs(value['matched_BEV_IoU'] - matrix[i][j]) > tolerance:
                        raise ValueError('claimed match geometry differs')
                    labels.append(i)
                else:
                    if value['matched_GT_track_id'] is not None or value['matched_BEV_IoU'] is not None:
                        raise ValueError('unmatched output has claimed GT identity')
                    labels.append(None)
                all_observations.append((box['score'], y))
                per_sequence[g['sid']].append((box['score'], y))
            audit_assignment(matrix, labels, tolerance=tolerance)
            audited_frames += 1
            frame_counts[g['sid']] += 1
            gt_counts[g['sid']] += len(g['boxes'])
        if next(rows, None) is not None:
            raise ValueError('extra prediction label rows')
    # Explicitly include empty sequences rather than inferring scope from observations.
    actual = dict(aggregate=scalar_statistics(all_observations, audited_frames, sum(gt_counts.values())),
                  per_sequence={sid: scalar_statistics(per_sequence[sid], counts[sid], gt_counts[sid]) for sid in sorted(counts)},
                  complete_sequence_frame_counts=counts, label_rows_sha256=sha(label_path),
                  strict_replay_pass=spec['strict_replay_pass'],
                  accepted_three_seed_statistics_eligible=spec['strict_replay_pass'])
    if native_metrics is not None:
        reference = native_metrics['trackeval']['car']['sequences']
        if set(counts) != set(reference):
            raise ValueError('independent native ROI sequence population differs')
        for sid in counts:
            if (counts[sid], gt_counts[sid], len(per_sequence[sid])) != (
                    reference[sid]['frames'], reference[sid]['gt_objects'], reference[sid]['predicted_objects']):
                raise ValueError('independent native ROI population differs: ' + sid)
    error = compare(actual, expected, tolerance)
    return dict(run_id=spec['run_id'], audited_frames=audited_frames, audited_prediction_labels=len(all_observations),
                audited_sequences=len(counts), independently_recomputed_scalars=actual,
                maximum_scalar_absolute_error=error, full_matching_alternating_cycle_check_passed=True,
                original_failed_replay_status_preserved=not spec['strict_replay_pass'])


def audit(audit_plan_path, input_root, result_root, output_path):
    protocol = read(audit_plan_path)
    for name, expected in protocol['audit_sources'].items():
        if sha(Path(__file__).parent / name) != expected:
            raise ValueError('independent audit source changed')
    plan_path = Path(input_root).parent / 'source/plan.json'
    if sha(plan_path) != protocol['scientific_plan_sha256']:
        raise ValueError('original frozen science plan differs')
    plan = read(plan_path)
    data, results = Path(input_root), Path(result_root)
    for item in plan['inputs']:
        p = data / item['member']
        if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise ValueError('complete original input byte binding differs')
    receipt_path = results / 'score-reliability-receipt.json'
    if sha(receipt_path) != protocol['science_receipt_sha256']:
        raise ValueError('original scientific output receipt differs')
    receipt = read(receipt_path)
    if receipt['plan_sha256'] != protocol['scientific_plan_sha256'] or receipt['complete_configs'] != 13:
        raise ValueError('original 13-config science scope differs')
    if set(receipt['runs']) != {s['run_id'] for s in plan['runs']}:
        raise ValueError('original config set differs')
    audits = []
    for spec in plan['runs']:
        audits.append(audit_run(spec, data / plan['GT_member'], data / spec['prediction_member'],
            read(data / spec['report_member']), results / (spec['run_id'] + '-labels.jsonl.gz'),
            receipt['runs'][spec['run_id']], protocol['absolute_tolerance'], read(data / spec['native_metrics_member'])))
        print('RBF_SCORE_AUDIT_RUN_COMPLETE ' + spec['run_id'], flush=True)
    for item in plan['inputs']:
        if sha(data / item['member']) != item['sha256']:
            raise ValueError('input changed during independent audit')
    result = dict(kind='complete13_SPD_emitted_score_label_and_scalar_independent_audit_v1',
                  checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  audits=audits, independent_matching_algorithm='contracted alternating-cycle residual graph',
                  producer_Hungarian_or_accumulator_executed=False,
                  same_qualified_native_geometry_and_commit_reader=True,
                  numerical_tolerance=protocol['absolute_tolerance'], formal_certificate=False,
                  full_K_complete=False, identity_posterior_calibration=False, paper_accepted=False)
    with Path(output_path).open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-plan', 'input-root', 'result-root', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    audit(a.audit_plan, a.input_root, a.result_root, a.output)
