#!/usr/bin/env python3
"""Actual unchanged-author matching for every distinct per-scene score subset.

This is evaluator-only CPU work, not a predictor, deployment or cost benchmark.
The full 40-threshold original cohort must be reconstructed before cache admission.
"""
import argparse
from collections import defaultdict
import datetime
import importlib.util
import math
from pathlib import Path
import time
import json
import numpy as np


def module(path, digest, name):
    import hashlib
    p = Path(path)
    if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
        raise ValueError('frozen independent source required')
    spec = importlib.util.spec_from_file_location(name, p)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def tracking_inputs(gt, pred):
    from nuscenes.eval.tracking.data_classes import TrackingBox
    ground, predicted, scores = defaultdict(dict), defaultdict(dict), defaultdict(list)
    for g, p in zip(gt, pred):
        sid, timestamp = g['sid'], g['time']; ground[sid][timestamp], predicted[sid][timestamp] = [], []
        for frame, destination, is_gt in [(g, ground[sid][timestamp], True), (p, predicted[sid][timestamp], False)]:
            for box in frame['boxes']:
                m = box['mean']; tid = sid + '/' + box['track_id']
                destination.append(TrackingBox(sample_token=sid + '/' + g['fid'], translation=tuple(m[:3]),
                    size=(m[4], m[3], m[5]), rotation=(math.cos(m[6] / 2), 0., 0., math.sin(m[6] / 2)),
                    velocity=tuple(m[7:9]), tracking_id=tid, tracking_name='car',
                    tracking_score=-1. if is_gt else float(box['score'])))
                if not is_gt: scores[sid, tid].append(box['score'])
    means = {key: float(np.mean(value)) for key, value in scores.items()}
    for sid, times in predicted.items():
        for boxes in times.values():
            for b in boxes: b.tracking_score = means[sid, b.tracking_id]
    return ground, predicted


def actual_statistics(acc, mh, oracle):
    from motmetrics.metrics import events_to_df_map
    result = mh.compute(acc, metrics=list(oracle.COUNT_KEYS), name='actual_score_subset').iloc[0]
    rows = [int(result[key]) for key in oracle.COUNT_KEYS]
    if any(result[key] != value for key, value in zip(oracle.COUNT_KEYS, rows)):
        raise ValueError('actual author count is not integer')
    rows.append(float(events_to_df_map(acc).noraw['D'].sum()))
    return rows


def evaluate(args):
    helper = module(args.helper_source, args.helper_sha256, 'independent_native_cache_input')
    oracle = module(args.oracle_source, args.oracle_sha256, 'independent_native_cache_aggregation')
    evidence = helper.read(args.evidence)
    report = helper.read(args.report, evidence['report_sha256'])
    expected = helper.read(args.metrics, evidence['metrics_sha256'])
    pins = {args.helper_source: args.helper_sha256, args.oracle_source: args.oracle_sha256,
        args.predictions: evidence['predictions_sha256'], args.ground_truth: helper.GT_SHA,
        args.ground_truth_manifest: helper.GT_MANIFEST_SHA, args.report: evidence['report_sha256'],
        args.metrics: evidence['metrics_sha256'], args.evidence: helper.sha(args.evidence),
        str(Path(__file__).absolute()): helper.sha(__file__)}
    for p, digest in pins.items():
        if helper.sha(helper.ordinary(p)) != digest: raise ValueError('original admitted cache input differs')
    if report['predictions_sha256'] != evidence['predictions_sha256'] or report['ground_truth_sha256'] != helper.GT_SHA or report['status'] != 'completed' or report['paper_eligible'] is not False:
        raise ValueError('original exploratory car protocol required')
    started = time.monotonic(); runtime_before = helper.runtime()
    gt, predictions, frame_counts = helper.load_frames(args.ground_truth, args.predictions, report)
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.algo import TrackingEvaluation
    from nuscenes.eval.tracking.utils import create_motmetrics
    # TrackingConfig initializes the author's process-global TRACKING_NAMES.
    # Native TrackingBox construction must follow that initialization in every
    # fresh process, exactly as the qualified original independent evaluator.
    cfg = config_factory('tracking_nips_2019'); mh = create_motmetrics()
    ground, predicted = tracking_inputs(gt, predictions)
    if cfg.min_recall != .1 or cfg.num_thresholds != 40 or cfg.dist_th_tp != 2.:
        raise ValueError('original threshold configuration differs')
    cache = dict(kind='rbf_stage1_actual_author_all_distinct_scene_score_subset_statistics_v1',
        run_id=evidence['run_id'], sequence_ids=list(ground), sequences={}, metric_worst=cfg.metric_worst,
        native_frame_counts=frame_counts, input_evidence=evidence, input_sha256=pins,
        runtime=runtime_before, source_sha256=helper.sha(__file__),
        strict_replay_pass=evidence['strict_replay_pass'], stage1_qualified=False, paper_metrics_complete=False)
    try:
        for sid in ground:
            engine = TrackingEvaluation({sid: ground[sid]}, {sid: predicted[sid]}, 'car', cfg.dist_fcn_callable,
                cfg.dist_th_tp, cfg.min_recall, cfg.num_thresholds, cfg.metric_worst, verbose=False)
            unfiltered, scores = engine.accumulate_threshold(None)
            cuts = sorted({b.tracking_score for boxes in predicted[sid].values() for b in boxes})
            rows = [actual_statistics(unfiltered, mh, oracle)]
            for threshold in cuts[1:] + ([float(np.nextafter(cuts[-1], np.inf))] if cuts else []):
                acc, _ = engine.accumulate_threshold(threshold)
                rows.append(actual_statistics(acc, mh, oracle))
            gt_boxes = [b for boxes in ground[sid].values() for b in boxes]
            cache['sequences'][sid] = dict(trajectory_score_cutoffs=cuts, rows=rows,
                unfiltered_MATCH_scores=scores, GT_box_count=len(gt_boxes),
                GT_track_count=len({b.tracking_id for b in gt_boxes}),
                all_distinct_prediction_score_subsets_actually_matched=True)
            print(json.dumps(dict(stage='actual_all_local_score_subsets', run_id=evidence['run_id'],
                sequence_id=sid, completed_sequences=len(cache['sequences']), total_sequences=21,
                actual_subset_rows=len(rows))), flush=True)
        headline, curves = oracle.evaluate(oracle.prepare(cache), np.ones(21, dtype=int))
        expected_headline = {key: expected['nuscenes']['label_metrics'][key]['car'] for key in headline}
        expected_curves = {key: expected['nuscenes_curves']['car'][key] for key in curves}
        cache['full_original_40_threshold_headline_and_curves_max_difference'] = max(
            helper.compare(helper.clean(headline), expected_headline),
            helper.compare(helper.clean(curves), expected_curves))
        if helper.runtime() != runtime_before: raise ValueError('original author runtime changed')
        for p, digest in pins.items():
            if helper.sha(p) != digest: raise ValueError('original inputs changed during full subset export')
        cache['all21_actual_score_subset_caches_original_full_cohort_admitted'] = True
    except BaseException as error:
        cache.update(all21_actual_score_subset_caches_original_full_cohort_admitted=False,
                     error_type=type(error).__name__, error=str(error))
    cache.update(seconds=time.monotonic()-started, checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    print('RBF_NUSCENES_CACHE_JSON ' + json.dumps(helper.clean(cache), sort_keys=True, allow_nan=False), flush=True)
    return 0 if cache['all21_actual_score_subset_caches_original_full_cohort_admitted'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('helper-source','helper-sha256','oracle-source','oracle-sha256','evidence',
                 'ground-truth','ground-truth-manifest','predictions','report','metrics'):
        parser.add_argument('--' + name, required=True)
    raise SystemExit(evaluate(parser.parse_args()))
