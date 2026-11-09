import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('nuscenes_statistics_oracle', Path(__file__).parents[2] / 'tools/event_track_v2x/stage1_nuscenes_statistics_oracle.py')
oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)
WORST = dict(gt=-1, tp=0, motar=0, mota=0, motp=2, faf=500, mt=0, ml=-1, fp=-1,
             fn=-1, ids=-1, frag=-1, recall=0, amota=0, amotp=2)


def raw(gt, matches, switches=0, fp=0, distance=.3):
    detections = matches + switches
    return [gt, gt, matches, gt - detections, switches, fp, detections, 0, int(detections == 0), 0, distance * detections]


def sequence(score, gt=2):
    return dict(trajectory_score_cutoffs=[score], unfiltered_MATCH_scores=[score], GT_box_count=gt,
                GT_track_count=1, rows=[raw(gt, 1), raw(gt, 0)])


def cache():
    return dict(sequence_ids=['A', 'B'], sequences={'A': sequence(.9), 'B': sequence(.1, 8)}, metric_worst=WORST)


def test_global_thresholds_are_recomputed_from_repeated_TP_scores_and_GT_counts():
    source = cache(); prepared = oracle.prepare(source)
    headline, curves = oracle.evaluate(prepared, [2, 1])
    recall = np.linspace(.1, 1, 40).round(12)
    scores = np.array([.9, .9, .1]); rec = np.arange(1, 4) / 12
    expected = np.interp(recall, rec, scores, right=0)
    expected[recall > .25] = np.nan
    np.testing.assert_array_equal(curves['confidence'], expected[::-1])
    assert np.all(curves['gt'][np.isfinite(curves['confidence'])] == 12)
    assert headline['amotp'] > .3  # Penalize unachieved recall with 2 meters.
    json.dumps(source)  # Preparing numerical caches must not mutate original JSON.


def test_counts_recombine_before_metrics_instead_of_averaging_sequence_metrics():
    prepared = oracle.prepare(cache())
    _, combined = oracle.evaluate(prepared, [1, 1])
    _, a = oracle.evaluate(prepared, [1, 0]); _, b = oracle.evaluate(prepared, [0, 1])
    assert not np.array_equal(combined['confidence'], (a['confidence'] + b['confidence']) / 2, equal_nan=True)
    assert combined['gt'][-1] == 10


def test_author_MATCH_based_MOTAR_differs_from_detection_based_recall():
    c = dict(sequence_ids=['A'], sequences={'A': dict(trajectory_score_cutoffs=[.7],
        unfiltered_MATCH_scores=[.7] * 3, GT_box_count=5, GT_track_count=1,
        rows=[raw(5, 3, switches=1), raw(5, 0)])}, metric_worst=WORST)
    headline, curves = oracle.evaluate(oracle.prepare(c), [1])
    assert headline['tp'] == 3 and headline['ids'] == 1
    assert headline['recall'] == .8 and headline['motp'] == pytest.approx(.3)
    assert headline['motar'] == pytest.approx(1.)  # Original MATCH-based correction.
    assert headline['mota'] == .6


def test_highest_recall_threshold_wins_an_equal_MOTA_tie():
    c = dict(sequence_ids=['A'], sequences={'A': dict(trajectory_score_cutoffs=[.2, .8],
        unfiltered_MATCH_scores=[.8, .2], GT_box_count=2, GT_track_count=1,
        rows=[raw(2, 2, fp=1), raw(2, 1), raw(2, 0)])}, metric_worst=WORST)
    headline, _ = oracle.evaluate(oracle.prepare(c), [1])
    assert headline['mota'] == .5 and headline['tp'] == 2 and headline['fp'] == 1


def test_no_matches_uses_author_worst_case_rules_with_unknown_error_counts():
    c = dict(sequence_ids=['A'], sequences={'A': dict(trajectory_score_cutoffs=[],
        unfiltered_MATCH_scores=[], GT_box_count=3, GT_track_count=2, rows=[raw(3, 0)])}, metric_worst=WORST)
    h, curves = oracle.evaluate(oracle.prepare(c), [2])
    assert h['amota'] == 0 and h['amotp'] == 2 and h['fn'] == h['gt'] == 6
    assert h['ml'] == 4 and np.isnan(h['fp']) and np.isnan(h['ids']) and np.isnan(h['frag'])
    assert np.isnan(curves['confidence']).all()


@pytest.mark.parametrize('corrupt', ['missing_subset', 'fractional_count', 'missing_TP', 'bad_GT'])
def test_invalid_sufficient_statistics_are_rejected(corrupt):
    c = copy.deepcopy(cache()); s = c['sequences']['A']
    if corrupt == 'missing_subset': s['rows'].pop()
    elif corrupt == 'fractional_count': s['rows'][0][0] = 1.2
    elif corrupt == 'missing_TP': s['unfiltered_MATCH_scores'] = []
    else: s['rows'][1][1] = 99
    with pytest.raises(ValueError): oracle.prepare(c)
