"""Sequence-resampled nuScenes aggregation from exact local score-subset caches.

Each cache row comes from unchanged author matching at an actual trajectory-score
cutoff. Resampling repeats independent complete scenes and recomputes global TP
score thresholds; it never averages per-scene AMOTA or frozen threshold curves.
"""
import numpy as np

COUNT_KEYS = ('num_frames', 'num_objects', 'num_matches', 'num_misses',
              'num_switches', 'num_false_positives', 'num_detections',
              'mostly_tracked', 'mostly_lost', 'num_fragmentations_custom')
RAW_KEYS = COUNT_KEYS + ('distance_sum',)
METRICS = ('gt', 'tp', 'motar', 'mota', 'motp', 'faf', 'mt', 'ml',
           'fp', 'fn', 'ids', 'frag', 'recall')


def prepare(cache):
    order = cache['sequence_ids']; sequences = {}
    if len(order) != len(set(order)) or set(order) != set(cache['sequences']):
        raise ValueError('complete unique original sequences required')
    scores, indexes = [], []
    for i, sid in enumerate(order):
        seq = cache['sequences'][sid]; cuts = np.asarray(seq['trajectory_score_cutoffs'], dtype=float)
        rows = np.asarray(seq['rows'], dtype=float)
        tp = np.asarray(seq['unfiltered_MATCH_scores'], dtype=float)
        if cuts.ndim != 1 or not np.isfinite(cuts).all() or np.any(np.diff(cuts) <= 0) or np.any((cuts < 0) | (cuts > 1)):
            raise ValueError('unique sorted actual trajectory scores required')
        if rows.shape != (len(cuts) + 1, len(RAW_KEYS)) or not np.isfinite(rows).all() or np.any(rows < 0):
            raise ValueError('complete finite score-subset count/distance cache required')
        if np.any(rows[:, :-1] != np.floor(rows[:, :-1])):
            raise ValueError('integer author event counts required')
        if not np.all(rows[:, 1] == seq['GT_box_count']) or seq['GT_track_count'] > seq['GT_box_count']:
            raise ValueError('GT count must be invariant to prediction cutoff')
        if np.any(rows[:, 6] != rows[:, 2] + rows[:, 4]):
            raise ValueError('author detections must include MATCH and SWITCH')
        if len(tp) != rows[0, 2] or not np.isfinite(tp).all() or np.any((tp < 0) | (tp > 1)):
            raise ValueError('unfiltered author MATCH score inventory differs')
        if np.any(~np.isin(tp, cuts)):
            raise ValueError('TP score not in actual trajectory score inventory')
        seq = dict(seq, cuts=cuts, raw=rows)
        sequences[sid] = seq
        scores.extend(tp); indexes.extend([i] * len(tp))
    scores = np.asarray(scores); indexes = np.asarray(indexes, dtype=int)
    permutation = np.argsort(-scores, kind='stable')
    return dict(sequence_ids=order, sequences=sequences, sorted_TP_scores=scores[permutation],
                sorted_TP_sequence_index=indexes[permutation], metric_worst=cache['metric_worst'])


def evaluate(prepared, weights):
    weights = np.asarray(weights)
    if weights.shape != (len(prepared['sequence_ids']),) or np.any(weights < 0) or np.any(weights != np.floor(weights)):
        raise ValueError('nonnegative integer complete-sequence multiplicities required')
    weights = weights.astype(int)
    seqs = [prepared['sequences'][sid] for sid in prepared['sequence_ids']]
    gt = int(sum(w * s['GT_box_count'] for w, s in zip(weights, seqs)))
    tracks = int(sum(w * s['GT_track_count'] for w, s in zip(weights, seqs)))
    if gt <= 0: raise ValueError('resampled cohort requires positive GT')
    scores = np.repeat(prepared['sorted_TP_scores'], weights[prepared['sorted_TP_sequence_index']])
    recalls = np.linspace(.1, 1., 40).round(12)
    if len(scores) > gt: raise ValueError('author MATCH count exceeds complete GT count')
    if len(scores):
        rec = np.arange(1, len(scores) + 1) / gt
        thresholds = np.interp(recalls, rec, scores, right=0.)
        thresholds[recalls > rec[-1]] = np.nan
        thresholds, recalls = thresholds[::-1].copy(), recalls[::-1].copy()
    else:
        thresholds = np.full(40, np.nan); recalls = np.full(40, np.nan)
    curves = dict(confidence=thresholds, recall_hypo=recalls)
    valid = np.isfinite(thresholds)
    if np.any(valid):
        raw = np.zeros((int(valid.sum()), len(RAW_KEYS)))
        for w, seq in zip(weights, seqs):
            if w:
                # Filtering is author score >= threshold; the final cache row
                # contains the genuine empty-prediction state above max score.
                indexes = np.searchsorted(seq['cuts'], thresholds[valid], side='left')
                raw += w * seq['raw'][indexes]
        values = dict(zip(RAW_KEYS, raw.T))
        matches, detections = values['num_matches'], values['num_detections']
        misses, switches, fp = (values[k] for k in ('num_misses', 'num_switches', 'num_false_positives'))
        with np.errstate(divide='ignore', invalid='ignore'):
            # Keep the original author's arithmetic order, including MATCH-
            # based MOTAR versus detection-based recall and MOTP.
            recall_for_motar = matches / gt
            numerator = (misses + switches + fp) - (1 - recall_for_motar) * gt
            motar = np.maximum(0., 1. - numerator / (recall_for_motar * gt))
            mota = np.maximum(0., 1. - (misses + switches + fp) / gt)
            motp = values['distance_sum'] / detections
            faf = fp / values['num_frames'] * 100
        derived = dict(gt=values['num_objects'], tp=matches, motar=motar, mota=mota, motp=motp, faf=faf,
                       mt=values['mostly_tracked'], ml=values['mostly_lost'], fp=fp, fn=misses,
                       ids=switches, frag=values['num_fragmentations_custom'], recall=detections / gt)
        for name, result in derived.items():
            curves[name] = np.full(40, np.nan); curves[name][valid] = result
    else:
        for name in METRICS:
            worst = prepared['metric_worst'][name]
            if worst == -1:
                if name == 'ml': worst = tracks
                elif name in ('gt', 'fn'): worst = gt
                elif name in ('fp', 'ids', 'frag'): worst = np.nan
                else: raise ValueError('unimplemented original worst-case metric')
            curves[name] = np.full(40, float(worst))
    # First nanargmax preserves the author's highest-recall tie rule.
    best = None if np.isnan(curves['mota']).all() else int(np.nanargmax(curves['mota']))
    headline = {k: float(curves[k][best]) for k in METRICS} if best is not None else {}
    for name, curve in (('amota', 'motar'), ('amotp', 'motp')):
        values = curves[curve]
        headline[name] = float(np.mean(np.where(np.isnan(values), prepared['metric_worst'][name], values))) if not np.isnan(values).all() else float('nan')
    return headline, curves
