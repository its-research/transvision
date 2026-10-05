"""Offline supervised examples for the all-class raw-score/top64 protocol.

This module does not fit on held-out data or certify dataset provenance. Callers
must verify manifests and sequence membership before supplying a frame.
"""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from transvision.models.event_track_v2x.prediction_features import (
    CLASSES, choose_candidates, match_predictions, wrap_angle, validate_pose,
    fit_score, fit_covariance,
)


def fit_groups(groups, *, score_config, covariance_config):
    """Fit one side's three classes; return explicit pooled fallback provenance.

    Each group is a list of frame_examples group records from admitted fit
    sequences only. No default or synthetic model is emitted for absent labels.
    """
    if set(groups) != set(CLASSES):
        raise ValueError('all three coarse classes required')
    minimum = covariance_config['minimum_class_samples']
    if type(minimum) is not int or minimum < 2:
        raise ValueError('invalid covariance support threshold')
    combined = {}
    for name in CLASSES:
        records = groups[name]
        if not records:
            raise ValueError('missing calibration frames for ' + name)
        scores = np.concatenate([r['scores'] for r in records])
        targets = np.concatenate([r['targets'] for r in records])
        residuals = np.concatenate([r['residuals'] for r in records])
        if (scores.ndim != 1 or targets.shape != scores.shape
                or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))
                or set(np.unique(targets)) != {0, 1}
                or residuals.ndim != 2 or residuals.shape[1] != 9
                or not np.isfinite(residuals).all()):
            raise ValueError('insufficient or invalid calibration examples for ' + name)
        combined[name] = (scores, targets, residuals)
    pooled = np.concatenate([combined[name][2] for name in CLASSES])
    if len(pooled) < minimum:
        raise ValueError('insufficient side residual support')
    result = {}
    for name, (scores, targets, residuals) in combined.items():
        support = len(residuals)
        fallback = support < minimum
        result[name] = dict(
            score=fit_score(scores, targets, score_config),
            covariance=fit_covariance(pooled if fallback else residuals, covariance_config),
            covariance_source='side-pooled-fallback' if fallback else 'side-class',
            class_covariance_support=support)
    return result


def sequence_velocity_targets(frames):
    """Return copied states and validity masks, keyed by (side, sequence, frame).

    Input rows contain metadata, gravity-centred states and source-local track
    IDs. Histories cannot cross sequences/sides. Ambiguous IDs break history.
    """
    history, result = {}, {}
    ordered = sorted(frames, key=lambda f: (
        f['metadata']['side'], f['metadata']['sequence_id'],
        f['metadata']['box_reference_timestamp_us'], f['metadata']['frame_id']))
    for frame in ordered:
        meta = frame['metadata']
        key = (meta['side'], meta['sequence_id'], meta['frame_id'])
        state = np.asarray(frame['state'], float).copy()
        tracks = [str(t) for t in frame['tracks']]
        timestamp = meta['box_reference_timestamp_us']
        if (key in result or state.shape != (len(tracks), 9)
                or not np.isfinite(state[:, :7]).all()
                or np.any(state[:, 3:6] <= 0)
                or type(timestamp) is not int):
            raise ValueError('invalid or duplicate supervision frame')
        rotation, translation = validate_pose(meta)
        world = state[:, :3] @ rotation + translation
        state[:, 7:9] = np.nan
        valid = np.zeros(len(state), dtype=bool)
        unique, counts = np.unique(tracks, return_counts=True)
        ambiguous = set(unique[counts > 1])
        for i, track in enumerate(tracks):
            identity = key[:2] + (track,)
            if track in ambiguous:
                history.pop(identity, None)
                continue
            previous = history.get(identity)
            if previous is not None:
                dt = (timestamp - previous[0]) / 1e6
                if dt <= 0:
                    raise ValueError('nonpositive identity history delta')
                velocity = ((world[i] - previous[1]) / dt) @ rotation.T
                if not np.isfinite(velocity).all():
                    raise ValueError('nonfinite reconstructed velocity')
                state[i, 7:9], valid[i] = velocity[:2], True
            history[identity] = (timestamp, world[i].copy())
        result[key] = (state, valid)
    return result


def frame_examples(state, scores, classes, gt_state, gt_classes, velocity_valid):
    state, gt_state = np.asarray(state, float), np.asarray(gt_state, float)
    scores, classes = np.asarray(scores, float), np.asarray(classes)
    gt_classes, velocity_valid = np.asarray(gt_classes), np.asarray(velocity_valid)
    n, g = len(state), len(gt_state)
    if (state.shape != (n, 9) or gt_state.shape != (g, 9)
            or scores.shape != (n,) or classes.shape != (n,)
            or gt_classes.shape != (g,) or velocity_valid.shape != (g,)
            or velocity_valid.dtype.kind != 'b'
            or classes.dtype.kind not in 'iu' or gt_classes.dtype.kind not in 'iu'
            or not np.isfinite(state).all() or not np.isfinite(scores).all()
            or not np.isfinite(gt_state[:, :7]).all()
            or np.any((scores < 0) | (scores > 1))
            or np.any((classes < 0) | (classes >= len(CLASSES)))
            or np.any((gt_classes < -1) | (gt_classes >= len(CLASSES)))
            or np.any(state[:, 3:6] <= 0) or np.any(gt_state[:, 3:6] <= 0)
            or not np.isfinite(gt_state[velocity_valid, 7:9]).all()):
        raise ValueError('invalid calibration frame or velocity validity')
    selected = choose_candidates(scores, {'minimum_raw_score': .05, 'maximum_per_side': 64})
    matches = match_predictions(state[selected], scores[selected], classes[selected],
                                gt_state, gt_classes, distance=2.)
    groups = {}
    for c, name in enumerate(CLASSES):
        local = np.flatnonzero(classes[selected] == c)
        positive = local[matches[local] >= 0]
        usable = positive[velocity_valid[matches[positive]]]
        residuals = state[selected[usable]] - gt_state[matches[usable]]
        residuals[:, 6] = wrap_angle(residuals[:, 6])
        groups[name] = dict(scores=scores[selected[local]].copy(),
                            targets=(matches[local] >= 0).astype(np.uint8),
                            residuals=residuals,
                            unknown_velocity_matches=len(positive) - len(usable))
    return dict(selected_raw_indices=selected, matches=matches, groups=groups)
