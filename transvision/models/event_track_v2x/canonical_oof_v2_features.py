"""Canonical role-bound V2 encoder, retaining the trained 203-feature recipe."""
import numpy as np

from .detection_cache_v2 import DetectionCacheV2
from .prediction_features import choose_candidates, encode_features


def frame_features(frame, reference, calibration, *, calibration_sha256,
                   fold_id, role, decision_us, origin_us):
    if not isinstance(frame, DetectionCacheV2) or not isinstance(reference, DetectionCacheV2):
        raise TypeError('sealed V2 source and vehicle reference required')
    if role not in {'fit', 'held_out'} or type(decision_us) is not int or type(origin_us) is not int:
        raise ValueError('explicit canonical role and integer decision/origin required')
    binding = calibration['canonical_oof_binding']
    if type(fold_id) is not int or binding['fold_id'] != fold_id:
        raise ValueError('canonical fold differs')
    fit, held = calibration['fit_sequences'], binding['held_out_sequence_ids']
    if set(fit) & set(held) or len(set(fit) | set(held)) != 46:
        raise ValueError('fold roles overlap or omit the train cohort')
    allowed = fit if role == 'fit' else held
    fm, rm = frame.metadata, reference.metadata
    if (rm['side'] != 'vehicle-side' or fm['sequence_id'] != rm['sequence_id']
            or fm['sequence_id'] not in allowed or fm['dataset_sha256'] != rm['dataset_sha256']):
        raise ValueError('source/reference role or coordinate cohort differs')
    for source in (frame, reference):
        if (source.metadata['dataset_split'] != 'train'
                or source.metadata['calibration_sha256'] != calibration_sha256
                or not source.available_at(decision_us)):
            raise ValueError('future source or mismatched fold calibration')
    selected = choose_candidates(frame.raw_scores, dict(minimum_raw_score=.05, maximum_per_side=64))
    # Filter only after the all-class top64 selection used in fit training.
    selected = selected[frame.class_indices[selected] == 0]
    arrays = dict(scores=frame.raw_scores, class_indices=frame.class_indices,
                  appearance_128=frame.appearance, appearance_valid=frame.appearance_valid)
    delta = (fm['box_reference_timestamp_us'] - rm['box_reference_timestamp_us']) / 1e6
    features = encode_features(frame.states, frame.covariances, arrays, selected, fm, rm,
                               delta, origin_us, calibration['sides'][fm['side']])
    if features.shape != (len(selected), 203) or not np.isfinite(features).all():
        raise ValueError('canonical V2 encoder produced invalid features')
    return selected, features
