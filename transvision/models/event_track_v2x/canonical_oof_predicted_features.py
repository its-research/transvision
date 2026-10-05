"""Prediction-only feature admission for canonical OOF association examples.

This component does not create labels, train a head, or grant paper eligibility.
It reuses the frozen 203-feature encoder and the calibrated V2 state recipe.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

from .prediction_features import (
    CLASSES, RAW_KEYS, choose_candidates, encode_features, raw_state,
)

RAW_METADATA_KEYS = frozenset({
    'sequence_id', 'frame_id', 'side', 'source_image_timestamp_us',
    'box_reference_timestamp_us', 'lidar_to_world_row_rotation',
    'lidar_to_world_translation', 'coordinate_system', 'image_sha256',
})


def load_fold_calibration(path, expected_sha256, *, fold_id):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('calibration must be a regular file')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('calibration bytes differ')
    value = json.loads(data)
    binding = value['canonical_oof_binding']
    fit, held = value['fit_sequences'], binding['held_out_sequence_ids']
    if (value['kind'] != 'eventtrack_train_calibration_v1'
            or type(fold_id) is not int or fold_id not in range(5)
            or binding['fold_id'] != fold_id
            or fit != sorted(set(fit)) or held != sorted(set(held))
            or set(fit) & set(held) or len(fit) + len(held) != 46
            or value['candidate_policy'] != 'raw-score>=0.05/all-class-top64'
            or value['evidence']['official_validation_used_for_selection'] is not False
            or value['evidence']['test_payloads_read'] is not False):
        raise ValueError('canonical fold calibration boundary differs')
    return value


def predicted_features(arrays, metadata, reference_metadata, calibration, *,
                       role, decision_time_us, origin_us):
    """Return raw query indices and features; GT is not an accepted argument.

    ``fit`` admits only the complementary training cohort. ``held_out`` is for
    inference features only, never association or calibration optimization.
    Arrival scheduling remains the caller's responsibility; both timestamps
    must already be available at the supplied decision time.
    """
    if role not in {'fit', 'held_out'}:
        raise ValueError('explicit fit or held_out role required')
    if set(arrays) != RAW_KEYS:
        raise ValueError('raw prediction schema differs')
    for meta in (metadata, reference_metadata):
        if set(meta) != RAW_METADATA_KEYS:
            raise ValueError('public raw metadata schema differs')
        if meta['side'] not in {'vehicle-side', 'infrastructure-side'}:
            raise ValueError('unknown source side')
        for key in ('source_image_timestamp_us', 'box_reference_timestamp_us'):
            if type(meta[key]) is not int or meta[key] > decision_time_us:
                raise ValueError('future or invalid source timestamp')
    if type(decision_time_us) is not int or type(origin_us) is not int:
        raise ValueError('timestamps must be integer microseconds')
    if metadata['sequence_id'] != reference_metadata['sequence_id']:
        raise ValueError('cross-sequence feature pairing forbidden')
    allowed = (calibration['fit_sequences'] if role == 'fit'
               else calibration['canonical_oof_binding']['held_out_sequence_ids'])
    if metadata['sequence_id'] not in allowed:
        raise ValueError('sequence outside explicit fold role')
    state = raw_state(arrays)
    n = len(state)
    shapes = {'scores': (n,), 'class_indices': (n,), 'appearance_128': (n, 128),
              'appearance_valid': (n,), 'image_rois_xyxy': (n, 4)}
    for key, shape in shapes.items():
        a = arrays[key]
        if a.shape != shape or a.dtype.hasobject or not np.isfinite(a).all():
            raise ValueError('invalid prediction array: ' + key)
    classes, scores = arrays['class_indices'], arrays['scores']
    if (classes.dtype.kind not in 'iu' or np.any((classes < 0) | (classes > 2))
            or arrays['appearance_valid'].dtype.kind != 'b'
            or np.any((scores < 0) | (scores > 1))):
        raise ValueError('invalid prediction class, score, or visibility')
    covariance = np.empty((n, 9, 9), dtype=np.float64)
    models = calibration['sides'][metadata['side']]
    for c, name in enumerate(CLASSES):
        matrix = np.asarray(models[name]['covariance']['matrix'], dtype=np.float64)
        if matrix.shape != (9, 9) or not np.isfinite(matrix).all():
            raise ValueError('invalid frozen class covariance')
        np.linalg.cholesky(matrix)
        covariance[classes == c] = matrix
    selected = choose_candidates(scores, {
        'minimum_raw_score': .05, 'maximum_per_side': 64,
    })
    delta = (metadata['box_reference_timestamp_us']
             - reference_metadata['box_reference_timestamp_us']) / 1e6
    features = encode_features(state, covariance, arrays, selected, metadata,
                               reference_metadata, delta, origin_us, models)
    return selected, features
