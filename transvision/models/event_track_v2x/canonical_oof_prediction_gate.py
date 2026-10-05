"""Prediction-only common-time geometry gate for canonical association data.

Uses the existing V2 physical-axis inversion, public world poses, constant
velocity propagation and conservative 2(P+R) bound. No GT inputs are accepted.
"""
import numpy as np
from scipy.stats import chi2

from .canonical_oof_predicted_features import RAW_METADATA_KEYS
from .prediction_features import CLASSES, raw_state, transform_state_covariance, wrap_angle


def world_predictions(arrays, metadata, selected, calibration, *, decision_time_us):
    if set(metadata) != RAW_METADATA_KEYS or type(decision_time_us) is not int:
        raise ValueError('public metadata and integer decision time required')
    if any(type(metadata[k]) is not int or metadata[k] > decision_time_us
           for k in ('source_image_timestamp_us', 'box_reference_timestamp_us')):
        raise ValueError('source has not arrived at decision time')
    selected = np.asarray(selected)
    if selected.ndim != 1 or selected.dtype.kind not in 'iu' or len(set(selected.tolist())) != len(selected):
        raise ValueError('unique selected prediction indices required')
    full_state = raw_state(arrays)
    if np.any(selected < 0) or np.any(selected >= len(full_state)):
        raise ValueError('prediction index out of range')
    state = full_state[selected].copy()
    classes = arrays['class_indices'][selected]
    cov = np.asarray([calibration['sides'][metadata['side']][CLASSES[int(c)]]['covariance']['matrix']
                      for c in classes], dtype=float).reshape(-1, 9, 9)
    if not np.isfinite(cov).all():
        raise ValueError('invalid calibrated covariance')
    np.linalg.cholesky(cov)
    state[:, [3, 4]] = state[:, [4, 3]]
    state[:, 6] = wrap_angle(-state[:, 6] - np.pi / 2)
    j = np.eye(9); j[[3, 4]] = j[[4, 3]]; j[6, 6] = -1
    cov = j @ cov @ j.T
    world = dict(lidar_to_world_row_rotation=np.eye(3), lidar_to_world_translation=np.zeros(3))
    state, cov, _, _ = transform_state_covariance(state, cov, metadata, world)
    delta = (decision_time_us - metadata['box_reference_timestamp_us']) / 1e6
    f = np.eye(9); f[0, 7] = delta; f[1, 8] = delta
    state = state @ f.T
    cov = f @ cov @ f.T + np.eye(9) * .1 * abs(delta)
    return state, cov, classes


def geometry_gate(left, right, *, probability=.99):
    if probability not in (.90, .95, .99):
        raise ValueError('gate confidence outside frozen grid')
    ls, lp, lc = left; rs, rp, rc = right
    innovation = ls[:, None, :3] - rs[None, :, :3]
    uncertainty = 2 * (lp[:, None, :3, :3] + rp[None, :, :3, :3])
    solved = np.linalg.solve(uncertainty, innovation[..., None])[..., 0]
    d2 = np.einsum('ijk,ijk->ij', innovation, solved)
    if not np.isfinite(d2).all() or np.any(d2 < -1e-9):
        raise ValueError('invalid innovation distance')
    gate = (lc[:, None] == rc[None, :]) & (d2 <= chi2.ppf(probability, 3))
    return gate, d2
