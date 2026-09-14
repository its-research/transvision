"""Decoupled PKF weighted Gaussian M-step (Cao et al., arXiv:2411.06378v2).

The information-form implementation is algebraically equivalent to the
expanded measurement R_j / beta_ij Kalman update, without dividing by tiny
weights. It is NOT JPDA moment matching or a full public tracking reproduction.
Association can use the same exact/LBP positive factors as jpda_filter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .jpda_filter import GaussianBelief, _angles, _covariance, _matrix, _scan_inputs, _wrap
from .jpda_marginals import JPDAMarginals, exact_jpda, lbp_jpda


def pkf_condition(prior, measurements, observation_matrix, measurement_covariances, weights, *,
                  angular_measurement_indices=(), angular_state_indices=()):
    """Maximise weighted log likelihood plus one Gaussian prior.

Weights are unconditional association probabilities. Do NOT renormalise them
over matched detections: their sum may be below one due to missed detections.
Zero weights give no information; no positive-weight cutoff is introduced.
"""
    if type(prior) is not GaussianBelief:
        raise TypeError('Gaussian prior required')
    measurements, measurement_covariances = tuple(measurements), tuple(measurement_covariances)
    if len(measurements) != len(measurement_covariances):
        raise ValueError('measurement covariance count differs')
    weight = _matrix(weights, (len(measurements),), 'association weights')
    if np.any(weight < 0) or weight.sum() > 1. + 1e-8:
        raise ValueError('unconditional association probabilities must sum to at most one')
    h = np.asarray(observation_matrix, dtype=float)
    if h.ndim != 2 or not all(h.shape):
        raise ValueError('nonempty observation matrix required')
    d, m = len(prior.mean), h.shape[0]
    _matrix(h, (m, d), 'observation matrix')
    state_angles, measurement_angles = _angles(angular_state_indices, d), _angles(angular_measurement_indices, m)
    information = np.linalg.solve(np.asarray(prior.covariance), np.eye(d))
    correction = np.zeros(d)
    for z, r, w in zip(measurements, measurement_covariances, weight):
        z, r = _matrix(z, (m,), 'measurement'), _covariance(r, m, 'measurement covariance')
        innovation = z - h @ np.asarray(prior.mean)
        for a in measurement_angles:
            innovation[a] = _wrap(innovation[a])
        information += w * h.T @ np.linalg.solve(r, h)
        correction += w * h.T @ np.linalg.solve(r, innovation)
    covariance = np.linalg.solve(information, np.eye(d))
    mean = np.asarray(prior.mean) + np.linalg.solve(information, correction)
    for a in state_angles:
        mean[a] = _wrap(mean[a])
    return GaussianBelief(mean, (covariance + covariance.T) / 2)


@dataclass(frozen=True)
class PKFScanUpdate:
    posteriors: tuple[GaussianBelief, ...]
    marginals: JPDAMarginals
    update_algorithm: str = 'pkf-decoupled-information-v1'
    retains_cross_scan_hypotheses: bool = False
    retains_cross_track_covariance: bool = False


def pkf_scan(priors, measurements, observation_matrix, measurement_covariances, factors, *,
             algorithm='exact', limits=None, angular_measurement_indices=(), angular_state_indices=()):
    """One marginal association step followed by the decoupled PKF M-step.

This does not claim EM iterations until convergence, a new E-step, correlated
cooperative fusion, or reproduction of the paper's detector/lifecycle system.
"""
    if algorithm not in ('exact', 'lbp'):
        raise ValueError('explicit exact or lbp PKF association algorithm required')
    priors, measurements, h, measurement_covariances = _scan_inputs(
        priors, measurements, observation_matrix, measurement_covariances, factors,
        angular_measurement_indices, angular_state_indices)
    marginals = (exact_jpda if algorithm == 'exact' else lbp_jpda)(factors, limits=limits)
    posteriors = tuple(pkf_condition(prior, measurements, h, measurement_covariances, marginals.pair[i],
        angular_measurement_indices=angular_measurement_indices, angular_state_indices=angular_state_indices)
        for i, prior in enumerate(priors))
    return PKFScanUpdate(posteriors, marginals)
