"""Classical single-scan JPDA Gaussian update, independent of the scorer.

Conditional updates assume a linear observation model and independent Gaussian
prior/measurement errors. This is NOT covariance intersection. Mixture collapse
keeps between-hypothesis variance but drops cross-track correlations/history.
Raw cooperative detections cannot be treated as independent without an explicit
protocol; this core is not, by itself, a V2 cooperative baseline reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .hypothesis_bank import LogAssociationFactors
from .jpda_marginals import JPDAMarginals, exact_jpda, lbp_jpda


def _matrix(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be finite with shape {shape}')
    return array


def _covariance(value, dimension, name):
    array = _matrix(value, (dimension, dimension), name)
    if not np.allclose(array, array.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f'{name} must be symmetric')
    array = (array + array.T) / 2
    try:
        np.linalg.cholesky(array)
    except np.linalg.LinAlgError as error:
        raise ValueError(f'{name} must be positive definite') from error
    return array


def _angles(indices, dimension):
    indices = tuple(indices)
    if (len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < dimension for i in indices)):
        raise ValueError('distinct valid angular coordinate indices required')
    return indices


def _wrap(values):
    return np.arctan2(np.sin(values), np.cos(values))


@dataclass(frozen=True)
class GaussianBelief:
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=float)
        if mean.ndim != 1 or not len(mean) or not np.all(np.isfinite(mean)):
            raise ValueError('nonempty finite Gaussian mean required')
        covariance = _covariance(self.covariance, len(mean), 'Gaussian covariance')
        object.__setattr__(self, 'mean', tuple(map(float, mean)))
        object.__setattr__(self, 'covariance', tuple(map(tuple, covariance)))


def gaussian_moment_match(components, weights, *, angular_state_indices=()):
    """First two moments including missed branch and between-branch variance.

For angular coordinates use the chart anchored at component zero (the prior in
JPDA), not a claimed globally Gaussian distribution on a circle. Only roundoff
normalisation of weights already summing to one within 1e-8 is allowed.
"""
    components = tuple(components)
    if not components or any(type(c) is not GaussianBelief for c in components):
        raise TypeError('nonempty Gaussian components required')
    dimension = len(components[0].mean)
    if any(len(c.mean) != dimension for c in components):
        raise ValueError('Gaussian mixture dimensions differ')
    mass = _matrix(weights, (len(components),), 'mixture weights')
    if np.any(mass < 0) or abs(math.fsum(mass) - 1.) > 1e-8:
        raise ValueError('mixture weights must be nonnegative and sum to one')
    mass = mass / math.fsum(mass)
    means = np.array([c.mean for c in components])
    angles = _angles(angular_state_indices, dimension)
    for a in angles:
        means[:, a] = means[0, a] + _wrap(means[:, a] - means[0, a])
    mean = mass @ means
    delta = means - mean
    covariance = np.einsum('k,kij->ij', mass, np.array([c.covariance for c in components]))
    covariance += (delta.T * mass) @ delta
    for a in angles:
        mean[a] = _wrap(mean[a])
    return GaussianBelief(mean, (covariance + covariance.T) / 2)


def kalman_condition(prior, measurement, observation_matrix, measurement_covariance, *,
                     angular_measurement_indices=(), angular_state_indices=()):
    """One ordinary Gaussian measurement update, with Joseph-form covariance."""
    if type(prior) is not GaussianBelief:
        raise TypeError('Gaussian prior required')
    z = np.asarray(measurement, dtype=float)
    if z.ndim != 1 or not len(z) or not np.all(np.isfinite(z)):
        raise ValueError('nonempty finite measurement vector required')
    d, m = len(prior.mean), len(z)
    h = _matrix(observation_matrix, (m, d), 'observation matrix')
    r = _covariance(measurement_covariance, m, 'measurement covariance')
    state_angles, measurement_angles = _angles(angular_state_indices, d), _angles(angular_measurement_indices, m)
    p, mean = np.asarray(prior.covariance), np.asarray(prior.mean)
    innovation = z - h @ mean
    for a in measurement_angles:
        innovation[a] = _wrap(innovation[a])
    s = h @ p @ h.T + r
    gain = np.linalg.solve(s, h @ p).T
    updated = mean + gain @ innovation
    residual = np.eye(d) - gain @ h
    covariance = residual @ p @ residual.T + gain @ r @ gain.T
    for a in state_angles:
        updated[a] = _wrap(updated[a])
    return GaussianBelief(updated, (covariance + covariance.T) / 2)


@dataclass(frozen=True)
class JPDAScanUpdate:
    posteriors: tuple[GaussianBelief, ...]
    marginals: JPDAMarginals
    retains_cross_scan_hypotheses: bool = False
    retains_cross_track_covariance: bool = False


def _scan_inputs(priors, measurements, observation_matrix, measurement_covariances, factors,
                 angular_measurement_indices, angular_state_indices):
    priors = tuple(priors)
    measurements, measurement_covariances = tuple(measurements), tuple(measurement_covariances)
    if (type(factors) is not LogAssociationFactors or factors.shape != (len(priors), len(measurements))
            or len(measurement_covariances) != len(measurements)
            or any(type(p) is not GaussianBelief for p in priors)):
        raise ValueError('JPDA priors, measurements, covariances and factor dimensions differ')
    # Validate every measurement even if gating leaves it entirely unmatched.
    h = np.asarray(observation_matrix, dtype=float)
    if h.ndim != 2 or not all(h.shape):
        raise ValueError('nonempty observation matrix required')
    d = len(priors[0].mean) if priors else h.shape[1]
    if any(len(p.mean) != d for p in priors):
        raise ValueError('prior dimensions differ')
    _matrix(h, (h.shape[0], d), 'observation matrix')
    _angles(angular_state_indices, d)
    _angles(angular_measurement_indices, h.shape[0])
    for z, r in zip(measurements, measurement_covariances):
        _matrix(z, (h.shape[0],), 'measurement')
        _covariance(r, h.shape[0], 'measurement covariance')
    return priors, measurements, h, measurement_covariances


def jpda_scan(priors, measurements, observation_matrix, measurement_covariances, factors, *,
              algorithm='exact', limits=None, angular_measurement_indices=(), angular_state_indices=()):
    """Collapse a complete single-scan posterior using caller-supplied factors.

Scorer and Gaussian update are separate: learned matched/unmatched potentials
are accepted unchanged. All candidate marginals enter the mixture (no Top-K).
The number of established objects is fixed for this core; births/deaths,
arrival ordering and V2 output commits belong to the tracker integration.
"""
    if algorithm not in ('exact', 'lbp'):
        raise ValueError('explicit exact or lbp JPDA algorithm required')
    priors, measurements, observation_matrix, measurement_covariances = _scan_inputs(
        priors, measurements, observation_matrix, measurement_covariances, factors,
        angular_measurement_indices, angular_state_indices)
    marginals = (exact_jpda if algorithm == 'exact' else lbp_jpda)(factors, limits=limits)
    updated = []
    for i, prior in enumerate(priors):
        components, weights = [prior], [marginals.left_unmatched[i]]
        for j, (measurement, covariance) in enumerate(zip(measurements, measurement_covariances)):
            if factors.allowed[i][j]:
                components.append(kalman_condition(prior, measurement, observation_matrix, covariance,
                    angular_measurement_indices=angular_measurement_indices, angular_state_indices=angular_state_indices))
                weights.append(marginals.pair[i][j])
        updated.append(gaussian_moment_match(components, weights, angular_state_indices=angular_state_indices))
    return JPDAScanUpdate(tuple(updated), marginals)
