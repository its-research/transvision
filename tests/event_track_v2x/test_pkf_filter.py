"""PKF expanded-model equivalence and explicit distinction from JPDA."""
import math

import numpy as np
import pytest
from scipy.linalg import block_diag

from transvision.models.event_track_v2x.hypothesis_bank import LogAssociationFactors
from transvision.models.event_track_v2x.jpda_filter import GaussianBelief, jpda_scan, kalman_condition
from transvision.models.event_track_v2x.pkf_filter import pkf_condition, pkf_scan


@pytest.mark.parametrize('seed', range(8))
def test_information_form_matches_independent_expanded_kalman_update(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 3))
    prior = GaussianBelief(rng.normal(size=3), a @ a.T + np.eye(3))
    h, measurements = rng.normal(size=(2, 3)), rng.normal(size=(4, 2))
    covariances = [np.eye(2) * (j + 1) for j in range(4)]
    weights = rng.uniform(.05, 1., 4)
    weights /= weights.sum() * 1.3
    updated = pkf_condition(prior, measurements, h, covariances, weights)
    expanded = kalman_condition(prior, measurements.reshape(-1), np.tile(h, (4, 1)),
                                block_diag(*(r / w for r, w in zip(covariances, weights))))
    np.testing.assert_allclose(updated.mean, expanded.mean, atol=1e-11)
    np.testing.assert_allclose(updated.covariance, expanded.covariance, atol=1e-11)


def test_pkf_and_jpda_share_association_but_differ_in_m_step():
    prior = GaussianBelief([0.], [[1.]])
    factors = LogAssociationFactors.from_positive([[1., 1.]], [1.], [1., 1.])
    args = ([prior], [[-2.], [2.]], [[1.]], [[[1.]], [[1.]]], factors)
    jpda, pkf = jpda_scan(*args), pkf_scan(*args)
    assert jpda.marginals == pkf.marginals
    assert pkf.posteriors[0].mean[0] == pytest.approx(0.)
    assert pkf.posteriors[0].covariance[0][0] == pytest.approx(3 / 5)
    assert jpda.posteriors[0].covariance[0][0] == pytest.approx(4 / 3)


def test_missing_mass_is_not_renormalised_away():
    prior = GaussianBelief([0.], [[1.]])
    result = pkf_condition(prior, [[2.]], [[1.]], [[[1.]]], [.1])
    assert result.mean[0] == pytest.approx(2 / 11)
    assert result.covariance[0][0] == pytest.approx(10 / 11)


def test_zero_and_tiny_weights_do_not_need_infinite_measurement_noise():
    prior = GaussianBelief([0.], [[1.]])
    result = pkf_condition(prior, [[2.], [3.]], [[1.]], [[[1.]], [[1.]]], [0., 1e-300])
    assert result.mean[0] == pytest.approx(3e-300, rel=1e-12, abs=0.)
    assert result.covariance == prior.covariance
    assert pkf_condition(prior, [], [[1.]], [], []) == prior


def test_angular_innovation_is_local_to_prior_chart():
    prior = GaussianBelief([math.pi - .1], [[1.]])
    result = pkf_condition(prior, [[-math.pi + .1]], [[1.]], [[[1.]]], [1.],
                           angular_state_indices=(0,), angular_measurement_indices=(0,))
    assert abs(result.mean[0]) == pytest.approx(math.pi)


@pytest.mark.parametrize('algorithm', ['exact', 'lbp'])
def test_no_targets_or_no_measurements(algorithm):
    empty = pkf_scan([], [[1.]], [[1.]], [[[1.]]], LogAssociationFactors([], [], [0.]), algorithm=algorithm)
    assert empty.posteriors == ()
    assert empty.marginals.right_unmatched == (1.,)
    prior = GaussianBelief([0.], [[1.]])
    missed = pkf_scan([prior], [], [[1.]], [], LogAssociationFactors([[]], [0.], []), algorithm=algorithm)
    assert missed.posteriors == (prior,)


@pytest.mark.parametrize('weights', [[-.1], [1.1], [float('nan')]])
def test_invalid_association_weights_fail(weights):
    with pytest.raises(ValueError):
        pkf_condition(GaussianBelief([0.], [[1.]]), [[1.]], [[1.]], [[[1.]]], weights)
