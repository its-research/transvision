"""Analytic JPDA mixture, recursive loss-of-history and Gaussian invariants."""
import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.hypothesis_bank import LogAssociationFactors
from transvision.models.event_track_v2x.jpda_filter import (
    GaussianBelief, gaussian_moment_match, kalman_condition, jpda_scan,
)


def test_jpda_keeps_missed_component_and_between_hypothesis_variance():
    prior = GaussianBelief([0.], [[1.]])
    factors = LogAssociationFactors.from_positive([[1., 1.]], [1.], [1., 1.])
    result = jpda_scan([prior], [[-2.], [2.]], [[1.]], [[[1.]], [[1.]]], factors)
    # Prior: N(0,1), conditionals: N(-1,.5), N(1,.5), all weights 1/3.
    assert result.posteriors[0].mean[0] == pytest.approx(0.)
    assert result.posteriors[0].covariance[0][0] == pytest.approx(4 / 3)
    assert result.marginals.left_unmatched[0] == pytest.approx(1 / 3)
    assert not result.retains_cross_scan_hypotheses
    assert not result.retains_cross_track_covariance


def test_globally_exclusive_associations_not_independent_row_softmax():
    prior = GaussianBelief([0.], [[1.]])
    f = LogAssociationFactors.from_positive([[1.], [1.]], [1., 1.], [1.])
    result = jpda_scan([prior, prior], [[3.]], [[1.]], [[[1.]]], f)
    for posterior in result.posteriors:
        assert posterior.mean[0] == pytest.approx(.5)  # 1/3 * Kalman posterior 1.5.
    assert sum(row[0] for row in result.marginals.pair) == pytest.approx(2 / 3)


def test_empty_measurements_preserve_all_priors():
    priors = [GaussianBelief([2., 1.], [[2., .2], [.2, 1.]])]
    f = LogAssociationFactors([[]], [0.], [])
    result = jpda_scan(priors, [], [[1., 0.]], [], f)
    assert result.posteriors == tuple(priors)


def test_no_priors_still_validates_measurements():
    f = LogAssociationFactors([], [], [0.])
    with pytest.raises(ValueError, match='finite'):
        jpda_scan([], [[float('nan')]], [[1.]], [[[1.]]], f)
    result = jpda_scan([], [[3.]], [[1.]], [[[1.]]], f)
    assert result.posteriors == ()
    assert result.marginals.right_unmatched == (1.,)


def test_kalman_matches_independent_information_form():
    prior = GaussianBelief([1., -1.], [[2., .3], [.3, 1.]])
    h, z, r = np.array([[1., .5], [0., 1.]]), np.array([2., -2.]), np.array([[1., .1], [.1, .5]])
    result = kalman_condition(prior, z, h, r)
    p, m = np.asarray(prior.covariance), np.asarray(prior.mean)
    covariance = np.linalg.inv(np.linalg.inv(p) + h.T @ np.linalg.solve(r, h))
    mean = covariance @ (np.linalg.solve(p, m) + h.T @ np.linalg.solve(r, z))
    np.testing.assert_allclose(result.mean, mean, atol=1e-12)
    np.testing.assert_allclose(result.covariance, covariance, atol=1e-12)


def test_gaussian_belief_copies_inputs_to_immutable_values():
    mean, covariance = np.zeros(2), np.eye(2)
    belief = GaussianBelief(mean, covariance)
    mean[0], covariance[0, 0] = 100., 100.
    assert belief.mean == (0., 0.)
    assert belief.covariance == ((1., 0.), (0., 1.))


def test_angle_chart_avoids_averaging_opposite_pi_sides_to_zero():
    a, b = GaussianBelief([math.pi - .1], [[.01]]), GaussianBelief([-math.pi + .1], [[.01]])
    result = gaussian_moment_match([a, b], [.5, .5], angular_state_indices=(0,))
    assert abs(result.mean[0]) == pytest.approx(math.pi)
    assert result.covariance[0][0] == pytest.approx(.02)
    posterior = kalman_condition(a, [-math.pi + .1], [[1.]], [[.01]],
                                 angular_state_indices=(0,), angular_measurement_indices=(0,))
    assert abs(posterior.mean[0]) == pytest.approx(math.pi)


def test_recursive_moment_collapse_is_not_recoverable_mixture_filtering():
    # A deliberately bimodal posterior is collapsed before discriminating data.
    components = [GaussianBelief([-3.], [[1.]]), GaussianBelief([3.], [[1.]])]
    collapsed = gaussian_moment_match(components, [.5, .5])
    after_collapse = kalman_condition(collapsed, [3.], [[1.]], [[1.]])
    posterior_components = [kalman_condition(c, [3.], [[1.]], [[1.]]) for c in components]
    weights = np.exp(np.array([-9., 0.]))
    weights /= weights.sum()
    preserved = gaussian_moment_match(posterior_components, weights)
    assert abs(after_collapse.mean[0] - preserved.mean[0]) > .2
    assert after_collapse.covariance[0][0] > preserved.covariance[0][0]


@pytest.mark.parametrize('algorithm', ['exact', 'lbp'])
def test_same_factors_are_preserved_through_update(algorithm):
    priors = [GaussianBelief([i], [[1.]]) for i in (0., 2.)]
    factors = LogAssociationFactors.from_positive([[2., 1.], [1., 3.]], [1., 1.], [1., 1.])
    digest = factors.digest()
    result = jpda_scan(priors, [[0.], [2.]], [[1.]], [[[1.]], [[1.]]], factors, algorithm=algorithm)
    assert result.marginals.factor_sha256 == digest == factors.digest()
    assert all(np.linalg.eigvalsh(p.covariance).min() > 0 for p in result.posteriors)


@pytest.mark.parametrize('weights', [[.1, .1], [-1., 2.], [float('nan'), 0.]])
def test_bad_mixture_weights_fail(weights):
    prior = GaussianBelief([0.], [[1.]])
    with pytest.raises(ValueError):
        gaussian_moment_match([prior, prior], weights)


@pytest.mark.parametrize('covariance', [[[0.]], [[-1.]], [[float('nan')]]])
def test_degenerate_or_invalid_covariance_is_rejected(covariance):
    with pytest.raises(ValueError):
        GaussianBelief([0.], covariance)
