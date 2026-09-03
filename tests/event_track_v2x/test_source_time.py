import numpy as np
import pytest

from transvision.models.event_track_v2x.source_time import (
    GaussianTimeComponent,
    SourceTimeMixture,
    SourceTimeMode,
    propagate_source_time,
)


def _state() -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(9)
    mean[7:9] = (2.0, -1.0)
    return mean, np.eye(9) * 0.25


def test_source_time_mixture_variance_decomposition() -> None:
    mixture = SourceTimeMixture(
        (
            GaussianTimeComponent(0.25, 1.0, 0.04),
            GaussianTimeComponent(0.75, 3.0, 0.16),
        ),
        causal_upper_bound=3.0,
    )
    assert mixture.mean == pytest.approx(2.5)
    expected = 0.25 * (0.04 + 1.5**2) + 0.75 * (0.16 + 0.5**2)
    assert mixture.variance == pytest.approx(expected, abs=1e-12)


def test_mixture_state_moments_match_component_decomposition() -> None:
    mean, covariance = _state()
    mixture = SourceTimeMixture(
        (
            GaussianTimeComponent(0.5, 1.0, 0.0),
            GaussianTimeComponent(0.5, 3.0, 0.0),
        ),
        causal_upper_bound=3.0,
    )
    result = propagate_source_time(
        mean, covariance, mixture, decision_time=4.0, mode="mixture"
    )
    component_means = np.stack([item.mean for item in result.components])
    expected_mean = np.mean(component_means, axis=0)
    residuals = component_means - expected_mean
    expected_covariance = np.mean(
        np.stack(
            [
                item.covariance + np.outer(residual, residual)
                for item, residual in zip(result.components, residuals)
            ]
        ),
        axis=0,
    )
    np.testing.assert_allclose(result.mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(result.covariance, expected_covariance, atol=1e-12)


def test_zero_time_variance_degenerates_to_point() -> None:
    mean, covariance = _state()
    mixture = SourceTimeMixture(
        (GaussianTimeComponent(1.0, 2.0, 0.0),), causal_upper_bound=2.0
    )
    point = propagate_source_time(
        mean, covariance, mixture, decision_time=3.0, mode=SourceTimeMode.POINT
    )
    first_order = propagate_source_time(
        mean,
        covariance,
        mixture,
        decision_time=3.0,
        mode=SourceTimeMode.FIRST_ORDER,
    )
    mixed = propagate_source_time(
        mean, covariance, mixture, decision_time=3.0, mode=SourceTimeMode.MIXTURE
    )
    np.testing.assert_allclose(point.mean, first_order.mean, atol=1e-12)
    np.testing.assert_allclose(point.mean, mixed.mean, atol=1e-12)
    np.testing.assert_allclose(point.covariance, first_order.covariance, atol=1e-12)
    np.testing.assert_allclose(point.covariance, mixed.covariance, atol=1e-12)


def test_first_order_adds_time_uncertainty_and_oracle_is_explicit() -> None:
    mean, covariance = _state()
    mixture = SourceTimeMixture(
        (GaussianTimeComponent(1.0, 2.0, 0.25),), causal_upper_bound=2.5
    )
    point = propagate_source_time(
        mean, covariance, mixture, decision_time=3.0, mode="point"
    )
    first_order = propagate_source_time(
        mean, covariance, mixture, decision_time=3.0, mode="first_order"
    )
    assert first_order.covariance[0, 0] - point.covariance[0, 0] == pytest.approx(
        1.0625
    )
    assert first_order.covariance[1, 1] - point.covariance[1, 1] == pytest.approx(
        0.3125
    )
    with pytest.raises(ValueError, match="requires oracle_time"):
        propagate_source_time(
            mean, covariance, mixture, decision_time=3.0, mode="oracle"
        )
    oracle = propagate_source_time(
        mean,
        covariance,
        mixture,
        decision_time=3.0,
        mode="oracle",
        oracle_time=2.5,
    )
    assert oracle.components[0].source_time_variance == 0.0


def test_source_time_probability_and_invalid_future_fail_closed() -> None:
    mixture = SourceTimeMixture(
        (
            GaussianTimeComponent(0.5, 1.0, 0.25),
            GaussianTimeComponent(0.5, 2.0, 0.25),
        ),
        causal_upper_bound=2.0,
    )
    assert np.isfinite(mixture.log_probability_density(1.5))
    assert mixture.log_interval_probability(1.5, half_width=0.0005) == pytest.approx(
        SourceTimeMixture(
            (
                GaussianTimeComponent(0.5, 1000.0, 250000.0),
                GaussianTimeComponent(0.5, 2000.0, 250000.0),
            ),
            causal_upper_bound=2000.0,
        ).log_interval_probability(1500.0, half_width=0.5),
        abs=1e-10,
    )
    mean, covariance = _state()
    with pytest.raises(ValueError, match="later"):
        propagate_source_time(
            mean, covariance, mixture, decision_time=0.0, mode="mixture"
        )


def test_source_time_atom_requires_interval_score_and_outputs_are_immutable() -> None:
    atom = SourceTimeMixture(
        (GaussianTimeComponent(1.0, 2.0, 0.0),), causal_upper_bound=2.0
    )
    with pytest.raises(ValueError, match="undefined"):
        atom.log_probability_density(2.0)
    assert atom.log_interval_probability(2.0, half_width=0.0005) == 0.0
    assert atom.log_interval_probability(3.0, half_width=0.0005) == -np.inf

    mean = np.zeros(9)
    covariance = np.eye(9)
    result = propagate_source_time(
        mean, covariance, atom, decision_time=3.0, mode="point"
    )
    with pytest.raises(ValueError, match="WRITEABLE"):
        result.mean.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        result.components[0].covariance.setflags(write=True)
