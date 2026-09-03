import numpy as np
import pytest

from transvision.models.event_track_v2x.diagnostics import (
    existence_calibration_v1,
    gaussian_calibration_v1,
    runtime_summary_v1,
)


def test_gaussian_calibration_matches_closed_form_zero_residual() -> None:
    residuals = np.zeros((4, 2))
    covariances = np.repeat(np.eye(2)[None, :, :], 4, axis=0)
    result = gaussian_calibration_v1(residuals, covariances)
    assert result.mean_nll == pytest.approx(np.log(2.0 * np.pi))
    assert result.mean_squared_normalized_error == 0.0
    assert (result.coverage_50, result.coverage_90, result.coverage_95) == (
        1.0,
        1.0,
        1.0,
    )


def test_gaussian_calibration_rejects_non_spd_covariance() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        gaussian_calibration_v1(
            np.zeros((1, 2)), np.asarray([[[1.0, 0.0], [0.0, -1.0]]])
        )


def test_existence_calibration_brier_and_ece() -> None:
    result = existence_calibration_v1(
        np.asarray([0.0, 0.25, 0.75, 1.0]),
        np.asarray([0, 0, 1, 1]),
        bin_count=2,
    )
    assert result.brier_score == pytest.approx(0.03125)
    assert result.expected_calibration_error == pytest.approx(0.125)


def test_runtime_summary_reports_registered_percentiles() -> None:
    result = runtime_summary_v1(
        np.asarray([0.01, 0.02, 0.03, 0.04]),
        peak_cpu_memory_bytes=100,
        peak_gpu_memory_bytes=200,
    )
    assert result.latency_p50_ms == pytest.approx(25.0)
    assert result.frames_per_second == pytest.approx(40.0)
    assert result.latency_p95_ms > result.latency_p50_ms
    assert result.latency_p99_ms >= result.latency_p95_ms
