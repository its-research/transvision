"""Calibration and runtime diagnostics used by the preregistered tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .association import chi_square_quantile


def _matrix_stack(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 3 or result.shape[1] != result.shape[2] or result.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty stack of square matrices")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    for matrix in result:
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name} matrices must be symmetric")
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{name} matrices must be positive definite") from exc
    return result


@dataclass(frozen=True, slots=True)
class GaussianCalibrationV1:
    mean_nll: float
    mean_squared_normalized_error: float
    coverage_50: float
    coverage_90: float
    coverage_95: float
    sample_count: int


def gaussian_calibration_v1(
    residuals: np.ndarray,
    covariances: np.ndarray,
) -> GaussianCalibrationV1:
    """Compute NLL and NEES/NIS-style diagnostics for Gaussian residuals."""

    residuals = np.asarray(residuals, dtype=np.float64)
    covariances = _matrix_stack(covariances, "covariances")
    if residuals.ndim != 2 or residuals.shape != covariances.shape[:2]:
        raise ValueError("residuals must have one vector per covariance")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals must be finite")
    dimension = residuals.shape[1]
    squared: list[float] = []
    nll: list[float] = []
    for residual, covariance in zip(residuals, covariances):
        solved = np.linalg.solve(covariance, residual)
        distance = float(residual @ solved)
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0.0:  # pragma: no cover - Cholesky already proves SPD.
            raise ValueError("covariance determinant must be positive")
        squared.append(distance)
        nll.append(0.5 * (dimension * np.log(2.0 * np.pi) + logdet + distance))
    squared_array = np.asarray(squared)

    def coverage(probability: float) -> float:
        threshold = chi_square_quantile(dimension, probability)
        return float(np.mean(squared_array <= threshold))

    return GaussianCalibrationV1(
        mean_nll=float(np.mean(nll)),
        mean_squared_normalized_error=float(np.mean(squared_array)),
        coverage_50=coverage(0.50),
        coverage_90=coverage(0.90),
        coverage_95=coverage(0.95),
        sample_count=len(residuals),
    )


@dataclass(frozen=True, slots=True)
class ExistenceCalibrationV1:
    brier_score: float
    expected_calibration_error: float
    sample_count: int
    bin_count: int


def existence_calibration_v1(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    bin_count: int = 10,
) -> ExistenceCalibrationV1:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    outcomes = np.asarray(outcomes)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be a non-empty vector")
    if outcomes.shape != probabilities.shape:
        raise ValueError("outcomes must have the same shape as probabilities")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.all(np.isin(outcomes, (0, 1, False, True))):
        raise ValueError("outcomes must be binary")
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count <= 0:
        raise ValueError("bin_count must be a positive integer")
    binary = outcomes.astype(np.float64)
    brier = float(np.mean((probabilities - binary) ** 2))
    # Right-closed only at p=1.0; all other boundaries go to the higher bin.
    indices = np.minimum((probabilities * bin_count).astype(int), bin_count - 1)
    ece = 0.0
    for index in range(bin_count):
        selected = indices == index
        if not np.any(selected):
            continue
        ece += float(np.mean(selected)) * abs(
            float(np.mean(probabilities[selected])) - float(np.mean(binary[selected]))
        )
    return ExistenceCalibrationV1(
        brier_score=brier,
        expected_calibration_error=ece,
        sample_count=probabilities.size,
        bin_count=bin_count,
    )


@dataclass(frozen=True, slots=True)
class RuntimeSummaryV1:
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    frames_per_second: float
    peak_cpu_memory_bytes: int
    peak_gpu_memory_bytes: int
    sample_count: int


def runtime_summary_v1(
    latencies_seconds: np.ndarray,
    *,
    peak_cpu_memory_bytes: int,
    peak_gpu_memory_bytes: int,
) -> RuntimeSummaryV1:
    values = np.asarray(latencies_seconds, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("latencies_seconds must be a non-empty finite vector")
    if np.any(values <= 0.0):
        raise ValueError("latencies_seconds must be positive")
    for name, value in (
        ("peak_cpu_memory_bytes", peak_cpu_memory_bytes),
        ("peak_gpu_memory_bytes", peak_gpu_memory_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    p50, p95, p99 = np.percentile(values * 1000.0, (50, 95, 99))
    return RuntimeSummaryV1(
        latency_p50_ms=float(p50),
        latency_p95_ms=float(p95),
        latency_p99_ms=float(p99),
        frames_per_second=float(1.0 / np.mean(values)),
        peak_cpu_memory_bytes=peak_cpu_memory_bytes,
        peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        sample_count=values.size,
    )


__all__ = [
    "ExistenceCalibrationV1",
    "GaussianCalibrationV1",
    "RuntimeSummaryV1",
    "existence_calibration_v1",
    "gaussian_calibration_v1",
    "runtime_summary_v1",
]
