"""Correlation-safe continuous-state fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arrays import immutable_float64


@dataclass(frozen=True, slots=True)
class CIFusion:
    mean: np.ndarray
    covariance: np.ndarray
    weight_first: float


def _inputs(mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if mean.ndim != 1 or not np.all(np.isfinite(mean)):
        raise ValueError("CI mean must be a finite vector")
    if (
        covariance.shape != (mean.size, mean.size)
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError("CI covariance must be finite, symmetric, and compatible")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("CI covariance must be positive definite") from exc
    return mean, covariance


def covariance_intersection(
    mean_first: np.ndarray,
    covariance_first: np.ndarray,
    mean_second: np.ndarray,
    covariance_second: np.ndarray,
    *,
    weight_first: float | None = None,
    objective: str = "logdet",
) -> CIFusion:
    """Fuse same-target beliefs without assuming a cross-covariance."""

    mean_first, covariance_first = _inputs(mean_first, covariance_first)
    mean_second, covariance_second = _inputs(mean_second, covariance_second)
    if mean_first.shape != mean_second.shape:
        raise ValueError("CI inputs must have the same state dimension")
    if objective not in {"logdet", "trace"}:
        raise ValueError("CI objective must be 'logdet' or 'trace'")
    information_first = np.linalg.inv(covariance_first)
    information_second = np.linalg.inv(covariance_second)

    def at(weight: float) -> tuple[np.ndarray, np.ndarray, float]:
        information = (
            weight * information_first + (1.0 - weight) * information_second
        )
        covariance = np.linalg.inv(information)
        mean = covariance @ (
            weight * information_first @ mean_first
            + (1.0 - weight) * information_second @ mean_second
        )
        if objective == "trace":
            score = float(np.trace(covariance))
        else:
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:  # pragma: no cover - convex SPD combination.
                raise ValueError("CI produced a non-positive covariance")
            score = float(logdet)
        return mean, covariance, score

    if weight_first is None:
        # Deterministic golden-section search plus both valid endpoints.
        lower = 0.0
        upper = 1.0
        ratio = (np.sqrt(5.0) - 1.0) / 2.0
        left = upper - ratio * (upper - lower)
        right = lower + ratio * (upper - lower)
        left_score = at(left)[2]
        right_score = at(right)[2]
        for _ in range(80):
            if left_score <= right_score:
                upper, right, right_score = right, left, left_score
                left = upper - ratio * (upper - lower)
                left_score = at(left)[2]
            else:
                lower, left, left_score = left, right, right_score
                right = lower + ratio * (upper - lower)
                right_score = at(right)[2]
        candidates = (0.0, 1.0, 0.5 * (lower + upper))
        weight_first = min(candidates, key=lambda weight: (at(weight)[2], weight))
    else:
        weight_first = float(weight_first)
        if not np.isfinite(weight_first) or not 0.0 <= weight_first <= 1.0:
            raise ValueError("weight_first must be in [0, 1]")
    mean, covariance, _ = at(weight_first)
    mean = immutable_float64(mean)
    covariance = immutable_float64(0.5 * (covariance + covariance.T))
    return CIFusion(mean=mean, covariance=covariance, weight_first=weight_first)


__all__ = ["CIFusion", "covariance_intersection"]
