#!/usr/bin/env python3
"""Local conformance evaluator for heteroscedastic 2D trajectory forecasts.

Version 2 accepts the final per-mode, per-step covariance emitted by the
forecast model after any reliability-dependent inflation.  It validates and
scores that covariance; it does not infer, calibrate, or modify it.  The
implementation is intentionally dependency-free and resource-bounded so that
small hand-computable vectors can detect integration regressions.

This is not an official V2X-Seq evaluator and cannot support scientific claims
until an independently frozen evaluator has passed the protocol parity gate.
The fixed-isotropic v1 evaluator remains a separate diagnostic contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


CONTRACT_ID = "RTPV2X-EVALUATOR-CONFORMANCE-v2"
MAX_SAMPLES = 4096
MAX_MODES = 12
MAX_HORIZON_STEPS = 100
MAX_ECE_BINS = 1000
MAX_IDENTIFIER_LENGTH = 255
MAX_PROTOCOL_BYTES = 1_048_576
MAX_ABS_COORDINATE_M = 1_000_000.0
MAX_MODE_WEIGHT = 1.0e100
MIN_COVARIANCE_EIGENVALUE_M2 = 1.0e-12
MAX_COVARIANCE_EIGENVALUE_M2 = 1.0e12
MAX_COVARIANCE_CONDITION_NUMBER = 1.0e8
SYMMETRY_ABSOLUTE_TOLERANCE_M2 = 0.0
CONFORMANCE_ABSOLUTE_TOLERANCE = 1.0e-12
_LOG_TWO_PI = math.log(2.0 * math.pi)

PROTOCOL_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "status",
        "scientific_claim_allowed",
        "official_evaluator_equivalence",
        "implementation",
        "scope",
        "forecasting",
        "numeric_policy",
        "resource_limits",
        "test_vectors",
        "parity_gate",
    }
)
FORECAST_CASE_NAMES = frozenset(
    {
        "isotropic_parity_with_v1",
        "anisotropic_single_step",
        "trajectory_level_mixture",
    }
)
FORECAST_EXPECTED_FIELDS = frozenset(
    {
        "min_ade_m",
        "min_fde_m",
        "miss_rate",
        "mixture_nll",
        "top_mode_brier",
        "top_mode_ece",
        "top_mode_aurc",
    }
)
RESOURCE_LIMITS = {
    "maximum_samples": MAX_SAMPLES,
    "maximum_modes_per_sample": MAX_MODES,
    "maximum_horizon_steps": MAX_HORIZON_STEPS,
    "maximum_ece_bins": MAX_ECE_BINS,
    "maximum_identifier_length": MAX_IDENTIFIER_LENGTH,
    "maximum_protocol_bytes": MAX_PROTOCOL_BYTES,
}
NUMERIC_LIMITS = {
    "maximum_absolute_coordinate_m": MAX_ABS_COORDINATE_M,
    "maximum_mode_weight": MAX_MODE_WEIGHT,
    "minimum_covariance_eigenvalue_m2": MIN_COVARIANCE_EIGENVALUE_M2,
    "maximum_covariance_eigenvalue_m2": MAX_COVARIANCE_EIGENVALUE_M2,
    "maximum_covariance_condition_number": MAX_COVARIANCE_CONDITION_NUMBER,
    "covariance_symmetry_absolute_tolerance_m2": SYMMETRY_ABSOLUTE_TOLERANCE_M2,
    "conformance_absolute_tolerance": CONFORMANCE_ABSOLUTE_TOLERANCE,
}
EXPECTED_SCOPE = {
    "purpose": (
        "freeze local heteroscedastic trajectory-forecast metric semantics on "
        "small hand-computable cases"
    ),
    "input_boundary": (
        "each mode supplies a 2D mean and final 2x2 covariance for every future step"
    ),
    "covariance_stage": "after model-side reliability inflation",
    "evaluator_does_not": [
        "compute reliability inflation",
        "interpret covariance as top-mode confidence",
        "read labels, datasets, network resources, or ClearML state",
        "replace the independent or official benchmark evaluator",
    ],
    "v1_boundary": (
        "RTPV2X-EVALUATOR-CONFORMANCE-v1 remains unchanged as a "
        "fixed-isotropic diagnostic"
    ),
}
EXPECTED_FORECASTING = {
    "space": "2D Cartesian positions in metres",
    "mode_probability": {
        "input": "finite non-negative weights, not logits",
        "normalization": "p_k = w_k / sum_j(w_j) using a finite positive sum",
        "positive_underflow": "reject if a positive input weight normalizes to zero",
        "top_mode_tie_break": "smallest mode index",
    },
    "covariance": {
        "input": (
            "final per-mode, per-step 2x2 covariance after model-side "
            "reliability inflation"
        ),
        "validation": (
            "finite, exactly symmetric, positive definite, and within the "
            "eigenvalue and condition-number bounds"
        ),
        "factorization": "2x2 Cholesky",
        "evaluator_transformation": "none",
    },
    "metrics": {
        "minADE": (
            "mean over samples of min_k mean_t ||prediction[k,t] - truth[t]||_2"
        ),
        "minFDE": (
            "mean over samples of min_k ||prediction[k,T] - truth[T]||_2; "
            "minimized independently from minADE"
        ),
        "MR": (
            "mean over samples of 1[min_k FDE_k > miss_threshold_m]; equality is a hit"
        ),
        "NLL": (
            "for each mode sum normalized 2D Gaussian log densities over all steps, "
            "add log p_k, combine modes with logsumexp, negate, then average across samples"
        ),
        "Brier": (
            "mean (top_mode_confidence - top_mode_hit)^2, with top_mode_hit = "
            "1[FDE_top <= miss_threshold_m]"
        ),
        "ECE": (
            "v1 top-mode equal-width bin semantics; covariance is not a confidence input"
        ),
        "AURC": (
            "v1 descending top-mode-confidence prefix-risk semantics with sample_id tie-break"
        ),
    },
}
EXPECTED_NUMERIC_POLICY = {
    "non_finite_input": "rejected",
    "boolean_as_number": "rejected",
    "miss_threshold": "finite and non-negative",
    **NUMERIC_LIMITS,
}
EXPECTED_PARITY_GATE = {
    "required_before_benchmark_use": True,
    "requirements": [
        "freeze an independent heteroscedastic evaluator and environment",
        "run both implementations on identical hashed prediction sidecars",
        "compare per-sample and aggregate metrics under a predeclared tolerance",
        "archive raw outputs, configuration, checkpoint, and environment hashes",
        (
            "keep the independent or official evaluator authoritative for thesis "
            "benchmark values"
        ),
    ],
}


class EvaluationInputError(ValueError):
    """Raised when an input makes the reference result unsafe or undefined."""


@dataclass(frozen=True)
class HeteroscedasticForecastMode:
    """One trajectory mean and its final 2D covariance at every future step."""

    means: tuple[tuple[float, float], ...]
    covariances: tuple[tuple[tuple[float, float], tuple[float, float]], ...]


@dataclass(frozen=True)
class HeteroscedasticForecastSample:
    """One realized trajectory and a finite Gaussian-mixture forecast."""

    sample_id: str
    truth: tuple[tuple[float, float], ...]
    modes: tuple[HeteroscedasticForecastMode, ...]
    mode_weights: tuple[float, ...]


@dataclass(frozen=True)
class ForecastMetrics:
    sample_count: int
    min_ade_m: float
    min_fde_m: float
    miss_rate: float
    mixture_nll: float
    top_mode_brier: float
    top_mode_ece: float
    top_mode_aurc: float


@dataclass(frozen=True)
class _CovarianceFactor:
    lower_00: float
    lower_10: float
    lower_11: float
    log_determinant: float


@dataclass(frozen=True)
class _ForecastCaseScore:
    sample_id: str
    min_ade_m: float
    min_fde_m: float
    missed: int
    mixture_nll: float
    top_mode_confidence: float
    top_mode_hit: int


def _require_identifier(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > MAX_IDENTIFIER_LENGTH
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise EvaluationInputError(
            f"{label} must be a non-empty identifier of at most "
            f"{MAX_IDENTIFIER_LENGTH} characters"
        )
    return value


def _require_finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise EvaluationInputError(f"{label} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationInputError(f"{label} must be a finite number") from exc
    if not math.isfinite(converted):
        raise EvaluationInputError(f"{label} must be a finite number")
    return converted


def _require_finite_bounded(
    value: object, *, label: str, absolute_bound: float
) -> float:
    converted = _require_finite(value, label)
    if abs(converted) > absolute_bound:
        raise EvaluationInputError(
            f"{label} exceeds the absolute bound {absolute_bound!r}"
        )
    return converted


def normalize_mode_probabilities(weights: Sequence[float]) -> tuple[float, ...]:
    """Normalize finite non-negative weights using the v1 L1 semantics."""

    if not isinstance(weights, (tuple, list)) or not weights:
        raise EvaluationInputError("mode_weights must be a non-empty sequence")
    if len(weights) > MAX_MODES:
        raise EvaluationInputError(f"mode count exceeds the limit {MAX_MODES}")
    validated: list[float] = []
    for index, value in enumerate(weights):
        converted = _require_finite(value, f"mode_weights[{index}]")
        if converted < 0.0:
            raise EvaluationInputError("mode_weights must be non-negative")
        if converted > MAX_MODE_WEIGHT:
            raise EvaluationInputError(
                f"mode_weights[{index}] exceeds the bound {MAX_MODE_WEIGHT!r}"
            )
        validated.append(converted)
    try:
        total = math.fsum(validated)
    except OverflowError as exc:
        raise EvaluationInputError("mode_weights sum overflowed") from exc
    if not math.isfinite(total) or total <= 0.0:
        raise EvaluationInputError("mode_weights must have a positive finite sum")
    normalized = tuple(value / total for value in validated)
    if any(
        value > 0.0 and probability == 0.0
        for value, probability in zip(validated, normalized)
    ):
        raise EvaluationInputError(
            "a positive mode weight underflowed during probability normalization"
        )
    if not math.isclose(math.fsum(normalized), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise EvaluationInputError("normalized mode probabilities do not sum to one")
    return normalized


def _validate_position(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise EvaluationInputError(f"{label} must contain exactly two coordinates")
    return (
        _require_finite_bounded(
            value[0], label=f"{label}[0]", absolute_bound=MAX_ABS_COORDINATE_M
        ),
        _require_finite_bounded(
            value[1], label=f"{label}[1]", absolute_bound=MAX_ABS_COORDINATE_M
        ),
    )


def _factor_covariance(value: object, label: str) -> _CovarianceFactor:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise EvaluationInputError(f"{label} must be a 2x2 covariance matrix")
    if any(not isinstance(row, (tuple, list)) or len(row) != 2 for row in value):
        raise EvaluationInputError(f"{label} must be a 2x2 covariance matrix")

    entries = tuple(
        _require_finite(entry, f"{label}[{row_index}][{column_index}]")
        for row_index, row in enumerate(value)
        for column_index, entry in enumerate(row)
    )
    covariance_00, covariance_01, covariance_10, covariance_11 = entries
    if covariance_01 != covariance_10:
        raise EvaluationInputError(
            f"{label} must be exactly symmetric; off-diagonal entries differ"
        )
    if covariance_00 <= 0.0 or covariance_11 <= 0.0:
        raise EvaluationInputError(f"{label} must be symmetric positive definite")

    try:
        lower_00 = math.sqrt(covariance_00)
        lower_10 = covariance_10 / lower_00
        residual = covariance_11 - lower_10 * lower_10
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise EvaluationInputError(f"{label} covariance factorization failed") from exc
    if not math.isfinite(residual) or residual <= 0.0:
        raise EvaluationInputError(f"{label} must be symmetric positive definite")
    lower_11 = math.sqrt(residual)

    try:
        discriminant = math.hypot(covariance_00 - covariance_11, 2.0 * covariance_01)
        maximum_eigenvalue = 0.5 * (covariance_00 + covariance_11 + discriminant)
        determinant = covariance_00 * residual
        minimum_eigenvalue = determinant / maximum_eigenvalue
        log_determinant = 2.0 * (math.log(lower_00) + math.log(lower_11))
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise EvaluationInputError(f"{label} covariance eigensystem failed") from exc
    if any(
        not math.isfinite(item)
        for item in (minimum_eigenvalue, maximum_eigenvalue, log_determinant)
    ):
        raise EvaluationInputError(f"{label} covariance calculation is non-finite")
    if minimum_eigenvalue < MIN_COVARIANCE_EIGENVALUE_M2:
        raise EvaluationInputError(
            f"{label} minimum eigenvalue is below {MIN_COVARIANCE_EIGENVALUE_M2!r} m^2"
        )
    if maximum_eigenvalue > MAX_COVARIANCE_EIGENVALUE_M2:
        raise EvaluationInputError(
            f"{label} maximum eigenvalue exceeds {MAX_COVARIANCE_EIGENVALUE_M2!r} m^2"
        )
    condition_number = maximum_eigenvalue / minimum_eigenvalue
    if (
        not math.isfinite(condition_number)
        or condition_number > MAX_COVARIANCE_CONDITION_NUMBER
    ):
        raise EvaluationInputError(
            f"{label} condition number exceeds {MAX_COVARIANCE_CONDITION_NUMBER!r}"
        )
    return _CovarianceFactor(lower_00, lower_10, lower_11, log_determinant)


def _validate_sample(
    sample: HeteroscedasticForecastSample,
) -> tuple[
    int,
    tuple[float, ...],
    tuple[tuple[_CovarianceFactor, ...], ...],
]:
    if not isinstance(sample, HeteroscedasticForecastSample):
        raise EvaluationInputError(
            "each forecast sample must be HeteroscedasticForecastSample"
        )
    sample_id = _require_identifier(sample.sample_id, "sample_id")
    if not isinstance(sample.truth, (tuple, list)) or not sample.truth:
        raise EvaluationInputError(
            f"sample {sample_id!r} has an empty truth trajectory"
        )
    if len(sample.truth) > MAX_HORIZON_STEPS:
        raise EvaluationInputError(
            f"sample {sample_id!r} horizon exceeds {MAX_HORIZON_STEPS} steps"
        )
    horizon = len(sample.truth)
    for time_index, position in enumerate(sample.truth):
        _validate_position(position, f"sample {sample_id!r} truth[{time_index}]")

    if not isinstance(sample.modes, (tuple, list)) or not sample.modes:
        raise EvaluationInputError(f"sample {sample_id!r} has no forecast modes")
    if len(sample.modes) > MAX_MODES:
        raise EvaluationInputError(
            f"sample {sample_id!r} mode count exceeds {MAX_MODES}"
        )
    factors_by_mode: list[tuple[_CovarianceFactor, ...]] = []
    for mode_index, mode in enumerate(sample.modes):
        if not isinstance(mode, HeteroscedasticForecastMode):
            raise EvaluationInputError(
                f"sample {sample_id!r} mode {mode_index} must be "
                "HeteroscedasticForecastMode"
            )
        if not isinstance(mode.means, (tuple, list)) or len(mode.means) != horizon:
            raise EvaluationInputError(
                f"sample {sample_id!r} mode {mode_index} means have inconsistent horizon"
            )
        if (
            not isinstance(mode.covariances, (tuple, list))
            or len(mode.covariances) != horizon
        ):
            raise EvaluationInputError(
                f"sample {sample_id!r} mode {mode_index} covariances have "
                "inconsistent horizon"
            )
        for time_index, position in enumerate(mode.means):
            _validate_position(
                position,
                f"sample {sample_id!r} mode[{mode_index}].means[{time_index}]",
            )
        factors_by_mode.append(
            tuple(
                _factor_covariance(
                    covariance,
                    f"sample {sample_id!r} mode[{mode_index}]"
                    f".covariances[{time_index}]",
                )
                for time_index, covariance in enumerate(mode.covariances)
            )
        )

    if not isinstance(sample.mode_weights, (tuple, list)):
        raise EvaluationInputError(
            f"sample {sample_id!r} mode_weights must be a sequence"
        )
    if len(sample.mode_weights) != len(sample.modes):
        raise EvaluationInputError(
            f"sample {sample_id!r} mode_weights length does not match mode count"
        )
    probabilities = normalize_mode_probabilities(sample.mode_weights)
    return horizon, probabilities, tuple(factors_by_mode)


def _logsumexp(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise EvaluationInputError("log-sum-exp requires at least one finite component")
    maximum = max(values)
    try:
        result = maximum + math.log(
            math.fsum(math.exp(value - maximum) for value in values)
        )
    except (OverflowError, ValueError) as exc:
        raise EvaluationInputError("log-sum-exp calculation failed") from exc
    if not math.isfinite(result):
        raise EvaluationInputError("log-sum-exp produced a non-finite result")
    return result


def _score_forecast_sample(
    sample: HeteroscedasticForecastSample, *, miss_threshold_m: float
) -> _ForecastCaseScore:
    horizon, probabilities, factors_by_mode = _validate_sample(sample)
    ade_by_mode: list[float] = []
    fde_by_mode: list[float] = []
    log_components: list[float] = []

    try:
        for mode_index, (mode, factors) in enumerate(
            zip(sample.modes, factors_by_mode)
        ):
            distances: list[float] = []
            step_log_densities: list[float] = []
            for mean, truth, factor in zip(mode.means, sample.truth, factors):
                delta_0 = float(truth[0]) - float(mean[0])
                delta_1 = float(truth[1]) - float(mean[1])
                distances.append(math.hypot(delta_0, delta_1))

                whitened_0 = delta_0 / factor.lower_00
                whitened_1 = (delta_1 - factor.lower_10 * whitened_0) / factor.lower_11
                quadratic = whitened_0 * whitened_0 + whitened_1 * whitened_1
                if not math.isfinite(quadratic):
                    raise EvaluationInputError("forecast NLL quadratic is non-finite")
                step_log_densities.append(
                    -_LOG_TWO_PI - 0.5 * factor.log_determinant - 0.5 * quadratic
                )

            ade_by_mode.append(math.fsum(distances) / horizon)
            fde_by_mode.append(distances[-1])
            trajectory_log_density = math.fsum(step_log_densities)
            if not math.isfinite(trajectory_log_density):
                raise EvaluationInputError("trajectory log-density is non-finite")
            probability = probabilities[mode_index]
            if probability > 0.0:
                log_components.append(math.log(probability) + trajectory_log_density)
    except EvaluationInputError:
        raise
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise EvaluationInputError("forecast calculation overflowed") from exc

    if any(not math.isfinite(value) for value in (*ade_by_mode, *fde_by_mode)):
        raise EvaluationInputError("forecast distance calculation is non-finite")
    mixture_nll = -_logsumexp(log_components)
    top_mode_index = min(
        range(len(probabilities)), key=lambda index: (-probabilities[index], index)
    )
    top_mode_confidence = probabilities[top_mode_index]
    top_mode_hit = int(fde_by_mode[top_mode_index] <= miss_threshold_m)
    return _ForecastCaseScore(
        sample_id=sample.sample_id,
        min_ade_m=min(ade_by_mode),
        min_fde_m=min(fde_by_mode),
        missed=int(min(fde_by_mode) > miss_threshold_m),
        mixture_nll=mixture_nll,
        top_mode_confidence=top_mode_confidence,
        top_mode_hit=top_mode_hit,
    )


def evaluate_forecasts(
    samples: Iterable[HeteroscedasticForecastSample],
    *,
    miss_threshold_m: float,
    ece_bins: int,
) -> ForecastMetrics:
    """Evaluate heteroscedastic trajectory NLL and unchanged v1 point metrics.

    For each mode, the evaluator sums the normalized 2D Gaussian log density
    over the complete horizon.  Mode log probabilities are then combined with
    ``logsumexp``.  Covariances are final model outputs and are never interpreted
    as top-mode confidence or modified using a reliability signal.
    """

    threshold = _require_finite_bounded(
        miss_threshold_m,
        label="miss_threshold_m",
        absolute_bound=MAX_ABS_COORDINATE_M,
    )
    if threshold < 0.0:
        raise EvaluationInputError("miss_threshold_m must be non-negative")
    if (
        isinstance(ece_bins, bool)
        or not isinstance(ece_bins, int)
        or ece_bins <= 0
        or ece_bins > MAX_ECE_BINS
    ):
        raise EvaluationInputError(
            f"ece_bins must be an integer in [1, {MAX_ECE_BINS}]"
        )

    rows: list[HeteroscedasticForecastSample] = []
    try:
        for row_index, sample in enumerate(samples):
            if row_index >= MAX_SAMPLES:
                raise EvaluationInputError(
                    f"sample count exceeds the limit {MAX_SAMPLES}"
                )
            rows.append(sample)
    except TypeError as exc:
        raise EvaluationInputError("samples must be an iterable of forecasts") from exc
    if not rows:
        raise EvaluationInputError("samples must contain at least one forecast")

    sample_ids: set[str] = set()
    scores: list[_ForecastCaseScore] = []
    for sample in rows:
        score = _score_forecast_sample(sample, miss_threshold_m=threshold)
        if score.sample_id in sample_ids:
            raise EvaluationInputError(f"duplicate sample_id {score.sample_id!r}")
        sample_ids.add(score.sample_id)
        scores.append(score)

    bin_counts = [0 for _ in range(ece_bins)]
    bin_confidence = [0.0 for _ in range(ece_bins)]
    bin_hits = [0 for _ in range(ece_bins)]
    for score in scores:
        bin_index = min(int(score.top_mode_confidence * ece_bins), ece_bins - 1)
        bin_counts[bin_index] += 1
        bin_confidence[bin_index] += score.top_mode_confidence
        bin_hits[bin_index] += score.top_mode_hit
    ece = 0.0
    for count, confidence_sum, hit_sum in zip(bin_counts, bin_confidence, bin_hits):
        if count:
            ece += (count / len(scores)) * abs(
                (confidence_sum / count) - (hit_sum / count)
            )

    ordered = sorted(
        scores, key=lambda score: (-score.top_mode_confidence, score.sample_id)
    )
    cumulative_misses = 0
    prefix_risks: list[float] = []
    for coverage_count, score in enumerate(ordered, start=1):
        cumulative_misses += 1 - score.top_mode_hit
        prefix_risks.append(cumulative_misses / coverage_count)

    try:
        result = ForecastMetrics(
            sample_count=len(scores),
            min_ade_m=math.fsum(score.min_ade_m for score in scores) / len(scores),
            min_fde_m=math.fsum(score.min_fde_m for score in scores) / len(scores),
            miss_rate=math.fsum(score.missed for score in scores) / len(scores),
            mixture_nll=math.fsum(score.mixture_nll for score in scores) / len(scores),
            top_mode_brier=math.fsum(
                (score.top_mode_confidence - score.top_mode_hit) ** 2
                for score in scores
            )
            / len(scores),
            top_mode_ece=ece,
            top_mode_aurc=math.fsum(prefix_risks) / len(prefix_risks),
        )
    except OverflowError as exc:
        raise EvaluationInputError("forecast metric aggregation overflowed") from exc
    if any(
        not math.isfinite(value)
        for value in (
            result.min_ade_m,
            result.min_fde_m,
            result.miss_rate,
            result.mixture_nll,
            result.top_mode_brier,
            result.top_mode_ece,
            result.top_mode_aurc,
        )
    ):
        raise EvaluationInputError("forecast metric aggregation is non-finite")
    return result


def _sample_from_json(
    row: object, *, case_name: str, index: int
) -> HeteroscedasticForecastSample:
    if not isinstance(row, dict) or set(row) != {
        "sample_id",
        "truth",
        "modes",
        "mode_weights",
    }:
        raise EvaluationInputError(
            f"protocol {case_name} sample {index} fields do not match the contract"
        )
    raw_modes = row["modes"]
    if not isinstance(raw_modes, list):
        raise EvaluationInputError(
            f"protocol {case_name} sample {index} modes must be a list"
        )
    modes: list[HeteroscedasticForecastMode] = []
    for mode_index, raw_mode in enumerate(raw_modes):
        if not isinstance(raw_mode, dict) or set(raw_mode) != {"means", "covariances"}:
            raise EvaluationInputError(
                f"protocol {case_name} sample {index} mode {mode_index} fields "
                "do not match the contract"
            )
        try:
            means = tuple(tuple(step) for step in raw_mode["means"])
            covariances = tuple(
                tuple(tuple(matrix_row) for matrix_row in covariance)
                for covariance in raw_mode["covariances"]
            )
        except TypeError as exc:
            raise EvaluationInputError(
                f"protocol {case_name} sample {index} mode {mode_index} shape is invalid"
            ) from exc
        modes.append(HeteroscedasticForecastMode(means, covariances))
    try:
        truth = tuple(tuple(step) for step in row["truth"])
        weights = tuple(row["mode_weights"])
    except TypeError as exc:
        raise EvaluationInputError(
            f"protocol {case_name} sample {index} shape is invalid"
        ) from exc
    return HeteroscedasticForecastSample(
        sample_id=row["sample_id"],
        truth=truth,
        modes=tuple(modes),
        mode_weights=weights,
    )


def _assert_expected_metrics(
    actual: dict[str, object], expected: dict[str, object], *, case_name: str
) -> None:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if not isinstance(expected_value, (int, float)) or isinstance(
            expected_value, bool
        ):
            raise EvaluationInputError(
                f"{case_name}: expected metric {key!r} must be numeric"
            )
        if not isinstance(actual_value, (int, float)) or not math.isclose(
            float(actual_value),
            float(expected_value),
            rel_tol=0.0,
            abs_tol=CONFORMANCE_ABSOLUTE_TOLERANCE,
        ):
            raise AssertionError(
                f"{case_name}: {key} expected {expected_value!r}, got {actual_value!r}"
            )


def run_protocol_conformance(protocol_path: Path) -> dict[str, object]:
    """Run every valid hand-computable vector in the v2 protocol."""

    def reject_constant(value: str) -> None:
        raise EvaluationInputError(f"protocol contains non-finite JSON number: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationInputError(f"protocol contains duplicate key: {key}")
            result[key] = value
        return result

    with protocol_path.open("rb") as protocol_stream:
        protocol_bytes = protocol_stream.read(MAX_PROTOCOL_BYTES + 1)
    if len(protocol_bytes) > MAX_PROTOCOL_BYTES:
        raise EvaluationInputError(
            f"protocol exceeds the resource limit {MAX_PROTOCOL_BYTES} bytes"
        )
    try:
        protocol = json.loads(
            protocol_bytes.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationInputError("protocol must be canonical UTF-8 JSON") from exc
    if not isinstance(protocol, dict) or set(protocol) != PROTOCOL_TOP_LEVEL_FIELDS:
        raise EvaluationInputError(
            "protocol top-level fields do not match the contract"
        )
    if protocol.get("schema_version") != 2:
        raise EvaluationInputError("protocol schema_version must be 2")
    if protocol.get("contract_id") != CONTRACT_ID:
        raise EvaluationInputError(f"protocol contract_id must be {CONTRACT_ID!r}")
    if protocol.get("status") != "reference_conformance_only":
        raise EvaluationInputError(
            "protocol status must remain reference_conformance_only"
        )
    if protocol.get("scientific_claim_allowed") is not False:
        raise EvaluationInputError("protocol cannot allow scientific claims")
    if protocol.get("official_evaluator_equivalence") is not False:
        raise EvaluationInputError(
            "protocol must explicitly deny official evaluator equivalence"
        )
    if protocol.get("implementation") != (
        "experiments/rtp_v2x/evaluator_conformance_v2.py"
    ):
        raise EvaluationInputError("protocol implementation path is invalid")
    if protocol.get("resource_limits") != RESOURCE_LIMITS:
        raise EvaluationInputError(
            "protocol resource_limits do not match the implementation"
        )
    if protocol.get("numeric_policy") != EXPECTED_NUMERIC_POLICY:
        raise EvaluationInputError(
            "protocol numeric_policy does not match the implementation"
        )
    if protocol.get("scope") != EXPECTED_SCOPE:
        raise EvaluationInputError("protocol scope does not match the contract")
    if protocol.get("forecasting") != EXPECTED_FORECASTING:
        raise EvaluationInputError(
            "protocol forecasting semantics do not match the contract"
        )
    if protocol.get("parity_gate") != EXPECTED_PARITY_GATE:
        raise EvaluationInputError("protocol parity_gate does not match the contract")

    test_vectors = protocol.get("test_vectors")
    if not isinstance(test_vectors, dict) or set(test_vectors) != {"forecasting"}:
        raise EvaluationInputError(
            "protocol test_vectors must contain forecasting exactly"
        )
    raw_cases = test_vectors["forecasting"]
    if not isinstance(raw_cases, list) or len(raw_cases) != len(FORECAST_CASE_NAMES):
        raise EvaluationInputError("protocol forecasting cases are incomplete")

    passed: list[str] = []
    seen_case_names: set[str] = set()
    for case_index, case in enumerate(raw_cases):
        if not isinstance(case, dict) or set(case) != {
            "name",
            "miss_threshold_m",
            "ece_bins",
            "samples",
            "expected",
        }:
            raise EvaluationInputError(
                f"protocol forecasting[{case_index}] fields do not match the contract"
            )
        case_name = case.get("name")
        if (
            not isinstance(case_name, str)
            or case_name not in FORECAST_CASE_NAMES
            or case_name in seen_case_names
        ):
            raise EvaluationInputError("protocol forecasting case names are invalid")
        expected = case.get("expected")
        if not isinstance(expected, dict) or set(expected) != FORECAST_EXPECTED_FIELDS:
            raise EvaluationInputError(
                f"protocol forecasting/{case_name} expected fields are incomplete"
            )
        raw_samples = case.get("samples")
        if not isinstance(raw_samples, list) or not raw_samples:
            raise EvaluationInputError(
                f"protocol forecasting/{case_name} samples are incomplete"
            )
        samples = tuple(
            _sample_from_json(row, case_name=case_name, index=index)
            for index, row in enumerate(raw_samples)
        )
        metrics = evaluate_forecasts(
            samples,
            miss_threshold_m=case["miss_threshold_m"],
            ece_bins=case["ece_bins"],
        )
        _assert_expected_metrics(
            asdict(metrics), expected, case_name=f"forecasting/{case_name}"
        )
        seen_case_names.add(case_name)
        passed.append(f"forecasting/{case_name}")
    if seen_case_names != FORECAST_CASE_NAMES:
        raise EvaluationInputError("protocol forecasting case names are incomplete")

    return {
        "contract_id": CONTRACT_ID,
        "scientific_claim_allowed": False,
        "official_evaluator_equivalence": False,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "passed_case_count": len(passed),
        "passed_cases": passed,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_protocol_conformance(args.protocol)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
