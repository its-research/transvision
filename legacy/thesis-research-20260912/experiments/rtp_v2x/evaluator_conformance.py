#!/usr/bin/env python3
"""Dependency-light reference evaluator for deterministic conformance cases.

This module is deliberately limited to small, hand-constructed inputs.  It
freezes metric semantics and catches integration regressions; it is not the
official V2X-Seq evaluator and cannot be used to publish benchmark results
until dual-implementation parity has been demonstrated on frozen predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence


CONTRACT_ID = "RTPV2X-EVALUATOR-CONFORMANCE-v1"
MAX_REFERENCE_IDENTITIES = 12
PROTOCOL_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "status",
        "scientific_claim_allowed",
        "official_evaluator_equivalence",
        "implementation",
        "scope",
        "causality",
        "tracking",
        "forecasting",
        "numeric_policy",
        "test_vectors",
        "parity_gate",
    }
)
TRACKING_CASE_EXPECTED_FIELDS = {
    "perfect_prediction": frozenset(
        {
            "true_positives",
            "false_positives",
            "false_negatives",
            "identity_switches",
            "fragmentations",
            "idf1",
            "hota_single_threshold",
        }
    ),
    "all_missed_detection": frozenset(
        {
            "true_positives",
            "false_positives",
            "false_negatives",
            "recall",
            "precision",
            "identity_switches",
            "fragmentations",
            "idf1",
            "hota_single_threshold",
        }
    ),
    "two_identity_swap": frozenset(
        {
            "true_positives",
            "identity_switches",
            "fragmentations",
            "id_true_positives",
            "id_false_positives",
            "id_false_negatives",
            "idf1",
            "association_accuracy",
            "hota_single_threshold",
        }
    ),
    "one_recovered_fragment": frozenset(
        {
            "true_positives",
            "false_positives",
            "false_negatives",
            "identity_switches",
            "fragmentations",
            "idf1",
            "detection_accuracy",
            "association_accuracy",
            "hota_single_threshold",
        }
    ),
}
FORECAST_CASE_EXPECTED_FIELDS = {
    name: frozenset(
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
    for name in (
        "perfect_single_mode",
        "all_modes_missed",
        "non_unit_weights_are_normalized",
    )
}


class EvaluationInputError(ValueError):
    """Raised when an input would make a metric undefined or non-reproducible."""


@dataclass(frozen=True)
class TrackPoint:
    """One 3D object center at one sequence frame."""

    frame_index: int
    track_id: str
    position_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class TimedObservation:
    """One message with capture and complete-message arrival timestamps."""

    message_id: str
    source_id: str
    event_time: int
    arrival_time: int


@dataclass(frozen=True)
class TrackingMetrics:
    ground_truth_detections: int
    predicted_detections: int
    true_positives: int
    false_positives: int
    false_negatives: int
    recall: float
    precision: float
    identity_switches: int
    fragmentations: int
    id_true_positives: int
    id_false_positives: int
    id_false_negatives: int
    idf1: float
    detection_accuracy: float
    association_accuracy: float
    hota_single_threshold: float


@dataclass(frozen=True)
class ForecastSample:
    """One multimodal forecast and its realized trajectory.

    Trajectories may be planar ``(x, y)`` or spatial ``(x, y, z)``.  Every
    mode must have the same horizon and dimensionality as ``truth``.
    """

    sample_id: str
    truth: tuple[tuple[float, ...], ...]
    modes: tuple[tuple[tuple[float, ...], ...], ...]
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
class _ForecastCaseScore:
    sample_id: str
    min_ade_m: float
    min_fde_m: float
    missed: int
    mixture_nll: float
    top_mode_confidence: float
    top_mode_hit: int


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationInputError(f"{label} must be a non-negative integer")
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


def _require_identifier(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > 255
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise EvaluationInputError(f"{label} must be a non-empty string")
    return value


def _validate_track_points(
    points: Iterable[TrackPoint], label: str
) -> tuple[TrackPoint, ...]:
    rows = tuple(points)
    seen: set[tuple[int, str]] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, TrackPoint):
            raise EvaluationInputError(f"{label}[{row_index}] must be TrackPoint")
        _require_nonnegative_int(row.frame_index, f"{label}[{row_index}].frame_index")
        _require_identifier(row.track_id, f"{label}[{row_index}].track_id")
        if not isinstance(row.position_xyz, (tuple, list)) or len(row.position_xyz) != 3:
            raise EvaluationInputError(
                f"{label}[{row_index}].position_xyz must contain exactly three coordinates"
            )
        for axis, value in zip("xyz", row.position_xyz):
            _require_finite(value, f"{label}[{row_index}].position_{axis}")
        key = (row.frame_index, row.track_id)
        if key in seen:
            raise EvaluationInputError(
                f"{label} contains duplicate (frame_index, track_id) {key!r}"
            )
        seen.add(key)
    return rows


def _assignment_is_better(
    candidate: tuple[int, float, tuple[tuple[int, int], ...]],
    incumbent: tuple[int, float, tuple[tuple[int, int], ...]],
) -> bool:
    if candidate[0] != incumbent[0]:
        return candidate[0] > incumbent[0]
    if candidate[1] != incumbent[1]:
        return candidate[1] < incumbent[1]
    return candidate[2] < incumbent[2]


def _match_frame(
    ground_truth: Sequence[TrackPoint],
    predictions: Sequence[TrackPoint],
    distance_threshold_m: float,
) -> tuple[tuple[int, int], ...]:
    """Return an exact small-set assignment.

    The objective is lexicographic: maximize the number of threshold-valid
    matches, minimize total Euclidean distance, then choose the smallest pair
    index sequence.  Input rows are already sorted by identity.
    """

    if len(ground_truth) > MAX_REFERENCE_IDENTITIES or len(predictions) > MAX_REFERENCE_IDENTITIES:
        raise EvaluationInputError(
            "reference assignment supports at most "
            f"{MAX_REFERENCE_IDENTITIES} ground-truth and predicted identities per frame"
        )

    distances: tuple[tuple[float, ...], ...] = tuple(
        tuple(math.dist(gt.position_xyz, prediction.position_xyz) for prediction in predictions)
        for gt in ground_truth
    )

    @lru_cache(maxsize=None)
    def solve(
        ground_truth_index: int, used_predictions: int
    ) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if ground_truth_index == len(ground_truth):
            return (0, 0.0, ())

        best = solve(ground_truth_index + 1, used_predictions)
        for prediction_index in range(len(predictions)):
            if used_predictions & (1 << prediction_index):
                continue
            distance = distances[ground_truth_index][prediction_index]
            if distance > distance_threshold_m:
                continue
            tail = solve(
                ground_truth_index + 1, used_predictions | (1 << prediction_index)
            )
            candidate = (
                tail[0] + 1,
                tail[1] + distance,
                ((ground_truth_index, prediction_index),) + tail[2],
            )
            if _assignment_is_better(candidate, best):
                best = candidate
        return best

    return solve(0, 0)[2]


def _maximum_identity_matches(
    counts: dict[tuple[str, str], int], gt_ids: Sequence[str], prediction_ids: Sequence[str]
) -> int:
    if len(gt_ids) > MAX_REFERENCE_IDENTITIES or len(prediction_ids) > MAX_REFERENCE_IDENTITIES:
        raise EvaluationInputError(
            "reference identity assignment supports at most "
            f"{MAX_REFERENCE_IDENTITIES} identities per side"
        )

    @lru_cache(maxsize=None)
    def solve(gt_index: int, used_predictions: int) -> int:
        if gt_index == len(gt_ids):
            return 0
        best = solve(gt_index + 1, used_predictions)
        for prediction_index, prediction_id in enumerate(prediction_ids):
            if used_predictions & (1 << prediction_index):
                continue
            best = max(
                best,
                counts.get((gt_ids[gt_index], prediction_id), 0)
                + solve(gt_index + 1, used_predictions | (1 << prediction_index)),
            )
        return best

    return solve(0, 0)


def evaluate_tracking(
    ground_truth: Iterable[TrackPoint],
    predictions: Iterable[TrackPoint],
    *,
    distance_threshold_m: float,
) -> TrackingMetrics:
    """Evaluate deterministic 3D tracking semantics at one distance threshold.

    This is a conformance reference, not an official AMOTA/HOTA implementation.
    In particular, ``hota_single_threshold`` is one threshold sample and is not
    the threshold-averaged HOTA value used by benchmark packages.
    """

    threshold = _require_finite(distance_threshold_m, "distance_threshold_m")
    if threshold < 0.0:
        raise EvaluationInputError("distance_threshold_m must be non-negative")
    gt_rows = _validate_track_points(ground_truth, "ground_truth")
    prediction_rows = _validate_track_points(predictions, "predictions")
    if not gt_rows:
        raise EvaluationInputError("ground_truth must contain at least one detection")

    gt_by_frame: dict[int, list[TrackPoint]] = {}
    prediction_by_frame: dict[int, list[TrackPoint]] = {}
    for row in gt_rows:
        gt_by_frame.setdefault(row.frame_index, []).append(row)
    for row in prediction_rows:
        prediction_by_frame.setdefault(row.frame_index, []).append(row)

    matches: list[tuple[int, str, str]] = []
    for frame_index in sorted(set(gt_by_frame) | set(prediction_by_frame)):
        frame_gt = sorted(gt_by_frame.get(frame_index, []), key=lambda row: row.track_id)
        frame_predictions = sorted(
            prediction_by_frame.get(frame_index, []), key=lambda row: row.track_id
        )
        for gt_index, prediction_index in _match_frame(
            frame_gt, frame_predictions, threshold
        ):
            matches.append(
                (
                    frame_index,
                    frame_gt[gt_index].track_id,
                    frame_predictions[prediction_index].track_id,
                )
            )

    true_positives = len(matches)
    false_positives = len(prediction_rows) - true_positives
    false_negatives = len(gt_rows) - true_positives
    recall = true_positives / len(gt_rows)
    precision = true_positives / len(prediction_rows) if prediction_rows else 0.0

    match_by_gt_observation = {
        (frame_index, gt_id): prediction_id
        for frame_index, gt_id, prediction_id in matches
    }
    identity_switches = 0
    fragmentations = 0
    for gt_id in sorted({row.track_id for row in gt_rows}):
        identity_rows = sorted(
            (row for row in gt_rows if row.track_id == gt_id),
            key=lambda row: row.frame_index,
        )
        previous_prediction_id: str | None = None
        matched_segments = 0
        in_matched_segment = False
        for row in identity_rows:
            prediction_id = match_by_gt_observation.get((row.frame_index, gt_id))
            if prediction_id is None:
                in_matched_segment = False
                continue
            if not in_matched_segment:
                matched_segments += 1
                in_matched_segment = True
            if previous_prediction_id is not None and prediction_id != previous_prediction_id:
                identity_switches += 1
            previous_prediction_id = prediction_id
        fragmentations += max(0, matched_segments - 1)

    gt_ids = sorted({row.track_id for row in gt_rows})
    prediction_ids = sorted({row.track_id for row in prediction_rows})
    pair_counts: dict[tuple[str, str], int] = {}
    for _, gt_id, prediction_id in matches:
        pair_counts[(gt_id, prediction_id)] = pair_counts.get((gt_id, prediction_id), 0) + 1
    id_true_positives = _maximum_identity_matches(pair_counts, gt_ids, prediction_ids)
    id_false_positives = len(prediction_rows) - id_true_positives
    id_false_negatives = len(gt_rows) - id_true_positives
    idf1_denominator = 2 * id_true_positives + id_false_positives + id_false_negatives
    idf1 = 2 * id_true_positives / idf1_denominator if idf1_denominator else 0.0

    detection_denominator = true_positives + false_positives + false_negatives
    detection_accuracy = (
        true_positives / detection_denominator if detection_denominator else 0.0
    )
    gt_identity_counts: dict[str, int] = {}
    prediction_identity_counts: dict[str, int] = {}
    for row in gt_rows:
        gt_identity_counts[row.track_id] = gt_identity_counts.get(row.track_id, 0) + 1
    for row in prediction_rows:
        prediction_identity_counts[row.track_id] = (
            prediction_identity_counts.get(row.track_id, 0) + 1
        )
    if true_positives:
        association_accuracy = sum(
            pair_counts[(gt_id, prediction_id)]
            / (
                gt_identity_counts[gt_id]
                + prediction_identity_counts[prediction_id]
                - pair_counts[(gt_id, prediction_id)]
            )
            for _, gt_id, prediction_id in matches
        ) / true_positives
    else:
        association_accuracy = 0.0
    hota_single_threshold = math.sqrt(detection_accuracy * association_accuracy)

    return TrackingMetrics(
        ground_truth_detections=len(gt_rows),
        predicted_detections=len(prediction_rows),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        recall=recall,
        precision=precision,
        identity_switches=identity_switches,
        fragmentations=fragmentations,
        id_true_positives=id_true_positives,
        id_false_positives=id_false_positives,
        id_false_negatives=id_false_negatives,
        idf1=idf1,
        detection_accuracy=detection_accuracy,
        association_accuracy=association_accuracy,
        hota_single_threshold=hota_single_threshold,
    )


def select_causal_observations(
    observations: Iterable[TimedObservation], *, decision_time: int
) -> tuple[TimedObservation, ...]:
    """Return the decision-time prefix in a canonical arrival order.

    Eligibility is inclusive: ``arrival_time <= decision_time``.  The function
    first validates every row, including future rows, so malformed data cannot
    be hidden beyond the current decision boundary.
    """

    cutoff = _require_nonnegative_int(decision_time, "decision_time")
    rows = tuple(observations)
    seen_message_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, TimedObservation):
            raise EvaluationInputError(
                f"observations[{row_index}] must be TimedObservation"
            )
        message_id = _require_identifier(
            row.message_id, f"observations[{row_index}].message_id"
        )
        _require_identifier(row.source_id, f"observations[{row_index}].source_id")
        event_time = _require_nonnegative_int(
            row.event_time, f"observations[{row_index}].event_time"
        )
        arrival_time = _require_nonnegative_int(
            row.arrival_time, f"observations[{row_index}].arrival_time"
        )
        if event_time > arrival_time:
            raise EvaluationInputError(
                f"observations[{row_index}] has event_time after arrival_time"
            )
        if message_id in seen_message_ids:
            raise EvaluationInputError(f"duplicate message_id {message_id!r}")
        seen_message_ids.add(message_id)
    return tuple(
        sorted(
            (row for row in rows if row.arrival_time <= cutoff),
            key=lambda row: (
                row.arrival_time,
                row.event_time,
                row.source_id,
                row.message_id,
            ),
        )
    )


def normalize_mode_probabilities(weights: Sequence[float]) -> tuple[float, ...]:
    """L1-normalize finite non-negative mode weights.

    Inputs are weights, not logits.  Negative values, non-finite values and an
    all-zero vector are rejected instead of repaired.
    """

    if not isinstance(weights, (tuple, list)) or not weights:
        raise EvaluationInputError("mode_weights must be a non-empty sequence")
    validated: list[float] = []
    for index, value in enumerate(weights):
        converted = _require_finite(value, f"mode_weights[{index}]")
        if converted < 0.0:
            raise EvaluationInputError("mode_weights must be non-negative")
        validated.append(converted)
    try:
        total = math.fsum(validated)
    except OverflowError as exc:
        raise EvaluationInputError("mode_weights sum must remain finite") from exc
    if not math.isfinite(total) or total <= 0.0:
        raise EvaluationInputError("mode_weights must have a positive finite sum")
    normalized = tuple(value / total for value in validated)
    # Make the invariant explicit even if a platform has unusual float behavior.
    if not math.isclose(math.fsum(normalized), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise EvaluationInputError("normalized mode probabilities do not sum to one")
    return normalized


def _validated_forecast_sample(sample: ForecastSample) -> tuple[int, int, tuple[float, ...]]:
    if not isinstance(sample, ForecastSample):
        raise EvaluationInputError("each forecast sample must be ForecastSample")
    _require_identifier(sample.sample_id, "sample_id")
    if not isinstance(sample.truth, (tuple, list)) or not sample.truth:
        raise EvaluationInputError(f"sample {sample.sample_id!r} has an empty truth trajectory")
    first_step = sample.truth[0]
    if not isinstance(first_step, (tuple, list)) or len(first_step) not in (2, 3):
        raise EvaluationInputError(
            f"sample {sample.sample_id!r} truth must use 2D or 3D coordinates"
        )
    dimension = len(first_step)
    horizon = len(sample.truth)
    for time_index, position in enumerate(sample.truth):
        if not isinstance(position, (tuple, list)) or len(position) != dimension:
            raise EvaluationInputError(
                f"sample {sample.sample_id!r} truth[{time_index}] has inconsistent dimension"
            )
        for axis_index, value in enumerate(position):
            _require_finite(
                value, f"sample {sample.sample_id!r} truth[{time_index}][{axis_index}]"
            )
    if not isinstance(sample.modes, (tuple, list)) or not sample.modes:
        raise EvaluationInputError(f"sample {sample.sample_id!r} has no forecast modes")
    if len(sample.modes) > MAX_REFERENCE_IDENTITIES:
        raise EvaluationInputError(
            f"reference evaluator supports at most {MAX_REFERENCE_IDENTITIES} modes"
        )
    for mode_index, mode in enumerate(sample.modes):
        if not isinstance(mode, (tuple, list)) or len(mode) != horizon:
            raise EvaluationInputError(
                f"sample {sample.sample_id!r} mode {mode_index} has inconsistent horizon"
            )
        for time_index, position in enumerate(mode):
            if not isinstance(position, (tuple, list)) or len(position) != dimension:
                raise EvaluationInputError(
                    f"sample {sample.sample_id!r} mode {mode_index} step {time_index} "
                    "has inconsistent dimension"
                )
            for axis_index, value in enumerate(position):
                _require_finite(
                    value,
                    f"sample {sample.sample_id!r} mode[{mode_index}]"
                    f"[{time_index}][{axis_index}]",
                )
    if len(sample.mode_weights) != len(sample.modes):
        raise EvaluationInputError(
            f"sample {sample.sample_id!r} mode_weights length does not match mode count"
        )
    probabilities = normalize_mode_probabilities(sample.mode_weights)
    return horizon, dimension, probabilities


def _logsumexp(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise EvaluationInputError("log-sum-exp requires at least one finite component")
    maximum = max(values)
    result = maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in values)
    )
    if not math.isfinite(result):
        raise EvaluationInputError("log-sum-exp produced a non-finite result")
    return result


def _score_forecast_sample(
    sample: ForecastSample, *, miss_threshold_m: float, nll_sigma_m: float
) -> _ForecastCaseScore:
    horizon, dimension, probabilities = _validated_forecast_sample(sample)
    ade_by_mode: list[float] = []
    fde_by_mode: list[float] = []
    squared_error_by_mode: list[float] = []
    try:
        for mode in sample.modes:
            distances = [
                math.dist(prediction, truth)
                for prediction, truth in zip(mode, sample.truth)
            ]
            ade_by_mode.append(math.fsum(distances) / horizon)
            fde_by_mode.append(distances[-1])
            squared_error_by_mode.append(
                math.fsum(
                    (float(predicted) - float(target)) ** 2
                    for prediction, truth in zip(mode, sample.truth)
                    for predicted, target in zip(prediction, truth)
                )
            )
    except (OverflowError, ValueError) as exc:
        raise EvaluationInputError("forecast distance calculation overflowed") from exc
    if any(
        not math.isfinite(value)
        for value in (*ade_by_mode, *fde_by_mode, *squared_error_by_mode)
    ):
        raise EvaluationInputError("forecast distance calculation is non-finite")

    gaussian_dimension = horizon * dimension
    try:
        log_normalizer = -0.5 * gaussian_dimension * (
            math.log(2.0 * math.pi) + 2.0 * math.log(nll_sigma_m)
        )
        log_components = []
        for probability, squared_error in zip(
            probabilities, squared_error_by_mode
        ):
            if probability <= 0.0:
                continue
            scaled_error = math.sqrt(squared_error) / nll_sigma_m
            quadratic = 0.5 * scaled_error * scaled_error
            if not math.isfinite(quadratic):
                raise EvaluationInputError("forecast NLL quadratic is non-finite")
            log_components.append(
                math.log(probability) + log_normalizer - quadratic
            )
    except (OverflowError, ValueError) as exc:
        raise EvaluationInputError("forecast NLL calculation overflowed") from exc
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
    samples: Iterable[ForecastSample],
    *,
    miss_threshold_m: float,
    nll_sigma_m: float,
    ece_bins: int,
) -> ForecastMetrics:
    """Evaluate deterministic multimodal trajectory and confidence metrics.

    ``minADE`` and ``minFDE`` minimize independently across modes. ``MR`` uses
    the oracle minimum FDE with an equality-as-hit threshold.  ``NLL`` is the
    full-horizon density under a fixed isotropic Gaussian mixture.  Brier, ECE
    and AURC evaluate the normalized highest-weight mode as a selective
    predictor; they do not claim to be official V2X-Seq calibration metrics.
    """

    threshold = _require_finite(miss_threshold_m, "miss_threshold_m")
    if threshold < 0.0:
        raise EvaluationInputError("miss_threshold_m must be non-negative")
    sigma = _require_finite(nll_sigma_m, "nll_sigma_m")
    if sigma <= 0.0:
        raise EvaluationInputError("nll_sigma_m must be positive")
    if isinstance(ece_bins, bool) or not isinstance(ece_bins, int) or ece_bins <= 0:
        raise EvaluationInputError("ece_bins must be a positive integer")

    rows = tuple(samples)
    if not rows:
        raise EvaluationInputError("samples must contain at least one forecast")
    sample_ids: set[str] = set()
    scores: list[_ForecastCaseScore] = []
    for sample in rows:
        if isinstance(sample, ForecastSample) and sample.sample_id in sample_ids:
            raise EvaluationInputError(f"duplicate sample_id {sample.sample_id!r}")
        score = _score_forecast_sample(
            sample, miss_threshold_m=threshold, nll_sigma_m=sigma
        )
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
        if count == 0:
            continue
        ece += (count / len(scores)) * abs((confidence_sum / count) - (hit_sum / count))

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


def _track_points_from_json(rows: Sequence[dict[str, object]]) -> tuple[TrackPoint, ...]:
    return tuple(
        TrackPoint(
            frame_index=row["frame_index"],  # type: ignore[arg-type]
            track_id=row["track_id"],  # type: ignore[arg-type]
            position_xyz=tuple(row["position_xyz"]),  # type: ignore[arg-type]
        )
        for row in rows
    )


def _forecast_samples_from_json(rows: Sequence[dict[str, object]]) -> tuple[ForecastSample, ...]:
    return tuple(
        ForecastSample(
            sample_id=row["sample_id"],  # type: ignore[arg-type]
            truth=tuple(tuple(step) for step in row["truth"]),  # type: ignore[arg-type]
            modes=tuple(
                tuple(tuple(step) for step in mode)
                for mode in row["modes"]  # type: ignore[union-attr]
            ),
            mode_weights=tuple(row["mode_weights"]),  # type: ignore[arg-type]
        )
        for row in rows
    )


def _assert_expected_metrics(
    actual: dict[str, object], expected: dict[str, object], *, case_name: str
) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise AssertionError(f"{case_name}: unknown expected metric {key!r}")
        actual_value = actual[key]
        if isinstance(expected_value, bool):
            if actual_value is not expected_value:
                raise AssertionError(
                    f"{case_name}: {key} expected {expected_value!r}, got {actual_value!r}"
                )
        elif isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            if not math.isclose(
                float(actual_value),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise AssertionError(
                    f"{case_name}: {key} expected {expected_value!r}, got {actual_value!r}"
                )
        elif actual_value != expected_value:
            raise AssertionError(
                f"{case_name}: {key} expected {expected_value!r}, got {actual_value!r}"
            )


def run_protocol_conformance(protocol_path: Path) -> dict[str, object]:
    """Execute the valid test vectors embedded in the versioned JSON contract."""

    def reject_constant(value: str) -> None:
        raise EvaluationInputError(f"protocol contains non-finite JSON number: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationInputError(f"protocol contains duplicate key: {key}")
            result[key] = value
        return result

    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(
        protocol_bytes.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(protocol, dict) or set(protocol) != PROTOCOL_TOP_LEVEL_FIELDS:
        raise EvaluationInputError("protocol top-level fields do not match the contract")
    if protocol.get("schema_version") != 1:
        raise EvaluationInputError("protocol schema_version must be 1")
    if protocol.get("contract_id") != CONTRACT_ID:
        raise EvaluationInputError(
            f"protocol contract_id must be {CONTRACT_ID!r}"
        )
    if protocol.get("official_evaluator_equivalence") is not False:
        raise EvaluationInputError("protocol must explicitly deny official evaluator equivalence")
    if protocol.get("status") != "reference_conformance_only":
        raise EvaluationInputError("protocol status must remain reference_conformance_only")
    if protocol.get("scientific_claim_allowed") is not False:
        raise EvaluationInputError("protocol cannot allow scientific claims")
    passed: list[str] = []
    test_vectors = protocol.get("test_vectors")
    if not isinstance(test_vectors, dict) or set(test_vectors) != {
        "tracking",
        "forecasting",
        "causal_ordering",
    }:
        raise EvaluationInputError("protocol test_vectors must contain all vector groups exactly")

    def validate_cases(
        raw_cases: object,
        *,
        expected_by_name: dict[str, frozenset[str]],
        case_fields: frozenset[str],
        group: str,
    ) -> list[dict[str, object]]:
        if not isinstance(raw_cases, list) or len(raw_cases) != len(expected_by_name):
            raise EvaluationInputError(f"protocol {group} cases are incomplete")
        cases: list[dict[str, object]] = []
        names: set[str] = set()
        for position, case in enumerate(raw_cases):
            if not isinstance(case, dict) or set(case) != case_fields:
                raise EvaluationInputError(
                    f"protocol {group}[{position}] fields do not match the contract"
                )
            name = case.get("name")
            if not isinstance(name, str) or name not in expected_by_name or name in names:
                raise EvaluationInputError(f"protocol {group} case names are invalid")
            expected = case.get("expected")
            if not isinstance(expected, dict) or set(expected) != expected_by_name[name]:
                raise EvaluationInputError(
                    f"protocol {group}/{name} expected fields are incomplete"
                )
            names.add(name)
            cases.append(case)
        if names != set(expected_by_name):
            raise EvaluationInputError(f"protocol {group} case names are incomplete")
        return cases

    tracking_cases = validate_cases(
        test_vectors["tracking"],
        expected_by_name=TRACKING_CASE_EXPECTED_FIELDS,
        case_fields=frozenset(
            {"name", "distance_threshold_m", "ground_truth", "predictions", "expected"}
        ),
        group="tracking",
    )
    forecast_cases = validate_cases(
        test_vectors["forecasting"],
        expected_by_name=FORECAST_CASE_EXPECTED_FIELDS,
        case_fields=frozenset(
            {
                "name",
                "miss_threshold_m",
                "nll_sigma_m",
                "ece_bins",
                "samples",
                "expected",
            }
        ),
        group="forecasting",
    )
    causal_case = test_vectors["causal_ordering"]
    if (
        not isinstance(causal_case, dict)
        or set(causal_case) != {"decision_time", "observations", "expected_message_ids"}
        or not isinstance(causal_case.get("observations"), list)
        or not causal_case["observations"]
        or not isinstance(causal_case.get("expected_message_ids"), list)
        or not causal_case["expected_message_ids"]
    ):
        raise EvaluationInputError("protocol causal_ordering case is incomplete")

    for case in tracking_cases:
        name = f"tracking/{case['name']}"
        metrics = evaluate_tracking(
            _track_points_from_json(case["ground_truth"]),
            _track_points_from_json(case["predictions"]),
            distance_threshold_m=case["distance_threshold_m"],
        )
        _assert_expected_metrics(asdict(metrics), case["expected"], case_name=name)
        passed.append(name)

    for case in forecast_cases:
        name = f"forecasting/{case['name']}"
        metrics = evaluate_forecasts(
            _forecast_samples_from_json(case["samples"]),
            miss_threshold_m=case["miss_threshold_m"],
            nll_sigma_m=case["nll_sigma_m"],
            ece_bins=case["ece_bins"],
        )
        _assert_expected_metrics(asdict(metrics), case["expected"], case_name=name)
        passed.append(name)

    messages = tuple(
        TimedObservation(
            message_id=row["message_id"],
            source_id=row["source_id"],
            event_time=row["event_time"],
            arrival_time=row["arrival_time"],
        )
        for row in causal_case["observations"]
    )
    selected = select_causal_observations(
        messages, decision_time=causal_case["decision_time"]
    )
    selected_ids = [row.message_id for row in selected]
    if selected_ids != causal_case["expected_message_ids"]:
        raise AssertionError(
            "causal_ordering expected "
            f"{causal_case['expected_message_ids']!r}, got {selected_ids!r}"
        )
    passed.append("causal_ordering")

    return {
        "contract_id": CONTRACT_ID,
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
