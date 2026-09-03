"""Deterministic multi-target reference adapter for detector-locked studies.

This is a deliberately lightweight reference implementation, not a learned
SOTA tracker.  It gives every experiment adapter the same lifecycle semantics:
strict no-future ingestion, deterministic association and births, explicit
survival/pruning, bounded identity hypotheses, and append-only commits.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import struct
from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from .association import chi_square_quantile, gaussian_nll, solve_one_to_one
from .commit import AppendOnlyCommitLog, CommitRecord
from .contracts import (
    DetectionCacheV1,
    IdentityHypothesisV1,
    TrackingPredictionV1,
)
from .schema import Lineage


_STATE_DIMENSION = 9
_CENTER_INDICES = np.asarray((0, 1, 2), dtype=np.int64)


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class ReferenceTrackerConfig:
    gate_probability: float = 0.99
    birth_score_threshold: float = 0.3
    prune_existence_threshold: float = 0.05
    survival_probability_per_second: float = 0.95
    missed_detection_factor: float = 0.75
    max_age_seconds: float = 2.0
    process_noise_per_second: float = 0.1
    unmatched_track_cost: float = 12.0
    unmatched_detection_cost: float = 12.0
    top_h: int = 3
    reference_agent_id: str = "vehicle"

    def __post_init__(self) -> None:
        for name in (
            "gate_probability",
            "birth_score_threshold",
            "prune_existence_threshold",
            "survival_probability_per_second",
            "missed_detection_factor",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "max_age_seconds",
            "unmatched_track_cost",
            "unmatched_detection_cost",
        ):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        process_noise = _finite(
            self.process_noise_per_second, "process_noise_per_second"
        )
        if process_noise < 0.0:
            raise ValueError("process_noise_per_second must be non-negative")
        object.__setattr__(self, "process_noise_per_second", process_noise)
        if isinstance(self.top_h, bool) or not isinstance(self.top_h, int):
            raise TypeError("top_h must be an integer")
        if self.top_h <= 0:
            raise ValueError("top_h must be positive")
        if self.birth_score_threshold < self.prune_existence_threshold:
            raise ValueError(
                "birth_score_threshold must be at least prune_existence_threshold"
            )
        if (
            not isinstance(self.reference_agent_id, str)
            or not self.reference_agent_id
            or self.reference_agent_id != self.reference_agent_id.strip()
        ):
            raise ValueError("reference_agent_id must be a trimmed non-empty string")


class TrackerIngestStatus(str, Enum):
    APPLIED = "applied"
    DUPLICATE_CACHE = "duplicate_cache"
    SEQUENCE_MISMATCH = "sequence_mismatch"
    COORDINATE_FRAME_MISMATCH = "coordinate_frame_mismatch"
    FUTURE_EVENT = "future_event"
    FUTURE_ARRIVAL = "future_arrival"
    CONFLICTING_CACHE_IDENTITY = "conflicting_cache_identity"


@dataclass(frozen=True, slots=True)
class TrackerIngestResult:
    status: TrackerIngestStatus
    cache_sha256: str
    matched_track_ids: tuple[str, ...] = ()
    born_track_ids: tuple[str, ...] = ()
    pruned_track_ids: tuple[str, ...] = ()

    @property
    def applied(self) -> bool:
        return self.status is TrackerIngestStatus.APPLIED


@runtime_checkable
class TrackerAdapter(Protocol):
    """Minimal lifecycle shared by all detector-locked tracking baselines."""

    @property
    def adapter_name(self) -> str: ...

    @property
    def current_time(self) -> float: ...

    def reset(self, *, sequence_id: str, initial_time: float) -> None: ...

    def ingest(
        self, detections: DetectionCacheV1, *, arrival_time: float
    ) -> TrackerIngestResult: ...

    def advance(self, decision_time: float) -> tuple[TrackingPredictionV1, ...]: ...

    def commit(self, decision_time: float) -> tuple[TrackingPredictionV1, ...]: ...

    def finalize(self) -> tuple[TrackingPredictionV1, ...]: ...


@dataclass(slots=True)
class _Track:
    track_id: str
    class_label: str
    mean: np.ndarray
    covariance: np.ndarray
    state_time: float
    last_update_time: float
    last_event_time: float
    last_arrival_time: float
    existence_probability: float
    identity_hypotheses: tuple[IdentityHypothesisV1, ...]
    other_identity_probability: float
    factor_ids: set[str]


def _transition(delta: float) -> np.ndarray:
    result = np.eye(_STATE_DIMENSION, dtype=np.float64)
    result[0, 7] = delta
    result[1, 8] = delta
    return result


def _process_covariance(delta: float, density: float) -> np.ndarray:
    # A small diagonal model is intentionally conservative for dimensions for
    # which the public cache does not provide angular/size rates.
    return np.eye(_STATE_DIMENSION, dtype=np.float64) * density * delta


def _predict_state(
    mean: np.ndarray,
    covariance: np.ndarray,
    delta: float,
    process_noise_per_second: float,
) -> tuple[np.ndarray, np.ndarray]:
    if delta < -1e-12:
        raise ValueError("state prediction cannot move backwards")
    delta = max(0.0, delta)
    transition = _transition(delta)
    predicted_mean = transition @ mean
    # NumPy 2.1 on macOS Accelerate may emit spurious floating warnings for a
    # finite matrix product; the explicit postcondition below remains strict.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        predicted_covariance = (
            transition @ covariance @ transition.T
            + _process_covariance(delta, process_noise_per_second)
        )
    if not np.all(np.isfinite(predicted_covariance)):
        raise FloatingPointError("state prediction produced non-finite covariance")
    predicted_covariance = 0.5 * (
        predicted_covariance + predicted_covariance.T
    )
    return predicted_mean, predicted_covariance


def _kalman_update(
    mean: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    measurement_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    innovation_covariance = covariance + measurement_covariance
    gain = np.linalg.solve(innovation_covariance, covariance).T
    updated_mean = mean + gain @ (measurement - mean)
    residual = np.eye(_STATE_DIMENSION) - gain
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        updated_covariance = (
            residual @ covariance @ residual.T
            + gain @ measurement_covariance @ gain.T
        )
    if not np.all(np.isfinite(updated_covariance)):
        raise FloatingPointError("Kalman update produced non-finite covariance")
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    return updated_mean, updated_covariance


def _canonical_detection_order(cache: DetectionCacheV1) -> tuple[int, ...]:
    states = cache.state_vectors()

    def key(index: int) -> tuple[object, ...]:
        covariance_digest = hashlib.sha256(
            cache.covariances[index].tobytes()
        ).hexdigest()
        return (
            cache.class_labels[index],
            *states[index].tolist(),
            cache.scores[index],
            covariance_digest,
            index,
        )

    return tuple(sorted(range(cache.count), key=key))


def _top_hypotheses(
    probabilities: dict[str, float], top_h: int
) -> tuple[tuple[IdentityHypothesisV1, ...], float]:
    finite = {
        label: float(probability)
        for label, probability in probabilities.items()
        if np.isfinite(probability) and probability > 0.0
    }
    if not finite:
        raise ValueError("identity mixture has no positive finite probability")
    selected = sorted(finite.items(), key=lambda item: (-item[1], item[0]))[:top_h]
    selected_probability = sum(probability for _, probability in selected)
    total_probability = sum(finite.values())
    hypotheses = tuple(
        IdentityHypothesisV1(label, probability) for label, probability in selected
    )
    return hypotheses, max(0.0, total_probability - selected_probability)


class ReferenceMultiTargetTracker:
    """Constant-velocity Kalman reference implementing :class:`TrackerAdapter`."""

    adapter_name = "eventtrack-reference-multi-target-v1"

    def __init__(self, config: ReferenceTrackerConfig | None = None) -> None:
        self._config = config or ReferenceTrackerConfig()
        self._initialized = False
        self._finalized = False
        self._sequence_id = ""
        self._coordinate_frame: str | None = None
        self._current_frame_id = ""
        self._current_time = 0.0
        self._tracks: dict[str, _Track] = {}
        self._next_track_number = 1
        self._processed_cache_ids: set[str] = set()
        self._processed_frame_identities: dict[tuple[str, str, str, float], str] = {}
        self._uncommitted_cache_ids: set[str] = set()
        self._commit_log = AppendOnlyCommitLog()
        self._committed_predictions: list[TrackingPredictionV1] = []
        self._matched_since_commit: set[str] = set()
        self._born_since_commit: set[str] = set()
        self._dirty = False

    @property
    def config(self) -> ReferenceTrackerConfig:
        return self._config

    @property
    def current_time(self) -> float:
        self._ensure_initialized()
        return self._current_time

    @property
    def commit_records(self) -> tuple[CommitRecord, ...]:
        return self._commit_log.records

    @property
    def active_track_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._tracks))

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("tracker must be reset before use")

    def _ensure_active(self) -> None:
        self._ensure_initialized()
        if self._finalized:
            raise RuntimeError("tracker has been finalized")

    def reset(self, *, sequence_id: str, initial_time: float) -> None:
        if self._initialized:
            raise RuntimeError(
                "tracker instances are single-sequence; create a new instance "
                "instead of erasing committed history"
            )
        if not isinstance(sequence_id, str) or not sequence_id.strip():
            raise ValueError("sequence_id must be a non-empty string")
        initial_time = _finite(initial_time, "initial_time")
        self._initialized = True
        self._finalized = False
        self._sequence_id = sequence_id
        self._coordinate_frame = None
        self._current_frame_id = ""
        self._current_time = initial_time
        self._tracks = {}
        self._next_track_number = 1
        self._processed_cache_ids = set()
        self._processed_frame_identities = {}
        self._uncommitted_cache_ids = set()
        self._commit_log = AppendOnlyCommitLog()
        self._committed_predictions = []
        self._matched_since_commit = set()
        self._born_since_commit = set()
        self._dirty = False

    def _new_track_id(self) -> str:
        result = f"{self._sequence_id}:{self._next_track_number:06d}"
        self._next_track_number += 1
        return result

    def _advance_tracks(self, decision_time: float) -> tuple[str, ...]:
        delta = decision_time - self._current_time
        if delta < 0.0:
            raise ValueError("decision_time cannot move backwards")
        if delta == 0.0:
            return ()
        for track in self._tracks.values():
            track.mean, track.covariance = _predict_state(
                track.mean,
                track.covariance,
                decision_time - track.state_time,
                self._config.process_noise_per_second,
            )
            track.state_time = decision_time
            track.existence_probability *= (
                self._config.survival_probability_per_second**delta
            )
        self._current_time = decision_time
        if self._tracks:
            self._dirty = True
        return self._prune()

    def _prune(self) -> tuple[str, ...]:
        removed = tuple(
            sorted(
                track_id
                for track_id, track in self._tracks.items()
                if track.existence_probability
                < self._config.prune_existence_threshold
                or self._current_time - track.last_update_time
                > self._config.max_age_seconds
            )
        )
        for track_id in removed:
            del self._tracks[track_id]
        if removed:
            self._dirty = True
        return removed

    def _prediction(self, track: _Track, *, committed: bool) -> TrackingPredictionV1:
        # Keep the deterministic fallback path-safe for PredictionArchiveV1.
        # Big-endian IEEE-754 bytes avoid locale/precision and host-endian drift.
        fallback_frame_id = f"time-{struct.pack('>d', self._current_time).hex()}"
        frame_id = self._current_frame_id or fallback_frame_id
        return TrackingPredictionV1(
            sequence_id=self._sequence_id,
            frame_id=frame_id,
            track_id=track.track_id,
            class_label=track.class_label,
            event_time=track.last_event_time,
            arrival_time=track.last_arrival_time,
            decision_time=self._current_time,
            mean=track.mean,
            covariance=track.covariance,
            existence_probability=track.existence_probability,
            identity_hypotheses=track.identity_hypotheses,
            other_identity_probability=track.other_identity_probability,
            lineage=Lineage(tuple(sorted(track.factor_ids)), True),
            committed=committed,
        )

    def _snapshot(self, *, committed: bool) -> tuple[TrackingPredictionV1, ...]:
        return tuple(
            self._prediction(self._tracks[track_id], committed=committed)
            for track_id in sorted(self._tracks)
        )

    def advance(self, decision_time: float) -> tuple[TrackingPredictionV1, ...]:
        self._ensure_active()
        decision_time = _finite(decision_time, "decision_time")
        self._advance_tracks(decision_time)
        return self._snapshot(committed=False)

    def _association_costs(
        self,
        tracks: list[_Track],
        states: np.ndarray,
        covariances: np.ndarray,
        class_labels: tuple[str, ...],
        scores: np.ndarray,
    ) -> np.ndarray:
        costs = np.full((len(tracks), len(states)), np.inf, dtype=np.float64)
        threshold = chi_square_quantile(3, self._config.gate_probability)
        for track_index, track in enumerate(tracks):
            track_center = track.mean[_CENTER_INDICES]
            track_covariance = track.covariance[np.ix_(_CENTER_INDICES, _CENTER_INDICES)]
            for detection_index, state in enumerate(states):
                if track.class_label != class_labels[detection_index]:
                    continue
                detection_covariance = covariances[detection_index][
                    np.ix_(_CENTER_INDICES, _CENTER_INDICES)
                ]
                innovation = state[_CENTER_INDICES] - track_center
                innovation_covariance = track_covariance + detection_covariance
                distance = float(
                    innovation
                    @ np.linalg.solve(innovation_covariance, innovation)
                )
                if distance > threshold:
                    continue
                costs[track_index, detection_index] = gaussian_nll(
                    innovation, innovation_covariance
                ) - np.log(max(float(scores[detection_index]), 1e-12))
        return costs

    def _identity_mixture(
        self,
        tracks: list[_Track],
        costs: np.ndarray,
        detection_index: int,
    ) -> tuple[tuple[IdentityHypothesisV1, ...], float]:
        candidates = np.flatnonzero(np.isfinite(costs[:, detection_index]))
        if candidates.size == 0:  # pragma: no cover - only called for a match.
            raise ValueError("matched detection has no finite identity candidate")
        candidate_costs = costs[candidates, detection_index]
        weights = np.exp(-(candidate_costs - np.min(candidate_costs)))
        weights /= np.sum(weights)
        probabilities: dict[str, float] = {}
        other_probability = 0.0
        for candidate, weight in zip(candidates.tolist(), weights.tolist()):
            for hypothesis in tracks[candidate].identity_hypotheses:
                probabilities[hypothesis.identity_id] = (
                    probabilities.get(hypothesis.identity_id, 0.0)
                    + weight * hypothesis.probability
                )
            other_probability += (
                weight * tracks[candidate].other_identity_probability
            )
        hypotheses, omitted_known_probability = _top_hypotheses(
            probabilities, self._config.top_h
        )
        return hypotheses, other_probability + omitted_known_probability

    def ingest(
        self, detections: DetectionCacheV1, *, arrival_time: float
    ) -> TrackerIngestResult:
        self._ensure_active()
        if not isinstance(detections, DetectionCacheV1):
            raise TypeError("detections must be DetectionCacheV1")
        arrival_time = _finite(arrival_time, "arrival_time")
        cache_sha256 = detections.digest()
        if cache_sha256 in self._processed_cache_ids:
            return TrackerIngestResult(
                TrackerIngestStatus.DUPLICATE_CACHE, cache_sha256
            )
        frame_identity = (
            detections.sequence_id,
            detections.frame_id,
            detections.agent_id,
            detections.event_time,
        )
        if frame_identity in self._processed_frame_identities:
            return TrackerIngestResult(
                TrackerIngestStatus.CONFLICTING_CACHE_IDENTITY, cache_sha256
            )
        if detections.sequence_id != self._sequence_id:
            return TrackerIngestResult(
                TrackerIngestStatus.SEQUENCE_MISMATCH, cache_sha256
            )
        if (
            self._coordinate_frame is not None
            and detections.coordinate_frame != self._coordinate_frame
        ):
            return TrackerIngestResult(
                TrackerIngestStatus.COORDINATE_FRAME_MISMATCH, cache_sha256
            )
        if detections.event_time > arrival_time + 1e-12:
            return TrackerIngestResult(
                TrackerIngestStatus.FUTURE_EVENT, cache_sha256
            )
        if arrival_time > self._current_time + 1e-12:
            return TrackerIngestResult(
                TrackerIngestStatus.FUTURE_ARRIVAL, cache_sha256
            )

        if self._coordinate_frame is None:
            self._coordinate_frame = detections.coordinate_frame
        if detections.agent_id == self._config.reference_agent_id:
            if (
                self._current_frame_id
                and self._uncommitted_cache_ids
                and detections.frame_id != self._current_frame_id
            ):
                raise ValueError(
                    "multiple reference-agent frames require an intervening commit"
                )
            self._current_frame_id = detections.frame_id
        order = _canonical_detection_order(detections)
        states_at_event = detections.state_vectors()[list(order)]
        covariances_at_event = detections.covariances[list(order)]
        scores = detections.scores[list(order)]
        class_labels = tuple(detections.class_labels[index] for index in order)
        states: list[np.ndarray] = []
        covariances: list[np.ndarray] = []
        for mean, covariance in zip(states_at_event, covariances_at_event):
            predicted_mean, predicted_covariance = _predict_state(
                mean,
                covariance,
                self._current_time - detections.event_time,
                self._config.process_noise_per_second,
            )
            states.append(predicted_mean)
            covariances.append(predicted_covariance)
        state_array = np.asarray(states, dtype=np.float64).reshape((-1, _STATE_DIMENSION))
        covariance_array = np.asarray(covariances, dtype=np.float64).reshape(
            (-1, _STATE_DIMENSION, _STATE_DIMENSION)
        )

        tracks = [self._tracks[track_id] for track_id in sorted(self._tracks)]
        costs = self._association_costs(
            tracks, state_array, covariance_array, class_labels, scores
        )
        assignment = solve_one_to_one(
            costs,
            unmatched_left_cost=self._config.unmatched_track_cost,
            unmatched_right_cost=self._config.unmatched_detection_cost,
        )
        matched_ids: list[str] = []
        for track_index, detection_index in assignment.pairs:
            track = tracks[track_index]
            factor_id = f"detection-cache:{cache_sha256}:{order[detection_index]}"
            identity_hypotheses, other_identity_probability = self._identity_mixture(
                tracks, costs, detection_index
            )
            track.mean, track.covariance = _kalman_update(
                track.mean,
                track.covariance,
                state_array[detection_index],
                covariance_array[detection_index],
            )
            track.last_update_time = self._current_time
            track.last_event_time = detections.event_time
            track.last_arrival_time = arrival_time
            track.existence_probability = 1.0 - (
                1.0 - track.existence_probability
            ) * (1.0 - float(scores[detection_index]))
            track.identity_hypotheses = identity_hypotheses
            track.other_identity_probability = other_identity_probability
            track.factor_ids.add(factor_id)
            matched_ids.append(track.track_id)
            self._matched_since_commit.add(track.track_id)

        born_ids: list[str] = []
        for detection_index in assignment.unmatched_right:
            score = float(scores[detection_index])
            if score < self._config.birth_score_threshold:
                continue
            track_id = self._new_track_id()
            factor_id = f"detection-cache:{cache_sha256}:{order[detection_index]}"
            self._tracks[track_id] = _Track(
                track_id=track_id,
                class_label=class_labels[detection_index],
                mean=state_array[detection_index].copy(),
                covariance=covariance_array[detection_index].copy(),
                state_time=self._current_time,
                last_update_time=self._current_time,
                last_event_time=detections.event_time,
                last_arrival_time=arrival_time,
                existence_probability=score,
                identity_hypotheses=(IdentityHypothesisV1(track_id, 1.0),),
                other_identity_probability=0.0,
                factor_ids={factor_id},
            )
            born_ids.append(track_id)
            self._born_since_commit.add(track_id)

        self._processed_cache_ids.add(cache_sha256)
        self._processed_frame_identities[frame_identity] = cache_sha256
        self._uncommitted_cache_ids.add(cache_sha256)
        self._dirty = True
        pruned = self._prune()
        return TrackerIngestResult(
            TrackerIngestStatus.APPLIED,
            cache_sha256,
            matched_track_ids=tuple(sorted(matched_ids)),
            born_track_ids=tuple(born_ids),
            pruned_track_ids=pruned,
        )

    def ingest_batch(
        self,
        batch: Iterable[tuple[DetectionCacheV1, float]],
    ) -> tuple[TrackerIngestResult, ...]:
        """Atomically ingest a canonicalized set of simultaneously available caches.

        The batch is fully materialized and preflighted before association starts.
        Unlike the idempotent single-cache :meth:`ingest`, this strict transaction
        rejects existing or within-batch duplicates together with every other
        non-applicable item; callers should remove retransmissions before batching.
        Association runs on a shadow copy and its ingest-mutated state is adopted
        only after every cache applies, so unexpected numerical failures cannot
        leave a partially ingested batch.
        """

        self._ensure_active()
        try:
            items = tuple(batch)
        except TypeError as error:
            raise TypeError("batch must be an iterable of (cache, arrival_time)") from error
        if not items:
            return ()

        prepared: list[tuple[float, int, str, str, str, DetectionCacheV1]] = []
        seen_cache_ids: set[str] = set()
        seen_frame_identities: dict[tuple[str, str, str, float], str] = {}
        reference_frame_ids: set[str] = set()
        batch_coordinate_frame: str | None = self._coordinate_frame

        for item_index, item in enumerate(items):
            try:
                detections, raw_arrival_time = item
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "batch items must be (DetectionCacheV1, arrival_time) pairs"
                ) from error
            if not isinstance(detections, DetectionCacheV1):
                raise TypeError(
                    f"batch item {item_index} detections must be DetectionCacheV1"
                )
            arrival_time = _finite(
                raw_arrival_time, f"batch[{item_index}].arrival_time"
            )
            cache_sha256 = detections.digest()
            if (
                cache_sha256 in self._processed_cache_ids
                or cache_sha256 in seen_cache_ids
            ):
                raise ValueError(
                    f"batch item {item_index} rejected before mutation: "
                    f"{TrackerIngestStatus.DUPLICATE_CACHE.value}"
                )

            frame_identity = (
                detections.sequence_id,
                detections.frame_id,
                detections.agent_id,
                detections.event_time,
            )
            previous_digest = self._processed_frame_identities.get(frame_identity)
            if previous_digest is None:
                previous_digest = seen_frame_identities.get(frame_identity)
            if previous_digest is not None:
                raise ValueError(
                    f"batch item {item_index} rejected before mutation: "
                    f"{TrackerIngestStatus.CONFLICTING_CACHE_IDENTITY.value}"
                )
            if detections.sequence_id != self._sequence_id:
                raise ValueError(
                    f"batch item {item_index} rejected before mutation: "
                    f"{TrackerIngestStatus.SEQUENCE_MISMATCH.value}"
                )
            if batch_coordinate_frame is None:
                batch_coordinate_frame = detections.coordinate_frame
            elif detections.coordinate_frame != batch_coordinate_frame:
                raise ValueError(
                    f"batch item {item_index} rejected before mutation: "
                    f"{TrackerIngestStatus.COORDINATE_FRAME_MISMATCH.value}"
                )
            if detections.event_time > arrival_time + 1e-12:
                raise ValueError(
                    f"batch item {item_index} rejected before mutation: "
                    f"{TrackerIngestStatus.FUTURE_EVENT.value}"
                )
            if arrival_time > self._current_time + 1e-12:
                raise ValueError(
                    f"batch item {item_index} rejected before mutation: "
                    f"{TrackerIngestStatus.FUTURE_ARRIVAL.value}"
                )
            if detections.agent_id == self._config.reference_agent_id:
                reference_frame_ids.add(detections.frame_id)

            seen_cache_ids.add(cache_sha256)
            seen_frame_identities[frame_identity] = cache_sha256
            prepared.append(
                (
                    arrival_time,
                    0
                    if detections.agent_id == self._config.reference_agent_id
                    else 1,
                    detections.agent_id,
                    detections.frame_id,
                    cache_sha256,
                    detections,
                )
            )

        if len(reference_frame_ids) > 1:
            raise ValueError(
                "batch rejected before mutation: multiple reference-agent frames "
                "require an intervening commit"
            )
        if (
            self._current_frame_id
            and self._uncommitted_cache_ids
            and reference_frame_ids
            and reference_frame_ids != {self._current_frame_id}
        ):
            raise ValueError(
                "batch rejected before mutation: multiple reference-agent frames "
                "require an intervening commit"
            )

        prepared.sort(key=lambda item: item[:5])
        shadow = copy.deepcopy(self)
        results: list[TrackerIngestResult] = []
        for arrival_time, _, _, _, _, detections in prepared:
            result = shadow.ingest(detections, arrival_time=arrival_time)
            if not result.applied:  # pragma: no cover - guarded by preflight.
                raise RuntimeError(
                    "batch preflight/application mismatch: " f"{result.status.value}"
                )
            results.append(result)

        for attribute in (
            "_coordinate_frame",
            "_current_frame_id",
            "_tracks",
            "_next_track_number",
            "_processed_cache_ids",
            "_processed_frame_identities",
            "_uncommitted_cache_ids",
            "_matched_since_commit",
            "_born_since_commit",
            "_dirty",
        ):
            setattr(self, attribute, getattr(shadow, attribute))
        return tuple(results)

    def commit(self, decision_time: float) -> tuple[TrackingPredictionV1, ...]:
        self._ensure_active()
        decision_time = _finite(decision_time, "decision_time")
        self._advance_tracks(decision_time)
        if self._uncommitted_cache_ids:
            for track_id, track in self._tracks.items():
                if (
                    track_id not in self._matched_since_commit
                    and track_id not in self._born_since_commit
                ):
                    track.existence_probability *= (
                        self._config.missed_detection_factor
                    )
            self._prune()
        predictions = tuple(
            replace(prediction, committed=True)
            for prediction in self._snapshot(committed=False)
        )
        self._commit_log.commit(
            decision_time=decision_time,
            snapshot=[prediction.to_primitive() for prediction in predictions],
            message_ids=tuple(sorted(self._uncommitted_cache_ids)),
        )
        self._committed_predictions.extend(predictions)
        self._uncommitted_cache_ids.clear()
        self._matched_since_commit.clear()
        self._born_since_commit.clear()
        self._current_frame_id = ""
        self._dirty = False
        return predictions

    def finalize(self) -> tuple[TrackingPredictionV1, ...]:
        self._ensure_active()
        if self._dirty:
            raise RuntimeError("tracker has uncommitted state; commit before finalize")
        self._finalized = True
        if not self._commit_log.verify():  # pragma: no cover - defensive invariant.
            raise RuntimeError("tracker commit chain failed verification")
        return tuple(self._committed_predictions)


__all__ = [
    "ReferenceMultiTargetTracker",
    "ReferenceTrackerConfig",
    "TrackerAdapter",
    "TrackerIngestResult",
    "TrackerIngestStatus",
]
