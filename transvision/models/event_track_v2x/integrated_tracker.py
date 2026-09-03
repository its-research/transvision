"""Unified deterministic EventTrack-V2X reference tracker.

The implementation connects the public detector-locked tracker lifecycle with
fixed-lag point-time replay, source-time moment matching, correlation-safe CI,
lineage accounting, and the analytic VoI scorer.  It is intentionally a
reference implementation: learned association/VoI and exact non-Gaussian
mixture replay are explicit unsupported capabilities.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable

import numpy as np

from .clock import AffineClockMap, MappedEventTimes, conservative_no_future_gate
from .contracts import DetectionCacheV1, TrackingPredictionV1
from .fusion import covariance_intersection
from .replay import (
    FixedLagReplay,
    IngestStatus,
    LinearGaussianDynamics,
    OutOfWindowPolicy,
)
from .schema import (
    CorrelatedTrackBelief,
    IndependentIncrement,
    Lineage,
    LocalTimestamps,
    ReceivedMessage,
    WireMessage,
)
from .source_time import (
    GaussianTimeComponent,
    SourceTimeMixture,
    SourceTimeMode,
    propagate_source_time,
)
from .tracker import (
    ReferenceMultiTargetTracker,
    ReferenceTrackerConfig,
    TrackerIngestResult,
    _Track,
    _top_hypotheses,
)
from .voi import (
    DEFAULT_VOI_SCORER_CONFIG_V1,
    CounterfactualVoiScoreV1,
    FrozenOnTimeProbabilityV1,
    VoiScorerConfigV1,
    score_counterfactual_voi_v1,
)


class EventTrackReferenceCapabilityError(RuntimeError):
    """Raised when a caller requires a capability the reference path lacks."""


@dataclass(frozen=True, slots=True)
class EventTrackReferenceCapabilitiesV1:
    """Fail-closed capability declaration for the unified reference path."""

    tracker_adapter_lifecycle: bool = field(init=False, default=True)
    multi_target_birth_death: bool = field(init=False, default=True)
    top_h_identity: bool = field(init=False, default=True)
    point_time_fixed_lag_oosm: bool = field(init=False, default=True)
    source_time_mixture_moment_matching: bool = field(init=False, default=True)
    lineage_duplicate_guard: bool = field(init=False, default=True)
    correlated_belief_ci: bool = field(init=False, default=True)
    analytic_counterfactual_voi: bool = field(init=False, default=True)
    exact_non_gaussian_mixture_replay: bool = field(init=False, default=False)
    learned_association: bool = field(init=False, default=False)
    learned_voi: bool = field(init=False, default=False)
    formal_experiment_or_sota_evidence: bool = field(init=False, default=False)

    def to_dict(self) -> dict[str, bool]:
        return {
            name: bool(getattr(self, name))
            for name in self.__dataclass_fields__
        }

    def require(self, *names: str) -> None:
        available = self.to_dict()
        unknown = sorted(set(names) - set(available))
        if unknown:
            raise EventTrackReferenceCapabilityError(
                f"unknown EventTrack reference capabilities: {unknown}"
            )
        unsupported = sorted(name for name in names if not available[name])
        if unsupported:
            raise EventTrackReferenceCapabilityError(
                f"unsupported EventTrack reference capabilities: {unsupported}"
            )


EVENTTRACK_REFERENCE_CAPABILITIES_V1 = EventTrackReferenceCapabilitiesV1()


@dataclass(frozen=True, slots=True)
class EventTrackReferenceConfig:
    tracker: ReferenceTrackerConfig = field(default_factory=ReferenceTrackerConfig)
    fixed_lag_seconds: float = 1.0
    source_time_mode: SourceTimeMode = SourceTimeMode.MIXTURE
    clock_gate_confidence: float = 0.99
    ci_objective: str = "logdet"
    out_of_window_policy: OutOfWindowPolicy = OutOfWindowPolicy.DROP

    def __post_init__(self) -> None:
        if not isinstance(self.tracker, ReferenceTrackerConfig):
            raise TypeError("tracker must be ReferenceTrackerConfig")
        lag = float(self.fixed_lag_seconds)
        if not np.isfinite(lag) or lag <= 0.0:
            raise ValueError("fixed_lag_seconds must be finite and positive")
        confidence = float(self.clock_gate_confidence)
        if not 0.5 < confidence < 1.0:
            raise ValueError("clock_gate_confidence must be in (0.5, 1)")
        mode = SourceTimeMode(self.source_time_mode)
        if mode is SourceTimeMode.ORACLE:
            raise ValueError("the reference tracker cannot use oracle source time")
        if self.ci_objective not in {"logdet", "trace"}:
            raise ValueError("ci_objective must be 'logdet' or 'trace'")
        object.__setattr__(self, "fixed_lag_seconds", lag)
        object.__setattr__(self, "clock_gate_confidence", confidence)
        object.__setattr__(self, "source_time_mode", mode)
        object.__setattr__(
            self,
            "out_of_window_policy",
            OutOfWindowPolicy(self.out_of_window_policy),
        )


class CooperativeIngestStatus(str, Enum):
    APPLIED_POINT_REPLAY = "applied_point_replay"
    APPLIED_MIXTURE_MOMENT = "applied_mixture_moment"
    APPLIED_CORRELATED_CI = "applied_correlated_ci"
    DUPLICATE = "duplicate"
    FUTURE_ARRIVAL = "future_arrival"
    FUTURE_INFORMATION = "future_information"
    MISSED_DEADLINE = "missed_deadline"
    EXPIRED_TTL = "expired_ttl"
    WRONG_TARGET = "wrong_target"
    COORDINATE_FRAME_MISMATCH = "coordinate_frame_mismatch"
    OUT_OF_WINDOW = "out_of_window"
    LINEAGE_OVERLAP = "lineage_overlap"
    UNSUPPORTED_PAYLOAD_SHAPE = "unsupported_payload_shape"


_APPLIED_COOPERATIVE_STATUSES = frozenset(
    {
        CooperativeIngestStatus.APPLIED_POINT_REPLAY,
        CooperativeIngestStatus.APPLIED_MIXTURE_MOMENT,
        CooperativeIngestStatus.APPLIED_CORRELATED_CI,
    }
)


@dataclass(frozen=True, slots=True)
class CooperativeIngestResult:
    status: CooperativeIngestStatus
    message_id: str
    target_track_id: str

    @property
    def applied(self) -> bool:
        return self.status in _APPLIED_COOPERATIVE_STATUSES


@dataclass(slots=True)
class _IntegratedMetadata:
    ancestor_message_ids: set[str]
    lineage_complete: bool


def _reference_dynamics(process_noise_per_second: float) -> LinearGaussianDynamics:
    def transition(delta: float) -> np.ndarray:
        delta = float(delta)
        if not np.isfinite(delta) or delta < 0.0:
            raise ValueError("delta must be finite and non-negative")
        result = np.eye(9, dtype=np.float64)
        result[0, 7] = delta
        result[1, 8] = delta
        return result

    def process_covariance(delta: float) -> np.ndarray:
        delta = float(delta)
        if not np.isfinite(delta) or delta < 0.0:
            raise ValueError("delta must be finite and non-negative")
        return np.eye(9, dtype=np.float64) * process_noise_per_second * delta

    return LinearGaussianDynamics(9, transition, process_covariance)


_IDENTITY_CLOCK = AffineClockMap(1.0, 0.0, np.zeros((2, 2), dtype=np.float64))


class EventTrackReferenceTracker(ReferenceMultiTargetTracker):
    """One deterministic TrackerAdapter-compatible EventTrack reference path.

    Point-time independent increments share the same per-track replay ledger as
    detector-cache updates.  Uncertain timestamps are supported only for full
    nine-dimensional state increments: their source-time mixture is propagated
    and moment-matched at the current decision before insertion.  A correlated
    posterior is propagated and fused by CI, then becomes a new replay boundary;
    older information arriving after that boundary is rejected rather than
    silently double counted.
    """

    adapter_name = "eventtrack-unified-reference-v1"
    capabilities = EVENTTRACK_REFERENCE_CAPABILITIES_V1

    def __init__(self, config: EventTrackReferenceConfig | None = None) -> None:
        self._eventtrack_config = config or EventTrackReferenceConfig()
        super().__init__(self._eventtrack_config.tracker)
        self._replays: dict[str, FixedLagReplay] = {}
        self._integrated_metadata: dict[str, _IntegratedMetadata] = {}
        self._cooperative_message_ids: set[str] = set()
        self._uncommitted_cooperative_message_ids: set[str] = set()
        self._dynamics = _reference_dynamics(
            self._eventtrack_config.tracker.process_noise_per_second
        )

    @property
    def eventtrack_config(self) -> EventTrackReferenceConfig:
        return self._eventtrack_config

    def reset(self, *, sequence_id: str, initial_time: float) -> None:
        super().reset(sequence_id=sequence_id, initial_time=initial_time)
        self._replays = {}
        self._integrated_metadata = {}
        self._cooperative_message_ids = set()
        self._uncommitted_cooperative_message_ids = set()

    def _adopt(self, shadow: "EventTrackReferenceTracker") -> None:
        self.__dict__.clear()
        self.__dict__.update(shadow.__dict__)

    @staticmethod
    def _cache_factor_index(track: _Track, cache_sha256: str) -> int:
        prefix = f"detection-cache:{cache_sha256}:"
        matches = sorted(
            factor_id for factor_id in track.factor_ids if factor_id.startswith(prefix)
        )
        if len(matches) != 1:
            raise RuntimeError("detection association did not produce one cache factor")
        suffix = matches[0][len(prefix) :]
        try:
            return int(suffix)
        except ValueError as error:  # pragma: no cover - internal factor contract.
            raise RuntimeError("detection factor index is not an integer") from error

    def _new_replay_from_detection(
        self,
        track_id: str,
        detections: DetectionCacheV1,
        cache_sha256: str,
    ) -> None:
        track = self._tracks[track_id]
        index = self._cache_factor_index(track, cache_sha256)
        factor_id = f"detection-cache:{cache_sha256}:{index}"
        replay = FixedLagReplay(
            track_id=track_id,
            initial_time=detections.event_time,
            initial_mean=detections.state_vectors()[index],
            initial_covariance=detections.covariances[index],
            dynamics=self._dynamics,
            lag=self._eventtrack_config.fixed_lag_seconds,
            absorbed_factor_ids=(factor_id,),
            out_of_window_policy=self._eventtrack_config.out_of_window_policy,
        )
        replay.advance_to(self._current_time)
        self._replays[track_id] = replay
        self._integrated_metadata[track_id] = _IntegratedMetadata(set(), True)
        self._synchronize_from_replay(track_id)

    def _detection_increment(
        self,
        track_id: str,
        detections: DetectionCacheV1,
        cache_sha256: str,
        arrival_time: float,
    ) -> ReceivedMessage:
        track = self._tracks[track_id]
        index = self._cache_factor_index(track, cache_sha256)
        factor_id = f"detection-cache:{cache_sha256}:{index}"
        timestamp = detections.event_time
        return ReceivedMessage(
            message=WireMessage(
                message_id=f"cache-factor:{cache_sha256}:{index}",
                source=detections.agent_id,
                sequence=index,
                timestamps=LocalTimestamps(timestamp, timestamp, timestamp, timestamp),
                deadline=self._current_time,
                coordinate_frame=detections.coordinate_frame,
                ttl=max(
                    1.0,
                    self._current_time
                    - timestamp
                    + self._eventtrack_config.fixed_lag_seconds,
                ),
                payload=IndependentIncrement(
                    target_id=track_id,
                    measurement=detections.state_vectors()[index],
                    measurement_matrix=np.eye(9, dtype=np.float64),
                    measurement_covariance=detections.covariances[index],
                    lineage=Lineage((factor_id,), True),
                ),
            ),
            received_at=arrival_time,
        )

    def _synchronize_from_replay(self, track_id: str) -> None:
        snapshot = self._replays[track_id].snapshot()
        track = self._tracks[track_id]
        track.mean = np.asarray(snapshot.mean, dtype=np.float64).copy()
        track.covariance = np.asarray(snapshot.covariance, dtype=np.float64).copy()
        track.state_time = snapshot.time

    def _integrate_detection_replays(
        self,
        detections: DetectionCacheV1,
        arrival_time: float,
        result: TrackerIngestResult,
    ) -> None:
        cache_sha256 = result.cache_sha256
        for track_id in result.born_track_ids:
            self._new_replay_from_detection(track_id, detections, cache_sha256)
        for track_id in result.matched_track_ids:
            replay = self._replays.get(track_id)
            if replay is None:
                raise RuntimeError("matched track has no integrated replay ledger")
            received = self._detection_increment(
                track_id,
                detections,
                cache_sha256,
                arrival_time,
            )
            replay_result = replay.ingest_increment(
                received,
                clock_map=_IDENTITY_CLOCK,
                decision_time=self._current_time,
                confidence=self._eventtrack_config.clock_gate_confidence,
            )
            if replay_result.status is not IngestStatus.APPLIED:
                raise ValueError(
                    "detector factor is outside the integrated replay contract: "
                    f"{replay_result.status.value}"
                )
            self._synchronize_from_replay(track_id)
        for track_id in result.pruned_track_ids:
            self._replays.pop(track_id, None)
            self._integrated_metadata.pop(track_id, None)

    def ingest(
        self,
        detections: DetectionCacheV1,
        *,
        arrival_time: float,
    ) -> TrackerIngestResult:
        shadow = copy.deepcopy(self)
        result = ReferenceMultiTargetTracker.ingest(
            shadow,
            detections,
            arrival_time=arrival_time,
        )
        if not result.applied:
            return result
        shadow._integrate_detection_replays(detections, float(arrival_time), result)
        self._adopt(shadow)
        return result

    def ingest_batch(
        self,
        batch: Iterable[tuple[DetectionCacheV1, float]],
    ) -> tuple[TrackerIngestResult, ...]:
        """Atomically ingest one canonical simultaneous/multi-agent cache batch."""

        self._ensure_active()
        try:
            items = tuple(batch)
        except TypeError as error:
            raise TypeError("batch must be an iterable of (cache, arrival_time)") from error
        if not items:
            return ()
        prepared: list[tuple[float, int, str, str, str, DetectionCacheV1]] = []
        for index, item in enumerate(items):
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("batch items must be (DetectionCacheV1, arrival_time) pairs")
            cache, raw_arrival = item
            if not isinstance(cache, DetectionCacheV1):
                raise TypeError(f"batch item {index} detections must be DetectionCacheV1")
            arrival = float(raw_arrival)
            if not np.isfinite(arrival):
                raise ValueError(f"batch[{index}].arrival_time must be finite")
            prepared.append(
                (
                    arrival,
                    0 if cache.agent_id == self.config.reference_agent_id else 1,
                    cache.agent_id,
                    cache.frame_id,
                    cache.digest(),
                    cache,
                )
            )
        prepared.sort(key=lambda item: item[:5])
        shadow = copy.deepcopy(self)
        results: list[TrackerIngestResult] = []
        for arrival, _, _, _, _, cache in prepared:
            result = shadow.ingest(cache, arrival_time=arrival)
            if not result.applied:
                raise ValueError(
                    "batch rejected before adoption: " f"{result.status.value}"
                )
            results.append(result)
        self._adopt(shadow)
        return tuple(results)

    def _advance_tracks(self, decision_time: float) -> tuple[str, ...]:
        removed = super()._advance_tracks(decision_time)
        for track_id in removed:
            self._replays.pop(track_id, None)
            self._integrated_metadata.pop(track_id, None)
        for track_id in sorted(set(self._replays).intersection(self._tracks)):
            self._replays[track_id].advance_to(decision_time)
            self._synchronize_from_replay(track_id)
        return removed

    def _prediction(self, track: _Track, *, committed: bool) -> TrackingPredictionV1:
        prediction = super()._prediction(track, committed=committed)
        metadata = self._integrated_metadata.get(track.track_id)
        if metadata is None:
            return prediction
        return replace(
            prediction,
            lineage=Lineage(
                tuple(sorted(track.factor_ids)),
                metadata.lineage_complete,
                tuple(sorted(metadata.ancestor_message_ids)),
            ),
        )

    def commit(self, decision_time: float) -> tuple[TrackingPredictionV1, ...]:
        """Commit caches and cooperative messages without inventing cache misses.

        Only an ingested detector cache activates the detector-miss existence
        update.  Cooperative message IDs are nevertheless bound into the same
        append-only commit record.
        """

        self._ensure_active()
        decision_time = float(decision_time)
        if not np.isfinite(decision_time):
            raise ValueError("decision_time must be finite")
        self._advance_tracks(decision_time)
        if self._uncommitted_cache_ids:
            for track_id, track in self._tracks.items():
                if (
                    track_id not in self._matched_since_commit
                    and track_id not in self._born_since_commit
                ):
                    track.existence_probability *= (
                        self.config.missed_detection_factor
                    )
            self._prune()
        predictions = tuple(
            replace(prediction, committed=True)
            for prediction in self._snapshot(committed=False)
        )
        committed_message_ids = tuple(
            sorted(
                self._uncommitted_cache_ids
                | self._uncommitted_cooperative_message_ids
            )
        )
        self._commit_log.commit(
            decision_time=decision_time,
            snapshot=[prediction.to_primitive() for prediction in predictions],
            message_ids=committed_message_ids,
        )
        self._committed_predictions.extend(predictions)
        self._uncommitted_cache_ids.clear()
        self._uncommitted_cooperative_message_ids.clear()
        self._matched_since_commit.clear()
        self._born_since_commit.clear()
        self._current_frame_id = ""
        self._dirty = False
        return predictions

    def _target_id(self, received: ReceivedMessage) -> str:
        payload = received.message.payload
        if isinstance(payload, IndependentIncrement):
            return payload.target_id
        return payload.track_id

    def _preflight_cooperative(
        self,
        received: ReceivedMessage,
        clock_map: AffineClockMap,
    ) -> tuple[CooperativeIngestResult | None, MappedEventTimes]:
        if not isinstance(received, ReceivedMessage):
            raise TypeError("received must be ReceivedMessage")
        if not isinstance(clock_map, AffineClockMap):
            raise TypeError("clock_map must be AffineClockMap")
        message = received.message
        target_id = self._target_id(received)
        if message.message_id in self._cooperative_message_ids:
            return (
                CooperativeIngestResult(
                    CooperativeIngestStatus.DUPLICATE,
                    message.message_id,
                    target_id,
                ),
                clock_map.map_message(received),
            )
        mapped = clock_map.map_message(received)
        if received.received_at > self._current_time + 1e-12:
            return (
                CooperativeIngestResult(
                    CooperativeIngestStatus.FUTURE_ARRIVAL,
                    message.message_id,
                    target_id,
                ),
                mapped,
            )
        if self._current_time > message.deadline:
            self._cooperative_message_ids.add(message.message_id)
            return (
                CooperativeIngestResult(
                    CooperativeIngestStatus.MISSED_DEADLINE,
                    message.message_id,
                    target_id,
                ),
                mapped,
            )
        if not conservative_no_future_gate(
            mapped.information_cutoff,
            self._current_time,
            confidence=self._eventtrack_config.clock_gate_confidence,
        ) or not conservative_no_future_gate(
            mapped.state_reference,
            self._current_time,
            confidence=self._eventtrack_config.clock_gate_confidence,
        ):
            return (
                CooperativeIngestResult(
                    CooperativeIngestStatus.FUTURE_INFORMATION,
                    message.message_id,
                    target_id,
                ),
                mapped,
            )
        if self._current_time > mapped.transmitted.mean + message.ttl:
            self._cooperative_message_ids.add(message.message_id)
            return (
                CooperativeIngestResult(
                    CooperativeIngestStatus.EXPIRED_TTL,
                    message.message_id,
                    target_id,
                ),
                mapped,
            )
        if target_id not in self._tracks:
            self._cooperative_message_ids.add(message.message_id)
            return (
                CooperativeIngestResult(
                    CooperativeIngestStatus.WRONG_TARGET,
                    message.message_id,
                    target_id,
                ),
                mapped,
            )
        if message.coordinate_frame != self._coordinate_frame:
            self._cooperative_message_ids.add(message.message_id)
            return (
                CooperativeIngestResult(
                    CooperativeIngestStatus.COORDINATE_FRAME_MISMATCH,
                    message.message_id,
                    target_id,
                ),
                mapped,
            )
        return None, mapped

    def _validated_source_time(
        self,
        mapped: MappedEventTimes,
        explicit: SourceTimeMixture | None,
    ) -> SourceTimeMixture:
        estimate = mapped.state_reference
        certified_upper = estimate.upper_quantile(
            self._eventtrack_config.clock_gate_confidence
        )
        if explicit is None:
            return SourceTimeMixture(
                (
                    GaussianTimeComponent(
                        1.0,
                        estimate.mean,
                        estimate.variance,
                    ),
                ),
                causal_upper_bound=certified_upper,
            )
        if not isinstance(explicit, SourceTimeMixture):
            raise TypeError("source_time_mixture must be SourceTimeMixture or None")
        if not np.isclose(explicit.mean, estimate.mean, rtol=0.0, atol=1e-12):
            raise ValueError("source-time mixture mean disagrees with the clock map")
        if not np.isclose(explicit.variance, estimate.variance, rtol=1e-10, atol=1e-12):
            raise ValueError("source-time mixture variance disagrees with the clock map")
        if explicit.causal_upper_bound + 1e-12 < certified_upper:
            raise ValueError(
                "source-time mixture bound is weaker than the clock causality gate"
            )
        if explicit.causal_upper_bound > self._current_time + 1e-12:
            raise ValueError("source-time mixture admits future information")
        return explicit

    @staticmethod
    def _is_point_source_time(source_time: SourceTimeMixture) -> bool:
        return (
            len(source_time.components) == 1
            and source_time.components[0].variance == 0.0
        )

    def _current_equivalent_increment(
        self,
        received: ReceivedMessage,
        source_time: SourceTimeMixture,
    ) -> ReceivedMessage | None:
        payload = received.message.payload
        assert isinstance(payload, IndependentIncrement)
        if (
            payload.measurement.shape != (9,)
            or payload.measurement_matrix.shape != (9, 9)
            or not np.array_equal(payload.measurement_matrix, np.eye(9))
            or payload.measurement_covariance.shape != (9, 9)
        ):
            return None
        propagated = propagate_source_time(
            payload.measurement,
            payload.measurement_covariance,
            source_time,
            decision_time=self._current_time,
            mode=self._eventtrack_config.source_time_mode,
            process_noise_per_second=(
                self._eventtrack_config.tracker.process_noise_per_second
            ),
        )
        message = received.message
        timestamp = self._current_time
        return ReceivedMessage(
            message=WireMessage(
                message_id=message.message_id,
                source=message.source,
                sequence=message.sequence,
                timestamps=LocalTimestamps(timestamp, timestamp, timestamp, timestamp),
                deadline=message.deadline,
                coordinate_frame=message.coordinate_frame,
                ttl=message.ttl,
                payload=IndependentIncrement(
                    target_id=payload.target_id,
                    measurement=propagated.mean,
                    measurement_matrix=np.eye(9, dtype=np.float64),
                    measurement_covariance=propagated.covariance,
                    lineage=payload.lineage,
                    log_normalizer=payload.log_normalizer,
                ),
            ),
            received_at=timestamp,
        )

    def _ingest_independent(
        self,
        received: ReceivedMessage,
        clock_map: AffineClockMap,
        mapped: MappedEventTimes,
        source_time_mixture: SourceTimeMixture | None,
    ) -> CooperativeIngestResult:
        payload = received.message.payload
        assert isinstance(payload, IndependentIncrement)
        track_id = payload.target_id
        replay = self._replays[track_id]
        source_time = self._validated_source_time(mapped, source_time_mixture)
        status = CooperativeIngestStatus.APPLIED_POINT_REPLAY
        replay_received = received
        replay_clock = clock_map
        if not self._is_point_source_time(source_time):
            replay_received = self._current_equivalent_increment(received, source_time)
            if replay_received is None:
                self._cooperative_message_ids.add(received.message.message_id)
                return CooperativeIngestResult(
                    CooperativeIngestStatus.UNSUPPORTED_PAYLOAD_SHAPE,
                    received.message.message_id,
                    track_id,
                )
            replay_clock = _IDENTITY_CLOCK
            status = CooperativeIngestStatus.APPLIED_MIXTURE_MOMENT
        replay_result = replay.ingest_increment(
            replay_received,
            clock_map=replay_clock,
            decision_time=self._current_time,
            confidence=self._eventtrack_config.clock_gate_confidence,
        )
        if replay_result.status is not IngestStatus.APPLIED:
            consumed = replay_result.status not in {
                IngestStatus.FUTURE_ARRIVAL,
                IngestStatus.FUTURE_INFORMATION,
            }
            if consumed:
                self._cooperative_message_ids.add(received.message.message_id)
            mapped_status = (
                CooperativeIngestStatus.LINEAGE_OVERLAP
                if replay_result.status is IngestStatus.LINEAGE_OVERLAP
                else CooperativeIngestStatus.OUT_OF_WINDOW
                if replay_result.status
                in {
                    IngestStatus.OUT_OF_WINDOW_DROPPED,
                    IngestStatus.OUT_OF_WINDOW_REJECTED,
                }
                else CooperativeIngestStatus.DUPLICATE
                if replay_result.status is IngestStatus.DUPLICATE
                else CooperativeIngestStatus.FUTURE_ARRIVAL
                if replay_result.status is IngestStatus.FUTURE_ARRIVAL
                else CooperativeIngestStatus.FUTURE_INFORMATION
            )
            return CooperativeIngestResult(
                mapped_status,
                received.message.message_id,
                track_id,
            )
        self._cooperative_message_ids.add(received.message.message_id)
        track = self._tracks[track_id]
        track.factor_ids.update(payload.lineage.factor_ids)
        metadata = self._integrated_metadata[track_id]
        metadata.ancestor_message_ids.update(payload.lineage.ancestor_message_ids)
        metadata.ancestor_message_ids.add(received.message.message_id)
        metadata.lineage_complete = metadata.lineage_complete and payload.lineage.complete
        self._uncommitted_cooperative_message_ids.add(received.message.message_id)
        self._matched_since_commit.add(track_id)
        track.last_event_time = source_time.mean
        track.last_arrival_time = received.received_at
        self._synchronize_from_replay(track_id)
        self._dirty = True
        return CooperativeIngestResult(
            status,
            received.message.message_id,
            track_id,
        )

    def _fuse_identities(
        self,
        track: _Track,
        belief: CorrelatedTrackBelief,
        weight_local: float,
    ) -> tuple[tuple[object, ...], float]:
        probabilities: dict[str, float] = {}
        for hypothesis in track.identity_hypotheses:
            probabilities[hypothesis.identity_id] = (
                probabilities.get(hypothesis.identity_id, 0.0)
                + weight_local * hypothesis.probability
            )
        for identity_id, probability in belief.identity_probabilities:
            probabilities[identity_id] = (
                probabilities.get(identity_id, 0.0)
                + (1.0 - weight_local) * probability
            )
        hypotheses, omitted = _top_hypotheses(
            probabilities,
            self._eventtrack_config.tracker.top_h,
        )
        other = weight_local * track.other_identity_probability + omitted
        return hypotheses, other

    def _ingest_correlated(
        self,
        received: ReceivedMessage,
        mapped: MappedEventTimes,
        source_time_mixture: SourceTimeMixture | None,
    ) -> CooperativeIngestResult:
        message = received.message
        payload = message.payload
        assert isinstance(payload, CorrelatedTrackBelief)
        track_id = payload.track_id
        if payload.mean.shape != (9,) or payload.covariance.shape != (9, 9):
            self._cooperative_message_ids.add(message.message_id)
            return CooperativeIngestResult(
                CooperativeIngestStatus.UNSUPPORTED_PAYLOAD_SHAPE,
                message.message_id,
                track_id,
            )
        source_time = self._validated_source_time(mapped, source_time_mixture)
        propagated = propagate_source_time(
            payload.mean,
            payload.covariance,
            source_time,
            decision_time=self._current_time,
            mode=self._eventtrack_config.source_time_mode,
            process_noise_per_second=(
                self._eventtrack_config.tracker.process_noise_per_second
            ),
        )
        track = self._tracks[track_id]
        fused = covariance_intersection(
            track.mean,
            track.covariance,
            propagated.mean,
            propagated.covariance,
            objective=self._eventtrack_config.ci_objective,
        )
        hypotheses, other_probability = self._fuse_identities(
            track,
            payload,
            fused.weight_first,
        )
        track.mean = np.asarray(fused.mean, dtype=np.float64).copy()
        track.covariance = np.asarray(fused.covariance, dtype=np.float64).copy()
        track.state_time = self._current_time
        track.last_update_time = self._current_time
        track.last_event_time = source_time.mean
        track.last_arrival_time = received.received_at
        track.existence_probability = (
            fused.weight_first * track.existence_probability
            + (1.0 - fused.weight_first) * payload.existence_probability
        )
        track.identity_hypotheses = hypotheses
        track.other_identity_probability = other_probability
        track.factor_ids.update(payload.lineage.factor_ids)
        metadata = self._integrated_metadata[track_id]
        metadata.ancestor_message_ids.update(payload.lineage.ancestor_message_ids)
        metadata.ancestor_message_ids.add(message.message_id)
        metadata.lineage_complete = metadata.lineage_complete and payload.lineage.complete
        self._cooperative_message_ids.add(message.message_id)
        self._uncommitted_cooperative_message_ids.add(message.message_id)
        self._matched_since_commit.add(track_id)
        self._replays[track_id] = FixedLagReplay(
            track_id=track_id,
            initial_time=self._current_time,
            initial_mean=track.mean,
            initial_covariance=track.covariance,
            dynamics=self._dynamics,
            lag=self._eventtrack_config.fixed_lag_seconds,
            absorbed_factor_ids=tuple(sorted(track.factor_ids)),
            absorbed_ancestor_message_ids=tuple(
                sorted(metadata.ancestor_message_ids)
            ),
            out_of_window_policy=self._eventtrack_config.out_of_window_policy,
        )
        self._dirty = True
        return CooperativeIngestResult(
            CooperativeIngestStatus.APPLIED_CORRELATED_CI,
            message.message_id,
            track_id,
        )

    def _ingest_cooperative_mutating(
        self,
        received: ReceivedMessage,
        *,
        clock_map: AffineClockMap,
        source_time_mixture: SourceTimeMixture | None,
    ) -> CooperativeIngestResult:
        rejected, mapped = self._preflight_cooperative(received, clock_map)
        if rejected is not None:
            return rejected
        if isinstance(received.message.payload, IndependentIncrement):
            return self._ingest_independent(
                received,
                clock_map,
                mapped,
                source_time_mixture,
            )
        return self._ingest_correlated(received, mapped, source_time_mixture)

    def ingest_cooperative(
        self,
        received: ReceivedMessage,
        *,
        clock_map: AffineClockMap,
        source_time_mixture: SourceTimeMixture | None = None,
    ) -> CooperativeIngestResult:
        """Atomically absorb one independent increment or correlated belief."""

        self._ensure_active()
        shadow = copy.deepcopy(self)
        result = shadow._ingest_cooperative_mutating(
            received,
            clock_map=clock_map,
            source_time_mixture=source_time_mixture,
        )
        if result.applied or result.status in {
            CooperativeIngestStatus.DUPLICATE,
            CooperativeIngestStatus.MISSED_DEADLINE,
            CooperativeIngestStatus.EXPIRED_TTL,
            CooperativeIngestStatus.WRONG_TARGET,
            CooperativeIngestStatus.COORDINATE_FRAME_MISMATCH,
            CooperativeIngestStatus.OUT_OF_WINDOW,
            CooperativeIngestStatus.LINEAGE_OVERLAP,
            CooperativeIngestStatus.UNSUPPORTED_PAYLOAD_SHAPE,
        }:
            self._adopt(shadow)
        return result

    def prediction_for_track(self, track_id: str) -> TrackingPredictionV1:
        """Return one non-committed current prediction for VoI/canary use."""

        self._ensure_active()
        if track_id not in self._tracks:
            raise KeyError(track_id)
        return self._prediction(self._tracks[track_id], committed=False)

    def score_cooperative_candidate(
        self,
        received: ReceivedMessage,
        *,
        clock_map: AffineClockMap,
        on_time_estimate: FrozenOnTimeProbabilityV1,
        source_time_mixture: SourceTimeMixture | None = None,
        scorer_config: VoiScorerConfigV1 = DEFAULT_VOI_SCORER_CONFIG_V1,
    ) -> CounterfactualVoiScoreV1:
        """Score a non-mutating causal counterfactual for one wire candidate."""

        self._ensure_active()
        track_id = self._target_id(received)
        before = self.prediction_for_track(track_id)
        shadow = copy.deepcopy(self)
        result = shadow.ingest_cooperative(
            received,
            clock_map=clock_map,
            source_time_mixture=source_time_mixture,
        )
        if not result.applied:
            raise ValueError(
                "candidate cannot form an applied counterfactual: "
                f"{result.status.value}"
            )
        after = shadow.prediction_for_track(track_id)
        return score_counterfactual_voi_v1(
            before,
            after,
            received.message,
            on_time_estimate,
            scorer_config,
        )


__all__ = [
    "CooperativeIngestResult",
    "CooperativeIngestStatus",
    "EVENTTRACK_REFERENCE_CAPABILITIES_V1",
    "EventTrackReferenceCapabilitiesV1",
    "EventTrackReferenceCapabilityError",
    "EventTrackReferenceConfig",
    "EventTrackReferenceTracker",
]
