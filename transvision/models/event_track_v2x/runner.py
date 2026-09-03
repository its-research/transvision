"""Deterministic reference sequence runner for local canaries.

``ReferenceSequenceRunnerV1`` exercises the tracker lifecycle and the frozen
network/input contracts.  It is deliberately a reference/canary runner: it is
not the formal experiment orchestrator, an evaluator, or evidence of SOTA
performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Mapping

import numpy as np

from .contracts import DetectionCacheV1, TrackingPredictionV1
from .network import NetworkConditionId, NetworkTraceEventV1, NetworkTraceV1, condition_plan_v1
from .network_disturbance import (
    ConditionInputEvidenceV1,
    ConditionInputManifestV1,
    build_condition_input_manifest_v1,
    realize_condition_input_v1,
    require_causal_at_arrival_v1,
)
from .tracker import TrackerAdapter, TrackerIngestResult
from .wire import canonical_json_bytes


REFERENCE_SEQUENCE_RUNNER_SCHEMA_V1 = "eventtrack-v2x.reference-sequence-runner.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


class ReferenceSequenceRunnerError(ValueError):
    """Raised before a canary run can consume ambiguous or non-causal input."""


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise ReferenceSequenceRunnerError(f"{name} must be a canonical identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ReferenceSequenceRunnerError(f"{name} must be a lowercase SHA-256")
    return value


def _decision_times(value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ReferenceSequenceRunnerError("decision_times must be an array")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ReferenceSequenceRunnerError("decision_times must be numeric")
        observed = float(item)
        if not np.isfinite(observed):
            raise ReferenceSequenceRunnerError("decision_times must be finite")
        result.append(observed)
    if not result or any(current <= previous for previous, current in zip(result, result[1:])):
        raise ReferenceSequenceRunnerError(
            "decision_times must be non-empty and strictly increasing"
        )
    return tuple(result)


def _cache_encoded_bytes(cache: DetectionCacheV1) -> int:
    return len(canonical_json_bytes(cache.to_primitive()))


def _cache_manifest_sha256(caches: Mapping[str, DetectionCacheV1]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            [
                {"cache_sha256": caches[message_id].digest(), "message_id": message_id}
                for message_id in sorted(caches)
            ]
        )
    ).hexdigest()


def _validate_disturbance_scope(event: NetworkTraceEventV1, condition: NetworkConditionId) -> None:
    clock = (
        event.clock_offset_seconds,
        event.clock_drift_ppm,
        event.clock_error_at_transmit_seconds,
    )
    pose = (*event.translation_noise_metres, event.yaw_noise_degrees)
    if condition not in {NetworkConditionId.C6, NetworkConditionId.C8} and any(clock):
        raise ReferenceSequenceRunnerError(
            f"{condition.value} trace carries an unauthorized clock disturbance"
        )
    if condition not in {NetworkConditionId.C7, NetworkConditionId.C8} and any(pose):
        raise ReferenceSequenceRunnerError(
            f"{condition.value} trace carries an unauthorized pose disturbance"
        )


@dataclass(frozen=True, slots=True)
class ReferenceSequenceRunResultV1:
    """Canonical output of one reference/canary sequence execution."""

    sequence_id: str
    adapter_name: str
    condition_id: NetworkConditionId
    run_config_sha256: str
    detection_cache_manifest_sha256: str
    network_trace_sha256: str
    decision_times: tuple[float, ...]
    predictions: tuple[TrackingPredictionV1, ...]
    condition_input_manifest: ConditionInputManifestV1 | None
    processed_message_ids: tuple[str, ...]
    dropped_message_ids: tuple[str, ...]
    late_message_ids: tuple[str, ...]
    schema_version: str = field(init=False, default=REFERENCE_SEQUENCE_RUNNER_SCHEMA_V1)
    kind: str = field(init=False, default="reference_sequence_run_result_v1")

    def __post_init__(self) -> None:
        for name in ("sequence_id", "adapter_name"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        condition = NetworkConditionId(self.condition_id)
        if condition.value == "C9":
            raise ReferenceSequenceRunnerError("C9 is not supported by the reference runner")
        object.__setattr__(self, "condition_id", condition)
        for name in (
            "run_config_sha256",
            "detection_cache_manifest_sha256",
            "network_trace_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "decision_times", _decision_times(self.decision_times))
        if not isinstance(self.predictions, (list, tuple)) or not all(
            isinstance(item, TrackingPredictionV1) for item in self.predictions
        ):
            raise TypeError("predictions must contain TrackingPredictionV1")
        predictions = tuple(self.predictions)
        if any(not item.committed for item in predictions):
            raise ReferenceSequenceRunnerError("runner predictions must be committed")
        decision_index = {
            decision_time: index
            for index, decision_time in enumerate(self.decision_times)
        }
        if any(item.sequence_id != self.sequence_id for item in predictions):
            raise ReferenceSequenceRunnerError(
                "runner prediction sequence does not match the requested sequence"
            )
        if any(item.decision_time not in decision_index for item in predictions):
            raise ReferenceSequenceRunnerError(
                "runner prediction decision_time is outside the committed schedule"
            )
        prediction_keys = tuple(
            (item.decision_time, item.track_id) for item in predictions
        )
        if len(prediction_keys) != len(set(prediction_keys)):
            raise ReferenceSequenceRunnerError(
                "runner predictions duplicate one track at a decision time"
            )
        expected_order = tuple(
            sorted(
                predictions,
                key=lambda item: (
                    decision_index[item.decision_time],
                    item.frame_id,
                    item.track_id,
                ),
            )
        )
        if predictions != expected_order:
            raise ReferenceSequenceRunnerError(
                "runner predictions are not in canonical commit/frame/track order"
            )
        object.__setattr__(self, "predictions", predictions)
        id_groups: list[tuple[str, ...]] = []
        for name in (
            "processed_message_ids",
            "dropped_message_ids",
            "late_message_ids",
        ):
            raw = getattr(self, name)
            if not isinstance(raw, (list, tuple)):
                raise TypeError(f"{name} must be an array")
            ids = tuple(_identifier(item, name) for item in raw)
            if len(ids) != len(set(ids)):
                raise ReferenceSequenceRunnerError(f"{name} must be unique")
            object.__setattr__(self, name, ids)
            id_groups.append(ids)
        combined = tuple(item for group in id_groups for item in group)
        if len(combined) != len(set(combined)):
            raise ReferenceSequenceRunnerError(
                "processed, dropped, and late message IDs must be disjoint"
            )
        requires_manifest = condition in {
            NetworkConditionId.C6,
            NetworkConditionId.C7,
            NetworkConditionId.C8,
        }
        if requires_manifest:
            if not isinstance(self.condition_input_manifest, ConditionInputManifestV1):
                raise ReferenceSequenceRunnerError("C6-C8 require a condition input manifest")
            manifest = self.condition_input_manifest
            if (
                manifest.condition_id is not condition
                or manifest.network_trace_sha256 != self.network_trace_sha256
                or manifest.run_config_sha256 != self.run_config_sha256
                or manifest.detection_cache_sha256
                != self.detection_cache_manifest_sha256
            ):
                raise ReferenceSequenceRunnerError(
                    "condition input manifest does not match the run"
                )
        elif self.condition_input_manifest is not None:
            raise ReferenceSequenceRunnerError("C0-C5 must not carry a condition input manifest")

    def to_primitive(self) -> dict[str, Any]:
        manifest = self.condition_input_manifest
        return {
            "adapter_name": self.adapter_name,
            "condition_id": self.condition_id.value,
            "condition_input_manifest": (
                None if manifest is None else manifest.to_primitive()
            ),
            "condition_input_manifest_sha256": (
                None if manifest is None else manifest.content_sha256
            ),
            "decision_times": list(self.decision_times),
            "detection_cache_manifest_sha256": self.detection_cache_manifest_sha256,
            "dropped_message_ids": list(self.dropped_message_ids),
            "kind": self.kind,
            "late_message_ids": list(self.late_message_ids),
            "network_trace_sha256": self.network_trace_sha256,
            "predictions": [item.to_primitive() for item in self.predictions],
            "processed_message_ids": list(self.processed_message_ids),
            "run_config_sha256": self.run_config_sha256,
            "schema_version": self.schema_version,
            "sequence_id": self.sequence_id,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_primitive())

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


class ReferenceSequenceRunnerV1:
    """Run one fresh :class:`TrackerAdapter` as a deterministic local canary."""

    def __init__(self, tracker: TrackerAdapter) -> None:
        if not isinstance(tracker, TrackerAdapter):
            raise TypeError("tracker must implement TrackerAdapter")
        self._tracker = tracker

    def run(
        self,
        *,
        caches_by_message_id: Mapping[str, DetectionCacheV1],
        network_trace: NetworkTraceV1,
        run_config_sha256: str,
        decision_times: tuple[float, ...],
    ) -> ReferenceSequenceRunResultV1:
        """Execute causal arrival/decision semantics without evaluating metrics."""

        raw_condition = getattr(network_trace, "condition_id", None)
        if raw_condition == "C9" or getattr(raw_condition, "value", None) == "C9":
            raise ReferenceSequenceRunnerError(
                "C9 measured traces are not supported by ReferenceSequenceRunnerV1"
            )
        if not isinstance(network_trace, NetworkTraceV1):
            raise TypeError("network_trace must be NetworkTraceV1")
        condition = network_trace.condition_id
        if condition.value == "C9":
            raise ReferenceSequenceRunnerError(
                "C9 measured traces are not supported by ReferenceSequenceRunnerV1"
            )
        if network_trace.condition_plan_sha256 != condition_plan_v1(condition).content_sha256:
            raise ReferenceSequenceRunnerError(
                "network trace does not use the frozen condition plan"
            )
        config_sha256 = _sha256(run_config_sha256, "run_config_sha256")
        decisions = _decision_times(decision_times)
        if not isinstance(caches_by_message_id, Mapping) or not all(
            type(key) is str for key in caches_by_message_id
        ):
            raise TypeError("caches_by_message_id must be a string-keyed mapping")
        caches = dict(caches_by_message_id)
        if not caches or not all(isinstance(item, DetectionCacheV1) for item in caches.values()):
            raise TypeError("caches_by_message_id must contain DetectionCacheV1 values")
        trace_ids = tuple(event.packet.message_id for event in network_trace.events)
        if set(caches) != set(trace_ids):
            raise ReferenceSequenceRunnerError(
                "cache mapping must contain exactly the trace message IDs"
            )
        if len(trace_ids) != len(set(trace_ids)):
            raise ReferenceSequenceRunnerError("one message would be counted twice")

        sequence_ids = {cache.sequence_id for cache in caches.values()}
        if len(sequence_ids) != 1:
            raise ReferenceSequenceRunnerError("cache sequence mismatch")
        sequence_id = next(iter(sequence_ids))
        cache_digests = tuple(cache.digest() for cache in caches.values())
        if len(cache_digests) != len(set(cache_digests)):
            raise ReferenceSequenceRunnerError(
                "distinct messages must not double-count one detection cache"
            )
        per_source_sequences: dict[str, list[int]] = {}
        for event in network_trace.events:
            packet = event.packet
            cache = caches[packet.message_id]
            if cache.agent_id != packet.source:
                raise ReferenceSequenceRunnerError("message/cache source mismatch")
            if _cache_encoded_bytes(cache) != packet.encoded_bytes:
                raise ReferenceSequenceRunnerError("message/cache encoded size mismatch")
            per_source_sequences.setdefault(packet.source, []).append(packet.sequence)
            _validate_disturbance_scope(event, condition)
        if any(
            len(values) != len(set(values))
            or any(current <= previous for previous, current in zip(values, values[1:]))
            for values in per_source_sequences.values()
        ):
            raise ReferenceSequenceRunnerError(
                "packet sequence must be unique and increasing per source"
            )

        final_decision = decisions[-1]
        delivered = network_trace.arrival_order
        dropped_ids = tuple(
            event.packet.message_id for event in network_trace.events if event.dropped
        )
        late_ids = tuple(
            event.packet.message_id
            for event in delivered
            if event.arrival_time is not None and event.arrival_time > final_decision
        )
        available = tuple(
            event
            for event in delivered
            if event.arrival_time is not None and event.arrival_time <= final_decision
        )

        realized_caches: dict[str, DetectionCacheV1] = {}
        realized_evidence: dict[str, ConditionInputEvidenceV1] = {}
        for event in delivered:
            message_id = event.packet.message_id
            cache = caches[message_id]
            assert event.arrival_time is not None
            if condition in {
                NetworkConditionId.C6,
                NetworkConditionId.C7,
                NetworkConditionId.C8,
            }:
                realized = realize_condition_input_v1(cache, network_trace, event)
                realized_caches[message_id] = require_causal_at_arrival_v1(realized)
                realized_evidence[message_id] = realized.evidence
            else:
                if cache.event_time > event.arrival_time + 1e-12:
                    raise ReferenceSequenceRunnerError(
                        "future detection event cannot be ingested"
                    )
                realized_caches[message_id] = cache

        groups_by_decision: list[tuple[float, tuple[NetworkTraceEventV1, ...]]] = []
        remaining = list(available)
        for decision in decisions:
            selected: list[NetworkTraceEventV1] = []
            while remaining and remaining[0].arrival_time is not None and remaining[0].arrival_time <= decision:
                selected.append(remaining.pop(0))
            groups_by_decision.append((decision, tuple(selected)))
        simultaneous = any(
            sum(item.arrival_time == event.arrival_time for item in events) > 1
            for _, events in groups_by_decision
            for event in events
        )
        batch_ingest = getattr(self._tracker, "ingest_batch", None)
        if simultaneous and not callable(batch_ingest):
            raise ReferenceSequenceRunnerError(
                "simultaneous arrivals require adapter ingest_batch support"
            )

        initial_time = min(
            decisions[0],
            min(event.packet.transmitted_at for event in network_trace.events),
        )
        self._tracker.reset(sequence_id=sequence_id, initial_time=initial_time)
        processed: list[str] = []
        processed_set: set[str] = set()
        evidence_in_ingest_order: list[ConditionInputEvidenceV1] = []
        for decision, events in groups_by_decision:
            self._tracker.advance(decision)
            offset = 0
            while offset < len(events):
                arrival = events[offset].arrival_time
                assert arrival is not None
                end = offset + 1
                while end < len(events) and events[end].arrival_time == arrival:
                    end += 1
                group = events[offset:end]
                for event in group:
                    if event.packet.message_id in processed_set:
                        raise ReferenceSequenceRunnerError(
                            "one message would be counted twice"
                        )
                if len(group) == 1:
                    event = group[0]
                    result = self._tracker.ingest(
                        realized_caches[event.packet.message_id],
                        arrival_time=arrival,
                    )
                    results = (result,)
                else:
                    assert callable(batch_ingest)
                    results = tuple(
                        batch_ingest(
                            (
                                realized_caches[event.packet.message_id],
                                arrival,
                            )
                            for event in group
                        )
                    )
                if len(results) != len(group) or not all(
                    isinstance(item, TrackerIngestResult) and item.applied
                    for item in results
                ):
                    raise ReferenceSequenceRunnerError(
                        "tracker rejected a preflighted detection cache"
                    )
                for event in group:
                    message_id = event.packet.message_id
                    processed.append(message_id)
                    processed_set.add(message_id)
                    if message_id in realized_evidence:
                        evidence_in_ingest_order.append(realized_evidence[message_id])
                offset = end
            self._tracker.commit(decision)
        predictions = self._tracker.finalize()

        cache_manifest_sha256 = _cache_manifest_sha256(caches)
        manifest: ConditionInputManifestV1 | None = None
        if condition in {
            NetworkConditionId.C6,
            NetworkConditionId.C7,
            NetworkConditionId.C8,
        }:
            if not evidence_in_ingest_order:
                raise ReferenceSequenceRunnerError(
                    "C6-C8 require at least one processed realized input"
                )
            run_id_seed = canonical_json_bytes(
                {
                    "condition_id": condition.value,
                    "run_config_sha256": config_sha256,
                    "sequence_id": sequence_id,
                    "trace_sha256": network_trace.content_sha256,
                }
            )
            run_id = f"reference-run-{hashlib.sha256(run_id_seed).hexdigest()[:24]}"
            manifest = build_condition_input_manifest_v1(
                run_id=run_id,
                trace=network_trace,
                run_config_sha256=config_sha256,
                detection_cache_sha256=cache_manifest_sha256,
                evidence=evidence_in_ingest_order,
            )

        return ReferenceSequenceRunResultV1(
            sequence_id=sequence_id,
            adapter_name=self._tracker.adapter_name,
            condition_id=condition,
            run_config_sha256=config_sha256,
            detection_cache_manifest_sha256=cache_manifest_sha256,
            network_trace_sha256=network_trace.content_sha256,
            decision_times=decisions,
            predictions=predictions,
            condition_input_manifest=manifest,
            processed_message_ids=tuple(processed),
            dropped_message_ids=dropped_ids,
            late_message_ids=late_ids,
        )


__all__ = [
    "REFERENCE_SEQUENCE_RUNNER_SCHEMA_V1",
    "ReferenceSequenceRunResultV1",
    "ReferenceSequenceRunnerError",
    "ReferenceSequenceRunnerV1",
]
