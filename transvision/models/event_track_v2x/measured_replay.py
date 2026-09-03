"""Deterministic, fail-closed C9 measured-packet replay contract.

Measured packet metadata do not contain application payload identity.  This
module therefore has two explicit capabilities:

* ``network_timing_only`` derives loss, arrival and measured byte ledgers only;
* ``application_messages`` additionally requires an exact, externally supplied
  mapping from packet sequence numbers to canonical ``DetectionCacheV1``
  payloads.

ACKs and retransmissions remain wire-ledger records.  They never create extra
application messages.  A fragmented application message is delivered exactly
once, at the latest first-arrival time of all of its packet sequences.  Missing
payload bindings are never filled with synthetic detections.

The output is intentionally separate from :class:`NetworkTraceV1`: that class
is sealed to the synthetic C0--C8 condition plan and synthetic wire accounting.
``application_network_events_v1`` provides the small reference mapping needed
by an application runner without mislabelling measured evidence as synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
from typing import ClassVar, Mapping

import numpy as np

from .contracts import DetectionCacheV1
from .measured_packet_trace import (
    MeasuredPacketRecordV1,
    MeasuredPacketTraceV1,
    validate_measured_packet_trace_segment_v1,
)
from .measured_trace import MeasuredTraceReceiptV1, MeasuredTraceSegmentV1, TraceRole
from .network import NetworkEvent, PacketRequest
from .wire import canonical_json_bytes


C9_MEASURED_REPLAY_SCHEMA_V1 = "eventtrack-v2x.c9-measured-replay.v1"
C9_MEASURED_REPLAY_MAPPING_SCHEMA_V1 = (
    "eventtrack-v2x.c9-detection-cache-packet-mapping.v1"
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class C9MeasuredReplayError(ValueError):
    """Raised when measured replay evidence is incomplete or ambiguous."""


class C9ReplayCapability(str, Enum):
    """Closed capability set; timing-only artifacts cannot feed a tracker."""

    NETWORK_TIMING_ONLY = "network_timing_only"
    APPLICATION_MESSAGES = "application_messages"


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise C9MeasuredReplayError(f"{name} must be a canonical identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise C9MeasuredReplayError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise C9MeasuredReplayError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise C9MeasuredReplayError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise C9MeasuredReplayError(f"{name} must be an integer")
    if value < minimum:
        raise C9MeasuredReplayError(f"{name} must be at least {minimum}")
    return value


def _strict_fields(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise C9MeasuredReplayError(f"{name} must be a string-keyed object")
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise C9MeasuredReplayError(
            f"{name} fields do not match schema; missing={missing}, unknown={unknown}"
        )
    return value


def _record_sha256(record: MeasuredPacketRecordV1) -> str:
    return hashlib.sha256(canonical_json_bytes(record.to_primitive())).hexdigest()


def _segment_sha256(segment: MeasuredTraceSegmentV1) -> str:
    return hashlib.sha256(canonical_json_bytes(segment.to_primitive())).hexdigest()


def _cache_bytes(cache: DetectionCacheV1) -> bytes:
    return canonical_json_bytes(cache.to_primitive())


_MAPPING_CONTRACT_V1: dict[str, object] = {
    "ack_rule": "ack_records_are_wire_only",
    "application_arrival_rule": (
        "max(first_successful_data_arrival_per_sequence)"
    ),
    "application_delivery_rule": "all_bound_sequences_must_arrive",
    "cache_payload_rule": "payload_is_canonical_detection_cache_v1_bytes",
    "duplicate_rule": "one_application_message_per_unique_message_and_cache",
    "future_rule": "cache_event_time_must_not_exceed_first_data_send",
    "retransmission_rule": "later_attempts_never_create_application_messages",
    "schema_version": C9_MEASURED_REPLAY_MAPPING_SCHEMA_V1,
}
C9_MEASURED_REPLAY_MAPPING_SHA256_V1 = hashlib.sha256(
    canonical_json_bytes(_MAPPING_CONTRACT_V1)
).hexdigest()


@dataclass(frozen=True, slots=True)
class C9ApplicationMessageBindingV1:
    """Exact external mapping from packet fragments to one cache payload."""

    message_id: str
    source: str
    sequence_numbers: tuple[int, ...]
    detection_cache_sha256: str
    payload_sha256: str
    payload_bytes: int
    deadline: float
    payload_kind: str = field(init=False, default="detection_cache_v1")
    schema_version: str = field(
        init=False,
        default=C9_MEASURED_REPLAY_MAPPING_SCHEMA_V1,
    )

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "deadline",
            "detection_cache_sha256",
            "message_id",
            "payload_bytes",
            "payload_kind",
            "payload_sha256",
            "schema_version",
            "sequence_numbers",
            "source",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "message_id", _identifier(self.message_id, "message_id"))
        object.__setattr__(self, "source", _identifier(self.source, "source"))
        if not isinstance(self.sequence_numbers, (list, tuple)):
            raise TypeError("sequence_numbers must be an array")
        sequences = tuple(
            _integer(item, "sequence_number") for item in self.sequence_numbers
        )
        if not sequences or sequences != tuple(sorted(sequences)):
            raise C9MeasuredReplayError(
                "sequence_numbers must be non-empty and strictly sorted"
            )
        if len(sequences) != len(set(sequences)):
            raise C9MeasuredReplayError("sequence_numbers must be unique")
        object.__setattr__(self, "sequence_numbers", sequences)
        for name in ("detection_cache_sha256", "payload_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.payload_sha256 != self.detection_cache_sha256:
            raise C9MeasuredReplayError(
                "detection cache bytes must be the exact application payload"
            )
        object.__setattr__(
            self,
            "payload_bytes",
            _integer(self.payload_bytes, "payload_bytes", minimum=1),
        )
        object.__setattr__(self, "deadline", _finite(self.deadline, "deadline"))

    def to_primitive(self) -> dict[str, object]:
        return {
            "deadline": self.deadline,
            "detection_cache_sha256": self.detection_cache_sha256,
            "message_id": self.message_id,
            "payload_bytes": self.payload_bytes,
            "payload_kind": self.payload_kind,
            "payload_sha256": self.payload_sha256,
            "schema_version": self.schema_version,
            "sequence_numbers": list(self.sequence_numbers),
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "C9ApplicationMessageBindingV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        if (
            item["payload_kind"] != "detection_cache_v1"
            or item["schema_version"] != C9_MEASURED_REPLAY_MAPPING_SCHEMA_V1
        ):
            raise C9MeasuredReplayError("unsupported C9 payload mapping schema")
        raw_sequences = item["sequence_numbers"]
        if not isinstance(raw_sequences, list):
            raise C9MeasuredReplayError("sequence_numbers must be an array")
        return cls(
            message_id=item["message_id"],  # type: ignore[arg-type]
            source=item["source"],  # type: ignore[arg-type]
            sequence_numbers=tuple(raw_sequences),
            detection_cache_sha256=item["detection_cache_sha256"],  # type: ignore[arg-type]
            payload_sha256=item["payload_sha256"],  # type: ignore[arg-type]
            payload_bytes=item["payload_bytes"],  # type: ignore[arg-type]
            deadline=item["deadline"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class C9MeasuredSequenceOutcomeV1:
    """One logical measured packet sequence after retry de-duplication."""

    sequence_number: int
    data_record_sha256s: tuple[str, ...]
    ack_record_sha256s: tuple[str, ...]
    first_send_time: float
    first_arrival_time: float | None
    dropped: bool
    data_attempts: int
    delivered_data_attempts: int
    retransmissions: int
    initial_data_on_wire_bytes: int
    data_on_wire_bytes: int
    ack_transmissions: int
    delivered_ack_transmissions: int
    ack_on_wire_bytes: int

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "ack_on_wire_bytes",
            "ack_record_sha256s",
            "ack_transmissions",
            "data_attempts",
            "data_on_wire_bytes",
            "data_record_sha256s",
            "delivered_ack_transmissions",
            "delivered_data_attempts",
            "dropped",
            "first_arrival_time",
            "first_send_time",
            "initial_data_on_wire_bytes",
            "retransmissions",
            "sequence_number",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence_number",
            _integer(self.sequence_number, "sequence_number"),
        )
        for name in ("data_record_sha256s", "ack_record_sha256s"):
            raw = getattr(self, name)
            if not isinstance(raw, (list, tuple)):
                raise TypeError(f"{name} must be an array")
            digests = tuple(_sha256(item, name) for item in raw)
            if len(digests) != len(set(digests)):
                raise C9MeasuredReplayError(f"{name} must be unique")
            object.__setattr__(self, name, digests)
        if not self.data_record_sha256s:
            raise C9MeasuredReplayError("sequence outcome requires a data record")
        first_send = _finite(self.first_send_time, "first_send_time")
        object.__setattr__(self, "first_send_time", first_send)
        if self.first_arrival_time is not None:
            arrival = _finite(self.first_arrival_time, "first_arrival_time")
            if arrival < first_send:
                raise C9MeasuredReplayError("arrival cannot precede first send")
            object.__setattr__(self, "first_arrival_time", arrival)
        if type(self.dropped) is not bool:
            raise TypeError("dropped must be bool")
        if self.dropped != (self.first_arrival_time is None):
            raise C9MeasuredReplayError(
                "dropped must be true exactly when no data attempt arrives"
            )
        for name, minimum in (
            ("data_attempts", 1),
            ("delivered_data_attempts", 0),
            ("retransmissions", 0),
            ("initial_data_on_wire_bytes", 1),
            ("data_on_wire_bytes", 1),
            ("ack_transmissions", 0),
            ("delivered_ack_transmissions", 0),
            ("ack_on_wire_bytes", 0),
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=minimum),
            )
        if self.data_attempts != len(self.data_record_sha256s):
            raise C9MeasuredReplayError("data_attempts does not match record hashes")
        if self.ack_transmissions != len(self.ack_record_sha256s):
            raise C9MeasuredReplayError("ack_transmissions does not match record hashes")
        if self.retransmissions != self.data_attempts - 1:
            raise C9MeasuredReplayError("retransmissions must exclude the initial send")
        if not 0 <= self.delivered_data_attempts <= self.data_attempts:
            raise C9MeasuredReplayError("delivered_data_attempts is impossible")
        if self.dropped != (self.delivered_data_attempts == 0):
            raise C9MeasuredReplayError("delivery count disagrees with dropped")
        if not 0 <= self.delivered_ack_transmissions <= self.ack_transmissions:
            raise C9MeasuredReplayError("delivered_ack_transmissions is impossible")
        if self.initial_data_on_wire_bytes > self.data_on_wire_bytes:
            raise C9MeasuredReplayError("initial data bytes exceed all attempt bytes")

    def to_primitive(self) -> dict[str, object]:
        return {
            "ack_on_wire_bytes": self.ack_on_wire_bytes,
            "ack_record_sha256s": list(self.ack_record_sha256s),
            "ack_transmissions": self.ack_transmissions,
            "data_attempts": self.data_attempts,
            "data_on_wire_bytes": self.data_on_wire_bytes,
            "data_record_sha256s": list(self.data_record_sha256s),
            "delivered_ack_transmissions": self.delivered_ack_transmissions,
            "delivered_data_attempts": self.delivered_data_attempts,
            "dropped": self.dropped,
            "first_arrival_time": self.first_arrival_time,
            "first_send_time": self.first_send_time,
            "initial_data_on_wire_bytes": self.initial_data_on_wire_bytes,
            "retransmissions": self.retransmissions,
            "sequence_number": self.sequence_number,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_primitive())

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> "C9MeasuredSequenceOutcomeV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        for name in ("data_record_sha256s", "ack_record_sha256s"):
            if not isinstance(item[name], list):
                raise C9MeasuredReplayError(f"{name} must be an array")
        return cls(
            sequence_number=item["sequence_number"],  # type: ignore[arg-type]
            data_record_sha256s=tuple(item["data_record_sha256s"]),  # type: ignore[arg-type]
            ack_record_sha256s=tuple(item["ack_record_sha256s"]),  # type: ignore[arg-type]
            first_send_time=item["first_send_time"],  # type: ignore[arg-type]
            first_arrival_time=item["first_arrival_time"],  # type: ignore[arg-type]
            dropped=item["dropped"],  # type: ignore[arg-type]
            data_attempts=item["data_attempts"],  # type: ignore[arg-type]
            delivered_data_attempts=item["delivered_data_attempts"],  # type: ignore[arg-type]
            retransmissions=item["retransmissions"],  # type: ignore[arg-type]
            initial_data_on_wire_bytes=item["initial_data_on_wire_bytes"],  # type: ignore[arg-type]
            data_on_wire_bytes=item["data_on_wire_bytes"],  # type: ignore[arg-type]
            ack_transmissions=item["ack_transmissions"],  # type: ignore[arg-type]
            delivered_ack_transmissions=item[  # type: ignore[arg-type]
                "delivered_ack_transmissions"
            ],
            ack_on_wire_bytes=item["ack_on_wire_bytes"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class C9ApplicationMessageReplayV1:
    """One de-duplicated tracker-facing message derived from measured packets."""

    binding: C9ApplicationMessageBindingV1
    detection_cache_sequence_id: str
    detection_cache_frame_id: str
    detection_cache_event_time: float
    sequence_outcome_sha256s: tuple[str, ...]
    transmitted_at: float
    arrival_time: float | None
    dropped: bool
    on_time: bool
    data_on_wire_bytes: int
    ack_on_wire_bytes: int

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "ack_on_wire_bytes",
            "arrival_time",
            "binding",
            "data_on_wire_bytes",
            "detection_cache_event_time",
            "detection_cache_frame_id",
            "detection_cache_sequence_id",
            "dropped",
            "on_time",
            "sequence_outcome_sha256s",
            "transmitted_at",
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.binding, C9ApplicationMessageBindingV1):
            raise TypeError("binding must be C9ApplicationMessageBindingV1")
        for name in ("detection_cache_sequence_id", "detection_cache_frame_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        event_time = _finite(
            self.detection_cache_event_time,
            "detection_cache_event_time",
        )
        transmitted = _finite(self.transmitted_at, "transmitted_at")
        if event_time > transmitted:
            raise C9MeasuredReplayError(
                "future detection cache cannot be mapped to an earlier packet send"
            )
        if self.binding.deadline < transmitted:
            raise C9MeasuredReplayError("message deadline cannot precede transmission")
        object.__setattr__(self, "detection_cache_event_time", event_time)
        object.__setattr__(self, "transmitted_at", transmitted)
        if self.arrival_time is not None:
            arrival = _finite(self.arrival_time, "arrival_time")
            if arrival < transmitted:
                raise C9MeasuredReplayError("message arrival cannot precede transmission")
            object.__setattr__(self, "arrival_time", arrival)
        for name in ("dropped", "on_time"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.dropped != (self.arrival_time is None):
            raise C9MeasuredReplayError("message dropped flag disagrees with arrival")
        expected_on_time = (
            self.arrival_time is not None
            and self.arrival_time <= self.binding.deadline
        )
        if self.on_time is not expected_on_time:
            raise C9MeasuredReplayError("message on_time is not derived from deadline")
        if not isinstance(self.sequence_outcome_sha256s, (list, tuple)):
            raise TypeError("sequence_outcome_sha256s must be an array")
        outcomes = tuple(
            _sha256(item, "sequence_outcome_sha256")
            for item in self.sequence_outcome_sha256s
        )
        if len(outcomes) != len(self.binding.sequence_numbers):
            raise C9MeasuredReplayError(
                "one sequence outcome is required for every bound packet sequence"
            )
        if len(outcomes) != len(set(outcomes)):
            raise C9MeasuredReplayError("sequence outcomes must be unique")
        object.__setattr__(self, "sequence_outcome_sha256s", outcomes)
        for name in ("data_on_wire_bytes", "ack_on_wire_bytes"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.data_on_wire_bytes < self.binding.payload_bytes:
            raise C9MeasuredReplayError(
                "measured data bytes cannot contain the declared cache payload"
            )

    def to_primitive(self) -> dict[str, object]:
        return {
            "ack_on_wire_bytes": self.ack_on_wire_bytes,
            "arrival_time": self.arrival_time,
            "binding": self.binding.to_primitive(),
            "data_on_wire_bytes": self.data_on_wire_bytes,
            "detection_cache_event_time": self.detection_cache_event_time,
            "detection_cache_frame_id": self.detection_cache_frame_id,
            "detection_cache_sequence_id": self.detection_cache_sequence_id,
            "dropped": self.dropped,
            "on_time": self.on_time,
            "sequence_outcome_sha256s": list(self.sequence_outcome_sha256s),
            "transmitted_at": self.transmitted_at,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "C9ApplicationMessageReplayV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        raw_outcomes = item["sequence_outcome_sha256s"]
        if not isinstance(raw_outcomes, list):
            raise C9MeasuredReplayError("sequence_outcome_sha256s must be an array")
        return cls(
            binding=C9ApplicationMessageBindingV1.from_mapping(item["binding"]),
            detection_cache_sequence_id=item["detection_cache_sequence_id"],  # type: ignore[arg-type]
            detection_cache_frame_id=item["detection_cache_frame_id"],  # type: ignore[arg-type]
            detection_cache_event_time=item["detection_cache_event_time"],  # type: ignore[arg-type]
            sequence_outcome_sha256s=tuple(raw_outcomes),
            transmitted_at=item["transmitted_at"],  # type: ignore[arg-type]
            arrival_time=item["arrival_time"],  # type: ignore[arg-type]
            dropped=item["dropped"],  # type: ignore[arg-type]
            on_time=item["on_time"],  # type: ignore[arg-type]
            data_on_wire_bytes=item["data_on_wire_bytes"],  # type: ignore[arg-type]
            ack_on_wire_bytes=item["ack_on_wire_bytes"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class C9MeasuredReplayV1:
    """Canonical cold-readable replay derivation for one receipt-bound trace."""

    receipt_id: str
    receipt_sha256: str
    collection_protocol_sha256: str
    trace_id: str
    trace_role: TraceRole
    receipt_segment_sha256: str
    packet_trace_sha256: str
    packet_record_count: int
    supports_clock_claims: bool
    capability: C9ReplayCapability
    sequence_outcomes: tuple[C9MeasuredSequenceOutcomeV1, ...]
    application_messages: tuple[C9ApplicationMessageReplayV1, ...]
    mapping_contract_sha256: str = field(
        init=False,
        default=C9_MEASURED_REPLAY_MAPPING_SHA256_V1,
    )
    kind: str = field(init=False, default="c9_measured_replay_v1")
    schema_version: str = field(init=False, default=C9_MEASURED_REPLAY_SCHEMA_V1)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "application_messages",
            "capability",
            "collection_protocol_sha256",
            "kind",
            "mapping_contract_sha256",
            "packet_record_count",
            "packet_trace_sha256",
            "receipt_id",
            "receipt_segment_sha256",
            "receipt_sha256",
            "schema_version",
            "sequence_outcomes",
            "supports_clock_claims",
            "trace_id",
            "trace_role",
        }
    )

    def __post_init__(self) -> None:
        for name in ("receipt_id", "trace_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "receipt_sha256",
            "collection_protocol_sha256",
            "receipt_segment_sha256",
            "packet_trace_sha256",
            "mapping_contract_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.mapping_contract_sha256 != C9_MEASURED_REPLAY_MAPPING_SHA256_V1:
            raise C9MeasuredReplayError("unsupported measured replay mapping contract")
        object.__setattr__(self, "trace_role", TraceRole(self.trace_role))
        object.__setattr__(self, "capability", C9ReplayCapability(self.capability))
        object.__setattr__(
            self,
            "packet_record_count",
            _integer(self.packet_record_count, "packet_record_count", minimum=1),
        )
        if type(self.supports_clock_claims) is not bool:
            raise TypeError("supports_clock_claims must be bool")
        if not isinstance(self.sequence_outcomes, (list, tuple)) or not all(
            isinstance(item, C9MeasuredSequenceOutcomeV1)
            for item in self.sequence_outcomes
        ):
            raise TypeError("sequence_outcomes must contain C9MeasuredSequenceOutcomeV1")
        outcomes = tuple(self.sequence_outcomes)
        if not outcomes:
            raise C9MeasuredReplayError("sequence_outcomes must be non-empty")
        if tuple(item.sequence_number for item in outcomes) != tuple(
            sorted(item.sequence_number for item in outcomes)
        ):
            raise C9MeasuredReplayError("sequence_outcomes must be sorted")
        if len({item.sequence_number for item in outcomes}) != len(outcomes):
            raise C9MeasuredReplayError("sequence_outcomes must be unique")
        record_hashes = tuple(
            digest
            for outcome in outcomes
            for digest in (*outcome.data_record_sha256s, *outcome.ack_record_sha256s)
        )
        if len(record_hashes) != self.packet_record_count:
            raise C9MeasuredReplayError("packet_record_count does not match outcomes")
        if len(record_hashes) != len(set(record_hashes)):
            raise C9MeasuredReplayError("one measured record would be counted twice")
        object.__setattr__(self, "sequence_outcomes", outcomes)
        if not isinstance(self.application_messages, (list, tuple)) or not all(
            isinstance(item, C9ApplicationMessageReplayV1)
            for item in self.application_messages
        ):
            raise TypeError(
                "application_messages must contain C9ApplicationMessageReplayV1"
            )
        messages = tuple(self.application_messages)
        if self.capability is C9ReplayCapability.NETWORK_TIMING_ONLY:
            if messages:
                raise C9MeasuredReplayError(
                    "timing-only replay cannot contain application messages"
                )
        else:
            if not messages:
                raise C9MeasuredReplayError(
                    "application replay requires payload-bound messages"
                )
            expected_order = tuple(
                sorted(
                    messages,
                    key=lambda item: (
                        item.transmitted_at,
                        item.binding.source,
                        item.binding.message_id,
                    ),
                )
            )
            if messages != expected_order:
                raise C9MeasuredReplayError("application_messages are not canonical")
            message_ids = tuple(item.binding.message_id for item in messages)
            cache_ids = tuple(
                item.binding.detection_cache_sha256 for item in messages
            )
            if len(message_ids) != len(set(message_ids)):
                raise C9MeasuredReplayError("application message IDs must be unique")
            if len(cache_ids) != len(set(cache_ids)):
                raise C9MeasuredReplayError(
                    "one detection cache cannot be counted as two messages"
                )
            mapped_sequences = tuple(
                sequence
                for message in messages
                for sequence in message.binding.sequence_numbers
            )
            if len(mapped_sequences) != len(set(mapped_sequences)):
                raise C9MeasuredReplayError(
                    "one measured packet sequence would be counted twice"
                )
            if set(mapped_sequences) != {
                outcome.sequence_number for outcome in outcomes
            }:
                raise C9MeasuredReplayError(
                    "application mapping must cover every measured data sequence exactly"
                )
            outcome_by_sequence = {
                item.sequence_number: item for item in outcomes
            }
            for message in messages:
                selected = tuple(
                    outcome_by_sequence[sequence]
                    for sequence in message.binding.sequence_numbers
                )
                expected_hashes = tuple(item.content_sha256 for item in selected)
                expected_transmitted_at = min(
                    item.first_send_time for item in selected
                )
                expected_dropped = any(item.dropped for item in selected)
                expected_arrival = (
                    None
                    if expected_dropped
                    else max(
                        item.first_arrival_time  # type: ignore[type-var]
                        for item in selected
                    )
                )
                expected_data_bytes = sum(
                    item.data_on_wire_bytes for item in selected
                )
                expected_ack_bytes = sum(
                    item.ack_on_wire_bytes for item in selected
                )
                if (
                    message.sequence_outcome_sha256s != expected_hashes
                    or message.transmitted_at != expected_transmitted_at
                    or message.dropped is not expected_dropped
                    or message.arrival_time != expected_arrival
                    or message.data_on_wire_bytes != expected_data_bytes
                    or message.ack_on_wire_bytes != expected_ack_bytes
                ):
                    raise C9MeasuredReplayError(
                        "application message is not derived from its sequence outcomes"
                    )
                if sum(
                    item.initial_data_on_wire_bytes for item in selected
                ) < message.binding.payload_bytes:
                    raise C9MeasuredReplayError(
                        "initial measured fragments are smaller than cache payload"
                    )
        object.__setattr__(self, "application_messages", messages)

    @property
    def application_bytes(self) -> int:
        return sum(item.binding.payload_bytes for item in self.application_messages)

    @property
    def data_on_wire_bytes(self) -> int:
        return sum(item.data_on_wire_bytes for item in self.sequence_outcomes)

    @property
    def ack_on_wire_bytes(self) -> int:
        return sum(item.ack_on_wire_bytes for item in self.sequence_outcomes)

    @property
    def total_on_wire_bytes(self) -> int:
        return self.data_on_wire_bytes + self.ack_on_wire_bytes

    def to_primitive(self) -> dict[str, object]:
        return {
            "application_messages": [
                item.to_primitive() for item in self.application_messages
            ],
            "capability": self.capability.value,
            "collection_protocol_sha256": self.collection_protocol_sha256,
            "kind": self.kind,
            "mapping_contract_sha256": self.mapping_contract_sha256,
            "packet_record_count": self.packet_record_count,
            "packet_trace_sha256": self.packet_trace_sha256,
            "receipt_id": self.receipt_id,
            "receipt_segment_sha256": self.receipt_segment_sha256,
            "receipt_sha256": self.receipt_sha256,
            "schema_version": self.schema_version,
            "sequence_outcomes": [
                item.to_primitive() for item in self.sequence_outcomes
            ],
            "supports_clock_claims": self.supports_clock_claims,
            "trace_id": self.trace_id,
            "trace_role": self.trace_role.value,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_primitive())

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> "C9MeasuredReplayV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        if (
            item["kind"] != "c9_measured_replay_v1"
            or item["schema_version"] != C9_MEASURED_REPLAY_SCHEMA_V1
        ):
            raise C9MeasuredReplayError("unsupported C9 measured replay schema")
        raw_outcomes = item["sequence_outcomes"]
        raw_messages = item["application_messages"]
        if not isinstance(raw_outcomes, list) or not isinstance(raw_messages, list):
            raise C9MeasuredReplayError("replay outcomes and messages must be arrays")
        return cls(
            receipt_id=item["receipt_id"],  # type: ignore[arg-type]
            receipt_sha256=item["receipt_sha256"],  # type: ignore[arg-type]
            collection_protocol_sha256=item["collection_protocol_sha256"],  # type: ignore[arg-type]
            trace_id=item["trace_id"],  # type: ignore[arg-type]
            trace_role=item["trace_role"],  # type: ignore[arg-type]
            receipt_segment_sha256=item["receipt_segment_sha256"],  # type: ignore[arg-type]
            packet_trace_sha256=item["packet_trace_sha256"],  # type: ignore[arg-type]
            packet_record_count=item["packet_record_count"],  # type: ignore[arg-type]
            supports_clock_claims=item["supports_clock_claims"],  # type: ignore[arg-type]
            capability=item["capability"],  # type: ignore[arg-type]
            sequence_outcomes=tuple(
                C9MeasuredSequenceOutcomeV1.from_mapping(raw)
                for raw in raw_outcomes
            ),
            application_messages=tuple(
                C9ApplicationMessageReplayV1.from_mapping(raw)
                for raw in raw_messages
            ),
        )


def _receipt_segment(
    receipt: MeasuredTraceReceiptV1,
    trace_id: str,
) -> MeasuredTraceSegmentV1:
    matches = tuple(item for item in receipt.segments if item.trace_id == trace_id)
    if len(matches) != 1:
        raise C9MeasuredReplayError(
            "packet trace must match exactly one receipt segment"
        )
    return matches[0]


def _sequence_outcomes(
    trace: MeasuredPacketTraceV1,
) -> tuple[C9MeasuredSequenceOutcomeV1, ...]:
    sequences = sorted({item.sequence_number for item in trace.records})
    outcomes: list[C9MeasuredSequenceOutcomeV1] = []
    for sequence in sequences:
        data = tuple(
            sorted(
                (
                    item
                    for item in trace.records
                    if item.sequence_number == sequence and not item.ack
                ),
                key=lambda item: (item.send_time, item.stable_order_key),
            )
        )
        acks = tuple(
            sorted(
                (
                    item
                    for item in trace.records
                    if item.sequence_number == sequence and item.ack
                ),
                key=lambda item: (item.send_time, item.stable_order_key),
            )
        )
        if not data:
            raise C9MeasuredReplayError("ACK-only sequence cannot be replayed")
        if any(item.packet_bytes != data[0].packet_bytes for item in data[1:]):
            raise C9MeasuredReplayError(
                "retransmitted data packet size changed for one sequence"
            )
        arrivals = tuple(
            item.arrival_time for item in data if item.arrival_time is not None
        )
        outcomes.append(
            C9MeasuredSequenceOutcomeV1(
                sequence_number=sequence,
                data_record_sha256s=tuple(_record_sha256(item) for item in data),
                ack_record_sha256s=tuple(_record_sha256(item) for item in acks),
                first_send_time=data[0].send_time,
                first_arrival_time=min(arrivals) if arrivals else None,
                dropped=not arrivals,
                data_attempts=len(data),
                delivered_data_attempts=len(arrivals),
                retransmissions=len(data) - 1,
                initial_data_on_wire_bytes=data[0].packet_bytes,
                data_on_wire_bytes=sum(item.packet_bytes for item in data),
                ack_transmissions=len(acks),
                delivered_ack_transmissions=sum(
                    item.arrival_time is not None for item in acks
                ),
                ack_on_wire_bytes=sum(item.packet_bytes for item in acks),
            )
        )
    return tuple(outcomes)


def build_c9_measured_replay_v1(
    trace: MeasuredPacketTraceV1,
    receipt: MeasuredTraceReceiptV1,
    *,
    bindings: tuple[C9ApplicationMessageBindingV1, ...] | None = None,
    caches_by_message_id: Mapping[str, DetectionCacheV1] | None = None,
) -> C9MeasuredReplayV1:
    """Build timing-only or payload-bound replay without inventing content."""

    if not isinstance(trace, MeasuredPacketTraceV1):
        raise TypeError("trace must be MeasuredPacketTraceV1")
    if not isinstance(receipt, MeasuredTraceReceiptV1):
        raise TypeError("receipt must be MeasuredTraceReceiptV1")
    segment = _receipt_segment(receipt, trace.trace_id)
    try:
        validate_measured_packet_trace_segment_v1(trace, segment)
    except (TypeError, ValueError) as exc:
        raise C9MeasuredReplayError(
            "packet trace is not bound to the receipt segment"
        ) from exc
    outcomes = _sequence_outcomes(trace)

    if bindings is None and caches_by_message_id is None:
        messages: tuple[C9ApplicationMessageReplayV1, ...] = ()
        capability = C9ReplayCapability.NETWORK_TIMING_ONLY
    elif bindings is None or caches_by_message_id is None:
        raise C9MeasuredReplayError(
            "application replay requires both bindings and detection caches"
        )
    else:
        if not isinstance(bindings, (list, tuple)) or not all(
            isinstance(item, C9ApplicationMessageBindingV1) for item in bindings
        ):
            raise TypeError("bindings must contain C9ApplicationMessageBindingV1")
        if not isinstance(caches_by_message_id, Mapping) or not all(
            type(key) is str for key in caches_by_message_id
        ):
            raise TypeError("caches_by_message_id must be a string-keyed mapping")
        cache_map = dict(caches_by_message_id)
        binding_ids = tuple(item.message_id for item in bindings)
        if len(binding_ids) != len(set(binding_ids)):
            raise C9MeasuredReplayError("binding message IDs must be unique")
        if set(cache_map) != set(binding_ids):
            raise C9MeasuredReplayError(
                "cache mapping must contain exactly the bound message IDs"
            )
        if not all(isinstance(item, DetectionCacheV1) for item in cache_map.values()):
            raise TypeError("cache mapping must contain DetectionCacheV1 values")
        outcome_by_sequence = {
            item.sequence_number: item for item in outcomes
        }
        messages_list: list[C9ApplicationMessageReplayV1] = []
        for binding in bindings:
            if any(
                sequence not in outcome_by_sequence
                for sequence in binding.sequence_numbers
            ):
                raise C9MeasuredReplayError(
                    "binding refers to an absent measured packet sequence"
                )
            cache = cache_map[binding.message_id]
            payload = _cache_bytes(cache)
            payload_sha256 = hashlib.sha256(payload).hexdigest()
            if (
                cache.digest() != binding.detection_cache_sha256
                or payload_sha256 != binding.payload_sha256
                or len(payload) != binding.payload_bytes
            ):
                raise C9MeasuredReplayError(
                    "binding does not match exact detection cache payload bytes"
                )
            if cache.agent_id != binding.source:
                raise C9MeasuredReplayError("binding source does not match cache agent")
            selected = tuple(
                outcome_by_sequence[sequence]
                for sequence in binding.sequence_numbers
            )
            transmitted_at = min(item.first_send_time for item in selected)
            if cache.event_time > transmitted_at:
                raise C9MeasuredReplayError(
                    "future detection cache cannot be transmitted"
                )
            if binding.deadline < transmitted_at:
                raise C9MeasuredReplayError(
                    "binding deadline cannot precede packet transmission"
                )
            initial_wire_bytes = sum(
                item.initial_data_on_wire_bytes for item in selected
            )
            if initial_wire_bytes < binding.payload_bytes:
                raise C9MeasuredReplayError(
                    "initial measured fragments are smaller than cache payload"
                )
            dropped = any(item.dropped for item in selected)
            arrival_time = (
                None
                if dropped
                else max(
                    item.first_arrival_time  # type: ignore[type-var]
                    for item in selected
                )
            )
            messages_list.append(
                C9ApplicationMessageReplayV1(
                    binding=binding,
                    detection_cache_sequence_id=cache.sequence_id,
                    detection_cache_frame_id=cache.frame_id,
                    detection_cache_event_time=cache.event_time,
                    sequence_outcome_sha256s=tuple(
                        item.content_sha256 for item in selected
                    ),
                    transmitted_at=transmitted_at,
                    arrival_time=arrival_time,
                    dropped=dropped,
                    on_time=(
                        arrival_time is not None
                        and arrival_time <= binding.deadline
                    ),
                    data_on_wire_bytes=sum(
                        item.data_on_wire_bytes for item in selected
                    ),
                    ack_on_wire_bytes=sum(
                        item.ack_on_wire_bytes for item in selected
                    ),
                )
            )
        messages = tuple(
            sorted(
                messages_list,
                key=lambda item: (
                    item.transmitted_at,
                    item.binding.source,
                    item.binding.message_id,
                ),
            )
        )
        capability = C9ReplayCapability.APPLICATION_MESSAGES

    return C9MeasuredReplayV1(
        receipt_id=receipt.receipt_id,
        receipt_sha256=receipt.content_sha256,
        collection_protocol_sha256=receipt.collection_protocol_sha256,
        trace_id=trace.trace_id,
        trace_role=segment.role,
        receipt_segment_sha256=_segment_sha256(segment),
        packet_trace_sha256=trace.content_sha256,
        packet_record_count=len(trace.records),
        supports_clock_claims=trace.supports_clock_claims,
        capability=capability,
        sequence_outcomes=outcomes,
        application_messages=messages,
    )


def decode_c9_measured_replay_v1(data: bytes) -> C9MeasuredReplayV1:
    """Cold-read exact canonical replay JSON and re-run local invariants."""

    if type(data) is not bytes:
        raise TypeError("C9 measured replay data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise C9MeasuredReplayError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                C9MeasuredReplayError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C9MeasuredReplayError("invalid C9 measured replay JSON") from exc
    try:
        replay = C9MeasuredReplayV1.from_mapping(value)
    except C9MeasuredReplayError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise C9MeasuredReplayError(f"invalid C9 measured replay content: {exc}") from exc
    if replay.canonical_bytes != data:
        raise C9MeasuredReplayError("C9 measured replay JSON is not canonical")
    return replay


def validate_c9_measured_replay_v1(
    replay: C9MeasuredReplayV1,
    trace: MeasuredPacketTraceV1,
    receipt: MeasuredTraceReceiptV1,
    *,
    caches_by_message_id: Mapping[str, DetectionCacheV1] | None = None,
) -> None:
    """Open the receipt/packet/cache inputs and reproduce the exact artifact."""

    if not isinstance(replay, C9MeasuredReplayV1):
        raise TypeError("replay must be C9MeasuredReplayV1")
    if replay.capability is C9ReplayCapability.NETWORK_TIMING_ONLY:
        if caches_by_message_id is not None:
            raise C9MeasuredReplayError(
                "timing-only replay validation must not imply payload capability"
            )
        rebuilt = build_c9_measured_replay_v1(trace, receipt)
    else:
        if caches_by_message_id is None:
            raise C9MeasuredReplayError(
                "application replay validation requires exact detection caches"
            )
        rebuilt = build_c9_measured_replay_v1(
            trace,
            receipt,
            bindings=tuple(item.binding for item in replay.application_messages),
            caches_by_message_id=caches_by_message_id,
        )
    if rebuilt.content_sha256 != replay.content_sha256:
        raise C9MeasuredReplayError(
            "C9 measured replay does not reproduce from receipt-bound inputs"
        )


def application_network_events_v1(
    replay: C9MeasuredReplayV1,
) -> tuple[NetworkEvent, ...]:
    """Map payload-bound C9 messages to de-duplicated reference events.

    This function deliberately returns application ``NetworkEvent`` values,
    not a synthetic ``NetworkTraceV1``.  The measured on-wire ledger remains
    in ``replay`` and must be reported from there.
    """

    if not isinstance(replay, C9MeasuredReplayV1):
        raise TypeError("replay must be C9MeasuredReplayV1")
    if replay.capability is not C9ReplayCapability.APPLICATION_MESSAGES:
        raise C9MeasuredReplayError(
            "packet metadata alone cannot produce application messages"
        )
    return tuple(
        NetworkEvent(
            packet=PacketRequest(
                message_id=item.binding.message_id,
                source=item.binding.source,
                sequence=min(item.binding.sequence_numbers),
                transmitted_at=item.transmitted_at,
                deadline=item.binding.deadline,
                encoded_bytes=item.binding.payload_bytes,
            ),
            arrival_time=item.arrival_time,
            dropped=item.dropped,
            channel_state=None,
        )
        for item in replay.application_messages
    )


__all__ = [
    "C9ApplicationMessageBindingV1",
    "C9ApplicationMessageReplayV1",
    "C9_MEASURED_REPLAY_MAPPING_SCHEMA_V1",
    "C9_MEASURED_REPLAY_MAPPING_SHA256_V1",
    "C9_MEASURED_REPLAY_SCHEMA_V1",
    "C9MeasuredReplayError",
    "C9MeasuredReplayV1",
    "C9MeasuredSequenceOutcomeV1",
    "C9ReplayCapability",
    "application_network_events_v1",
    "build_c9_measured_replay_v1",
    "decode_c9_measured_replay_v1",
    "validate_c9_measured_replay_v1",
]
