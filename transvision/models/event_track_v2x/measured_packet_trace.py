"""Strict, privacy-minimal artifact contract for measured C9 packet metadata.

The artifact contains only the preregistered packet fields plus the minimum
trace header needed to bind a 600-second collection segment.  Images, precise
coordinates, device identifiers, and every other unknown field are rejected.
This module validates metadata artifacts; it does not implement C9 replay or
claim that measured data have been collected.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping

import numpy as np

from .measured_trace import (
    REQUIRED_PACKET_FIELDS_V1,
    ClockReference,
    MeasuredTraceSegmentV1,
)
from .wire import canonical_json_bytes


MEASURED_PACKET_TRACE_SCHEMA_V1 = "eventtrack-v2x.measured-packet-trace.v1"
MEASURED_PACKET_COLLECTION_SECONDS_V1 = 600.0
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


class MeasuredPacketTraceError(ValueError):
    """Raised when measured packet metadata are unsafe or inconsistent."""


class SynchronizationStatus(str, Enum):
    """Closed synchronization state recorded for each packet event."""

    SYNCHRONIZED = "synchronized"
    UNSYNCHRONIZED = "unsynchronized"


_TRUSTED_CLOCK_REFERENCES_V1 = frozenset(
    {
        ClockReference.GNSS_PPS,
        ClockReference.PTP,
        ClockReference.OTHER_VERIFIED,
    }
)


def _strict_fields(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise MeasuredPacketTraceError(f"{name} must be a string-keyed object")
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise MeasuredPacketTraceError(
            f"{name} fields do not match schema; missing={missing}, unknown={unknown}"
        )
    return value


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise MeasuredPacketTraceError(f"{name} must be a canonical identifier")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasuredPacketTraceError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise MeasuredPacketTraceError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _wall_time(value: object) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise MeasuredPacketTraceError("wall_time must be canonical RFC3339 UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise MeasuredPacketTraceError(
            "wall_time must be canonical RFC3339 UTC"
        ) from exc
    if parsed.isoformat().replace("+00:00", "Z") != value:
        raise MeasuredPacketTraceError("wall_time must be canonical RFC3339 UTC")
    return value


@dataclass(frozen=True, slots=True)
class MeasuredPacketRecordV1:
    """One record containing exactly the ten preregistered packet fields."""

    monotonic_time: float
    wall_time: str
    sequence_number: int
    send_time: float
    arrival_time: float | None
    packet_bytes: int
    loss: bool
    retransmission: bool
    ack: bool
    synchronization_status: SynchronizationStatus

    _FIELDS = frozenset(REQUIRED_PACKET_FIELDS_V1)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "monotonic_time", _finite(self.monotonic_time, "monotonic_time")
        )
        object.__setattr__(self, "wall_time", _wall_time(self.wall_time))
        if type(self.sequence_number) is not int or self.sequence_number < 0:
            raise MeasuredPacketTraceError(
                "sequence_number must be a non-negative integer"
            )
        object.__setattr__(self, "send_time", _finite(self.send_time, "send_time"))
        if self.arrival_time is not None:
            arrival = _finite(self.arrival_time, "arrival_time")
            if arrival < self.send_time:
                raise MeasuredPacketTraceError("arrival_time cannot precede send_time")
            object.__setattr__(self, "arrival_time", arrival)
        if type(self.packet_bytes) is not int or self.packet_bytes <= 0:
            raise MeasuredPacketTraceError("packet_bytes must be a positive integer")
        for name in ("loss", "retransmission", "ack"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.loss != (self.arrival_time is None):
            raise MeasuredPacketTraceError(
                "loss must be true exactly when arrival_time is absent"
            )
        if self.ack and self.retransmission:
            raise MeasuredPacketTraceError(
                "ACK records cannot be marked as data retransmissions"
            )
        try:
            synchronization_status = SynchronizationStatus(
                self.synchronization_status
            )
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(item.value for item in SynchronizationStatus)
            raise MeasuredPacketTraceError(
                "synchronization_status must be one of: " + allowed
            ) from exc
        object.__setattr__(
            self,
            "synchronization_status",
            synchronization_status,
        )

    @property
    def stable_order_key(self) -> tuple[object, ...]:
        return (
            self.monotonic_time,
            self.sequence_number,
            self.ack,
            self.retransmission,
            self.send_time,
            self.arrival_time is None,
            0.0 if self.arrival_time is None else self.arrival_time,
            self.packet_bytes,
            self.loss,
            self.wall_time,
            self.synchronization_status.value,
        )

    def to_primitive(self) -> dict[str, object]:
        return {
            "ack": self.ack,
            "arrival_time": self.arrival_time,
            "loss": self.loss,
            "monotonic_time": self.monotonic_time,
            "packet_bytes": self.packet_bytes,
            "retransmission": self.retransmission,
            "send_time": self.send_time,
            "sequence_number": self.sequence_number,
            "synchronization_status": self.synchronization_status.value,
            "wall_time": self.wall_time,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "MeasuredPacketRecordV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        return cls(
            monotonic_time=item["monotonic_time"],  # type: ignore[arg-type]
            wall_time=item["wall_time"],  # type: ignore[arg-type]
            sequence_number=item["sequence_number"],  # type: ignore[arg-type]
            send_time=item["send_time"],  # type: ignore[arg-type]
            arrival_time=item["arrival_time"],  # type: ignore[arg-type]
            packet_bytes=item["packet_bytes"],  # type: ignore[arg-type]
            loss=item["loss"],  # type: ignore[arg-type]
            retransmission=item["retransmission"],  # type: ignore[arg-type]
            ack=item["ack"],  # type: ignore[arg-type]
            synchronization_status=item["synchronization_status"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class MeasuredPacketTraceV1:
    """Canonical packet-metadata artifact for one exact 600-second segment."""

    trace_id: str
    collection_start_monotonic_time: float
    collection_end_monotonic_time: float
    clock_reference: ClockReference
    records: tuple[MeasuredPacketRecordV1, ...]

    _FIELDS = frozenset(
        {
            "clock_reference",
            "collection_end_monotonic_time",
            "collection_start_monotonic_time",
            "kind",
            "records",
            "schema_version",
            "trace_id",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_id", _identifier(self.trace_id, "trace_id"))
        start = _finite(
            self.collection_start_monotonic_time,
            "collection_start_monotonic_time",
        )
        end = _finite(
            self.collection_end_monotonic_time,
            "collection_end_monotonic_time",
        )
        if end - start != MEASURED_PACKET_COLLECTION_SECONDS_V1:
            raise MeasuredPacketTraceError(
                "measured packet collection window must equal exactly 600 seconds"
            )
        object.__setattr__(self, "collection_start_monotonic_time", start)
        object.__setattr__(self, "collection_end_monotonic_time", end)
        object.__setattr__(self, "clock_reference", ClockReference(self.clock_reference))
        if not isinstance(self.records, (list, tuple)) or not all(
            isinstance(item, MeasuredPacketRecordV1) for item in self.records
        ):
            raise TypeError("records must contain MeasuredPacketRecordV1")
        records = tuple(self.records)
        if not records:
            raise MeasuredPacketTraceError("records must be non-empty")
        keys = tuple(item.stable_order_key for item in records)
        if keys != tuple(sorted(keys)):
            raise MeasuredPacketTraceError(
                "records must use non-decreasing monotonic time and stable tie-break"
            )
        if len(keys) != len(set(keys)):
            raise MeasuredPacketTraceError("duplicate packet records are forbidden")
        if any(
            not start <= item.monotonic_time <= end
            or not start <= item.send_time <= end
            or (
                item.arrival_time is not None
                and not start <= item.arrival_time <= end
            )
            for item in records
        ):
            raise MeasuredPacketTraceError(
                "packet record times must remain inside the 600-second window"
            )
        self._validate_protocol(records)
        object.__setattr__(self, "records", records)

    @staticmethod
    def _validate_protocol(records: tuple[MeasuredPacketRecordV1, ...]) -> None:
        data_attempts: dict[int, list[MeasuredPacketRecordV1]] = {}
        acknowledgements: dict[int, list[MeasuredPacketRecordV1]] = {}
        for record in records:
            sequence = record.sequence_number
            if record.ack:
                acknowledgements.setdefault(sequence, []).append(record)
            else:
                data_attempts.setdefault(sequence, []).append(record)

        for sequence in sorted(set(data_attempts) | set(acknowledgements)):
            attempts = tuple(
                sorted(
                    data_attempts.get(sequence, ()),
                    key=lambda item: (item.send_time, item.stable_order_key),
                )
            )
            acks = tuple(acknowledgements.get(sequence, ()))
            if acks and not attempts:
                raise MeasuredPacketTraceError(
                    "ACK requires a previously delivered data attempt"
                )
            send_times = tuple(item.send_time for item in attempts)
            if len(send_times) != len(set(send_times)):
                raise MeasuredPacketTraceError(
                    "data attempts for one sequence require unique send_time values"
                )
            for attempt_index, attempt in enumerate(attempts):
                if attempt.retransmission != (attempt_index > 0):
                    raise MeasuredPacketTraceError(
                        "first data attempt must not be retransmission and later "
                        "attempts must be"
                    )

            delivered_data = tuple(
                item for item in attempts if item.arrival_time is not None
            )
            for ack in acks:
                if not any(
                    item.arrival_time is not None
                    and item.arrival_time <= ack.send_time
                    for item in delivered_data
                ):
                    raise MeasuredPacketTraceError(
                        "ACK requires a previously delivered data attempt"
                    )

            delivered_acks = tuple(
                item for item in acks if item.arrival_time is not None
            )
            for attempt in attempts[1:]:
                if any(
                    ack.arrival_time is not None
                    and ack.arrival_time <= attempt.send_time
                    for ack in delivered_acks
                ):
                    raise MeasuredPacketTraceError(
                        "data cannot retransmit after a delivered ACK"
                    )

    @property
    def supports_clock_claims(self) -> bool:
        return self.clock_reference in _TRUSTED_CLOCK_REFERENCES_V1 and all(
            item.synchronization_status is SynchronizationStatus.SYNCHRONIZED
            for item in self.records
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "clock_reference": self.clock_reference.value,
            "collection_end_monotonic_time": self.collection_end_monotonic_time,
            "collection_start_monotonic_time": (
                self.collection_start_monotonic_time
            ),
            "kind": "measured_packet_trace_v1",
            "records": [item.to_primitive() for item in self.records],
            "schema_version": MEASURED_PACKET_TRACE_SCHEMA_V1,
            "trace_id": self.trace_id,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_primitive())

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> "MeasuredPacketTraceV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        if (
            item["kind"] != "measured_packet_trace_v1"
            or item["schema_version"] != MEASURED_PACKET_TRACE_SCHEMA_V1
        ):
            raise MeasuredPacketTraceError("unsupported measured packet trace schema")
        raw_records = item["records"]
        if not isinstance(raw_records, list):
            raise MeasuredPacketTraceError("records must be an array")
        return cls(
            trace_id=item["trace_id"],  # type: ignore[arg-type]
            collection_start_monotonic_time=item[  # type: ignore[arg-type]
                "collection_start_monotonic_time"
            ],
            collection_end_monotonic_time=item[  # type: ignore[arg-type]
                "collection_end_monotonic_time"
            ],
            clock_reference=item["clock_reference"],  # type: ignore[arg-type]
            records=tuple(
                MeasuredPacketRecordV1.from_mapping(record)
                for record in raw_records
            ),
        )


def decode_measured_packet_trace_v1(data: bytes) -> MeasuredPacketTraceV1:
    """Decode exact canonical JSON while rejecting duplicates and NaN/Inf."""

    if type(data) is not bytes:
        raise TypeError("measured packet trace data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise MeasuredPacketTraceError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                MeasuredPacketTraceError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeasuredPacketTraceError("invalid measured packet trace JSON") from exc
    trace = MeasuredPacketTraceV1.from_mapping(value)
    if trace.canonical_bytes != data:
        raise MeasuredPacketTraceError("measured packet trace JSON is not canonical")
    return trace


def validate_measured_packet_trace_segment_v1(
    trace: MeasuredPacketTraceV1,
    segment: MeasuredTraceSegmentV1,
) -> None:
    """Require exact identity/hash/count/schema/capability agreement with receipt."""

    if not isinstance(trace, MeasuredPacketTraceV1):
        raise TypeError("trace must be MeasuredPacketTraceV1")
    if not isinstance(segment, MeasuredTraceSegmentV1):
        raise TypeError("segment must be MeasuredTraceSegmentV1")
    expected = (
        trace.trace_id,
        int(MEASURED_PACKET_COLLECTION_SECONDS_V1),
        len(trace.records),
        REQUIRED_PACKET_FIELDS_V1,
        trace.clock_reference,
        trace.content_sha256,
        trace.supports_clock_claims,
    )
    observed = (
        segment.trace_id,
        segment.duration_seconds,
        segment.packet_count,
        segment.packet_fields,
        segment.clock_reference,
        segment.packet_metadata_sha256,
        segment.supports_clock_claims,
    )
    if observed != expected:
        raise MeasuredPacketTraceError(
            "measured packet trace does not match receipt segment"
        )


__all__ = [
    "MEASURED_PACKET_COLLECTION_SECONDS_V1",
    "MEASURED_PACKET_TRACE_SCHEMA_V1",
    "MeasuredPacketRecordV1",
    "MeasuredPacketTraceError",
    "MeasuredPacketTraceV1",
    "SynchronizationStatus",
    "decode_measured_packet_trace_v1",
    "validate_measured_packet_trace_segment_v1",
]
