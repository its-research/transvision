"""Privacy-preserving contract for held-out measured V2X link traces.

The contract stores only identifiers, coverage strata, packet-metadata hashes,
and collection capabilities.  Raw road images, coordinates, and device
identifiers are forbidden.  It therefore proves the identity and minimum
coverage of a C9 replay cohort without turning the paper evidence bundle into
a location-data archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping

from .wire import canonical_json_bytes


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
REQUIRED_PACKET_FIELDS_V1 = (
    "ack",
    "arrival_time",
    "loss",
    "monotonic_time",
    "packet_bytes",
    "retransmission",
    "send_time",
    "sequence_number",
    "synchronization_status",
    "wall_time",
)


class MeasuredTraceError(ValueError):
    """Raised when a measured-link receipt cannot support the C9 protocol."""


class TraceRole(str, Enum):
    SETUP = "setup"
    HELD_OUT = "held_out"


class ClockReference(str, Enum):
    NONE = "none"
    GNSS_PPS = "gnss_pps"
    PTP = "ptp"
    OTHER_VERIFIED = "other_verified"


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise MeasuredTraceError(f"{name} must be a canonical identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise MeasuredTraceError(f"{name} must be a lowercase SHA-256")
    return value


def _strict_mapping(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise MeasuredTraceError(f"{name} must be a string-keyed object")
    if frozenset(value) != expected:
        raise MeasuredTraceError(f"{name} has missing or unknown fields")
    return value


@dataclass(frozen=True, slots=True)
class MeasuredTraceSegmentV1:
    """One ten-minute trace segment represented by a metadata-file digest."""

    trace_id: str
    role: TraceRole
    date_cohort_id: str
    road_coverage_type: str
    duration_seconds: int
    packet_count: int
    packet_metadata_sha256: str
    packet_fields: tuple[str, ...]
    clock_reference: ClockReference
    contains_road_images: bool = False
    contains_precise_coordinates: bool = False
    contains_device_identifiers: bool = False

    _FIELDS = frozenset(
        {
            "clock_reference",
            "contains_device_identifiers",
            "contains_precise_coordinates",
            "contains_road_images",
            "date_cohort_id",
            "duration_seconds",
            "packet_count",
            "packet_fields",
            "packet_metadata_sha256",
            "road_coverage_type",
            "role",
            "trace_id",
        }
    )

    def __post_init__(self) -> None:
        for name in ("trace_id", "date_cohort_id", "road_coverage_type"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "role", TraceRole(self.role))
        object.__setattr__(
            self, "clock_reference", ClockReference(self.clock_reference)
        )
        if type(self.duration_seconds) is not int or self.duration_seconds != 600:
            raise MeasuredTraceError("measured trace duration_seconds must equal 600")
        if type(self.packet_count) is not int or self.packet_count <= 0:
            raise MeasuredTraceError("packet_count must be a positive integer")
        object.__setattr__(
            self,
            "packet_metadata_sha256",
            _sha256(self.packet_metadata_sha256, "packet_metadata_sha256"),
        )
        if not isinstance(self.packet_fields, (list, tuple)):
            raise MeasuredTraceError("packet_fields must be an array")
        fields = tuple(_identifier(item, "packet field") for item in self.packet_fields)
        if fields != REQUIRED_PACKET_FIELDS_V1:
            raise MeasuredTraceError(
                "packet_fields must equal the preregistered packet metadata schema"
            )
        object.__setattr__(self, "packet_fields", fields)
        for name in (
            "contains_road_images",
            "contains_precise_coordinates",
            "contains_device_identifiers",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
            if getattr(self, name):
                raise MeasuredTraceError(
                    "C9 receipt must not contain images, precise coordinates, "
                    "or device identifiers"
                )

    @property
    def supports_clock_claims(self) -> bool:
        return self.clock_reference is not ClockReference.NONE

    def to_primitive(self) -> dict[str, object]:
        return {
            "clock_reference": self.clock_reference.value,
            "contains_device_identifiers": self.contains_device_identifiers,
            "contains_precise_coordinates": self.contains_precise_coordinates,
            "contains_road_images": self.contains_road_images,
            "date_cohort_id": self.date_cohort_id,
            "duration_seconds": self.duration_seconds,
            "packet_count": self.packet_count,
            "packet_fields": list(self.packet_fields),
            "packet_metadata_sha256": self.packet_metadata_sha256,
            "road_coverage_type": self.road_coverage_type,
            "role": self.role.value,
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "MeasuredTraceSegmentV1":
        result = _strict_mapping(value, cls._FIELDS, cls.__name__)
        return cls(**result)


@dataclass(frozen=True, slots=True)
class MeasuredTraceReceiptV1:
    """Sealed minimum-coverage receipt for the 10/20 setup/test split."""

    receipt_id: str
    segments: tuple[MeasuredTraceSegmentV1, ...]
    collection_protocol_sha256: str

    _FIELDS = frozenset(
        {
            "collection_protocol_sha256",
            "kind",
            "receipt_id",
            "schema_version",
            "segments",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_id", _identifier(self.receipt_id, "receipt_id"))
        object.__setattr__(
            self,
            "collection_protocol_sha256",
            _sha256(
                self.collection_protocol_sha256, "collection_protocol_sha256"
            ),
        )
        if not isinstance(self.segments, (list, tuple)) or not all(
            isinstance(item, MeasuredTraceSegmentV1) for item in self.segments
        ):
            raise TypeError("segments must contain MeasuredTraceSegmentV1 values")
        segments = tuple(sorted(self.segments, key=lambda item: item.trace_id))
        if len(segments) != 30 or len({item.trace_id for item in segments}) != 30:
            raise MeasuredTraceError("C9 receipt requires 30 unique trace segments")
        setup = tuple(item for item in segments if item.role is TraceRole.SETUP)
        held_out = tuple(item for item in segments if item.role is TraceRole.HELD_OUT)
        if len(setup) != 10 or len(held_out) != 20:
            raise MeasuredTraceError("C9 receipt requires 10 setup and 20 held-out traces")
        if len({item.date_cohort_id for item in segments}) < 3:
            raise MeasuredTraceError("C9 receipt requires at least three date cohorts")
        if len({item.road_coverage_type for item in segments}) < 3:
            raise MeasuredTraceError(
                "C9 receipt requires at least three road or coverage types"
            )
        object.__setattr__(self, "segments", segments)

    @property
    def held_out_trace_ids(self) -> tuple[str, ...]:
        return tuple(
            item.trace_id
            for item in self.segments
            if item.role is TraceRole.HELD_OUT
        )

    @property
    def held_out_packet_metadata_sha256s(self) -> tuple[tuple[str, str], ...]:
        """Return the exact replay artifact bound to every held-out trace ID."""

        return tuple(
            (item.trace_id, item.packet_metadata_sha256)
            for item in self.segments
            if item.role is TraceRole.HELD_OUT
        )

    @property
    def setup_trace_ids(self) -> tuple[str, ...]:
        return tuple(
            item.trace_id for item in self.segments if item.role is TraceRole.SETUP
        )

    @property
    def supports_clock_claims(self) -> bool:
        return all(item.supports_clock_claims for item in self.segments)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "collection_protocol_sha256": self.collection_protocol_sha256,
            "kind": "measured_trace_receipt_v1",
            "receipt_id": self.receipt_id,
            "schema_version": 1,
            "segments": [item.to_primitive() for item in self.segments],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_primitive())

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> "MeasuredTraceReceiptV1":
        result = _strict_mapping(value, cls._FIELDS, cls.__name__)
        if result["kind"] != "measured_trace_receipt_v1" or result["schema_version"] != 1:
            raise MeasuredTraceError("unsupported measured trace receipt schema")
        raw_segments = result["segments"]
        if not isinstance(raw_segments, list):
            raise MeasuredTraceError("segments must be an array")
        return cls(
            receipt_id=result["receipt_id"],
            collection_protocol_sha256=result["collection_protocol_sha256"],
            segments=tuple(
                MeasuredTraceSegmentV1.from_mapping(item) for item in raw_segments
            ),
        )


def decode_measured_trace_receipt(data: bytes) -> MeasuredTraceReceiptV1:
    """Decode canonical JSON while rejecting duplicate keys and NaN values."""

    if type(data) is not bytes:
        raise TypeError("measured trace receipt data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise MeasuredTraceError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                MeasuredTraceError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeasuredTraceError("invalid measured trace receipt JSON") from exc
    receipt = MeasuredTraceReceiptV1.from_mapping(value)
    if receipt.canonical_bytes != data:
        raise MeasuredTraceError("measured trace receipt JSON is not canonical")
    return receipt


__all__ = [
    "ClockReference",
    "MeasuredTraceError",
    "MeasuredTraceReceiptV1",
    "MeasuredTraceSegmentV1",
    "REQUIRED_PACKET_FIELDS_V1",
    "TraceRole",
    "decode_measured_trace_receipt",
]
