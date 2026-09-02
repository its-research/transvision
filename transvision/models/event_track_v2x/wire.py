"""Canonical, deterministic EventTrack-V2X wire encoding."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .schema import (
    CorrelatedTrackBelief,
    IndependentIncrement,
    Lineage,
    LocalTimestamps,
    WireMessage,
)


SCHEMA_VERSION = 1


def _number(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("canonical encoding rejects non-finite numbers")
    return 0.0 if value == 0.0 else value


def _array(value: np.ndarray) -> list[Any]:
    return np.asarray(value, dtype=np.float64).tolist()


def _lineage(value: Lineage) -> dict[str, Any]:
    return {
        "ancestor_message_ids": list(value.ancestor_message_ids),
        "complete": value.complete,
        "factor_ids": list(value.factor_ids),
    }


def message_to_primitive(message: WireMessage) -> dict[str, Any]:
    """Convert a message to the versioned canonical wire object."""

    timestamps = message.timestamps
    common: dict[str, Any] = {
        "coordinate_frame": message.coordinate_frame,
        "deadline": _number(message.deadline),
        "message_id": message.message_id,
        "schema_version": SCHEMA_VERSION,
        "sequence": message.sequence,
        "source": message.source,
        "timestamps": {
            "generated": _number(timestamps.generated),
            "information_cutoff": _number(timestamps.information_cutoff),
            "state_reference": _number(timestamps.state_reference),
            "transmitted": _number(timestamps.transmitted),
        },
        "ttl": _number(message.ttl),
    }
    payload = message.payload
    if isinstance(payload, IndependentIncrement):
        common["payload"] = {
            "kind": payload.kind,
            "lineage": _lineage(payload.lineage),
            "log_normalizer": _number(payload.log_normalizer),
            "measurement": _array(payload.measurement),
            "measurement_covariance": _array(payload.measurement_covariance),
            "measurement_matrix": _array(payload.measurement_matrix),
            "target_id": payload.target_id,
        }
    elif isinstance(payload, CorrelatedTrackBelief):
        common["payload"] = {
            "covariance": _array(payload.covariance),
            "existence_probability": _number(payload.existence_probability),
            "identity_probabilities": [
                [label, _number(probability)]
                for label, probability in payload.identity_probabilities
            ],
            "kind": payload.kind,
            "lineage": _lineage(payload.lineage),
            "mean": _array(payload.mean),
            "track_id": payload.track_id,
        }
    else:  # pragma: no cover - WireMessage prevents this state.
        raise TypeError(f"unsupported payload: {type(payload)!r}")
    return common


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON-compatible data with stable keys and no insignificant bytes."""

    def normalize(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return normalize(item.tolist())
        if isinstance(item, np.generic):
            return normalize(item.item())
        if isinstance(item, float):
            return _number(item)
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise TypeError("canonical JSON object keys must be strings")
            return {key: normalize(item[key]) for key in sorted(item)}
        if isinstance(item, (list, tuple)):
            return [normalize(element) for element in item]
        if item is None or isinstance(item, (str, int, bool)):
            return item
        raise TypeError(f"unsupported canonical JSON value: {type(item)!r}")

    return json.dumps(
        normalize(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def encode_message(message: WireMessage) -> bytes:
    """Return the exact bytes charged to the communication budget."""

    return canonical_json_bytes(message_to_primitive(message))


def wire_size(message: WireMessage) -> int:
    return len(encode_message(message))


def wire_digest(message: WireMessage) -> str:
    return hashlib.sha256(encode_message(message)).hexdigest()


def _decode_lineage(value: dict[str, Any]) -> Lineage:
    return Lineage(
        factor_ids=tuple(value["factor_ids"]),
        complete=value["complete"],
        ancestor_message_ids=tuple(value["ancestor_message_ids"]),
    )


def decode_message(data: bytes) -> WireMessage:
    """Decode canonical bytes and reject non-canonical or unknown schemas."""

    if not isinstance(data, bytes):
        raise TypeError("wire data must be bytes")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid EventTrack-V2X wire message") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported EventTrack-V2X schema version")
    timestamps = LocalTimestamps(**value["timestamps"])
    payload_value = value["payload"]
    lineage = _decode_lineage(payload_value["lineage"])
    if payload_value["kind"] == "independent_increment":
        payload = IndependentIncrement(
            target_id=payload_value["target_id"],
            measurement=np.asarray(payload_value["measurement"], dtype=np.float64),
            measurement_matrix=np.asarray(
                payload_value["measurement_matrix"], dtype=np.float64
            ),
            measurement_covariance=np.asarray(
                payload_value["measurement_covariance"], dtype=np.float64
            ),
            lineage=lineage,
            log_normalizer=payload_value["log_normalizer"],
        )
    elif payload_value["kind"] == "correlated_track_belief":
        payload = CorrelatedTrackBelief(
            track_id=payload_value["track_id"],
            existence_probability=payload_value["existence_probability"],
            mean=np.asarray(payload_value["mean"], dtype=np.float64),
            covariance=np.asarray(payload_value["covariance"], dtype=np.float64),
            identity_probabilities=tuple(
                (label, probability)
                for label, probability in payload_value["identity_probabilities"]
            ),
            lineage=lineage,
        )
    else:
        raise ValueError("unknown EventTrack-V2X payload kind")
    message = WireMessage(
        message_id=value["message_id"],
        source=value["source"],
        sequence=value["sequence"],
        timestamps=timestamps,
        deadline=value["deadline"],
        coordinate_frame=value["coordinate_frame"],
        ttl=value["ttl"],
        payload=payload,
    )
    if encode_message(message) != data:
        raise ValueError("wire message is valid JSON but not canonical")
    return message


__all__ = [
    "SCHEMA_VERSION",
    "canonical_json_bytes",
    "decode_message",
    "encode_message",
    "message_to_primitive",
    "wire_digest",
    "wire_size",
]
