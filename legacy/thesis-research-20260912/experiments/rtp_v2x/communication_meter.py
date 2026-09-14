#!/usr/bin/env python3
"""Dependency-light, fail-closed communication measurement for RTP-V2X.

The module measures bytes produced by the actual encoder, rather than tensor
shapes or sparsity estimates.  The generic callback runner is deliberately
diagnostic-only: a future real-data runner must register and audit transport,
timer, synchronization, consumer, and frozen-frame adapters separately.  This
module never writes to ``results/registry.json`` or grants a scientific claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROTOCOL_ID = "RTPV2X-COMM-MEASURE-v1"
SCHEMA_VERSION = 1
EVIDENCE_TIERS = frozenset({"diagnostic_only", "candidate_real"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
WIRE_SCOPE = "application_message_bytes_including_protocol_headers"
DIAGNOSTIC_WIRE_SCOPE = "diagnostic_canonical_json_bytes"
REAL_PIPELINE_SCOPE = "encode_transport_decode_consumer"
REAL_CONTEXT_FIELDS = frozenset(
    {
        "code_commit",
        "config_sha256",
        "dataset_identity_sha256",
        "pipeline_scope",
        "split_sha256",
        "transport_id",
        "wire_scope",
    }
)
REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "evidence_tier",
        "scientific_claim_allowed",
        "context",
        "measurement",
        "metrics",
        "frame_manifest",
        "samples",
    }
)
MEASUREMENT_FIELDS = frozenset(
    {
        "timer_id",
        "synchronization_id",
        "warmup_repetitions",
        "repetitions",
        "unique_frames",
        "sample_count",
        "wire_scope",
        "p95_estimator",
    }
)
METRIC_FIELDS = frozenset(
    {
        "communication_bytes_per_frame",
        "communication_encode_ms",
        "communication_decode_ms",
        "communication_consumer_ms",
        "communication_e2e_mean_ms",
        "communication_e2e_p95_ms",
    }
)
SAMPLE_FIELDS = frozenset(
    {
        "frame_id",
        "repetition",
        "wire_bytes",
        "wire_sha256",
        "encode_ns",
        "decode_ns",
        "consumer_ns",
        "e2e_ns",
    }
)
FRAME_MANIFEST_FIELDS = frozenset({"frame_id", "payload_sha256"})


class MeasurementError(ValueError):
    """Raised when a measurement cannot satisfy the frozen contract."""


@dataclass(frozen=True)
class Frame:
    """One unique application frame and its pre-serialization payload."""

    frame_id: str
    payload: object


def _assert_finite_payload(value: object, *, path: str = "payload") -> None:
    """Accept JSON-like values and byte buffers, rejecting ambiguous objects."""

    if value is None or isinstance(value, (bool, str, bytes, bytearray, memoryview)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MeasurementError(f"{path} contains a non-finite number")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_payload(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise MeasurementError(f"{path} keys must be non-empty strings")
            _assert_finite_payload(item, path=f"{path}.{key}")
        return
    raise MeasurementError(
        f"{path} has unsupported type {type(value).__name__}; "
        "convert tensors to a finite byte buffer before measurement"
    )


def payload_sha256(value: object) -> str:
    """Hash supported payloads without serializing mutable Python objects."""

    _assert_finite_payload(value)
    digest = hashlib.sha256()

    def token(raw: bytes) -> None:
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)

    def visit(item: object) -> None:
        if item is None:
            token(b"none")
        elif isinstance(item, bool):
            token(b"bool:1" if item else b"bool:0")
        elif isinstance(item, int):
            token(b"int")
            token(str(item).encode("ascii"))
        elif isinstance(item, float):
            token(b"float")
            token(item.hex().encode("ascii"))
        elif isinstance(item, str):
            token(b"str")
            token(item.encode("utf-8"))
        elif isinstance(item, (bytes, bytearray, memoryview)):
            token(b"bytes")
            token(bytes(item))
        elif isinstance(item, list):
            token(b"list")
            token(str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
        elif isinstance(item, tuple):
            token(b"tuple")
            token(str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
        elif isinstance(item, Mapping):
            token(b"mapping")
            keys = sorted(item)
            token(str(len(keys)).encode("ascii"))
            for key in keys:
                token(key.encode("utf-8"))
                visit(item[key])
        else:  # pragma: no cover - guarded by _assert_finite_payload
            raise MeasurementError("unsupported payload type")

    visit(value)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise MeasurementError(f"non-finite JSON number is forbidden: {value}")


def canonical_json_encoder(value: object) -> bytes:
    """Canonical JSON encoder used only by the loopback diagnostic CLI."""

    _assert_finite_payload(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MeasurementError("payload is not canonical-JSON serializable") from exc


def canonical_json_decoder(value: bytes) -> object:
    try:
        decoded = json.loads(value.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeasurementError("wire payload is not valid canonical JSON") from exc
    if canonical_json_encoder(decoded) != value:
        raise MeasurementError("wire payload is not in canonical JSON form")
    return decoded


def _validate_frames(frames: Iterable[Frame]) -> list[Frame]:
    rows = list(frames)
    if not rows:
        raise MeasurementError("at least one frame is required")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Frame):
            raise MeasurementError("each input must be a Frame")
        if not isinstance(row.frame_id, str) or not SAFE_LABEL_RE.fullmatch(row.frame_id):
            raise MeasurementError("frame_id must be a safe non-empty label")
        if row.frame_id in seen:
            raise MeasurementError(f"duplicate frame_id: {row.frame_id}")
        seen.add(row.frame_id)
        _assert_finite_payload(row.payload, path=f"frame[{row.frame_id}].payload")
    return sorted(rows, key=lambda row: row.frame_id)


def _positive_integer(value: object, *, name: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise MeasurementError(f"{name} must be a {qualifier} integer")
    return value


def _safe_label(value: object, *, name: str) -> str:
    if not isinstance(value, str) or SAFE_LABEL_RE.fullmatch(value) is None:
        raise MeasurementError(f"{name} must be a safe non-empty label")
    return value


def _validated_context(evidence_tier: str, context: Mapping[str, object] | None) -> dict[str, str]:
    if evidence_tier not in EVIDENCE_TIERS:
        raise MeasurementError(f"evidence_tier must be one of {sorted(EVIDENCE_TIERS)}")
    supplied = dict(context or {})
    if evidence_tier == "diagnostic_only":
        if supplied:
            raise MeasurementError("diagnostic_only context must be empty")
        return {}
    if set(supplied) != REAL_CONTEXT_FIELDS:
        raise MeasurementError(
            "candidate_real context must contain exactly: "
            + ", ".join(sorted(REAL_CONTEXT_FIELDS))
        )
    for field in (
        "config_sha256",
        "dataset_identity_sha256",
        "split_sha256",
    ):
        value = supplied[field]
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise MeasurementError(f"context.{field} must be 64 lowercase hex characters")
    commit = supplied["code_commit"]
    if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
        raise MeasurementError("context.code_commit must be a full 40- or 64-hex revision")
    transport_id = _safe_label(supplied["transport_id"], name="context.transport_id")
    if transport_id.lower() == "loopback":
        raise MeasurementError("candidate_real transport_id cannot be loopback")
    if supplied["wire_scope"] != WIRE_SCOPE:
        raise MeasurementError(f"context.wire_scope must be {WIRE_SCOPE}")
    if supplied["pipeline_scope"] != REAL_PIPELINE_SCOPE:
        raise MeasurementError(f"context.pipeline_scope must be {REAL_PIPELINE_SCOPE}")
    return {key: str(supplied[key]) for key in sorted(supplied)}


def _as_wire_bytes(value: object, *, stage: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise MeasurementError(f"{stage} must return bytes-like data")
    result = bytes(value)
    if not result:
        raise MeasurementError(f"{stage} returned an empty wire message")
    return result


def _stamp(clock_ns: Callable[[], int]) -> int:
    value = clock_ns()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MeasurementError("clock_ns must return a non-negative integer")
    return value


def _duration(start: int, end: int, *, stage: str) -> int:
    if end < start:
        raise MeasurementError(f"{stage} clock moved backwards")
    return end - start


def _call_sync(synchronize: Callable[[], None]) -> None:
    try:
        synchronize()
    except Exception as exc:
        raise MeasurementError("synchronization callback failed") from exc


def _run_once(
    frame: Frame,
    *,
    encoder: Callable[[object], object],
    transport: Callable[[bytes], object],
    decoder: Callable[[bytes], object],
    consumer: Callable[[object], object],
    verifier: Callable[[object, object], bool],
    clock_ns: Callable[[], int],
    synchronize: Callable[[], None],
) -> dict[str, int | str]:
    _call_sync(synchronize)
    e2e_start = _stamp(clock_ns)

    _call_sync(synchronize)
    encode_start = _stamp(clock_ns)
    try:
        wire = _as_wire_bytes(encoder(frame.payload), stage="encoder")
    except MeasurementError:
        raise
    except Exception as exc:
        raise MeasurementError(f"encoder failed for frame {frame.frame_id}") from exc
    _call_sync(synchronize)
    encode_end = _stamp(clock_ns)

    try:
        received = _as_wire_bytes(transport(wire), stage="transport")
    except MeasurementError:
        raise
    except Exception as exc:
        raise MeasurementError(f"transport failed for frame {frame.frame_id}") from exc
    if received != wire:
        raise MeasurementError(f"wire integrity failed for frame {frame.frame_id}")

    _call_sync(synchronize)
    decode_start = _stamp(clock_ns)
    try:
        decoded = decoder(received)
    except MeasurementError:
        raise
    except Exception as exc:
        raise MeasurementError(f"decoder failed for frame {frame.frame_id}") from exc
    _call_sync(synchronize)
    decode_end = _stamp(clock_ns)
    try:
        verified = verifier(frame.payload, decoded)
    except Exception as exc:
        raise MeasurementError(f"round-trip verifier failed for frame {frame.frame_id}") from exc
    if not isinstance(verified, bool) or not verified:
        raise MeasurementError(f"round-trip semantic mismatch for frame {frame.frame_id}")

    _call_sync(synchronize)
    consumer_start = _stamp(clock_ns)
    try:
        consumer(decoded)
    except Exception as exc:
        raise MeasurementError(f"consumer failed for frame {frame.frame_id}") from exc
    _call_sync(synchronize)
    consumer_end = _stamp(clock_ns)

    _call_sync(synchronize)
    e2e_end = _stamp(clock_ns)
    timestamps = (
        e2e_start,
        encode_start,
        encode_end,
        decode_start,
        decode_end,
        consumer_start,
        consumer_end,
        e2e_end,
    )
    if any(later < earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise MeasurementError("clock moved backwards across measurement stages")
    return {
        "wire_bytes": len(wire),
        "wire_sha256": hashlib.sha256(wire).hexdigest(),
        "encode_ns": _duration(encode_start, encode_end, stage="encode"),
        "decode_ns": _duration(decode_start, decode_end, stage="decode"),
        "consumer_ns": _duration(consumer_start, consumer_end, stage="consumer"),
        "e2e_ns": _duration(e2e_start, e2e_end, stage="end-to-end"),
    }


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        raise MeasurementError("percentile requires at least one sample")
    if not 0.0 < percentile <= 1.0:
        raise MeasurementError("percentile must be in (0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def measure_frames(
    frames: Iterable[Frame],
    *,
    encoder: Callable[[object], object],
    transport: Callable[[bytes], object],
    decoder: Callable[[bytes], object],
    verifier: Callable[[object, object], bool],
    consumer: Callable[[object], object] | None = None,
    warmup_repetitions: int = 1,
    repetitions: int = 10,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    synchronize: Callable[[], None] = lambda: None,
    timer_id: str = "perf_counter_ns",
    synchronization_id: str = "none_cpu",
    evidence_tier: str = "diagnostic_only",
    context: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Measure a synchronous encode/transport/decode/consumer path.

    ``transport`` must return only after the same application message has
    arrived.  ``consumer`` is the downstream endpoint included in the
    end-to-end diagnostic.  ``candidate_real`` is reserved and always rejected
    until a registered audited runner exists.
    """

    rows = _validate_frames(frames)
    warmup_repetitions = _positive_integer(
        warmup_repetitions, name="warmup_repetitions", allow_zero=True
    )
    repetitions = _positive_integer(repetitions, name="repetitions")
    if not all(callable(item) for item in (encoder, transport, decoder, verifier)):
        raise MeasurementError("encoder, transport, decoder, and verifier must be callable")
    if consumer is not None and not callable(consumer):
        raise MeasurementError("consumer must be callable")
    if not callable(clock_ns) or not callable(synchronize):
        raise MeasurementError("clock_ns and synchronize must be callable")
    timer_id = _safe_label(timer_id, name="timer_id")
    synchronization_id = _safe_label(
        synchronization_id, name="synchronization_id"
    )
    validated_context = _validated_context(evidence_tier, context)
    if evidence_tier == "candidate_real":
        raise MeasurementError(
            "candidate_real is disabled until a registered audited transport runner exists"
        )
    downstream = consumer if consumer is not None else (lambda value: value)
    payload_digests = {frame.frame_id: payload_sha256(frame.payload) for frame in rows}

    def run_immutable(frame: Frame) -> dict[str, int | str]:
        expected = payload_digests[frame.frame_id]
        if payload_sha256(frame.payload) != expected:
            raise MeasurementError(f"payload changed before frame {frame.frame_id}")
        measured = _run_once(frame, **common)
        if payload_sha256(frame.payload) != expected:
            raise MeasurementError(f"encoder mutated payload for frame {frame.frame_id}")
        return measured

    common = {
        "encoder": encoder,
        "transport": transport,
        "decoder": decoder,
        "consumer": downstream,
        "verifier": verifier,
        "clock_ns": clock_ns,
        "synchronize": synchronize,
    }
    for _ in range(warmup_repetitions):
        for frame in rows:
            run_immutable(frame)

    samples: list[dict[str, int | str]] = []
    for repetition in range(repetitions):
        for frame in rows:
            measured = run_immutable(frame)
            samples.append(
                {
                    "frame_id": frame.frame_id,
                    "repetition": repetition,
                    **measured,
                }
            )

    wire_bytes = [int(row["wire_bytes"]) for row in samples]
    encode_ns = [int(row["encode_ns"]) for row in samples]
    decode_ns = [int(row["decode_ns"]) for row in samples]
    consumer_ns = [int(row["consumer_ns"]) for row in samples]
    e2e_ns = [int(row["e2e_ns"]) for row in samples]
    metrics = {
        "communication_bytes_per_frame": statistics.fmean(wire_bytes),
        "communication_encode_ms": statistics.fmean(encode_ns) / 1_000_000.0,
        "communication_decode_ms": statistics.fmean(decode_ns) / 1_000_000.0,
        "communication_consumer_ms": statistics.fmean(consumer_ns) / 1_000_000.0,
        "communication_e2e_mean_ms": statistics.fmean(e2e_ns) / 1_000_000.0,
        "communication_e2e_p95_ms": _nearest_rank(e2e_ns, 0.95) / 1_000_000.0,
    }
    if any(not math.isfinite(value) or value < 0 for value in metrics.values()):
        raise MeasurementError("aggregate metrics are not finite non-negative values")

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "evidence_tier": evidence_tier,
        "scientific_claim_allowed": False,
        "context": validated_context,
        "measurement": {
            "timer_id": timer_id,
            "synchronization_id": synchronization_id,
            "warmup_repetitions": warmup_repetitions,
            "repetitions": repetitions,
            "unique_frames": len(rows),
            "sample_count": len(samples),
            "wire_scope": (
                WIRE_SCOPE
                if evidence_tier == "candidate_real"
                else DIAGNOSTIC_WIRE_SCOPE
            ),
            "p95_estimator": "nearest_rank_ceiling",
        },
        "metrics": metrics,
        "frame_manifest": [
            {
                "frame_id": frame.frame_id,
                "payload_sha256": payload_digests[frame.frame_id],
            }
            for frame in rows
        ],
        "samples": samples,
    }


def canonical_bytes(value: object) -> bytes:
    _assert_finite_payload(value, path="document")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def validate_report(report: Mapping[str, object]) -> None:
    """Recompute every aggregate before evidence is accepted for writing."""

    if not isinstance(report, Mapping) or set(report) != REPORT_FIELDS:
        raise MeasurementError("report fields do not match the frozen schema")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise MeasurementError("report schema_version does not match the contract")
    if report.get("protocol_id") != PROTOCOL_ID:
        raise MeasurementError("report protocol_id does not match the contract")
    if report.get("scientific_claim_allowed") is not False:
        raise MeasurementError("measurement reports cannot grant scientific claims")
    tier = report.get("evidence_tier")
    if not isinstance(tier, str):
        raise MeasurementError("report evidence_tier must be a string")
    context = report.get("context")
    if not isinstance(context, Mapping):
        raise MeasurementError("report context must be an object")
    _validated_context(tier, context)
    if tier == "candidate_real":
        raise MeasurementError(
            "candidate_real evidence is disabled until an audited runner exists"
        )

    measurement = report.get("measurement")
    if not isinstance(measurement, Mapping) or set(measurement) != MEASUREMENT_FIELDS:
        raise MeasurementError("measurement fields do not match the frozen schema")
    _safe_label(measurement.get("timer_id"), name="measurement.timer_id")
    _safe_label(
        measurement.get("synchronization_id"),
        name="measurement.synchronization_id",
    )
    warmups = _positive_integer(
        measurement.get("warmup_repetitions"),
        name="measurement.warmup_repetitions",
        allow_zero=True,
    )
    del warmups
    repetitions = _positive_integer(
        measurement.get("repetitions"), name="measurement.repetitions"
    )
    unique_frames = _positive_integer(
        measurement.get("unique_frames"), name="measurement.unique_frames"
    )
    sample_count = _positive_integer(
        measurement.get("sample_count"), name="measurement.sample_count"
    )
    expected_wire_scope = WIRE_SCOPE if tier == "candidate_real" else DIAGNOSTIC_WIRE_SCOPE
    if measurement.get("wire_scope") != expected_wire_scope:
        raise MeasurementError("measurement.wire_scope does not match evidence_tier")
    if measurement.get("p95_estimator") != "nearest_rank_ceiling":
        raise MeasurementError("measurement.p95_estimator does not match the contract")

    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        raise MeasurementError("report.samples must be a non-empty array")
    if sample_count != len(samples) or sample_count != unique_frames * repetitions:
        raise MeasurementError("sample_count does not match frames and repetitions")
    frame_manifest = report.get("frame_manifest")
    if not isinstance(frame_manifest, list) or not frame_manifest:
        raise MeasurementError("report.frame_manifest must be a non-empty array")
    manifest_ids: set[str] = set()
    for index, entry in enumerate(frame_manifest):
        if not isinstance(entry, Mapping) or set(entry) != FRAME_MANIFEST_FIELDS:
            raise MeasurementError(
                f"frame_manifest[{index}] fields do not match the frozen schema"
            )
        frame_id = _safe_label(
            entry.get("frame_id"), name=f"frame_manifest[{index}].frame_id"
        )
        digest = entry.get("payload_sha256")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise MeasurementError(
                f"frame_manifest[{index}].payload_sha256 is invalid"
            )
        if frame_id in manifest_ids:
            raise MeasurementError(f"duplicate frame manifest ID: {frame_id}")
        manifest_ids.add(frame_id)
    if len(manifest_ids) != unique_frames:
        raise MeasurementError("unique_frames does not match frame_manifest")

    observed_pairs: set[tuple[str, int]] = set()
    frame_ids: set[str] = set()
    wire_bytes: list[int] = []
    encode_ns: list[int] = []
    decode_ns: list[int] = []
    consumer_ns: list[int] = []
    e2e_ns: list[int] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or set(sample) != SAMPLE_FIELDS:
            raise MeasurementError(f"sample[{index}] fields do not match the frozen schema")
        frame_id = _safe_label(sample.get("frame_id"), name=f"sample[{index}].frame_id")
        repetition = sample.get("repetition")
        if (
            isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or not 0 <= repetition < repetitions
        ):
            raise MeasurementError(f"sample[{index}].repetition is out of range")
        pair = (frame_id, repetition)
        if pair in observed_pairs:
            raise MeasurementError(f"duplicate measured frame/repetition pair: {pair!r}")
        observed_pairs.add(pair)
        frame_ids.add(frame_id)
        wire_digest = sample.get("wire_sha256")
        if not isinstance(wire_digest, str) or SHA256_RE.fullmatch(wire_digest) is None:
            raise MeasurementError(f"sample[{index}].wire_sha256 is invalid")
        values: dict[str, int] = {}
        for field in ("wire_bytes", "encode_ns", "decode_ns", "consumer_ns", "e2e_ns"):
            value = sample.get(field)
            minimum = 1 if field == "wire_bytes" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise MeasurementError(f"sample[{index}].{field} is invalid")
            values[field] = value
        if values["e2e_ns"] < (
            values["encode_ns"] + values["decode_ns"] + values["consumer_ns"]
        ):
            raise MeasurementError(f"sample[{index}].e2e_ns is shorter than timed stages")
        wire_bytes.append(values["wire_bytes"])
        encode_ns.append(values["encode_ns"])
        decode_ns.append(values["decode_ns"])
        consumer_ns.append(values["consumer_ns"])
        e2e_ns.append(values["e2e_ns"])
    if frame_ids != manifest_ids:
        raise MeasurementError("sample frame IDs do not match frame_manifest")

    expected_metrics = {
        "communication_bytes_per_frame": statistics.fmean(wire_bytes),
        "communication_encode_ms": statistics.fmean(encode_ns) / 1_000_000.0,
        "communication_decode_ms": statistics.fmean(decode_ns) / 1_000_000.0,
        "communication_consumer_ms": statistics.fmean(consumer_ns) / 1_000_000.0,
        "communication_e2e_mean_ms": statistics.fmean(e2e_ns) / 1_000_000.0,
        "communication_e2e_p95_ms": _nearest_rank(e2e_ns, 0.95) / 1_000_000.0,
    }
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != METRIC_FIELDS:
        raise MeasurementError("metric fields do not match the frozen schema")
    for key, expected in expected_metrics.items():
        observed = metrics.get(key)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(observed)
            or observed < 0
            or not math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise MeasurementError(f"metric {key} does not match raw samples")


def evidence_document(report: Mapping[str, object]) -> dict[str, object]:
    validate_report(report)
    raw = canonical_bytes(report)
    return {
        "report": dict(report),
        "report_sha256": hashlib.sha256(raw).hexdigest(),
    }


def write_evidence(path: Path, report: Mapping[str, object]) -> str:
    """Write an immutable, self-hashed evidence document without overwriting."""

    document = evidence_document(report)
    rendered = canonical_bytes(document) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise MeasurementError(f"evidence output already exists: {path}") from exc
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return str(document["report_sha256"])


def _read_jsonl(path: Path) -> list[Frame]:
    frames: list[Frame] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, MeasurementError) as exc:
            raise MeasurementError(f"invalid JSON on line {line_number}") from exc
        if not isinstance(row, dict) or set(row) != {"frame_id", "payload"}:
            raise MeasurementError(
                f"line {line_number} must contain exactly frame_id and payload"
            )
        frames.append(Frame(frame_id=row["frame_id"], payload=row["payload"]))
    return frames


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a diagnostic-only canonical-JSON communication loopback."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        frames = _read_jsonl(args.input_jsonl)
        report = measure_frames(
            frames,
            encoder=canonical_json_encoder,
            transport=lambda value: value,
            decoder=canonical_json_decoder,
            verifier=lambda expected, observed: expected == observed,
            warmup_repetitions=args.warmup_repetitions,
            repetitions=args.repetitions,
            evidence_tier="diagnostic_only",
        )
        digest = write_evidence(args.output, report)
    except (MeasurementError, OSError, ValueError) as exc:
        print(f"communication measurement failed: {exc}", file=os.sys.stderr)
        return 2
    print(f"wrote diagnostic-only communication evidence sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
