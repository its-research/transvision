#!/usr/bin/env python3
"""Inference-only consumer for canonical V2X-Seq SPD fixed detections.

This entry point validates detector output through the frozen TransVision SPD
fixed-detection contract and applies a small deterministic association canary.
It does not read annotations, train a model, evaluate predictions, access
ClearML, or write the thesis result registry.  Every output record is marked
``diagnostic_only`` and ``scientific_claim_allowed=false``.

The reference association links detections only between consecutive frames of
the same sequence.  It deliberately performs no motion extrapolation: the
previous observed bottom centre is the complete query state.  This makes the
data-flow and causality boundary auditable without presenting the canary as an
RTP-V2X tracking implementation or as scientific evidence.

The input v1 contract also freezes ``Car -> vehicle`` while the scientific
tracking class mapping remains unverified.  Consequently, this consumer cannot
promote v1 fixed detections into production experiment input; data read-back
must precede a separately versioned v2 contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.adapters.transvision_spd import schema as detection_schema  # noqa: E402
from experiments.rtp_v2x.model import association as association_module  # noqa: E402


SCHEMA_VERSION = 1
CONTRACT_ID = "RTPV2X-DIAGNOSTIC-FIXED-TRACKS-v1"
RECORD_KIND = "diagnostic_fixed_track_frame"
TRACKER_ID = "consecutive_frame_bottom_center_xy_v1"
MAX_OUTPUT_BYTES = detection_schema.MAX_JSONL_BYTES
MAX_OUTPUT_RECORD_BYTES = detection_schema.MAX_RECORD_BYTES
MAX_SIGNED_INT64 = 2**63 - 1
FIXED_DETECTION_PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "clearml"
    / "protocols"
    / "fixed-detections-v1.json"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        frozen[key] = _freeze_mapping(item) if isinstance(item, Mapping) else item
    return MappingProxyType(frozen)


CAPABILITY_BOUNDARY: Mapping[str, Any] = _freeze_mapping(
    {
        "purpose": "diagnostic_baseline_plumbing_only",
        "fixed_detection_contract_id": detection_schema.CONTRACT_ID,
        "per_agent_source_available": False,
        "arrival_time_available": False,
        "spatial_reliability_available": False,
        "detection_score_role": (
            "passthrough_only_not_spatial_reliability_or_association_term"
        ),
        "runs_rtp_v2x": False,
        "runs_h1_reliability_association": False,
        "fixed_v1_class_mapping": "Car_to_vehicle_only",
        "class_mapping_unverified_for_scientific_protocol": True,
        "production_fixed_detection_input_allowed": False,
        "production_contract_upgrade_requirement": (
            "data_readback_then_separately_versioned_v2"
        ),
        "runtime_source_hash_is_execution_proof": False,
        "scientific_evidence_eligible": False,
    }
)

ASSOCIATION_POLICY: Mapping[str, Any] = _freeze_mapping(
    {
        "tracker_id": TRACKER_ID,
        "state_scope": "previous_frame_of_same_sequence_only",
        "query_state": "last_observed_xy_bottom_center_without_extrapolation",
        "query_order": "ascending_track_id",
        "candidate_order": "canonical_fixed_detection_order",
        "distance": "euclidean_xy_bottom_center_metres",
        "gate": {
            "comparison": "less_than_or_equal",
            "maximum_distance_m": 6.0,
        },
        "query_unmatched_cost": 4.0,
        "candidate_new_cost": 4.0,
        "solver": "repository_hungarian_with_explicit_unmatched_v1",
        "tie_resolution": "stable_query_then_candidate_scan_order",
        "matched_state_update": "replace_with_current_detection",
        "unmatched_state_update": "retire_before_current_output",
        "track_id_allocation": "per_sequence_monotone_new_candidate_order",
        "capability_boundary": CAPABILITY_BOUNDARY,
    }
)


class TrackingInferenceError(ValueError):
    """Raised when an inference-only tracking artifact is unsafe or ambiguous."""


@dataclass(frozen=True, slots=True)
class TrackingArtifact:
    """Canonical diagnostic tracking output and its provenance digests."""

    records: tuple[dict[str, Any], ...]
    jsonl_bytes: bytes
    input_document_sha256: str
    output_document_sha256: str
    association_contract_sha256: str
    consumer_implementation_sha256: str
    association_implementation_sha256: str
    fixed_detection_contract_sha256: str
    detection_schema_implementation_sha256: str


@dataclass(frozen=True, slots=True)
class _TrackState:
    track_id: str
    box_3d: tuple[float, float, float, float, float, float, float]
    vehicle_frame_id: str
    detection_id: str
    decision_time_ns: int


@dataclass(slots=True)
class _SequenceState:
    previous_tracks: tuple[_TrackState, ...] = ()
    last_decision_time_ns: int | None = None
    next_track_ordinal: int = 0


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TrackingInferenceError(
            f"cannot hash required implementation file: {path.name}"
        ) from exc
    return digest.hexdigest()


def _canonical_float_text(value: float) -> str:
    if not math.isfinite(value):
        raise TrackingInferenceError("tracking output contains a non-finite float")
    normalized = 0.0 if value == 0.0 else value
    return format(normalized, ".16e")


def _canonical_json_text(value: Any) -> str:
    """Serialize the bounded output with a frozen, deterministic spelling."""

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if not -MAX_SIGNED_INT64 - 1 <= value <= MAX_SIGNED_INT64:
            raise TrackingInferenceError("tracking output integer exceeds int64")
        return str(value)
    if isinstance(value, float):
        return _canonical_float_text(value)
    if isinstance(value, str):
        try:
            return json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise TrackingInferenceError(
                "tracking output contains an invalid string"
            ) from exc
    if isinstance(value, list):
        return "[" + ",".join(_canonical_json_text(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TrackingInferenceError(
                "tracking output contains a non-string object key"
            )
        return (
            "{"
            + ",".join(
                _canonical_json_text(key) + ":" + _canonical_json_text(value[key])
                for key in sorted(value)
            )
            + "}"
        )
    raise TrackingInferenceError(
        f"tracking output contains unsupported type {type(value).__name__}"
    )


def _canonical_object_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_text(value).encode("utf-8"))


ASSOCIATION_CONTRACT_SHA256 = _canonical_object_sha256(ASSOCIATION_POLICY)


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json_text(value))


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TrackingInferenceError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _normalize_input_bytes(data: bytes | str) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        try:
            return data.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise TrackingInferenceError(
                "fixed-detection input is not valid UTF-8"
            ) from exc
    raise TrackingInferenceError("fixed-detection input must be bytes or text")


def _implementation_hashes() -> tuple[str, str, str, str]:
    association_file = getattr(association_module, "__file__", None)
    detection_schema_file = getattr(detection_schema, "__file__", None)
    if (
        not isinstance(association_file, str)
        or not association_file
        or not isinstance(detection_schema_file, str)
        or not detection_schema_file
    ):
        raise TrackingInferenceError(
            "a required implementation has no hashable source identity"
        )
    return (
        _sha256_file(Path(__file__).resolve()),
        _sha256_file(Path(association_file).resolve()),
        _sha256_file(FIXED_DETECTION_PROTOCOL_PATH.resolve()),
        _sha256_file(Path(detection_schema_file).resolve()),
    )


def _association_costs(
    queries: Sequence[_TrackState], detections: Sequence[Mapping[str, Any]]
) -> list[list[float]]:
    maximum_distance = ASSOCIATION_POLICY["gate"]["maximum_distance_m"]
    costs: list[list[float]] = []
    for query in queries:
        row: list[float] = []
        query_x, query_y = query.box_3d[:2]
        for detection in detections:
            candidate_x, candidate_y = detection["box_3d"][:2]
            distance = math.hypot(candidate_x - query_x, candidate_y - query_y)
            row.append(distance if distance <= maximum_distance else math.inf)
        costs.append(row)
    return costs


def _allocate_track_id(state: _SequenceState) -> str:
    if state.next_track_ordinal > MAX_SIGNED_INT64:
        raise TrackingInferenceError("per-sequence track ordinal exceeds int64")
    track_id = f"diag-track-{state.next_track_ordinal:08d}"
    state.next_track_ordinal += 1
    return track_id


def _track_one_frame(
    *,
    record: Mapping[str, Any],
    state: _SequenceState,
) -> tuple[list[dict[str, Any]], tuple[_TrackState, ...]]:
    frame = record["frame"]
    decision_time = frame["decision_time_ns"]
    if (
        state.last_decision_time_ns is not None
        and decision_time <= state.last_decision_time_ns
    ):
        raise TrackingInferenceError(
            "decision_time_ns must be strictly increasing within each sequence"
        )
    queries = tuple(sorted(state.previous_tracks, key=lambda item: item.track_id))
    detections = record["detections"]
    result = association_module.solve_with_unmatched(
        _association_costs(queries, detections),
        query_unmatched_costs=[
            ASSOCIATION_POLICY["query_unmatched_cost"] for _ in queries
        ],
        candidate_new_costs=[
            ASSOCIATION_POLICY["candidate_new_cost"] for _ in detections
        ],
    )
    matched_by_candidate = {
        candidate_index: queries[query_index]
        for query_index, candidate_index in result.matches
    }
    new_candidates = set(result.new_candidates)
    if set(range(len(detections))) != set(matched_by_candidate) | new_candidates:
        raise TrackingInferenceError(
            "association solver did not account for every current detection"
        )

    output_tracks: list[dict[str, Any]] = []
    current_states: list[_TrackState] = []
    for candidate_index, detection in enumerate(detections):
        predecessor = matched_by_candidate.get(candidate_index)
        if predecessor is None:
            if candidate_index not in new_candidates:
                raise TrackingInferenceError(
                    "association solver returned an inconsistent new-candidate set"
                )
            track_id = _allocate_track_id(state)
            lifecycle = "new"
            predecessor_record = None
        else:
            if predecessor.decision_time_ns >= decision_time:
                raise TrackingInferenceError(
                    "association predecessor violates the causal time boundary"
                )
            track_id = predecessor.track_id
            lifecycle = "matched"
            predecessor_record = {
                "vehicle_frame_id": predecessor.vehicle_frame_id,
                "detection_id": predecessor.detection_id,
                "decision_time_ns": predecessor.decision_time_ns,
            }
        box = tuple(float(value) for value in detection["box_3d"])
        current_states.append(
            _TrackState(
                track_id=track_id,
                box_3d=box,
                vehicle_frame_id=frame["vehicle_frame_id"],
                detection_id=detection["detection_id"],
                decision_time_ns=decision_time,
            )
        )
        output_tracks.append(
            {
                "track_id": track_id,
                "detection_id": detection["detection_id"],
                "lifecycle": lifecycle,
                "predecessor": predecessor_record,
                "class_name": detection["class_name"],
                "box_3d": list(box),
                "score": float(detection["score"]),
            }
        )
    state.previous_tracks = tuple(current_states)
    state.last_decision_time_ns = decision_time
    return output_tracks, state.previous_tracks


def _output_record(
    *,
    fixed_record: Mapping[str, Any],
    tracks: list[dict[str, Any]],
    input_document_sha256: str,
    consumer_implementation_sha256: str,
    association_implementation_sha256: str,
    fixed_detection_contract_sha256: str,
    detection_schema_implementation_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "record_kind": RECORD_KIND,
        "diagnostic_only": True,
        "scientific_claim_allowed": False,
        "capability_boundary": _json_copy(CAPABILITY_BOUNDARY),
        "dataset": copy.deepcopy(fixed_record["dataset"]),
        "frame": copy.deepcopy(fixed_record["frame"]),
        "coordinate_system": copy.deepcopy(fixed_record["coordinate_system"]),
        "detector_source": copy.deepcopy(fixed_record["source"]),
        "input_provenance": {
            "fixed_detection_contract_id": detection_schema.CONTRACT_ID,
            "fixed_detection_contract_sha256": fixed_detection_contract_sha256,
            "detection_schema_implementation_sha256": (
                detection_schema_implementation_sha256
            ),
            "fixed_detection_document_sha256": input_document_sha256,
            "fixed_detection_record_sha256": detection_schema.record_sha256(
                fixed_record
            ),
        },
        "tracker_provenance": {
            "tracker_id": TRACKER_ID,
            "consumer_implementation_sha256": consumer_implementation_sha256,
            "association_implementation_sha256": association_implementation_sha256,
            "association_contract_sha256": ASSOCIATION_CONTRACT_SHA256,
        },
        "association_policy": _json_copy(ASSOCIATION_POLICY),
        "tracks": tracks,
    }


def _canonical_output_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    if not records:
        raise TrackingInferenceError(
            "tracking output must contain at least one frame record"
        )
    payload = bytearray()
    for index, record in enumerate(records):
        line = (_canonical_json_text(record) + "\n").encode("utf-8")
        if len(line) > MAX_OUTPUT_RECORD_BYTES:
            raise TrackingInferenceError(
                f"tracking output record {index} exceeds the byte limit"
            )
        payload.extend(line)
        if len(payload) > MAX_OUTPUT_BYTES:
            raise TrackingInferenceError("tracking output exceeds the byte limit")
    return bytes(payload)


def run_tracking_inference(
    data: bytes | str, *, expected_input_document_sha256: str
) -> TrackingArtifact:
    """Validate fixed detections and return provenance-bound diagnostic tracks.

    The expected input digest is mandatory, so a caller cannot accidentally run
    a different detector artifact than the one selected during preflight.
    """

    raw = _normalize_input_bytes(data)
    expected = _require_sha256(
        expected_input_document_sha256, "expected input document SHA-256"
    )
    observed = _sha256_bytes(raw)
    if observed != expected:
        raise TrackingInferenceError(
            "fixed-detection input SHA-256 does not match the trusted expectation"
        )
    try:
        fixed_records = detection_schema.loads_jsonl(raw)
    except detection_schema.FixedDetectionContractError as exc:
        raise TrackingInferenceError(
            "fixed-detection input failed the frozen canonical contract"
        ) from exc
    canonical_digest = detection_schema.document_sha256(fixed_records)
    if canonical_digest != observed:
        raise TrackingInferenceError(
            "validated fixed-detection bytes do not match their canonical digest"
        )

    (
        consumer_hash,
        association_hash,
        fixed_detection_contract_hash,
        detection_schema_hash,
    ) = _implementation_hashes()
    states: dict[str, _SequenceState] = {}
    output_records: list[dict[str, Any]] = []
    for fixed_record in fixed_records:
        sequence_id = fixed_record["frame"]["sequence_id"]
        state = states.setdefault(sequence_id, _SequenceState())
        tracks, _ = _track_one_frame(record=fixed_record, state=state)
        output_records.append(
            _output_record(
                fixed_record=fixed_record,
                tracks=tracks,
                input_document_sha256=observed,
                consumer_implementation_sha256=consumer_hash,
                association_implementation_sha256=association_hash,
                fixed_detection_contract_sha256=fixed_detection_contract_hash,
                detection_schema_implementation_sha256=detection_schema_hash,
            )
        )
    if _implementation_hashes() != (
        consumer_hash,
        association_hash,
        fixed_detection_contract_hash,
        detection_schema_hash,
    ):
        raise TrackingInferenceError(
            "a required implementation changed while tracking inference was running"
        )
    payload = _canonical_output_bytes(output_records)
    return TrackingArtifact(
        records=tuple(output_records),
        jsonl_bytes=payload,
        input_document_sha256=observed,
        output_document_sha256=_sha256_bytes(payload),
        association_contract_sha256=ASSOCIATION_CONTRACT_SHA256,
        consumer_implementation_sha256=consumer_hash,
        association_implementation_sha256=association_hash,
        fixed_detection_contract_sha256=fixed_detection_contract_hash,
        detection_schema_implementation_sha256=detection_schema_hash,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate diagnostic-only tracks from canonical fixed-detection JSONL"
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    return parser.parse_args()


def _read_bounded_input(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        path_metadata = path.lstat()
        if stat.S_ISLNK(path_metadata.st_mode):
            raise TrackingInferenceError(
                "fixed-detection input must not be a symbolic link"
            )
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TrackingInferenceError("cannot open fixed-detection input") from exc
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise TrackingInferenceError(
                    "fixed-detection input must be a regular file"
                )
            if (metadata.st_dev, metadata.st_ino) != (
                path_metadata.st_dev,
                path_metadata.st_ino,
            ):
                raise TrackingInferenceError(
                    "fixed-detection input changed while it was opened"
                )
            if metadata.st_size > detection_schema.MAX_JSONL_BYTES:
                raise TrackingInferenceError(
                    "fixed-detection input exceeds the byte limit"
                )
            payload = bytearray()
            while True:
                remaining = detection_schema.MAX_JSONL_BYTES + 1 - len(payload)
                if remaining <= 0:
                    raise TrackingInferenceError(
                        "fixed-detection input exceeds the byte limit"
                    )
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                payload.extend(chunk)
            if len(payload) > detection_schema.MAX_JSONL_BYTES:
                raise TrackingInferenceError(
                    "fixed-detection input exceeds the byte limit"
                )
            return bytes(payload)
    except OSError as exc:
        raise TrackingInferenceError("cannot read fixed-detection input") from exc


def _write_new_file(path: Path, payload: bytes) -> None:
    temporary_path: Path | None = None
    try:
        parent = path.parent.resolve(strict=True)
        if not parent.is_dir():
            raise TrackingInferenceError(
                "tracking output parent directory is missing"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path, follow_symlinks=False)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError as exc:
        raise TrackingInferenceError(
            "tracking output already exists; refusing to overwrite it"
        ) from exc
    except OSError as exc:
        raise TrackingInferenceError("cannot write tracking output") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def main() -> None:
    args = _parse_args()
    try:
        if args.input.resolve() == args.output.resolve():
            raise TrackingInferenceError("input and output paths must differ")
        raw = _read_bounded_input(args.input)
        artifact = run_tracking_inference(
            raw,
            expected_input_document_sha256=args.expected_input_sha256,
        )
        _write_new_file(args.output, artifact.jsonl_bytes)
    except TrackingInferenceError as exc:
        raise SystemExit(str(exc)) from exc
    summary = {
        "contract_id": CONTRACT_ID,
        "diagnostic_only": True,
        "scientific_claim_allowed": False,
        "record_count": len(artifact.records),
        "input_document_sha256": artifact.input_document_sha256,
        "output_document_sha256": artifact.output_document_sha256,
        "association_contract_sha256": artifact.association_contract_sha256,
    }
    print(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()


__all__ = (
    "ASSOCIATION_CONTRACT_SHA256",
    "ASSOCIATION_POLICY",
    "CAPABILITY_BOUNDARY",
    "CONTRACT_ID",
    "RECORD_KIND",
    "SCHEMA_VERSION",
    "TRACKER_ID",
    "TrackingArtifact",
    "TrackingInferenceError",
    "run_tracking_inference",
)
