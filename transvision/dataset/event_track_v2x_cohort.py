"""Immutable native-10 Hz and frozen-2 Hz SPD cohort manifests."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
from typing import Iterable

from transvision.models.event_track_v2x.wire import canonical_json_bytes

from .event_track_v2x_spd import SPDMetadata, SPDPair


class SPDCohortError(ValueError):
    """Raised when a requested SPD cohort cannot be reproduced exactly."""


def _sha256(value: str, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SPDCohortError(f"{name} must be a lowercase SHA-256")
    return value


def _split_ids(values: Iterable[str]) -> tuple[str, ...]:
    result = tuple(values)
    if not result or any(
        type(value) is not str or len(value) != 6 or not value.isdecimal()
        for value in result
    ):
        raise SPDCohortError("split frame IDs must be canonical six-digit strings")
    if len(set(result)) != len(result):
        raise SPDCohortError("split frame IDs must be unique")
    return result


def _pair_key(pair: SPDPair) -> tuple[int, str, str]:
    return (
        pair.vehicle.pointcloud_timestamp_us,
        pair.vehicle.frame_id,
        pair.infrastructure.frame_id,
    )


def _label_sha256(pair: SPDPair) -> str:
    labels = [asdict(label) for label in pair.labels]
    return hashlib.sha256(canonical_json_bytes(labels)).hexdigest()


def build_spd_cohort_manifest(
    metadata: SPDMetadata,
    *,
    split_name: str,
    split_frame_ids: Iterable[str],
    split_sha256: str,
    dataset_manifest_sha256: str,
    target_rate_hz: int,
    phase: int = 0,
) -> dict[str, object]:
    """Build a sealed cohort without altering labels or sequence identity.

    Native 10 Hz keeps every split pair.  Frozen 2 Hz keeps every fifth pair
    independently inside each sequence, starting at ``phase``.
    """

    if split_name not in {"train", "val"}:
        raise SPDCohortError(
            "EventTrack-V2X supports only SPD train and val; test/test_A are excluded"
        )
    split_ids = _split_ids(split_frame_ids)
    split_set = set(split_ids)
    _sha256(split_sha256, "split_sha256")
    _sha256(dataset_manifest_sha256, "dataset_manifest_sha256")
    if target_rate_hz not in {2, 10}:
        raise SPDCohortError("target_rate_hz must be 2 or 10")
    stride = metadata.nominal_rate_hz // target_rate_hz
    if metadata.nominal_rate_hz != 10 or stride * target_rate_hz != 10:
        raise SPDCohortError("metadata rate cannot produce the requested cohort")
    if type(phase) is not int or not 0 <= phase < stride:
        raise SPDCohortError("phase must be an integer in [0, stride)")

    selected_source = [
        pair for pair in metadata.pairs if pair.vehicle.frame_id in split_set
    ]
    observed_ids = {pair.vehicle.frame_id for pair in selected_source}
    missing = sorted(split_set - observed_ids)
    if missing:
        raise SPDCohortError(
            f"split contains {len(missing)} frame IDs absent from cooperative pairs"
        )
    if len(selected_source) != len(split_ids):
        raise SPDCohortError("cooperative pairs are not one-to-one with split frame IDs")

    by_sequence: dict[str, list[SPDPair]] = defaultdict(list)
    for pair in selected_source:
        by_sequence[pair.sequence_id].append(pair)
    selected: list[SPDPair] = []
    for sequence_id in sorted(by_sequence):
        ordered = sorted(by_sequence[sequence_id], key=_pair_key)
        selected.extend(ordered[phase::stride])

    records = [
        {
            "infrastructure_event_time_us": pair.infrastructure.pointcloud_timestamp_us,
            "infrastructure_frame_id": pair.infrastructure.frame_id,
            "label_count": len(pair.labels),
            "label_sha256": _label_sha256(pair),
            "sequence_id": pair.sequence_id,
            "vehicle_event_time_us": pair.vehicle.pointcloud_timestamp_us,
            "vehicle_frame_id": pair.vehicle.frame_id,
        }
        for pair in selected
    ]
    payload: dict[str, object] = {
        "cohort_id": f"v2x-seq-spd-{target_rate_hz}hz-frozen-v1-{split_name}",
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "label_policy": "preserve-source-labels-byte-semantically",
        "pair_count": len(records),
        "pairs": records,
        "phase": phase,
        "schema_version": 1,
        "source_rate_hz": metadata.nominal_rate_hz,
        "split_name": split_name,
        "split_sha256": split_sha256,
        "stride": stride,
        "target_rate_hz": target_rate_hz,
    }
    return {
        **payload,
        "content_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    }


def verify_spd_cohort_manifest(value: dict[str, object]) -> str:
    """Return the content hash after exact-key and seal validation."""

    expected = {
        "cohort_id",
        "content_sha256",
        "dataset_manifest_sha256",
        "label_policy",
        "pair_count",
        "pairs",
        "phase",
        "schema_version",
        "source_rate_hz",
        "split_name",
        "split_sha256",
        "stride",
        "target_rate_hz",
    }
    if type(value) is not dict or set(value) != expected:
        raise SPDCohortError("cohort manifest has missing or unknown fields")
    observed = _sha256(value["content_sha256"], "content_sha256")
    payload = {key: item for key, item in value.items() if key != "content_sha256"}
    expected_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if observed != expected_hash:
        raise SPDCohortError("cohort manifest content SHA-256 mismatch")
    if type(value["pairs"]) is not list or value["pair_count"] != len(value["pairs"]):
        raise SPDCohortError("cohort pair_count does not match pairs")
    return observed


__all__ = [
    "SPDCohortError",
    "build_spd_cohort_manifest",
    "verify_spd_cohort_manifest",
]
