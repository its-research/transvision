#!/usr/bin/env python3
"""Audit a pinned third-party V2X-Seq-SPD metadata subset without mutating it.

This program is deliberately limited to file-integrity and schema findings.  It
does not authenticate the mirror as an official release, inspect raw sensor
payloads, run a benchmark, or produce evidence that may be cited as scientific
validation.  The only output is deterministic JSON written to standard output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


AUDIT_SCHEMA_VERSION = 1
MANIFEST_NAME = "mirror-manifest.json"
V1_DIRECTORY = "v1.0-trainval"
EXPECTED_V1_TABLES = (
    "attribute",
    "calibrated_sensor",
    "category",
    "ego_pose",
    "instance",
    "log",
    "map",
    "sample",
    "sample_annotation",
    "sample_data",
    "scene",
    "sensor",
    "visibility",
)
RAW_SENSOR_SUFFIXES = (
    ".bin",
    ".pcd",
    ".jpg",
    ".jpeg",
    ".png",
    ".npy",
    ".npz",
)
SENTINEL_ID = "-1"


class AuditError(RuntimeError):
    """Raised when provenance or required metadata cannot be audited safely."""


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON file {path.name}: {type(exc).__name__}") from exc


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditError(f"{label} must be a JSON object")
    return value


def require_object_rows(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise AuditError(f"{label} must be a JSON array")
    if any(not isinstance(row, dict) for row in value):
        raise AuditError(f"{label} must contain only JSON objects")
    return value


def safe_manifest_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise AuditError(f"unsafe manifest path: {relative_path!r}")
    target = (root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise AuditError(f"manifest path escapes dataset root: {relative_path!r}") from exc
    return target


def table_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    coverage: Counter[str] = Counter()
    variants: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        fields = tuple(sorted(str(key) for key in row))
        coverage.update(fields)
        variants[fields] += 1
    return {
        "field_presence_counts": dict(sorted(coverage.items())),
        "observed_fields": sorted(coverage),
        "row_count": len(rows),
        "schema_variant_count": len(variants),
    }


def integer_like(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (stripped.isdigit() or (stripped[0] == "-" and stripped[1:].isdigit())):
            return int(stripped)
    return None


def numeric_stats(values: Iterable[int]) -> dict[str, int | float | None]:
    rows = list(values)
    if not rows:
        return {
            "count": 0,
            "maximum": None,
            "median": None,
            "minimum": None,
            "unique_count": 0,
        }
    return {
        "count": len(rows),
        "maximum": max(rows),
        "median": statistics.median(rows),
        "minimum": min(rows),
        "unique_count": len(set(rows)),
    }


def linked_chain_integrity(
    parents: Sequence[Mapping[str, Any]],
    children: Sequence[Mapping[str, Any]],
    *,
    child_parent_field: str,
    declared_count_field: str,
    first_field: str,
    last_field: str,
) -> dict[str, int]:
    child_by_token = {
        row.get("token"): row for row in children if isinstance(row.get("token"), str)
    }
    children_by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    missing_parent_reference_count = 0
    parent_tokens = {
        row.get("token") for row in parents if isinstance(row.get("token"), str)
    }
    for row in children:
        parent_token = row.get(child_parent_field)
        if isinstance(parent_token, str):
            children_by_parent[parent_token].append(row)
            if parent_token not in parent_tokens:
                missing_parent_reference_count += 1

    counters: Counter[str] = Counter()
    for parent in parents:
        parent_token = parent.get("token")
        if not isinstance(parent_token, str):
            counters["malformed_parent_token_count"] += 1
            continue
        rows = children_by_parent.get(parent_token, [])
        if parent.get(declared_count_field) != len(rows):
            counters["declared_count_mismatch_count"] += 1

        first = parent.get(first_field)
        last = parent.get(last_field)
        if rows and not isinstance(first, str):
            counters["malformed_first_token_count"] += 1
            continue
        if rows and first not in child_by_token:
            counters["missing_first_token_count"] += 1
            continue

        seen: set[str] = set()
        current = first if isinstance(first, str) else ""
        previous = ""
        while current:
            if current in seen:
                counters["cycle_count"] += 1
                break
            row = child_by_token.get(current)
            if row is None:
                counters["broken_next_link_count"] += 1
                break
            if row.get(child_parent_field) != parent_token:
                counters["cross_parent_link_count"] += 1
            if row.get("prev") != previous:
                counters["previous_link_mismatch_count"] += 1
            seen.add(current)
            previous = current
            next_value = row.get("next")
            if not isinstance(next_value, str):
                counters["malformed_next_token_count"] += 1
                break
            current = next_value
        if len(seen) != len(rows):
            counters["unreached_child_set_count"] += 1
        if rows and previous != last:
            counters["last_token_mismatch_count"] += 1

    return {
        "broken_next_link_count": counters["broken_next_link_count"],
        "cross_parent_link_count": counters["cross_parent_link_count"],
        "cycle_count": counters["cycle_count"],
        "declared_count_mismatch_count": counters["declared_count_mismatch_count"],
        "duplicate_child_token_count": len(children) - len(child_by_token),
        "last_token_mismatch_count": counters["last_token_mismatch_count"],
        "malformed_link_field_count": (
            counters["malformed_first_token_count"]
            + counters["malformed_next_token_count"]
            + counters["malformed_parent_token_count"]
        ),
        "missing_first_token_count": counters["missing_first_token_count"],
        "missing_parent_reference_count": missing_parent_reference_count,
        "previous_link_mismatch_count": counters["previous_link_mismatch_count"],
        "unreached_child_set_count": counters["unreached_child_set_count"],
    }


def verify_manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    manifest_path = root / MANIFEST_NAME
    manifest = require_object(load_json(manifest_path), MANIFEST_NAME)
    if manifest.get("official_source") is not False:
        raise AuditError("manifest must explicitly declare official_source=false")
    if manifest.get("scientific_claim_allowed") is not False:
        raise AuditError("manifest must explicitly declare scientific_claim_allowed=false")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise AuditError("manifest entries must be an array of objects")

    observed: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    status_counts: Counter[str] = Counter()
    for entry in sorted(entries, key=lambda row: str(row.get("path", ""))):
        relative_path = entry.get("path")
        size = entry.get("size")
        expected_hash = entry.get("sha256")
        if not isinstance(relative_path, str) or not relative_path:
            raise AuditError("manifest entry path must be a non-empty string")
        if relative_path in seen_paths:
            raise AuditError(f"duplicate manifest path: {relative_path}")
        seen_paths.add(relative_path)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise AuditError(f"invalid manifest size for {relative_path}")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise AuditError(f"invalid manifest sha256 for {relative_path}")
        target = safe_manifest_path(root, relative_path)
        if not target.is_file():
            raise AuditError(f"manifest file missing: {relative_path}")
        observed_size = target.stat().st_size
        if observed_size != size:
            raise AuditError(f"manifest size mismatch: {relative_path}")
        observed_hash = file_sha256(target)
        if observed_hash != expected_hash:
            raise AuditError(f"manifest hash mismatch: {relative_path}")
        status_counts[str(entry.get("status", "missing"))] += 1
        observed.append(
            {"path": relative_path, "sha256": observed_hash, "size": observed_size}
        )

    declared_count = manifest.get("file_count")
    declared_bytes = manifest.get("total_bytes")
    if declared_count != len(observed):
        raise AuditError("manifest file_count does not match entries")
    if declared_bytes != sum(row["size"] for row in observed):
        raise AuditError("manifest total_bytes does not match entries")

    all_regular_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    }
    unlisted_paths = sorted(all_regular_files - seen_paths)
    provenance = {
        "content_set_sha256": canonical_sha256(observed),
        "declared_file_count": declared_count,
        "declared_total_bytes": declared_bytes,
        "entry_status_counts": dict(sorted(status_counts.items())),
        "hash_algorithm": "sha256",
        "manifest_filename": MANIFEST_NAME,
        "manifest_sha256": file_sha256(manifest_path),
        "mirror_prefix": manifest.get("prefix"),
        "mirror_revision": manifest.get("revision"),
        "mirror_source": manifest.get("source"),
        "unlisted_regular_file_count": len(unlisted_paths),
        "unlisted_regular_files": unlisted_paths,
        "verified_entry_count": len(observed),
        "verified_total_bytes": sum(row["size"] for row in observed),
    }
    return manifest, provenance, seen_paths


def audit_dataset(dataset_root: Path) -> dict[str, Any]:
    root = dataset_root.resolve()
    if not root.is_dir():
        raise AuditError("dataset root is not a directory")
    manifest, provenance, manifest_paths = verify_manifest(root)

    data_info = require_object_rows(load_json(root / "data_info.json"), "data_info.json")
    v1_rows: dict[str, list[dict[str, Any]]] = {}
    for table in EXPECTED_V1_TABLES:
        path = root / V1_DIRECTORY / f"{table}.json"
        v1_rows[table] = require_object_rows(load_json(path), path.name)
    required_manifest_paths = {
        "data_info.json",
        *(f"{V1_DIRECTORY}/{table}.json" for table in EXPECTED_V1_TABLES),
    }
    missing_required_manifest_paths = required_manifest_paths - manifest_paths
    if missing_required_manifest_paths:
        raise AuditError(
            "required metadata is outside the verified manifest: "
            + ", ".join(sorted(missing_required_manifest_paths))
        )

    label_paths = sorted((root / "label").glob("*.json"), key=lambda path: path.name)
    manifest_label_paths = {
        path for path in manifest_paths if path.startswith("label/") and path.endswith(".json")
    }
    filesystem_label_paths = {path.relative_to(root).as_posix() for path in label_paths}
    if filesystem_label_paths != manifest_label_paths:
        raise AuditError("label file set does not match the verified manifest")

    table_reports = {
        "data_info": table_summary(data_info),
        **{table: table_summary(v1_rows[table]) for table in EXPECTED_V1_TABLES},
    }
    data_info_by_vehicle_frame = {
        row.get("vehicle_frame"): row
        for row in data_info
        if isinstance(row.get("vehicle_frame"), str)
    }
    duplicate_data_info_vehicle_frame_count = len(data_info) - len(data_info_by_vehicle_frame)

    samples = v1_rows["sample"]
    scenes = v1_rows["scene"]
    annotations = v1_rows["sample_annotation"]
    instances = v1_rows["instance"]
    sample_data = v1_rows["sample_data"]
    annotation_by_token = {
        row.get("token"): row
        for row in annotations
        if isinstance(row.get("token"), str)
    }
    sample_by_token = {
        row.get("token"): row for row in samples if isinstance(row.get("token"), str)
    }

    label_fields: Counter[str] = Counter()
    label_schema_variants: Counter[tuple[str, ...]] = Counter()
    label_row_count = 0
    label_tokens: set[str] = set()
    linked_annotation_tokens: set[str] = set()
    label_token_link_count = 0
    vehicle_frame_filename_mismatch_count = 0
    from_side_counts: Counter[str] = Counter()
    object_type_counts: Counter[str] = Counter()
    id_value_counts: dict[str, Counter[str]] = {
        field: Counter()
        for field in ("track_id", "veh_track_id", "inf_track_id", "veh_token", "inf_token")
    }
    track_to_sequences: dict[str, set[str]] = defaultdict(set)
    track_to_instances: dict[str, set[str]] = defaultdict(set)
    sequence_track_to_instances: dict[tuple[str, str], set[str]] = defaultdict(set)
    sequence_instance_to_tracks: dict[tuple[str, str], set[str]] = defaultdict(set)
    timestamp_tuple_cardinalities: list[int] = []
    vehicle_infrastructure_offsets: list[int] = []
    vehicle_sample_offsets: list[int] = []
    parseable_vehicle_timestamp_rows = 0
    parseable_infrastructure_timestamp_rows = 0

    for label_path in label_paths:
        rows = require_object_rows(load_json(label_path), label_path.name)
        frame_timestamp_tuples: set[tuple[Any, Any, Any, Any]] = set()
        info_row = data_info_by_vehicle_frame.get(label_path.stem)
        sequence = info_row.get("vehicle_sequence") if isinstance(info_row, dict) else None
        for row in rows:
            label_row_count += 1
            fields = tuple(sorted(str(key) for key in row))
            label_fields.update(fields)
            label_schema_variants[fields] += 1
            if row.get("veh_frame_id") != label_path.stem:
                vehicle_frame_filename_mismatch_count += 1
            if isinstance(row.get("token"), str):
                token = row["token"]
                label_tokens.add(token)
                annotation = annotation_by_token.get(token)
                if annotation is not None:
                    label_token_link_count += 1
                    linked_annotation_tokens.add(token)
                    track = row.get("track_id")
                    instance = annotation.get("instance_token")
                    if isinstance(track, str) and isinstance(instance, str):
                        track_to_instances[track].add(instance)
                        if isinstance(sequence, str):
                            track_to_sequences[track].add(sequence)
                            sequence_track_to_instances[(sequence, track)].add(instance)
                            sequence_instance_to_tracks[(sequence, instance)].add(track)
            if isinstance(row.get("from_side"), str):
                from_side_counts[row["from_side"]] += 1
            if isinstance(row.get("type"), str):
                object_type_counts[row["type"]] += 1
            for field, counts in id_value_counts.items():
                value = row.get(field)
                if isinstance(value, str):
                    counts[value] += 1
            vehicle_timestamp = integer_like(row.get("veh_pointcloud_timestamp"))
            infrastructure_timestamp = integer_like(row.get("inf_pointcloud_timestamp"))
            if vehicle_timestamp is not None:
                parseable_vehicle_timestamp_rows += 1
            if infrastructure_timestamp is not None:
                parseable_infrastructure_timestamp_rows += 1
            frame_timestamp_tuples.add(
                (
                    row.get("veh_pointcloud_timestamp"),
                    row.get("inf_pointcloud_timestamp"),
                    row.get("veh_frame_id"),
                    row.get("inf_frame_id"),
                )
            )
        timestamp_tuple_cardinalities.append(len(frame_timestamp_tuples))
        if len(frame_timestamp_tuples) == 1:
            vehicle_raw, infrastructure_raw, _, _ = next(iter(frame_timestamp_tuples))
            vehicle_timestamp = integer_like(vehicle_raw)
            infrastructure_timestamp = integer_like(infrastructure_raw)
            if vehicle_timestamp is not None and infrastructure_timestamp is not None:
                vehicle_infrastructure_offsets.append(infrastructure_timestamp - vehicle_timestamp)
            sample_timestamp = integer_like(
                sample_by_token.get(label_path.stem, {}).get("timestamp")
            )
            if vehicle_timestamp is not None and sample_timestamp is not None:
                vehicle_sample_offsets.append(vehicle_timestamp - sample_timestamp)

    table_reports["cooperative_label_files"] = {
        "field_presence_counts": dict(sorted(label_fields.items())),
        "file_count": len(label_paths),
        "observed_fields": sorted(label_fields),
        "row_count": label_row_count,
        "schema_variant_count": len(label_schema_variants),
    }

    vehicle_sequences = {
        row.get("vehicle_sequence")
        for row in data_info
        if isinstance(row.get("vehicle_sequence"), str)
    }
    infrastructure_sequences = {
        row.get("infrastructure_sequence")
        for row in data_info
        if isinstance(row.get("infrastructure_sequence"), str)
    }
    sequence_pairs = {
        (row.get("vehicle_sequence"), row.get("infrastructure_sequence"))
        for row in data_info
        if isinstance(row.get("vehicle_sequence"), str)
        and isinstance(row.get("infrastructure_sequence"), str)
    }
    scene_tokens = {
        row.get("token") for row in scenes if isinstance(row.get("token"), str)
    }
    samples_by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in samples:
        if isinstance(row.get("scene_token"), str):
            samples_by_scene[row["scene_token"]].append(row)
    sample_counts = [len(rows) for rows in samples_by_scene.values()]
    scene_chain = linked_chain_integrity(
        scenes,
        samples,
        child_parent_field="scene_token",
        declared_count_field="nbr_samples",
        first_field="first_sample_token",
        last_field="last_sample_token",
    )
    instance_chain = linked_chain_integrity(
        instances,
        annotations,
        child_parent_field="instance_token",
        declared_count_field="nbr_annotations",
        first_field="first_annotation_token",
        last_field="last_annotation_token",
    )

    sample_timestamps = [
        value
        for value in (integer_like(row.get("timestamp")) for row in samples)
        if value is not None
    ]
    sample_data_timestamps = [
        value
        for value in (integer_like(row.get("timestamp")) for row in sample_data)
        if value is not None
    ]
    within_scene_deltas: list[int] = []
    for rows in samples_by_scene.values():
        timestamps = sorted(
            value
            for value in (integer_like(row.get("timestamp")) for row in rows)
            if value is not None
        )
        within_scene_deltas.extend(
            timestamps[index + 1] - timestamps[index]
            for index in range(len(timestamps) - 1)
        )

    all_observed_fields = set(label_fields)
    for report in table_reports.values():
        all_observed_fields.update(report.get("observed_fields", []))
    arrival_fields = sorted(
        field
        for field in all_observed_fields
        if "arrival" in field.lower() or "receive" in field.lower()
    )
    network_fault_fields = sorted(
        field
        for field in all_observed_fields
        if any(term in field.lower() for term in ("latency", "delay", "packet", "drop", "loss"))
    )

    label_stems = {path.stem for path in label_paths}
    vehicle_frames = set(data_info_by_vehicle_frame)
    sample_tokens = set(sample_by_token)
    raw_sensor_entries = sorted(
        path for path in manifest_paths if path.lower().endswith(RAW_SENSOR_SUFFIXES)
    )
    referenced_sensor_filenames = {
        row.get("filename")
        for row in sample_data
        if isinstance(row.get("filename"), str) and row.get("filename")
    }

    track_fields = {}
    for field, counts in id_value_counts.items():
        non_sentinel_values = {value for value in counts if value != SENTINEL_ID}
        track_fields[field] = {
            "non_sentinel_string_row_count": sum(counts[value] for value in non_sentinel_values),
            "present_string_row_count": sum(counts.values()),
            "sentinel_row_count": counts[SENTINEL_ID],
            "unique_non_sentinel_count": len(non_sentinel_values),
        }

    sequence_track_ambiguity_count = sum(
        len(instance_tokens) > 1 for instance_tokens in sequence_track_to_instances.values()
    )
    if sequence_track_ambiguity_count == 0 and sequence_track_to_instances:
        track_scope_finding = (
            "The observed direct track_id is not globally unique; the composite "
            "(vehicle_sequence, track_id) maps without ambiguity to instance_token "
            "inside this audited subset only."
        )
    else:
        track_scope_finding = (
            "The audited subset does not support an unambiguous track key; protocol identity "
            "scope remains unresolved."
        )

    report: dict[str, Any] = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "boundary": {
            "allowed_purpose": manifest.get("purpose"),
            "dataset_classification": "third_party_mirror_metadata_subset",
            "official_source": False,
            "scientific_claim_allowed": False,
            "scientific_validation_performed": False,
            "statement": (
                "This report verifies only the pinned mirror subset's file integrity and "
                "observed metadata schema; it is not official-dataset or scientific validation."
            ),
        },
        "file_counts": {
            "data_info_files": sum(path == "data_info.json" for path in manifest_paths),
            "filesystem_regular_files_including_manifest": (
                len(manifest_paths) + 1 + provenance["unlisted_regular_file_count"]
            ),
            "json_payload_files": sum(path.endswith(".json") for path in manifest_paths),
            "label_files": len(label_paths),
            "manifest_entries": len(manifest_paths),
            "raw_sensor_payload_files": len(raw_sensor_entries),
            "referenced_sensor_payload_paths": len(referenced_sensor_filenames),
            "v1_table_files": sum(
                path.startswith(f"{V1_DIRECTORY}/") and path.endswith(".json")
                for path in manifest_paths
            ),
        },
        "integrity_findings": {
            "data_info_duplicate_vehicle_frame_count": duplicate_data_info_vehicle_frame_count,
            "data_info_vehicle_frame_missing_label_file_count": len(vehicle_frames - label_stems),
            "label_file_missing_data_info_vehicle_frame_count": len(label_stems - vehicle_frames),
            "label_file_missing_sample_token_count": len(label_stems - sample_tokens),
            "label_vehicle_frame_filename_mismatch_row_count": (
                vehicle_frame_filename_mismatch_count
            ),
            "sample_scene_chain": scene_chain,
            "sample_annotation_instance_chain": instance_chain,
        },
        "provenance": provenance,
        "sequence_findings": {
            "data_info_infrastructure_frame_unique_count": len(
                {
                    row.get("infrastructure_frame")
                    for row in data_info
                    if isinstance(row.get("infrastructure_frame"), str)
                }
            ),
            "data_info_infrastructure_sequence_count": len(infrastructure_sequences),
            "data_info_row_count": len(data_info),
            "data_info_sequence_pair_count": len(sequence_pairs),
            "data_info_vehicle_frame_unique_count": len(vehicle_frames),
            "data_info_vehicle_infrastructure_sequence_mismatch_row_count": sum(
                row.get("vehicle_sequence") != row.get("infrastructure_sequence")
                for row in data_info
            ),
            "data_info_vehicle_sequence_count": len(vehicle_sequences),
            "infrastructure_sequence_not_in_scene_count": len(
                infrastructure_sequences - scene_tokens
            ),
            "sample_count": len(samples),
            "scene_count": len(scenes),
            "scene_sample_count_maximum": max(sample_counts) if sample_counts else None,
            "scene_sample_count_minimum": min(sample_counts) if sample_counts else None,
            "scene_token_count": len(scene_tokens),
            "vehicle_sequence_not_in_scene_count": len(vehicle_sequences - scene_tokens),
        },
        "table_findings": table_reports,
        "timestamp_findings": {
            "arrival_timestamp_candidate_fields": arrival_fields,
            "arrival_timestamp_field_found": bool(arrival_fields),
            "clock_timebase_from_subset": None,
            "label_infrastructure_minus_vehicle_offset_raw": numeric_stats(
                vehicle_infrastructure_offsets
            ),
            "label_timestamp_tuple_cardinality_per_file": numeric_stats(
                timestamp_tuple_cardinalities
            ),
            "label_vehicle_minus_sample_offset_raw": numeric_stats(vehicle_sample_offsets),
            "network_fault_candidate_fields": network_fault_fields,
            "network_fault_field_found": bool(network_fault_fields),
            "parseable_infrastructure_pointcloud_timestamp_row_count": (
                parseable_infrastructure_timestamp_rows
            ),
            "parseable_vehicle_pointcloud_timestamp_row_count": parseable_vehicle_timestamp_rows,
            "sample_data_timestamp_raw": numeric_stats(sample_data_timestamps),
            "sample_timestamp_raw": numeric_stats(sample_timestamps),
            "timestamp_unit_from_subset": None,
            "within_scene_consecutive_sample_delta_raw": numeric_stats(within_scene_deltas),
        },
        "track_id_findings": {
            "annotation_instance_token_unique_count": len(
                {
                    row.get("instance_token")
                    for row in annotations
                    if isinstance(row.get("instance_token"), str)
                }
            ),
            "annotation_row_count": len(annotations),
            "annotation_rows_without_cooperative_label_token_count": (
                len(annotation_by_token) - len(linked_annotation_tokens)
            ),
            "cooperative_label_file_count": len(label_paths),
            "cooperative_label_row_count": label_row_count,
            "cooperative_label_token_duplicate_row_count": label_row_count - len(label_tokens),
            "cooperative_label_token_linked_to_annotation_count": label_token_link_count,
            "cooperative_label_token_unlinked_row_count": label_row_count - label_token_link_count,
            "from_side_counts": dict(sorted(from_side_counts.items())),
            "global_track_ids_reused_across_sequences_count": sum(
                len(sequences) > 1 for sequences in track_to_sequences.values()
            ),
            "global_track_to_multiple_instance_count": sum(
                len(instance_tokens) > 1 for instance_tokens in track_to_instances.values()
            ),
            "id_fields": track_fields,
            "instance_record_count": len(instances),
            "object_type_counts": dict(sorted(object_type_counts.items())),
            "sequence_instance_to_multiple_track_count": sum(
                len(track_ids) > 1 for track_ids in sequence_instance_to_tracks.values()
            ),
            "sequence_scoped_track_key_count": len(sequence_track_to_instances),
            "sequence_track_to_multiple_instance_count": sequence_track_ambiguity_count,
            "track_id_scope_finding": track_scope_finding,
        },
        "unresolved_protocol_fields": [
            {
                "field": "authoritative_dataset_identity_and_license_binding",
                "reason": (
                    "The manifest identifies a third-party mirror and provides no verified "
                    "official-release checksum or license binding for this subset."
                ),
                "status": "unresolved",
            },
            {
                "field": "official_train_validation_test_split",
                "reason": (
                    "The subset is named trainval, but no authoritative scene-level train, "
                    "validation, or test assignment is present."
                ),
                "status": "unresolved",
            },
            {
                "field": "timestamp_unit_and_clock_timebase",
                "reason": (
                    "Timestamp values are parseable, but the subset contains no authoritative "
                    "unit, epoch, synchronization, or clock-domain declaration."
                ),
                "status": "unresolved",
            },
            {
                "field": "message_arrival_and_network_fault_ground_truth",
                "reason": (
                    "No arrival-time, packet-loss, delay, or network-event field is observed; "
                    "sensor timestamp offsets are not network latency labels."
                ),
                "status": "unresolved",
            },
            {
                "field": "raw_sensor_payload_and_decode_integrity",
                "reason": (
                    "sample_data references sensor filenames, but the verified mirror subset "
                    "contains no raw sensor payload file."
                ),
                "status": "unresolved",
            },
            {
                "field": "prediction_horizon_sampling_and_target_definition",
                "reason": (
                    "The metadata exposes temporal links but no authoritative prediction "
                    "horizon, resampling rule, target mask, or miss threshold."
                ),
                "status": "unresolved",
            },
            {
                "field": "benchmark_metrics_thresholds_and_class_mapping",
                "reason": (
                    "Observed category and object-type fields do not define official tracking "
                    "or forecasting metrics, thresholds, filters, or benchmark class mapping."
                ),
                "status": "unresolved",
            },
            {
                "field": "full_spd_tfd_and_test_coverage",
                "reason": (
                    "This metadata-only SPD trainval subset cannot establish coverage of raw "
                    "SPD data, TFD data, or held-out test assets."
                ),
                "status": "unresolved",
            },
        ],
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only schema audit for a pinned non-official V2X-Seq metadata subset."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_dataset(args.dataset_root)
    except AuditError as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
