"""Read-only audit of an SPD metadata archive against a frozen split file."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

from transvision.models.event_track_v2x.wire import canonical_json_bytes


class SPDReleaseAuditError(ValueError):
    """Raised when release metadata cannot establish a reproducible cohort."""


def _load_json(raw: bytes, name: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise SPDReleaseAuditError(f"duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SPDReleaseAuditError(f"invalid JSON in {name}") from exc


def audit_spd_release(
    archive_path: Path,
    split_path: Path,
    *,
    metadata_member: str = "V2X-Seq-SPD/cooperative/data_info.json",
    label_prefix: str = "V2X-Seq-SPD/cooperative/label/",
) -> dict[str, object]:
    """Return a sealed availability report without extracting dataset bytes."""

    archive = Path(archive_path)
    split_file = Path(split_path)
    for path, name in ((archive, "archive"), (split_file, "split file")):
        if path.is_symlink() or not path.is_file():
            raise SPDReleaseAuditError(f"{name} must be a regular file")
    split_raw = split_file.read_bytes()
    split_document = _load_json(split_raw, "split file")
    if type(split_document) is not dict or set(split_document) != {
        "batch_split",
        "cooperative_split",
        "infrastructure_split",
        "vehicle_split",
    }:
        raise SPDReleaseAuditError("split document has missing or unknown sections")
    cooperative = split_document["cooperative_split"]
    if type(cooperative) is not dict:
        raise SPDReleaseAuditError("cooperative_split must be an object")

    try:
        with zipfile.ZipFile(archive) as archive_file:
            metadata_raw = archive_file.read(metadata_member)
            members = set(archive_file.namelist())
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise SPDReleaseAuditError("cannot read SPD metadata archive") from exc
    metadata = _load_json(metadata_raw, metadata_member)
    if type(metadata) is not list or not metadata:
        raise SPDReleaseAuditError("SPD data_info must be a non-empty array")
    required_row_fields = {
        "vehicle_frame",
        "infrastructure_frame",
        "vehicle_sequence",
        "infrastructure_sequence",
    }
    rows: dict[str, dict[str, object]] = {}
    for item in metadata:
        if type(item) is not dict or not required_row_fields <= set(item):
            raise SPDReleaseAuditError("SPD data_info row is malformed")
        frame_id = item["vehicle_frame"]
        if type(frame_id) is not str or not frame_id:
            raise SPDReleaseAuditError("vehicle_frame must be a non-empty string")
        if frame_id in rows:
            raise SPDReleaseAuditError("vehicle_frame must be unique in data_info")
        rows[frame_id] = item

    protocol_roles = {"train": "development", "val": "confirmatory"}
    expected_frame_counts = {"train": 7445, "val": 3316}
    expected_sequence_counts = {"train": 46, "val": 21}
    split_frame_sets: dict[str, frozenset[str]] = {}
    split_vehicle_sequence_sets: dict[str, frozenset[str]] = {}
    split_infrastructure_sequence_sets: dict[str, frozenset[str]] = {}
    reports: dict[str, object] = {}
    for split_name in sorted(cooperative):
        frame_ids = cooperative[split_name]
        if type(frame_ids) is not list or not all(
            type(frame_id) is str and frame_id for frame_id in frame_ids
        ):
            raise SPDReleaseAuditError(f"split {split_name} must be a string array")
        if len(set(frame_ids)) != len(frame_ids):
            raise SPDReleaseAuditError(f"split {split_name} contains duplicate frames")
        matched = [rows[frame_id] for frame_id in frame_ids if frame_id in rows]
        missing_metadata = sorted(set(frame_ids) - set(rows))
        missing_labels = sorted(
            frame_id
            for frame_id in frame_ids
            if f"{label_prefix}{frame_id}.json" not in members
        )
        vehicle_sequences = sorted(
            {str(row["vehicle_sequence"]) for row in matched}
        )
        infrastructure_sequences = sorted(
            {str(row["infrastructure_sequence"]) for row in matched}
        )
        required = split_name in protocol_roles
        split_frame_sets[split_name] = frozenset(frame_ids)
        split_vehicle_sequence_sets[split_name] = frozenset(vehicle_sequences)
        split_infrastructure_sequence_sets[split_name] = frozenset(
            infrastructure_sequences
        )
        reports[split_name] = {
            "protocol_ready": required
            and bool(frame_ids)
            and not missing_metadata
            and not missing_labels
            and len(frame_ids) == expected_frame_counts.get(split_name)
            and len(vehicle_sequences) == expected_sequence_counts.get(split_name)
            and len(infrastructure_sequences)
            == expected_sequence_counts.get(split_name),
            "protocol_required": required,
            "protocol_role": protocol_roles.get(split_name, "excluded"),
            "infrastructure_sequence_count": len(infrastructure_sequences),
            "label_count": len(frame_ids) - len(missing_labels),
            "matched_metadata_count": len(matched),
            "missing_label_count": len(missing_labels),
            "missing_label_preview": missing_labels[:10],
            "missing_metadata_count": len(missing_metadata),
            "missing_metadata_preview": missing_metadata[:10],
            "requested_frame_count": len(frame_ids),
            "vehicle_sequence_count": len(vehicle_sequences),
        }

    protocol_partition_disjoint = not (
        split_frame_sets.get("train", frozenset()).intersection(
            split_frame_sets.get("val", frozenset())
        )
        or split_vehicle_sequence_sets.get("train", frozenset()).intersection(
            split_vehicle_sequence_sets.get("val", frozenset())
        )
        or split_infrastructure_sequence_sets.get("train", frozenset()).intersection(
            split_infrastructure_sequence_sets.get("val", frozenset())
        )
    )
    if not protocol_partition_disjoint:
        for split_name in protocol_roles:
            report = reports.get(split_name)
            if isinstance(report, dict):
                report["protocol_ready"] = False

    payload: dict[str, object] = {
        "archive_name": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "cooperative_metadata_count": len(rows),
        "data_info_sha256": hashlib.sha256(metadata_raw).hexdigest(),
        "kind": "event_track_v2x_spd_release_audit_v1",
        "metadata_member": metadata_member,
        "official_test_policy": "excluded",
        "protocol_partition_disjoint": protocol_partition_disjoint,
        "schema_version": 1,
        "split_reports": reports,
        "split_sha256": hashlib.sha256(split_raw).hexdigest(),
    }
    return {
        **payload,
        "content_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    }


__all__ = ["SPDReleaseAuditError", "audit_spd_release"]
