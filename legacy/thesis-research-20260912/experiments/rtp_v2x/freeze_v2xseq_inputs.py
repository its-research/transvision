#!/usr/bin/env python3
"""Freeze auditable V2X-Seq-SPD identities and sequence-level split files.

The utility records bytes that already exist; it never authenticates a mirror
or grants a dataset license.  The operator must obtain the official release and
explicitly acknowledge its terms before an identity file can be emitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v2xseq_real_canary import (
    OFFICIAL_REPOSITORY,
    ContractError,
    atomic_write,
    canonical_bytes,
    load_json,
    require_object,
    require_string,
    safe_dataset_path,
    sha256_bytes,
    sha256_file,
    validate_official_asset_path,
)


def sequence_index(
    root: Path, cooperative_info: str
) -> tuple[set[str], dict[str, str]]:
    rows = load_json(safe_dataset_path(root, cooperative_info))
    if not isinstance(rows, list):
        raise ContractError("cooperative data_info must be an array")
    frames = {
        str(row["vehicle_frame"]): str(row["vehicle_sequence"])
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("vehicle_frame"), str)
        and isinstance(row.get("vehicle_sequence"), str)
    }
    if len(frames) != len(rows):
        raise ContractError(
            "cooperative data_info contains malformed or duplicate vehicle frames"
        )
    return set(frames.values()), frames


def split_values(
    source: dict[str, Any], key: str, aliases: tuple[str, ...]
) -> list[str]:
    present = [name for name in aliases if name in source]
    if len(present) != 1:
        raise ContractError(
            f"split source must contain exactly one alias for {key}: {aliases}"
        )
    values = source[present[0]]
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise ContractError(f"split source {present[0]} must be a sequence-ID array")
    normalized = sorted(set(values))
    if len(normalized) != len(values):
        raise ContractError(f"split source {present[0]} contains duplicates")
    return normalized


def freeze_split(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.dataset_root).resolve()
    source_path = Path(args.source).resolve()
    source_root = require_object(load_json(source_path), "split source")
    if args.source_key:
        source = require_object(
            source_root.get(args.source_key), f"split source {args.source_key}"
        )
    else:
        source = source_root
    source_partitions = {
        "train": split_values(source, "train", ("train", "training")),
        "validation": split_values(source, "validation", ("validation", "val")),
        "test": split_values(source, "test", ("test", "testing")),
    }
    all_ids = [value for values in source_partitions.values() for value in values]
    if len(all_ids) != len(set(all_ids)):
        raise ContractError("split source partitions overlap")
    known_sequences, frame_to_sequence = sequence_index(root, args.cooperative_info)
    source_ids = set(all_ids)
    if source_ids.issubset(known_sequences):
        source_unit = "sequence"
        partitions = source_partitions
        source_universe = known_sequences
    elif source_ids.issubset(frame_to_sequence):
        source_unit = "vehicle_frame"
        partitions = {
            name: sorted({frame_to_sequence[frame] for frame in frames})
            for name, frames in source_partitions.items()
        }
        converted = [value for values in partitions.values() for value in values]
        if len(converted) != len(set(converted)):
            raise ContractError(
                "frame-level split leaks at least one sequence across partitions"
            )
        source_universe = set(frame_to_sequence)
    else:
        unknown = sorted(source_ids - known_sequences - set(frame_to_sequence))
        raise ContractError(f"split source references unknown IDs: {unknown}")
    if args.require_complete and source_ids != source_universe:
        missing = sorted(source_universe - source_ids)
        raise ContractError(
            f"split source does not cover its dataset universe: {missing}"
        )
    return {
        "schema_version": 1,
        "dataset": "V2X-Seq-SPD",
        "protocol_id": require_string(args.protocol_id, "protocol_id"),
        "source_path_record": source_path.name,
        "source_sha256": sha256_file(source_path),
        "source_unit": source_unit,
        "source_partition_counts": {
            name: len(values) for name, values in source_partitions.items()
        },
        "sequence_universe_sha256": sha256_bytes(
            canonical_bytes(sorted(known_sequences))
        ),
        "partition_sha256": sha256_bytes(canonical_bytes(partitions)),
        "partitions": partitions,
    }


def parse_asset_template(value: str) -> tuple[str, str, str]:
    parts = value.split(":", 2)
    if len(parts) != 3 or any(not part for part in parts):
        raise ContractError("asset template must be AGENT:MODALITY:POSIX_PATH_TEMPLATE")
    agent, modality, template = parts
    if agent not in {"vehicle", "infrastructure"}:
        raise ContractError("asset template agent must be vehicle or infrastructure")
    if "{vehicle_frame}" not in template and "{infrastructure_frame}" not in template:
        raise ContractError("asset template must contain a frame placeholder")
    return agent, modality, template


def file_entry(root: Path, relative: str, **metadata: Any) -> dict[str, Any]:
    path = safe_dataset_path(root, relative)
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"cannot seal missing/non-regular file: {relative}")
    if path.stat().st_size <= 0:
        raise ContractError(f"cannot seal empty file: {relative}")
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        **metadata,
    }


def freeze_identity(args: argparse.Namespace) -> dict[str, Any]:
    if not args.acknowledge_license:
        raise ContractError("--acknowledge-license is required")
    root = Path(args.dataset_root).resolve()
    raw_sequences = args.sequence
    if isinstance(raw_sequences, str):
        raw_sequences = [raw_sequences]
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise ContractError("at least one --sequence is required")
    sequence_ids = sorted(
        require_string(value, f"sequence[{index}]")
        for index, value in enumerate(raw_sequences)
    )
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ContractError("--sequence values must be unique")
    cooperative_info = args.cooperative_info
    data_info = load_json(safe_dataset_path(root, cooperative_info))
    if not isinstance(data_info, list):
        raise ContractError("cooperative data_info must be an array")
    vehicle_info = args.vehicle_info
    vehicle_rows = load_json(safe_dataset_path(root, vehicle_info))
    if not isinstance(vehicle_rows, list):
        raise ContractError("vehicle data_info must be an array")
    vehicle_frames = {
        str(row.get("frame_id")): row
        for row in vehicle_rows
        if isinstance(row, dict) and isinstance(row.get("frame_id"), str)
    }
    if len(vehicle_frames) != len(vehicle_rows):
        raise ContractError(
            "vehicle data_info contains malformed or duplicate frame IDs"
        )
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ContractError("--max-frames must be positive")
    templates = [parse_asset_template(value) for value in args.asset_template]
    agents = {agent for agent, _modality, _template in templates}
    if agents != {"vehicle", "infrastructure"}:
        raise ContractError("asset templates must cover vehicle and infrastructure")

    entries: list[dict[str, Any]] = []
    entries.append(file_entry(root, cooperative_info, role="cooperative_index"))
    entries.append(file_entry(root, vehicle_info, role="vehicle_index"))
    for sequence_id in sequence_ids:
        frames = [
            row
            for row in data_info
            if isinstance(row, dict) and row.get("vehicle_sequence") == sequence_id
        ]
        frames.sort(key=lambda row: str(row.get("vehicle_frame", "")))
        if not frames:
            raise ContractError(f"unknown sequence: {sequence_id}")
        if args.max_frames is not None:
            frames = frames[: args.max_frames]
        for pair in frames:
            vehicle_frame = require_string(pair.get("vehicle_frame"), "vehicle_frame")
            vehicle_row = vehicle_frames.get(vehicle_frame)
            if vehicle_row is None:
                raise ContractError(
                    f"vehicle data_info is missing cooperative frame: {vehicle_frame}"
                )
            if vehicle_row.get("sequence_id") != sequence_id:
                raise ContractError(
                    f"vehicle data_info sequence mismatch for frame: {vehicle_frame}"
                )
            timestamp = vehicle_row.get("pointcloud_timestamp")
            if isinstance(timestamp, bool) or not isinstance(timestamp, (str, int)):
                raise ContractError(
                    f"vehicle data_info pointcloud_timestamp is missing for: {vehicle_frame}"
                )
            try:
                int(timestamp)
            except (TypeError, ValueError) as exc:
                raise ContractError(
                    f"vehicle data_info pointcloud_timestamp is invalid for: {vehicle_frame}"
                ) from exc
            infrastructure_frame = require_string(
                pair.get("infrastructure_frame"), "infrastructure_frame"
            )
            label_relative = args.label_template.format(
                vehicle_frame=vehicle_frame,
                infrastructure_frame=infrastructure_frame,
            )
            entries.append(
                file_entry(
                    root,
                    label_relative,
                    role="label",
                    sequence=sequence_id,
                    vehicle_frame=vehicle_frame,
                )
            )
            for agent, modality, template in templates:
                relative = template.format(
                    vehicle_frame=vehicle_frame,
                    infrastructure_frame=infrastructure_frame,
                )
                validate_official_asset_path(
                    agent, modality, relative, "asset template result"
                )
                entries.append(
                    file_entry(
                        root,
                        relative,
                        role="raw_sensor",
                        sequence=sequence_id,
                        vehicle_frame=vehicle_frame,
                        infrastructure_frame=infrastructure_frame,
                        agent=agent,
                        modality=modality,
                    )
                )
    entries.sort(key=lambda row: row["path"])
    if len({row["path"] for row in entries}) != len(entries):
        raise ContractError("identity would contain duplicate paths")
    return {
        "schema_version": 1,
        "dataset": "V2X-Seq-SPD",
        "release_id": require_string(args.release_id, "release_id"),
        "source_repository": OFFICIAL_REPOSITORY,
        "source_kind": "official_release",
        "license_acknowledged": True,
        "scope": "canary_subset",
        "sequence_ids": sequence_ids,
        "content_sha256": sha256_bytes(canonical_bytes(entries)),
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split")
    split.add_argument("--dataset-root", required=True)
    split.add_argument("--source", required=True)
    split.add_argument("--protocol-id", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--require-complete", action="store_true")
    split.add_argument("--cooperative-info", default="cooperative/data_info.json")
    split.add_argument("--source-key")

    identity = subparsers.add_parser("identity")
    identity.add_argument("--dataset-root", required=True)
    identity.add_argument(
        "--sequence",
        action="append",
        required=True,
        help="sequence ID to seal; repeat for train and validation canary sequences",
    )
    identity.add_argument("--release-id", required=True)
    identity.add_argument("--acknowledge-license", action="store_true")
    identity.add_argument("--max-frames", type=int)
    identity.add_argument("--cooperative-info", default="cooperative/data_info.json")
    identity.add_argument("--vehicle-info", default="vehicle-side/data_info.json")
    identity.add_argument(
        "--label-template",
        default="cooperative/label/{vehicle_frame}.json",
    )
    identity.add_argument(
        "--asset-template",
        action="append",
        required=True,
        help="AGENT:MODALITY:POSIX_PATH_TEMPLATE; repeat for each required sensor",
    )
    identity.add_argument("--output", required=True)

    args = parser.parse_args()
    try:
        document = (
            freeze_split(args) if args.command == "split" else freeze_identity(args)
        )
        output = Path(args.output).resolve()
        atomic_write(output, canonical_bytes(document))
    except ContractError as exc:
        raise SystemExit(f"V2X-Seq input freeze rejected: {exc}") from exc
    print(
        json.dumps(
            {"output": str(output), "sha256": sha256_file(output)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
