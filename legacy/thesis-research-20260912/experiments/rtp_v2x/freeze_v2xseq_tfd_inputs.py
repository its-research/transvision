#!/usr/bin/env python3
"""Freeze auditable V2X-Seq-TFD scene identities and three-way splits.

The utility hashes bytes that already exist under an operator-provided official
release root.  It does not authenticate mirrors, accept terms on behalf of an
operator, infer an unpublished split, or supply missing TFD files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v2xseq_real_canary import (
    ContractError,
    atomic_write,
    canonical_bytes,
    load_json,
    require_object,
    require_string,
    sha256_bytes,
    sha256_file,
)
from v2xseq_tfd_canary import (
    BASE_TRAJECTORY_COLUMNS,
    COOPERATIVE_EXTRA_COLUMNS,
    DATASET_NAME,
    OFFICIAL_REPOSITORY,
    TRAFFIC_LIGHT_COLUMNS,
    read_csv_rows,
    safe_dataset_path,
)


def split_values(source: dict[str, Any], key: str, aliases: tuple[str, ...]) -> list[str]:
    present = [name for name in aliases if name in source]
    if len(present) != 1:
        raise ContractError(f"TFD split source must contain exactly one alias for {key}: {aliases}")
    values = source[present[0]]
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise ContractError(f"TFD split source {present[0]} must be a scene-ID array")
    normalized = sorted(set(values))
    if len(normalized) != len(values):
        raise ContractError(f"TFD split source {present[0]} contains duplicates")
    if not normalized:
        raise ContractError(f"TFD split source {present[0]} must not be empty")
    return normalized


def scene_universe(root: Path, cooperative_directory: str) -> set[str]:
    directory = safe_dataset_path(root, cooperative_directory)
    if not directory.is_dir():
        raise ContractError("TFD cooperative trajectory directory is missing")
    scenes = {path.stem for path in directory.iterdir() if path.is_file() and path.suffix == ".csv"}
    if not scenes:
        raise ContractError("TFD cooperative trajectory directory contains no CSV scenes")
    return scenes


def freeze_split(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.dataset_root).resolve()
    source_path = Path(args.source).resolve()
    source_root = require_object(load_json(source_path), "TFD split source")
    if args.source_key:
        source = require_object(source_root.get(args.source_key), f"TFD split source {args.source_key}")
    else:
        source = source_root
    partitions = {
        "train": split_values(source, "train", ("train", "training")),
        "validation": split_values(source, "validation", ("validation", "val")),
        "test": split_values(source, "test", ("test", "testing")),
    }
    all_ids = [scene for values in partitions.values() for scene in values]
    if len(all_ids) != len(set(all_ids)):
        raise ContractError("TFD split source partitions overlap")
    universe = scene_universe(root, args.cooperative_directory)
    unknown = sorted(set(all_ids) - universe)
    if unknown:
        raise ContractError(f"TFD split source references unknown scene IDs: {unknown}")
    if args.require_complete and set(all_ids) != universe:
        missing = sorted(universe - set(all_ids))
        raise ContractError(f"TFD split source does not cover the scene universe: {missing}")
    return {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "protocol_id": require_string(args.protocol_id, "protocol_id"),
        "source_path_record": source_path.name,
        "source_sha256": sha256_file(source_path),
        "source_unit": "scene_id",
        "source_partition_counts": {name: len(values) for name, values in partitions.items()},
        "scene_universe_sha256": sha256_bytes(canonical_bytes(sorted(universe))),
        "partition_sha256": sha256_bytes(canonical_bytes(partitions)),
        "partitions": partitions,
    }


def file_entry(root: Path, relative: str, **metadata: Any) -> dict[str, Any]:
    path = safe_dataset_path(root, relative)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ContractError(f"cannot seal missing or empty TFD file: {relative}")
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        **metadata,
    }


def formatted_path(template: str, *, scene_id: str, intersection_id: str | None = None) -> str:
    try:
        return template.format(scene_id=scene_id, intersection_id=intersection_id)
    except (KeyError, ValueError) as exc:
        raise ContractError(f"invalid TFD path template: {template}") from exc


def freeze_identity(args: argparse.Namespace) -> dict[str, Any]:
    if not args.acknowledge_license:
        raise ContractError("--acknowledge-license is required")
    root = Path(args.dataset_root).resolve()
    scenes = sorted(set(args.scene))
    if len(scenes) != len(args.scene) or any(not scene for scene in scenes):
        raise ContractError("--scene values must be non-empty and unique")

    entries: list[dict[str, Any]] = []
    map_entries: dict[str, dict[str, Any]] = {}
    for scene_id in scenes:
        cooperative_relative = formatted_path(args.cooperative_template, scene_id=scene_id)
        cooperative_rows = read_csv_rows(
            safe_dataset_path(root, cooperative_relative),
            BASE_TRAJECTORY_COLUMNS | COOPERATIVE_EXTRA_COLUMNS,
            f"scene {scene_id} cooperative trajectory",
        )
        intersections = sorted({str(row["intersect_id"]) for row in cooperative_rows if row["intersect_id"]})
        if len(intersections) != 1:
            raise ContractError(f"scene {scene_id} must contain exactly one intersection ID")
        intersection_id = intersections[0]
        role_templates = (
            ("cooperative_trajectory", cooperative_relative, BASE_TRAJECTORY_COLUMNS | COOPERATIVE_EXTRA_COLUMNS),
            (
                "vehicle_trajectory",
                formatted_path(args.vehicle_template, scene_id=scene_id),
                BASE_TRAJECTORY_COLUMNS,
            ),
            (
                "infrastructure_trajectory",
                formatted_path(args.infrastructure_template, scene_id=scene_id),
                BASE_TRAJECTORY_COLUMNS,
            ),
            (
                "traffic_light",
                formatted_path(args.traffic_light_template, scene_id=scene_id),
                TRAFFIC_LIGHT_COLUMNS,
            ),
        )
        for role, relative, required_columns in role_templates:
            rows = read_csv_rows(
                safe_dataset_path(root, relative),
                required_columns,
                f"scene {scene_id} {role}",
            )
            observed_intersections = {
                str(row["intersect_id"]) for row in rows if row.get("intersect_id") not in {None, ""}
            }
            if observed_intersections != {intersection_id}:
                raise ContractError(f"scene {scene_id} {role} intersection identity does not match")
            entries.append(
                file_entry(
                    root,
                    relative,
                    role=role,
                    scene_id=scene_id,
                    intersection_id=intersection_id,
                )
            )
        if intersection_id not in map_entries:
            map_relative = formatted_path(
                args.map_template,
                scene_id=scene_id,
                intersection_id=intersection_id,
            )
            map_document = load_json(safe_dataset_path(root, map_relative))
            if not isinstance(map_document, dict) or not map_document:
                raise ContractError(f"intersection {intersection_id} HD map must be a non-empty object")
            map_entries[intersection_id] = file_entry(
                root,
                map_relative,
                role="hd_map",
                intersection_id=intersection_id,
            )
    entries.extend(map_entries.values())
    entries.sort(key=lambda row: row["path"])
    if len({entry["path"] for entry in entries}) != len(entries):
        raise ContractError("TFD identity would contain duplicate paths")
    return {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "release_id": require_string(args.release_id, "release_id"),
        "source_repository": OFFICIAL_REPOSITORY,
        "source_kind": "official_release",
        "license_acknowledged": True,
        "scope": "canary_subset",
        "scene_ids": scenes,
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
    split.add_argument("--source-key")
    split.add_argument("--require-complete", action="store_true")
    split.add_argument(
        "--cooperative-directory",
        default="cooperative-vehicle-infrastructure/cooperative-trajectories",
    )

    identity = subparsers.add_parser("identity")
    identity.add_argument("--dataset-root", required=True)
    identity.add_argument("--scene", action="append", required=True)
    identity.add_argument("--release-id", required=True)
    identity.add_argument("--acknowledge-license", action="store_true")
    identity.add_argument(
        "--cooperative-template",
        default="cooperative-vehicle-infrastructure/cooperative-trajectories/{scene_id}.csv",
    )
    identity.add_argument(
        "--vehicle-template",
        default="cooperative-vehicle-infrastructure/vehicle-trajectories/{scene_id}.csv",
    )
    identity.add_argument(
        "--infrastructure-template",
        default="cooperative-vehicle-infrastructure/infrastructure-trajectories/{scene_id}.csv",
    )
    identity.add_argument(
        "--traffic-light-template",
        default="cooperative-vehicle-infrastructure/traffic-light/{scene_id}.csv",
    )
    identity.add_argument("--map-template", default="maps/hdmap{intersection_id}.json")
    identity.add_argument("--output", required=True)

    args = parser.parse_args()
    try:
        document = freeze_split(args) if args.command == "split" else freeze_identity(args)
        output = Path(args.output).resolve()
        atomic_write(output, canonical_bytes(document))
    except ContractError as exc:
        raise SystemExit(f"V2X-Seq-TFD input freeze rejected: {exc}") from exc
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
