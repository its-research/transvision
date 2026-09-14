#!/usr/bin/env python3
"""Fail-closed real V2X-Seq-TFD forecasting canary.

This runner verifies frozen official-release bytes, a three-way scene split,
the forecasting protocol, a licensed pinned upstream revision, and the exact
repository adapter bytes before any model call.  It uses one training scene
for forward/backward/optimizer phases and disjoint validation scenes for
inference and evaluation.  Validation inference receives history only; future
ground truth is supplied after predictions have been materialized.

The runner is deliberately incapable of publishing a scientific result.  All
outputs are marked ``diagnostic_only`` and ``scientific_claim_allowed=false``.
It never edits ``results/registry.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from v2xseq_real_canary import (
    ContractError,
    atomic_write,
    canonical_bytes,
    git_identity,
    json_safe,
    load_json,
    require_object,
    require_string,
    resolve_repository_file,
    sha256_bytes,
    sha256_file,
    validate_digest,
)
from evaluator_conformance import (
    EvaluationInputError,
    ForecastSample as ReferenceForecastSample,
    evaluate_forecasts,
)
from history_adapter_subprocess import (
    HistoryAdapterError,
    HistoryOnlyAdapterProcess,
    validate_isolation_contract,
)
from provenance_seals import (
    SealError,
    loads_json_strict,
    validate_git_seal,
    validate_installed_environment_manifest,
)


SCHEMA_VERSION = 1
DATASET_NAME = "V2X-Seq-TFD"
OFFICIAL_REPOSITORY = "https://github.com/AIR-THU/DAIR-V2X-Seq"
CANARY_TYPE = "real_v2xseq_tfd_forecast"
REQUIRED_SCENE_ROLES = {
    "cooperative_trajectory",
    "vehicle_trajectory",
    "infrastructure_trajectory",
    "traffic_light",
}
REQUIRED_ARTIFACTS = (
    "checkpoint.bin",
    "environment.json",
    "metrics.json",
    "predictions.jsonl",
    "stderr.log",
    "stdout.log",
)
REQUIRED_SOURCE_SEAL_PATHS = frozenset(
    {
        "experiments/clearml/configs/v2xseq_forecast_canary.json",
        "experiments/clearml/protocols/evaluator-conformance-v1.json",
        "experiments/clearml/protocols/history-only-subprocess-v1.json",
        "experiments/clearml/protocols/upstream-models-v1.json",
        "experiments/clearml/protocols/v2xseq-forecast-v1.json",
        "experiments/rtp_v2x/evaluator_conformance.py",
        "experiments/rtp_v2x/freeze_v2xseq_tfd_inputs.py",
        "experiments/rtp_v2x/history_adapter_subprocess.py",
        "experiments/rtp_v2x/history_adapter_worker.py",
        "experiments/rtp_v2x/provenance_seals.py",
        "experiments/rtp_v2x/v2xseq_real_canary.py",
        "experiments/rtp_v2x/v2xseq_tfd_canary.py",
    }
)

BASE_TRAJECTORY_COLUMNS = {
    "city",
    "timestamp",
    "id",
    "type",
    "sub_type",
    "tag",
    "x",
    "y",
    "z",
    "length",
    "width",
    "height",
    "theta",
    "v_x",
    "v_y",
    "intersect_id",
}
COOPERATIVE_EXTRA_COLUMNS = {
    "vic_tag",
    "from_side",
    "car_side_id",
    "road_side_id",
}
TRAFFIC_LIGHT_COLUMNS = {
    "city",
    "timestamp",
    "x",
    "y",
    "direction",
    "lane_id",
    "color_1",
    "remain_1",
    "color_2",
    "remain_2",
    "color_3",
    "remain_3",
    "intersect_id",
}


def safe_dataset_path(root: Path, relative: str) -> Path:
    """Resolve a manifest path under ``root`` and reject symlink traversal."""

    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ContractError(f"unsafe dataset path: {relative!r}")
    root = root.resolve()
    unresolved = root.joinpath(*pure.parts)
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ContractError(
                f"dataset evidence may not traverse a symlink: {relative}"
            )
    target = unresolved.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"dataset path escapes root: {relative!r}") from exc
    return target


def validate_identity(root: Path, document: object) -> dict[str, dict[str, Any]]:
    identity = require_object(document, "TFD dataset identity")
    if identity.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("TFD identity schema_version must be 1")
    if identity.get("dataset") != DATASET_NAME:
        raise ContractError(f"TFD identity must name {DATASET_NAME}")
    if identity.get("source_repository") != OFFICIAL_REPOSITORY:
        raise ContractError("TFD identity must point to the official repository")
    if identity.get("source_kind") != "official_release":
        raise ContractError("TFD identity source_kind must be official_release")
    if identity.get("license_acknowledged") is not True:
        raise ContractError("TFD data-use terms must be acknowledged explicitly")
    require_string(identity.get("release_id"), "TFD identity release_id")
    if identity.get("scope") not in {"canary_subset", "complete_release"}:
        raise ContractError(
            "TFD identity scope must be canary_subset or complete_release"
        )
    scene_ids = identity.get("scene_ids")
    if (
        not isinstance(scene_ids, list)
        or not scene_ids
        or any(not isinstance(item, str) or not item for item in scene_ids)
        or scene_ids != sorted(set(scene_ids))
    ):
        raise ContractError(
            "TFD identity scene_ids must be sorted, unique, and non-empty"
        )
    entries = identity.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ContractError("TFD identity entries must be a non-empty array")

    index: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(entries):
        entry = require_object(raw, f"TFD identity entry {position}")
        relative = require_string(entry.get("path"), f"entry {position}.path")
        if relative in index:
            raise ContractError(f"duplicate TFD identity path: {relative}")
        expected_size = entry.get("size")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
        ):
            raise ContractError(f"entry {position}.size must be a positive integer")
        expected_hash = validate_digest(entry.get("sha256"), f"entry {position}.sha256")
        role = require_string(entry.get("role"), f"entry {position}.role")
        if role not in REQUIRED_SCENE_ROLES | {"hd_map"}:
            raise ContractError(f"unsupported TFD identity role: {role}")
        target = safe_dataset_path(root, relative)
        if not target.is_file():
            raise ContractError(f"TFD identity file is missing: {relative}")
        if target.stat().st_size != expected_size:
            raise ContractError(f"TFD identity size mismatch: {relative}")
        if sha256_file(target) != expected_hash:
            raise ContractError(f"TFD identity hash mismatch: {relative}")
        normalized = {**entry, "role": role}
        if role in REQUIRED_SCENE_ROLES:
            scene_id = require_string(
                entry.get("scene_id"), f"entry {position}.scene_id"
            )
            if scene_id not in scene_ids:
                raise ContractError(
                    f"TFD identity entry references undeclared scene: {scene_id}"
                )
            require_string(
                entry.get("intersection_id"), f"entry {position}.intersection_id"
            )
        else:
            require_string(
                entry.get("intersection_id"), f"entry {position}.intersection_id"
            )
        index[relative] = normalized

    expected_content_hash = validate_digest(
        identity.get("content_sha256"), "TFD identity content_sha256"
    )
    sorted_entries = sorted(index.values(), key=lambda row: row["path"])
    if sha256_bytes(canonical_bytes(sorted_entries)) != expected_content_hash:
        raise ContractError("TFD identity content_sha256 mismatch")

    for scene_id in scene_ids:
        scene_entries = [
            entry for entry in index.values() if entry.get("scene_id") == scene_id
        ]
        roles = [entry["role"] for entry in scene_entries]
        if set(roles) != REQUIRED_SCENE_ROLES or len(roles) != len(
            REQUIRED_SCENE_ROLES
        ):
            raise ContractError(
                f"scene {scene_id} must seal each required trajectory/light role exactly once"
            )
        intersections = {entry["intersection_id"] for entry in scene_entries}
        if len(intersections) != 1:
            raise ContractError(f"scene {scene_id} has ambiguous intersection identity")
        intersection = next(iter(intersections))
        maps = [
            entry
            for entry in index.values()
            if entry["role"] == "hd_map"
            and entry.get("intersection_id") == intersection
        ]
        if len(maps) != 1:
            raise ContractError(
                f"scene {scene_id} must resolve to exactly one sealed HD map"
            )
    return index


def validate_split(document: object, *, protocol_id: str) -> dict[str, list[str]]:
    split = require_object(document, "TFD frozen split")
    if split.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("TFD frozen split schema_version must be 1")
    if split.get("dataset") != DATASET_NAME:
        raise ContractError(f"TFD frozen split must name {DATASET_NAME}")
    if split.get("protocol_id") != protocol_id:
        raise ContractError("TFD frozen split protocol_id does not match protocol")
    if split.get("source_unit") != "scene_id":
        raise ContractError("TFD frozen split source_unit must be scene_id")
    validate_digest(split.get("source_sha256"), "TFD frozen split source_sha256")
    validate_digest(split.get("scene_universe_sha256"), "TFD scene_universe_sha256")
    partitions = split.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {
        "train",
        "validation",
        "test",
    }:
        raise ContractError(
            "TFD frozen split must contain train, validation, and test exactly"
        )
    normalized: dict[str, list[str]] = {}
    seen: set[str] = set()
    for name in ("train", "validation", "test"):
        values = partitions[name]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(item, str) or not item for item in values)
            or values != sorted(set(values))
        ):
            raise ContractError(
                f"TFD split {name} must be sorted, unique, and non-empty"
            )
        overlap = seen.intersection(values)
        if overlap:
            raise ContractError(f"TFD split partitions overlap: {sorted(overlap)}")
        seen.update(values)
        normalized[name] = list(values)
    counts = split.get("source_partition_counts")
    if not isinstance(counts, dict) or counts != {
        name: len(values) for name, values in normalized.items()
    }:
        raise ContractError("TFD frozen split source_partition_counts mismatch")
    if sha256_bytes(canonical_bytes(normalized)) != validate_digest(
        split.get("partition_sha256"), "TFD partition_sha256"
    ):
        raise ContractError("TFD frozen split partition_sha256 mismatch")
    if sha256_bytes(canonical_bytes(sorted(seen))) != split["scene_universe_sha256"]:
        raise ContractError(
            "TFD frozen split does not cover its complete scene universe"
        )
    return normalized


def _reject_unfrozen_keys(value: Any, label: str = "protocol") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).endswith("_not_frozen"):
                raise ContractError(f"{label} contains an unfrozen section: {key}")
            _reject_unfrozen_keys(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_unfrozen_keys(child, f"{label}[{index}]")


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _positive_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ContractError(f"{label} must be finite numeric")
    if value <= 0:
        raise ContractError(f"{label} must be positive")
    return float(value)


def validate_protocol(document: object) -> dict[str, Any]:
    protocol = require_object(document, "TFD forecasting protocol")
    _reject_unfrozen_keys(protocol)
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("TFD protocol schema_version must be 1")
    require_string(protocol.get("protocol_id"), "TFD protocol_id")
    if protocol.get("status") != "frozen":
        raise ContractError("TFD forecasting protocol status must be frozen")
    if protocol.get("dataset") != DATASET_NAME:
        raise ContractError(f"TFD forecasting protocol must name {DATASET_NAME}")
    if (
        protocol.get("causal_rule")
        != "all history observations must have arrival_time <= decision_time"
    ):
        raise ContractError(
            "TFD protocol causal rule is not the inclusive frozen cutoff"
        )
    if protocol.get("unresolved") != []:
        raise ContractError("TFD protocol unresolved list must be empty")
    require_string(protocol.get("coordinate_frame"), "TFD protocol coordinate_frame")
    require_string(protocol.get("target_tag"), "TFD protocol target_tag")
    if protocol.get("arrival_model") != "arrival_time_ns = event_time_ns":
        raise ContractError(
            "TFD canary baseline arrival model must be frozen to zero latency"
        )

    time_model = require_object(protocol.get("time_model"), "TFD protocol time_model")
    require_string(time_model.get("native_timestamp_unit"), "native_timestamp_unit")
    native_scale = _positive_int(
        time_model.get("native_timestamp_scale_to_ns"), "native_timestamp_scale_to_ns"
    )
    sampling_interval = _positive_int(
        time_model.get("sampling_interval_ns"), "sampling_interval_ns"
    )
    history_steps = _positive_int(protocol.get("history_steps"), "history_steps")
    future_steps = _positive_int(protocol.get("future_steps"), "future_steps")
    if protocol.get("history_duration_ns") != history_steps * sampling_interval:
        raise ContractError(
            "history_duration_ns must equal history_steps * sampling_interval_ns"
        )
    if protocol.get("forecast_duration_ns") != future_steps * sampling_interval:
        raise ContractError(
            "forecast_duration_ns must equal future_steps * sampling_interval_ns"
        )
    modes = _positive_int(protocol.get("modes"), "modes")
    if modes < 2:
        raise ContractError(
            "multimodal forecast protocol must require at least two modes"
        )
    dimensions = _positive_int(protocol.get("spatial_dimensions"), "spatial_dimensions")
    if dimensions not in {2, 3}:
        raise ContractError("spatial_dimensions must be 2 or 3")

    metrics = protocol.get("metrics")
    if (
        not isinstance(metrics, list)
        or not metrics
        or any(not isinstance(item, str) or not item for item in metrics)
        or len(metrics) != len(set(metrics))
    ):
        raise ContractError(
            "TFD protocol metrics must be a unique non-empty string array"
        )
    required_metrics = {"minADE", "minFDE", "MR", "NLL", "ECE", "Brier", "AURC"}
    if set(metrics) != required_metrics:
        raise ContractError(
            "TFD protocol metrics must contain the seven frozen thesis metrics exactly"
        )
    _positive_number(protocol.get("miss_rate_threshold_m"), "miss_rate_threshold_m")
    calibration = require_object(
        protocol.get("calibration_definition"), "calibration_definition"
    )
    require_string(calibration.get("ece_event"), "calibration_definition.ece_event")
    _positive_int(calibration.get("ece_bins"), "calibration_definition.ece_bins")
    require_string(calibration.get("brier_event"), "calibration_definition.brier_event")
    _positive_number(
        calibration.get("nll_sigma_m"), "calibration_definition.nll_sigma_m"
    )
    if calibration.get("aurc_coverage_grid") != "all_prefixes":
        raise ContractError("aurc_coverage_grid must be frozen to all_prefixes")
    fault_grid = require_object(protocol.get("fault_grid"), "TFD protocol fault_grid")
    if not fault_grid:
        raise ContractError("TFD protocol fault_grid must be frozen and non-empty")
    if native_scale <= 0 or sampling_interval <= 0:  # guarded above; documents intent
        raise ContractError("TFD timestamp normalization is invalid")
    return protocol


def validate_upstream(
    document: object,
    *,
    source_id: str,
    expected_revision: str,
) -> dict[str, Any]:
    manifest = require_object(document, "upstream source manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("upstream source manifest schema_version must be 1")
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ContractError("upstream source manifest sources must be an array")
    matches = [
        source
        for source in sources
        if isinstance(source, dict) and source.get("id") == source_id
    ]
    if len(matches) != 1:
        raise ContractError("model upstream_id must resolve exactly once")
    source = matches[0]
    revision = require_string(source.get("revision"), "upstream revision")
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ContractError("upstream revision must be a full Git object ID")
    if expected_revision != revision:
        raise ContractError(
            "model upstream_revision does not match the pinned source manifest"
        )
    repository = require_string(source.get("repository"), "upstream repository")
    if not repository.startswith("https://github.com/"):
        raise ContractError("upstream repository must be a pinned HTTPS GitHub source")
    if source.get("license_file_observed") is not True:
        raise ContractError("forecasting upstream has no verified license file")
    if source.get("redistribution_allowed_by_this_manifest") is not True:
        raise ContractError("forecasting upstream is not approved for this integration")
    capabilities = require_object(source.get("capabilities"), "upstream capabilities")
    if capabilities.get("v2x_seq_tfd_forecasting") is not True:
        raise ContractError(
            "pinned upstream does not declare V2X-Seq-TFD forecasting support"
        )
    return source


def validate_reference_evaluator_contract(
    repository_root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    reference = require_object(
        protocol.get("reference_evaluator_contract"),
        "reference_evaluator_contract",
    )
    if set(reference) != {"contract_id", "path", "sha256"}:
        raise ContractError(
            "reference_evaluator_contract must contain contract_id, path, and sha256 exactly"
        )
    contract_id = require_string(
        reference.get("contract_id"), "reference evaluator contract_id"
    )
    path = resolve_repository_file(
        repository_root,
        require_string(reference.get("path"), "reference evaluator path"),
        "reference evaluator contract",
    )
    expected_hash = validate_digest(
        reference.get("sha256"), "reference evaluator sha256"
    )
    if sha256_file(path) != expected_hash:
        raise ContractError("reference evaluator contract SHA-256 mismatch")
    document = require_object(load_json(path), "reference evaluator contract")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("contract_id") != contract_id
    ):
        raise ContractError("reference evaluator contract identity mismatch")
    if document.get("status") != "reference_conformance_only":
        raise ContractError(
            "reference evaluator contract must remain reference_conformance_only"
        )
    if document.get("scientific_claim_allowed") is not False:
        raise ContractError(
            "reference evaluator contract cannot allow scientific claims"
        )
    if document.get("official_evaluator_equivalence") is not False:
        raise ContractError(
            "reference evaluator contract cannot assert official equivalence"
        )
    if document.get("implementation") != "experiments/rtp_v2x/evaluator_conformance.py":
        raise ContractError("reference evaluator implementation path is not frozen")
    resolve_repository_file(
        repository_root,
        document["implementation"],
        "reference evaluator implementation",
    )
    forecasting = require_object(
        document.get("forecasting"), "reference evaluator forecasting"
    )
    metric_definitions = require_object(
        forecasting.get("metrics"), "reference evaluator forecasting metrics"
    )
    if set(metric_definitions) != {
        "minADE",
        "minFDE",
        "MR",
        "NLL",
        "Brier",
        "ECE",
        "AURC",
    }:
        raise ContractError("reference evaluator forecasting metrics are incomplete")
    numeric_policy = require_object(
        document.get("numeric_policy"), "reference evaluator numeric_policy"
    )
    tolerance = numeric_policy.get("conformance_absolute_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ContractError("reference evaluator conformance tolerance is invalid")
    return path, document


def read_csv_rows(
    path: Path, required_columns: set[str], label: str
) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(required_columns - fieldnames)
            if missing:
                raise ContractError(f"{label} is missing required columns: {missing}")
            extra = sorted(fieldnames - required_columns)
            if extra:
                raise ContractError(f"{label} contains unfrozen extra columns: {extra}")
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContractError(f"cannot read {label}: {type(exc).__name__}") from exc
    if not rows:
        raise ContractError(f"{label} contains no data rows")
    return rows


def timestamp_ns(value: str, scale_to_ns: int, label: str) -> int:
    try:
        normalized = Decimal(value) * Decimal(scale_to_ns)
        if not normalized.is_finite():
            raise InvalidOperation
        integral = normalized.to_integral_value()
        if normalized != integral:
            raise InvalidOperation
        return int(integral)
    except (InvalidOperation, TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{label} has a non-numeric timestamp") from exc


def finite_coordinate(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ContractError(f"{label} is not finite")
    return number


def normalized_history_rows(
    rows: Sequence[dict[str, str]],
    *,
    scale_to_ns: int,
    history_timestamps_ns: set[int],
    label: str,
) -> list[dict[str, Any]]:
    if not history_timestamps_ns:
        raise ContractError(f"{label} received an empty frozen history window")
    decision_time_ns = max(history_timestamps_ns)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        event_time = timestamp_ns(row["timestamp"], scale_to_ns, label)
        arrival_time = event_time
        if event_time in history_timestamps_ns and arrival_time <= decision_time_ns:
            normalized.append(
                {
                    **row,
                    "event_time_ns": event_time,
                    "arrival_time_ns": arrival_time,
                }
            )
    normalized.sort(
        key=lambda row: (
            row["event_time_ns"],
            str(row.get("id", "")),
            str(row.get("lane_id", "")),
        )
    )
    return normalized


@dataclass(frozen=True)
class ForecastSample:
    scene_id: str
    target_id: str
    history: dict[str, Any]
    future: dict[str, Any]
    input_sha256: str

    def inference_payload(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "target_id": self.target_id,
            "history": self.history,
            "input_sha256": self.input_sha256,
        }

    def training_payload(self) -> dict[str, Any]:
        return {**self.inference_payload(), "ground_truth": self.future}

    def evaluation_target(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "target_id": self.target_id,
            "ground_truth": self.future,
            "input_sha256": self.input_sha256,
        }


def _entry_for_scene(
    identity_index: Mapping[str, dict[str, Any]], scene_id: str, role: str
) -> dict[str, Any]:
    matches = [
        entry
        for entry in identity_index.values()
        if entry.get("scene_id") == scene_id and entry.get("role") == role
    ]
    if len(matches) != 1:
        raise ContractError(f"scene {scene_id} must resolve exactly one {role}")
    return matches[0]


def load_forecast_sample(
    root: Path,
    identity_index: Mapping[str, dict[str, Any]],
    *,
    scene_id: str,
    protocol: Mapping[str, Any],
) -> ForecastSample:
    cooperative_entry = _entry_for_scene(
        identity_index, scene_id, "cooperative_trajectory"
    )
    vehicle_entry = _entry_for_scene(identity_index, scene_id, "vehicle_trajectory")
    infrastructure_entry = _entry_for_scene(
        identity_index, scene_id, "infrastructure_trajectory"
    )
    light_entry = _entry_for_scene(identity_index, scene_id, "traffic_light")
    intersection_id = cooperative_entry["intersection_id"]
    map_matches = [
        entry
        for entry in identity_index.values()
        if entry.get("role") == "hd_map"
        and entry.get("intersection_id") == intersection_id
    ]
    if len(map_matches) != 1:
        raise ContractError(f"scene {scene_id} must resolve one HD map")

    cooperative = read_csv_rows(
        safe_dataset_path(root, cooperative_entry["path"]),
        BASE_TRAJECTORY_COLUMNS | COOPERATIVE_EXTRA_COLUMNS,
        f"scene {scene_id} cooperative trajectory",
    )
    vehicle = read_csv_rows(
        safe_dataset_path(root, vehicle_entry["path"]),
        BASE_TRAJECTORY_COLUMNS,
        f"scene {scene_id} vehicle trajectory",
    )
    infrastructure = read_csv_rows(
        safe_dataset_path(root, infrastructure_entry["path"]),
        BASE_TRAJECTORY_COLUMNS,
        f"scene {scene_id} infrastructure trajectory",
    )
    lights = read_csv_rows(
        safe_dataset_path(root, light_entry["path"]),
        TRAFFIC_LIGHT_COLUMNS,
        f"scene {scene_id} traffic light",
    )
    map_document = load_json(safe_dataset_path(root, map_matches[0]["path"]))
    if not isinstance(map_document, dict) or not map_document:
        raise ContractError(f"scene {scene_id} HD map must be a non-empty JSON object")

    scale = protocol["time_model"]["native_timestamp_scale_to_ns"]
    timestamp_values = sorted(
        {
            timestamp_ns(
                row["timestamp"], scale, f"scene {scene_id} cooperative trajectory"
            )
            for row in cooperative
        }
    )
    total_steps = protocol["history_steps"] + protocol["future_steps"]
    if len(timestamp_values) != total_steps:
        raise ContractError(
            f"scene {scene_id} has {len(timestamp_values)} timestamps; protocol requires {total_steps}"
        )
    for previous, current in zip(timestamp_values, timestamp_values[1:]):
        if current - previous != protocol["time_model"]["sampling_interval_ns"]:
            raise ContractError(
                f"scene {scene_id} timestamps do not match sampling_interval_ns"
            )
    history_timestamps = timestamp_values[: protocol["history_steps"]]
    future_timestamps = timestamp_values[protocol["history_steps"] :]
    decision_time = history_timestamps[-1]

    target_tag = protocol["target_tag"]
    target_ids = sorted(
        {str(row["id"]) for row in cooperative if row.get("tag") == target_tag}
    )
    if len(target_ids) != 1:
        raise ContractError(
            f"scene {scene_id} must contain exactly one {target_tag} identity"
        )
    target_id = target_ids[0]
    target_rows: dict[int, dict[str, str]] = {}
    for row in cooperative:
        if str(row["id"]) != target_id or row.get("tag") != target_tag:
            continue
        timestamp = timestamp_ns(row["timestamp"], scale, f"scene {scene_id} target")
        if timestamp in target_rows:
            raise ContractError(
                f"scene {scene_id} target has duplicate timestamp {timestamp}"
            )
        target_rows[timestamp] = row
    if set(target_rows) != set(timestamp_values):
        raise ContractError(f"scene {scene_id} target trajectory is incomplete")

    dimensions = protocol["spatial_dimensions"]
    coordinate_names = ("x", "y", "z")[:dimensions]
    future_positions = [
        [
            finite_coordinate(
                target_rows[timestamp][axis], f"scene {scene_id} target {axis}"
            )
            for axis in coordinate_names
        ]
        for timestamp in future_timestamps
    ]
    history = {
        "decision_time_ns": decision_time,
        "timestamps_ns": history_timestamps,
        "cooperative_trajectories": normalized_history_rows(
            cooperative,
            scale_to_ns=scale,
            history_timestamps_ns=set(history_timestamps),
            label=f"scene {scene_id} cooperative trajectory",
        ),
        "vehicle_trajectories": normalized_history_rows(
            vehicle,
            scale_to_ns=scale,
            history_timestamps_ns=set(history_timestamps),
            label=f"scene {scene_id} vehicle trajectory",
        ),
        "infrastructure_trajectories": normalized_history_rows(
            infrastructure,
            scale_to_ns=scale,
            history_timestamps_ns=set(history_timestamps),
            label=f"scene {scene_id} infrastructure trajectory",
        ),
        "traffic_lights": normalized_history_rows(
            lights,
            scale_to_ns=scale,
            history_timestamps_ns=set(history_timestamps),
            label=f"scene {scene_id} traffic light",
        ),
        "hd_map": map_document,
        "coordinate_frame": protocol["coordinate_frame"],
    }
    for key in (
        "cooperative_trajectories",
        "vehicle_trajectories",
        "infrastructure_trajectories",
        "traffic_lights",
    ):
        if not history[key]:
            raise ContractError(f"scene {scene_id} has no causal {key} rows")
        if any(row["arrival_time_ns"] > decision_time for row in history[key]):
            raise ContractError(
                f"scene {scene_id} violates the inclusive causal cutoff"
            )
    future = {
        "timestamps_ns": future_timestamps,
        "positions": future_positions,
        "coordinate_frame": protocol["coordinate_frame"],
    }
    input_hash = sha256_bytes(canonical_bytes(history))
    return ForecastSample(
        scene_id=scene_id,
        target_id=target_id,
        history=history,
        future=future,
        input_sha256=input_hash,
    )


def validate_backward_report(value: object) -> dict[str, Any]:
    report = require_object(json_safe(value, "backward report"), "backward report")
    if set(report) != {"loss", "gradient_norm", "backward_completed"}:
        raise ContractError(
            "backward report must contain loss, gradient_norm, and backward_completed exactly"
        )
    if report.get("backward_completed") is not True:
        raise ContractError("backward report must confirm backward_completed=true")
    _positive_number(report.get("gradient_norm"), "backward gradient_norm")
    loss = report.get("loss")
    if (
        isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(loss)
    ):
        raise ContractError("backward loss must be finite numeric")
    return report


def validate_optimizer_report(value: object) -> dict[str, Any]:
    report = require_object(json_safe(value, "optimizer report"), "optimizer report")
    if set(report) != {"parameter_update_norm", "optimizer_step_completed"}:
        raise ContractError(
            "optimizer report must contain parameter_update_norm and optimizer_step_completed exactly"
        )
    if report.get("optimizer_step_completed") is not True:
        raise ContractError(
            "optimizer report must confirm optimizer_step_completed=true"
        )
    _positive_number(
        report.get("parameter_update_norm"), "optimizer parameter_update_norm"
    )
    return report


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ContractError(f"{label} must be finite numeric")
    return float(value)


def validate_prediction(
    value: object,
    *,
    sample: ForecastSample,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    prediction = require_object(
        json_safe(value, "forecast prediction"), "forecast prediction"
    )
    if set(prediction) != {"scene_id", "target_id", "modes"}:
        raise ContractError(
            "forecast prediction must contain scene_id, target_id, and modes exactly"
        )
    if (
        prediction["scene_id"] != sample.scene_id
        or str(prediction["target_id"]) != sample.target_id
    ):
        raise ContractError(
            "forecast prediction identity does not match the validation sample"
        )
    modes = prediction["modes"]
    if not isinstance(modes, list) or len(modes) != protocol["modes"]:
        raise ContractError(
            "forecast prediction mode count does not match the frozen protocol"
        )
    mode_ids: set[str] = set()
    probabilities: list[float] = []
    normalized_modes: list[dict[str, Any]] = []
    dimensions = protocol["spatial_dimensions"]
    future_steps = protocol["future_steps"]
    for index, raw_mode in enumerate(modes):
        mode = require_object(raw_mode, f"forecast mode {index}")
        if set(mode) != {"mode_id", "probability", "trajectory"}:
            raise ContractError(
                "each forecast mode must contain mode_id, probability, and trajectory exactly"
            )
        mode_id = require_string(mode["mode_id"], f"forecast mode {index}.mode_id")
        if mode_id in mode_ids:
            raise ContractError("forecast prediction contains duplicate mode_id")
        mode_ids.add(mode_id)
        probability = _finite_number(
            mode["probability"], f"forecast mode {index}.probability"
        )
        if probability < 0 or probability > 1:
            raise ContractError("forecast probabilities must be in [0, 1]")
        probabilities.append(probability)
        trajectory = mode["trajectory"]
        if not isinstance(trajectory, list) or len(trajectory) != future_steps:
            raise ContractError(
                "forecast trajectory length does not match future_steps"
            )
        normalized_trajectory: list[list[float]] = []
        for step, point in enumerate(trajectory):
            if not isinstance(point, list) or len(point) != dimensions:
                raise ContractError(
                    "forecast trajectory dimensionality does not match protocol"
                )
            normalized_trajectory.append(
                [
                    _finite_number(number, f"mode {index} trajectory step {step}")
                    for number in point
                ]
            )
        normalized_modes.append(
            {
                "mode_id": mode_id,
                "probability": probability,
                "trajectory": normalized_trajectory,
            }
        )
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ContractError("forecast mode probabilities must sum to one")
    return {
        "scene_id": sample.scene_id,
        "target_id": sample.target_id,
        "modes": normalized_modes,
    }


def validate_metrics(
    metrics: object, expected_names: Iterable[str]
) -> dict[str, float | int]:
    result = require_object(json_safe(metrics, "forecast metrics"), "forecast metrics")
    if set(result) != set(expected_names):
        raise ContractError("forecast metrics do not match the frozen protocol exactly")
    normalized: dict[str, float | int] = {}
    for name, value in result.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ContractError(f"forecast metric is not finite numeric: {name}")
        normalized[name] = value
    for name in ("minADE", "minFDE"):
        if normalized[name] < 0:
            raise ContractError(
                f"forecast distance metric must be non-negative: {name}"
            )
    for name in ("MR", "ECE", "Brier", "AURC"):
        if normalized[name] < 0 or normalized[name] > 1:
            raise ContractError(
                f"forecast probability/risk metric must be in [0, 1]: {name}"
            )
    return normalized


def detached_json(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} cannot be detached as finite JSON") from exc


def reference_forecast_metrics(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, float]:
    target_index = {str(target["scene_id"]): target for target in targets}
    samples: list[ReferenceForecastSample] = []
    for prediction in predictions:
        scene_id = str(prediction["scene_id"])
        if scene_id not in target_index:
            raise ContractError(
                "reference evaluator cannot resolve prediction ground truth"
            )
        target = target_index[scene_id]
        samples.append(
            ReferenceForecastSample(
                sample_id=scene_id,
                truth=tuple(
                    tuple(step) for step in target["ground_truth"]["positions"]
                ),
                modes=tuple(
                    tuple(tuple(step) for step in mode["trajectory"])
                    for mode in prediction["modes"]
                ),
                mode_weights=tuple(mode["probability"] for mode in prediction["modes"]),
            )
        )
    calibration = protocol["calibration_definition"]
    try:
        result = evaluate_forecasts(
            samples,
            miss_threshold_m=protocol["miss_rate_threshold_m"],
            nll_sigma_m=calibration["nll_sigma_m"],
            ece_bins=calibration["ece_bins"],
        )
    except EvaluationInputError as exc:
        raise ContractError(
            "reference forecast evaluation rejected canary outputs"
        ) from exc
    return {
        "minADE": result.min_ade_m,
        "minFDE": result.min_fde_m,
        "MR": result.miss_rate,
        "NLL": result.mixture_nll,
        "ECE": result.top_mode_ece,
        "Brier": result.top_mode_brier,
        "AURC": result.top_mode_aurc,
    }


def probe_gpu(required_name_substring: str) -> dict[str, Any]:
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ContractError("nvidia-smi GPU preflight failed") from exc
    rows = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if not rows or not any(
        required_name_substring in row.split(",", 1)[0] for row in rows
    ):
        raise ContractError(
            f"no visible GPU matches required substring: {required_name_substring}"
        )
    return {"query": "name,driver_version,memory.total", "rows": rows}


def ensure_read_only_dataset(root: Path) -> None:
    if not root.is_dir():
        raise ContractError("TFD dataset root is not a directory")
    if os.access(root, os.W_OK):
        raise ContractError("TFD dataset root must be mounted read-only for the canary")


def ensure_read_only_contracts_root(root: Path) -> None:
    if not root.is_dir():
        raise ContractError("external contracts root is not a directory")
    if os.access(root, os.W_OK):
        raise ContractError("external contracts root must be mounted read-only")


def validate_scene_universe(
    root: Path, relative_directory: str, expected_hash: str
) -> None:
    directory = safe_dataset_path(root, relative_directory)
    if not directory.is_dir():
        raise ContractError("TFD cooperative trajectory directory is missing")
    scenes = sorted(
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix == ".csv"
    )
    if not scenes or len(scenes) != len(set(scenes)):
        raise ContractError(
            "TFD cooperative trajectory scene universe is empty or ambiguous"
        )
    if sha256_bytes(canonical_bytes(scenes)) != expected_hash:
        raise ContractError(
            "TFD frozen split does not match the mounted scene universe"
        )


def validate_clearml_task_id(value: Any) -> str:
    task_id = require_string(value, "ClearML task ID")
    if len(task_id) != 32 or any(c not in "0123456789abcdef" for c in task_id):
        raise ContractError(
            "ClearML task ID must be a 32-character lowercase hexadecimal ID"
        )
    environment_id = os.environ.get("CLEARML_TASK_ID") or os.environ.get(
        "TRAINS_TASK_ID"
    )
    if environment_id != task_id:
        raise ContractError(
            "ClearML task ID must come from the active ClearML agent environment"
        )
    return task_id


def evidence_hashes(output: Path) -> dict[str, str]:
    return {name: sha256_file(output / name) for name in REQUIRED_ARTIFACTS}


def _config_file_with_hash(
    repository_root: Path,
    path_value: Any,
    hash_value: Any,
    label: str,
) -> Path:
    path = resolve_repository_file(
        repository_root, require_string(path_value, f"{label}.path"), label
    )
    expected = validate_digest(hash_value, f"{label}.sha256")
    if sha256_file(path) != expected:
        raise ContractError(f"{label} SHA-256 does not match the frozen config")
    return path


def read_stable_regular_file(path: Path, label: str) -> bytes:
    """Read one regular non-symlink file and reject replacement during the read."""

    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise ContractError(f"{label} must be a non-symlink regular file")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise ContractError(f"{label} could not be read") from exc
    stable_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
    )
    stable_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    )
    if stable_before != stable_after or not stat.S_ISREG(after.st_mode):
        raise ContractError(f"{label} changed while it was read")
    return payload


def resolve_contract_file(contracts_root: Path, relative: str, label: str) -> Path:
    """Resolve a regular file below an external, non-symlink contracts root."""

    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "." in pure.parts:
        raise ContractError(f"unsafe {label} path: {relative!r}")
    if pure.as_posix() != relative:
        raise ContractError(f"non-canonical {label} path: {relative!r}")
    raw_root = Path(contracts_root)
    if raw_root.is_symlink():
        raise ContractError("contracts root may not be a symlink")
    try:
        root = raw_root.resolve(strict=True)
    except OSError as exc:
        raise ContractError("contracts root is unavailable") from exc
    if not root.is_dir():
        raise ContractError("contracts root must be a directory")
    cursor = root
    for component in pure.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ContractError(f"{label} may not traverse a symlink")
    try:
        metadata = cursor.lstat()
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ContractError(f"{label} is unavailable below the contracts root") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"{label} must be a regular file")
    return resolved


def _resolve_repository_directory(
    repository_root: Path, relative: str, label: str
) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "." in pure.parts:
        raise ContractError(f"unsafe {label} path: {relative!r}")
    if pure.as_posix() != relative:
        raise ContractError(f"non-canonical {label} path: {relative!r}")
    root = repository_root.resolve()
    cursor = root
    for component in pure.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ContractError(f"{label} may not traverse a symlink")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ContractError(
            f"{label} is unavailable below the repository root"
        ) from exc
    if not resolved.is_dir():
        raise ContractError(f"{label} must be a directory")
    return resolved


def _package_tree_paths(repository_root: Path, package_root_relative: str) -> set[str]:
    package_root = _resolve_repository_directory(
        repository_root, package_root_relative, "adapter.package_root"
    )
    paths: set[str] = set()
    for current, directory_names, file_names in os.walk(
        package_root, followlinks=False
    ):
        current_path = Path(current)
        for name in directory_names:
            candidate = current_path / name
            if name == ".git":
                raise ContractError(
                    "adapter package_root may not contain nested Git metadata"
                )
            if candidate.is_symlink():
                raise ContractError(
                    "adapter package_root may not contain directory symlinks"
                )
        for name in file_names:
            candidate = current_path / name
            if name == ".git":
                raise ContractError(
                    "adapter package_root may not contain nested Git metadata"
                )
            try:
                metadata = candidate.lstat()
            except OSError as exc:
                raise ContractError("adapter package file became unavailable") from exc
            if not stat.S_ISREG(metadata.st_mode) or candidate.is_symlink():
                raise ContractError(
                    "adapter package_root may contain regular files only"
                )
            paths.add(candidate.relative_to(repository_root.resolve()).as_posix())
    if not paths:
        raise ContractError(
            "adapter package_root must contain at least one regular file"
        )
    return paths


def _load_external_seal(
    contracts_root: Path,
    reference_value: Any,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    reference = require_object(reference_value, label)
    if set(reference) != {"manifest_path", "manifest_file_sha256"}:
        raise ContractError(
            f"{label} must contain manifest_path and manifest_file_sha256 exactly"
        )
    relative = require_string(reference.get("manifest_path"), f"{label}.manifest_path")
    expected = validate_digest(
        reference.get("manifest_file_sha256"), f"{label}.manifest_file_sha256"
    )
    path = resolve_contract_file(contracts_root, relative, label)
    payload = read_stable_regular_file(path, label)
    observed = sha256_bytes(payload)
    if observed != expected:
        raise ContractError(f"{label} file SHA-256 does not match the resolved config")
    try:
        document = require_object(loads_json_strict(payload, label=label), label)
    except SealError as exc:
        raise ContractError(f"{label} is not a valid strict JSON seal: {exc}") from exc
    return path, document, observed


def _seal_entry_paths(document: Mapping[str, Any]) -> set[str]:
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ContractError("Git seal entries must be an array")
    return {
        require_string(require_object(entry, "Git seal entry").get("path"), "path")
        for entry in entries
    }


def enforce_production_execution_gate(
    model: Mapping[str, Any],
    adapter: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
    contracts_root: Path | None = None,
    resolved_config_path: Path | None = None,
    resolved_config_sha256: str | None = None,
    template_config: Mapping[str, Any] | None = None,
    required_source_paths: Iterable[str] = REQUIRED_SOURCE_SEAL_PATHS,
    required_package_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Verify external seals and the causal subprocess contract before execution."""

    if (
        adapter.get("execution_enabled") is not True
        or adapter.get("execution_status") != "enabled"
    ):
        raise ContractError(
            "adapter execution must set execution_enabled=true and status=enabled"
        )
    isolation = require_object(adapter.get("isolation"), "adapter.isolation")
    if isolation.get("mode") != "history_only_subprocess":
        raise ContractError("adapter isolation.mode must be history_only_subprocess")
    if (
        repository_root is None
        or contracts_root is None
        or resolved_config_path is None
        or resolved_config_sha256 is None
    ):
        raise ContractError(
            "enabled adapter execution requires an external resolved config root"
        )
    repository_root = repository_root.resolve()
    if contracts_root.is_symlink():
        raise ContractError("contracts root may not be a symlink")
    ensure_read_only_contracts_root(contracts_root)
    contracts_root = contracts_root.resolve()
    resolved_config_path = resolved_config_path.resolve()
    try:
        resolved_config_path.relative_to(contracts_root)
    except ValueError as exc:
        raise ContractError(
            "enabled config must be below the external contracts root"
        ) from exc
    expected_config_sha = validate_digest(
        resolved_config_sha256, "resolved config SHA-256"
    )
    if sha256_file(resolved_config_path) != expected_config_sha:
        raise ContractError("resolved config changed after it was parsed")
    commit, dirty = git_identity(repository_root)
    if dirty:
        raise ContractError(
            "enabled adapter execution requires a clean Git working tree"
        )

    _, source_document, source_file_sha = _load_external_seal(
        contracts_root, model.get("source_tree_seal"), "model.source_tree_seal"
    )
    _, package_document, package_file_sha = _load_external_seal(
        contracts_root, adapter.get("package_seal"), "adapter.package_seal"
    )
    _, environment_document, environment_file_sha = _load_external_seal(
        contracts_root,
        model.get("installed_environment_seal"),
        "model.installed_environment_seal",
    )
    try:
        source_validated = validate_git_seal(
            repository_root, source_document, expected_kind="source_tree"
        )
        package_validated = validate_git_seal(
            repository_root, package_document, expected_kind="package"
        )
        environment_validated = validate_installed_environment_manifest(
            environment_document, verify_current=True
        )
    except SealError as exc:
        raise ContractError(f"execution provenance seal rejected: {exc}") from exc
    if (
        source_validated["git_commit"] != commit
        or package_validated["git_commit"] != commit
    ):
        raise ContractError("source/package seals must bind the current Git HEAD")

    source_paths = _seal_entry_paths(source_validated)
    missing_sources = sorted(set(required_source_paths) - source_paths)
    if missing_sources:
        raise ContractError(f"source-tree seal omits required paths: {missing_sources}")

    package_root_relative = require_string(
        adapter.get("package_root"), "adapter.package_root"
    )
    observed_package_paths = _package_tree_paths(repository_root, package_root_relative)
    package_paths = _seal_entry_paths(package_validated)
    if package_paths != observed_package_paths:
        raise ContractError(
            "package seal must cover the complete adapter.package_root file set"
        )
    missing_package_paths = sorted(set(required_package_paths) - package_paths)
    if missing_package_paths:
        raise ContractError(
            f"package seal omits required runtime paths: {missing_package_paths}"
        )

    template = require_object(template_config, "template_config")
    if set(template) != {"path", "sha256"}:
        raise ContractError("template_config must contain path and sha256 exactly")
    template_relative = require_string(template.get("path"), "template_config.path")
    template_path = resolve_repository_file(
        repository_root, template_relative, "TFD pending template config"
    )
    template_sha = validate_digest(template.get("sha256"), "template_config.sha256")
    if sha256_file(template_path) != template_sha:
        raise ContractError("pending template config SHA-256 mismatch")
    if template_relative not in source_paths:
        raise ContractError("source-tree seal must include the pending template config")

    if set(isolation) != {"mode", "manifest_path", "manifest_file_sha256"}:
        raise ContractError(
            "adapter.isolation must contain mode, manifest_path, and manifest_file_sha256 exactly"
        )
    isolation_relative = require_string(
        isolation.get("manifest_path"), "adapter.isolation.manifest_path"
    )
    isolation_path = resolve_repository_file(
        repository_root, isolation_relative, "history-only subprocess contract"
    )
    isolation_sha = validate_digest(
        isolation.get("manifest_file_sha256"),
        "adapter.isolation.manifest_file_sha256",
    )
    if sha256_file(isolation_path) != isolation_sha:
        raise ContractError("history-only subprocess contract SHA-256 mismatch")
    if isolation_relative not in source_paths:
        raise ContractError("source-tree seal must include the isolation contract")
    try:
        validate_isolation_contract(isolation_path, isolation_sha)
    except HistoryAdapterError as exc:
        raise ContractError(
            f"history-only subprocess contract rejected: {exc}"
        ) from exc

    return {
        "resolved_config_sha256": expected_config_sha,
        "template_config_sha256": template_sha,
        "source_tree": {
            "file_sha256": source_file_sha,
            "manifest_sha256": source_validated["manifest_sha256"],
            "git_commit": source_validated["git_commit"],
        },
        "adapter_package": {
            "file_sha256": package_file_sha,
            "manifest_sha256": package_validated["manifest_sha256"],
            "git_commit": package_validated["git_commit"],
        },
        "installed_environment": {
            "file_sha256": environment_file_sha,
            "manifest_sha256": environment_validated["manifest_sha256"],
        },
        "isolation_contract_sha256": isolation_sha,
    }


def run_canary(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = Path(args.repository_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    ensure_read_only_dataset(dataset_root)
    contracts_root_value = getattr(args, "contracts_root", None)
    contracts_root = (
        Path(contracts_root_value) if contracts_root_value is not None else None
    )
    if contracts_root is None:
        config_path = resolve_repository_file(
            repository_root, args.config, "TFD canary config"
        )
    else:
        config_path = resolve_contract_file(
            contracts_root, args.config, "resolved TFD canary config"
        )
    config_payload = read_stable_regular_file(config_path, "TFD canary config")
    config_sha256 = sha256_bytes(config_payload)
    try:
        config = require_object(
            loads_json_strict(config_payload, label="TFD canary config"),
            "TFD canary config",
        )
    except SealError as exc:
        raise ContractError(f"TFD canary config is not strict JSON: {exc}") from exc
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("TFD canary config schema_version must be 1")
    if config.get("canary_type") != CANARY_TYPE:
        raise ContractError(f"TFD canary config canary_type must be {CANARY_TYPE}")
    if config.get("scientific_claim_allowed") is not False:
        raise ContractError("TFD canary must set scientific_claim_allowed=false")
    if config.get("stage") != "canary":
        raise ContractError("TFD diagnostic runner accepts stage=canary only")

    clearml = require_object(config.get("clearml"), "config.clearml")
    require_string(clearml.get("project"), "config.clearml.project")
    require_string(clearml.get("queue"), "config.clearml.queue")
    if clearml.get("require_task_id") is not True:
        raise ContractError("TFD canary config must require a ClearML task ID")
    clearml_task_id = validate_clearml_task_id(args.clearml_task_id)

    data = require_object(config.get("data"), "config.data")
    if data.get("dataset") != DATASET_NAME:
        raise ContractError(f"config.data.dataset must be {DATASET_NAME}")
    if data.get("read_only_mount_required") is not True:
        raise ContractError("config.data must require a read-only dataset mount")
    protocol_path = resolve_repository_file(
        repository_root,
        require_string(data.get("protocol_path"), "config.data.protocol_path"),
        "TFD protocol",
    )
    protocol = validate_protocol(load_json(protocol_path))
    reference_evaluator_path, reference_evaluator = (
        validate_reference_evaluator_contract(repository_root, protocol)
    )
    protocol_id = protocol["protocol_id"]
    if config.get("protocol") != protocol_id:
        raise ContractError(
            "TFD canary config protocol does not match the protocol document"
        )

    identity_path = Path(
        require_string(data.get("identity_manifest"), "identity_manifest")
    ).resolve()
    split_path = Path(
        require_string(data.get("frozen_split"), "frozen_split")
    ).resolve()
    identity_hash = validate_digest(
        data.get("identity_manifest_sha256"), "identity_manifest_sha256"
    )
    split_hash = validate_digest(data.get("frozen_split_sha256"), "frozen_split_sha256")
    if sha256_file(identity_path) != identity_hash:
        raise ContractError(
            "TFD identity manifest SHA-256 does not match the frozen config"
        )
    if sha256_file(split_path) != split_hash:
        raise ContractError("TFD split SHA-256 does not match the frozen config")
    identity = require_object(load_json(identity_path), "TFD identity")
    identity_index = validate_identity(dataset_root, identity)
    split_document = require_object(load_json(split_path), "TFD frozen split")
    partitions = validate_split(split_document, protocol_id=protocol_id)
    cooperative_directory = require_string(
        data.get("cooperative_directory"), "config.data.cooperative_directory"
    )
    validate_scene_universe(
        dataset_root,
        cooperative_directory,
        split_document["scene_universe_sha256"],
    )

    training_scene = require_string(data.get("training_scene_id"), "training_scene_id")
    evaluation_scenes = data.get("evaluation_scene_ids")
    if (
        not isinstance(evaluation_scenes, list)
        or not evaluation_scenes
        or any(not isinstance(scene, str) or not scene for scene in evaluation_scenes)
        or evaluation_scenes != sorted(set(evaluation_scenes))
    ):
        raise ContractError(
            "evaluation_scene_ids must be sorted, unique, and non-empty"
        )
    if training_scene not in partitions["train"]:
        raise ContractError(
            "TFD training scene is absent from the frozen train partition"
        )
    if any(scene not in partitions["validation"] for scene in evaluation_scenes):
        raise ContractError(
            "TFD evaluation scene is absent from the frozen validation partition"
        )
    if training_scene in evaluation_scenes:
        raise ContractError(
            "TFD training and validation canary scenes must be disjoint"
        )
    sealed_scenes = set(identity.get("scene_ids", []))
    if not {training_scene, *evaluation_scenes}.issubset(sealed_scenes):
        raise ContractError("TFD identity does not seal every configured canary scene")

    training_sample = load_forecast_sample(
        dataset_root, identity_index, scene_id=training_scene, protocol=protocol
    )
    evaluation_samples = [
        load_forecast_sample(
            dataset_root, identity_index, scene_id=scene, protocol=protocol
        )
        for scene in evaluation_scenes
    ]

    commit, dirty = git_identity(repository_root)
    if dirty and not args.allow_dirty_diagnostic:
        raise ContractError(
            "repository is dirty; commit the exact TFD canary source before running"
        )

    model = require_object(config.get("model"), "config.model")
    if model.get("implementation_status") != "ready":
        raise ContractError("TFD canary model implementation_status must be ready")
    require_string(model.get("name"), "config.model.name")
    if model.get("initialization") != "from_scratch":
        raise ContractError(
            "TFD canary model initialization must be frozen to from_scratch"
        )
    if (
        model.get("checkpoint_path") is not None
        or model.get("checkpoint_sha256") is not None
    ):
        raise ContractError(
            "from_scratch TFD canary cannot declare an input checkpoint"
        )
    upstream_manifest_path = resolve_repository_file(
        repository_root,
        require_string(model.get("upstream_manifest_path"), "upstream_manifest_path"),
        "upstream source manifest",
    )
    upstream_id = require_string(model.get("upstream_id"), "model.upstream_id")
    upstream_revision = require_string(
        model.get("upstream_revision"), "model.upstream_revision"
    )
    if len(upstream_revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in upstream_revision
    ):
        raise ContractError("model.upstream_revision must be a full Git object ID")
    upstream = validate_upstream(
        load_json(upstream_manifest_path),
        source_id=upstream_id,
        expected_revision=upstream_revision,
    )
    dependency_lock_relative = require_string(
        model.get("dependency_lock_path"), "model.dependency_lock_path"
    )
    dependency_lock = _config_file_with_hash(
        repository_root,
        dependency_lock_relative,
        model.get("dependency_lock_sha256"),
        "dependency lock",
    )

    adapter_config = require_object(config.get("adapter"), "config.adapter")
    adapter_relative = require_string(adapter_config.get("path"), "adapter.path")
    adapter_path = resolve_repository_file(
        repository_root,
        adapter_relative,
        "TFD model adapter",
    )
    adapter_hash = validate_digest(adapter_config.get("sha256"), "adapter.sha256")
    if sha256_file(adapter_path) != adapter_hash:
        raise ContractError(
            "TFD model adapter SHA-256 does not match the frozen config"
        )
    factory = require_string(adapter_config.get("factory"), "adapter.factory")
    if getattr(args, "allow_dirty_diagnostic", False):
        raise ContractError(
            "enabled TFD subprocess execution does not allow dirty diagnostics"
        )
    execution_provenance = enforce_production_execution_gate(
        model,
        adapter_config,
        repository_root=repository_root,
        contracts_root=contracts_root,
        resolved_config_path=config_path,
        resolved_config_sha256=config_sha256,
        template_config=config.get("template_config"),
        required_package_paths={adapter_relative, dependency_lock_relative},
    )

    runtime = require_object(config.get("runtime"), "config.runtime")
    if runtime.get("epochs") != 1 or runtime.get("batch_size") != 1:
        raise ContractError("TFD canary must freeze epochs=1 and batch_size=1")
    required_gpu = require_string(
        runtime.get("require_gpu_name_substring"), "require_gpu_name_substring"
    )
    gpu_environment = probe_gpu(required_gpu)
    seed = _positive_int(config.get("seed"), "config.seed")
    evidence = require_object(config.get("evidence"), "config.evidence")
    required_evidence_flags = {
        "save_multimodal_predictions",
        "save_metrics",
        "save_checkpoint",
        "sha256_manifest",
    }
    if any(evidence.get(key) is not True for key in required_evidence_flags):
        raise ContractError("TFD canary evidence flags must all be true")
    if evidence.get("registry_write_allowed") is not False:
        raise ContractError("TFD canary evidence must set registry_write_allowed=false")
    context = {
        "protocol": protocol,
        "model": {
            "name": model["name"],
            "upstream_id": upstream_id,
            "upstream_repository": upstream["repository"],
            "upstream_revision": upstream_revision,
        },
        "seed": seed,
        "reference_evaluator": {
            "contract_id": reference_evaluator["contract_id"],
            "official_evaluator_equivalence": False,
        },
        "claim_scope": "diagnostic_only",
        "scientific_claim_allowed": False,
    }
    output = Path(args.output).resolve()
    if output.exists():
        if not output.is_dir():
            raise ContractError("TFD canary output must be a directory path")
        if any(output.iterdir()):
            raise ContractError(
                "TFD canary output directory must not already contain files"
            )
    output.mkdir(parents=True, exist_ok=True)
    isolation = require_object(adapter_config.get("isolation"), "adapter.isolation")
    isolation_path = resolve_repository_file(
        repository_root,
        require_string(
            isolation.get("manifest_path"), "adapter.isolation.manifest_path"
        ),
        "history-only subprocess contract",
    )
    isolation_sha = validate_digest(
        isolation.get("manifest_file_sha256"),
        "adapter.isolation.manifest_file_sha256",
    )
    timeout_seconds = runtime.get("subprocess_timeout_seconds", 30.0)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise ContractError(
            "runtime.subprocess_timeout_seconds must be finite and positive"
        )

    events: list[str] = ["preflight_complete"]
    try:
        session = HistoryOnlyAdapterProcess(
            adapter_path=adapter_path,
            factory=factory,
            context=detached_json(context, "adapter context"),
            expected_inference_count=len(evaluation_samples),
            timeout_seconds=float(timeout_seconds),
            isolation_contract_path=isolation_path,
            expected_isolation_contract_sha256=isolation_sha,
        )
    except HistoryAdapterError as exc:
        raise ContractError(
            f"cannot construct history-only adapter process: {exc}"
        ) from exc

    try:
        session.start()
        runtime_report = require_object(
            json_safe(session.runtime_environment(), "subprocess runtime environment"),
            "subprocess runtime environment",
        )
        adapter_environment = require_object(
            runtime_report.get("adapter"), "adapter runtime environment"
        )
        worker_environment = require_object(
            runtime_report.get("worker"), "worker runtime environment"
        )
        if set(adapter_environment) != {
            "framework",
            "framework_version",
            "device_name",
        }:
            raise ContractError(
                "adapter runtime environment must contain framework, framework_version, and device_name exactly"
            )
        if required_gpu not in require_string(
            adapter_environment.get("device_name"),
            "adapter runtime environment device_name",
        ):
            raise ContractError(
                "adapter runtime environment does not identify the required GPU"
            )
        require_string(
            adapter_environment.get("framework"),
            "adapter runtime environment framework",
        )
        require_string(
            adapter_environment.get("framework_version"),
            "adapter runtime environment framework_version",
        )

        training_record = session.train(training_sample.training_payload())
        events.append("training_forward_complete")
        backward_report = validate_backward_report(training_record.backward_report)
        events.append("backward_complete")
        optimizer_report = validate_optimizer_report(training_record.optimizer_report)
        events.append("optimizer_step_complete")

        predictions: list[dict[str, Any]] = []
        normalized_predictions: list[dict[str, Any]] = []
        prediction_hashes: dict[str, dict[str, str]] = {}
        for sample in evaluation_samples:
            record = session.infer(sample.inference_payload())
            prediction = validate_prediction(
                record.prediction, sample=sample, protocol=protocol
            )
            normalized_prediction_sha = sha256_bytes(canonical_bytes(prediction))
            normalized_predictions.append(prediction)
            prediction_hashes[sample.scene_id] = {
                "worker_payload_sha256": record.prediction_sha256,
                "persisted_prediction_sha256": normalized_prediction_sha,
            }
            predictions.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scene_id": sample.scene_id,
                    "target_id": sample.target_id,
                    "input_sha256": sample.input_sha256,
                    "prediction_sha256": normalized_prediction_sha,
                    "worker_prediction_sha256": record.prediction_sha256,
                    "prediction": prediction,
                }
            )
        events.append("evaluation_forward_complete")
        persisted_prediction_set_sha = sha256_bytes(
            canonical_bytes(normalized_predictions)
        )

        evaluation_targets = [
            sample.evaluation_target() for sample in evaluation_samples
        ]
        metric_values = validate_metrics(
            reference_forecast_metrics(
                normalized_predictions, evaluation_targets, protocol
            ),
            protocol["metrics"],
        )
        evaluation_record = session.evaluate(evaluation_targets)
        adapter_metric_values = validate_metrics(
            evaluation_record.metrics, protocol["metrics"]
        )
        tolerance = reference_evaluator["numeric_policy"][
            "conformance_absolute_tolerance"
        ]
        if any(
            not math.isclose(
                float(adapter_metric_values[name]),
                float(metric_values[name]),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            for name in protocol["metrics"]
        ):
            raise ContractError(
                "adapter evaluation does not match the frozen reference evaluator"
            )
        events.append("evaluation_complete")

        checkpoint_record = session.save_checkpoint(output / "checkpoint.bin")
        events.append("checkpoint_complete")
        session.close()
        events.append("subprocess_closed")
    except HistoryAdapterError as exc:
        session.abort()
        raise ContractError(
            f"history-only adapter process rejected execution: {exc}"
        ) from exc
    except BaseException:
        session.abort()
        raise

    post_execution_provenance = enforce_production_execution_gate(
        model,
        adapter_config,
        repository_root=repository_root,
        contracts_root=contracts_root,
        resolved_config_path=config_path,
        resolved_config_sha256=config_sha256,
        template_config=config.get("template_config"),
        required_package_paths={adapter_relative, dependency_lock_relative},
    )
    if canonical_bytes(post_execution_provenance) != canonical_bytes(
        execution_provenance
    ):
        raise ContractError(
            "execution provenance changed while the subprocess was running"
        )
    transcript_records = list(session.transcript_records)
    transcript_sha = session.transcript_sha256
    stderr_summary = session.stderr_summary

    atomic_write(
        output / "predictions.jsonl",
        b"".join(canonical_bytes(row) for row in predictions),
    )
    metrics_document = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "claim_scope": "diagnostic_only",
        "scientific_claim_allowed": False,
        "training_scene_id": training_scene,
        "evaluation_scene_ids": evaluation_scenes,
        "sample_count": len(evaluation_samples),
        "backward_report": backward_report,
        "optimizer_report": optimizer_report,
        "adapter_metrics": adapter_metric_values,
        "metrics": metric_values,
        "prediction_set_sha256": evaluation_record.prediction_set_sha256,
        "persisted_prediction_set_sha256": persisted_prediction_set_sha,
        "target_set_sha256": evaluation_record.target_set_sha256,
    }
    atomic_write(output / "metrics.json", canonical_bytes(metrics_document))
    environment = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "diagnostic_only",
        "scientific_claim_allowed": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "adapter_sha256": adapter_hash,
        "dependency_lock_sha256": sha256_file(dependency_lock),
        "upstream_id": upstream_id,
        "upstream_repository": upstream["repository"],
        "upstream_revision": upstream_revision,
        "gpu": gpu_environment,
        "adapter": adapter_environment,
        "worker": worker_environment,
        "history_only_subprocess": {
            "parent_source_sha256": session.parent_source_sha256,
            "worker_source_sha256": session.worker_sha256,
            "python_executable_basename": session.python_executable.name,
            "python_executable_sha256": session.python_executable_sha256,
            "isolation_contract_sha256": session.isolation_contract_sha256,
            "context_sha256": session.context_sha256,
        },
    }
    atomic_write(output / "environment.json", canonical_bytes(environment))
    atomic_write(output / "stdout.log", ("\n".join(events) + "\n").encode("utf-8"))
    atomic_write(output / "stderr.log", canonical_bytes(stderr_summary))

    unexpected = sorted(
        path.name for path in output.iterdir() if path.name not in REQUIRED_ARTIFACTS
    )
    if unexpected:
        raise ContractError("TFD adapter created unexpected output artifacts")

    artifact_hashes = evidence_hashes(output)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "diagnostic_only",
        "scientific_claim_allowed": False,
        "clearml_task_id": clearml_task_id,
        "protocol_id": protocol_id,
        "dataset": DATASET_NAME,
        "dataset_release_id": identity["release_id"],
        "dataset_identity_sha256": identity_hash,
        "frozen_split_sha256": split_hash,
        "protocol_sha256": sha256_file(protocol_path),
        "config_sha256": config_sha256,
        "template_config_sha256": execution_provenance["template_config_sha256"],
        "upstream_manifest_sha256": sha256_file(upstream_manifest_path),
        "reference_evaluator_contract_id": reference_evaluator["contract_id"],
        "reference_evaluator_contract_sha256": sha256_file(reference_evaluator_path),
        "upstream_id": upstream_id,
        "upstream_revision": upstream_revision,
        "dependency_lock_sha256": sha256_file(dependency_lock),
        "adapter_sha256": adapter_hash,
        "code_commit": commit,
        "dirty_diagnostic": dirty,
        "registry_write_allowed": False,
        "training_scene_id": training_scene,
        "evaluation_scene_ids": evaluation_scenes,
        "training_input_sha256": training_sample.input_sha256,
        "training_target_sha256": sha256_bytes(
            canonical_bytes(training_sample.training_payload()["ground_truth"])
        ),
        "evaluation_input_sha256": {
            sample.scene_id: sample.input_sha256 for sample in evaluation_samples
        },
        "evaluation_target_sha256": {
            sample.scene_id: sha256_bytes(canonical_bytes(sample.evaluation_target()))
            for sample in evaluation_samples
        },
        "prediction_sha256": prediction_hashes,
        "prediction_set_sha256": evaluation_record.prediction_set_sha256,
        "persisted_prediction_set_sha256": persisted_prediction_set_sha,
        "target_set_sha256": evaluation_record.target_set_sha256,
        "checkpoint_sha256": checkpoint_record.sha256,
        "checkpoint_size_bytes": checkpoint_record.size_bytes,
        "training_response_sha256": training_record.response_payload_sha256,
        "evaluation_response_sha256": evaluation_record.response_payload_sha256,
        "checkpoint_response_sha256": checkpoint_record.response_payload_sha256,
        "transcript_sha256": transcript_sha,
        "transcript_records": transcript_records,
        "stderr_summary": stderr_summary,
        "execution_provenance": execution_provenance,
        "artifacts": artifact_hashes,
    }
    atomic_write(output / "run_manifest.json", canonical_bytes(manifest))
    sums = "".join(
        f"{sha256_file(output / name)}  {name}\n"
        for name in sorted((*REQUIRED_ARTIFACTS, "run_manifest.json"))
    )
    atomic_write(output / "SHA256SUMS", sums.encode("ascii"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root", default=str(Path(__file__).resolve().parents[2])
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--config",
        required=True,
        help="repository-relative pending config, or contracts-root-relative resolved config",
    )
    parser.add_argument(
        "--contracts-root",
        help="external read-only root containing an enabled resolved config and provenance seals",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--clearml-task-id",
        default=os.environ.get("CLEARML_TASK_ID") or os.environ.get("TRAINS_TASK_ID"),
    )
    parser.add_argument(
        "--allow-dirty-diagnostic",
        action="store_true",
        help="diagnostic only; dirty state remains ineligible for native result evidence",
    )
    args = parser.parse_args()
    try:
        manifest = run_canary(args)
    except ContractError as exc:
        raise SystemExit(f"real V2X-Seq-TFD canary rejected: {exc}") from exc
    print(
        json.dumps(
            {
                "claim_scope": manifest["claim_scope"],
                "evaluation_sample_count": len(manifest["evaluation_scene_ids"]),
                "protocol_id": manifest["protocol_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
