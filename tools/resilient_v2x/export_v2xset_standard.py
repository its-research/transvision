#!/usr/bin/env python3
"""Export an auditable V2XSet-Standard manifest from a loader trace.

This module deliberately does not parse a V2XSet archive or run a detector.
The repository does not contain the source V2XSet multi-agent dataloader, so
the only supported input is a canonical trace emitted by that actual loader.
Unknown protocol fields and incomplete traces are rejected instead of inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.dataset.resilient_v2x_manifest import (  # noqa: E402
    canonical_json_bytes,
    content_sha256,
)


TRACE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
TRACE_TYPE = "v2xset_standard_dataloader_trace"
MANIFEST_TYPE = "v2xset_standard_manifest"
DATASET_NAME = "V2XSet"
PROTOCOL_SCOPE = "v2xset_standard_multi_agent_lidar_only"
COMMUNICATION_RANGE_M = 70.0
SOURCE_ALIGNED_LATENCIES_MS = frozenset({0, 200, 300})
SUPPLEMENTAL_LATENCIES_MS = frozenset({100})
IDENTIFIER = re.compile(r"-?[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
UNRESOLVED = frozenset(
    {
        "",
        "n/a",
        "na",
        "none",
        "nr",
        "pending",
        "tbd",
        "todo",
        "unknown",
        "unspecified",
        "【填写】",
    }
)
TRUNCATION_RULE = "prefix_after_ordering_to_max_cav"
METHOD_ELIGIBILITY = {
    "method_id": "resilient_v2x.one_ego_one_rsu",
    "status": "N/A",
    "multi_agent_adapter_implemented": False,
    "reason_code": "standard_multi_agent_adapter_not_implemented",
}
MEASUREMENT_BOUNDARY = {
    "contains_model_outputs": False,
    "contains_evaluator_outputs": False,
    "controlled_result_id": None,
}


class V2XSetStandardError(ValueError):
    """Raised when a Standard trace or manifest is incomplete or inconsistent."""


_TRACE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "dataset_name",
        "split",
        "dataset_identity",
        "loader_identity",
        "selection_policy",
        "delay_policy",
        "records",
        "content_sha256",
    }
)
_DATASET_IDENTITY_FIELDS = frozenset(
    {
        "release_id",
        "release_inventory_sha256",
        "split_id",
        "split_sha256",
    }
)
_LOADER_IDENTITY_FIELDS = frozenset(
    {
        "implementation",
        "repository_commit",
        "config_path",
        "config_sha256",
        "trace_hook",
    }
)
_SELECTION_POLICY_FIELDS = frozenset(
    {
        "communication_range_m",
        "max_cav",
        "agent_ordering_rule",
        "agent_ordering_source",
        "truncation_rule",
        "truncation_source",
        "ego_selection_rule",
        "ego_selection_source",
    }
)
_DELAY_POLICY_FIELDS = frozenset(
    {
        "nominal_latency_ms",
        "condition_kind",
        "unit",
        "generation_rule",
        "causal_frame_mapping_rule",
        "uniform_across_remote_agents",
        "source_location",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "sample_id",
        "scene_id",
        "target_timestamp_us",
        "epoch",
        "draw_index",
        "ego_selection",
        "available_agents",
        "ordered_in_range_agent_ids",
        "admitted_agents",
        "truncated_agent_ids",
    }
)
_EGO_SELECTION_FIELDS = frozenset(
    {
        "mode",
        "candidate_vehicle_agent_ids",
        "selected_index",
        "rng_seed",
        "rng_state_before_sha256",
    }
)
_AVAILABLE_AGENT_FIELDS = frozenset({"agent_id", "agent_type", "distance_to_ego_m"})
_ADMITTED_AGENT_FIELDS = frozenset(
    {
        "agent_id",
        "agent_type",
        "distance_to_ego_m",
        "lidar",
        "delay",
    }
)
_PAYLOAD_FIELDS = frozenset({"relative_path", "size_bytes", "sha256"})
_DELAY_FIELDS = frozenset(
    {
        "generated_delay_ms",
        "source_timestamp_us",
        "arrival_timestamp_us",
        "causal_frame_offset",
    }
)
_INPUT_TRACE_FIELDS = frozenset({"artifact_type", "file_sha256", "content_sha256"})
_METHOD_FIELDS = frozenset(METHOD_ELIGIBILITY)
_MEASUREMENT_FIELDS = frozenset(MEASUREMENT_BOUNDARY)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "protocol_scope",
        "dataset_name",
        "split",
        "modalities",
        "input_trace",
        "dataset_identity",
        "loader_identity",
        "selection_policy",
        "delay_policy",
        "method_eligibility",
        "measurement_boundary",
        "record_count",
        "scene_count",
        "epoch_ids",
        "records",
        "content_sha256",
    }
)


def _expect_fields(
    value: object,
    expected: frozenset[str],
    context: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise V2XSetStandardError(f"{context} must be an object")
    if not all(type(key) is str for key in value):
        raise V2XSetStandardError(f"{context} keys must be strings")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise V2XSetStandardError(
            f"{context} fields mismatch: missing={missing}, extra={extra}"
        )
    return dict(value)


def _exact_int(
    value: object,
    context: str,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise V2XSetStandardError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise V2XSetStandardError(f"{context} must be >= {minimum}")
    return value


def _exact_bool(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise V2XSetStandardError(f"{context} must be a boolean")
    return value


def _finite_float(
    value: object,
    context: str,
    minimum: float | None = None,
) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise V2XSetStandardError(f"{context} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise V2XSetStandardError(f"{context} must be >= {minimum}")
    return result


def _identifier(value: object, context: str) -> str:
    if type(value) is not str or IDENTIFIER.fullmatch(value) is None:
        raise V2XSetStandardError(f"{context} is not a canonical identifier")
    return value


def _resolved_string(value: object, context: str) -> str:
    if type(value) is not str:
        raise V2XSetStandardError(f"{context} must be a resolved string")
    result = value.strip()
    unresolved_words = re.findall(r"[a-z]+", result.casefold())
    if (
        result != value
        or result.casefold() in UNRESOLVED
        or any(
            word in {"pending", "tbd", "todo", "unknown", "unspecified"}
            for word in unresolved_words
        )
        or any(marker in result for marker in ("【", "】", "<", ">"))
    ):
        raise V2XSetStandardError(f"{context} must be a resolved string")
    return result


def _sha256(value: object, context: str) -> str:
    if type(value) is not str or SHA256.fullmatch(value) is None:
        raise V2XSetStandardError(
            f"{context} must be 64 lowercase hexadecimal characters"
        )
    return value


def _git_oid(value: object, context: str) -> str:
    if type(value) is not str or GIT_OID.fullmatch(value) is None:
        raise V2XSetStandardError(
            f"{context} must be a 40- or 64-character lowercase git object ID"
        )
    return value


def _relative_path(value: object, context: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise V2XSetStandardError(f"{context} must be a canonical relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise V2XSetStandardError(f"{context} must be a canonical relative path")
    return value


def _string_list(value: object, context: str) -> list[str]:
    if not isinstance(value, list):
        raise V2XSetStandardError(f"{context} must be an array")
    result = [
        _identifier(item, f"{context}[{index}]") for index, item in enumerate(value)
    ]
    if len(set(result)) != len(result):
        raise V2XSetStandardError(f"{context} must contain unique identifiers")
    return result


def _normalize_payload(value: object, context: str) -> dict[str, object]:
    payload = _expect_fields(value, _PAYLOAD_FIELDS, context)
    return {
        "relative_path": _relative_path(
            payload["relative_path"],
            f"{context}.relative_path",
        ),
        "size_bytes": _exact_int(
            payload["size_bytes"],
            f"{context}.size_bytes",
            0,
        ),
        "sha256": _sha256(payload["sha256"], f"{context}.sha256"),
    }


def _verify_payload(
    payload: Mapping[str, object],
    data_root: Path,
    cache: dict[tuple[str, int, str], None],
) -> None:
    relative = payload["relative_path"]
    size = payload["size_bytes"]
    digest = payload["sha256"]
    assert (
        isinstance(relative, str) and isinstance(size, int) and isinstance(digest, str)
    )
    key = (relative, size, digest)
    if key in cache:
        return
    candidate = data_root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(data_root)
    except (OSError, ValueError) as error:
        raise V2XSetStandardError(
            f"LiDAR payload is missing or outside data_root: {relative}"
        ) from error
    try:
        descriptor = os.open(
            resolved,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as error:
        raise V2XSetStandardError(
            f"LiDAR payload is missing or non-regular: {relative}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise V2XSetStandardError(
                f"LiDAR payload is not a regular file: {relative}"
            )
        if before.st_size != size:
            raise V2XSetStandardError(f"LiDAR payload size mismatch: {relative}")
        hasher = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            hasher.update(block)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity:
            raise V2XSetStandardError(
                f"LiDAR payload changed during verification: {relative}"
            )
        if hasher.hexdigest() != digest:
            raise V2XSetStandardError(f"LiDAR payload sha256 mismatch: {relative}")
    finally:
        os.close(descriptor)
    cache[key] = None


def _normalize_dataset_identity(value: object) -> dict[str, object]:
    identity = _expect_fields(
        value,
        _DATASET_IDENTITY_FIELDS,
        "dataset_identity",
    )
    return {
        "release_id": _resolved_string(
            identity["release_id"],
            "dataset_identity.release_id",
        ),
        "release_inventory_sha256": _sha256(
            identity["release_inventory_sha256"],
            "dataset_identity.release_inventory_sha256",
        ),
        "split_id": _resolved_string(
            identity["split_id"],
            "dataset_identity.split_id",
        ),
        "split_sha256": _sha256(
            identity["split_sha256"],
            "dataset_identity.split_sha256",
        ),
    }


def _normalize_loader_identity(value: object) -> dict[str, object]:
    identity = _expect_fields(
        value,
        _LOADER_IDENTITY_FIELDS,
        "loader_identity",
    )
    return {
        "implementation": _resolved_string(
            identity["implementation"],
            "loader_identity.implementation",
        ),
        "repository_commit": _git_oid(
            identity["repository_commit"],
            "loader_identity.repository_commit",
        ),
        "config_path": _relative_path(
            identity["config_path"],
            "loader_identity.config_path",
        ),
        "config_sha256": _sha256(
            identity["config_sha256"],
            "loader_identity.config_sha256",
        ),
        "trace_hook": _resolved_string(
            identity["trace_hook"],
            "loader_identity.trace_hook",
        ),
    }


def _normalize_selection_policy(
    value: object,
    split: str,
) -> dict[str, object]:
    policy = _expect_fields(
        value,
        _SELECTION_POLICY_FIELDS,
        "selection_policy",
    )
    communication_range = _finite_float(
        policy["communication_range_m"],
        "selection_policy.communication_range_m",
        0.0,
    )
    if communication_range != COMMUNICATION_RANGE_M:
        raise V2XSetStandardError("selection_policy.communication_range_m must be 70.0")
    max_cav = _exact_int(policy["max_cav"], "selection_policy.max_cav", 2)
    truncation_rule = _resolved_string(
        policy["truncation_rule"],
        "selection_policy.truncation_rule",
    )
    if truncation_rule != TRUNCATION_RULE:
        raise V2XSetStandardError(
            f"selection_policy.truncation_rule must be {TRUNCATION_RULE}"
        )
    expected_ego_rule = "random_per_epoch" if split == "train" else "fixed_per_scene"
    ego_rule = _resolved_string(
        policy["ego_selection_rule"],
        "selection_policy.ego_selection_rule",
    )
    if ego_rule != expected_ego_rule:
        raise V2XSetStandardError(
            f"selection_policy.ego_selection_rule must be {expected_ego_rule}"
        )
    return {
        "communication_range_m": communication_range,
        "max_cav": max_cav,
        "agent_ordering_rule": _resolved_string(
            policy["agent_ordering_rule"],
            "selection_policy.agent_ordering_rule",
        ),
        "agent_ordering_source": _resolved_string(
            policy["agent_ordering_source"],
            "selection_policy.agent_ordering_source",
        ),
        "truncation_rule": truncation_rule,
        "truncation_source": _resolved_string(
            policy["truncation_source"],
            "selection_policy.truncation_source",
        ),
        "ego_selection_rule": ego_rule,
        "ego_selection_source": _resolved_string(
            policy["ego_selection_source"],
            "selection_policy.ego_selection_source",
        ),
    }


def _normalize_delay_policy(value: object) -> dict[str, object]:
    policy = _expect_fields(value, _DELAY_POLICY_FIELDS, "delay_policy")
    nominal = _exact_int(
        policy["nominal_latency_ms"],
        "delay_policy.nominal_latency_ms",
        0,
    )
    condition_kind = policy["condition_kind"]
    if condition_kind not in ("source_aligned", "supplemental"):
        raise V2XSetStandardError(
            "delay_policy.condition_kind must be source_aligned or supplemental"
        )
    if (
        condition_kind == "source_aligned"
        and nominal not in SOURCE_ALIGNED_LATENCIES_MS
    ):
        raise V2XSetStandardError(
            "source_aligned nominal latency must be one of 0, 200, 300 ms"
        )
    if condition_kind == "supplemental" and nominal not in SUPPLEMENTAL_LATENCIES_MS:
        raise V2XSetStandardError("supplemental nominal latency must be 100 ms")
    if policy["unit"] != "millisecond":
        raise V2XSetStandardError("delay_policy.unit must be millisecond")
    return {
        "nominal_latency_ms": nominal,
        "condition_kind": condition_kind,
        "unit": "millisecond",
        "generation_rule": _resolved_string(
            policy["generation_rule"],
            "delay_policy.generation_rule",
        ),
        "causal_frame_mapping_rule": _resolved_string(
            policy["causal_frame_mapping_rule"],
            "delay_policy.causal_frame_mapping_rule",
        ),
        "uniform_across_remote_agents": _exact_bool(
            policy["uniform_across_remote_agents"],
            "delay_policy.uniform_across_remote_agents",
        ),
        "source_location": _resolved_string(
            policy["source_location"],
            "delay_policy.source_location",
        ),
    }


def _normalize_ego_selection(
    value: object,
    split: str,
    available_vehicle_ids: list[str],
) -> tuple[dict[str, object], str]:
    selection = _expect_fields(value, _EGO_SELECTION_FIELDS, "ego_selection")
    expected_mode = "random" if split == "train" else "fixed"
    if selection["mode"] != expected_mode:
        raise V2XSetStandardError(
            f"ego_selection.mode must be {expected_mode} for split {split}"
        )
    candidates = _string_list(
        selection["candidate_vehicle_agent_ids"],
        "ego_selection.candidate_vehicle_agent_ids",
    )
    if set(candidates) != set(available_vehicle_ids):
        raise V2XSetStandardError(
            "ego_selection candidates must equal all available vehicle agents"
        )
    selected_index = _exact_int(
        selection["selected_index"],
        "ego_selection.selected_index",
        0,
    )
    if selected_index >= len(candidates):
        raise V2XSetStandardError(
            "ego_selection.selected_index is outside candidate list"
        )
    if split == "train":
        rng_seed = _exact_int(selection["rng_seed"], "ego_selection.rng_seed", 0)
        rng_state = _sha256(
            selection["rng_state_before_sha256"],
            "ego_selection.rng_state_before_sha256",
        )
    else:
        if (
            selection["rng_seed"] is not None
            or selection["rng_state_before_sha256"] is not None
        ):
            raise V2XSetStandardError(
                "fixed validation Ego must not contain RNG fields"
            )
        rng_seed = None
        rng_state = None
    normalized = {
        "mode": expected_mode,
        "candidate_vehicle_agent_ids": candidates,
        "selected_index": selected_index,
        "rng_seed": rng_seed,
        "rng_state_before_sha256": rng_state,
    }
    return normalized, candidates[selected_index]


def _normalize_available_agents(
    value: object,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    if not isinstance(value, list) or not value:
        raise V2XSetStandardError("available_agents must be a non-empty array")
    normalized: list[dict[str, object]] = []
    by_id: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(value):
        agent = _expect_fields(
            raw,
            _AVAILABLE_AGENT_FIELDS,
            f"available_agents[{index}]",
        )
        agent_id = _identifier(
            agent["agent_id"],
            f"available_agents[{index}].agent_id",
        )
        if agent_id in by_id:
            raise V2XSetStandardError(f"duplicate available agent ID: {agent_id}")
        agent_type = agent["agent_type"]
        if agent_type not in ("vehicle", "infrastructure"):
            raise V2XSetStandardError(
                f"available_agents[{index}].agent_type is invalid"
            )
        normalized_agent = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "distance_to_ego_m": _finite_float(
                agent["distance_to_ego_m"],
                f"available_agents[{index}].distance_to_ego_m",
                0.0,
            ),
        }
        normalized.append(normalized_agent)
        by_id[agent_id] = normalized_agent
    return normalized, by_id


def _normalize_delay(
    value: object,
    context: str,
    *,
    is_ego: bool,
    target_timestamp_us: int,
) -> dict[str, object]:
    delay = _expect_fields(value, _DELAY_FIELDS, context)
    generated_delay_ms = _exact_int(
        delay["generated_delay_ms"],
        f"{context}.generated_delay_ms",
        0,
    )
    source_timestamp_us = _exact_int(
        delay["source_timestamp_us"],
        f"{context}.source_timestamp_us",
        0,
    )
    arrival_timestamp_us = _exact_int(
        delay["arrival_timestamp_us"],
        f"{context}.arrival_timestamp_us",
        0,
    )
    frame_offset = _exact_int(
        delay["causal_frame_offset"],
        f"{context}.causal_frame_offset",
        0,
    )
    if arrival_timestamp_us != source_timestamp_us + generated_delay_ms * 1000:
        raise V2XSetStandardError(
            f"{context} arrival must equal source + generated delay"
        )
    if arrival_timestamp_us > target_timestamp_us:
        raise V2XSetStandardError(f"{context} is not causally arrived")
    if (source_timestamp_us == target_timestamp_us) != (frame_offset == 0):
        raise V2XSetStandardError(
            f"{context} causal frame offset disagrees with source timestamp"
        )
    if is_ego and (
        generated_delay_ms != 0
        or source_timestamp_us != target_timestamp_us
        or arrival_timestamp_us != target_timestamp_us
        or frame_offset != 0
    ):
        raise V2XSetStandardError(
            f"{context} must record a current zero-delay Ego observation"
        )
    return {
        "generated_delay_ms": generated_delay_ms,
        "source_timestamp_us": source_timestamp_us,
        "arrival_timestamp_us": arrival_timestamp_us,
        "causal_frame_offset": frame_offset,
    }


def _normalize_record(
    value: object,
    *,
    split: str,
    selection_policy: Mapping[str, object],
    delay_policy: Mapping[str, object],
    data_root: Path | None,
    payload_cache: dict[tuple[str, int, str], None] | None,
) -> dict[str, object]:
    record = _expect_fields(value, _RECORD_FIELDS, "record")
    sample_id = _identifier(record["sample_id"], "record.sample_id")
    scene_id = _identifier(record["scene_id"], "record.scene_id")
    target_timestamp_us = _exact_int(
        record["target_timestamp_us"],
        "record.target_timestamp_us",
        0,
    )
    if split == "train":
        epoch: int | None = _exact_int(record["epoch"], "record.epoch", 0)
    else:
        if record["epoch"] is not None:
            raise V2XSetStandardError("validation record.epoch must be null")
        epoch = None
    draw_index = _exact_int(record["draw_index"], "record.draw_index", 0)

    available, available_by_id = _normalize_available_agents(record["available_agents"])
    available_vehicle_ids = [
        item["agent_id"] for item in available if item["agent_type"] == "vehicle"
    ]
    assert all(isinstance(item, str) for item in available_vehicle_ids)
    if not available_vehicle_ids:
        raise V2XSetStandardError("record has no available vehicle Ego candidate")
    ego_selection, ego_id = _normalize_ego_selection(
        record["ego_selection"],
        split,
        available_vehicle_ids,
    )
    ego_available = available_by_id[ego_id]
    if ego_available["distance_to_ego_m"] != 0.0:
        raise V2XSetStandardError("selected Ego distance must be exactly 0.0")

    ordered = _string_list(
        record["ordered_in_range_agent_ids"],
        "record.ordered_in_range_agent_ids",
    )
    expected_in_range = {
        agent_id
        for agent_id, agent in available_by_id.items()
        if float(agent["distance_to_ego_m"]) <= COMMUNICATION_RANGE_M
    }
    if set(ordered) != expected_in_range:
        raise V2XSetStandardError(
            "ordered_in_range_agent_ids must contain exactly the <=70 m agents"
        )
    if not ordered or ordered[0] != ego_id:
        raise V2XSetStandardError(
            "ordered_in_range_agent_ids must start with the selected Ego"
        )
    max_cav = int(selection_policy["max_cav"])
    expected_admitted_ids = ordered[:max_cav]
    expected_truncated_ids = ordered[max_cav:]
    truncated = _string_list(
        record["truncated_agent_ids"],
        "record.truncated_agent_ids",
    )
    if truncated != expected_truncated_ids:
        raise V2XSetStandardError(
            "truncated_agent_ids must be the ordered suffix after max_cav"
        )

    raw_admitted = record["admitted_agents"]
    if not isinstance(raw_admitted, list):
        raise V2XSetStandardError("record.admitted_agents must be an array")
    admitted: list[dict[str, object]] = []
    admitted_ids: list[str] = []
    remote_delays: list[int] = []
    for index, raw in enumerate(raw_admitted):
        context = f"record.admitted_agents[{index}]"
        agent = _expect_fields(raw, _ADMITTED_AGENT_FIELDS, context)
        agent_id = _identifier(agent["agent_id"], f"{context}.agent_id")
        admitted_ids.append(agent_id)
        if agent_id not in available_by_id:
            raise V2XSetStandardError(f"{context}.agent_id is not an available agent")
        available_agent = available_by_id[agent_id]
        if agent["agent_type"] != available_agent["agent_type"]:
            raise V2XSetStandardError(
                f"{context}.agent_type disagrees with available_agents"
            )
        distance = _finite_float(
            agent["distance_to_ego_m"],
            f"{context}.distance_to_ego_m",
            0.0,
        )
        if distance != available_agent["distance_to_ego_m"]:
            raise V2XSetStandardError(
                f"{context}.distance_to_ego_m disagrees with available_agents"
            )
        payload = _normalize_payload(agent["lidar"], f"{context}.lidar")
        if data_root is not None:
            assert payload_cache is not None
            _verify_payload(payload, data_root, payload_cache)
        is_ego = agent_id == ego_id
        normalized_delay = _normalize_delay(
            agent["delay"],
            f"{context}.delay",
            is_ego=is_ego,
            target_timestamp_us=target_timestamp_us,
        )
        if not is_ego:
            remote_delays.append(int(normalized_delay["generated_delay_ms"]))
        admitted.append(
            {
                "agent_id": agent_id,
                "agent_type": agent["agent_type"],
                "distance_to_ego_m": distance,
                "lidar": payload,
                "delay": normalized_delay,
            }
        )
    if admitted_ids != expected_admitted_ids:
        raise V2XSetStandardError(
            "admitted_agents must be the ordered prefix through max_cav"
        )

    nominal = int(delay_policy["nominal_latency_ms"])
    uniform = bool(delay_policy["uniform_across_remote_agents"])
    if nominal == 0 and any(delay != 0 for delay in remote_delays):
        raise V2XSetStandardError(
            "0 ms condition contains a nonzero remote-agent delay"
        )
    if uniform and any(delay != nominal for delay in remote_delays):
        raise V2XSetStandardError(
            "uniform delay trace must assign nominal latency to every remote agent"
        )

    return {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "target_timestamp_us": target_timestamp_us,
        "epoch": epoch,
        "draw_index": draw_index,
        "ego_selection": ego_selection,
        "available_agents": available,
        "ordered_in_range_agent_ids": ordered,
        "admitted_agents": admitted,
        "truncated_agent_ids": truncated,
    }


def _load_trace(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V2XSetStandardError("unable to read Standard dataloader trace") from error

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise V2XSetStandardError(f"duplicate trace key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise V2XSetStandardError(f"non-finite trace value: {value}")

    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except V2XSetStandardError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise V2XSetStandardError("invalid Standard dataloader trace JSON") from error
    trace = _expect_fields(decoded, _TRACE_FIELDS, "Standard dataloader trace")
    if canonical_json_bytes(trace) != raw:
        raise V2XSetStandardError("Standard dataloader trace JSON is not canonical")
    if trace["schema_version"] != TRACE_SCHEMA_VERSION:
        raise V2XSetStandardError("unsupported Standard trace schema_version")
    if trace["artifact_type"] != TRACE_TYPE:
        raise V2XSetStandardError("unexpected Standard trace artifact_type")
    if trace["dataset_name"] != DATASET_NAME:
        raise V2XSetStandardError("Standard trace dataset_name must be V2XSet")
    digest = _sha256(trace["content_sha256"], "trace.content_sha256")
    if content_sha256(trace) != digest:
        raise V2XSetStandardError("Standard trace content hash mismatch")
    return trace, raw


def export_standard_manifest(
    trace_path: Path,
    data_root: Path,
) -> dict[str, object]:
    """Validate a real-loader trace and export a self-hashed Standard manifest."""

    try:
        resolved_root = Path(data_root).resolve(strict=True)
    except OSError as error:
        raise V2XSetStandardError("data_root does not exist") from error
    if not resolved_root.is_dir():
        raise V2XSetStandardError("data_root must be a directory")

    trace, raw = _load_trace(Path(trace_path))
    split = trace["split"]
    if split not in ("train", "validation"):
        raise V2XSetStandardError("trace.split must be train or validation")
    assert isinstance(split, str)
    dataset_identity = _normalize_dataset_identity(trace["dataset_identity"])
    loader_identity = _normalize_loader_identity(trace["loader_identity"])
    selection_policy = _normalize_selection_policy(
        trace["selection_policy"],
        split,
    )
    delay_policy = _normalize_delay_policy(trace["delay_policy"])

    raw_records = trace["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise V2XSetStandardError("trace.records must be a non-empty array")
    payload_cache: dict[tuple[str, int, str], None] = {}
    records = [
        _normalize_record(
            record,
            split=split,
            selection_policy=selection_policy,
            delay_policy=delay_policy,
            data_root=resolved_root,
            payload_cache=payload_cache,
        )
        for record in raw_records
    ]

    record_keys: set[tuple[object, ...]] = set()
    fixed_ego_by_scene: dict[str, str] = {}
    for record in records:
        if split == "train":
            key = (record["epoch"], record["draw_index"])
        else:
            key = (record["sample_id"],)
        if key in record_keys:
            raise V2XSetStandardError("duplicate Standard trace record key")
        record_keys.add(key)
        if split == "validation":
            scene_id = str(record["scene_id"])
            candidates = record["ego_selection"]["candidate_vehicle_agent_ids"]
            index = record["ego_selection"]["selected_index"]
            ego_id = candidates[index]
            previous = fixed_ego_by_scene.setdefault(scene_id, ego_id)
            if previous != ego_id:
                raise V2XSetStandardError(
                    "validation Ego must remain fixed within each scene"
                )

    epoch_ids = (
        sorted({int(record["epoch"]) for record in records}) if split == "train" else []
    )
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": MANIFEST_TYPE,
        "protocol_scope": PROTOCOL_SCOPE,
        "dataset_name": DATASET_NAME,
        "split": split,
        "modalities": ["lidar"],
        "input_trace": {
            "artifact_type": TRACE_TYPE,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "content_sha256": trace["content_sha256"],
        },
        "dataset_identity": dataset_identity,
        "loader_identity": loader_identity,
        "selection_policy": selection_policy,
        "delay_policy": delay_policy,
        "method_eligibility": dict(METHOD_ELIGIBILITY),
        "measurement_boundary": dict(MEASUREMENT_BOUNDARY),
        "record_count": len(records),
        "scene_count": len({record["scene_id"] for record in records}),
        "epoch_ids": epoch_ids,
        "records": records,
    }
    manifest["content_sha256"] = content_sha256(manifest)
    validate_standard_manifest(manifest)
    return manifest


def validate_standard_manifest(value: object) -> dict[str, object]:
    """Fail closed on malformed or scope-expanding Standard manifests."""

    manifest = _expect_fields(value, _MANIFEST_FIELDS, "Standard manifest")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise V2XSetStandardError("Standard manifest schema_version mismatch")
    if manifest["artifact_type"] != MANIFEST_TYPE:
        raise V2XSetStandardError("Standard manifest artifact_type mismatch")
    if manifest["protocol_scope"] != PROTOCOL_SCOPE:
        raise V2XSetStandardError("Standard manifest protocol_scope mismatch")
    if manifest["dataset_name"] != DATASET_NAME:
        raise V2XSetStandardError("Standard manifest dataset_name mismatch")
    if manifest["split"] not in ("train", "validation"):
        raise V2XSetStandardError("Standard manifest split mismatch")
    if manifest["modalities"] != ["lidar"]:
        raise V2XSetStandardError("Standard manifest must be LiDAR-only")
    input_trace = _expect_fields(
        manifest["input_trace"],
        _INPUT_TRACE_FIELDS,
        "Standard manifest input_trace",
    )
    if input_trace["artifact_type"] != TRACE_TYPE:
        raise V2XSetStandardError("Standard manifest trace type mismatch")
    _sha256(input_trace["file_sha256"], "input_trace.file_sha256")
    _sha256(input_trace["content_sha256"], "input_trace.content_sha256")
    _normalize_dataset_identity(manifest["dataset_identity"])
    _normalize_loader_identity(manifest["loader_identity"])
    split = str(manifest["split"])
    selection_policy = _normalize_selection_policy(
        manifest["selection_policy"],
        split,
    )
    delay_policy = _normalize_delay_policy(manifest["delay_policy"])
    method = _expect_fields(
        manifest["method_eligibility"],
        _METHOD_FIELDS,
        "method_eligibility",
    )
    if method != METHOD_ELIGIBILITY:
        raise V2XSetStandardError(
            "current method eligibility must remain N/A without an adapter"
        )
    boundary = _expect_fields(
        manifest["measurement_boundary"],
        _MEASUREMENT_FIELDS,
        "measurement_boundary",
    )
    if boundary != MEASUREMENT_BOUNDARY:
        raise V2XSetStandardError(
            "Standard manifest must not contain model or evaluator outputs"
        )
    records = manifest["records"]
    if not isinstance(records, list) or not records:
        raise V2XSetStandardError("Standard manifest records must be non-empty")
    normalized_records = [
        _normalize_record(
            record,
            split=split,
            selection_policy=selection_policy,
            delay_policy=delay_policy,
            data_root=None,
            payload_cache=None,
        )
        for record in records
    ]
    if canonical_json_bytes(normalized_records) != canonical_json_bytes(records):
        raise V2XSetStandardError("Standard manifest records are not normalized")
    record_count = _exact_int(
        manifest["record_count"],
        "Standard manifest record_count",
        1,
    )
    if record_count != len(records):
        raise V2XSetStandardError("Standard manifest record_count mismatch")
    scene_count = len(
        {
            _identifier(
                _expect_fields(record, _RECORD_FIELDS, "manifest record")["scene_id"],
                "manifest record.scene_id",
            )
            for record in records
        }
    )
    recorded_scene_count = _exact_int(
        manifest["scene_count"],
        "Standard manifest scene_count",
        1,
    )
    if recorded_scene_count != scene_count:
        raise V2XSetStandardError("Standard manifest scene_count mismatch")
    epochs = manifest["epoch_ids"]
    if not isinstance(epochs, list) or any(
        type(epoch) is not int or epoch < 0 for epoch in epochs
    ):
        raise V2XSetStandardError("Standard manifest epoch_ids are invalid")
    if epochs != sorted(set(epochs)):
        raise V2XSetStandardError(
            "Standard manifest epoch_ids must be unique and sorted"
        )
    expected_epochs = (
        sorted(
            {
                _exact_int(
                    _expect_fields(record, _RECORD_FIELDS, "manifest record")["epoch"],
                    "manifest record.epoch",
                    0,
                )
                for record in records
            }
        )
        if split == "train"
        else []
    )
    if epochs != expected_epochs:
        raise V2XSetStandardError("Standard manifest epoch_ids mismatch")
    record_keys: set[tuple[object, ...]] = set()
    fixed_ego_by_scene: dict[str, str] = {}
    for record in normalized_records:
        key = (
            (record["epoch"], record["draw_index"])
            if split == "train"
            else (record["sample_id"],)
        )
        if key in record_keys:
            raise V2XSetStandardError("duplicate Standard manifest record key")
        record_keys.add(key)
        if split == "validation":
            scene_id = str(record["scene_id"])
            selection = record["ego_selection"]
            candidates = selection["candidate_vehicle_agent_ids"]
            ego_id = candidates[selection["selected_index"]]
            previous = fixed_ego_by_scene.setdefault(scene_id, ego_id)
            if previous != ego_id:
                raise V2XSetStandardError(
                    "validation Ego must remain fixed within each scene"
                )
    digest = _sha256(
        manifest["content_sha256"],
        "Standard manifest content_sha256",
    )
    if content_sha256(manifest) != digest:
        raise V2XSetStandardError("Standard manifest content hash mismatch")
    return manifest


def _regular_bytes(path: Path, context: str) -> bytes:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as error:
        raise V2XSetStandardError(f"unable to inspect {context}") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise V2XSetStandardError(f"{context} must be a regular file")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        return b"".join(blocks)
    finally:
        os.close(descriptor)


def write_standard_manifest(
    output: Path,
    manifest: Mapping[str, object],
) -> Path:
    """Atomically publish a validated manifest without overwriting conflicts."""

    validate_standard_manifest(manifest)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(manifest)
    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            staging,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short Standard manifest write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(staging, output, follow_symlinks=False)
        except FileExistsError:
            if _regular_bytes(output, "existing Standard manifest") != raw:
                raise V2XSetStandardError("Standard manifest destination conflict")
        directory = os.open(
            output.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return output
    except V2XSetStandardError:
        raise
    except OSError as error:
        raise V2XSetStandardError("Standard manifest publication failed") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        staging.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a trace emitted by the actual V2XSet multi-agent "
            "dataloader and export a LiDAR-only Standard manifest. This tool "
            "does not implement that loader, an evaluator, or a model adapter."
        )
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = export_standard_manifest(args.trace, args.data_root)
        output = write_standard_manifest(args.out, manifest)
    except (OSError, V2XSetStandardError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_type": MANIFEST_TYPE,
                "content_sha256": manifest["content_sha256"],
                "split": manifest["split"],
                "record_count": manifest["record_count"],
                "output": str(output.resolve()),
                "standard_loader_implemented_in_repository": False,
                "resilient_v2x_method_eligibility": "N/A",
                "contains_model_results": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "COMMUNICATION_RANGE_M",
    "MANIFEST_TYPE",
    "METHOD_ELIGIBILITY",
    "SOURCE_ALIGNED_LATENCIES_MS",
    "SUPPLEMENTAL_LATENCIES_MS",
    "TRACE_TYPE",
    "TRUNCATION_RULE",
    "V2XSetStandardError",
    "export_standard_manifest",
    "main",
    "parse_args",
    "validate_standard_manifest",
    "write_standard_manifest",
)
