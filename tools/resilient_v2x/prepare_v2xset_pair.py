#!/usr/bin/env python3
"""Prepare the paper-scoped deterministic V2XSet Pair adapter manifest.

This is an offline adapter boundary, not an implementation of the V2XSet
Standard multi-agent protocol.
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


INVENTORY_SCHEMA_VERSION = 2
PAIR_SCHEMA_VERSION = 2
TARGET_SCHEMA_VERSION = 1
INVENTORY_TYPE = "v2xset_normalized_inventory"
PAIR_TYPE = "v2xset_pair_manifest"
TARGET_TYPE = "v2xset_pair_target_manifest"
DATASET_NAME = "V2XSet"
CAMERA_IDS = ("camera_0", "camera_1", "camera_2", "camera_3")
SPLITS = ("train", "validation")
AGENT_TYPES = ("av", "infrastructure")
MAIN_CONDITION_NAMES = (
    "full_lc",
    "l_fail_e_plus_r",
    "c_fail_e_plus_r",
)
DIAGNOSTIC_CONDITION_NAMES = (
    "l_fail_ego_only",
    "c_fail_ego_only",
    "l_fail_infrastructure_only",
    "c_fail_infrastructure_only",
)
CONDITION_NAMES = MAIN_CONDITION_NAMES + DIAGNOSTIC_CONDITION_NAMES
MAX_COLLABORATOR_DISTANCE_M = 70.0
TRANSFORM_ATOL = 1e-5
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
SHA256 = re.compile(r"[0-9a-f]{64}")
DISCLAIMER = (
    "V2XSet Standard multi-agent evaluation is not implemented; this artifact "
    "contains only the paper-scoped deterministic one-ego/one-collaborator "
    "Pair adapter."
)


class V2XSetPairError(ValueError):
    """Raised when the normalized inventory or Pair output is invalid."""


def _expect_fields(
    value: object,
    expected: frozenset[str],
    context: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise V2XSetPairError(f"{context} must be an object")
    if frozenset(value) != expected:
        raise V2XSetPairError(f"{context} fields mismatch")
    return dict(value)


def _identifier(value: object, context: str) -> str:
    if type(value) is not str or IDENTIFIER.fullmatch(value) is None:
        raise V2XSetPairError(f"{context} is not a canonical identifier")
    return value


def _exact_int(value: object, context: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise V2XSetPairError(f"{context} must be an integer >= {minimum}")
    return value


def _matrix(
    value: object,
    rows: int,
    columns: int,
) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != rows:
        return None
    result: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            return None
        normalized: list[float] = []
        for item in row:
            if type(item) not in (int, float) or not math.isfinite(float(item)):
                return None
            normalized.append(float(item))
        result.append(normalized)
    return result


def _determinant_3x3(matrix: list[list[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _rigid_transform(value: object) -> list[list[float]] | None:
    matrix = _matrix(value, 4, 4)
    if matrix is None:
        return None
    expected_bottom = (0.0, 0.0, 0.0, 1.0)
    if any(
        abs(matrix[3][index] - expected_bottom[index]) > TRANSFORM_ATOL
        for index in range(4)
    ):
        return None
    rotation = [row[:3] for row in matrix[:3]]
    for first in range(3):
        for second in range(3):
            dot = sum(rotation[row][first] * rotation[row][second] for row in range(3))
            expected = 1.0 if first == second else 0.0
            if abs(dot - expected) > TRANSFORM_ATOL:
                return None
    if abs(_determinant_3x3(rotation) - 1.0) > TRANSFORM_ATOL:
        return None
    return matrix


def _intrinsic(value: object) -> list[list[float]] | None:
    matrix = _matrix(value, 3, 3)
    if matrix is None or matrix[0][0] <= 0 or matrix[1][1] <= 0:
        return None
    if any(
        abs(matrix[2][index] - expected) > TRANSFORM_ATOL
        for index, expected in enumerate((0.0, 0.0, 1.0))
    ):
        return None
    return matrix


def _relative_path(value: object) -> str | None:
    if type(value) is not str or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return None
    if path.as_posix() != value:
        return None
    return value


def _artifact_identity(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    if frozenset(value) != {"relative_path", "size_bytes", "sha256"}:
        return None
    relative = _relative_path(value.get("relative_path"))
    size = value.get("size_bytes")
    digest = value.get("sha256")
    if (
        relative is None
        or type(size) is not int
        or size < 0
        or type(digest) is not str
        or SHA256.fullmatch(digest) is None
    ):
        return None
    return {"relative_path": relative, "size_bytes": size, "sha256": digest}


def _verify_payload(
    identity: dict[str, object],
    data_root: Path,
    cache: dict[tuple[str, int, str], str | None],
) -> str | None:
    key = (
        identity["relative_path"],
        identity["size_bytes"],
        identity["sha256"],
    )
    assert (
        isinstance(key[0], str) and isinstance(key[1], int) and isinstance(key[2], str)
    )
    if key in cache:
        return cache[key]
    candidate = data_root.joinpath(*PurePosixPath(key[0]).parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(data_root)
    except (OSError, ValueError):
        cache[key] = "missing_or_outside_data_root"
        return cache[key]
    try:
        descriptor = os.open(
            candidate,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError:
        cache[key] = "missing_or_nonregular"
        return cache[key]
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            cache[key] = "missing_or_nonregular"
            return cache[key]
        if before.st_size != key[1]:
            cache[key] = "size_mismatch"
            return cache[key]
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            cache[key] = "payload_changed_during_verification"
        elif digest.hexdigest() != key[2]:
            cache[key] = "sha256_mismatch"
        else:
            cache[key] = None
        return cache[key]
    finally:
        os.close(descriptor)


def _payload_reason(
    value: object,
    data_root: Path,
    cache: dict[tuple[str, int, str], str | None],
    code: str,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    identity = _artifact_identity(value)
    if identity is None:
        return None, {"code": code, "failure": "invalid_artifact_identity"}
    failure = _verify_payload(identity, data_root, cache)
    if failure is not None:
        return None, {
            "code": code,
            "relative_path": identity["relative_path"],
            "failure": failure,
        }
    return identity, None


def _normalize_agent(
    value: object,
    data_root: Path,
    cache: dict[tuple[str, int, str], str | None],
) -> tuple[
    str,
    int,
    str,
    dict[str, object] | None,
    list[dict[str, object]],
]:
    agent = _expect_fields(
        value,
        frozenset(
            {
                "agent_id",
                "numeric_id",
                "agent_type",
                "world_from_agent",
                "lidar",
                "cameras",
            }
        ),
        "agent",
    )
    agent_id = _identifier(agent["agent_id"], "agent_id")
    numeric_id = _exact_int(agent["numeric_id"], "agent numeric_id")
    agent_type = agent["agent_type"]
    if agent_type not in AGENT_TYPES:
        raise V2XSetPairError("agent_type must be av or infrastructure")
    reasons: list[dict[str, object]] = []
    world_from_agent = _rigid_transform(agent["world_from_agent"])
    if world_from_agent is None:
        reasons.append({"code": "invalid_world_from_agent"})

    lidar_value = agent["lidar"]
    lidar_payload: dict[str, object] | None = None
    agent_from_lidar: list[list[float]] | None = None
    if not isinstance(lidar_value, Mapping) or frozenset(lidar_value) != {
        "payload",
        "agent_from_lidar",
    }:
        reasons.append({"code": "invalid_lidar_record"})
    else:
        agent_from_lidar = _rigid_transform(lidar_value["agent_from_lidar"])
        if agent_from_lidar is None:
            reasons.append({"code": "invalid_lidar_extrinsic"})
        lidar_payload, reason = _payload_reason(
            lidar_value["payload"],
            data_root,
            cache,
            "invalid_lidar_payload",
        )
        if reason is not None:
            reasons.append(reason)

    camera_values = agent["cameras"]
    cameras: list[dict[str, object]] = []
    camera_by_id: dict[str, object] = {}
    if not isinstance(camera_values, list):
        reasons.append({"code": "camera_set_mismatch", "present_camera_ids": []})
    else:
        for index, camera_value in enumerate(camera_values):
            if not isinstance(camera_value, Mapping):
                reasons.append({"code": "invalid_camera_record", "index": index})
                continue
            if frozenset(camera_value) != {
                "camera_id",
                "image",
                "intrinsic",
                "agent_from_camera",
            }:
                reasons.append({"code": "invalid_camera_record", "index": index})
                continue
            camera_id = camera_value.get("camera_id")
            if type(camera_id) is not str or camera_id not in CAMERA_IDS:
                reasons.append({"code": "invalid_camera_id", "index": index})
                continue
            if camera_id in camera_by_id:
                reasons.append({"code": "duplicate_camera_id", "camera_id": camera_id})
                continue
            camera_by_id[camera_id] = camera_value
        if set(camera_by_id) != set(CAMERA_IDS):
            reasons.append(
                {
                    "code": "camera_set_mismatch",
                    "required_camera_ids": list(CAMERA_IDS),
                    "present_camera_ids": sorted(camera_by_id),
                }
            )
        for camera_id in CAMERA_IDS:
            camera_value = camera_by_id.get(camera_id)
            if not isinstance(camera_value, Mapping):
                continue
            intrinsic = _intrinsic(camera_value["intrinsic"])
            extrinsic = _rigid_transform(camera_value["agent_from_camera"])
            image, reason = _payload_reason(
                camera_value["image"],
                data_root,
                cache,
                "invalid_camera_payload",
            )
            if intrinsic is None:
                reasons.append(
                    {"code": "invalid_camera_intrinsic", "camera_id": camera_id}
                )
            if extrinsic is None:
                reasons.append(
                    {"code": "invalid_camera_extrinsic", "camera_id": camera_id}
                )
            if reason is not None:
                reason["camera_id"] = camera_id
                reasons.append(reason)
            if intrinsic is not None and extrinsic is not None and image is not None:
                cameras.append(
                    {
                        "camera_id": camera_id,
                        "image": image,
                        "intrinsic": intrinsic,
                        "agent_from_camera": extrinsic,
                    }
                )

    if reasons:
        return agent_id, numeric_id, agent_type, None, reasons
    assert (
        world_from_agent is not None
        and lidar_payload is not None
        and agent_from_lidar is not None
        and len(cameras) == 4
    )
    return (
        agent_id,
        numeric_id,
        agent_type,
        {
            "agent_id": agent_id,
            "numeric_id": numeric_id,
            "agent_type": agent_type,
            "world_from_agent": world_from_agent,
            "lidar": {
                "payload": lidar_payload,
                "agent_from_lidar": agent_from_lidar,
            },
            "cameras": cameras,
        },
        [],
    )


def _load_inventory(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise V2XSetPairError("unable to read normalized inventory") from error

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise V2XSetPairError(f"duplicate inventory key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise V2XSetPairError(f"non-finite inventory value: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except V2XSetPairError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise V2XSetPairError("invalid normalized inventory JSON") from error
    inventory = _expect_fields(
        value,
        frozenset(
            {
                "schema_version",
                "artifact_type",
                "dataset_name",
                "coordinate_unit",
                "timestamp_unit",
                "configured_delays_ms",
                "history_limit",
                "frame_interval_us",
                "scenes",
                "content_sha256",
            }
        ),
        "normalized inventory",
    )
    if canonical_json_bytes(inventory) != raw:
        raise V2XSetPairError("normalized inventory JSON is not canonical")
    if inventory["schema_version"] != INVENTORY_SCHEMA_VERSION:
        raise V2XSetPairError("unsupported normalized inventory schema_version")
    if inventory["artifact_type"] != INVENTORY_TYPE:
        raise V2XSetPairError("unexpected normalized inventory artifact_type")
    if inventory["dataset_name"] != DATASET_NAME:
        raise V2XSetPairError("normalized inventory dataset_name must be V2XSet")
    if inventory["coordinate_unit"] != "meter":
        raise V2XSetPairError("normalized inventory coordinate_unit must be meter")
    if inventory["timestamp_unit"] != "microsecond":
        raise V2XSetPairError("normalized inventory timestamp_unit must be microsecond")
    digest = inventory["content_sha256"]
    if (
        type(digest) is not str
        or SHA256.fullmatch(digest) is None
        or content_sha256(inventory) != digest
    ):
        raise V2XSetPairError("normalized inventory content hash mismatch")
    if not isinstance(inventory["scenes"], list):
        raise V2XSetPairError("normalized inventory scenes must be an array")
    return inventory, raw


def _stable_id(prefix: str, domain: bytes, value: Mapping[str, object]) -> str:
    digest = hashlib.sha256(domain + b"\x00" + canonical_json_bytes(value)).hexdigest()
    return f"{prefix}-{digest}"


def _sequence_pair_id(key: Mapping[str, object]) -> str:
    return _stable_id("pair", b"v2xset-sequence-pair-v2", key)


def _target_sample_id(key: Mapping[str, object]) -> str:
    return _stable_id("sample", b"v2xset-pair-target-v1", key)


def _frame_id(scene_id: str, timestamp_us: int) -> str:
    return _stable_id(
        "frame",
        b"v2xset-pair-frame-v1",
        {"scene_id": scene_id, "timestamp_us": timestamp_us},
    )


def _id_list_sha256(domain: bytes, identifiers: Sequence[str]) -> str:
    return hashlib.sha256(
        domain + b"\x00" + canonical_json_bytes(list(identifiers))
    ).hexdigest()


def _id_inventory(domain: bytes, identifiers: Sequence[str]) -> dict[str, object]:
    values = list(identifiers)
    if len(values) != len(set(values)):
        raise V2XSetPairError("derived identifier inventory contains duplicates")
    return {
        "count": len(values),
        "ids": values,
        "ids_sha256": _id_list_sha256(domain, values),
    }


def _protocol_values(
    inventory: Mapping[str, object],
) -> tuple[list[int], int, int]:
    raw_delays = inventory["configured_delays_ms"]
    if not isinstance(raw_delays, list) or not raw_delays:
        raise V2XSetPairError("configured_delays_ms must be a non-empty array")
    delays = [
        _exact_int(value, f"configured_delays_ms[{index}]")
        for index, value in enumerate(raw_delays)
    ]
    if delays != sorted(set(delays)) or delays[0] != 0:
        raise V2XSetPairError(
            "configured_delays_ms must be unique, ascending, and start at zero"
        )
    history_limit = _exact_int(inventory["history_limit"], "history_limit")
    frame_interval_us = _exact_int(
        inventory["frame_interval_us"],
        "frame_interval_us",
        minimum=1,
    )
    for delay_ms in delays:
        if (delay_ms * 1000) % frame_interval_us:
            raise V2XSetPairError(
                "every configured delay must map exactly to the frame interval"
            )
    return delays, history_limit, frame_interval_us


def _sequence_selection_parameters(
    delays_ms: Sequence[int],
    history_limit: int,
    frame_interval_us: int,
) -> dict[str, object]:
    return {
        "train_anchor": "earliest target with at least one eligible AV/Infra pair",
        "train_ego_order": ["numeric_id_ascending", "agent_id_ascending"],
        "validation_ego_source": "scene.standard_fixed_ego_agent_id",
        "validation_ego_substitution": False,
        "pair_freeze_scope": "sequence",
        "max_collaborator_distance_m": MAX_COLLABORATOR_DISTANCE_M,
        "distance_metric": "Euclidean XY distance between target-time world_from_agent translations",
        "infrastructure_order": [
            "squared_distance_ascending",
            "numeric_id_ascending",
            "agent_id_ascending",
        ],
        "required_camera_ids": list(CAMERA_IDS),
        "required_lidar_payload": True,
        "payload_verification": ["regular_file", "size_bytes", "sha256"],
        "transform_validation_atol": TRANSFORM_ATOL,
        "configured_delays_ms": list(delays_ms),
        "history_limit": history_limit,
        "history_horizons": list(range(history_limit + 1)),
        "frame_interval_us": frame_interval_us,
        "causal_timestamp_rule": (
            "ego=t-h*frame_interval_us; "
            "infrastructure=t-delay_ms*1000-h*frame_interval_us"
        ),
        "target_generation_stage": "before_latency_selection_and_fault_injection",
        "unselected_agent_policy": "exclude_before_encoding_fusion_and_target_eligibility",
    }


def _role_agent(agent: Mapping[str, object], role: str) -> dict[str, object]:
    return {**dict(agent), "role": role}


def _ordered_agent_ids(frame: Mapping[str, object]) -> list[str]:
    metadata = frame["metadata"]
    assert isinstance(metadata, Mapping)
    type_order = {"av": 0, "infrastructure": 1}
    return sorted(
        metadata,
        key=lambda agent_id: (
            type_order[metadata[agent_id]["agent_type"]],
            metadata[agent_id]["numeric_id"],
            agent_id,
        ),
    )


def _endpoint_record(
    *,
    frames: Mapping[int, Mapping[str, object]],
    agent_id: str,
    expected_numeric_id: int,
    expected_agent_type: str,
    role: str,
    timestamp_us: int,
    horizon: int,
    delay_ms: int | None,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    location = {
        "member": role,
        "delay_ms": delay_ms,
        "horizon": horizon,
        "required_timestamp_us": timestamp_us,
    }
    frame = frames.get(timestamp_us)
    if frame is None:
        return None, [{"code": "required_endpoint_frame_missing", **location}]
    metadata = frame["metadata"]
    assert isinstance(metadata, Mapping)
    observed = metadata.get(agent_id)
    if not isinstance(observed, Mapping):
        return None, [
            {
                "code": "agent_missing_at_required_endpoint",
                "agent_id": agent_id,
                **location,
            }
        ]
    reasons: list[dict[str, object]] = []
    if observed["agent_type"] != expected_agent_type:
        reasons.append(
            {
                "code": "agent_type_changed_at_required_endpoint",
                "agent_id": agent_id,
                "expected_agent_type": expected_agent_type,
                "observed_agent_type": observed["agent_type"],
                **location,
            }
        )
    if observed["numeric_id"] != expected_numeric_id:
        reasons.append(
            {
                "code": "numeric_id_changed_at_required_endpoint",
                "agent_id": agent_id,
                "expected_numeric_id": expected_numeric_id,
                "observed_numeric_id": observed["numeric_id"],
                **location,
            }
        )
    invalid = frame["invalid"]
    normalized = frame["normalized"]
    assert isinstance(invalid, Mapping) and isinstance(normalized, Mapping)
    if agent_id in invalid:
        reasons.append(
            {
                "code": "agent_invalid_at_required_endpoint",
                "agent_id": agent_id,
                "details": invalid[agent_id],
                **location,
            }
        )
    agent = normalized.get(agent_id)
    if reasons or not isinstance(agent, Mapping):
        return None, reasons
    record: dict[str, object] = {
        "horizon": horizon,
        "timestamp_us": timestamp_us,
        "agent": _role_agent(agent, role),
    }
    if delay_ms is not None:
        record["delay_ms"] = delay_ms
    return record, []


def _pair_eligibility(
    *,
    frames: Mapping[int, Mapping[str, object]],
    target_timestamp_us: int,
    ego_agent_id: str,
    ego_numeric_id: int,
    infrastructure_agent_id: str,
    infrastructure_numeric_id: int,
    delays_ms: Sequence[int],
    history_limit: int,
    frame_interval_us: int,
) -> tuple[float | None, dict[str, object] | None, list[dict[str, object]]]:
    ego_history: list[dict[str, object]] = []
    infrastructure_histories: list[dict[str, object]] = []
    reasons: list[dict[str, object]] = []
    for horizon in range(history_limit + 1):
        timestamp_us = target_timestamp_us - horizon * frame_interval_us
        record, failures = _endpoint_record(
            frames=frames,
            agent_id=ego_agent_id,
            expected_numeric_id=ego_numeric_id,
            expected_agent_type="av",
            role="ego",
            timestamp_us=timestamp_us,
            horizon=horizon,
            delay_ms=None,
        )
        reasons.extend(failures)
        if record is not None:
            ego_history.append(record)
    for delay_ms in delays_ms:
        history: list[dict[str, object]] = []
        for horizon in range(history_limit + 1):
            timestamp_us = (
                target_timestamp_us - delay_ms * 1000 - horizon * frame_interval_us
            )
            record, failures = _endpoint_record(
                frames=frames,
                agent_id=infrastructure_agent_id,
                expected_numeric_id=infrastructure_numeric_id,
                expected_agent_type="infrastructure",
                role="infrastructure",
                timestamp_us=timestamp_us,
                horizon=horizon,
                delay_ms=delay_ms,
            )
            reasons.extend(failures)
            if record is not None:
                history.append(record)
        infrastructure_histories.append({"delay_ms": delay_ms, "history": history})

    distance: float | None = None
    target_frame = frames.get(target_timestamp_us)
    if target_frame is not None:
        normalized = target_frame["normalized"]
        assert isinstance(normalized, Mapping)
        ego = normalized.get(ego_agent_id)
        infrastructure = normalized.get(infrastructure_agent_id)
        if isinstance(ego, Mapping) and isinstance(infrastructure, Mapping):
            ego_pose = ego["world_from_agent"]
            infrastructure_pose = infrastructure["world_from_agent"]
            assert isinstance(ego_pose, list) and isinstance(infrastructure_pose, list)
            squared = math.fsum(
                (
                    (infrastructure_pose[0][3] - ego_pose[0][3]) ** 2,
                    (infrastructure_pose[1][3] - ego_pose[1][3]) ** 2,
                )
            )
            distance = math.sqrt(squared)
            if distance > MAX_COLLABORATOR_DISTANCE_M:
                reasons.append(
                    {
                        "code": "outside_max_collaborator_distance",
                        "distance_m": distance,
                        "max_distance_m": MAX_COLLABORATOR_DISTANCE_M,
                    }
                )
    if reasons:
        return distance, None, reasons
    if distance is None:
        raise V2XSetPairError("eligible pair is missing target-time poses")
    coverage = {
        "ego_history": ego_history,
        "infrastructure_histories": infrastructure_histories,
    }
    return distance, coverage, []


def _anchor_selection(
    *,
    split: str,
    standard_fixed_ego_agent_id: str | None,
    target_timestamp_us: int,
    frames: Mapping[int, Mapping[str, object]],
    delays_ms: Sequence[int],
    history_limit: int,
    frame_interval_us: int,
) -> tuple[dict[str, object], dict[str, object] | None]:
    target = frames[target_timestamp_us]
    metadata = target["metadata"]
    assert isinstance(metadata, Mapping)
    all_av_ids = sorted(
        (agent_id for agent_id, item in metadata.items() if item["agent_type"] == "av"),
        key=lambda agent_id: (metadata[agent_id]["numeric_id"], agent_id),
    )
    infrastructure_ids = sorted(
        (
            agent_id
            for agent_id, item in metadata.items()
            if item["agent_type"] == "infrastructure"
        ),
        key=lambda agent_id: (metadata[agent_id]["numeric_id"], agent_id),
    )
    ignored_av_ids: list[str] = []
    frame_reasons: list[dict[str, object]] = []
    if split == "validation":
        assert standard_fixed_ego_agent_id is not None
        ignored_av_ids = [
            agent_id
            for agent_id in all_av_ids
            if agent_id != standard_fixed_ego_agent_id
        ]
        fixed = metadata.get(standard_fixed_ego_agent_id)
        if not isinstance(fixed, Mapping):
            av_ids: list[str] = []
            frame_reasons.append(
                {
                    "code": "standard_fixed_ego_missing_no_substitution",
                    "agent_id": standard_fixed_ego_agent_id,
                }
            )
        elif fixed["agent_type"] != "av":
            av_ids = []
            frame_reasons.append(
                {
                    "code": "standard_fixed_ego_not_av_no_substitution",
                    "agent_id": standard_fixed_ego_agent_id,
                }
            )
        else:
            av_ids = [standard_fixed_ego_agent_id]
    else:
        av_ids = all_av_ids
    if not av_ids and not frame_reasons:
        frame_reasons.append({"code": "no_av_candidate_at_target"})
    if not infrastructure_ids:
        frame_reasons.append({"code": "no_infrastructure_candidate_at_target"})

    candidates: list[dict[str, object]] = []
    eligible: list[tuple[int, str, float, int, str, dict[str, object]]] = []
    for ego_agent_id in av_ids:
        ego_numeric_id = metadata[ego_agent_id]["numeric_id"]
        for infrastructure_agent_id in infrastructure_ids:
            infrastructure_numeric_id = metadata[infrastructure_agent_id]["numeric_id"]
            distance, coverage, reasons = _pair_eligibility(
                frames=frames,
                target_timestamp_us=target_timestamp_us,
                ego_agent_id=ego_agent_id,
                ego_numeric_id=ego_numeric_id,
                infrastructure_agent_id=infrastructure_agent_id,
                infrastructure_numeric_id=infrastructure_numeric_id,
                delays_ms=delays_ms,
                history_limit=history_limit,
                frame_interval_us=frame_interval_us,
            )
            candidate = {
                "ego_agent_id": ego_agent_id,
                "ego_numeric_id": ego_numeric_id,
                "infrastructure_agent_id": infrastructure_agent_id,
                "infrastructure_numeric_id": infrastructure_numeric_id,
                "distance_m": distance,
                "eligible": not reasons,
                "reasons": reasons,
            }
            candidates.append(candidate)
            if not reasons:
                assert distance is not None and coverage is not None
                eligible.append(
                    (
                        ego_numeric_id,
                        ego_agent_id,
                        distance,
                        infrastructure_numeric_id,
                        infrastructure_agent_id,
                        coverage,
                    )
                )
    candidates.sort(
        key=lambda item: (
            item["ego_numeric_id"],
            item["ego_agent_id"],
            item["infrastructure_numeric_id"],
            item["infrastructure_agent_id"],
        )
    )
    audit = {
        "timestamp_us": target_timestamp_us,
        "ignored_av_agent_ids": ignored_av_ids,
        "frame_reasons": frame_reasons,
        "candidate_pairs": candidates,
    }
    if not eligible:
        return audit, None
    chosen_ego_numeric_id, chosen_ego_agent_id = min(
        (item[0], item[1]) for item in eligible
    )
    eligible_for_ego = [
        item
        for item in eligible
        if item[0] == chosen_ego_numeric_id and item[1] == chosen_ego_agent_id
    ]
    selected = min(
        eligible_for_ego,
        key=lambda item: (item[2] * item[2], item[3], item[4]),
    )
    return audit, {
        "ego_numeric_id": selected[0],
        "ego_agent_id": selected[1],
        "distance_m": selected[2],
        "infrastructure_numeric_id": selected[3],
        "infrastructure_agent_id": selected[4],
        "endpoint_coverage": selected[5],
    }


def _parse_sequence_frames(
    *,
    scene_id: str,
    frame_values: object,
    root: Path,
    payload_cache: dict[tuple[str, int, str], str | None],
) -> tuple[dict[int, dict[str, object]], int]:
    if not isinstance(frame_values, list):
        raise V2XSetPairError("scene frames must be an array")
    frames: dict[int, dict[str, object]] = {}
    total_agents = 0
    for frame_index, frame_value in enumerate(frame_values):
        frame = _expect_fields(
            frame_value,
            frozenset({"timestamp_us", "agents"}),
            f"scene {scene_id} frame {frame_index}",
        )
        timestamp_us = _exact_int(frame["timestamp_us"], "timestamp_us")
        if timestamp_us in frames:
            raise V2XSetPairError("frame timestamps must be unique per scene")
        agent_values = frame["agents"]
        if not isinstance(agent_values, list):
            raise V2XSetPairError("frame agents must be an array")
        total_agents += len(agent_values)
        normalized: dict[str, dict[str, object]] = {}
        invalid: dict[str, list[dict[str, object]]] = {}
        metadata: dict[str, dict[str, object]] = {}
        numeric_keys: set[tuple[str, int]] = set()
        for agent_value in agent_values:
            agent_id, numeric_id, agent_type, agent, reasons = _normalize_agent(
                agent_value,
                root,
                payload_cache,
            )
            if agent_id in metadata:
                raise V2XSetPairError("agent IDs must be unique per frame")
            numeric_key = (agent_type, numeric_id)
            if numeric_key in numeric_keys:
                raise V2XSetPairError(
                    "numeric IDs must be unique within each agent type and frame"
                )
            numeric_keys.add(numeric_key)
            metadata[agent_id] = {
                "numeric_id": numeric_id,
                "agent_type": agent_type,
            }
            if agent is None:
                invalid[agent_id] = reasons
            else:
                normalized[agent_id] = agent
        frames[timestamp_us] = {
            "timestamp_us": timestamp_us,
            "normalized": normalized,
            "invalid": invalid,
            "metadata": metadata,
        }
    return dict(sorted(frames.items())), total_agents


def _build_target_manifest(pairs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    sample_ids = [str(pair["target_sample_id"]) for pair in pairs]
    split_counts = {
        split: sum(pair["split"] == split for pair in pairs) for split in SPLITS
    }
    value: dict[str, object] = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "artifact_type": TARGET_TYPE,
        "target_generation_stage": "before_latency_selection_and_fault_injection",
        "sample_count": len(sample_ids),
        "split_counts": split_counts,
        "sample_ids": sample_ids,
        "sample_ids_sha256": _id_list_sha256(
            b"v2xset-pair-target-sample-ids-v1",
            sample_ids,
        ),
    }
    value["content_sha256"] = content_sha256(value)
    return value


def _condition_target_references(
    delays_ms: Sequence[int],
    target_manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    main_references = [
        {
            "condition_id": f"{condition}@{delay_ms}ms",
            "condition": condition,
            "delay_ms": delay_ms,
            "target_manifest_content_sha256": target_manifest["content_sha256"],
            "target_sample_count": target_manifest["sample_count"],
        }
        for delay_ms in delays_ms
        for condition in MAIN_CONDITION_NAMES
    ]
    diagnostic_references = [
        {
            "condition_id": f"{condition}@0ms",
            "condition": condition,
            "delay_ms": 0,
            "target_manifest_content_sha256": target_manifest["content_sha256"],
            "target_sample_count": target_manifest["sample_count"],
        }
        for condition in DIAGNOSTIC_CONDITION_NAMES
    ]
    return main_references + diagnostic_references


def prepare_pair_manifest(
    inventory_path: Path,
    data_root: Path,
) -> dict[str, object]:
    """Derive the deterministic sequence-frozen V2XSet-Pair manifest."""

    inventory, inventory_raw = _load_inventory(Path(inventory_path))
    try:
        root = Path(data_root).resolve(strict=True)
    except OSError as error:
        raise V2XSetPairError("data_root does not exist") from error
    if not root.is_dir():
        raise V2XSetPairError("data_root must be a directory")
    delays_ms, history_limit, frame_interval_us = _protocol_values(inventory)
    scenes_value = inventory["scenes"]
    assert isinstance(scenes_value, list)
    payload_cache: dict[tuple[str, int, str], str | None] = {}
    seen_scene_ids: set[str] = set()
    total_frames = 0
    total_agents = 0
    split_scene_counts = {split: 0 for split in SPLITS}
    sequence_records: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    excluded_frames: list[dict[str, object]] = []

    parsed_scenes: list[tuple[str, str, str | None, dict[int, dict[str, object]]]] = []
    for scene_index, scene_value in enumerate(scenes_value):
        scene = _expect_fields(
            scene_value,
            frozenset(
                {
                    "scene_id",
                    "split",
                    "standard_fixed_ego_agent_id",
                    "frames",
                }
            ),
            f"scenes[{scene_index}]",
        )
        scene_id = _identifier(scene["scene_id"], "scene_id")
        if scene_id in seen_scene_ids:
            raise V2XSetPairError("scene IDs must be unique")
        seen_scene_ids.add(scene_id)
        split = scene["split"]
        if split not in SPLITS:
            raise V2XSetPairError("scene split must be train or validation")
        fixed_ego = scene["standard_fixed_ego_agent_id"]
        if split == "validation":
            fixed_ego = _identifier(
                fixed_ego,
                "validation standard_fixed_ego_agent_id",
            )
        elif fixed_ego is not None:
            raise V2XSetPairError(
                "train scene standard_fixed_ego_agent_id must be null"
            )
        frames, agent_count = _parse_sequence_frames(
            scene_id=scene_id,
            frame_values=scene["frames"],
            root=root,
            payload_cache=payload_cache,
        )
        total_frames += len(frames)
        total_agents += agent_count
        split_scene_counts[split] += 1
        parsed_scenes.append((scene_id, split, fixed_ego, frames))

    for scene_id, split, fixed_ego, frames in sorted(parsed_scenes):
        attempts: list[dict[str, object]] = []
        anchor_timestamp_us: int | None = None
        selection: dict[str, object] | None = None
        for timestamp_us in frames:
            audit, candidate = _anchor_selection(
                split=split,
                standard_fixed_ego_agent_id=fixed_ego,
                target_timestamp_us=timestamp_us,
                frames=frames,
                delays_ms=delays_ms,
                history_limit=history_limit,
                frame_interval_us=frame_interval_us,
            )
            attempts.append(audit)
            if candidate is not None:
                anchor_timestamp_us = timestamp_us
                selection = candidate
                break

        sequence_included_ids: list[str] = []
        sequence_excluded_ids: list[str] = []
        if selection is None or anchor_timestamp_us is None:
            for timestamp_us, frame in frames.items():
                frame_identifier = _frame_id(scene_id, timestamp_us)
                sequence_excluded_ids.append(frame_identifier)
                attempt = next(
                    item for item in attempts if item["timestamp_us"] == timestamp_us
                )
                excluded_frames.append(
                    {
                        "frame_id": frame_identifier,
                        "scene_id": scene_id,
                        "split": split,
                        "timestamp_us": timestamp_us,
                        "ego_agent_id": fixed_ego,
                        "infrastructure_agent_id": None,
                        "excluded_agent_ids": _ordered_agent_ids(frame),
                        "reasons": [
                            {"code": "sequence_has_no_eligible_anchor"},
                            {
                                "code": "no_eligible_pair_at_target",
                                "selection_audit": attempt,
                            },
                        ],
                    }
                )
            sequence_records.append(
                {
                    "scene_id": scene_id,
                    "split": split,
                    "status": "excluded_no_anchor",
                    "standard_fixed_ego_agent_id": fixed_ego,
                    "anchor_timestamp_us": None,
                    "ego_agent_id": None,
                    "ego_numeric_id": None,
                    "infrastructure_agent_id": None,
                    "infrastructure_numeric_id": None,
                    "anchor_distance_m": None,
                    "anchor_attempts": attempts,
                    "included_target_sample_ids": [],
                    "excluded_frame_ids": sequence_excluded_ids,
                }
            )
            continue

        ego_agent_id = str(selection["ego_agent_id"])
        ego_numeric_id = int(selection["ego_numeric_id"])
        infrastructure_agent_id = str(selection["infrastructure_agent_id"])
        infrastructure_numeric_id = int(selection["infrastructure_numeric_id"])
        anchor_distance = float(selection["distance_m"])
        for timestamp_us, frame in frames.items():
            current_agent_ids = _ordered_agent_ids(frame)
            excluded_agent_ids = [
                agent_id
                for agent_id in current_agent_ids
                if agent_id not in {ego_agent_id, infrastructure_agent_id}
            ]
            if timestamp_us < anchor_timestamp_us:
                frame_identifier = _frame_id(scene_id, timestamp_us)
                sequence_excluded_ids.append(frame_identifier)
                attempt = next(
                    item for item in attempts if item["timestamp_us"] == timestamp_us
                )
                excluded_frames.append(
                    {
                        "frame_id": frame_identifier,
                        "scene_id": scene_id,
                        "split": split,
                        "timestamp_us": timestamp_us,
                        "ego_agent_id": ego_agent_id,
                        "infrastructure_agent_id": infrastructure_agent_id,
                        "excluded_agent_ids": excluded_agent_ids,
                        "reasons": [
                            {
                                "code": "before_sequence_anchor",
                                "anchor_timestamp_us": anchor_timestamp_us,
                            },
                            {
                                "code": "no_eligible_pair_at_target",
                                "selection_audit": attempt,
                            },
                        ],
                    }
                )
                continue
            distance, coverage, reasons = _pair_eligibility(
                frames=frames,
                target_timestamp_us=timestamp_us,
                ego_agent_id=ego_agent_id,
                ego_numeric_id=ego_numeric_id,
                infrastructure_agent_id=infrastructure_agent_id,
                infrastructure_numeric_id=infrastructure_numeric_id,
                delays_ms=delays_ms,
                history_limit=history_limit,
                frame_interval_us=frame_interval_us,
            )
            if reasons:
                frame_identifier = _frame_id(scene_id, timestamp_us)
                sequence_excluded_ids.append(frame_identifier)
                excluded_frames.append(
                    {
                        "frame_id": frame_identifier,
                        "scene_id": scene_id,
                        "split": split,
                        "timestamp_us": timestamp_us,
                        "ego_agent_id": ego_agent_id,
                        "infrastructure_agent_id": infrastructure_agent_id,
                        "excluded_agent_ids": excluded_agent_ids,
                        "reasons": [
                            {
                                "code": "frozen_pair_ineligible",
                                "substitution_forbidden": True,
                                "details": reasons,
                            }
                        ],
                    }
                )
                continue
            assert distance is not None and coverage is not None
            normalized = frame["normalized"]
            assert isinstance(normalized, Mapping)
            ego = normalized[ego_agent_id]
            infrastructure = normalized[infrastructure_agent_id]
            target_key = {
                "scene_id": scene_id,
                "split": split,
                "timestamp_us": timestamp_us,
                "ego_agent_id": ego_agent_id,
                "infrastructure_agent_id": infrastructure_agent_id,
            }
            target_id = _target_sample_id(target_key)
            pair_key = {
                "scene_id": scene_id,
                "timestamp_us": timestamp_us,
                "ego_agent_id": ego_agent_id,
                "infrastructure_agent_id": infrastructure_agent_id,
            }
            sequence_included_ids.append(target_id)
            pairs.append(
                {
                    "pair_id": _sequence_pair_id(pair_key),
                    "target_sample_id": target_id,
                    "scene_id": scene_id,
                    "split": split,
                    "timestamp_us": timestamp_us,
                    "anchor_timestamp_us": anchor_timestamp_us,
                    "ego_agent_id": ego_agent_id,
                    "ego_numeric_id": ego_numeric_id,
                    "infrastructure_agent_id": infrastructure_agent_id,
                    "infrastructure_numeric_id": infrastructure_numeric_id,
                    "anchor_distance_m": anchor_distance,
                    "infrastructure_distance_m": distance,
                    "target_eligibility_agent_ids": [
                        ego_agent_id,
                        infrastructure_agent_id,
                    ],
                    "excluded_agent_ids": excluded_agent_ids,
                    "endpoint_coverage": coverage,
                    "agents": [
                        _role_agent(ego, "ego"),
                        _role_agent(infrastructure, "infrastructure"),
                    ],
                }
            )
        sequence_records.append(
            {
                "scene_id": scene_id,
                "split": split,
                "status": "included",
                "standard_fixed_ego_agent_id": fixed_ego,
                "anchor_timestamp_us": anchor_timestamp_us,
                "ego_agent_id": ego_agent_id,
                "ego_numeric_id": ego_numeric_id,
                "infrastructure_agent_id": infrastructure_agent_id,
                "infrastructure_numeric_id": infrastructure_numeric_id,
                "anchor_distance_m": anchor_distance,
                "anchor_attempts": attempts,
                "included_target_sample_ids": sequence_included_ids,
                "excluded_frame_ids": sequence_excluded_ids,
            }
        )

    pairs.sort(key=lambda item: (item["scene_id"], item["timestamp_us"]))
    excluded_frames.sort(key=lambda item: (item["scene_id"], item["timestamp_us"]))
    included_sequence_ids = [
        item["scene_id"] for item in sequence_records if item["status"] == "included"
    ]
    excluded_sequence_ids = [
        item["scene_id"]
        for item in sequence_records
        if item["status"] == "excluded_no_anchor"
    ]
    target_manifest = _build_target_manifest(pairs)
    derivation_ids = {
        "included_sequence_ids": _id_inventory(
            b"v2xset-pair-included-sequences-v1", included_sequence_ids
        ),
        "excluded_sequence_ids": _id_inventory(
            b"v2xset-pair-excluded-sequences-v1", excluded_sequence_ids
        ),
        "included_target_sample_ids": _id_inventory(
            b"v2xset-pair-included-targets-v1",
            target_manifest["sample_ids"],
        ),
        "excluded_frame_ids": _id_inventory(
            b"v2xset-pair-excluded-frames-v1",
            [item["frame_id"] for item in excluded_frames],
        ),
    }
    payload = {
        "protocol_scope": "paper_pair_sequence_offline_adapter_only",
        "protocol_disclaimer": DISCLAIMER,
        "selection_parameters": _sequence_selection_parameters(
            delays_ms,
            history_limit,
            frame_interval_us,
        ),
        "input_summary": {
            "inventory_file_sha256": hashlib.sha256(inventory_raw).hexdigest(),
            "inventory_content_sha256": inventory["content_sha256"],
            "dataset_name": DATASET_NAME,
            "scene_count": len(parsed_scenes),
            "train_scene_count": split_scene_counts["train"],
            "validation_scene_count": split_scene_counts["validation"],
            "frame_count": total_frames,
            "agent_record_count": total_agents,
        },
        "sequence_count": len(sequence_records),
        "included_sequence_count": len(included_sequence_ids),
        "excluded_sequence_count": len(excluded_sequence_ids),
        "pair_count": len(pairs),
        "excluded_frame_count": len(excluded_frames),
        "derivation_ids": derivation_ids,
        "sequences": sequence_records,
        "pairs": pairs,
        "excluded_frames": excluded_frames,
        "target_manifest": target_manifest,
        "condition_target_references": _condition_target_references(
            delays_ms,
            target_manifest,
        ),
    }
    manifest: dict[str, object] = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "artifact_type": PAIR_TYPE,
        **payload,
    }
    manifest["content_sha256"] = content_sha256(manifest)
    _validate_pair_manifest(manifest)
    return manifest


def _validate_sequence_output_agent(value: object, expected_role: str) -> str:
    agent = _expect_fields(
        value,
        frozenset(
            {
                "agent_id",
                "numeric_id",
                "agent_type",
                "role",
                "world_from_agent",
                "lidar",
                "cameras",
            }
        ),
        "Pair agent",
    )
    agent_id = _identifier(agent["agent_id"], "Pair agent_id")
    _exact_int(agent["numeric_id"], "Pair numeric_id")
    expected_type = "av" if expected_role == "ego" else "infrastructure"
    if agent["role"] != expected_role or agent["agent_type"] != expected_type:
        raise V2XSetPairError("Pair agent role/type mismatch")
    if _rigid_transform(agent["world_from_agent"]) is None:
        raise V2XSetPairError("Pair agent pose is invalid")
    lidar = _expect_fields(
        agent["lidar"],
        frozenset({"payload", "agent_from_lidar"}),
        "Pair LiDAR",
    )
    if _artifact_identity(lidar["payload"]) is None:
        raise V2XSetPairError("Pair LiDAR payload identity is invalid")
    if _rigid_transform(lidar["agent_from_lidar"]) is None:
        raise V2XSetPairError("Pair LiDAR extrinsic is invalid")
    cameras = agent["cameras"]
    if not isinstance(cameras, list) or len(cameras) != len(CAMERA_IDS):
        raise V2XSetPairError("Pair agent must contain exactly four cameras")
    observed_camera_ids: list[str] = []
    for camera in cameras:
        record = _expect_fields(
            camera,
            frozenset({"camera_id", "image", "intrinsic", "agent_from_camera"}),
            "Pair camera",
        )
        camera_id = record["camera_id"]
        if type(camera_id) is not str:
            raise V2XSetPairError("Pair camera_id is invalid")
        observed_camera_ids.append(camera_id)
        if _artifact_identity(record["image"]) is None:
            raise V2XSetPairError("Pair camera payload identity is invalid")
        if _intrinsic(record["intrinsic"]) is None:
            raise V2XSetPairError("Pair camera intrinsic is invalid")
        if _rigid_transform(record["agent_from_camera"]) is None:
            raise V2XSetPairError("Pair camera extrinsic is invalid")
    if observed_camera_ids != list(CAMERA_IDS):
        raise V2XSetPairError("Pair cameras are not in required slot order")
    return agent_id


def _validate_endpoint_coverage(
    coverage: object,
    *,
    target_timestamp_us: int,
    ego_agent_id: str,
    infrastructure_agent_id: str,
    delays_ms: Sequence[int],
    history_limit: int,
    frame_interval_us: int,
) -> None:
    value = _expect_fields(
        coverage,
        frozenset({"ego_history", "infrastructure_histories"}),
        "endpoint_coverage",
    )
    ego_history = value["ego_history"]
    if not isinstance(ego_history, list) or len(ego_history) != history_limit + 1:
        raise V2XSetPairError("ego endpoint coverage length mismatch")
    for horizon, endpoint in enumerate(ego_history):
        record = _expect_fields(
            endpoint,
            frozenset({"horizon", "timestamp_us", "agent"}),
            "ego endpoint",
        )
        if record["horizon"] != horizon or record["timestamp_us"] != (
            target_timestamp_us - horizon * frame_interval_us
        ):
            raise V2XSetPairError("ego endpoint timestamp mismatch")
        if _validate_sequence_output_agent(record["agent"], "ego") != ego_agent_id:
            raise V2XSetPairError("ego endpoint identity mismatch")
    infrastructure_histories = value["infrastructure_histories"]
    if not isinstance(infrastructure_histories, list) or len(
        infrastructure_histories
    ) != len(delays_ms):
        raise V2XSetPairError("infrastructure delay coverage mismatch")
    for delay_ms, item in zip(delays_ms, infrastructure_histories, strict=True):
        delay_record = _expect_fields(
            item,
            frozenset({"delay_ms", "history"}),
            "infrastructure delay coverage",
        )
        if delay_record["delay_ms"] != delay_ms:
            raise V2XSetPairError("infrastructure delay order mismatch")
        history = delay_record["history"]
        if not isinstance(history, list) or len(history) != history_limit + 1:
            raise V2XSetPairError("infrastructure history length mismatch")
        for horizon, endpoint in enumerate(history):
            record = _expect_fields(
                endpoint,
                frozenset({"delay_ms", "horizon", "timestamp_us", "agent"}),
                "infrastructure endpoint",
            )
            expected_timestamp = (
                target_timestamp_us - delay_ms * 1000 - horizon * frame_interval_us
            )
            if (
                record["delay_ms"] != delay_ms
                or record["horizon"] != horizon
                or record["timestamp_us"] != expected_timestamp
            ):
                raise V2XSetPairError("infrastructure endpoint timestamp mismatch")
            if (
                _validate_sequence_output_agent(record["agent"], "infrastructure")
                != infrastructure_agent_id
            ):
                raise V2XSetPairError("infrastructure endpoint identity mismatch")


def _validate_id_inventory(
    value: object,
    *,
    domain: bytes,
    expected_ids: Sequence[str],
    context: str,
) -> None:
    record = _expect_fields(
        value,
        frozenset({"count", "ids", "ids_sha256"}),
        context,
    )
    if record["count"] != len(expected_ids) or record["ids"] != list(expected_ids):
        raise V2XSetPairError(f"{context} count or IDs mismatch")
    if record["ids_sha256"] != _id_list_sha256(domain, expected_ids):
        raise V2XSetPairError(f"{context} hash mismatch")


def _validate_anchor_attempt(
    value: object,
    *,
    split: str,
    fixed_ego_agent_id: str | None,
) -> tuple[int, list[dict[str, object]]]:
    attempt = _expect_fields(
        value,
        frozenset(
            {
                "timestamp_us",
                "ignored_av_agent_ids",
                "frame_reasons",
                "candidate_pairs",
            }
        ),
        "anchor attempt",
    )
    timestamp_us = _exact_int(attempt["timestamp_us"], "anchor attempt timestamp")
    ignored = attempt["ignored_av_agent_ids"]
    if not isinstance(ignored, list) or len(ignored) != len(set(ignored)):
        raise V2XSetPairError("anchor ignored AV IDs are invalid")
    for agent_id in ignored:
        _identifier(agent_id, "anchor ignored AV agent_id")
        if split == "validation" and agent_id == fixed_ego_agent_id:
            raise V2XSetPairError("fixed validation Ego appears in ignored AV IDs")
    frame_reasons = attempt["frame_reasons"]
    if not isinstance(frame_reasons, list):
        raise V2XSetPairError("anchor frame_reasons must be an array")
    for reason in frame_reasons:
        if not isinstance(reason, Mapping) or type(reason.get("code")) is not str:
            raise V2XSetPairError("anchor attempt contains an invalid frame reason")
    raw_candidates = attempt["candidate_pairs"]
    if not isinstance(raw_candidates, list):
        raise V2XSetPairError("anchor candidate_pairs must be an array")
    candidates: list[dict[str, object]] = []
    candidate_keys: list[tuple[int, str, int, str]] = []
    for raw_candidate in raw_candidates:
        candidate = _expect_fields(
            raw_candidate,
            frozenset(
                {
                    "ego_agent_id",
                    "ego_numeric_id",
                    "infrastructure_agent_id",
                    "infrastructure_numeric_id",
                    "distance_m",
                    "eligible",
                    "reasons",
                }
            ),
            "anchor candidate",
        )
        ego_agent_id = _identifier(candidate["ego_agent_id"], "anchor candidate Ego")
        infrastructure_agent_id = _identifier(
            candidate["infrastructure_agent_id"],
            "anchor candidate infrastructure",
        )
        ego_numeric_id = _exact_int(
            candidate["ego_numeric_id"], "anchor candidate Ego numeric ID"
        )
        infrastructure_numeric_id = _exact_int(
            candidate["infrastructure_numeric_id"],
            "anchor candidate infrastructure numeric ID",
        )
        if split == "validation" and ego_agent_id != fixed_ego_agent_id:
            raise V2XSetPairError("validation anchor considered a substitute Ego")
        eligible = candidate["eligible"]
        reasons = candidate["reasons"]
        if type(eligible) is not bool or not isinstance(reasons, list):
            raise V2XSetPairError("anchor candidate eligibility is invalid")
        for reason in reasons:
            if not isinstance(reason, Mapping) or type(reason.get("code")) is not str:
                raise V2XSetPairError("anchor candidate contains an invalid reason")
        distance = candidate["distance_m"]
        if distance is not None and (
            type(distance) not in (int, float)
            or not math.isfinite(float(distance))
            or float(distance) < 0
        ):
            raise V2XSetPairError("anchor candidate distance is invalid")
        if eligible:
            if (
                reasons
                or distance is None
                or float(distance) > MAX_COLLABORATOR_DISTANCE_M
            ):
                raise V2XSetPairError("eligible anchor candidate has invalid evidence")
        elif not reasons and not frame_reasons:
            raise V2XSetPairError("ineligible anchor candidate has no reason")
        candidate_keys.append(
            (
                ego_numeric_id,
                ego_agent_id,
                infrastructure_numeric_id,
                infrastructure_agent_id,
            )
        )
        candidates.append(candidate)
    if candidate_keys != sorted(candidate_keys) or len(candidate_keys) != len(
        set(candidate_keys)
    ):
        raise V2XSetPairError("anchor candidates are duplicated or unordered")
    return timestamp_us, candidates


def _validate_pair_manifest(manifest: Mapping[str, object]) -> None:
    value = _expect_fields(
        manifest,
        frozenset(
            {
                "schema_version",
                "artifact_type",
                "protocol_scope",
                "protocol_disclaimer",
                "selection_parameters",
                "input_summary",
                "sequence_count",
                "included_sequence_count",
                "excluded_sequence_count",
                "pair_count",
                "excluded_frame_count",
                "derivation_ids",
                "sequences",
                "pairs",
                "excluded_frames",
                "target_manifest",
                "condition_target_references",
                "content_sha256",
            }
        ),
        "Pair manifest",
    )
    if (
        value["schema_version"] != PAIR_SCHEMA_VERSION
        or value["artifact_type"] != PAIR_TYPE
    ):
        raise V2XSetPairError("Pair manifest schema/type mismatch")
    if value["protocol_scope"] != "paper_pair_sequence_offline_adapter_only":
        raise V2XSetPairError("Pair manifest protocol_scope mismatch")
    if value["protocol_disclaimer"] != DISCLAIMER:
        raise V2XSetPairError("Pair manifest protocol disclaimer mismatch")
    parameters = value["selection_parameters"]
    if not isinstance(parameters, Mapping):
        raise V2XSetPairError("selection_parameters must be an object")
    delays_ms = parameters.get("configured_delays_ms")
    history_limit = parameters.get("history_limit")
    frame_interval_us = parameters.get("frame_interval_us")
    if not isinstance(delays_ms, list):
        raise V2XSetPairError("selection delay grid is invalid")
    validated_delays = [_exact_int(item, "selection delay") for item in delays_ms]
    validated_history = _exact_int(history_limit, "selection history_limit")
    validated_interval = _exact_int(
        frame_interval_us,
        "selection frame_interval_us",
        minimum=1,
    )
    if (
        not validated_delays
        or validated_delays != sorted(set(validated_delays))
        or validated_delays[0] != 0
        or any((delay_ms * 1000) % validated_interval for delay_ms in validated_delays)
    ):
        raise V2XSetPairError("selection delay grid is invalid")
    if parameters != _sequence_selection_parameters(
        validated_delays,
        validated_history,
        validated_interval,
    ):
        raise V2XSetPairError("Pair manifest selection parameters mismatch")
    summary = _expect_fields(
        value["input_summary"],
        frozenset(
            {
                "inventory_file_sha256",
                "inventory_content_sha256",
                "dataset_name",
                "scene_count",
                "train_scene_count",
                "validation_scene_count",
                "frame_count",
                "agent_record_count",
            }
        ),
        "Pair input_summary",
    )
    if summary["dataset_name"] != DATASET_NAME:
        raise V2XSetPairError("Pair input dataset_name mismatch")
    for field in ("inventory_file_sha256", "inventory_content_sha256"):
        if type(summary[field]) is not str or SHA256.fullmatch(summary[field]) is None:
            raise V2XSetPairError(f"Pair input {field} is invalid")
    for field in (
        "scene_count",
        "train_scene_count",
        "validation_scene_count",
        "frame_count",
        "agent_record_count",
    ):
        _exact_int(summary[field], f"Pair input {field}")
    if (
        summary["train_scene_count"] + summary["validation_scene_count"]
        != summary["scene_count"]
    ):
        raise V2XSetPairError("Pair input split scene counts mismatch")

    sequences = value["sequences"]
    pairs = value["pairs"]
    exclusions = value["excluded_frames"]
    if (
        not isinstance(sequences, list)
        or not isinstance(pairs, list)
        or not isinstance(exclusions, list)
    ):
        raise V2XSetPairError("sequence, pair, and exclusion records must be arrays")
    if value["sequence_count"] != len(sequences):
        raise V2XSetPairError("sequence count mismatch")
    if value["pair_count"] != len(pairs) or value["excluded_frame_count"] != len(
        exclusions
    ):
        raise V2XSetPairError("pair or exclusion count mismatch")
    sequence_by_id: dict[str, Mapping[str, object]] = {}
    included_sequence_ids: list[str] = []
    excluded_sequence_ids: list[str] = []
    for sequence in sequences:
        sequence = _expect_fields(
            sequence,
            frozenset(
                {
                    "scene_id",
                    "split",
                    "status",
                    "standard_fixed_ego_agent_id",
                    "anchor_timestamp_us",
                    "ego_agent_id",
                    "ego_numeric_id",
                    "infrastructure_agent_id",
                    "infrastructure_numeric_id",
                    "anchor_distance_m",
                    "anchor_attempts",
                    "included_target_sample_ids",
                    "excluded_frame_ids",
                }
            ),
            "sequence record",
        )
        scene_id = _identifier(sequence["scene_id"], "sequence scene_id")
        if scene_id in sequence_by_id:
            raise V2XSetPairError("duplicate sequence record")
        split = sequence["split"]
        status = sequence["status"]
        if split not in SPLITS or status not in {"included", "excluded_no_anchor"}:
            raise V2XSetPairError("invalid sequence split/status")
        if split == "validation":
            _identifier(
                sequence["standard_fixed_ego_agent_id"],
                "sequence fixed validation ego",
            )
        elif sequence["standard_fixed_ego_agent_id"] is not None:
            raise V2XSetPairError("train sequence cannot have a fixed Standard ego")
        if not isinstance(sequence["anchor_attempts"], list):
            raise V2XSetPairError("sequence anchor_attempts must be an array")
        if not isinstance(
            sequence["included_target_sample_ids"], list
        ) or not isinstance(sequence["excluded_frame_ids"], list):
            raise V2XSetPairError("sequence derived ID lists are invalid")
        attempt_records = [
            _validate_anchor_attempt(
                attempt,
                split=split,
                fixed_ego_agent_id=sequence["standard_fixed_ego_agent_id"],
            )
            for attempt in sequence["anchor_attempts"]
        ]
        attempt_timestamps = [item[0] for item in attempt_records]
        if attempt_timestamps != sorted(set(attempt_timestamps)):
            raise V2XSetPairError("sequence anchor attempts are unordered")
        if status == "included":
            included_sequence_ids.append(scene_id)
            anchor = _exact_int(sequence["anchor_timestamp_us"], "sequence anchor")
            ego_id = _identifier(sequence["ego_agent_id"], "sequence ego")
            infrastructure_id = _identifier(
                sequence["infrastructure_agent_id"],
                "sequence infrastructure",
            )
            ego_numeric_id = _exact_int(
                sequence["ego_numeric_id"], "sequence ego numeric ID"
            )
            infrastructure_numeric_id = _exact_int(
                sequence["infrastructure_numeric_id"],
                "sequence infrastructure numeric ID",
            )
            if (
                split == "validation"
                and ego_id != sequence["standard_fixed_ego_agent_id"]
            ):
                raise V2XSetPairError("validation fixed Ego was substituted")
            distance = sequence["anchor_distance_m"]
            if (
                type(distance) not in (int, float)
                or not 0 <= float(distance) <= MAX_COLLABORATOR_DISTANCE_M
            ):
                raise V2XSetPairError("sequence anchor distance is invalid")
            if not attempt_records or attempt_records[-1][0] != anchor:
                raise V2XSetPairError("sequence anchor is not the final anchor attempt")
            if any(
                candidate["eligible"]
                for _, candidates in attempt_records[:-1]
                for candidate in candidates
            ):
                raise V2XSetPairError("sequence anchor is not earliest eligible target")
            eligible_candidates = [
                candidate
                for candidate in attempt_records[-1][1]
                if candidate["eligible"]
            ]
            if not eligible_candidates:
                raise V2XSetPairError("included sequence anchor has no eligible pair")
            selected_ego_key = min(
                (candidate["ego_numeric_id"], candidate["ego_agent_id"])
                for candidate in eligible_candidates
            )
            eligible_for_ego = [
                candidate
                for candidate in eligible_candidates
                if (
                    candidate["ego_numeric_id"],
                    candidate["ego_agent_id"],
                )
                == selected_ego_key
            ]
            selected_candidate = min(
                eligible_for_ego,
                key=lambda candidate: (
                    float(candidate["distance_m"]) ** 2,
                    candidate["infrastructure_numeric_id"],
                    candidate["infrastructure_agent_id"],
                ),
            )
            if (
                selected_candidate["ego_agent_id"] != ego_id
                or selected_candidate["ego_numeric_id"] != ego_numeric_id
                or selected_candidate["infrastructure_agent_id"] != infrastructure_id
                or selected_candidate["infrastructure_numeric_id"]
                != infrastructure_numeric_id
                or abs(float(selected_candidate["distance_m"]) - float(distance)) > 1e-9
            ):
                raise V2XSetPairError(
                    "frozen pair does not match anchor selection rule"
                )
            if not sequence["included_target_sample_ids"]:
                raise V2XSetPairError("included sequence has no included target")
        else:
            excluded_sequence_ids.append(scene_id)
            for field in (
                "anchor_timestamp_us",
                "ego_agent_id",
                "ego_numeric_id",
                "infrastructure_agent_id",
                "infrastructure_numeric_id",
                "anchor_distance_m",
            ):
                if sequence[field] is not None:
                    raise V2XSetPairError("excluded sequence has frozen pair fields")
            if sequence["included_target_sample_ids"]:
                raise V2XSetPairError("excluded sequence contains target sample IDs")
            if any(
                candidate["eligible"]
                for _, candidates in attempt_records
                for candidate in candidates
            ):
                raise V2XSetPairError("no-anchor sequence contains an eligible pair")
        sequence_by_id[scene_id] = sequence
    if list(sequence_by_id) != sorted(sequence_by_id):
        raise V2XSetPairError("sequence records must be ordered by scene ID")
    if value["included_sequence_count"] != len(included_sequence_ids) or value[
        "excluded_sequence_count"
    ] != len(excluded_sequence_ids):
        raise V2XSetPairError("included/excluded sequence count mismatch")

    pair_keys: list[tuple[str, int]] = []
    target_ids: list[str] = []
    pair_ids: set[str] = set()
    target_ids_by_scene: dict[str, list[str]] = {
        scene_id: [] for scene_id in sequence_by_id
    }
    for pair in pairs:
        record = _expect_fields(
            pair,
            frozenset(
                {
                    "pair_id",
                    "target_sample_id",
                    "scene_id",
                    "split",
                    "timestamp_us",
                    "anchor_timestamp_us",
                    "ego_agent_id",
                    "ego_numeric_id",
                    "infrastructure_agent_id",
                    "infrastructure_numeric_id",
                    "anchor_distance_m",
                    "infrastructure_distance_m",
                    "target_eligibility_agent_ids",
                    "excluded_agent_ids",
                    "endpoint_coverage",
                    "agents",
                }
            ),
            "Pair record",
        )
        scene_id = _identifier(record["scene_id"], "Pair scene_id")
        timestamp_us = _exact_int(record["timestamp_us"], "Pair timestamp_us")
        sequence = sequence_by_id.get(scene_id)
        if sequence is None or sequence["status"] != "included":
            raise V2XSetPairError("Pair references a non-included sequence")
        if (
            record["split"] != sequence["split"]
            or record["anchor_timestamp_us"] != sequence["anchor_timestamp_us"]
            or timestamp_us < record["anchor_timestamp_us"]
        ):
            raise V2XSetPairError("Pair split/anchor mismatch")
        ego_id = _identifier(record["ego_agent_id"], "Pair ego_agent_id")
        infrastructure_id = _identifier(
            record["infrastructure_agent_id"],
            "Pair infrastructure_agent_id",
        )
        if (
            ego_id != sequence["ego_agent_id"]
            or infrastructure_id != sequence["infrastructure_agent_id"]
        ):
            raise V2XSetPairError("frozen Pair identity changed inside sequence")
        if (
            record["ego_numeric_id"] != sequence["ego_numeric_id"]
            or record["infrastructure_numeric_id"]
            != sequence["infrastructure_numeric_id"]
        ):
            raise V2XSetPairError("frozen Pair numeric identity changed")
        if (
            abs(
                float(record["anchor_distance_m"])
                - float(sequence["anchor_distance_m"])
            )
            > 1e-9
        ):
            raise V2XSetPairError("Pair anchor distance changed inside sequence")
        pair_key = {
            "scene_id": scene_id,
            "timestamp_us": timestamp_us,
            "ego_agent_id": ego_id,
            "infrastructure_agent_id": infrastructure_id,
        }
        expected_pair_id = _sequence_pair_id(pair_key)
        if record["pair_id"] != expected_pair_id or expected_pair_id in pair_ids:
            raise V2XSetPairError("Pair ID is invalid or duplicated")
        pair_ids.add(expected_pair_id)
        target_key = {**pair_key, "split": record["split"]}
        expected_target_id = _target_sample_id(target_key)
        if record["target_sample_id"] != expected_target_id:
            raise V2XSetPairError("target sample ID is invalid")
        if record["target_eligibility_agent_ids"] != [ego_id, infrastructure_id]:
            raise V2XSetPairError("target eligibility contains unselected agents")
        excluded_ids = record["excluded_agent_ids"]
        if (
            not isinstance(excluded_ids, list)
            or any(not isinstance(item, str) for item in excluded_ids)
            or {ego_id, infrastructure_id}.intersection(excluded_ids)
        ):
            raise V2XSetPairError("Pair excluded-agent audit is invalid")
        agents = record["agents"]
        if not isinstance(agents, list) or len(agents) != 2:
            raise V2XSetPairError("Pair must contain exactly two target agents")
        if (
            _validate_sequence_output_agent(agents[0], "ego") != ego_id
            or _validate_sequence_output_agent(agents[1], "infrastructure")
            != infrastructure_id
        ):
            raise V2XSetPairError("Pair target agent identity mismatch")
        if (
            agents[0]["numeric_id"] != record["ego_numeric_id"]
            or agents[1]["numeric_id"] != record["infrastructure_numeric_id"]
        ):
            raise V2XSetPairError("Pair target numeric identity mismatch")
        ego_pose = agents[0]["world_from_agent"]
        infrastructure_pose = agents[1]["world_from_agent"]
        expected_distance = math.sqrt(
            math.fsum(
                (
                    (infrastructure_pose[0][3] - ego_pose[0][3]) ** 2,
                    (infrastructure_pose[1][3] - ego_pose[1][3]) ** 2,
                )
            )
        )
        if (
            abs(float(record["infrastructure_distance_m"]) - expected_distance) > 1e-9
            or expected_distance > MAX_COLLABORATOR_DISTANCE_M
        ):
            raise V2XSetPairError("Pair target distance mismatch")
        _validate_endpoint_coverage(
            record["endpoint_coverage"],
            target_timestamp_us=timestamp_us,
            ego_agent_id=ego_id,
            infrastructure_agent_id=infrastructure_id,
            delays_ms=validated_delays,
            history_limit=validated_history,
            frame_interval_us=validated_interval,
        )
        if record["endpoint_coverage"]["ego_history"][0]["agent"] != agents[0]:
            raise V2XSetPairError("Pair current Ego differs from horizon-zero endpoint")
        if (
            record["endpoint_coverage"]["infrastructure_histories"][0]["history"][0][
                "agent"
            ]
            != agents[1]
        ):
            raise V2XSetPairError(
                "Pair current infrastructure differs from zero-delay endpoint"
            )
        pair_keys.append((scene_id, timestamp_us))
        target_ids.append(expected_target_id)
        target_ids_by_scene[scene_id].append(expected_target_id)
    if pair_keys != sorted(pair_keys) or len(pair_keys) != len(set(pair_keys)):
        raise V2XSetPairError("Pair records must be unique and scene/time ordered")

    exclusion_keys: list[tuple[str, int]] = []
    excluded_frame_ids: list[str] = []
    excluded_ids_by_scene: dict[str, list[str]] = {
        scene_id: [] for scene_id in sequence_by_id
    }
    for exclusion in exclusions:
        exclusion = _expect_fields(
            exclusion,
            frozenset(
                {
                    "frame_id",
                    "scene_id",
                    "split",
                    "timestamp_us",
                    "ego_agent_id",
                    "infrastructure_agent_id",
                    "excluded_agent_ids",
                    "reasons",
                }
            ),
            "excluded frame record",
        )
        scene_id = _identifier(exclusion["scene_id"], "excluded scene_id")
        sequence = sequence_by_id.get(scene_id)
        if sequence is None:
            raise V2XSetPairError("excluded frame references unknown sequence")
        timestamp_us = _exact_int(exclusion["timestamp_us"], "excluded timestamp")
        expected_frame_id = _frame_id(scene_id, timestamp_us)
        if exclusion["frame_id"] != expected_frame_id:
            raise V2XSetPairError("excluded frame ID mismatch")
        if (
            exclusion["split"] != sequence["split"]
            or not isinstance(exclusion["reasons"], list)
            or not exclusion["reasons"]
        ):
            raise V2XSetPairError("excluded frame split/reasons invalid")
        for reason in exclusion["reasons"]:
            if not isinstance(reason, Mapping) or type(reason.get("code")) is not str:
                raise V2XSetPairError("excluded frame contains invalid reason")
        excluded_agent_ids = exclusion["excluded_agent_ids"]
        if not isinstance(excluded_agent_ids, list) or len(excluded_agent_ids) != len(
            set(excluded_agent_ids)
        ):
            raise V2XSetPairError("excluded frame agent audit missing")
        for agent_id in excluded_agent_ids:
            _identifier(agent_id, "excluded frame agent_id")
        first_reason_code = exclusion["reasons"][0]["code"]
        if sequence["status"] == "excluded_no_anchor":
            if first_reason_code != "sequence_has_no_eligible_anchor":
                raise V2XSetPairError("no-anchor frame has the wrong exclusion reason")
            expected_ego = sequence["standard_fixed_ego_agent_id"]
            if (
                exclusion["ego_agent_id"] != expected_ego
                or exclusion["infrastructure_agent_id"] is not None
            ):
                raise V2XSetPairError("no-anchor frame contains a substituted pair")
        else:
            if (
                exclusion["ego_agent_id"] != sequence["ego_agent_id"]
                or exclusion["infrastructure_agent_id"]
                != sequence["infrastructure_agent_id"]
            ):
                raise V2XSetPairError("excluded frame changed the frozen pair")
            if {
                sequence["ego_agent_id"],
                sequence["infrastructure_agent_id"],
            }.intersection(excluded_agent_ids):
                raise V2XSetPairError("excluded-agent audit removes a frozen member")
            expected_code = (
                "before_sequence_anchor"
                if timestamp_us < sequence["anchor_timestamp_us"]
                else "frozen_pair_ineligible"
            )
            if first_reason_code != expected_code:
                raise V2XSetPairError(
                    "included-sequence frame has wrong exclusion reason"
                )
        exclusion_keys.append((scene_id, timestamp_us))
        excluded_frame_ids.append(expected_frame_id)
        excluded_ids_by_scene[scene_id].append(expected_frame_id)
    if exclusion_keys != sorted(exclusion_keys) or len(exclusion_keys) != len(
        set(exclusion_keys)
    ):
        raise V2XSetPairError("excluded frames must be unique and scene/time ordered")
    if set(pair_keys).intersection(exclusion_keys):
        raise V2XSetPairError("a frame cannot be both paired and excluded")
    if len(pair_keys) + len(exclusion_keys) != summary["frame_count"]:
        raise V2XSetPairError("not every input frame was accounted for")
    for scene_id, sequence in sequence_by_id.items():
        if (
            sequence["included_target_sample_ids"] != target_ids_by_scene[scene_id]
            or sequence["excluded_frame_ids"] != excluded_ids_by_scene[scene_id]
        ):
            raise V2XSetPairError("sequence derived ID membership mismatch")

    target_manifest = _expect_fields(
        value["target_manifest"],
        frozenset(
            {
                "schema_version",
                "artifact_type",
                "target_generation_stage",
                "sample_count",
                "split_counts",
                "sample_ids",
                "sample_ids_sha256",
                "content_sha256",
            }
        ),
        "target manifest",
    )
    if (
        target_manifest["schema_version"] != TARGET_SCHEMA_VERSION
        or target_manifest["artifact_type"] != TARGET_TYPE
        or target_manifest["target_generation_stage"]
        != "before_latency_selection_and_fault_injection"
    ):
        raise V2XSetPairError("target manifest schema/type/stage mismatch")
    if (
        target_manifest["sample_count"] != len(target_ids)
        or target_manifest["sample_ids"] != target_ids
    ):
        raise V2XSetPairError("target manifest sample IDs mismatch")
    if (
        target_manifest["sample_ids_sha256"]
        != _id_list_sha256(b"v2xset-pair-target-sample-ids-v1", target_ids)
        or content_sha256(target_manifest) != target_manifest["content_sha256"]
    ):
        raise V2XSetPairError("target manifest hash mismatch")
    expected_split_counts = {
        split: sum(pair["split"] == split for pair in pairs) for split in SPLITS
    }
    if target_manifest["split_counts"] != expected_split_counts:
        raise V2XSetPairError("target manifest split counts mismatch")
    references = value["condition_target_references"]
    expected_references = _condition_target_references(
        validated_delays, target_manifest
    )
    if references != expected_references:
        raise V2XSetPairError("condition target manifests are not invariant")

    derivation = _expect_fields(
        value["derivation_ids"],
        frozenset(
            {
                "included_sequence_ids",
                "excluded_sequence_ids",
                "included_target_sample_ids",
                "excluded_frame_ids",
            }
        ),
        "derivation_ids",
    )
    _validate_id_inventory(
        derivation["included_sequence_ids"],
        domain=b"v2xset-pair-included-sequences-v1",
        expected_ids=included_sequence_ids,
        context="included_sequence_ids",
    )
    _validate_id_inventory(
        derivation["excluded_sequence_ids"],
        domain=b"v2xset-pair-excluded-sequences-v1",
        expected_ids=excluded_sequence_ids,
        context="excluded_sequence_ids",
    )
    _validate_id_inventory(
        derivation["included_target_sample_ids"],
        domain=b"v2xset-pair-included-targets-v1",
        expected_ids=target_ids,
        context="included_target_sample_ids",
    )
    _validate_id_inventory(
        derivation["excluded_frame_ids"],
        domain=b"v2xset-pair-excluded-frames-v1",
        expected_ids=excluded_frame_ids,
        context="excluded_frame_ids",
    )
    digest = value["content_sha256"]
    if (
        type(digest) is not str
        or SHA256.fullmatch(digest) is None
        or content_sha256(value) != digest
    ):
        raise V2XSetPairError("Pair manifest content hash mismatch")
    canonical_json_bytes(value)


def _regular_bytes(path: Path, context: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise V2XSetPairError(f"unable to inspect {context}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise V2XSetPairError(f"{context} must be a regular file")
    return path.read_bytes()


def write_pair_manifest(path: Path, manifest: Mapping[str, object]) -> Path:
    _validate_pair_manifest(manifest)
    output = Path(path)
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
                raise OSError("short pair manifest write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(staging, output, follow_symlinks=False)
        except FileExistsError:
            if _regular_bytes(output, "existing pair manifest") != raw:
                raise V2XSetPairError("pair manifest destination conflict")
        directory = os.open(
            output.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return output
    except V2XSetPairError:
        raise
    except OSError as error:
        raise V2XSetPairError("pair manifest publication failed") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        staging.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a normalized V2XSet inventory and prepare deterministic "
            "sequence-frozen one-Ego/one-Infra Pair records with complete "
            "configured causal endpoints. This does not implement "
            "V2XSet Standard multi-agent evaluation."
        )
    )
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = prepare_pair_manifest(args.inventory, args.data_root)
        output = write_pair_manifest(args.out, manifest)
    except (OSError, V2XSetPairError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_type": PAIR_TYPE,
                "content_sha256": manifest["content_sha256"],
                "pair_count": manifest["pair_count"],
                "excluded_frame_count": manifest["excluded_frame_count"],
                "included_sequence_count": manifest["included_sequence_count"],
                "excluded_sequence_count": manifest["excluded_sequence_count"],
                "target_manifest_content_sha256": manifest["target_manifest"][
                    "content_sha256"
                ],
                "target_sample_count": manifest["target_manifest"]["sample_count"],
                "output": str(output.resolve()),
                "standard_multi_agent_protocol_implemented": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CAMERA_IDS",
    "CONDITION_NAMES",
    "DISCLAIMER",
    "INVENTORY_TYPE",
    "MAX_COLLABORATOR_DISTANCE_M",
    "PAIR_TYPE",
    "TARGET_TYPE",
    "V2XSetPairError",
    "main",
    "parse_args",
    "prepare_pair_manifest",
    "write_pair_manifest",
)
