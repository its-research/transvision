#!/usr/bin/env python3
"""Pure-standard-library V2X-Graph/HistoryOnly preprocessing comparator.

Version 1 deliberately reports *candidate checks* only.  It can prove that a
sealed export is internally consistent with an explicitly bound HistoryOnly
sample for actor order, causal time steps, coordinate transforms, masks, and
association edges.  It cannot prove complete preprocessing parity because the
official scene bytes, generated ``v2x_fusion`` provenance, DAIR map replay,
and two geometric edge masks are not yet independently covered.

Consequently no successful call from this module returns
``parity_verified=true`` or permits a scientific claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
CONTRACT_ID = "V2XGRAPH-TFD-PREPROCESS-PARITY-v1"
EXPORT_KIND = "v2xgraph_tfd_temporaldata_history_projection"
UPSTREAM_REPOSITORY = "https://github.com/AIR-THU/V2X-Graph"
UPSTREAM_COMMIT = "b6cf34b3a67db1d60b77a1fcf73ce234c0df439c"
HISTORY_STEPS = 50
FUTURE_STEPS = 50
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_TENSOR_ELEMENTS = 20_000_000
COORDINATE_RULE = "v2xgraph-b6cf34b3-row-vector-world-minus-origin-times-rotation-v1"

UPSTREAM_SOURCE_SHA256 = {
    "datasets/tfd_dataset.py": (
        "f4c4b0e0e50681d103a0fa91f2995daaafa8964c3376885e23138e191704e131"
    ),
    "preprocess.py": (
        "eabcf103bd4a71fb76ed17c86430a2e8f56d81fb77bf34dc96c3a4fb58e9c1ed"
    ),
    "traj_match_labels/generate_label.py": (
        "5183e9790f89cf8b660f1a92faf175a747ebe4d2e214e04c23acafce4b21c3ab"
    ),
    "utils.py": ("4f7f02241cf69f010f5da764c106291fce7f8cc953b268df3c623270d83533af"),
}

PROJECTED_FIELDS = frozenset(
    {
        "x",
        "positions",
        "edge_index",
        "v2x_edge_index",
        "v2x_pesudo_mask",  # spelling is fixed by the pinned upstream source
        "v2x_mask",
        "v2x_aa_mask",
        "v2x_ins_mask",
        "v2x_type_mask",
        "interact_ego_mask",
        "interact_road_mask",
        "actor_type",
        "num_nodes",
        "num_car_actors",
        "num_road_actors",
        "ego_mask",
        "road_mask",
        "padding_mask",
        "bos_mask",
        "rotate_angles",
        "lane_vectors",
        "is_intersections",
        "turn_directions",
        "traffic_controls",
        "lane_actor_index",
        "lane_actor_vectors",
        "seq_id",
        "av_index",
        "agent_index",
        "city",
        "origin",
        "theta",
        "last_positions",
    }
)

TENSOR_FIELDS = PROJECTED_FIELDS - {
    "num_nodes",
    "num_car_actors",
    "num_road_actors",
    "seq_id",
    "av_index",
    "agent_index",
    "city",
}

TRAJECTORY_STREAMS = frozenset(
    {
        "cooperative_trajectories",
        "vehicle_trajectories",
        "infrastructure_trajectories",
    }
)

BASE_TRAJECTORY_FIELDS = frozenset(
    {
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
        "event_time_ns",
        "arrival_time_ns",
    }
)
COOPERATIVE_TRAJECTORY_FIELDS = BASE_TRAJECTORY_FIELDS | {
    "vic_tag",
    "from_side",
    "car_side_id",
    "road_side_id",
}
TRAFFIC_LIGHT_FIELDS = frozenset(
    {
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
        "event_time_ns",
        "arrival_time_ns",
    }
)
HISTORY_ROW_FIELDS = {
    "cooperative_trajectories": COOPERATIVE_TRAJECTORY_FIELDS,
    "vehicle_trajectories": BASE_TRAJECTORY_FIELDS,
    "infrastructure_trajectories": BASE_TRAJECTORY_FIELDS,
    "traffic_lights": TRAFFIC_LIGHT_FIELDS,
}

UNRESOLVED_CATEGORIES = (
    "official TFD identity manifest and source-byte readback",
    "generated v2x_fusion provenance back to the official cooperative source",
    "replay provenance from sealed raw inputs to the processed TemporalData bytes",
    "lane and map feature replay with the pinned DAIR map API",
    "independent v2x_aa_mask and v2x_ins_mask geometric recomputation",
    "installed upstream dependency environment seal",
    "future targets and future masks intentionally withheld from history-only comparison",
    "model forward/backward and checkpoint numerical parity",
)

FORBIDDEN_HISTORY_KEYS = frozenset(
    {
        "future",
        "futures",
        "ground_truth",
        "groundtruth",
        "label",
        "labels",
        "supervision",
        "target_positions",
        "target_trajectory",
        "target_timestamps",
        "truth",
    }
)


class ParityContractError(RuntimeError):
    """Raised when evidence is ambiguous, incomplete, or inconsistent."""


def _duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ParityContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ParityContractError(f"non-finite JSON constant is forbidden: {value}")


def _validate_json(value: Any, label: str = "JSON value", depth: int = 0) -> Any:
    if depth > 64:
        raise ParityContractError(f"{label} exceeds maximum nesting depth")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if abs(value) > 2**63 - 1:
            raise ParityContractError(f"{label} contains an out-of-range integer")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ParityContractError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{label}[{index}]", depth + 1)
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ParityContractError(f"{label} contains a non-string key")
            _validate_json(item, f"{label}.{key}", depth + 1)
        return value
    raise ParityContractError(
        f"{label} contains a non-JSON value: {type(value).__name__}"
    )


def loads_strict(data: bytes | str, label: str = "JSON document") -> Any:
    if isinstance(data, bytes):
        if len(data) > MAX_JSON_BYTES:
            raise ParityContractError(f"{label} exceeds the byte limit")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ParityContractError(f"{label} is not UTF-8") from exc
    else:
        text = data
        if len(text.encode("utf-8")) > MAX_JSON_BYTES:
            raise ParityContractError(f"{label} exceeds the byte limit")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_object,
            parse_constant=_reject_constant,
        )
    except ParityContractError:
        raise
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ParityContractError(f"{label} is not strict JSON") from exc
    return _validate_json(value, label)


def canonical_bytes(value: Any) -> bytes:
    _validate_json(value)
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ParityContractError("value cannot be canonically encoded") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ParityContractError(f"cannot hash evidence file: {path.name}") from exc
    return digest.hexdigest()


def load_json_file(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ParityContractError(f"{label} must be a non-symlink regular file")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ParityContractError(f"cannot read {label}") from exc
    return loads_strict(data, label)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ParityContractError(f"cannot write output: {path.name}") from exc


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParityContractError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ParityContractError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ParityContractError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParityContractError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ParityContractError(f"{label} must be >= {minimum}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ParityContractError(f"{label} must be finite")
    return result


def _numeric_text(value: Any, label: str) -> float:
    if not isinstance(value, str) or not value.strip():
        raise ParityContractError(f"{label} must be a numeric string")
    try:
        number = float(value)
    except ValueError as exc:
        raise ParityContractError(f"{label} must be a numeric string") from exc
    if not math.isfinite(number):
        raise ParityContractError(f"{label} must be finite")
    return number


def _float32(value: float, label: str) -> float:
    """Round one finite Python float through IEEE-754 binary32 like Tensor.float."""

    try:
        rounded = struct.unpack(">f", struct.pack(">f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ParityContractError(f"{label} is outside finite float32 range") from exc
    if not math.isfinite(rounded):
        raise ParityContractError(f"{label} becomes non-finite after float32 rounding")
    return rounded


def _digest(value: Any, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ParityContractError(f"{label} must be a lowercase SHA-256 digest")
    return text


def decimal_token(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParityContractError(f"{label} must be a decimal string")
    try:
        number = Decimal(value.strip())
    except (InvalidOperation, ValueError) as exc:
        raise ParityContractError(f"{label} is not decimal") from exc
    if not number.is_finite():
        raise ParityContractError(f"{label} is not finite")
    text = format(number.normalize(), "f")
    return "0" if text in {"-0", ""} else text


def identity_token(value: Any, label: str) -> str:
    text = _string(value, label).strip()
    try:
        return decimal_token(text, label)
    except ParityContractError:
        return text


def tensor_descriptor(
    dtype: str, shape: Sequence[int], values: Sequence[Any]
) -> dict[str, Any]:
    """Build the canonical tensor descriptor used by fixtures and exporter."""

    core = {"dtype": dtype, "shape": list(shape), "values": list(values)}
    return {"kind": "tensor", **core, "sha256": sha256_json(core)}


def scalar_descriptor(value_type: str, value: Any) -> dict[str, Any]:
    core = {"type": value_type, "value": value}
    return {"kind": "scalar", **core, "sha256": sha256_json(core)}


def _shape_size(shape: Sequence[int]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def validate_tensor_descriptor(value: Any, label: str) -> dict[str, Any]:
    descriptor = _mapping(value, label)
    _exact(descriptor, {"kind", "dtype", "shape", "values", "sha256"}, label)
    if descriptor.get("kind") != "tensor":
        raise ParityContractError(f"{label}.kind must be tensor")
    dtype = _string(descriptor.get("dtype"), f"{label}.dtype")
    if dtype not in {
        "torch.bool",
        "torch.float32",
        "torch.float64",
        "torch.int64",
        "torch.uint8",
    }:
        raise ParityContractError(f"{label} has unsupported dtype {dtype}")
    shape = descriptor.get("shape")
    if not isinstance(shape, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in shape
    ):
        raise ParityContractError(f"{label}.shape must contain non-negative integers")
    size = _shape_size(shape)
    if size > MAX_TENSOR_ELEMENTS:
        raise ParityContractError(f"{label} exceeds tensor element limit")
    values = descriptor.get("values")
    if not isinstance(values, list) or len(values) != size:
        raise ParityContractError(f"{label}.values length does not match shape")
    for index, item in enumerate(values):
        item_label = f"{label}.values[{index}]"
        if dtype == "torch.bool":
            if not isinstance(item, bool):
                raise ParityContractError(f"{item_label} must be boolean")
        elif dtype in {"torch.int64", "torch.uint8"}:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ParityContractError(f"{item_label} must be integer")
            if dtype == "torch.uint8" and not 0 <= item <= 255:
                raise ParityContractError(f"{item_label} is outside uint8")
        else:
            _finite(item, item_label)
    core = {"dtype": dtype, "shape": shape, "values": values}
    if _digest(descriptor.get("sha256"), f"{label}.sha256") != sha256_json(core):
        raise ParityContractError(f"{label} descriptor SHA-256 mismatch")
    return descriptor


def validate_scalar_descriptor(value: Any, label: str) -> dict[str, Any]:
    descriptor = _mapping(value, label)
    _exact(descriptor, {"kind", "type", "value", "sha256"}, label)
    if descriptor.get("kind") != "scalar":
        raise ParityContractError(f"{label}.kind must be scalar")
    value_type = _string(descriptor.get("type"), f"{label}.type")
    raw = descriptor.get("value")
    if value_type == "int":
        _integer(raw, f"{label}.value")
    elif value_type == "string":
        _string(raw, f"{label}.value")
    else:
        raise ParityContractError(f"{label}.type is unsupported")
    core = {"type": value_type, "value": raw}
    if _digest(descriptor.get("sha256"), f"{label}.sha256") != sha256_json(core):
        raise ParityContractError(f"{label} descriptor SHA-256 mismatch")
    return descriptor


def _descriptor_value(projection: Mapping[str, Any], key: str) -> Any:
    descriptor = projection[key]
    if descriptor["kind"] == "scalar":
        return descriptor["value"]
    return descriptor


def _tensor_at(descriptor: Mapping[str, Any], *indices: int) -> Any:
    shape = descriptor["shape"]
    if len(indices) != len(shape):
        raise ParityContractError("tensor index rank mismatch")
    offset = 0
    for index, dimension in zip(indices, shape):
        if not 0 <= index < dimension:
            raise ParityContractError("tensor index is out of range")
        offset = offset * dimension + index
    return descriptor["values"][offset]


def _edge_pairs(descriptor: Mapping[str, Any], label: str) -> list[tuple[int, int]]:
    shape = descriptor["shape"]
    if len(shape) != 2 or shape[0] != 2:
        raise ParityContractError(f"{label} must have shape [2, E]")
    count = shape[1]
    values = descriptor["values"]
    return [(int(values[index]), int(values[count + index])) for index in range(count)]


def _file_seal(value: Any, label: str) -> dict[str, Any]:
    seal = _mapping(value, label)
    _exact(seal, {"basename", "size", "sha256"}, label)
    _string(seal.get("basename"), f"{label}.basename")
    _integer(seal.get("size"), f"{label}.size", minimum=1)
    _digest(seal.get("sha256"), f"{label}.sha256")
    return seal


def _validate_withheld(value: Any, label: str) -> dict[str, Any]:
    item = _mapping(value, label)
    _exact(item, {"present", "dtype", "shape", "canonical_values_sha256"}, label)
    if not isinstance(item.get("present"), bool):
        raise ParityContractError(f"{label}.present must be boolean")
    if item["present"]:
        _string(item.get("dtype"), f"{label}.dtype")
        shape = item.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(part, bool) or not isinstance(part, int) or part < 0
            for part in shape
        ):
            raise ParityContractError(f"{label}.shape is invalid")
        _digest(item.get("canonical_values_sha256"), f"{label}.canonical_values_sha256")
    elif any(
        item.get(name) is not None
        for name in ("dtype", "shape", "canonical_values_sha256")
    ):
        raise ParityContractError(f"{label} absent tensor metadata must be null")
    return item


def validate_export_document(value: Any) -> dict[str, Any]:
    export = _mapping(value, "upstream export")
    _exact(
        export,
        {
            "schema_version",
            "contract_id",
            "export_kind",
            "claim_scope",
            "scientific_claim_allowed",
            "upstream",
            "runtime",
            "scene",
            "temporal_data",
            "exporter",
            "content_sha256",
        },
        "upstream export",
    )
    if (
        export.get("schema_version") != SCHEMA_VERSION
        or export.get("contract_id") != CONTRACT_ID
    ):
        raise ParityContractError("upstream export contract identity mismatch")
    if export.get("export_kind") != EXPORT_KIND:
        raise ParityContractError("upstream export kind mismatch")
    if export.get("claim_scope") != "diagnostic_preprocessing_evidence_only":
        raise ParityContractError("upstream export claim scope is not diagnostic-only")
    if export.get("scientific_claim_allowed") is not False:
        raise ParityContractError("upstream export cannot allow scientific claims")

    upstream = _mapping(export.get("upstream"), "upstream source")
    _exact(upstream, {"repository", "revision", "source_files"}, "upstream source")
    if (
        upstream.get("repository") != UPSTREAM_REPOSITORY
        or upstream.get("revision") != UPSTREAM_COMMIT
    ):
        raise ParityContractError("upstream repository or revision drift")
    sources = _mapping(upstream.get("source_files"), "upstream source files")
    if set(sources) != set(UPSTREAM_SOURCE_SHA256):
        raise ParityContractError("upstream source-file inventory drift")
    for path, expected in UPSTREAM_SOURCE_SHA256.items():
        if _digest(sources[path], f"upstream source {path}") != expected:
            raise ParityContractError(f"upstream source hash drift: {path}")

    runtime = _mapping(export.get("runtime"), "upstream runtime")
    _exact(runtime, {"python", "torch", "torch_geometric"}, "upstream runtime")
    for name in runtime:
        _string(runtime[name], f"upstream runtime {name}")

    scene = _mapping(export.get("scene"), "upstream scene")
    _exact(
        scene,
        {
            "scene_id",
            "split",
            "processed_file",
            "inputs",
            "history_steps",
            "future_steps",
            "native_history_timestamps",
            "actors",
            "association_rows",
        },
        "upstream scene",
    )
    _string(scene.get("scene_id"), "upstream scene_id")
    if scene.get("split") not in {"sample", "train", "val"}:
        raise ParityContractError(
            "upstream split must be sample, train, or val; pinned test preprocessing dereferences null y"
        )
    _file_seal(scene.get("processed_file"), "processed file seal")
    inputs = _mapping(scene.get("inputs"), "upstream inputs")
    _exact(inputs, {"ego_raw", "road_raw", "association_raw"}, "upstream inputs")
    for name in inputs:
        _file_seal(inputs[name], f"upstream input {name}")
    if (
        scene.get("history_steps") != HISTORY_STEPS
        or scene.get("future_steps") != FUTURE_STEPS
    ):
        raise ParityContractError("upstream temporal horizon drift")
    native_times = scene.get("native_history_timestamps")
    if not isinstance(native_times, list) or len(native_times) != HISTORY_STEPS:
        raise ParityContractError("upstream history must expose 50 native timestamps")
    normalized_times = [
        decimal_token(item, f"native_history_timestamps[{index}]")
        for index, item in enumerate(native_times)
    ]
    if normalized_times != native_times or len(set(native_times)) != HISTORY_STEPS:
        raise ParityContractError(
            "native history timestamps are not canonical and unique"
        )
    if [Decimal(item) for item in native_times] != sorted(
        Decimal(item) for item in native_times
    ):
        raise ParityContractError("native history timestamps are not sorted")

    actors = scene.get("actors")
    if not isinstance(actors, list) or not actors:
        raise ParityContractError("upstream actors must be a non-empty array")
    actor_ids: list[str] = []
    for index, raw in enumerate(actors):
        actor = _mapping(raw, f"upstream actor {index}")
        _exact(actor, {"index", "actor_id", "side", "tags"}, f"upstream actor {index}")
        if actor.get("index") != index:
            raise ParityContractError(
                "upstream actor indices must be contiguous and ordered"
            )
        actor_id = identity_token(
            actor.get("actor_id"), f"upstream actor {index}.actor_id"
        )
        if actor_id != actor.get("actor_id") or actor_id in actor_ids:
            raise ParityContractError("upstream actor IDs must be canonical and unique")
        actor_ids.append(actor_id)
        if actor.get("side") not in {"ego", "road"}:
            raise ParityContractError("upstream actor side is invalid")
        tags = actor.get("tags")
        if (
            not isinstance(tags, list)
            or any(not isinstance(tag, str) or not tag for tag in tags)
            or tags != sorted(set(tags))
        ):
            raise ParityContractError("upstream actor tags must be sorted and unique")

    association_rows = scene.get("association_rows")
    if not isinstance(association_rows, list):
        raise ParityContractError("association_rows must be an array")
    for index, raw in enumerate(association_rows):
        row = _mapping(raw, f"association row {index}")
        _exact(
            row,
            {"timestamp", "id", "tag", "ego_side_id", "coop_side_id"},
            f"association row {index}",
        )
        if (
            decimal_token(row.get("timestamp"), f"association row {index}.timestamp")
            not in native_times
        ):
            raise ParityContractError("association row is outside the history window")
        identity_token(row.get("id"), f"association row {index}.id")
        _string(row.get("tag"), f"association row {index}.tag")
        for key in ("ego_side_id", "coop_side_id"):
            if row.get(key) is not None:
                identity_token(row[key], f"association row {index}.{key}")

    temporal = _mapping(export.get("temporal_data"), "TemporalData export")
    _exact(
        temporal,
        {"field_inventory", "history_projection", "withheld_future_fields"},
        "TemporalData export",
    )
    inventory = temporal.get("field_inventory")
    if (
        not isinstance(inventory, list)
        or any(not isinstance(name, str) or not name for name in inventory)
        or inventory != sorted(set(inventory))
    ):
        raise ParityContractError(
            "TemporalData field inventory must be sorted and unique"
        )
    projection = _mapping(temporal.get("history_projection"), "history projection")
    if set(projection) != PROJECTED_FIELDS:
        raise ParityContractError("history projection field inventory drift")
    if "y" in projection:
        raise ParityContractError("future ground truth leaked into history projection")
    for name in sorted(TENSOR_FIELDS):
        validate_tensor_descriptor(projection[name], f"history projection {name}")
    for name in sorted(PROJECTED_FIELDS - TENSOR_FIELDS):
        validate_scalar_descriptor(projection[name], f"history projection {name}")
    withheld = _mapping(
        temporal.get("withheld_future_fields"), "withheld future fields"
    )
    _exact(withheld, {"positions_future", "y"}, "withheld future fields")
    _validate_withheld(withheld["positions_future"], "withheld positions_future")
    _validate_withheld(withheld["y"], "withheld y")

    exporter = _mapping(export.get("exporter"), "exporter evidence")
    _exact(exporter, {"basename", "sha256", "contract_sha256"}, "exporter evidence")
    if exporter.get("basename") != "export_upstream.py":
        raise ParityContractError("unexpected exporter basename")
    expected_exporter = sha256_file(Path(__file__).with_name("export_upstream.py"))
    expected_contract = sha256_file(Path(__file__))
    if _digest(exporter.get("sha256"), "exporter sha256") != expected_exporter:
        raise ParityContractError("exporter source binding mismatch")
    if _digest(exporter.get("contract_sha256"), "contract sha256") != expected_contract:
        raise ParityContractError("parity contract source binding mismatch")

    without_hash = {
        key: item for key, item in export.items() if key != "content_sha256"
    }
    if _digest(export.get("content_sha256"), "export content_sha256") != sha256_json(
        without_hash
    ):
        raise ParityContractError("upstream export content SHA-256 mismatch")

    _validate_temporal_relationships(export)
    return export


def _validate_temporal_relationships(export: Mapping[str, Any]) -> None:
    scene = export["scene"]
    projection = export["temporal_data"]["history_projection"]
    num_nodes = _integer(
        _descriptor_value(projection, "num_nodes"), "num_nodes", minimum=1
    )
    if num_nodes != len(scene["actors"]):
        raise ParityContractError("num_nodes does not match actor witness")
    num_car = _integer(
        _descriptor_value(projection, "num_car_actors"), "num_car_actors", minimum=1
    )
    num_road = _integer(
        _descriptor_value(projection, "num_road_actors"), "num_road_actors", minimum=1
    )
    if num_car + num_road != num_nodes:
        raise ParityContractError("car and road actor counts do not sum to num_nodes")
    expected_sides = ["ego"] * num_car + ["road"] * num_road
    if [actor["side"] for actor in scene["actors"]] != expected_sides:
        raise ParityContractError(
            "actor witness side order does not match upstream counts"
        )

    withheld = export["temporal_data"]["withheld_future_fields"]
    positions_future = withheld["positions_future"]
    if (
        positions_future["present"] is not True
        or positions_future["dtype"] != "torch.float32"
        or positions_future["shape"] != [num_nodes, FUTURE_STEPS, 2]
    ):
        raise ParityContractError(
            "withheld future positions must be present float32 [N, 50, 2]"
        )
    y = withheld["y"]
    if y["present"] is not True:
        raise ParityContractError("withheld y presence does not match upstream split")
    if y["dtype"] != "torch.float32" or y["shape"] != [num_nodes, FUTURE_STEPS, 2]:
        raise ParityContractError("withheld y must be float32 [N, 50, 2]")
    if "y" not in export["temporal_data"]["field_inventory"]:
        raise ParityContractError(
            "TemporalData y inventory does not match upstream split"
        )

    expected_shapes = {
        "x": [num_nodes, HISTORY_STEPS, 2],
        "positions": [num_nodes, HISTORY_STEPS, 2],
        "padding_mask": [num_nodes, HISTORY_STEPS],
        "bos_mask": [num_nodes, HISTORY_STEPS],
        "rotate_angles": [num_nodes],
        "actor_type": [num_nodes],
        "ego_mask": [num_nodes],
        "road_mask": [num_nodes],
        "last_positions": [num_nodes, 2],
        "origin": [1, 2],
        "theta": [],
    }
    for name, shape in expected_shapes.items():
        if projection[name]["shape"] != shape:
            raise ParityContractError(f"{name} shape mismatch")
    for index in range(num_nodes):
        ego_expected = index < num_car
        if _tensor_at(projection["ego_mask"], index) is not ego_expected:
            raise ParityContractError("ego_mask does not match actor order")
        if _tensor_at(projection["road_mask"], index) is not (not ego_expected):
            raise ParityContractError("road_mask does not match actor order")

    edge_pairs = _edge_pairs(projection["edge_index"], "edge_index")
    expected_edges = {
        (source, target)
        for source in range(num_nodes)
        for target in range(num_nodes)
        if source != target
    }
    if len(edge_pairs) != len(expected_edges) or set(edge_pairs) != expected_edges:
        raise ParityContractError("edge_index is not the complete directed actor graph")
    edge_count = len(edge_pairs)
    for name in (
        "v2x_pesudo_mask",
        "v2x_mask",
        "v2x_aa_mask",
        "v2x_ins_mask",
        "v2x_type_mask",
        "interact_ego_mask",
        "interact_road_mask",
    ):
        if projection[name]["shape"] != [edge_count]:
            raise ParityContractError(f"{name} shape does not match edge_index")

    actor_types = projection["actor_type"]["values"]
    association_edges = _association_edges(scene["association_rows"], scene["actors"])
    exported_association_edges = set(
        _edge_pairs(projection["v2x_edge_index"], "v2x_edge_index")
    )
    if exported_association_edges != association_edges:
        raise ParityContractError("v2x_edge_index does not match association witness")
    for position, pair in enumerate(edge_pairs):
        source, target = pair
        cross_side = (source < num_car) != (target < num_car)
        same_type = actor_types[source] == actor_types[target]
        same_ego = source < num_car and target < num_car
        same_road = source >= num_car and target >= num_car
        expected = {
            "v2x_pesudo_mask": pair in association_edges,
            "v2x_mask": cross_side,
            "v2x_type_mask": same_type and cross_side,
            "interact_ego_mask": same_ego,
            "interact_road_mask": same_road,
        }
        for name, expected_value in expected.items():
            if _tensor_at(projection[name], position) is not expected_value:
                raise ParityContractError(f"{name} is inconsistent at edge {pair}")

    lane_vectors = projection["lane_vectors"]["shape"]
    if len(lane_vectors) != 2 or lane_vectors[1] != 2 or lane_vectors[0] <= 0:
        raise ParityContractError("lane_vectors must be non-empty [L, 2]")
    lane_count = lane_vectors[0]
    for name in ("is_intersections", "turn_directions", "traffic_controls"):
        if projection[name]["shape"] != [lane_count]:
            raise ParityContractError(f"{name} does not match lane count")
    lane_actor_index = projection["lane_actor_index"]["shape"]
    if len(lane_actor_index) != 2 or lane_actor_index[0] != 2:
        raise ParityContractError("lane_actor_index must have shape [2, E]")
    if projection["lane_actor_vectors"]["shape"] != [lane_actor_index[1], 2]:
        raise ParityContractError("lane_actor_vectors does not match lane_actor_index")
    for axis in range(lane_actor_index[1]):
        lane_index = int(_tensor_at(projection["lane_actor_index"], 0, axis))
        actor_index = int(_tensor_at(projection["lane_actor_index"], 1, axis))
        if not 0 <= lane_index < lane_count or not 0 <= actor_index < num_nodes:
            raise ParityContractError("lane_actor_index contains an out-of-range index")


def _association_edges(
    rows: Sequence[Mapping[str, Any]], actors: Sequence[Mapping[str, Any]]
) -> set[tuple[int, int]]:
    actor_index = {actor["actor_id"]: actor["index"] for actor in actors}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["tag"] == "AV":
            continue
        grouped.setdefault(row["id"], []).append(row)
    relations: set[tuple[int, int]] = set()
    for group in grouped.values():
        ego_ids: list[str] = []
        coop_ids: list[str] = []
        for row in group:
            for key, destination in (
                ("ego_side_id", ego_ids),
                ("coop_side_id", coop_ids),
            ):
                actor_id = row[key]
                if actor_id is not None and actor_id not in destination:
                    destination.append(actor_id)
        try:
            ego_indices = [actor_index[item] for item in ego_ids]
            coop_indices = [actor_index[item] for item in coop_ids]
        except KeyError as exc:
            raise ParityContractError(
                "association witness references an actor outside actor order"
            ) from exc
        for coop in coop_indices:
            for ego in ego_indices:
                relations.add((coop, ego))
                relations.add((ego, coop))
    return relations


def _reject_future_keys(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = "".join(
                character for character in key.casefold() if character.isalnum()
            )
            forbidden = (
                key.casefold() in FORBIDDEN_HISTORY_KEYS
                or "future" in normalized
                or "groundtruth" in normalized
                or normalized.startswith("label")
                or normalized.startswith("target")
                or normalized.startswith("supervision")
            )
            if forbidden:
                raise ParityContractError(
                    f"{label} contains future-bearing key {key!r}"
                )
            _reject_future_keys(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_future_keys(item, f"{label}[{index}]")


def validate_history_sample(value: Any) -> dict[str, Any]:
    sample = _mapping(value, "HistoryOnly sample")
    _exact(
        sample,
        {"scene_id", "target_id", "history", "input_sha256"},
        "HistoryOnly sample",
    )
    _string(sample.get("scene_id"), "HistoryOnly scene_id")
    _string(sample.get("target_id"), "HistoryOnly target_id")
    history = _mapping(sample.get("history"), "HistoryOnly history")
    _exact(
        history,
        {
            "decision_time_ns",
            "timestamps_ns",
            "cooperative_trajectories",
            "vehicle_trajectories",
            "infrastructure_trajectories",
            "traffic_lights",
            "hd_map",
            "coordinate_frame",
        },
        "HistoryOnly history",
    )
    _reject_future_keys(history, "HistoryOnly history")
    timestamps = history.get("timestamps_ns")
    if (
        not isinstance(timestamps, list)
        or len(timestamps) != HISTORY_STEPS
        or any(
            isinstance(item, bool) or not isinstance(item, int) for item in timestamps
        )
        or timestamps != sorted(set(timestamps))
    ):
        raise ParityContractError(
            "HistoryOnly timestamps must contain 50 sorted unique integers"
        )
    if history.get("decision_time_ns") != timestamps[-1]:
        raise ParityContractError(
            "HistoryOnly decision time must equal the final history timestamp"
        )
    for stream in sorted(TRAJECTORY_STREAMS | {"traffic_lights"}):
        rows = history.get(stream)
        if not isinstance(rows, list) or not rows:
            raise ParityContractError(f"HistoryOnly {stream} must be a non-empty array")
        seen: set[tuple[str, int]] = set()
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"HistoryOnly {stream}[{index}]")
            expected_fields = set(HISTORY_ROW_FIELDS[stream])
            _exact(row, expected_fields, f"HistoryOnly {stream}[{index}]")
            for field in sorted(expected_fields - {"event_time_ns", "arrival_time_ns"}):
                if not isinstance(row[field], str):
                    raise ParityContractError(
                        f"HistoryOnly {stream}[{index}].{field} must preserve the CSV string value"
                    )
            event = _integer(
                row.get("event_time_ns"), f"{stream}[{index}].event_time_ns"
            )
            arrival = _integer(
                row.get("arrival_time_ns"), f"{stream}[{index}].arrival_time_ns"
            )
            if event not in timestamps or arrival > history["decision_time_ns"]:
                raise ParityContractError(
                    f"HistoryOnly {stream} violates the causal window"
                )
            identity = str(row.get("id", row.get("lane_id", "")))
            key = (identity, event)
            if key in seen:
                raise ParityContractError(
                    f"HistoryOnly {stream} has a duplicate identity/time row"
                )
            seen.add(key)
    if not isinstance(history.get("hd_map"), dict) or not history["hd_map"]:
        raise ParityContractError("HistoryOnly hd_map must be a non-empty object")
    _string(history.get("coordinate_frame"), "HistoryOnly coordinate_frame")
    expected_hash = sha256_json(history)
    if _digest(sample.get("input_sha256"), "HistoryOnly input_sha256") != expected_hash:
        raise ParityContractError("HistoryOnly input SHA-256 mismatch")
    return sample


def validate_bindings(value: Any) -> dict[str, Any]:
    bindings = _mapping(value, "parity bindings")
    _exact(
        bindings,
        {
            "schema_version",
            "contract_id",
            "status",
            "scene_id",
            "upstream_export_content_sha256",
            "history_input_sha256",
            "canonical_coordinate_frame",
            "coordinate_rule",
            "absolute_tolerance",
            "timestamp_bindings",
            "actor_bindings",
        },
        "parity bindings",
    )
    if (
        bindings.get("schema_version") != SCHEMA_VERSION
        or bindings.get("contract_id") != CONTRACT_ID
    ):
        raise ParityContractError("parity binding contract identity mismatch")
    if bindings.get("status") != "candidate_unverified":
        raise ParityContractError(
            "binding status must remain candidate_unverified until official parity is independently gated"
        )
    _string(bindings.get("scene_id"), "binding scene_id")
    _digest(
        bindings.get("upstream_export_content_sha256"), "binding upstream export hash"
    )
    _digest(bindings.get("history_input_sha256"), "binding history input hash")
    _string(bindings.get("canonical_coordinate_frame"), "binding coordinate frame")
    if bindings.get("coordinate_rule") != COORDINATE_RULE:
        raise ParityContractError("unsupported or drifted coordinate rule")
    tolerance = _finite(bindings.get("absolute_tolerance"), "absolute_tolerance")
    if tolerance <= 0 or tolerance > 1e-4:
        raise ParityContractError("absolute_tolerance must be in (0, 1e-4]")
    timestamp_bindings = bindings.get("timestamp_bindings")
    if (
        not isinstance(timestamp_bindings, list)
        or len(timestamp_bindings) != HISTORY_STEPS
    ):
        raise ParityContractError("timestamp bindings must cover exactly 50 steps")
    for index, raw in enumerate(timestamp_bindings):
        item = _mapping(raw, f"timestamp binding {index}")
        _exact(
            item,
            {"upstream_index", "native_timestamp", "event_time_ns"},
            f"timestamp binding {index}",
        )
        if item.get("upstream_index") != index:
            raise ParityContractError(
                "timestamp bindings must be ordered and contiguous"
            )
        if decimal_token(
            item.get("native_timestamp"), f"timestamp binding {index}.native_timestamp"
        ) != item.get("native_timestamp"):
            raise ParityContractError(
                "timestamp binding native timestamp is not canonical"
            )
        _integer(item.get("event_time_ns"), f"timestamp binding {index}.event_time_ns")
    actor_bindings = bindings.get("actor_bindings")
    if not isinstance(actor_bindings, list) or not actor_bindings:
        raise ParityContractError("actor bindings must be a non-empty array")
    canonical_keys: set[tuple[str, str]] = set()
    for index, raw in enumerate(actor_bindings):
        item = _mapping(raw, f"actor binding {index}")
        _exact(
            item,
            {
                "upstream_index",
                "upstream_actor_id",
                "canonical_stream",
                "canonical_actor_id",
            },
            f"actor binding {index}",
        )
        if item.get("upstream_index") != index:
            raise ParityContractError("actor bindings must be ordered and contiguous")
        if identity_token(
            item.get("upstream_actor_id"), f"actor binding {index}.upstream_actor_id"
        ) != item.get("upstream_actor_id"):
            raise ParityContractError("upstream actor binding ID is not canonical")
        if item.get("canonical_stream") not in TRAJECTORY_STREAMS:
            raise ParityContractError("actor binding canonical stream is invalid")
        canonical_id = _string(
            item.get("canonical_actor_id"), f"actor binding {index}.canonical_actor_id"
        )
        key = (item["canonical_stream"], canonical_id)
        if key in canonical_keys:
            raise ParityContractError("canonical actor binding is duplicated")
        canonical_keys.add(key)
    return bindings


def _close(actual: Any, expected: float, tolerance: float, label: str) -> None:
    number = _finite(actual, label)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ParityContractError(
            f"{label} mismatch: actual={number!r}, expected={expected!r}, tolerance={tolerance!r}"
        )


def _canonical_actor_rows(
    history: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[int, Mapping[str, Any]]:
    rows: dict[int, Mapping[str, Any]] = {}
    actor_id = binding["canonical_actor_id"]
    for row in history[binding["canonical_stream"]]:
        if str(row.get("id")) != actor_id:
            continue
        event = int(row["event_time_ns"])
        if event in rows:
            raise ParityContractError("canonical actor has duplicate observations")
        rows[event] = row
    if not rows:
        raise ParityContractError(
            f"canonical actor is absent: {binding['canonical_stream']}:{actor_id}"
        )
    return rows


def compare_candidate(
    export_value: Any,
    history_value: Any,
    binding_value: Any,
) -> dict[str, Any]:
    """Compare sealed evidence and return a permanently non-claimable report."""

    export = validate_export_document(export_value)
    sample = validate_history_sample(history_value)
    bindings = validate_bindings(binding_value)
    scene = export["scene"]
    history = sample["history"]
    projection = export["temporal_data"]["history_projection"]

    if len({scene["scene_id"], sample["scene_id"], bindings["scene_id"]}) != 1:
        raise ParityContractError(
            "scene identity differs across export, sample, and bindings"
        )
    if bindings["upstream_export_content_sha256"] != export["content_sha256"]:
        raise ParityContractError("bindings do not seal the upstream export")
    if bindings["history_input_sha256"] != sample["input_sha256"]:
        raise ParityContractError("bindings do not seal the HistoryOnly sample")
    if bindings["canonical_coordinate_frame"] != history["coordinate_frame"]:
        raise ParityContractError(
            "binding coordinate frame differs from HistoryOnly history"
        )

    time_bindings = bindings["timestamp_bindings"]
    if [item["native_timestamp"] for item in time_bindings] != scene[
        "native_history_timestamps"
    ]:
        raise ParityContractError(
            "native timestamp binding differs from upstream witness"
        )
    if [item["event_time_ns"] for item in time_bindings] != history["timestamps_ns"]:
        raise ParityContractError(
            "event-time binding differs from HistoryOnly timestamps"
        )

    actor_bindings = bindings["actor_bindings"]
    if len(actor_bindings) != len(scene["actors"]):
        raise ParityContractError(
            "actor bindings do not cover the complete upstream actor order"
        )
    for binding, actor in zip(actor_bindings, scene["actors"]):
        if (
            binding["upstream_index"] != actor["index"]
            or binding["upstream_actor_id"] != actor["actor_id"]
        ):
            raise ParityContractError("actor binding differs from upstream actor order")

    tolerance = float(bindings["absolute_tolerance"])
    actor_rows = [_canonical_actor_rows(history, binding) for binding in actor_bindings]
    actor_tags = [{str(row.get("tag")) for row in rows.values()} for rows in actor_rows]
    target_indices = [
        index for index, tags in enumerate(actor_tags) if "TARGET_AGENT" in tags
    ]
    av_indices = [index for index, tags in enumerate(actor_tags) if "AV" in tags]
    if len(target_indices) != 1:
        raise ParityContractError("bound actors must contain exactly one TARGET_AGENT")
    if len(av_indices) != 1:
        raise ParityContractError("bound actors must contain exactly one AV")

    agent_index = int(_descriptor_value(projection, "agent_index"))
    av_index = int(_descriptor_value(projection, "av_index"))
    if target_indices[0] != agent_index:
        raise ParityContractError(
            "TARGET_AGENT binding does not match upstream agent_index"
        )
    target_binding = actor_bindings[agent_index]
    if target_binding["canonical_actor_id"] != sample["target_id"]:
        raise ParityContractError(
            "HistoryOnly target_id differs from TARGET_AGENT binding"
        )
    if av_indices[0] != av_index:
        raise ParityContractError("AV binding does not match upstream av_index")

    final_event_time = history["timestamps_ns"][-1]
    av_rows = actor_rows[av_index]
    if set(av_rows) != set(history["timestamps_ns"]):
        raise ParityContractError(
            "bound AV must have one observation at every history timestamp"
        )
    av_final = av_rows[final_event_time]
    raw_origin_x = _numeric_text(av_final.get("x"), "final AV x")
    raw_origin_y = _numeric_text(av_final.get("y"), "final AV y")
    raw_theta = _numeric_text(av_final.get("theta"), "final AV theta")
    exported_origin_x = float(_tensor_at(projection["origin"], 0, 0))
    exported_origin_y = float(_tensor_at(projection["origin"], 0, 1))
    exported_theta = float(_tensor_at(projection["theta"]))
    _close(
        exported_origin_x,
        _float32(raw_origin_x, "final AV x"),
        tolerance,
        "origin[0,0] after float32 rounding",
    )
    _close(
        exported_origin_y,
        _float32(raw_origin_y, "final AV y"),
        tolerance,
        "origin[0,1] after float32 rounding",
    )
    _close(
        exported_theta,
        _float32(raw_theta, "final AV theta"),
        tolerance,
        "theta after float32 rounding",
    )
    cosine = math.cos(raw_theta)
    sine = math.sin(raw_theta)

    for actor_index, rows in enumerate(actor_rows):
        previous_raw_position: tuple[float, float] | None = None
        observed_world: list[tuple[float, float, Mapping[str, Any]]] = []
        for step, time_binding in enumerate(time_bindings):
            event_time = time_binding["event_time_ns"]
            row = rows.get(event_time)
            padded = bool(_tensor_at(projection["padding_mask"], actor_index, step))
            if row is None:
                if not padded:
                    raise ParityContractError(
                        "padding_mask marks an absent canonical observation valid"
                    )
            else:
                if padded:
                    raise ParityContractError(
                        "padding_mask hides a canonical observation"
                    )
                if (
                    decimal_token(str(row.get("timestamp")), "canonical row timestamp")
                    != time_binding["native_timestamp"]
                ):
                    raise ParityContractError(
                        "canonical raw timestamp differs from explicit binding"
                    )
                world_x = _numeric_text(row.get("x"), "canonical actor x")
                world_y = _numeric_text(row.get("y"), "canonical actor y")
                delta_x = world_x - raw_origin_x
                delta_y = world_y - raw_origin_y
                raw_position = (
                    delta_x * cosine + delta_y * sine,
                    -delta_x * sine + delta_y * cosine,
                )
                observed_world.append((world_x, world_y, row))
            if row is None:
                raw_position = (0.0, 0.0)
            expected_position = (
                _float32(raw_position[0], "upstream position x"),
                _float32(raw_position[1], "upstream position y"),
            )
            _close(
                _tensor_at(projection["positions"], actor_index, step, 0),
                expected_position[0],
                tolerance,
                f"positions[{actor_index},{step},0]",
            )
            _close(
                _tensor_at(projection["positions"], actor_index, step, 1),
                expected_position[1],
                tolerance,
                f"positions[{actor_index},{step},1]",
            )
            if step == 0 or padded or previous_raw_position is None:
                raw_delta = (0.0, 0.0)
            else:
                raw_delta = (
                    raw_position[0] - previous_raw_position[0],
                    raw_position[1] - previous_raw_position[1],
                )
            expected_delta = (
                _float32(raw_delta[0], "upstream displacement x"),
                _float32(raw_delta[1], "upstream displacement y"),
            )
            _close(
                _tensor_at(projection["x"], actor_index, step, 0),
                expected_delta[0],
                tolerance,
                f"x[{actor_index},{step},0]",
            )
            _close(
                _tensor_at(projection["x"], actor_index, step, 1),
                expected_delta[1],
                tolerance,
                f"x[{actor_index},{step},1]",
            )
            previous_padded = (
                True
                if step == 0
                else bool(_tensor_at(projection["padding_mask"], actor_index, step - 1))
            )
            expected_bos = (not padded) and (step == 0 or previous_padded)
            if (
                bool(_tensor_at(projection["bos_mask"], actor_index, step))
                is not expected_bos
            ):
                raise ParityContractError(
                    "bos_mask differs from canonical observation continuity"
                )
            previous_raw_position = None if padded else raw_position

        if not observed_world:
            raise ParityContractError("bound actor has no observed history")
        last_world_x, last_world_y, last_row = observed_world[-1]
        _close(
            _tensor_at(projection["last_positions"], actor_index, 0),
            _float32(last_world_x, "upstream last position x"),
            tolerance,
            f"last_positions[{actor_index},0]",
        )
        _close(
            _tensor_at(projection["last_positions"], actor_index, 1),
            _float32(last_world_y, "upstream last position y"),
            tolerance,
            f"last_positions[{actor_index},1]",
        )
        expected_actor_type = {
            "None": 0,
            "PEDESTRIAN": 1,
            "BICYCLE": 2,
            "VEHICLE": 3,
        }.get(str(last_row.get("type")))
        if expected_actor_type is None:
            raise ParityContractError(
                "canonical actor type is outside pinned upstream mapping"
            )
        if (
            int(_tensor_at(projection["actor_type"], actor_index))
            != expected_actor_type
        ):
            raise ParityContractError("actor_type differs from pinned upstream mapping")
        if len(observed_world) > 1:
            expected_angle = _numeric_text(
                last_row.get("theta"), "canonical actor theta"
            )
        else:
            expected_angle = 0.0
        _close(
            _tensor_at(projection["rotate_angles"], actor_index),
            _float32(expected_angle, "upstream rotate angle"),
            tolerance,
            f"rotate_angles[{actor_index}]",
        )

    report_without_hash = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "comparison_status": "candidate_checks_passed",
        "parity_verified": False,
        "scientific_claim_allowed": False,
        "scene_id": scene["scene_id"],
        "upstream_export_content_sha256": export["content_sha256"],
        "history_input_sha256": sample["input_sha256"],
        "verified_categories": [
            "pinned upstream source identity",
            "strict history-row schemas and future-field exclusion",
            "actor order and ego-road masks",
            "explicit native-to-event timestamp binding",
            "float64 world transform and field-specific float32 rounding",
            "history positions displacements padding and BOS masks",
            "actor type heading and last-position projection",
            "complete actor graph and source-derived association edges",
        ],
        "unresolved_categories": list(UNRESOLVED_CATEGORIES),
    }
    return {**report_without_hash, "content_sha256": sha256_json(report_without_hash)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-export", required=True)
    parser.add_argument("--history-sample", required=True)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = compare_candidate(
            load_json_file(Path(args.upstream_export), "upstream export"),
            load_json_file(Path(args.history_sample), "HistoryOnly sample"),
            load_json_file(Path(args.bindings), "parity bindings"),
        )
        output = Path(args.output)
        if output.exists() and (output.is_symlink() or output.is_dir()):
            raise ParityContractError("output must not be a symlink or directory")
        atomic_write(output, canonical_bytes(report))
    except ParityContractError as exc:
        raise SystemExit(f"V2X-Graph preprocessing comparison rejected: {exc}") from exc
    print(
        json.dumps(
            {"output": str(output), "sha256": sha256_file(output)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
