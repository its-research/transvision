#!/usr/bin/env python3
"""Export a history-only witness from pinned V2X-Graph ``TemporalData``.

Run this file inside the separately installed, trusted V2X-Graph environment
at the exact pinned commit.  ``torch.load`` uses Python pickle semantics; never
run it on an untrusted ``.pt`` file.  PyTorch and PyG are imported lazily so the
local verifier and its tests remain pure standard library.

The exporter does not translate a HistoryOnly sample and does not claim
preprocessing parity.  It serializes what the pinned upstream produced,
reconstructs only the actor/timestamp ordering stated by that source, withholds
future values, and seals every input byte for later independent comparison.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import platform
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import parity
except ImportError:  # direct execution in a copied upstream environment
    import parity  # type: ignore[no-redef]


MINIMUM_TRAJECTORY_COLUMNS = frozenset(
    {"timestamp", "id", "type", "tag", "x", "y", "theta"}
)
MINIMUM_ASSOCIATION_COLUMNS = frozenset(
    {"timestamp", "id", "tag", "ego_side_id", "coop_side_id"}
)


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise parity.ParityContractError(
            f"{label} must be a non-empty non-symlink regular file"
        )
    return path.resolve()


def _run_git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise parity.ParityContractError(
            "cannot verify pinned upstream Git tree"
        ) from exc
    return completed.stdout.strip()


def validate_upstream_root(root: Path) -> dict[str, str]:
    root = root.resolve()
    if not root.is_dir():
        raise parity.ParityContractError("upstream root is not a directory")
    revision = _run_git(root, "rev-parse", "HEAD")
    if revision != parity.UPSTREAM_COMMIT:
        raise parity.ParityContractError("upstream HEAD does not match pinned commit")
    source_hashes: dict[str, str] = {}
    for relative, expected in parity.UPSTREAM_SOURCE_SHA256.items():
        source = _regular_file(root / relative, f"upstream source {relative}")
        actual = parity.sha256_file(source)
        if actual != expected:
            raise parity.ParityContractError(
                f"upstream source bytes drifted: {relative}"
            )
        source_hashes[relative] = actual
    return source_hashes


def _read_snapshot(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise parity.ParityContractError(f"cannot read {label}") from exc
    if not data:
        raise parity.ParityContractError(f"{label} contains no bytes")
    return data, {
        "basename": path.name,
        "size": len(data),
        "sha256": parity.sha256_bytes(data),
    }


def _read_csv_snapshot(
    data: bytes, required: frozenset[str], label: str
) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise parity.ParityContractError(
                f"{label} is missing required columns: {missing}"
            )
        rows = [dict(row) for row in reader]
    except parity.ParityContractError:
        raise
    except (UnicodeError, csv.Error) as exc:
        raise parity.ParityContractError(f"cannot parse {label}") from exc
    if not rows:
        raise parity.ParityContractError(f"{label} contains no rows")
    return rows


def _first_seen(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def build_scene_witness(
    ego_rows: Sequence[Mapping[str, str]],
    road_rows: Sequence[Mapping[str, str]],
    association_rows: Sequence[Mapping[str, str]],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reproduce only ordering rules stated by pinned ``tfd_dataset.py``."""

    native_timestamps = sorted(
        {
            parity.decimal_token(row["timestamp"], "trajectory timestamp")
            for row in [*ego_rows, *road_rows]
        },
        key=Decimal,
    )
    if len(native_timestamps) != parity.HISTORY_STEPS + parity.FUTURE_STEPS:
        raise parity.ParityContractError(
            "pinned V2X-Graph scene must expose exactly 100 trajectory timestamps"
        )
    history_times = native_timestamps[: parity.HISTORY_STEPS]
    history_set = set(history_times)

    def history_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
        return [
            row
            for row in rows
            if parity.decimal_token(row["timestamp"], "trajectory timestamp")
            in history_set
        ]

    ego_history = history_rows(ego_rows)
    road_history = history_rows(road_rows)
    ego_ids = _first_seen(
        [parity.identity_token(row["id"], "ego actor id") for row in ego_history]
    )
    road_ids = _first_seen(
        [parity.identity_token(row["id"], "road actor id") for row in road_history]
    )
    if not ego_ids or not road_ids:
        raise parity.ParityContractError(
            "both ego and road actor sets must be non-empty"
        )
    overlap = set(ego_ids).intersection(road_ids)
    if overlap:
        raise parity.ParityContractError(
            f"pinned upstream actor sets overlap across sides: {sorted(overlap)}"
        )
    try:
        ordered = Decimal(ego_ids[-1]) < Decimal(road_ids[0])
    except InvalidOperation:
        ordered = ego_ids[-1] < road_ids[0]
    if not ordered:
        raise parity.ParityContractError(
            "raw actor IDs violate the pinned ego-last < road-first assertion"
        )

    actors: list[dict[str, Any]] = []
    for side, rows, actor_ids in (
        ("ego", ego_history, ego_ids),
        ("road", road_history, road_ids),
    ):
        for actor_id in actor_ids:
            tags = sorted(
                {
                    str(row["tag"])
                    for row in rows
                    if parity.identity_token(row["id"], "actor id") == actor_id
                    and str(row["tag"])
                }
            )
            if not tags:
                raise parity.ParityContractError("actor witness has no tag")
            actors.append(
                {
                    "index": len(actors),
                    "actor_id": actor_id,
                    "side": side,
                    "tags": tags,
                }
            )
    if sum("AV" in actor["tags"] for actor in actors) != 1:
        raise parity.ParityContractError("actor witness must contain exactly one AV")
    if sum("TARGET_AGENT" in actor["tags"] for actor in actors) != 1:
        raise parity.ParityContractError(
            "actor witness must contain exactly one TARGET_AGENT"
        )

    non_av_associations = [row for row in association_rows if row["tag"] != "AV"]
    association_timestamps = sorted(
        {
            parity.decimal_token(row["timestamp"], "association timestamp")
            for row in non_av_associations
        },
        key=Decimal,
    )
    association_history_times = association_timestamps[: parity.HISTORY_STEPS]
    if association_history_times != history_times:
        raise parity.ParityContractError(
            "association and trajectory history timestamp windows differ"
        )

    normalized_associations: list[dict[str, Any]] = []
    association_history_set = set(association_history_times)
    for row in non_av_associations:
        timestamp = parity.decimal_token(row["timestamp"], "association timestamp")
        if timestamp not in association_history_set:
            continue

        def association_identity(key: str) -> str | None:
            raw = str(row[key]).strip()
            if not raw:
                return None
            value = parity.identity_token(raw, f"association {key}")
            return None if value == "-1" else value

        normalized_associations.append(
            {
                "timestamp": timestamp,
                "id": parity.identity_token(row["id"], "association id"),
                "tag": parity._string(row["tag"], "association tag"),
                "ego_side_id": association_identity("ego_side_id"),
                "coop_side_id": association_identity("coop_side_id"),
            }
        )
    if not normalized_associations:
        raise parity.ParityContractError(
            "association source contains no history-window evidence"
        )
    return history_times, actors, normalized_associations


def _tensor_core(tensor: Any, label: str) -> dict[str, Any]:
    required = ("detach", "cpu", "contiguous", "reshape", "tolist", "shape", "dtype")
    if any(not hasattr(tensor, name) for name in required):
        raise parity.ParityContractError(f"{label} is not a tensor")
    try:
        detached = tensor.detach().cpu().contiguous()
        shape = [int(item) for item in detached.shape]
        values = detached.reshape(-1).tolist()
    except Exception as exc:  # upstream tensor libraries define their own exceptions
        raise parity.ParityContractError(f"cannot serialize {label}") from exc
    if not isinstance(values, list):
        values = [values]
    for index, item in enumerate(values):
        if isinstance(item, float) and not math.isfinite(item):
            raise parity.ParityContractError(f"{label}[{index}] is non-finite")
        if not isinstance(item, (bool, int, float)):
            raise parity.ParityContractError(f"{label}[{index}] is not a JSON scalar")
    return {"dtype": str(detached.dtype), "shape": shape, "values": values}


def tensor_descriptor(tensor: Any, label: str) -> dict[str, Any]:
    core = _tensor_core(tensor, label)
    return {"kind": "tensor", **core, "sha256": parity.sha256_json(core)}


def withheld_tensor(tensor: Any | None, label: str) -> dict[str, Any]:
    if tensor is None:
        return {
            "present": False,
            "dtype": None,
            "shape": None,
            "canonical_values_sha256": None,
        }
    core = _tensor_core(tensor, label)
    return {
        "present": True,
        "dtype": core["dtype"],
        "shape": core["shape"],
        "canonical_values_sha256": parity.sha256_json(core),
    }


def _scalar(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, bool):
        raise parity.ParityContractError(f"{label} cannot be boolean")
    if isinstance(value, int):
        return parity.scalar_descriptor("int", value)
    if isinstance(value, str) and value:
        return parity.scalar_descriptor("string", value)
    raise parity.ParityContractError(f"{label} has unsupported scalar type")


def _data_keys(data: Any) -> list[str]:
    try:
        keys = data.keys
        values = keys() if callable(keys) else keys
        result = sorted(str(item) for item in values)
    except Exception as exc:
        raise parity.ParityContractError(
            "cannot enumerate TemporalData fields"
        ) from exc
    if not result or result != sorted(set(result)):
        raise parity.ParityContractError("TemporalData keys are empty or duplicated")
    return result


def _attribute(data: Any, name: str) -> Any:
    try:
        value = getattr(data, name)
    except Exception as exc:
        raise parity.ParityContractError(f"TemporalData is missing {name}") from exc
    if value is None:
        raise parity.ParityContractError(f"TemporalData field {name} is null")
    return value


def build_temporal_projection(data: Any) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    scalar_fields = {
        "num_nodes",
        "num_car_actors",
        "num_road_actors",
        "seq_id",
        "av_index",
        "agent_index",
        "city",
    }
    for name in sorted(parity.PROJECTED_FIELDS):
        value = _attribute(data, name)
        if name == "positions":
            value = value[:, : parity.HISTORY_STEPS]
        elif name == "padding_mask":
            value = value[:, : parity.HISTORY_STEPS]
        if name in scalar_fields:
            projection[name] = _scalar(value, f"TemporalData {name}")
        else:
            projection[name] = tensor_descriptor(value, f"TemporalData {name}")
    positions = _attribute(data, "positions")
    if (
        len(positions.shape) != 3
        or int(positions.shape[1]) != parity.HISTORY_STEPS + parity.FUTURE_STEPS
    ):
        raise parity.ParityContractError("TemporalData positions must have 100 steps")
    y = getattr(data, "y", None)
    return {
        "field_inventory": _data_keys(data),
        "history_projection": projection,
        "withheld_future_fields": {
            "positions_future": withheld_tensor(
                positions[:, parity.HISTORY_STEPS :], "TemporalData future positions"
            ),
            "y": withheld_tensor(y, "TemporalData y"),
        },
    }


def _verify_witness_against_projection(
    temporal: Mapping[str, Any], actors: Sequence[Mapping[str, Any]]
) -> None:
    projection = temporal["history_projection"]
    num_nodes = projection["num_nodes"]["value"]
    num_car = projection["num_car_actors"]["value"]
    num_road = projection["num_road_actors"]["value"]
    if num_nodes != len(actors):
        raise parity.ParityContractError(
            "processed num_nodes differs from raw actor witness"
        )
    if num_car != sum(actor["side"] == "ego" for actor in actors):
        raise parity.ParityContractError(
            "processed num_car_actors differs from raw actor witness"
        )
    if num_road != sum(actor["side"] == "road" for actor in actors):
        raise parity.ParityContractError(
            "processed num_road_actors differs from raw actor witness"
        )
    av_indices = [actor["index"] for actor in actors if "AV" in actor["tags"]]
    target_indices = [
        actor["index"] for actor in actors if "TARGET_AGENT" in actor["tags"]
    ]
    if projection["av_index"]["value"] != av_indices[0]:
        raise parity.ParityContractError(
            "processed av_index differs from raw actor witness"
        )
    if projection["agent_index"]["value"] != target_indices[0]:
        raise parity.ParityContractError(
            "processed agent_index differs from raw actor witness"
        )


def export_document(args: argparse.Namespace) -> dict[str, Any]:
    upstream_root = Path(args.upstream_root).resolve()
    source_hashes = validate_upstream_root(upstream_root)
    processed = _regular_file(Path(args.processed), "processed TemporalData")
    ego_raw = _regular_file(Path(args.ego_raw), "upstream ego raw CSV")
    road_raw = _regular_file(Path(args.road_raw), "upstream road raw CSV")
    association_raw = _regular_file(
        Path(args.association_raw), "upstream association raw CSV"
    )
    scene_id = parity._string(args.scene_id, "scene_id")
    if processed.stem != scene_id:
        raise parity.ParityContractError(
            "processed filename stem must equal the declared scene_id"
        )
    for raw_path in (ego_raw, road_raw, association_raw):
        if raw_path.stem != scene_id:
            raise parity.ParityContractError(
                "every upstream CSV filename stem must equal scene_id"
            )

    processed_bytes, processed_seal = _read_snapshot(
        processed, "processed TemporalData"
    )
    ego_bytes, ego_seal = _read_snapshot(ego_raw, "upstream ego raw CSV")
    road_bytes, road_seal = _read_snapshot(road_raw, "upstream road raw CSV")
    association_bytes, association_seal = _read_snapshot(
        association_raw, "upstream association raw CSV"
    )
    ego_rows = _read_csv_snapshot(
        ego_bytes, MINIMUM_TRAJECTORY_COLUMNS, "upstream ego raw CSV"
    )
    road_rows = _read_csv_snapshot(
        road_bytes, MINIMUM_TRAJECTORY_COLUMNS, "upstream road raw CSV"
    )
    association_rows = _read_csv_snapshot(
        association_bytes,
        MINIMUM_ASSOCIATION_COLUMNS,
        "upstream association raw CSV",
    )
    history_times, actors, normalized_associations = build_scene_witness(
        ego_rows, road_rows, association_rows
    )

    try:
        import torch
        import torch_geometric
    except ImportError as exc:
        raise parity.ParityContractError(
            "exporter must run inside the pinned V2X-Graph PyTorch/PyG environment"
        ) from exc
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    try:
        # This is intentionally limited to an operator-designated trusted file.
        data = torch.load(io.BytesIO(processed_bytes), map_location="cpu")
    except Exception as exc:
        raise parity.ParityContractError(
            "cannot load trusted processed TemporalData"
        ) from exc
    temporal = build_temporal_projection(data)
    _verify_witness_against_projection(temporal, actors)
    if validate_upstream_root(upstream_root) != source_hashes:
        raise parity.ParityContractError(
            "upstream source identity changed during export"
        )

    document_without_hash = {
        "schema_version": parity.SCHEMA_VERSION,
        "contract_id": parity.CONTRACT_ID,
        "export_kind": parity.EXPORT_KIND,
        "claim_scope": "diagnostic_preprocessing_evidence_only",
        "scientific_claim_allowed": False,
        "upstream": {
            "repository": parity.UPSTREAM_REPOSITORY,
            "revision": parity.UPSTREAM_COMMIT,
            "source_files": source_hashes,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_geometric": str(torch_geometric.__version__),
        },
        "scene": {
            "scene_id": scene_id,
            "split": args.split,
            "processed_file": processed_seal,
            "inputs": {
                "ego_raw": ego_seal,
                "road_raw": road_seal,
                "association_raw": association_seal,
            },
            "history_steps": parity.HISTORY_STEPS,
            "future_steps": parity.FUTURE_STEPS,
            "native_history_timestamps": history_times,
            "actors": actors,
            "association_rows": normalized_associations,
        },
        "temporal_data": temporal,
        "exporter": {
            "basename": Path(__file__).name,
            "sha256": parity.sha256_file(Path(__file__)),
            "contract_sha256": parity.sha256_file(Path(parity.__file__)),
        },
    }
    document = {
        **document_without_hash,
        "content_sha256": parity.sha256_json(document_without_hash),
    }
    parity.validate_export_document(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--ego-raw", required=True)
    parser.add_argument("--road-raw", required=True)
    parser.add_argument("--association-raw", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--split", choices=("sample", "train", "val"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        output = Path(args.output)
        if output.exists() and (output.is_symlink() or output.is_dir()):
            raise parity.ParityContractError(
                "output must not be a symlink or directory"
            )
        document = export_document(args)
        parity.atomic_write(output, parity.canonical_bytes(document))
    except parity.ParityContractError as exc:
        raise SystemExit(f"V2X-Graph upstream export rejected: {exc}") from exc
    print(
        json.dumps(
            {"output": str(output), "sha256": parity.sha256_file(output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
