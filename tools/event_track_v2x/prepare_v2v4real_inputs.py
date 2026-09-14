#!/usr/bin/env python3
"""Copy native V2V4Real LiDAR and pose-only inputs into a GT-free directory.

Preparation only: does not run a detector, choose a class mapping, manufacture
source timestamps, certify official membership, or evaluate tracking metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transvision.models.event_track_v2x.v2v4real_inputs import (
    MAX_YAML_BYTES, OFFICIAL_SOURCE_COMMIT, POSE_CONVENTION, PROJECTION_KIND,
    V2V4RealInputError, load_raw_yaml, pose_projection,
)

def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise V2V4RealInputError(f"not a regular file: {path}")
    return info


def _stamp(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read(path: Path, limit: int) -> bytes:
    before = _regular(path)
    if before.st_size > limit:
        raise V2V4RealInputError(f"metadata too large: {path}")
    with path.open("rb") as stream:
        if _stamp(os.fstat(stream.fileno())) != _stamp(before):
            raise V2V4RealInputError(f"source changed before read: {path}")
        raw = stream.read(limit + 1)
    if len(raw) > limit or _stamp(_regular(path)) != _stamp(before):
        raise V2V4RealInputError(f"source changed during read: {path}")
    return raw


def _copy_pcd(source: Path, target: Path) -> dict[str, Any]:
    before = _regular(source)
    if before.st_size == 0:
        raise V2V4RealInputError(f"empty point cloud: {source}")
    digest = hashlib.sha256()
    size = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as inp, target.open("xb") as out:
        if _stamp(os.fstat(inp.fileno())) != _stamp(before):
            raise V2V4RealInputError(f"source changed before copy: {source}")
        while chunk := inp.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            out.write(chunk)
    if size != before.st_size or _stamp(_regular(source)) != _stamp(before):
        raise V2V4RealInputError(f"source changed during copy: {source}")
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def inventory_native_root(source_root: Path) -> dict[str, Any]:
    """List exact two-CAV frame keys; no YAML/PCD payload is opened here."""
    source_root = Path(source_root).absolute()
    if source_root.is_symlink() or not source_root.is_dir():
        raise V2V4RealInputError("source root must be an existing non-symlink directory")
    # Audit the whole selected split, not its siblings. Never silently follow a
    # camera/metadata symlink into another split or a caller's private directory.
    for folder, directories, files in os.walk(source_root, followlinks=False):
        for name in (*directories, *files):
            path = Path(folder) / name
            mode = path.lstat().st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise V2V4RealInputError(f"unsupported filesystem entry: {path}")
    sequences = {}
    ignored = []
    for sequence in sorted(source_root.iterdir()):
        if not sequence.is_dir():
            ignored.append(sequence.relative_to(source_root).as_posix())
            continue
        cavs = sorted(item for item in sequence.iterdir() if item.is_dir())
        if len(cavs) != 2 or any(not re.fullmatch(r"[0-9]+", c.name) for c in cavs):
            raise V2V4RealInputError(f"expected exactly two numeric CAV directories: {sequence}")
        frame_keys = None
        for cav in cavs:
            entries = list(cav.iterdir())
            frames = sorted(p.stem for p in entries if p.suffix == ".yaml"
                            and "additional" not in p.name and "camera_gt" not in p.name)
            if (not frames or any(not re.fullmatch(r"[0-9]+", key) for key in frames)
                    or len({len(key) for key in frames}) != 1):
                raise V2V4RealInputError(f"invalid or empty native frame-key sequence: {cav}")
            pcds = sorted(p.stem for p in entries if p.suffix == ".pcd")
            if pcds != frames:
                raise V2V4RealInputError(f"YAML/PCD frame coverage mismatch: {cav}")
            if frame_keys is not None and frames != frame_keys:
                raise V2V4RealInputError(f"CAV frame coverage mismatch: {sequence}")
            frame_keys = frames
            used = {key + suffix for key in frames for suffix in (".yaml", ".pcd")}
            ignored.extend(p.relative_to(source_root).as_posix() for p in entries if p.name not in used)
        ignored.extend(p.relative_to(source_root).as_posix()
                       for p in sequence.iterdir() if not p.is_dir())
        sequences[sequence.name] = {"cav_ids": [c.name for c in cavs], "frame_keys": frame_keys}
    if not sequences:
        raise V2V4RealInputError("no native two-CAV sequences found")
    return {"sequences": sequences, "ignored_entries": sorted(ignored),
            "paired_frame_count": sum(len(s["frame_keys"]) for s in sequences.values()),
            "source_frame_count": 2 * sum(len(s["frame_keys"]) for s in sequences.values())}


def prepare_native_inputs(
    source_root: Path, output: Path, *, dataset_split: str,
    ego_agents: dict[str, str], source_evidence: Path,
) -> dict[str, Any]:
    """Create a new projection; existing outputs are never intentionally replaced.

    ``source_evidence`` is a locally reviewed provenance document whose bytes are
    retained, NOT a magic official-split certificate. This tool only binds it.
    Raw YAML (which can contain GT) is parsed by this offline preparer. Only pose
    and copied PCD enter ``inputs/``; hashes/source paths stay in ``audit.json``.
    """
    if dataset_split not in ("train", "test"):
        raise V2V4RealInputError("native split must be explicit train or test, never a val alias")
    source_root, output = Path(source_root).absolute(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise V2V4RealInputError("output already exists; choose a fresh output directory")
    source_real, output_real = source_root.resolve(), output.resolve()
    if source_real in output_real.parents or output_real in source_real.parents or source_real == output_real:
        raise V2V4RealInputError("source and output trees must be disjoint")
    inventory = inventory_native_root(source_root)
    sequences = inventory["sequences"]
    if (not isinstance(ego_agents, dict) or set(ego_agents) != set(sequences)
            or any(type(ego_agents[s]) is not str or ego_agents[s] not in sequences[s]["cav_ids"]
                   for s in sequences)):
        raise V2V4RealInputError("explicit ego CAV mapping must cover every sequence exactly")
    evidence = _read(Path(source_evidence), MAX_YAML_BYTES)
    if not evidence.strip():
        raise V2V4RealInputError("source provenance evidence must not be empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.preparing-", dir=output.parent))
    sources = []
    source_stamps = {}
    records = []
    try:
        for sequence_id, sequence in sequences.items():
            for ordinal, key in enumerate(sequence["frame_keys"]):
                for cav_id in sequence["cav_ids"]:
                    relative = Path(sequence_id) / cav_id / key
                    yaml_path = source_root / relative.with_suffix(".yaml")
                    pcd_source = source_root / relative.with_suffix(".pcd")
                    for path in (yaml_path, pcd_source):
                        source_stamps[path] = _stamp(_regular(path))
                    raw = _read(yaml_path, MAX_YAML_BYTES)
                    pose = pose_projection(load_raw_yaml(raw))
                    pcd_path = Path("pcd") / relative.with_suffix(".pcd")
                    pcd = _copy_pcd(source_root / relative.with_suffix(".pcd"), staging / "inputs" / pcd_path)
                    records.append({
                        "sequence_id": sequence_id, "frame_key": key, "frame_ordinal": ordinal,
                        "cav_id": cav_id, "is_ego": cav_id == ego_agents[sequence_id],
                        "pcd_path": pcd_path.as_posix(), "pcd_sha256": pcd["sha256"],
                        "pcd_size_bytes": pcd["size_bytes"], **pose,
                    })
                    sources.append({"yaml_path": relative.with_suffix(".yaml").as_posix(),
                                    "yaml_sha256": _digest(raw), "yaml_size_bytes": len(raw),
                                    "pcd_path": relative.with_suffix(".pcd").as_posix(), **pcd})
        if inventory_native_root(source_root) != inventory:
            raise V2V4RealInputError("source frame inventory changed during preparation")
        if any(_stamp(_regular(path)) != stamp for path, stamp in source_stamps.items()):
            raise V2V4RealInputError("source payload changed during preparation")
        frames = b"".join(_json_bytes(record) for record in records)
        (staging / "inputs" / "frames.jsonl").write_bytes(frames)
        manifest = {
            "kind": PROJECTION_KIND, "dataset": "V2V4Real", "dataset_split": dataset_split,
            "time_basis": "ordinal-only-no-clock", "pose_convention": POSE_CONVENTION,
            "source_frame_count": len(records), "paired_frame_count": inventory["paired_frame_count"],
            "sequence_count": len(sequences), "ego_agents": ego_agents,
            "frames_path": "frames.jsonl", "frames_sha256": _digest(frames),
            "gt_in_projection": False, "detection_cache_created": False,
            "official_split_membership_verified": False, "paper_eligible": False,
        }
        manifest_bytes = _json_bytes(manifest)
        (staging / "inputs" / "manifest.json").write_bytes(manifest_bytes)
        (staging / "source-evidence.bin").write_bytes(evidence)
        receipt = {
            "kind": "v2v4real_input_preparation_receipt_v1", "dataset_split": dataset_split,
            "input_manifest_sha256": _digest(manifest_bytes),
            "source_evidence_sha256": _digest(evidence), "source_root": str(source_real),
            "native_source_commit": OFFICIAL_SOURCE_COMMIT,
            "raw_yaml_parsed_by_preparer": True, "annotation_records_exported": False,
            "pcd_structure_validated": False,
            "detector_or_tracker_run": False, "performance_viewed": False,
            "official_split_membership_verified": False, "paper_eligible": False,
            "inventory": inventory, "sources": sources,
        }
        (staging / "audit.json").write_bytes(_json_bytes(receipt))
        if output.exists() or output.is_symlink():
            raise V2V4RealInputError("output appeared during preparation; not replacing it")
        staging.rename(output)
        return receipt
    except BaseException:
        # Only the unique directory created by this invocation is removed.
        # The original data, evidence, and any caller-owned output remain intact.
        shutil.rmtree(staging)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true", help="No payload reads or output writes")
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--ego-agents", type=Path, help="JSON object: exact sequence name to explicit CAV ID string")
    parser.add_argument("--source-evidence", type=Path, help="Locally reviewed official-download/split provenance document")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.inventory_only:
            if any((args.split, args.ego_agents, args.source_evidence, args.output)):
                parser.error("inventory-only does not accept preparation arguments")
            result = inventory_native_root(args.source_root)
        else:
            if not all((args.split, args.ego_agents, args.source_evidence, args.output)):
                parser.error("preparation requires split, ego-agents, source-evidence and output")
            def unique_pairs(pairs):
                value = {}
                for key, item in pairs:
                    if key in value:
                        raise V2V4RealInputError("duplicate ego mapping key")
                    value[key] = item
                return value
            ego = json.loads(_read(args.ego_agents, MAX_YAML_BYTES), object_pairs_hook=unique_pairs)
            receipt = prepare_native_inputs(args.source_root, args.output, dataset_split=args.split,
                                           ego_agents=ego, source_evidence=args.source_evidence)
            result = {key: value for key, value in receipt.items() if key not in ("sources", "inventory")}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except (V2V4RealInputError, OSError, ValueError) as exc:
        print(f"V2V4Real preparation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
