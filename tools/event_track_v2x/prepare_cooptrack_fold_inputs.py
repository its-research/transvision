#!/usr/bin/env python3
"""Copy an immutable SPD train camera view for a fold or full-train run.

This is input preparation, not a formal-training authorization. The upstream
label updater, frame-removal lists and occlusion interpolation are not run.
Only train metadata is read; held-out label/image payloads are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.development_split import (  # noqa: E402
    DevelopmentSplitManifestV1,
)
from transvision.models.event_track_v2x.wire import canonical_json_bytes  # noqa: E402


SIDES = ("vehicle-side", "infrastructure-side")
PAYLOAD_KEYS = frozenset({
    "image_path", "label_camera_std_path", "label_lidar_std_path",
    "calib_camera_intrinsic_path", "calib_lidar_to_camera_path",
    "calib_lidar_to_novatel_path", "calib_novatel_to_world_path",
    "calib_virtuallidar_to_camera_path", "calib_virtuallidar_to_world_path",
})


def safe_file(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if (not value or relative.is_absolute() or relative.as_posix() != value
            or ".." in relative.parts or "\\" in value):
        raise ValueError("invalid relative payload path")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("symlink input is forbidden")
    if not stat.S_ISREG(candidate.stat().st_mode):
        raise ValueError("input must be a regular file")
    return candidate


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def prepare(data_root: Path, split_path: Path, manifest_path: Path,
            fold_id: int | None, output: Path, *, cohort: str = "fit-fold") -> dict:
    if data_root.is_symlink() or output.is_symlink():
        raise ValueError("symlink roots are forbidden")
    data_root = data_root.resolve(strict=True)
    if output.exists():
        raise FileExistsError("output is create-once")
    output_resolved = output.resolve()
    if output_resolved == data_root or data_root in output_resolved.parents:
        raise ValueError("output must be outside the source dataset")
    manifest_raw = manifest_path.read_bytes()
    manifest = DevelopmentSplitManifestV1.from_mapping(json.loads(manifest_raw))
    split_raw = split_path.read_bytes()
    if digest(split_raw) != manifest.split_sha256:
        raise ValueError("official split hash mismatch")
    split = json.loads(split_raw)
    if set(split["batch_split"]["train"]) != set(manifest.sequence_ids):
        raise ValueError("official train sequences mismatch")
    if cohort == "fit-fold":
        if type(fold_id) is not int or not 0 <= fold_id < len(manifest.folds):
            raise ValueError("fold must be in 0..4")
        fit_sequences = manifest.folds[fold_id].fit_sequence_ids
        held_out_sequences = manifest.folds[fold_id].held_out_sequence_ids
    elif cohort == "full-train":
        if fold_id is not None:
            raise ValueError("full-train must not specify a fold")
        fit_sequences = manifest.sequence_ids
        held_out_sequences = ()
    else:
        raise ValueError("unknown training cohort")
    selected = set(fit_sequences)
    source_files: dict[str, Path] = {}
    projections: dict[str, bytes] = {}
    metadata_hashes = {}
    counts = {}
    for side in SIDES:
        raw = safe_file(data_root, f"{side}/data_info.json").read_bytes()
        rows = json.loads(raw)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("invalid side metadata")
        if {row["sequence_id"] for row in rows} != set(manifest.sequence_ids):
            raise ValueError("side metadata must contain only the complete official train")
        frames = [row["frame_id"] for row in rows]
        if len(set(frames)) != len(frames):
            raise ValueError("duplicate frame IDs")
        fit_rows = [row for row in rows if row["sequence_id"] in selected]
        if {row["sequence_id"] for row in fit_rows} != selected:
            raise ValueError("fit sequence missing from metadata")
        for row in fit_rows:
            if not {"image_path", "label_lidar_std_path"} <= row.keys():
                raise ValueError("required fit payload reference missing")
            for key in PAYLOAD_KEYS & row.keys():
                relative = f"{side}/{row[key]}"
                path = safe_file(data_root, relative)
                expected_suffix = ".jpg" if key == "image_path" else ".json"
                if path.suffix != expected_suffix:
                    raise ValueError("unexpected payload type")
                source_files[relative] = path
        projections[f"{side}/data_info.json"] = canonical_json_bytes(fit_rows)
        metadata_hashes[side] = digest(raw)
        counts[side] = len(fit_rows)
    output.mkdir(parents=True, exist_ok=False)
    inventory = []
    for relative, source in sorted(source_files.items()):
        raw = source.read_bytes()
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        if digest(target.read_bytes()) != digest(raw):
            raise ValueError("copied payload hash mismatch")
        inventory.append({"path": relative, "size_bytes": len(raw), "sha256": digest(raw)})
    fold_split = {"batch_split": {"train": list(fit_sequences), "val": [],
                                   "test": [], "test_A": []}}
    projections["fold-split.json"] = canonical_json_bytes(fold_split)
    for relative, raw in sorted(projections.items()):
        with (output / relative).open("xb") as stream:
            stream.write(raw)
        inventory.append({"path": relative, "size_bytes": len(raw), "sha256": digest(raw)})
    report = {
        "kind": "cooptrack_fit_input_preparation_v1", "schema_version": 1,
        "cohort": cohort, "fold_id": fold_id, "fit_sequence_ids": list(fit_sequences),
        "excluded_held_out_sequence_ids": list(held_out_sequences),
        "official_split_sha256": manifest.split_sha256,
        "development_manifest_sha256": digest(manifest_raw),
        "source_metadata_sha256": metadata_hashes, "frame_counts": counts,
        "payload_inventory": sorted(inventory, key=lambda item: item["path"]),
        "raw_labels_modified": False, "point_clouds_copied": False,
        "official_val_read": False, "test_or_test_A_read": False,
        "formal_training_authorized_by_this_artifact": False,
    }
    report["content_sha256"] = digest(canonical_json_bytes(report))
    with (output / "input-manifest.json").open("xb") as stream:
        stream.write(canonical_json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-split", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--cohort", choices=("fit-fold", "full-train"), default="fit-fold")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.data_root, args.official_split, args.development_manifest,
                     args.fold, args.output, cohort=args.cohort)
    print(json.dumps({key: report[key] for key in
                     ("kind", "fold_id", "frame_counts", "content_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
