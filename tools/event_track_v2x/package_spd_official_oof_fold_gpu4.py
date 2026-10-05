#!/usr/bin/env python3
"""Package one verified SPD canonical OOF fit fold for isolated GPU training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

from run_cooptrack_official_oof_gpu4 import validate_archive, validate_cohort


ROOT = Path("/Volumes/Data/test/recover-before-fuse")
CANONICAL = ROOT / "receipts/spd-official-train-canonical-development-fivefold-20260930.json"
CANONICAL_SHA256 = "1c9d23f291be09e686a2bdcc60ef19ea58a5c2894978cc5f6a6b2a81f43c2bd6"
OFFICIAL_SPLIT_SHA256 = "0a59332221f3618639f34fb70678d03c5221ab80fd69f2cf41acffca0d8859fd"
SHARED = (
    ("resnet50-0676ba61.pth", 102530333,
     "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a",
     "9a7e7a9213954b57a35403f777e74561"),
    ("runtime.tar.gz", 2504425555,
     "b6b39c66eec6921c0f0f51d2b9cd4e5571fcafdc44d3605112dfb349af8fab32",
     "9a7e7a9213954b57a35403f777e74561"),
    ("source.tar.gz", 545130,
     "33250c65975d47e1aaf3f86e6c8553c4384fd6b38b8fb05f572a4192870bafbc",
     "5da9693dfce54d85b57ab0ca663dbb41"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def verify_inventory(root: Path, items: list[dict], allowed_extras: set[str]) -> None:
    expected = {}
    for item in items:
        relative = item["path"]
        if relative in expected or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("unsafe or duplicate view inventory path")
        expected[relative] = item
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink is forbidden in OOF package")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            observed.add(relative)
            if relative in expected:
                item = expected[relative]
                if path.stat().st_size != item.get("size_bytes", item.get("bytes")) \
                        or sha256(path) != item["sha256"]:
                    raise ValueError("fit file differs from frozen inventory")
    if observed != set(expected) | allowed_extras:
        raise ValueError("view contains missing or extra payload files")


def package(fold_id: int, view: Path, converted: Path, output: Path) -> dict:
    if type(fold_id) is not int or not 0 <= fold_id < 5:
        raise ValueError("fold_id must be in 0..4")
    if output.exists() or output.is_symlink():
        raise FileExistsError("fold package is create-once")
    if sha256(CANONICAL) != CANONICAL_SHA256:
        raise ValueError("canonical fivefold manifest byte identity differs")
    canonical = json.loads(CANONICAL.read_bytes())
    fold = canonical["folds"][fold_id]
    accepted = json.loads((view / "acceptance.json").read_bytes())
    input_path = view / "input-manifest.json"
    inputs = json.loads(input_path.read_bytes())
    if (accepted.get("status") != "fit_input_bytes_verified"
            or accepted.get("fold_id") != fold_id
            or accepted.get("input_manifest_sha256") != sha256(input_path)
            or inputs.get("cohort") != "fit-fold"
            or inputs.get("fold_id") != fold_id
            or inputs.get("fit_sequence_ids") != fold["fit_sequence_ids"]
            or inputs.get("excluded_held_out_sequence_ids") != fold["held_out_sequence_ids"]
            or inputs.get("official_split_sha256") != OFFICIAL_SPLIT_SHA256
            or inputs.get("development_manifest_sha256") != CANONICAL_SHA256):
        raise ValueError("fit input view is not accepted canonical fold")
    verify_inventory(view, inputs["payload_inventory"],
                     {"input-manifest.json", "acceptance.json"})
    conversion_path = converted / "conversion-manifest.json"
    conversion_raw = json.loads(conversion_path.read_bytes())
    conversion = dict(conversion_raw)
    content_sha = conversion.pop("content_sha256")
    if (hashlib.sha256(canonical_json(conversion)).hexdigest() != content_sha
            or conversion.get("cohort") != "fit-fold"
            or conversion.get("fold_id") != fold_id
            or conversion.get("fit_sequence_ids") != fold["fit_sequence_ids"]
            or conversion.get("input_manifest_sha256") != sha256(input_path)
            or conversion.get("removed_frames") != 0
            or conversion.get("interpolated_annotations") != 0
            or conversion.get("raw_labels_modified") is not False
            or conversion.get("forecasting_targets_generated") is not False):
        raise ValueError("fold conversion is not accepted")
    verify_inventory(converted, conversion["inventory"], {"conversion-manifest.json"})
    manifest = {
        "kind": "eventtrack_spd_official_oof_fold_package_v1",
        "cohort": "official-oof-fold-fit",
        "fold_id": fold_id,
        "official_train_sequence_ids": canonical["sequence_ids"],
        "fit_sequence_ids": fold["fit_sequence_ids"],
        "held_out_sequence_ids": fold["held_out_sequence_ids"],
        "train_sequence_count": len(fold["fit_sequence_ids"]),
        "official_split_sha256": OFFICIAL_SPLIT_SHA256,
        "canonical_fivefold_manifest_sha256": CANONICAL_SHA256,
        "input_manifest_sha256": sha256(input_path),
        "conversion_manifest_sha256": sha256(conversion_path),
        "val_or_test_included": False,
        "pretrained_origin": "ImageNet R50",
        "velocity_loss_guard_source_sha256": SHARED[2][2],
    }
    validate_cohort(manifest)
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / "train-inputs.tar.gz"
    expected_files = len(inputs["payload_inventory"]) + 2 + len(conversion["inventory"]) + 1
    packed_files = 0
    started = time.monotonic()

    def progress(member: tarfile.TarInfo) -> tarfile.TarInfo:
        nonlocal packed_files
        if member.isfile():
            packed_files += 1
            if packed_files % 10000 == 0:
                elapsed = max(time.monotonic() - started, 0.001)
                eta = max(expected_files - packed_files, 0) * elapsed / packed_files
                print(f"SPD OOF fold {fold_id} package {packed_files}/{expected_files} files "
                      f"ETA={eta:.1f}s", flush=True)
        return member

    with tarfile.open(archive_path, mode="x:gz", compresslevel=5) as archive:
        archive.add(view, arcname="inputs", filter=progress)
        archive.add(converted, arcname="converted", filter=progress)
    if packed_files != expected_files:
        raise ValueError("packaged file count differs from verified fold inventories")
    validate_archive(archive_path, ["inputs", "converted"])
    manifest["inventory"] = [
        {"path": name, "bytes": size, "sha256": digest,
         "artifact_task_id": task_id}
        for name, size, digest, task_id in SHARED
    ] + [{"path": "train-inputs.tar.gz", "bytes": archive_path.stat().st_size,
          "sha256": sha256(archive_path)}]
    with (output / "package-manifest.json").open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"fold_id": fold_id,
                      "package_manifest_sha256": sha256(output / "package-manifest.json"),
                      "train_archive_sha256": manifest["inventory"][-1]["sha256"]}),
          flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.fold, args.view, args.converted, args.output)
