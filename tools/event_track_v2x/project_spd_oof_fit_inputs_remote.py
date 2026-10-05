#!/usr/bin/env python3
"""Project verified ClearML SPD raw bytes into canonical OOF fit views on Linux.

The small projected metadata is transferred from the independently accepted
local fit views. Payloads are read only from the verified ClearML raw root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import time


EXPECTED_INVENTORY_SHA256 = "e04f07ad6e4eeb7a505ca33eed31bafd27d4157e40cc34f73e9e961a07136ce0"
EXPECTED_SPLIT_SHA256 = "0a59332221f3618639f34fb70678d03c5221ab80fd69f2cf41acffca0d8859fd"
EXPECTED_MANIFEST_SHA256 = "1c9d23f291be09e686a2bdcc60ef19ea58a5c2894978cc5f6a6b2a81f43c2bd6"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or p.as_posix() != relative or ".." in p.parts or "\\" in relative:
        raise ValueError("unsafe relative path")
    target = root
    for component in p.parts:
        target = target / component
        if target.is_symlink():
            raise ValueError("symlink source path")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("fold output is create-once")
    if digest(args.inventory) != EXPECTED_INVENTORY_SHA256:
        raise ValueError("ClearML verified inventory differs")
    inventory = json.loads(args.inventory.read_bytes())
    listed = {}
    for row in inventory["entries"]:
        name = row["path"]
        if name.startswith("inputs/"):
            listed[name[7:]] = row
    if len(listed) != 106536:
        raise ValueError("raw inventory count differs")
    metadata_root = args.metadata / f"fold-{args.fold}-fit-inputs"
    manifest_path = metadata_root / "input-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if (manifest["kind"] != "cooptrack_fit_input_preparation_v1"
            or manifest["fold_id"] != args.fold
            or manifest["cohort"] != "fit-fold"
            or manifest["official_split_sha256"] != EXPECTED_SPLIT_SHA256
            or manifest["development_manifest_sha256"] != EXPECTED_MANIFEST_SHA256
            or manifest["official_val_read"] or manifest["test_or_test_A_read"]
            or manifest["raw_labels_modified"]):
        raise ValueError("fold manifest violates canonical fit protocol")
    acceptance = json.loads((metadata_root / "acceptance.json").read_bytes())
    if (acceptance["input_manifest_sha256"] != digest(manifest_path)
            or acceptance["fold_id"] != args.fold
            or acceptance["status"] != "fit_input_bytes_verified"):
        raise ValueError("local fit acceptance mismatch")
    payload = manifest["payload_inventory"]
    expected_paths = [row["path"] for row in payload]
    if expected_paths != sorted(set(expected_paths)):
        raise ValueError("fold payload inventory is not sorted unique")
    metadata_files = {"fold-split.json", "vehicle-side/data_info.json",
                      "infrastructure-side/data_info.json"}
    if not metadata_files <= set(expected_paths):
        raise ValueError("fit metadata incomplete")
    for row in payload:
        relative = row["path"]
        if relative in metadata_files:
            path = safe_path(metadata_root, relative)
            if digest(path) != row["sha256"] or path.stat().st_size != row["size_bytes"]:
                raise ValueError("transferred fit metadata differs")
        else:
            original = listed.get(relative)
            if original is None or original["sha256"] != row["sha256"] or original["bytes"] != row["size_bytes"]:
                raise ValueError("fit payload not bound to ClearML inventory")
    args.output.mkdir(parents=True, exist_ok=False)
    for relative in sorted(metadata_files | {"input-manifest.json"}):
        source = safe_path(metadata_root, relative)
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    started = time.monotonic()
    copied = 0
    bytes_copied = 0
    for row in payload:
        relative = row["path"]
        if relative in metadata_files:
            continue
        source = safe_path(args.source, relative)
        info = source.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != row["size_bytes"]:
            raise ValueError("raw source type or size differs")
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        with source.open("rb") as src, target.open("xb") as dst:
            for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                h.update(block)
                dst.write(block)
        if h.hexdigest() != row["sha256"] or digest(target) != row["sha256"]:
            raise ValueError("copied payload differs")
        copied += 1
        bytes_copied += info.st_size
        if copied % 15000 == 0:
            elapsed = max(time.monotonic() - started, 0.001)
            remaining = (len(payload) - len(metadata_files) - copied) * elapsed / copied
            print(f"SPD OOF fold {args.fold} payload {copied}/{len(payload)-len(metadata_files)} ETA={remaining:.1f}s", flush=True)
    receipt = {"kind": "spd_official_oof_remote_fit_projection_v1", "fold_id": args.fold,
               "status": "payload_bytes_verified", "file_count": copied,
               "payload_bytes": bytes_copied, "input_manifest_sha256": digest(args.output / "input-manifest.json"),
               "clearml_inventory_sha256": EXPECTED_INVENTORY_SHA256,
               "official_val_or_test_read": False}
    (args.output / "remote-projection-receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
