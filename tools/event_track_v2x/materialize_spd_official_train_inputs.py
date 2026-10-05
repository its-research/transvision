#!/usr/bin/env python3
"""Verify every frozen SPD train-package file and materialize only raw train inputs.

The package's existing full-train conversion is audited but never copied into
this OOF preparation view. This entrypoint is create-once and keeps partial
output on failure for inspection.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time


ROOT = Path("/Volumes/Data/test/recover-before-fuse/artifacts/"
            "spd-official-train-source-20260930")
ARCHIVE = ROOT / "train-inputs.tar.gz"
SOURCE_RECEIPT = ROOT / "source-readback.json"
INVENTORY = ROOT / "verified-inventory/verified-inventory.bin"
INVENTORY_SHA256 = "e04f07ad6e4eeb7a505ca33eed31bafd27d4157e40cc34f73e9e961a07136ce0"
ARCHIVE_SHA256 = "c793bc74ec35140c894fd7f3d14fd2cce231f0b919eee3fdbeca7515b3d1e0c1"
ARCHIVE_BYTES = 4070034766
OUTPUT = ROOT / "materialized-inputs"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_name(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or name.startswith("/") or "\\" in name
            or ".." in path.parts or path.as_posix() != name
            or path.parts[0] not in {"inputs", "converted"}):
        raise ValueError("unsafe or out-of-scope archive path")
    return name


def materialize() -> dict:
    if OUTPUT.exists():
        raise FileExistsError("materialized input output is create-once")
    source = json.loads(SOURCE_RECEIPT.read_bytes())
    if (source.get("status") != "byte_verified"
            or source.get("sha256") != ARCHIVE_SHA256
            or source.get("bytes") != ARCHIVE_BYTES
            or ARCHIVE.stat().st_size != ARCHIVE_BYTES
            or sha256_file(ARCHIVE) != ARCHIVE_SHA256):
        raise ValueError("complete archive byte identity is unverified")
    if sha256_file(INVENTORY) != INVENTORY_SHA256:
        raise ValueError("verified inventory byte identity differs")
    raw_entries = json.loads(INVENTORY.read_bytes())["entries"]
    expected = {}
    for item in raw_entries:
        name = checked_name(item["path"])
        if name in expected or type(item["bytes"]) is not int or item["bytes"] < 0:
            raise ValueError("invalid or duplicate expected inventory row")
        expected[name] = item
    OUTPUT.mkdir(parents=True, exist_ok=False)
    seen = set()
    counts = {"inputs": 0, "converted": 0}
    byte_counts = {"inputs": 0, "converted": 0}
    started = time.monotonic()
    with tarfile.open(ARCHIVE, mode="r|gz") as archive:
        for member in archive:
            name = checked_name(member.name)
            if member.isdir():
                continue
            if not member.isfile() or name in seen or name not in expected:
                raise ValueError("unexpected archive member or duplicate path")
            if member.size != expected[name]["bytes"]:
                raise ValueError("archive member size differs from frozen inventory")
            seen.add(name)
            prefix = PurePosixPath(name).parts[0]
            source_stream = archive.extractfile(member)
            if source_stream is None:
                raise ValueError("regular archive member has no payload")
            target = OUTPUT / name if prefix == "inputs" else None
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            received = 0
            with source_stream:
                if target is None:
                    while chunk := source_stream.read(8 * 1024 * 1024):
                        digest.update(chunk)
                        received += len(chunk)
                else:
                    with target.open("xb") as output_stream:
                        while chunk := source_stream.read(8 * 1024 * 1024):
                            digest.update(chunk)
                            output_stream.write(chunk)
                            received += len(chunk)
            if (received != member.size or digest.hexdigest() != expected[name]["sha256"]
                    or (target is not None and sha256_file(target) != digest.hexdigest())):
                raise ValueError("archive or materialized file byte readback differs")
            counts[prefix] += 1
            byte_counts[prefix] += received
            if len(seen) % 10000 == 0:
                elapsed = max(time.monotonic() - started, 0.001)
                eta = (len(expected) - len(seen)) * elapsed / len(seen)
                print(f"SPD archive {len(seen)}/{len(expected)} files ETA={eta:.1f}s",
                      flush=True)
    if seen != set(expected):
        raise ValueError("archive does not cover the entire frozen inventory")
    if counts["inputs"] == 0 or counts["converted"] == 0:
        raise ValueError("source archive omitted one required subtree")
    report = {
        "kind": "spd_official_train_input_materialization_v1",
        "status": "file_inventory_verified",
        "clearml_data_task_id": source["task_id"],
        "archive_sha256": ARCHIVE_SHA256,
        "inventory_sha256": INVENTORY_SHA256,
        "file_counts": counts,
        "byte_counts": byte_counts,
        "raw_input_root": str(OUTPUT / "inputs"),
        "fulltrain_converted_copied": False,
        "official_val_or_test_read": False,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (OUTPUT / "readback.json").open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print("SPD input materialization accepted receipt_sha256="
          + sha256_file(OUTPUT / "readback.json"), flush=True)
    return report


if __name__ == "__main__":
    materialize()
