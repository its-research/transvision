#!/usr/bin/env python3
"""Independently re-read every canonical SPD OOF fit input file."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


ROOT = Path("/Volumes/Data/test/recover-before-fuse")
FOLDS = ROOT / "artifacts/spd-official-oof-fivefold-20260930"
CANONICAL = ROOT / "receipts/spd-official-train-canonical-development-fivefold-20260930.json"
CANONICAL_SHA256 = "1c9d23f291be09e686a2bdcc60ef19ea58a5c2894978cc5f6a6b2a81f43c2bd6"
SOURCE = ROOT / "artifacts/spd-official-train-source-20260930/materialized-inputs/readback.json"
OUTPUT = ROOT / "receipts/spd-official-oof-fivefold-inputs-independent-readback-20260930.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def audit() -> dict:
    if OUTPUT.exists():
        raise FileExistsError("independent readback receipt is create-once")
    if sha256(CANONICAL) != CANONICAL_SHA256:
        raise ValueError("canonical fivefold source differs")
    manifest = json.loads(CANONICAL.read_bytes())
    source = json.loads(SOURCE.read_bytes())
    if source.get("status") != "file_inventory_verified":
        raise ValueError("upstream train materialization was not accepted")
    held_counts = {sequence_id: 0 for sequence_id in manifest["sequence_ids"]}
    reports = []
    started = time.monotonic()
    processed = 0
    total = 0
    for fold_id in range(5):
        path = FOLDS / f"fold-{fold_id}-fit-inputs"
        receipt_path = path / "acceptance.json"
        input_path = path / "input-manifest.json"
        receipt = json.loads(receipt_path.read_bytes())
        prepared = json.loads(input_path.read_bytes())
        fold = manifest["folds"][fold_id]
        if (receipt.get("status") != "fit_input_bytes_verified"
                or receipt.get("fold_id") != fold_id
                or receipt.get("input_manifest_sha256") != sha256(input_path)
                or prepared.get("fit_sequence_ids") != fold["fit_sequence_ids"]
                or prepared.get("excluded_held_out_sequence_ids")
                   != fold["held_out_sequence_ids"]
                or prepared.get("development_manifest_sha256") != CANONICAL_SHA256
                or prepared.get("official_split_sha256") != manifest["split_sha256"]
                or prepared.get("official_val_read") is not False
                or prepared.get("test_or_test_A_read") is not False
                or prepared.get("raw_labels_modified") is not False):
            raise ValueError("fold view or acceptance disagrees with fixed protocol")
        content = dict(prepared)
        content_sha = content.pop("content_sha256")
        if hashlib.sha256(canonical_bytes(content)).hexdigest() != content_sha:
            raise ValueError("fold input manifest content hash differs")
        inventory = prepared["payload_inventory"]
        expected = {}
        for item in inventory:
            relative = item["path"]
            if relative in expected or relative.startswith("/") or ".." in Path(relative).parts:
                raise ValueError("invalid fold inventory path")
            expected[relative] = item
        total += len(expected)
        observed = set()
        bytes_read = 0
        for file in path.rglob("*"):
            if file.is_symlink():
                raise ValueError("fold view contains symlink")
            if not file.is_file():
                continue
            relative = file.relative_to(path).as_posix()
            observed.add(relative)
            if relative in expected:
                item = expected[relative]
                if (file.stat().st_size != item["size_bytes"]
                        or sha256(file) != item["sha256"]):
                    raise ValueError("fold payload byte readback differs")
                processed += 1
                bytes_read += item["size_bytes"]
                if processed % 50000 == 0:
                    print(f"SPD OOF independent readback {processed} files; "
                          f"elapsed={time.monotonic()-started:.1f}s ETA=unknown",
                          flush=True)
        if observed != set(expected) | {"input-manifest.json", "acceptance.json"}:
            raise ValueError("fold view contains missing or extra files")
        for sequence_id in fold["held_out_sequence_ids"]:
            held_counts[sequence_id] += 1
        reports.append({
            "fold_id": fold_id,
            "fit_sequence_count": len(fold["fit_sequence_ids"]),
            "held_out_sequence_count": len(fold["held_out_sequence_ids"]),
            "frame_counts": receipt["frame_counts"],
            "payload_file_count": len(expected),
            "payload_bytes": bytes_read,
            "input_manifest_sha256": sha256(input_path),
            "acceptance_sha256": sha256(receipt_path),
            "status": "independent_file_readback_verified",
        })
        print(f"SPD canonical fold {fold_id} independently accepted", flush=True)
    if set(held_counts.values()) != {1}:
        raise ValueError("official train sequence holdout coverage is not exactly once")
    result = {
        "kind": "spd_official_oof_fivefold_input_independent_readback_v1",
        "status": "all_five_folds_file_verified",
        "canonical_manifest_sha256": CANONICAL_SHA256,
        "materialized_source_receipt_sha256": sha256(SOURCE),
        "source_clearml_task_id": source["clearml_data_task_id"],
        "folds": reports,
        "holdout_coverage": "each_of_46_train_sequences_exactly_once",
        "payload_files_total": processed,
        "official_val_or_test_read": False,
        "conversion_started": False,
        "detector_training_started": False,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if processed != total:
        raise ValueError("independent payload count mismatch")
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print("SPD OOF fivefold independent readback receipt_sha256=" + sha256(OUTPUT),
          flush=True)
    return result


if __name__ == "__main__":
    audit()
