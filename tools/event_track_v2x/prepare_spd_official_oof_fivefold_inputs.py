#!/usr/bin/env python3
"""Prepare the repository's fixed SPD fivefold fit views from verified train bytes.

This creates isolated train-only fit inputs. It neither converts the inputs nor
authorizes detector training; both remain separate acceptance steps.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from prepare_cooptrack_fold_inputs import prepare


ROOT = Path("/Volumes/Data/test/recover-before-fuse")
SOURCE = ROOT / "artifacts/spd-official-train-source-20260930"
MATERIALIZED = SOURCE / "materialized-inputs"
SPLIT = SOURCE / "official-split-source/cooperative-split-data-spd.json"
SPLIT_SHA256 = "0a59332221f3618639f34fb70678d03c5221ab80fd69f2cf41acffca0d8859fd"
MANIFEST = ROOT / "receipts/spd-official-train-canonical-development-fivefold-20260930.json"
MANIFEST_SHA256 = "1c9d23f291be09e686a2bdcc60ef19ea58a5c2894978cc5f6a6b2a81f43c2bd6"
OUTPUT = ROOT / "artifacts/spd-official-oof-fivefold-20260930"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    source_receipt = json.loads((MATERIALIZED / "readback.json").read_bytes())
    if (source_receipt.get("status") != "file_inventory_verified"
            or source_receipt.get("raw_input_root") != str(MATERIALIZED / "inputs")
            or source_receipt.get("fulltrain_converted_copied") is not False):
        raise ValueError("full train source has not passed file-level admission")
    if sha256(SPLIT) != SPLIT_SHA256 or sha256(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("fixed official split or canonical fivefold manifest differs")
    manifest = json.loads(MANIFEST.read_bytes())
    if (manifest.get("fold_count") != 5 or manifest.get("split_sha256") != SPLIT_SHA256
            or manifest.get("fold_salt") != "eventtrack-v2x-spd-development-5fold-v1"):
        raise ValueError("fivefold protocol is not the fixed repository protocol")
    if OUTPUT.is_symlink() or OUTPUT.exists():
        raise FileExistsError("fivefold output root is create-once")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    for fold_id in range(5):
        print(f"SPD canonical fold {fold_id} fit-input preparation started ETA=unknown",
              flush=True)
        destination = OUTPUT / f"fold-{fold_id}-fit-inputs"
        report = prepare(MATERIALIZED / "inputs", SPLIT, MANIFEST, fold_id,
                         destination, cohort="fit-fold")
        if (report["fold_id"] != fold_id
                or report["development_manifest_sha256"] != MANIFEST_SHA256
                or report["official_split_sha256"] != SPLIT_SHA256
                or report["official_val_read"] or report["test_or_test_A_read"]
                or report["raw_labels_modified"]):
            raise ValueError("fit-only fold preparation violated protocol")
        receipt = {
            "kind": "spd_official_oof_canonical_fivefold_fit_view_v1",
            "status": "fit_input_bytes_verified",
            "fold_id": fold_id,
            "fit_sequence_ids": report["fit_sequence_ids"],
            "held_out_sequence_ids": report["excluded_held_out_sequence_ids"],
            "frame_counts": report["frame_counts"],
            "input_manifest_sha256": sha256(destination / "input-manifest.json"),
            "content_sha256": report["content_sha256"],
            "source_materialization_receipt_sha256": sha256(MATERIALIZED / "readback.json"),
            "canonical_fivefold_manifest_sha256": MANIFEST_SHA256,
            "official_split_sha256": SPLIT_SHA256,
            "official_val_or_test_read": False,
            "detector_training_started": False,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with (destination / "acceptance.json").open("x") as stream:
            json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        elapsed = max(time.monotonic() - started, 0.001)
        eta = elapsed / (fold_id + 1) * (4 - fold_id)
        print(f"SPD canonical fold {fold_id} accepted; remaining fit-input ETA={eta:.1f}s",
              flush=True)
    print("SPD fivefold fit-input preparation accepted; conversion and training pending",
          flush=True)


if __name__ == "__main__":
    main()
