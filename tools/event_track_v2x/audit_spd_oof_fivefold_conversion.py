#!/usr/bin/env python3
"""Independently read back all canonical SPD OOF converted fit outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat


ROOT = Path("/Volumes/Data/test/recover-before-fuse")
FOLDS = ROOT / "artifacts/spd-official-oof-fivefold-20260930"
CONVERTED = FOLDS / "remote-conversion"
OUT = ROOT / "receipts/spd-official-oof-fivefold-conversion-independent-readback-20260930.json"
SOURCE_SHA = "abe990d81039b71afc6191ad886978c971ec6607377f1792677ba2f9893d9bb5"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def main() -> None:
    if OUT.exists():
        raise FileExistsError("conversion acceptance is create-once")
    rows = []
    seen_holdouts = set()
    for fold in range(5):
        view = FOLDS / f"fold-{fold}-fit-inputs"
        converted = CONVERTED / f"fold-{fold}-converted"
        projection = json.loads((CONVERTED / f"fold-{fold}-projection-receipt.json").read_bytes())
        input_path = view / "input-manifest.json"
        inputs = json.loads(input_path.read_bytes())
        acceptance = json.loads((view / "acceptance.json").read_bytes())
        if (projection["fold_id"] != fold or projection["status"] != "payload_bytes_verified"
                or projection["input_manifest_sha256"] != sha(input_path)
                or acceptance["input_manifest_sha256"] != sha(input_path)
                or inputs["fold_id"] != fold or inputs["official_val_read"]
                or inputs["test_or_test_A_read"] or inputs["raw_labels_modified"]):
            raise ValueError("fit source or remote projection differs")
        seen_holdouts.update(inputs["excluded_held_out_sequence_ids"])
        manifest_path = converted / "conversion-manifest.json"
        document = json.loads(manifest_path.read_bytes())
        body = dict(document)
        content_sha = body.pop("content_sha256")
        if (hashlib.sha256(canonical(body)).hexdigest() != content_sha
                or document["kind"] != "cooptrack_fit_conversion_v1"
                or document["fold_id"] != fold
                or document["cohort"] != "fit-fold"
                or document["fit_sequence_ids"] != inputs["fit_sequence_ids"]
                or document["input_manifest_sha256"] != sha(input_path)
                or document["source_binding"]["sha256"] != SOURCE_SHA
                or document["removed_frames"] != 0
                or document["interpolated_annotations"] != 0
                or document["raw_labels_modified"]
                or document["forecasting_targets_generated"]):
            raise ValueError("conversion manifest violates accepted protocol")
        expected = {}
        total_bytes = 0
        for item in document["inventory"]:
            rel = item["path"]
            if rel.startswith("/") or ".." in rel.split("/") or rel in expected:
                raise ValueError("unsafe or duplicate output path")
            expected[rel] = item
            path = converted / rel
            if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("converted artifact type differs")
            if path.stat().st_size != item["size_bytes"] or sha(path) != item["sha256"]:
                raise ValueError("converted artifact bytes differ")
            total_bytes += item["size_bytes"]
        actual = {p.relative_to(converted).as_posix() for p in converted.rglob("*") if p.is_file()}
        if actual != set(expected) | {"conversion-manifest.json"}:
            raise ValueError("converted artifact set differs")
        for side in ("vehicle-side", "infrastructure-side"):
            source_rows = json.loads((view / side / "data_info.json").read_bytes())
            target = converted / side / "v1.0-trainval"
            samples = json.loads((target / "sample.json").read_bytes())
            annotations = json.loads((target / "sample_annotation.json").read_bytes())
            frame_tokens = {row["frame_id"] for row in source_rows}
            sample_tokens = [row["token"] for row in samples]
            if len(samples) != len(source_rows) or set(sample_tokens) != frame_tokens or len(set(sample_tokens)) != len(samples):
                raise ValueError("converted sample coverage differs")
            original_annotation_tokens = set()
            for row in source_rows:
                labels = json.loads((view / side / row["label_lidar_std_path"]).read_bytes())
                original_annotation_tokens.update(item["token"] for item in labels)
            new_tokens = [row["token"] for row in annotations]
            if (len(new_tokens) != len(original_annotation_tokens)
                    or set(new_tokens) != original_annotation_tokens
                    or len(set(new_tokens)) != len(new_tokens)
                    or any(row["sample_token"] not in frame_tokens for row in annotations)
                    or document["counts"][side]["frames"] != len(samples)
                    or document["counts"][side]["annotations"] != len(annotations)):
                raise ValueError("converted annotation coverage differs")
        row = {"fold_id": fold, "conversion_manifest_sha256": sha(manifest_path),
               "input_manifest_sha256": sha(input_path), "converted_file_count": len(expected),
               "converted_bytes": total_bytes, "counts": document["counts"],
               "content_sha256": content_sha}
        rows.append(row)
        print(f"SPD OOF conversion accepted fold={fold} files={len(expected)} bytes={total_bytes} ETA=unknown", flush=True)
    if len(seen_holdouts) != 46:
        raise ValueError("held-out union incomplete")
    report = {"kind": "spd_official_oof_fivefold_conversion_independent_readback_v1",
              "status": "all_five_folds_file_and_coverage_verified", "folds": rows,
              "official_val_or_test_read": False, "detector_training_started": False,
              "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    with OUT.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print("SPD OOF CONVERSION ACCEPTED", sha(OUT), flush=True)


if __name__ == "__main__":
    main()
