#!/usr/bin/env python3
"""Read back the pinned CoopTrack source and freeze its official SPD split."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


TASK_ID = "9a7e7a9213954b57a35403f777e74561"
SOURCE_SHA256 = "abe990d81039b71afc6191ad886978c971ec6607377f1792677ba2f9893d9bb5"
SOURCE_BYTES = 551492
SPLIT_SHA256 = "0a59332221f3618639f34fb70678d03c5221ab80fd69f2cf41acffca0d8859fd"
MEMBER = "workspace/CoopTrack/data/split_datas/cooperative-split-data-spd.json"
MANIFEST = Path("/Volumes/Data/test/recover-before-fuse/receipts/"
                "spd-official-train-canonical-development-fivefold-20260930.json")
OUTPUT = Path("/Volumes/Data/test/recover-before-fuse/artifacts/"
              "spd-official-train-source-20260930/official-split-source")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    from clearml import Task

    if OUTPUT.exists():
        raise FileExistsError("official split source is create-once")
    manifest_raw = MANIFEST.read_bytes()
    manifest = json.loads(manifest_raw)
    task = Task.get_task(task_id=TASK_ID)
    artifact = task.artifacts["source"]
    if (task.status != "completed" or artifact.hash != SOURCE_SHA256
            or artifact.size != SOURCE_BYTES):
        raise ValueError("ClearML source identity changed")
    archive_path = Path(artifact.get_local_copy())
    archive_raw = archive_path.read_bytes()
    if len(archive_raw) != SOURCE_BYTES or digest(archive_raw) != SOURCE_SHA256:
        raise ValueError("source archive byte readback failed")
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:gz") as archive:
        matches = [entry for entry in archive if entry.name == MEMBER]
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError("official split member absent or ambiguous")
        split_raw = archive.extractfile(matches[0]).read()
    if digest(split_raw) != SPLIT_SHA256 or manifest["split_sha256"] != SPLIT_SHA256:
        raise ValueError("official split SHA-256 mismatch")
    split = json.loads(split_raw)
    if set(split["batch_split"]["train"]) != set(manifest["sequence_ids"]):
        raise ValueError("official train and canonical fivefold membership differ")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with (OUTPUT / "source.tar.gz").open("xb") as target:
        target.write(archive_raw)
    with (OUTPUT / PurePosixPath(MEMBER).name).open("xb") as target:
        target.write(split_raw)
    receipt = {
        "kind": "spd_official_split_source_readback_v1",
        "status": "byte_verified",
        "task_id": TASK_ID,
        "source_archive_bytes": SOURCE_BYTES,
        "source_archive_sha256": SOURCE_SHA256,
        "source_member": MEMBER,
        "split_bytes": len(split_raw),
        "split_sha256": SPLIT_SHA256,
        "train_sequence_count": len(manifest["sequence_ids"]),
        "canonical_fivefold_manifest_sha256": digest(manifest_raw),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "detector_training_started": False,
    }
    with (OUTPUT / "readback.json").open("x") as target:
        json.dump(receipt, target, sort_keys=True, indent=2)
        target.write("\n")
    print(json.dumps({"status": receipt["status"], "split_sha256": SPLIT_SHA256,
                      "receipt_sha256": digest((OUTPUT / "readback.json").read_bytes())}))


if __name__ == "__main__":
    main()
