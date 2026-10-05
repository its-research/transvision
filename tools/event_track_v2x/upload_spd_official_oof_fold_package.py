#!/usr/bin/env python3
"""Publish one byte-frozen canonical SPD OOF fit package to ClearML."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_cooptrack_official_oof_gpu4 import validate_cohort


PROJECT = "Thesis/EventTrack-V2X/Training"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    from clearml import Task

    manifest_path = args.package / "package-manifest.json"
    manifest_sha = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["fold_id"] != args.fold:
        raise ValueError("package fold differs")
    validate_cohort(manifest)
    archive = args.package / "train-inputs.tar.gz"
    row = next(item for item in manifest["inventory"] if item["path"] == "train-inputs.tar.gz")
    if row.get("artifact_task_id") or archive.stat().st_size != row["bytes"] or sha256(archive) != row["sha256"]:
        raise ValueError("local OOF package bytes differ from accepted manifest")
    name = f"SPD canonical official-train OOF fold-{args.fold} fit package {manifest_sha[:12]}"
    receipt = args.package / "clearml-package-task.json"
    if receipt.exists():
        prior = json.loads(receipt.read_bytes())
        if prior["manifest_sha256"] != manifest_sha or prior["name"] != name:
            raise ValueError("existing upload receipt belongs to another package")
        task = Task.get_task(task_id=prior["task_id"])
    else:
        existing = Task.get_tasks(project_name=PROJECT, task_name=name)
        if len(existing) > 1:
            raise RuntimeError("ambiguous matching ClearML package tasks")
        if existing:
            task = existing[0]
            if task.get_parameters().get("General/package_manifest_sha256") != manifest_sha:
                raise ValueError("existing package task has different manifest")
        else:
            task = Task.create(project_name=PROJECT, task_name=name,
                               task_type=Task.TaskTypes.data_processing)
            task.output_uri = True
            task.add_tags(["SPD", "canonical-OOF", f"fold-{args.fold}", "fit-only"])
            task.set_parameters({"package_manifest_sha256": manifest_sha,
                                 "train_inputs_sha256": row["sha256"],
                                 "fold_id": args.fold,
                                 "val_or_test_included": False,
                                 "source_upload_authorization": "user-explicit-2026-09-29"})
        with receipt.open("x") as stream:
            json.dump({"task_id": task.id, "manifest_sha256": manifest_sha,
                       "name": name}, stream, sort_keys=True, indent=2)
            stream.write("\n")
    current = Task.get_task(task_id=task.id)
    if current.status != "completed":
        task.mark_started(force=True)
    for artifact_name, path, expected in [
        ("train-inputs", archive, row["sha256"]),
        ("package-manifest", manifest_path, manifest_sha),
    ]:
        current = Task.get_task(task_id=task.id)
        if artifact_name in current.artifacts and current.artifacts[artifact_name].hash == expected:
            print("ALREADY_UPLOADED", artifact_name, flush=True)
            continue
        if not task.upload_artifact(artifact_name, artifact_object=path, wait_on_upload=True):
            raise RuntimeError("upload failed: " + artifact_name)
        current = Task.get_task(task_id=task.id)
        if artifact_name not in current.artifacts or current.artifacts[artifact_name].hash != expected:
            raise RuntimeError("uploaded artifact hash mismatch: " + artifact_name)
        print("UPLOADED_AND_HASH_VERIFIED", artifact_name, expected, flush=True)
    task.mark_completed(force=True)
    accepted = {"kind": "spd_official_oof_fold_package_clearml_upload_v1",
                "fold_id": args.fold, "task_id": task.id,
                "manifest_sha256": manifest_sha, "train_inputs_sha256": row["sha256"],
                "status": "uploaded_and_hash_verified", "detector_training_started": False}
    out = args.package / "clearml-upload-acceptance.json"
    if out.exists():
        if json.loads(out.read_bytes()) != accepted:
            raise ValueError("upload acceptance differs")
    else:
        with out.open("x") as stream:
            json.dump(accepted, stream, sort_keys=True, indent=2)
            stream.write("\n")
    print("SPD_OOF_PACKAGE_UPLOAD_ACCEPTED", json.dumps(accepted, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
