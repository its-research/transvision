#!/usr/bin/env python3
"""Publish the byte-frozen, GPU-model-neutral diagnostic inference source."""
import hashlib
import json
from pathlib import Path
import re

from clearml import Task


ARCHIVE = Path("/Volumes/Data/test/recover-before-fuse/artifacts/"
               "v2v4real-overlap-v1-diagnostic-inference-source-gpu4-20260930/"
               "diagnostic-inference-source.tar.gz")
EXPECTED_SHA256 = "e511ea54695ccd5154561331f9d6f1ca32f7a9d9d4c59c55825a4efb215b9d6f"
PROJECT = "Thesis/Recover-Before-Fuse/Training"
NAME = "V2V4Real overlap v1 diagnostic inference GPU4 source " + EXPECTED_SHA256[:12]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if ARCHIVE.is_symlink() or not ARCHIVE.is_file() or sha256(ARCHIVE) != EXPECTED_SHA256:
        raise ValueError("frozen inference source archive differs")
    existing = Task.get_tasks(project_name=PROJECT, task_name="^" + re.escape(NAME) + "$")
    if len(existing) > 1:
        raise ValueError("duplicate source task identity")
    if existing:
        task = existing[0]
        artifact = task.artifacts.get("source")
        if (str(task.status) != "completed" or artifact is None
                or artifact.hash != EXPECTED_SHA256 or artifact.size != ARCHIVE.stat().st_size):
            raise ValueError("existing diagnostic inference source differs")
        duplicate = True
    else:
        task = Task.create(project_name=PROJECT, task_name=NAME,
                           task_type=Task.TaskTypes.data_processing)
        task.add_tags(["Recover-Before-Fuse", "V2V4Real", "diagnostic-inference-source",
                       "overlap-controlled-v1", "train-validate-only", "non-formal",
                       "four-GPU", "model-neutral"])
        task.set_parameters({"variant_id": "v2v4real-train-overlap-controlled-v1",
                             "source_sha256": EXPECTED_SHA256,
                             "selection_task_id": "35ae26a544d24a9e9a314f4789e9648f",
                             "accepted_checkpoint_seeds": "1337,2027,3407",
                             "four_gpu_count_required": True,
                             "gpu_model_restriction": False,
                             "paper_eligible": False,
                             "formal_independent_test_eligible": False})
        if not task.upload_artifact("source", artifact_object=ARCHIVE, wait_on_upload=True):
            raise RuntimeError("source upload failed; inspect created task before retry")
        task.reload()
        artifact = task.artifacts.get("source")
        if (artifact is None or artifact.hash != EXPECTED_SHA256
                or artifact.size != ARCHIVE.stat().st_size):
            raise ValueError("published inference source artifact differs")
        task.mark_completed(force=True)
        duplicate = False
    print(json.dumps({"task_id": task.id, "status": str(task.status),
                      "source_sha256": EXPECTED_SHA256,
                      "duplicate_not_created": duplicate,
                      "paper_eligible": False,
                      "formal_independent_test_eligible": False}, sort_keys=True))


if __name__ == "__main__":
    main()
