#!/usr/bin/env python3
"""Byte-freeze historical three-seed SPD full-train identity runs without rerun."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlparse

from clearml import Task
from clearml.storage.helper import StorageHelper

TASKS = {1337: "2cc70989d6fd48f7bfd6c715db9984f2",
         2027: "fbd1abc3470544bea4fd18b271309930",
         3407: "6d91e8945e1449aab02582f29bc12cb8"}
KEYS = {"checkpoint-manifest", "epoch-progress", "identity-checkpoint",
        "training-plan", "training-receipt"}
HOSTS = {"10.100.34.118:8081", "10.100.35.118:8081"}


def must(value, message):
    if not value:
        raise ValueError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def download(artifact):
    url = artifact.url
    must(urlparse(url).netloc in HOSTS, "unapproved ClearML artifact host")
    url = url.replace("10.100.34.118:8081", "10.100.35.118:8081", 1)
    payload = b"".join(StorageHelper.get(url).download_as_stream(url))
    must(len(payload) == artifact.size and sha(payload) == artifact.hash,
         "artifact byte identity differs")
    return payload


def verify(seed, payloads):
    checkpoint = json.loads(payloads["checkpoint-manifest"])
    plan = json.loads(payloads["training-plan"])
    receipt = json.loads(payloads["training-receipt"])
    epochs = [json.loads(line) for line in payloads["epoch-progress"].splitlines()]
    must(checkpoint["seed"] == seed and checkpoint["data_split"] == "train"
         and checkpoint["full_official_train"] is True
         and checkpoint["fixed_final_epoch"] == 10
         and checkpoint["weights"]["sha256"] == sha(payloads["identity-checkpoint"])
         and checkpoint["plan_sha256"] == sha(payloads["training-plan"])
         and checkpoint["strict_pipeline_isolated_selection"] is False
         and checkpoint["validation_or_test_selection"] is False
         and checkpoint["paper_eligible"] is False,
         "checkpoint or source boundary differs")
    must(plan["full_official_train"] is True and
         len(plan["dataset_sequences"]) == 46 and
         plan["supervised_rows"] == 90651 and
         plan["world_size"] == 4 and
         plan["global_batch_size"] == 64 and
         plan["selection"] == "fixed_final_epoch_no_validation_search" and
         plan["strict_pipeline_isolated_selection"] is False and
         plan["paper_eligible"] is False,
         "full-train plan differs")
    must(receipt["status"] == "complete" and
         receipt["plan_sha256"] == checkpoint["plan_sha256"] and
         receipt["full_official_train"] is True and
         receipt["world_size"] == 4 and
         receipt["complete_three_seed_campaign"] is False and
         receipt["tracking_validation_performed"] is False and
         receipt["paper_eligible"] is False,
         "single-run receipt boundary differs")
    must(len(epochs) == 10 and [row["epoch"] for row in epochs] == list(range(1, 11)),
         "epoch coverage differs")
    for row in epochs:
        must(row["seed"] == seed and row["supervised_rows"] == 90651 and
             row["global_batches"] == 1440 and
             row["validation_metrics_read"] is False and
             len(row["rank_progress"]) == 4,
             "epoch progress or rank coverage differs")
    return {"epoch_count": len(epochs), "global_batches_per_epoch": 1440,
            "supervised_rows_per_epoch": 90651,
            "weights_sha256": checkpoint["weights"]["sha256"],
            "plan_sha256": checkpoint["plan_sha256"],
            "dataset_sha256": checkpoint["dataset_sha256"],
            "row_protocol_sha256": sha(json.dumps(checkpoint["row_protocol"],
                                                  sort_keys=True,
                                                  separators=(",", ":")).encode()),
            "class_scope": checkpoint["row_protocol"]["class_scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    must(not args.output.exists() and not args.output.is_symlink(),
         "create-once output already exists")
    args.output.mkdir(parents=True)
    rows = []
    started = time.monotonic()
    for seed, task_id in TASKS.items():
        task = Task.get_task(task_id=task_id)
        must(task.status == "completed" and set(task.artifacts) == KEYS,
             "historical task status or artifact inventory differs")
        payloads = {key: download(task.artifacts[key]) for key in sorted(KEYS)}
        validation = verify(seed, payloads)
        directory = args.output / f"seed{seed}"
        directory.mkdir()
        artifact_rows = {}
        for key, payload in payloads.items():
            extension = ".pt" if key == "identity-checkpoint" else (
                ".jsonl" if key == "epoch-progress" else ".json")
            path = directory / (key + extension)
            with path.open("xb") as stream:
                stream.write(payload)
            artifact_rows[key] = {"path": str(path), "sha256": sha(payload),
                                  "bytes": len(payload)}
        rows.append({"seed": seed, "task_id": task_id,
                     "worker_id": task.data.last_worker,
                     "artifacts": artifact_rows, "validation": validation})
        eta = (time.monotonic() - started) / len(rows) * (len(TASKS) - len(rows))
        print(f"SPD historical full-train identity freeze {len(rows)}/{len(TASKS)} "
              f"seed={seed} ETA={eta:.1f}s", flush=True)
    must(len({x["validation"]["dataset_sha256"] for x in rows}) == 1 and
         len({x["validation"]["row_protocol_sha256"] for x in rows}) == 1 and
         all(x["validation"]["class_scope"] == ["car"] for x in rows) and
         len({x["validation"]["weights_sha256"] for x in rows}) == 3,
         "three-seed campaign data/protocol/model identity differs")
    result = {"kind": "spd_historical_fulltrain_identity_three_seed_byte_freeze_v1",
              "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "rows": rows, "full_train_identity_software_training_complete": True,
              "strict_pipeline_isolated_selection": False,
              "tracking_validation_performed": False,
              "paper_performance_complete": False}
    path = args.output / "acceptance-receipt.json"
    with path.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"receipt": str(path), "sha256": sha(path.read_bytes()),
                      "seeds": sorted(TASKS)}, sort_keys=True))


if __name__ == "__main__":
    main()
