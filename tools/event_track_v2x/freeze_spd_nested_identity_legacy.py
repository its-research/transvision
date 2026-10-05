#!/usr/bin/env python3
"""Independently freeze two completed historical SPD nested identity runs."""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
from urllib.parse import urlparse

from clearml import Task
from clearml.storage.helper import StorageHelper

TASKS = {1337: "b20f15594b6a48b68731868cc1d095f0",
         2027: "f7da8e72146742f191ded68d8b57f481"}
KEYS = {"checkpoint", "identity-training", "launch", "plan", "receipt"}
HOSTS = {"10.100.34.118:8081", "10.100.35.118:8081"}


def must(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_artifact(artifact):
    url = artifact.url
    must(urlparse(url).netloc in HOSTS, "unapproved ClearML file host")
    url = url.replace("10.100.34.118:8081", "10.100.35.118:8081", 1)
    data = b"".join(StorageHelper.get(url).download_as_stream(url))
    must(len(data) == artifact.size and sha(data) == artifact.hash,
         "ClearML artifact byte identity differs")
    return data


def inspect_archive(data, seed, checkpoint, sibling):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        names = {m.name for m in members}
        prefix = f"training/seed-{seed}"
        expected = {"training", "training/plan.json", "training/receipt.json",
                    prefix, f"{prefix}/checkpoint.json", f"{prefix}/epochs.json",
                    f"{prefix}/weights.pt"}
        must(names == expected and len(members) == len(expected),
             "archive member inventory differs")
        must(all(not m.issym() and not m.islnk() and not m.name.startswith("/")
                 and ".." not in Path(m.name).parts for m in members),
             "unsafe archive member")
        member_bytes = {name: archive.extractfile(name).read()
                        for name in expected if name not in {"training", prefix}}
    must(member_bytes["training/plan.json"] == sibling["plan"] and
         member_bytes["training/receipt.json"] == sibling["receipt"] and
         member_bytes[f"{prefix}/checkpoint.json"] == sibling["checkpoint"],
         "archive and standalone artifacts differ")
    must(sha(member_bytes[f"{prefix}/weights.pt"]) == checkpoint["weights"]["sha256"],
         "selected weights hash differs")
    epochs = json.loads(member_bytes[f"{prefix}/epochs.json"])
    must(len(epochs) == 10 and [x["epoch"] for x in epochs] == list(range(1, 11)),
         "training epoch coverage differs")
    best = min(epochs, key=lambda x: (x["holdout"]["macro_row_surrogate"], x["epoch"]))
    must(best["epoch"] == checkpoint["selected_epoch"],
         "checkpoint selection does not match train-internal holdout")
    return {"member_count": len(members), "epoch_count": len(epochs),
            "selected_epoch": best["epoch"],
            "selected_weights_sha256": checkpoint["weights"]["sha256"],
            "epochs_sha256": sha(member_bytes[f"{prefix}/epochs.json"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    must(not args.output.exists() and not args.output.is_symlink(),
         "create-once output already exists")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    rows = []
    for seed, task_id in TASKS.items():
        task = Task.get_task(task_id=task_id)
        must(task.status == "completed" and set(task.artifacts) == KEYS,
             "historical task status or artifact inventory differs")
        payloads = {key: read_artifact(task.artifacts[key]) for key in sorted(KEYS)}
        checkpoint = json.loads(payloads["checkpoint"])
        receipt = json.loads(payloads["receipt"])
        must(checkpoint["seed"] == seed and checkpoint["dataset"] == "spd"
             and checkpoint["data_split"] == "train" and checkpoint["fixture"] is False
             and checkpoint["paper_eligible"] is False
             and checkpoint["strict_pipeline_isolated_selection"] is True
             and checkpoint["validation_or_test_selection"] is False
             and checkpoint["selection"] == "official_train_internal_holdout"
             and checkpoint["row_protocol"]["candidate_protocol"] == "rbf-all-class-top64-v1",
             "nested training checkpoint boundary differs")
        fit = set(checkpoint["partition"]["fit"])
        holdout = set(checkpoint["partition"]["holdout"])
        must(len(fit) == 37 and len(holdout) == 9 and not fit & holdout,
             "nested train fit/selection partition differs")
        must(checkpoint["plan_sha256"] == sha(payloads["plan"])
             and receipt["plan_sha256"] == checkpoint["plan_sha256"]
             and receipt["status"] == "software_training_completed"
             and receipt["paper_results_verified"] is False,
             "plan or completion receipt differs")
        archive = inspect_archive(payloads["identity-training"], seed, checkpoint, payloads)
        directory = args.output / f"seed{seed}"
        directory.mkdir()
        files = {}
        for key, data in payloads.items():
            filename = {"identity-training": "identity-training.tar.gz"}.get(key, key + ".json")
            path = directory / filename
            with path.open("xb") as stream:
                stream.write(data)
            files[key] = {"path": str(path), "sha256": sha(data), "bytes": len(data)}
        rows.append({"seed": seed, "task_id": task_id,
                     "worker_id": task.data.last_worker, "artifacts": files,
                     "archive_check": archive,
                     "plan_sha256": checkpoint["plan_sha256"],
                     "partition_fit": len(fit), "partition_holdout": len(holdout)})
        eta = (time.monotonic() - started) / len(rows) * (len(TASKS) - len(rows))
        print(f"SPD nested identity byte freeze {len(rows)}/{len(TASKS)} "
              f"seed={seed} ETA={eta:.1f}s", flush=True)
    result = {"kind": "spd_nested_identity_legacy_two_seed_byte_freeze_v1",
              "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "rows": rows, "software_training_completed": True,
              "final_full_train_training_completed": False,
              "independent_validation_completed": False,
              "paper_performance_complete": False}
    path = args.output / "acceptance-receipt.json"
    with path.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"receipt": str(path), "sha256": sha(path.read_bytes()),
                      "seeds": sorted(TASKS)}, sort_keys=True))


if __name__ == "__main__":
    main()
