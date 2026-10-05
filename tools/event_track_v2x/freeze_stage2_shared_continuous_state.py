#!/usr/bin/env python3
"""Freeze raw continuous observations beside accepted Stage2 finite event streams.

This extracts existing ClearML bytes; it does not create comparator outcomes or
assert a common state-update/resource contract.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from urllib.parse import urlparse

from clearml import Task
from clearml.storage.helper import StorageHelper

PARENTS = {1337: "88b25f8e71704e909a5ba50f3aaabb5b",
           2027: "939704d8e6d14dd78d213ed6e7a8dd28",
           3407: "5b7535e981054bbca8f5b2053fbea57c"}
HOSTS = {"10.100.34.118:8081", "10.100.35.118:8081"}
ROOT = Path("/Volumes/Data/test/recover-before-fuse")
THIN = ROOT / "artifacts/stage2-late-event-shared-candidate-stream-20260930"
OUTPUT = ROOT / "artifacts/stage2-late-event-shared-continuous-input-20260930"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def download(artifact):
    url = artifact.url
    if urlparse(url).netloc not in HOSTS:
        raise ValueError("unapproved ClearML file host")
    url = url.replace("10.100.34.118:8081", "10.100.35.118:8081", 1)
    data = b"".join(StorageHelper.get(url).download_as_stream(url))
    if len(data) != artifact.size or digest(data) != artifact.hash:
        raise ValueError("parent artifact byte identity differs")
    return data


def source_rows(seed, task, thin):
    artifact = task.artifacts["tracker-database"]
    data = download(artifact)
    if artifact.hash != thin["source_database_sha256"]:
        raise ValueError("thin stream parent database differs")
    connection = sqlite3.connect(":memory:")
    connection.deserialize(data)
    try:
        connection.execute("PRAGMA query_only=ON")
        observations = []
        rows = connection.execute(
            "SELECT i,node_id,source,frame,detection_index,state_us,arrival_us,"
            "score,raw,sha FROM observations ORDER BY i")
        for i, node_id, source, frame, detection_index, state_us, arrival_us, score, raw, raw_sha in rows:
            if digest(raw) != raw_sha:
                raise ValueError("raw observation hash differs")
            payload = json.loads(raw)
            identity = {"index": i, "node_id": node_id, "source_id": source,
                        "frame_id": frame, "information_us": state_us,
                        "arrival_us": arrival_us}
            if identity != thin["observations"][len(observations)]:
                raise ValueError("thin observation identity differs")
            if payload["node"] != {k: identity[k] for k in
                                   ("node_id", "source_id", "frame_id",
                                    "information_us", "arrival_us")}:
                raise ValueError("raw observation node identity differs")
            if (payload["score"] != score or payload["state_us"] != state_us or
                    payload["detection_index"] != detection_index or
                    len(payload["mean"]) != 9 or len(payload["covariance"]) != 9 or
                    any(len(row) != 9 for row in payload["covariance"])):
                raise ValueError("continuous observation shape differs")
            observations.append({**identity, "detection_index": detection_index,
                                 "score": score, "raw_sha256": raw_sha,
                                 "raw": payload})
        potentials = [{"index": i, "parent_index": p, "log_potential": w}
                      for i, p, w in connection.execute(
                          "SELECT i,p,w FROM potentials ORDER BY i,p")]
        if sorted(potentials, key=lambda x: (x["index"], x["parent_index"])) != sorted(
                thin["potentials"], key=lambda x: (x["index"], x["parent_index"])):
            raise ValueError("thin potential set differs")
        events = [row[0] for row in connection.execute(
            "SELECT event_id FROM events ORDER BY ordinal")]
        if events != [row["event_id"] for row in thin["events"]]:
            raise ValueError("thin event sequence differs")
        meta = {k: json.loads(v) for k, v in connection.execute("SELECT k,v FROM meta")}
        if not isinstance(meta.get("config", {}).get("state"), dict):
            raise ValueError("source state configuration missing")
    finally:
        connection.close()
    if (len(observations), len(potentials), len(events)) != (5, 9, 14):
        raise ValueError("parent finite stream counts differ")
    return {"kind": "stage2_shared_continuous_input_freeze_v1",
            "seed": seed, "source_task_id": task.id,
            "source_database_sha256": artifact.hash,
            "source_result_sha256": thin["source_result_sha256"],
            "thin_stream_sha256": digest((THIN / f"seed{seed}.json").read_bytes()),
            "observations": observations, "potentials": potentials,
            "events": thin["events"], "source_tracker_meta": meta,
            "comparator_outcomes_computed": False,
            "common_state_update_contract_admitted": False,
            "same_resource_contract_admitted": False,
            "full_stage_two_complete": False}


def main():
    paths = [OUTPUT / f"seed{seed}.json" for seed in PARENTS]
    receipt_path = OUTPUT / "freeze-receipt.json"
    if any(p.exists() or p.is_symlink() for p in paths + [receipt_path]):
        raise ValueError("create-once continuous input path already exists")
    prepared = []
    for seed, task_id in PARENTS.items():
        task = Task.get_task(task_id=task_id)
        if task.status != "completed":
            raise ValueError("parent task incomplete")
        thin_path = THIN / f"seed{seed}.json"
        thin = json.loads(thin_path.read_text())
        if thin["source_task_id"] != task_id or thin["seed"] != seed:
            raise ValueError("thin stream identity differs")
        prepared.append((seed, source_rows(seed, task, thin)))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed, stream in prepared:
        data = (json.dumps(stream, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        path = OUTPUT / f"seed{seed}.json"
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        rows.append({"seed": seed, "path": path.name, "bytes": len(data),
                     "sha256": digest(data), "source_task_id": PARENTS[seed],
                     "source_database_sha256": stream["source_database_sha256"]})
    receipt = {"kind": "stage2_shared_continuous_input_freeze_receipt_v1",
               "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "streams": rows, "comparator_outcomes_computed": False,
               "common_state_update_contract_admitted": False,
               "same_resource_contract_admitted": False,
               "full_stage_two_complete": False}
    data = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(receipt_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(data)
    print(json.dumps({"receipt": str(receipt_path), "sha256": digest(data),
                      "streams": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
