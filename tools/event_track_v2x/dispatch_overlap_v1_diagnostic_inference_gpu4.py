#!/usr/bin/env python3
"""Dispatch protocol-gated four-GPU diagnostic inference after acceptance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

from clearml import Task
from clearml.backend_api import Session
from clearml.backend_api.session.client import APIClient
from clearml.storage.helper import StorageHelper


ROOT = Path("/Volumes/Data/test/recover-before-fuse/artifacts/"
            "v2v4real-overlap-v1-diagnostic-inference-source-v3-20260929")
PROJECT = "Thesis/Recover-Before-Fuse/Training"
SOURCE_SHA = "e511ea54695ccd5154561331f9d6f1ca32f7a9d9d4c59c55825a4efb215b9d6f"
BOOTSTRAP = ROOT / "bootstrap_overlap_v1_diagnostic.py"
BOOTSTRAP_SHA = "dfe1af5b8893b11ee5df793c45248f5a4e9ddf01a61587043af5f42e1affcb35"
CONFIG = ROOT / "tree/project/configs/event_track_v2x/v2v4real-nominal-10hz-formal-v1.json"
CONFIG_SHA = "40479859c407db27cc7e609c889a447057319905f94e887769ef919d57004334"
VALIDATOR = ROOT / "tree/project/transvision/models/event_track_v2x/paper_nominal_clock.py"
VALIDATOR_SHA = "d45dcf4d7572192a59384ddb55035dad47aa3204e91df29a592e5ab572b6ed3a"
PROJECTION_TASK = "72599160425d47ca9b7e8916bbb77b42"
PROJECTION_SHA = "2a022771710632db4d1011ded447fd6677e241bdff7c51e4f6d572a0c18523d5"
SELECTION_TASK = "35ae26a544d24a9e9a314f4789e9648f"
SELECTION_SHA = "266593c647c99cd33778de5e35a2c5fb14493f6808631309fb949f0b67d4efaa"
TRAINING = {
    1337: ("669833fb759742a4a857f99f87341e0b", "7f58517f4c62452cbef9b98c79ff1420",
           "93b3f63cb1058c725224e275c713566e2a11090812026060684c77867a3c3a69"),
    2027: ("f945af3976e34a479e40b56ef2a7be24", "7f58517f4c62452cbef9b98c79ff1420",
           "93b3f63cb1058c725224e275c713566e2a11090812026060684c77867a3c3a69"),
}
IMAGE = ("gitlab.zhht.ai.com:5000/aitech/"
         "model_infer:py-3.12-cuda-12.6.2-torch-2.10-ultralytics-8.4.13-dvc-v2-onnx-clearml")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formal_gate():
    for path, expected in ((CONFIG, CONFIG_SHA), (VALIDATOR, VALIDATOR_SHA),
                           (BOOTSTRAP, BOOTSTRAP_SHA)):
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError("frozen protocol or bootstrap source differs")
    sys.path.insert(0, str(ROOT / "tree/project"))
    from transvision.models.event_track_v2x.paper_nominal_clock import validate_formal_variant
    config = validate_formal_variant(json.loads(CONFIG.read_bytes()))
    if config["protocol_id"] != "v2v4real-nominal-10hz-formal-v1":
        raise ValueError("nominal protocol identity differs")
    return config


def verify_freeze(seed, path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("ordinary diagnostic byte freeze receipt required")
    data = json.loads(path.read_bytes())
    training_id, source_id, source_sha = TRAINING[seed]
    if (data.get("kind") != "v2v4real_overlap_controlled_v1_diagnostic_checkpoint_byte_freeze_v1"
            or data.get("task_id") != training_id or data.get("seed") != seed
            or data.get("source_task_id") != source_id
            or data.get("source_sha256") != source_sha
            or data.get("all_60_epochs_verified") is not True
            or data.get("checkpoint_bytes_frozen") is not True
            or data.get("paper_eligible") is not False
            or data.get("formal_independent_test_eligible") is not False):
        raise ValueError("completed diagnostic checkpoint freeze differs")
    for key in ("best-checkpoint", "training-receipt"):
        row = data.get("artifacts", {}).get(key, {})
        artifact_path = Path(row.get("path", ""))
        if (artifact_path.is_symlink() or not artifact_path.is_file()
                or artifact_path.stat().st_size != row.get("bytes")
                or sha256(artifact_path) != row.get("sha256")):
            raise ValueError("frozen diagnostic artifact bytes differ: " + key)
    return data


def upstream(task_id, key, digest):
    task = Task.get_task(task_id=task_id)
    artifact = task.artifacts.get(key)
    if (str(task.status) != "completed" or artifact is None
            or artifact.hash != digest):
        raise ValueError("completed upstream artifact identity differs: " + task_id + "/" + key)
    return task, artifact


def acceptance_readback(task_id, seed, training_id, checkpoint_sha, source_id, source_sha):
    if not re.fullmatch(r"[0-9a-f]{32}", task_id):
        raise ValueError("exact acceptance task ID required")
    task = Task.get_task(task_id=task_id)
    artifact = task.artifacts.get("checkpoint-acceptance")
    if str(task.status) != "completed" or artifact is None:
        raise ValueError("completed four-GPU checkpoint acceptance required")
    parsed = urlparse(artifact.url)
    if (parsed.scheme not in ("http", "https")
            or parsed.netloc not in {"10.100.34.118:8081", "10.100.35.118:8081"}):
        raise ValueError("acceptance artifact outside approved file service")
    url = artifact.url.replace("10.100.34.118:8081", "10.100.35.118:8081", 1)
    payload = b"".join(StorageHelper.get(url).download_as_stream(url))
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != artifact.size or digest != artifact.hash:
        raise ValueError("four-GPU acceptance artifact readback differs")
    data = json.loads(payload)
    if (data.get("kind") != "v2v4real_overlap_diagnostic_checkpoint_gpu4_acceptance_v1"
            or data.get("task_id") != task_id
            or data.get("checkpoint_task_id") != training_id
            or data.get("checkpoint_sha256") != checkpoint_sha
            or data.get("source_task_id") != source_id
            or data.get("source_sha256") != source_sha
            or data.get("seed") != seed
            or data.get("diagnostic_checkpoint_envelope_verified") is not True
            or data.get("strict_state_dict_load_verified") is not True
            or data.get("raw_head_forward_verified") is not True
            or data.get("dataset_read") is not False
            or data.get("gt_read") is not False
            or data.get("paper_eligible") is not False
            or data.get("formal_independent_test_eligible") is not False
            or re.fullmatch(r"[^:]+:gpu\d+,\d+,\d+,\d+",
                            str(data.get("worker_id", ""))) is None
            or len(data.get("devices", [])) != 4
            or sorted(device.get("index") for device in data["devices"]) != [0, 1, 2, 3]
            or any(not device.get("name", "")
                   for device in data["devices"])):
        raise ValueError("independent four-GPU acceptance differs")
    return digest


def idle_workers(queue_id):
    api = APIClient()
    queue = api.queues.get_by_id(queue=queue_id)
    if not queue.name.startswith("GPU4-"):
        raise ValueError("four-GPU queue required")
    now = datetime.now(timezone.utc)
    workers = api.workers.get_all()
    target = [worker for worker in workers if any(
        entry.id == queue_id for entry in worker.queues or [])]
    if not target:
        raise ValueError("four-GPU queue has no live worker")
    busy_devices = {}
    for worker in workers:
        if ":gpu" not in worker.id:
            continue
        host, suffix = worker.id.split(":gpu", 1)
        try:
            devices = {int(value) for value in suffix.split(",")}
        except ValueError as exc:
            raise ValueError("GPU worker binding differs") from exc
        if not devices or not devices <= set(range(16)):
            raise ValueError("GPU worker binding differs")
        if worker.task is not None:
            busy_devices.setdefault(host, set()).update(devices)
    idle = []
    for worker in target:
        if ":gpu" not in worker.id:
            continue
        host, suffix = worker.id.split(":gpu", 1)
        values = suffix.split(",")
        if len(values) != 4 or len(set(values)) != 4:
            continue
        devices = {int(value) for value in values}
        if (worker.task is None and worker.last_activity_time is not None
                and (now - worker.last_activity_time).total_seconds() < 120
                and not devices & busy_devices.get(host, set())):
            idle.append(worker.id)
    if len(idle) <= len(queue.entries or []):
        raise ValueError("no verified idle collision-free four-GPU worker")
    return sorted(idle), queue.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=TRAINING, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--acceptance-task-id", required=True)
    parser.add_argument("--source-task-id", required=True)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--dispatch-receipt", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{32}", args.source_task_id):
        raise ValueError("exact published source task ID required")
    formal_gate()
    freeze = verify_freeze(args.seed, args.freeze_receipt)
    training_id, training_source_id, training_source_sha = TRAINING[args.seed]
    checkpoint_sha = freeze["artifacts"]["best-checkpoint"]["sha256"]
    training_receipt_sha = freeze["artifacts"]["training-receipt"]["sha256"]
    upstream(args.source_task_id, "source", SOURCE_SHA)
    upstream(SELECTION_TASK, "selection", SELECTION_SHA)
    upstream(PROJECTION_TASK, "projection", PROJECTION_SHA)
    upstream(training_source_id, "source", training_source_sha)
    upstream(training_id, "best-checkpoint", checkpoint_sha)
    upstream(training_id, "training-receipt", training_receipt_sha)
    acceptance_sha = acceptance_readback(args.acceptance_task_id, args.seed,
                                         training_id, checkpoint_sha,
                                         training_source_id, training_source_sha)
    if urlparse(Session.get_api_server_host()).hostname != "10.100.35.118":
        raise ValueError("private ClearML API host differs")
    os.environ["CLEARML_FILES_HOST"] = "http://10.100.35.118:8081"
    prefix = f"V2V4Real overlap v1 diagnostic validate seed{args.seed} "
    existing = Task.get_tasks(project_name=PROJECT,
                              task_name="^" + re.escape(prefix) + ".*")
    name = prefix + SOURCE_SHA[:12] + " 4xGPU"
    if existing:
        if len(existing) != 1 or existing[0].name != name:
            raise ValueError("existing same-seed diagnostic inference requires audit")
        task = existing[0]
        params = task.get_parameters()
        if (params.get("General/source_task_id") != args.source_task_id
                or params.get("General/source_sha256") != SOURCE_SHA
                or params.get("General/checkpoint_task_id") != training_id
                or params.get("General/checkpoint_sha256") != checkpoint_sha
                or params.get("General/formal_config_sha256") != CONFIG_SHA):
            raise ValueError("existing same-seed inference configuration differs")
        result = {"task_id": task.id, "status": str(task.status),
                  "duplicate_not_created": True}
    else:
        idle, queue_name = idle_workers(args.queue_id)
        task = Task.create(project_name=PROJECT, task_name=name,
                           task_type=Task.TaskTypes.testing, binary="python3")
        task.set_script(repository="", branch="", commit="", working_dir=".",
                        entry_point=BOOTSTRAP.name, diff=BOOTSTRAP.read_text())
        task.set_packages(["clearml==2.1.2"])
        task.set_base_docker(
            IMAGE, docker_arguments="-e CLEARML_AGENT_FORCE_TASK_INIT=0 --shm-size 64g "
            "--network host --env NVIDIA_DRIVER_CAPABILITIES=compute,utility "
            "--env CLEARML_FILES_HOST=http://10.100.34.118:8081")
        task.set_parameters({"source_task_id": args.source_task_id,
                             "source_sha256": SOURCE_SHA,
                             "bootstrap_sha256": BOOTSTRAP_SHA,
                             "formal_config_sha256": CONFIG_SHA,
                             "selection_task_id": SELECTION_TASK,
                             "selection_sha256": SELECTION_SHA,
                             "projection_task_id": PROJECTION_TASK,
                             "projection_sha256": PROJECTION_SHA,
                             "checkpoint_task_id": training_id,
                             "checkpoint_sha256": checkpoint_sha,
                             "checkpoint_seed": args.seed,
                             "training_receipt_sha256": training_receipt_sha,
                             "checkpoint_acceptance_task_id": args.acceptance_task_id,
                             "checkpoint_acceptance_sha256": acceptance_sha,
                             "variant_id": "v2v4real-train-overlap-controlled-v1",
                             "dataset_split": "train",
                             "data_role": "train_internal_validate",
                             "gt_read": False, "official_test_read": False,
                             "paper_eligible": False,
                             "formal_independent_test_eligible": False})
        task.add_tags(["Recover-Before-Fuse", "V2V4Real", "diagnostic-inference",
                       "overlap-controlled-v1", "train-validate-only", "non-formal",
                       "protocol-gated", "4xGPU", f"seed-{args.seed}"])
        Task.enqueue(task, queue_id=args.queue_id)
        result = {"task_id": task.id, "status": str(task.status),
                  "queue_id": args.queue_id, "queue_name": queue_name,
                  "idle_workers_at_dispatch": idle,
                  "duplicate_not_created": False}
    receipt = {"kind": "v2v4real_overlap_v1_diagnostic_inference_protocol_gated_dispatch_v1",
               "checked_at_utc": datetime.now(timezone.utc).isoformat(),
               "seed": args.seed, "source_task_id": args.source_task_id,
               "source_sha256": SOURCE_SHA, "formal_config_sha256": CONFIG_SHA,
               "training_task_id": training_id,
               "checkpoint_sha256": checkpoint_sha,
               "training_receipt_sha256": training_receipt_sha,
               "checkpoint_byte_freeze_receipt_sha256": sha256(args.freeze_receipt),
               "checkpoint_acceptance_task_id": args.acceptance_task_id,
               "checkpoint_acceptance_sha256": acceptance_sha,
               "selection_task_id": SELECTION_TASK,
               "selection_sha256": SELECTION_SHA,
               "projection_task_id": PROJECTION_TASK,
               "projection_sha256": PROJECTION_SHA,
               "paper_eligible": False,
               "formal_independent_test_eligible": False,
               "inference_completed": False, "dispatch": result}
    output = args.dispatch_receipt.absolute()
    if output.is_symlink():
        raise ValueError("dispatch receipt cannot be a symlink")
    if output.exists():
        prior = json.loads(output.read_bytes())
        if (prior.get("kind") != receipt["kind"]
                or prior.get("dispatch", {}).get("task_id") != result["task_id"]
                or prior.get("source_sha256") != SOURCE_SHA
                or prior.get("checkpoint_sha256") != checkpoint_sha):
            raise ValueError("existing dispatch receipt differs")
        print(json.dumps(prior, sort_keys=True))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
