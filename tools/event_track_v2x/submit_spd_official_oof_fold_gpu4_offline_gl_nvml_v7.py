#!/usr/bin/env python3
"""Queue an independently byte-read-back SPD OOF fold on four compatible GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from run_cooptrack_official_oof_gpu4_offline_gl_v6 import validate_cohort


PROJECT = "Thesis/EventTrack-V2X/Training"
DOCKER = "gitlab.zhht.ai.com:5000/aitech/ultralytics_rfdetr@sha256:e5b249d993f9675971b328152aab4e1023930f3480a80dd9cfd6e9620b208bfb"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def physical_binding(worker_id: str) -> tuple[str, set[int]]:
    match = re.fullmatch(r"([^:]+):gpu([0-9]+(?:,[0-9]+)*)", worker_id)
    if not match:
        raise ValueError("unparseable GPU Worker identity: " + worker_id)
    indices = [int(v) for v in match.group(2).split(",")]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate physical GPU index")
    return match.group(1), set(indices)


def eligible_workers(client, queue_name: str) -> list[str]:
    queues = [q for q in client.queues.get_all(name=queue_name) if q.name == queue_name]
    if len(queues) != 1:
        raise RuntimeError("queue name must resolve uniquely")
    queue = queues[0]
    if getattr(queue, "entries", None):
        raise RuntimeError("queue has pending tasks; avoid accidental duplicate allocation")
    workers = list(client.workers.get_all())
    candidates = []
    for worker in workers:
        if not any(q.id == queue.id for q in worker.queues or []):
            continue
        host, devices = physical_binding(worker.id)
        if len(devices) != 4 or getattr(getattr(worker, "task", None), "id", None):
            continue
        collision = False
        for other in workers:
            active = getattr(getattr(other, "task", None), "id", None)
            if not active:
                continue
            other_host, other_devices = physical_binding(other.id)
            if other_host == host and devices & other_devices:
                collision = True
                break
        if not collision:
            candidates.append(worker.id)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, choices=(1337, 2027, 3407), required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--retry-of")
    parser.add_argument("--dependency-receipt", required=True, type=Path)
    parser.add_argument("--communication-receipt", required=True, type=Path)
    args = parser.parse_args()
    from clearml import Task
    from clearml.backend_api.session.client import APIClient

    communication = json.loads(args.communication_receipt.read_bytes())
    if (communication.get("task_id") != "cb451dd7590c41c5a84d4a5ac9bca545"
            or communication.get("status") != "independent_bytes_verified"
            or communication.get("communication_accepted") is not True):
        raise ValueError("NVML utility communication proof not accepted")
    comm_task = Task.get_task(task_id=communication["task_id"])
    if comm_task.status != "completed":
        raise ValueError("communication probe no longer completed")
    report_path = args.communication_receipt.parent / "nccl-diagnostic.json"
    report = json.loads(report_path.read_bytes())
    report_row = next(r for r in communication["artifacts"] if r["name"] == "nccl-diagnostic")
    if (sha256(report_path) != report_row["sha256"]
            or comm_task.artifacts["nccl-diagnostic"].hash != report_row["sha256"]
            or comm_task.artifacts["nccl-diagnostic"].size != report_row["bytes"]
            or report.get("controller_source_sha256") != "07501f1c458c9597ed1576ee120feae46562b186f0faabc43732d6d2df761601"
            or report.get("runtime_sha256") != "b6b39c66eec6921c0f0f51d2b9cd4e5571fcafdc44d3605112dfb349af8fab32"
            or report.get("gt_or_dataset_loaded") is not False
            or report.get("model_or_optimizer_created") is not False
            or "NVIDIA_DRIVER_CAPABILITIES=compute,utility" not in comm_task.get_base_docker()):
        raise ValueError("communication proof/source/runtime/container mismatch")
    baseline = next(r for r in report["results"] if r["case"] == "baseline")
    if (not baseline["communication_accepted"] or baseline["returncode"] != 0
            or baseline["environment_overrides"] or baseline["timeout"]
            or sorted(r["rank"] for r in baseline["ranks"]) != list(range(4))
            or not all(r["broadcast_passed"] and r["all_reduce_passed"] for r in baseline["ranks"])):
        raise ValueError("default transport four-rank communication not accepted")

    dependency = json.loads(args.dependency_receipt.read_bytes())
    if (dependency.get("kind") != "spd_offline_libgl_dependency_clearml_independent_readback_v1"
            or dependency.get("status") != "independent_bytes_verified"
            or dependency.get("target_runtime_import_accepted") is not True
            or dependency["artifacts"]["libgl"]["sha256"] != "5ac68c58a292e435a0ee55c98f0bd2720a9b088343afc813c262bdc552cc0e10"
            or dependency["artifacts"]["manifest"]["sha256"] != "2905061e79777c2858471fa2ee3ccbfaf3cf064fa62429dfa4e1424d34530f82"):
        raise ValueError("offline dependency was not independently admitted")
    dep_task = Task.get_task(task_id=dependency["task_id"])
    if dep_task.status != "completed" or any(dep_task.artifacts[n].hash != r["sha256"] or dep_task.artifacts[n].size != r["bytes"] for n,r in dependency["artifacts"].items()):
        raise ValueError("offline dependency current artifacts differ")
    manifest_path = args.package / "package-manifest.json"
    manifest_sha = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["fold_id"] != args.fold:
        raise ValueError("fold/package mismatch")
    validate_cohort(manifest)
    upload = json.loads((args.package / "clearml-upload-acceptance.json").read_bytes())
    if (upload["fold_id"] != args.fold or upload["manifest_sha256"] != manifest_sha
            or upload["status"] != "uploaded_and_hash_verified"):
        raise ValueError("ClearML package upload has not passed acceptance")
    package_task = Task.get_task(task_id=upload["task_id"])
    readback_path = args.package / "clearml-independent-readback.json"
    readback = json.loads(readback_path.read_bytes())
    if (readback.get("kind") != "spd_official_oof_fold_clearml_independent_readback_v1"
            or readback.get("status") != "independent_bytes_verified"
            or readback.get("task_id") != package_task.id
            or readback.get("fold_id") != args.fold
            or readback.get("manifest_sha256") != manifest_sha
            or readback.get("artifacts", {}).get("package-manifest", {}).get("sha256") != manifest_sha
            or readback.get("artifacts", {}).get("train-inputs", {}).get("sha256") != upload["train_inputs_sha256"]
            or readback["artifacts"]["package-manifest"]["bytes"] != manifest_path.stat().st_size
            or readback["artifacts"]["train-inputs"]["bytes"] != next(
                r["bytes"] for r in manifest["inventory"] if r["path"] == "train-inputs.tar.gz")):
        raise ValueError("package has not passed independent ClearML byte readback")
    if package_task.status != "completed":
        raise RuntimeError("package task not completed")
    expected_artifacts = {"package-manifest": manifest_sha,
                          "train-inputs": upload["train_inputs_sha256"]}
    if any(name not in package_task.artifacts or package_task.artifacts[name].hash != digest
           for name, digest in expected_artifacts.items()):
        raise RuntimeError("ClearML package artifact hash differs")
    controller = Path(__file__).with_name("run_cooptrack_official_oof_gpu4_offline_gl_v6.py")
    controller_raw = controller.read_bytes()
    controller_sha = hashlib.sha256(controller_raw).hexdigest()
    freeze = Path("/Volumes/Data/test/recover-before-fuse/source-freezes/"
                  "spd-official-oof-offline-gl-v6-20261001/run_cooptrack_official_oof_gpu4_offline_gl_v6.py")
    if freeze.read_bytes() != controller_raw:
        raise ValueError("controller differs from frozen source")
    base_name = f"D2 SPD official OOF fold-{args.fold} seed-{args.seed} detector {manifest_sha[:12]}"
    name = base_name + " offline-gl-nvml-v7 " + controller_sha[:12]
    existing = Task.get_tasks(project_name=PROJECT, task_name="^" + re.escape(base_name) + ".*$")
    same = [task for task in existing if task.name == name]
    if same:
        if len(same) != 1:
            raise RuntimeError("duplicate retry tasks require investigation")
        print("EXISTING_TRAINING_TASK", same[0].id, same[0].status, flush=True)
        return
    if existing:
        if not args.retry_of or args.retry_of not in {task.id for task in existing}:
            raise RuntimeError("existing failures require explicit preserved retry linkage")
        for prior in existing:
            if prior.status != "failed" or prior.get_last_iteration() != 0:
                raise RuntimeError("all prior fits must be failed startup before optimizer artifacts")
            if prior.artifacts:
                proof_path = Path("/Volumes/Data/test/recover-before-fuse/artifacts/spd-oof-fold2-v100-v6-startup-failure-readback-20261001/acceptance-receipt.json")
                if (prior.id != "d0d2b630acdb47adbdb297173d687d9f"
                        or sha256(proof_path) != "0c486cd1ed42bc8fb939bf812cf040e41397df3595dbf821d36f18d318fca48b"
                        or set(prior.artifacts) != {"probe-b2-vehicle-side-profile"}):
                    raise RuntimeError("prior task contains unadmitted artifacts")
                proof = json.loads(proof_path.read_bytes());row = proof["artifacts"]["probe-b2-vehicle-side-profile"]
                profile_path = proof_path.parent / row["path"]
                profile = json.loads(profile_path.read_bytes())
                artifact = prior.artifacts["probe-b2-vehicle-side-profile"]
                if (sha256(profile_path) != row["sha256"] or artifact.hash != row["sha256"]
                        or artifact.size != row["bytes"] or not profile["batch_probe_only"]
                        or profile["success"] or profile["returncode"] == 0 or profile["ranks"]):
                    raise RuntimeError("prior failure profile no longer proves zero-rank startup")
            params = prior.get_parameters()
            if (params.get("General/package_manifest_sha256") != manifest_sha
                    or str(params.get("General/seed")) != str(args.seed)
                    or str(params.get("General/fold_id")) != str(args.fold)):
                raise ValueError("prior failure fold/seed/package differs")
    elif args.retry_of:
        raise RuntimeError("specified failed task is not the same configuration")
    candidates = eligible_workers(APIClient(), args.queue)
    if not candidates:
        raise RuntimeError("no idle collision-free four-GPU Worker in selected queue")
    task = Task.create(project_name=PROJECT, task_name=name,
                       task_type=Task.TaskTypes.training, binary="python3.12")
    task.set_script(repository="", branch="", commit="", working_dir=".",
                    entry_point=controller.name, diff=controller_raw.decode())
    task.set_packages(["clearml==2.1.2"])
    task.set_base_docker(DOCKER,
                         docker_arguments="-e CLEARML_AGENT_FORCE_TASK_INIT=0 --shm-size 32g --env NCCL_P2P_DISABLE=1 --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env CLEARML_FILES_HOST=http://10.100.34.118:8081 ")
    task.set_parameters({"package_task_id": package_task.id,
                         "package_manifest_sha256": manifest_sha,
                         "package_independent_readback_sha256": sha256(readback_path),
                         "controller_sha256": controller_sha,
                         "environment_dependency_task_id": dep_task.id,
                         "environment_dependency_receipt_sha256": sha256(args.dependency_receipt),
                         "fold_id": args.fold, "seed": args.seed,
                         "cohort": "official-oof-fold-fit",
                         "fit_sequence_count": manifest["train_sequence_count"],
                         "world_size": 4, "epochs_per_side": 24,
                         "supersedes_failed_startup_task_id": args.retry_of or "",
                         "container_system_dependencies": "offline-libgl-no-network",
                         "nvidia_driver_capabilities": "compute,utility",
                         "communication_probe_task_id": comm_task.id,
                         "communication_receipt_sha256": sha256(args.communication_receipt),
                         "val_or_test_included": False,
                         "queue": args.queue})
    task.add_tags(["SPD", "canonical-OOF", f"fold-{args.fold}",
                   f"seed-{args.seed}", "fit-only", "paper-evidence-pending"])
    receipt = {"kind": "spd_official_oof_gpu4_training_dispatch_v1",
               "task_id": task.id, "package_task_id": package_task.id,
               "package_manifest_sha256": manifest_sha,
               "package_independent_readback_sha256": sha256(readback_path),
               "controller_sha256": controller_sha,
                         "environment_dependency_task_id": dep_task.id,
                         "environment_dependency_receipt_sha256": sha256(args.dependency_receipt), "fold_id": args.fold,
               "seed": args.seed, "queue": args.queue,
               "eligible_physical_workers_at_dispatch": candidates,
               "supersedes_failed_startup_task_id": args.retry_of,
               "communication_probe_task_id": comm_task.id,
               "communication_receipt_sha256": sha256(args.communication_receipt),
               "nvidia_driver_capabilities": "compute,utility",
               "status": "created_before_enqueue"}
    output = args.package / f"fold-{args.fold}-seed-{args.seed}-offline-gl-nvml-v7-training-dispatch.json"
    with output.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    Task.enqueue(task, queue_name=args.queue)
    queued = Task.get_task(task_id=task.id)
    if queued.status not in ("queued", "in_progress"):
        raise RuntimeError("ClearML did not confirm queued or running state for created task")
    accepted = args.package / f"fold-{args.fold}-seed-{args.seed}-offline-gl-nvml-v7-training-enqueue-acceptance.json"
    with accepted.open("x") as stream:
        json.dump({**receipt, "status": queued.status}, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print("SPD_OOF_TRAINING_ENQUEUED",json.dumps(receipt,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
