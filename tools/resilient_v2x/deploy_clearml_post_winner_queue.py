#!/usr/bin/env python3
"""Publish and enqueue the remaining sealed post-winner paper jobs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
PROJECT = "ResilientV2X/Training"
DATASET_PROJECT = "ResilientV2X/Datasets"
FILES_SERVER_URI = "http://10.100.34.118:8081"
BASE_EVALUATION_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
BASE_EVALUATION_TASK_ID = "7deb18e532324850bee1fb4279a838b7"
SELECTION_TASK_ID = "d7ea54ce540d4b0486904c5910100885"
SELECTED_IDENTITY_SEAL = (
    "7e24831bc1420b202abedf9c748423223875687c8d19fc2eeadc4ad14f944aad"
)
WINNER = {
    "subject": "resilient_v2x",
    "task_id": "54d28bc513794051810fd383140ae96e",
    "model_id": "ea23e399bc554b07bc2ce07e6b68700e",
    "checkpoint_sha256": (
        "f6a5683a1df7126a3069edce577177eefb80208d9548ba3b61a8907bbf148a67"
    ),
}
CONCAT = {
    "subject": "concat_capacity_matched",
    "task_id": "175e6087f66646e282a93439fecf292a",
    "model_id": "2a3378058b9a453fa47e15a98aa4ca15",
    "checkpoint_sha256": (
        "f47d962959e9f990b0af1020e26b232e7ed1d72b69365ca19c68358316adeb56"
    ),
}
P05_TASK_ID = "59d1c2bbb91741e98f46564accc58477"
SERVICE_QUEUE = "services"
SERVICE_DOCKER = "ubuntu:24.04"
SERVICE_DOCKER_ARGS = (
    "--network host "
    "--add-host apiserver:10.100.34.118 "
    "--add-host fileserver:10.100.34.118 "
    "--add-host webserver:10.100.34.118 "
    "-e CLEARML_API_HOST=http://10.100.34.118:8008 "
    "-e CLEARML_WEB_HOST=http://10.100.34.118:8080 "
    "-e CLEARML_FILES_HOST=http://10.100.34.118:8081"
)
BOOTSTRAP = ROOT / "tools/resilient_v2x/clearml_5090_bootstrap.py"
EVALUATION_INPUT = (
    ROOT / "artifacts/resilient_v2x/post-winner/formal-evaluation-input"
)
RECEIPT_DIR = ROOT / "artifacts/resilient_v2x/post-winner/deployment"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="20260815-v1")
    parser.add_argument(
        "--job-set",
        choices=("all", "duration", "profile"),
        default="all",
        help=(
            "deploy every remaining job, only the six duration jobs, or only "
            "the two complexity profiles and pair validator"
        ),
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _source_package(directory: Path) -> dict[str, object]:
    root = directory.resolve(strict=True)
    inventory_path = root / "source-inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    archives = sorted(root.glob("resilient-v2x-source-*.tar.zst"))
    if len(archives) != 1:
        raise RuntimeError("source package directory must contain exactly one archive")
    archive = archives[0]
    tree = inventory.get("tree_sha256")
    if type(tree) is not str or len(tree) != 64:
        raise RuntimeError("source inventory tree SHA is invalid")
    if archive.name != f"resilient-v2x-source-{tree[:12]}.tar.zst":
        raise RuntimeError("source archive name does not match its inventory")
    return {
        "root": root,
        "archive": archive,
        "archive_name": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "inventory": inventory_path,
        "inventory_sha256": _sha256(inventory_path),
        "tree_sha256": tree,
    }


def _publish_source(package: Mapping[str, object], version: str) -> str:
    from clearml import Dataset

    tree = str(package["tree_sha256"])
    dataset = Dataset.create(
        dataset_project=DATASET_PROJECT,
        dataset_name=f"ResilientV2X post-winner sealed source {tree[:12]}",
        dataset_version=version,
        dataset_tags=["ResilientV2X", "source", "post-winner", "sealed"],
        output_uri=FILES_SERVER_URI,
        description=json.dumps(
            {
                "purpose": "Table VI/VII and E-only/R-only formal evaluation",
                "selected_identity_seal": SELECTED_IDENTITY_SEAL,
                "tree_sha256": tree,
                "archive_sha256": package["archive_sha256"],
                "inventory_sha256": package["inventory_sha256"],
            },
            sort_keys=True,
        ),
    )
    dataset.add_files(
        package["root"],
        wildcard=[str(package["archive_name"]), "source-inventory.json"],
        local_base_folder=str(package["root"]),
        dataset_path="source",
        recursive=False,
        verbose=True,
        max_workers=2,
    )
    dataset.upload(show_progress=True, verbose=True, preview=False, max_workers=2)
    if not dataset.finalize(verbose=True, raise_on_error=True, auto_upload=False):
        raise RuntimeError("post-winner source Dataset did not finalize")
    return dataset.id


def _publish_evaluation_dataset(version: str) -> str:
    from clearml import Dataset

    root = EVALUATION_INPUT.resolve(strict=True)
    registry = root / "formal-evaluation-input-registry.json"
    registry_document = json.loads(registry.read_text(encoding="utf-8"))
    if registry_document.get("sample_count") != 1337:
        raise RuntimeError("post-winner evaluation registry sample count drifted")
    if registry_document.get("sample_ids_sha256") != (
        "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"
    ):
        raise RuntimeError("post-winner evaluation registry cohort drifted")
    dataset = Dataset.create(
        dataset_project=DATASET_PROJECT,
        dataset_name="DAIR-V2X-C ResilientV2X post-winner formal evaluation",
        dataset_version=version,
        dataset_tags=[
            "DAIR-V2X-C",
            "ResilientV2X",
            "post-winner",
            "evaluation",
            "single-seed",
        ],
        parent_datasets=[BASE_EVALUATION_DATASET_ID],
        output_uri=FILES_SERVER_URI,
        description=json.dumps(
            {
                "parent_dataset_id": BASE_EVALUATION_DATASET_ID,
                "selected_identity_seal": SELECTED_IDENTITY_SEAL,
                "protocol_id": "DAIR-CAUSAL-1337-v1",
                "sample_count": 1337,
                "ground_truth_count": 11330,
                "sample_ids_sha256": registry_document["sample_ids_sha256"],
                "registry_sha256": _sha256(registry),
            },
            sort_keys=True,
        ),
    )
    dataset.add_files(
        root / "protocols",
        local_base_folder=str(root),
        recursive=True,
        verbose=True,
        max_workers=4,
    )
    dataset.add_files(
        registry,
        local_base_folder=str(root),
        dataset_path="metadata/post_winner",
        recursive=False,
        verbose=True,
        max_workers=1,
    )
    dataset.upload(show_progress=True, verbose=True, preview=False, max_workers=4)
    if not dataset.finalize(verbose=True, raise_on_error=True, auto_upload=False):
        raise RuntimeError("post-winner evaluation Dataset did not finalize")
    return dataset.id


def _base_parameters(task: object) -> dict[str, object]:
    parameters = dict(task.get_parameters(cast=False))
    for key in tuple(parameters):
        if key.startswith("PostWinner/"):
            parameters.pop(key)
    return parameters


def _embedded_protocol_files(
    source: Path,
    *,
    target_prefix: str,
) -> dict[str, dict[str, object]]:
    root = source.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("embedded protocol source is not a directory")
    result: dict[str, dict[str, object]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise RuntimeError(f"embedded protocol source is a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        target = f"{target_prefix.rstrip('/')}/{relative}"
        payload = path.read_bytes()
        result[target] = {
            "base64": base64.b64encode(payload).decode("ascii"),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if not result:
        raise RuntimeError("embedded protocol source is empty")
    return result


def _task_bootstrap_source(
    *,
    embedded_root: Path | None,
    embedded_prefix: str | None,
) -> str:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    future = "from __future__ import annotations\n"
    initializer = """try:
    from allegroai import Task
except ImportError:
    from clearml import Task
(__name__ != "__main__") or Task.init()
"""
    if source.count(future) != 1:
        raise RuntimeError("post-winner bootstrap future-import anchor drifted")
    source = source.replace(future, future + initializer, 1)
    files: dict[str, dict[str, object]] | None = None
    if embedded_root is not None:
        if not embedded_prefix:
            raise RuntimeError("embedded protocol prefix is required")
        files = _embedded_protocol_files(
            embedded_root,
            target_prefix=embedded_prefix,
        )
    evaluator = (
        ROOT / "tools/resilient_v2x/evaluate_controlled_baselines.py"
    ).read_text(encoding="utf-8")
    profiler = (ROOT / "tools/resilient_v2x/profile.py").read_text(encoding="utf-8")
    files_anchor = (
        "POST_WINNER_EMBEDDED_FILES: Mapping[str, Mapping[str, object]] = {}"
    )
    evaluator_anchor = "POST_WINNER_EMBEDDED_EVALUATOR_SOURCE: str | None = None"
    profiler_anchor = "POST_WINNER_EMBEDDED_PROFILER_SOURCE: str | None = None"
    if (
        source.count(files_anchor) != 1
        or source.count(evaluator_anchor) != 1
        or source.count(profiler_anchor) != 1
    ):
        raise RuntimeError("post-winner bootstrap embed anchors drifted")
    if files is not None:
        source = source.replace(
            files_anchor,
            f"POST_WINNER_EMBEDDED_FILES = {files!r}",
        )
    source = source.replace(
        evaluator_anchor,
        f"POST_WINNER_EMBEDDED_EVALUATOR_SOURCE = {evaluator!r}",
    )
    source = source.replace(
        profiler_anchor,
        f"POST_WINNER_EMBEDDED_PROFILER_SOURCE = {profiler!r}",
    )
    return source


def _create_gpu_task(
    *,
    base: object,
    name: str,
    stage: str,
    source_dataset_id: str,
    package: Mapping[str, object],
    evaluation_dataset_id: str,
    job_id: str,
    subject: Mapping[str, str],
    tags: Sequence[str],
    extra: Mapping[str, object] | None = None,
    auto_model: bool = False,
    embedded_root: Path | None = None,
    embedded_prefix: str | None = None,
) -> object:
    from clearml import Task

    task = Task.clone(
        source_task=base,
        name=name,
        parent=SELECTION_TASK_ID,
        project=str(getattr(base, "project", "") or ""),
    )
    if str(task.status) != "created":
        raise RuntimeError(f"cloned task is not created: {task.id} {task.status}")
    parameters = _base_parameters(base)
    parameters.update(
        {
            "Args/training_dataset_id": evaluation_dataset_id,
            "Args/stage": stage,
            "Args/controlled_baseline": subject["subject"],
            "Args/controlled_baseline_task_id": subject["task_id"],
            "Args/predecessor_task_id": subject["task_id"],
            "Args/post_winner_job_id": job_id,
            "Args/post_winner_selected_identity_seal": SELECTED_IDENTITY_SEAL,
            "PostWinner/job_id": job_id,
            "PostWinner/selected_identity_seal": SELECTED_IDENTITY_SEAL,
            "PostWinner/runtime_source_dataset_id": source_dataset_id,
            "PostWinner/evaluation_dataset_id": evaluation_dataset_id,
        }
    )
    if auto_model:
        parameters.pop("Args/controlled_baseline_model_id", None)
        parameters.pop("Args/controlled_baseline_checkpoint_sha256", None)
    else:
        parameters.update(
            {
                "Args/controlled_baseline_model_id": subject["model_id"],
                "Args/controlled_baseline_checkpoint_sha256": subject[
                    "checkpoint_sha256"
                ],
            }
        )
    if extra:
        parameters.update(extra)
    bootstrap_source = _task_bootstrap_source(
        embedded_root=embedded_root,
        embedded_prefix=embedded_prefix,
    )
    task.set_script(
        repository="",
        branch="",
        commit="",
        diff=bootstrap_source,
        working_dir=".",
        entry_point=BOOTSTRAP.name,
    )
    task.set_parameters(parameters)
    task.output_uri = FILES_SERVER_URI
    task.set_parent(SELECTION_TASK_ID)
    task.set_tags(
        sorted(
            set(tags)
            | {
                "ResilientV2X-suite",
                "post-winner",
                "single-seed",
                "seed-20250218",
                "sealed-source",
            }
        )
    )
    task.flush(wait_for_uploads=True)
    task.reload()
    if hashlib.sha256((task.data.script.diff or "").encode()).hexdigest() != (
        hashlib.sha256(bootstrap_source.encode()).hexdigest()
    ):
        raise RuntimeError(f"task script readback drifted: {task.id}")
    return task


def _enqueue(task: object, queue: str) -> None:
    from clearml import Task

    error: Exception | None = None
    try:
        response = Task.enqueue(task=task, queue_name=queue)
        if response is None or response is False:
            error = RuntimeError("ClearML did not confirm enqueue")
    except Exception as caught:  # authoritative readback decides acceptance
        error = caught
    for attempt in range(5):
        authoritative = Task.get_task(task_id=task.id)
        if str(authoritative.status) in {"queued", "in_progress", "completed"}:
            return
        if attempt != 4:
            time.sleep(1)
    raise RuntimeError(f"enqueue failed for {task.id}: {error!r}")


DISPATCHER_SOURCE = r'''#!/usr/bin/env python3
import json, time
from clearml import Task

current = Task.init()
p = current.get_parameters(cast=False)
training_id = p["Dispatcher/training_task_id"]
evaluation_id = p["Dispatcher/evaluation_task_id"]
queue = p["Dispatcher/queue"]
while True:
    training = Task.get_task(task_id=training_id)
    status = str(training.status)
    if status == "completed":
        models = [m for m in training.get_models().get("output", ()) if m.name == "ResilientV2X resilient_v2x final checkpoint"]
        if len(models) != 1:
            raise RuntimeError("completed p=0.5 task has no unique final OutputModel")
        target = Task.get_task(task_id=evaluation_id)
        target_status = str(target.status)
        if target_status == "created":
            try:
                Task.enqueue(task=target, queue_name=queue)
            except Exception:
                pass
        for _ in range(5):
            target = Task.get_task(task_id=evaluation_id)
            if str(target.status) in {"queued", "in_progress", "completed"}:
                receipt = {"training_task_id": training_id, "model_id": models[0].id, "evaluation_task_id": evaluation_id, "queue": queue, "status": str(target.status)}
                current.upload_artifact("p05_evaluation_dispatch", artifact_object=receipt, wait_on_upload=True)
                current.flush(wait_for_uploads=True)
                raise SystemExit(0)
            time.sleep(1)
        raise RuntimeError("p=0.5 evaluation enqueue was not authoritative")
    if status in {"failed", "aborted", "closed", "published"}:
        raise RuntimeError(f"p=0.5 training reached terminal status {status!r}")
    time.sleep(60)
'''


PAIR_VALIDATOR_SOURCE = r'''#!/usr/bin/env python3
import hashlib, json, time
from clearml import Task

current = Task.init()
p = current.get_parameters(cast=False)
ids = [p["Pair/winner_task_id"], p["Pair/concat_task_id"]]
tasks = []
while True:
    tasks = [Task.get_task(task_id=value) for value in ids]
    statuses = [str(task.status) for task in tasks]
    if all(value == "completed" for value in statuses):
        break
    if any(value in {"failed", "aborted", "closed", "published"} for value in statuses):
        raise RuntimeError(f"profile dependency failed: {statuses}")
    time.sleep(30)
docs = []
for task in tasks:
    artifact = task.artifacts.get("deployment_profile")
    if artifact is None:
        raise RuntimeError("completed profile task has no deployment_profile")
    document = artifact.get()
    if not isinstance(document, dict):
        raise RuntimeError("deployment_profile is not an object")
    docs.append(document)
left, right = docs
if left["parameters"]["parameter_count"] != right["parameters"]["parameter_count"]:
    raise RuntimeError("capacity-matched concat parameter count differs from winner")
for key in ("batch_size", "measurement_boundary"):
    if left[key] != right[key]:
        raise RuntimeError(f"profile pair differs in {key}")
for key in ("name", "total_memory_bytes", "compute_capability", "torch_version", "cuda_runtime", "cudnn_version"):
    if left["device"].get(key) != right["device"].get(key):
        raise RuntimeError(f"profile pair device differs in {key}")
for key in ("warmup_iterations", "measured_iterations", "synchronized"):
    if left["runtime"]["latency"].get(key) != right["runtime"]["latency"].get(key):
        raise RuntimeError(f"profile pair timing differs in {key}")
def overlay_hash(document):
    values = [item["sha256"] for item in document["artifacts"] if item["role"] == "evaluation_overlay_index"]
    if len(values) != 1:
        raise RuntimeError("profile has no unique evaluation overlay identity")
    return values[0]
if overlay_hash(left) != overlay_hash(right):
    raise RuntimeError("profile pair input overlay differs")
receipt = {"schema_version": 1, "status": "pass", "winner_task_id": ids[0], "concat_task_id": ids[1], "parameter_count": left["parameters"]["parameter_count"], "device": left["device"]["name"], "overlay_sha256": overlay_hash(left)}
receipt["content_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
current.upload_artifact("complexity_pair_validation", artifact_object=receipt, wait_on_upload=True)
current.flush(wait_for_uploads=True)
'''


def _create_service_task(
    *,
    name: str,
    entry_point: str,
    source: str,
    parameters: Mapping[str, object],
    tags: Sequence[str],
) -> object:
    from clearml import Task

    task = Task.create(
        project_name=PROJECT,
        task_name=name,
        task_type=Task.TaskTypes.controller,
        script=None,
        packages=None,
        docker=None,
        docker_args=None,
        add_task_init_call=False,
        binary="python",
        detect_repository=False,
    )
    task.set_script(
        repository="",
        branch="",
        commit="",
        diff=source,
        working_dir=".",
        entry_point=entry_point,
    )
    task.set_packages(["clearml==2.1.11"])
    task.set_base_docker(
        docker_image=SERVICE_DOCKER,
        docker_arguments=SERVICE_DOCKER_ARGS,
    )
    task.set_parent(SELECTION_TASK_ID)
    task.output_uri = FILES_SERVER_URI
    task.set_parameters(dict(parameters))
    task.set_tags(sorted(set(tags) | {"post-winner", "controller"}))
    task.flush(wait_for_uploads=True)
    return task


def _write_deployment_receipt(
    *,
    version: str,
    job_set: str,
    source_dataset_id: str,
    evaluation_dataset_id: str,
    tasks: Mapping[str, object],
) -> int:
    receipt = {
        "schema_version": 1,
        "job_set": job_set,
        "source_dataset_id": source_dataset_id,
        "bootstrap_sha256": _sha256(BOOTSTRAP),
        "evaluator_sha256": _sha256(
            ROOT / "tools/resilient_v2x/evaluate_controlled_baselines.py"
        ),
        "overlay_delivery": "sealed-task-script-embedded",
        "evaluation_dataset_id": evaluation_dataset_id,
        "selected_identity_seal": SELECTED_IDENTITY_SEAL,
        "tasks": {
            key: {"task_id": value.id, "status": str(value.status)}
            for key, value in tasks.items()
        },
    }
    receipt["content_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = RECEIPT_DIR / f"post-winner-deployment-{version}.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"receipt": str(receipt_path), **receipt}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from clearml import Task

    base = Task.get_task(task_id=BASE_EVALUATION_TASK_ID)
    if str(base.status) != "completed":
        raise RuntimeError("formal evaluation base task is not completed")
    base_parameters = _base_parameters(base)
    source_dataset_id = str(base_parameters["Args/source_dataset_id"])
    evaluation_dataset_id = BASE_EVALUATION_DATASET_ID
    package: dict[str, object] = {}
    tasks: dict[str, object] = {}
    duration_specs = [
        (1, "lidar", "L-Fail", "GPU4-5090"),
        (1, "camera", "C-Fail", "GPU4-5090"),
        (2, "lidar", "L-Fail", "GPU4-A100"),
        (2, "camera", "C-Fail", "GPU4-A100"),
        (3, "lidar", "L-Fail", "GPU4-5090"),
        (3, "camera", "C-Fail", "GPU4-A100"),
    ]
    if args.job_set in {"all", "duration"}:
        for duration, modality, condition, queue in duration_specs:
            key = f"q{duration}_{modality}"
            job_id = f"table_vii.duration.q{duration}.{modality}.evaluate"
            task = _create_gpu_task(
                base=base,
                name=f"ResilientV2X TableVII q={duration} {modality} E+R formal",
                stage="post_winner_evaluate",
                source_dataset_id=source_dataset_id,
                package=package,
                evaluation_dataset_id=evaluation_dataset_id,
                job_id=job_id,
                subject=WINNER,
                tags=["table-vii", "duration", f"q-{duration}", modality, "E+R"],
                extra={
                    "Args/post_winner_overlay_index": (
                        f"protocols/post_winner/q_{duration}/evaluation_overlays.json"
                    ),
                    "Args/post_winner_delays": [0],
                    "Args/post_winner_conditions": [condition],
                    "Args/post_winner_agent_scope": "E+R",
                    "Args/post_winner_duration_ticks": duration,
                    "Args/post_winner_expected_runs": 1,
                },
                embedded_root=(
                    ROOT
                    / f"artifacts/resilient_v2x/post-winner/duration/q_{duration}"
                ),
                embedded_prefix=f"protocols/post_winner/q_{duration}",
            )
            tasks[key] = task
            _enqueue(task, queue)

    if args.job_set == "duration":
        return _write_deployment_receipt(
            version=args.version,
            job_set=args.job_set,
            source_dataset_id=source_dataset_id,
            evaluation_dataset_id=evaluation_dataset_id,
            tasks=tasks,
        )

    if args.job_set == "all":
        for scope, queue in (("E-only", "GPU4-5090"), ("R-only", "GPU4-A100")):
            key = scope.lower().replace("-", "_")
            task = _create_gpu_task(
                base=base,
                name=f"ResilientV2X diagnostic {scope} 1337x8 formal",
                stage="post_winner_evaluate",
                source_dataset_id=source_dataset_id,
                package=package,
                evaluation_dataset_id=evaluation_dataset_id,
                job_id=f"diagnostic.{key}.evaluate",
                subject=WINNER,
                tags=["diagnostic", scope, "8-condition"],
                extra={
                    "Args/post_winner_overlay_index": (
                        f"protocols/post_winner/{key}/evaluation_overlays.json"
                    ),
                    "Args/post_winner_delays": [0, 100, 200, 300],
                    "Args/post_winner_conditions": ["L-Fail", "C-Fail"],
                    "Args/post_winner_agent_scope": scope,
                    "Args/post_winner_duration_ticks": 1,
                    "Args/post_winner_expected_runs": 8,
                },
                embedded_root=(EVALUATION_INPUT / "protocols/dair_v2"),
                embedded_prefix=f"protocols/post_winner/{key}",
            )
            tasks[key] = task
            _enqueue(task, queue)

    for key, subject, config in (
        ("winner_profile", WINNER, "winner"),
        ("concat_profile", CONCAT, "concat"),
    ):
        task = _create_gpu_task(
            base=base,
            name=f"ResilientV2X TableVII complexity {config} formal profile",
            stage="post_winner_profile",
            source_dataset_id=source_dataset_id,
            package=package,
            evaluation_dataset_id=evaluation_dataset_id,
            job_id=f"table_vii.complexity.{config}.profile",
            subject=subject,
            tags=["table-vii", "complexity", "profile", config],
            extra={
                "Args/post_winner_profile_config": config,
                "Args/post_winner_profile_warmup": 10,
                "Args/post_winner_profile_iterations": 100,
            },
        )
        tasks[key] = task
        _enqueue(task, "GPU4-5090")

    if args.job_set == "profile":
        pair = _create_service_task(
            name="ResilientV2X TableVII complexity pair validator",
            entry_point="post_winner_complexity_pair_validator.py",
            source=PAIR_VALIDATOR_SOURCE,
            parameters={
                "Pair/winner_task_id": tasks["winner_profile"].id,
                "Pair/concat_task_id": tasks["concat_profile"].id,
                "Pair/selected_identity_seal": SELECTED_IDENTITY_SEAL,
            },
            tags=["table-vii", "complexity", "pair-validation"],
        )
        tasks["pair_validator"] = pair
        _enqueue(pair, SERVICE_QUEUE)
        return _write_deployment_receipt(
            version=args.version,
            job_set=args.job_set,
            source_dataset_id=source_dataset_id,
            evaluation_dataset_id=evaluation_dataset_id,
            tasks=tasks,
        )

    p05_eval = _create_gpu_task(
        base=base,
        name="ResilientV2X TableVI formal-v2 eval p=0.5 seed20250218 auto",
        stage="post_winner_auto_validate",
        source_dataset_id=source_dataset_id,
        package=package,
        evaluation_dataset_id=BASE_EVALUATION_DATASET_ID,
        job_id="table_vi.p_0_5.evaluate",
        subject={"subject": "resilient_v2x", "task_id": P05_TASK_ID},
        tags=["table-vi", "p-0_5", "12-condition", "dependency-gated"],
        auto_model=True,
    )
    tasks["p05_eval"] = p05_eval

    dispatcher = _create_service_task(
        name="ResilientV2X TableVI p=0.5 evaluation dispatcher",
        entry_point="post_winner_p05_dispatcher.py",
        source=DISPATCHER_SOURCE,
        parameters={
            "Dispatcher/training_task_id": P05_TASK_ID,
            "Dispatcher/evaluation_task_id": p05_eval.id,
            "Dispatcher/queue": "GPU4-A100",
            "Dispatcher/selected_identity_seal": SELECTED_IDENTITY_SEAL,
        },
        tags=["table-vi", "p-0_5", "dependency-watcher"],
    )
    tasks["p05_dispatcher"] = dispatcher
    _enqueue(dispatcher, SERVICE_QUEUE)

    pair = _create_service_task(
        name="ResilientV2X TableVII complexity pair validator",
        entry_point="post_winner_complexity_pair_validator.py",
        source=PAIR_VALIDATOR_SOURCE,
        parameters={
            "Pair/winner_task_id": tasks["winner_profile"].id,
            "Pair/concat_task_id": tasks["concat_profile"].id,
            "Pair/selected_identity_seal": SELECTED_IDENTITY_SEAL,
        },
        tags=["table-vii", "complexity", "pair-validation"],
    )
    tasks["pair_validator"] = pair
    _enqueue(pair, SERVICE_QUEUE)

    return _write_deployment_receipt(
        version=args.version,
        job_set=args.job_set,
        source_dataset_id=source_dataset_id,
        evaluation_dataset_id=evaluation_dataset_id,
        tasks=tasks,
    )


if __name__ == "__main__":
    raise SystemExit(main())
