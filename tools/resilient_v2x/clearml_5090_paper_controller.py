#!/usr/bin/env python3
"""Run the formal student-to-validation ClearML chain without a pipeline DAG.

The controller deliberately creates no validation task until the formal student
task is completed and its final OutputModel has been downloaded and hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urlsplit


WORKER_QUEUE = "GPU4-5090"
FILES_SERVER_URI = "http://10.100.34.118:8081"
FILES_SERVER_HOST = "10.100.34.118"
FILES_SERVER_PORT = 8081
EXPECTED_GPUS = 4
EXPECTED_TRAIN_BATCH_SIZE_PER_GPU = 1
EXPECTED_MAX_EPOCHS = 50
_CUDA_VISIBLE_DEVICES_ARG = re.compile(
    r"(?:--env|-e)\s+CUDA_VISIBLE_DEVICES=[^\s]+"
)
EXPECTED_CONDITION_COUNT = 12
CLEAN_TEACHER_MODEL_NAME = "ResilientV2X clean teacher"
DISTILLED_STUDENT_MODEL_NAME = "ResilientV2X distilled student"
VALIDATION_SUMMARY_DOCUMENT_TYPE = "resilient_v2x_validation_summary"
CONDITION_RESULT_DOCUMENT_TYPE = "resilient_v2x_condition_result"
PAPER_CONTROLLER_SUMMARY_DOCUMENT_TYPE = "resilient_v2x_paper_controller_summary"
BOOTSTRAP_ENTRY_POINT = "tools/resilient_v2x/clearml_5090_bootstrap.py"
CLEARML_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
TERMINAL_NONCOMPLETED_STATUSES = frozenset({"failed", "stopped", "closed", "published"})
EXPECTED_CONDITIONS = tuple(
    f"delay_{delay_ms:03d}_{condition}"
    for delay_ms in (0, 100, 200, 300)
    for condition in ("full", "l_fail", "c_fail")
)
PROVENANCE_PARAMETER_KEYS = (
    "Args/source_dataset_id",
    "Args/source_archive_name",
    "Args/source_archive_bytes",
    "Args/source_archive_sha256",
    "Args/training_dataset_id",
    "Args/native_bundle_bytes",
    "Args/native_bundle_sha256",
    "Args/build_manifest_sha256",
)
FORBIDDEN_LOCAL_HANDOFF_PARAMETERS = (
    "Args/teacher_checkpoint",
    "Args/student_checkpoint",
    "Args/experiment_from_task",
    "Args/predecessor_task_id",
)
REQUIRED_AP_METRICS = (
    "resilient_v2x/car_bev_ap_r40_0.50",
    "resilient_v2x/car_bev_ap_r40_0.70",
    "resilient_v2x/car_3d_ap_r40_0.50",
    "resilient_v2x/car_3d_ap_r40_0.70",
)


class ModelPin(NamedTuple):
    task_id: str
    model_id: str
    checkpoint_sha256: str


class ControllerResult(NamedTuple):
    validation_task_id: str
    student_model_id: str
    student_checkpoint_sha256: str
    validation_summary_sha256: str
    condition_count: int


def _clearml_id_argument(value: str) -> str:
    if CLEARML_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("value must be a lowercase 32-hex ClearML ID")
    return value


def _sha256_argument(value: str) -> str:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("value must be a lowercase SHA-256")
    return value


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student-task-id",
        required=True,
        type=_clearml_id_argument,
        help="formal completed student task; it is also the verified clone source",
    )
    parser.add_argument(
        "--student-model-id",
        type=_clearml_id_argument,
        help="optional expected final student OutputModel ID",
    )
    parser.add_argument(
        "--student-checkpoint-sha256",
        type=_sha256_argument,
        help="optional expected final student checkpoint SHA-256",
    )
    parser.add_argument(
        "--teacher-task-id",
        required=True,
        type=_clearml_id_argument,
    )
    parser.add_argument(
        "--teacher-model-id",
        required=True,
        type=_clearml_id_argument,
    )
    parser.add_argument(
        "--teacher-checkpoint-sha256",
        required=True,
        type=_sha256_argument,
    )
    parser.add_argument(
        "--allow-failed-teacher-task",
        action="store_true",
        help="reuse a failed teacher task only through its sealed epoch-50 output",
    )
    parser.add_argument("--project", default="ResilientV2X/Training")
    parser.add_argument(
        "--name",
        default="ResilientV2X RTX5090 formal 12-condition validation",
    )
    parser.add_argument("--poll-seconds", type=_positive_float, default=30.0)
    parser.add_argument(
        "--timeout-seconds",
        type=_nonnegative_float,
        default=0.0,
        help="zero waits indefinitely",
    )
    return parser


def _normalized_task_status(task: object) -> str:
    status = getattr(task, "status", None)
    if callable(status):
        status = status()
    if status is None:
        getter = getattr(task, "get_status", None)
        if callable(getter):
            status = getter()
    value = getattr(status, "value", status)
    return str(value).rsplit(".", 1)[-1].lower()


def _task_id(task: object, *, context: str) -> str:
    value = str(getattr(task, "id", "") or "")
    if CLEARML_ID_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"{context} has an invalid ClearML ID: {value!r}")
    return value


def _project_id(task_class: object, project: str) -> str:
    if CLEARML_ID_PATTERN.fullmatch(project) is not None:
        return project
    getter = getattr(task_class, "get_project_id", None)
    if not callable(getter):
        raise RuntimeError("ClearML Task class cannot resolve a project ID")
    try:
        value = getter(project_name=project, search_hidden=True)
    except TypeError:
        value = getter(project_name=project)
    result = str(value or "")
    if CLEARML_ID_PATTERN.fullmatch(result) is None:
        raise RuntimeError(
            f"ClearML project does not resolve to an ID: {project!r}"
        )
    return result


def _connect_controller_task(task_class: object) -> object:
    current_getter = getattr(task_class, "current_task", None)
    current = current_getter() if callable(current_getter) else None
    if current is not None:
        return current
    initializer = getattr(task_class, "init", None)
    if not callable(initializer):
        raise RuntimeError("ClearML Task class cannot initialize the controller")
    return initializer(
        project_name="ResilientV2X/Training",
        task_name="ResilientV2X paper validation controller",
        reuse_last_task_id=False,
        output_uri=FILES_SERVER_URI,
        auto_connect_arg_parser=True,
    )


def _reload(task: object) -> None:
    reloader = getattr(task, "reload", None)
    if callable(reloader):
        reloader()


def _wait_for_completed_task(
    task: object,
    *,
    context: str,
    poll_seconds: float,
    timeout_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> object:
    """Explicitly poll one task and accept only the completed state."""

    if poll_seconds <= 0.0:
        raise ValueError("poll interval must be positive")
    if timeout_seconds < 0.0:
        raise ValueError("timeout must be non-negative")
    started = monotonic_fn()
    while True:
        _reload(task)
        status = _normalized_task_status(task)
        if status == "completed":
            return task
        if status in TERMINAL_NONCOMPLETED_STATUSES:
            raise RuntimeError(f"{context} ended without completion: {status!r}")
        if timeout_seconds and monotonic_fn() - started >= timeout_seconds:
            raise TimeoutError(
                f"timed out waiting for {context}; last status={status!r}"
            )
        sleep_fn(poll_seconds)


def _require_files_server_url(uri: object, *, context: str) -> str:
    value = str(uri or "")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != FILES_SERVER_HOST
        or parsed.port != FILES_SERVER_PORT
        or parsed.username is not None
        or parsed.password is not None
        or (parsed.path and not parsed.path.startswith("/"))
    ):
        raise RuntimeError(
            f"{context} must use the trusted files server {FILES_SERVER_URI}: {value!r}"
        )
    return value


def _parameters(task: object, *, context: str) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise RuntimeError(f"{context} cannot enumerate parameters")
    value = getter()
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} returned invalid parameters")
    return dict(value)


def _parameter_matches(actual: object, expected: object) -> bool:
    if type(expected) is bool:
        if type(actual) is bool:
            return actual is expected
        return str(actual).casefold() == str(expected).casefold()
    if type(expected) is int:
        return str(actual) == str(expected)
    return actual == expected


def _is_empty_parameter(value: object) -> bool:
    return value is None or str(value).strip().casefold() in {"", "none", "null"}


def _nested_value(value: object, *names: str) -> object:
    current = value
    for name in names:
        if isinstance(current, Mapping):
            current = current.get(name)
        else:
            current = getattr(current, name, None)
    return current


def _require_bootstrap_entry_point(task: object, *, context: str) -> str:
    entry_point = str(_nested_value(task, "data", "script", "entry_point") or "")
    normalized = entry_point.replace("\\", "/")
    if not (
        normalized == Path(BOOTSTRAP_ENTRY_POINT).name
        or normalized == BOOTSTRAP_ENTRY_POINT
        or normalized.endswith(f"/{BOOTSTRAP_ENTRY_POINT}")
    ):
        raise RuntimeError(
            f"{context} does not execute the sealed RTX5090 bootstrap: {entry_point!r}"
        )
    return entry_point


def _artifact(task: object, name: str, *, context: str) -> object:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or name not in artifacts:
        raise RuntimeError(f"{context} has no {name!r} artifact")
    return artifacts[name]


def _artifact_value(artifact: object, *, context: str) -> object:
    getter = getattr(artifact, "get", None)
    if not callable(getter):
        raise RuntimeError(f"{context} cannot be downloaded")
    try:
        value = getter(force_download=True)
    except TypeError:
        value = getter()
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, (str, Path)):
        raise RuntimeError(f"{context} returned an unsupported value")
    path = Path(value)
    if path.is_symlink():
        raise RuntimeError(f"{context} must not be a symlink: {path}")
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"{context} is not a non-empty regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{context} is not valid JSON: {path}") from error
    return payload


def _json_artifact(task: object, name: str, *, context: str) -> dict[str, object]:
    value = _artifact_value(_artifact(task, name, context=context), context=context)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} is not a JSON object")
    return dict(value)


def _require_run_contract(
    task: object,
    *,
    task_id: str,
    stage: str,
    dataset_id: str,
    context: str,
) -> dict[str, object]:
    contract = _json_artifact(task, "run_contract", context=f"{context} run_contract")
    expected = {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "runtime_profile": "rtx5090",
        "gpus": EXPECTED_GPUS,
        "global_batch_size": EXPECTED_GPUS * EXPECTED_TRAIN_BATCH_SIZE_PER_GPU,
        "max_epochs": EXPECTED_MAX_EPOCHS,
        "amp": False,
        "checkpoint_policy": "final_epoch",
        "evaluation_condition_count": EXPECTED_CONDITION_COUNT,
        "stage": stage,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(
                f"{context} run_contract {key} mismatch: "
                f"expected {value!r}, got {contract.get(key)!r}"
            )
    return contract


def _require_model_mapping(task: object, *, context: str) -> Sequence[object]:
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise RuntimeError(f"{context} cannot enumerate models")
    models = getter()
    if not isinstance(models, Mapping):
        raise RuntimeError(f"{context} returned an invalid model mapping")
    outputs = models.get("output")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise RuntimeError(f"{context} has no output model sequence")
    return outputs


def _require_unique_output_model(
    task: object,
    *,
    model_name: str,
    expected_task_id: str,
    expected_model_id: str | None,
    context: str,
) -> object:
    outputs = _require_model_mapping(task, context=context)
    candidates = [
        model for model in outputs if getattr(model, "name", None) == model_name
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"{context} must expose exactly one {model_name!r} OutputModel; "
            f"found {len(candidates)}"
        )
    model = candidates[0]
    owner = str(getattr(model, "task", "") or "")
    if owner != expected_task_id:
        raise RuntimeError(
            f"{model_name} OutputModel task mismatch: "
            f"expected {expected_task_id!r}, got {owner!r}"
        )
    model_id = str(getattr(model, "id", "") or "")
    if CLEARML_ID_PATTERN.fullmatch(model_id) is None:
        raise RuntimeError(f"{model_name} OutputModel has invalid ID: {model_id!r}")
    if expected_model_id is not None and model_id != expected_model_id:
        raise RuntimeError(
            f"{model_name} OutputModel ID mismatch: "
            f"expected {expected_model_id!r}, got {model_id!r}"
        )
    model_url = _require_files_server_url(
        getattr(model, "url", ""),
        context=f"{model_name} OutputModel URL",
    )
    expected_filename = (
        "teacher_epoch_50.pth"
        if model_name == CLEAN_TEACHER_MODEL_NAME
        else "epoch_50.pth"
    )
    if Path(unquote(urlsplit(model_url).path)).name != expected_filename:
        raise RuntimeError(
            f"{model_name} OutputModel is not the final epoch-50 checkpoint"
        )
    return model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_and_hash_model(
    model: object,
    *,
    expected_sha256: str | None,
    context: str,
) -> tuple[Path, str]:
    getter = getattr(model, "get_local_copy", None)
    if not callable(getter):
        raise RuntimeError(f"{context} cannot be downloaded")
    value = getter(
        extract_archive=False,
        raise_on_error=True,
        force_download=True,
    )
    if not value:
        raise RuntimeError(f"{context} returned no local checkpoint")
    path = Path(value)
    if path.is_symlink():
        raise RuntimeError(f"{context} must not be a symlink: {path}")
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"{context} is not a non-empty regular file: {path}")
    observed = _sha256(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(
            f"{context} SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )
    return path, observed


def _require_provenance_parameters(
    parameters: Mapping[str, object],
    *,
    context: str,
) -> str:
    for key in PROVENANCE_PARAMETER_KEYS:
        if _is_empty_parameter(parameters.get(key)):
            raise RuntimeError(f"{context} has no sealed provenance parameter {key}")
    for key in ("Args/source_dataset_id", "Args/training_dataset_id"):
        if CLEARML_ID_PATTERN.fullmatch(str(parameters[key])) is None:
            raise RuntimeError(f"{context} has invalid {key}")
    archive_name = str(parameters["Args/source_archive_name"])
    if Path(archive_name).name != archive_name or Path(archive_name).is_absolute():
        raise RuntimeError(f"{context} has invalid source archive name")
    for key in ("Args/source_archive_bytes", "Args/native_bundle_bytes"):
        try:
            size = int(str(parameters[key]))
        except ValueError as error:
            raise RuntimeError(f"{context} has invalid {key}") from error
        if size <= 0:
            raise RuntimeError(f"{context} has non-positive {key}")
    for key in (
        "Args/source_archive_sha256",
        "Args/native_bundle_sha256",
        "Args/build_manifest_sha256",
    ):
        if SHA256_PATTERN.fullmatch(str(parameters[key])) is None:
            raise RuntimeError(f"{context} has invalid {key}")
    return str(parameters["Args/training_dataset_id"])


def _require_failed_teacher_contract(
    task: object,
    *,
    teacher_pin: ModelPin,
    dataset_id: str,
) -> None:
    contract = _json_artifact(
        task,
        "run_contract",
        context="failed teacher run_contract",
    )
    expected = {
        "task_id": teacher_pin.task_id,
        "dataset_id": dataset_id,
        "runtime_profile": "rtx5090",
        "gpus": EXPECTED_GPUS,
        "max_epochs": EXPECTED_MAX_EPOCHS,
        "checkpoint_policy": "final_epoch",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(
                f"failed teacher run_contract {key} mismatch: "
                f"expected {value!r}, got {contract.get(key)!r}"
            )
    if contract.get("stage") not in {"all", "teacher"}:
        raise RuntimeError("failed teacher run_contract did not train a teacher")


def _require_teacher_model(
    task_class: object,
    *,
    teacher_pin: ModelPin,
    allow_failed_teacher_task: bool,
    dataset_id: str,
) -> object:
    task = task_class.get_task(task_id=teacher_pin.task_id)
    status = _normalized_task_status(task)
    if status == "failed" and allow_failed_teacher_task:
        _require_failed_teacher_contract(
            task,
            teacher_pin=teacher_pin,
            dataset_id=dataset_id,
        )
    elif status != "completed":
        raise RuntimeError(f"teacher task is not reusable: {status!r}")
    return _require_unique_output_model(
        task,
        model_name=CLEAN_TEACHER_MODEL_NAME,
        expected_task_id=teacher_pin.task_id,
        expected_model_id=teacher_pin.model_id,
        context="teacher task",
    )


def _require_checkpoint_handoff_contract(
    task: object,
    *,
    teacher_pin: ModelPin,
    allow_failed_teacher_task: bool,
) -> None:
    contract = _json_artifact(
        task,
        "checkpoint_handoff_contract",
        context="student checkpoint handoff contract",
    )
    teacher = contract.get("teacher")
    if not isinstance(teacher, Mapping):
        raise RuntimeError("student checkpoint handoff contract has no teacher pin")
    expected = {
        "task_id": teacher_pin.task_id,
        "model_id": teacher_pin.model_id,
        "name": CLEAN_TEACHER_MODEL_NAME,
        "sha256": teacher_pin.checkpoint_sha256,
        "expected_sha256": teacher_pin.checkpoint_sha256,
    }
    for key, value in expected.items():
        if teacher.get(key) != value:
            raise RuntimeError(
                f"student checkpoint handoff teacher {key} mismatch: "
                f"expected {value!r}, got {teacher.get(key)!r}"
            )
    _require_files_server_url(
        teacher.get("url"),
        context="student checkpoint handoff teacher URL",
    )
    if contract.get("failed_task_salvage") is not allow_failed_teacher_task:
        raise RuntimeError("student checkpoint handoff salvage policy mismatch")


def _require_formal_student_bootstrap(
    task: object,
    *,
    student_task_id: str,
    teacher_pin: ModelPin,
    allow_failed_teacher_task: bool,
) -> tuple[dict[str, object], str]:
    if _normalized_task_status(task) != "completed":
        raise RuntimeError("formal student bootstrap task is not completed")
    if _task_id(task, context="formal student task") != student_task_id:
        raise RuntimeError("formal student task ID mismatch")
    _require_bootstrap_entry_point(task, context="formal student task")
    parameters = _parameters(task, context="formal student task")
    dataset_id = _require_provenance_parameters(
        parameters,
        context="formal student task",
    )
    expected = {
        "Args/gpus": EXPECTED_GPUS,
        "Args/stage": "student",
        "Args/max_epochs": EXPECTED_MAX_EPOCHS,
        "Args/amp": False,
        "Args/teacher_task_id": teacher_pin.task_id,
        "Args/teacher_model_id": teacher_pin.model_id,
        "Args/teacher_checkpoint_sha256": teacher_pin.checkpoint_sha256,
        "Args/allow_failed_teacher_task": allow_failed_teacher_task,
    }
    for key, value in expected.items():
        if not _parameter_matches(parameters.get(key), value):
            raise RuntimeError(
                f"formal student task {key} mismatch: "
                f"expected {value!r}, got {parameters.get(key)!r}"
            )
    for key in FORBIDDEN_LOCAL_HANDOFF_PARAMETERS:
        if not _is_empty_parameter(parameters.get(key)):
            raise RuntimeError(f"formal student task contains forbidden {key}")
    for key in (
        "Args/student_task_id",
        "Args/student_model_id",
        "Args/student_checkpoint_sha256",
    ):
        if not _is_empty_parameter(parameters.get(key)):
            raise RuntimeError(f"formal student task contains stale {key}")
    _require_run_contract(
        task,
        task_id=student_task_id,
        stage="student",
        dataset_id=dataset_id,
        context="formal student task",
    )
    _require_checkpoint_handoff_contract(
        task,
        teacher_pin=teacher_pin,
        allow_failed_teacher_task=allow_failed_teacher_task,
    )
    return parameters, dataset_id


def _set_parameter(task: object, name: str, value: object) -> None:
    setter = getattr(task, "set_parameter", None)
    if not callable(setter):
        raise RuntimeError("cloned validation task cannot set parameters")
    setter(name=name, value=value)


def _strip_cuda_visible_devices(docker_arguments: str) -> str:
    """Drop host GPU pin envs so GPU4-5090 workers keep remapped device 0..N-1."""
    cleaned = _CUDA_VISIBLE_DEVICES_ARG.sub("", docker_arguments)
    return " ".join(cleaned.split())


def _prepare_validation_worker_runtime(task: object) -> None:
    """Ensure the cloned validate task can run on GPU4-5090 without a CUDA pin."""
    setter = getattr(task, "set_base_docker", None)
    if not callable(setter):
        return
    container = _nested_value(task, "data", "container")
    if isinstance(container, Mapping):
        image = str(container.get("image") or "")
        arguments = str(container.get("arguments") or "")
        setup = str(container.get("setup_shell_script") or "")
    else:
        image = str(getattr(container, "image", "") or "") if container else ""
        arguments = (
            str(getattr(container, "arguments", "") or "") if container else ""
        )
        setup = (
            str(getattr(container, "setup_shell_script", "") or "")
            if container
            else ""
        )
    if not image:
        return
    cleaned = _strip_cuda_visible_devices(arguments)
    if cleaned == arguments:
        return
    setter(
        docker_image=image,
        docker_arguments=cleaned,
        docker_setup_bash_script=setup,
    )


def _require_empty_output_state(task: object) -> None:
    getter = getattr(task, "get_models", None)
    if callable(getter):
        models = getter()
        if isinstance(models, Mapping) and models.get("output"):
            raise RuntimeError("cloned validation draft contains stale output models")
    artifacts = getattr(task, "artifacts", None)
    if isinstance(artifacts, Mapping) and "validation_summary" in artifacts:
        raise RuntimeError("cloned validation draft contains stale validation summary")


def _require_validation_draft(
    task: object,
    *,
    source_parameters: Mapping[str, object],
    expected_parameters: Mapping[str, object],
) -> None:
    if _normalized_task_status(task) != "created":
        raise RuntimeError("validation task must remain created until all checks pass")
    _task_id(task, context="validation draft")
    _require_bootstrap_entry_point(task, context="validation draft")
    parameters = _parameters(task, context="validation draft")
    for key in PROVENANCE_PARAMETER_KEYS:
        if not _parameter_matches(parameters.get(key), source_parameters.get(key)):
            raise RuntimeError(f"validation draft changed provenance parameter {key}")
    for key, value in expected_parameters.items():
        if not _parameter_matches(parameters.get(key), value):
            raise RuntimeError(
                f"validation draft {key} mismatch: "
                f"expected {value!r}, got {parameters.get(key)!r}"
            )
    for key in FORBIDDEN_LOCAL_HANDOFF_PARAMETERS:
        if not _is_empty_parameter(parameters.get(key)):
            raise RuntimeError(f"validation draft contains forbidden {key}")
    _require_files_server_url(
        getattr(task, "output_uri", ""),
        context="validation draft output URI",
    )
    _require_empty_output_state(task)


def _enqueue_acknowledged(response: object) -> bool:
    if isinstance(response, Mapping):
        queued = response.get("queued")
        updated = response.get("updated")
    else:
        queued = getattr(response, "queued", None)
        updated = getattr(response, "updated", None)
    return queued == 1 and updated == 1


def _record_validation_task_id(
    controller_task: object | None,
    validation_task_id: str,
) -> None:
    if controller_task is None:
        return
    _set_parameter(
        controller_task,
        "Controller/validation_task_id",
        validation_task_id,
    )
    flusher = getattr(controller_task, "flush", None)
    if callable(flusher):
        flusher(wait_for_uploads=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _document_content_sha256(document: Mapping[str, object]) -> str:
    payload = dict(document)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _upload_controller_summary(
    controller_task: object | None,
    *,
    student_task_id: str,
    result: ControllerResult,
) -> None:
    if controller_task is None:
        return
    controller_task_id = _task_id(controller_task, context="paper controller task")
    summary: dict[str, object] = {
        "schema_version": 1,
        "document_type": PAPER_CONTROLLER_SUMMARY_DOCUMENT_TYPE,
        "controller_task_id": controller_task_id,
        "student_task_id": student_task_id,
        "validation_task_id": result.validation_task_id,
        "student_model": {
            "name": DISTILLED_STUDENT_MODEL_NAME,
            "model_id": result.student_model_id,
            "checkpoint_sha256": result.student_checkpoint_sha256,
        },
        "validation_summary_sha256": result.validation_summary_sha256,
        "condition_count": result.condition_count,
        "worker_queue": WORKER_QUEUE,
    }
    summary["content_sha256"] = _document_content_sha256(summary)
    controller_task.output_uri = FILES_SERVER_URI
    uploader = getattr(controller_task, "upload_artifact", None)
    if not callable(uploader):
        raise RuntimeError("paper controller task cannot upload its summary")
    with tempfile.TemporaryDirectory(
        prefix="resilient-v2x-paper-controller-"
    ) as temporary_dir:
        path = Path(temporary_dir) / "paper_controller_summary.json"
        path.write_bytes(_canonical_json_bytes(summary) + b"\n")
        if not uploader(
            "paper_controller_summary",
            artifact_object=str(path),
            wait_on_upload=True,
        ):
            raise RuntimeError("failed to upload paper_controller_summary")
    flusher = getattr(controller_task, "flush", None)
    if callable(flusher):
        flusher(wait_for_uploads=True)


def _require_sealed_document(
    value: object,
    *,
    expected_type: str,
    context: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} is not an object")
    document = dict(value)
    if document.get("schema_version") != 1:
        raise RuntimeError(f"{context} has unsupported schema_version")
    if document.get("document_type") != expected_type:
        raise RuntimeError(f"{context} has wrong document_type")
    digest = document.get("content_sha256")
    if type(digest) is not str or SHA256_PATTERN.fullmatch(digest) is None:
        raise RuntimeError(f"{context} has invalid content_sha256")
    try:
        observed = _document_content_sha256(document)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{context} is outside canonical JSON") from error
    if observed != digest:
        raise RuntimeError(f"{context} content hash mismatch")
    return document


def _require_positive_integer(value: object, *, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"{context} must be a positive integer")
    return value


def _require_condition_result(
    value: object,
    *,
    condition_name: str,
    validation_task_id: str,
    dataset_id: str,
    teacher_sha256: str,
    student_sha256: str,
    evaluation_sample_count: int,
    ground_truth_count: int,
) -> None:
    result = _require_sealed_document(
        value,
        expected_type=CONDITION_RESULT_DOCUMENT_TYPE,
        context=f"condition {condition_name}",
    )
    expected = {
        "task_id": validation_task_id,
        "dataset_id": dataset_id,
        "runtime_profile": "rtx5090",
        "artifact_name": f"validation_{condition_name}",
        "condition_id": condition_name,
        "evaluation_sample_count": evaluation_sample_count,
    }
    for key, expected_value in expected.items():
        if result.get(key) != expected_value:
            raise RuntimeError(
                f"condition {condition_name} {key} mismatch: "
                f"expected {expected_value!r}, got {result.get(key)!r}"
            )
    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise RuntimeError(f"condition {condition_name} has no checkpoints")
    checkpoint_shas = {"teacher": teacher_sha256, "student": student_sha256}
    for role, expected_sha256 in checkpoint_shas.items():
        checkpoint = checkpoints.get(role)
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(f"condition {condition_name} has no {role} checkpoint")
        if checkpoint.get("sha256") != expected_sha256:
            raise RuntimeError(
                f"condition {condition_name} {role} checkpoint SHA mismatch"
            )
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError(f"condition {condition_name} has no metrics")
    if metrics.get("resilient_v2x/sample_count") != evaluation_sample_count:
        raise RuntimeError(f"condition {condition_name} sample count mismatch")
    if metrics.get("resilient_v2x/car_ground_truth_count") != ground_truth_count:
        raise RuntimeError(f"condition {condition_name} ground-truth count mismatch")
    for key in REQUIRED_AP_METRICS:
        metric = metrics.get(key)
        if type(metric) not in {int, float} or not math.isfinite(float(metric)):
            raise RuntimeError(f"condition {condition_name} has invalid metric {key}")
        if not 0.0 <= float(metric) <= 100.0:
            raise RuntimeError(
                f"condition {condition_name} metric {key} is out of range"
            )


def _download_validation_summary(
    task: object,
    *,
    dataset_id: str,
    teacher_sha256: str,
    student_sha256: str,
) -> tuple[str, int]:
    validation_task_id = _task_id(task, context="validation task")
    artifact = _artifact(
        task,
        "validation_summary",
        context="completed validation task",
    )
    _require_files_server_url(
        getattr(artifact, "url", ""),
        context="validation_summary artifact URL",
    )
    getter = getattr(artifact, "get", None)
    if not callable(getter):
        raise RuntimeError("validation_summary artifact cannot be downloaded")
    try:
        value = getter(force_download=True)
    except TypeError:
        value = getter()
    if not isinstance(value, (str, Path)):
        raise RuntimeError("validation_summary artifact is not a downloaded file")
    path = Path(value)
    if path.is_symlink():
        raise RuntimeError("validation_summary artifact must not be a symlink")
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("validation_summary artifact is not a non-empty file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("validation_summary artifact is not valid JSON") from error
    summary = _require_sealed_document(
        value,
        expected_type=VALIDATION_SUMMARY_DOCUMENT_TYPE,
        context="validation_summary",
    )
    expected = {
        "task_id": validation_task_id,
        "dataset_id": dataset_id,
        "runtime_profile": "rtx5090",
        "condition_count": EXPECTED_CONDITION_COUNT,
    }
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            raise RuntimeError(
                f"validation_summary {key} mismatch: "
                f"expected {expected_value!r}, got {summary.get(key)!r}"
            )
    evaluation_sample_count = _require_positive_integer(
        summary.get("evaluation_sample_count"),
        context="validation_summary evaluation_sample_count",
    )
    ground_truth_count = _require_positive_integer(
        summary.get("ground_truth_count"),
        context="validation_summary ground_truth_count",
    )
    checkpoint_shas = {"teacher": teacher_sha256, "student": student_sha256}
    for role, expected_sha256 in checkpoint_shas.items():
        checkpoint = summary.get(f"{role}_checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(f"validation_summary has no {role} checkpoint")
        if checkpoint.get("sha256") != expected_sha256:
            raise RuntimeError(f"validation_summary {role} checkpoint SHA mismatch")
    conditions = summary.get("conditions")
    if not isinstance(conditions, Mapping):
        raise RuntimeError("validation_summary conditions is not an object")
    if set(conditions) != set(EXPECTED_CONDITIONS):
        raise RuntimeError(
            "validation_summary does not contain the exact 12-condition matrix"
        )
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("validation task has no artifact mapping")
    for condition_name in EXPECTED_CONDITIONS:
        artifact_name = f"validation_{condition_name}"
        if artifact_name not in artifacts:
            raise RuntimeError(f"validation task has no {artifact_name!r} artifact")
        _require_files_server_url(
            getattr(artifacts[artifact_name], "url", ""),
            context=f"{artifact_name} artifact URL",
        )
        _require_condition_result(
            conditions[condition_name],
            condition_name=condition_name,
            validation_task_id=validation_task_id,
            dataset_id=dataset_id,
            teacher_sha256=teacher_sha256,
            student_sha256=student_sha256,
            evaluation_sample_count=evaluation_sample_count,
            ground_truth_count=ground_truth_count,
        )
    return hashlib.sha256(raw).hexdigest(), len(conditions)


def run_paper_controller(
    task_class: object,
    *,
    student_task_id: str,
    teacher_task_id: str,
    teacher_model_id: str,
    teacher_checkpoint_sha256: str,
    expected_student_model_id: str | None = None,
    expected_student_checkpoint_sha256: str | None = None,
    allow_failed_teacher_task: bool = False,
    project: str = "ResilientV2X/Training",
    name: str = "ResilientV2X RTX5090 formal 12-condition validation",
    poll_seconds: float = 30.0,
    timeout_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> ControllerResult:
    """Wait, verify, clone, enqueue, wait, and verify the paper validation."""

    for value, context in (
        (student_task_id, "student task ID"),
        (teacher_task_id, "teacher task ID"),
        (teacher_model_id, "teacher model ID"),
    ):
        if CLEARML_ID_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{context} must be a lowercase 32-hex ID")
    if SHA256_PATTERN.fullmatch(teacher_checkpoint_sha256) is None:
        raise ValueError("teacher checkpoint SHA must be lowercase hexadecimal")
    if (
        expected_student_model_id is not None
        and CLEARML_ID_PATTERN.fullmatch(expected_student_model_id) is None
    ):
        raise ValueError("expected student model ID must be lowercase 32-hex")
    if (
        expected_student_checkpoint_sha256 is not None
        and SHA256_PATTERN.fullmatch(expected_student_checkpoint_sha256) is None
    ):
        raise ValueError("expected student checkpoint SHA is invalid")
    if not project.strip() or not name.strip():
        raise ValueError("project and task name must be non-empty")

    teacher_pin = ModelPin(
        teacher_task_id,
        teacher_model_id,
        teacher_checkpoint_sha256,
    )
    student_task = task_class.get_task(task_id=student_task_id)
    _wait_for_completed_task(
        student_task,
        context="formal student task",
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
    )
    source_parameters, dataset_id = _require_formal_student_bootstrap(
        student_task,
        student_task_id=student_task_id,
        teacher_pin=teacher_pin,
        allow_failed_teacher_task=allow_failed_teacher_task,
    )
    _require_teacher_model(
        task_class,
        teacher_pin=teacher_pin,
        allow_failed_teacher_task=allow_failed_teacher_task,
        dataset_id=dataset_id,
    )
    student_model = _require_unique_output_model(
        student_task,
        model_name=DISTILLED_STUDENT_MODEL_NAME,
        expected_task_id=student_task_id,
        expected_model_id=expected_student_model_id,
        context="formal student task",
    )
    _, observed_student_sha256 = _download_and_hash_model(
        student_model,
        expected_sha256=expected_student_checkpoint_sha256,
        context="formal student checkpoint",
    )
    student_model_id = str(getattr(student_model, "id", "") or "")

    draft = task_class.clone(
        source_task=student_task,
        name=name,
        comment=(
            "Validation successor created only after the formal student task "
            "completed and its unique epoch-50 OutputModel SHA was verified."
        ),
        parent=student_task_id,
        project=_project_id(task_class, project),
    )
    validation_task_id = _task_id(draft, context="cloned validation task")
    if validation_task_id == student_task_id:
        raise RuntimeError("ClearML clone reused the formal student task ID")
    draft.output_uri = FILES_SERVER_URI
    expected_parameters: dict[str, object] = {
        "Args/gpus": EXPECTED_GPUS,
        "Args/stage": "validate",
        "Args/max_epochs": EXPECTED_MAX_EPOCHS,
        "Args/amp": False,
        "Args/teacher_task_id": teacher_pin.task_id,
        "Args/teacher_model_id": teacher_pin.model_id,
        "Args/teacher_checkpoint_sha256": teacher_pin.checkpoint_sha256,
        "Args/allow_failed_teacher_task": allow_failed_teacher_task,
        "Args/student_task_id": student_task_id,
        "Args/student_model_id": student_model_id,
        "Args/student_checkpoint_sha256": observed_student_sha256,
    }
    for key in sorted(expected_parameters):
        _set_parameter(draft, key, expected_parameters[key])
    _prepare_validation_worker_runtime(draft)
    _reload(draft)
    _require_validation_draft(
        draft,
        source_parameters=source_parameters,
        expected_parameters=expected_parameters,
    )
    enqueue_response = task_class.enqueue(
        task=draft,
        queue_name=WORKER_QUEUE,
        force=False,
    )
    if not _enqueue_acknowledged(enqueue_response):
        raise RuntimeError(
            f"ClearML did not acknowledge exactly one enqueue: {enqueue_response!r}"
        )
    _wait_for_completed_task(
        draft,
        context="formal 12-condition validation task",
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
    )
    summary_sha256, condition_count = _download_validation_summary(
        draft,
        dataset_id=dataset_id,
        teacher_sha256=teacher_pin.checkpoint_sha256,
        student_sha256=observed_student_sha256,
    )
    result = ControllerResult(
        validation_task_id=validation_task_id,
        student_model_id=student_model_id,
        student_checkpoint_sha256=observed_student_sha256,
        validation_summary_sha256=summary_sha256,
        condition_count=condition_count,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    from clearml import Task

    if argv is None:
        _connect_controller_task(Task)
    args = _parser().parse_args(argv)

    result = run_paper_controller(
        Task,
        student_task_id=args.student_task_id,
        teacher_task_id=args.teacher_task_id,
        teacher_model_id=args.teacher_model_id,
        teacher_checkpoint_sha256=args.teacher_checkpoint_sha256,
        expected_student_model_id=args.student_model_id,
        expected_student_checkpoint_sha256=args.student_checkpoint_sha256,
        allow_failed_teacher_task=args.allow_failed_teacher_task,
        project=args.project,
        name=args.name,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    controller_task = Task.current_task()
    _record_validation_task_id(
        controller_task,
        result.validation_task_id,
    )
    _upload_controller_summary(
        controller_task,
        student_task_id=args.student_task_id,
        result=result,
    )

    print(
        json.dumps(
            {
                "event": "resilient_v2x_paper_validation_complete",
                **result._asdict(),
                "worker_queue": WORKER_QUEUE,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ControllerResult",
    "FILES_SERVER_URI",
    "WORKER_QUEUE",
    "main",
    "run_paper_controller",
)
