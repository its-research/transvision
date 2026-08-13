#!/usr/bin/env python3
"""Gate a completed clean teacher against the controlled FFNet reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

try:
    from allegroai import Task
except ImportError:
    try:
        from clearml import Task
    except ImportError:  # pragma: no cover - exercised only without the SDK
        Task = None  # type: ignore[assignment]


DEFAULT_PROJECT = "ResilientV2X/Training"
FILES_SERVER_URI = "http://10.100.34.118:8081"
EXPECTED_FILES_SERVER_HOST = "10.100.34.118"
EXPECTED_FILES_SERVER_PORT = 8081
RUN_CONTRACT_ARTIFACT = "run_contract"
TEACHER_CHECKPOINT_ARTIFACT = "teacher_checkpoint_contract"
QUALITY_GATE_ARTIFACT = "teacher_quality_gate"
QUALITY_GATE_DOCUMENT_TYPE = "resilient_v2x_teacher_quality_gate"
CLEAN_TEACHER_MODEL_NAME = "ResilientV2X clean teacher"
CLEAN_TEACHER_SELECTION_PROTOCOL = "DAIR-CLEAN-PAIR1789-v1"
CLEAN_TEACHER_SELECTION_METRIC = "resilient_v2x/car_bev_ap_r40_0.70"
CLEAN_TEACHER_SELECTION_RULE = "greater"
CLEAN_TEACHER_CHECKPOINT_POLICY = "clean_validation_best_for_teacher_handoff"
DEFAULT_TRAINING_SEED = 20250218
TRAINING_OVERLAY_PROTOCOL_SEED = 20250218
EXPECTED_VALIDATION_EPOCHS = (10, 20, 30, 40, 50)
EXPECTED_CLEAN_SAMPLE_COUNT = 1_789
EXPECTED_CLEAN_GROUND_TRUTH_COUNT = 15_337
EXPECTED_CLEAN_UNSUPPORTED_SAMPLE_COUNT = 0
FFNET_STYLE_REFERENCE_TASK_ID = "ddbeeec499fb4b55bcd13bc14823b9df"
FFNET_STYLE_REFERENCE_BEV_AP70 = 60.6257
FFNET_STYLE_REFERENCE_3D_AP70 = 34.9827
FFNET_STYLE_BEV_NONINFERIORITY_MARGIN = 1.0
DEFAULT_MIN_BEV_AP70 = 59.6257
DEFAULT_MIN_3D_AP70 = 30.0
FFNET_STYLE_POINT_CLOUD_RANGE = [0.0, -40.0, -3.0, 80.0, 40.0, 1.0]
EXPECTED_SOURCE_DATASET_ID = "4f7fac0078a4419a907fec6ff9e306c8"
EXPECTED_SOURCE_ARCHIVE_NAME = "resilient-v2x-source-5c984ad49b52.tar.zst"
EXPECTED_SOURCE_ARCHIVE_BYTES = 1_222_481
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "655f33d9b684c26ff4ff577d5c5f3175d4fc4eac40426eb55820e7a8aa8bc40d"
)
EXPECTED_TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
EXPECTED_NATIVE_BUNDLE_BYTES = 753_382_966
EXPECTED_NATIVE_BUNDLE_SHA256 = (
    "19b8e7f5edc8216d4b43cb17854dccafe4fc9fe46995a88803e6342eeaa22b21"
)
EXPECTED_BUILD_MANIFEST_SHA256 = (
    "21c6ab7a6e9a2823ba111289a42e7f882c5f4ba02c73175106ff251fe5864a43"
)
EXPECTED_TEACHER_SCRIPT_DIFF_SHA256 = (
    "4dfe2e9d2ee3076df1679818211f40b7c2ebcbc73efb4cdae2f230c971485b67"
)
LOG_ROUNDING_UNIT = 0.0001
CONSOLE_REPORT_LIMIT = 10_000
CONSOLE_LINE_LIMIT = 16_384
EXPECTED_MANIFEST_CONTENT_SHA256 = (
    "715ac6f7a14225e20327eed0650c55abdc0cb98431830164e84545238099645d"
)
OFFICIAL_SPLIT_SHA256 = (
    "d048aeeca548fb194c548b798e6fc08488c4dd350ad223a154c028ed0a58de6c"
)
EXPECTED_EVALUATION_INDEX_CONTENT_SHA256 = (
    "77bd4585dbb02901f862b8da6aa208a504674b824a3d55cf15005aacbeeeaaff"
)
EXPECTED_EVALUATION_SAMPLE_IDS_SHA256 = (
    "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"
)
EXPECTED_EVALUATION_SAMPLE_COUNT = 1_337
EXPECTED_EVALUATION_CONDITION_COUNT = 12
AP_METRIC_KEYS = (
    "resilient_v2x/car_bev_ap_r40_0.50",
    "resilient_v2x/car_bev_ap_r40_0.70",
    "resilient_v2x/car_3d_ap_r40_0.50",
    "resilient_v2x/car_3d_ap_r40_0.70",
)
AP_FIELD_BY_KEY = {
    "resilient_v2x/car_bev_ap_r40_0.50": "car_bev_ap_r40_0.50",
    "resilient_v2x/car_bev_ap_r40_0.70": "car_bev_ap_r40_0.70",
    "resilient_v2x/car_3d_ap_r40_0.50": "car_3d_ap_r40_0.50",
    "resilient_v2x/car_3d_ap_r40_0.70": "car_3d_ap_r40_0.70",
}
BEV_AP70_FIELD = AP_FIELD_BY_KEY[CLEAN_TEACHER_SELECTION_METRIC]
THREE_D_AP70_FIELD = AP_FIELD_BY_KEY["resilient_v2x/car_3d_ap_r40_0.70"]
WAITABLE_STATUSES = frozenset({"created", "queued", "in_progress"})
TERMINAL_FAILURE_STATUSES = frozenset(
    {
        "failed",
        "stopped",
        "closed",
        "published",
        "publishing",
        "rejected",
        "unknown",
    }
)
CLEARML_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ANSI_ESCAPE_PATTERN = re.compile(
    r"(?:\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[@-_][0-?]*[ -/]*[@-~])"
)
VALIDATION_PREFIX_PATTERN = re.compile(
    r"Epoch\(val\)\s+\[\s*(?P<epoch>\d+)\s*\]"
    r"\[\s*(?P<iteration>\d+)\s*/\s*(?P<total>\d+)\s*\]"
)
NUMBER_PATTERN = (
    r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"|inf(?:inity)?|nan)"
)
LEGACY_SOURCE_C_RUN_CONTRACT_KEYS = {
    "task_id",
    "dataset_id",
    "dataset_local_copy",
    "gpus",
    "training_world_size",
    "global_batch_size",
    "train_batch_size_per_gpu",
    "eval_batch_size_per_gpu",
    "vehicle_global_batch_size",
    "vehicle_train_batch_size_per_gpu",
    "vehicle_eval_batch_size_per_gpu",
    "learning_rate",
    "auto_scale_lr",
    "expected_optimizer_steps",
    "max_epochs",
    "val_interval",
    "amp",
    "runtime_profile",
    "python_safe_path",
    "checkpoint_policy",
    "manifest_content_sha256",
    "split_sha256",
    "evaluation_index_content_sha256",
    "evaluation_sample_ids_sha256",
    "evaluation_sample_count",
    "evaluation_condition_count",
    "output_uri_scheme",
    "stage",
}
SEEDED_RUN_CONTRACT_KEYS = LEGACY_SOURCE_C_RUN_CONTRACT_KEYS | {
    "training_seed",
    "training_overlay_protocol_seed",
    "seed",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-task-id", required=True)
    parser.add_argument("--min-bev-ap70", type=float, default=DEFAULT_MIN_BEV_AP70)
    parser.add_argument("--min-3d-ap70", type=float, default=DEFAULT_MIN_3D_AP70)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=168.0)
    return parser


def _clearml_id(value: object, context: str) -> str:
    result = str(value or "")
    if CLEARML_ID_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{context} must be a lowercase 32-hex ClearML ID")
    return result


def _sha256(value: object, context: str) -> str:
    result = str(value or "")
    if SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return result


def _validate_json_domain(value: object, context: str = "document") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{context} mapping keys must be strings")
            _validate_json_domain(item, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_domain(item, f"{context}[{index}]")
        return
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float and math.isfinite(value):
        return
    raise ValueError(f"{context} is outside the canonical JSON domain")


def _canonical_json(value: object) -> str:
    _validate_json_domain(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _mapping_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seal(payload: Mapping[str, object]) -> dict[str, object]:
    reserved = {"schema_version", "document_type", "content_sha256"}
    if reserved.intersection(payload):
        raise ValueError("quality gate payload contains a reserved document field")
    document: dict[str, object] = {
        "schema_version": 1,
        "document_type": QUALITY_GATE_DOCUMENT_TYPE,
        **dict(payload),
    }
    _validate_json_domain(document)
    document["content_sha256"] = _mapping_sha256(document)
    return document


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise RuntimeError(
            f"{context} schema mismatch: "
            f"missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )


def _require_literal(
    value: Mapping[str, object],
    key: str,
    expected: object,
    *,
    context: str,
) -> None:
    actual = value.get(key)
    if type(actual) is not type(expected) or actual != expected:
        raise RuntimeError(
            f"{context} {key} mismatch: expected {expected!r}, got {actual!r}"
        )


def _parameter_matches(actual: object, expected: object) -> bool:
    if type(actual) is bool or type(expected) is bool:
        return str(actual).casefold() == str(expected).casefold()
    if type(actual) is int or type(expected) is int:
        return str(actual) == str(expected)
    return actual == expected


def _status(task: object) -> str:
    status = getattr(task, "status", None)
    if callable(status):
        status = status()
    if status is None:
        getter = getattr(task, "get_status", None)
        status = getter() if callable(getter) else None
    value = getattr(status, "value", status)
    return str(value or "").rsplit(".", 1)[-1].lower()


def _reload(task: object) -> None:
    reloader = getattr(task, "reload", None)
    if callable(reloader):
        reloader()


def _wait_for_completed(
    task: object,
    *,
    deadline: float,
    poll_seconds: float,
    monotonic: Callable[[], float],
    sleeper: Callable[[float], None],
) -> None:
    while True:
        _reload(task)
        status = _status(task)
        if status == "completed":
            return
        if status in TERMINAL_FAILURE_STATUSES:
            raise RuntimeError(f"teacher task ended as {status!r}")
        if status not in WAITABLE_STATUSES:
            raise RuntimeError(f"teacher task has unexpected status {status!r}")
        if monotonic() >= deadline:
            raise TimeoutError("timed out waiting for the teacher task")
        sleeper(poll_seconds)


def _parameters(task: object) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise RuntimeError("teacher task cannot enumerate parameters")
    try:
        value = getter(backwards_compatibility=False, cast=False)
    except TypeError:
        try:
            value = getter(cast=False)
        except TypeError:
            value = getter()
    if not isinstance(value, Mapping):
        raise RuntimeError("teacher task returned invalid parameters")
    return {str(key): item for key, item in value.items()}


def _artifact_mapping(task: object, name: str) -> dict[str, object]:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or name not in artifacts:
        raise RuntimeError(f"teacher task lacks artifact {name!r}")
    getter = getattr(artifacts[name], "get", None)
    if not callable(getter):
        raise RuntimeError(f"teacher artifact {name!r} cannot be read")
    value = getter()
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, (str, Path)):
        raise RuntimeError(f"teacher artifact {name!r} is not a JSON object")
    try:
        path = Path(value).resolve(strict=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"teacher artifact {name!r} cannot be read as JSON"
        ) from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"teacher artifact {name!r} is not a JSON object")
    return dict(payload)


def _require_files_server_url(value: object, *, context: str) -> str:
    result = str(value or "")
    parsed = urlsplit(result)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != EXPECTED_FILES_SERVER_HOST
        or parsed.port != EXPECTED_FILES_SERVER_PORT
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise RuntimeError(
            f"{context} must use {EXPECTED_FILES_SERVER_HOST}:"
            f"{EXPECTED_FILES_SERVER_PORT}: {result!r}"
        )
    return result


def _positive_integer(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"{context} must be a positive integer")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_script_diff_sha256(task: object) -> str:
    data = getattr(task, "data", None)
    script_object = getattr(data, "script", None)
    to_dict = getattr(script_object, "to_dict", None)
    script = to_dict() if callable(to_dict) else None
    if not isinstance(script, Mapping):
        getter = getattr(task, "get_script", None)
        script = getter() if callable(getter) else None
    if not isinstance(script, Mapping) or type(script.get("diff")) is not str:
        raise RuntimeError("teacher task cannot expose its script diff")
    return hashlib.sha256(script["diff"].encode("utf-8")).hexdigest()


def _validate_run_contract(
    task: object,
    *,
    teacher_task_id: str,
) -> tuple[dict[str, object], str, dict[str, object]]:
    parameters = _parameters(task)
    expected_parameters = {
        "Args/source_dataset_id": EXPECTED_SOURCE_DATASET_ID,
        "Args/source_archive_name": EXPECTED_SOURCE_ARCHIVE_NAME,
        "Args/source_archive_bytes": EXPECTED_SOURCE_ARCHIVE_BYTES,
        "Args/source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "Args/training_dataset_id": EXPECTED_TRAINING_DATASET_ID,
        "Args/native_bundle_bytes": EXPECTED_NATIVE_BUNDLE_BYTES,
        "Args/native_bundle_sha256": EXPECTED_NATIVE_BUNDLE_SHA256,
        "Args/build_manifest_sha256": EXPECTED_BUILD_MANIFEST_SHA256,
        "Args/stage": "teacher",
        "Args/max_epochs": 50,
        "Args/gpus": 4,
        "Args/amp": False,
    }
    for key, expected in expected_parameters.items():
        if not _parameter_matches(parameters.get(key), expected):
            raise RuntimeError(f"teacher task parameter {key} mismatch")
    dataset_id = _clearml_id(
        parameters.get("Args/training_dataset_id"), "teacher dataset"
    )
    if _task_script_diff_sha256(task) != EXPECTED_TEACHER_SCRIPT_DIFF_SHA256:
        raise RuntimeError("teacher task script diff SHA-256 mismatch")

    contract = _artifact_mapping(task, RUN_CONTRACT_ARTIFACT)
    if set(contract) == LEGACY_SOURCE_C_RUN_CONTRACT_KEYS:
        schema_kind = "legacy_source_c"
        seed_evidence = "legacy_fixed_by_sealed_source"
        seed_binding: dict[str, object] = {
            "teacher_script_diff_sha256": EXPECTED_TEACHER_SCRIPT_DIFF_SHA256,
            "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        }
    elif set(contract) == SEEDED_RUN_CONTRACT_KEYS:
        schema_kind = "seeded"
        seed_evidence = "run_contract_explicit"
        seed_binding = {
            "training_seed": DEFAULT_TRAINING_SEED,
            "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
        }
    else:
        raise RuntimeError("teacher run contract schema mismatch")
    literals = {
        "task_id": teacher_task_id,
        "dataset_id": dataset_id,
        "gpus": 4,
        "training_world_size": 4,
        "global_batch_size": 8,
        "train_batch_size_per_gpu": 2,
        "eval_batch_size_per_gpu": 4,
        "vehicle_global_batch_size": 8,
        "vehicle_train_batch_size_per_gpu": 2,
        "vehicle_eval_batch_size_per_gpu": 4,
        "learning_rate": 0.0001,
        "auto_scale_lr": False,
        "expected_optimizer_steps": None,
        "max_epochs": 50,
        "val_interval": 10,
        "amp": False,
        "runtime_profile": "rtx5090",
        "python_safe_path": "1",
        "checkpoint_policy": CLEAN_TEACHER_CHECKPOINT_POLICY,
        "manifest_content_sha256": EXPECTED_MANIFEST_CONTENT_SHA256,
        "split_sha256": OFFICIAL_SPLIT_SHA256,
        "evaluation_index_content_sha256": (EXPECTED_EVALUATION_INDEX_CONTENT_SHA256),
        "evaluation_sample_ids_sha256": EXPECTED_EVALUATION_SAMPLE_IDS_SHA256,
        "evaluation_sample_count": EXPECTED_EVALUATION_SAMPLE_COUNT,
        "evaluation_condition_count": EXPECTED_EVALUATION_CONDITION_COUNT,
        "stage": "teacher",
    }
    if schema_kind == "seeded":
        literals.update(
            {
                "training_seed": DEFAULT_TRAINING_SEED,
                "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
                "seed": DEFAULT_TRAINING_SEED,
            }
        )
    for key, expected in literals.items():
        _require_literal(contract, key, expected, context="teacher run contract")
    local_copy = contract.get("dataset_local_copy")
    if (
        type(local_copy) is not str
        or not local_copy.startswith("/")
        or ".." in PurePosixPath(local_copy).parts
    ):
        raise RuntimeError("teacher run contract dataset_local_copy is invalid")
    output_uri_scheme = contract.get("output_uri_scheme")
    if output_uri_scheme not in {"http", "https", "clearml-default"}:
        raise RuntimeError("teacher run contract output_uri_scheme is invalid")
    return (
        contract,
        dataset_id,
        {
            "run_contract_schema": schema_kind,
            "training_seed_evidence": seed_evidence,
            "training_seed_binding": seed_binding,
            "source_identity": {
                "source_dataset_id": EXPECTED_SOURCE_DATASET_ID,
                "source_archive_name": EXPECTED_SOURCE_ARCHIVE_NAME,
                "source_archive_bytes": EXPECTED_SOURCE_ARCHIVE_BYTES,
                "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
                "training_dataset_id": EXPECTED_TRAINING_DATASET_ID,
                "teacher_script_diff_sha256": EXPECTED_TEACHER_SCRIPT_DIFF_SHA256,
            },
        },
    )


def _teacher_output_model(task: object) -> object:
    getter = getattr(task, "get_models", None)
    models = getter() if callable(getter) else None
    if not isinstance(models, Mapping):
        raise RuntimeError("teacher task returned an invalid model mapping")
    outputs = models.get("output")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise RuntimeError("teacher task has no output model sequence")
    if len(outputs) != 1:
        raise RuntimeError("teacher task must expose exactly one OutputModel")
    model = outputs[0]
    if getattr(model, "name", None) != CLEAN_TEACHER_MODEL_NAME:
        raise RuntimeError("teacher OutputModel name mismatch")
    return model


def _download_and_verify_model(
    model: object,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
) -> None:
    getter = getattr(model, "get_local_copy", None)
    if not callable(getter):
        raise RuntimeError("teacher OutputModel cannot be downloaded")
    value = getter(
        extract_archive=False,
        raise_on_error=True,
        force_download=True,
    )
    if not value:
        raise RuntimeError("teacher OutputModel returned no local copy")
    path = Path(value)
    if path.is_symlink():
        raise RuntimeError("teacher checkpoint must not be a symlink")
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("teacher checkpoint does not exist") from error
    if not path.is_file():
        raise RuntimeError("teacher checkpoint is not a regular file")
    if path.stat().st_size != expected_size_bytes:
        raise RuntimeError("teacher checkpoint size mismatch")
    if _sha256_path(path) != expected_sha256:
        raise RuntimeError("teacher checkpoint SHA-256 mismatch")


def _validate_checkpoint_contract(
    task: object,
    *,
    teacher_task_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    contract = _artifact_mapping(task, TEACHER_CHECKPOINT_ARTIFACT)
    _require_exact_keys(
        contract,
        {
            "schema_version",
            "selection_protocol",
            "selection_metric",
            "selection_rule",
            "selected_epoch",
            "selected_checkpoint",
            "trained_epochs",
            "final_epoch",
            "final_checkpoint",
            "downstream_role",
        },
        context="teacher checkpoint contract",
    )
    literals = {
        "schema_version": 1,
        "selection_protocol": CLEAN_TEACHER_SELECTION_PROTOCOL,
        "selection_metric": CLEAN_TEACHER_SELECTION_METRIC,
        "selection_rule": CLEAN_TEACHER_SELECTION_RULE,
        "trained_epochs": 50,
        "final_epoch": 50,
        "downstream_role": "frozen teacher and trainable student initialization",
    }
    for key, expected in literals.items():
        _require_literal(
            contract,
            key,
            expected,
            context="teacher checkpoint contract",
        )
    selected_epoch = _positive_integer(
        contract.get("selected_epoch"), "teacher selected_epoch"
    )
    if selected_epoch not in EXPECTED_VALIDATION_EPOCHS:
        raise RuntimeError("teacher selected_epoch is not a scheduled validation")

    selected = contract.get("selected_checkpoint")
    if not isinstance(selected, Mapping):
        raise RuntimeError("teacher selected_checkpoint is not an object")
    selected = dict(selected)
    _require_exact_keys(
        selected,
        {"model_id", "name", "url", "filename", "size_bytes", "sha256"},
        context="teacher selected checkpoint",
    )
    model_id = _clearml_id(selected.get("model_id"), "selected teacher model")
    _require_literal(
        selected,
        "name",
        CLEAN_TEACHER_MODEL_NAME,
        context="teacher selected checkpoint",
    )
    filename = selected.get("filename")
    expected_filename = (
        f"best_resilient_v2x_car_bev_ap_r40_0.70_teacher_epoch_{selected_epoch}.pth"
    )
    if filename != expected_filename:
        raise RuntimeError("teacher selected checkpoint filename mismatch")
    model_url = _require_files_server_url(
        selected.get("url"), context="selected teacher model URL"
    )
    if Path(unquote(urlsplit(model_url).path)).name != filename:
        raise RuntimeError("selected teacher model URL filename mismatch")
    size_bytes = _positive_integer(
        selected.get("size_bytes"), "selected teacher checkpoint size"
    )
    checkpoint_sha256 = _sha256(selected.get("sha256"), "selected teacher checkpoint")

    final_checkpoint = contract.get("final_checkpoint")
    if not isinstance(final_checkpoint, Mapping):
        raise RuntimeError("teacher final_checkpoint is not an object")
    final_checkpoint = dict(final_checkpoint)
    _require_exact_keys(
        final_checkpoint,
        {"filename", "size_bytes", "sha256"},
        context="teacher final checkpoint",
    )
    _require_literal(
        final_checkpoint,
        "filename",
        "teacher_epoch_50.pth",
        context="teacher final checkpoint",
    )
    final_size_bytes = _positive_integer(
        final_checkpoint.get("size_bytes"), "teacher final checkpoint size"
    )
    final_sha256 = _sha256(final_checkpoint.get("sha256"), "teacher final checkpoint")

    model = _teacher_output_model(task)
    if str(getattr(model, "id", "") or "") != model_id:
        raise RuntimeError("teacher OutputModel ID mismatch")
    if str(getattr(model, "task", "") or "") != teacher_task_id:
        raise RuntimeError("teacher OutputModel ownership mismatch")
    actual_url = _require_files_server_url(
        getattr(model, "url", ""), context="teacher OutputModel URL"
    )
    if actual_url != model_url:
        raise RuntimeError("teacher OutputModel URL mismatch")
    _download_and_verify_model(
        model,
        expected_size_bytes=size_bytes,
        expected_sha256=checkpoint_sha256,
    )
    reference = {
        "task_id": teacher_task_id,
        "model_id": model_id,
        "model_name": CLEAN_TEACHER_MODEL_NAME,
        "model_url": model_url,
        "checkpoint_filename": filename,
        "checkpoint_size_bytes": size_bytes,
        "checkpoint_sha256": checkpoint_sha256,
        "selected_epoch": selected_epoch,
        "final_checkpoint_filename": "teacher_epoch_50.pth",
        "final_checkpoint_size_bytes": final_size_bytes,
        "final_checkpoint_sha256": final_sha256,
    }
    return contract, reference


def _console_chunks(task: object) -> list[str]:
    getter = getattr(task, "get_reported_console_output", None)
    if not callable(getter):
        raise RuntimeError("teacher task cannot expose console output")
    try:
        value = getter(
            number_of_reports=CONSOLE_REPORT_LIMIT,
            max_line_length=CONSOLE_LINE_LIMIT,
            order="asc",
        )
    except TypeError:
        try:
            value = getter(CONSOLE_REPORT_LIMIT, CONSOLE_LINE_LIMIT, "asc")
        except TypeError:
            value = getter(CONSOLE_REPORT_LIMIT)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError("teacher console output is not a sequence")
    chunks = list(value)
    if not chunks or any(type(chunk) is not str for chunk in chunks):
        raise RuntimeError("teacher console output has no usable text")
    return chunks


def _number_from_line(line: str, key: str) -> float:
    pattern = re.compile(
        rf"(?<!\S){re.escape(key)}\s*:\s*(?P<value>{NUMBER_PATTERN})(?=\s|$)",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(line))
    if len(matches) != 1:
        raise RuntimeError(f"validation line has {len(matches)} values for {key}")
    value = float(matches[0].group("value"))
    if not math.isfinite(value):
        raise RuntimeError(f"validation value {key} is not finite")
    return value


def _metric_from_line(line: str, key: str) -> float:
    value = _number_from_line(line, key)
    if not 0.0 <= value <= 100.0:
        raise RuntimeError(f"validation metric {key} is outside [0, 100]")
    return value


def _parse_validation_logs(chunks: Sequence[str]) -> list[dict[str, object]]:
    text = ANSI_ESCAPE_PATTERN.sub("", "\n".join(chunks))
    by_epoch: dict[int, dict[str, object]] = {}
    unexpected_epochs: set[int] = set()
    for line in text.splitlines():
        prefix = VALIDATION_PREFIX_PATTERN.search(line)
        if prefix is None or CLEAN_TEACHER_SELECTION_METRIC not in line:
            continue
        epoch = int(prefix.group("epoch"))
        iteration = int(prefix.group("iteration"))
        total = int(prefix.group("total"))
        if iteration != total or total <= 0:
            continue
        metrics = {
            AP_FIELD_BY_KEY[key]: _metric_from_line(line, key) for key in AP_METRIC_KEYS
        }
        sample_count = _number_from_line(line, "resilient_v2x/sample_count")
        if sample_count != float(EXPECTED_CLEAN_SAMPLE_COUNT):
            raise RuntimeError(f"validation epoch {epoch} clean sample_count mismatch")
        ground_truth_count = _number_from_line(
            line, "resilient_v2x/car_ground_truth_count"
        )
        if ground_truth_count != float(EXPECTED_CLEAN_GROUND_TRUTH_COUNT):
            raise RuntimeError(
                f"validation epoch {epoch} car_ground_truth_count mismatch"
            )
        unsupported_count = _number_from_line(
            line, "resilient_v2x/unsupported_sample_count"
        )
        if unsupported_count != float(EXPECTED_CLEAN_UNSUPPORTED_SAMPLE_COUNT):
            raise RuntimeError(
                f"validation epoch {epoch} unsupported_sample_count mismatch"
            )
        record: dict[str, object] = {
            "epoch": epoch,
            "validation_iterations": total,
            "sample_count": EXPECTED_CLEAN_SAMPLE_COUNT,
            "car_ground_truth_count": EXPECTED_CLEAN_GROUND_TRUTH_COUNT,
            "unsupported_sample_count": EXPECTED_CLEAN_UNSUPPORTED_SAMPLE_COUNT,
            **metrics,
        }
        if epoch not in EXPECTED_VALIDATION_EPOCHS:
            unexpected_epochs.add(epoch)
            continue
        previous = by_epoch.get(epoch)
        if previous is not None and previous != record:
            raise RuntimeError(f"validation epoch {epoch} has conflicting results")
        by_epoch[epoch] = record
    if unexpected_epochs:
        raise RuntimeError(
            "teacher emitted unexpected terminal validation epochs: "
            f"{sorted(unexpected_epochs)!r}"
        )
    missing = sorted(set(EXPECTED_VALIDATION_EPOCHS) - set(by_epoch))
    if missing or len(by_epoch) < 5:
        raise RuntimeError(
            "teacher must expose at least five scheduled validations; "
            f"missing={missing!r}"
        )
    return [by_epoch[epoch] for epoch in EXPECTED_VALIDATION_EPOCHS]


def _validate_quality(
    validations: Sequence[Mapping[str, object]],
    *,
    selected_epoch: int,
    min_bev_ap70: float,
    min_3d_ap70: float,
) -> dict[str, object]:
    by_epoch = {int(item["epoch"]): item for item in validations}
    if set(by_epoch) != set(EXPECTED_VALIDATION_EPOCHS):
        raise RuntimeError("validation epoch set is incomplete")
    selected = by_epoch.get(selected_epoch)
    if selected is None:
        raise RuntimeError("selected teacher epoch has no validation result")
    maximum_bev_ap70 = max(float(item[BEV_AP70_FIELD]) for item in validations)
    best_epochs = [
        int(item["epoch"])
        for item in validations
        if maximum_bev_ap70 - float(item[BEV_AP70_FIELD]) <= LOG_ROUNDING_UNIT + 1e-12
    ]
    if selected_epoch not in best_epochs:
        raise RuntimeError(
            "teacher selected_epoch does not match the BEV AP70 validation best"
        )
    selected_bev_ap70 = float(selected[BEV_AP70_FIELD])
    selected_3d_ap70 = float(selected[THREE_D_AP70_FIELD])
    if selected_bev_ap70 < min_bev_ap70:
        raise RuntimeError(
            "teacher selected BEV AP70 failed the FFNet non-inferiority gate: "
            f"{selected_bev_ap70:.4f} < {min_bev_ap70:.4f}"
        )
    if selected_3d_ap70 < min_3d_ap70:
        raise RuntimeError(
            "teacher selected 3D AP70 failed the geometry safety gate: "
            f"{selected_3d_ap70:.4f} < {min_3d_ap70:.4f}"
        )
    return {
        "selected_epoch": selected_epoch,
        "maximum_bev_ap_r40_0.70": maximum_bev_ap70,
        "rounding_equivalent_best_epochs": best_epochs,
        "log_rounding_unit": LOG_ROUNDING_UNIT,
        "selected_metrics": {key: selected[key] for key in AP_FIELD_BY_KEY.values()},
    }


def _publish(task: object, payload: Mapping[str, object]) -> None:
    artifacts = getattr(task, "artifacts", None)
    if isinstance(artifacts, Mapping) and QUALITY_GATE_ARTIFACT in artifacts:
        observed = _artifact_mapping(task, QUALITY_GATE_ARTIFACT)
        if observed != payload:
            raise RuntimeError("existing teacher quality gate artifact drifted")
        return
    uploader = getattr(task, "upload_artifact", None)
    if not callable(uploader) or not uploader(
        QUALITY_GATE_ARTIFACT,
        artifact_object=dict(payload),
        wait_on_upload=True,
    ):
        raise RuntimeError("failed to publish the teacher quality gate artifact")
    flusher = getattr(task, "flush", None)
    if callable(flusher):
        flusher(wait_for_uploads=True)


def _thresholds(min_bev_ap70: float, min_3d_ap70: float) -> dict[str, object]:
    return {
        "minimum_car_bev_ap_r40_0.70": min_bev_ap70,
        "minimum_car_3d_ap_r40_0.70": min_3d_ap70,
        "bev_gate_semantics": "ffnet_style_reference_minus_noninferiority_margin",
        "three_d_gate_semantics": "independent_geometry_safety_floor",
        "reference": {
            "task_id": FFNET_STYLE_REFERENCE_TASK_ID,
            "sample_count": EXPECTED_CLEAN_SAMPLE_COUNT,
            "car_ground_truth_count": EXPECTED_CLEAN_GROUND_TRUTH_COUNT,
            "unsupported_sample_count": EXPECTED_CLEAN_UNSUPPORTED_SAMPLE_COUNT,
            "point_cloud_range": FFNET_STYLE_POINT_CLOUD_RANGE,
            "car_bev_ap_r40_0.70": FFNET_STYLE_REFERENCE_BEV_AP70,
            "car_3d_ap_r40_0.70": FFNET_STYLE_REFERENCE_3D_AP70,
            "bev_noninferiority_margin_ap": FFNET_STYLE_BEV_NONINFERIORITY_MARGIN,
        },
    }


def _safe_error_message(error: BaseException) -> str:
    message = str(error).strip()
    sensitive_markers = (
        "://",
        "\\",
        "password",
        "secret",
        "credential",
        "authorization",
        "token=",
    )
    if (
        not message
        or len(message) > 400
        or "\n" in message
        or any(marker in message.casefold() for marker in sensitive_markers)
    ):
        return "teacher quality validation failed; inspect protected task logs"
    return message


def run(
    args: argparse.Namespace,
    *,
    task_class: object = Task,
    output_task: object | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    numeric_arguments = {
        "poll_seconds": args.poll_seconds,
        "timeout_hours": args.timeout_hours,
        "min_bev_ap70": args.min_bev_ap70,
        "min_3d_ap70": args.min_3d_ap70,
    }
    for name, value in numeric_arguments.items():
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if args.poll_seconds <= 0 or args.timeout_hours <= 0:
        raise ValueError("poll interval and timeout must be positive")
    if not 0.0 <= args.min_bev_ap70 <= 100.0:
        raise ValueError("min_bev_ap70 must be inside [0, 100]")
    if not 0.0 <= args.min_3d_ap70 <= 100.0:
        raise ValueError("min_3d_ap70 must be inside [0, 100]")
    if task_class is None:
        raise RuntimeError("ClearML SDK is unavailable")

    teacher_task_id = _clearml_id(args.teacher_task_id, "teacher task")
    if output_task is None:
        current_getter = getattr(task_class, "current_task", None)
        output_task = current_getter() if callable(current_getter) else None
    if output_task is None:
        raise RuntimeError("teacher quality gate requires a current ClearML task")
    output_task_id = _clearml_id(getattr(output_task, "id", ""), "quality gate task")
    if output_task_id == teacher_task_id:
        raise RuntimeError("teacher quality gate must run as an independent task")
    tagger = getattr(output_task, "set_tags", None)
    if callable(tagger):
        tagger(
            [
                "ResilientV2X-suite",
                "teacher-quality-gate",
                CLEAN_TEACHER_SELECTION_PROTOCOL,
                "cpu-controller",
            ]
        )

    teacher = task_class.get_task(task_id=teacher_task_id)
    deadline = monotonic() + float(args.timeout_hours) * 3600.0
    _wait_for_completed(
        teacher,
        deadline=deadline,
        poll_seconds=float(args.poll_seconds),
        monotonic=monotonic,
        sleeper=sleeper,
    )
    thresholds = _thresholds(float(args.min_bev_ap70), float(args.min_3d_ap70))
    validations: list[dict[str, object]] = []
    contract_evidence: dict[str, object] = {}
    reference: dict[str, object] = {"task_id": teacher_task_id}
    try:
        if _clearml_id(getattr(teacher, "id", ""), "resolved teacher task") != (
            teacher_task_id
        ):
            raise RuntimeError("resolved teacher task identity mismatch")
        run_contract, dataset_id, contract_evidence = _validate_run_contract(
            teacher,
            teacher_task_id=teacher_task_id,
        )
        validations = _parse_validation_logs(_console_chunks(teacher))
        checkpoint_contract, reference = _validate_checkpoint_contract(
            teacher,
            teacher_task_id=teacher_task_id,
        )
        best = _validate_quality(
            validations,
            selected_epoch=int(reference["selected_epoch"]),
            min_bev_ap70=float(args.min_bev_ap70),
            min_3d_ap70=float(args.min_3d_ap70),
        )
    except Exception as error:
        failure = _seal(
            {
                "gate_type": (
                    "controlled_ffnet_style_bev_noninferiority_and_geometry_safety"
                ),
                "passed": False,
                "teacher_task_id": teacher_task_id,
                "quality_gate_task_id": output_task_id,
                "selection_protocol": CLEAN_TEACHER_SELECTION_PROTOCOL,
                "selection_metric": CLEAN_TEACHER_SELECTION_METRIC,
                "expected_validation_epochs": list(EXPECTED_VALIDATION_EPOCHS),
                "validation_count": len(validations),
                "validations": validations,
                "thresholds": thresholds,
                "teacher": reference,
                "training_seed_evidence": contract_evidence.get(
                    "training_seed_evidence", "unverified"
                ),
                "error_type": type(error).__name__,
                "error_message": _safe_error_message(error),
            }
        )
        try:
            _publish(output_task, failure)
        except Exception:
            pass
        raise
    payload = _seal(
        {
            "gate_type": (
                "controlled_ffnet_style_bev_noninferiority_and_geometry_safety"
            ),
            "passed": True,
            "teacher_task_id": teacher_task_id,
            "quality_gate_task_id": output_task_id,
            "dataset_id": dataset_id,
            "selection_protocol": CLEAN_TEACHER_SELECTION_PROTOCOL,
            "selection_metric": CLEAN_TEACHER_SELECTION_METRIC,
            "expected_validation_epochs": list(EXPECTED_VALIDATION_EPOCHS),
            "validation_count": len(validations),
            "validations": list(validations),
            "best": best,
            "thresholds": thresholds,
            "teacher": reference,
            **contract_evidence,
            "contracts": {
                "run_contract_artifact": RUN_CONTRACT_ARTIFACT,
                "run_contract_sha256": _mapping_sha256(run_contract),
                "checkpoint_contract_artifact": TEACHER_CHECKPOINT_ARTIFACT,
                "checkpoint_contract_sha256": _mapping_sha256(checkpoint_contract),
            },
        }
    )
    _publish(output_task, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if Task is None:
        raise RuntimeError("ClearML SDK is unavailable")
    task = Task.init(
        project_name=DEFAULT_PROJECT,
        task_name="ResilientV2X clean teacher quality gate",
        reuse_last_task_id=False,
        output_uri=FILES_SERVER_URI,
    )
    run(args, task_class=Task, output_task=task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "AP_METRIC_KEYS",
    "DEFAULT_MIN_3D_AP70",
    "DEFAULT_MIN_BEV_AP70",
    "EXPECTED_VALIDATION_EPOCHS",
    "QUALITY_GATE_ARTIFACT",
    "run",
)
