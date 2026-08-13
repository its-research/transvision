#!/usr/bin/env python3
"""Launch the sealed E1/E2/E3 candidate fastlane without bypassing provenance.

The default invocation is a local, deterministic dry-run.  ``--execute`` and
``--recover-exact-partials`` are the only remote-writing paths and each requires
its own exact execution token.
The launcher deliberately releases only E1 and E2 at first; E3 is kept in the
ClearML ``created`` state until E1 has completed, which makes the maximum
training concurrency two independently of queue worker topology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

try:
    from tools.resilient_v2x import clearml_sota_candidate_controller as candidate
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from tools.resilient_v2x import clearml_sota_candidate_controller as candidate


LAUNCHER_TYPE = "resilient_v2x_sota_candidate_fastlane_v1"
LAUNCH_RECEIPT_ARTIFACT = "sota_candidate_fastlane_launch_receipt"
EXECUTE_TOKEN = "LAUNCH_EXACT_E1_E2_E3_FASTLANE"
RECOVERY_EXECUTE_TOKEN = "RECOVER_EXACT_E1_E2_E3_FASTLANE_PARTIALS"
RECOVERY_TASK_IDS = (
    "f0c3082f3aa34a81805903e0ffdc8610",
    "969c8fce6d24446299561772b3955274",
    "dc037315c0684c3d854a2fd7c19a2a2f",
)
RECOVERY_RECEIPT_CREATED_AT = "2026-08-11T20:14:57.817000+00:00"
RECOVERY_RECEIPT_SEAL_SHA256 = (
    "16d119a4260824b1f870d56722e25c4de14c1b3b2ff1f41d132401e111b13492"
)

TRAINING_PROJECT = "ResilientV2X/Training"
FILES_SERVER_URI = "http://10.100.34.118:8081"
FILES_SERVER_HOST = "10.100.34.118"
FILES_SERVER_PORT = 8081

MAIN_CONTROLLER_TASK_ID = "1011e98e10f64c428880af1d4b1d542b"
TEMPLATE_TASK_ID = "4b8d48c8a64b4391bbef8969d1b39e3e"
TEMPLATE_NAME = "ResilientV2X candidate bootstrap template [f02edab673ea]"
TEMPLATE_ENTRY_POINT = "clearml_5090_bootstrap.py"
TEMPLATE_SCRIPT_SHA256 = (
    "4dfe2e9d2ee3076df1679818211f40b7c2ebcbc73efb4cdae2f230c971485b67"
)
PATCHED_TEMPLATE_SCRIPT_SHA256 = (
    "84825bfbcb63bd283be9a2703f019471393023279fed9e0bbc9c079d93c360bc"
)
SOURCE_TRANSITION_ARTIFACT = "sota_candidate_source_transition"
SOURCE_TRANSITION_ARTIFACT_SHA256 = (
    "f6d7dbda06e8bdff7943d31511407de75ce9a7ae9b14997707a291f35abde6b2"
)
SOURCE_TRANSITION_ARTIFACT_BYTES = 2_744
SOURCE_TRANSITION_SEAL_SHA256 = (
    "f02edab673ea7d6e78e81f15bf9819653c2ba3e943a7308bb1d80778409d82ad"
)

SOURCE_DATASET_ID = "c9ca3075434e44e29951e4aca36e9046"
SOURCE_DATASET_NAME = "ResilientV2X sealed SOTA candidate source 8a22d6d600a5"
SOURCE_DATASET_VERSION = "8a22d6d600a5"
SOURCE_DATA_ARTIFACT_SHA256 = (
    "defcbdbdb08b16e691828d9e74b475e0cc1285a5b4b8beafff4f461806ecc421"
)
SOURCE_DATA_ARTIFACT_BYTES = 1_323_977
SOURCE_STATE_ARTIFACT_SHA256 = (
    "88d99c9a21d6d65961e39ba3004a2d6a03c96f8fb5887a488c2c0a848d24effb"
)
SOURCE_STATE_ARTIFACT_BYTES = 977

TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
TEACHER_TASK_ID = "487dab2664a8485fa0cc7c4e2a0c3df8"
TEACHER_TASK_NAME = "ResilientV2X v4c clean teacher 5c984ad4 [robust materialization]"
TEACHER_MODEL_ID = "d962f6bae8474260b54e170a7a5f0418"
TEACHER_MODEL_NAME = "ResilientV2X clean teacher"
TEACHER_CHECKPOINT_FILENAME = (
    "best_resilient_v2x_car_bev_ap_r40_0.70_teacher_epoch_30.pth"
)
TEACHER_CHECKPOINT_SHA256 = (
    "7516eb82c7d025f49877c97bfc96a28e7a62853056007289fddd196ce2c231fb"
)
TEACHER_CHECKPOINT_BYTES = 146_189_677
TEACHER_CONTRACT_ARTIFACT = "teacher_checkpoint_contract"
TEACHER_CONTRACT_ARTIFACT_SHA256 = (
    "c017041480d23b94520dcbb31d5e55a0ea5eda04aaf6865cbf01bbac0fa1dd76"
)
TEACHER_CONTRACT_ARTIFACT_BYTES = 1_091

PREDECESSOR_TASK_ID = "21368e8260cc4e5392fe2dbdf116e36f"
PREDECESSOR_TASK_NAME = (
    "ResilientV2X post-main 18 fcooper [f8c36e508c7d453dadc766207a5b25b2]"
)
PREDECESSOR_RUN_CONTRACT_SHA256 = (
    "69660954be12ad503214834e9d1953ecdbb5fb6da672b2904b6dbe4b430069e7"
)
PREDECESSOR_RUN_CONTRACT_BYTES = 5_036

WORKER_QUEUES = ("GPU4-5090", "GPU4-V100")
MAX_PARALLEL = 2
TRAINING_SEED = 20_250_218
OVERLAY_PROTOCOL_SEED = 20_250_218
GPU_COUNT = 4
BATCH_SIZE_PER_GPU = 2
GLOBAL_BATCH_SIZE = 8
MAX_EPOCHS = 50
VAL_INTERVAL = 10
PRECISION = "FP32"

_CLEARML_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ACTIVE_STATUSES = frozenset({"queued", "in_progress"})
_PHASE_ONE_WAITABLE = frozenset({"queued", "in_progress"})
_FAILED_STATUSES = frozenset({"failed", "stopped", "closed"})

_RECOVERY_TRAINING_ARTIFACT_SEQUENCE = (
    "run_contract",
    "common_teacher_initialization_audit",
    "final_checkpoint_contract",
    "best_checkpoint_contract",
)

CANDIDATES = (
    {
        "index": 1,
        "label": "E1",
        "experiment": "support_residual_linear",
        "config": "configs/resilient_v2x/improvements/support_residual_linear.py",
        "config_sha256": (
            "a105124cf16a693a8c6fb176e07720abe0b05a6ff9d187b076b83d46c30bb4c0"
        ),
        "phase": 1,
        "queue": WORKER_QUEUES[0],
    },
    {
        "index": 2,
        "label": "E2",
        "experiment": "no_reliability_linear",
        "config": "configs/resilient_v2x/improvements/no_reliability_linear.py",
        "config_sha256": (
            "a42f6b28bf1678f663458ccc6f98a409e2483d6786423aa50b2c72e68b721e5d"
        ),
        "phase": 1,
        "queue": WORKER_QUEUES[1],
    },
    {
        "index": 3,
        "label": "E3",
        "experiment": "support_residual_no_reliability",
        "config": (
            "configs/resilient_v2x/improvements/support_residual_no_reliability.py"
        ),
        "config_sha256": (
            "86f124aa6ae542891575c26d68560f6b4835b9ebe96516fed40074dfd5af1efa"
        ),
        "phase": 2,
        "queue": WORKER_QUEUES[0],
        "release_after_completed_label": "E1",
    },
)

EXPECTED_TEMPLATE_PARAMETERS = {
    "Args/allow_failed_teacher_task": False,
    "Args/amp": False,
    "Args/build_manifest_sha256": (
        "21c6ab7a6e9a2823ba111289a42e7f882c5f4ba02c73175106ff251fe5864a43"
    ),
    "Args/controlled_baseline": "",
    "Args/controlled_baseline_checkpoint_sha256": "",
    "Args/controlled_baseline_model_id": "",
    "Args/controlled_baseline_task_id": "",
    "Args/experiment_from_task": "",
    "Args/gpus": 4,
    "Args/max_epochs": 50,
    "Args/native_bundle_bytes": 753_382_966,
    "Args/native_bundle_sha256": (
        "19b8e7f5edc8216d4b43cb17854dccafe4fc9fe46995a88803e6342eeaa22b21"
    ),
    "Args/predecessor_task_id": "",
    "Args/source_archive_bytes": 1_222_568,
    "Args/source_archive_name": "resilient-v2x-source-8a22d6d600a5.tar.zst",
    "Args/source_archive_sha256": (
        "a249546f16f236d42c8346a33b137dea27c037bf0df73c643da4132c8e589557"
    ),
    "Args/source_dataset_id": SOURCE_DATASET_ID,
    "Args/stage": "teacher",
    "Args/student_checkpoint": "",
    "Args/student_checkpoint_sha256": "",
    "Args/student_model_id": "",
    "Args/student_task_id": "",
    "Args/teacher_checkpoint": "",
    "Args/teacher_checkpoint_sha256": "",
    "Args/teacher_model_id": "",
    "Args/teacher_task_id": "",
    "Args/training_dataset_id": TRAINING_DATASET_ID,
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sealed(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = hashlib.sha256(
        _canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def _require_valid_seal(payload: Mapping[str, object], *, context: str) -> None:
    observed = payload.get("seal_sha256")
    if type(observed) is not str or _SHA256.fullmatch(observed) is None:
        raise RuntimeError(f"{context} has no valid seal_sha256")
    if observed != _sealed(payload)["seal_sha256"]:
        raise RuntimeError(f"{context} seal_sha256 mismatch")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status(task: object) -> str:
    value = getattr(task, "status", None)
    if callable(value):
        value = value()
    if value is None:
        getter = getattr(task, "get_status", None)
        value = getter() if callable(getter) else None
    return str(getattr(value, "value", value)).rsplit(".", 1)[-1].lower()


def _reload(task: object) -> None:
    reloader = getattr(task, "reload", None)
    if callable(reloader):
        reloader()


def _task_id(task: object, *, context: str) -> str:
    value = str(getattr(task, "id", "") or "")
    if _CLEARML_ID.fullmatch(value) is None:
        raise RuntimeError(f"{context} has an invalid ClearML task ID")
    return value


def _task_parent(task: object) -> str:
    value = getattr(task, "parent", "")
    if callable(value):
        value = value()
    return str(value or "")


def _parameters(task: object) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise RuntimeError("task cannot enumerate parameters")
    try:
        value = getter(backwards_compatibility=False, cast=False)
    except TypeError:
        value = getter()
    if not isinstance(value, Mapping):
        raise RuntimeError("task returned invalid parameters")
    return {str(key): item for key, item in value.items()}


def _script(task: object) -> dict[str, object]:
    data = getattr(task, "data", None)
    value = getattr(data, "script", None)
    converter = getattr(value, "to_dict", None)
    value = converter() if callable(converter) else value
    if not isinstance(value, Mapping):
        getter = getattr(task, "get_script", None)
        value = getter() if callable(getter) else None
    if not isinstance(value, Mapping):
        raise RuntimeError("task returned invalid script metadata")
    return {str(key): item for key, item in value.items()}


def _matches(actual: object, expected: object) -> bool:
    if type(expected) is bool:
        return str(actual).casefold() == str(expected).casefold()
    if type(expected) is int:
        return str(actual) == str(expected)
    return actual == expected


def _require_parameters_exact(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    context: str,
) -> None:
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"{context} parameter keys drifted: missing={missing}, extra={extra}"
        )
    for key, value in expected.items():
        if not _matches(actual.get(key), value):
            raise RuntimeError(f"{context} parameter drifted: {key}")


def _artifact_records(task: object) -> dict[str, dict[str, object]]:
    data = getattr(task, "data", None)
    execution = getattr(data, "execution", None)
    values = getattr(execution, "artifacts", None)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeError("task cannot expose server artifact metadata")
    result: dict[str, dict[str, object]] = {}
    for value in values:
        converter = getattr(value, "to_dict", None)
        record = converter() if callable(converter) else value
        if not isinstance(record, Mapping):
            raise RuntimeError("task returned invalid artifact metadata")
        key = str(record.get("key") or "")
        if not key or key in result:
            raise RuntimeError("task returned duplicate or unnamed artifact metadata")
        result[key] = dict(record)
    return result


def _artifact_record(
    task: object,
    name: str,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, object]:
    records = _artifact_records(task)
    if name not in records:
        raise RuntimeError(f"task is missing required artifact {name!r}")
    record = records[name]
    if record.get("hash") != expected_sha256:
        raise RuntimeError(f"artifact {name!r} SHA-256 drifted")
    if record.get("content_size") != expected_bytes:
        raise RuntimeError(f"artifact {name!r} size drifted")
    uri = str(record.get("uri") or "")
    parsed = urlsplit(uri)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != FILES_SERVER_HOST
        or parsed.port != FILES_SERVER_PORT
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeError(f"artifact {name!r} has an invalid fileserver URI")
    return record


def _artifact_preview_payload(
    record: Mapping[str, object], *, name: str
) -> dict[str, object]:
    type_data = record.get("type_data")
    if not isinstance(type_data, Mapping):
        raise RuntimeError(f"artifact {name!r} has no type metadata")
    preview = type_data.get("preview")
    if type(preview) is not str:
        raise RuntimeError(f"artifact {name!r} has no JSON preview")
    try:
        value = json.loads(preview)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"artifact {name!r} preview is not complete JSON") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"artifact {name!r} preview is not a JSON object")
    return dict(value)


def _source_transition_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "transition_type": "exact_additive_sota_candidate_source_revision",
        "controller_type": "resilient_v2x_sota_candidate_training_v1",
        "base_source": {
            "dataset_id": "351feedbbe81481fa31f1e9ae11a3f4e",
            "archive_name": "resilient-v2x-source-ad511d88b731.tar.zst",
            "archive_bytes": 1_222_492,
            "archive_sha256": (
                "b94a01c2acf2cc456fe9729f7c40e990e6d44b65e789c6fed11989a673f4f6da"
            ),
            "tree_sha256": (
                "ad511d88b731cb45ef2defb873712bdb2a325c648634b66c349fe1c1459510e4"
            ),
            "inventory_sha256": (
                "39b1a42af65ad5df935945bd0a4eeac6e7f6e1cfdd1cc608f3dd8a708e9c5ca0"
            ),
            "inventory_bytes": 101_195,
            "file_count": 631,
            "source_bytes": 8_926_102,
        },
        "target_source": {
            "dataset_id": SOURCE_DATASET_ID,
            "archive_name": EXPECTED_TEMPLATE_PARAMETERS["Args/source_archive_name"],
            "archive_bytes": 1_222_568,
            "archive_sha256": EXPECTED_TEMPLATE_PARAMETERS[
                "Args/source_archive_sha256"
            ],
            "tree_sha256": (
                "8a22d6d600a52117d01cfde5fa5f20fa1de269ec9c9ba040228f5e8e3a5e8767"
            ),
            "inventory_sha256": (
                "5bd96277989838760f3a2f9b6f3d86a4ce8a0ef79d5a21d366540b4fa7c7604f"
            ),
            "inventory_bytes": 101_714,
            "file_count": 634,
            "source_bytes": 8_927_627,
        },
        "inventory_delta": {
            "base_file_count": 631,
            "target_file_count": 634,
            "unchanged_file_count": 631,
            "modified_file_count": 0,
            "added_file_count": 3,
            "removed_file_count": 0,
            "modified_files": [],
            "added_files": [
                {"path": item["config"], "sha256": item["config_sha256"]}
                for item in CANDIDATES
            ],
        },
        "candidate_order": [item["experiment"] for item in CANDIDATES],
        "protocol": {
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "gpu_count": GPU_COUNT,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "max_epochs": MAX_EPOCHS,
            "val_interval": VAL_INTERVAL,
            "training_seed": TRAINING_SEED,
            "training_overlay_protocol_seed": OVERLAY_PROTOCOL_SEED,
            "precision": PRECISION,
        },
    }


def _expected_source_transition() -> dict[str, object]:
    payload = _source_transition_payload()
    observed = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    if observed != SOURCE_TRANSITION_SEAL_SHA256:
        raise RuntimeError(
            "embedded candidate source transition no longer matches its seal"
        )
    return {**payload, "seal_sha256": observed}


def _static_plan_payload() -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "launcher_type": LAUNCHER_TYPE,
        "remote_state_changed": False,
        "fixed_inputs": {
            "main_controller_task_id": MAIN_CONTROLLER_TASK_ID,
            "template_task_id": TEMPLATE_TASK_ID,
            "template_script_sha256": TEMPLATE_SCRIPT_SHA256,
            "patched_template_script_sha256": PATCHED_TEMPLATE_SCRIPT_SHA256,
            "source_dataset_id": SOURCE_DATASET_ID,
            "source_transition_seal_sha256": SOURCE_TRANSITION_SEAL_SHA256,
            "teacher_task_id": TEACHER_TASK_ID,
            "teacher_model_id": TEACHER_MODEL_ID,
            "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
            "predecessor_task_id": PREDECESSOR_TASK_ID,
            "training_dataset_id": TRAINING_DATASET_ID,
        },
        "protocol": {
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "gpu_count": GPU_COUNT,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "max_epochs": MAX_EPOCHS,
            "val_interval": VAL_INTERVAL,
            "training_seed": TRAINING_SEED,
            "training_overlay_protocol_seed": OVERLAY_PROTOCOL_SEED,
            "precision": PRECISION,
            "amp": False,
        },
        "worker_queues": list(WORKER_QUEUES),
        "max_parallel": MAX_PARALLEL,
        "candidates": [dict(item) for item in CANDIDATES],
        "release_policy": {
            "phase_1": ["E1", "E2"],
            "phase_2": ["E3"],
            "phase_2_condition": "E1 task status is exactly completed",
            "e3_pre_enqueue_state": "created",
            "all_training_predecessors": PREDECESSOR_TASK_ID,
        },
        "execution": {
            "requires_execute_flag": True,
            "requires_exact_token": EXECUTE_TOKEN,
            "remote_mutations": [
                "clone three tasks",
                "patch candidate identities into each clone",
                "set and verify sealed parameters",
                "upload one common sealed launch receipt to every clone",
                "enqueue E1/E2, then enqueue E3 only after E1 completes",
            ],
        },
    }
    return _sealed(payload)


def _patched_template_diff(diff: str) -> str:
    if hashlib.sha256(diff.encode("utf-8")).hexdigest() != TEMPLATE_SCRIPT_SHA256:
        raise RuntimeError("candidate template raw script SHA-256 drifted")
    patched = candidate._apply_candidate_experiment_patch(diff)
    observed = hashlib.sha256(patched.encode("utf-8")).hexdigest()
    if observed != PATCHED_TEMPLATE_SCRIPT_SHA256:
        raise RuntimeError("candidate template patched script SHA-256 drifted")
    for item in CANDIDATES:
        if patched.count(f'"{item["experiment"]}"') < 2:
            raise RuntimeError(f"patched template lacks {item['label']} identity")
        if patched.count(f'"{item["config"]}"') != 1:
            raise RuntimeError(
                f"patched template config is ambiguous for {item['label']}"
            )
    return patched


def _validate_fixed_inputs(task_class: object) -> tuple[object, str]:
    main = task_class.get_task(task_id=MAIN_CONTROLLER_TASK_ID)
    if _task_id(main, context="main controller") != MAIN_CONTROLLER_TASK_ID:
        raise RuntimeError("main controller task ID drifted")
    if _status(main) != "in_progress":
        raise RuntimeError(
            "fastlane is authorized only while the main controller is in progress"
        )

    source = task_class.get_task(task_id=SOURCE_DATASET_ID)
    if _task_id(source, context="candidate source Dataset") != SOURCE_DATASET_ID:
        raise RuntimeError("candidate source Dataset task ID drifted")
    if (
        _status(source) != "completed"
        or getattr(source, "name", "") != SOURCE_DATASET_NAME
    ):
        raise RuntimeError("candidate source Dataset identity drifted")
    source_parameters = _parameters(source)
    _require_parameters_exact(
        source_parameters,
        {"properties/version": SOURCE_DATASET_VERSION},
        context="candidate source Dataset",
    )
    _artifact_record(
        source,
        "data",
        expected_sha256=SOURCE_DATA_ARTIFACT_SHA256,
        expected_bytes=SOURCE_DATA_ARTIFACT_BYTES,
    )
    _artifact_record(
        source,
        "state",
        expected_sha256=SOURCE_STATE_ARTIFACT_SHA256,
        expected_bytes=SOURCE_STATE_ARTIFACT_BYTES,
    )

    teacher = task_class.get_task(task_id=TEACHER_TASK_ID)
    if (
        _status(teacher) != "completed"
        or getattr(teacher, "name", "") != TEACHER_TASK_NAME
    ):
        raise RuntimeError("sealed teacher task identity drifted")
    models = teacher.get_models()
    output_models = models.get("output", ()) if isinstance(models, Mapping) else ()
    if not isinstance(output_models, Sequence) or len(output_models) != 1:
        raise RuntimeError("sealed teacher must expose exactly one OutputModel")
    model = output_models[0]
    if (
        str(getattr(model, "id", "") or "") != TEACHER_MODEL_ID
        or str(getattr(model, "task", "") or "") != TEACHER_TASK_ID
        or str(getattr(model, "name", "") or "") != TEACHER_MODEL_NAME
    ):
        raise RuntimeError("sealed teacher OutputModel binding drifted")
    model_url = str(getattr(model, "url", "") or "")
    parsed_model_url = urlsplit(model_url)
    if (
        parsed_model_url.hostname != FILES_SERVER_HOST
        or parsed_model_url.port != FILES_SERVER_PORT
        or Path(unquote(parsed_model_url.path)).name != TEACHER_CHECKPOINT_FILENAME
    ):
        raise RuntimeError("sealed teacher OutputModel URL drifted")
    teacher_record = _artifact_record(
        teacher,
        TEACHER_CONTRACT_ARTIFACT,
        expected_sha256=TEACHER_CONTRACT_ARTIFACT_SHA256,
        expected_bytes=TEACHER_CONTRACT_ARTIFACT_BYTES,
    )
    teacher_contract = _artifact_preview_payload(
        teacher_record, name=TEACHER_CONTRACT_ARTIFACT
    )
    selected = teacher_contract.get("selected_checkpoint")
    if not isinstance(selected, Mapping):
        raise RuntimeError("teacher checkpoint contract has no selected checkpoint")
    expected_selected = {
        "model_id": TEACHER_MODEL_ID,
        "name": TEACHER_MODEL_NAME,
        "filename": TEACHER_CHECKPOINT_FILENAME,
        "size_bytes": TEACHER_CHECKPOINT_BYTES,
        "sha256": TEACHER_CHECKPOINT_SHA256,
    }
    for key, value in expected_selected.items():
        if selected.get(key) != value:
            raise RuntimeError(f"teacher checkpoint contract drifted: {key}")
    if (
        teacher_contract.get("selected_epoch") != 30
        or teacher_contract.get("trained_epochs") != 50
    ):
        raise RuntimeError("teacher checkpoint epoch contract drifted")

    predecessor = task_class.get_task(task_id=PREDECESSOR_TASK_ID)
    if (
        _status(predecessor) != "completed"
        or getattr(predecessor, "name", "") != PREDECESSOR_TASK_NAME
    ):
        raise RuntimeError("fastlane predecessor task is not the exact completed gate")
    predecessor_parameters = _parameters(predecessor)
    predecessor_expected = {
        "Args/experiment_from_task": "fcooper",
        "Args/gpus": 4,
        "Args/max_epochs": 50,
        "Args/training_seed": TRAINING_SEED,
        "Args/amp": False,
        "Args/teacher_task_id": TEACHER_TASK_ID,
        "Args/teacher_model_id": TEACHER_MODEL_ID,
        "Args/teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        "Args/training_dataset_id": TRAINING_DATASET_ID,
    }
    for key, value in predecessor_expected.items():
        if not _matches(predecessor_parameters.get(key), value):
            raise RuntimeError(f"fastlane predecessor parameter drifted: {key}")
    predecessor_record = _artifact_record(
        predecessor,
        "run_contract",
        expected_sha256=PREDECESSOR_RUN_CONTRACT_SHA256,
        expected_bytes=PREDECESSOR_RUN_CONTRACT_BYTES,
    )
    predecessor_contract = _artifact_preview_payload(
        predecessor_record, name="run_contract"
    )
    predecessor_contract_expected = {
        "task_id": PREDECESSOR_TASK_ID,
        "experiment": "fcooper",
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "gpus": GPU_COUNT,
        "max_epochs": MAX_EPOCHS,
        "val_interval": VAL_INTERVAL,
        "seed": TRAINING_SEED,
        "precision": PRECISION,
        "training_dataset_id": TRAINING_DATASET_ID,
    }
    for key, value in predecessor_contract_expected.items():
        if predecessor_contract.get(key) != value:
            raise RuntimeError(f"fastlane predecessor run contract drifted: {key}")

    template = task_class.get_task(task_id=TEMPLATE_TASK_ID)
    if (
        _task_id(template, context="candidate template") != TEMPLATE_TASK_ID
        or _status(template) != "completed"
        or getattr(template, "name", "") != TEMPLATE_NAME
        or _task_parent(template) != MAIN_CONTROLLER_TASK_ID
    ):
        raise RuntimeError("candidate template identity drifted")
    _require_parameters_exact(
        _parameters(template),
        EXPECTED_TEMPLATE_PARAMETERS,
        context="candidate template",
    )
    template_script = _script(template)
    if template_script.get("entry_point") != TEMPLATE_ENTRY_POINT:
        raise RuntimeError("candidate template entry point drifted")
    diff = str(template_script.get("diff") or "")
    patched = _patched_template_diff(diff)
    transition_record = _artifact_record(
        template,
        SOURCE_TRANSITION_ARTIFACT,
        expected_sha256=SOURCE_TRANSITION_ARTIFACT_SHA256,
        expected_bytes=SOURCE_TRANSITION_ARTIFACT_BYTES,
    )
    observed_transition = _artifact_preview_payload(
        transition_record, name=SOURCE_TRANSITION_ARTIFACT
    )
    if observed_transition != _expected_source_transition():
        raise RuntimeError("candidate template source transition payload drifted")
    return template, patched


def _task_parameters(experiment: str) -> dict[str, object]:
    parameters = dict(EXPECTED_TEMPLATE_PARAMETERS)
    parameters.update(
        {
            "Args/stage": "all",
            "Args/experiment_from_task": experiment,
            "Args/predecessor_task_id": PREDECESSOR_TASK_ID,
            "Args/teacher_task_id": TEACHER_TASK_ID,
            "Args/teacher_model_id": TEACHER_MODEL_ID,
            "Args/teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
            "Args/allow_failed_teacher_task": False,
            "Args/training_seed": TRAINING_SEED,
            "Args/gpus": GPU_COUNT,
            "Args/max_epochs": MAX_EPOCHS,
            "Args/amp": False,
        }
    )
    return parameters


def _task_name(item: Mapping[str, object], plan_seal: str) -> str:
    return (
        f"ResilientV2X fastlane {item['label']} {item['experiment']} [{plan_seal[:12]}]"
    )


def _require_no_existing_tasks(task_class: object, *, plan_seal: str) -> None:
    getter = getattr(task_class, "get_tasks", None)
    if not callable(getter):
        raise RuntimeError("ClearML task class cannot perform duplicate preflight")
    for item in CANDIDATES:
        name = _task_name(item, plan_seal)
        values = getter(
            task_name=f"^{re.escape(name)}$",
            task_filter={"parent": PREDECESSOR_TASK_ID},
        )
        exact = [task for task in values or () if getattr(task, "name", "") == name]
        if exact:
            raise RuntimeError(f"refusing duplicate fastlane task name: {name}")


def _edit_script(task_id: str, diff: str) -> None:
    from clearml.backend_api.session.client import APIClient

    response = APIClient().tasks.edit(task=task_id, script={"diff": diff})
    if response is None or response is False:
        raise RuntimeError("ClearML did not confirm cloned task script edit")


def _validate_clone(
    task: object,
    *,
    item: Mapping[str, object],
    name: str,
    patched_diff: str,
    expected_parameters: Mapping[str, object],
) -> None:
    if (
        _status(task) != "created"
        or getattr(task, "name", "") != name
        or _task_parent(task) != PREDECESSOR_TASK_ID
    ):
        raise RuntimeError(f"{item['label']} clone identity drifted")
    task_script = _script(task)
    if (
        task_script.get("entry_point") != TEMPLATE_ENTRY_POINT
        or hashlib.sha256(
            str(task_script.get("diff") or "").encode("utf-8")
        ).hexdigest()
        != PATCHED_TEMPLATE_SCRIPT_SHA256
        or str(task_script.get("diff") or "") != patched_diff
    ):
        raise RuntimeError(f"{item['label']} clone script drifted")
    _require_parameters_exact(
        _parameters(task), expected_parameters, context=f"{item['label']} clone"
    )


def _upload_receipt(task: object, receipt: Mapping[str, object]) -> None:
    uploader = getattr(task, "upload_artifact", None)
    if not callable(uploader) or not uploader(
        LAUNCH_RECEIPT_ARTIFACT,
        artifact_object=dict(receipt),
        wait_on_upload=True,
    ):
        raise RuntimeError("failed to upload sealed fastlane launch receipt")
    flusher = getattr(task, "flush", None)
    if callable(flusher):
        flusher(wait_for_uploads=True)


def _enqueue(task_class: object, task: object, *, queue: str) -> None:
    response = task_class.enqueue(task=task, queue_name=queue)
    if response is None or response is False:
        raise RuntimeError(f"ClearML did not confirm enqueue to {queue!r}")
    _reload(task)
    if _status(task) not in _ACTIVE_STATUSES:
        raise RuntimeError(f"task did not enter queue {queue!r}")


def _launch_receipt(
    *,
    plan: Mapping[str, object],
    tasks: Sequence[object],
    created_at: str,
) -> dict[str, object]:
    if len(tasks) != len(CANDIDATES):
        raise RuntimeError("fastlane receipt task count mismatch")
    task_entries = []
    for item, task in zip(CANDIDATES, tasks, strict=True):
        parameters = _task_parameters(str(item["experiment"]))
        task_entries.append(
            {
                **dict(item),
                "task_id": _task_id(task, context=f"{item['label']} clone"),
                "task_name": getattr(task, "name", ""),
                "parent_task_id": PREDECESSOR_TASK_ID,
                "training_predecessor_task_id": PREDECESSOR_TASK_ID,
                "parameters_sha256": hashlib.sha256(
                    _canonical_json(parameters).encode("utf-8")
                ).hexdigest(),
                "script_sha256": PATCHED_TEMPLATE_SCRIPT_SHA256,
                "receipt_upload_state": "before_any_enqueue",
            }
        )
    receipt = _sealed(
        {
            "schema_version": 1,
            "receipt_type": LAUNCHER_TYPE,
            "created_at": created_at,
            "plan_seal_sha256": plan["seal_sha256"],
            "template_task_id": TEMPLATE_TASK_ID,
            "source_dataset_id": SOURCE_DATASET_ID,
            "source_transition_seal_sha256": SOURCE_TRANSITION_SEAL_SHA256,
            "teacher": {
                "task_id": TEACHER_TASK_ID,
                "model_id": TEACHER_MODEL_ID,
                "checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
            },
            "protocol": dict(plan["protocol"]),
            "worker_queues": list(WORKER_QUEUES),
            "max_parallel": MAX_PARALLEL,
            "release_policy": dict(plan["release_policy"]),
            "tasks": task_entries,
        }
    )
    _require_valid_seal(receipt, context="fastlane launch receipt")
    return receipt


def _execution_queue(task: object) -> str | None:
    data = getattr(task, "data", None)
    execution = getattr(data, "execution", None)
    value = getattr(execution, "queue", None)
    return str(value) if value not in (None, "") else None


def _require_exact_recovery_partials(
    task_class: object,
    *,
    plan: Mapping[str, object],
    patched_diff: str,
) -> list[object]:
    getter = getattr(task_class, "get_tasks", None)
    if not callable(getter):
        raise RuntimeError("ClearML task class cannot verify recovery task uniqueness")
    tasks: list[object] = []
    for item, expected_task_id in zip(CANDIDATES, RECOVERY_TASK_IDS, strict=True):
        name = _task_name(item, str(plan["seal_sha256"]))
        matches = getter(
            task_name=f"^{re.escape(name)}$",
            task_filter={"parent": PREDECESSOR_TASK_ID},
        )
        exact = [task for task in matches or () if getattr(task, "name", "") == name]
        if len(exact) != 1:
            raise RuntimeError(
                f"{item['label']} recovery requires exactly one named partial task"
            )
        discovered_id = _task_id(
            exact[0], context=f"{item['label']} discovered partial"
        )
        if discovered_id != expected_task_id:
            raise RuntimeError(f"{item['label']} recovery task ID is not allowlisted")
        task = task_class.get_task(task_id=expected_task_id)
        _reload(task)
        if (
            _task_id(task, context=f"{item['label']} recovery partial")
            != expected_task_id
        ):
            raise RuntimeError(f"{item['label']} recovery task lookup drifted")
        if _status(task) != "created":
            raise RuntimeError(f"{item['label']} recovery partial is not created")
        if _execution_queue(task) is not None:
            raise RuntimeError(f"{item['label']} recovery partial already has a queue")
        if _artifact_records(task):
            raise RuntimeError(
                f"{item['label']} recovery partial artifacts are not empty"
            )
        output_models = task.get_models().get("output", ())
        if output_models:
            raise RuntimeError(f"{item['label']} recovery partial has output models")
        if str(getattr(task, "output_uri", "") or "") != FILES_SERVER_URI:
            raise RuntimeError(f"{item['label']} recovery partial output URI drifted")
        _validate_clone(
            task,
            item=item,
            name=name,
            patched_diff=patched_diff,
            expected_parameters=_task_parameters(str(item["experiment"])),
        )
        tasks.append(task)
    return tasks


def _recovery_receipt(
    *,
    plan: Mapping[str, object],
    tasks: Sequence[object],
) -> dict[str, object]:
    receipt = _launch_receipt(
        plan=plan,
        tasks=tasks,
        created_at=RECOVERY_RECEIPT_CREATED_AT,
    )
    receipt.pop("seal_sha256")
    receipt["recovery"] = {
        "mode": "adopt_exact_created_pre_enqueue_partial_v1",
        "incident": "fileserver_401_before_first_receipt_upload",
        "observed_task_ids": list(RECOVERY_TASK_IDS),
        "observed_statuses": ["created"] * len(RECOVERY_TASK_IDS),
        "observed_execution_queues": [None] * len(RECOVERY_TASK_IDS),
        "observed_artifact_names": [[] for _ in RECOVERY_TASK_IDS],
        "prior_enqueue_count": 0,
        "server_time_anchor": "latest_clone_created_at",
        "all_clone_identity_checks": "pass",
    }
    sealed = _sealed(receipt)
    _require_valid_seal(sealed, context="fastlane recovery launch receipt")
    if sealed["seal_sha256"] != RECOVERY_RECEIPT_SEAL_SHA256:
        raise RuntimeError("fastlane recovery launch receipt seal drifted")
    return sealed


def _force_server_artifact_payload(task: object, name: str) -> dict[str, object]:
    _reload(task)
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or name not in artifacts:
        raise RuntimeError(f"task is missing uploaded artifact {name!r}")
    getter = getattr(artifacts[name], "get", None)
    if not callable(getter):
        raise RuntimeError(f"uploaded artifact {name!r} cannot be read back")
    try:
        value = getter(force_download=True)
    except TypeError as error:
        raise RuntimeError(
            f"uploaded artifact {name!r} cannot force a server readback"
        ) from error
    if isinstance(value, Mapping):
        return dict(value)
    try:
        path = Path(value).resolve(strict=True)
    except (OSError, TypeError) as error:
        raise RuntimeError(
            f"uploaded artifact {name!r} is not readable JSON"
        ) from error
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"uploaded artifact {name!r} is not a regular JSON file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"uploaded artifact {name!r} is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"uploaded artifact {name!r} is not a JSON object")
    return dict(payload)


def _recovery_receipt_readback(
    task: object,
    *,
    receipt: Mapping[str, object],
) -> dict[str, object]:
    _reload(task)
    status = _status(task)
    records = _artifact_records(task)
    observed = frozenset(records)
    receipt_only = frozenset({LAUNCH_RECEIPT_ARTIFACT})
    completed = receipt_only | frozenset(_RECOVERY_TRAINING_ARTIFACT_SEQUENCE)
    if status in {"created", "queued"}:
        if observed != receipt_only:
            raise RuntimeError(
                "created/queued recovery task artifact inventory is not receipt-only"
            )
    elif status == "in_progress":
        allowed = {
            receipt_only
            | frozenset(_RECOVERY_TRAINING_ARTIFACT_SEQUENCE[:prefix_length])
            for prefix_length in range(len(_RECOVERY_TRAINING_ARTIFACT_SEQUENCE) + 1)
        }
        if observed not in allowed:
            raise RuntimeError(
                "in-progress recovery task training artifact inventory drifted"
            )
    elif status == "completed":
        if observed != completed:
            raise RuntimeError(
                "completed recovery task checkpoint-contract artifact inventory drifted"
            )
    else:
        raise RuntimeError(
            f"recovery task status cannot authorize artifact inventory: {status!r}"
        )
    record = records[LAUNCH_RECEIPT_ARTIFACT]
    digest = str(record.get("hash") or "")
    if _SHA256.fullmatch(digest) is None:
        raise RuntimeError("recovery receipt artifact has no valid content hash")
    size = record.get("content_size")
    if type(size) is not int or size <= 0:
        raise RuntimeError("recovery receipt artifact has no positive content size")
    uri = str(record.get("uri") or "")
    parsed = urlsplit(uri)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != FILES_SERVER_HOST
        or parsed.port != FILES_SERVER_PORT
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeError("recovery receipt artifact has an invalid fileserver URI")
    preview = _artifact_preview_payload(record, name=LAUNCH_RECEIPT_ARTIFACT)
    if preview != receipt:
        raise RuntimeError("recovery receipt server metadata preview drifted")
    first = _force_server_artifact_payload(task, LAUNCH_RECEIPT_ARTIFACT)
    second = _force_server_artifact_payload(task, LAUNCH_RECEIPT_ARTIFACT)
    if first != receipt or second != receipt or first != second:
        raise RuntimeError("recovery receipt forced server readback drifted")
    _require_valid_seal(first, context="recovery receipt server readback")
    return {"hash": digest, "content_size": size, "uri": uri}


def _require_common_recovery_receipts(
    tasks: Sequence[object],
    *,
    receipt: Mapping[str, object],
) -> list[dict[str, object]]:
    observations = [_recovery_receipt_readback(task, receipt=receipt) for task in tasks]
    if len({item["hash"] for item in observations}) != 1:
        raise RuntimeError("recovery receipt artifact hashes differ across tasks")
    if len({item["content_size"] for item in observations}) != 1:
        raise RuntimeError("recovery receipt artifact sizes differ across tasks")
    return observations


def _validate_recovery_ready_task(
    task: object,
    *,
    item: Mapping[str, object],
    plan: Mapping[str, object],
    patched_diff: str,
    receipt: Mapping[str, object],
) -> None:
    _reload(task)
    if _status(task) != "created":
        raise RuntimeError(f"{item['label']} changed state before recovery enqueue")
    if _execution_queue(task) is not None:
        raise RuntimeError(f"{item['label']} acquired a queue before recovery enqueue")
    _validate_clone(
        task,
        item=item,
        name=_task_name(item, str(plan["seal_sha256"])),
        patched_diff=patched_diff,
        expected_parameters=_task_parameters(str(item["experiment"])),
    )
    _recovery_receipt_readback(task, receipt=receipt)


def recover_exact_partials(
    *,
    execute_token: str,
    poll_seconds: float,
    task_class: object,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Adopt only the three fixed pre-enqueue partials and resume the fastlane."""

    if execute_token != RECOVERY_EXECUTE_TOKEN:
        raise PermissionError(
            f"recovery mutation requires --execute-token {RECOVERY_EXECUTE_TOKEN}"
        )
    if poll_seconds <= 0:
        raise ValueError("poll seconds must be positive")

    plan = _static_plan_payload()
    _require_valid_seal(plan, context="fastlane static plan")
    _template, patched_diff = _validate_fixed_inputs(task_class)
    tasks = _require_exact_recovery_partials(
        task_class,
        plan=plan,
        patched_diff=patched_diff,
    )
    receipt = _recovery_receipt(plan=plan, tasks=tasks)

    # This is the recovery transaction boundary: no enqueue is permitted until
    # every upload and two independent forced fileserver readbacks have passed.
    for task in tasks:
        _upload_receipt(task, receipt)
    receipt_observations = _require_common_recovery_receipts(
        tasks,
        receipt=receipt,
    )
    for item, task in zip(CANDIDATES, tasks, strict=True):
        _validate_recovery_ready_task(
            task,
            item=item,
            plan=plan,
            patched_diff=patched_diff,
            receipt=receipt,
        )

    _enqueue(task_class, tasks[0], queue=str(CANDIDATES[0]["queue"]))
    _enqueue(task_class, tasks[1], queue=str(CANDIDATES[1]["queue"]))
    if sum(_status(task) == "in_progress" for task in tasks) > MAX_PARALLEL:
        raise RuntimeError("recovered fastlane exceeded its concurrency limit")

    while True:
        _reload(tasks[0])
        e1_status = _status(tasks[0])
        if e1_status == "completed":
            break
        if e1_status in _FAILED_STATUSES:
            raise RuntimeError(f"recovered E1 ended before E3 release: {e1_status!r}")
        if e1_status not in _PHASE_ONE_WAITABLE:
            raise RuntimeError(
                f"recovered E1 has unexpected pre-release status: {e1_status!r}"
            )
        _reload(tasks[2])
        if _status(tasks[2]) != "created" or _execution_queue(tasks[2]) is not None:
            raise RuntimeError("recovered E3 changed state before its release gate")
        sleeper(poll_seconds)

    _reload(tasks[1])
    e2_status = _status(tasks[1])
    if e2_status in _FAILED_STATUSES:
        raise RuntimeError(f"recovered E2 failed before E3 release: {e2_status!r}")
    if e2_status not in {"queued", "in_progress", "completed"}:
        raise RuntimeError(
            f"recovered E2 has unexpected status before E3 release: {e2_status!r}"
        )
    _validate_recovery_ready_task(
        tasks[2],
        item=CANDIDATES[2],
        plan=plan,
        patched_diff=patched_diff,
        receipt=receipt,
    )
    _require_common_recovery_receipts(tasks, receipt=receipt)
    _enqueue(task_class, tasks[2], queue=str(CANDIDATES[2]["queue"]))
    if sum(_status(task) == "in_progress" for task in tasks) > MAX_PARALLEL:
        raise RuntimeError("recovered fastlane exceeded its concurrency limit")

    return _sealed(
        {
            "schema_version": 1,
            "result_type": f"{LAUNCHER_TYPE}_exact_partial_recovery_release",
            "remote_state_changed": True,
            "launch_receipt_seal_sha256": receipt["seal_sha256"],
            "receipt_artifact_hash": receipt_observations[0]["hash"],
            "receipt_artifact_bytes": receipt_observations[0]["content_size"],
            "e1_release_status": "completed",
            "e2_release_status": e2_status,
            "tasks": [
                {
                    "label": item["label"],
                    "task_id": task_id,
                    "queue": item["queue"],
                    "status": _status(task),
                }
                for item, task_id, task in zip(
                    CANDIDATES, RECOVERY_TASK_IDS, tasks, strict=True
                )
            ],
        }
    )


def execute_fastlane(
    *,
    execute_token: str,
    poll_seconds: float,
    task_class: object,
    script_editor: Callable[[str, str], None] = _edit_script,
    sleeper: Callable[[float], None] = time.sleep,
    now: Callable[[], str] = _now,
) -> dict[str, object]:
    """Create, seal, and release the exact three-task two-lane schedule."""

    if execute_token != EXECUTE_TOKEN:
        raise PermissionError(
            f"remote mutation requires --execute-token {EXECUTE_TOKEN}"
        )
    if poll_seconds <= 0:
        raise ValueError("poll seconds must be positive")

    plan = _static_plan_payload()
    _require_valid_seal(plan, context="fastlane static plan")
    template, patched_diff = _validate_fixed_inputs(task_class)
    _require_no_existing_tasks(task_class, plan_seal=str(plan["seal_sha256"]))

    tasks: list[object] = []
    for item in CANDIDATES:
        name = _task_name(item, str(plan["seal_sha256"]))
        task = task_class.clone(
            source_task=template,
            name=name,
            parent=PREDECESSOR_TASK_ID,
        )
        if _status(task) != "created":
            raise RuntimeError(f"new {item['label']} clone is not in created state")
        task_id = _task_id(task, context=f"{item['label']} clone")
        script_editor(task_id, patched_diff)
        _reload(task)
        parameters = _task_parameters(str(item["experiment"]))
        setter = getattr(task, "set_parameters", None)
        if not callable(setter):
            raise RuntimeError("cloned task cannot set sealed parameters")
        setter(parameters)
        setattr(task, "output_uri", FILES_SERVER_URI)
        _reload(task)
        _validate_clone(
            task,
            item=item,
            name=name,
            patched_diff=patched_diff,
            expected_parameters=parameters,
        )
        tasks.append(task)

    receipt = _launch_receipt(plan=plan, tasks=tasks, created_at=now())
    for task in tasks:
        _upload_receipt(task, receipt)

    # No enqueue occurs until all three clones have the same sealed receipt.
    _enqueue(task_class, tasks[0], queue=str(CANDIDATES[0]["queue"]))
    _enqueue(task_class, tasks[1], queue=str(CANDIDATES[1]["queue"]))
    if sum(_status(task) == "in_progress" for task in tasks) > MAX_PARALLEL:
        raise RuntimeError("fastlane exceeded its sealed concurrency limit")

    while True:
        _reload(tasks[0])
        e1_status = _status(tasks[0])
        if e1_status == "completed":
            break
        if e1_status in _FAILED_STATUSES:
            raise RuntimeError(f"E1 ended before E3 release: {e1_status!r}")
        if e1_status not in _PHASE_ONE_WAITABLE:
            raise RuntimeError(f"E1 has unexpected pre-release status: {e1_status!r}")
        sleeper(poll_seconds)

    _reload(tasks[1])
    e2_status = _status(tasks[1])
    if e2_status in _FAILED_STATUSES:
        raise RuntimeError(f"E2 failed before E3 release: {e2_status!r}")
    if e2_status not in {"queued", "in_progress", "completed"}:
        raise RuntimeError(f"E2 has unexpected status before E3 release: {e2_status!r}")
    _validate_clone(
        tasks[2],
        item=CANDIDATES[2],
        name=_task_name(CANDIDATES[2], str(plan["seal_sha256"])),
        patched_diff=patched_diff,
        expected_parameters=_task_parameters(str(CANDIDATES[2]["experiment"])),
    )
    _enqueue(task_class, tasks[2], queue=str(CANDIDATES[2]["queue"]))
    if sum(_status(task) == "in_progress" for task in tasks) > MAX_PARALLEL:
        raise RuntimeError("fastlane exceeded its sealed concurrency limit")

    result = _sealed(
        {
            "schema_version": 1,
            "result_type": f"{LAUNCHER_TYPE}_release",
            "remote_state_changed": True,
            "launch_receipt_seal_sha256": receipt["seal_sha256"],
            "e1_release_status": "completed",
            "e2_release_status": e2_status,
            "tasks": [
                {
                    "label": item["label"],
                    "experiment": item["experiment"],
                    "task_id": _task_id(task, context=f"{item['label']} task"),
                    "queue": item["queue"],
                    "status": _status(task),
                }
                for item, task in zip(CANDIDATES, tasks, strict=True)
            ],
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--execute",
        action="store_true",
        help="perform the sealed remote launch instead of printing a dry-run",
    )
    modes.add_argument(
        "--recover-exact-partials",
        action="store_true",
        help="adopt only the three allowlisted pre-enqueue partial tasks",
    )
    parser.add_argument("--execute-token", default="")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute and not args.recover_exact_partials:
        if args.execute_token:
            raise ValueError("--execute-token is invalid without an execution mode")
        print(json.dumps(_static_plan_payload(), sort_keys=True))
        return 0
    expected_token = (
        RECOVERY_EXECUTE_TOKEN if args.recover_exact_partials else EXECUTE_TOKEN
    )
    if args.execute_token != expected_token:
        raise PermissionError(
            f"remote mutation requires --execute-token {expected_token}"
        )
    from clearml import Task

    if args.recover_exact_partials:
        result = recover_exact_partials(
            execute_token=args.execute_token,
            poll_seconds=args.poll_seconds,
            task_class=Task,
        )
    else:
        result = execute_fastlane(
            execute_token=args.execute_token,
            poll_seconds=args.poll_seconds,
            task_class=Task,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CANDIDATES",
    "EXECUTE_TOKEN",
    "LAUNCH_RECEIPT_ARTIFACT",
    "MAX_PARALLEL",
    "PATCHED_TEMPLATE_SCRIPT_SHA256",
    "PREDECESSOR_TASK_ID",
    "RECOVERY_EXECUTE_TOKEN",
    "RECOVERY_RECEIPT_SEAL_SHA256",
    "RECOVERY_TASK_IDS",
    "SOURCE_DATASET_ID",
    "TEMPLATE_TASK_ID",
    "TEACHER_CHECKPOINT_SHA256",
    "TEACHER_MODEL_ID",
    "TEACHER_TASK_ID",
    "WORKER_QUEUES",
    "execute_fastlane",
    "recover_exact_partials",
)
