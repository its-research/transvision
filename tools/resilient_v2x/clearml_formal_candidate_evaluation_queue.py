#!/usr/bin/env python3
"""Create and supervise the five final-only formal candidate evaluations.

The default command is a deterministic dry-run and performs no ClearML writes.
Remote writes require both ``--execute`` and the exact execution token.  In
execute mode this process may be queued as a ClearML service before training
finishes; it creates an evaluation only after the corresponding training task
is authoritatively completed and its final checkpoint contracts validate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

try:
    from allegroai import Task
except ImportError:
    try:
        from clearml import Task
    except ImportError:  # pragma: no cover - pure unit-test environments
        Task = None  # type: ignore[assignment]


PROJECT_NAME = "ResilientV2X/Training"
PROJECT_ID = "6e43f972e5ea4cee901a7c8855fce8cd"
CONTROLLER_NAME = "ResilientV2X formal 1337 candidate evaluation queue"
CONTROLLER_ARTIFACT = "formal_1337_candidate_evaluation_manifest"
EXECUTE_TOKEN = "EXECUTE_EXACT_FORMAL_1337_CANDIDATE_EVALUATIONS"
DEFAULT_SERVICE_QUEUE = "services"

PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
TRAINING_SEED = 20_250_218
SAMPLE_COUNT = 1_337
GROUND_TRUTH_COUNT = 11_330
UNSUPPORTED_SAMPLE_COUNT = 0
DELAYS_MS = (0, 100, 200, 300)
CONDITIONS = ("Full", "L-Fail", "C-Fail")
RUN_COUNT = 12
CHECKPOINT_POLICY = "epoch_50_final_only"
MANIFEST_CONTENT_SHA256 = (
    "715ac6f7a14225e20327eed0650c55abdc0cb98431830164e84545238099645d"
)
OVERLAY_INDEX_CONTENT_SHA256 = (
    "77bd4585dbb02901f862b8da6aa208a504674b824a3d55cf15005aacbeeeaaff"
)
SAMPLE_IDS_SHA256 = (
    "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"
)

TEACHER_TASK_ID = "487dab2664a8485fa0cc7c4e2a0c3df8"
TEACHER_MODEL_ID = "d962f6bae8474260b54e170a7a5f0418"
TEACHER_CHECKPOINT_SHA256 = (
    "7516eb82c7d025f49877c97bfc96a28e7a62853056007289fddd196ce2c231fb"
)
PREDECESSOR_TASK_ID = "21368e8260cc4e5392fe2dbdf116e36f"
TEMPLATE_PARENT_TASK_ID = "1011e98e10f64c428880af1d4b1d542b"
TEMPLATE_SCRIPT_SHA256 = (
    "4dfe2e9d2ee3076df1679818211f40b7c2ebcbc73efb4cdae2f230c971485b67"
)
TEMPLATE_ENTRY_POINT = "clearml_5090_bootstrap.py"
SOURCE_EVALUATOR_SHA256 = (
    "d233e054f2b608bc441de25833fb89d117197d841752812a0054e0514995ad36"
)

NATIVE_BUNDLE_SHA256 = (
    "19b8e7f5edc8216d4b43cb17854dccafe4fc9fe46995a88803e6342eeaa22b21"
)
BUILD_MANIFEST_SHA256 = (
    "21c6ab7a6e9a2823ba111289a42e7f882c5f4ba02c73175106ff251fe5864a43"
)

FILES_SERVER_HOST = "10.100.34.118"
FILES_SERVER_PORT = 8081
MAX_JSON_ARTIFACT_BYTES = 16 * 1024 * 1024
QUEUE_IDS = {
    "GPU4-A100": "9350f33af13a448da8339eb7bea52fdf",
    "GPU4-V100": "3925e906ce484620a941e6ccedc4bdbd",
    "GPU4-5090": "5a84454c072349069e7b61af38637c6d",
}
FORMAL_CONTROLLER_TASK_ID = "1011e98e10f64c428880af1d4b1d542b"
FORMAL_CORE_EVALUATIONS = (
    {
        "subject": "ffnet",
        "task_id": "144397bfa9c242bc9a92a1279922558b",
        "index": 11,
        "queue": "GPU4-V100",
    },
    {
        "subject": "coformernet",
        "task_id": "d8fac86325e047e5aa25f5ce899902b6",
        "index": 6,
        "queue": "GPU4-A100",
    },
    {
        "subject": "v2x_vit",
        "task_id": "cb2675d7e0e845268420f5c0d5248ece",
        "index": 13,
        "queue": "GPU4-A100",
    },
    {
        "subject": "cobevt",
        "task_id": "23f4d7f082284aa2be098a90f0a596e8",
        "index": 14,
        "queue": "GPU4-A100",
    },
    {
        "subject": "bevfusion",
        "task_id": "c3b87760b78742cb8e7de3a506a08999",
        "index": 12,
        "queue": "GPU4-5090",
    },
    {
        "subject": "resilient_v2x",
        "task_id": "7deb18e532324850bee1fb4279a838b7",
        "index": 26,
        "queue": "GPU4-A100",
    },
)
AP_METRIC_KEYS = (
    "resilient_v2x/car_bev_ap_r40_0.50",
    "resilient_v2x/car_bev_ap_r40_0.70",
    "resilient_v2x/car_3d_ap_r40_0.50",
    "resilient_v2x/car_3d_ap_r40_0.70",
)
COUNT_METRICS = {
    "resilient_v2x/sample_count": SAMPLE_COUNT,
    "resilient_v2x/car_ground_truth_count": GROUND_TRUTH_COUNT,
    "resilient_v2x/unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
}
FAILED_STATUSES = frozenset(
    {"failed", "stopped", "closed", "published", "publishing", "rejected", "unknown"}
)
WAITING_STATUSES = frozenset({"created", "queued", "in_progress"})
ACTIVE_EVALUATION_STATUSES = frozenset({"queued", "in_progress"})
RESOURCE_BLOCKING_STATUSES = frozenset({"queued", "in_progress"})
MAX_ACTIVE_CANDIDATE_EVALUATIONS = 3

_CLEARML_ID = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
WORKER_GPU_ID_PATTERN = re.compile(
    r"(?P<host>[^:]+):gpu(?P<gpu>[0-9]+(?:,[0-9]+)*)$"
)


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    subject: str
    training_task_id: str
    template_task_id: str
    queue: str
    config_path: str
    config_sha256: str
    training_script_sha256: str
    source_dataset_id: str
    source_archive_name: str
    source_archive_bytes: int
    source_archive_sha256: str
    source_transition_artifact: str
    source_transition_artifact_sha256: str
    source_transition_artifact_bytes: int
    launch_receipt_artifact: str
    launch_receipt_seal_sha256: str


_E123_SOURCE = {
    "source_dataset_id": "c9ca3075434e44e29951e4aca36e9046",
    "source_archive_name": "resilient-v2x-source-8a22d6d600a5.tar.zst",
    "source_archive_bytes": 1_222_568,
    "source_archive_sha256": (
        "a249546f16f236d42c8346a33b137dea27c037bf0df73c643da4132c8e589557"
    ),
    "source_transition_artifact": "sota_candidate_source_transition",
    "source_transition_artifact_sha256": (
        "f6d7dbda06e8bdff7943d31511407de75ce9a7ae9b14997707a291f35abde6b2"
    ),
    "source_transition_artifact_bytes": 2_744,
    "launch_receipt_artifact": "sota_candidate_fastlane_launch_receipt",
    "launch_receipt_seal_sha256": (
        "16d119a4260824b1f870d56722e25c4de14c1b3b2ff1f41d132401e111b13492"
    ),
}

CANDIDATES = (
    CandidateSpec(
        label="E1",
        subject="support_residual_linear",
        training_task_id="f0c3082f3aa34a81805903e0ffdc8610",
        template_task_id="4b8d48c8a64b4391bbef8969d1b39e3e",
        queue="GPU4-5090",
        config_path="configs/resilient_v2x/improvements/support_residual_linear.py",
        config_sha256=(
            "a105124cf16a693a8c6fb176e07720abe0b05a6ff9d187b076b83d46c30bb4c0"
        ),
        training_script_sha256=(
            "84825bfbcb63bd283be9a2703f019471393023279fed9e0bbc9c079d93c360bc"
        ),
        **_E123_SOURCE,
    ),
    CandidateSpec(
        label="E2",
        subject="no_reliability_linear",
        training_task_id="969c8fce6d24446299561772b3955274",
        template_task_id="4b8d48c8a64b4391bbef8969d1b39e3e",
        queue="GPU4-V100",
        config_path="configs/resilient_v2x/improvements/no_reliability_linear.py",
        config_sha256=(
            "a42f6b28bf1678f663458ccc6f98a409e2483d6786423aa50b2c72e68b721e5d"
        ),
        training_script_sha256=(
            "84825bfbcb63bd283be9a2703f019471393023279fed9e0bbc9c079d93c360bc"
        ),
        **_E123_SOURCE,
    ),
    CandidateSpec(
        label="E3",
        subject="support_residual_no_reliability",
        training_task_id="dc037315c0684c3d854a2fd7c19a2a2f",
        template_task_id="4b8d48c8a64b4391bbef8969d1b39e3e",
        queue="GPU4-A100",
        config_path=(
            "configs/resilient_v2x/improvements/support_residual_no_reliability.py"
        ),
        config_sha256=(
            "86f124aa6ae542891575c26d68560f6b4835b9ebe96516fed40074dfd5af1efa"
        ),
        training_script_sha256=(
            "84825bfbcb63bd283be9a2703f019471393023279fed9e0bbc9c079d93c360bc"
        ),
        **_E123_SOURCE,
    ),
    CandidateSpec(
        label="P0",
        subject="support_residual_no_reliability_linear",
        training_task_id="8883c51ced4f4951a45edbaefe6342d4",
        template_task_id="13f87c0fd45d4621b3c93c0ff88702d8",
        queue="GPU4-A100",
        config_path=(
            "configs/resilient_v2x/improvements/"
            "support_residual_no_reliability_linear.py"
        ),
        config_sha256=(
            "e8208673e2633231c5a60d3783e0735aa3fe0ccaede1a386f4e337597414d1fc"
        ),
        training_script_sha256=(
            "430a0c55a3b84393cc6696fc76df51d1c90f287963c655ccae0659d60ada3582"
        ),
        source_dataset_id="ca2ef9dd8a984df6b05693fb02e89f34",
        source_archive_name="resilient-v2x-source-bff6f84c0e49.tar.zst",
        source_archive_bytes=1_223_924,
        source_archive_sha256=(
            "8a33214c68b956730da1ab6664a008a50f8a6163cd20f299ac21f2b9158536c3"
        ),
        source_transition_artifact="sota_round2_p0_source_transition",
        source_transition_artifact_sha256=(
            "ca3d46d17efe35a61613f514a78616b373e779a8102f27b2a204d2c8d88f2108"
        ),
        source_transition_artifact_bytes=2_475,
        launch_receipt_artifact="sota_round2_p0_launch_receipt",
        launch_receipt_seal_sha256=(
            "c6e59472d3103eea02ce20e4de09e973761f573f67ab0e9b85309f9c79252b3c"
        ),
    ),
    CandidateSpec(
        label="P2",
        subject="support_residual_no_reliability_linear_bbox25",
        training_task_id="f5d3820b4cdf416183c8f1fee566abe3",
        template_task_id="ac35b8f7403c41e1a8c50137abded09b",
        queue="GPU4-A100",
        config_path=(
            "configs/resilient_v2x/improvements/"
            "support_residual_no_reliability_linear_bbox25.py"
        ),
        config_sha256=(
            "eb9e47fde3709b8e04a2b827ea458834fc8db31584369c4c84e2c28403359a88"
        ),
        training_script_sha256=(
            "faa58279f3a81d09a207f42d2c80f90ed328eec367ee8f89410e298fce958995"
        ),
        source_dataset_id="858238049cad4d13918384bb8faec630",
        source_archive_name="resilient-v2x-source-25d1a9a1b67f.tar.zst",
        source_archive_bytes=1_223_663,
        source_archive_sha256=(
            "54e0cf371a6c8b68b041b84f8568efa72c432f0d38c9dfce89f005b2605ee1b3"
        ),
        source_transition_artifact="sota_round2_p2_source_transition",
        source_transition_artifact_sha256=(
            "8b6df34c9aaca561e36a28559c2d2cc3c23a585418949ec9fb170ca4cec760d6"
        ),
        source_transition_artifact_bytes=5_235,
        launch_receipt_artifact="sota_round2_p2_launch_receipt",
        launch_receipt_seal_sha256=(
            "c031ed8e65487aedad3d4c3e721686fdb1949d8199b7ac2a144a642871ac38d8"
        ),
    ),
)
CANDIDATE_ORDER = tuple(spec.subject for spec in CANDIDATES)


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is outside canonical JSON: {error}") from error


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = hashlib.sha256(
        _canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def _require_seal(value: Mapping[str, object], *, context: str) -> str:
    observed = _sha256(value.get("seal_sha256"), f"{context} seal")
    if _sealed(value)["seal_sha256"] != observed:
        raise RuntimeError(f"{context} seal mismatch")
    return observed


def _task_id(value: object, context: str) -> str:
    result = str(value or "")
    if _CLEARML_ID.fullmatch(result) is None:
        raise RuntimeError(f"{context} is not a lowercase ClearML ID")
    return result


def _sha256(value: object, context: str) -> str:
    result = str(value or "")
    if _SHA256.fullmatch(result) is None:
        raise RuntimeError(f"{context} is not a lowercase SHA-256")
    return result


def _status(task: object) -> str:
    reloader = getattr(task, "reload", None)
    if callable(reloader):
        reloader()
    value = getattr(task, "status", "")
    if callable(value):
        value = value()
    return str(getattr(value, "value", value) or "").rsplit(".", 1)[-1].lower()


def _task_parent(task: object) -> str:
    value = getattr(task, "parent", "")
    if callable(value):
        value = value()
    if value in {None, ""}:
        value = getattr(getattr(task, "data", None), "parent", "")
    return str(value or "")


def _task_project_id(task: object, *, context: str) -> str:
    values = {
        str(value)
        for value in (
            getattr(task, "project", None),
            getattr(getattr(task, "data", None), "project", None),
            getattr(getattr(task, "_data", None), "project", None),
        )
        if value not in {None, ""}
    }
    if values != {PROJECT_ID}:
        raise RuntimeError(f"{context} project ID drifted")
    return PROJECT_ID


def _parameters(task: object) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise RuntimeError("task cannot expose parameters")
    try:
        value = getter(backwards_compatibility=False, cast=False)
    except TypeError:
        value = getter(cast=False)
    if not isinstance(value, Mapping):
        raise RuntimeError("task returned invalid parameters")
    return {str(key): item for key, item in value.items()}


def _script(task: object) -> dict[str, object]:
    value = getattr(getattr(task, "data", None), "script", None)
    converter = getattr(value, "to_dict", None)
    value = converter() if callable(converter) else value
    if not isinstance(value, Mapping):
        raise RuntimeError("task cannot expose script metadata")
    return {str(key): item for key, item in value.items()}


def _script_sha256(task: object, *, context: str) -> str:
    script = _script(task)
    if (
        str(script.get("repository") or "") != ""
        or str(script.get("working_dir") or "") != "."
        or str(script.get("entry_point") or "") != TEMPLATE_ENTRY_POINT
    ):
        raise RuntimeError(f"{context} script identity mismatch")
    diff = str(script.get("diff") or "")
    if not diff:
        raise RuntimeError(f"{context} standalone script is empty")
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def _matches(actual: object, expected: object) -> bool:
    if type(expected) is bool:
        return str(actual).casefold() == str(expected).casefold()
    if type(expected) is int:
        return str(actual) == str(expected)
    return actual == expected


def _require_parameters(
    actual: Mapping[str, object], expected: Mapping[str, object], *, context: str
) -> None:
    for key, value in expected.items():
        if key not in actual or not _matches(actual[key], value):
            raise RuntimeError(f"{context} parameter {key} mismatch")


def _artifact_records(task: object) -> dict[str, dict[str, object]]:
    values = getattr(getattr(getattr(task, "data", None), "execution", None), "artifacts", None)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeError("task cannot expose authoritative artifact metadata")
    records: dict[str, dict[str, object]] = {}
    for value in values:
        converter = getattr(value, "to_dict", None)
        value = converter() if callable(converter) else value
        if not isinstance(value, Mapping):
            raise RuntimeError("task returned malformed artifact metadata")
        key = str(value.get("key") or "")
        if not key or key in records:
            raise RuntimeError("task returned duplicate or unnamed artifacts")
        records[key] = dict(value)
    return records


def _artifact_record(task: object, name: str) -> dict[str, object]:
    records = _artifact_records(task)
    if name not in records:
        raise RuntimeError(f"task lacks required artifact {name!r}")
    record = records[name]
    _sha256(record.get("hash"), f"artifact {name} hash")
    size = record.get("content_size")
    if type(size) is not int or size <= 0:
        raise RuntimeError(f"artifact {name} has invalid size")
    uri = str(record.get("uri") or "")
    parsed = urlsplit(uri)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != FILES_SERVER_HOST
        or parsed.port != FILES_SERVER_PORT
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeError(f"artifact {name} has an untrusted URI")
    return record


def _artifact_payload(task: object, name: str) -> dict[str, object]:
    record = _artifact_record(task, name)
    type_data = record.get("type_data")
    if isinstance(type_data, Mapping) and type(type_data.get("preview")) is str:
        try:
            value = json.loads(type_data["preview"])
        except json.JSONDecodeError:
            value = None
        if isinstance(value, Mapping):
            return dict(value)
    artifacts = getattr(task, "artifacts", None)
    artifact = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    getter = getattr(artifact, "get", None)
    if not callable(getter):
        raise RuntimeError(f"artifact {name} has no readable JSON payload")
    value = getter()
    if isinstance(value, Mapping):
        return dict(value)
    path_value: object = value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 1:
            raise RuntimeError(f"artifact {name} did not resolve to one JSON file")
        path_value = value[0]
    try:
        path_value = os.fspath(path_value)
    except TypeError:
        raise RuntimeError(f"artifact {name} is not a JSON object or file")
    path = Path(path_value)
    try:
        before = path.lstat()
    except OSError as error:
        raise RuntimeError(f"artifact {name} JSON file is unavailable") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > MAX_JSON_ARTIFACT_BYTES
    ):
        raise RuntimeError(f"artifact {name} JSON file size/type drifted")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise RuntimeError(f"artifact {name} JSON file cannot be opened") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size != before.st_size
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise RuntimeError(f"artifact {name} JSON file changed before read")
        raw = b""
        while len(raw) <= MAX_JSON_ARTIFACT_BYTES:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_JSON_ARTIFACT_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.lstat()
    if (
        len(raw) != before.st_size
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or current.st_dev != before.st_dev
        or current.st_ino != before.st_ino
        or current.st_size != before.st_size
        or hashlib.sha256(raw).hexdigest() != record["hash"]
    ):
        raise RuntimeError(f"artifact {name} JSON file changed during read")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"artifact {name} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"artifact {name} is not a JSON object")
    return dict(value)


def _artifact_object(task: object, name: str) -> object:
    _artifact_record(task, name)
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or name not in artifacts:
        raise RuntimeError(f"task cannot expose artifact object {name!r}")
    return artifacts[name]


def _validate_template(template: object, spec: CandidateSpec) -> None:
    if _task_id(getattr(template, "id", ""), "template task") != spec.template_task_id:
        raise RuntimeError(f"{spec.label} template ID drifted")
    if _status(template) != "completed":
        raise RuntimeError(f"{spec.label} template is not completed")
    _task_project_id(template, context=f"{spec.label} template")
    if _task_parent(template) != TEMPLATE_PARENT_TASK_ID:
        raise RuntimeError(f"{spec.label} template parent drifted")
    if _script_sha256(template, context=f"{spec.label} template") != TEMPLATE_SCRIPT_SHA256:
        raise RuntimeError(f"{spec.label} template script drifted")
    _require_parameters(
        _parameters(template),
        {
            "Args/source_dataset_id": spec.source_dataset_id,
            "Args/source_archive_name": spec.source_archive_name,
            "Args/source_archive_bytes": spec.source_archive_bytes,
            "Args/source_archive_sha256": spec.source_archive_sha256,
            "Args/training_dataset_id": TRAINING_DATASET_ID,
            "Args/native_bundle_sha256": NATIVE_BUNDLE_SHA256,
            "Args/build_manifest_sha256": BUILD_MANIFEST_SHA256,
            "Args/gpus": 4,
            "Args/max_epochs": 50,
            "Args/amp": False,
        },
        context=f"{spec.label} template",
    )
    transition = _artifact_record(template, spec.source_transition_artifact)
    if (
        transition["hash"] != spec.source_transition_artifact_sha256
        or transition["content_size"] != spec.source_transition_artifact_bytes
    ):
        raise RuntimeError(f"{spec.label} source transition artifact drifted")


def _validate_launch_receipt(task: object, spec: CandidateSpec) -> None:
    receipt = _artifact_payload(task, spec.launch_receipt_artifact)
    seal = _require_seal(receipt, context=f"{spec.label} launch receipt")
    if seal != spec.launch_receipt_seal_sha256:
        raise RuntimeError(f"{spec.label} launch receipt identity drifted")
    expected_common = {
        "source_dataset_id": spec.source_dataset_id,
        "template_task_id": spec.template_task_id,
    }
    for key, expected in expected_common.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"{spec.label} launch receipt {key} drifted")
    if spec.label.startswith("E"):
        tasks = receipt.get("tasks")
        if not isinstance(tasks, list):
            raise RuntimeError(f"{spec.label} launch receipt tasks are invalid")
        matches = [
            row
            for row in tasks
            if isinstance(row, Mapping) and row.get("label") == spec.label
        ]
        if len(matches) != 1:
            raise RuntimeError(f"{spec.label} launch receipt binding is ambiguous")
        row = matches[0]
        expected = {
            "experiment": spec.subject,
            "config": spec.config_path,
            "config_sha256": spec.config_sha256,
            "task_id": spec.training_task_id,
            "parent_task_id": PREDECESSOR_TASK_ID,
            "training_predecessor_task_id": PREDECESSOR_TASK_ID,
            "script_sha256": spec.training_script_sha256,
        }
    else:
        row = receipt
        expected = {
            "task_id": spec.training_task_id,
            "parent_task_id": PREDECESSOR_TASK_ID,
            "patched_script_sha256": spec.training_script_sha256,
            "teacher_task_id": TEACHER_TASK_ID,
            "teacher_model_id": TEACHER_MODEL_ID,
            "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        }
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError(f"{spec.label} launch receipt {key} drifted")


def _validate_run_contract(
    task: object, spec: CandidateSpec
) -> tuple[dict[str, object], str]:
    record = _artifact_record(task, "run_contract")
    contract = _artifact_payload(task, "run_contract")
    expected = {
        "schema_version": 1,
        "mode": "experiment_from_task",
        "task_id": spec.training_task_id,
        "experiment": spec.subject,
        "source_dataset_id": spec.source_dataset_id,
        "training_dataset_id": TRAINING_DATASET_ID,
        "predecessor_task_id": PREDECESSOR_TASK_ID,
        "gpus": 4,
        "global_batch_size": 8,
        "max_epochs": 50,
        "val_interval": 10,
        "precision": "FP32",
        "amp": False,
        "seed": TRAINING_SEED,
        "native_bundle_sha256": NATIVE_BUNDLE_SHA256,
        "build_manifest_sha256": BUILD_MANIFEST_SHA256,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(f"{spec.label} run contract {key} drifted")
    source = contract.get("source_archive")
    if source != {
        "name": spec.source_archive_name,
        "sha256": spec.source_archive_sha256,
        "size_bytes": spec.source_archive_bytes,
    }:
        raise RuntimeError(f"{spec.label} run contract source archive drifted")
    config = contract.get("config")
    if not isinstance(config, Mapping) or any(
        config.get(key) != value
        for key, value in {
            "declared": spec.config_path,
            "config_sha256": spec.config_sha256,
            "resolved_config_sha256": spec.config_sha256,
        }.items()
    ):
        raise RuntimeError(f"{spec.label} run contract config drifted")
    teacher = contract.get("teacher")
    if not isinstance(teacher, Mapping) or any(
        teacher.get(key) != value
        for key, value in {
            "task_id": TEACHER_TASK_ID,
            "model_id": TEACHER_MODEL_ID,
            "sha256": TEACHER_CHECKPOINT_SHA256,
            "expected_sha256": TEACHER_CHECKPOINT_SHA256,
        }.items()
    ):
        raise RuntimeError(f"{spec.label} run contract teacher drifted")
    initialization = contract.get("common_teacher_initialization")
    if not isinstance(initialization, Mapping) or any(
        initialization.get(key) != value
        for key, value in {
            "audit_artifact_name": "common_teacher_initialization_audit",
            "contract": "shared-only-clean-teacher-initialization-v1",
            "expected_nested_teacher": True,
            "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        }.items()
    ):
        raise RuntimeError(f"{spec.label} initialization contract drifted")
    return contract, str(record["hash"])


def _validate_static_training(task: object, spec: CandidateSpec) -> str:
    if _task_id(getattr(task, "id", ""), "training task") != spec.training_task_id:
        raise RuntimeError(f"{spec.label} training task ID drifted")
    status = _status(task)
    if status in FAILED_STATUSES:
        raise RuntimeError(f"{spec.label} training ended as {status!r}")
    if status not in WAITING_STATUSES | {"completed"}:
        raise RuntimeError(f"{spec.label} training has unexpected status {status!r}")
    _task_project_id(task, context=f"{spec.label} training")
    if _task_parent(task) != PREDECESSOR_TASK_ID:
        raise RuntimeError(f"{spec.label} training parent drifted")
    _require_parameters(
        _parameters(task),
        {
            "Args/stage": "all",
            "Args/experiment_from_task": spec.subject,
            "Args/predecessor_task_id": PREDECESSOR_TASK_ID,
            "Args/source_dataset_id": spec.source_dataset_id,
            "Args/source_archive_name": spec.source_archive_name,
            "Args/source_archive_bytes": spec.source_archive_bytes,
            "Args/source_archive_sha256": spec.source_archive_sha256,
            "Args/training_dataset_id": TRAINING_DATASET_ID,
            "Args/teacher_task_id": TEACHER_TASK_ID,
            "Args/teacher_model_id": TEACHER_MODEL_ID,
            "Args/teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
            "Args/training_seed": TRAINING_SEED,
            "Args/gpus": 4,
            "Args/max_epochs": 50,
            "Args/amp": False,
        },
        context=f"{spec.label} training",
    )
    if _script_sha256(task, context=f"{spec.label} training") != spec.training_script_sha256:
        raise RuntimeError(f"{spec.label} training script drifted")
    _validate_launch_receipt(task, spec)
    _validate_run_contract(task, spec)
    return status


def _model_inventory(task: object, kind: str) -> list[object]:
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise RuntimeError("task cannot expose models")
    models = getter()
    values = models.get(kind) if isinstance(models, Mapping) else getattr(models, kind, None)
    if values is None or isinstance(values, (str, bytes, Mapping)):
        raise RuntimeError("task returned invalid model inventory")
    try:
        return list(values)
    except TypeError as error:
        raise RuntimeError("task returned invalid model inventory") from error


def _validate_completed_training(task: object, spec: CandidateSpec) -> dict[str, object]:
    if _validate_static_training(task, spec) != "completed":
        raise RuntimeError(f"{spec.label} training is not completed")
    audit = _artifact_record(task, "common_teacher_initialization_audit")
    final_record = _artifact_record(task, "final_checkpoint_contract")
    final = _artifact_payload(task, "final_checkpoint_contract")
    if final.get("filename") != "epoch_50.pth":
        raise RuntimeError(f"{spec.label} final checkpoint is not epoch_50.pth")
    if "best" in str(final.get("filename") or "").casefold():
        raise RuntimeError(f"{spec.label} clean-best checkpoint is forbidden")
    model_name = f"ResilientV2X {spec.subject} final checkpoint"
    outputs = [
        model
        for model in _model_inventory(task, "output")
        if str(getattr(model, "name", "") or "") == model_name
        and str(getattr(model, "task", "") or "") == spec.training_task_id
    ]
    if len(outputs) != 1:
        raise RuntimeError(f"{spec.label} must expose exactly one matching final model")
    model = outputs[0]
    model_id = _task_id(getattr(model, "id", ""), f"{spec.label} final model")
    model_url = str(getattr(model, "url", "") or "")
    parsed = urlsplit(model_url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != FILES_SERVER_HOST
        or parsed.port != FILES_SERVER_PORT
        or PurePosixPath(unquote(parsed.path)).name != f"{spec.subject}_epoch_50.pth"
    ):
        raise RuntimeError(f"{spec.label} final model URL drifted")
    checkpoint_sha = _sha256(final.get("sha256"), f"{spec.label} final checkpoint")
    if any(
        final.get(key) != value
        for key, value in {
            "model_id": model_id,
            "name": model_name,
            "url": model_url,
        }.items()
    ):
        raise RuntimeError(f"{spec.label} final checkpoint binding drifted")
    size = final.get("size_bytes")
    if type(size) is not int or size <= 0:
        raise RuntimeError(f"{spec.label} final checkpoint size is invalid")
    return {
        "model_id": model_id,
        "model_name": model_name,
        "model_url": model_url,
        "checkpoint_filename": "epoch_50.pth",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": size,
        "run_contract_artifact_sha256": _artifact_record(task, "run_contract")["hash"],
        "initialization_audit_artifact_sha256": audit["hash"],
        "initialization_audit_artifact_bytes": audit["content_size"],
        "final_checkpoint_contract_artifact_sha256": final_record["hash"],
    }


_EVALUATOR_SUBJECT_ANCHOR = '''IMPROVEMENTS = (
    "support_residual",
    "linear_no_distillation",
    "no_distillation_peak_lr_3e4",
)
'''
_EXECUTION_ANCHOR = (
    "    evaluator = _apply_controlled_evaluator_headless_compatibility(source_root)\n"
    "    command = [\n"
)
_HELPER_ANCHOR = "def _execute_controlled_baseline_validation(\n"
_GPU_PREFLIGHT_ANCHOR = (
    "def _capture_gpu_runtime() -> dict[str, object]:\n"
    "    import torch\n"
)
_GPU_PREFLIGHT_REPLACEMENT = '''def _capture_gpu_runtime() -> dict[str, object]:
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [
        [value.strip() for value in line.split(",")]
        for line in gpu_result.stdout.splitlines()
        if line.strip()
    ]
    if len(rows) not in {4, 8} or any(len(row) != 5 for row in rows):
        raise RuntimeError("GPU memory preflight inventory is invalid")
    gpus = []
    for row in rows:
        try:
            index = int(row[0])
            total_mib, used_mib, free_mib = map(int, row[2:])
        except ValueError as error:
            raise RuntimeError("GPU memory preflight values are invalid") from error
        if (
            index < 0
            or total_mib <= 0
            or used_mib < 0
            or free_mib < 0
            or used_mib > 1024
            or free_mib < total_mib - 1024
        ):
            raise RuntimeError("assigned GPUs are not idle")
        gpus.append(
            {
                "index": index,
                "uuid": row[1],
                "total_mib": total_mib,
                "used_mib": used_mib,
                "free_mib": free_mib,
            }
        )
    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    if any(line.strip() for line in process_result.stdout.splitlines()):
        raise RuntimeError("assigned GPUs already have compute processes")
    print(
        json.dumps(
            {
                "event": "gpu_memory_preflight_pass",
                "policy": "no_compute_process_and_at_most_1024_MiB_used_per_gpu",
                "gpu_count": len(gpus),
                "gpus": gpus,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    import torch
'''


def _inject_gpu_memory_preflight(source: str, *, context: str) -> str:
    if source.count(_GPU_PREFLIGHT_ANCHOR) != 1:
        raise RuntimeError(f"{context} GPU preflight anchor drifted")
    return source.replace(
        _GPU_PREFLIGHT_ANCHOR,
        _GPU_PREFLIGHT_REPLACEMENT,
        1,
    )


def _candidate_script_patch(spec: CandidateSpec, template_diff: str) -> str:
    if hashlib.sha256(template_diff.encode("utf-8")).hexdigest() != TEMPLATE_SCRIPT_SHA256:
        raise RuntimeError(f"{spec.label} template bytes are not sealed")
    if template_diff.count(_EXECUTION_ANCHOR) != 1 or template_diff.count(_HELPER_ANCHOR) != 1:
        raise RuntimeError(f"{spec.label} evaluator script patch anchors drifted")
    helper = f'''CANDIDATE_EVALUATION_SUBJECT = {spec.subject!r}
CANDIDATE_EVALUATION_CONFIG = {spec.config_path!r}
CANDIDATE_EVALUATION_CONFIG_SHA256 = {spec.config_sha256!r}
CANDIDATE_SOURCE_EVALUATOR_SHA256 = {SOURCE_EVALUATOR_SHA256!r}


def _apply_formal_candidate_evaluator_subject(target: Path, baseline: str) -> Path:
    if baseline != CANDIDATE_EVALUATION_SUBJECT:
        raise RuntimeError("formal candidate evaluator subject drifted")
    target = target.resolve(strict=True)
    original = target.read_bytes()
    observed = hashlib.sha256(original).hexdigest()
    if observed != CANDIDATE_SOURCE_EVALUATOR_SHA256:
        raise RuntimeError("formal candidate source evaluator bytes drifted")
    config = target.parents[2] / CANDIDATE_EVALUATION_CONFIG
    if not config.is_file() or config.is_symlink():
        raise RuntimeError("formal candidate evaluation config is unavailable")
    if hashlib.sha256(config.read_bytes()).hexdigest() != CANDIDATE_EVALUATION_CONFIG_SHA256:
        raise RuntimeError("formal candidate evaluation config bytes drifted")
    text = original.decode("utf-8")
    anchor = {_EVALUATOR_SUBJECT_ANCHOR!r}
    if text.count(anchor) != 1:
        raise RuntimeError("formal candidate evaluator registry anchor drifted")
    replacement = anchor[:-2] + f'    "{{baseline}}",\\n)\\n'
    encoded = text.replace(anchor, replacement, 1).encode("utf-8")
    temporary = target.with_name(target.name + ".formal-candidate.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError("formal candidate evaluator temporary path exists")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
        temporary.chmod(target.stat().st_mode & 0o777)
        os.replace(temporary, target)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return target


'''
    patched = template_diff.replace(_HELPER_ANCHOR, helper + _HELPER_ANCHOR, 1)
    replacement = (
        "    evaluator = _apply_controlled_evaluator_headless_compatibility(source_root)\n"
        "    evaluator = _apply_formal_candidate_evaluator_subject(evaluator, baseline)\n"
        "    command = [\n"
    )
    patched = patched.replace(_EXECUTION_ANCHOR, replacement, 1)
    if patched == template_diff:
        raise RuntimeError(f"{spec.label} evaluation script patch did not apply")
    return _inject_gpu_memory_preflight(
        patched, context=f"{spec.label} evaluation script"
    )


def _evaluation_parameters(
    template: object, spec: CandidateSpec, binding: Mapping[str, object]
) -> dict[str, object]:
    parameters = dict(_parameters(template))
    for key in list(parameters):
        if key in {
            "Args/experiment_from_task",
            "Args/allow_failed_teacher_task",
            "Args/training_seed",
        } or key.startswith("Args/teacher_") or key.startswith("Args/student_"):
            parameters.pop(key, None)
    parameters.update(
        {
            "Args/stage": "baseline_validate",
            "Args/controlled_baseline": spec.subject,
            "Args/controlled_baseline_task_id": spec.training_task_id,
            "Args/controlled_baseline_model_id": binding["model_id"],
            "Args/controlled_baseline_checkpoint_sha256": binding[
                "checkpoint_sha256"
            ],
            "Args/predecessor_task_id": spec.training_task_id,
            "Args/gpus": "4",
            "Args/max_epochs": "50",
            "Args/amp": "False",
        }
    )
    return parameters


_RUNTIME_EMPTY_EVALUATION_PARAMETERS = {
    "Args/allow_failed_teacher_task": "False",
    "Args/experiment_from_task": "",
    "Args/student_checkpoint": "",
    "Args/student_checkpoint_sha256": "",
    "Args/student_model_id": "",
    "Args/student_task_id": "",
    "Args/teacher_checkpoint": "",
    "Args/teacher_checkpoint_sha256": "",
    "Args/teacher_model_id": "",
    "Args/teacher_task_id": "",
}


def _normalized_evaluation_parameters(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Remove only bootstrap defaults materialized after worker execution."""

    result = dict(value)
    for key, expected in _RUNTIME_EMPTY_EVALUATION_PARAMETERS.items():
        if key not in result:
            continue
        if result[key] != expected:
            raise RuntimeError(f"evaluation runtime parameter {key} drifted")
        result.pop(key)
    return result


def _evaluation_name(spec: CandidateSpec) -> str:
    return (
        f"ResilientV2X formal1337 candidate eval {spec.label} {spec.subject} "
        f"[{spec.training_task_id[:12]}]"
    )


def _find_evaluation(task_class: object, spec: CandidateSpec, template: object) -> object | None:
    del template
    query = getattr(task_class, "query_tasks", None)
    getter = getattr(task_class, "get_task", None)
    if not callable(query) or not callable(getter):
        raise RuntimeError(
            "ClearML task class cannot authoritatively inspect queue occupancy "
            "or duplicate lookup"
        )
    name = _evaluation_name(spec)
    values = query(
        task_filter={
            "project": [PROJECT_ID],
            "parent": spec.training_task_id,
        }
    )
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeError("ClearML returned an invalid duplicate lookup result")
    matches = [
        task
        for value in values
        for raw_id in [value.get("id") if isinstance(value, Mapping) else value]
        for task in [getter(task_id=_task_id(raw_id, "candidate evaluation lookup"))]
        if str(getattr(task, "name", "") or "") == name
        and _task_parent(task) == spec.training_task_id
        and _task_project_id(task, context=f"{spec.label} evaluation") == PROJECT_ID
    ]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate formal candidate evaluation for {spec.label}")
    return matches[0] if matches else None


def _query_task_ids_by_status(
    task_class: object, statuses: frozenset[str]
) -> list[str]:
    query = getattr(task_class, "query_tasks", None)
    if not callable(query):
        raise RuntimeError(
            "ClearML task class cannot authoritatively inspect queue occupancy"
        )
    values = query(task_filter={"status": sorted(statuses)})
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeError("ClearML returned an invalid queue occupancy result")
    task_ids: list[str] = []
    for index, value in enumerate(values):
        raw_id = value.get("id") if isinstance(value, Mapping) else value
        task_ids.append(_task_id(raw_id, f"queue occupancy task {index}"))
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("ClearML returned duplicate queue occupancy tasks")
    return task_ids


def _record_value(record: object, key: str) -> object:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def _worker_resource(
    worker_id: object, *, context: str
) -> tuple[str, frozenset[int]]:
    if type(worker_id) is not str:
        raise RuntimeError(f"{context} worker ID is invalid")
    match = WORKER_GPU_ID_PATTERN.fullmatch(worker_id.strip())
    if match is None:
        raise RuntimeError(f"{context} worker ID has no GPU resource identity")
    gpu_ids = tuple(int(value) for value in match.group("gpu").split(","))
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise RuntimeError(f"{context} worker GPU inventory is invalid")
    return match.group("host"), frozenset(gpu_ids)


def _worker_queue_ids(worker: object, *, context: str) -> tuple[str, ...]:
    values = _record_value(worker, "queues")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeError(f"{context} queue inventory is invalid")
    result = tuple(
        _task_id(_record_value(value, "id"), f"{context} queue {index}")
        for index, value in enumerate(values)
    )
    if len(set(result)) != len(result):
        raise RuntimeError(f"{context} queue inventory contains duplicates")
    return result


def _live_worker_rows() -> list[object]:
    try:
        from clearml.backend_api.session.client import APIClient

        rows = list(APIClient().workers.get_all())
    except Exception as error:
        raise RuntimeError(
            "ClearML worker inventory is unavailable for candidate resource gating"
        ) from error
    if not rows:
        raise RuntimeError("ClearML worker inventory is empty")
    return rows


def _resources_overlap(
    left: tuple[str, frozenset[int]], right: tuple[str, frozenset[int]]
) -> bool:
    return left[0] == right[0] and not left[1].isdisjoint(right[1])


def _resource_gate_snapshot(
    task_class: object,
    *,
    candidate_evaluation_task_ids: set[str],
    target_queue_id: str,
    worker_rows: Sequence[object] | None = None,
) -> dict[str, object]:
    """Prove a selected candidate queue still has an auditable idle worker."""

    rows = list(worker_rows) if worker_rows is not None else _live_worker_rows()
    if not rows:
        raise RuntimeError("ClearML worker inventory is empty")
    target_queue_ids = frozenset(QUEUE_IDS.values())
    selected_queue_id = _task_id(target_queue_id, "candidate target queue")
    if selected_queue_id not in target_queue_ids:
        raise ValueError("candidate target queue is unsupported")
    queue_resources: dict[str, set[tuple[str, frozenset[int]]]] = {}
    worker_resources: dict[str, tuple[str, frozenset[int]]] = {}
    queue_workers: dict[str, set[str]] = {}
    for index, worker in enumerate(rows):
        worker_id = _record_value(worker, "id")
        queues = _worker_queue_ids(worker, context=f"worker {index}")
        if not queues:
            continue
        if not WORKER_GPU_ID_PATTERN.fullmatch(str(worker_id or "").strip()):
            if selected_queue_id in queues:
                raise RuntimeError("candidate target queue is served by an invalid worker")
            continue
        resource = _worker_resource(worker_id, context=f"worker {index}")
        normalized_worker_id = str(worker_id).strip()
        if normalized_worker_id in worker_resources:
            raise RuntimeError("ClearML worker inventory contains duplicate workers")
        worker_resources[normalized_worker_id] = resource
        for queue_id in queues:
            queue_resources.setdefault(queue_id, set()).add(resource)
            queue_workers.setdefault(queue_id, set()).add(normalized_worker_id)
    if selected_queue_id not in queue_resources:
        raise RuntimeError("candidate target queue has no auditable live GPU worker")
    target_resources = set(queue_resources[selected_queue_id])
    target_worker_ids = set(queue_workers[selected_queue_id])
    queued_blockers: list[dict[str, object]] = []
    overlapping_active: list[dict[str, object]] = []
    for task_id in _query_task_ids_by_status(task_class, RESOURCE_BLOCKING_STATUSES):
        task = task_class.get_task(task_id=task_id)
        observed_id = _task_id(getattr(task, "id", ""), "queue occupancy task")
        if observed_id != task_id:
            raise RuntimeError("queue occupancy task identity drifted")
        status = _status(task)
        if status not in RESOURCE_BLOCKING_STATUSES:
            continue
        execution = getattr(getattr(task, "data", None), "execution", None)
        queue_id = _task_id(
            getattr(execution, "queue", ""), "queue occupancy execution queue"
        )
        last_worker = str(
            getattr(getattr(task, "data", None), "last_worker", None) or ""
        ).strip()
        if status == "queued":
            if queue_id == selected_queue_id:
                queued_blockers.append(
                    {
                        "task_id": task_id,
                        "status": status,
                        "queue_id": queue_id,
                        "last_worker": last_worker or None,
                        "reasons": ["target_queue_waiting_task"],
                    }
                )
            continue
        queue_can_overlap = any(
            _resources_overlap(resource, target_resource)
            for resource in queue_resources.get(queue_id, set())
            for target_resource in target_resources
        )
        if not last_worker:
            if queue_id == selected_queue_id or queue_can_overlap:
                raise RuntimeError("active queue occupancy task has no last_worker")
            continue
        try:
            active_resource = worker_resources.get(last_worker) or _worker_resource(
                last_worker, context="active queue occupancy task"
            )
        except RuntimeError:
            if queue_id == selected_queue_id or queue_can_overlap:
                raise
            continue
        overlapping_workers = sorted(
            worker_id
            for worker_id in target_worker_ids
            if _resources_overlap(active_resource, worker_resources[worker_id])
        )
        if overlapping_workers:
            overlapping_active.append(
                {
                    "task_id": task_id,
                    "status": status,
                    "queue_id": queue_id,
                    "last_worker": last_worker,
                    "active_resource": (
                        f"{active_resource[0]}:gpu"
                        + ",".join(str(value) for value in sorted(active_resource[1]))
                    ),
                    "overlapping_target_workers": overlapping_workers,
                }
            )
    occupied_target_worker_ids = {
        worker_id
        for item in overlapping_active
        for worker_id in item["overlapping_target_workers"]
    }
    idle_target_worker_ids = sorted(target_worker_ids - occupied_target_worker_ids)
    active_blockers = []
    if not idle_target_worker_ids:
        active_blockers = [
            {
                "task_id": item["task_id"],
                "status": item["status"],
                "queue_id": item["queue_id"],
                "last_worker": item["last_worker"],
                "reasons": ["overlapping_active_worker"],
            }
            for item in overlapping_active
        ]
    blockers = [*queued_blockers, *active_blockers]
    blockers.sort(key=lambda row: (row["queue_id"], row["task_id"]))
    selected_idle_worker = idle_target_worker_ids[0] if idle_target_worker_ids else None
    return {
        "ready": not queued_blockers and selected_idle_worker is not None,
        "policy": (
            "queued_target_queue_blocks_and_in_progress_occupies_physical_worker"
        ),
        "blocking_statuses": sorted(RESOURCE_BLOCKING_STATUSES),
        "target_queue_id": selected_queue_id,
        "target_workers": [
            {
                "worker_id": worker_id,
                "resource": (
                    f"{worker_resources[worker_id][0]}:gpu"
                    + ",".join(
                        str(value)
                        for value in sorted(worker_resources[worker_id][1])
                    )
                ),
            }
            for worker_id in sorted(target_worker_ids)
        ],
        "idle_target_worker_ids": idle_target_worker_ids,
        "selected_idle_target_worker_id": selected_idle_worker,
        "occupied_target_worker_ids": sorted(occupied_target_worker_ids),
        "observed_overlapping_active_tasks": overlapping_active,
        "external_blockers": blockers,
    }


def _validate_formal_core_metrics(task: object, *, subject: str) -> None:
    value = _artifact_payload(task, "controlled_baseline_metrics")
    expected = {
        "schema_version": 1,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "complete": True,
        "planned_run_count": RUN_COUNT,
        "baseline": subject,
        "protocol_id": PROTOCOL_ID,
        "expected_sample_count": SAMPLE_COUNT,
        "expected_ground_truth_count": GROUND_TRUTH_COUNT,
        "expected_unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"formal core {subject} metrics {key} drifted")
    runs = value.get("runs")
    if not isinstance(runs, list) or len(runs) != RUN_COUNT:
        raise RuntimeError(f"formal core {subject} metrics run count drifted")
    expected_matrix = [
        (delay, condition)
        for delay in DELAYS_MS
        for condition in CONDITIONS
    ]
    for index, (run, (delay, condition)) in enumerate(
        zip(runs, expected_matrix, strict=True)
    ):
        if not isinstance(run, Mapping):
            raise RuntimeError(f"formal core {subject} run {index} is invalid")
        for key, expected_value in {
            "delay_ms": delay,
            "condition": condition,
            "sample_count": SAMPLE_COUNT,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        }.items():
            if run.get(key) != expected_value:
                raise RuntimeError(
                    f"formal core {subject} run {index} {key} drifted"
                )
        _sha256(
            run.get("prediction_sha256"),
            f"formal core {subject} run {index} prediction",
        )
        _sha256(
            run.get("prediction_content_sha256"),
            f"formal core {subject} run {index} prediction content",
        )
        metrics = run.get("metrics")
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"formal core {subject} run {index} lacks metrics")
        for key in AP_METRIC_KEYS:
            _require_ap(
                metrics.get(key),
                context=f"formal core {subject} run {index} {key}",
            )


def _formal_core_phase_snapshot(task_class: object) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    ready = True
    for spec in FORMAL_CORE_EVALUATIONS:
        task = task_class.get_task(task_id=spec["task_id"])
        context = f"formal core {spec['subject']} evaluation"
        if _task_id(getattr(task, "id", ""), context) != spec["task_id"]:
            raise RuntimeError(f"{context} ID drifted")
        _task_project_id(task, context=context)
        expected_name = (
            f"ResilientV2X formal1337 eval {spec['index']:02d} "
            f"{spec['subject']} [{FORMAL_CONTROLLER_TASK_ID}]"
        )
        if str(getattr(task, "name", "") or "") != expected_name:
            raise RuntimeError(f"{context} name drifted")
        if _task_parent(task) != FORMAL_CONTROLLER_TASK_ID:
            raise RuntimeError(f"{context} parent drifted")
        status = _status(task)
        if status in FAILED_STATUSES:
            raise RuntimeError(f"{context} ended as {status!r}")
        if status not in WAITING_STATUSES | {"completed"}:
            raise RuntimeError(f"{context} status is invalid")
        execution = getattr(getattr(task, "data", None), "execution", None)
        queue_id = str(getattr(execution, "queue", "") or "")
        expected_queue_id = QUEUE_IDS[spec["queue"]]
        if status in {"queued", "in_progress", "completed"} and (
            queue_id != expected_queue_id
        ):
            raise RuntimeError(f"{context} queue drifted")
        if status == "completed":
            _validate_formal_core_metrics(task, subject=str(spec["subject"]))
        else:
            ready = False
        entries.append(
            {
                "subject": spec["subject"],
                "task_id": spec["task_id"],
                "status": status,
                "queue": spec["queue"],
                "execution_queue_id": queue_id,
            }
        )
    return {
        "policy": "formal_core_six_completed_before_candidate_e1",
        "ready": ready,
        "entries": entries,
    }


def _remove_all_input_models(task: object, *, context: str) -> None:
    models = _model_inventory(task, "input")
    if not models:
        return
    ids = [_task_id(getattr(model, "id", ""), f"{context} input model") for model in models]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{context} inherited duplicate input models")
    remover = getattr(task, "remove_input_models", None)
    if not callable(remover):
        raise RuntimeError(f"{context} cannot remove inherited input models")
    remover(ids)
    if _model_inventory(task, "input"):
        raise RuntimeError(f"{context} retained inherited input models")


def _set_exact_parameters(task: object, expected: Mapping[str, object], *, context: str) -> None:
    setter = getattr(task, "set_parameters", None)
    if not callable(setter):
        raise RuntimeError(f"{context} cannot replace parameters")
    setter(dict(expected))
    if _parameters(task) != dict(expected):
        raise RuntimeError(f"{context} parameter replacement drifted")


def _default_script_editor(task_id: str, diff: str) -> None:
    from clearml.backend_api.session.client import APIClient

    response = APIClient().tasks.edit(task=task_id, script={"diff": diff})
    if response is None or response is False:
        raise RuntimeError("ClearML did not confirm evaluator script edit")


def _bind_final_input_model(task: object, spec: CandidateSpec, binding: Mapping[str, object]) -> None:
    setter = getattr(task, "set_input_model", None)
    if not callable(setter):
        raise RuntimeError(f"{spec.label} evaluation cannot bind its final model")
    setter(
        model_id=str(binding["model_id"]),
        name=f"{spec.subject}_final_checkpoint",
        update_task_design=False,
        update_task_labels=False,
    )


def _validate_input_binding(task: object, spec: CandidateSpec, binding: Mapping[str, object]) -> None:
    inputs = _model_inventory(task, "input")
    observed = [str(getattr(model, "id", "") or "") for model in inputs]
    if observed != [binding["model_id"]]:
        raise RuntimeError(f"{spec.label} evaluation final input model drifted")


def _validate_evaluation_identity(
    task: object,
    template: object,
    spec: CandidateSpec,
    binding: Mapping[str, object],
) -> str:
    _task_project_id(task, context=f"{spec.label} evaluation")
    if _task_parent(task) != spec.training_task_id:
        raise RuntimeError(f"{spec.label} evaluation parent drifted")
    if str(getattr(task, "name", "") or "") != _evaluation_name(spec):
        raise RuntimeError(f"{spec.label} evaluation name drifted")
    status = _status(task)
    if status in FAILED_STATUSES:
        raise RuntimeError(f"{spec.label} evaluation ended as {status!r}")
    if status not in WAITING_STATUSES | {"completed"}:
        raise RuntimeError(f"{spec.label} evaluation status is invalid")
    expected_parameters = _evaluation_parameters(template, spec, binding)
    if _normalized_evaluation_parameters(_parameters(task)) != expected_parameters:
        raise RuntimeError(f"{spec.label} evaluation parameters drifted")
    expected_diff = _candidate_script_patch(spec, str(_script(template)["diff"]))
    expected_script_sha = hashlib.sha256(expected_diff.encode("utf-8")).hexdigest()
    if _script_sha256(task, context=f"{spec.label} evaluation") != expected_script_sha:
        raise RuntimeError(f"{spec.label} evaluation script drifted")
    _validate_input_binding(task, spec, binding)
    execution = getattr(getattr(task, "data", None), "execution", None)
    queue_id = str(getattr(execution, "queue", "") or "")
    if status == "created" and queue_id:
        raise RuntimeError(f"{spec.label} created evaluation already has a queue")
    if status in {"queued", "in_progress", "completed"} and queue_id != QUEUE_IDS[spec.queue]:
        raise RuntimeError(f"{spec.label} evaluation queue drifted")
    return status


def _create_evaluation(
    task_class: object,
    template: object,
    spec: CandidateSpec,
    binding: Mapping[str, object],
    *,
    script_editor: Callable[[str, str], None],
) -> object:
    task = task_class.clone(
        source_task=template,
        name=_evaluation_name(spec),
        parent=spec.training_task_id,
    )
    if _status(task) != "created":
        raise RuntimeError(f"{spec.label} evaluation clone is not created")
    _remove_all_input_models(task, context=f"{spec.label} evaluation clone")
    _set_exact_parameters(
        task,
        _evaluation_parameters(template, spec, binding),
        context=f"{spec.label} evaluation clone",
    )
    patched = _candidate_script_patch(spec, str(_script(template)["diff"]))
    script_editor(_task_id(getattr(task, "id", ""), "evaluation task"), patched)
    reloader = getattr(task, "reload", None)
    if callable(reloader):
        reloader()
    _bind_final_input_model(task, spec, binding)
    if callable(reloader):
        reloader()
    return task


def _enqueue(
    task_class: object,
    task: object,
    spec: CandidateSpec,
    *,
    sleeper: Callable[[float], None],
) -> str:
    response = task_class.enqueue(task=task, queue_name=spec.queue)
    if response is None or response is False:
        raise RuntimeError(f"failed to enqueue {spec.label} evaluation")
    for _ in range(10):
        task = task_class.get_task(
            task_id=_task_id(
                getattr(task, "id", ""), f"{spec.label} evaluation task"
            )
        )
        status = _status(task)
        if status in {"queued", "in_progress", "completed"}:
            return status
        sleeper(1.0)
    raise RuntimeError(f"{spec.label} evaluation enqueue was not authoritative")


def _require_ap(value: object, *, context: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise RuntimeError(f"{context} is not a finite AP value")
    result = float(value)
    if result < 0.0 or result > 100.0:
        raise RuntimeError(f"{context} is outside [0, 100]")
    return result


def _validate_metrics(
    value: Mapping[str, object], spec: CandidateSpec, binding: Mapping[str, object]
) -> list[dict[str, object]]:
    expected_top = {
        "schema_version": 1,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "complete": True,
        "planned_run_count": RUN_COUNT,
        "baseline": spec.subject,
        "protocol_id": PROTOCOL_ID,
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": SAMPLE_IDS_SHA256,
        "expected_sample_count": SAMPLE_COUNT,
        "expected_ground_truth_count": GROUND_TRUTH_COUNT,
        "expected_unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
    }
    for key, expected in expected_top.items():
        if value.get(key) != expected:
            raise RuntimeError(f"{spec.label} metrics {key} drifted")
    checkpoint = str(value.get("checkpoint") or "")
    checkpoint_name = PurePosixPath(checkpoint).name
    expected_checkpoint_name = f"{spec.subject}_epoch_50.pth"
    if checkpoint_name != expected_checkpoint_name and re.fullmatch(
        rf"[0-9a-f]{{32}}\.{re.escape(expected_checkpoint_name)}",
        checkpoint_name,
    ) is None:
        raise RuntimeError(f"{spec.label} metrics checkpoint is not final-only")
    raw_runs = value.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != RUN_COUNT:
        raise RuntimeError(f"{spec.label} metrics run count drifted")
    normalized: list[dict[str, object]] = []
    for index, (run, delay, condition) in enumerate(
        (
            (run, delay, condition)
            for run, (delay, condition) in zip(
                raw_runs,
                (
                    (delay, condition)
                    for delay in DELAYS_MS
                    for condition in CONDITIONS
                ),
                strict=True,
            )
        )
    ):
        if not isinstance(run, Mapping):
            raise RuntimeError(f"{spec.label} metrics run {index} is invalid")
        condition_id = f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        expected = {
            "condition_id": condition_id,
            "delay_ms": delay,
            "condition": condition,
            "sample_count": SAMPLE_COUNT,
            "sample_ids_sha256": SAMPLE_IDS_SHA256,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        }
        for key, expected_value in expected.items():
            if run.get(key) != expected_value:
                raise RuntimeError(f"{spec.label} metrics run {index} {key} drifted")
        prediction_sha = _sha256(
            run.get("prediction_sha256"), f"{spec.label} run {index} prediction"
        )
        prediction_content_sha = _sha256(
            run.get("prediction_content_sha256"),
            f"{spec.label} run {index} prediction content",
        )
        metrics = run.get("metrics")
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"{spec.label} metrics run {index} lacks metrics")
        for key, expected_count in COUNT_METRICS.items():
            raw = metrics.get(key)
            if (
                type(raw) not in {int, float}
                or not math.isfinite(float(raw))
                or not float(raw).is_integer()
                or int(raw) != expected_count
            ):
                raise RuntimeError(f"{spec.label} metrics run {index} {key} drifted")
        ap = {
            key: _require_ap(metrics.get(key), context=f"{spec.label} run {index} {key}")
            for key in AP_METRIC_KEYS
        }
        normalized.append(
            {
                "condition_id": condition_id,
                "delay_ms": delay,
                "condition": condition,
                "agent_scope": "E+R",
                "prediction_sha256": prediction_sha,
                "prediction_content_sha256": prediction_content_sha,
                "metrics": ap,
            }
        )
    return normalized


def _validate_evaluation_plan(
    value: Mapping[str, object], spec: CandidateSpec, binding: Mapping[str, object]
) -> None:
    expected = {
        "schema_version": 1,
        "plan_type": "resilient_v2x_controlled_baseline_evaluation",
        "protocol_id": PROTOCOL_ID,
        "baseline": spec.subject,
        "evaluation_subject_type": "improvement",
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": SAMPLE_IDS_SHA256,
        "expected_sample_count": SAMPLE_COUNT,
        "expected_ground_truth_count": GROUND_TRUTH_COUNT,
        "expected_unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"{spec.label} evaluation plan {key} drifted")
    runs = value.get("runs")
    if not isinstance(runs, list) or len(runs) != RUN_COUNT:
        raise RuntimeError(f"{spec.label} evaluation plan run count drifted")
    for index, (run, delay, condition) in enumerate(
        (row, delay, condition)
        for row, (delay, condition) in zip(
            runs,
            ((delay, condition) for delay in DELAYS_MS for condition in CONDITIONS),
            strict=True,
        )
    ):
        if not isinstance(run, Mapping) or any(
            run.get(key) != expected_value
            for key, expected_value in {
                "condition_id": f"delay_{delay:03d}_{condition.lower().replace('-', '_')}",
                "delay_ms": delay,
                "condition": condition,
                "agent_scope": "E+R",
                "duration_ticks": 1,
            }.items()
        ):
            raise RuntimeError(f"{spec.label} evaluation plan run {index} drifted")


def _read_evidence_archive(artifact: object) -> Path:
    getter = getattr(artifact, "get_local_copy", None)
    if not callable(getter):
        raise RuntimeError("prediction evidence archive cannot be downloaded")
    try:
        value = getter(extract_archive=False, raise_on_error=True, force_download=True)
    except TypeError:
        value = getter()
    if not value:
        raise RuntimeError("prediction evidence archive returned no local path")
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("prediction evidence archive is not a regular file")
    return path


def _validate_prediction_document(
    raw: bytes,
    *,
    expected_sha256: str,
    expected_content_sha256: str,
    context: str,
) -> None:
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError(f"{context} byte SHA-256 drifted")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{context} is not JSON") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} is not a JSON object")
    content_sha = _sha256(value.get("content_sha256"), f"{context} content")
    payload = dict(value)
    payload.pop("content_sha256", None)
    if (
        content_sha != expected_content_sha256
        or hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        != content_sha
    ):
        raise RuntimeError(f"{context} content SHA-256 drifted")
    samples = value.get("samples")
    if value.get("sample_count") != SAMPLE_COUNT or not isinstance(samples, list) or len(samples) != SAMPLE_COUNT:
        raise RuntimeError(f"{context} sample count drifted")
    sample_ids: list[str] = []
    ground_truth_count = 0
    for sample in samples:
        if not isinstance(sample, Mapping) or type(sample.get("sample_id")) is not str:
            raise RuntimeError(f"{context} has an invalid sample")
        sample_ids.append(sample["sample_id"])
        boxes = sample.get("ground_truth_boxes_lidar_bottom_center")
        labels = sample.get("ground_truth_labels")
        if not isinstance(boxes, list) or not isinstance(labels, list) or len(boxes) != len(labels):
            raise RuntimeError(f"{context} has invalid ground-truth arrays")
        ground_truth_count += len(boxes)
    if (
        hashlib.sha256(_canonical_json(sample_ids).encode("utf-8")).hexdigest()
        != SAMPLE_IDS_SHA256
        or ground_truth_count != GROUND_TRUTH_COUNT
    ):
        raise RuntimeError(f"{context} cohort evidence drifted")


def _validate_prediction_archive(artifact: object, runs: Sequence[Mapping[str, object]]) -> str:
    path = _read_evidence_archive(artifact)
    archive_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path, "r", allowZip64=False) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        expected_names = {"evaluation_plan.json", "metrics.json"}
        for delay in DELAYS_MS:
            for condition in CONDITIONS:
                condition_id = f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
                expected_names.update(
                    {
                        f"{condition_id}/resolved_config.py",
                        f"{condition_id}/predictions.json",
                        f"{condition_id}/checkpoint.sha256",
                    }
                )
        observed_names = set(names)
        extras = observed_names - expected_names
        timestamp_evidence: dict[str, set[str]] = {}
        valid_extra_layout = True
        for name in extras:
            match = re.fullmatch(
                r"(delay_[0-9]{3}_(?:full|l_fail|c_fail))/"
                r"([0-9]{8}_[0-9]{6})/([0-9]{8}_[0-9]{6})\.(json|log)",
                name,
            )
            if match is not None and match.group(2) == match.group(3):
                timestamp_evidence.setdefault(match.group(1), set()).add(
                    match.group(4)
                )
                continue
            if re.fullmatch(
                r"delay_000_full/[0-9]{8}_[0-9]{6}/vis_data/config\.py",
                name,
            ) is None:
                valid_extra_layout = False
                break
        expected_condition_ids = {
            f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
            for delay in DELAYS_MS
            for condition in CONDITIONS
        }
        exact_runtime_extras = (
            len(extras) == 25
            and valid_extra_layout
            and set(timestamp_evidence) == expected_condition_ids
            and all(value == {"json", "log"} for value in timestamp_evidence.values())
            and any(name.endswith("/vis_data/config.py") for name in extras)
        )
        if (
            len(names) != len(observed_names)
            or not expected_names.issubset(observed_names)
            or (extras and not exact_runtime_extras)
        ):
            raise RuntimeError("prediction evidence archive inventory drifted")
        for member in members:
            relative = PurePosixPath(member.filename)
            unix_mode = (member.external_attr >> 16) & 0xFFFF
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or member.is_dir()
                or member.file_size <= 0
                or member.file_size > 128 * 1024 * 1024
                or member.compress_size <= 0
                or member.file_size / member.compress_size > 128.0
                or member.extra
                or member.comment
                or not stat.S_ISREG(unix_mode)
            ):
                raise RuntimeError("prediction evidence archive has an unsafe member")
        for run in runs:
            name = f"{run['condition_id']}/predictions.json"
            raw = archive.read(name)
            _validate_prediction_document(
                raw,
                expected_sha256=str(run["prediction_sha256"]),
                expected_content_sha256=str(run["prediction_content_sha256"]),
                context=name,
            )
    return archive_sha


def _validate_completed_evaluation(
    task: object, spec: CandidateSpec, binding: Mapping[str, object]
) -> dict[str, object]:
    records = _artifact_records(task)
    expected_names = {
        "run_contract",
        "evaluation_plan",
        "controlled_baseline_metrics",
        "controlled_baseline_evidence",
    }
    if set(records) != expected_names:
        raise RuntimeError(f"{spec.label} completed evaluation artifact inventory drifted")
    run_contract = _artifact_payload(task, "run_contract")
    run_expected = {
        "schema_version": 1,
        "mode": "baseline_validate",
        "task_id": _task_id(getattr(task, "id", ""), "evaluation task"),
        "baseline": spec.subject,
        "baseline_task_id": spec.training_task_id,
        "predecessor_task_id": spec.training_task_id,
        "training_dataset_id": TRAINING_DATASET_ID,
        "protocol_id": PROTOCOL_ID,
        "expected_sample_count": SAMPLE_COUNT,
        "expected_ground_truth_count": GROUND_TRUTH_COUNT,
        "expected_run_count": RUN_COUNT,
        "expected_manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "expected_overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "expected_sample_ids_sha256": SAMPLE_IDS_SHA256,
    }
    for key, expected in run_expected.items():
        if run_contract.get(key) != expected:
            raise RuntimeError(f"{spec.label} evaluation run contract {key} drifted")
    checkpoint = run_contract.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or any(
        checkpoint.get(key) != value
        for key, value in {
            "task_id": spec.training_task_id,
            "model_id": binding["model_id"],
            "name": binding["model_name"],
            "sha256": binding["checkpoint_sha256"],
            "size_bytes": binding["checkpoint_size_bytes"],
        }.items()
    ):
        raise RuntimeError(f"{spec.label} evaluation checkpoint binding drifted")
    plan = _artifact_payload(task, "evaluation_plan")
    _validate_evaluation_plan(plan, spec, binding)
    metrics = _artifact_payload(task, "controlled_baseline_metrics")
    runs = _validate_metrics(metrics, spec, binding)
    evidence = _artifact_object(task, "controlled_baseline_evidence")
    evidence_record = records["controlled_baseline_evidence"]
    evidence_url = str(evidence_record.get("uri") or "")
    if PurePosixPath(urlsplit(evidence_url).path).suffix.casefold() != ".zip":
        raise RuntimeError(f"{spec.label} prediction evidence is not a ZIP artifact")
    evidence_archive_sha = _validate_prediction_archive(evidence, runs)
    if evidence_record["hash"] != evidence_archive_sha:
        raise RuntimeError(
            f"{spec.label} prediction evidence artifact SHA-256 drifted"
        )
    return {
        "metrics_artifact_sha256": records["controlled_baseline_metrics"]["hash"],
        "evaluation_plan_artifact_sha256": records["evaluation_plan"]["hash"],
        "prediction_evidence_artifact_sha256": evidence_record["hash"],
        "prediction_evidence_archive_sha256": evidence_archive_sha,
        "runs": runs,
    }


def _evidence_fingerprint() -> dict[str, object]:
    value = {
        "protocol_id": PROTOCOL_ID,
        "sample_count": SAMPLE_COUNT,
        "ground_truth_count": GROUND_TRUTH_COUNT,
        "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": SAMPLE_IDS_SHA256,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "run_count": RUN_COUNT,
        "agent_scope": "E+R",
        "checkpoint_policy": CHECKPOINT_POLICY,
        "training_seed": TRAINING_SEED,
    }
    return {
        **value,
        "fingerprint_sha256": hashlib.sha256(
            _canonical_json(value).encode("utf-8")
        ).hexdigest(),
    }


def _manifest(
    controller_task_id: str,
    entries: list[dict[str, object]],
    *,
    formal_core_gate: Mapping[str, object],
) -> dict[str, object]:
    validated = [row for row in entries if row["evidence_status"] == "validated"]
    training_barrier_ready = all(
        row.get("training_status") == "completed"
        and isinstance(row.get("training_final"), Mapping)
        for row in entries
    )
    selector_candidates = {
        str(row["subject"]): row["formal_evidence"]["runs"]
        for row in validated
        if isinstance(row.get("formal_evidence"), Mapping)
    }
    selector_fingerprints = {
        str(row["subject"]): _evidence_fingerprint() for row in validated
    }
    return _sealed(
        {
            "schema_version": 1,
            "document_type": "resilient_v2x_formal_1337_candidate_evaluation_manifest",
            "controller_task_id": controller_task_id,
            "protocol_id": PROTOCOL_ID,
            "training_seed": TRAINING_SEED,
            "training_dataset_id": TRAINING_DATASET_ID,
            "checkpoint_policy": CHECKPOINT_POLICY,
            "sample_count": SAMPLE_COUNT,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
            "sample_ids_sha256": SAMPLE_IDS_SHA256,
            "delays_ms": list(DELAYS_MS),
            "conditions": list(CONDITIONS),
            "run_count_per_candidate": RUN_COUNT,
            "agent_scope": "E+R",
            "candidate_order": list(CANDIDATE_ORDER),
            "candidate_count": len(CANDIDATES),
            "validated_candidate_count": len(validated),
            "all_candidates_validated": len(validated) == len(CANDIDATES),
            "training_release_barrier": {
                "policy": "all_five_completed_and_final_contract_verified",
                "ready": training_barrier_ready,
                "required_labels": [spec.label for spec in CANDIDATES],
            },
            "formal_core_release_barrier": dict(formal_core_gate),
            "entries": entries,
            "sota_selector_input": {
                "candidate_runs_by_subject": selector_candidates,
                "evidence_fingerprints_by_subject": selector_fingerprints,
                "missing_candidates_are_pending": True,
                "provided_invalid_evidence_policy": "fail_closed",
            },
        }
    )


def reconcile_once(
    *,
    task_class: object,
    controller_task: object | None,
    execute: bool,
    script_editor: Callable[[str, str], None] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Reconcile one authoritative snapshot; only ``execute=True`` may write."""

    editor = script_editor or _default_script_editor
    entries: list[dict[str, object]] = []
    controller_id = (
        _task_id(getattr(controller_task, "id", ""), "candidate evaluation controller")
        if controller_task is not None
        else "0" * 32
    )
    snapshots: list[tuple[CandidateSpec, object, object, str, object | None]] = []
    formal_core_gate = _formal_core_phase_snapshot(task_class)
    formal_core_ready = formal_core_gate["ready"] is True
    candidate_evaluation_task_ids: set[str] = set()
    for spec in CANDIDATES:
        training = task_class.get_task(task_id=spec.training_task_id)
        template = task_class.get_task(task_id=spec.template_task_id)
        _validate_template(template, spec)
        training_status = _validate_static_training(training, spec)
        evaluation = _find_evaluation(task_class, spec, template)
        if evaluation is not None:
            candidate_evaluation_task_ids.add(
                _task_id(
                    getattr(evaluation, "id", ""),
                    f"{spec.label} evaluation task",
                )
            )
        snapshots.append((spec, training, template, training_status, evaluation))

    all_training_bindings: dict[str, dict[str, object]] = {}
    all_training_ready = all(
        training_status == "completed"
        for _, _, _, training_status, _ in snapshots
    )
    if all_training_ready:
        for spec, training, _, _, _ in snapshots:
            all_training_bindings[spec.subject] = _validate_completed_training(
                training, spec
            )

    active_count = 0
    for spec, training, template, training_status, evaluation in snapshots:
        if training_status != "completed":
            if evaluation is not None:
                raise RuntimeError(
                    f"{spec.label} evaluation exists before completed training"
                )
            continue
        binding = _validate_completed_training(training, spec)
        if evaluation is None:
            continue
        status = _validate_evaluation_identity(evaluation, template, spec, binding)
        if status != "created" and not formal_core_ready:
            raise RuntimeError(
                f"{spec.label} evaluation was released before formal core six completed"
            )
        if status in ACTIVE_EVALUATION_STATUSES:
            active_count += 1
    if active_count > MAX_ACTIVE_CANDIDATE_EVALUATIONS:
        raise RuntimeError("candidate evaluation concurrency limit exceeded")

    for index, (spec, training, template, training_status, evaluation) in enumerate(
        snapshots, start=1
    ):
        row: dict[str, object] = {
            "index": index,
            "label": spec.label,
            "subject": spec.subject,
            "training_task_id": spec.training_task_id,
            "template_task_id": spec.template_task_id,
            "planned_queue": spec.queue,
            "parent_binding": "evaluation_parent_is_training_task",
            "training_status": training_status,
            "training_final": None,
            "evaluation_task_id": None,
            "evaluation_status": "not_created",
            "evidence_status": "pending_training",
            "formal_evidence": None,
            "release_gate": "waiting_for_training",
        }
        if training_status != "completed":
            entries.append(row)
            continue
        binding = all_training_bindings.get(spec.subject) or (
            _validate_completed_training(training, spec)
        )
        row["training_final"] = binding
        row["evidence_status"] = "pending_evaluation"
        if (
            evaluation is None
            and execute
            and all_training_ready
            and formal_core_ready
        ):
            evaluation = _create_evaluation(
                task_class,
                template,
                spec,
                binding,
                script_editor=editor,
            )
            duplicates = _find_evaluation(task_class, spec, template)
            if duplicates is None or getattr(duplicates, "id", None) != getattr(
                evaluation, "id", None
            ):
                raise RuntimeError(f"{spec.label} post-clone duplicate guard failed")
            candidate_evaluation_task_ids.add(
                _task_id(
                    getattr(evaluation, "id", ""),
                    f"{spec.label} evaluation task",
                )
            )
        if evaluation is None:
            row["release_gate"] = (
                "waiting_for_formal_core"
                if not formal_core_ready
                else "ready_for_creation"
                if all_training_ready
                else "waiting_for_all_training_final_contracts"
            )
            entries.append(row)
            continue
        status = _validate_evaluation_identity(
            evaluation, template, spec, binding
        )
        if status == "created":
            row["release_gate"] = (
                "waiting_for_formal_core"
                if not formal_core_ready
                else "ready_for_resource_gate"
                if all_training_ready
                else "waiting_for_all_training_final_contracts"
            )
            if (
                execute
                and all_training_ready
                and formal_core_ready
            ):
                if active_count >= MAX_ACTIVE_CANDIDATE_EVALUATIONS:
                    row["release_gate"] = "waiting_for_parallel_slot"
                else:
                    resource_gate = _resource_gate_snapshot(
                        task_class,
                        candidate_evaluation_task_ids=candidate_evaluation_task_ids,
                        target_queue_id=QUEUE_IDS[spec.queue],
                    )
                    row["resource_gate"] = resource_gate
                    if resource_gate["ready"] is True:
                        _enqueue(task_class, evaluation, spec, sleeper=sleeper)
                        evaluation = task_class.get_task(task_id=evaluation.id)
                        status = _validate_evaluation_identity(
                            evaluation, template, spec, binding
                        )
                        row["release_gate"] = "released"
                        active_count += 1
                    else:
                        row["release_gate"] = "waiting_for_gpu_resources"
        elif status in ACTIVE_EVALUATION_STATUSES:
            row["release_gate"] = "evaluation_active"
        else:
            row["release_gate"] = "terminal"
        row["evaluation_task_id"] = _task_id(
            getattr(evaluation, "id", ""), f"{spec.label} evaluation task"
        )
        row["evaluation_status"] = status
        if status == "completed":
            row["formal_evidence"] = _validate_completed_evaluation(
                evaluation, spec, binding
            )
            row["evidence_status"] = "validated"
        entries.append(row)
    return _manifest(
        controller_id, entries, formal_core_gate=formal_core_gate
    )


def _publish_manifest(task: object, manifest: Mapping[str, object]) -> None:
    existing = getattr(task, "artifacts", None)
    if isinstance(existing, Mapping) and CONTROLLER_ARTIFACT in existing:
        try:
            current = _artifact_payload(task, CONTROLLER_ARTIFACT)
        except RuntimeError:
            current = None
        if current == dict(manifest):
            return
    uploader = getattr(task, "upload_artifact", None)
    if not callable(uploader) or not uploader(
        CONTROLLER_ARTIFACT,
        artifact_object=dict(manifest),
        wait_on_upload=True,
    ):
        raise RuntimeError("failed to publish candidate evaluation manifest")
    flusher = getattr(task, "flush", None)
    if callable(flusher):
        flusher(wait_for_uploads=True)


def static_plan() -> dict[str, object]:
    return _sealed(
        {
            "schema_version": 1,
            "plan_type": "resilient_v2x_formal_1337_candidate_evaluation_queue",
            "remote_state_changed": False,
            "execute_token": EXECUTE_TOKEN,
            "protocol_id": PROTOCOL_ID,
            "training_seed": TRAINING_SEED,
            "checkpoint_policy": CHECKPOINT_POLICY,
            "candidate_order": list(CANDIDATE_ORDER),
            "release_semantics": (
                "formal_core_six_then_all_five_training_final_barrier_then_"
                "up_to_three_parallel_candidate_evaluations_with_external_gpu_queue_gate"
            ),
            "max_active_candidate_evaluations": MAX_ACTIVE_CANDIDATE_EVALUATIONS,
            "entries": [
                {
                    **asdict(spec),
                    "parent_binding": "evaluation_parent_is_training_task",
                }
                for spec in CANDIDATES
            ],
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-token", default="")
    parser.add_argument("--execute-remotely", action="store_true")
    parser.add_argument("--service-queue", default=DEFAULT_SERVICE_QUEUE)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=168.0)
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Execute one reconciliation snapshot instead of supervising to completion.",
    )
    return parser


def run(args: argparse.Namespace, *, task_class: object = Task) -> dict[str, object]:
    if not args.execute:
        if args.execute_token or args.execute_remotely:
            raise ValueError("execute options require --execute")
        plan = static_plan()
        print(json.dumps(plan, sort_keys=True, indent=2))
        return plan
    if args.execute_token != EXECUTE_TOKEN:
        raise PermissionError(f"exact execute token required: {EXECUTE_TOKEN}")
    if args.poll_seconds <= 0 or args.timeout_hours <= 0:
        raise ValueError("poll interval and timeout must be positive")
    if task_class is None:
        raise RuntimeError("ClearML is required for execute mode")
    controller = task_class.init(
        project_name=PROJECT_NAME,
        task_name=CONTROLLER_NAME,
        task_type=getattr(getattr(task_class, "TaskTypes", None), "service", None),
        reuse_last_task_id=False,
    )
    if args.execute_remotely:
        executor = getattr(controller, "execute_remotely", None)
        if not callable(executor):
            raise RuntimeError("ClearML task cannot execute remotely")
        executor(queue_name=args.service_queue, exit_process=True)
    setter = getattr(controller, "set_tags", None)
    if callable(setter):
        setter(
            [
                "ResilientV2X-suite",
                "formal-candidate-evaluation-queue",
                PROTOCOL_ID,
                "single-seed-20250218",
                "cpu-controller",
            ]
        )
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    while True:
        manifest = reconcile_once(
            task_class=task_class,
            controller_task=controller,
            execute=True,
        )
        _publish_manifest(controller, manifest)
        print(json.dumps(manifest, sort_keys=True), flush=True)
        if args.single_pass or manifest["all_candidates_validated"] is True:
            return manifest
        if time.monotonic() >= deadline:
            raise TimeoutError("candidate evaluation queue timed out")
        time.sleep(args.poll_seconds)


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CANDIDATES",
    "CANDIDATE_ORDER",
    "EXECUTE_TOKEN",
    "_candidate_script_patch",
    "_validate_completed_evaluation",
    "_validate_completed_training",
    "reconcile_once",
    "run",
    "static_plan",
)
