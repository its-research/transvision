#!/usr/bin/env python3
"""Audit the completed formal 26x12 ClearML chain without mutating it."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from urllib.parse import unquote, urlsplit

try:
    from allegroai import Task
    from allegroai.backend_api.services import queues as clearml_queues
    from allegroai.binding.artifacts import Artifact as ClearMLArtifact
except ImportError:
    from clearml import Task
    from clearml.backend_api.services import queues as clearml_queues
    from clearml.binding.artifacts import Artifact as ClearMLArtifact


DEFAULT_PROJECT = "ResilientV2X/Training"
FILES_SERVER_URI = "http://10.100.34.118:8081"
AUDIT_ARTIFACT = "formal_1337_comparability_audit"
TRAINING_MANIFEST_ARTIFACT = "formal_1337_training_manifest"
TRAINING_PROGRESS_ARTIFACT = "post_main_training_progress"
TRAINING_SUMMARY_ARTIFACT = "post_main_training_summary"
TRAINING_PROVENANCE_ARTIFACT = "formal_1337_training_provenance_equivalence"
EVALUATION_PLAN_ARTIFACT = "formal_1337_evaluation_plan"
LEADERBOARD_ARTIFACT = "formal_1337_leaderboard"
RUN_CONTRACT_ARTIFACT = "run_contract"
FINAL_CHECKPOINT_ARTIFACT = "final_checkpoint_contract"
BEST_CHECKPOINT_ARTIFACT = "best_checkpoint_contract"
BASELINE_DRY_RUN_PLAN_ARTIFACT = "baseline_dry_run_plan"
BASELINE_RESOLVED_CONFIG_ARTIFACT = "baseline_resolved_config"
METRICS_ARTIFACT = "controlled_baseline_metrics"
CONTROLLED_EVIDENCE_ARTIFACT = "controlled_baseline_evidence"
CONTROLLED_EVALUATION_PLAN_ARTIFACT = "evaluation_plan"
COMMON_TEACHER_AUDIT_ARTIFACT = "common_teacher_initialization_audit"
MODEL_BYTES_VERIFIER = "tools/resilient_v2x/collect_clearml_formal_models.py"

PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
SOURCE_DATASET_ID = "4f7fac0078a4419a907fec6ff9e306c8"
SOURCE_ARCHIVE_NAME = "resilient-v2x-source-5c984ad49b52.tar.zst"
SOURCE_ARCHIVE_BYTES = 1_222_481
SOURCE_ARCHIVE_SHA256 = (
    "655f33d9b684c26ff4ff577d5c5f3175d4fc4eac40426eb55820e7a8aa8bc40d"
)
SOURCE_TREE_SHA256 = (
    "5c984ad49b5232d7f6d053fb641895283477efcbf2de40b36d9b3f3c6f8e28b6"
)
SOURCE_INVENTORY_BYTES = 101_195
SOURCE_INVENTORY_SHA256 = (
    "bed72cd86438f2ba932edda14052e3cd3d9589ec201d09d18a315e18a2f7cff2"
)
SOURCE_FILE_COUNT = 631
SOURCE_BYTES = 8_926_106
NEW_SOURCE_DATASET_ID = "351feedbbe81481fa31f1e9ae11a3f4e"
NEW_SOURCE_ARCHIVE_NAME = "resilient-v2x-source-ad511d88b731.tar.zst"
NEW_SOURCE_ARCHIVE_BYTES = 1_222_492
NEW_SOURCE_ARCHIVE_SHA256 = (
    "b94a01c2acf2cc456fe9729f7c40e990e6d44b65e789c6fed11989a673f4f6da"
)
NEW_SOURCE_TREE_SHA256 = (
    "ad511d88b731cb45ef2defb873712bdb2a325c648634b66c349fe1c1459510e4"
)
NEW_SOURCE_INVENTORY_BYTES = 101_195
NEW_SOURCE_INVENTORY_SHA256 = (
    "39b1a42af65ad5df935945bd0a4eeac6e7f6e1cfdd1cc608f3dd8a708e9c5ca0"
)
NEW_SOURCE_FILE_COUNT = 631
NEW_SOURCE_BYTES = 8_926_102
SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256 = (
    "29de9700cac66f9998be643e85a8ec646c04ec17fddbb6207bc1438e9e73941b"
)
SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256 = (
    "c3170f4a88b080f9cc7267053f354640dd260687cb2c35a6f7ffed73d69f4154"
)
SOURCE_REVISION_TRANSITION_SEAL_SHA256 = (
    "1c08edf7daeea7676ceb806d68dc51a4dedd5650d54665d675fe45ab750c6b78"
)
SOURCE_REVISION_TARGET_PREDECESSOR_TASK_ID = (
    "21368e8260cc4e5392fe2dbdf116e36f"
)
TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
NATIVE_BUNDLE_SHA256 = (
    "19b8e7f5edc8216d4b43cb17854dccafe4fc9fe46995a88803e6342eeaa22b21"
)
BUILD_MANIFEST_SHA256 = (
    "21c6ab7a6e9a2823ba111289a42e7f882c5f4ba02c73175106ff251fe5864a43"
)
TEACHER_CHECKPOINT_SHA256 = (
    "7516eb82c7d025f49877c97bfc96a28e7a62853056007289fddd196ce2c231fb"
)
DOCKER_COMMAND_SHA256 = (
    "f68e2426a223b700bc58e7bd60ca040773d355323ae32cd70fd61ee2e322569b"
)
TRAINING_SEED = 20_250_218
SAMPLE_COUNT = 1_337
GROUND_TRUTH_COUNT = 11_330
UNSUPPORTED_SAMPLE_COUNT = 0
DELAYS_MS = (0, 100, 200, 300)
CONDITIONS = ("Full", "L-Fail", "C-Fail")
CHECKPOINT_POLICY = "epoch_50_final_only"
RELEASE_SEMANTICS = "formal_manifest_after_full_training_suite_completion"
MANIFEST_CONTENT_SHA256 = (
    "715ac6f7a14225e20327eed0650c55abdc0cb98431830164e84545238099645d"
)
OVERLAY_INDEX_CONTENT_SHA256 = (
    "77bd4585dbb02901f862b8da6aa208a504674b824a3d55cf15005aacbeeeaaff"
)
SAMPLE_IDS_SHA256 = "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"

# These pins are intentionally injectable in tests until the new W/A tasks are
# created.  The training bootstrap policy is per subject and is independently
# recomputed from raw script bytes below.
CONTROLLER_SCRIPT_SHA256 = (
    "22fe485f3eb999c85b464e260a4376bc81a577c15e6a429585e48df06ffd0678"
)
WATCHER_SCRIPT_SHA256 = (
    "be493eae42b0f50204dbbad3d2dd60671d8480226d1c0176bd912bf2a47f2624"
)
LEADERBOARD_SCRIPT_SHA256 = (
    "1da0cf5dd4435c6ae85d5a474b5a67ec8435f50e9878a3d8f0c1d2872356524"
)
LEGACY_TRAINING_SCRIPT_SHA256 = (
    "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
)
CANONICAL_TRAINING_SCRIPT_SHA256 = (
    "5d9584f1212bc021068e2a948bdf151aed97240c9994d5b994249316692f5066"
)
TRAINING_SCRIPT_SHA256 = CANONICAL_TRAINING_SCRIPT_SHA256
EVALUATION_SCRIPT_SHA256 = CANONICAL_TRAINING_SCRIPT_SHA256
TRAINING_CONTROLLER_TASK_ID = "f8c36e508c7d453dadc766207a5b25b2"
SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID = TRAINING_CONTROLLER_TASK_ID
SOURCE_REVISION_SOURCE_TEMPLATE_TASK_ID = "487dab2664a8485fa0cc7c4e2a0c3df8"
SOURCE_REVISION_TARGET_TEMPLATE_TASK_ID = "d377543f6a574449a5d4b28cb9275dbc"
EVALUATION_TEMPLATE_TASK_ID = "8b77a3674dfe405388aae39ef82d06ef"
LEGACY_PARENT_TASK_ID = "6525107e60ae4104a2800731d74ecd4e"
NO_DISTILLATION_PARENT_TASK_ID = "d4b83d9b68704050aeb24a2e34540d8a"
RECOVERY_PARENT_TASK_ID = "bfcf18a3fd484776adcc5efd48a2c95e"
LEGACY_SCRIPT_ALLOWED_SUBJECTS = frozenset({"support_residual", "no_distillation"})
LEGACY_NESTED_TEACHER_EXPERIMENTS = frozenset({"resilient_v2x", "support_residual"})
CANONICAL_NESTED_TEACHER_EXPERIMENTS = frozenset(
    {
        "support_residual",
        "ptf_none",
        "ptf_linear",
        "router_static",
        "router_uniform",
        "no_reliability",
        "no_delay_metadata",
        "concat_capacity_matched",
        "resilient_v2x",
    }
)
PROVENANCE_PRODUCER_ENTRY_POINT = "clearml_formal_training_provenance.py"
PROVENANCE_FORMAL_TAGS = (
    "ResilientV2X-suite",
    "formal-training-provenance-equivalence",
    PROTOCOL_ID,
    "cpu-controller",
)
TRAINING_PROVENANCE_SCRIPT_SHA256 = (
    "981abba51adf5f8c3b78960a14c2200a2fd68260e74273a52984de869825f77a"
)

EXPECTED_FILES_HOST = "10.100.34.118"
EXPECTED_FILES_PORT = 8081
CLEARML_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SEALED_SUCCESSOR_PIN_ARGS = (
    ("evaluation_plan_amendment_producer_task_id", "task_id"),
    ("evaluation_plan_amendment_receipt_seal_sha256", "sha256"),
    ("evaluation_plan_amendment_revised_plan_seal_sha256", "sha256"),
    ("evaluation_plan_amendment_evidence_seal_sha256", "sha256"),
    ("evaluation_plan_amendment_worker_evidence_seal_sha256", "sha256"),
    ("evaluation_plan_amendment_task_ids_sha256", "sha256"),
    ("exact_eval_runtime_recovery_receipt_seal_sha256", "sha256"),
    ("exact_eval_runtime_recovery_attempt_seal_sha256", "sha256"),
    ("exact_eval_runtime_ffnet_source_sha256", "sha256"),
    ("exact_eval_runtime_ffnet_parameters_sha256", "sha256"),
    ("exact_eval_runtime_candidate_source_sha256", "sha256"),
    ("exact_eval_runtime_candidate_parameters_sha256", "sha256"),
)
# Exact authority installed by the fresh unchanged-plan V100 recovery.
ORIGINAL_QUEUE_RUNTIME_PINS = {
    "Args/original_queue_runtime_recovery_receipt_path": (
        "/home/lbin/Desktop/transvision/artifacts/resilient_v2x/"
        "formal-original-queue-recovery/ffnet-recovery-20260812T1822CST.json"
    ),
    "Args/original_queue_runtime_recovery_receipt_seal_sha256": (
        "dc0b9bcb5041144d5a43204c06b1c30a97274c1b48db4f5439a88e44665eb58e"
    ),
    "Args/original_queue_runtime_recovery_attempt_seal_sha256": (
        "f4ca0740a102c180c3e2b696820c9927c779a3e1d850a267c8b7c99bbc946b7b"
    ),
    "Args/original_queue_runtime_ffnet_source_sha256": (
        "982f607cc7c7cc34872c68ce9c662bb7635d66d15bc7e0fc8cfc9ce70d7c3c3c"
    ),
    "Args/original_queue_runtime_ffnet_parameters_sha256": (
        "583df2fe1aba3a2e714e46e0cec4b4bb8c385cc4c35d450cc28d8bab9b3f8741"
    ),
    "Args/original_queue_runtime_ffnet_task_id": "144397bfa9c242bc9a92a1279922558b",
    "Args/original_queue_runtime_ffnet_planned_queue": "GPU4-V100",
    "Args/original_queue_runtime_coformer_task_id": (
        "d8fac86325e047e5aa25f5ce899902b6"
    ),
    "Args/original_queue_runtime_coformer_artifacts_sha256": (
        "a277b668da6a98875c774e2f245960f8021ff3f3c7d09089a226e51fa19c86cb"
    ),
}
# ClearML argparse integration materializes these exact empty defaults at run time.
OUTPUT_RUNTIME_EMPTY_DEFAULT_PARAMETERS = {
    f"Args/{name}": "" for name, _kind in SEALED_SUCCESSOR_PIN_ARGS
}
SAFE_FILENAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
MAX_JSON_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_SNAPSHOT_ARTIFACT_BYTES = 16 * 1024 * 1024 * 1024
MAX_SNAPSHOT_DIRECTORY_ENTRIES = 1_000_000
PRODUCER_ENTRY_POINT = "clearml_formal_comparability_audit.py"
FORMAL_TAGS = (
    "ResilientV2X-suite",
    "formal-1337-comparability-audit",
    PROTOCOL_ID,
    "cpu-controller",
)
# This is a stable, evidence-bearing projection of a raw ClearML task record.
# Volatile timestamps/metrics are deliberately excluded: artifact publication
# and tag commits update them even when the formal contract is unchanged.
RAW_AUTHORITY_FIELDS = (
    "id",
    "name",
    "user",
    "company",
    "type",
    "status",
    "comment",
    "parent",
    "project",
    "output",
    "execution",
    "script",
    "tags",
    "system_tags",
    "status_message",
    "status_reason",
    "last_worker",
    "hyperparams",
    "configuration",
)
WAITING_STATUSES = frozenset({"created", "queued", "in_progress"})
FAILED_STATUSES = frozenset(
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
SUPPORTED_QUEUES = frozenset({"GPU4-A100", "GPU4-V100", "GPU4-5090"})

SUBJECT_ORDER = (
    "support_residual",
    "ptf_none",
    "ptf_linear",
    "router_static",
    "no_distillation",
    "coformernet",
    "router_uniform",
    "no_reliability",
    "no_delay_metadata",
    "concat_capacity_matched",
    "ffnet",
    "bevfusion",
    "v2x_vit",
    "cobevt",
    "linear_no_distillation",
    "no_distillation_peak_lr_3e4",
    "ego_only",
    "fcooper",
    "attfuse",
    "v2vnet",
    "when2com",
    "where2comm",
    "late_fusion",
    "disconet",
    "how2comm",
    "resilient_v2x",
)
NEW_SOURCE_SUBJECTS = frozenset(
    {"where2comm", "late_fusion", "disconet", "how2comm", "resilient_v2x"}
)
SOURCE_REVISION_BY_SUBJECT = {
    subject: ("new" if subject in NEW_SOURCE_SUBJECTS else "old")
    for subject in SUBJECT_ORDER
}
SOURCE_BY_REVISION = {
    "old": {
        "dataset_id": SOURCE_DATASET_ID,
        "archive_name": SOURCE_ARCHIVE_NAME,
        "archive_size_bytes": SOURCE_ARCHIVE_BYTES,
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
        "tree_sha256": SOURCE_TREE_SHA256,
    },
    "new": {
        "dataset_id": NEW_SOURCE_DATASET_ID,
        "archive_name": NEW_SOURCE_ARCHIVE_NAME,
        "archive_size_bytes": NEW_SOURCE_ARCHIVE_BYTES,
        "archive_sha256": NEW_SOURCE_ARCHIVE_SHA256,
        "tree_sha256": NEW_SOURCE_TREE_SHA256,
    },
}
SOURCE_REVISION_CERTIFICATE_BY_TREE = {
    SOURCE_TREE_SHA256: {
        "dataset_id": SOURCE_DATASET_ID,
        "tree_sha256": SOURCE_TREE_SHA256,
        "file_count": SOURCE_FILE_COUNT,
        "source_bytes": SOURCE_BYTES,
        "archive": {
            "name": SOURCE_ARCHIVE_NAME,
            "size_bytes": SOURCE_ARCHIVE_BYTES,
            "sha256": SOURCE_ARCHIVE_SHA256,
        },
        "inventory": {
            "name": "source-inventory.json",
            "size_bytes": SOURCE_INVENTORY_BYTES,
            "sha256": SOURCE_INVENTORY_SHA256,
        },
    },
    NEW_SOURCE_TREE_SHA256: {
        "dataset_id": NEW_SOURCE_DATASET_ID,
        "tree_sha256": NEW_SOURCE_TREE_SHA256,
        "file_count": NEW_SOURCE_FILE_COUNT,
        "source_bytes": NEW_SOURCE_BYTES,
        "archive": {
            "name": NEW_SOURCE_ARCHIVE_NAME,
            "size_bytes": NEW_SOURCE_ARCHIVE_BYTES,
            "sha256": NEW_SOURCE_ARCHIVE_SHA256,
        },
        "inventory": {
            "name": "source-inventory.json",
            "size_bytes": NEW_SOURCE_INVENTORY_BYTES,
            "sha256": NEW_SOURCE_INVENTORY_SHA256,
        },
    },
}
BASELINE_SUBJECTS = (
    "coformernet",
    "ffnet",
    "bevfusion",
    "v2x_vit",
    "cobevt",
    "ego_only",
    "fcooper",
    "attfuse",
    "v2vnet",
    "when2com",
    "where2comm",
    "late_fusion",
    "disconet",
    "how2comm",
)
ABLATION_SUBJECTS = frozenset(
    {
        "ptf_none",
        "ptf_linear",
        "router_static",
        "no_distillation",
        "router_uniform",
        "no_reliability",
        "no_delay_metadata",
        "concat_capacity_matched",
    }
)
IMPROVEMENT_SUBJECTS = frozenset(
    {
        "support_residual",
        "linear_no_distillation",
        "no_distillation_peak_lr_3e4",
    }
)
PRIMARY_SUBJECT = "resilient_v2x"
SUBJECT_KIND = {
    subject: (
        "baseline"
        if subject in BASELINE_SUBJECTS
        else "ablation"
        if subject in ABLATION_SUBJECTS
        else "improvement"
        if subject in IMPROVEMENT_SUBJECTS
        else "primary_method"
    )
    for subject in SUBJECT_ORDER
}
AP_METRIC_KEYS = (
    "resilient_v2x/car_bev_ap_r40_0.50",
    "resilient_v2x/car_bev_ap_r40_0.70",
    "resilient_v2x/car_3d_ap_r40_0.50",
    "resilient_v2x/car_3d_ap_r40_0.70",
)
LEADERSHIP_METRIC = "resilient_v2x/car_bev_ap_r40_0.70"
COUNT_METRICS = {
    "resilient_v2x/sample_count": SAMPLE_COUNT,
    "resilient_v2x/car_ground_truth_count": GROUND_TRUTH_COUNT,
    "resilient_v2x/unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
}


def _training_artifact_inventory(subject: str) -> set[str]:
    result = {
        RUN_CONTRACT_ARTIFACT,
        FINAL_CHECKPOINT_ARTIFACT,
        BEST_CHECKPOINT_ARTIFACT,
        COMMON_TEACHER_AUDIT_ARTIFACT,
    }
    if subject in BASELINE_SUBJECTS:
        result.update(
            {
                BASELINE_DRY_RUN_PLAN_ARTIFACT,
                BASELINE_RESOLVED_CONFIG_ARTIFACT,
            }
        )
    return result


def _evaluation_artifact_inventory() -> set[str]:
    return {
        RUN_CONTRACT_ARTIFACT,
        CONTROLLED_EVALUATION_PLAN_ARTIFACT,
        METRICS_ARTIFACT,
        CONTROLLED_EVIDENCE_ARTIFACT,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-controller-task-id", required=True)
    parser.add_argument("--training-provenance-task-id", required=True)
    parser.add_argument("--watcher-task-id", required=True)
    parser.add_argument("--leaderboard-task-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=720.0)
    for name, _kind in SEALED_SUCCESSOR_PIN_ARGS:
        parser.add_argument(f"--{name.replace('_', '-')}", default="")
    return parser


def _require_json_domain(
    value: object,
    *,
    context: str,
    active: set[int] | None = None,
) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        object_id = id(value)
        active = set() if active is None else active
        if object_id in active:
            raise ValueError(f"{context} contains a cycle")
        active.add(object_id)
        try:
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError(f"{context} contains a non-string object key")
                _require_json_domain(
                    item,
                    context=f"{context}.{key}",
                    active=active,
                )
        finally:
            active.remove(object_id)
        return
    if isinstance(value, (list, tuple)):
        object_id = id(value)
        active = set() if active is None else active
        if object_id in active:
            raise ValueError(f"{context} contains a cycle")
        active.add(object_id)
        try:
            for index, item in enumerate(value):
                _require_json_domain(
                    item,
                    context=f"{context}[{index}]",
                    active=active,
                )
        finally:
            active.remove(object_id)
        return
    raise ValueError(f"{context} contains a non-JSON value")


def _canonical_json(value: object) -> str:
    _require_json_domain(value, context="canonical JSON")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_copy(value: object, *, context: str) -> object:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{context} is outside the strict JSON domain") from error


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


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


def _sealed_successor_pins(args: argparse.Namespace) -> dict[str, str]:
    raw = {
        name: str(getattr(args, name, "") or "")
        for name, _kind in SEALED_SUCCESSOR_PIN_ARGS
    }
    if not any(raw.values()):
        return {}
    if not all(raw.values()):
        raise ValueError("all amended plan and runtime recovery pins are required")
    result: dict[str, str] = {}
    for name, kind in SEALED_SUCCESSOR_PIN_ARGS:
        value = (
            _clearml_id(raw[name], name)
            if kind == "task_id"
            else _sha256(raw[name], name)
        )
        result[f"Args/{name}"] = value
    return result


def _require_seal(value: Mapping[str, object], *, context: str) -> str:
    observed = _sha256(value.get("seal_sha256"), f"{context} seal")
    if _sealed(value)["seal_sha256"] != observed:
        raise ValueError(f"{context} seal SHA-256 mismatch")
    return observed


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], *, context: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{context} keys mismatch; missing={missing!r}, extra={extra!r}"
        )


def _task_data(task: object, *, context: str) -> object:
    data = getattr(task, "data", None)
    if data is None or isinstance(data, (bool, int, float, str, bytes, bytearray)):
        raise RuntimeError(f"{context} has no installed raw server snapshot")
    return data


def _task_id(task: object, *, context: str) -> str:
    return _clearml_id(
        getattr(_task_data(task, context=context), "id", None),
        f"{context} raw task ID",
    )


def _status(task: object, *, context: str) -> str:
    status = getattr(_task_data(task, context=context), "status", None)
    value = getattr(status, "value", status)
    allowed = WAITING_STATUSES | FAILED_STATUSES | {"completed"}
    if type(value) is not str or value not in allowed:
        raise RuntimeError(f"{context} raw status is unavailable")
    return value


def _reload(task: object, *, context: str) -> None:
    expected_id = _clearml_id(getattr(task, "id", None), f"{context} local ID")
    if bool(getattr(task, "_offline_mode", False)):
        raise RuntimeError(f"{context} cannot use offline reload")
    method = getattr(task, "_reload", None)
    if not callable(method):
        raise RuntimeError(f"{context} cannot be server-reloaded")
    has_flag = hasattr(task, "_reload_skip_flag")
    old_flag = getattr(task, "_reload_skip_flag", None)
    try:
        if has_flag:
            setattr(task, "_reload_skip_flag", False)
        snapshot = method()
    except Exception as error:
        raise RuntimeError(f"{context} server reload failed") from error
    finally:
        if has_flag:
            setattr(task, "_reload_skip_flag", old_flag)
    if snapshot is None or isinstance(
        snapshot, (bool, int, float, str, bytes, bytearray)
    ):
        raise RuntimeError(f"{context} server reload returned no snapshot")
    snapshot_id = _clearml_id(
        getattr(snapshot, "id", None), f"{context} server snapshot ID"
    )
    if snapshot_id != expected_id:
        raise RuntimeError(f"{context} server snapshot identity mismatch")
    try:
        setattr(task, "_data", snapshot)
    except Exception as error:
        raise RuntimeError(f"{context} server snapshot cannot be installed") from error
    if (
        getattr(task, "_data", None) is not snapshot
        or getattr(task, "data", None) is not snapshot
    ):
        raise RuntimeError(f"{context} server snapshot installation did not persist")


def _wait_for_completed(
    tasks: Sequence[tuple[str, object]],
    *,
    deadline: float,
    poll_seconds: float,
    monotonic_clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> None:
    while True:
        pending = False
        for context, task in tasks:
            _reload(task, context=context)
            status = _status(task, context=context)
            if status == "completed":
                continue
            if status in FAILED_STATUSES:
                raise RuntimeError(f"{context} ended as {status!r}")
            if status not in WAITING_STATUSES:
                raise RuntimeError(f"{context} has unsupported status {status!r}")
            pending = True
        if not pending:
            return
        if monotonic_clock() >= deadline:
            raise TimeoutError(
                "timed out waiting for formal comparability dependencies"
            )
        sleeper(poll_seconds)


def _server_artifacts(task: object, *, context: str) -> dict[str, object]:
    execution = getattr(_task_data(task, context=context), "execution", None)
    raw_artifacts = getattr(execution, "artifacts", None)
    if raw_artifacts is None:
        raw_artifacts = ()
    if type(raw_artifacts) not in {list, tuple}:
        raise RuntimeError(f"{context} raw server artifacts are invalid")
    result: dict[str, object] = {}
    for raw in raw_artifacts:
        name = getattr(raw, "key", None)
        if type(name) is not str or not name or name in result:
            raise RuntimeError(f"{context} raw server artifact inventory is invalid")
        try:
            result[name] = ClearMLArtifact(raw)
        except Exception as error:
            raise RuntimeError(
                f"{context} raw server artifact {name!r} cannot be materialized"
            ) from error
    return result


def _artifact_names(task: object, *, context: str) -> tuple[str, ...]:
    return tuple(_server_artifacts(task, context=context))


def _require_artifact_inventory(
    task: object,
    expected: set[str],
    *,
    context: str,
) -> None:
    observed_names = _artifact_names(task, context=context)
    observed = set(observed_names)
    if len(observed) != len(observed_names) or observed != expected:
        raise RuntimeError(
            f"{context} artifact inventory mismatch; "
            f"expected={sorted(expected)!r}, observed={sorted(observed)!r}"
        )


def _read_json_path(value: str | Path, *, context: str) -> dict[str, object]:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError(f"{context} secure local-file flags are unavailable")
    path = Path(value)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        before = path.lstat()
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or before.st_nlink != 1
                or opened.st_nlink != 1
                or before.st_dev != opened.st_dev
                or before.st_ino != opened.st_ino
                or opened.st_size > MAX_JSON_ARTIFACT_BYTES
            ):
                raise RuntimeError(f"{context} local artifact path is unsafe")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(1024 * 1024, MAX_JSON_ARTIFACT_BYTES + 1 - total),
                )
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_JSON_ARTIFACT_BYTES:
                    raise RuntimeError(f"{context} local artifact is too large")
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        final_path = path.lstat()
    except RuntimeError:
        raise
    except OSError as error:
        raise RuntimeError(f"{context} local artifact cannot be read") from error
    identities = (
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        (
            final_path.st_dev,
            final_path.st_ino,
            final_path.st_size,
            final_path.st_mtime_ns,
        ),
    )
    if identities[0] != identities[1] or identities[1] != identities[2]:
        raise RuntimeError(f"{context} local artifact changed during read")
    try:
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{context} local artifact is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{context} local artifact is not a JSON object")
    copied = _json_copy(payload, context=context)
    if not isinstance(copied, dict):  # pragma: no cover - guarded above
        raise RuntimeError(f"{context} local artifact is not a JSON object")
    return copied


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )


def _hash_regular_descriptor(
    descriptor: int,
    *,
    before: os.stat_result,
    context: str,
    remaining_bytes: int,
) -> tuple[dict[str, object], int]:
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or before.st_nlink != 1
        or opened.st_nlink != 1
        or before.st_dev != opened.st_dev
        or before.st_ino != opened.st_ino
        or opened.st_size > remaining_bytes
    ):
        raise RuntimeError(f"{context} local artifact file is unsafe")
    digest = hashlib.sha256()
    total = 0
    while True:
        block = os.read(
            descriptor,
            min(1024 * 1024, remaining_bytes + 1 - total),
        )
        if not block:
            break
        total += len(block)
        if total > remaining_bytes:
            raise RuntimeError(f"{context} local artifact content is too large")
        digest.update(block)
    after = os.fstat(descriptor)
    if _stat_signature(opened) != _stat_signature(after) or total != opened.st_size:
        raise RuntimeError(f"{context} local artifact file changed during read")
    return (
        {
            "kind": "file",
            "size_bytes": total,
            "sha256": digest.hexdigest(),
        },
        total,
    )


def _snapshot_regular_path(path: Path, *, context: str) -> dict[str, object]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        before = path.lstat()
        descriptor = os.open(path, flags)
        try:
            snapshot, _used = _hash_regular_descriptor(
                descriptor,
                before=before,
                context=context,
                remaining_bytes=MAX_SNAPSHOT_ARTIFACT_BYTES,
            )
        finally:
            os.close(descriptor)
        final_path = path.lstat()
    except RuntimeError:
        raise
    except OSError as error:
        raise RuntimeError(f"{context} local artifact file cannot be read") from error
    if _stat_signature(before) != _stat_signature(final_path):
        raise RuntimeError(f"{context} local artifact file changed during read")
    return snapshot


def _snapshot_directory_path(path: Path, *, context: str) -> dict[str, object]:
    if not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError(f"{context} secure directory flags are unavailable")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    file_flags |= getattr(os, "O_CLOEXEC", 0)
    entries: list[dict[str, object]] = []
    total_bytes = 0

    def add_entry(entry: dict[str, object]) -> None:
        if len(entries) >= MAX_SNAPSHOT_DIRECTORY_ENTRIES:
            raise RuntimeError(f"{context} local artifact has too many entries")
        entries.append(entry)

    def walk(descriptor: int, prefix: str) -> None:
        nonlocal total_bytes
        directory_before = os.fstat(descriptor)
        try:
            names = os.listdir(descriptor)
        except OSError as error:
            raise RuntimeError(
                f"{context} local artifact directory cannot be listed"
            ) from error
        if any(type(name) is not str or name in {"", ".", ".."} for name in names):
            raise RuntimeError(f"{context} local artifact directory entry is invalid")
        for name in sorted(names):
            relative = f"{prefix}/{name}" if prefix else name
            try:
                before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise RuntimeError(
                    f"{context} local artifact entry cannot be inspected"
                ) from error
            if stat.S_ISREG(before.st_mode):
                try:
                    child = os.open(name, file_flags, dir_fd=descriptor)
                    try:
                        snapshot, used = _hash_regular_descriptor(
                            child,
                            before=before,
                            context=f"{context} entry {relative!r}",
                            remaining_bytes=MAX_SNAPSHOT_ARTIFACT_BYTES - total_bytes,
                        )
                    finally:
                        os.close(child)
                    final = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                except RuntimeError:
                    raise
                except OSError as error:
                    raise RuntimeError(
                        f"{context} local artifact file cannot be read"
                    ) from error
                if _stat_signature(before) != _stat_signature(final):
                    raise RuntimeError(
                        f"{context} local artifact file changed during read"
                    )
                total_bytes += used
                add_entry({"path": relative, **snapshot})
                continue
            if stat.S_ISDIR(before.st_mode):
                try:
                    child = os.open(name, directory_flags, dir_fd=descriptor)
                    opened = os.fstat(child)
                    if (
                        not stat.S_ISDIR(opened.st_mode)
                        or before.st_dev != opened.st_dev
                        or before.st_ino != opened.st_ino
                    ):
                        raise RuntimeError(
                            f"{context} local artifact directory is unsafe"
                        )
                    add_entry({"path": relative, "kind": "directory"})
                    try:
                        walk(child, relative)
                        after = os.fstat(child)
                    finally:
                        os.close(child)
                    final = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                except RuntimeError:
                    raise
                except OSError as error:
                    raise RuntimeError(
                        f"{context} local artifact directory cannot be read"
                    ) from error
                if _stat_signature(opened) != _stat_signature(after) or _stat_signature(
                    before
                ) != _stat_signature(final):
                    raise RuntimeError(
                        f"{context} local artifact directory changed during read"
                    )
                continue
            raise RuntimeError(f"{context} local artifact contains a non-regular entry")
        directory_after = os.fstat(descriptor)
        if _stat_signature(directory_before) != _stat_signature(directory_after):
            raise RuntimeError(
                f"{context} local artifact directory changed during read"
            )

    try:
        root_before = path.lstat()
        root = os.open(path, directory_flags)
        try:
            root_opened = os.fstat(root)
            if (
                not stat.S_ISDIR(root_before.st_mode)
                or not stat.S_ISDIR(root_opened.st_mode)
                or root_before.st_dev != root_opened.st_dev
                or root_before.st_ino != root_opened.st_ino
            ):
                raise RuntimeError(f"{context} local artifact directory is unsafe")
            walk(root, "")
            root_after = os.fstat(root)
        finally:
            os.close(root)
        root_final = path.lstat()
    except RuntimeError:
        raise
    except OSError as error:
        raise RuntimeError(
            f"{context} local artifact directory cannot be read"
        ) from error
    if _stat_signature(root_opened) != _stat_signature(root_after) or _stat_signature(
        root_before
    ) != _stat_signature(root_final):
        raise RuntimeError(f"{context} local artifact directory changed during read")
    return {
        "kind": "directory",
        "entry_count": len(entries),
        "total_size_bytes": total_bytes,
        "entries": entries,
    }


def _snapshot_artifact_value(value: object, *, context: str) -> dict[str, object]:
    if isinstance(value, Mapping):
        canonical = _canonical_json(value)
        if len(canonical.encode("utf-8")) > MAX_JSON_ARTIFACT_BYTES:
            raise RuntimeError(f"{context} JSON artifact is too large")
        copied = json.loads(canonical)
        if not isinstance(copied, dict):  # pragma: no cover - guarded above
            raise RuntimeError(f"{context} artifact is not a JSON object")
        return {"kind": "json", "value": copied}
    if not isinstance(value, (str, Path)):
        raise RuntimeError(f"{context} artifact has an unsupported local value")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError(f"{context} secure local-file flags are unavailable")
    path = Path(value)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise RuntimeError(f"{context} local artifact cannot be inspected") from error
    if stat.S_ISREG(mode):
        return _snapshot_regular_path(path, context=context)
    if stat.S_ISDIR(mode):
        return _snapshot_directory_path(path, context=context)
    raise RuntimeError(f"{context} local artifact path is unsafe")


def _artifact_content_snapshot(
    artifact: object,
    *,
    name: str,
    context: str,
) -> dict[str, object]:
    getter = getattr(artifact, "get", None)
    if not callable(getter):
        raise RuntimeError(f"{context} artifact {name!r} cannot be read")
    try:
        value = getter(force_download=True)
    except Exception as error:
        raise RuntimeError(
            f"{context} artifact {name!r} cannot be force-downloaded"
        ) from error
    return _snapshot_artifact_value(
        value,
        context=f"{context} artifact {name!r}",
    )


def _artifact_mapping(task: object, name: str, *, context: str) -> dict[str, object]:
    artifacts = _server_artifacts(task, context=context)
    if name not in artifacts:
        raise RuntimeError(f"{context} lacks artifact {name!r}")
    getter = getattr(artifacts[name], "get", None)
    if not callable(getter):
        raise RuntimeError(f"{context} artifact {name!r} cannot be read")
    try:
        value = getter(force_download=True)
    except Exception as error:
        raise RuntimeError(
            f"{context} artifact {name!r} cannot be force-downloaded"
        ) from error
    if isinstance(value, Mapping):
        copied = _json_copy(value, context=f"{context} artifact {name!r}")
        if not isinstance(copied, dict):  # pragma: no cover - guarded above
            raise RuntimeError(f"{context} artifact {name!r} is not a JSON object")
        return copied
    if isinstance(value, (str, Path)):
        return _read_json_path(value, context=f"{context} artifact {name!r}")
    raise RuntimeError(f"{context} artifact {name!r} is not a JSON object")


def _parameters(task: object, *, context: str) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise RuntimeError(f"{context} cannot expose parameters")
    try:
        value = getter(cast=False)
    except TypeError:
        value = getter()
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} parameters are invalid")
    copied = _json_copy(value, context=f"{context} parameters")
    if not isinstance(copied, dict):  # pragma: no cover - guarded above
        raise RuntimeError(f"{context} parameters are invalid")
    return copied


def _task_parent(task: object, *, context: str) -> str:
    value = getattr(_task_data(task, context=context), "parent", None)
    if value in {None, ""}:
        return ""
    if type(value) is not str:
        raise RuntimeError(f"{context} raw parent must be a string")
    return _clearml_id(value, f"{context} raw parent")


def _raw_script_source(task: object, *, entry_point: str, context: str) -> str:
    script = getattr(_task_data(task, context=context), "script", None)
    if script is None:
        raise RuntimeError(f"{context} has no script metadata")
    repository = getattr(script, "repository", None)
    working_dir = getattr(script, "working_dir", None)
    observed_entry_point = getattr(script, "entry_point", None)
    diff = getattr(script, "diff", None)
    if type(repository) is not str or repository != "":
        raise RuntimeError(f"{context} is not a standalone script")
    if type(working_dir) is not str or working_dir != ".":
        raise RuntimeError(f"{context} working directory drifted")
    if type(observed_entry_point) is not str or observed_entry_point != entry_point:
        raise RuntimeError(f"{context} entry point drifted")
    if type(diff) is not str or not diff:
        raise RuntimeError(f"{context} standalone source is empty")
    return diff


def _script_sha256(task: object, *, entry_point: str, context: str) -> str:
    source = _raw_script_source(task, entry_point=entry_point, context=context)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _require_training_provenance_script_sha256(observed: str) -> None:
    if observed != TRAINING_PROVENANCE_SCRIPT_SHA256:
        raise RuntimeError("training provenance producer script SHA-256 mismatch")


def _nested_teacher_projection(
    source: str, *, context: str
) -> tuple[str, str, frozenset[str], tuple[str, ...]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise RuntimeError(f"{context} is not valid Python") from error
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "NESTED_TEACHER_EXPERIMENTS"
    ]
    if len(assignments) != 1:
        raise RuntimeError(
            f"{context} must define NESTED_TEACHER_EXPERIMENTS exactly once"
        )
    assignment = assignments[0]
    value = assignment.value
    if (
        not isinstance(value, ast.Call)
        or not isinstance(value.func, ast.Name)
        or value.func.id != "frozenset"
        or len(value.args) != 1
        or value.keywords
        or not isinstance(value.args[0], ast.Set)
    ):
        raise RuntimeError(f"{context} nested-teacher assignment AST drifted")
    members: list[str] = []
    for element in value.args[0].elts:
        if not isinstance(element, ast.Constant) or type(element.value) is not str:
            raise RuntimeError(f"{context} nested-teacher member is not literal")
        members.append(element.value)
    if len(members) != len(set(members)):
        raise RuntimeError(f"{context} nested-teacher membership is duplicated")
    if (
        assignment.lineno is None
        or assignment.end_lineno is None
        or assignment.col_offset != 0
        or assignment.end_col_offset is None
    ):
        raise RuntimeError(f"{context} nested-teacher source coordinates drifted")
    lines = source.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: assignment.lineno - 1])
    end = (
        sum(len(line) for line in lines[: assignment.end_lineno - 1])
        + assignment.end_col_offset
    )
    text_projection = source[:start] + "__NESTED_TEACHER_ASSIGNMENT__" + source[end:]
    normalized = copy.deepcopy(tree)
    normalized_assignments = [
        node
        for node in ast.walk(normalized)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "NESTED_TEACHER_EXPERIMENTS"
    ]
    normalized_assignments[0].value = ast.Constant(value="__NESTED_TEACHER_MEMBERS__")
    ast.fix_missing_locations(normalized)
    ast_projection = ast.dump(
        normalized, annotate_fields=True, include_attributes=False
    )
    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    usage_sites: set[str] = set()
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "NESTED_TEACHER_EXPERIMENTS"
        and isinstance(node.ctx, ast.Load)
    ]
    if len(loads) != 2:
        raise RuntimeError(f"{context} nested-teacher Load inventory drifted")
    for load in loads:
        compare = parents.get(id(load))
        if (
            not isinstance(compare, ast.Compare)
            or len(compare.ops) != 1
            or not isinstance(compare.ops[0], ast.In)
            or len(compare.comparators) != 1
            or compare.comparators[0] is not load
            or not isinstance(compare.left, ast.Attribute)
            or not isinstance(compare.left.value, ast.Name)
            or compare.left.value.id != "spec"
            or compare.left.attr != "name"
        ):
            raise RuntimeError(f"{context} nested-teacher Load semantic drifted")
        container = parents.get(id(compare))
        if (
            isinstance(container, ast.keyword)
            and container.arg == "expect_nested_teacher"
        ):
            usage_sites.add("expect_nested_teacher_keyword")
            continue
        if isinstance(container, ast.Dict):
            matches = [
                key
                for key, item in zip(container.keys, container.values, strict=True)
                if item is compare
            ]
            if (
                len(matches) == 1
                and isinstance(matches[0], ast.Constant)
                and matches[0].value == "expected_nested_teacher"
            ):
                usage_sites.add("expected_nested_teacher_contract_field")
                continue
        raise RuntimeError(f"{context} nested-teacher Load call-site drifted")
    usages = tuple(sorted(usage_sites))
    if set(usages) != {
        "expect_nested_teacher_keyword",
        "expected_nested_teacher_contract_field",
    }:
        raise RuntimeError(f"{context} nested-teacher usage closure drifted")
    return text_projection, ast_projection, frozenset(members), usages


def _verify_training_script_equivalence(
    source_by_sha: Mapping[str, str], script_by_subject: Mapping[str, str]
) -> dict[str, object]:
    expected_shas = {
        LEGACY_TRAINING_SCRIPT_SHA256,
        CANONICAL_TRAINING_SCRIPT_SHA256,
    }
    if set(source_by_sha) != expected_shas:
        raise RuntimeError(
            "training chain does not expose exactly both reviewed scripts"
        )
    if set(script_by_subject) != set(SUBJECT_ORDER) or len(script_by_subject) != len(
        SUBJECT_ORDER
    ):
        raise RuntimeError("training script subject inventory drifted")
    if any(sha not in expected_shas for sha in script_by_subject.values()):
        raise RuntimeError("training script subject binding uses an unreviewed script")
    projections = {
        sha: _nested_teacher_projection(source, context=f"training script {sha}")
        for sha, source in source_by_sha.items()
    }
    legacy = projections[LEGACY_TRAINING_SCRIPT_SHA256]
    canonical = projections[CANONICAL_TRAINING_SCRIPT_SHA256]
    if legacy[0] != canonical[0] or legacy[1] != canonical[1]:
        raise RuntimeError("training scripts differ outside NESTED_TEACHER_EXPERIMENTS")
    if legacy[2] != LEGACY_NESTED_TEACHER_EXPERIMENTS:
        raise RuntimeError("legacy nested-teacher membership drifted")
    if canonical[2] != CANONICAL_NESTED_TEACHER_EXPERIMENTS:
        raise RuntimeError("canonical nested-teacher membership drifted")
    if legacy[3] != canonical[3]:
        raise RuntimeError("nested-teacher runtime usage closure drifted")
    legacy_subjects = [
        subject
        for subject in SUBJECT_ORDER
        if script_by_subject[subject] == LEGACY_TRAINING_SCRIPT_SHA256
    ]
    expected_legacy_subjects = [
        subject
        for subject in SUBJECT_ORDER
        if subject in LEGACY_SCRIPT_ALLOWED_SUBJECTS
    ]
    if legacy_subjects != expected_legacy_subjects:
        raise RuntimeError("legacy script subject inventory drifted")
    for subject in legacy_subjects:
        if (subject in legacy[2]) != (subject in canonical[2]):
            raise RuntimeError(
                f"legacy script changes nested-teacher behavior for {subject}"
            )
    return {
        "legacy_script_sha256": LEGACY_TRAINING_SCRIPT_SHA256,
        "canonical_script_sha256": CANONICAL_TRAINING_SCRIPT_SHA256,
        "legacy_script_subjects": legacy_subjects,
        "only_difference": "NESTED_TEACHER_EXPERIMENTS membership",
        "runtime_usage_closure": list(legacy[3]),
    }


def _fileserver_url(value: object, *, expected_filename: str, context: str) -> str:
    url = str(value or "")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"{context} URL is invalid") from error
    raw_filename = parsed.path.rsplit("/", 1)[-1]
    filename = unquote(raw_filename)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != EXPECTED_FILES_HOST
        or port != EXPECTED_FILES_PORT
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
        or not parsed.path.startswith("/")
        or filename != expected_filename
        or SAFE_FILENAME_PATTERN.fullmatch(filename) is None
    ):
        raise ValueError(f"{context} is not the expected fileserver URL")
    return url


def _parameter_matches(value: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return str(value).casefold() == str(expected).casefold()
    return str(value) == str(expected)


def _require_parameters(
    parameters: Mapping[str, object], expected: Mapping[str, object], *, context: str
) -> None:
    for key, value in expected.items():
        if not _parameter_matches(parameters.get(key), value):
            raise RuntimeError(f"{context} parameter {key} mismatch")


def _validate_manifest(
    payload: Mapping[str, object],
) -> tuple[list[dict[str, object]], str, bool]:
    manifest = dict(payload)
    seal = _require_seal(manifest, context="formal training manifest")
    seed_fields = {"training_seed", "training_overlay_protocol_seed"}
    top_seed_fields = set(manifest) & seed_fields
    if top_seed_fields not in (set(), seed_fields):
        raise ValueError("formal training manifest seed metadata is partial")
    seeded = top_seed_fields == seed_fields
    expected_keys = {
        "schema_version",
        "manifest_type",
        "protocol_id",
        "sample_count",
        "delays_ms",
        "conditions",
        "run_count",
        "checkpoint_policy",
        "evaluation_release_semantics",
        "subject_order",
        "subject_count",
        "entries",
        "seal_sha256",
    } | (seed_fields if seeded else set())
    _require_exact_keys(manifest, expected_keys, context="formal training manifest")
    expected = {
        "schema_version": 1,
        "manifest_type": "resilient_v2x_formal_1337_training_inputs",
        "protocol_id": PROTOCOL_ID,
        "sample_count": SAMPLE_COUNT,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "run_count": len(DELAYS_MS) * len(CONDITIONS),
        "checkpoint_policy": CHECKPOINT_POLICY,
        "evaluation_release_semantics": RELEASE_SEMANTICS,
        "subject_order": list(SUBJECT_ORDER),
        "subject_count": len(SUBJECT_ORDER),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"formal training manifest {key} mismatch")
    if seeded and (
        manifest.get("training_seed") != TRAINING_SEED
        or manifest.get("training_overlay_protocol_seed") != TRAINING_SEED
    ):
        raise ValueError("formal training manifest seed mismatch")
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != len(SUBJECT_ORDER):
        raise ValueError("formal training manifest entry count mismatch")
    entries: list[dict[str, object]] = []
    task_ids: set[str] = set()
    model_ids: set[str] = set()
    entry_seed_fields = seed_fields if seeded else set()
    base_entry_keys = {
        "index",
        "subject",
        "kind",
        "training_task_id",
        "training_predecessor_task_id",
        "model_id",
        "model_name",
        "model_url",
        "checkpoint_filename",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "common_teacher_initialization_audit_artifact",
        "common_teacher_initialization_audit_sha256",
    }
    for index, (raw_entry, subject) in enumerate(
        zip(raw_entries, SUBJECT_ORDER, strict=True), start=1
    ):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"formal training entry {index} is not an object")
        entry = dict(raw_entry)
        _require_exact_keys(
            entry,
            base_entry_keys | entry_seed_fields,
            context=f"formal training entry {subject}",
        )
        if (
            entry.get("index") != index
            or entry.get("subject") != subject
            or entry.get("kind") != SUBJECT_KIND[subject]
        ):
            raise ValueError(f"formal training entry {index} identity mismatch")
        task_id = _clearml_id(entry.get("training_task_id"), f"{subject} task")
        _clearml_id(entry.get("training_predecessor_task_id"), f"{subject} predecessor")
        model_id = _clearml_id(entry.get("model_id"), f"{subject} model")
        if task_id in task_ids or model_id in model_ids:
            raise ValueError("formal training task/model IDs are not unique")
        task_ids.add(task_id)
        model_ids.add(model_id)
        expected_model_name = f"ResilientV2X {subject} final checkpoint"
        if entry.get("model_name") != expected_model_name:
            raise ValueError(f"{subject} final model name mismatch")
        _fileserver_url(
            entry.get("model_url"),
            expected_filename=f"{subject}_epoch_50.pth",
            context=f"{subject} final model URL",
        )
        if entry.get("checkpoint_filename") != "epoch_50.pth":
            raise ValueError(f"{subject} checkpoint filename mismatch")
        _sha256(entry.get("checkpoint_sha256"), f"{subject} checkpoint")
        size = entry.get("checkpoint_size_bytes")
        if type(size) is not int or size <= 0:
            raise ValueError(f"{subject} checkpoint size is invalid")
        if entry.get("common_teacher_initialization_audit_artifact") != (
            "common_teacher_initialization_audit"
        ):
            raise ValueError(f"{subject} initialization audit artifact mismatch")
        _sha256(
            entry.get("common_teacher_initialization_audit_sha256"),
            f"{subject} initialization audit",
        )
        if seeded and (
            entry.get("training_seed") != TRAINING_SEED
            or entry.get("training_overlay_protocol_seed") != TRAINING_SEED
        ):
            raise ValueError(f"{subject} manifest seed mismatch")
        entries.append(entry)
    return entries, seal, seeded


def _validate_summary(
    payload: Mapping[str, object],
    *,
    controller_task_id: str,
    controller_parameters: Mapping[str, object],
    manifest: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
) -> str:
    summary = dict(payload)
    seal = _require_seal(summary, context="training summary")
    expected = {
        "schema_version": 1,
        "summary_type": "resilient_v2x_post_main_sequential_training",
        "status": "completed",
        "controller_task_id": controller_task_id,
        "experiment_order": list(SUBJECT_ORDER),
        "task_count": len(SUBJECT_ORDER),
        "formal_1337_manifest_artifact": TRAINING_MANIFEST_ARTIFACT,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"training summary {key} mismatch")
    gate_task_id = _clearml_id(
        controller_parameters.get("Args/gate_task_id"), "controller quality gate"
    )
    if summary.get("gate_task_id") != gate_task_id:
        raise ValueError("training summary gate task mismatch")
    if summary.get("formal_1337_evaluation_manifest") != manifest:
        raise ValueError("standalone and nested formal training manifests differ")
    template = summary.get("template")
    if not isinstance(template, Mapping):
        raise ValueError("training summary template is missing")
    template_task_id = _clearml_id(
        controller_parameters.get("Args/template_task_id"), "controller template"
    )
    if template.get("task_id") != template_task_id:
        raise ValueError("training summary template task mismatch")
    source_parameters = template.get("source_parameters")
    if not isinstance(source_parameters, Mapping):
        raise ValueError("training summary template source parameters are missing")
    _require_parameters(
        source_parameters,
        {
            "Args/source_dataset_id": NEW_SOURCE_DATASET_ID,
            "Args/source_archive_name": NEW_SOURCE_ARCHIVE_NAME,
            "Args/source_archive_bytes": NEW_SOURCE_ARCHIVE_BYTES,
            "Args/source_archive_sha256": NEW_SOURCE_ARCHIVE_SHA256,
            "Args/training_dataset_id": TRAINING_DATASET_ID,
        },
        context="training summary template",
    )
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != len(entries):
        raise ValueError("training summary result count mismatch")
    for index, (result, entry, subject) in enumerate(
        zip(results, entries, SUBJECT_ORDER, strict=True), start=1
    ):
        if not isinstance(result, Mapping):
            raise ValueError(f"training summary result {index} is invalid")
        expected_result = {
            "index": index,
            "experiment": subject,
            "task_id": entry["training_task_id"],
            "predecessor_task_id": entry["training_predecessor_task_id"],
            "model_id": entry["model_id"],
            "model_name": entry["model_name"],
            "model_url": entry["model_url"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "checkpoint_size_bytes": entry["checkpoint_size_bytes"],
        }
        for key, value in expected_result.items():
            if result.get(key) != value:
                raise ValueError(f"training summary {subject} result {key} mismatch")
    return seal


def _require_unique_output_model(
    task: object, *, entry: Mapping[str, object], context: str
) -> object:
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise RuntimeError(f"{context} cannot expose models")
    models = getter()
    if not isinstance(models, Mapping):
        raise RuntimeError(f"{context} model mapping is invalid")
    outputs = models.get("output")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise RuntimeError(f"{context} output models are invalid")
    candidates = [
        model
        for model in outputs
        if str(getattr(model, "name", "") or "") == str(entry["model_name"])
        and str(getattr(model, "task", "") or "") == str(entry["training_task_id"])
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"{context} must expose exactly one named final model")
    model = candidates[0]
    expected = {
        "id": entry["model_id"],
        "task": entry["training_task_id"],
        "name": entry["model_name"],
        "url": entry["model_url"],
    }
    for attribute, value in expected.items():
        if str(getattr(model, attribute, "") or "") != str(value):
            raise RuntimeError(f"{context} OutputModel {attribute} mismatch")
    return model


def _run_contract_seed(
    contract: Mapping[str, object],
    *,
    subject: str,
    manifest_seeded: bool,
) -> tuple[int, list[str]]:
    seed_fields = {"seed", "training_seed", "training_overlay_protocol_seed"}
    required_fields = seed_fields if manifest_seeded else {"seed"}
    observed_fields = set(contract) & seed_fields
    if observed_fields != required_fields:
        raise RuntimeError(
            f"{subject} run contract seed schema mismatch; "
            f"required={sorted(required_fields)!r}, "
            f"observed={sorted(observed_fields)!r}"
        )
    if any(contract[key] != TRAINING_SEED for key in required_fields):
        raise RuntimeError(f"{subject} run contract seed mismatch")
    return TRAINING_SEED, sorted(observed_fields)


def _expected_training_parent(
    subject: str,
    controller_task_id: str,
    *,
    recovery: Mapping[str, object] | None = None,
) -> str:
    if subject == "support_residual":
        return LEGACY_PARENT_TASK_ID
    if subject == "no_distillation":
        return NO_DISTILLATION_PARENT_TASK_ID
    if subject in {"ptf_none", "ptf_linear", "router_static"}:
        return RECOVERY_PARENT_TASK_ID
    if subject in NEW_SOURCE_SUBJECTS:
        if recovery is not None:
            for key in (
                "recovery_target_adoptions",
                "recovered_pending_target_children",
            ):
                observations = recovery.get(key)
                if not isinstance(observations, Mapping):
                    continue
                observation = observations.get(subject)
                if isinstance(observation, Mapping):
                    return _clearml_id(
                        observation.get("parent_controller_task_id"),
                        f"{subject} recovery parent controller",
                    )
        return controller_task_id
    return SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID


def _expected_training_script(subject: str) -> str:
    if subject in LEGACY_SCRIPT_ALLOWED_SUBJECTS:
        return LEGACY_TRAINING_SCRIPT_SHA256
    return CANONICAL_TRAINING_SCRIPT_SHA256


def _validate_recovery_progress(
    progress: Mapping[str, object],
    *,
    controller_task_id: str,
    training_entries: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
    seal = _require_seal(progress, context="training recovery progress")
    _require_exact_keys(
        progress,
        {
            "controller_task_id",
            "controller_type",
            "created_at",
            "experiment_order",
            "gate_policy",
            "gate_task_id",
            "max_parallel_training_tasks",
            "recovery",
            "revision",
            "schema_version",
            "seal_sha256",
            "steps",
            "teacher",
            "template",
            "training_overlay_protocol_seed",
            "training_seed",
            "updated_at",
            "worker_queue",
            "worker_queues",
        },
        context="training recovery progress",
    )
    if (
        progress.get("controller_task_id") != controller_task_id
        or progress.get("controller_type")
        != "resilient_v2x_post_main_sequential_training"
        or progress.get("experiment_order") != list(SUBJECT_ORDER)
        or progress.get("schema_version") != 1
        or progress.get("gate_policy")
        != "exact_task_must_be_completed_before_any_clone"
        or progress.get("max_parallel_training_tasks") != 4
        or progress.get("training_seed") != TRAINING_SEED
        or progress.get("training_overlay_protocol_seed") != TRAINING_SEED
        or progress.get("worker_queue") != "GPU4-A100"
        or progress.get("worker_queues")
        != ["GPU4-A100", "GPU4-A100", "GPU4-V100", "GPU4-5090"]
    ):
        raise RuntimeError("training recovery progress root contract drifted")
    raw_steps = progress.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) != len(SUBJECT_ORDER):
        raise RuntimeError("training recovery progress step inventory drifted")
    for index, (subject, raw_step, entry) in enumerate(
        zip(SUBJECT_ORDER, raw_steps, training_entries, strict=True), start=1
    ):
        if not isinstance(raw_step, Mapping):
            raise RuntimeError(f"training recovery step {subject} is invalid")
        if (
            raw_step.get("index") != index
            or raw_step.get("experiment") != subject
            or raw_step.get("state") != "completed"
            or raw_step.get("task_id") != entry["training_task_id"]
            or raw_step.get("predecessor_task_id")
            != entry["training_predecessor_task_id"]
        ):
            raise RuntimeError(f"training recovery step {subject} binding drifted")
        result = raw_step.get("result")
        if not isinstance(result, Mapping) or result.get("task_id") != raw_step.get(
            "task_id"
        ):
            raise RuntimeError(f"training recovery result {subject} drifted")
    recovery = progress.get("recovery")
    if not isinstance(recovery, Mapping):
        raise RuntimeError("training recovery contract is missing")
    _require_exact_keys(
        recovery,
        {
            "adopted_predecessor_task_ids",
            "adopted_task_binding_receipts",
            "adopted_task_ids",
            "adopted_task_template_roles",
            "completion_validation_retries",
            "mode",
            "recovered_pending_target_children",
            "recovery_target_adoptions",
            "rerun_experiments",
            "rerun_predecessor_task_ids",
            "rerun_source_task_ids",
            "rerun_task_observations",
            "schema_version",
            "source_controller_status",
            "source_controller_task_id",
            "source_patch",
            "source_progress_artifact_readback",
            "source_progress_revision",
            "source_progress_seal_sha256",
            "source_recovery_chain",
            "source_revision_transition",
            "source_template_script_sha256",
            "target_template_script_sha256",
            "target_template_predecessor_task_ids",
            "transition_replaced_source_tasks",
        },
        context="training recovery contract",
    )
    adopted = recovery.get("adopted_task_ids")
    adopted_predecessors = recovery.get("adopted_predecessor_task_ids")
    adopted_roles = recovery.get("adopted_task_template_roles")
    adopted_receipts = recovery.get("adopted_task_binding_receipts")
    expected_old_adopted = set(SUBJECT_ORDER) - set(NEW_SOURCE_SUBJECTS)
    if (
        recovery.get("schema_version") != 4
        or recovery.get("mode") != "failed_controller_immutable_fork"
        or recovery.get("source_controller_status") != "failed"
        or not isinstance(adopted, Mapping)
        or not isinstance(adopted_predecessors, Mapping)
        or not isinstance(adopted_roles, Mapping)
        or not isinstance(adopted_receipts, Mapping)
        or not expected_old_adopted <= set(adopted)
        or not set(adopted) <= set(SUBJECT_ORDER)
        or set(adopted_predecessors) != set(adopted)
        or set(adopted_roles) != set(adopted)
        or set(adopted_receipts) != set(adopted)
    ):
        raise RuntimeError("training recovery adopted inventory drifted")
    entries_by_subject = {str(entry["subject"]): entry for entry in training_entries}
    for subject in adopted:
        entry = entries_by_subject[subject]
        expected_role = "new" if subject in NEW_SOURCE_SUBJECTS else "old"
        if (
            adopted.get(subject) != entry["training_task_id"]
            or adopted_predecessors.get(subject)
            != entry["training_predecessor_task_id"]
            or adopted_roles.get(subject)
            != ("target" if expected_role == "new" else "source")
        ):
            raise RuntimeError(f"training recovery adoption {subject} drifted")
    target_predecessors = recovery.get("target_template_predecessor_task_ids")
    rerun_predecessors = recovery.get("rerun_predecessor_task_ids")
    if (
        not isinstance(target_predecessors, Mapping)
        or set(target_predecessors) != set(NEW_SOURCE_SUBJECTS)
        or not isinstance(rerun_predecessors, Mapping)
    ):
        raise RuntimeError(
            "training recovery target-template predecessor inventory drifted"
        )
    steps_by_subject = {
        str(step.get("experiment")): step
        for step in raw_steps
        if isinstance(step, Mapping)
    }
    for subject in NEW_SOURCE_SUBJECTS:
        predecessor = _clearml_id(
            target_predecessors.get(subject),
            f"training recovery target-template predecessor {subject}",
        )
        entry = entries_by_subject[subject]
        step = steps_by_subject.get(subject)
        if (
            predecessor != SOURCE_REVISION_TARGET_PREDECESSOR_TASK_ID
            or entry.get("training_predecessor_task_id") != predecessor
            or not isinstance(step, Mapping)
            or step.get("predecessor_task_id") != predecessor
            or (
                subject in adopted_predecessors
                and adopted_predecessors[subject] != predecessor
            )
            or (
                subject in rerun_predecessors
                and rerun_predecessors[subject] != predecessor
            )
        ):
            raise RuntimeError(
                f"training recovery target-template predecessor {subject} drifted"
            )
    if recovery.get("source_controller_task_id") != (
        SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
    ):
        raise RuntimeError("training recovery source controller drifted")
    transition = recovery.get("source_revision_transition")
    if not isinstance(transition, Mapping):
        raise RuntimeError("training recovery source transition is missing")
    transition_seal = _require_seal(
        transition, context="training recovery source revision transition"
    )
    source_identity = transition.get("source_template_identity")
    target_identity = transition.get("target_template_identity")
    transition_source_revision = transition.get("source_revision")
    transition_target_revision = transition.get("target_revision")
    if (
        transition_seal != SOURCE_REVISION_TRANSITION_SEAL_SHA256
        or transition.get("source_controller_task_id")
        != SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
        or not isinstance(transition_source_revision, Mapping)
        or transition_source_revision.get("tree_sha256")
        != SOURCE_TREE_SHA256
        or not isinstance(transition_target_revision, Mapping)
        or transition_target_revision.get("tree_sha256")
        != NEW_SOURCE_TREE_SHA256
        or not isinstance(source_identity, Mapping)
        or not isinstance(target_identity, Mapping)
        or source_identity.get("task_id")
        != SOURCE_REVISION_SOURCE_TEMPLATE_TASK_ID
        or target_identity.get("task_id")
        != SOURCE_REVISION_TARGET_TEMPLATE_TASK_ID
        or progress.get("template") != target_identity
        or recovery.get("source_template_script_sha256")
        != source_identity.get("script_sha256")
        or recovery.get("target_template_script_sha256")
        != target_identity.get("script_sha256")
    ):
        raise RuntimeError("training recovery source transition drifted")
    observed_subjects: set[str] = set()
    for key in ("recovery_target_adoptions", "recovered_pending_target_children"):
        values = recovery.get(key)
        if not isinstance(values, Mapping) or not set(values) <= set(adopted):
            raise RuntimeError(f"training recovery {key} inventory drifted")
        for subject, raw_observation in values.items():
            if subject in observed_subjects:
                raise RuntimeError(
                    f"training recovery observation {subject} is duplicated"
                )
            observed_subjects.add(subject)
            if not isinstance(raw_observation, Mapping):
                raise RuntimeError(
                    f"training recovery observation {subject} drifted"
                )
            entry = entries_by_subject[subject]
            required_observation_keys = {
                "task_id",
                "predecessor_task_id",
                "parent_controller_task_id",
            }
            if (
                not required_observation_keys <= set(raw_observation)
                or raw_observation.get("task_id") != adopted[subject]
                or raw_observation.get("task_id") != entry["training_task_id"]
                or raw_observation.get("predecessor_task_id")
                != adopted_predecessors[subject]
                or raw_observation.get("predecessor_task_id")
                != entry["training_predecessor_task_id"]
            ):
                raise RuntimeError(
                    f"training recovery observation {subject} drifted"
                )
            observed_parent = _clearml_id(
                raw_observation.get("parent_controller_task_id"),
                f"training recovery observation {subject} parent controller",
            )
            if subject not in NEW_SOURCE_SUBJECTS and observed_parent != (
                _expected_training_parent(subject, controller_task_id)
            ):
                raise RuntimeError(
                    f"training recovery observation {subject} drifted"
                )
    replaced = recovery.get("transition_replaced_source_tasks")
    if not isinstance(replaced, Mapping) or not set(replaced) <= set(
        NEW_SOURCE_SUBJECTS
    ):
        raise RuntimeError(
            "training recovery transition replacement inventory drifted"
        )
    return seal, _content_sha256(recovery)


def _validate_training_provenance_payload(
    payload: Mapping[str, object],
    *,
    controller_task_id: str,
    progress: Mapping[str, object],
    progress_seal: str,
    recovery_sha256: str,
    manifest: Mapping[str, object],
    manifest_seal: str,
    training_entries: Sequence[Mapping[str, object]],
    script_equivalence: Mapping[str, object],
    training_records: Sequence[Mapping[str, object]],
) -> str:
    seal = _require_seal(payload, context="formal training provenance")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "document_type",
            "protocol_id",
            "passed",
            "controller_task_id",
            "controller_status",
            "controller_raw_authority_sha256",
            "subject_order",
            "subject_count",
            "all_training_tasks_completed",
            "authoritative_metadata_read",
            "progress_artifact",
            "progress_seal_sha256",
            "progress_content_sha256",
            "formal_training_manifest_artifact",
            "formal_training_manifest_seal_sha256",
            "formal_training_manifest_content_sha256",
            "recovery_contract_sha256",
            "recursive_progress_chain",
            "bootstrap_equivalence",
            "source_revision_equivalence",
            "source_revision_equivalence_seal_sha256",
            "source_revision_subject_map",
            "source_revision_subject_map_seal_sha256",
            "run_contract_equivalence",
            "capacity_matched_hardware",
            "recovery_parent_controllers",
            "training_tasks",
            "seal_sha256",
        },
        context="formal training provenance",
    )
    expected = {
        "schema_version": 2,
        "document_type": "resilient_v2x_formal_training_provenance_equivalence",
        "protocol_id": PROTOCOL_ID,
        "passed": True,
        "controller_task_id": controller_task_id,
        "controller_status": "completed",
        "subject_order": list(SUBJECT_ORDER),
        "subject_count": len(SUBJECT_ORDER),
        "all_training_tasks_completed": True,
        "progress_artifact": TRAINING_PROGRESS_ARTIFACT,
        "progress_seal_sha256": progress_seal,
        "progress_content_sha256": _content_sha256(progress),
        "formal_training_manifest_artifact": TRAINING_MANIFEST_ARTIFACT,
        "formal_training_manifest_seal_sha256": manifest_seal,
        "formal_training_manifest_content_sha256": _content_sha256(manifest),
        "recovery_contract_sha256": recovery_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"formal training provenance {key} mismatch")
    if (
        payload.get("authoritative_metadata_read") != "single_batch_per_snapshot"
        or SHA256_PATTERN.fullmatch(
            str(payload.get("controller_raw_authority_sha256") or "")
        )
        is None
    ):
        raise RuntimeError("formal training provenance controller authority drifted")
    bootstrap = payload.get("bootstrap_equivalence")
    if not isinstance(bootstrap, Mapping):
        raise RuntimeError("formal training provenance bootstrap evidence is missing")
    _require_exact_keys(
        bootstrap,
        {
            "equivalence_contract",
            "verified_from_actual_script_bytes",
            "legacy_script_sha256",
            "expanded_script_sha256",
            "legacy_nested_teacher_experiments",
            "expanded_nested_teacher_experiments",
            "common_text_projection_sha256",
            "common_ast_projection_sha256",
            "only_text_difference",
            "only_ast_difference",
            "capacity_matched_hardware_contract",
            "runtime_guard_evidence",
            "tf32_override",
            "homogeneous_gpu_count",
            "allowed_compute_capabilities",
            "legacy_script_subjects",
            "expanded_script_subjects",
        },
        context="formal training provenance bootstrap evidence",
    )
    for key in (
        "legacy_script_sha256",
        "legacy_script_subjects",
    ):
        if bootstrap.get(key) != script_equivalence.get(key):
            raise RuntimeError(f"formal training provenance bootstrap {key} drifted")
    if bootstrap.get("expanded_script_sha256") != script_equivalence.get(
        "canonical_script_sha256"
    ):
        raise RuntimeError("formal training provenance expanded script drifted")
    expected_bootstrap_values = {
        "equivalence_contract": "nested-teacher-membership-only-v1",
        "verified_from_actual_script_bytes": True,
        "legacy_nested_teacher_experiments": sorted(LEGACY_NESTED_TEACHER_EXPERIMENTS),
        "expanded_nested_teacher_experiments": sorted(
            CANONICAL_NESTED_TEACHER_EXPERIMENTS
        ),
        "only_text_difference": "NESTED_TEACHER_EXPERIMENTS assignment",
        "only_ast_difference": "NESTED_TEACHER_EXPERIMENTS frozenset members",
        "capacity_matched_hardware_contract": "capacity-matched-hardware-v1",
        "runtime_guard_evidence": "reviewed_completed_bootstrap_bytes",
        "tf32_override": "0",
        "homogeneous_gpu_count": 4,
        "allowed_compute_capabilities": [[7, 0], [8, 0], [12, 0]],
        "expanded_script_subjects": [
            subject
            for subject in SUBJECT_ORDER
            if subject not in LEGACY_SCRIPT_ALLOWED_SUBJECTS
        ],
    }
    for key, value in expected_bootstrap_values.items():
        if bootstrap.get(key) != value:
            raise RuntimeError(f"formal training provenance bootstrap {key} drifted")
    for key in ("common_text_projection_sha256", "common_ast_projection_sha256"):
        if SHA256_PATTERN.fullmatch(str(bootstrap.get(key) or "")) is None:
            raise RuntimeError(f"formal training provenance bootstrap {key} drifted")
    source_equivalence = payload.get("source_revision_equivalence")
    if not isinstance(source_equivalence, Mapping):
        raise RuntimeError("formal training provenance source equivalence is missing")
    source_equivalence_seal = _require_seal(
        source_equivalence,
        context="formal training provenance source equivalence",
    )
    if (
        source_equivalence_seal != SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256
        or payload.get("source_revision_equivalence_seal_sha256")
        != source_equivalence_seal
    ):
        raise RuntimeError("formal training provenance source equivalence drifted")
    if source_equivalence.get("source_revisions") != (
        SOURCE_REVISION_CERTIFICATE_BY_TREE
    ):
        raise RuntimeError("formal training provenance source certificates drifted")
    source_subject_map = payload.get("source_revision_subject_map")
    if not isinstance(source_subject_map, Mapping):
        raise RuntimeError("formal training provenance source subject map is missing")
    source_subject_map_seal = _require_seal(
        source_subject_map,
        context="formal training provenance source subject map",
    )
    expected_subject_map = {
        subject: SOURCE_BY_REVISION[SOURCE_REVISION_BY_SUBJECT[subject]][
            "tree_sha256"
        ]
        for subject in SUBJECT_ORDER
    }
    if (
        source_subject_map_seal != SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256
        or payload.get("source_revision_subject_map_seal_sha256")
        != source_subject_map_seal
        or source_subject_map.get("source_revision_by_subject")
        != expected_subject_map
    ):
        raise RuntimeError("formal training provenance source subject map drifted")
    expected_run_contract = {
        "source_fields_vary_only_by_sealed_subject_revision_map": True,
        "source_revision_equivalence_seal_sha256": source_equivalence_seal,
        "source_revision_subject_map_seal_sha256": source_subject_map_seal,
        "source_revision_counts": {
            SOURCE_TREE_SHA256: 21,
            NEW_SOURCE_TREE_SHA256: 5,
        },
        "training_dataset_id": TRAINING_DATASET_ID,
        "native_bundle_sha256": NATIVE_BUNDLE_SHA256,
        "build_manifest_sha256": BUILD_MANIFEST_SHA256,
        "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
        "training_seed": TRAINING_SEED,
        "global_batch_size": 8,
        "precision": "FP32",
        "max_epochs": 50,
        "val_interval": 10,
        "amp": False,
    }
    if payload.get("run_contract_equivalence") != expected_run_contract:
        raise RuntimeError("formal training provenance run-contract evidence drifted")
    expected_hardware = {
        "contract": "capacity-matched-hardware-v1",
        "docker_command_sha256": DOCKER_COMMAND_SHA256,
        "docker_command_identical_across_26_tasks": True,
        "gpu_count": 4,
        "homogeneous_per_task": True,
        "allowed_compute_capabilities": [[7, 0], [8, 0], [12, 0]],
        "gpu_class_may_vary_across_tasks": True,
        "tf32_override": "0",
        "precision": "FP32",
        "global_batch_size": 8,
        "runtime_guard_evidence": "reviewed_completed_bootstrap_bytes",
    }
    if payload.get("capacity_matched_hardware") != expected_hardware:
        raise RuntimeError("formal training provenance hardware evidence drifted")
    lineage = payload.get("recursive_progress_chain")
    expected_lineage_ids = (
        controller_task_id,
        SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID,
        RECOVERY_PARENT_TASK_ID,
        LEGACY_PARENT_TASK_ID,
    )
    if not isinstance(lineage, list) or len(lineage) != len(expected_lineage_ids):
        raise RuntimeError("formal training provenance recursive lineage is missing")
    progress_seals_by_controller: dict[str, str] = {}
    progress_content_by_controller: dict[str, str] = {}
    for index, (record, expected_task_id) in enumerate(
        zip(lineage, expected_lineage_ids, strict=True)
    ):
        task_id = str(record.get("task_id")) if isinstance(record, Mapping) else ""
        expected_role = (
            "final_controller" if index == 0 else "recursive_recovery_source_controller"
        )
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "task_id",
                "role",
                "progress_artifact",
                "progress_revision",
                "progress_seal_sha256",
                "progress_content_sha256",
                "template_script_identity_sha256",
            }
            or task_id != expected_task_id
            or record.get("progress_artifact") != TRAINING_PROGRESS_ARTIFACT
            or record.get("role") != expected_role
            or type(record.get("progress_revision")) is not int
            or SHA256_PATTERN.fullmatch(str(record.get("progress_seal_sha256") or ""))
            is None
            or SHA256_PATTERN.fullmatch(
                str(record.get("progress_content_sha256") or "")
            )
            is None
            or SHA256_PATTERN.fullmatch(
                str(record.get("template_script_identity_sha256") or "")
            )
            is None
        ):
            raise RuntimeError("formal training provenance lineage record drifted")
        progress_seals_by_controller[task_id] = str(record["progress_seal_sha256"])
        progress_content_by_controller[task_id] = str(record["progress_content_sha256"])
    if progress_seals_by_controller[
        controller_task_id
    ] != progress_seal or progress_content_by_controller[
        controller_task_id
    ] != _content_sha256(progress):
        raise RuntimeError("formal training provenance root lineage binding drifted")
    parent_records = payload.get("recovery_parent_controllers")
    if not isinstance(parent_records, list):
        raise RuntimeError("formal training provenance parent inventory is missing")
    parents = {
        str(record.get("task_id")): record
        for record in parent_records
        if isinstance(record, Mapping)
    }
    expected_parent_subjects = {
        LEGACY_PARENT_TASK_ID: ["support_residual"],
        NO_DISTILLATION_PARENT_TASK_ID: ["no_distillation"],
        RECOVERY_PARENT_TASK_ID: ["ptf_none", "ptf_linear", "router_static"],
        SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID: [
            subject
            for subject in SUBJECT_ORDER
            if subject not in NEW_SOURCE_SUBJECTS
            and subject
            not in {
                "support_residual",
                "no_distillation",
                "ptf_none",
                "ptf_linear",
                "router_static",
            }
        ],
    }
    progress_recovery = progress.get("recovery")
    if not isinstance(progress_recovery, Mapping):
        raise RuntimeError("formal training provenance root recovery is missing")
    for subject in SUBJECT_ORDER:
        if subject not in NEW_SOURCE_SUBJECTS:
            continue
        parent_id = _expected_training_parent(
            subject,
            controller_task_id,
            recovery=progress_recovery,
        )
        if parent_id != controller_task_id:
            expected_parent_subjects.setdefault(parent_id, []).append(subject)
    if set(parents) != set(expected_parent_subjects):
        raise RuntimeError("formal training provenance parent inventory drifted")
    for parent_id, subjects in expected_parent_subjects.items():
        record = parents[parent_id]
        has_recursive_progress = parent_id in progress_seals_by_controller
        expected_roles = [
            *(["recursive_recovery_source"] if has_recursive_progress else []),
            "actual_training_parent",
        ]
        if (
            set(record)
            != {
                "task_id",
                "terminal_status",
                "roles",
                "actual_parent_subjects",
                "progress_artifact",
                "progress_seal_sha256",
                "progress_content_sha256",
                "raw_authority_sha256",
            }
            or record.get("terminal_status") != "failed"
            or record.get("roles") != expected_roles
            or record.get("actual_parent_subjects") != subjects
            or record.get("progress_artifact")
            != (TRAINING_PROGRESS_ARTIFACT if has_recursive_progress else None)
            or (
                has_recursive_progress
                and (
                    SHA256_PATTERN.fullmatch(
                        str(record.get("progress_seal_sha256") or "")
                    )
                    is None
                    or record.get("progress_seal_sha256")
                    != progress_seals_by_controller[parent_id]
                    or SHA256_PATTERN.fullmatch(
                        str(record.get("progress_content_sha256") or "")
                    )
                    is None
                    or record.get("progress_content_sha256")
                    != progress_content_by_controller[parent_id]
                )
            )
            or (
                not has_recursive_progress
                and (
                    record.get("progress_seal_sha256") is not None
                    or record.get("progress_content_sha256") is not None
                )
            )
            or SHA256_PATTERN.fullmatch(str(record.get("raw_authority_sha256") or ""))
            is None
        ):
            raise RuntimeError(f"formal training provenance parent {parent_id} drifted")
    records = payload.get("training_tasks")
    if not isinstance(records, list) or len(records) != len(SUBJECT_ORDER):
        raise RuntimeError("formal training provenance task inventory drifted")
    record_by_subject = {
        str(record.get("subject")): record
        for record in records
        if isinstance(record, Mapping)
    }
    if set(record_by_subject) != set(SUBJECT_ORDER):
        raise RuntimeError("formal training provenance subject inventory drifted")
    audit_by_subject = {str(record["subject"]): record for record in training_records}
    entry_by_subject = {str(entry["subject"]): entry for entry in training_entries}
    for index, subject in enumerate(SUBJECT_ORDER, start=1):
        record = record_by_subject[subject]
        _require_exact_keys(
            record,
            {
                "index",
                "subject",
                "training_task_id",
                "predecessor_task_id",
                "parent_controller_task_id",
                "parent_binding",
                "recovery_lineage",
                "raw_bootstrap_script_sha256",
                "normalized_task_script_identity_sha256",
                "sealed_progress_script_identity_sha256",
                "script_equivalence_class",
                "manifest_model_id",
                "manifest_model_name",
                "manifest_model_url",
                "checkpoint_sha256",
                "checkpoint_size_bytes",
                "source_revision_tree_sha256",
                "source_dataset_id",
                "source_archive_name",
                "source_archive_bytes",
                "source_archive_sha256",
                "observed_parameter_keys",
                "observed_parameters_sha256",
                "docker_command_sha256",
                "execution_queue_id",
                "last_worker",
                "raw_authority_sha256",
                "run_contract_content_sha256",
                "final_checkpoint_contract_content_sha256",
                "common_teacher_initialization_audit_content_sha256",
            },
            context=f"formal training provenance task {subject}",
        )
        raw_lineage = record.get("recovery_lineage")
        if not isinstance(raw_lineage, list) or not raw_lineage:
            raise RuntimeError(
                f"formal training provenance {subject} recovery lineage drifted"
            )
        lineage_controller = controller_task_id
        resolved_parent_id: str | None = None
        resolved_binding: str | None = None
        for lineage_index, raw_step in enumerate(raw_lineage):
            if not isinstance(raw_step, Mapping):
                raise RuntimeError(
                    f"formal training provenance {subject} lineage step drifted"
                )
            decision = str(raw_step.get("decision") or "")
            expected_keys = {
                "controller_task_id",
                "decision",
                "progress_seal_sha256",
            }
            target_key = None
            if decision == "adopted_from_source_progress":
                target_key = "source_controller_task_id"
            elif decision in {
                "recovered_pending_target_children",
                "recovery_target_adoptions",
            }:
                target_key = "parent_controller_task_id"
            elif decision != "controller_created":
                raise RuntimeError(
                    f"formal training provenance {subject} lineage decision drifted"
                )
            if target_key is not None:
                expected_keys.add(target_key)
            if (
                set(raw_step) != expected_keys
                or raw_step.get("controller_task_id") != lineage_controller
                or raw_step.get("progress_seal_sha256")
                != progress_seals_by_controller[lineage_controller]
            ):
                raise RuntimeError(
                    f"formal training provenance {subject} lineage step drifted"
                )
            terminal = lineage_index == len(raw_lineage) - 1
            if decision == "adopted_from_source_progress":
                if terminal:
                    raise RuntimeError(
                        f"formal training provenance {subject} lineage is truncated"
                    )
                next_controller = _clearml_id(
                    raw_step.get("source_controller_task_id"),
                    f"formal training provenance {subject} source controller",
                )
                next_step = raw_lineage[lineage_index + 1]
                if (
                    not isinstance(next_step, Mapping)
                    or next_step.get("controller_task_id") != next_controller
                ):
                    raise RuntimeError(
                        f"formal training provenance {subject} lineage is cross-spliced"
                    )
                lineage_controller = next_controller
            elif decision == "controller_created":
                if not terminal:
                    raise RuntimeError(
                        f"formal training provenance {subject} lineage continues past origin"
                    )
                resolved_parent_id = lineage_controller
                resolved_binding = (
                    "current_controller_created"
                    if lineage_controller == controller_task_id
                    else "recursive_source_controller_created"
                )
            else:
                if not terminal:
                    raise RuntimeError(
                        f"formal training provenance {subject} lineage continues past parent"
                    )
                resolved_parent_id = _clearml_id(
                    raw_step.get("parent_controller_task_id"),
                    f"formal training provenance {subject} observed parent",
                )
                resolved_binding = decision
        expected_first_decision = (
            "controller_created"
            if subject in NEW_SOURCE_SUBJECTS
            else "adopted_from_source_progress"
        )
        for observation_key in (
            "recovery_target_adoptions",
            "recovered_pending_target_children",
        ):
            observations = progress_recovery.get(observation_key)
            if isinstance(observations, Mapping) and subject in observations:
                expected_first_decision = observation_key
                break
        first_lineage = raw_lineage[0]
        if (
            not isinstance(first_lineage, Mapping)
            or first_lineage.get("decision") != expected_first_decision
            or (
                expected_first_decision == "adopted_from_source_progress"
                and first_lineage.get("source_controller_task_id")
                != SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID
            )
            or record.get("parent_binding") != resolved_binding
            or record.get("parent_controller_task_id") != resolved_parent_id
        ):
            raise RuntimeError(
                f"formal training provenance {subject} root lineage role drifted"
            )
        audit = audit_by_subject[subject]
        entry = entry_by_subject[subject]
        source_revision = SOURCE_REVISION_BY_SUBJECT[subject]
        expected_source = SOURCE_BY_REVISION[source_revision]
        expected_record = {
            "index": index,
            "training_task_id": entry["training_task_id"],
            "predecessor_task_id": entry["training_predecessor_task_id"],
            "parent_controller_task_id": _expected_training_parent(
                subject,
                controller_task_id,
                recovery=progress.get("recovery"),
            ),
            "raw_bootstrap_script_sha256": audit["script_sha256"],
            "manifest_model_id": entry["model_id"],
            "manifest_model_name": entry["model_name"],
            "manifest_model_url": entry["model_url"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "checkpoint_size_bytes": entry["checkpoint_size_bytes"],
            "source_revision_tree_sha256": expected_source["tree_sha256"],
            "source_dataset_id": expected_source["dataset_id"],
            "source_archive_name": expected_source["archive_name"],
            "source_archive_bytes": expected_source["archive_size_bytes"],
            "source_archive_sha256": expected_source["archive_sha256"],
            "script_equivalence_class": (
                "legacy_nested_teacher_membership"
                if subject in LEGACY_SCRIPT_ALLOWED_SUBJECTS
                else "expanded_nested_teacher_membership"
            ),
            "docker_command_sha256": DOCKER_COMMAND_SHA256,
            "run_contract_content_sha256": audit["run_contract_sha256"],
            "final_checkpoint_contract_content_sha256": audit[
                "final_checkpoint_contract_content_sha256"
            ],
            "common_teacher_initialization_audit_content_sha256": audit[
                "common_teacher_initialization_audit_sha256"
            ],
            "execution_queue_id": audit["execution_queue_id"],
            "last_worker": audit["last_worker"],
            "raw_authority_sha256": audit["raw_authority_sha256"],
        }
        for key, value in expected_record.items():
            if record.get(key) != value:
                raise RuntimeError(
                    f"formal training provenance {subject} {key} drifted"
                )
        for key in (
            "normalized_task_script_identity_sha256",
            "sealed_progress_script_identity_sha256",
            "observed_parameters_sha256",
        ):
            if SHA256_PATTERN.fullmatch(str(record.get(key) or "")) is None:
                raise RuntimeError(
                    f"formal training provenance {subject} {key} drifted"
                )
    return seal


def _validate_training_task(
    task: object,
    *,
    controller_task_id: str,
    entry: Mapping[str, object],
    manifest_seeded: bool,
    recovery: Mapping[str, object],
) -> dict[str, object]:
    subject = str(entry["subject"])
    task_id = str(entry["training_task_id"])
    context = f"training task {subject}"
    if _task_id(task, context=context) != task_id:
        raise RuntimeError(f"{context} identity mismatch")
    if _status(task, context=context) != "completed":
        raise RuntimeError(f"{context} is not completed")
    expected_parent = _expected_training_parent(
        subject, controller_task_id, recovery=recovery
    )
    if _task_parent(task, context=context) != expected_parent:
        raise RuntimeError(f"{context} parent mismatch")
    _require_artifact_inventory(
        task,
        _training_artifact_inventory(subject),
        context=context,
    )
    source = _raw_script_source(
        task, entry_point="clearml_5090_bootstrap.py", context=context
    )
    script_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if script_sha != _expected_training_script(subject):
        raise RuntimeError(f"{context} script SHA-256 mismatch")
    source_revision = SOURCE_REVISION_BY_SUBJECT[subject]
    expected_source = SOURCE_BY_REVISION[source_revision]
    parameters = _parameters(task, context=context)
    _require_parameters(
        parameters,
        {
            "Args/experiment_from_task": subject,
            "Args/source_dataset_id": expected_source["dataset_id"],
            "Args/source_archive_name": expected_source["archive_name"],
            "Args/source_archive_bytes": expected_source["archive_size_bytes"],
            "Args/source_archive_sha256": expected_source["archive_sha256"],
            "Args/training_dataset_id": TRAINING_DATASET_ID,
            "Args/predecessor_task_id": entry["training_predecessor_task_id"],
            "Args/stage": "all",
            "Args/max_epochs": 50,
            "Args/gpus": 4,
            "Args/amp": False,
        },
        context=context,
    )
    contract = _artifact_mapping(task, RUN_CONTRACT_ARTIFACT, context=context)
    expected_contract = {
        "schema_version": 1,
        "mode": "experiment_from_task",
        "task_id": task_id,
        "experiment": subject,
        "source_dataset_id": expected_source["dataset_id"],
        "training_dataset_id": TRAINING_DATASET_ID,
        "predecessor_task_id": entry["training_predecessor_task_id"],
        "gpus": 4,
        "global_batch_size": 8,
        "ddp_processes": 4,
        "max_epochs": 50,
        "amp": False,
        "precision": "FP32",
        "val_interval": 10,
        "condition_evaluation": False,
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise RuntimeError(f"{context} run contract {key} mismatch")
    _, run_contract_seed_fields = _run_contract_seed(
        contract,
        subject=subject,
        manifest_seeded=manifest_seeded,
    )
    source_archive = contract.get("source_archive")
    if not isinstance(source_archive, Mapping):
        raise RuntimeError(f"{context} run contract source archive is missing")
    if source_archive != {
        "name": expected_source["archive_name"],
        "size_bytes": expected_source["archive_size_bytes"],
        "sha256": expected_source["archive_sha256"],
    }:
        raise RuntimeError(f"{context} run contract source archive identity mismatch")
    final_contract = _artifact_mapping(task, FINAL_CHECKPOINT_ARTIFACT, context=context)
    final_contract_content_sha256 = _content_sha256(final_contract)
    expected_final = {
        "model_id": entry["model_id"],
        "name": entry["model_name"],
        "url": entry["model_url"],
        "filename": "epoch_50.pth",
        "size_bytes": entry["checkpoint_size_bytes"],
        "sha256": entry["checkpoint_sha256"],
    }
    for key, value in expected_final.items():
        if final_contract.get(key) != value:
            raise RuntimeError(f"{context} final checkpoint {key} mismatch")
    _require_unique_output_model(task, entry=entry, context=context)
    initialization_audit = _artifact_mapping(
        task,
        COMMON_TEACHER_AUDIT_ARTIFACT,
        context=context,
    )
    initialization_audit_sha256 = _content_sha256(initialization_audit)
    if (
        initialization_audit_sha256
        != entry["common_teacher_initialization_audit_sha256"]
    ):
        raise RuntimeError(f"{context} initialization audit SHA-256 mismatch")
    return {
        "index": entry["index"],
        "subject": subject,
        "training_task_id": task_id,
        "model_id": entry["model_id"],
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "parent_controller_task_id": expected_parent,
        "script_sha256": script_sha,
        "script_source": source,
        "run_contract_sha256": _content_sha256(contract),
        "final_checkpoint_contract_content_sha256": (final_contract_content_sha256),
        "run_contract_seed_fields": run_contract_seed_fields,
        "training_seed": TRAINING_SEED,
        "source_revision_tree_sha256": expected_source["tree_sha256"],
        "source_dataset_id": expected_source["dataset_id"],
        "source_archive_name": expected_source["archive_name"],
        "source_archive_bytes": expected_source["archive_size_bytes"],
        "source_archive_sha256": expected_source["archive_sha256"],
        "training_dataset_id": TRAINING_DATASET_ID,
        "gpus": 4,
        "precision": "FP32",
        "max_epochs": 50,
        "val_interval": 10,
        "common_teacher_initialization_audit_sha256": (initialization_audit_sha256),
        "final_model_verification_level": "clearml_metadata_contract_only",
        "checkpoint_bytes_sha256_recomputed": False,
        "checkpoint_bytes_verifier": MODEL_BYTES_VERIFIER,
    }


def _validate_evaluation_plan(
    payload: Mapping[str, object],
    *,
    controller_task_id: str,
    evaluation_template_task_id: str,
) -> tuple[list[dict[str, str]], str, dict[str, object]]:
    plan = dict(payload)
    seal = _require_seal(plan, context="formal evaluation plan")
    _require_exact_keys(
        plan,
        {
            "schema_version",
            "plan_type",
            "training_controller_task_id",
            "training_provenance_task_id",
            "training_provenance_seal_sha256",
            "source_revision_equivalence",
            "source_revision_equivalence_seal_sha256",
            "source_revision_subject_map",
            "source_revision_subject_map_seal_sha256",
            "evaluation_source_revision_tree_sha256",
            "evaluation_source_revision",
            "evaluation_template_task_id",
            "protocol_id",
            "sample_count",
            "delays_ms",
            "conditions",
            "run_count",
            "subject_order",
            "entries",
            "seal_sha256",
        },
        context="formal evaluation plan",
    )
    expected = {
        "schema_version": 2,
        "plan_type": "resilient_v2x_formal_1337_evaluation_tasks",
        "training_controller_task_id": controller_task_id,
        "evaluation_template_task_id": evaluation_template_task_id,
        "protocol_id": PROTOCOL_ID,
        "sample_count": SAMPLE_COUNT,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "run_count": len(DELAYS_MS) * len(CONDITIONS),
        "subject_order": list(SUBJECT_ORDER),
        "evaluation_source_revision_tree_sha256": SOURCE_TREE_SHA256,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"formal evaluation plan {key} mismatch")
    provenance_task_id = _clearml_id(
        plan.get("training_provenance_task_id"), "training provenance task"
    )
    provenance_seal = _sha256(
        plan.get("training_provenance_seal_sha256"), "training provenance seal"
    )
    source_equivalence = plan.get("source_revision_equivalence")
    if not isinstance(source_equivalence, Mapping):
        raise ValueError("formal evaluation plan source equivalence is missing")
    source_equivalence_seal = _require_seal(
        source_equivalence, context="formal evaluation plan source equivalence"
    )
    if (
        source_equivalence_seal != SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256
        or plan.get("source_revision_equivalence_seal_sha256")
        != source_equivalence_seal
    ):
        raise ValueError("formal evaluation plan source equivalence mismatch")
    if source_equivalence.get("source_revisions") != (
        SOURCE_REVISION_CERTIFICATE_BY_TREE
    ):
        raise ValueError("formal evaluation plan source certificates mismatch")
    source_subject_map = plan.get("source_revision_subject_map")
    if not isinstance(source_subject_map, Mapping):
        raise ValueError("formal evaluation plan source subject map is missing")
    source_subject_map_seal = _require_seal(
        source_subject_map, context="formal evaluation plan source subject map"
    )
    expected_subject_map = {
        subject: SOURCE_BY_REVISION[SOURCE_REVISION_BY_SUBJECT[subject]][
            "tree_sha256"
        ]
        for subject in SUBJECT_ORDER
    }
    if (
        source_subject_map_seal != SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256
        or plan.get("source_revision_subject_map_seal_sha256")
        != source_subject_map_seal
        or source_subject_map.get("source_revision_by_subject")
        != expected_subject_map
    ):
        raise ValueError("formal evaluation plan source subject map mismatch")
    evaluation_source_revision = plan.get("evaluation_source_revision")
    if (
        not isinstance(evaluation_source_revision, Mapping)
        or dict(evaluation_source_revision)
        != SOURCE_REVISION_CERTIFICATE_BY_TREE[SOURCE_TREE_SHA256]
        or source_equivalence["source_revisions"].get(SOURCE_TREE_SHA256)
        != evaluation_source_revision
    ):
        raise ValueError("formal evaluation plan old source certificate mismatch")
    raw_entries = plan.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != len(SUBJECT_ORDER):
        raise ValueError("formal evaluation plan entry count mismatch")
    entries: list[dict[str, str]] = []
    task_ids: set[str] = set()
    for index, (raw_entry, subject) in enumerate(
        zip(raw_entries, SUBJECT_ORDER, strict=True), start=1
    ):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"formal evaluation plan entry {index} is invalid")
        _require_exact_keys(
            raw_entry,
            {"subject", "evaluation_task_id", "queue"},
            context=f"formal evaluation plan entry {subject}",
        )
        if raw_entry.get("subject") != subject:
            raise ValueError(f"formal evaluation plan entry {index} subject mismatch")
        task_id = _clearml_id(
            raw_entry.get("evaluation_task_id"), f"{subject} evaluation task"
        )
        if task_id in task_ids:
            raise ValueError("formal evaluation task IDs are not unique")
        task_ids.add(task_id)
        queue = str(raw_entry.get("queue") or "")
        if queue not in SUPPORTED_QUEUES:
            raise ValueError(f"{subject} evaluation queue is unsupported")
        entries.append(
            {"subject": subject, "evaluation_task_id": task_id, "queue": queue}
        )
    return (
        entries,
        seal,
        {
            "training_provenance_task_id": provenance_task_id,
            "training_provenance_seal_sha256": provenance_seal,
            "source_revision_equivalence": dict(source_equivalence),
            "source_revision_equivalence_seal_sha256": source_equivalence_seal,
            "source_revision_subject_map": dict(source_subject_map),
            "source_revision_subject_map_seal_sha256": source_subject_map_seal,
            "evaluation_source_revision_tree_sha256": SOURCE_TREE_SHA256,
            "evaluation_source_revision": dict(evaluation_source_revision),
        },
    )


def _require_ap(value: object, *, context: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise ValueError(f"{context} must be finite and within [0, 100]")
    return result


def _validate_metrics(
    payload: Mapping[str, object],
    *,
    subject: str,
    checkpoint_sha256: str,
) -> dict[tuple[int, str], dict[str, float]]:
    metrics_document = dict(payload)
    _require_exact_keys(
        metrics_document,
        {
            "schema_version",
            "result_type",
            "complete",
            "planned_run_count",
            "baseline",
            "protocol_id",
            "checkpoint",
            "checkpoint_sha256",
            "manifest_content_sha256",
            "overlay_index_content_sha256",
            "sample_ids_sha256",
            "expected_sample_count",
            "expected_ground_truth_count",
            "expected_unsupported_sample_count",
            "runs",
        },
        context=f"{subject} metrics",
    )
    expected = {
        "schema_version": 1,
        "result_type": "resilient_v2x_controlled_baseline_metrics",
        "complete": True,
        "planned_run_count": len(DELAYS_MS) * len(CONDITIONS),
        "baseline": subject,
        "protocol_id": PROTOCOL_ID,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": SAMPLE_IDS_SHA256,
        "expected_sample_count": SAMPLE_COUNT,
        "expected_ground_truth_count": GROUND_TRUTH_COUNT,
        "expected_unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
    }
    for key, value in expected.items():
        if metrics_document.get(key) != value:
            raise ValueError(f"{subject} metrics {key} mismatch")
    if (
        type(metrics_document.get("checkpoint")) is not str
        or not metrics_document["checkpoint"]
    ):
        raise ValueError(f"{subject} metrics checkpoint path is invalid")
    raw_runs = metrics_document.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 12:
        raise ValueError(f"{subject} metrics run count mismatch")
    result: dict[tuple[int, str], dict[str, float]] = {}
    pairs = ((delay, condition) for delay in DELAYS_MS for condition in CONDITIONS)
    for index, (raw_run, pair) in enumerate(zip(raw_runs, pairs, strict=True), start=1):
        if not isinstance(raw_run, Mapping):
            raise ValueError(f"{subject} metrics run {index} is invalid")
        _require_exact_keys(
            raw_run,
            {
                "condition_id",
                "delay_ms",
                "condition",
                "metrics",
                "predictions",
                "prediction_sha256",
                "prediction_content_sha256",
                "sample_count",
                "sample_ids_sha256",
                "ground_truth_count",
                "unsupported_sample_count",
            },
            context=f"{subject} metrics run {index}",
        )
        delay, condition = pair
        condition_id = f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        if (
            raw_run.get("delay_ms") != delay
            or raw_run.get("condition") != condition
            or raw_run.get("condition_id") != condition_id
        ):
            raise ValueError(f"{subject} metrics run {index} order mismatch")
        expected_run = {
            "sample_count": SAMPLE_COUNT,
            "sample_ids_sha256": SAMPLE_IDS_SHA256,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        }
        for key, value in expected_run.items():
            if raw_run.get(key) != value:
                raise ValueError(f"{subject} metrics run {index} {key} mismatch")
        _sha256(raw_run.get("prediction_sha256"), f"{subject} prediction")
        _sha256(
            raw_run.get("prediction_content_sha256"),
            f"{subject} prediction content",
        )
        if type(raw_run.get("predictions")) is not str or not raw_run["predictions"]:
            raise ValueError(f"{subject} metrics run {index} predictions path invalid")
        run_metrics = raw_run.get("metrics")
        if not isinstance(run_metrics, Mapping):
            raise ValueError(f"{subject} metrics run {index} has no metrics object")
        for count_key, expected_count in COUNT_METRICS.items():
            raw_count = run_metrics.get(count_key)
            if (
                type(raw_count) not in {int, float}
                or not math.isfinite(float(raw_count))
                or not float(raw_count).is_integer()
                or int(raw_count) != expected_count
            ):
                raise ValueError(f"{subject} metrics run {index} {count_key} mismatch")
        result[pair] = {
            key: _require_ap(
                run_metrics.get(key), context=f"{subject} run {index} metric {key}"
            )
            for key in AP_METRIC_KEYS
        }
    return result


def _validate_evaluation_task(
    task: object,
    *,
    controller_task_id: str,
    training_entry: Mapping[str, object],
    evaluation_entry: Mapping[str, str],
    queue_name_resolver: Callable[[str], str],
) -> tuple[dict[tuple[int, str], dict[str, float]], dict[str, object]]:
    subject = str(training_entry["subject"])
    task_id = evaluation_entry["evaluation_task_id"]
    context = f"evaluation task {subject}"
    if _task_id(task, context=context) != task_id:
        raise RuntimeError(f"{context} identity mismatch")
    if _status(task, context=context) != "completed":
        raise RuntimeError(f"{context} is not completed")
    if _task_parent(task, context=context) != controller_task_id:
        raise RuntimeError(f"{context} parent mismatch")
    _require_artifact_inventory(
        task,
        _evaluation_artifact_inventory(),
        context=context,
    )
    execution = getattr(getattr(task, "data", None), "execution", None)
    execution_queue_id = _clearml_id(
        getattr(execution, "queue", ""),
        f"{context} execution queue",
    )
    execution_queue_name = str(queue_name_resolver(execution_queue_id) or "")
    if execution_queue_name != evaluation_entry["queue"]:
        raise RuntimeError(f"{context} execution queue mismatch")
    script_sha = _script_sha256(
        task, entry_point="clearml_5090_bootstrap.py", context=context
    )
    if script_sha != EVALUATION_SCRIPT_SHA256:
        raise RuntimeError(f"{context} script SHA-256 mismatch")
    parameters = _parameters(task, context=context)
    _require_parameters(
        parameters,
        {
            "Args/stage": "baseline_validate",
            "Args/controlled_baseline": subject,
            "Args/controlled_baseline_task_id": training_entry["training_task_id"],
            "Args/controlled_baseline_model_id": training_entry["model_id"],
            "Args/controlled_baseline_checkpoint_sha256": training_entry[
                "checkpoint_sha256"
            ],
            "Args/predecessor_task_id": training_entry["training_task_id"],
            "Args/source_dataset_id": SOURCE_DATASET_ID,
            "Args/source_archive_name": SOURCE_ARCHIVE_NAME,
            "Args/source_archive_bytes": SOURCE_ARCHIVE_BYTES,
            "Args/source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "Args/training_dataset_id": TRAINING_DATASET_ID,
            "Args/gpus": 4,
            "Args/max_epochs": 50,
            "Args/amp": False,
        },
        context=context,
    )
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise RuntimeError(f"{context} cannot expose input models")
    models = getter()
    if not isinstance(models, Mapping):
        raise RuntimeError(f"{context} model mapping is invalid")
    inputs = models.get("input")
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        raise RuntimeError(f"{context} input models are invalid")
    input_ids = [str(getattr(model, "id", "") or "") for model in inputs]
    if input_ids != [training_entry["model_id"]]:
        raise RuntimeError(f"{context} input model mismatch")
    metrics_document = _artifact_mapping(task, METRICS_ARTIFACT, context=context)
    runs = _validate_metrics(
        metrics_document,
        subject=subject,
        checkpoint_sha256=str(training_entry["checkpoint_sha256"]),
    )
    record = {
        "index": training_entry["index"],
        "subject": subject,
        "evaluation_task_id": task_id,
        "training_task_id": training_entry["training_task_id"],
        "model_id": training_entry["model_id"],
        "checkpoint_sha256": training_entry["checkpoint_sha256"],
        "source_revision_tree_sha256": SOURCE_TREE_SHA256,
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_archive_name": SOURCE_ARCHIVE_NAME,
        "source_archive_bytes": SOURCE_ARCHIVE_BYTES,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "script_sha256": script_sha,
        "planned_queue": evaluation_entry["queue"],
        "execution_queue_id": execution_queue_id,
        "execution_queue_name": execution_queue_name,
        "last_worker": str(
            getattr(getattr(task, "data", None), "last_worker", "") or ""
        ),
        "metrics_sha256": _content_sha256(metrics_document),
        "run_count": 12,
        "sample_count_per_run": SAMPLE_COUNT,
        "ground_truth_count_per_run": GROUND_TRUTH_COUNT,
        "unsupported_sample_count_per_run": UNSUPPORTED_SAMPLE_COUNT,
        "metric_keys": list(AP_METRIC_KEYS),
    }
    return runs, record


def _clearml_queue_name(task_class: object, queue_id: str) -> str:
    session_getter = getattr(task_class, "_get_default_session", None)
    if not callable(session_getter):
        raise RuntimeError("ClearML Task class cannot resolve execution queues")
    request_type = getattr(clearml_queues, "GetAllRequest", None)
    if request_type is None:
        raise RuntimeError("ClearML queue lookup API is unavailable")
    result = session_getter().send(
        request_type(
            id=[queue_id],
            only_fields=["id", "name"],
            page=0,
            page_size=2,
        )
    )
    response = getattr(result, "response", None)
    queues = getattr(response, "queues", None)
    if not isinstance(queues, Sequence) or isinstance(queues, (str, bytes)):
        raise RuntimeError("ClearML queue lookup returned an invalid response")

    def field(item: object, name: str) -> object:
        if isinstance(item, Mapping):
            return item.get(name)
        return getattr(item, name, None)

    matches = [
        str(field(item, "name") or "")
        for item in queues
        if str(field(item, "id") or "") == queue_id
    ]
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError(f"ClearML execution queue {queue_id} is not unique")
    return matches[0]


def _metric_summary(
    runs: Mapping[tuple[int, str], Mapping[str, float]], metric_key: str
) -> dict[str, object]:
    ordered = [
        (delay, condition, runs[(delay, condition)][metric_key])
        for delay in DELAYS_MS
        for condition in CONDITIONS
    ]
    clean = runs[(0, "Full")][metric_key]
    full_300 = runs[(300, "Full")][metric_key]
    worst_delay, worst_condition, worst_value = min(ordered, key=lambda item: item[2])
    return {
        "clean_0ms": clean,
        "mean_12": math.fsum(item[2] for item in ordered) / len(ordered),
        "worst_12": {
            "value": worst_value,
            "delay_ms": worst_delay,
            "condition": worst_condition,
        },
        "full_mean": math.fsum(runs[(delay, "Full")][metric_key] for delay in DELAYS_MS)
        / len(DELAYS_MS),
        "l_fail_mean": math.fsum(
            runs[(delay, "L-Fail")][metric_key] for delay in DELAYS_MS
        )
        / len(DELAYS_MS),
        "c_fail_mean": math.fsum(
            runs[(delay, "C-Fail")][metric_key] for delay in DELAYS_MS
        )
        / len(DELAYS_MS),
        "full_300": full_300,
        "pdr": None if clean == 0.0 else 100.0 * (clean - full_300) / clean,
    }


def _leadership_dimension(
    summaries: Mapping[str, Mapping[str, Mapping[str, object]]], dimension: str
) -> dict[str, object]:
    def value_for(subject: str) -> float:
        value = summaries[subject][LEADERSHIP_METRIC][dimension]
        if dimension == "worst_12":
            if not isinstance(value, Mapping):
                raise ValueError("worst_12 summary is invalid")
            return float(value["value"])
        return float(value)

    baseline_values = {subject: value_for(subject) for subject in BASELINE_SUBJECTS}
    best_value = max(baseline_values.values())
    best_subjects = [
        subject
        for subject in BASELINE_SUBJECTS
        if baseline_values[subject] == best_value
    ]
    primary_value = value_for(PRIMARY_SUBJECT)
    margin = primary_value - best_value
    return {
        "resilient_v2x_value": primary_value,
        "best_baseline_value": best_value,
        "best_baseline_subject": best_subjects[0],
        "best_baseline_subjects": best_subjects,
        "margin": margin,
        "leads": margin > 0.0,
    }


def _leadership_condition(
    runs_by_subject: Mapping[str, Mapping[tuple[int, str], Mapping[str, float]]],
    delay: int,
    condition: str,
) -> dict[str, object]:
    baseline_values = {
        subject: runs_by_subject[subject][(delay, condition)][LEADERSHIP_METRIC]
        for subject in BASELINE_SUBJECTS
    }
    best_value = max(baseline_values.values())
    best_subjects = [
        subject
        for subject in BASELINE_SUBJECTS
        if baseline_values[subject] == best_value
    ]
    primary_value = runs_by_subject[PRIMARY_SUBJECT][(delay, condition)][
        LEADERSHIP_METRIC
    ]
    margin = primary_value - best_value
    return {
        "condition_id": f"delay_{delay:03d}_{condition.lower().replace('-', '_')}",
        "delay_ms": delay,
        "condition": condition,
        "resilient_v2x_value": primary_value,
        "best_baseline_value": best_value,
        "best_baseline_subject": best_subjects[0],
        "best_baseline_subjects": best_subjects,
        "margin": margin,
        "leads": margin > 0.0,
    }


def _expected_leadership(
    runs_by_subject: Mapping[str, Mapping[tuple[int, str], Mapping[str, float]]],
    summaries: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    aggregate = {
        dimension: _leadership_dimension(summaries, dimension)
        for dimension in ("clean_0ms", "mean_12", "worst_12")
    }
    directional = {
        dimension: _leadership_dimension(summaries, dimension)
        for dimension in ("full_mean", "l_fail_mean", "c_fail_mean", "full_300")
    }
    comparisons = [
        _leadership_condition(runs_by_subject, delay, condition)
        for delay in DELAYS_MS
        for condition in CONDITIONS
    ]
    won = sum(bool(item["leads"]) for item in comparisons)
    return {
        "metric_key": LEADERSHIP_METRIC,
        "comparison_pool": list(BASELINE_SUBJECTS),
        "aggregate_dimensions": aggregate,
        "leads_all_aggregate": all(bool(item["leads"]) for item in aggregate.values()),
        "directional_summary_dimensions": directional,
        "condition_comparisons": comparisons,
        "conditions_won": won,
        "conditions_total": len(comparisons),
        "conditions_won_fraction": f"{won}/{len(comparisons)}",
        "leads_all_conditions": won == len(comparisons),
    }


def _validate_leaderboard(
    payload: Mapping[str, object],
    *,
    controller_task_id: str,
    watcher_task_id: str,
    manifest_seal: str,
    plan_seal: str,
    provenance_binding: Mapping[str, object],
    training_entries: Sequence[Mapping[str, object]],
    evaluation_entries: Sequence[Mapping[str, str]],
    runs_by_subject: Mapping[str, Mapping[tuple[int, str], Mapping[str, float]]],
) -> str:
    leaderboard = dict(payload)
    seal = _require_seal(leaderboard, context="formal leaderboard")
    seed_fields = {"training_seed", "training_overlay_protocol_seed"}
    present_seed_fields = set(leaderboard) & seed_fields
    if present_seed_fields not in (set(), seed_fields):
        raise ValueError("formal leaderboard seed metadata is partial")
    seeded = present_seed_fields == seed_fields
    expected_keys = {
        "schema_version",
        "leaderboard_type",
        "protocol_id",
        "sample_count",
        "ground_truth_count",
        "unsupported_sample_count",
        "delays_ms",
        "conditions",
        "run_count_per_subject",
        "training_controller_task_id",
        "watcher_task_id",
        "training_manifest_seal_sha256",
        "evaluation_plan_seal_sha256",
        "training_provenance_task_id",
        "training_provenance_seal_sha256",
        "source_revision_equivalence",
        "source_revision_equivalence_seal_sha256",
        "source_revision_subject_map",
        "source_revision_subject_map_seal_sha256",
        "evaluation_source_revision_tree_sha256",
        "evaluation_source_revision",
        "subject_order",
        "subject_count",
        "baseline_subjects",
        "baseline_count",
        "metric_keys",
        "results",
        "leadership",
        "seal_sha256",
    } | (seed_fields if seeded else set())
    _require_exact_keys(leaderboard, expected_keys, context="formal leaderboard")
    expected = {
        "schema_version": 3,
        "leaderboard_type": "resilient_v2x_formal_1337_leaderboard",
        "protocol_id": PROTOCOL_ID,
        "sample_count": SAMPLE_COUNT,
        "ground_truth_count": GROUND_TRUTH_COUNT,
        "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "run_count_per_subject": len(DELAYS_MS) * len(CONDITIONS),
        "training_controller_task_id": controller_task_id,
        "watcher_task_id": watcher_task_id,
        "training_manifest_seal_sha256": manifest_seal,
        "evaluation_plan_seal_sha256": plan_seal,
        "training_provenance_task_id": provenance_binding[
            "training_provenance_task_id"
        ],
        "training_provenance_seal_sha256": provenance_binding[
            "training_provenance_seal_sha256"
        ],
        "source_revision_equivalence": provenance_binding[
            "source_revision_equivalence"
        ],
        "source_revision_equivalence_seal_sha256": provenance_binding[
            "source_revision_equivalence_seal_sha256"
        ],
        "source_revision_subject_map": provenance_binding[
            "source_revision_subject_map"
        ],
        "source_revision_subject_map_seal_sha256": provenance_binding[
            "source_revision_subject_map_seal_sha256"
        ],
        "evaluation_source_revision_tree_sha256": provenance_binding[
            "evaluation_source_revision_tree_sha256"
        ],
        "evaluation_source_revision": provenance_binding[
            "evaluation_source_revision"
        ],
        "subject_order": list(SUBJECT_ORDER),
        "subject_count": len(SUBJECT_ORDER),
        "baseline_subjects": list(BASELINE_SUBJECTS),
        "baseline_count": len(BASELINE_SUBJECTS),
        "metric_keys": list(AP_METRIC_KEYS),
    }
    for key, value in expected.items():
        if leaderboard.get(key) != value:
            raise ValueError(f"formal leaderboard {key} mismatch")
    if seeded and (
        leaderboard.get("training_seed") != TRAINING_SEED
        or leaderboard.get("training_overlay_protocol_seed") != TRAINING_SEED
    ):
        raise ValueError("formal leaderboard seed mismatch")
    results = leaderboard.get("results")
    if not isinstance(results, list) or len(results) != len(SUBJECT_ORDER):
        raise ValueError("formal leaderboard result count mismatch")
    summaries: dict[str, dict[str, dict[str, object]]] = {}
    for index, (result, training, evaluation, subject) in enumerate(
        zip(
            results,
            training_entries,
            evaluation_entries,
            SUBJECT_ORDER,
            strict=True,
        ),
        start=1,
    ):
        if not isinstance(result, Mapping):
            raise ValueError(f"formal leaderboard result {index} is invalid")
        _require_exact_keys(
            result,
            {
                "index",
                "subject",
                "kind",
                "training_task_id",
                "training_model_id",
                "training_checkpoint_sha256",
                "source_revision_tree_sha256",
                "evaluation_task_id",
                "metrics",
            },
            context=f"formal leaderboard result {subject}",
        )
        expected_result = {
            "index": index,
            "subject": subject,
            "kind": SUBJECT_KIND[subject],
            "training_task_id": training["training_task_id"],
            "training_model_id": training["model_id"],
            "training_checkpoint_sha256": training["checkpoint_sha256"],
            "source_revision_tree_sha256": SOURCE_BY_REVISION[
                SOURCE_REVISION_BY_SUBJECT[subject]
            ]["tree_sha256"],
            "evaluation_task_id": evaluation["evaluation_task_id"],
        }
        for key, value in expected_result.items():
            if result.get(key) != value:
                raise ValueError(f"formal leaderboard {subject} {key} mismatch")
        expected_metrics = {
            metric_key: _metric_summary(runs_by_subject[subject], metric_key)
            for metric_key in AP_METRIC_KEYS
        }
        if result.get("metrics") != expected_metrics:
            raise ValueError(f"formal leaderboard {subject} metric summary mismatch")
        summaries[subject] = expected_metrics
    expected_leadership = _expected_leadership(runs_by_subject, summaries)
    if leaderboard.get("leadership") != expected_leadership:
        raise ValueError("formal leaderboard leadership summary mismatch")
    return seal


def _model_inventory(task: object, *, context: str) -> dict[str, object]:
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise RuntimeError(f"{context} cannot expose models")
    models = getter()
    if not isinstance(models, Mapping):
        raise RuntimeError(f"{context} model mapping is invalid")
    result: dict[str, object] = {}
    for key, values in models.items():
        if type(key) is not str:
            raise RuntimeError(f"{context} model mapping key is invalid")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise RuntimeError(f"{context} model sequence {key!r} is invalid")
        records: list[dict[str, str]] = []
        for model in values:
            record: dict[str, str] = {}
            for attribute in ("id", "task", "name", "url"):
                value = getattr(model, attribute, None)
                if value is None:
                    value = ""
                if type(value) is not str:
                    raise RuntimeError(f"{context} model {attribute} must be a string")
                record[attribute] = value
            records.append(record)
        result[key] = records
    copied = _json_copy(result, context=f"{context} model inventory")
    if not isinstance(copied, dict):  # pragma: no cover - guarded above
        raise RuntimeError(f"{context} model inventory is invalid")
    return copied


def _backend_record_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    serializer = getattr(value, "to_dict", None)
    if not callable(serializer):
        raise RuntimeError(f"{context} is not a raw backend record")
    try:
        serialized = serializer()
    except Exception as error:
        raise RuntimeError(f"{context} cannot be serialized") from error
    if not isinstance(serialized, Mapping):
        raise RuntimeError(f"{context} raw serialization is invalid")
    return serialized


def _raw_tag_list(value: object, *, context: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError(f"{context} is not a tag sequence")
    result: list[str] = []
    for tag in value:
        if type(tag) is not str:
            raise RuntimeError(f"{context} contains a non-string tag")
        result.append(tag)
    return result


def _unique_raw_tag_set(value: object, *, context: str) -> frozenset[str]:
    tags = _raw_tag_list(value, context=context)
    unique = frozenset(tags)
    if len(unique) != len(tags):
        raise RuntimeError(f"{context} contains duplicate tags")
    return unique


def _raw_task_authority(value: object, *, context: str) -> dict[str, object]:
    """Return the exact stable contract projection of one backend task record."""
    raw = _backend_record_mapping(value, context=context)
    selected = {field: raw.get(field) for field in RAW_AUTHORITY_FIELDS}
    selected["tags"] = _raw_tag_list(
        selected.get("tags"),
        context=f"{context} tags",
    )
    selected["system_tags"] = _raw_tag_list(
        selected.get("system_tags"),
        context=f"{context} system tags",
    )
    execution = selected.get("execution")
    if execution is None:
        execution = {}
    if not isinstance(execution, Mapping):
        raise RuntimeError(f"{context} execution record is invalid")
    execution_copy = _json_copy(execution, context=f"{context} execution record")
    if not isinstance(execution_copy, dict):  # pragma: no cover - guarded above
        raise RuntimeError(f"{context} execution record is invalid")
    artifacts = execution_copy.pop("artifacts", None) or []
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise RuntimeError(f"{context} raw artifact records are invalid")
    artifact_records: list[dict[str, object]] = []
    artifact_names: set[str] = set()
    for index, artifact in enumerate(artifacts):
        artifact_mapping = _backend_record_mapping(
            artifact,
            context=f"{context} raw artifact record {index}",
        )
        artifact_copy = _json_copy(
            artifact_mapping,
            context=f"{context} raw artifact record {index}",
        )
        if not isinstance(artifact_copy, dict):  # pragma: no cover - guarded above
            raise RuntimeError(f"{context} raw artifact record {index} is invalid")
        name = artifact_copy.get("key")
        if type(name) is not str or not name:
            raise RuntimeError(f"{context} raw artifact record {index} has no key")
        if name in artifact_names:
            raise RuntimeError(f"{context} has duplicate raw artifact {name!r}")
        artifact_names.add(name)
        artifact_records.append(artifact_copy)
    artifact_records.sort(key=lambda record: str(record["key"]))
    selected["execution"] = execution_copy
    selected["artifacts"] = artifact_records
    copied = _json_copy(selected, context=f"{context} raw authority")
    if not isinstance(copied, dict):  # pragma: no cover - guarded above
        raise RuntimeError(f"{context} raw authority is invalid")
    _clearml_id(copied.get("id"), f"{context} raw authority ID")
    return copied


def _batch_authority_snapshot(
    task_class: object,
    task_ids: Sequence[str],
    *,
    context: str,
) -> dict[str, dict[str, object]]:
    """Linearize all formal tasks in one backend GetAll request."""
    ordered_ids = list(task_ids)
    if not ordered_ids or len(set(ordered_ids)) != len(ordered_ids):
        raise RuntimeError(f"{context} task ID set is invalid")
    query = getattr(task_class, "_query_tasks", None)
    if not callable(query):
        raise RuntimeError(f"{context} cannot issue one authoritative batch read")
    try:
        # In the installed SDK, get_tasks(task_ids=...) creates lazy wrappers;
        # _query_tasks is the API path that emits one tasks.GetAllRequest and
        # returns the raw records from that single response.
        records = query(
            task_ids=ordered_ids,
            fetch_only_first_page=True,
            only_fields=list(RAW_AUTHORITY_FIELDS),
            search_hidden=True,
        )
    except Exception as error:
        raise RuntimeError(f"{context} authoritative batch read failed") from error
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise RuntimeError(f"{context} authoritative batch response is invalid")
    result: dict[str, dict[str, object]] = {}
    for index, record in enumerate(records):
        snapshot = _raw_task_authority(
            record,
            context=f"{context} record {index}",
        )
        task_id = str(snapshot["id"])
        if task_id in result:
            raise RuntimeError(f"{context} returned duplicate task {task_id}")
        result[task_id] = snapshot
    if set(result) != set(ordered_ids):
        raise RuntimeError(
            f"{context} task inventory mismatch; "
            f"expected={sorted(ordered_ids)!r}, observed={sorted(result)!r}"
        )
    return {task_id: result[task_id] for task_id in ordered_ids}


def _dependency_snapshot(
    task: object,
    *,
    expected_artifacts: set[str],
    context: str,
) -> dict[str, object]:
    _reload(task, context=context)
    data = _task_data(task, context=context)
    server_artifacts = _server_artifacts(task, context=context)
    observed = set(server_artifacts)
    if observed != expected_artifacts:
        raise RuntimeError(
            f"{context} artifact inventory mismatch; "
            f"expected={sorted(expected_artifacts)!r}, observed={sorted(observed)!r}"
        )
    artifacts = {
        name: _artifact_content_snapshot(
            server_artifacts[name],
            name=name,
            context=context,
        )
        for name in sorted(expected_artifacts)
    }
    script = getattr(data, "script", None)
    if script is None:
        raise RuntimeError(f"{context} has no script metadata")
    script_values: dict[str, object] = {}
    for attribute in ("repository", "working_dir", "entry_point", "diff"):
        value = getattr(script, attribute, None)
        if type(value) is not str:
            raise RuntimeError(f"{context} script {attribute} is not raw text")
        script_values[attribute] = value
    execution = getattr(data, "execution", None)
    queue = getattr(execution, "queue", None)
    if queue is None:
        queue = ""
    if type(queue) is not str:
        raise RuntimeError(f"{context} execution queue is invalid")
    last_worker = getattr(data, "last_worker", None)
    if last_worker is None:
        last_worker = ""
    if type(last_worker) is not str:
        raise RuntimeError(f"{context} last worker is invalid")
    snapshot = {
        "task_id": _task_id(task, context=context),
        "status": _status(task, context=context),
        "parent_task_id": _task_parent(task, context=context),
        "script": script_values,
        "parameters": _parameters(task, context=context),
        "artifacts": artifacts,
        "execution_queue_id": queue,
        "last_worker": last_worker,
        "models": _model_inventory(task, context=context),
        "raw_authority": _raw_task_authority(
            data,
            context=f"{context} installed raw record",
        ),
    }
    if getattr(task, "data", None) is not data:
        raise RuntimeError(f"{context} raw snapshot changed while it was inspected")
    copied = _json_copy(snapshot, context=f"{context} dependency snapshot")
    if not isinstance(copied, dict):  # pragma: no cover - guarded above
        raise RuntimeError(f"{context} dependency snapshot is invalid")
    return copied


def _runtime_source() -> str:
    try:
        source = Path(__file__).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError("cannot read the running comparability producer") from error
    if not source:
        raise RuntimeError("running comparability producer is empty")
    return source


def _output_uri(task: object, *, context: str) -> str:
    value = getattr(
        getattr(_task_data(task, context=context), "output", None),
        "destination",
        None,
    )
    if value != FILES_SERVER_URI:
        raise RuntimeError(f"{context} output destination drifted")
    return value


def _validate_output_inventory(names: Sequence[str]) -> None:
    observed = set(names)
    if len(observed) != len(names) or observed not in (set(), {AUDIT_ARTIFACT}):
        raise RuntimeError("audit output artifact inventory drifted")


def _output_snapshot(
    task: object,
    *,
    task_id: str,
    leaderboard_task_id: str,
    expected_source: str,
    expected_parameters: Mapping[str, object],
) -> dict[str, object]:
    _reload(task, context="audit output")
    data = _task_data(task, context="audit output")
    if _task_id(task, context="audit output") != task_id:
        raise RuntimeError("audit output identity drifted")
    if _task_parent(task, context="audit output") != leaderboard_task_id:
        raise RuntimeError("audit output parent drifted")
    status = _status(task, context="audit output")
    if status not in WAITING_STATUSES:
        raise RuntimeError("audit output status is not writable")
    script_sha = _script_sha256(
        task,
        entry_point=PRODUCER_ENTRY_POINT,
        context="audit output",
    )
    if script_sha != hashlib.sha256(expected_source.encode("utf-8")).hexdigest():
        raise RuntimeError("audit output producer SHA-256 drifted")
    script = getattr(_task_data(task, context="audit output"), "script", None)
    if getattr(script, "diff", None) != expected_source:
        raise RuntimeError("audit output producer bytes drifted")
    parameters = _parameters(task, context="audit output")
    normalized_parameters = dict(parameters)
    for key, expected in OUTPUT_RUNTIME_EMPTY_DEFAULT_PARAMETERS.items():
        if key in expected_parameters:
            continue
        if key not in normalized_parameters:
            continue
        if normalized_parameters.pop(key) != expected:
            raise RuntimeError(f"audit output runtime default {key!r} drifted")
    observed_original_keys = set(normalized_parameters) & set(
        ORIGINAL_QUEUE_RUNTIME_PINS
    )
    if observed_original_keys and observed_original_keys != set(
        ORIGINAL_QUEUE_RUNTIME_PINS
    ):
        raise RuntimeError("audit output original-queue authority is incomplete")
    if observed_original_keys and any(
        normalized_parameters[key] != expected
        for key, expected in ORIGINAL_QUEUE_RUNTIME_PINS.items()
    ):
        raise RuntimeError("audit output original-queue authority drifted")
    expected = dict(expected_parameters)
    if observed_original_keys:
        expected.update(ORIGINAL_QUEUE_RUNTIME_PINS)
    if _canonical_json(normalized_parameters) != _canonical_json(expected):
        raise RuntimeError("audit output Args contract drifted")
    output_uri = _output_uri(task, context="audit output")
    names = _artifact_names(task, context="audit output")
    _validate_output_inventory(names)
    raw_authority = _raw_task_authority(
        data,
        context="audit output installed raw record",
    )
    if getattr(task, "data", None) is not data:
        raise RuntimeError("audit output raw snapshot changed while it was inspected")
    return {
        "task_id": task_id,
        "parent_task_id": leaderboard_task_id,
        "status": status,
        "producer_script_sha256": script_sha,
        "parameters": normalized_parameters,
        "output_uri": output_uri,
        "artifact_names": list(names),
        "raw_authority": raw_authority,
    }


def _read_expected_audit(
    task: object,
    *,
    expected: Mapping[str, object],
) -> dict[str, object]:
    observed = _artifact_mapping(task, AUDIT_ARTIFACT, context="audit output")
    if _canonical_json(observed) != _canonical_json(expected):
        raise RuntimeError("published formal comparability audit bytes drifted")
    return observed


def _publish(
    task: object,
    payload: Mapping[str, object],
    *,
    validate_bindings: Callable[[], list[dict[str, object]]],
    authoritative_readback: Callable[[str], dict[str, dict[str, object]]],
) -> dict[str, dict[str, object]]:
    copied = _json_copy(payload, context="formal comparability audit payload")
    if not isinstance(copied, dict):  # pragma: no cover - guarded above
        raise RuntimeError("formal comparability audit payload is invalid")
    precommit_value = _json_copy(
        validate_bindings(),
        context="precommit dependency bindings",
    )
    if not isinstance(precommit_value, list):  # pragma: no cover - fixed callback
        raise RuntimeError("precommit dependency bindings are invalid")
    precommit = precommit_value
    precommit_authority = _json_copy(
        authoritative_readback("precommit"),
        context="precommit authoritative batch",
    )
    if not isinstance(precommit_authority, dict):  # pragma: no cover - callback
        raise RuntimeError("precommit authoritative batch is invalid")

    def require_precommit_match() -> None:
        current = validate_bindings()
        if _canonical_json(current) != _canonical_json(precommit):
            raise RuntimeError("formal dependency bindings changed after precommit")

    names = _artifact_names(task, context="audit output")
    _validate_output_inventory(names)
    uploader = getattr(task, "upload_artifact", None)
    flusher = getattr(task, "flush", None)
    if not callable(uploader) or not callable(flusher):
        raise RuntimeError("audit output cannot publish artifacts")
    if AUDIT_ARTIFACT not in names:
        require_precommit_match()
        uploaded = uploader(
            AUDIT_ARTIFACT,
            artifact_object=copied,
            wait_on_upload=True,
        )
        if uploaded is not True:
            raise RuntimeError("failed to upload formal comparability audit")
        require_precommit_match()
        flushed = flusher(wait_for_uploads=True)
        if flushed is not None and flushed is not True:
            raise RuntimeError("failed to flush formal comparability audit")
        require_precommit_match()
    # Per-object sweeps retain detailed artifact/content diagnostics. They are
    # not the acceptance boundary: the single committed batch below is the
    # terminal linearized server snapshot for output plus every dependency.
    for _snapshot_pass in range(2):
        require_precommit_match()
        _read_expected_audit(task, expected=copied)
    committed_authority = _json_copy(
        authoritative_readback("committed"),
        context="committed authoritative batch",
    )
    if not isinstance(committed_authority, dict):  # pragma: no cover - callback
        raise RuntimeError("committed authoritative batch is invalid")
    return committed_authority


def _authority_artifact_names(
    snapshot: Mapping[str, object],
    *,
    context: str,
) -> tuple[str, ...]:
    records = snapshot.get("artifacts")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise RuntimeError(f"{context} raw artifact authority is invalid")
    names: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise RuntimeError(f"{context} raw artifact authority {index} is invalid")
        name = record.get("key")
        if type(name) is not str or not name:
            raise RuntimeError(f"{context} raw artifact authority {index} has no key")
        names.append(name)
    return tuple(names)


def _authority_with_tags(
    authority: Mapping[str, object],
    *,
    task_id: str,
    tags: Sequence[str],
    context: str,
) -> dict[str, object]:
    copied = _json_copy(authority, context=context)
    if not isinstance(copied, dict):  # pragma: no cover - guarded by callback
        raise RuntimeError(f"{context} is invalid")
    task_snapshot = copied.get(task_id)
    if not isinstance(task_snapshot, dict):
        raise RuntimeError(f"{context} lacks audit output")
    task_snapshot["tags"] = list(tags)
    return copied


def _commit_formal_tags(
    task: object,
    *,
    task_id: str,
    original_tags: Sequence[str],
    committed_authority: Mapping[str, object],
    batch_readback: Callable[[str], dict[str, dict[str, object]]],
) -> None:
    """Commit tags by server state, compensating every rejected transaction."""
    desired_tags = list(FORMAL_TAGS)
    original = list(original_tags)
    expected = _authority_with_tags(
        committed_authority,
        task_id=task_id,
        tags=desired_tags,
        context="expected tagged authoritative batch",
    )
    tagger = getattr(task, "set_tags", None)
    callback_error: Exception | None = None
    if original != desired_tags:
        if not callable(tagger):
            callback_error = RuntimeError("audit output cannot set formal tags")
        else:
            try:
                # The callback result is advisory. Only a fresh batch read can
                # establish whether the server committed the exact tag set.
                tagger(list(desired_tags))
            except Exception as error:
                callback_error = error

    observed: dict[str, dict[str, object]] | None = None
    readback_error: Exception | None = None
    try:
        observed = batch_readback("post-tag authoritative batch")
    except Exception as error:
        readback_error = error
    if observed is not None and _canonical_json(observed) == _canonical_json(expected):
        return

    if observed is not None:
        output_snapshot = observed.get(task_id)
        if not isinstance(output_snapshot, Mapping):
            failure = RuntimeError("post-tag authoritative batch lacks audit output")
        elif output_snapshot.get("tags") != desired_tags:
            failure = RuntimeError("formal tags were not committed by the server")
        else:
            failure = RuntimeError("formal task bindings changed during tag commit")
    else:
        failure = RuntimeError("formal tags have no fresh authoritative readback")

    restore_callback_error: Exception | None = None
    observed_tags: object = None
    if observed is not None and isinstance(observed.get(task_id), Mapping):
        observed_tags = observed[task_id].get("tags")
    # Once a tag mutation was attempted, always issue the inverse write on a
    # rejected transaction. This also covers delayed commits that were not yet
    # visible in the first failed readback.
    must_restore = original != desired_tags and callable(tagger)
    if must_restore or observed_tags != original:
        if not callable(tagger):
            restore_callback_error = RuntimeError(
                "audit output cannot restore original tags"
            )
        else:
            try:
                tagger(list(original))
            except Exception as error:
                restore_callback_error = error
    compensation_error: Exception | None = None
    try:
        restored = batch_readback("tag compensation authoritative batch")
        restored_output = restored.get(task_id)
        if (
            not isinstance(restored_output, Mapping)
            or restored_output.get("tags") != original
        ):
            raise RuntimeError("original tags were not restored by the server")
    except Exception as error:
        compensation_error = restore_callback_error or error
    if compensation_error is not None:
        raise RuntimeError(
            "formal tag compensation failed; success-tag state is uncertain"
        ) from compensation_error
    cause = readback_error or callback_error
    if cause is not None:
        raise failure from cause
    raise failure


def run(
    args: argparse.Namespace,
    *,
    task_class: object = Task,
    output_task: object | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    queue_name_resolver: Callable[[str], str] | None = None,
) -> dict[str, object]:
    sealed_successor_pins = _sealed_successor_pins(args)
    if (
        type(args.poll_seconds) not in {int, float}
        or not math.isfinite(float(args.poll_seconds))
        or args.poll_seconds <= 0
        or type(args.timeout_hours) not in {int, float}
        or not math.isfinite(float(args.timeout_hours))
        or args.timeout_hours <= 0
    ):
        raise ValueError("poll interval and timeout must be finite and positive")
    controller_task_id = _clearml_id(
        args.training_controller_task_id, "training controller"
    )
    if controller_task_id == SOURCE_REVISION_SOURCE_CONTROLLER_TASK_ID:
        raise ValueError(
            "comparability audit requires the schema-v4 mixed-source successor"
        )
    provenance_task_id = _clearml_id(
        args.training_provenance_task_id, "training provenance"
    )
    watcher_task_id = _clearml_id(args.watcher_task_id, "evaluation watcher")
    leaderboard_task_id = _clearml_id(args.leaderboard_task_id, "leaderboard")
    if (
        len(
            {
                controller_task_id,
                provenance_task_id,
                watcher_task_id,
                leaderboard_task_id,
            }
        )
        != 4
    ):
        raise ValueError(
            "formal controller, provenance, watcher, and leaderboard IDs must differ"
        )
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise RuntimeError("ClearML Task class cannot resolve formal tasks")
    controller = getter(task_id=controller_task_id)
    provenance_task = getter(task_id=provenance_task_id)
    watcher = getter(task_id=watcher_task_id)
    leaderboard_task = getter(task_id=leaderboard_task_id)
    if output_task is None:
        current = getattr(task_class, "current_task", None)
        output_task = current() if callable(current) else None
    if output_task is None:
        raise RuntimeError("formal comparability audit requires a current ClearML task")
    audit_task_id = _clearml_id(
        getattr(output_task, "id", ""),
        "formal comparability audit task",
    )
    if audit_task_id in {
        controller_task_id,
        provenance_task_id,
        watcher_task_id,
        leaderboard_task_id,
    }:
        raise RuntimeError("formal comparability audit task must be outside C2/W2/L2")
    if (
        len(
            {
                id(controller),
                id(provenance_task),
                id(watcher),
                id(leaderboard_task),
                id(output_task),
            }
        )
        != 5
    ):
        raise RuntimeError("formal roots and audit output must not alias")
    runtime_source = _runtime_source()
    expected_output_parameters = {
        "Args/training_controller_task_id": controller_task_id,
        "Args/training_provenance_task_id": provenance_task_id,
        "Args/watcher_task_id": watcher_task_id,
        "Args/leaderboard_task_id": leaderboard_task_id,
        "Args/poll_seconds": str(float(args.poll_seconds)),
        "Args/timeout_hours": str(float(args.timeout_hours)),
        **sealed_successor_pins,
    }
    initial_output_snapshot = _output_snapshot(
        output_task,
        task_id=audit_task_id,
        leaderboard_task_id=leaderboard_task_id,
        expected_source=runtime_source,
        expected_parameters=expected_output_parameters,
    )
    deadline = monotonic_clock() + float(args.timeout_hours) * 3600.0
    _wait_for_completed(
        (
            ("training controller", controller),
            ("training provenance", provenance_task),
            ("evaluation watcher", watcher),
            ("leaderboard", leaderboard_task),
        ),
        deadline=deadline,
        poll_seconds=float(args.poll_seconds),
        monotonic_clock=monotonic_clock,
        sleeper=sleeper,
    )
    seen_task_ids = {audit_task_id}
    seen_task_objects = {id(output_task)}
    bindings: list[tuple[object, set[str], str, dict[str, object]]] = []

    def bind_dependency(
        task: object,
        *,
        expected_task_id: str,
        expected_artifacts: set[str],
        context: str,
    ) -> dict[str, object]:
        if expected_task_id in seen_task_ids or id(task) in seen_task_objects:
            raise RuntimeError(f"{context} aliases another formal task")
        snapshot = _dependency_snapshot(
            task,
            expected_artifacts=expected_artifacts,
            context=context,
        )
        if snapshot["task_id"] != expected_task_id:
            raise RuntimeError(f"{context} identity mismatch")
        seen_task_ids.add(expected_task_id)
        seen_task_objects.add(id(task))
        bindings.append((task, set(expected_artifacts), context, snapshot))
        return snapshot

    bind_dependency(
        controller,
        expected_task_id=controller_task_id,
        expected_artifacts={
            TRAINING_PROGRESS_ARTIFACT,
            TRAINING_MANIFEST_ARTIFACT,
            TRAINING_SUMMARY_ARTIFACT,
        },
        context="training controller",
    )
    bind_dependency(
        provenance_task,
        expected_task_id=provenance_task_id,
        expected_artifacts={TRAINING_PROVENANCE_ARTIFACT},
        context="training provenance",
    )
    bind_dependency(
        watcher,
        expected_task_id=watcher_task_id,
        expected_artifacts={EVALUATION_PLAN_ARTIFACT},
        context="evaluation watcher",
    )
    bind_dependency(
        leaderboard_task,
        expected_task_id=leaderboard_task_id,
        expected_artifacts={LEADERBOARD_ARTIFACT},
        context="leaderboard",
    )

    if (
        _task_parent(provenance_task, context="training provenance")
        != controller_task_id
    ):
        raise RuntimeError("training provenance parent mismatch")
    if _task_parent(watcher, context="evaluation watcher") != provenance_task_id:
        raise RuntimeError("evaluation watcher parent mismatch")
    if _task_parent(leaderboard_task, context="leaderboard") != watcher_task_id:
        raise RuntimeError("leaderboard parent mismatch")
    orchestrators = (
        (
            controller,
            "clearml_5090_training_controller.py",
            CONTROLLER_SCRIPT_SHA256,
            "training controller",
        ),
        (
            watcher,
            "clearml_1337_dependency_watcher.py",
            WATCHER_SCRIPT_SHA256,
            "evaluation watcher",
        ),
        (
            leaderboard_task,
            "clearml_1337_leaderboard.py",
            LEADERBOARD_SCRIPT_SHA256,
            "leaderboard",
        ),
    )
    for task, entry_point, expected_sha, context in orchestrators:
        observed = _script_sha256(task, entry_point=entry_point, context=context)
        if observed != expected_sha:
            raise RuntimeError(f"{context} script SHA-256 mismatch")
    provenance_script_sha = _script_sha256(
        provenance_task,
        entry_point=PROVENANCE_PRODUCER_ENTRY_POINT,
        context="training provenance",
    )
    _require_training_provenance_script_sha256(provenance_script_sha)
    expected_provenance_parameters = {
        "Args/training_controller_task_id": controller_task_id,
        "Args/poll_seconds": "60.0",
        "Args/timeout_hours": "720.0",
    }
    if (
        _parameters(provenance_task, context="training provenance")
        != expected_provenance_parameters
    ):
        raise RuntimeError("training provenance exact Args contract drifted")
    if _unique_raw_tag_set(
        getattr(
            _task_data(provenance_task, context="training provenance"), "tags", None
        ),
        context="training provenance tags",
    ) != _unique_raw_tag_set(
        PROVENANCE_FORMAL_TAGS,
        context="expected training provenance tags",
    ):
        raise RuntimeError("training provenance formal tags drifted")

    controller_parameters = _parameters(controller, context="training controller")
    watcher_parameters = _parameters(watcher, context="evaluation watcher")
    leaderboard_parameters = _parameters(leaderboard_task, context="leaderboard")
    _require_parameters(
        watcher_parameters,
        {
            "Args/training_controller_task_id": controller_task_id,
            "Args/training_provenance_task_id": provenance_task_id,
            "Args/evaluation_template_task_id": EVALUATION_TEMPLATE_TASK_ID,
            "Args/expected_training_provenance_script_sha256": (
                TRAINING_PROVENANCE_SCRIPT_SHA256
            ),
            "Args/expected_eval_script_sha256": EVALUATION_SCRIPT_SHA256,
            "Args/expected_source_dataset_id": SOURCE_DATASET_ID,
            "Args/expected_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "Args/expected_training_dataset_id": TRAINING_DATASET_ID,
        },
        context="evaluation watcher",
    )
    if any(
        watcher_parameters.get(key) not in {None, ""}
        for key in (
            "Args/expected_training_source_dataset_id",
            "Args/expected_training_source_archive_sha256",
        )
    ):
        raise RuntimeError(
            "evaluation watcher has a forbidden single training source pin"
        )
    _require_parameters(
        leaderboard_parameters,
        {
            "Args/training_controller_task_id": controller_task_id,
            "Args/watcher_task_id": watcher_task_id,
        },
        context="leaderboard",
    )
    if sealed_successor_pins:
        _require_parameters(
            watcher_parameters,
            sealed_successor_pins,
            context="evaluation watcher sealed authority",
        )
        _require_parameters(
            leaderboard_parameters,
            sealed_successor_pins,
            context="leaderboard sealed authority",
        )

    template_task_id = EVALUATION_TEMPLATE_TASK_ID
    template = getter(task_id=template_task_id)
    if _task_id(template, context="evaluation template") != template_task_id:
        raise RuntimeError("evaluation template identity mismatch")
    if _status(template, context="training template") != "completed":
        raise RuntimeError("training template is not completed")
    template_script_sha = _script_sha256(
        template,
        entry_point="clearml_5090_bootstrap.py",
        context="training template",
    )
    if template_script_sha != EVALUATION_SCRIPT_SHA256:
        raise RuntimeError("evaluation template script SHA-256 mismatch")
    _require_parameters(
        _parameters(template, context="training template"),
        {
            "Args/source_dataset_id": SOURCE_DATASET_ID,
            "Args/source_archive_name": SOURCE_ARCHIVE_NAME,
            "Args/source_archive_bytes": SOURCE_ARCHIVE_BYTES,
            "Args/source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "Args/training_dataset_id": TRAINING_DATASET_ID,
            "Args/gpus": 4,
            "Args/max_epochs": 50,
            "Args/amp": False,
        },
        context="training template",
    )

    manifest = _artifact_mapping(
        controller, TRAINING_MANIFEST_ARTIFACT, context="training controller"
    )
    training_entries, manifest_seal, manifest_seeded = _validate_manifest(manifest)
    progress = _artifact_mapping(
        controller, TRAINING_PROGRESS_ARTIFACT, context="training controller"
    )
    progress_seal, recovery_sha256 = _validate_recovery_progress(
        progress,
        controller_task_id=controller_task_id,
        training_entries=training_entries,
    )
    recovery = progress.get("recovery")
    if not isinstance(recovery, Mapping):  # pragma: no cover - validator owns shape
        raise RuntimeError("validated training recovery is missing")
    summary = _artifact_mapping(
        controller, TRAINING_SUMMARY_ARTIFACT, context="training controller"
    )
    summary_seal = _validate_summary(
        summary,
        controller_task_id=controller_task_id,
        controller_parameters=controller_parameters,
        manifest=manifest,
        entries=training_entries,
    )
    plan = _artifact_mapping(watcher, EVALUATION_PLAN_ARTIFACT, context="watcher")
    evaluation_entries, plan_seal, plan_provenance_binding = (
        _validate_evaluation_plan(
        plan,
        controller_task_id=controller_task_id,
        evaluation_template_task_id=template_task_id,
        )
    )
    if sealed_successor_pins and plan_seal != sealed_successor_pins[
        "Args/evaluation_plan_amendment_revised_plan_seal_sha256"
    ]:
        raise RuntimeError("amended evaluation plan seal pin mismatch")
    training_ids = {str(entry["training_task_id"]) for entry in training_entries}
    evaluation_ids = {entry["evaluation_task_id"] for entry in evaluation_entries}
    if training_ids & evaluation_ids:
        raise ValueError("training and evaluation task IDs must be disjoint")

    training_records: list[dict[str, object]] = []
    training_source_by_sha: dict[str, str] = {}
    training_script_by_subject: dict[str, str] = {}
    for entry in training_entries:
        task = getter(task_id=str(entry["training_task_id"]))
        training_snapshot = bind_dependency(
            task,
            expected_task_id=str(entry["training_task_id"]),
            expected_artifacts=_training_artifact_inventory(str(entry["subject"])),
            context=f"training task {entry['subject']}",
        )
        record = _validate_training_task(
            task,
            controller_task_id=controller_task_id,
            entry=entry,
            manifest_seeded=manifest_seeded,
            recovery=recovery,
        )
        record["raw_authority_sha256"] = _content_sha256(
            training_snapshot["raw_authority"]
        )
        record["execution_queue_id"] = training_snapshot["execution_queue_id"]
        record["last_worker"] = training_snapshot["last_worker"]
        source = str(record.pop("script_source"))
        script_sha = str(record["script_sha256"])
        previous_source = training_source_by_sha.setdefault(script_sha, source)
        if previous_source != source:
            raise RuntimeError("training script SHA maps to inconsistent source bytes")
        training_script_by_subject[str(entry["subject"])] = script_sha
        training_records.append(record)

    training_script_equivalence = _verify_training_script_equivalence(
        training_source_by_sha, training_script_by_subject
    )
    provenance_payload = _artifact_mapping(
        provenance_task,
        TRAINING_PROVENANCE_ARTIFACT,
        context="training provenance",
    )
    provenance_seal = _validate_training_provenance_payload(
        provenance_payload,
        controller_task_id=controller_task_id,
        progress=progress,
        progress_seal=progress_seal,
        recovery_sha256=recovery_sha256,
        manifest=manifest,
        manifest_seal=manifest_seal,
        training_entries=training_entries,
        script_equivalence=training_script_equivalence,
        training_records=training_records,
    )
    expected_plan_provenance_binding = {
        "training_provenance_task_id": provenance_task_id,
        "training_provenance_seal_sha256": provenance_seal,
        "source_revision_equivalence": provenance_payload[
            "source_revision_equivalence"
        ],
        "source_revision_equivalence_seal_sha256": (
            SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256
        ),
        "source_revision_subject_map": provenance_payload[
            "source_revision_subject_map"
        ],
        "source_revision_subject_map_seal_sha256": (
            SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256
        ),
        "evaluation_source_revision_tree_sha256": SOURCE_TREE_SHA256,
        "evaluation_source_revision": SOURCE_REVISION_CERTIFICATE_BY_TREE[
            SOURCE_TREE_SHA256
        ],
    }
    if plan_provenance_binding != expected_plan_provenance_binding:
        raise RuntimeError(
            "formal evaluation plan/provenance source binding mismatch"
        )

    if queue_name_resolver is None:

        def default_queue_name_resolver(queue_id: str) -> str:
            return _clearml_queue_name(task_class, queue_id)

        queue_name_resolver = default_queue_name_resolver
    queue_name_cache: dict[str, str] = {}

    def resolve_queue_name(queue_id: str) -> str:
        if queue_id not in queue_name_cache:
            queue_name_cache[queue_id] = queue_name_resolver(queue_id)
        return queue_name_cache[queue_id]

    evaluation_records: list[dict[str, object]] = []
    runs_by_subject: dict[str, dict[tuple[int, str], dict[str, float]]] = {}
    for training_entry, evaluation_entry in zip(
        training_entries, evaluation_entries, strict=True
    ):
        task = getter(task_id=evaluation_entry["evaluation_task_id"])
        bind_dependency(
            task,
            expected_task_id=evaluation_entry["evaluation_task_id"],
            expected_artifacts=_evaluation_artifact_inventory(),
            context=f"evaluation task {training_entry['subject']}",
        )
        runs, record = _validate_evaluation_task(
            task,
            controller_task_id=controller_task_id,
            training_entry=training_entry,
            evaluation_entry=evaluation_entry,
            queue_name_resolver=resolve_queue_name,
        )
        subject = str(training_entry["subject"])
        runs_by_subject[subject] = runs
        evaluation_records.append(record)

    leaderboard = _artifact_mapping(
        leaderboard_task, LEADERBOARD_ARTIFACT, context="leaderboard"
    )
    leaderboard_seal = _validate_leaderboard(
        leaderboard,
        controller_task_id=controller_task_id,
        watcher_task_id=watcher_task_id,
        manifest_seal=manifest_seal,
        plan_seal=plan_seal,
        provenance_binding=plan_provenance_binding,
        training_entries=training_entries,
        evaluation_entries=evaluation_entries,
        runs_by_subject=runs_by_subject,
    )
    payload = _sealed(
        {
            "schema_version": 3,
            "document_type": "resilient_v2x_formal_1337_comparability_audit",
            "passed": True,
            "audit_scope": "protocol_and_clearml_metadata_comparability",
            "audit_task_id": audit_task_id,
            "training_controller_task_id": controller_task_id,
            "training_provenance_task_id": provenance_task_id,
            "watcher_task_id": watcher_task_id,
            "leaderboard_task_id": leaderboard_task_id,
            "protocol_id": PROTOCOL_ID,
            "evaluation_source_revision_tree_sha256": SOURCE_TREE_SHA256,
            "evaluation_source_revision": SOURCE_REVISION_CERTIFICATE_BY_TREE[
                SOURCE_TREE_SHA256
            ],
            "source_revision_equivalence": provenance_payload[
                "source_revision_equivalence"
            ],
            "source_revision_equivalence_seal_sha256": (
                SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256
            ),
            "source_revision_subject_map": provenance_payload[
                "source_revision_subject_map"
            ],
            "source_revision_subject_map_seal_sha256": (
                SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256
            ),
            "source_revision_counts": {
                SOURCE_TREE_SHA256: 21,
                NEW_SOURCE_TREE_SHA256: 5,
            },
            "training_dataset_id": TRAINING_DATASET_ID,
            "training_seed": TRAINING_SEED,
            "training_seed_evidence": "all_26_live_run_contracts",
            "legacy_unseeded_training_manifest": not manifest_seeded,
            "legacy_source_c_seed_schema": (
                "explicit_run_contract_seed_only" if not manifest_seeded else None
            ),
            "training_script_equivalence": training_script_equivalence,
            "evaluation_script_sha256": EVALUATION_SCRIPT_SHA256,
            "checkpoint_policy": CHECKPOINT_POLICY,
            "sample_count": SAMPLE_COUNT,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
            "delays_ms": list(DELAYS_MS),
            "conditions": list(CONDITIONS),
            "run_count_per_subject": 12,
            "subject_order": list(SUBJECT_ORDER),
            "subject_count": len(SUBJECT_ORDER),
            "total_evaluation_run_count": len(SUBJECT_ORDER) * 12,
            "metric_keys": list(AP_METRIC_KEYS),
            "final_model_verification_level": ("clearml_metadata_contract_only"),
            "checkpoint_bytes_sha256_recomputed": False,
            "checkpoint_bytes_verifier": MODEL_BYTES_VERIFIER,
            "checkpoint_bytes_verification_dependency": (
                "separate_collect_clearml_formal_models_byte_audit"
            ),
            "training_manifest_seal_sha256": manifest_seal,
            "training_progress_seal_sha256": progress_seal,
            "training_provenance_seal_sha256": provenance_seal,
            "training_summary_seal_sha256": summary_seal,
            "evaluation_plan_seal_sha256": plan_seal,
            "leaderboard_seal_sha256": leaderboard_seal,
            "training_tasks": training_records,
            "evaluation_tasks": evaluation_records,
        }
    )
    if len(bindings) != 4 + 2 * len(SUBJECT_ORDER):
        raise RuntimeError("formal dependency binding count mismatch")

    def capture_binding_sweep(*, phase: str) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for task, expected_artifacts, context, expected in bindings:
            current = _dependency_snapshot(
                task,
                expected_artifacts=expected_artifacts,
                context=f"{context} {phase}",
            )
            if _canonical_json(current) != _canonical_json(expected):
                raise RuntimeError(f"{context} changed during publication")
            result.append({"context": context, "snapshot": current})
        return result

    initial_output_authority = initial_output_snapshot.get("raw_authority")
    if not isinstance(initial_output_authority, dict):  # pragma: no cover - helper
        raise RuntimeError("initial audit output raw authority is invalid")
    latest_output_authority = initial_output_authority

    def validate_bindings() -> list[dict[str, object]]:
        nonlocal latest_output_authority
        before_output = capture_binding_sweep(phase="pre-output binding")
        output_snapshot = _output_snapshot(
            output_task,
            task_id=audit_task_id,
            leaderboard_task_id=leaderboard_task_id,
            expected_source=runtime_source,
            expected_parameters=expected_output_parameters,
        )
        output_raw = output_snapshot.get("raw_authority")
        if not isinstance(output_raw, dict):  # pragma: no cover - helper
            raise RuntimeError("audit output raw authority is invalid")
        latest_output_authority = output_raw
        after_output = capture_binding_sweep(phase="post-output binding")
        if _canonical_json(before_output) != _canonical_json(after_output):
            raise RuntimeError("formal dependencies changed across output validation")
        return after_output

    original_tags = _raw_tag_list(
        initial_output_authority.get("tags"),
        context="initial audit output tags",
    )
    authority_task_ids = [audit_task_id]
    expected_dependency_authority: list[tuple[str, str, Mapping[str, object]]] = []
    for _task, _artifacts, context, expected in bindings:
        expected_task_id = str(expected["task_id"])
        expected_raw = expected.get("raw_authority")
        if not isinstance(expected_raw, Mapping):  # pragma: no cover - helper
            raise RuntimeError(f"{context} raw authority is invalid")
        authority_task_ids.append(expected_task_id)
        expected_dependency_authority.append((expected_task_id, context, expected_raw))

    def batch_readback(context: str) -> dict[str, dict[str, object]]:
        return _batch_authority_snapshot(
            task_class,
            authority_task_ids,
            context=context,
        )

    def authoritative_readback(stage: str) -> dict[str, dict[str, object]]:
        observed = batch_readback(f"{stage} authoritative batch")
        for expected_task_id, context, expected_raw in expected_dependency_authority:
            if _canonical_json(observed[expected_task_id]) != _canonical_json(
                expected_raw
            ):
                raise RuntimeError(f"{context} changed in authoritative batch")
        output_authority = observed[audit_task_id]
        if _canonical_json(output_authority) != _canonical_json(
            latest_output_authority
        ):
            raise RuntimeError(
                f"audit output changed after its last verified {stage} reload"
            )
        if stage == "committed":
            if output_authority.get("tags") != original_tags:
                raise RuntimeError("audit output tags changed before tag commit")
            if _authority_artifact_names(
                output_authority,
                context="committed audit output",
            ) != (AUDIT_ARTIFACT,):
                raise RuntimeError(
                    "audit output lacks authoritative committed artifact"
                )
        elif stage != "precommit":  # pragma: no cover - fixed caller
            raise RuntimeError(f"unsupported authoritative readback stage {stage!r}")
        return observed

    committed_authority = _publish(
        output_task,
        payload,
        validate_bindings=validate_bindings,
        authoritative_readback=authoritative_readback,
    )
    _commit_formal_tags(
        output_task,
        task_id=audit_task_id,
        original_tags=original_tags,
        committed_authority=committed_authority,
        batch_readback=batch_readback,
    )
    return payload


def main() -> int:
    args = _parser().parse_args()
    task = Task.init(
        project_name=DEFAULT_PROJECT,
        task_name="ResilientV2X formal 1337 comparability audit",
        reuse_last_task_id=False,
        output_uri=FILES_SERVER_URI,
    )
    run(args, output_task=task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "AUDIT_ARTIFACT",
    "AP_METRIC_KEYS",
    "SUBJECT_ORDER",
    "run",
)
