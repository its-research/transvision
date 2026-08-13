#!/usr/bin/env python3
"""Build a sealed 26-method leaderboard from completed formal 1337 tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

try:
    from allegroai import Task
except ImportError:
    from clearml import Task


FILES_SERVER_URI = "http://10.100.34.118:8081"
DEFAULT_PROJECT = "ResilientV2X/Training"
TRAINING_MANIFEST_ARTIFACT = "formal_1337_training_manifest"
EVALUATION_PLAN_ARTIFACT = "formal_1337_evaluation_plan"
METRICS_ARTIFACT = "controlled_baseline_metrics"
LEADERBOARD_ARTIFACT = "formal_1337_leaderboard"
PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
SAMPLE_COUNT = 1_337
GROUND_TRUTH_COUNT = 11_330
UNSUPPORTED_SAMPLE_COUNT = 0
DELAYS_MS = (0, 100, 200, 300)
CONDITIONS = ("Full", "L-Fail", "C-Fail")
CHECKPOINT_POLICY = "epoch_50_final_only"
DEFAULT_TRAINING_SEED = 20_250_218
TRAINING_OVERLAY_PROTOCOL_SEED = 20_250_218
RELEASE_SEMANTICS = "formal_manifest_after_full_training_suite_completion"
OLD_SOURCE_DATASET_ID = "4f7fac0078a4419a907fec6ff9e306c8"
OLD_SOURCE_ARCHIVE_NAME = "resilient-v2x-source-5c984ad49b52.tar.zst"
OLD_SOURCE_ARCHIVE_BYTES = 1_222_481
OLD_SOURCE_ARCHIVE_SHA256 = (
    "655f33d9b684c26ff4ff577d5c5f3175d4fc4eac40426eb55820e7a8aa8bc40d"
)
OLD_SOURCE_INVENTORY_BYTES = 101_195
OLD_SOURCE_INVENTORY_SHA256 = (
    "bed72cd86438f2ba932edda14052e3cd3d9589ec201d09d18a315e18a2f7cff2"
)
OLD_SOURCE_FILE_COUNT = 631
OLD_SOURCE_BYTES = 8_926_106
OLD_SOURCE_TREE_SHA256 = (
    "5c984ad49b5232d7f6d053fb641895283477efcbf2de40b36d9b3f3c6f8e28b6"
)
NEW_SOURCE_DATASET_ID = "351feedbbe81481fa31f1e9ae11a3f4e"
NEW_SOURCE_ARCHIVE_NAME = "resilient-v2x-source-ad511d88b731.tar.zst"
NEW_SOURCE_ARCHIVE_BYTES = 1_222_492
NEW_SOURCE_ARCHIVE_SHA256 = (
    "b94a01c2acf2cc456fe9729f7c40e990e6d44b65e789c6fed11989a673f4f6da"
)
NEW_SOURCE_INVENTORY_BYTES = 101_195
NEW_SOURCE_INVENTORY_SHA256 = (
    "39b1a42af65ad5df935945bd0a4eeac6e7f6e1cfdd1cc608f3dd8a708e9c5ca0"
)
NEW_SOURCE_FILE_COUNT = 631
NEW_SOURCE_BYTES = 8_926_102
NEW_SOURCE_TREE_SHA256 = (
    "ad511d88b731cb45ef2defb873712bdb2a325c648634b66c349fe1c1459510e4"
)
SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256 = (
    "29de9700cac66f9998be643e85a8ec646c04ec17fddbb6207bc1438e9e73941b"
)
SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256 = (
    "c3170f4a88b080f9cc7267053f354640dd260687cb2c35a6f7ffed73d69f4154"
)
MANIFEST_CONTENT_SHA256 = (
    "715ac6f7a14225e20327eed0650c55abdc0cb98431830164e84545238099645d"
)
OVERLAY_INDEX_CONTENT_SHA256 = (
    "77bd4585dbb02901f862b8da6aa208a504674b824a3d55cf15005aacbeeeaaff"
)
SAMPLE_IDS_SHA256 = "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"
CLEARML_ID_LENGTH = 32
SHA256_LENGTH = 64
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

FORMAL_SUBJECT_ORDER = (
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
    for subject in FORMAL_SUBJECT_ORDER
}
SOURCE_TREE_BY_REVISION = {
    "old": OLD_SOURCE_TREE_SHA256,
    "new": NEW_SOURCE_TREE_SHA256,
}
SOURCE_REVISION_CERTIFICATE_BY_TREE = {
    OLD_SOURCE_TREE_SHA256: {
        "dataset_id": OLD_SOURCE_DATASET_ID,
        "tree_sha256": OLD_SOURCE_TREE_SHA256,
        "file_count": OLD_SOURCE_FILE_COUNT,
        "source_bytes": OLD_SOURCE_BYTES,
        "archive": {
            "name": OLD_SOURCE_ARCHIVE_NAME,
            "size_bytes": OLD_SOURCE_ARCHIVE_BYTES,
            "sha256": OLD_SOURCE_ARCHIVE_SHA256,
        },
        "inventory": {
            "name": "source-inventory.json",
            "size_bytes": OLD_SOURCE_INVENTORY_BYTES,
            "sha256": OLD_SOURCE_INVENTORY_SHA256,
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
    for subject in FORMAL_SUBJECT_ORDER
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watcher-task-id", required=True)
    parser.add_argument("--training-controller-task-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=168.0)
    for name, _kind in SEALED_SUCCESSOR_PIN_ARGS:
        parser.add_argument(f"--{name.replace('_', '-')}", default="")
    return parser


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _seal(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = hashlib.sha256(
        _canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _task_id(value: object, context: str) -> str:
    result = str(value or "")
    if not _is_lower_hex(result, CLEARML_ID_LENGTH):
        raise ValueError(f"{context} must be a lowercase 32-hex ClearML ID")
    return result


def _sha256(value: object, context: str) -> str:
    result = str(value or "")
    if not _is_lower_hex(result, SHA256_LENGTH):
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
            _task_id(raw[name], name)
            if kind == "task_id"
            else _sha256(raw[name], name)
        )
        result[f"Args/{name}"] = value
    return result


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{context} keys mismatch; missing={missing!r}, extra={extra!r}"
        )


def _require_seal(value: Mapping[str, object], *, context: str) -> str:
    observed = _sha256(value.get("seal_sha256"), f"{context} seal")
    if _seal(value)["seal_sha256"] != observed:
        raise ValueError(f"{context} seal SHA-256 mismatch")
    return observed


def _artifact_mapping(task: object, name: str, *, context: str) -> Mapping[str, object]:
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, Mapping) or name not in artifacts:
        raise RuntimeError(f"{context} lacks artifact {name!r}")
    getter = getattr(artifacts[name], "get", None)
    if not callable(getter):
        raise RuntimeError(f"{context} artifact {name!r} cannot be read")
    value = getter()
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, (str, Path)):
        raise RuntimeError(f"{context} artifact {name!r} is not a JSON object")
    path = Path(value)
    try:
        if not path.is_file():
            raise RuntimeError(
                f"{context} artifact {name!r} path is not a regular file"
            )
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"{context} artifact {name!r} path cannot be read as JSON"
        ) from error
    if not isinstance(document, Mapping):
        raise RuntimeError(f"{context} artifact {name!r} is not a JSON object")
    return document


def _parameters(task: object, *, context: str) -> Mapping[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise RuntimeError(f"{context} cannot expose parameters")
    try:
        value = getter(cast=False)
    except TypeError:
        value = getter()
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} parameters are invalid")
    return value


def _status(task: object) -> str:
    reloader = getattr(task, "reload", None)
    if callable(reloader):
        reloader()
    return str(getattr(task, "status", "") or "").lower()


def _wait_for_completed_tasks(
    tasks: Sequence[tuple[str, object]],
    *,
    deadline: float,
    poll_seconds: float,
    monotonic: Callable[[], float],
    sleeper: Callable[[float], None],
) -> None:
    while True:
        pending = False
        for context, task in tasks:
            status = _status(task)
            if status == "completed":
                continue
            if status in FAILED_STATUSES:
                raise RuntimeError(f"{context} ended as {status!r}")
            if status not in WAITING_STATUSES:
                raise RuntimeError(f"{context} has unexpected status {status!r}")
            pending = True
        if not pending:
            return
        if monotonic() >= deadline:
            raise TimeoutError("timed out waiting for formal 1337 dependencies")
        sleeper(poll_seconds)


def _parse_training_manifest(
    value: Mapping[str, object],
) -> tuple[list[dict[str, object]], str]:
    seal = _require_seal(value, context="formal 1337 training manifest")
    seed_keys = {"training_seed", "training_overlay_protocol_seed"}
    observed_seed_keys = set(value) & seed_keys
    if observed_seed_keys not in (set(), seed_keys):
        raise ValueError(
            "formal 1337 training manifest seed metadata must be all-or-none"
        )
    has_seed_metadata = observed_seed_keys == seed_keys
    _require_exact_keys(
        value,
        {
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
        }
        | (seed_keys if has_seed_metadata else set()),
        context="formal 1337 training manifest",
    )
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
        "subject_order": list(FORMAL_SUBJECT_ORDER),
        "subject_count": len(FORMAL_SUBJECT_ORDER),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"formal 1337 training manifest {key} mismatch")
    if has_seed_metadata:
        training_seed = value.get("training_seed")
        if type(training_seed) is not int or training_seed < 0:
            raise ValueError("formal 1337 training manifest training seed is invalid")
        if value.get("training_overlay_protocol_seed") != (
            TRAINING_OVERLAY_PROTOCOL_SEED
        ):
            raise ValueError(
                "formal 1337 training manifest overlay protocol seed mismatch"
            )
    else:
        training_seed = None
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != len(FORMAL_SUBJECT_ORDER):
        raise ValueError("formal 1337 training manifest entry count mismatch")
    result: list[dict[str, object]] = []
    task_ids: set[str] = set()
    model_ids: set[str] = set()
    for index, (entry, subject) in enumerate(
        zip(entries, FORMAL_SUBJECT_ORDER, strict=True),
        start=1,
    ):
        if not isinstance(entry, Mapping):
            raise ValueError(f"training manifest entry {index} is not an object")
        observed_entry_seed_keys = set(entry) & seed_keys
        expected_entry_seed_keys = seed_keys if has_seed_metadata else set()
        if observed_entry_seed_keys != expected_entry_seed_keys:
            raise ValueError(
                f"training manifest entry {subject} seed metadata mismatch"
            )
        _require_exact_keys(
            entry,
            {
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
            | expected_entry_seed_keys,
            context=f"training manifest entry {subject}",
        )
        if (
            entry.get("index") != index
            or entry.get("subject") != subject
            or entry.get("kind") != SUBJECT_KIND[subject]
        ):
            raise ValueError(f"training manifest entry {index} identity mismatch")
        task_id = _task_id(entry.get("training_task_id"), f"{subject} training task")
        predecessor_task_id = _task_id(
            entry.get("training_predecessor_task_id"),
            f"{subject} training predecessor",
        )
        model_id = _task_id(entry.get("model_id"), f"{subject} final model")
        if task_id in task_ids or model_id in model_ids:
            raise ValueError("formal training task/model IDs must be unique")
        task_ids.add(task_id)
        model_ids.add(model_id)
        expected_name = f"ResilientV2X {subject} final checkpoint"
        if entry.get("model_name") != expected_name:
            raise ValueError(f"{subject} final model name mismatch")
        model_url = str(entry.get("model_url") or "")
        parsed = urlparse(model_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"{subject} final model URL is invalid")
        if PurePosixPath(unquote(parsed.path)).name != f"{subject}_epoch_50.pth":
            raise ValueError(f"{subject} final model filename mismatch")
        if entry.get("checkpoint_filename") != "epoch_50.pth":
            raise ValueError(f"{subject} checkpoint policy mismatch")
        checkpoint_sha256 = _sha256(
            entry.get("checkpoint_sha256"), f"{subject} checkpoint"
        )
        size = entry.get("checkpoint_size_bytes")
        if type(size) is not int or size <= 0:
            raise ValueError(f"{subject} checkpoint size is invalid")
        if entry.get("common_teacher_initialization_audit_artifact") != (
            "common_teacher_initialization_audit"
        ):
            raise ValueError(f"{subject} initialization audit artifact mismatch")
        audit_sha256 = _sha256(
            entry.get("common_teacher_initialization_audit_sha256"),
            f"{subject} initialization audit",
        )
        parsed_entry = {
            "index": index,
            "subject": subject,
            "kind": SUBJECT_KIND[subject],
            "training_task_id": task_id,
            "training_predecessor_task_id": predecessor_task_id,
            "model_id": model_id,
            "model_name": expected_name,
            "model_url": model_url,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size_bytes": size,
            "initialization_audit_sha256": audit_sha256,
        }
        if has_seed_metadata:
            if entry.get("training_seed") != training_seed:
                raise ValueError(f"{subject} training seed mismatch")
            if entry.get("training_overlay_protocol_seed") != (
                TRAINING_OVERLAY_PROTOCOL_SEED
            ):
                raise ValueError(f"{subject} overlay protocol seed mismatch")
            parsed_entry.update(
                {
                    "training_seed": training_seed,
                    "training_overlay_protocol_seed": (TRAINING_OVERLAY_PROTOCOL_SEED),
                }
            )
        result.append(parsed_entry)
    return result, seal


def _parse_evaluation_plan(
    value: Mapping[str, object],
    *,
    controller_task_id: str,
) -> tuple[list[dict[str, str]], str, dict[str, object]]:
    seal = _require_seal(value, context="formal 1337 evaluation plan")
    _require_exact_keys(
        value,
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
        context="formal 1337 evaluation plan",
    )
    expected = {
        "schema_version": 2,
        "plan_type": "resilient_v2x_formal_1337_evaluation_tasks",
        "training_controller_task_id": controller_task_id,
        "protocol_id": PROTOCOL_ID,
        "sample_count": SAMPLE_COUNT,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "run_count": len(DELAYS_MS) * len(CONDITIONS),
        "subject_order": list(FORMAL_SUBJECT_ORDER),
        "evaluation_source_revision_tree_sha256": OLD_SOURCE_TREE_SHA256,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"formal 1337 evaluation plan {key} mismatch")
    _task_id(value.get("evaluation_template_task_id"), "evaluation template task")
    provenance_task_id = _task_id(
        value.get("training_provenance_task_id"), "training provenance task"
    )
    provenance_seal = _sha256(
        value.get("training_provenance_seal_sha256"), "training provenance seal"
    )
    source_equivalence = value.get("source_revision_equivalence")
    if not isinstance(source_equivalence, Mapping):
        raise ValueError("source revision equivalence is not an object")
    source_equivalence_seal = _require_seal(
        source_equivalence, context="source revision equivalence"
    )
    if (
        source_equivalence_seal != SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256
        or value.get("source_revision_equivalence_seal_sha256")
        != source_equivalence_seal
    ):
        raise ValueError("source revision equivalence seal mismatch")
    source_revisions = source_equivalence.get("source_revisions")
    if not isinstance(source_revisions, Mapping) or dict(source_revisions) != (
        SOURCE_REVISION_CERTIFICATE_BY_TREE
    ):
        raise ValueError("source revision certificate inventory mismatch")
    source_subject_map = value.get("source_revision_subject_map")
    if not isinstance(source_subject_map, Mapping):
        raise ValueError("source revision subject map is not an object")
    source_subject_map_seal = _require_seal(
        source_subject_map, context="source revision subject map"
    )
    if (
        source_subject_map_seal != SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256
        or value.get("source_revision_subject_map_seal_sha256")
        != source_subject_map_seal
    ):
        raise ValueError("source revision subject map seal mismatch")
    expected_subject_map = {
        subject: SOURCE_TREE_BY_REVISION[SOURCE_REVISION_BY_SUBJECT[subject]]
        for subject in FORMAL_SUBJECT_ORDER
    }
    if source_subject_map.get("source_revision_by_subject") != expected_subject_map:
        raise ValueError("source revision subject split mismatch")
    evaluation_source_revision = value.get("evaluation_source_revision")
    if (
        not isinstance(evaluation_source_revision, Mapping)
        or dict(evaluation_source_revision)
        != SOURCE_REVISION_CERTIFICATE_BY_TREE[OLD_SOURCE_TREE_SHA256]
        or source_revisions.get(OLD_SOURCE_TREE_SHA256)
        != evaluation_source_revision
    ):
        raise ValueError("evaluation source revision certificate mismatch")
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != len(FORMAL_SUBJECT_ORDER):
        raise ValueError("formal 1337 evaluation plan entry count mismatch")
    result: list[dict[str, str]] = []
    evaluation_ids: set[str] = set()
    for index, (entry, subject) in enumerate(
        zip(entries, FORMAL_SUBJECT_ORDER, strict=True),
        start=1,
    ):
        if not isinstance(entry, Mapping):
            raise ValueError(f"evaluation plan entry {index} is not an object")
        _require_exact_keys(
            entry,
            {"subject", "evaluation_task_id", "queue"},
            context=f"evaluation plan entry {subject}",
        )
        if entry.get("subject") != subject:
            raise ValueError(f"evaluation plan entry {index} subject mismatch")
        evaluation_task_id = _task_id(
            entry.get("evaluation_task_id"), f"{subject} evaluation task"
        )
        if evaluation_task_id in evaluation_ids:
            raise ValueError("formal evaluation task IDs must be unique")
        evaluation_ids.add(evaluation_task_id)
        queue = str(entry.get("queue") or "")
        if queue not in SUPPORTED_QUEUES:
            raise ValueError(f"{subject} evaluation queue is unsupported")
        result.append(
            {
                "subject": subject,
                "evaluation_task_id": evaluation_task_id,
                "queue": queue,
            }
        )
    return (
        result,
        seal,
        {
            "training_provenance_task_id": provenance_task_id,
            "training_provenance_seal_sha256": provenance_seal,
            "source_revision_equivalence": dict(source_equivalence),
            "source_revision_equivalence_seal_sha256": source_equivalence_seal,
            "source_revision_subject_map": dict(source_subject_map),
            "source_revision_subject_map_seal_sha256": source_subject_map_seal,
            "evaluation_source_revision_tree_sha256": OLD_SOURCE_TREE_SHA256,
            "evaluation_source_revision": dict(evaluation_source_revision),
        },
    )


def _require_evaluation_join(
    task: object,
    *,
    subject: str,
    evaluation_task_id: str,
    training: Mapping[str, object],
) -> None:
    if _task_id(getattr(task, "id", ""), f"{subject} evaluation task") != (
        evaluation_task_id
    ):
        raise RuntimeError(f"{subject} evaluation task ID mismatch")
    if _status(task) != "completed":
        raise RuntimeError(f"{subject} evaluation task is not completed")
    parameters = _parameters(task, context=f"{subject} evaluation task")
    expected_parameters = {
        "Args/stage": "baseline_validate",
        "Args/source_dataset_id": OLD_SOURCE_DATASET_ID,
        "Args/source_archive_name": OLD_SOURCE_ARCHIVE_NAME,
        "Args/source_archive_bytes": str(OLD_SOURCE_ARCHIVE_BYTES),
        "Args/source_archive_sha256": OLD_SOURCE_ARCHIVE_SHA256,
        "Args/controlled_baseline": subject,
        "Args/controlled_baseline_task_id": training["training_task_id"],
        "Args/controlled_baseline_model_id": training["model_id"],
        "Args/controlled_baseline_checkpoint_sha256": training["checkpoint_sha256"],
        "Args/predecessor_task_id": training["training_task_id"],
        "Args/gpus": "4",
        "Args/max_epochs": "50",
        "Args/amp": "False",
    }
    for key, expected in expected_parameters.items():
        if str(parameters.get(key)) != str(expected):
            raise RuntimeError(f"{subject} evaluation parameter {key} mismatch")
    models_getter = getattr(task, "get_models", None)
    if not callable(models_getter):
        raise RuntimeError(f"{subject} evaluation task cannot expose input models")
    models = models_getter()
    if not isinstance(models, Mapping):
        raise RuntimeError(f"{subject} evaluation task model mapping is invalid")
    input_models = models.get("input")
    if not isinstance(input_models, Sequence) or isinstance(input_models, (str, bytes)):
        raise RuntimeError(f"{subject} evaluation input models are invalid")
    input_model_ids = [
        _task_id(getattr(model, "id", ""), f"{subject} evaluation input model")
        for model in input_models
    ]
    if input_model_ids != [training["model_id"]]:
        raise RuntimeError(f"{subject} evaluation input model mismatch")


def _require_ap(value: object, *, context: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise ValueError(f"{context} must be finite and within [0, 100]")
    return result


def _parse_metrics(
    value: Mapping[str, object],
    *,
    subject: str,
    checkpoint_sha256: str,
) -> dict[tuple[int, str], dict[str, float]]:
    _require_exact_keys(
        value,
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
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"{subject} metrics {key} mismatch")
    if type(value.get("checkpoint")) is not str or not value["checkpoint"]:
        raise ValueError(f"{subject} metrics checkpoint path is invalid")
    runs = value.get("runs")
    if not isinstance(runs, list) or len(runs) != len(DELAYS_MS) * len(CONDITIONS):
        raise ValueError(f"{subject} metrics run count mismatch")
    result: dict[tuple[int, str], dict[str, float]] = {}
    for index, (run, expected_pair) in enumerate(
        zip(
            runs,
            ((delay, condition) for delay in DELAYS_MS for condition in CONDITIONS),
            strict=True,
        ),
        start=1,
    ):
        if not isinstance(run, Mapping):
            raise ValueError(f"{subject} metrics run {index} is not an object")
        _require_exact_keys(
            run,
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
        delay, condition = expected_pair
        if (
            run.get("delay_ms") != delay
            or run.get("condition") != condition
            or run.get("condition_id")
            != f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        ):
            raise ValueError(f"{subject} metrics run {index} order mismatch")
        run_expected = {
            "sample_count": SAMPLE_COUNT,
            "ground_truth_count": GROUND_TRUTH_COUNT,
            "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
            "sample_ids_sha256": SAMPLE_IDS_SHA256,
        }
        for key, expected_value in run_expected.items():
            if run.get(key) != expected_value:
                raise ValueError(f"{subject} metrics run {index} {key} mismatch")
        _sha256(run.get("prediction_sha256"), f"{subject} run prediction")
        _sha256(
            run.get("prediction_content_sha256"),
            f"{subject} run prediction content",
        )
        if type(run.get("predictions")) is not str or not run["predictions"]:
            raise ValueError(f"{subject} metrics run {index} predictions path invalid")
        metrics = run.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{subject} metrics run {index} has no metric object")
        for count_key, expected_count in COUNT_METRICS.items():
            raw_count = metrics.get(count_key)
            if (
                type(raw_count) not in {int, float}
                or not math.isfinite(float(raw_count))
                or not float(raw_count).is_integer()
                or int(raw_count) != expected_count
            ):
                raise ValueError(f"{subject} metrics run {index} {count_key} mismatch")
        result[(delay, condition)] = {
            key: _require_ap(
                metrics.get(key),
                context=f"{subject} run {index} metric {key}",
            )
            for key in AP_METRIC_KEYS
        }
    return result


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _metric_summary(
    runs: Mapping[tuple[int, str], Mapping[str, float]],
    metric_key: str,
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
        "mean_12": _mean([item[2] for item in ordered]),
        "worst_12": {
            "value": worst_value,
            "delay_ms": worst_delay,
            "condition": worst_condition,
        },
        "full_mean": _mean([runs[(delay, "Full")][metric_key] for delay in DELAYS_MS]),
        "l_fail_mean": _mean(
            [runs[(delay, "L-Fail")][metric_key] for delay in DELAYS_MS]
        ),
        "c_fail_mean": _mean(
            [runs[(delay, "C-Fail")][metric_key] for delay in DELAYS_MS]
        ),
        "full_300": full_300,
        "pdr": None if clean == 0.0 else 100.0 * (clean - full_300) / clean,
    }


def _leadership_dimension(
    results_by_subject: Mapping[str, Mapping[str, object]],
    *,
    dimension: str,
) -> dict[str, object]:
    def value_for(subject: str) -> float:
        metric_summary = results_by_subject[subject]["metrics"][LEADERSHIP_METRIC]
        if dimension == "worst_12":
            return float(metric_summary[dimension]["value"])
        return float(metric_summary[dimension])

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
    *,
    delay_ms: int,
    condition: str,
) -> dict[str, object]:
    baseline_values = {
        subject: runs_by_subject[subject][(delay_ms, condition)][LEADERSHIP_METRIC]
        for subject in BASELINE_SUBJECTS
    }
    best_value = max(baseline_values.values())
    best_subjects = [
        subject
        for subject in BASELINE_SUBJECTS
        if baseline_values[subject] == best_value
    ]
    primary_value = runs_by_subject[PRIMARY_SUBJECT][(delay_ms, condition)][
        LEADERSHIP_METRIC
    ]
    margin = primary_value - best_value
    return {
        "condition_id": (f"delay_{delay_ms:03d}_{condition.lower().replace('-', '_')}"),
        "delay_ms": delay_ms,
        "condition": condition,
        "resilient_v2x_value": primary_value,
        "best_baseline_value": best_value,
        "best_baseline_subject": best_subjects[0],
        "best_baseline_subjects": best_subjects,
        "margin": margin,
        "leads": margin > 0.0,
    }


def build_leaderboard(
    *,
    task_class: object,
    controller_task_id: str,
    watcher_task_id: str,
    training_manifest: Mapping[str, object],
    evaluation_plan: Mapping[str, object],
) -> dict[str, object]:
    training, training_seal = _parse_training_manifest(training_manifest)
    evaluations, evaluation_seal, provenance_binding = _parse_evaluation_plan(
        evaluation_plan,
        controller_task_id=controller_task_id,
    )
    results: list[dict[str, object]] = []
    runs_by_subject: dict[str, dict[tuple[int, str], dict[str, float]]] = {}
    for training_entry, evaluation_entry in zip(training, evaluations, strict=True):
        subject = str(training_entry["subject"])
        if evaluation_entry["subject"] != subject:
            raise ValueError(f"{subject} training/evaluation subject mismatch")
        evaluation_task_id = evaluation_entry["evaluation_task_id"]
        evaluation_task = task_class.get_task(task_id=evaluation_task_id)
        _require_evaluation_join(
            evaluation_task,
            subject=subject,
            evaluation_task_id=evaluation_task_id,
            training=training_entry,
        )
        metrics = _artifact_mapping(
            evaluation_task,
            METRICS_ARTIFACT,
            context=f"{subject} evaluation task",
        )
        runs = _parse_metrics(
            metrics,
            subject=subject,
            checkpoint_sha256=str(training_entry["checkpoint_sha256"]),
        )
        runs_by_subject[subject] = runs
        results.append(
            {
                "index": training_entry["index"],
                "subject": subject,
                "kind": training_entry["kind"],
                "training_task_id": training_entry["training_task_id"],
                "training_model_id": training_entry["model_id"],
                "training_checkpoint_sha256": training_entry["checkpoint_sha256"],
                "source_revision_tree_sha256": SOURCE_TREE_BY_REVISION[
                    SOURCE_REVISION_BY_SUBJECT[subject]
                ],
                "evaluation_task_id": evaluation_task_id,
                "metrics": {
                    metric_key: _metric_summary(runs, metric_key)
                    for metric_key in AP_METRIC_KEYS
                },
            }
        )
    results_by_subject = {str(item["subject"]): item for item in results}
    aggregate_dimensions = {
        dimension: _leadership_dimension(
            results_by_subject,
            dimension=dimension,
        )
        for dimension in ("clean_0ms", "mean_12", "worst_12")
    }
    directional_summary_dimensions = {
        dimension: _leadership_dimension(
            results_by_subject,
            dimension=dimension,
        )
        for dimension in ("full_mean", "l_fail_mean", "c_fail_mean", "full_300")
    }
    condition_comparisons = [
        _leadership_condition(
            runs_by_subject,
            delay_ms=delay_ms,
            condition=condition,
        )
        for delay_ms in DELAYS_MS
        for condition in CONDITIONS
    ]
    conditions_won = sum(bool(item["leads"]) for item in condition_comparisons)
    has_seed_metadata = "training_seed" in training[0]
    payload: dict[str, object] = {
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
        "training_manifest_seal_sha256": training_seal,
        "evaluation_plan_seal_sha256": evaluation_seal,
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
        "subject_order": list(FORMAL_SUBJECT_ORDER),
        "subject_count": len(FORMAL_SUBJECT_ORDER),
        "baseline_subjects": list(BASELINE_SUBJECTS),
        "baseline_count": len(BASELINE_SUBJECTS),
        "metric_keys": list(AP_METRIC_KEYS),
        "results": results,
        "leadership": {
            "metric_key": LEADERSHIP_METRIC,
            "comparison_pool": list(BASELINE_SUBJECTS),
            "aggregate_dimensions": aggregate_dimensions,
            "leads_all_aggregate": all(
                bool(item["leads"]) for item in aggregate_dimensions.values()
            ),
            "directional_summary_dimensions": directional_summary_dimensions,
            "condition_comparisons": condition_comparisons,
            "conditions_won": conditions_won,
            "conditions_total": len(condition_comparisons),
            "conditions_won_fraction": (
                f"{conditions_won}/{len(condition_comparisons)}"
            ),
            "leads_all_conditions": conditions_won == len(condition_comparisons),
        },
    }
    if has_seed_metadata:
        payload.update(
            {
                "training_seed": training[0]["training_seed"],
                "training_overlay_protocol_seed": training[0][
                    "training_overlay_protocol_seed"
                ],
            }
        )
    return _seal(payload)


def _publish(task: object, payload: Mapping[str, object]) -> None:
    artifacts = getattr(task, "artifacts", None)
    if isinstance(artifacts, Mapping) and LEADERBOARD_ARTIFACT in artifacts:
        observed = _artifact_mapping(
            task,
            LEADERBOARD_ARTIFACT,
            context="leaderboard task",
        )
        if observed != payload:
            raise RuntimeError("existing formal 1337 leaderboard artifact drifted")
        return
    uploader = getattr(task, "upload_artifact", None)
    if not callable(uploader) or not uploader(
        LEADERBOARD_ARTIFACT,
        artifact_object=dict(payload),
        wait_on_upload=True,
    ):
        raise RuntimeError("failed to publish formal 1337 leaderboard")
    flusher = getattr(task, "flush", None)
    if callable(flusher):
        flusher(wait_for_uploads=True)


def run(
    args: argparse.Namespace,
    *,
    task_class: object = Task,
    output_task: object | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
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
    watcher_task_id = _task_id(args.watcher_task_id, "watcher task")
    controller_task_id = _task_id(
        args.training_controller_task_id,
        "training controller task",
    )
    if watcher_task_id == controller_task_id:
        raise ValueError("watcher and training controller task IDs must differ")
    if output_task is None:
        current_getter = getattr(task_class, "current_task", None)
        output_task = current_getter() if callable(current_getter) else None
    if output_task is None:
        raise RuntimeError("leaderboard requires a current ClearML task")
    tagger = getattr(output_task, "set_tags", None)
    if callable(tagger):
        tagger(
            [
                "ResilientV2X-suite",
                "formal-1337-leaderboard",
                PROTOCOL_ID,
                "cpu-controller",
            ]
        )
    controller = task_class.get_task(task_id=controller_task_id)
    watcher = task_class.get_task(task_id=watcher_task_id)
    deadline = monotonic() + float(args.timeout_hours) * 3600.0
    _wait_for_completed_tasks(
        (
            ("training controller", controller),
            ("evaluation watcher", watcher),
        ),
        deadline=deadline,
        poll_seconds=float(args.poll_seconds),
        monotonic=monotonic,
        sleeper=sleeper,
    )
    if _task_id(getattr(controller, "id", ""), "training controller task") != (
        controller_task_id
    ):
        raise RuntimeError("training controller task identity mismatch")
    if _task_id(getattr(watcher, "id", ""), "watcher task") != watcher_task_id:
        raise RuntimeError("watcher task identity mismatch")
    training_manifest = _artifact_mapping(
        controller,
        TRAINING_MANIFEST_ARTIFACT,
        context="training controller",
    )
    evaluation_plan = _artifact_mapping(
        watcher,
        EVALUATION_PLAN_ARTIFACT,
        context="evaluation watcher",
    )
    if sealed_successor_pins:
        watcher_parameters = _parameters(watcher, context="evaluation watcher")
        for key, expected in sealed_successor_pins.items():
            if str(watcher_parameters.get(key) or "") != expected:
                raise RuntimeError(f"evaluation watcher parameter {key!r} mismatch")
        if _require_seal(
            evaluation_plan, context="formal evaluation plan"
        ) != sealed_successor_pins[
            "Args/evaluation_plan_amendment_revised_plan_seal_sha256"
        ]:
            raise RuntimeError("amended evaluation plan seal pin mismatch")
    payload = build_leaderboard(
        task_class=task_class,
        controller_task_id=controller_task_id,
        watcher_task_id=watcher_task_id,
        training_manifest=training_manifest,
        evaluation_plan=evaluation_plan,
    )
    _publish(output_task, payload)
    return payload


def main() -> int:
    args = _parser().parse_args()
    task = Task.init(
        project_name=DEFAULT_PROJECT,
        task_name="ResilientV2X formal 1337 leaderboard",
        reuse_last_task_id=False,
        output_uri=FILES_SERVER_URI,
    )
    run(args, output_task=task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "AP_METRIC_KEYS",
    "BASELINE_SUBJECTS",
    "FORMAL_SUBJECT_ORDER",
    "LEADERBOARD_ARTIFACT",
    "build_leaderboard",
    "run",
)
