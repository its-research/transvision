#!/usr/bin/env python3
"""Create a sealed, one-field amendment of the formal 1337 evaluation plan.

The default mode performs authoritative reads and writes only a local planning
receipt.  Remote ClearML mutation requires ``--execute``, the exact execute
token, and two distinct write-once receipt paths.  The only permitted plan
change is FFNet's queue from GPU4-V100 to GPU4-A100; all evaluation task IDs,
subjects, protocol fields, and every other byte-valued field remain fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

try:
    from tools.resilient_v2x import clearml_1337_dependency_watcher as formal
    from tools.resilient_v2x import (
        recover_clearml_formal_evaluation_attempts as recovery,
    )
except ModuleNotFoundError:  # direct execution from tools/resilient_v2x
    import clearml_1337_dependency_watcher as formal
    import recover_clearml_formal_evaluation_attempts as recovery


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts/resilient_v2x/formal-evaluation-plan-amendment"
DEFAULT_RECOVERY_RECEIPT = (
    ROOT
    / "artifacts/resilient_v2x/formal-evaluation-recovery"
    / "recovery-receipt-20260812T1450CST.json"
)
DEFAULT_ATTEMPT_EVIDENCE = (
    ROOT
    / "artifacts/resilient_v2x/formal-evaluation-recovery"
    / "attempt-evidence-20260812T1450CST.json"
)
DEFAULT_DEPLOYMENT_RECEIPT = (
    ROOT
    / "artifacts/resilient_v2x/formal-successor-chain"
    / "deployment-receipt-final-inputs-v2-20260812T1540CST.json"
)

EXECUTE_TOKEN = "AMEND_FFNET_QUEUE_GPU4_V100_TO_GPU4_A100"
PROJECT_NAME = "ResilientV2X/Training"
PROJECT_ID = "6e43f972e5ea4cee901a7c8855fce8cd"
TASK_NAME = "ResilientV2X formal 1337 FFNet queue amendment producer"
TASK_TAGS = frozenset(
    {
        "DAIR-CAUSAL-1337-v1",
        "ResilientV2X-suite",
        "cpu-controller",
        "formal-evaluation-plan-amendment",
    }
)
FILES_SERVER_URI = "http://10.100.34.118:8081"
ENTRY_POINT = Path(__file__).name
SERVICE_REQUIREMENTS = ("clearml==2.1.11",)
REVISED_PLAN_ARTIFACT = "formal_1337_evaluation_plan"
AMENDMENT_ARTIFACT = "formal_1337_evaluation_plan_amendment"
BASE_PLAN_PRODUCER_TASK_ID = "dbc05b28fbd044bf89edc3872742843a"
BASE_PLAN_SEAL_SHA256 = (
    "c5d14aa8021d06609c7a7e9a5401f4b5a0a417601fa8c95163f083eeb397e0f3"
)
RECOVERY_RECEIPT_SEAL_SHA256 = (
    "d95e2bc2bb93ea6c8c69ea683d31ba09ec2bf219c7a0c09591c0ff12de60fcdf"
)
ATTEMPT_EVIDENCE_SEAL_SHA256 = (
    "3342f4e1bf2cba08e5b3e71afe2719752ae362c288964c2fee3589ee57a6b5bf"
)
DEPLOYMENT_RECEIPT_SEAL_SHA256 = (
    "f54d6a51c1fba1f5c776e236ed79087922da61b71d4b4997f76c0bb1c6a5bb6d"
)
TRAINING_CONTROLLER_TASK_ID = "1011e98e10f64c428880af1d4b1d542b"
TRAINING_PROVENANCE_TASK_ID = "7e244a711751469b8cfdb25d77b05269"
EVALUATION_TEMPLATE_TASK_ID = "8b77a3674dfe405388aae39ef82d06ef"
FFNET_SUBJECT = "ffnet"
FFNET_EVALUATION_TASK_ID = "144397bfa9c242bc9a92a1279922558b"
SOURCE_QUEUE = "GPU4-V100"
TARGET_QUEUE = "GPU4-A100"
V100_WORKER_IDS = (
    "10.100.34.26-V100:gpu0,1,2,3",
    "10.100.34.26-V100:gpu4,5,6,7",
)
A100_TARGET_WORKER_ID = "10.100.34.18-A100:gpu4,5,6,7"
ORIGINAL_FFNET_WORKER_ID = "10.100.34.26-V100:gpu0,1,2,3"
GPU_MEMORY_IDLE_LIMIT_MIB = 1024.0
GPU_USAGE_IDLE_LIMIT_PERCENT = 1.0
GPU_TELEMETRY_WINDOW_SECONDS = 180
GPU_TELEMETRY_INTERVAL_SECONDS = 10
FIXED_BASELINES = ("ffnet", "coformernet", "v2x_vit", "cobevt", "bevfusion")
SUCCESSOR_ROLES = ("W", "L", "A", "S")


class AmendmentError(RuntimeError):
    """Raised when the one-field amendment cannot be proven."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seal(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _strict_json(path: Path, *, context: str) -> dict[str, object]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise AmendmentError(f"{context} cannot be resolved") from error
    if path.is_symlink() or not resolved.is_file():
        raise AmendmentError(f"{context} must be a regular non-symlink file")
    raw = resolved.read_bytes()
    if not raw or len(raw) > 16 * 1024 * 1024:
        raise AmendmentError(f"{context} size is invalid")

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise AmendmentError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda item: (_ for _ in ()).throw(
                AmendmentError(f"{context} contains non-finite value {item}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AmendmentError(f"{context} is not strict JSON") from error
    if not isinstance(value, dict):
        raise AmendmentError(f"{context} must contain one JSON object")
    return value


def _require_valid_seal(value: Mapping[str, object], *, context: str) -> str:
    observed = value.get("seal_sha256")
    if (
        type(observed) is not str
        or formal.SHA256_PATTERN.fullmatch(observed) is None
        or _seal(value)["seal_sha256"] != observed
    ):
        raise AmendmentError(f"{context} seal mismatch")
    return observed


def validate_deployment_receipt(path: Path) -> dict[str, object]:
    value = _strict_json(path, context="formal successor deployment receipt")
    seal = _require_valid_seal(value, context="formal successor deployment receipt")
    if seal != DEPLOYMENT_RECEIPT_SEAL_SHA256:
        raise AmendmentError("formal successor deployment receipt pin mismatch")
    exact = {
        "schema_version": 1,
        "receipt_type": "resilient_v2x_formal_successor_chain_deployment",
        "mode": "completed_provenance_successor_execute",
        "status": "deployed",
        "remote_state_changed": True,
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise AmendmentError(f"formal successor deployment {key} drifted")
    task_ids = value.get("task_ids")
    if not isinstance(task_ids, Mapping) or set(task_ids) != {"P", *SUCCESSOR_ROLES}:
        raise AmendmentError("formal successor task inventory drifted")
    normalized_ids = {
        role: formal._task_id(task_ids[role], f"successor {role} task")
        for role in ("P", *SUCCESSOR_ROLES)
    }
    if len(set(normalized_ids.values())) != len(normalized_ids):
        raise AmendmentError("formal successor task IDs are not unique")
    parameters = value.get("parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != set(SUCCESSOR_ROLES):
        raise AmendmentError("formal successor parameter inventory drifted")
    watcher = parameters.get("W")
    if not isinstance(watcher, Mapping):
        raise AmendmentError("formal successor watcher parameters are invalid")
    if (
        watcher.get("Args/authoritative_evaluation_plan_producer_task_id")
        != BASE_PLAN_PRODUCER_TASK_ID
        or watcher.get("Args/authoritative_evaluation_plan_seal_sha256")
        != BASE_PLAN_SEAL_SHA256
    ):
        raise AmendmentError("formal successor base plan binding drifted")
    recovery_binding = value.get("evaluation_recovery_receipt")
    if (
        not isinstance(recovery_binding, Mapping)
        or recovery_binding.get("receipt_seal_sha256")
        != RECOVERY_RECEIPT_SEAL_SHA256
        or recovery_binding.get("all_tasks_created_unqueued") is not True
        or recovery_binding.get("replacement_tasks_created") is not False
    ):
        raise AmendmentError("formal successor recovery binding drifted")
    return {
        "path": str(path.resolve()),
        "receipt_seal_sha256": seal,
        "task_ids": normalized_ids,
    }


def validate_amendment_receipt(path: Path) -> dict[str, object]:
    """Validate an executed amendment receipt for deploy/recovery consumers."""

    value = _strict_json(path, context="formal plan amendment receipt")
    seal = _require_valid_seal(value, context="formal plan amendment receipt")
    expected_keys = {
        "schema_version",
        "receipt_type",
        "generated_at_utc",
        "mode",
        "status",
        "remote_state_changed",
        "execute_token",
        "base_plan_producer_task_id",
        "base_plan_seal_sha256",
        "revised_plan_seal_sha256",
        "change",
        "ffnet_attempt_evidence",
        "recovery_receipt",
        "deployment_receipt",
        "worker_stats_evidence",
        "staging_amendment_evidence",
        "producer_source_sha256",
        "remote_write_contract",
        "attempt_receipt_seal_sha256",
        "producer",
        "amendment_evidence",
        "seal_sha256",
    }
    if set(value) != expected_keys:
        raise AmendmentError("formal plan amendment receipt key inventory drifted")
    exact = {
        "schema_version": 1,
        "receipt_type": "resilient_v2x_formal_evaluation_plan_amendment",
        "mode": "execute",
        "status": "completed",
        "remote_state_changed": True,
        "execute_token": EXECUTE_TOKEN,
        "base_plan_producer_task_id": BASE_PLAN_PRODUCER_TASK_ID,
        "base_plan_seal_sha256": BASE_PLAN_SEAL_SHA256,
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise AmendmentError(f"formal plan amendment receipt {key} drifted")
    recovery_binding = value.get("recovery_receipt")
    deployment_binding = value.get("deployment_receipt")
    attempt_binding = value.get("ffnet_attempt_evidence")
    if (
        not isinstance(recovery_binding, Mapping)
        or recovery_binding.get("receipt_seal_sha256")
        != RECOVERY_RECEIPT_SEAL_SHA256
        or recovery_binding.get("all_tasks_created_unqueued") is not True
        or recovery_binding.get("replacement_tasks_created") is not False
        or not isinstance(deployment_binding, Mapping)
        or deployment_binding.get("receipt_seal_sha256")
        != DEPLOYMENT_RECEIPT_SEAL_SHA256
        or not isinstance(attempt_binding, Mapping)
        or attempt_binding.get("receipt_seal_sha256")
        != ATTEMPT_EVIDENCE_SEAL_SHA256
        or attempt_binding.get("task_id") != FFNET_EVALUATION_TASK_ID
        or attempt_binding.get("last_worker") != ORIGINAL_FFNET_WORKER_ID
    ):
        raise AmendmentError("formal plan amendment authority receipt drifted")
    change = value.get("change")
    if not isinstance(change, Mapping):
        raise AmendmentError("formal plan amendment change record is invalid")
    expected_change = {
        "base_plan_seal_sha256": BASE_PLAN_SEAL_SHA256,
        "revised_plan_seal_sha256": value.get("revised_plan_seal_sha256"),
        "changed_json_pointer": (
            f"/entries/{formal.FORMAL_SUBJECT_ORDER.index(FFNET_SUBJECT)}/queue"
        ),
        "subject": FFNET_SUBJECT,
        "evaluation_task_id": FFNET_EVALUATION_TASK_ID,
        "from_queue": SOURCE_QUEUE,
        "to_queue": TARGET_QUEUE,
        "fixed_baselines": list(FIXED_BASELINES),
        "protocol_id": formal.EXPECTED_PROTOCOL_ID,
        "sample_count": formal.EXPECTED_SAMPLE_COUNT,
        "changed_field_count": 1,
    }
    if set(change) != {*expected_change, "evaluation_task_ids_sha256"} or any(
        change.get(key) != expected for key, expected in expected_change.items()
    ):
        raise AmendmentError("formal plan amendment is not the exact FFNet queue change")
    evaluation_ids_sha256 = change.get("evaluation_task_ids_sha256")
    if (
        type(evaluation_ids_sha256) is not str
        or formal.SHA256_PATTERN.fullmatch(evaluation_ids_sha256) is None
    ):
        raise AmendmentError("formal plan amendment task-ID inventory seal is invalid")
    revised_seal = value.get("revised_plan_seal_sha256")
    source_sha = value.get("producer_source_sha256")
    if (
        type(revised_seal) is not str
        or formal.SHA256_PATTERN.fullmatch(revised_seal) is None
        or type(source_sha) is not str
        or formal.SHA256_PATTERN.fullmatch(source_sha) is None
    ):
        raise AmendmentError("formal plan amendment plan/source seal is invalid")
    worker = validate_worker_evidence(value["worker_stats_evidence"])
    producer = value.get("producer")
    evidence = value.get("amendment_evidence")
    if not isinstance(producer, Mapping) or not isinstance(evidence, Mapping):
        raise AmendmentError("formal plan amendment producer evidence is invalid")
    producer_id = formal._task_id(producer.get("task_id"), "amendment producer")
    evidence_seal = _require_valid_seal(evidence, context="amendment evidence")
    if (
        producer.get("status") != "completed"
        or producer.get("name") != TASK_NAME
        or producer.get("project") != PROJECT_NAME
        or producer.get("source_sha256") != source_sha
        or producer.get("parent_task_id") != BASE_PLAN_PRODUCER_TASK_ID
        or producer.get("tags") != sorted(TASK_TAGS)
        or producer.get("artifact_seals")
        != {
            REVISED_PLAN_ARTIFACT: revised_seal,
            AMENDMENT_ARTIFACT: evidence_seal,
        }
        or producer.get("authoritative_readback_count") != 2
        or evidence.get("producer_task_id") != producer_id
        or evidence.get("worker_stats_evidence") != worker
        or evidence.get("amendment") != change
        or not isinstance(evidence.get("authorization"), Mapping)
        or evidence["authorization"].get("ffnet_attempt_evidence")
        != attempt_binding
    ):
        raise AmendmentError("formal plan amendment producer/evidence binding drifted")
    parameters = producer.get("parameters")
    if parameters != _producer_parameters(
        revised_plan_seal=revised_seal,
        worker_evidence_seal=str(worker["seal_sha256"]),
    ):
        raise AmendmentError("formal plan amendment producer parameters drifted")
    return {
        "path": str(path.resolve()),
        "receipt_seal_sha256": seal,
        "producer_task_id": producer_id,
        "producer_source_sha256": source_sha,
        "revised_plan_seal_sha256": revised_seal,
        "amendment_evidence_seal_sha256": evidence_seal,
        "worker_stats_evidence_seal_sha256": worker["seal_sha256"],
        "evaluation_task_ids_sha256": evaluation_ids_sha256,
        "from_queue": SOURCE_QUEUE,
        "to_queue": TARGET_QUEUE,
        "ffnet_evaluation_task_id": FFNET_EVALUATION_TASK_ID,
    }


def validate_recovery_receipt(path: Path) -> dict[str, object]:
    try:
        binding = recovery.validate_recovery_receipt(path)
    except recovery.RecoveryError as error:
        raise AmendmentError(f"formal evaluation recovery receipt invalid: {error}") from error
    if binding.get("receipt_seal_sha256") != RECOVERY_RECEIPT_SEAL_SHA256:
        raise AmendmentError("formal evaluation recovery receipt pin mismatch")
    return binding


def validate_attempt_evidence(path: Path) -> dict[str, object]:
    value = _strict_json(path, context="formal evaluation attempt evidence")
    seal = _require_valid_seal(value, context="formal evaluation attempt evidence")
    if (
        seal != ATTEMPT_EVIDENCE_SEAL_SHA256
        or value.get("schema_version") != 1
        or value.get("receipt_type")
        != "resilient_v2x_evaluation_attempt_evidence"
        or value.get("remote_state_changed") is not False
        or value.get("authoritative_plan_producer_task_id")
        != BASE_PLAN_PRODUCER_TASK_ID
        or value.get("authoritative_plan_seal_sha256") != BASE_PLAN_SEAL_SHA256
    ):
        raise AmendmentError("formal evaluation attempt evidence identity drifted")
    attempts = value.get("attempts")
    planned = value.get("planned_tasks")
    if not isinstance(attempts, list) or not isinstance(planned, list):
        raise AmendmentError("formal evaluation attempt inventory is invalid")
    ffnet_attempts = [
        row
        for row in attempts
        if isinstance(row, Mapping)
        and row.get("task_id") == FFNET_EVALUATION_TASK_ID
        and row.get("subject") == FFNET_SUBJECT
    ]
    ffnet_plans = [
        row
        for row in planned
        if isinstance(row, Mapping)
        and row.get("task_id") == FFNET_EVALUATION_TASK_ID
        and row.get("subject") == FFNET_SUBJECT
    ]
    if len(ffnet_attempts) != 1 or len(ffnet_plans) != 1:
        raise AmendmentError("sealed FFNet attempt is not unique")
    attempt = dict(ffnet_attempts[0])
    plan = dict(ffnet_plans[0])
    if (
        attempt.get("status") != "failed"
        or attempt.get("execution_queue_id")
        != formal.EXPECTED_QUEUE_IDS[SOURCE_QUEUE]
        or attempt.get("last_worker") != ORIGINAL_FFNET_WORKER_ID
        or plan.get("queue") != SOURCE_QUEUE
        or plan.get("queue_id") != formal.EXPECTED_QUEUE_IDS[SOURCE_QUEUE]
    ):
        raise AmendmentError("sealed FFNet failed-attempt worker/queue drifted")
    return {
        "path": str(path.resolve()),
        "receipt_seal_sha256": seal,
        "task_id": FFNET_EVALUATION_TASK_ID,
        "subject": FFNET_SUBJECT,
        "status": "failed",
        "queue": SOURCE_QUEUE,
        "queue_id": formal.EXPECTED_QUEUE_IDS[SOURCE_QUEUE],
        "last_worker": ORIGINAL_FFNET_WORKER_ID,
        "attempt_evidence_sha256": attempt.get("attempt_evidence_sha256"),
    }


def validate_base_plan(value: Mapping[str, object]) -> dict[str, object]:
    plan = dict(value)
    seal = _require_valid_seal(plan, context="base formal evaluation plan")
    if seal != BASE_PLAN_SEAL_SHA256:
        raise AmendmentError("base formal evaluation plan seal pin mismatch")
    exact = {
        "schema_version": 2,
        "plan_type": "resilient_v2x_formal_1337_evaluation_tasks",
        "protocol_id": formal.EXPECTED_PROTOCOL_ID,
        "sample_count": formal.EXPECTED_SAMPLE_COUNT,
        "delays_ms": list(formal.EXPECTED_DELAYS_MS),
        "conditions": list(formal.EXPECTED_CONDITIONS),
        "run_count": 12,
        "subject_order": list(formal.FORMAL_SUBJECT_ORDER),
    }
    for key, expected in exact.items():
        if plan.get(key) != expected:
            raise AmendmentError(f"base formal evaluation plan {key} drifted")
    entries = plan.get("entries")
    if not isinstance(entries, list) or len(entries) != len(formal.FORMAL_SUBJECT_ORDER):
        raise AmendmentError("base formal evaluation plan entries drifted")
    observed_ids: list[str] = []
    by_subject: dict[str, Mapping[str, object]] = {}
    for index, (entry, subject) in enumerate(
        zip(entries, formal.FORMAL_SUBJECT_ORDER, strict=True)
    ):
        if not isinstance(entry, Mapping) or set(entry) != {
            "subject",
            "evaluation_task_id",
            "queue",
        }:
            raise AmendmentError(f"base evaluation entry {index} shape drifted")
        if entry.get("subject") != subject:
            raise AmendmentError(f"base evaluation entry {index} subject drifted")
        observed_ids.append(
            formal._task_id(entry.get("evaluation_task_id"), f"{subject} evaluation")
        )
        queue = str(entry.get("queue") or "")
        if queue not in formal.EXPECTED_QUEUE_IDS:
            raise AmendmentError(f"base evaluation {subject} queue drifted")
        by_subject[subject] = entry
    if len(set(observed_ids)) != 26:
        raise AmendmentError("base evaluation task IDs are not unique")
    if set(FIXED_BASELINES) - set(by_subject):
        raise AmendmentError("fixed five-baseline inventory drifted")
    ffnet = by_subject[FFNET_SUBJECT]
    if (
        ffnet.get("evaluation_task_id") != FFNET_EVALUATION_TASK_ID
        or ffnet.get("queue") != SOURCE_QUEUE
    ):
        raise AmendmentError("base FFNet task/queue binding drifted")
    return plan


def amend_plan(base_plan: Mapping[str, object]) -> dict[str, object]:
    base = validate_base_plan(base_plan)
    revised = json.loads(_canonical_json(base))
    entries = revised["entries"]
    ffnet_indexes = [
        index
        for index, entry in enumerate(entries)
        if entry.get("subject") == FFNET_SUBJECT
    ]
    if len(ffnet_indexes) != 1:
        raise AmendmentError("FFNet plan entry is not unique")
    entries[ffnet_indexes[0]]["queue"] = TARGET_QUEUE
    revised.pop("seal_sha256", None)
    revised = _seal(revised)
    validate_one_field_amendment(base, revised)
    return revised


def validate_one_field_amendment(
    base_plan: Mapping[str, object], revised_plan: Mapping[str, object]
) -> dict[str, object]:
    base = validate_base_plan(base_plan)
    revised = dict(revised_plan)
    revised_seal = _require_valid_seal(revised, context="revised formal evaluation plan")
    expected = json.loads(_canonical_json(base))
    expected_entries = expected["entries"]
    ffnet_index = next(
        index
        for index, entry in enumerate(expected_entries)
        if entry["subject"] == FFNET_SUBJECT
    )
    expected_entries[ffnet_index]["queue"] = TARGET_QUEUE
    expected.pop("seal_sha256", None)
    expected = _seal(expected)
    if revised != expected:
        raise AmendmentError("revised plan changed more than FFNet's queue field")
    entries = revised.get("entries")
    if not isinstance(entries, list):  # covered by equality; keeps typing explicit
        raise AmendmentError("revised plan entries are invalid")
    observed_ids = [entry.get("evaluation_task_id") for entry in entries]
    if observed_ids != [entry.get("evaluation_task_id") for entry in base["entries"]]:
        raise AmendmentError("revised plan evaluation task IDs drifted")
    if revised.get("protocol_id") != formal.EXPECTED_PROTOCOL_ID or revised.get(
        "sample_count"
    ) != formal.EXPECTED_SAMPLE_COUNT:
        raise AmendmentError("revised plan protocol/sample count drifted")
    return {
        "base_plan_seal_sha256": BASE_PLAN_SEAL_SHA256,
        "revised_plan_seal_sha256": revised_seal,
        "changed_json_pointer": f"/entries/{ffnet_index}/queue",
        "subject": FFNET_SUBJECT,
        "evaluation_task_id": FFNET_EVALUATION_TASK_ID,
        "from_queue": SOURCE_QUEUE,
        "to_queue": TARGET_QUEUE,
        "evaluation_task_ids_sha256": _content_sha256(observed_ids),
        "fixed_baselines": list(FIXED_BASELINES),
        "protocol_id": formal.EXPECTED_PROTOCOL_ID,
        "sample_count": formal.EXPECTED_SAMPLE_COUNT,
        "changed_field_count": 1,
    }


def _finite_nonnegative_values(value: object, *, context: str) -> list[float]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) not in {int, float} for item in value)
    ):
        raise AmendmentError(f"{context} values are invalid")
    result = [float(item) for item in value]
    if any(not (0.0 <= item < float("inf")) for item in result):
        raise AmendmentError(f"{context} values are not finite and non-negative")
    return result


def _validate_gpu_telemetry_record(
    value: object, *, expected_worker_id: str, context: str
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "worker_id",
        "window_from_unix",
        "window_to_unix",
        "interval_seconds",
        "gpu_memory_used_mib",
        "gpu_usage_percent",
        "max_gpu_memory_used_mib",
        "max_gpu_usage_percent",
    }:
        raise AmendmentError(f"{context} telemetry shape drifted")
    record = dict(value)
    if record.get("worker_id") != expected_worker_id:
        raise AmendmentError(f"{context} worker identity drifted")
    if (
        type(record.get("window_from_unix")) not in {int, float}
        or type(record.get("window_to_unix")) not in {int, float}
        or float(record["window_to_unix"]) <= float(record["window_from_unix"])
        or record.get("interval_seconds") != GPU_TELEMETRY_INTERVAL_SECONDS
    ):
        raise AmendmentError(f"{context} telemetry window drifted")
    memory = _finite_nonnegative_values(
        record.get("gpu_memory_used_mib"), context=f"{context} GPU memory"
    )
    usage = _finite_nonnegative_values(
        record.get("gpu_usage_percent"), context=f"{context} GPU usage"
    )
    if len(memory) != len(usage):
        raise AmendmentError(f"{context} metric sample count drifted")
    if record.get("max_gpu_memory_used_mib") != max(memory) or record.get(
        "max_gpu_usage_percent"
    ) != max(usage):
        raise AmendmentError(f"{context} telemetry summary drifted")
    return record


def validate_worker_evidence(value: Mapping[str, object]) -> dict[str, object]:
    evidence = dict(value)
    if set(evidence) != {
        "captured_at_utc",
        "formal_evaluation_task_ids",
        "gpu4_v100",
        "gpu4_a100",
        "gpu_telemetry",
        "gpu_telemetry_policy",
        "seal_sha256",
    }:
        raise AmendmentError("worker evidence key inventory drifted")
    _require_valid_seal(evidence, context="worker stats evidence")
    ids = evidence.get("formal_evaluation_task_ids")
    if not isinstance(ids, list) or set(ids) != {
        row[2] for row in recovery.EXPECTED_RECOVERY_TASKS
    }:
        raise AmendmentError("worker evidence formal task inventory drifted")
    v100 = evidence.get("gpu4_v100")
    a100 = evidence.get("gpu4_a100")
    if not isinstance(v100, Mapping) or not isinstance(a100, Mapping):
        raise AmendmentError("worker resource snapshots are invalid")
    if v100.get("target_queue_id") != formal.EXPECTED_QUEUE_IDS[SOURCE_QUEUE]:
        raise AmendmentError("GPU4-V100 queue identity drifted")
    if (
        a100.get("target_queue_id") != formal.EXPECTED_QUEUE_IDS[TARGET_QUEUE]
        or a100.get("ready") is not True
        or type(a100.get("selected_idle_target_worker_id")) is not str
        or not a100["selected_idle_target_worker_id"]
    ):
        raise AmendmentError("GPU4-A100 lacks one authoritative idle worker")
    telemetry = evidence.get("gpu_telemetry")
    if not isinstance(telemetry, Mapping) or set(telemetry) != {
        *V100_WORKER_IDS,
        A100_TARGET_WORKER_ID,
    }:
        raise AmendmentError("worker GPU telemetry inventory drifted")
    parsed = {
        worker_id: _validate_gpu_telemetry_record(
            telemetry[worker_id], expected_worker_id=worker_id, context=worker_id
        )
        for worker_id in (*V100_WORKER_IDS, A100_TARGET_WORKER_ID)
    }
    if (
        parsed[ORIGINAL_FFNET_WORKER_ID]["max_gpu_memory_used_mib"]
        <= GPU_MEMORY_IDLE_LIMIT_MIB
    ):
        raise AmendmentError("FFNet's original V100 worker is no longer busy")
    a100_telemetry = parsed[A100_TARGET_WORKER_ID]
    if (
        a100_telemetry["max_gpu_memory_used_mib"] > GPU_MEMORY_IDLE_LIMIT_MIB
        or a100_telemetry["max_gpu_usage_percent"] > GPU_USAGE_IDLE_LIMIT_PERCENT
    ):
        raise AmendmentError("A100 gpu4-7 telemetry is not idle")
    if evidence.get("gpu_telemetry_policy") != {
        "source_workers": list(V100_WORKER_IDS),
        "target_worker": A100_TARGET_WORKER_ID,
        "window_seconds": GPU_TELEMETRY_WINDOW_SECONDS,
        "interval_seconds": GPU_TELEMETRY_INTERVAL_SECONDS,
        "v100_busy_worker": ORIGINAL_FFNET_WORKER_ID,
        "v100_busy_min_memory_used_mib_exclusive": GPU_MEMORY_IDLE_LIMIT_MIB,
        "a100_idle_max_memory_used_mib_inclusive": GPU_MEMORY_IDLE_LIMIT_MIB,
        "a100_idle_max_usage_percent_inclusive": GPU_USAGE_IDLE_LIMIT_PERCENT,
    }:
        raise AmendmentError("worker GPU telemetry policy drifted")
    return evidence


def _metric_values(metric: object, *, context: str) -> list[float]:
    value = metric.to_dict() if hasattr(metric, "to_dict") else metric
    if not isinstance(value, Mapping) or value.get("metric") != context:
        raise AmendmentError(f"worker {context} metric identity drifted")
    stats = value.get("stats")
    if not isinstance(stats, list) or len(stats) != 1:
        raise AmendmentError(f"worker {context} metric aggregation drifted")
    stat = stats[0]
    if not isinstance(stat, Mapping) or stat.get("aggregation") != "avg":
        raise AmendmentError(f"worker {context} metric aggregation drifted")
    return _finite_nonnegative_values(
        stat.get("values"), context=f"worker {context}"
    )


def _worker_gpu_telemetry(api_client: object, worker_id: str) -> dict[str, object]:
    end = time.time()
    start = end - GPU_TELEMETRY_WINDOW_SECONDS
    response = api_client.workers.get_stats(
        from_date=start,
        to_date=end,
        interval=GPU_TELEMETRY_INTERVAL_SECONDS,
        items=[
            {"key": "gpu_memory_used", "category": "max"},
            {"key": "gpu_usage", "category": "max"},
        ],
        worker_ids=[worker_id],
        split_by_variant=True,
    )
    payload = response.to_dict() if hasattr(response, "to_dict") else response
    workers = payload.get("workers") if isinstance(payload, Mapping) else None
    if not isinstance(workers, list) or len(workers) != 1:
        raise AmendmentError(f"worker {worker_id} telemetry response drifted")
    worker = workers[0]
    if not isinstance(worker, Mapping) or worker.get("worker") != worker_id:
        raise AmendmentError(f"worker {worker_id} telemetry identity drifted")
    metrics = worker.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != 2:
        raise AmendmentError(f"worker {worker_id} telemetry metrics drifted")
    by_name: dict[str, object] = {}
    for item in metrics:
        normalized = item.to_dict() if hasattr(item, "to_dict") else item
        if isinstance(normalized, Mapping):
            by_name[str(normalized.get("metric") or "")] = item
    if set(by_name) != {"gpu_memory_used", "gpu_usage"}:
        raise AmendmentError(f"worker {worker_id} telemetry metric inventory drifted")
    memory = _metric_values(by_name["gpu_memory_used"], context="gpu_memory_used")
    usage = _metric_values(by_name["gpu_usage"], context="gpu_usage")
    if len(memory) != len(usage):
        raise AmendmentError(f"worker {worker_id} telemetry sample count drifted")
    return {
        "worker_id": worker_id,
        "window_from_unix": start,
        "window_to_unix": end,
        "interval_seconds": GPU_TELEMETRY_INTERVAL_SECONDS,
        "gpu_memory_used_mib": memory,
        "gpu_usage_percent": usage,
        "max_gpu_memory_used_mib": max(memory),
        "max_gpu_usage_percent": max(usage),
    }


def capture_worker_evidence(task_class: object, api_client: object) -> dict[str, object]:
    rows = formal._live_worker_rows()
    formal_ids = {row[2] for row in recovery.EXPECTED_RECOVERY_TASKS}
    value = _seal({
        "captured_at_utc": _utc_timestamp(),
        "formal_evaluation_task_ids": sorted(formal_ids),
        "gpu4_v100": formal._formal_resource_gate_snapshot(
            task_class,
            formal_evaluation_task_ids=formal_ids,
            target_queue_id=formal.EXPECTED_QUEUE_IDS[SOURCE_QUEUE],
            worker_rows=rows,
        ),
        "gpu4_a100": formal._formal_resource_gate_snapshot(
            task_class,
            formal_evaluation_task_ids=formal_ids,
            target_queue_id=formal.EXPECTED_QUEUE_IDS[TARGET_QUEUE],
            worker_rows=rows,
        ),
        "gpu_telemetry": {
            worker_id: _worker_gpu_telemetry(api_client, worker_id)
            for worker_id in (*V100_WORKER_IDS, A100_TARGET_WORKER_ID)
        },
        "gpu_telemetry_policy": {
            "source_workers": list(V100_WORKER_IDS),
            "target_worker": A100_TARGET_WORKER_ID,
            "window_seconds": GPU_TELEMETRY_WINDOW_SECONDS,
            "interval_seconds": GPU_TELEMETRY_INTERVAL_SECONDS,
            "v100_busy_worker": ORIGINAL_FFNET_WORKER_ID,
            "v100_busy_min_memory_used_mib_exclusive": GPU_MEMORY_IDLE_LIMIT_MIB,
            "a100_idle_max_memory_used_mib_inclusive": GPU_MEMORY_IDLE_LIMIT_MIB,
            "a100_idle_max_usage_percent_inclusive": GPU_USAGE_IDLE_LIMIT_PERCENT,
        },
    })
    return validate_worker_evidence(value)


def build_amendment_evidence(
    *,
    change: Mapping[str, object],
    attempt_binding: Mapping[str, object],
    recovery_binding: Mapping[str, object],
    deployment_binding: Mapping[str, object],
    worker_evidence: Mapping[str, object],
    producer_task_id: str | None,
) -> dict[str, object]:
    return _seal(
        {
            "schema_version": 1,
            "evidence_type": "resilient_v2x_formal_1337_evaluation_plan_amendment",
            "generated_at_utc": _utc_timestamp(),
            "producer_task_id": producer_task_id,
            "base_authority": {
                "producer_task_id": BASE_PLAN_PRODUCER_TASK_ID,
                "artifact_name": REVISED_PLAN_ARTIFACT,
                "plan_seal_sha256": BASE_PLAN_SEAL_SHA256,
            },
            "authorization": {
                "ffnet_attempt_evidence": dict(attempt_binding),
                "evaluation_recovery_receipt_seal_sha256": recovery_binding[
                    "receipt_seal_sha256"
                ],
                "successor_deployment_receipt_seal_sha256": deployment_binding[
                    "receipt_seal_sha256"
                ],
                "successor_task_ids": deployment_binding["task_ids"],
            },
            "amendment": dict(change),
            "rationale": (
                "The sealed failed FFNet attempt ran on V100 gpu0-3, which fresh "
                "worker telemetry still shows externally busy; GPU4-V100 cannot "
                "deterministically target its currently idle half, while A100 "
                "gpu4-7 is authoritatively <=1024 MiB and idle. Move only FFNet's "
                "queue assignment without changing evaluation identity or protocol."
            ),
            "worker_stats_evidence": validate_worker_evidence(worker_evidence),
            "invariants": {
                "one_field_amendment": True,
                "evaluation_task_ids_unchanged": True,
                "evaluation_parameters_unchanged": True,
                "evaluation_sources_unchanged": True,
                "evaluation_models_unchanged": True,
                "protocol_and_sample_count_unchanged": True,
                "fixed_five_baselines_unchanged": True,
                "replacement_tasks_created": False,
            },
        }
    )


def _load_base_plan(task_class: object, downloader: Callable[[object, str], bytes]) -> dict[str, object]:
    # The existing watcher validator proves the old producer's task, source,
    # parameters, tags, artifact metadata, preview bytes, and plan seal.
    formal._load_authoritative_evaluation_plan(
        task_class=task_class,
        producer_task_id=BASE_PLAN_PRODUCER_TASK_ID,
        expected_seal_sha256=BASE_PLAN_SEAL_SHA256,
        expected_project_id=PROJECT_ID,
        controller_task_id=TRAINING_CONTROLLER_TASK_ID,
        provenance_task_id=TRAINING_PROVENANCE_TASK_ID,
        template_task_id=EVALUATION_TEMPLATE_TASK_ID,
    )
    producer = task_class.get_task(task_id=BASE_PLAN_PRODUCER_TASK_ID)
    raw = downloader(producer, REVISED_PLAN_ARTIFACT)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AmendmentError("base formal evaluation plan artifact is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise AmendmentError("base formal evaluation plan artifact is not an object")
    return validate_base_plan(value)


def _producer_parameters(
    *, revised_plan_seal: str, worker_evidence_seal: str
) -> dict[str, str]:
    return {
        "Args/base_plan_producer_task_id": BASE_PLAN_PRODUCER_TASK_ID,
        "Args/base_plan_seal_sha256": BASE_PLAN_SEAL_SHA256,
        "Args/evaluation_recovery_receipt_seal_sha256": (
            RECOVERY_RECEIPT_SEAL_SHA256
        ),
        "Args/ffnet_attempt_evidence_seal_sha256": (
            ATTEMPT_EVIDENCE_SEAL_SHA256
        ),
        "Args/ffnet_original_worker_id": ORIGINAL_FFNET_WORKER_ID,
        "Args/successor_deployment_receipt_seal_sha256": (
            DEPLOYMENT_RECEIPT_SEAL_SHA256
        ),
        "Args/subject": FFNET_SUBJECT,
        "Args/evaluation_task_id": FFNET_EVALUATION_TASK_ID,
        "Args/from_queue": SOURCE_QUEUE,
        "Args/to_queue": TARGET_QUEUE,
        "Args/revised_plan_seal_sha256": revised_plan_seal,
        "Args/worker_evidence_seal_sha256": worker_evidence_seal,
    }


def _artifact_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _producer_authoritative_readback(
    task: object,
    *,
    source: str,
    revised_plan: Mapping[str, object],
    evidence: Mapping[str, object],
    expected_parameters: Mapping[str, str],
) -> dict[str, object]:
    expected_artifacts = {
        REVISED_PLAN_ARTIFACT: dict(revised_plan),
        AMENDMENT_ARTIFACT: dict(evidence),
    }
    observed_artifacts: dict[str, Mapping[str, object]] | None = None
    for read_index in range(2):
        reloader = getattr(task, "reload", None)
        if callable(reloader):
            reloader()
        if formal._status(task) != "completed":
            raise AmendmentError("amendment producer did not remain completed")
        if formal._task_id(getattr(task, "id", ""), "amendment producer") == (
            BASE_PLAN_PRODUCER_TASK_ID
        ):
            raise AmendmentError("amendment producer aliases the base producer")
        if str(getattr(task, "name", "") or "") != TASK_NAME:
            raise AmendmentError("amendment producer name drifted")
        project_getter = getattr(task, "get_project_name", None)
        if not callable(project_getter) or project_getter() != PROJECT_NAME:
            raise AmendmentError("amendment producer project drifted")
        if formal._task_parent(task) != BASE_PLAN_PRODUCER_TASK_ID:
            raise AmendmentError("amendment producer parent drifted")
        script = getattr(getattr(task, "data", None), "script", None)
        if (
            str(getattr(script, "repository", "") or "") != ""
            or str(getattr(script, "working_dir", "") or "") != "."
            or str(getattr(script, "entry_point", "") or "") != ENTRY_POINT
            or str(getattr(script, "diff", "") or "") != source
        ):
            raise AmendmentError("amendment producer source drifted")
        parameters = formal._parameters(task)
        if set(parameters) != set(expected_parameters) or any(
            str(parameters.get(key)) != value
            for key, value in expected_parameters.items()
        ):
            raise AmendmentError("amendment producer exact parameters drifted")
        raw_tags = getattr(task, "tags", None)
        if raw_tags is None:
            raw_tags = getattr(getattr(task, "data", None), "tags", None)
        if (
            not isinstance(raw_tags, Sequence)
            or isinstance(raw_tags, (str, bytes))
            or len(raw_tags) != len(set(raw_tags))
            or set(raw_tags) != set(TASK_TAGS)
        ):
            raise AmendmentError("amendment producer exact tags drifted")
        artifacts = getattr(task, "artifacts", None)
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(
            expected_artifacts
        ):
            raise AmendmentError("amendment producer artifact inventory drifted")
        current = {
            name: formal._artifact_mapping(artifacts[name], name)
            for name in sorted(expected_artifacts)
        }
        if any(current[name] != expected_artifacts[name] for name in current):
            raise AmendmentError("amendment producer artifact content drifted")
        if observed_artifacts is not None and current != observed_artifacts:
            raise AmendmentError("amendment producer changed across readbacks")
        observed_artifacts = current
        if read_index == 0:
            time.sleep(0.25)
    return {
        "task_id": formal._task_id(getattr(task, "id", ""), "amendment producer"),
        "status": "completed",
        "name": TASK_NAME,
        "project": PROJECT_NAME,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "parameters": dict(expected_parameters),
        "tags": sorted(TASK_TAGS),
        "parent_task_id": BASE_PLAN_PRODUCER_TASK_ID,
        "artifact_seals": {
            REVISED_PLAN_ARTIFACT: revised_plan["seal_sha256"],
            AMENDMENT_ARTIFACT: evidence["seal_sha256"],
        },
        "authoritative_readback_count": 2,
    }


def execute_producer(
    task_class: object,
    *,
    source: str,
    revised_plan: Mapping[str, object],
    attempt_binding: Mapping[str, object],
    recovery_binding: Mapping[str, object],
    deployment_binding: Mapping[str, object],
    worker_evidence: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    task_types = getattr(task_class, "TaskTypes", None)
    task = task_class.create(
        project_name=PROJECT_NAME,
        task_name=TASK_NAME,
        task_type=getattr(task_types, "controller", "controller"),
        script=None,
        packages=None,
        docker=None,
        docker_args=None,
        add_task_init_call=False,
        binary="python",
        detect_repository=False,
    )
    task_id = formal._task_id(getattr(task, "id", ""), "amendment producer")
    if formal._status(task) != "created":
        raise AmendmentError("amendment producer was not created")
    # Recompute the public change record without trusting caller-provided prose.
    change = {
        "base_plan_seal_sha256": BASE_PLAN_SEAL_SHA256,
        "revised_plan_seal_sha256": revised_plan["seal_sha256"],
        "changed_json_pointer": (
            f"/entries/{formal.FORMAL_SUBJECT_ORDER.index(FFNET_SUBJECT)}/queue"
        ),
        "subject": FFNET_SUBJECT,
        "evaluation_task_id": FFNET_EVALUATION_TASK_ID,
        "from_queue": SOURCE_QUEUE,
        "to_queue": TARGET_QUEUE,
        "evaluation_task_ids_sha256": _content_sha256(
            [entry["evaluation_task_id"] for entry in revised_plan["entries"]]
        ),
        "fixed_baselines": list(FIXED_BASELINES),
        "protocol_id": formal.EXPECTED_PROTOCOL_ID,
        "sample_count": formal.EXPECTED_SAMPLE_COUNT,
        "changed_field_count": 1,
    }
    evidence = build_amendment_evidence(
        change=change,
        attempt_binding=attempt_binding,
        recovery_binding=recovery_binding,
        deployment_binding=deployment_binding,
        worker_evidence=worker_evidence,
        producer_task_id=task_id,
    )
    task.set_script(
        repository="",
        branch="",
        commit="",
        diff=source,
        working_dir=".",
        entry_point=ENTRY_POINT,
    )
    task.set_packages(list(SERVICE_REQUIREMENTS))
    task.set_parent(BASE_PLAN_PRODUCER_TASK_ID)
    task.output_uri = FILES_SERVER_URI
    task.set_parameters(
        _producer_parameters(
            revised_plan_seal=str(revised_plan["seal_sha256"]),
            worker_evidence_seal=str(worker_evidence["seal_sha256"]),
        )
    )
    if task.set_tags(sorted(TASK_TAGS)) is False:
        raise AmendmentError("amendment producer rejected exact tags")
    if not task.upload_artifact(
        REVISED_PLAN_ARTIFACT,
        artifact_object=dict(revised_plan),
        wait_on_upload=True,
    ):
        raise AmendmentError("revised plan artifact upload failed")
    if not task.upload_artifact(
        AMENDMENT_ARTIFACT,
        artifact_object=evidence,
        wait_on_upload=True,
    ):
        raise AmendmentError("amendment evidence artifact upload failed")
    if task.flush(wait_for_uploads=True) is False:
        raise AmendmentError("amendment producer flush was not confirmed")
    task.mark_completed(status_message="sealed FFNet queue-only amendment", force=True)
    reloader = getattr(task, "reload", None)
    if callable(reloader):
        reloader()
    expected_parameters = _producer_parameters(
        revised_plan_seal=str(revised_plan["seal_sha256"]),
        worker_evidence_seal=str(worker_evidence["seal_sha256"]),
    )
    producer = _producer_authoritative_readback(
        task,
        source=source,
        revised_plan=revised_plan,
        evidence=evidence,
        expected_parameters=expected_parameters,
    )
    if producer["task_id"] != task_id:
        raise AmendmentError("amendment producer identity changed after creation")
    return evidence, producer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-receipt", type=Path, default=DEFAULT_RECOVERY_RECEIPT)
    parser.add_argument("--attempt-evidence", type=Path, default=DEFAULT_ATTEMPT_EVIDENCE)
    parser.add_argument("--deployment-receipt", type=Path, default=DEFAULT_DEPLOYMENT_RECEIPT)
    parser.add_argument("--base-plan-json", type=Path)
    parser.add_argument("--worker-evidence-json", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--attempt-receipt", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-token", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and args.execute_token != EXECUTE_TOKEN:
        raise AmendmentError(f"exact execute token required: {EXECUTE_TOKEN}")
    if not args.execute and args.execute_token:
        raise AmendmentError("--execute-token requires --execute")
    if args.execute and args.attempt_receipt is None:
        raise AmendmentError("--attempt-receipt is required with --execute")
    if not args.execute and args.attempt_receipt is not None:
        raise AmendmentError("--attempt-receipt requires --execute")
    if args.attempt_receipt is not None and args.attempt_receipt.resolve() == args.receipt.resolve():
        raise AmendmentError("attempt and final receipts must use distinct paths")

    recovery_binding = validate_recovery_receipt(args.recovery_receipt)
    attempt_binding = validate_attempt_evidence(args.attempt_evidence)
    deployment_binding = validate_deployment_receipt(args.deployment_receipt)
    task_class = None
    api_client = None
    downloader = None
    if args.base_plan_json is None or args.worker_evidence_json is None or args.execute:
        try:
            from clearml import Task
            from clearml.backend_api.session import Session
            from clearml.backend_api.session.client import APIClient
        except ImportError as error:
            raise AmendmentError("ClearML is required for authoritative inspection") from error
        task_class = Task
        api_client = APIClient()
        session = Session()
        downloader = recovery.AuthenticatedArtifactDownloader(
            files_server_url=str(Session.get_files_server_host()),
            add_auth_headers=session.add_auth_headers,
        )
    base_plan = (
        validate_base_plan(_strict_json(args.base_plan_json, context="base plan fixture"))
        if args.base_plan_json is not None
        else _load_base_plan(task_class, downloader)
    )
    worker_evidence = (
        validate_worker_evidence(
            _strict_json(args.worker_evidence_json, context="worker evidence fixture")
        )
        if args.worker_evidence_json is not None
        else capture_worker_evidence(task_class, api_client)
    )
    revised = amend_plan(base_plan)
    change = validate_one_field_amendment(base_plan, revised)
    staging_evidence = build_amendment_evidence(
        change=change,
        attempt_binding=attempt_binding,
        recovery_binding=recovery_binding,
        deployment_binding=deployment_binding,
        worker_evidence=worker_evidence,
        producer_task_id=None,
    )
    source = Path(__file__).read_text(encoding="utf-8")
    precommit = _seal(
        {
            "schema_version": 1,
            "receipt_type": "resilient_v2x_formal_evaluation_plan_amendment",
            "generated_at_utc": _utc_timestamp(),
            "mode": "execute_precommit" if args.execute else "dry_run",
            "status": "staged",
            "remote_state_changed": False,
            "execute_token": EXECUTE_TOKEN,
            "base_plan_producer_task_id": BASE_PLAN_PRODUCER_TASK_ID,
            "base_plan_seal_sha256": BASE_PLAN_SEAL_SHA256,
            "revised_plan_seal_sha256": revised["seal_sha256"],
            "change": change,
            "ffnet_attempt_evidence": attempt_binding,
            "recovery_receipt": recovery_binding,
            "deployment_receipt": deployment_binding,
            "worker_stats_evidence": worker_evidence,
            "staging_amendment_evidence": staging_evidence,
            "producer_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "remote_write_contract": {
                "new_cpu_producer_only": True,
                "evaluation_tasks_mutated": False,
                "replacement_evaluation_tasks_created": False,
                "downstream_redeployment_is_separate": True,
            },
        }
    )
    if not args.execute:
        _write_new(args.receipt, precommit)
        print(json.dumps({"receipt": str(args.receipt), "seal_sha256": precommit["seal_sha256"], "revised_plan_seal_sha256": revised["seal_sha256"], "remote_state_changed": False}, sort_keys=True))
        return 0

    _write_new(args.attempt_receipt, precommit)
    evidence, producer = execute_producer(
        task_class,
        source=source,
        revised_plan=revised,
        attempt_binding=attempt_binding,
        recovery_binding=recovery_binding,
        deployment_binding=deployment_binding,
        worker_evidence=worker_evidence,
    )
    receipt = _seal(
        {
            **precommit,
            "generated_at_utc": _utc_timestamp(),
            "mode": "execute",
            "status": "completed",
            "remote_state_changed": True,
            "attempt_receipt_seal_sha256": precommit["seal_sha256"],
            "producer": producer,
            "amendment_evidence": evidence,
        }
    )
    _write_new(args.receipt, receipt)
    print(json.dumps({"receipt": str(args.receipt), "seal_sha256": receipt["seal_sha256"], "producer_task_id": producer["task_id"], "revised_plan_seal_sha256": revised["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
