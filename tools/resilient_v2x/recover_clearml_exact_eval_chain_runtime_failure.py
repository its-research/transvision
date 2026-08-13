#!/usr/bin/env python3
"""Recover failed FFNet and the exact candidate controller without cloning.

The default mode is read-only.  It validates the sealed 28-task recovery, the
deployed W/L/A/S chain, and both candidate-controller receipts, then writes a
local sealed plan.  Remote mutation requires ``--execute``, the exact token,
and distinct write-once attempt/final receipt paths.

Recovery never clones a task.  It resets and reinstalls only the frozen exact
IDs for FFNet and the candidate evaluation controller.  The failed old W/L/A/S
chain is retained as read-only causal evidence; a separately sealed amendment
owns the replacement W/L/A/S chain.  CoFormer and all other evaluation tasks
are checked but never mutated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

try:
    from tools.resilient_v2x import amend_clearml_formal_evaluation_plan as amendment
    from tools.resilient_v2x import (
        clearml_1337_dependency_watcher as formal,
    )
    from tools.resilient_v2x import (
        clearml_formal_candidate_evaluation_queue as candidate_queue,
    )
    from tools.resilient_v2x import (
        deploy_clearml_formal_candidate_evaluation_queue as candidate_deploy,
    )
    from tools.resilient_v2x import (
        deploy_clearml_formal_successor_chain as successor_deploy,
    )
    from tools.resilient_v2x import (
        recover_clearml_formal_candidate_controller as controller_recovery,
    )
    from tools.resilient_v2x import (
        recover_clearml_formal_evaluation_attempts as evaluation_recovery,
    )
except ModuleNotFoundError:  # direct execution from tools/resilient_v2x
    import amend_clearml_formal_evaluation_plan as amendment
    import clearml_1337_dependency_watcher as formal
    import clearml_formal_candidate_evaluation_queue as candidate_queue
    import deploy_clearml_formal_candidate_evaluation_queue as candidate_deploy
    import deploy_clearml_formal_successor_chain as successor_deploy
    import recover_clearml_formal_candidate_controller as controller_recovery
    import recover_clearml_formal_evaluation_attempts as evaluation_recovery


FFNET_TASK_ID = "144397bfa9c242bc9a92a1279922558b"
COFORMER_TASK_ID = "d8fac86325e047e5aa25f5ce899902b6"
SUCCESSOR_TASK_IDS = {
    "W": "40094c850391452da03df4f6b2499b90",
    "L": "70f1ceb80c1a4fceb4cf88343e360231",
    "A": "d985d344576b4718b59b4c207e2442c2",
    "S": "edb75f8a5c81413186eb866f9bd9855b",
}
CANDIDATE_CONTROLLER_TASK_ID = "b6f0fbab32a5478183a45b3ca833fc01"
RESET_ORDER = ("FFNet", "C")
# FFNet deliberately remains created+unqueued for the amended W to release.
ENQUEUE_ORDER = ("C",)
TARGET_TASK_IDS = {
    "FFNet": FFNET_TASK_ID,
    "C": CANDIDATE_CONTROLLER_TASK_ID,
}

EXPECTED_EVALUATION_RECOVERY_SEAL = (
    "d95e2bc2bb93ea6c8c69ea683d31ba09ec2bf219c7a0c09591c0ff12de60fcdf"
)
EXPECTED_SUCCESSOR_DEPLOYMENT_SEAL = (
    "f54d6a51c1fba1f5c776e236ed79087922da61b71d4b4997f76c0bb1c6a5bb6d"
)
EXPECTED_CONTROLLER_RECOVERY_SEAL = (
    "2974d130bd95131a4f31ac96f51c3362876ee7ab89ed547564540d524f6459b4"
)
EXPECTED_CANDIDATE_DEPLOYMENT_SEAL = (
    "e1231be77286bf5a8faf024a1a82ed9037ecfd855adadb226d3051543225eaaa"
)
EXPECTED_SOURCE_SHA256 = {
    "FFNet": "982f607cc7c7cc34872c68ce9c662bb7635d66d15bc7e0fc8cfc9ce70d7c3c3c",
    "W": "6fb8ac2bdcfb3806c22e6331cd404c2255f7d8e33e16f46f9959534ed1cd11f2",
    "L": "1da0cf5dd4435c6ae85d5a474b5a67ec8435f50e9878a3d8f0c1d2872356524b",
    "A": "5381c8e3f444cb3798c4bfca23b42a9007332c09e3178fb03652bfb75fd2fe58",
    "S": "8dd9352d987c14c9c03945c0f9e5abe32091051ba28cb8c49a9a03f49127028c",
    "C": "a45c090020d6fd3c66f27e92baee88956e96056f702fcff6df70bb091dae2e02",
}
EXPECTED_PARAMETERS_SHA256 = {
    "FFNet": "583df2fe1aba3a2e714e46e0cec4b4bb8c385cc4c35d450cc28d8bab9b3f8741",
    "W": "82c3cf80e3ed86ba9145ce1ca7e264c5cd265fb4b29f322f7f53ccec7b860b13",
    "L": "e946129860492fc371a984cfb2e75805a61930a4f41dd32fa2b187e022ccb178",
    "A": "aa9b02ab707f239271e6e022172f8a4f9264c9bb1ddcd0a0643702eb8a8293c2",
    "S": "34b4deabfc959233598c5bcc8634ddbaa6a66c80fc31e047d6eb949a64c16355",
    "C": "065349e5c5c1e9c2f3c10d43d208fca4212d5fd7947c9c58cd9ce5e4c64c7533",
}
EXPECTED_FAILURE_MESSAGES = {
    "FFNet": "assigned GPUs are not idle",
    "W": "evaluation task ffnet ended as 'failed'",
    "L": "evaluation watcher ended as 'failed'",
    "A": "evaluation watcher ended as 'failed'",
    "S": "comparability audit ended as 'failed'",
    "C": "formal core ffnet evaluation ended as 'failed'",
}
EXPECTED_W_PLAN_ARTIFACT_SHA256 = (
    "66ee8180ee13a34ff1519b204ddb71437e729f168466a38dbefa26c7eb8a4f57"
)
EXPECTED_W_PLAN_ARTIFACT_BYTES = 16_552
EXPECTED_W_PLAN_SEAL = (
    "c5d14aa8021d06609c7a7e9a5401f4b5a0a417601fa8c95163f083eeb397e0f3"
)
AMENDMENT_RECEIPT_TYPE = "resilient_v2x_formal_evaluation_plan_amendment"
AMENDMENT_SOURCE_QUEUE = "GPU4-V100"
AMENDMENT_TARGET_QUEUE = "GPU4-A100"
AMENDMENT_PRODUCER_NAME = amendment.TASK_NAME
AMENDMENT_PRODUCER_ENTRY_POINT = amendment.ENTRY_POINT
AMENDMENT_PLAN_ARTIFACT = amendment.REVISED_PLAN_ARTIFACT
AMENDMENT_EVIDENCE_ARTIFACT = amendment.AMENDMENT_ARTIFACT
AMENDED_CANDIDATE_TAG = "ffnet-queue-amended-gpu4-a100"
SERVICE_ENVIRONMENT_POLICY = {
    "binary": "python",
    "requirement": successor_deploy.SERVICE_REQUIREMENTS[0],
    "docker": " ".join(
        (
            f"{successor_deploy.SERVICE_DOCKER_IMAGE} "
            f"{successor_deploy.SERVICE_DOCKER_ARGS}"
        ).split()
    ),
    "output_uri": successor_deploy.FILES_SERVER_URI,
    "allowed_requirement_forms": ["canonical", "agent_materialized"],
}

EXECUTE_TOKEN = (
    "RESET_EXACT_FFNET_AND_CANDIDATE_CONTROLLER_"
    "144397BFA9C242BC9A92A1279922558B"
)
ATTEMPT_RECEIPT_TYPE = "resilient_v2x_exact_eval_chain_runtime_recovery_attempt"
PLAN_RECEIPT_TYPE = "resilient_v2x_exact_eval_chain_runtime_recovery_plan"
RECOVERY_RECEIPT_TYPE = "resilient_v2x_exact_eval_chain_runtime_recovery"
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

FORMAL_EVALUATION_ARTIFACTS = frozenset(
    {
        "run_contract",
        "evaluation_plan",
        "controlled_baseline_metrics",
        "controlled_baseline_evidence",
    }
)
ROLE_OUTPUT_ARTIFACTS = {
    "W": frozenset({"formal_1337_evaluation_plan"}),
    "L": frozenset({"formal_1337_leaderboard"}),
    "A": frozenset({"formal_1337_comparability_audit"}),
    "S": frozenset({"final_selector_formal_inputs", "formal_candidate_selection"}),
    "C": frozenset({"formal_1337_candidate_evaluation_manifest"}),
}
BOOTSTRAP_EMPTY_DEFAULT_PARAMETERS = frozenset(
    {
        "Args/teacher_checkpoint",
        "Args/student_checkpoint",
        "Args/experiment_from_task",
        "Args/teacher_task_id",
        "Args/teacher_model_id",
        "Args/teacher_checkpoint_sha256",
        "Args/allow_failed_teacher_task",
        "Args/student_task_id",
        "Args/student_model_id",
        "Args/student_checkpoint_sha256",
    }
)
SUCCESSOR_RUNTIME_DEFAULT_PARAMETERS = {
    "W": {
        "Args/dependencies_json": "",
        "Args/evaluation_plan_json": "",
        "Args/expected_training_script_sha256": "",
        "Args/expected_training_source_archive_sha256": "",
        "Args/expected_training_source_dataset_id": "",
        "Args/metadata_only_artifact_gate": "False",
        "Args/training_manifest_json": "",
    },
    "S": {"Args/formal_inputs_artifact": "final_selector_formal_inputs"},
}


class RuntimeRecoveryError(RuntimeError):
    """Raised when exact-ID runtime recovery cannot be proven."""


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


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
        raise RuntimeRecoveryError("recovery evidence is not canonical JSON") from error


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seal(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


def _sha256(value: object, *, context: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise RuntimeRecoveryError(f"{context} is not a lowercase SHA-256")
    return value


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if len(payload.encode("utf-8")) > MAX_RECEIPT_BYTES:
        raise RuntimeRecoveryError("write-once receipt exceeds its byte cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _require_new(path: Path, *, context: str) -> None:
    if path.is_symlink() or path.exists():
        raise RuntimeRecoveryError(f"{context} must be a new write-once path")


def _read_sealed(path: Path, *, context: str) -> tuple[dict[str, object], Path]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeRecoveryError(f"{context} cannot be resolved") from error
    if path.is_symlink() or not resolved.is_file():
        raise RuntimeRecoveryError(f"{context} must be a regular non-symlink file")
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise RuntimeRecoveryError(f"{context} cannot be read") from error
    if not raw or len(raw) > MAX_RECEIPT_BYTES:
        raise RuntimeRecoveryError(f"{context} size is invalid")
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise RuntimeRecoveryError(
                    f"{context} contains duplicate key {key!r}"
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda item: (_ for _ in ()).throw(
                RuntimeRecoveryError(
                    f"{context} contains non-finite value {item}"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeRecoveryError(f"{context} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeRecoveryError(f"{context} is not a JSON object")
    observed = _sha256(value.get("seal_sha256"), context=f"{context} seal")
    if _seal(value)["seal_sha256"] != observed:
        raise RuntimeRecoveryError(f"{context} seal mismatch")
    return value, resolved


def _evaluation_binding(path: Path) -> dict[str, object]:
    try:
        value = evaluation_recovery.validate_recovery_receipt(path)
    except evaluation_recovery.RecoveryError as error:
        raise RuntimeRecoveryError("evaluation recovery receipt is invalid") from error
    if value.get("receipt_seal_sha256") != EXPECTED_EVALUATION_RECOVERY_SEAL:
        raise RuntimeRecoveryError("evaluation recovery receipt is not the frozen d95e receipt")
    return dict(value)


def _successor_binding(
    path: Path, *, evaluation_binding: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    value, resolved = _read_sealed(path, context="successor deployment receipt")
    expected = {
        "schema_version": 1,
        "receipt_type": "resilient_v2x_formal_successor_chain_deployment",
        "status": "deployed",
        "mode": "completed_provenance_successor_execute",
        "remote_state_changed": True,
        "seal_sha256": EXPECTED_SUCCESSOR_DEPLOYMENT_SEAL,
        "created_roles": ["W", "L", "A", "S"],
        "created_task_ids": dict(SUCCESSOR_TASK_IDS),
        "enqueue_order": ["W", "L", "A", "S"],
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise RuntimeRecoveryError(f"successor deployment receipt {key} drifted")
    expected_ids = {
        "P": successor_deploy.COMPLETED_PROVENANCE_TASK_ID,
        **SUCCESSOR_TASK_IDS,
    }
    if value.get("task_ids") != expected_ids:
        raise RuntimeRecoveryError("successor deployment exact task IDs drifted")
    if value.get("evaluation_recovery_receipt") != evaluation_binding:
        raise RuntimeRecoveryError("successor deployment evaluation binding drifted")
    parents = value.get("parents")
    parameters = value.get("parameters")
    sources = value.get("sources")
    queued = value.get("queued_authority")
    if not all(isinstance(item, Mapping) for item in (parents, parameters, sources, queued)):
        raise RuntimeRecoveryError("successor deployment material is incomplete")
    expected_parents = {
        "W": successor_deploy.COMPLETED_PROVENANCE_TASK_ID,
        "L": SUCCESSOR_TASK_IDS["W"],
        "A": SUCCESSOR_TASK_IDS["L"],
        "S": SUCCESSOR_TASK_IDS["A"],
    }
    if dict(parents) != expected_parents:
        raise RuntimeRecoveryError("successor deployment parents drifted")
    for role in ("W", "L", "A", "S"):
        source = sources.get(role)
        authority = queued.get(role)
        role_parameters = parameters.get(role)
        if not isinstance(source, Mapping) or not isinstance(authority, Mapping):
            raise RuntimeRecoveryError(f"successor role {role} evidence is absent")
        if not isinstance(role_parameters, Mapping):
            raise RuntimeRecoveryError(f"successor role {role} parameters are absent")
        if (
            source.get("deployed_sha256") != EXPECTED_SOURCE_SHA256[role]
            or authority.get("script_sha256") != EXPECTED_SOURCE_SHA256[role]
            or authority.get("parameters_sha256") != EXPECTED_PARAMETERS_SHA256[role]
            or _content_sha256({str(k): str(v) for k, v in role_parameters.items()})
            != EXPECTED_PARAMETERS_SHA256[role]
            or authority.get("task_id") != SUCCESSOR_TASK_IDS[role]
            or authority.get("parent_task_id") != expected_parents[role]
            or authority.get("queue") != successor_deploy.SERVICES_QUEUE
        ):
            raise RuntimeRecoveryError(f"successor role {role} authority drifted")
    return (
        {
            "path": str(resolved),
            "receipt_seal_sha256": EXPECTED_SUCCESSOR_DEPLOYMENT_SEAL,
            "evaluation_recovery_receipt_seal_sha256": (
                EXPECTED_EVALUATION_RECOVERY_SEAL
            ),
            "task_ids": dict(SUCCESSOR_TASK_IDS),
        },
        value,
    )


def _controller_recovery_binding(path: Path) -> dict[str, object]:
    try:
        value = controller_recovery.validate_controller_recovery_receipt(path)
    except controller_recovery.ControllerRecoveryError as error:
        raise RuntimeRecoveryError("candidate controller recovery receipt is invalid") from error
    expected = {
        "receipt_seal_sha256": EXPECTED_CONTROLLER_RECOVERY_SEAL,
        "controller_task_id": CANDIDATE_CONTROLLER_TASK_ID,
        "evaluation_recovery_receipt_seal_sha256": (
            EXPECTED_EVALUATION_RECOVERY_SEAL
        ),
        "target_source_sha256": EXPECTED_SOURCE_SHA256["C"],
        "same_id_only": True,
        "replacement_controller_created": False,
        "created_unqueued": True,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise RuntimeRecoveryError(f"candidate controller recovery {key} drifted")
    return dict(value)


def _candidate_deployment_binding(
    path: Path,
    *,
    evaluation_binding: Mapping[str, object],
    controller_binding: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    value, resolved = _read_sealed(path, context="candidate deployment receipt")
    expected = {
        "schema_version": 1,
        "receipt_type": "resilient_v2x_formal_candidate_evaluation_queue_deployment",
        "status": "deployed",
        "mode": "execute",
        "remote_state_changed": True,
        "created_shell": False,
        "disposition": "resumed_exact_created_shell",
        "task_id": CANDIDATE_CONTROLLER_TASK_ID,
        "seal_sha256": EXPECTED_CANDIDATE_DEPLOYMENT_SEAL,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise RuntimeRecoveryError(f"candidate deployment receipt {key} drifted")
    if value.get("evaluation_recovery_receipt") != evaluation_binding:
        raise RuntimeRecoveryError("candidate deployment evaluation binding drifted")
    if value.get("candidate_controller_recovery_receipt") != controller_binding:
        raise RuntimeRecoveryError("candidate deployment controller binding drifted")
    source = value.get("source")
    authority = value.get("queued_authority")
    if (
        not isinstance(source, Mapping)
        or source.get("sha256") != EXPECTED_SOURCE_SHA256["C"]
        or not isinstance(authority, Mapping)
        or authority.get("task_id") != CANDIDATE_CONTROLLER_TASK_ID
        or authority.get("source_sha256") != EXPECTED_SOURCE_SHA256["C"]
        or authority.get("parameters_sha256") != EXPECTED_PARAMETERS_SHA256["C"]
        or authority.get("queue") != candidate_deploy.SERVICES_QUEUE
        or authority.get("queue_id") != candidate_deploy.SERVICES_QUEUE_ID
    ):
        raise RuntimeRecoveryError("candidate deployment authority drifted")
    return (
        {
            "path": str(resolved),
            "receipt_seal_sha256": EXPECTED_CANDIDATE_DEPLOYMENT_SEAL,
            "controller_recovery_receipt_seal_sha256": (
                EXPECTED_CONTROLLER_RECOVERY_SEAL
            ),
            "evaluation_recovery_receipt_seal_sha256": (
                EXPECTED_EVALUATION_RECOVERY_SEAL
            ),
            "controller_task_id": CANDIDATE_CONTROLLER_TASK_ID,
        },
        value,
    )


def _amendment_binding(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    try:
        validated = amendment.validate_amendment_receipt(path)
    except amendment.AmendmentError as error:
        raise RuntimeRecoveryError("formal plan amendment receipt is invalid") from error
    value, resolved = _read_sealed(path, context="formal plan amendment receipt")
    expected = {
        "schema_version": 1,
        "receipt_type": AMENDMENT_RECEIPT_TYPE,
        "mode": "execute",
        "status": "completed",
        "remote_state_changed": True,
        "base_plan_producer_task_id": amendment.BASE_PLAN_PRODUCER_TASK_ID,
        "base_plan_seal_sha256": EXPECTED_W_PLAN_SEAL,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise RuntimeRecoveryError(f"formal plan amendment {key} drifted")
    if validated.get("path") != str(resolved):
        raise RuntimeRecoveryError("formal plan amendment resolved path drifted")
    recovery = value.get("recovery_receipt")
    deployment = value.get("deployment_receipt")
    if (
        not isinstance(recovery, Mapping)
        or recovery.get("receipt_seal_sha256")
        != EXPECTED_EVALUATION_RECOVERY_SEAL
        or recovery.get("all_tasks_created_unqueued") is not True
        or recovery.get("replacement_tasks_created") is not False
        or not isinstance(deployment, Mapping)
        or deployment.get("receipt_seal_sha256")
        != EXPECTED_SUCCESSOR_DEPLOYMENT_SEAL
    ):
        raise RuntimeRecoveryError("formal plan amendment authorization drifted")
    change = value.get("change")
    revised_seal = _sha256(
        value.get("revised_plan_seal_sha256"),
        context="amendment revised plan seal",
    )
    expected_change = {
        "base_plan_seal_sha256": EXPECTED_W_PLAN_SEAL,
        "revised_plan_seal_sha256": revised_seal,
        "changed_json_pointer": (
            f"/entries/{formal.FORMAL_SUBJECT_ORDER.index('ffnet')}/queue"
        ),
        "subject": "ffnet",
        "evaluation_task_id": FFNET_TASK_ID,
        "from_queue": AMENDMENT_SOURCE_QUEUE,
        "to_queue": AMENDMENT_TARGET_QUEUE,
        "evaluation_task_ids_sha256": _content_sha256(
            [row[2] for row in evaluation_recovery.EXPECTED_RECOVERY_TASKS[:26]]
        ),
        "fixed_baselines": list(amendment.FIXED_BASELINES),
        "protocol_id": formal.EXPECTED_PROTOCOL_ID,
        "sample_count": formal.EXPECTED_SAMPLE_COUNT,
        "changed_field_count": 1,
    }
    if change != expected_change:
        raise RuntimeRecoveryError("formal plan amendment changed more than FFNet queue")
    producer = value.get("producer")
    evidence = value.get("amendment_evidence")
    worker_evidence = value.get("worker_stats_evidence")
    attempt_evidence = value.get("ffnet_attempt_evidence")
    if not isinstance(producer, Mapping) or not isinstance(evidence, Mapping):
        raise RuntimeRecoveryError("formal plan amendment producer evidence is absent")
    producer_id = str(producer.get("task_id") or "")
    if len(producer_id) != 32 or any(
        character not in "0123456789abcdef" for character in producer_id
    ):
        raise RuntimeRecoveryError("formal plan amendment producer ID is invalid")
    producer_source_sha = _sha256(
        producer.get("source_sha256"), context="amendment producer source"
    )
    if (
        producer_source_sha != value.get("producer_source_sha256")
        or producer.get("status") != "completed"
        or producer.get("parent_task_id") != amendment.BASE_PLAN_PRODUCER_TASK_ID
        or not isinstance(producer.get("parameters"), Mapping)
        or producer.get("tags") != sorted(amendment.TASK_TAGS)
    ):
        raise RuntimeRecoveryError("formal plan amendment producer identity drifted")
    evidence_seal = _sha256(
        evidence.get("seal_sha256"), context="formal plan amendment evidence"
    )
    worker_evidence_seal = _sha256(
        validated.get("worker_stats_evidence_seal_sha256"),
        context="formal plan amendment worker evidence",
    )
    evaluation_ids_sha256 = _sha256(
        validated.get("evaluation_task_ids_sha256"),
        context="formal plan amendment evaluation inventory",
    )
    if (
        _seal(evidence)["seal_sha256"] != evidence_seal
        or evidence.get("schema_version") != 1
        or evidence.get("evidence_type")
        != "resilient_v2x_formal_1337_evaluation_plan_amendment"
        or evidence.get("producer_task_id") != producer_id
        or evidence.get("amendment") != expected_change
        or not isinstance(worker_evidence, Mapping)
        or worker_evidence.get("seal_sha256") != worker_evidence_seal
        or evidence.get("worker_stats_evidence") != worker_evidence
        or not isinstance(evidence.get("invariants"), Mapping)
        or set(evidence["invariants"].values()) != {True, False}
        or evidence["invariants"].get("replacement_tasks_created") is not False
        or any(
            evidence["invariants"].get(key) is not True
            for key in (
                "one_field_amendment",
                "evaluation_task_ids_unchanged",
                "evaluation_parameters_unchanged",
                "evaluation_sources_unchanged",
                "evaluation_models_unchanged",
                "protocol_and_sample_count_unchanged",
                "fixed_five_baselines_unchanged",
            )
        )
    ):
        raise RuntimeRecoveryError("formal plan amendment evidence drifted")
    authorization = evidence.get("authorization")
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("evaluation_recovery_receipt_seal_sha256")
        != EXPECTED_EVALUATION_RECOVERY_SEAL
        or authorization.get("successor_deployment_receipt_seal_sha256")
        != EXPECTED_SUCCESSOR_DEPLOYMENT_SEAL
        or not isinstance(attempt_evidence, Mapping)
        or authorization.get("ffnet_attempt_evidence") != attempt_evidence
    ):
        raise RuntimeRecoveryError("formal plan amendment evidence authorization drifted")
    validated_expected = {
        "receipt_seal_sha256": value["seal_sha256"],
        "producer_task_id": producer_id,
        "producer_source_sha256": producer_source_sha,
        "revised_plan_seal_sha256": revised_seal,
        "amendment_evidence_seal_sha256": evidence_seal,
        "worker_stats_evidence_seal_sha256": worker_evidence_seal,
        "evaluation_task_ids_sha256": evaluation_ids_sha256,
        "from_queue": AMENDMENT_SOURCE_QUEUE,
        "to_queue": AMENDMENT_TARGET_QUEUE,
        "ffnet_evaluation_task_id": FFNET_TASK_ID,
    }
    for key, wanted in validated_expected.items():
        if validated.get(key) != wanted:
            raise RuntimeRecoveryError(f"formal plan amendment validator {key} drifted")
    return (
        {
            "path": str(resolved),
            "receipt_seal_sha256": value["seal_sha256"],
            "producer_task_id": producer_id,
            "producer_source_sha256": producer_source_sha,
            "producer_parameters": dict(producer["parameters"]),
            "producer_tags": list(producer["tags"]),
            "amendment_evidence_seal_sha256": evidence_seal,
            "worker_stats_evidence_seal_sha256": worker_evidence_seal,
            "evaluation_task_ids_sha256": evaluation_ids_sha256,
            "revised_plan_seal_sha256": revised_seal,
            "change": dict(expected_change),
        },
        value,
    )


def _candidate_amendment_source(
    base_source: str, *, amendment_binding: Mapping[str, object]
) -> str:
    old_core = '''    {
        "subject": "ffnet",
        "task_id": "144397bfa9c242bc9a92a1279922558b",
        "index": 11,
        "queue": "GPU4-V100",
    },'''
    new_core = old_core.replace('"queue": "GPU4-V100"', '"queue": "GPU4-A100"')
    if base_source.count(old_core) != 1:
        raise RuntimeRecoveryError("candidate FFNet core queue anchor drifted")
    source = base_source.replace(old_core, new_core, 1)
    anchor = "FORMAL_CORE_EVALUATIONS = (\n"
    if source.count(anchor) != 1:
        raise RuntimeRecoveryError("candidate amendment constant anchor drifted")
    constants = (
        f'FORMAL_PLAN_AMENDMENT_PRODUCER_TASK_ID = '
        f'{amendment_binding["producer_task_id"]!r}\n'
        f'FORMAL_PLAN_AMENDMENT_PRODUCER_SOURCE_SHA256 = '
        f'{amendment_binding["producer_source_sha256"]!r}\n'
        f'FORMAL_PLAN_AMENDMENT_RECEIPT_SEAL_SHA256 = '
        f'{amendment_binding["receipt_seal_sha256"]!r}\n'
        f'FORMAL_PLAN_AMENDMENT_EVIDENCE_SEAL_SHA256 = '
        f'{amendment_binding["amendment_evidence_seal_sha256"]!r}\n'
        f'FORMAL_PLAN_AMENDMENT_WORKER_STATS_SEAL_SHA256 = '
        f'{amendment_binding["worker_stats_evidence_seal_sha256"]!r}\n'
        f'FORMAL_EVALUATION_TASK_IDS_SHA256 = '
        f'{amendment_binding["evaluation_task_ids_sha256"]!r}\n'
        f'REVISED_FORMAL_PLAN_SEAL_SHA256 = '
        f'{amendment_binding["revised_plan_seal_sha256"]!r}\n'
        f'FORMAL_PLAN_AMENDMENT_PRODUCER_PARAMETERS = '
        f'{dict(amendment_binding["producer_parameters"])!r}\n'
        f'FORMAL_PLAN_AMENDMENT_PRODUCER_TAGS = '
        f'{tuple(amendment_binding["producer_tags"])!r}\n\n'
    )
    source = source.replace(anchor, constants + anchor, 1)
    function_anchor = "def _formal_core_phase_snapshot(task_class: object) -> dict[str, object]:\n"
    if source.count(function_anchor) != 1:
        raise RuntimeRecoveryError("candidate amendment validation anchor drifted")
    validation_source = '''def _validate_formal_plan_amendment(task_class: object) -> dict[str, object]:
    task = task_class.get_task(task_id=FORMAL_PLAN_AMENDMENT_PRODUCER_TASK_ID)
    context = "formal plan amendment producer"
    if (
        _task_id(getattr(task, "id", ""), context)
        != FORMAL_PLAN_AMENDMENT_PRODUCER_TASK_ID
        or _task_project_id(task, context=context) != PROJECT_ID
        or str(getattr(task, "name", "") or "")
        != "ResilientV2X formal 1337 FFNet queue amendment producer"
        or _task_parent(task) != "dbc05b28fbd044bf89edc3872742843a"
        or _status(task) != "completed"
    ):
        raise RuntimeError("formal plan amendment producer identity drifted")
    script = _script(task)
    if (
        str(script.get("repository") or "") != ""
        or str(script.get("working_dir") or "") != "."
        or str(script.get("entry_point") or "")
        != "amend_clearml_formal_evaluation_plan.py"
        or hashlib.sha256(str(script.get("diff") or "").encode("utf-8")).hexdigest()
        != FORMAL_PLAN_AMENDMENT_PRODUCER_SOURCE_SHA256
        or _parameters(task) != FORMAL_PLAN_AMENDMENT_PRODUCER_PARAMETERS
    ):
        raise RuntimeError("formal plan amendment producer source/parameters drifted")
    tags = getattr(task, "tags", None)
    if tags is None:
        tags = getattr(getattr(task, "data", None), "tags", None)
    if tuple(sorted(str(item) for item in (tags or []))) != tuple(
        sorted(FORMAL_PLAN_AMENDMENT_PRODUCER_TAGS)
    ):
        raise RuntimeError("formal plan amendment producer tags drifted")
    if set(_artifact_records(task)) != {
        "formal_1337_evaluation_plan",
        "formal_1337_evaluation_plan_amendment",
    }:
        raise RuntimeError("formal plan amendment artifact inventory drifted")
    plan = _artifact_payload(task, "formal_1337_evaluation_plan")
    if (
        _require_seal(plan, context="revised formal evaluation plan")
        != REVISED_FORMAL_PLAN_SEAL_SHA256
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("sample_count") != SAMPLE_COUNT
        or plan.get("run_count") != RUN_COUNT
    ):
        raise RuntimeError("revised formal evaluation plan identity drifted")
    entries = plan.get("entries")
    if not isinstance(entries, list) or len(entries) != 26:
        raise RuntimeError("revised formal evaluation entries drifted")
    ffnet = [entry for entry in entries if isinstance(entry, Mapping) and entry.get("subject") == "ffnet"]
    if len(ffnet) != 1 or ffnet[0] != {
        "subject": "ffnet",
        "evaluation_task_id": "144397bfa9c242bc9a92a1279922558b",
        "queue": "GPU4-A100",
    }:
        raise RuntimeError("revised formal FFNet queue binding drifted")
    evidence = _artifact_payload(task, "formal_1337_evaluation_plan_amendment")
    if (
        _require_seal(evidence, context="formal plan amendment evidence")
        != FORMAL_PLAN_AMENDMENT_EVIDENCE_SEAL_SHA256
        or evidence.get("producer_task_id")
        != FORMAL_PLAN_AMENDMENT_PRODUCER_TASK_ID
        or evidence.get("amendment", {}).get("revised_plan_seal_sha256")
        != REVISED_FORMAL_PLAN_SEAL_SHA256
        or evidence.get("amendment", {}).get("changed_field_count") != 1
        or evidence.get("amendment", {}).get("evaluation_task_id")
        != "144397bfa9c242bc9a92a1279922558b"
        or evidence.get("amendment", {}).get("from_queue") != "GPU4-V100"
        or evidence.get("amendment", {}).get("to_queue") != "GPU4-A100"
        or evidence.get("amendment", {}).get("evaluation_task_ids_sha256")
        != FORMAL_EVALUATION_TASK_IDS_SHA256
        or not isinstance(evidence.get("worker_stats_evidence"), Mapping)
        or _require_seal(
            evidence["worker_stats_evidence"],
            context="formal plan amendment worker evidence",
        )
        != FORMAL_PLAN_AMENDMENT_WORKER_STATS_SEAL_SHA256
    ):
        raise RuntimeError("formal plan amendment evidence drifted")
    return {
        "producer_task_id": FORMAL_PLAN_AMENDMENT_PRODUCER_TASK_ID,
        "receipt_seal_sha256": FORMAL_PLAN_AMENDMENT_RECEIPT_SEAL_SHA256,
        "amendment_evidence_seal_sha256": FORMAL_PLAN_AMENDMENT_EVIDENCE_SEAL_SHA256,
        "worker_stats_evidence_seal_sha256": FORMAL_PLAN_AMENDMENT_WORKER_STATS_SEAL_SHA256,
        "evaluation_task_ids_sha256": FORMAL_EVALUATION_TASK_IDS_SHA256,
        "revised_plan_seal_sha256": REVISED_FORMAL_PLAN_SEAL_SHA256,
        "ffnet_queue": "GPU4-A100",
    }


'''
    source = source.replace(function_anchor, validation_source + function_anchor, 1)
    reconcile_anchor = "    formal_core_gate = _formal_core_phase_snapshot(task_class)\n"
    if source.count(reconcile_anchor) != 1:
        raise RuntimeRecoveryError("candidate reconcile amendment anchor drifted")
    reconcile = (
        "    formal_plan_amendment = _validate_formal_plan_amendment(task_class)\n"
        "    formal_core_gate = _formal_core_phase_snapshot(task_class)\n"
        "    formal_core_gate = {\n"
        "        **formal_core_gate,\n"
        "        \"formal_plan_amendment\": formal_plan_amendment,\n"
        "    }\n"
    )
    source = source.replace(reconcile_anchor, reconcile, 1)
    runtime_tags_anchor = '''                "single-seed-20250218",
                "cpu-controller",
            ]'''
    amended_runtime_tags = '''                "single-seed-20250218",
                "cpu-controller",
                "ffnet-queue-amended-gpu4-a100",
            ]'''
    if source.count(runtime_tags_anchor) != 1:
        raise RuntimeRecoveryError("candidate runtime tag anchor drifted")
    source = source.replace(runtime_tags_anchor, amended_runtime_tags, 1)
    try:
        compile(source, "clearml_formal_candidate_evaluation_queue.py", "exec")
    except SyntaxError as error:
        raise RuntimeRecoveryError("amendment-aware candidate source is invalid") from error
    return source


def _candidate_target_material(
    *,
    pre_reset: Mapping[str, object],
    amendment_binding: Mapping[str, object],
) -> dict[str, object]:
    base_source = str(pre_reset.get("source") or "")
    source = _candidate_amendment_source(
        base_source, amendment_binding=amendment_binding
    )
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    runtime_parameters = {
        key: value
        for key, value in dict(pre_reset["parameters"]).items()
        if key.startswith("Args/")
    }
    runtime_parameters.update(
        {
            "Args/formal_plan_amendment_producer_task_id": amendment_binding[
                "producer_task_id"
            ],
            "Args/formal_plan_amendment_seal_sha256": amendment_binding[
                "amendment_evidence_seal_sha256"
            ],
            "Args/formal_plan_amendment_receipt_seal_sha256": amendment_binding[
                "receipt_seal_sha256"
            ],
            "Args/revised_formal_plan_seal_sha256": amendment_binding[
                "revised_plan_seal_sha256"
            ],
        }
    )
    deployment_contract = {
        "schema_version": 2,
        "deployment_type": (
            "resilient_v2x_formal_candidate_evaluation_queue_amended"
        ),
        "base_candidate_deployment_receipt_seal_sha256": (
            EXPECTED_CANDIDATE_DEPLOYMENT_SEAL
        ),
        "formal_plan_amendment_receipt_seal_sha256": amendment_binding[
            "receipt_seal_sha256"
        ],
        "formal_plan_amendment_producer_task_id": amendment_binding[
            "producer_task_id"
        ],
        "formal_plan_amendment_evidence_seal_sha256": amendment_binding[
            "amendment_evidence_seal_sha256"
        ],
        "revised_formal_plan_seal_sha256": amendment_binding[
            "revised_plan_seal_sha256"
        ],
        "source_sha256": source_sha,
        "runtime_parameters": {str(key): str(value) for key, value in runtime_parameters.items()},
        "parent_task_id": candidate_deploy.PARENT_TASK_ID,
        "queue": candidate_deploy.SERVICES_QUEUE,
        "queue_id": candidate_deploy.SERVICES_QUEUE_ID,
        "same_controller_task_id": CANDIDATE_CONTROLLER_TASK_ID,
    }
    deployment_seal = _content_sha256(deployment_contract)
    parameters = {
        **runtime_parameters,
        "Deployment/schema_version": "2",
        "Deployment/deployment_seal_sha256": deployment_seal,
        "Deployment/source_sha256": source_sha,
        "Deployment/services_queue_id": candidate_deploy.SERVICES_QUEUE_ID,
        "Deployment/parent_task_id": candidate_deploy.PARENT_TASK_ID,
    }
    parameters = {str(key): str(value) for key, value in parameters.items()}
    tags = sorted(
        {
            *(str(item) for item in pre_reset["tags"]),
            AMENDED_CANDIDATE_TAG,
        }
    )
    return {
        **dict(pre_reset),
        "source": source,
        "source_sha256": source_sha,
        "source_bytes": len(source.encode("utf-8")),
        "parameters": parameters,
        "parameters_sha256": _content_sha256(parameters),
        "tags": tags,
        "pre_reset_source_sha256": pre_reset["source_sha256"],
        "pre_reset_source": pre_reset["source"],
        "pre_reset_parameters_sha256": pre_reset["parameters_sha256"],
        "pre_reset_parameters": dict(pre_reset["parameters"]),
        "pre_reset_tags": list(pre_reset["tags"]),
        "pre_reset_queue": pre_reset["queue"],
        "pre_reset_queue_id": pre_reset["queue_id"],
        "deployment_contract": deployment_contract,
        "deployment_seal_sha256": deployment_seal,
        "formal_plan_amendment": dict(amendment_binding),
        "service_environment_policy": dict(SERVICE_ENVIRONMENT_POLICY),
    }


def _validate_remote_amendment_producer(
    task_class: object,
    *,
    binding: Mapping[str, object],
    base_plan: Mapping[str, object],
) -> dict[str, object]:
    task_id = str(binding.get("producer_task_id") or "")
    task = _authoritative_task(
        task_class, task_id, context="formal plan amendment producer"
    )
    if (
        _status(task) != "completed"
        or str(getattr(task, "name", "") or "") != AMENDMENT_PRODUCER_NAME
        or successor_deploy._task_parent(task)
        != amendment.BASE_PLAN_PRODUCER_TASK_ID
        or _exact_project_id(task, context="formal plan amendment producer")
        != amendment.PROJECT_ID
        or successor_deploy._project_name(task) != amendment.PROJECT_NAME
    ):
        raise RuntimeRecoveryError("remote amendment producer identity drifted")
    source = _source(
        task,
        entry_point=AMENDMENT_PRODUCER_ENTRY_POINT,
        context="formal plan amendment producer",
    )
    parameters = _parameters(task, context="formal plan amendment producer")
    tags = _tags(task, field="tags", context="formal plan amendment producer")
    if (
        hashlib.sha256(source.encode("utf-8")).hexdigest()
        != binding.get("producer_source_sha256")
        or parameters != binding.get("producer_parameters")
        or sorted(tags) != sorted(binding.get("producer_tags", []))
    ):
        raise RuntimeRecoveryError("remote amendment producer contract drifted")
    try:
        records = candidate_queue._artifact_records(task)
        if set(records) != {AMENDMENT_PLAN_ARTIFACT, AMENDMENT_EVIDENCE_ARTIFACT}:
            raise RuntimeRecoveryError(
                "remote amendment producer artifact inventory drifted"
            )
        revised = candidate_queue._artifact_payload(task, AMENDMENT_PLAN_ARTIFACT)
        evidence = candidate_queue._artifact_payload(
            task, AMENDMENT_EVIDENCE_ARTIFACT
        )
    except Exception as error:
        if isinstance(error, RuntimeRecoveryError):
            raise
        raise RuntimeRecoveryError(
            "remote amendment producer artifacts are invalid"
        ) from error
    try:
        change = amendment.validate_one_field_amendment(base_plan, revised)
    except amendment.AmendmentError as error:
        raise RuntimeRecoveryError("remote revised plan is not one-field") from error
    if (
        change != binding.get("change")
        or revised.get("seal_sha256")
        != binding.get("revised_plan_seal_sha256")
        or _seal(evidence)["seal_sha256"] != evidence.get("seal_sha256")
        or evidence.get("seal_sha256")
        != binding.get("amendment_evidence_seal_sha256")
        or evidence.get("producer_task_id") != task_id
        or evidence.get("amendment") != change
    ):
        raise RuntimeRecoveryError("remote amendment payload binding drifted")
    return {
        "task_id": task_id,
        "project_id": amendment.PROJECT_ID,
        "status": "completed",
        "source_sha256": binding["producer_source_sha256"],
        "parameters_sha256": _content_sha256(parameters),
        "tags": sorted(tags),
        "artifact_names": sorted(records),
        "revised_plan_seal_sha256": binding["revised_plan_seal_sha256"],
        "amendment_evidence_seal_sha256": binding[
            "amendment_evidence_seal_sha256"
        ],
    }


def _authoritative_task(task_class: object, task_id: str, *, context: str) -> object:
    try:
        return successor_deploy._authoritative_task(
            task_class, task_id, context=context
        )
    except successor_deploy.DeploymentError as error:
        raise RuntimeRecoveryError(f"{context} authority readback failed") from error


def _status(task: object) -> str:
    return successor_deploy._normalized_status(task)


def _source(task: object, *, entry_point: str, context: str) -> str:
    try:
        return successor_deploy._script_source(
            task, entry_point=entry_point, context=context
        )
    except successor_deploy.DeploymentError as error:
        raise RuntimeRecoveryError(f"{context} source identity drifted") from error


def _parameters(task: object, *, context: str) -> dict[str, object]:
    try:
        return successor_deploy._task_parameters(task, context=context)
    except successor_deploy.DeploymentError as error:
        raise RuntimeRecoveryError(f"{context} parameters are unavailable") from error


def _evaluation_parameters_sha256(parameters: Mapping[str, object]) -> str:
    normalized = {str(key): str(value) for key, value in parameters.items()}
    allowed_empty = {
        key: normalized.pop(key)
        for key in sorted(BOOTSTRAP_EMPTY_DEFAULT_PARAMETERS & set(normalized))
    }
    if any(value not in {"", "False"} for value in allowed_empty.values()) or (
        allowed_empty.get("Args/allow_failed_teacher_task", "False") != "False"
    ):
        raise RuntimeRecoveryError("evaluation bootstrap default parameters drifted")
    return _content_sha256(normalized)


def _successor_deployment_parameters(
    label: str, parameters: Mapping[str, object]
) -> dict[str, str]:
    normalized = {str(key): str(value) for key, value in parameters.items()}
    defaults = SUCCESSOR_RUNTIME_DEFAULT_PARAMETERS.get(label, {})
    for key, expected in defaults.items():
        if key in normalized and normalized.pop(key) != expected:
            raise RuntimeRecoveryError(
                f"{label} runtime default parameter {key} drifted"
            )
    return normalized


def _exact_project_id(task: object, *, context: str) -> str:
    expected = evaluation_recovery.PROJECT_ID
    values = {
        str(value)
        for value in (
            getattr(task, "project", None),
            getattr(getattr(task, "data", None), "project", None),
            getattr(getattr(task, "_data", None), "project", None),
        )
        if value not in {None, ""}
    }
    if values != {expected}:
        raise RuntimeRecoveryError(f"{context} project ID drifted")
    return expected


def _tags(task: object, *, field: str, context: str) -> list[str]:
    try:
        return list(
            successor_deploy._task_tag_values(task, field=field, context=context)
        )
    except successor_deploy.DeploymentError as error:
        raise RuntimeRecoveryError(f"{context} {field} drifted") from error


def _artifact_inventory(task: object) -> list[dict[str, object]]:
    return evaluation_recovery._artifact_inventory(task)


def _artifact_inventory_is_monotonic(
    frozen: object,
    current: object,
    *,
    allowed_names: frozenset[str],
) -> bool:
    if not isinstance(frozen, list) or not isinstance(current, list):
        return False
    if any(not isinstance(row, Mapping) for row in frozen + current):
        return False
    frozen_by_name = {str(row.get("name") or ""): dict(row) for row in frozen}
    current_by_name = {str(row.get("name") or ""): dict(row) for row in current}
    if (
        len(frozen_by_name) != len(frozen)
        or len(current_by_name) != len(current)
        or not set(current_by_name).issubset(allowed_names)
        or not set(frozen_by_name).issubset(current_by_name)
    ):
        return False
    return all(current_by_name[name] == row for name, row in frozen_by_name.items())


def _artifact_names(task: object) -> set[str]:
    return {str(item["name"]) for item in _artifact_inventory(task)}


def _model_inventory(task: object, *, context: str) -> dict[str, list[str]]:
    getter = getattr(task, "get_models", None)
    value = getter() if callable(getter) else None
    if not isinstance(value, Mapping):
        raise RuntimeRecoveryError(f"{context} model inventory is unavailable")
    result: dict[str, list[str]] = {}
    for role in ("input", "output"):
        rows = value.get(role, [])
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            raise RuntimeRecoveryError(f"{context} model inventory is invalid")
        ids = [str(getattr(item, "id", "") or "") for item in rows]
        if any(
            len(item) != 32
            or any(character not in "0123456789abcdef" for character in item)
            for item in ids
        ) or len(set(ids)) != len(ids):
            raise RuntimeRecoveryError(f"{context} model inventory is invalid")
        result[role] = ids
    return result


def _queue_identity(task: object, *, context: str) -> tuple[str, str]:
    getter = getattr(task, "get_executed_queue", None)
    if not callable(getter):
        raise RuntimeRecoveryError(f"{context} cannot expose its execution queue")
    return str(getter(return_name=True) or ""), str(
        getter(return_name=False) or ""
    )


def _console_failure(task: object, *, label: str) -> dict[str, object]:
    if _status(task) != "failed":
        raise RuntimeRecoveryError(f"{label} is not failed")
    getter = getattr(task, "get_reported_console_output", None)
    if not callable(getter):
        raise RuntimeRecoveryError(f"{label} console evidence is unavailable")
    try:
        lines = [str(item) for item in (getter(200) or [])][-200:]
    except Exception as error:
        raise RuntimeRecoveryError(f"{label} console evidence cannot be read") from error
    raw = "\n".join(lines).encode("utf-8")[-262_144:]
    text = raw.decode("utf-8", errors="replace")
    expected = EXPECTED_FAILURE_MESSAGES[label]
    runtime_messages = {
        line.split("RuntimeError:", 1)[1].strip()
        for line in text.splitlines()
        if "RuntimeError:" in line
    }
    if runtime_messages != {expected}:
        raise RuntimeRecoveryError(f"{label} failure reason is not exactly {expected!r}")
    data = getattr(task, "data", None)
    if str(getattr(data, "status_reason", "") or "") != "worker execution exit code 1":
        raise RuntimeRecoveryError(f"{label} status reason drifted")
    return {
        "content_stored": False,
        "tail_line_count": len(lines),
        "tail_bytes": len(raw),
        "tail_sha256": hashlib.sha256(raw).hexdigest(),
        "runtime_error": expected,
        "status_reason": "worker execution exit code 1",
    }


def _validate_w_plan(task: object) -> dict[str, object]:
    if _artifact_names(task) != {formal.FORMAL_EVALUATION_PLAN_ARTIFACT}:
        raise RuntimeRecoveryError("W must contain only its non-final evaluation plan")
    try:
        artifact, record, _ = formal._unique_artifact_metadata(
            task, formal.FORMAL_EVALUATION_PLAN_ARTIFACT
        )
        hashes = {
            str(value)
            for value in (
                formal._metadata_value(artifact, "hash", "sha256"),
                formal._metadata_value(record, "hash", "sha256"),
            )
            if value not in {None, ""}
        }
        sizes = {
            int(value)
            for value in (
                formal._metadata_value(artifact, "content_size", "size", "size_bytes"),
                formal._metadata_value(record, "content_size", "size", "size_bytes"),
            )
            if value not in {None, ""}
        }
        plan = dict(
            formal._artifact_preview_mapping(
                artifact, record, formal.FORMAL_EVALUATION_PLAN_ARTIFACT
            )
        )
    except Exception as error:
        raise RuntimeRecoveryError("W evaluation plan evidence is invalid") from error
    raw = json.dumps(plan, indent=4, sort_keys=True).encode("utf-8")
    if (
        hashes != {EXPECTED_W_PLAN_ARTIFACT_SHA256}
        or sizes != {EXPECTED_W_PLAN_ARTIFACT_BYTES}
        or len(raw) != EXPECTED_W_PLAN_ARTIFACT_BYTES
        or hashlib.sha256(raw).hexdigest() != EXPECTED_W_PLAN_ARTIFACT_SHA256
        or plan.get("seal_sha256") != EXPECTED_W_PLAN_SEAL
        or _seal(plan)["seal_sha256"] != EXPECTED_W_PLAN_SEAL
    ):
        raise RuntimeRecoveryError("W evaluation plan hash, size, or seal drifted")
    return plan


def _evaluation_contracts(
    recovery_path: Path,
    *,
    plan: Mapping[str, object],
) -> list[dict[str, object]]:
    receipt, _ = _read_sealed(recovery_path, context="evaluation recovery receipt")
    rows = receipt.get("recovered_tasks")
    entries = plan.get("entries")
    if not isinstance(rows, list) or not isinstance(entries, list):
        raise RuntimeRecoveryError("evaluation task evidence is incomplete")
    compact = {
        (str(row.get("scope") or ""), str(row.get("subject") or "")): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if len(compact) != 28:
        raise RuntimeRecoveryError("evaluation recovery inventory is not exactly 28")
    contracts: list[dict[str, object]] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping):
            raise RuntimeRecoveryError("formal evaluation plan entry is invalid")
        subject = str(entry.get("subject") or "")
        row = compact.get(("formal_w3", subject))
        if not isinstance(row, Mapping):
            raise RuntimeRecoveryError(f"formal evaluation {subject} is absent")
        contracts.append(
            {
                "scope": "formal_w3",
                "subject": subject,
                "task_id": str(entry.get("evaluation_task_id") or ""),
                "task_name": formal._planned_evaluation_name(
                    controller_task_id=evaluation_recovery.TRAINING_CONTROLLER_TASK_ID,
                    index=index,
                    subject=subject,
                ),
                "parent_task_id": evaluation_recovery.TRAINING_CONTROLLER_TASK_ID,
                "queue": str(entry.get("queue") or ""),
                "queue_id": formal.EXPECTED_QUEUE_IDS.get(str(entry.get("queue") or "")),
                "script_sha256": row.get("script_sha256"),
                "parameters_sha256": row.get("parameters_sha256"),
                "input_model_ids": row.get("input_model_ids"),
            }
        )
    specs = {spec.label: spec for spec in candidate_queue.CANDIDATES}
    for label in ("E1", "E3"):
        spec = specs[label]
        row = compact.get(("candidate", spec.subject))
        if not isinstance(row, Mapping):
            raise RuntimeRecoveryError(f"candidate evaluation {label} is absent")
        contracts.append(
            {
                "scope": "candidate",
                "subject": spec.subject,
                "task_id": evaluation_recovery.CANDIDATE_EVALUATION_IDS[label],
                "task_name": candidate_queue._evaluation_name(spec),
                "parent_task_id": spec.training_task_id,
                "queue": spec.queue,
                "queue_id": candidate_queue.QUEUE_IDS[spec.queue],
                "script_sha256": row.get("script_sha256"),
                "parameters_sha256": row.get("parameters_sha256"),
                "input_model_ids": row.get("input_model_ids"),
            }
        )
    identities = tuple(
        (item["scope"], item["subject"], item["task_id"]) for item in contracts
    )
    if identities != evaluation_recovery.EXPECTED_RECOVERY_TASKS:
        raise RuntimeRecoveryError("evaluation exact task identity/order drifted")
    return contracts


def _validate_evaluations(
    task_class: object,
    contracts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for contract in contracts:
        task_id = str(contract["task_id"])
        subject = str(contract["subject"])
        context = f"{subject} exact evaluation"
        task = _authoritative_task(task_class, task_id, context=context)
        if (
            str(getattr(task, "name", "") or "") != contract["task_name"]
            or successor_deploy._task_parent(task) != contract["parent_task_id"]
            or str(getattr(task, "project", "") or "")
            != evaluation_recovery.PROJECT_ID
            or str(getattr(task, "task_type", "") or "").rsplit(".", 1)[-1].lower()
            != "training"
        ):
            raise RuntimeRecoveryError(f"{context} identity drifted")
        source = _source(
            task, entry_point="clearml_5090_bootstrap.py", context=context
        )
        parameters = _parameters(task, context=context)
        models = _model_inventory(task, context=context)
        if (
            hashlib.sha256(source.encode("utf-8")).hexdigest()
            != contract["script_sha256"]
            or _evaluation_parameters_sha256(parameters)
            != contract["parameters_sha256"]
            or models["input"] != contract["input_model_ids"]
            or models["output"]
        ):
            raise RuntimeRecoveryError(f"{context} source, parameters, or models drifted")
        status = _status(task)
        queue_name, queue_id = _queue_identity(task, context=context)
        artifacts = _artifact_inventory(task)
        if task_id == FFNET_TASK_ID:
            if (
                status != "failed"
                or queue_id != contract["queue_id"]
                or queue_name != contract["queue"]
                or artifacts
            ):
                raise RuntimeRecoveryError("FFNet failed attempt identity drifted")
        elif task_id == COFORMER_TASK_ID:
            if status not in {"queued", "in_progress", "completed"}:
                raise RuntimeRecoveryError("CoFormer is not running or completed")
            if queue_id != contract["queue_id"] or queue_name != contract["queue"]:
                raise RuntimeRecoveryError("CoFormer queue drifted")
            if not _artifact_names(task).issubset(FORMAL_EVALUATION_ARTIFACTS):
                raise RuntimeRecoveryError("CoFormer output artifact inventory drifted")
        elif status != "created" or queue_name or queue_id or artifacts:
            raise RuntimeRecoveryError(f"{context} is not unchanged created+unqueued")
        records.append(
            {
                "scope": contract["scope"],
                "subject": subject,
                "task_id": task_id,
                "project_id": evaluation_recovery.PROJECT_ID,
                "parent_task_id": contract["parent_task_id"],
                "task_name": contract["task_name"],
                "task_type": "training",
                "status": status,
                "queue": queue_name,
                "queue_id": queue_id,
                "script_sha256": contract["script_sha256"],
                "parameters_sha256": contract["parameters_sha256"],
                "input_model_ids": list(models["input"]),
                "output_model_ids": [],
                "tags": sorted(_tags(task, field="tags", context=context)),
                "system_tags": sorted(
                    _tags(task, field="system_tags", context=context)
                ),
                "artifact_names": sorted(_artifact_names(task)),
                "artifact_inventory": _artifact_inventory(task),
            }
        )
    return {
        "task_count": 28,
        "exact_task_ids_sha256": _content_sha256(
            [str(item["task_id"]) for item in contracts]
        ),
        "records": records,
        "coformer_preservation": next(
            item for item in records if item["task_id"] == COFORMER_TASK_ID
        ),
        "all_non_target_non_coformer_created_unqueued": True,
    }


def _service_requirements_valid(task: object, *, context: str) -> dict[str, object]:
    raw = successor_deploy._raw_requirements(task)
    canonical = "\n".join(successor_deploy.SERVICE_REQUIREMENTS)
    if raw == {"pip": canonical}:
        return {
            "form": "canonical",
            "sha256": _content_sha256(raw),
            "canonical_requirement": canonical,
        }
    status = _status(task)
    if status not in {"in_progress", "completed", "stopped", "failed"}:
        raise RuntimeRecoveryError(
            f"{context} cannot have agent-materialized requirements while {status}"
        )
    if not isinstance(raw, Mapping) or set(raw) != {"org_pip", "pip"}:
        raise RuntimeRecoveryError(f"{context} materialized requirements drifted")
    if raw.get("org_pip") != canonical:
        raise RuntimeRecoveryError(f"{context} original requirements drifted")
    resolved = raw.get("pip")
    if not isinstance(resolved, Sequence) or isinstance(
        resolved, (str, bytes, bytearray)
    ) or any(type(item) is not str for item in resolved):
        raise RuntimeRecoveryError(f"{context} materialized requirements drifted")
    packages = list(resolved)
    if (
        not packages
        or any(not item or item != item.strip() for item in packages)
        or len(set(packages)) != len(packages)
        or packages.count(canonical) != 1
    ):
        raise RuntimeRecoveryError(f"{context} materialized requirements drifted")
    return {
        "form": "agent_materialized",
        "sha256": _content_sha256(raw),
        "canonical_requirement": canonical,
        "resolved_count": len(packages),
        "resolved_packages": packages,
    }


def _service_environment(task: object, *, context: str) -> dict[str, object]:
    requirements = _service_requirements_valid(task, context=context)
    expected_docker = " ".join(
        (
            f"{successor_deploy.SERVICE_DOCKER_IMAGE} "
            f"{successor_deploy.SERVICE_DOCKER_ARGS}"
        ).split()
    )
    try:
        docker = successor_deploy._task_docker(task)
    except successor_deploy.DeploymentError as error:
        raise RuntimeRecoveryError(f"{context} Docker is unavailable") from error
    output_uri = successor_deploy._output_destination(task)
    binary = str(successor_deploy._task_script(task).get("binary") or "")
    if (
        docker != expected_docker
        or output_uri != successor_deploy.FILES_SERVER_URI
        or binary != "python"
    ):
        raise RuntimeRecoveryError(f"{context} service environment drifted")
    return {
        "policy": dict(SERVICE_ENVIRONMENT_POLICY),
        "requirements": requirements,
        "docker": docker,
        "output_uri": output_uri,
        "binary": binary,
    }


def _execution_environment(task: object, *, context: str) -> dict[str, object]:
    try:
        script = successor_deploy._task_script(task)
        requirements = successor_deploy._raw_requirements(task)
    except successor_deploy.DeploymentError as error:
        raise RuntimeRecoveryError(f"{context} execution environment is unavailable") from error
    getter = getattr(task, "get_base_docker", None)
    if not callable(getter):
        raise RuntimeRecoveryError(f"{context} cannot expose its Docker command")
    docker = getter()
    if type(docker) is not str:
        raise RuntimeRecoveryError(f"{context} Docker command is invalid")
    return {
        "binary": str(script.get("binary") or ""),
        "requirements": requirements,
        "docker": " ".join(docker.split()),
        "output_uri": successor_deploy._output_destination(task),
    }


def _target_snapshot(
    task_class: object,
    *,
    label: str,
    entry_point: str,
    parent_task_id: str,
    task_name: str,
    parameters: Mapping[str, object] | None,
    allowed_artifacts: frozenset[str],
) -> tuple[dict[str, object], object]:
    task_id = (
        SUCCESSOR_TASK_IDS[label]
        if label in SUCCESSOR_TASK_IDS
        else TARGET_TASK_IDS[label]
    )
    context = f"{label} runtime recovery target"
    task = _authoritative_task(task_class, task_id, context=context)
    if _status(task) != "failed":
        raise RuntimeRecoveryError(f"{context} is not failed")
    if (
        str(getattr(task, "name", "") or "") != task_name
        or successor_deploy._task_parent(task) != parent_task_id
        or _exact_project_id(task, context=context) != evaluation_recovery.PROJECT_ID
        or successor_deploy._project_name(task) != successor_deploy.PROJECT_NAME
        or str(getattr(task, "task_type", "") or "").rsplit(".", 1)[-1].lower()
        not in ({"training"} if label == "FFNet" else {"controller"})
    ):
        raise RuntimeRecoveryError(f"{context} static identity drifted")
    source = _source(task, entry_point=entry_point, context=context)
    observed_parameters = _parameters(task, context=context)
    normalized_parameters = (
        _successor_deployment_parameters(label, observed_parameters)
        if label in SUCCESSOR_RUNTIME_DEFAULT_PARAMETERS
        else {str(key): str(value) for key, value in observed_parameters.items()}
    )
    if parameters is not None and normalized_parameters != {
        str(key): str(value) for key, value in parameters.items()
    }:
        raise RuntimeRecoveryError(f"{context} exact parameters drifted")
    if (
        hashlib.sha256(source.encode("utf-8")).hexdigest()
        != EXPECTED_SOURCE_SHA256[label]
        or (
            _evaluation_parameters_sha256(observed_parameters)
            if label == "FFNet"
            else _content_sha256(normalized_parameters)
        )
        != EXPECTED_PARAMETERS_SHA256[label]
    ):
        raise RuntimeRecoveryError(f"{context} source or parameter hash drifted")
    queue_name, queue_id = _queue_identity(task, context=context)
    expected_queue = "GPU4-V100" if label == "FFNet" else successor_deploy.SERVICES_QUEUE
    expected_queue_id = (
        formal.EXPECTED_QUEUE_IDS[expected_queue]
        if label == "FFNet"
        else successor_deploy.SERVICES_QUEUE_ID
    )
    if queue_name != expected_queue or queue_id != expected_queue_id:
        raise RuntimeRecoveryError(f"{context} queue drifted")
    artifacts = _artifact_names(task)
    if artifacts != set(allowed_artifacts):
        raise RuntimeRecoveryError(f"{context} artifact inventory drifted")
    models = _model_inventory(task, context=context)
    if models != {"input": [], "output": []}:
        raise RuntimeRecoveryError(f"{context} unexpectedly owns models")
    execution_environment = _execution_environment(task, context=context)
    service_environment = (
        None if label == "FFNet" else _service_environment(task, context=context)
    )
    result = {
        "label": label,
        "task_id": task_id,
        "project_id": evaluation_recovery.PROJECT_ID,
        "status": "failed",
        "task_name": task_name,
        "task_type": "training" if label == "FFNet" else "controller",
        "parent_task_id": parent_task_id,
        "entry_point": entry_point,
        "source": source,
        "source_sha256": EXPECTED_SOURCE_SHA256[label],
        "source_bytes": len(source.encode("utf-8")),
        "parameters": (
            {
                key: value
                for key, value in observed_parameters.items()
                if key not in BOOTSTRAP_EMPTY_DEFAULT_PARAMETERS
            }
            if label == "FFNet"
            else dict(normalized_parameters)
        ),
        "parameters_sha256": EXPECTED_PARAMETERS_SHA256[label],
        "tags": sorted(_tags(task, field="tags", context=context)),
        "system_tags": sorted(_tags(task, field="system_tags", context=context)),
        "queue": expected_queue,
        "queue_id": expected_queue_id,
        "artifact_names": sorted(artifacts),
        "artifact_inventory": _artifact_inventory(task),
        "model_inventory": models,
        "pre_reset_execution_environment": execution_environment,
        "service_environment": service_environment,
        "failure_evidence": _console_failure(task, label=label),
    }
    return result, task


def _exact_name_inventory(
    task_class: object,
    *,
    task_name: str,
    parent_task_id: str,
    expected_task_id: str,
) -> list[str]:
    getter = getattr(task_class, "get_tasks", None)
    if not callable(getter):
        raise RuntimeRecoveryError("ClearML cannot perform exact-name lookup")
    rows = getter(
        task_name=f"^{re.escape(task_name)}$",
        task_filter={"parent": parent_task_id},
        allow_archived=True,
    )
    if rows is None or isinstance(rows, (str, bytes, bytearray)):
        raise RuntimeRecoveryError("exact-name lookup returned invalid data")
    ids: list[str] = []
    for row in rows:
        task_id = str(getattr(row, "id", "") or "")
        task = _authoritative_task(
            task_class, task_id, context="exact-name inventory task"
        )
        if (
            str(getattr(task, "name", "") or "") != task_name
            or successor_deploy._task_parent(task) != parent_task_id
        ):
            raise RuntimeRecoveryError("exact-name inventory identity drifted")
        ids.append(task_id)
    if len(ids) != len(set(ids)) or expected_task_id not in ids:
        raise RuntimeRecoveryError("exact-name inventory is duplicated or misses target")
    return sorted(ids)


def _same_name_history_snapshot(
    task_class: object,
    *,
    label: str,
    inventory: Sequence[str],
) -> list[dict[str, object]]:
    target_id = TARGET_TASK_IDS[label]
    rows: list[dict[str, object]] = []
    for task_id in inventory:
        if task_id == target_id:
            continue
        if len(task_id) != 32 or any(
            character not in "0123456789abcdef" for character in task_id
        ):
            raise RuntimeRecoveryError(f"{label} history task ID is invalid")
        task = _authoritative_task(
            task_class,
            task_id,
            context=f"{label} preexisting same-name task",
        )
        context = f"{label} preexisting same-name task {task_id}"
        source = _source(
            task,
            entry_point=(
                "clearml_5090_bootstrap.py"
                if label == "FFNet"
                else candidate_deploy.ENTRY_POINT
            ),
            context=context,
        )
        parameters = _parameters(task, context=context)
        queue_name, queue_id = _queue_identity(task, context=context)
        rows.append(
            {
                "task_id": task_id,
                "status": _status(task),
                "project": str(getattr(task, "project", "") or ""),
                "task_name": str(getattr(task, "name", "") or ""),
                "parent_task_id": successor_deploy._task_parent(task),
                "task_type": str(getattr(task, "task_type", "") or "")
                .rsplit(".", 1)[-1]
                .lower(),
                "source_sha256": hashlib.sha256(
                    source.encode("utf-8")
                ).hexdigest(),
                "parameters_sha256": _content_sha256(parameters),
                "queue": queue_name,
                "queue_id": queue_id,
                "artifact_inventory": _artifact_inventory(task),
                "model_inventory": _model_inventory(task, context=context),
                "tags": sorted(_tags(task, field="tags", context=context)),
                "system_tags": sorted(
                    _tags(task, field="system_tags", context=context)
                ),
                "execution_environment": _execution_environment(
                    task, context=context
                ),
            }
        )
    return rows


def _revalidate_old_successor_failures(
    task_class: object,
    *,
    evidence: Mapping[str, object],
) -> dict[str, object]:
    if set(evidence) != {"W", "L", "A", "S"}:
        raise RuntimeRecoveryError("old successor failure inventory drifted")
    result: dict[str, object] = {}
    for role in ("W", "L", "A", "S"):
        frozen = evidence[role]
        if not isinstance(frozen, Mapping):
            raise RuntimeRecoveryError(f"old successor {role} evidence is invalid")
        task = _authoritative_task(
            task_class,
            SUCCESSOR_TASK_IDS[role],
            context=f"old successor {role} preservation",
        )
        source = _source(
            task,
            entry_point=str(successor_deploy.ROLE_DEFINITIONS[role]["entry_point"]),
            context=f"old successor {role} preservation",
        )
        parameters = _parameters(
            task, context=f"old successor {role} preservation"
        )
        normalized_parameters = _successor_deployment_parameters(
            role, parameters
        )
        queue_name, queue_id = _queue_identity(
            task, context=f"old successor {role} preservation"
        )
        failure = _console_failure(task, label=role)
        if (
            _status(task) != "failed"
            or str(getattr(task, "name", "") or "") != frozen.get("task_name")
            or _exact_project_id(
                task, context=f"old successor {role} preservation"
            )
            != evaluation_recovery.PROJECT_ID
            or successor_deploy._project_name(task) != successor_deploy.PROJECT_NAME
            or str(getattr(task, "task_type", "") or "").rsplit(".", 1)[-1].lower()
            != "controller"
            or hashlib.sha256(source.encode("utf-8")).hexdigest()
            != frozen.get("source_sha256")
            or _content_sha256(normalized_parameters)
            != frozen.get("parameters_sha256")
            or successor_deploy._task_parent(task)
            != frozen.get("parent_task_id")
            or queue_name != frozen.get("queue")
            or queue_id != frozen.get("queue_id")
            or sorted(_artifact_names(task)) != frozen.get("artifact_names")
            or _artifact_inventory(task) != frozen.get("artifact_inventory")
            or sorted(_tags(task, field="tags", context=f"old successor {role}"))
            != sorted(frozen.get("tags", []))
            or sorted(
                _tags(task, field="system_tags", context=f"old successor {role}")
            )
            != sorted(frozen.get("system_tags", []))
            or _model_inventory(task, context=f"old successor {role}")
            != frozen.get("model_inventory")
            or _execution_environment(task, context=f"old successor {role}")
            != frozen.get("pre_reset_execution_environment")
            or _service_environment(task, context=f"old successor {role}")
            != frozen.get("service_environment")
            or failure != frozen.get("failure_evidence")
        ):
            raise RuntimeRecoveryError(f"old successor {role} changed")
        if role == "W":
            _validate_w_plan(task)
        result[role] = {
            "task_id": SUCCESSOR_TASK_IDS[role],
            "project_id": evaluation_recovery.PROJECT_ID,
            "task_name": frozen["task_name"],
            "task_type": frozen["task_type"],
            "parent_task_id": frozen["parent_task_id"],
            "status": "failed",
            "queue": queue_name,
            "queue_id": queue_id,
            "source_sha256": frozen["source_sha256"],
            "parameters_sha256": frozen["parameters_sha256"],
            "artifact_names": sorted(_artifact_names(task)),
            "artifact_inventory": _artifact_inventory(task),
            "tags": sorted(_tags(task, field="tags", context=f"old successor {role}")),
            "system_tags": sorted(
                _tags(task, field="system_tags", context=f"old successor {role}")
            ),
            "model_inventory": _model_inventory(
                task, context=f"old successor {role}"
            ),
            "pre_reset_execution_environment": _execution_environment(
                task, context=f"old successor {role}"
            ),
            "service_environment": _service_environment(
                task, context=f"old successor {role}"
            ),
            "failure_evidence": failure,
        }
    return result


def _load_bindings(
    *,
    evaluation_recovery_receipt: Path,
    successor_deployment_receipt: Path,
    controller_recovery_receipt: Path,
    candidate_deployment_receipt: Path,
    amendment_receipt: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    evaluation = _evaluation_binding(evaluation_recovery_receipt)
    successor_binding, successor_receipt = _successor_binding(
        successor_deployment_receipt, evaluation_binding=evaluation
    )
    controller = _controller_recovery_binding(controller_recovery_receipt)
    candidate_binding, candidate_receipt = _candidate_deployment_binding(
        candidate_deployment_receipt,
        evaluation_binding=evaluation,
        controller_binding=controller,
    )
    amendment_binding, amendment_receipt_value = _amendment_binding(
        amendment_receipt
    )
    bindings = {
        "evaluation_recovery": evaluation,
        "successor_deployment": successor_binding,
        "candidate_controller_recovery": controller,
        "candidate_deployment": candidate_binding,
        "formal_plan_amendment": amendment_binding,
    }
    return (
        bindings,
        successor_receipt,
        candidate_receipt,
        amendment_receipt_value,
    )


def build_attempt_receipt(
    task_class: object,
    *,
    evaluation_recovery_receipt: Path,
    successor_deployment_receipt: Path,
    controller_recovery_receipt: Path,
    candidate_deployment_receipt: Path,
    amendment_receipt: Path,
) -> dict[str, object]:
    (
        bindings,
        successor_receipt,
        _candidate_receipt,
        _amendment_receipt_value,
    ) = _load_bindings(
        evaluation_recovery_receipt=evaluation_recovery_receipt,
        successor_deployment_receipt=successor_deployment_receipt,
        controller_recovery_receipt=controller_recovery_receipt,
        candidate_deployment_receipt=candidate_deployment_receipt,
        amendment_receipt=amendment_receipt,
    )
    w = _authoritative_task(
        task_class, SUCCESSOR_TASK_IDS["W"], context="W plan producer"
    )
    plan = _validate_w_plan(w)
    amendment_authority = _validate_remote_amendment_producer(
        task_class,
        binding=bindings["formal_plan_amendment"],
        base_plan=plan,
    )
    contracts = _evaluation_contracts(
        evaluation_recovery_receipt, plan=plan
    )
    evaluations = _validate_evaluations(task_class, contracts)
    task_material: dict[str, object] = {}
    old_successor_failure_evidence: dict[str, object] = {}
    target_objects: dict[str, object] = {}

    ffnet_contract = next(
        item for item in contracts if item["task_id"] == FFNET_TASK_ID
    )
    ffnet, target_objects["FFNet"] = _target_snapshot(
        task_class,
        label="FFNet",
        entry_point="clearml_5090_bootstrap.py",
        parent_task_id=str(ffnet_contract["parent_task_id"]),
        task_name=str(ffnet_contract["task_name"]),
        parameters=None,
        allowed_artifacts=frozenset(),
    )
    ffnet["pre_reset_queue"] = ffnet["queue"]
    ffnet["pre_reset_queue_id"] = ffnet["queue_id"]
    ffnet["pre_reset_tags"] = list(ffnet["tags"])
    ffnet["queue"] = AMENDMENT_TARGET_QUEUE
    ffnet["queue_id"] = formal.EXPECTED_QUEUE_IDS[AMENDMENT_TARGET_QUEUE]
    ffnet["tags"] = [
        value for value in ffnet["tags"] if value != "dependency-released"
    ]
    task_material["FFNet"] = ffnet

    parents = successor_receipt["parents"]
    parameters = successor_receipt["parameters"]
    for role in ("W", "L", "A", "S"):
        material, target_objects[role] = _target_snapshot(
            task_class,
            label=role,
            entry_point=str(successor_deploy.ROLE_DEFINITIONS[role]["entry_point"]),
            parent_task_id=str(parents[role]),
            task_name=str(successor_deploy.ROLE_DEFINITIONS[role]["task_name"]),
            parameters=parameters[role],
            allowed_artifacts=(
                frozenset({formal.FORMAL_EVALUATION_PLAN_ARTIFACT})
                if role == "W"
                else frozenset()
            ),
        )
        old_successor_failure_evidence[role] = {
            key: value
            for key, value in material.items()
            if key not in {"source", "parameters"}
        }

    candidate_pre_reset, target_objects["C"] = _target_snapshot(
        task_class,
        label="C",
        entry_point=candidate_deploy.ENTRY_POINT,
        parent_task_id=candidate_deploy.PARENT_TASK_ID,
        task_name=candidate_deploy.CONTROLLER_NAME,
        parameters=None,
        allowed_artifacts=frozenset(),
    )
    if set(candidate_pre_reset["tags"]) != candidate_deploy.TAGS:
        raise RuntimeRecoveryError("candidate controller tag contract drifted")
    task_material["C"] = _candidate_target_material(
        pre_reset=candidate_pre_reset,
        amendment_binding=bindings["formal_plan_amendment"],
    )

    duplicate_inventories = {
        label: _exact_name_inventory(
            task_class,
            task_name=str(task_material[label]["task_name"]),
            parent_task_id=str(task_material[label]["parent_task_id"]),
            expected_task_id=TARGET_TASK_IDS[label],
        )
        for label in RESET_ORDER
    }
    same_name_history = {
        label: _same_name_history_snapshot(
            task_class,
            label=label,
            inventory=duplicate_inventories[label],
        )
        for label in RESET_ORDER
    }
    return _seal(
        {
            "schema_version": 1,
            "receipt_type": ATTEMPT_RECEIPT_TYPE,
            "generated_at_utc": _utc_timestamp(),
            "status": "frozen_before_reset",
            "remote_state_changed": False,
            "same_ids_only": True,
            "replacement_tasks_created": False,
            "bindings": bindings,
            "evaluation_plan": {
                "producer_task_id": SUCCESSOR_TASK_IDS["W"],
                "artifact_name": formal.FORMAL_EVALUATION_PLAN_ARTIFACT,
                "artifact_sha256": EXPECTED_W_PLAN_ARTIFACT_SHA256,
                "artifact_bytes": EXPECTED_W_PLAN_ARTIFACT_BYTES,
                "seal_sha256": EXPECTED_W_PLAN_SEAL,
                "entry_count": len(plan["entries"]),
            },
            "evaluation_inventory": evaluations,
            "amendment_producer_authority": amendment_authority,
            "target_material": task_material,
            "old_successor_failure_evidence": old_successor_failure_evidence,
            "exact_name_inventories": duplicate_inventories,
            "preexisting_same_name_tasks": same_name_history,
            "reset_order": list(RESET_ORDER),
            "enqueue_order": list(ENQUEUE_ORDER),
            "coformer_mutation_allowed": False,
            "planned_actions": [
                "durably_write_attempt_receipt",
                "authoritative_pre_reset_recheck",
                "reset_and_reinstall_two_exact_ids",
                "authoritative_created_unqueued_readback",
                "leave_ffnet_created_unqueued_for_amended_w",
                "enqueue_candidate_controller_services",
                "authoritative_queue_readback",
                "verify_no_replacement_and_coformer_preserved",
            ],
        }
    )


def _wait_created(
    task_class: object,
    task_id: str,
    *,
    sleeper: Callable[[float], None],
) -> object:
    last = ""
    for attempt in range(5):
        task = _authoritative_task(
            task_class, task_id, context="reset created readback"
        )
        last = _status(task)
        if last == "created":
            return task
        if attempt != 4:
            sleeper(1.0)
    raise RuntimeRecoveryError(
        f"task {task_id} did not reach created after reset; observed {last!r}"
    )


def _set_tags(task: object, tags: Sequence[str], *, context: str) -> None:
    setter = getattr(task, "set_tags", None)
    if not callable(setter):
        raise RuntimeRecoveryError(f"{context} cannot restore tags")
    callback_error: Exception | None = None
    try:
        if setter(list(tags)) is False:
            callback_error = RuntimeRecoveryError(f"{context} tag callback rejected")
    except Exception as error:
        callback_error = error
    expected = [str(item) for item in tags]
    observed = [
        str(item) for item in (getattr(getattr(task, "data", None), "tags", None) or [])
    ]
    if (
        len(expected) != len(set(expected))
        or len(observed) != len(set(observed))
        or set(observed) != set(expected)
    ):
        if callback_error is not None:
            raise RuntimeRecoveryError(f"{context} tag authority readback failed") from callback_error
        raise RuntimeRecoveryError(f"{context} tag authority readback failed")


def _configure_target(task: object, material: Mapping[str, object]) -> None:
    label = str(material["label"])
    task.set_script(
        repository="",
        branch="",
        commit="",
        diff=str(material["source"]),
        working_dir=".",
        entry_point=str(material["entry_point"]),
    )
    if label != "FFNet":
        task.set_packages(list(successor_deploy.SERVICE_REQUIREMENTS))
        task.set_base_docker(
            docker_image=successor_deploy.SERVICE_DOCKER_IMAGE,
            docker_arguments=successor_deploy.SERVICE_DOCKER_ARGS,
        )
        task.output_uri = successor_deploy.FILES_SERVER_URI
    setter = getattr(task, "set_parent", None)
    if not callable(setter):
        raise RuntimeRecoveryError(f"{label} cannot restore its parent")
    setter(str(material["parent_task_id"]))
    task.set_parameters(dict(material["parameters"]))
    _set_tags(task, list(material["tags"]), context=f"{label} target")
    flushed = task.flush(wait_for_uploads=True)
    if flushed is False:
        raise RuntimeRecoveryError(f"{label} target flush was not confirmed")


def _validate_created_target(
    task_class: object,
    *,
    label: str,
    material: Mapping[str, object],
) -> dict[str, object]:
    task = _authoritative_task(
        task_class, TARGET_TASK_IDS[label], context=f"{label} created target"
    )
    if (
        _status(task) != "created"
        or successor_deploy._task_parent(task) != material["parent_task_id"]
        or str(getattr(task, "name", "") or "") != material["task_name"]
        or _exact_project_id(task, context=f"{label} created target")
        != evaluation_recovery.PROJECT_ID
        or successor_deploy._project_name(task) != successor_deploy.PROJECT_NAME
        or str(getattr(task, "task_type", "") or "").rsplit(".", 1)[-1].lower()
        != str(material["task_type"])
    ):
        raise RuntimeRecoveryError(f"{label} created identity drifted")
    source = _source(
        task,
        entry_point=str(material["entry_point"]),
        context=f"{label} created target",
    )
    parameters = _parameters(task, context=f"{label} created target")
    queue_name, queue_id = _queue_identity(task, context=f"{label} created target")
    if (
        hashlib.sha256(source.encode("utf-8")).hexdigest()
        != material["source_sha256"]
        or (
            _evaluation_parameters_sha256(parameters)
            if label == "FFNet"
            else _content_sha256(parameters)
        )
        != material["parameters_sha256"]
        or queue_name
        or queue_id
        or _artifact_names(task)
        or _model_inventory(task, context=f"{label} created target")
        != {"input": [], "output": []}
        or sorted(
            _tags(task, field="system_tags", context=f"{label} created target")
        )
        != sorted(material["system_tags"])
        or sorted(_tags(task, field="tags", context=f"{label} created target"))
        != sorted(material["tags"])
    ):
        raise RuntimeRecoveryError(f"{label} created contract drifted")
    service_environment = (
        None
        if label == "FFNet"
        else _service_environment(task, context=f"{label} created target")
    )
    execution_environment = _execution_environment(
        task, context=f"{label} created target"
    )
    if label == "FFNet" and execution_environment != material[
        "pre_reset_execution_environment"
    ]:
        raise RuntimeRecoveryError("FFNet created execution environment drifted")
    return {
        "label": label,
        "task_id": TARGET_TASK_IDS[label],
        "project_id": evaluation_recovery.PROJECT_ID,
        "task_name": material["task_name"],
        "task_type": material["task_type"],
        "status": "created",
        "queue": None,
        "queue_id": None,
        "source_sha256": material["source_sha256"],
        "parameters_sha256": material["parameters_sha256"],
        "parent_task_id": material["parent_task_id"],
        "artifact_names": [],
        "model_inventory": {"input": [], "output": []},
        "tags": sorted(material["tags"]),
        "system_tags": sorted(material["system_tags"]),
        "service_environment": service_environment,
        "execution_environment": execution_environment,
    }


def _validate_failed_target_against_material(
    task_class: object,
    *,
    label: str,
    material: Mapping[str, object],
    context: str,
) -> object:
    """Immediately recheck the frozen failed target before a destructive reset."""
    task = _authoritative_task(task_class, TARGET_TASK_IDS[label], context=context)
    task_type = str(getattr(task, "task_type", "") or "").rsplit(".", 1)[-1].lower()
    if (
        _status(task) != "failed"
        or str(getattr(task, "name", "") or "") != material["task_name"]
        or successor_deploy._task_parent(task) != material["parent_task_id"]
        or _exact_project_id(task, context=context) != evaluation_recovery.PROJECT_ID
        or successor_deploy._project_name(task) != successor_deploy.PROJECT_NAME
        or task_type != material["task_type"]
    ):
        raise RuntimeRecoveryError(f"{context} identity/status drifted")
    source = _source(
        task,
        entry_point=str(material["entry_point"]),
        context=context,
    )
    parameters = _parameters(task, context=context)
    queue_name, queue_id = _queue_identity(task, context=context)
    pre_source_sha = material.get(
        "pre_reset_source_sha256", material["source_sha256"]
    )
    pre_parameters_sha = material.get(
        "pre_reset_parameters_sha256", material["parameters_sha256"]
    )
    pre_tags = material.get("pre_reset_tags", material["tags"])
    pre_queue = material.get("pre_reset_queue", material["queue"])
    pre_queue_id = material.get("pre_reset_queue_id", material["queue_id"])
    if (
        hashlib.sha256(source.encode("utf-8")).hexdigest() != pre_source_sha
        or (
            _evaluation_parameters_sha256(parameters)
            if label == "FFNet"
            else _content_sha256(parameters)
        )
        != pre_parameters_sha
        or sorted(_tags(task, field="tags", context=context)) != sorted(pre_tags)
        or sorted(_tags(task, field="system_tags", context=context))
        != sorted(material["system_tags"])
        or queue_name != pre_queue
        or queue_id != pre_queue_id
        or _artifact_inventory(task) != material["artifact_inventory"]
        or _model_inventory(task, context=context) != {"input": [], "output": []}
        or _execution_environment(task, context=context)
        != material["pre_reset_execution_environment"]
        or _console_failure(task, label=label) != material["failure_evidence"]
    ):
        raise RuntimeRecoveryError(f"{context} changed after attempt freeze")
    if label != "FFNet" and _service_environment(task, context=context) != material[
        "service_environment"
    ]:
        raise RuntimeRecoveryError(f"{context} service environment drifted")
    return task


def _created_unqueued_write_guard(
    task_class: object,
    *,
    label: str,
    material: Mapping[str, object],
    context: str,
    exact_contract: bool,
) -> object:
    """Reject a write if the exact target acquired a queue, output, or new identity."""
    task = _authoritative_task(task_class, TARGET_TASK_IDS[label], context=context)
    queue_name, queue_id = _queue_identity(task, context=context)
    task_type = str(getattr(task, "task_type", "") or "").rsplit(".", 1)[-1].lower()
    if (
        _status(task) != "created"
        or str(getattr(task, "name", "") or "") != material["task_name"]
        or successor_deploy._task_parent(task) != material["parent_task_id"]
        or _exact_project_id(task, context=context) != evaluation_recovery.PROJECT_ID
        or successor_deploy._project_name(task) != successor_deploy.PROJECT_NAME
        or task_type != material["task_type"]
        or queue_name
        or queue_id
        or _artifact_names(task)
        or _model_inventory(task, context=context) != {"input": [], "output": []}
    ):
        raise RuntimeRecoveryError(f"{context} is not the exact created+unqueued shell")
    if not exact_contract:
        return task
    source = _source(
        task,
        entry_point=str(material["entry_point"]),
        context=context,
    )
    parameters = _parameters(task, context=context)
    if (
        hashlib.sha256(source.encode("utf-8")).hexdigest()
        != material["source_sha256"]
        or (
            _evaluation_parameters_sha256(parameters)
            if label == "FFNet"
            else _content_sha256(parameters)
        )
        != material["parameters_sha256"]
        or sorted(_tags(task, field="tags", context=context))
        != sorted(material["tags"])
        or sorted(_tags(task, field="system_tags", context=context))
        != sorted(material["system_tags"])
    ):
        raise RuntimeRecoveryError(f"{context} exact contract drifted")
    execution_environment = _execution_environment(task, context=context)
    if label == "FFNet":
        if execution_environment != material["pre_reset_execution_environment"]:
            raise RuntimeRecoveryError(f"{context} execution environment drifted")
    elif _service_environment(task, context=context)["policy"] != material[
        "service_environment_policy"
    ]:
        raise RuntimeRecoveryError(f"{context} service environment drifted")
    return task


def _created_shell_snapshot(
    task_class: object,
    *,
    label: str,
    material: Mapping[str, object],
    context: str,
) -> tuple[object, dict[str, object]]:
    task = _created_unqueued_write_guard(
        task_class,
        label=label,
        material=material,
        context=context,
        exact_contract=False,
    )
    source = _source(
        task,
        entry_point=str(material["entry_point"]),
        context=context,
    )
    return task, {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "parameters": _parameters(task, context=context),
        "tags": sorted(_tags(task, field="tags", context=context)),
        "system_tags": sorted(_tags(task, field="system_tags", context=context)),
        "execution_environment": _execution_environment(task, context=context),
    }


def _reset_and_install(
    task_class: object,
    *,
    label: str,
    material: Mapping[str, object],
    sleeper: Callable[[float], None],
) -> dict[str, object]:
    task_id = TARGET_TASK_IDS[label]
    task = _validate_failed_target_against_material(
        task_class,
        label=label,
        material=material,
        context=f"{label} immediate pre-reset guard",
    )
    resetter = getattr(task, "reset", None)
    if not callable(resetter):
        raise RuntimeRecoveryError(f"{label} target cannot be reset")
    callback_error: Exception | None = None
    try:
        if resetter(force=True) is False:
            callback_error = RuntimeRecoveryError(f"{label} reset callback rejected")
    except Exception as error:
        callback_error = error
    try:
        _wait_created(task_class, task_id, sleeper=sleeper)
    except Exception as error:
        if callback_error is not None:
            raise RuntimeRecoveryError(
                f"{label} reset authority readback failed"
            ) from callback_error
        raise error
    task, first_snapshot = _created_shell_snapshot(
        task_class,
        label=label,
        material=material,
        context=f"{label} post-reset frozen shell",
    )
    task, confirm_snapshot = _created_shell_snapshot(
        task_class,
        label=label,
        material=material,
        context=f"{label} immediate configure write guard",
    )
    if first_snapshot != confirm_snapshot:
        raise RuntimeRecoveryError(f"{label} post-reset shell changed before configure")
    configure_error: Exception | None = None
    try:
        _configure_target(task, material)
    except Exception as error:
        configure_error = error
    try:
        first = _validate_created_target(task_class, label=label, material=material)
        confirm = _validate_created_target(task_class, label=label, material=material)
        if first != confirm:
            raise RuntimeRecoveryError(f"{label} created readbacks differ")
        return first
    except Exception as error:
        if configure_error is not None:
            raise RuntimeRecoveryError(
                f"{label} exact reinstall callback/readback failed"
            ) from configure_error
        raise RuntimeRecoveryError(f"{label} exact reinstall failed") from error


def _completed_candidate_manifest_contract(
    material: Mapping[str, object],
) -> dict[str, object]:
    amendment_binding = material["formal_plan_amendment"]
    return {
        "schema_version": 1,
        "document_type": "resilient_v2x_formal_1337_candidate_evaluation_manifest",
        "controller_task_id": CANDIDATE_CONTROLLER_TASK_ID,
        "protocol_id": candidate_queue.PROTOCOL_ID,
        "training_seed": candidate_queue.TRAINING_SEED,
        "training_dataset_id": candidate_queue.TRAINING_DATASET_ID,
        "checkpoint_policy": candidate_queue.CHECKPOINT_POLICY,
        "sample_count": candidate_queue.SAMPLE_COUNT,
        "ground_truth_count": candidate_queue.GROUND_TRUTH_COUNT,
        "unsupported_sample_count": candidate_queue.UNSUPPORTED_SAMPLE_COUNT,
        "sample_ids_sha256": candidate_queue.SAMPLE_IDS_SHA256,
        "delays_ms": list(candidate_queue.DELAYS_MS),
        "conditions": list(candidate_queue.CONDITIONS),
        "run_count_per_candidate": candidate_queue.RUN_COUNT,
        "agent_scope": "E+R",
        "candidate_order": list(candidate_queue.CANDIDATE_ORDER),
        "candidate_count": len(candidate_queue.CANDIDATES),
        "validated_candidate_count": len(candidate_queue.CANDIDATES),
        "all_candidates_validated": True,
        "formal_plan_amendment": {
            "producer_task_id": amendment_binding["producer_task_id"],
            "receipt_seal_sha256": amendment_binding["receipt_seal_sha256"],
            "amendment_evidence_seal_sha256": amendment_binding[
                "amendment_evidence_seal_sha256"
            ],
            "worker_stats_evidence_seal_sha256": amendment_binding[
                "worker_stats_evidence_seal_sha256"
            ],
            "evaluation_task_ids_sha256": amendment_binding[
                "evaluation_task_ids_sha256"
            ],
            "revised_plan_seal_sha256": amendment_binding[
                "revised_plan_seal_sha256"
            ],
            "ffnet_queue": AMENDMENT_TARGET_QUEUE,
        },
    }


def _validate_completed_candidate_manifest(
    task: object,
    *,
    material: Mapping[str, object],
) -> dict[str, object]:
    context = "completed candidate evaluation manifest"
    try:
        record = candidate_queue._artifact_record(
            task, candidate_queue.CONTROLLER_ARTIFACT
        )
        manifest = candidate_queue._artifact_payload(
            task, candidate_queue.CONTROLLER_ARTIFACT
        )
        manifest_seal = candidate_queue._require_seal(manifest, context=context)
    except (RuntimeError, ValueError) as error:
        raise RuntimeRecoveryError(f"{context} is invalid") from error
    contract = _completed_candidate_manifest_contract(material)
    if any(manifest.get(key) != value for key, value in contract.items() if key != "formal_plan_amendment"):
        raise RuntimeRecoveryError(f"{context} protocol identity drifted")
    entries = manifest.get("entries")
    if (
        not isinstance(entries, list)
        or len(entries) != len(candidate_queue.CANDIDATES)
        or [row.get("subject") if isinstance(row, Mapping) else None for row in entries]
        != list(candidate_queue.CANDIDATE_ORDER)
        or any(
            not isinstance(row, Mapping)
            or row.get("evidence_status") != "validated"
            or row.get("training_status") != "completed"
            or not isinstance(row.get("training_final"), Mapping)
            or not isinstance(row.get("formal_evidence"), Mapping)
            for row in entries
        )
    ):
        raise RuntimeRecoveryError(f"{context} candidate completion drifted")
    training_gate = manifest.get("training_release_barrier")
    if training_gate != {
        "policy": "all_five_completed_and_final_contract_verified",
        "ready": True,
        "required_labels": [spec.label for spec in candidate_queue.CANDIDATES],
    }:
        raise RuntimeRecoveryError(f"{context} training barrier drifted")
    formal_gate = manifest.get("formal_core_release_barrier")
    core_entries = formal_gate.get("entries") if isinstance(formal_gate, Mapping) else None
    expected_core = [
        {
            "subject": spec["subject"],
            "task_id": spec["task_id"],
            "status": "completed",
            "queue": (
                AMENDMENT_TARGET_QUEUE
                if spec["subject"] == "ffnet"
                else spec["queue"]
            ),
            "execution_queue_id": formal.EXPECTED_QUEUE_IDS[
                AMENDMENT_TARGET_QUEUE
                if spec["subject"] == "ffnet"
                else spec["queue"]
            ],
        }
        for spec in candidate_queue.FORMAL_CORE_EVALUATIONS
    ]
    if (
        not isinstance(formal_gate, Mapping)
        or formal_gate.get("policy")
        != "formal_core_six_completed_before_candidate_e1"
        or formal_gate.get("ready") is not True
        or core_entries != expected_core
        or formal_gate.get("formal_plan_amendment")
        != contract["formal_plan_amendment"]
    ):
        raise RuntimeRecoveryError(f"{context} formal-core gate drifted")
    artifact_sha = _sha256(record.get("hash"), context=f"{context} artifact hash")
    artifact_bytes = record.get("content_size")
    if type(artifact_bytes) is not int or artifact_bytes <= 0:
        raise RuntimeRecoveryError(f"{context} artifact size is invalid")
    return {
        **contract,
        "manifest_seal_sha256": manifest_seal,
        "artifact_sha256": artifact_sha,
        "artifact_bytes": artifact_bytes,
        "entries_sha256": _content_sha256(entries),
        "formal_core_entries_sha256": _content_sha256(core_entries),
    }


def _validate_queued_target(
    task_class: object,
    *,
    label: str,
    material: Mapping[str, object],
) -> dict[str, object]:
    task = _authoritative_task(
        task_class, TARGET_TASK_IDS[label], context=f"{label} queued target"
    )
    status = _status(task)
    if status not in {"queued", "in_progress", "completed"}:
        raise RuntimeRecoveryError(f"{label} did not remain queued/running/completed")
    source = _source(
        task,
        entry_point=str(material["entry_point"]),
        context=f"{label} queued target",
    )
    parameters = _parameters(task, context=f"{label} queued target")
    queue_name, queue_id = _queue_identity(task, context=f"{label} queued target")
    task_type = str(getattr(task, "task_type", "") or "").rsplit(".", 1)[-1].lower()
    if (
        str(getattr(task, "name", "") or "") != material["task_name"]
        or _exact_project_id(task, context=f"{label} queued target")
        != evaluation_recovery.PROJECT_ID
        or successor_deploy._project_name(task) != successor_deploy.PROJECT_NAME
        or task_type != material["task_type"]
    ):
        raise RuntimeRecoveryError(f"{label} queued identity drifted")
    allowed_artifacts = (
        FORMAL_EVALUATION_ARTIFACTS
        if label == "FFNet"
        else ROLE_OUTPUT_ARTIFACTS[label]
    )
    if (
        hashlib.sha256(source.encode("utf-8")).hexdigest()
        != material["source_sha256"]
        or (
            _evaluation_parameters_sha256(parameters)
            if label == "FFNet"
            else _content_sha256(parameters)
        )
        != material["parameters_sha256"]
        or successor_deploy._task_parent(task) != material["parent_task_id"]
        or queue_name != material["queue"]
        or queue_id != material["queue_id"]
        or not _artifact_names(task).issubset(allowed_artifacts)
        or sorted(_tags(task, field="tags", context=f"{label} queued target"))
        != sorted(material["tags"])
        or _model_inventory(task, context=f"{label} queued target")
        != {"input": [], "output": []}
        or sorted(
            _tags(task, field="system_tags", context=f"{label} queued target")
        )
        != sorted(material["system_tags"])
    ):
        raise RuntimeRecoveryError(f"{label} queued authority drifted")
    completion_evidence = None
    if status == "completed":
        if _artifact_names(task) != ROLE_OUTPUT_ARTIFACTS[label]:
            raise RuntimeRecoveryError(f"{label} completed without exact output evidence")
        if label == "C":
            completion_evidence = _validate_completed_candidate_manifest(
                task, material=material
            )
    service_environment = (
        None
        if label == "FFNet"
        else _service_environment(task, context=f"{label} queued target")
    )
    execution_environment = _execution_environment(
        task, context=f"{label} queued target"
    )
    if label == "FFNet" and execution_environment != material[
        "pre_reset_execution_environment"
    ]:
        raise RuntimeRecoveryError("FFNet queued execution environment drifted")
    return {
        "label": label,
        "task_id": TARGET_TASK_IDS[label],
        "project_id": evaluation_recovery.PROJECT_ID,
        "task_name": material["task_name"],
        "task_type": material["task_type"],
        "status": status,
        "queue": queue_name,
        "queue_id": queue_id,
        "source_sha256": material["source_sha256"],
        "parameters_sha256": material["parameters_sha256"],
        "parent_task_id": material["parent_task_id"],
        "artifact_names": sorted(_artifact_names(task)),
        "model_inventory": {"input": [], "output": []},
        "tags": sorted(material["tags"]),
        "system_tags": sorted(material["system_tags"]),
        "service_environment": service_environment,
        "execution_environment": execution_environment,
        "completion_evidence": completion_evidence,
    }


def _candidate_authority_transition_valid(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> bool:
    ranks = {"queued": 0, "in_progress": 1, "completed": 2}
    first_status = str(first.get("status") or "")
    second_status = str(second.get("status") or "")
    if first_status not in ranks or second_status not in ranks:
        return False
    stable_keys = (
        "label",
        "task_id",
        "project_id",
        "task_name",
        "task_type",
        "queue",
        "queue_id",
        "source_sha256",
        "parameters_sha256",
        "parent_task_id",
        "model_inventory",
        "tags",
        "system_tags",
    )
    return (
        ranks[second_status] >= ranks[first_status]
        and all(first.get(key) == second.get(key) for key in stable_keys)
        and set(first.get("artifact_names", [])).issubset(
            set(second.get("artifact_names", []))
        )
        and isinstance(first.get("service_environment"), Mapping)
        and isinstance(second.get("service_environment"), Mapping)
        and first["service_environment"].get("policy")
        == second["service_environment"].get("policy")
        and (
            first.get("completion_evidence") is None
            or first.get("completion_evidence") == second.get("completion_evidence")
        )
    )


def _enqueue_with_authority(
    task_class: object,
    *,
    label: str,
    material: Mapping[str, object],
    sleeper: Callable[[float], None],
) -> dict[str, object]:
    first = _validate_created_target(task_class, label=label, material=material)
    confirm = _validate_created_target(task_class, label=label, material=material)
    if first != confirm:
        raise RuntimeRecoveryError(f"{label} pre-enqueue readbacks differ")
    task = _created_unqueued_write_guard(
        task_class,
        label=label,
        material=material,
        context=f"{label} immediate pre-enqueue guard",
        exact_contract=True,
    )
    enqueuer = getattr(task_class, "enqueue", None)
    if not callable(enqueuer):
        raise RuntimeRecoveryError("ClearML Task class cannot enqueue tasks")
    callback_error: Exception | None = None
    try:
        response = enqueuer(task=task, queue_name=str(material["queue"]))
        if response is None or response is False:
            callback_error = RuntimeRecoveryError(f"{label} enqueue callback rejected")
    except Exception as error:
        callback_error = error
    last_error: Exception | None = callback_error
    for attempt in range(5):
        try:
            return _validate_queued_target(
                task_class, label=label, material=material
            )
        except Exception as error:
            last_error = error
        if attempt != 4:
            sleeper(1.0)
    raise RuntimeRecoveryError(f"{label} enqueue authority readback failed") from last_error


def _validate_attempt_structure(value: Mapping[str, object]) -> None:
    expected = {
        "schema_version": 1,
        "receipt_type": ATTEMPT_RECEIPT_TYPE,
        "status": "frozen_before_reset",
        "remote_state_changed": False,
        "same_ids_only": True,
        "replacement_tasks_created": False,
        "reset_order": list(RESET_ORDER),
        "enqueue_order": list(ENQUEUE_ORDER),
        "coformer_mutation_allowed": False,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise RuntimeRecoveryError(f"attempt receipt {key} drifted")
    material = value.get("target_material")
    if not isinstance(material, Mapping) or set(material) != set(RESET_ORDER):
        raise RuntimeRecoveryError("attempt target material inventory drifted")
    bindings = value.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "evaluation_recovery",
        "successor_deployment",
        "candidate_controller_recovery",
        "candidate_deployment",
        "formal_plan_amendment",
    }:
        raise RuntimeRecoveryError("attempt upstream binding inventory drifted")
    for label in RESET_ORDER:
        row = material[label]
        expected_source_sha = (
            EXPECTED_SOURCE_SHA256[label]
            if label == "FFNet"
            else row.get("source_sha256")
        )
        expected_parameters_sha = (
            EXPECTED_PARAMETERS_SHA256[label]
            if label == "FFNet"
            else row.get("parameters_sha256")
        )
        if (
            not isinstance(row, Mapping)
            or row.get("task_id") != TARGET_TASK_IDS[label]
            or _sha256(expected_source_sha, context=f"attempt {label} source")
            != row.get("source_sha256")
            or _sha256(
                expected_parameters_sha, context=f"attempt {label} parameters"
            )
            != row.get("parameters_sha256")
            or hashlib.sha256(str(row.get("source") or "").encode("utf-8")).hexdigest()
            != expected_source_sha
            or _content_sha256(row.get("parameters"))
            != expected_parameters_sha
        ):
            raise RuntimeRecoveryError(f"attempt target {label} material drifted")
    candidate = material["C"]
    amendment_binding = bindings.get("formal_plan_amendment")
    deployment_contract = candidate.get("deployment_contract")
    pre_reset_source = candidate.get("pre_reset_source")
    pre_reset_parameters = candidate.get("pre_reset_parameters")
    if (
        not isinstance(amendment_binding, Mapping)
        or candidate.get("formal_plan_amendment") != amendment_binding
        or candidate.get("pre_reset_source_sha256") != EXPECTED_SOURCE_SHA256["C"]
        or candidate.get("pre_reset_parameters_sha256")
        != EXPECTED_PARAMETERS_SHA256["C"]
        or type(pre_reset_source) is not str
        or hashlib.sha256(pre_reset_source.encode("utf-8")).hexdigest()
        != EXPECTED_SOURCE_SHA256["C"]
        or not isinstance(pre_reset_parameters, Mapping)
        or _content_sha256(pre_reset_parameters) != EXPECTED_PARAMETERS_SHA256["C"]
        or not isinstance(deployment_contract, Mapping)
        or deployment_contract.get(
            "formal_plan_amendment_receipt_seal_sha256"
        )
        != amendment_binding.get("receipt_seal_sha256")
    ):
        raise RuntimeRecoveryError("attempt amended candidate binding drifted")
    reconstructed = _candidate_target_material(
        pre_reset={
            **dict(candidate),
            "source": pre_reset_source,
            "source_sha256": EXPECTED_SOURCE_SHA256["C"],
            "parameters": dict(pre_reset_parameters),
            "parameters_sha256": EXPECTED_PARAMETERS_SHA256["C"],
            "tags": list(candidate.get("pre_reset_tags", [])),
            "queue": candidate.get("pre_reset_queue"),
            "queue_id": candidate.get("pre_reset_queue_id"),
        },
        amendment_binding=amendment_binding,
    )
    for key in (
        "source",
        "source_sha256",
        "source_bytes",
        "parameters",
        "parameters_sha256",
        "tags",
        "deployment_contract",
        "deployment_seal_sha256",
        "formal_plan_amendment",
        "service_environment_policy",
    ):
        if candidate.get(key) != reconstructed.get(key):
            raise RuntimeRecoveryError(
                f"attempt amended candidate {key} is not deterministic"
            )
    ffnet = material["FFNet"]
    if (
        ffnet.get("pre_reset_queue") != AMENDMENT_SOURCE_QUEUE
        or ffnet.get("queue") != AMENDMENT_TARGET_QUEUE
        or ffnet.get("queue_id")
        != formal.EXPECTED_QUEUE_IDS[AMENDMENT_TARGET_QUEUE]
        or "dependency-released" in ffnet.get("tags", [])
    ):
        raise RuntimeRecoveryError("attempt amended FFNet queue target drifted")
    evaluation_inventory = value.get("evaluation_inventory")
    records = (
        evaluation_inventory.get("records")
        if isinstance(evaluation_inventory, Mapping)
        else None
    )
    expected_ids = [
        task_id
        for _scope, _subject, task_id in evaluation_recovery.EXPECTED_RECOVERY_TASKS
    ]
    if (
        not isinstance(evaluation_inventory, Mapping)
        or evaluation_inventory.get("task_count") != len(expected_ids)
        or evaluation_inventory.get("exact_task_ids_sha256")
        != _content_sha256(expected_ids)
        or not isinstance(records, list)
        or len(records) != len(expected_ids)
        or any(not isinstance(row, Mapping) for row in records)
        or [row.get("task_id") for row in records] != expected_ids
    ):
        raise RuntimeRecoveryError("attempt exact evaluation inventory drifted")
    old_successors = value.get("old_successor_failure_evidence")
    if not isinstance(old_successors, Mapping) or set(old_successors) != {
        "W",
        "L",
        "A",
        "S",
    }:
        raise RuntimeRecoveryError("attempt old successor evidence drifted")
    name_inventories = value.get("exact_name_inventories")
    if not isinstance(name_inventories, Mapping) or set(name_inventories) != set(
        RESET_ORDER
    ):
        raise RuntimeRecoveryError("attempt exact-name inventory drifted")
    for label in RESET_ORDER:
        ids = name_inventories[label]
        if (
            not isinstance(ids, list)
            or TARGET_TASK_IDS[label] not in ids
            or len(ids) != len(set(ids))
        ):
            raise RuntimeRecoveryError(f"attempt {label} exact-name inventory drifted")
    same_name_history = value.get("preexisting_same_name_tasks")
    if not isinstance(same_name_history, Mapping) or set(same_name_history) != set(
        RESET_ORDER
    ):
        raise RuntimeRecoveryError("attempt preexisting same-name evidence drifted")
    for label in RESET_ORDER:
        rows = same_name_history[label]
        expected_history_ids = [
            task_id
            for task_id in name_inventories[label]
            if task_id != TARGET_TASK_IDS[label]
        ]
        if (
            not isinstance(rows, list)
            or any(not isinstance(row, Mapping) for row in rows)
            or [row.get("task_id") for row in rows] != expected_history_ids
        ):
            raise RuntimeRecoveryError(
                f"attempt {label} preexisting same-name evidence drifted"
            )


def _revalidate_preserved_evaluations(
    task_class: object,
    *,
    attempt_receipt: Mapping[str, object],
) -> dict[str, object]:
    records = attempt_receipt["evaluation_inventory"]["records"]
    preserved: list[dict[str, object]] = []
    for record in records:
        task_id = str(record["task_id"])
        if task_id == FFNET_TASK_ID:
            continue
        task = _authoritative_task(
            task_class, task_id, context="preserved exact evaluation"
        )
        status = _status(task)
        queue_name, queue_id = _queue_identity(
            task, context="preserved exact evaluation"
        )
        source = _source(
            task,
            entry_point="clearml_5090_bootstrap.py",
            context="preserved exact evaluation",
        )
        parameters = _parameters(task, context="preserved exact evaluation")
        models = _model_inventory(task, context="preserved exact evaluation")
        artifacts = _artifact_inventory(task)
        if (
            _exact_project_id(task, context="preserved exact evaluation")
            != record["project_id"]
            or successor_deploy._task_parent(task) != record["parent_task_id"]
            or str(getattr(task, "name", "") or "") != record["task_name"]
            or str(getattr(task, "task_type", "") or "")
            .rsplit(".", 1)[-1]
            .lower()
            != record["task_type"]
            or hashlib.sha256(source.encode("utf-8")).hexdigest()
            != record["script_sha256"]
            or _evaluation_parameters_sha256(parameters)
            != record["parameters_sha256"]
            or models["input"] != record["input_model_ids"]
            or models["output"] != record["output_model_ids"]
            or sorted(
                _tags(task, field="tags", context="preserved exact evaluation")
            )
            != record["tags"]
            or sorted(
                _tags(task, field="system_tags", context="preserved exact evaluation")
            )
            != record["system_tags"]
        ):
            raise RuntimeRecoveryError("a preserved evaluation contract drifted")
        if task_id == COFORMER_TASK_ID:
            prior = str(record["status"])
            allowed = (
                {"queued", "in_progress", "completed"}
                if prior == "queued"
                else {"in_progress", "completed"}
                if prior == "in_progress"
                else {"completed"}
            )
            if (
                status not in allowed
                or queue_name != record["queue"]
                or queue_id != record["queue_id"]
                or not _artifact_inventory_is_monotonic(
                    record.get("artifact_inventory"),
                    artifacts,
                    allowed_names=FORMAL_EVALUATION_ARTIFACTS,
                )
            ):
                raise RuntimeRecoveryError("CoFormer preservation contract drifted")
        elif (
            status != "created"
            or queue_name
            or queue_id
            or artifacts != record.get("artifact_inventory")
        ):
            raise RuntimeRecoveryError("a non-target evaluation changed during recovery")
        preserved.append(
            {
                "task_id": task_id,
                "subject": record["subject"],
                "status": status,
                "queue": queue_name,
                "queue_id": queue_id,
                "project_id": evaluation_recovery.PROJECT_ID,
                "parent_task_id": successor_deploy._task_parent(task),
                "task_name": str(getattr(task, "name", "") or ""),
                "task_type": str(getattr(task, "task_type", "") or "")
                .rsplit(".", 1)[-1]
                .lower(),
                "script_sha256": hashlib.sha256(
                    source.encode("utf-8")
                ).hexdigest(),
                "parameters_sha256": record["parameters_sha256"],
                "input_model_ids": list(models["input"]),
                "output_model_ids": list(models["output"]),
                "tags": sorted(
                    _tags(task, field="tags", context="preserved exact evaluation")
                ),
                "system_tags": sorted(
                    _tags(
                        task,
                        field="system_tags",
                        context="preserved exact evaluation",
                    )
                ),
                "artifact_names": sorted(_artifact_names(task)),
                "artifact_inventory": artifacts,
            }
        )
    return {
        "task_count": len(preserved),
        "coformer": next(
            item for item in preserved if item["task_id"] == COFORMER_TASK_ID
        ),
        "records": preserved,
        "records_sha256": _content_sha256(preserved),
        "no_preserved_task_mutated_by_recovery": True,
    }


def _revalidate_write_dependency_barrier(
    task_class: object,
    *,
    attempt_receipt: Mapping[str, object],
    old_plan: Mapping[str, object],
    phase: str,
) -> dict[str, object]:
    materials = attempt_receipt["target_material"]
    inventories = {
        label: _exact_name_inventory(
            task_class,
            task_name=str(materials[label]["task_name"]),
            parent_task_id=str(materials[label]["parent_task_id"]),
            expected_task_id=TARGET_TASK_IDS[label],
        )
        for label in RESET_ORDER
    }
    if inventories != attempt_receipt["exact_name_inventories"]:
        raise RuntimeRecoveryError(f"{phase} exact-name inventory drifted")
    history = {
        label: _same_name_history_snapshot(
            task_class,
            label=label,
            inventory=inventories[label],
        )
        for label in RESET_ORDER
    }
    if history != attempt_receipt["preexisting_same_name_tasks"]:
        raise RuntimeRecoveryError(f"{phase} same-name history drifted")
    preserved = _revalidate_preserved_evaluations(
        task_class, attempt_receipt=attempt_receipt
    )
    old_successors = _revalidate_old_successor_failures(
        task_class,
        evidence=attempt_receipt["old_successor_failure_evidence"],
    )
    amendment_authority = _validate_remote_amendment_producer(
        task_class,
        binding=attempt_receipt["bindings"]["formal_plan_amendment"],
        base_plan=old_plan,
    )
    if amendment_authority != attempt_receipt["amendment_producer_authority"]:
        raise RuntimeRecoveryError(f"{phase} amendment producer drifted")
    ffnet_first = _validate_created_target(
        task_class, label="FFNet", material=materials["FFNet"]
    )
    ffnet_confirm = _validate_created_target(
        task_class, label="FFNet", material=materials["FFNet"]
    )
    if ffnet_first != ffnet_confirm:
        raise RuntimeRecoveryError(f"{phase} FFNet authority readbacks differ")
    return {
        "phase": phase,
        "ffnet": ffnet_first,
        "preserved_records_sha256": preserved["records_sha256"],
        "old_successors_sha256": _content_sha256(old_successors),
        "amendment_producer": amendment_authority,
        "exact_name_inventories": inventories,
    }


def execute_recovery(
    task_class: object,
    *,
    attempt_receipt: Mapping[str, object],
    attempt_receipt_path: Path,
    sleeper: Callable[[float], None] = time.sleep,
    journal: dict[str, object] | None = None,
) -> dict[str, object]:
    _validate_attempt_structure(attempt_receipt)
    frozen_attempt, frozen_attempt_path = _read_sealed(
        attempt_receipt_path,
        context="runtime recovery attempt receipt",
    )
    if frozen_attempt_path != attempt_receipt_path.resolve() or frozen_attempt != dict(
        attempt_receipt
    ):
        raise RuntimeRecoveryError("runtime recovery attempt was not durably frozen")
    journal = {} if journal is None else journal
    materials = attempt_receipt["target_material"]
    current_name_inventories = {
        label: _exact_name_inventory(
            task_class,
            task_name=str(materials[label]["task_name"]),
            parent_task_id=str(materials[label]["parent_task_id"]),
            expected_task_id=TARGET_TASK_IDS[label],
        )
        for label in RESET_ORDER
    }
    if current_name_inventories != attempt_receipt["exact_name_inventories"]:
        raise RuntimeRecoveryError("exact-name inventory changed after attempt freeze")
    if {
        label: _same_name_history_snapshot(
            task_class,
            label=label,
            inventory=current_name_inventories[label],
        )
        for label in RESET_ORDER
    } != attempt_receipt["preexisting_same_name_tasks"]:
        raise RuntimeRecoveryError("a preexisting same-name task changed before reset")
    old_w = _authoritative_task(
        task_class, SUCCESSOR_TASK_IDS["W"], context="old W plan recheck"
    )
    old_plan = _validate_w_plan(old_w)
    amendment_authority = _validate_remote_amendment_producer(
        task_class,
        binding=attempt_receipt["bindings"]["formal_plan_amendment"],
        base_plan=old_plan,
    )
    if amendment_authority != attempt_receipt["amendment_producer_authority"]:
        raise RuntimeRecoveryError("amendment producer changed after attempt freeze")
    old_successors_before = _revalidate_old_successor_failures(
        task_class,
        evidence=attempt_receipt["old_successor_failure_evidence"],
    )
    # Recheck every failed target and its exact failure evidence before mutation.
    for label in RESET_ORDER:
        task = _authoritative_task(
            task_class, TARGET_TASK_IDS[label], context=f"{label} pre-reset recheck"
        )
        if _status(task) != "failed":
            raise RuntimeRecoveryError(f"{label} status changed after attempt freeze")
        evidence = _console_failure(task, label=label)
        if evidence != materials[label]["failure_evidence"]:
            raise RuntimeRecoveryError(f"{label} failure evidence changed after freeze")
        if _artifact_names(task) != set(materials[label]["artifact_names"]):
            raise RuntimeRecoveryError(f"{label} artifacts changed after freeze")
        source = _source(
            task,
            entry_point=str(materials[label]["entry_point"]),
            context=f"{label} pre-reset recheck",
        )
        parameters = _parameters(task, context=f"{label} pre-reset recheck")
        tags = _tags(task, field="tags", context=f"{label} pre-reset recheck")
        queue_name, queue_id = _queue_identity(
            task, context=f"{label} pre-reset recheck"
        )
        pre_source_sha = materials[label].get(
            "pre_reset_source_sha256", materials[label]["source_sha256"]
        )
        pre_parameters_sha = materials[label].get(
            "pre_reset_parameters_sha256", materials[label]["parameters_sha256"]
        )
        pre_tags = materials[label].get("pre_reset_tags", materials[label]["tags"])
        pre_queue = materials[label].get("pre_reset_queue", materials[label]["queue"])
        pre_queue_id = materials[label].get(
            "pre_reset_queue_id", materials[label]["queue_id"]
        )
        if (
            hashlib.sha256(source.encode("utf-8")).hexdigest() != pre_source_sha
            or (
                _evaluation_parameters_sha256(parameters)
                if label == "FFNet"
                else _content_sha256(parameters)
            )
            != pre_parameters_sha
            or sorted(tags) != sorted(pre_tags)
            or queue_name != pre_queue
            or queue_id != pre_queue_id
            or _model_inventory(task, context=f"{label} pre-reset recheck")
            != {"input": [], "output": []}
        ):
            raise RuntimeRecoveryError(f"{label} changed after attempt freeze")
    preserved_before = _revalidate_preserved_evaluations(
        task_class, attempt_receipt=attempt_receipt
    )
    journal["authoritative_pre_reset_recheck"] = True
    journal["preserved_before"] = preserved_before
    journal["old_successors_before"] = old_successors_before

    created: dict[str, object] = {}
    for label in RESET_ORDER:
        if label == "C":
            journal["pre_c_reset_dependency_barrier"] = (
                _revalidate_write_dependency_barrier(
                    task_class,
                    attempt_receipt=attempt_receipt,
                    old_plan=old_plan,
                    phase="pre_c_reset",
                )
            )
        journal.setdefault("reset_attempted", []).append(label)
        created[label] = _reset_and_install(
            task_class,
            label=label,
            material=materials[label],
            sleeper=sleeper,
        )
        journal.setdefault("created_unqueued", []).append(label)
    if set(created) != set(RESET_ORDER):
        raise RuntimeRecoveryError("both exact tasks must reach created+unqueued")
    preserved_after_reset = _revalidate_preserved_evaluations(
        task_class, attempt_receipt=attempt_receipt
    )
    journal["preserved_after_reset"] = preserved_after_reset
    if {
        label: _same_name_history_snapshot(
            task_class,
            label=label,
            inventory=current_name_inventories[label],
        )
        for label in RESET_ORDER
    } != attempt_receipt["preexisting_same_name_tasks"]:
        raise RuntimeRecoveryError("a preexisting same-name task changed after reset")

    queued: dict[str, object] = {}
    for label in ENQUEUE_ORDER:
        journal["pre_c_enqueue_dependency_barrier"] = (
            _revalidate_write_dependency_barrier(
                task_class,
                attempt_receipt=attempt_receipt,
                old_plan=old_plan,
                phase="pre_c_enqueue",
            )
        )
        queued[label] = _enqueue_with_authority(
            task_class,
            label=label,
            material=materials[label],
            sleeper=sleeper,
        )
        journal.setdefault("enqueued", []).append(label)
    preserved_final = _revalidate_preserved_evaluations(
        task_class, attempt_receipt=attempt_receipt
    )
    old_successors_final = _revalidate_old_successor_failures(
        task_class,
        evidence=attempt_receipt["old_successor_failure_evidence"],
    )
    amendment_authority_final = _validate_remote_amendment_producer(
        task_class,
        binding=attempt_receipt["bindings"]["formal_plan_amendment"],
        base_plan=old_plan,
    )
    if amendment_authority_final != amendment_authority:
        raise RuntimeRecoveryError("amendment producer changed during recovery")
    final_name_inventories = {
        label: _exact_name_inventory(
            task_class,
            task_name=str(materials[label]["task_name"]),
            parent_task_id=str(materials[label]["parent_task_id"]),
            expected_task_id=TARGET_TASK_IDS[label],
        )
        for label in RESET_ORDER
    }
    if final_name_inventories != attempt_receipt["exact_name_inventories"]:
        raise RuntimeRecoveryError("recovery created a replacement or duplicate task")
    preexisting_same_name_final = {
        label: _same_name_history_snapshot(
            task_class,
            label=label,
            inventory=final_name_inventories[label],
        )
        for label in RESET_ORDER
    }
    if preexisting_same_name_final != attempt_receipt["preexisting_same_name_tasks"]:
        raise RuntimeRecoveryError("a preexisting same-name task changed")
    ffnet_final = _validate_created_target(
        task_class,
        label="FFNet",
        material=materials["FFNet"],
    )
    ffnet_confirm = _validate_created_target(
        task_class,
        label="FFNet",
        material=materials["FFNet"],
    )
    if ffnet_final != ffnet_confirm:
        raise RuntimeRecoveryError("FFNet final created readbacks differ")
    candidate_final = _validate_queued_target(
        task_class,
        label="C",
        material=materials["C"],
    )
    candidate_confirm = _validate_queued_target(
        task_class,
        label="C",
        material=materials["C"],
    )
    if not _candidate_authority_transition_valid(
        candidate_final, candidate_confirm
    ):
        raise RuntimeRecoveryError("candidate final queued readbacks drifted")
    queued["C"] = candidate_confirm
    return _seal(
        {
            "schema_version": 1,
            "receipt_type": RECOVERY_RECEIPT_TYPE,
            "generated_at_utc": _utc_timestamp(),
            "status": "recovered_and_enqueued",
            "remote_state_changed": True,
            "same_ids_only": True,
            "replacement_tasks_created": False,
            "attempt_receipt_path": str(attempt_receipt_path.resolve()),
            "attempt_receipt_seal_sha256": attempt_receipt["seal_sha256"],
            "bindings": dict(attempt_receipt["bindings"]),
            "target_task_ids": dict(TARGET_TASK_IDS),
            "reset_order": list(RESET_ORDER),
            "enqueue_order": list(ENQUEUE_ORDER),
            "created_unqueued_authority": created,
            "queued_authority": queued,
            "ffnet_final_authority": ffnet_final,
            "preserved_evaluations": preserved_final,
            "old_successors_preserved": old_successors_final,
            "amendment_producer_authority": amendment_authority_final,
            "coformer_mutated": False,
            "exact_name_inventories": final_name_inventories,
            "preexisting_same_name_tasks_preserved": preexisting_same_name_final,
            "no_new_exact_name_task_ids": True,
        }
    )


def _validate_service_environment_evidence(
    value: object,
    *,
    material: Mapping[str, object],
    status: str,
    context: str,
) -> None:
    if not isinstance(value, Mapping) or value.get("policy") != material[
        "service_environment_policy"
    ]:
        raise RuntimeRecoveryError(f"{context} service policy drifted")
    policy = material["service_environment_policy"]
    if (
        value.get("binary") != policy["binary"]
        or value.get("docker") != policy["docker"]
        or value.get("output_uri") != policy["output_uri"]
    ):
        raise RuntimeRecoveryError(f"{context} service environment drifted")
    requirements = value.get("requirements")
    if not isinstance(requirements, Mapping):
        raise RuntimeRecoveryError(f"{context} requirements evidence is absent")
    canonical = str(policy["requirement"])
    form = requirements.get("form")
    if requirements.get("canonical_requirement") != canonical:
        raise RuntimeRecoveryError(f"{context} canonical requirement drifted")
    if form == "canonical":
        expected_raw: dict[str, object] = {"pip": canonical}
        if set(requirements) != {"form", "sha256", "canonical_requirement"}:
            raise RuntimeRecoveryError(f"{context} canonical evidence drifted")
    elif form == "agent_materialized":
        if status not in {"in_progress", "completed", "stopped", "failed"}:
            raise RuntimeRecoveryError(
                f"{context} materialized requirements are invalid for {status}"
            )
        packages = requirements.get("resolved_packages")
        if (
            not isinstance(packages, list)
            or any(type(item) is not str for item in packages)
            or not packages
            or any(not item or item != item.strip() for item in packages)
            or len(set(packages)) != len(packages)
            or packages.count(canonical) != 1
            or requirements.get("resolved_count") != len(packages)
            or set(requirements)
            != {
                "form",
                "sha256",
                "canonical_requirement",
                "resolved_count",
                "resolved_packages",
            }
        ):
            raise RuntimeRecoveryError(f"{context} materialized evidence drifted")
        expected_raw = {"org_pip": canonical, "pip": packages}
    else:
        raise RuntimeRecoveryError(f"{context} requirement form drifted")
    if requirements.get("sha256") != _content_sha256(expected_raw):
        raise RuntimeRecoveryError(f"{context} requirement hash drifted")


def validate_runtime_recovery_receipt(path: Path) -> dict[str, object]:
    value, resolved = _read_sealed(path, context="runtime recovery receipt")
    expected = {
        "schema_version": 1,
        "receipt_type": RECOVERY_RECEIPT_TYPE,
        "status": "recovered_and_enqueued",
        "remote_state_changed": True,
        "same_ids_only": True,
        "replacement_tasks_created": False,
        "target_task_ids": dict(TARGET_TASK_IDS),
        "reset_order": list(RESET_ORDER),
        "enqueue_order": list(ENQUEUE_ORDER),
        "coformer_mutated": False,
        "no_new_exact_name_task_ids": True,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise RuntimeRecoveryError(f"runtime recovery receipt {key} drifted")
    attempt_path = value.get("attempt_receipt_path")
    if type(attempt_path) is not str or not attempt_path:
        raise RuntimeRecoveryError("runtime recovery attempt path is invalid")
    attempt, attempt_resolved = _read_sealed(
        Path(attempt_path), context="runtime recovery attempt receipt"
    )
    _validate_attempt_structure(attempt)
    if value.get("attempt_receipt_seal_sha256") != attempt.get("seal_sha256"):
        raise RuntimeRecoveryError("runtime recovery attempt binding drifted")
    if value.get("bindings") != attempt.get("bindings"):
        raise RuntimeRecoveryError("runtime recovery upstream bindings drifted")
    bindings = attempt["bindings"]
    try:
        revalidated_bindings, _successor, _candidate, _amendment = _load_bindings(
            evaluation_recovery_receipt=Path(
                str(bindings["evaluation_recovery"]["path"])
            ),
            successor_deployment_receipt=Path(
                str(bindings["successor_deployment"]["path"])
            ),
            controller_recovery_receipt=Path(
                str(bindings["candidate_controller_recovery"]["path"])
            ),
            candidate_deployment_receipt=Path(
                str(bindings["candidate_deployment"]["path"])
            ),
            amendment_receipt=Path(
                str(bindings["formal_plan_amendment"]["path"])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeRecoveryError(
            "runtime recovery upstream receipt paths are invalid"
        ) from error
    if revalidated_bindings != bindings:
        raise RuntimeRecoveryError("runtime recovery upstream receipts changed")
    created = value.get("created_unqueued_authority")
    queued = value.get("queued_authority")
    if not isinstance(created, Mapping) or not isinstance(queued, Mapping):
        raise RuntimeRecoveryError("runtime recovery authority is incomplete")
    if set(created) != set(RESET_ORDER) or set(queued) != set(ENQUEUE_ORDER):
        raise RuntimeRecoveryError("runtime recovery authority order drifted")
    for label in RESET_ORDER:
        authority = created[label]
        material = attempt["target_material"][label]
        if (
            not isinstance(authority, Mapping)
            or authority.get("task_id") != TARGET_TASK_IDS[label]
            or authority.get("status") != "created"
            or authority.get("queue") is not None
            or authority.get("queue_id") is not None
            or authority.get("source_sha256")
            != attempt["target_material"][label]["source_sha256"]
            or authority.get("parameters_sha256")
            != material["parameters_sha256"]
            or authority.get("project_id") != evaluation_recovery.PROJECT_ID
            or authority.get("task_name") != material["task_name"]
            or authority.get("task_type") != material["task_type"]
            or authority.get("parent_task_id") != material["parent_task_id"]
            or authority.get("tags") != sorted(material["tags"])
            or authority.get("system_tags") != sorted(material["system_tags"])
            or authority.get("artifact_names") != []
            or authority.get("model_inventory") != {"input": [], "output": []}
            or (
                label == "FFNet"
                and authority.get("service_environment") is not None
            )
            or (
                label != "FFNet"
                and (
                    not isinstance(authority.get("service_environment"), Mapping)
                    or authority["service_environment"].get("policy")
                    != material["service_environment_policy"]
                )
            )
        ):
            raise RuntimeRecoveryError(f"runtime recovery {label} authority drifted")
        if label == "FFNet":
            if authority.get("execution_environment") != material[
                "pre_reset_execution_environment"
            ]:
                raise RuntimeRecoveryError(
                    "runtime recovery FFNet execution environment drifted"
                )
        else:
            _validate_service_environment_evidence(
                authority.get("service_environment"),
                material=material,
                status="created",
                context=f"runtime recovery {label} created authority",
            )
    for label in ENQUEUE_ORDER:
        authority = queued[label]
        material = attempt["target_material"][label]
        if (
            not isinstance(authority, Mapping)
            or authority.get("task_id") != TARGET_TASK_IDS[label]
            or authority.get("status")
            not in {"queued", "in_progress", "completed"}
            or authority.get("queue")
            != attempt["target_material"][label]["queue"]
            or authority.get("queue_id")
            != attempt["target_material"][label]["queue_id"]
            or authority.get("source_sha256")
            != attempt["target_material"][label]["source_sha256"]
            or authority.get("parameters_sha256")
            != material["parameters_sha256"]
            or authority.get("project_id") != evaluation_recovery.PROJECT_ID
            or authority.get("task_name") != material["task_name"]
            or authority.get("task_type") != material["task_type"]
            or authority.get("parent_task_id") != material["parent_task_id"]
            or authority.get("tags") != sorted(material["tags"])
            or authority.get("system_tags") != sorted(material["system_tags"])
            or authority.get("model_inventory") != {"input": [], "output": []}
            or not set(authority.get("artifact_names", [])).issubset(
                ROLE_OUTPUT_ARTIFACTS[label]
            )
        ):
            raise RuntimeRecoveryError(f"runtime recovery {label} queue drifted")
        status = str(authority["status"])
        _validate_service_environment_evidence(
            authority.get("service_environment"),
            material=material,
            status=status,
            context=f"runtime recovery {label} queued authority",
        )
        completion = authority.get("completion_evidence")
        if status == "completed":
            expected_completion = _completed_candidate_manifest_contract(material)
            if (
                not isinstance(completion, Mapping)
                or any(
                    completion.get(key) != expected
                    for key, expected in expected_completion.items()
                )
                or _sha256(
                    completion.get("manifest_seal_sha256"),
                    context="runtime recovery completed manifest seal",
                )
                != completion.get("manifest_seal_sha256")
                or _sha256(
                    completion.get("artifact_sha256"),
                    context="runtime recovery completed manifest artifact",
                )
                != completion.get("artifact_sha256")
                or type(completion.get("artifact_bytes")) is not int
                or completion["artifact_bytes"] <= 0
                or _sha256(
                    completion.get("entries_sha256"),
                    context="runtime recovery completed entries",
                )
                != completion.get("entries_sha256")
                or _sha256(
                    completion.get("formal_core_entries_sha256"),
                    context="runtime recovery completed formal core entries",
                )
                != completion.get("formal_core_entries_sha256")
            ):
                raise RuntimeRecoveryError(
                    "runtime recovery completed candidate evidence drifted"
                )
        elif completion is not None:
            raise RuntimeRecoveryError(
                "runtime recovery non-completed candidate has completion evidence"
            )
    if value.get("ffnet_final_authority") != created.get("FFNet"):
        raise RuntimeRecoveryError("FFNet final created+unqueued authority drifted")
    preserved = value.get("preserved_evaluations")
    preserved_records = (
        preserved.get("records") if isinstance(preserved, Mapping) else None
    )
    if (
        not isinstance(preserved, Mapping)
        or preserved.get("task_count") != 27
        or preserved.get("no_preserved_task_mutated_by_recovery") is not True
        or not isinstance(preserved.get("coformer"), Mapping)
        or preserved["coformer"].get("task_id") != COFORMER_TASK_ID
        or not isinstance(preserved_records, list)
        or len(preserved_records) != 27
        or preserved.get("records_sha256") != _content_sha256(preserved_records)
    ):
        raise RuntimeRecoveryError("runtime recovery preservation evidence drifted")
    attempt_records = {
        str(row["task_id"]): row
        for row in attempt["evaluation_inventory"]["records"]
        if str(row["task_id"]) != FFNET_TASK_ID
    }
    if set(attempt_records) != {
        str(row.get("task_id") or "")
        for row in preserved_records
        if isinstance(row, Mapping)
    }:
        raise RuntimeRecoveryError("runtime recovery preserved task IDs drifted")
    for row in preserved_records:
        if not isinstance(row, Mapping):
            raise RuntimeRecoveryError("runtime recovery preserved record is invalid")
        frozen = attempt_records[str(row["task_id"])]
        stable_expected = {
            "subject": frozen["subject"],
            "project_id": frozen["project_id"],
            "parent_task_id": frozen["parent_task_id"],
            "task_name": frozen["task_name"],
            "task_type": frozen["task_type"],
            "script_sha256": frozen["script_sha256"],
            "parameters_sha256": frozen["parameters_sha256"],
            "input_model_ids": frozen["input_model_ids"],
            "output_model_ids": frozen["output_model_ids"],
            "tags": frozen["tags"],
            "system_tags": frozen["system_tags"],
        }
        if any(row.get(key) != expected for key, expected in stable_expected.items()):
            raise RuntimeRecoveryError("runtime recovery preserved record drifted")
        if row["task_id"] == COFORMER_TASK_ID:
            prior = str(frozen["status"])
            allowed = (
                {"queued", "in_progress", "completed"}
                if prior == "queued"
                else {"in_progress", "completed"}
                if prior == "in_progress"
                else {"completed"}
            )
            if (
                row.get("status") not in allowed
                or row.get("queue") != frozen["queue"]
                or row.get("queue_id") != frozen["queue_id"]
                or not _artifact_inventory_is_monotonic(
                    frozen.get("artifact_inventory"),
                    row.get("artifact_inventory"),
                    allowed_names=FORMAL_EVALUATION_ARTIFACTS,
                )
                or row.get("artifact_names")
                != sorted(
                    str(item.get("name") or "")
                    for item in row.get("artifact_inventory", [])
                    if isinstance(item, Mapping)
                )
            ):
                raise RuntimeRecoveryError("runtime recovery CoFormer record drifted")
        elif (
            row.get("status") != "created"
            or row.get("queue")
            or row.get("queue_id")
            or row.get("artifact_names") != []
            or row.get("artifact_inventory") != frozen.get("artifact_inventory")
        ):
            raise RuntimeRecoveryError("runtime recovery preserved record drifted")
    old_successors = value.get("old_successors_preserved")
    if not isinstance(old_successors, Mapping) or set(old_successors) != {
        "W",
        "L",
        "A",
        "S",
    }:
        raise RuntimeRecoveryError("runtime recovery old-successor evidence drifted")
    for role in ("W", "L", "A", "S"):
        row = old_successors[role]
        frozen = attempt["old_successor_failure_evidence"][role]
        if (
            not isinstance(row, Mapping)
            or not isinstance(frozen, Mapping)
            or row.get("task_id") != SUCCESSOR_TASK_IDS[role]
            or row.get("status") != "failed"
            or any(
                row.get(key) != frozen.get(key)
                for key in (
                    "project_id",
                    "task_name",
                    "task_type",
                    "parent_task_id",
                    "queue",
                    "queue_id",
                    "source_sha256",
                    "parameters_sha256",
                    "artifact_names",
                    "artifact_inventory",
                    "tags",
                    "system_tags",
                    "model_inventory",
                    "pre_reset_execution_environment",
                    "service_environment",
                    "failure_evidence",
                )
            )
        ):
            raise RuntimeRecoveryError(
                f"runtime recovery old successor {role} drifted"
            )
    if value.get("amendment_producer_authority") != attempt.get(
        "amendment_producer_authority"
    ):
        raise RuntimeRecoveryError("runtime recovery amendment authority drifted")
    if value.get("exact_name_inventories") != attempt.get("exact_name_inventories"):
        raise RuntimeRecoveryError("runtime recovery exact-name evidence drifted")
    if value.get("preexisting_same_name_tasks_preserved") != attempt.get(
        "preexisting_same_name_tasks"
    ):
        raise RuntimeRecoveryError(
            "runtime recovery preexisting same-name evidence drifted"
        )
    amendment_binding = attempt["bindings"]["formal_plan_amendment"]
    evaluation_ids_sha256 = attempt["evaluation_inventory"][
        "exact_task_ids_sha256"
    ]
    return {
        "path": str(resolved),
        "receipt_seal_sha256": value["seal_sha256"],
        "attempt_receipt_path": str(attempt_resolved),
        "attempt_receipt_seal_sha256": attempt["seal_sha256"],
        "target_task_ids": dict(TARGET_TASK_IDS),
        "ffnet_created_unqueued": True,
        "ffnet_authority": dict(created["FFNet"]),
        "candidate_services": True,
        "candidate_authority": dict(queued["C"]),
        "formal_plan_amendment_receipt_seal_sha256": amendment_binding[
            "receipt_seal_sha256"
        ],
        "formal_plan_amendment_producer_task_id": amendment_binding[
            "producer_task_id"
        ],
        "revised_formal_plan_seal_sha256": amendment_binding[
            "revised_plan_seal_sha256"
        ],
        "exact_evaluation_task_ids_sha256": evaluation_ids_sha256,
        "exact_evaluation_task_count": 28,
        "same_ids_only": True,
        "replacement_tasks_created": False,
        "coformer_mutated": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-token", default="")
    parser.add_argument("--evaluation-recovery-receipt", type=Path, required=True)
    parser.add_argument("--successor-deployment-receipt", type=Path, required=True)
    parser.add_argument("--controller-recovery-receipt", type=Path, required=True)
    parser.add_argument("--candidate-deployment-receipt", type=Path, required=True)
    parser.add_argument("--amendment-receipt", type=Path, required=True)
    parser.add_argument("--attempt-receipt", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def _print_summary(path: Path, value: Mapping[str, object]) -> None:
    print(
        json.dumps(
            {
                "receipt": str(path),
                "receipt_type": value.get("receipt_type"),
                "status": value.get("status"),
                "remote_state_changed": value.get("remote_state_changed"),
                "target_task_ids": value.get("target_task_ids", TARGET_TASK_IDS),
                "same_ids_only": value.get("same_ids_only"),
                "replacement_tasks_created": value.get("replacement_tasks_created"),
                "coformer_mutated": value.get("coformer_mutated"),
                "seal_sha256": value.get("seal_sha256"),
            },
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and args.execute_token != EXECUTE_TOKEN:
        raise RuntimeRecoveryError(f"exact execute token required: {EXECUTE_TOKEN}")
    if not args.execute and args.execute_token:
        raise RuntimeRecoveryError("--execute-token requires --execute")
    if args.execute and args.attempt_receipt is None:
        raise RuntimeRecoveryError("--attempt-receipt is required with --execute")
    if not args.execute and args.attempt_receipt is not None:
        raise RuntimeRecoveryError("--attempt-receipt requires --execute")
    paths = [
        args.evaluation_recovery_receipt,
        args.successor_deployment_receipt,
        args.controller_recovery_receipt,
        args.candidate_deployment_receipt,
        args.amendment_receipt,
        args.receipt,
        *([] if args.attempt_receipt is None else [args.attempt_receipt]),
    ]
    resolved = [item.resolve() for item in paths]
    if len(resolved) != len(set(resolved)):
        raise RuntimeRecoveryError("all input and output receipt paths must differ")
    _require_new(args.receipt, context="runtime recovery receipt")
    if args.attempt_receipt is not None:
        _require_new(args.attempt_receipt, context="runtime recovery attempt receipt")
    try:
        from clearml import Task
    except ImportError as error:
        raise RuntimeRecoveryError("ClearML is required for recovery inspection") from error
    attempt = build_attempt_receipt(
        Task,
        evaluation_recovery_receipt=args.evaluation_recovery_receipt,
        successor_deployment_receipt=args.successor_deployment_receipt,
        controller_recovery_receipt=args.controller_recovery_receipt,
        candidate_deployment_receipt=args.candidate_deployment_receipt,
        amendment_receipt=args.amendment_receipt,
    )
    if not args.execute:
        plan = dict(attempt)
        plan.pop("target_material", None)
        plan.update(
            {
                "receipt_type": PLAN_RECEIPT_TYPE,
                "status": "planned",
                "execute_token": EXECUTE_TOKEN,
                "target_task_ids": dict(TARGET_TASK_IDS),
            }
        )
        plan = _seal(plan)
        _write_new(args.receipt, plan)
        _print_summary(args.receipt, plan)
        return 0
    if args.attempt_receipt is None:  # defensive after CLI validation
        raise RuntimeRecoveryError("runtime recovery attempt receipt is absent")
    _write_new(args.attempt_receipt, attempt)
    journal: dict[str, object] = {}
    try:
        recovered = execute_recovery(
            Task,
            attempt_receipt=attempt,
            attempt_receipt_path=args.attempt_receipt,
            journal=journal,
        )
    except Exception as error:
        failure = _seal(
            {
                "schema_version": 1,
                "receipt_type": RECOVERY_RECEIPT_TYPE,
                "generated_at_utc": _utc_timestamp(),
                "status": "failed_closed",
                "remote_state_changed": bool(journal.get("reset_attempted")),
                "same_ids_only": True,
                "replacement_tasks_created": False,
                "attempt_receipt_path": str(args.attempt_receipt.resolve()),
                "attempt_receipt_seal_sha256": attempt["seal_sha256"],
                "bindings": dict(attempt["bindings"]),
                "target_task_ids": dict(TARGET_TASK_IDS),
                "partial_journal": journal,
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _write_new(args.receipt, failure)
        raise
    _write_new(args.receipt, recovered)
    _print_summary(args.receipt, recovered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ATTEMPT_RECEIPT_TYPE",
    "CANDIDATE_CONTROLLER_TASK_ID",
    "COFORMER_TASK_ID",
    "ENQUEUE_ORDER",
    "EXECUTE_TOKEN",
    "FFNET_TASK_ID",
    "PLAN_RECEIPT_TYPE",
    "RECOVERY_RECEIPT_TYPE",
    "RESET_ORDER",
    "RuntimeRecoveryError",
    "SUCCESSOR_TASK_IDS",
    "TARGET_TASK_IDS",
    "build_attempt_receipt",
    "execute_recovery",
    "main",
    "validate_runtime_recovery_receipt",
)
