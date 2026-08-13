#!/usr/bin/env python3
"""Recover the one stopped formal candidate controller without cloning it.

The default mode is read-only and writes a sealed local plan.  Remote mutation
requires the exact execute token, the already-completed 26+2 evaluation recovery
receipt, and two distinct write-once receipt paths.  The attempt receipt is
durably written before the exact task ID is force-reset.
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
    from tools.resilient_v2x import (
        deploy_clearml_formal_candidate_evaluation_queue as deployment,
    )
    from tools.resilient_v2x import (
        recover_clearml_formal_evaluation_attempts as evaluation_recovery,
    )
except ModuleNotFoundError:  # direct execution from tools/resilient_v2x
    import deploy_clearml_formal_candidate_evaluation_queue as deployment
    import recover_clearml_formal_evaluation_attempts as evaluation_recovery


CONTROLLER_TASK_ID = "b6f0fbab32a5478183a45b3ca833fc01"
EXECUTE_TOKEN = (
    "RESET_EXACT_STOPPED_FORMAL_CANDIDATE_CONTROLLER_"
    "B6F0FBAB32A5478183A45B3CA833FC01"
)
ATTEMPT_RECEIPT_TYPE = "resilient_v2x_formal_candidate_controller_recovery_attempt"
PLAN_RECEIPT_TYPE = "resilient_v2x_formal_candidate_controller_recovery_plan"
RECOVERY_RECEIPT_TYPE = "resilient_v2x_formal_candidate_controller_recovery"
LEGACY_SOURCE_SHA256 = (
    "9055b41235d3e8c34583a03fe64e2a2ab42f91e99995ffac13c5a7e77cf10adb"
)
LEGACY_SOURCE_BYTES = 62_919
LEGACY_PARAMETERS_SHA256 = (
    "74d8a4c4cac3dff674df53e62f676b975c83b481645ee45d793eea5eacc251ec"
)
LEGACY_DEPLOYMENT_SEAL_SHA256 = (
    "6803b34e9c7bee2818e2507fda740b35098e0d477e2ef07b0a17f4748516f654"
)
LEGACY_DEPLOYMENT_RECEIPT_SEAL_SHA256 = (
    "4696f549011988b953fa4537f1c42369fc28e1abc32daf4c9a58901e3b497cbf"
)
LEGACY_ARTIFACT_NAMES = ("formal_1337_candidate_evaluation_manifest",)
LEGACY_MANIFEST_SHA256 = (
    "2275743af40dfa5fbcf49fbeae54d4665662821ebda376acc4470a1ec5e6a0d8"
)
LEGACY_MANIFEST_BYTES = 6_735
LEGACY_PARAMETERS = {
    "Args/execute": "True",
    "Args/execute_remotely": "False",
    "Args/execute_token": "EXECUTE_EXACT_FORMAL_1337_CANDIDATE_EVALUATIONS",
    "Args/poll_seconds": "60.0",
    "Args/service_queue": "services",
    "Args/single_pass": "False",
    "Args/timeout_hours": "168.0",
    "Deployment/deployment_seal_sha256": LEGACY_DEPLOYMENT_SEAL_SHA256,
    "Deployment/parent_task_id": deployment.PARENT_TASK_ID,
    "Deployment/schema_version": "1",
    "Deployment/services_queue_id": deployment.SERVICES_QUEUE_ID,
    "Deployment/source_sha256": LEGACY_SOURCE_SHA256,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_RECEIPT_BYTES = 4_194_304


class ControllerRecoveryError(RuntimeError):
    """Raised when exact-ID controller recovery cannot be proven."""


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ControllerRecoveryError("recovery evidence is not canonical JSON") from error


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seal(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


def _sha256(value: object, *, context: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ControllerRecoveryError(f"{context} is not a lowercase SHA-256")
    return value


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
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


def _require_new_output(path: Path, *, context: str) -> None:
    if path.is_symlink() or path.exists():
        raise ControllerRecoveryError(f"{context} must be a new write-once path")


def _read_receipt(path: Path, *, context: str) -> tuple[dict[str, object], Path]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ControllerRecoveryError(f"{context} cannot be resolved") from error
    if path.is_symlink() or not resolved.is_file():
        raise ControllerRecoveryError(f"{context} must be a regular file")
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise ControllerRecoveryError(f"{context} cannot be read") from error
    if not raw or len(raw) > MAX_RECEIPT_BYTES:
        raise ControllerRecoveryError(f"{context} size is invalid")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ControllerRecoveryError(f"{context} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ControllerRecoveryError(f"{context} is not a JSON object")
    seal = _sha256(value.get("seal_sha256"), context=f"{context} seal")
    if _seal(value)["seal_sha256"] != seal:
        raise ControllerRecoveryError(f"{context} seal mismatch")
    return value, resolved


def _artifact_metadata(task: object) -> tuple[tuple[str, ...], str]:
    value = getattr(task, "artifacts", None)
    if not isinstance(value, Mapping):
        raise ControllerRecoveryError("candidate controller artifacts are unavailable")
    names = tuple(sorted(str(name) for name in value))
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ControllerRecoveryError("candidate controller artifact names are invalid")
    safe: list[dict[str, object]] = []
    for name in names:
        artifact = value[name]
        record: dict[str, object] = {"name": name}
        for key in ("hash", "size", "content_size", "timestamp", "type", "mode"):
            item = getattr(artifact, key, None)
            if item is not None:
                record[key] = str(item)
        safe.append(record)
    return names, _content_sha256(safe)


def _model_inventory(task: object) -> dict[str, list[str]]:
    getter = getattr(task, "get_models", None)
    if not callable(getter):
        raise ControllerRecoveryError("candidate controller models are unavailable")
    value = getter()
    if not isinstance(value, Mapping) or set(value) != {"input", "output"}:
        raise ControllerRecoveryError("candidate controller model inventory drifted")
    result: dict[str, list[str]] = {}
    for role in ("input", "output"):
        rows = value[role]
        if not isinstance(rows, Sequence) or isinstance(
            rows, (str, bytes, bytearray)
        ):
            raise ControllerRecoveryError("candidate controller model inventory drifted")
        ids = [str(getattr(item, "id", "") or "") for item in rows]
        if any(
            len(item) != 32
            or any(character not in "0123456789abcdef" for character in item)
            for item in ids
        ) or len(set(ids)) != len(ids):
            raise ControllerRecoveryError("candidate controller model inventory drifted")
        result[role] = ids
    return result


def _legacy_receipt_binding() -> dict[str, object]:
    path = (
        Path(__file__).resolve().parents[2]
        / "artifacts/resilient_v2x/formal-candidate-evaluation-deployment"
        / f"recovery-receipt-{CONTROLLER_TASK_ID}.json"
    )
    value, resolved = _read_receipt(path, context="legacy deployment receipt")
    expected = {
        "receipt_type": (
            "resilient_v2x_formal_candidate_evaluation_queue_deployment"
        ),
        "status": "deployed",
        "task_id": CONTROLLER_TASK_ID,
        "created_shell": False,
        "disposition": "resumed_exact_created_shell",
        "deployment_seal_sha256": LEGACY_DEPLOYMENT_SEAL_SHA256,
        "seal_sha256": LEGACY_DEPLOYMENT_RECEIPT_SEAL_SHA256,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ControllerRecoveryError(f"legacy deployment receipt {key} drifted")
    source = value.get("source")
    authority = value.get("queued_authority")
    if (
        not isinstance(source, Mapping)
        or source.get("sha256") != LEGACY_SOURCE_SHA256
        or source.get("bytes") != LEGACY_SOURCE_BYTES
        or not isinstance(authority, Mapping)
        or authority.get("task_id") != CONTROLLER_TASK_ID
        or authority.get("source_sha256") != LEGACY_SOURCE_SHA256
        or authority.get("parameters_sha256") != LEGACY_PARAMETERS_SHA256
    ):
        raise ControllerRecoveryError("legacy deployment receipt authority drifted")
    return {
        "path": str(resolved),
        "receipt_seal_sha256": LEGACY_DEPLOYMENT_RECEIPT_SEAL_SHA256,
        "deployment_seal_sha256": LEGACY_DEPLOYMENT_SEAL_SHA256,
        "source_sha256": LEGACY_SOURCE_SHA256,
    }


def _target_binding(prepared: Mapping[str, object]) -> dict[str, object]:
    source = prepared.get("source")
    source_record = prepared.get("source_record")
    parameters = prepared.get("parameters")
    if (
        type(source) is not str
        or not isinstance(source_record, Mapping)
        or not isinstance(parameters, Mapping)
    ):
        raise ControllerRecoveryError("target deployment material is invalid")
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if (
        source_record.get("sha256") != source_sha
        or source_record.get("bytes") != len(source.encode("utf-8"))
    ):
        raise ControllerRecoveryError("target deployment source record drifted")
    return {
        "controller_task_id": CONTROLLER_TASK_ID,
        "task_name": deployment.CONTROLLER_NAME,
        "parent_task_id": deployment.PARENT_TASK_ID,
        "project_id": deployment.PROJECT_ID,
        "source_sha256": source_sha,
        "source_bytes": len(source.encode("utf-8")),
        "parameters_sha256": _content_sha256(
            deployment._normalized_parameters(parameters)
        ),
        "deployment_seal_sha256": _sha256(
            prepared.get("deployment_seal_sha256"),
            context="target deployment seal",
        ),
        "services_queue_id": deployment.SERVICES_QUEUE_ID,
    }


def _pre_reset_snapshot(task_class: object) -> dict[str, object]:
    parent = deployment._validate_parent(task_class)
    exact = deployment._exact_named_tasks(task_class)
    exact_ids = [
        deployment._task_id(
            getattr(task, "id", ""), "candidate controller recovery duplicate"
        )
        for task in exact
    ]
    if exact_ids != [CONTROLLER_TASK_ID]:
        raise ControllerRecoveryError(
            "candidate controller exact-name inventory is not the one frozen task ID"
        )
    task = deployment._authoritative_task(
        task_class,
        CONTROLLER_TASK_ID,
        context="candidate controller recovery target",
    )
    if deployment._normalized_status(task) != "stopped":
        raise ControllerRecoveryError("candidate controller is not stopped")
    project = deployment._project_identity(
        task, context="candidate controller recovery target"
    )
    if str(getattr(task, "name", "") or "") != deployment.CONTROLLER_NAME:
        raise ControllerRecoveryError("candidate controller name drifted")
    task_type = str(getattr(task, "task_type", "") or "").rsplit(".", 1)[-1]
    if task_type.lower() != "controller":
        raise ControllerRecoveryError("candidate controller type drifted")
    if deployment._task_parent(task) != deployment.PARENT_TASK_ID:
        raise ControllerRecoveryError("candidate controller parent drifted")
    if deployment._output_destination(task) != deployment.FILES_SERVER_URI:
        raise ControllerRecoveryError("candidate controller output URI drifted")
    script = deployment._task_script(task)
    source = script.get("diff")
    if (
        str(script.get("repository") or "") != ""
        or str(script.get("working_dir") or "") != "."
        or str(script.get("entry_point") or "") != deployment.ENTRY_POINT
        or str(script.get("binary") or "") != "python"
        or type(source) is not str
        or len(source.encode("utf-8")) != LEGACY_SOURCE_BYTES
        or hashlib.sha256(source.encode("utf-8")).hexdigest()
        != LEGACY_SOURCE_SHA256
    ):
        raise ControllerRecoveryError("candidate controller legacy source drifted")
    requirements = deployment._validated_requirements(
        script,
        status="stopped",
        context="candidate controller recovery target",
    )
    parameters = deployment._observed_parameters(
        task, context="candidate controller recovery target"
    )
    normalized = deployment._normalized_parameters(parameters)
    if normalized != LEGACY_PARAMETERS or _content_sha256(normalized) != (
        LEGACY_PARAMETERS_SHA256
    ):
        raise ControllerRecoveryError("candidate controller legacy parameters drifted")
    if set(deployment._tags(task, context="candidate controller recovery target")) != (
        deployment.TAGS
    ):
        raise ControllerRecoveryError("candidate controller tags drifted")
    expected_docker = " ".join(
        f"{deployment.SERVICE_DOCKER_IMAGE} {deployment.SERVICE_DOCKER_ARGS}".split()
    )
    if deployment._task_docker(task) != expected_docker:
        raise ControllerRecoveryError("candidate controller Docker contract drifted")
    queue_name, queue_id = deployment._queue_identity(
        task, context="candidate controller recovery target"
    )
    if (
        queue_name != deployment.SERVICES_QUEUE
        or queue_id != deployment.SERVICES_QUEUE_ID
    ):
        raise ControllerRecoveryError("candidate controller legacy queue drifted")
    artifact_names, artifact_metadata_sha = _artifact_metadata(task)
    if artifact_names != LEGACY_ARTIFACT_NAMES:
        raise ControllerRecoveryError("candidate controller legacy artifact inventory drifted")
    manifest = getattr(task, "artifacts")[LEGACY_ARTIFACT_NAMES[0]]
    manifest_hash = str(getattr(manifest, "hash", "") or "")
    manifest_size = getattr(manifest, "size", None)
    if manifest_hash != LEGACY_MANIFEST_SHA256 or manifest_size != (
        LEGACY_MANIFEST_BYTES
    ):
        raise ControllerRecoveryError("candidate controller legacy manifest drifted")
    model_inventory = _model_inventory(task)
    if model_inventory != {"input": [], "output": []}:
        raise ControllerRecoveryError("candidate controller unexpectedly owns models")
    return {
        "controller_task_id": CONTROLLER_TASK_ID,
        "status": "stopped",
        "task_name": deployment.CONTROLLER_NAME,
        "task_type": "controller",
        "parent_task_id": deployment.PARENT_TASK_ID,
        **project,
        "source_sha256": LEGACY_SOURCE_SHA256,
        "source_bytes": LEGACY_SOURCE_BYTES,
        "parameters_sha256": LEGACY_PARAMETERS_SHA256,
        "requirements": requirements,
        "deployment_seal_sha256": LEGACY_DEPLOYMENT_SEAL_SHA256,
        "queue": queue_name,
        "queue_id": queue_id,
        "artifact_names": list(artifact_names),
        "manifest_sha256": manifest_hash,
        "manifest_bytes": manifest_size,
        "artifact_metadata_sha256": artifact_metadata_sha,
        "model_inventory": model_inventory,
        "console_attempt_evidence": evaluation_recovery._console_evidence(
            task, status="stopped"
        ),
        "parent_authority": parent,
    }


def build_attempt_receipt(
    task_class: object,
    *,
    evaluation_recovery_binding: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    prepared = deployment._prepare()
    target = _target_binding(prepared)
    pre_reset = _pre_reset_snapshot(task_class)
    attempt = _seal(
        {
            "schema_version": 1,
            "receipt_type": ATTEMPT_RECEIPT_TYPE,
            "generated_at_utc": _utc_timestamp(),
            "status": "frozen_before_reset",
            "remote_state_changed": False,
            "controller_task_id": CONTROLLER_TASK_ID,
            "same_id_only": True,
            "replacement_controller_created": False,
            "evaluation_recovery_receipt": dict(evaluation_recovery_binding),
            "legacy_deployment_receipt": _legacy_receipt_binding(),
            "pre_reset_authority": pre_reset,
            "target_deployment": target,
            "planned_actions": [
                "authoritative_pre_reset_recheck",
                "reset_force",
                "install_frozen_target_deployment",
                "authoritative_created_unqueued_readback",
            ],
        }
    )
    return attempt, prepared


def _wait_created(
    task_class: object,
    *,
    sleeper: Callable[[float], None],
) -> object:
    last = ""
    for attempt in range(5):
        task = deployment._authoritative_task(
            task_class,
            CONTROLLER_TASK_ID,
            context="candidate controller recovery readback",
        )
        last = deployment._normalized_status(task)
        if last == "created":
            return task
        if attempt != 4:
            sleeper(1.0)
    raise ControllerRecoveryError(
        f"candidate controller did not reach created; observed {last!r}"
    )


def _empty_artifacts(task: object) -> None:
    names, _ = _artifact_metadata(task)
    if names:
        raise ControllerRecoveryError(
            "candidate controller reset retained stale output artifacts"
        )


def _empty_models(task: object) -> None:
    if _model_inventory(task) != {"input": [], "output": []}:
        raise ControllerRecoveryError(
            "candidate controller reset retained stale model bindings"
        )


def execute_recovery(
    task_class: object,
    *,
    attempt_receipt: Mapping[str, object],
    prepared: Mapping[str, object],
    attempt_receipt_path: Path,
    sleeper: Callable[[float], None] = time.sleep,
    journal: dict[str, object] | None = None,
) -> dict[str, object]:
    journal = {} if journal is None else journal
    observed = _pre_reset_snapshot(task_class)
    if observed != attempt_receipt.get("pre_reset_authority"):
        raise ControllerRecoveryError(
            "candidate controller changed after attempt receipt was frozen"
        )
    journal["authoritative_pre_reset_recheck"] = True
    task = deployment._authoritative_task(
        task_class,
        CONTROLLER_TASK_ID,
        context="candidate controller reset target",
    )
    resetter = getattr(task, "reset", None)
    if not callable(resetter):
        raise ControllerRecoveryError("candidate controller cannot be reset")
    journal["reset_attempted"] = True
    callback_error: Exception | None = None
    try:
        result = resetter(force=True)
        if result is False:
            callback_error = ControllerRecoveryError(
                "candidate controller reset callback was not confirmed"
            )
    except Exception as error:  # authority can still prove reset acceptance
        callback_error = error
    try:
        task = _wait_created(task_class, sleeper=sleeper)
    except Exception as error:
        if callback_error is not None:
            raise ControllerRecoveryError(
                "candidate controller reset authority readback failed"
            ) from callback_error
        raise error
    journal["reset_authoritatively_created"] = True
    _empty_artifacts(task)
    _empty_models(task)
    journal["configure_attempted"] = True
    authority: dict[str, object] | None = None
    configure_error: Exception | None = None
    for attempt in range(3):
        try:
            deployment._configure_shell(task, prepared=prepared)
            configure_error = None
        except Exception as error:
            configure_error = error
        try:
            first = deployment._validate_task(
                task_class,
                task_id=CONTROLLER_TASK_ID,
                prepared=prepared,
                expected_statuses=("created",),
            )
            confirm = deployment._validate_task(
                task_class,
                task_id=CONTROLLER_TASK_ID,
                prepared=prepared,
                expected_statuses=("created",),
            )
            if first != confirm:
                raise ControllerRecoveryError(
                    "candidate controller final authority readbacks differ"
                )
            authority = first
            break
        except Exception as error:
            if configure_error is None:
                configure_error = error
        if attempt != 2:
            sleeper(1.0)
    if authority is None:
        raise ControllerRecoveryError(
            "candidate controller target deployment readback failed"
        ) from configure_error
    final_task = deployment._authoritative_task(
        task_class,
        CONTROLLER_TASK_ID,
        context="candidate controller final artifact readback",
    )
    _empty_artifacts(final_task)
    _empty_models(final_task)
    exact_ids = [
        deployment._task_id(
            getattr(item, "id", ""), "candidate controller final duplicate"
        )
        for item in deployment._exact_named_tasks(task_class)
    ]
    if exact_ids != [CONTROLLER_TASK_ID]:
        raise ControllerRecoveryError(
            "candidate controller recovery created a replacement or duplicate"
        )
    journal["target_authoritatively_created_unqueued"] = True
    target = _target_binding(prepared)
    return _seal(
        {
            "schema_version": 1,
            "receipt_type": RECOVERY_RECEIPT_TYPE,
            "generated_at_utc": _utc_timestamp(),
            "status": "recovered",
            "remote_state_changed": True,
            "controller_task_id": CONTROLLER_TASK_ID,
            "same_id_only": True,
            "replacement_controller_created": False,
            "all_controller_artifacts_cleared": True,
            "all_controller_models_absent": True,
            "created_unqueued": True,
            "attempt_receipt_path": str(attempt_receipt_path.resolve()),
            "attempt_receipt_seal_sha256": attempt_receipt["seal_sha256"],
            "evaluation_recovery_receipt": dict(
                attempt_receipt["evaluation_recovery_receipt"]
            ),
            "pre_reset_authority": dict(attempt_receipt["pre_reset_authority"]),
            "target_deployment": target,
            "final_authority": authority,
            "actions": [
                "authoritative_pre_reset_recheck",
                "reset_force",
                "install_frozen_target_deployment",
                "authoritative_created_unqueued_readback",
            ],
        }
    )


def _validate_attempt_receipt(path: Path) -> tuple[dict[str, object], Path]:
    value, resolved = _read_receipt(path, context="controller attempt receipt")
    expected = {
        "schema_version": 1,
        "receipt_type": ATTEMPT_RECEIPT_TYPE,
        "status": "frozen_before_reset",
        "remote_state_changed": False,
        "controller_task_id": CONTROLLER_TASK_ID,
        "same_id_only": True,
        "replacement_controller_created": False,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ControllerRecoveryError(f"controller attempt receipt {key} drifted")
    return value, resolved


def validate_controller_recovery_receipt(path: Path) -> dict[str, object]:
    """Validate a successful exact-ID controller recovery for deployment pinning."""

    value, resolved = _read_receipt(path, context="controller recovery receipt")
    expected = {
        "schema_version": 1,
        "receipt_type": RECOVERY_RECEIPT_TYPE,
        "status": "recovered",
        "remote_state_changed": True,
        "controller_task_id": CONTROLLER_TASK_ID,
        "same_id_only": True,
        "replacement_controller_created": False,
        "all_controller_artifacts_cleared": True,
        "all_controller_models_absent": True,
        "created_unqueued": True,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ControllerRecoveryError(f"controller recovery receipt {key} drifted")
    attempt_path = value.get("attempt_receipt_path")
    if type(attempt_path) is not str or not attempt_path:
        raise ControllerRecoveryError("controller attempt receipt path is invalid")
    attempt, attempt_resolved = _validate_attempt_receipt(Path(attempt_path))
    attempt_seal = _sha256(
        value.get("attempt_receipt_seal_sha256"),
        context="controller recovery attempt seal",
    )
    if attempt.get("seal_sha256") != attempt_seal:
        raise ControllerRecoveryError("controller attempt receipt binding drifted")
    evaluation_binding = value.get("evaluation_recovery_receipt")
    if not isinstance(evaluation_binding, Mapping):
        raise ControllerRecoveryError("evaluation recovery binding is absent")
    evaluation_seal = _sha256(
        evaluation_binding.get("receipt_seal_sha256"),
        context="evaluation recovery binding seal",
    )
    task_ids_sha = _sha256(
        evaluation_binding.get("task_ids_sha256"),
        context="evaluation recovery task IDs",
    )
    if (
        evaluation_binding.get("task_count") != 28
        or evaluation_binding.get("all_tasks_created_unqueued") is not True
        or evaluation_binding.get("replacement_tasks_created") is not False
        or attempt.get("evaluation_recovery_receipt") != evaluation_binding
    ):
        raise ControllerRecoveryError("evaluation recovery binding drifted")
    target = value.get("target_deployment")
    current_target = _target_binding(deployment._prepare())
    if target != current_target or attempt.get("target_deployment") != current_target:
        raise ControllerRecoveryError("controller target deployment drifted")
    pre_reset = value.get("pre_reset_authority")
    if (
        not isinstance(pre_reset, Mapping)
        or pre_reset.get("status") != "stopped"
        or pre_reset.get("source_sha256") != LEGACY_SOURCE_SHA256
        or pre_reset.get("parameters_sha256") != LEGACY_PARAMETERS_SHA256
        or attempt.get("pre_reset_authority") != pre_reset
    ):
        raise ControllerRecoveryError("controller pre-reset authority drifted")
    final = value.get("final_authority")
    if (
        not isinstance(final, Mapping)
        or final.get("task_id") != CONTROLLER_TASK_ID
        or final.get("status") != "created"
        or final.get("source_sha256") != current_target["source_sha256"]
        or final.get("parameters_sha256") != current_target["parameters_sha256"]
        or final.get("deployment_seal_sha256")
        != current_target["deployment_seal_sha256"]
        or final.get("queue") is not None
        or final.get("queue_id") is not None
    ):
        raise ControllerRecoveryError("controller final authority drifted")
    if value.get("actions") != [
        "authoritative_pre_reset_recheck",
        "reset_force",
        "install_frozen_target_deployment",
        "authoritative_created_unqueued_readback",
    ]:
        raise ControllerRecoveryError("controller recovery action contract drifted")
    return {
        "path": str(resolved),
        "receipt_seal_sha256": value["seal_sha256"],
        "attempt_receipt_path": str(attempt_resolved),
        "attempt_receipt_seal_sha256": attempt_seal,
        "controller_task_id": CONTROLLER_TASK_ID,
        "evaluation_recovery_receipt_seal_sha256": evaluation_seal,
        "evaluation_recovery_task_ids_sha256": task_ids_sha,
        "target_source_sha256": current_target["source_sha256"],
        "target_deployment_seal_sha256": current_target[
            "deployment_seal_sha256"
        ],
        "same_id_only": True,
        "replacement_controller_created": False,
        "created_unqueued": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-token", default="")
    parser.add_argument("--evaluation-recovery-receipt", type=Path, required=True)
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
                "controller_task_id": value.get("controller_task_id"),
                "same_id_only": value.get("same_id_only"),
                "replacement_controller_created": value.get(
                    "replacement_controller_created"
                ),
                "seal_sha256": value.get("seal_sha256"),
            },
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and args.execute_token != EXECUTE_TOKEN:
        raise ControllerRecoveryError(f"exact execute token required: {EXECUTE_TOKEN}")
    if args.execute and args.attempt_receipt is None:
        raise ControllerRecoveryError("--attempt-receipt is required with --execute")
    if not args.execute and args.execute_token:
        raise ControllerRecoveryError("--execute-token requires --execute")
    if not args.execute and args.attempt_receipt is not None:
        raise ControllerRecoveryError("--attempt-receipt requires --execute")
    if args.attempt_receipt is not None and (
        args.attempt_receipt.resolve() == args.receipt.resolve()
    ):
        raise ControllerRecoveryError("attempt and recovery receipts must differ")
    if args.attempt_receipt is not None and (
        args.attempt_receipt.resolve()
        == args.evaluation_recovery_receipt.resolve()
    ):
        raise ControllerRecoveryError(
            "attempt and evaluation recovery receipts must differ"
        )
    if args.receipt.resolve() == args.evaluation_recovery_receipt.resolve():
        raise ControllerRecoveryError("controller receipt must not overwrite evaluation recovery")
    _require_new_output(args.receipt, context="controller recovery receipt")
    if args.attempt_receipt is not None:
        _require_new_output(args.attempt_receipt, context="controller attempt receipt")
    evaluation_binding = evaluation_recovery.validate_recovery_receipt(
        args.evaluation_recovery_receipt
    )
    try:
        from clearml import Task
    except ImportError as error:
        raise ControllerRecoveryError("ClearML is required for recovery inspection") from error
    attempt, prepared = build_attempt_receipt(
        Task,
        evaluation_recovery_binding=evaluation_binding,
    )
    if not args.execute:
        plan = _seal(
            {
                **attempt,
                "receipt_type": PLAN_RECEIPT_TYPE,
                "status": "planned",
                "execute_token": EXECUTE_TOKEN,
            }
        )
        _write_new(args.receipt, plan)
        _print_summary(args.receipt, plan)
        return 0
    if args.attempt_receipt is None:  # defensive: execute preflight requires it
        raise ControllerRecoveryError(
            "controller attempt receipt is absent after execute preflight"
        )
    _write_new(args.attempt_receipt, attempt)
    journal: dict[str, object] = {}
    try:
        recovered = execute_recovery(
            Task,
            attempt_receipt=attempt,
            prepared=prepared,
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
                "controller_task_id": CONTROLLER_TASK_ID,
                "same_id_only": True,
                "replacement_controller_created": False,
                "attempt_receipt_path": str(args.attempt_receipt.resolve()),
                "attempt_receipt_seal_sha256": attempt["seal_sha256"],
                "evaluation_recovery_receipt": dict(evaluation_binding),
                "partial_journal": journal,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
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
    "CONTROLLER_TASK_ID",
    "ControllerRecoveryError",
    "EXECUTE_TOKEN",
    "PLAN_RECEIPT_TYPE",
    "RECOVERY_RECEIPT_TYPE",
    "build_attempt_receipt",
    "execute_recovery",
    "main",
    "validate_controller_recovery_receipt",
)
