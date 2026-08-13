#!/usr/bin/env python3
"""Plan, deploy, or verify the formal candidate-evaluation service controller.

The default invocation is local-only.  Remote mutation requires both
``--execute`` and the exact deployment token.  A deterministic deployment seal
and exact source bytes make repeated execution idempotent: an existing exact
controller is reused, while any same-name drift fails closed without creating a
second shell.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

try:
    from tools.resilient_v2x import (
        recover_clearml_formal_evaluation_attempts as evaluation_recovery,
    )
except ModuleNotFoundError:  # direct execution from tools/resilient_v2x
    import recover_clearml_formal_evaluation_attempts as evaluation_recovery


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    ROOT / "tools/resilient_v2x/clearml_formal_candidate_evaluation_queue.py"
)
ENTRY_POINT = SOURCE_PATH.name
ARTIFACT_DIR = (
    ROOT / "artifacts/resilient_v2x/formal-candidate-evaluation-deployment"
)

PROJECT_NAME = "ResilientV2X/Training"
PROJECT_ID = "6e43f972e5ea4cee901a7c8855fce8cd"
CONTROLLER_NAME = "ResilientV2X formal 1337 candidate evaluation queue"
PARENT_TASK_ID = "7e244a711751469b8cfdb25d77b05269"
PARENT_TASK_NAME = "ResilientV2X formal training provenance successor"
FILES_SERVER_URI = "http://10.100.34.118:8081"
SERVICES_QUEUE = "services"
SERVICES_QUEUE_ID = "85707f0acbd04a4d95b49202474866b3"
SERVICE_DOCKER_IMAGE = "ubuntu:24.04"
SERVICE_DOCKER_ARGS = (
    "--network host "
    "--add-host apiserver:10.100.34.118 "
    "--add-host fileserver:10.100.34.118 "
    "--add-host webserver:10.100.34.118 "
    "-e CLEARML_API_HOST=http://10.100.34.118:8008 "
    "-e CLEARML_WEB_HOST=http://10.100.34.118:8080 "
    "-e CLEARML_FILES_HOST=http://10.100.34.118:8081"
)
SERVICE_REQUIREMENTS = ("clearml==2.1.11",)

DEPLOY_EXECUTE_TOKEN = "DEPLOY_EXACT_FORMAL_1337_CANDIDATE_EVALUATION_QUEUE"
CONTROLLER_EXECUTE_TOKEN = "EXECUTE_EXACT_FORMAL_1337_CANDIDATE_EVALUATIONS"
PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
TAGS = frozenset(
    {
        "ResilientV2X-suite",
        "formal-candidate-evaluation-queue",
        PROTOCOL_ID,
        "single-seed-20250218",
        "cpu-controller",
    }
)
RUNTIME_PARAMETERS: dict[str, object] = {
    "Args/execute": True,
    "Args/execute_token": CONTROLLER_EXECUTE_TOKEN,
    "Args/execute_remotely": False,
    "Args/service_queue": SERVICES_QUEUE,
    "Args/poll_seconds": 60.0,
    "Args/timeout_hours": 168.0,
    "Args/single_pass": False,
}


class DeploymentError(RuntimeError):
    """Raised when a deployment invariant cannot be proven."""


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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seal(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


def _validate_evaluation_recovery_receipt(path: Path) -> dict[str, object]:
    try:
        return evaluation_recovery.validate_recovery_receipt(path)
    except evaluation_recovery.RecoveryError as error:
        raise DeploymentError(
            f"evaluation recovery receipt is invalid: {error}"
        ) from error


def _validate_controller_recovery_receipt(path: Path) -> dict[str, object]:
    try:
        try:
            from tools.resilient_v2x import (
                recover_clearml_formal_candidate_controller as recovery,
            )
        except ModuleNotFoundError:  # direct execution from tools/resilient_v2x
            import recover_clearml_formal_candidate_controller as recovery
        return recovery.validate_controller_recovery_receipt(path)
    except Exception as error:
        raise DeploymentError(
            f"candidate controller recovery receipt is invalid: {error}"
        ) from error


def _bind_evaluation_recovery(
    receipt: Mapping[str, object], binding: Mapping[str, object]
) -> dict[str, object]:
    value = dict(receipt)
    value.pop("seal_sha256", None)
    value["evaluation_recovery_receipt"] = dict(binding)
    return _seal(value)


def _bind_controller_recovery(
    receipt: Mapping[str, object], binding: Mapping[str, object]
) -> dict[str, object]:
    value = dict(receipt)
    value.pop("seal_sha256", None)
    value["candidate_controller_recovery_receipt"] = dict(binding)
    return _seal(value)


def _validate_recovery_cross_binding(
    evaluation: Mapping[str, object], controller: Mapping[str, object]
) -> None:
    if controller.get("evaluation_recovery_receipt_seal_sha256") != (
        evaluation.get("receipt_seal_sha256")
    ):
        raise DeploymentError(
            "controller recovery does not bind the supplied evaluation recovery receipt"
        )
    if controller.get("evaluation_recovery_task_ids_sha256") != evaluation.get(
        "task_ids_sha256"
    ):
        raise DeploymentError(
            "controller recovery evaluation task inventory binding drifted"
        )


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _task_id(value: object, context: str) -> str:
    result = str(value or "")
    if not _is_lower_hex(result, 32):
        raise DeploymentError(f"{context} must be a lowercase 32-hex ClearML ID")
    return result


def _read_source() -> str:
    try:
        resolved = SOURCE_PATH.resolve(strict=True)
    except OSError as error:
        raise DeploymentError("candidate controller source cannot be resolved") from error
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise DeploymentError("candidate controller source escapes the repository") from error
    if SOURCE_PATH.is_symlink() or not resolved.is_file():
        raise DeploymentError("candidate controller source must be a regular file")
    try:
        source = resolved.read_text(encoding="utf-8")
        compile(source, str(resolved), "exec")
    except (OSError, UnicodeError, SyntaxError) as error:
        raise DeploymentError("candidate controller source is not valid UTF-8 Python") from error
    if not source:
        raise DeploymentError("candidate controller source is empty")
    return source


def _literal_assignment(source: str, name: str) -> object:
    tree = ast.parse(source)
    matches: list[ast.expr] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == name:
                    matches.append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name and node.value is not None:
                matches.append(node.value)
    if len(matches) != 1:
        raise DeploymentError(f"candidate source must assign {name} exactly once")
    try:
        return ast.literal_eval(matches[0])
    except (ValueError, TypeError) as error:
        raise DeploymentError(f"candidate source assignment {name} is not literal") from error


def _source_record(source: str) -> dict[str, object]:
    expected = {
        "PROJECT_NAME": PROJECT_NAME,
        "CONTROLLER_NAME": CONTROLLER_NAME,
        "EXECUTE_TOKEN": CONTROLLER_EXECUTE_TOKEN,
        "DEFAULT_SERVICE_QUEUE": SERVICES_QUEUE,
    }
    for name, value in expected.items():
        if _literal_assignment(source, name) != value:
            raise DeploymentError(f"candidate source {name} contract drifted")
    encoded = source.encode("utf-8")
    return {
        "path": str(SOURCE_PATH.relative_to(ROOT)),
        "entry_point": ENTRY_POINT,
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "deployment_mode": "exact_standalone_diff",
    }


_ISOLATED_SMOKE_DRIVER = r"""
import hashlib
import importlib.util
import json
import pathlib
import sys
import types

path = pathlib.Path(sys.argv[1]).resolve(strict=True)
expected_sha = sys.argv[2]
repository_root = pathlib.Path(sys.argv[3]).resolve(strict=True)
for entry in sys.path:
    if not entry:
        continue
    candidate = pathlib.Path(entry).resolve()
    if candidate == repository_root or repository_root in candidate.parents:
        raise RuntimeError("isolated smoke sys.path contains the repository")
source = path.read_text(encoding="utf-8")
observed_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
if observed_sha != expected_sha:
    raise RuntimeError("standalone source SHA-256 drifted")
stub = types.ModuleType("allegroai")
stub.Task = object
sys.modules["allegroai"] = stub
spec = importlib.util.spec_from_file_location(
    "_resilient_v2x_candidate_evaluation_standalone_smoke", path
)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot build standalone import spec")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
result = {
    "source_sha256": observed_sha,
    "project": module.PROJECT_NAME,
    "task_name": module.CONTROLLER_NAME,
    "execute_token": module.EXECUTE_TOKEN,
    "service_queue": module.DEFAULT_SERVICE_QUEUE,
}
print("RESILIENT_V2X_CANDIDATE_DEPLOY_SMOKE=" + json.dumps(result, sort_keys=True))
"""


def _isolated_import_smoke(
    source: str, *, timeout_seconds: float = 120.0
) -> dict[str, object]:
    if (
        type(timeout_seconds) not in {int, float}
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise DeploymentError("standalone smoke timeout must be finite and positive")
    source_sha256 = _sha256_text(source)
    with tempfile.TemporaryDirectory(
        prefix="resilient-v2x-candidate-eval-smoke-"
    ) as raw:
        directory = Path(raw)
        path = directory / ENTRY_POINT
        path.write_text(source, encoding="utf-8")
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    _ISOLATED_SMOKE_DRIVER,
                    str(path),
                    source_sha256,
                    str(ROOT),
                ],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise DeploymentError("standalone isolated import smoke failed") from error
    marker = "RESILIENT_V2X_CANDIDATE_DEPLOY_SMOKE="
    marked = [
        line for line in completed.stdout.splitlines() if line.startswith(marker)
    ]
    if completed.returncode != 0 or len(marked) != 1:
        details = (completed.stderr or completed.stdout).strip()[-2000:]
        raise DeploymentError(f"standalone isolated import smoke failed: {details}")
    try:
        observed = json.loads(marked[0][len(marker) :])
    except (json.JSONDecodeError, TypeError) as error:
        raise DeploymentError("standalone smoke result is invalid JSON") from error
    expected = {
        "source_sha256": source_sha256,
        "project": PROJECT_NAME,
        "task_name": CONTROLLER_NAME,
        "execute_token": CONTROLLER_EXECUTE_TOKEN,
        "service_queue": SERVICES_QUEUE,
    }
    if observed != expected:
        raise DeploymentError("standalone smoke identity drifted")
    return {
        "result": "pass",
        "isolation": "python_-I_without_repository_on_sys_path",
        "interpreter": str(Path(sys.executable).resolve()),
        **expected,
    }


def _normalized_parameters(value: Mapping[str, object]) -> dict[str, str]:
    return {str(key): str(item) for key, item in value.items()}


def _deployment_contract(source_record: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "deployment_type": "resilient_v2x_formal_candidate_evaluation_queue",
        "project": PROJECT_NAME,
        "task_name": CONTROLLER_NAME,
        "task_type": "controller",
        "parent_task_id": PARENT_TASK_ID,
        "source": dict(source_record),
        "runtime_parameters": _normalized_parameters(RUNTIME_PARAMETERS),
        "tags": sorted(TAGS),
        "output_uri": FILES_SERVER_URI,
        "requirements": list(SERVICE_REQUIREMENTS),
        "docker_image": SERVICE_DOCKER_IMAGE,
        "docker_args": SERVICE_DOCKER_ARGS,
        "queue": SERVICES_QUEUE,
        "queue_id": SERVICES_QUEUE_ID,
    }


def _task_parameters(
    *, deployment_seal_sha256: str, source_sha256: str
) -> dict[str, object]:
    return {
        **RUNTIME_PARAMETERS,
        "Deployment/schema_version": 1,
        "Deployment/deployment_seal_sha256": deployment_seal_sha256,
        "Deployment/source_sha256": source_sha256,
        "Deployment/services_queue_id": SERVICES_QUEUE_ID,
        "Deployment/parent_task_id": PARENT_TASK_ID,
    }


def _prepare() -> dict[str, object]:
    source = _read_source()
    source_record = _source_record(source)
    smoke = _isolated_import_smoke(source)
    contract = _deployment_contract(source_record)
    deployment_seal = _content_sha256(contract)
    parameters = _task_parameters(
        deployment_seal_sha256=deployment_seal,
        source_sha256=str(source_record["sha256"]),
    )
    return {
        "source": source,
        "source_record": source_record,
        "standalone_import_smoke": smoke,
        "deployment_contract": contract,
        "deployment_seal_sha256": deployment_seal,
        "parameters": parameters,
    }


def _normalized_status(task: object) -> str:
    value = getattr(task, "status", None)
    if value is None:
        getter = getattr(task, "get_status", None)
        value = getter() if callable(getter) else None
    value = getattr(value, "value", value)
    return str(value or "").rsplit(".", 1)[-1].lower()


def _authoritative_task(task_class: object, task_id: str, *, context: str) -> object:
    getter = getattr(task_class, "get_task", None)
    if not callable(getter):
        raise DeploymentError("ClearML Task class cannot resolve tasks")
    task = getter(task_id=task_id)
    if _task_id(getattr(task, "id", ""), context) != task_id:
        raise DeploymentError(f"{context} identity mismatch")
    if bool(getattr(task, "_offline_mode", False)):
        raise DeploymentError(f"{context} cannot be verified in offline mode")
    reloader = getattr(task, "_reload", None)
    if not callable(reloader):
        raise DeploymentError(f"{context} has no authoritative reload")
    has_skip = hasattr(task, "_reload_skip_flag")
    previous_skip = getattr(task, "_reload_skip_flag", None)
    try:
        if has_skip:
            setattr(task, "_reload_skip_flag", False)
        snapshot = reloader()
    except Exception as error:
        raise DeploymentError(f"{context} authoritative reload failed") from error
    finally:
        if has_skip:
            setattr(task, "_reload_skip_flag", previous_skip)
    if snapshot is None or isinstance(
        snapshot, (bool, int, float, str, bytes, bytearray)
    ):
        raise DeploymentError(f"{context} authoritative reload returned no record")
    if _task_id(getattr(snapshot, "id", ""), f"{context} record") != task_id:
        raise DeploymentError(f"{context} authoritative record identity mismatch")
    try:
        setattr(task, "_data", snapshot)
    except Exception as error:
        raise DeploymentError(f"{context} cannot install authoritative record") from error
    return task


def _task_script(task: object) -> dict[str, object]:
    script = getattr(getattr(task, "data", None), "script", None)
    to_dict = getattr(script, "to_dict", None)
    value = to_dict() if callable(to_dict) else None
    if not isinstance(value, Mapping):
        value = script if isinstance(script, Mapping) else None
    if not isinstance(value, Mapping):
        raise DeploymentError("task script metadata is unavailable")
    return dict(value)


def _project_name(task: object) -> str:
    getter = getattr(task, "get_project_name", None)
    if not callable(getter):
        raise DeploymentError("task cannot expose its project name")
    return str(getter() or "")


def _project_identity(task: object, *, context: str) -> dict[str, str]:
    direct_id = str(getattr(task, "project", None) or "")
    record_id = str(getattr(getattr(task, "data", None), "project", None) or "")
    if direct_id != PROJECT_ID or record_id != PROJECT_ID:
        raise DeploymentError(f"{context} project ID drifted")
    name = _project_name(task)
    if name != PROJECT_NAME:
        raise DeploymentError(f"{context} project name drifted")
    return {"project_id": PROJECT_ID, "project_name": PROJECT_NAME}


def _task_parent(task: object) -> str:
    return str(getattr(getattr(task, "data", None), "parent", None) or "")


def _output_destination(task: object) -> str:
    return str(
        getattr(
            getattr(getattr(task, "data", None), "output", None), "destination", None
        )
        or ""
    )


def _observed_parameters(task: object, *, context: str) -> dict[str, object]:
    getter = getattr(task, "get_parameters", None)
    if not callable(getter):
        raise DeploymentError(f"{context} cannot expose parameters")
    try:
        value = getter(backwards_compatibility=False, cast=False)
    except TypeError:
        try:
            value = getter(cast=False)
        except TypeError:
            value = getter()
    if not isinstance(value, Mapping):
        raise DeploymentError(f"{context} parameters are invalid")
    return {str(key): item for key, item in value.items()}


def _parameter_equal(observed: object, expected: object) -> bool:
    if type(expected) is bool:
        return str(observed).casefold() == str(expected).casefold()
    return str(observed) == str(expected)


def _task_docker(task: object) -> str:
    getter = getattr(task, "get_base_docker", None)
    if not callable(getter):
        raise DeploymentError("task cannot expose its base Docker command")
    value = getter()
    if type(value) is not str or not value.strip():
        raise DeploymentError("task base Docker command is empty")
    return " ".join(value.split())


def _validated_requirements(
    script: Mapping[str, object], *, status: str, context: str
) -> dict[str, object]:
    requirements = script.get("requirements")
    canonical = "\n".join(SERVICE_REQUIREMENTS)
    if requirements == {"pip": canonical}:
        return {
            "form": "canonical",
            "sha256": _content_sha256(requirements),
        }
    if status not in {"in_progress", "completed", "stopped"}:
        raise DeploymentError(f"{context} package requirements drifted")
    if not isinstance(requirements, Mapping) or set(requirements) != {
        "org_pip",
        "pip",
    }:
        raise DeploymentError(f"{context} materialized requirements drifted")
    if requirements.get("org_pip") != canonical:
        raise DeploymentError(f"{context} original package requirements drifted")
    resolved = requirements.get("pip")
    if not isinstance(resolved, Sequence) or isinstance(
        resolved, (str, bytes, bytearray)
    ):
        raise DeploymentError(f"{context} resolved package requirements drifted")
    if any(type(item) is not str for item in resolved):
        raise DeploymentError(f"{context} resolved package requirements drifted")
    packages = list(resolved)
    if (
        not packages
        or any(not item or item != item.strip() for item in packages)
        or len(set(packages)) != len(packages)
        or canonical not in packages
    ):
        raise DeploymentError(f"{context} resolved package requirements drifted")
    return {
        "form": "agent_materialized",
        "sha256": _content_sha256(
            {"org_pip": canonical, "pip": packages}
        ),
        "resolved_count": len(packages),
    }


def _tags(task: object, *, context: str) -> tuple[str, ...]:
    value = getattr(getattr(task, "data", None), "tags", None)
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise DeploymentError(f"{context} tag inventory is unavailable")
    result = tuple(str(item) for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise DeploymentError(f"{context} tag inventory is invalid or duplicated")
    return result


def _queue_identity(task: object, *, context: str) -> tuple[str, str]:
    getter = getattr(task, "get_executed_queue", None)
    if not callable(getter):
        raise DeploymentError(f"{context} cannot expose execution queue")
    return str(getter(return_name=True) or ""), str(
        getter(return_name=False) or ""
    )


def _validate_parent(task_class: object) -> dict[str, object]:
    task = _authoritative_task(
        task_class, PARENT_TASK_ID, context="candidate controller parent"
    )
    if _normalized_status(task) != "completed":
        raise DeploymentError("candidate controller parent is not completed")
    project = _project_identity(task, context="candidate controller parent")
    if str(getattr(task, "name", "") or "") != PARENT_TASK_NAME:
        raise DeploymentError("candidate controller parent name drifted")
    return {
        "task_id": PARENT_TASK_ID,
        "task_name": PARENT_TASK_NAME,
        **project,
        "status": "completed",
    }


def _validate_task(
    task_class: object,
    *,
    task_id: str,
    prepared: Mapping[str, object],
    expected_statuses: Sequence[str],
) -> dict[str, object]:
    context = "formal candidate evaluation controller"
    task = _authoritative_task(task_class, task_id, context=context)
    status = _normalized_status(task)
    if status not in expected_statuses:
        raise DeploymentError(
            f"{context} status {status!r} is outside {tuple(expected_statuses)!r}"
        )
    project = _project_identity(task, context=context)
    if str(getattr(task, "name", "") or "") != CONTROLLER_NAME:
        raise DeploymentError(f"{context} name drifted")
    task_type = str(getattr(task, "task_type", "") or "").rsplit(".", 1)[-1]
    if task_type.lower() != "controller":
        raise DeploymentError(f"{context} type drifted")
    if _task_parent(task) != PARENT_TASK_ID:
        raise DeploymentError(f"{context} parent drifted")
    if _output_destination(task) != FILES_SERVER_URI:
        raise DeploymentError(f"{context} output destination drifted")
    script = _task_script(task)
    if (
        str(script.get("repository") or "") != ""
        or str(script.get("working_dir") or "") != "."
        or str(script.get("entry_point") or "") != ENTRY_POINT
        or str(script.get("binary") or "") != "python"
    ):
        raise DeploymentError(f"{context} script identity drifted")
    source = script.get("diff")
    expected_source = prepared.get("source")
    if type(source) is not str or source != expected_source:
        raise DeploymentError(f"{context} source bytes drifted")
    source_record = prepared.get("source_record")
    if not isinstance(source_record, Mapping) or _sha256_text(source) != str(
        source_record.get("sha256") or ""
    ):
        raise DeploymentError(f"{context} source SHA-256 drifted")
    requirements = _validated_requirements(
        script, status=status, context=context
    )
    expected_docker = " ".join(
        f"{SERVICE_DOCKER_IMAGE} {SERVICE_DOCKER_ARGS}".split()
    )
    if _task_docker(task) != expected_docker:
        raise DeploymentError(f"{context} Docker command drifted")
    parameters = prepared.get("parameters")
    if not isinstance(parameters, Mapping):
        raise DeploymentError("prepared parameter contract is invalid")
    observed_parameters = _observed_parameters(task, context=context)
    if set(observed_parameters) != set(parameters) or any(
        not _parameter_equal(observed_parameters[key], expected)
        for key, expected in parameters.items()
    ):
        raise DeploymentError(f"{context} exact parameter contract drifted")
    observed_tags = _tags(task, context=context)
    if set(observed_tags) != TAGS:
        raise DeploymentError(f"{context} formal tag set drifted")
    queue_name: str | None = None
    queue_id: str | None = None
    if status != "created":
        queue_name, queue_id = _queue_identity(task, context=context)
        if queue_name != SERVICES_QUEUE or queue_id != SERVICES_QUEUE_ID:
            raise DeploymentError(f"{context} execution queue drifted")
    return {
        "task_id": task_id,
        "status": status,
        **project,
        "task_name": CONTROLLER_NAME,
        "task_type": "controller",
        "parent_task_id": PARENT_TASK_ID,
        "entry_point": ENTRY_POINT,
        "source_sha256": str(source_record["sha256"]),
        "requirements": requirements,
        "deployment_seal_sha256": str(
            prepared["deployment_seal_sha256"]
        ),
        "parameters_sha256": _content_sha256(
            _normalized_parameters(parameters)
        ),
        "output_uri": FILES_SERVER_URI,
        "queue": queue_name,
        "queue_id": queue_id,
    }


def _exact_named_tasks(task_class: object) -> list[object]:
    getter = getattr(task_class, "get_tasks", None)
    if not callable(getter):
        raise DeploymentError("ClearML Task class cannot perform duplicate lookup")
    value = getter(
        task_name=f"^{re.escape(CONTROLLER_NAME)}$",
        task_filter={"parent": PARENT_TASK_ID},
        allow_archived=True,
    )
    if value is None or isinstance(value, (str, bytes, bytearray)):
        raise DeploymentError("ClearML duplicate lookup returned an invalid inventory")
    try:
        candidates = list(value)
    except TypeError as error:
        raise DeploymentError("ClearML duplicate lookup is not iterable") from error
    exact: list[object] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_id = _task_id(
            getattr(candidate, "id", ""), "candidate duplicate lookup task"
        )
        if candidate_id in seen:
            raise DeploymentError("candidate duplicate lookup returned a duplicate ID")
        seen.add(candidate_id)
        task = _authoritative_task(
            task_class, candidate_id, context="candidate duplicate lookup task"
        )
        if str(getattr(task, "name", "") or "") != CONTROLLER_NAME:
            raise DeploymentError("candidate duplicate lookup task name drifted")
        if _task_parent(task) != PARENT_TASK_ID:
            raise DeploymentError("candidate duplicate lookup task parent drifted")
        _project_identity(task, context="candidate duplicate lookup task")
        exact.append(task)
    return exact


def _create_shell(task_class: object) -> object:
    creator = getattr(task_class, "create", None)
    if not callable(creator):
        raise DeploymentError("ClearML Task class cannot create a controller shell")
    task_types = getattr(task_class, "TaskTypes", None)
    task = creator(
        project_name=PROJECT_NAME,
        task_name=CONTROLLER_NAME,
        task_type=getattr(task_types, "controller", "controller"),
        script=None,
        packages=None,
        docker=None,
        docker_args=None,
        add_task_init_call=False,
        binary="python",
        detect_repository=False,
    )
    _task_id(getattr(task, "id", ""), "created candidate controller shell")
    if _normalized_status(task) != "created":
        raise DeploymentError("candidate controller shell was not created")
    return task


def _configure_shell(task: object, *, prepared: Mapping[str, object]) -> None:
    source = prepared.get("source")
    parameters = prepared.get("parameters")
    if type(source) is not str or not isinstance(parameters, Mapping):
        raise DeploymentError("prepared deployment material is invalid")
    task.set_script(
        repository="",
        branch="",
        commit="",
        diff=source,
        working_dir=".",
        entry_point=ENTRY_POINT,
    )
    task.set_packages(list(SERVICE_REQUIREMENTS))
    task.set_base_docker(
        docker_image=SERVICE_DOCKER_IMAGE,
        docker_arguments=SERVICE_DOCKER_ARGS,
    )
    task.set_parent(PARENT_TASK_ID)
    task.output_uri = FILES_SERVER_URI
    task.set_parameters(dict(parameters))
    tagger = getattr(task, "set_tags", None)
    if not callable(tagger) or tagger(sorted(TAGS)) is False:
        raise DeploymentError("candidate controller shell tags were not confirmed")
    flushed = task.flush(wait_for_uploads=True)
    if flushed is False:
        raise DeploymentError("candidate controller shell flush was not confirmed")


def _enqueue(
    task_class: object,
    task: object,
    *,
    task_id: str,
    prepared: Mapping[str, object],
    sleeper: Callable[[float], None],
) -> dict[str, object]:
    enqueuer = getattr(task_class, "enqueue", None)
    if not callable(enqueuer):
        raise DeploymentError("ClearML Task class cannot enqueue tasks")
    callback_error: Exception | None = None
    try:
        response = enqueuer(task=task, queue_name=SERVICES_QUEUE)
        if response is None or response is False:
            callback_error = DeploymentError(
                "ClearML did not confirm candidate controller enqueue"
            )
    except Exception as error:  # authority can still prove server-side acceptance
        callback_error = error
    last_error: Exception | None = callback_error
    for attempt in range(5):
        try:
            return _validate_task(
                task_class,
                task_id=task_id,
                prepared=prepared,
                expected_statuses=("queued", "in_progress", "completed"),
            )
        except Exception as error:
            last_error = error
        if attempt != 4:
            sleeper(1.0)
    raise DeploymentError(
        "candidate controller enqueue authority readback failed"
    ) from last_error


def _deploy(
    task_class: object,
    *,
    prepared: Mapping[str, object],
    sleeper: Callable[[float], None] = time.sleep,
    journal: dict[str, object] | None = None,
) -> dict[str, object]:
    journal = {} if journal is None else journal
    journal["parent"] = _validate_parent(task_class)
    existing = _exact_named_tasks(task_class)
    journal["duplicate_preflight_task_ids"] = [
        _task_id(getattr(task, "id", ""), "existing candidate controller")
        for task in existing
    ]
    if len(existing) > 1:
        raise DeploymentError("multiple exact-name candidate controllers already exist")
    created_shell = False
    if existing:
        task = existing[0]
        task_id = _task_id(getattr(task, "id", ""), "existing candidate controller")
        authority = _validate_task(
            task_class,
            task_id=task_id,
            prepared=prepared,
            expected_statuses=("created", "queued", "in_progress", "completed"),
        )
        journal["existing_authority"] = authority
        if authority["status"] == "created":
            journal["enqueue_attempted_task_id"] = task_id
            queued = _enqueue(
                task_class,
                task,
                task_id=task_id,
                prepared=prepared,
                sleeper=sleeper,
            )
            disposition = "resumed_exact_created_shell"
        else:
            queued = authority
            disposition = "reused_exact_deployment"
    else:
        journal["create_attempted"] = True
        task = _create_shell(task_class)
        created_shell = True
        task_id = _task_id(getattr(task, "id", ""), "created candidate controller")
        journal["created_task_id"] = task_id
        _configure_shell(task, prepared=prepared)
        journal["configured_task_id"] = task_id
        if _read_source() != prepared.get("source"):
            raise DeploymentError("candidate controller source changed after rendering")
        post_create = _exact_named_tasks(task_class)
        post_ids = [
            _task_id(getattr(item, "id", ""), "post-create candidate controller")
            for item in post_create
        ]
        journal["duplicate_post_create_task_ids"] = post_ids
        if post_ids != [task_id]:
            raise DeploymentError(
                "candidate controller post-create duplicate guard did not find exactly "
                "the created task"
            )
        authority = _validate_task(
            task_class,
            task_id=task_id,
            prepared=prepared,
            expected_statuses=("created",),
        )
        journal["pre_enqueue_authority"] = authority
        journal["enqueue_attempted_task_id"] = task_id
        queued = _enqueue(
            task_class,
            task,
            task_id=task_id,
            prepared=prepared,
            sleeper=sleeper,
        )
        disposition = "created_and_enqueued"
    journal["queued_authority"] = queued
    return {
        "task_id": task_id,
        "created_shell": created_shell,
        "disposition": disposition,
        "parent": journal["parent"],
        "deployment_seal_sha256": prepared["deployment_seal_sha256"],
        "queued_authority": queued,
    }


def _fixed_bindings(prepared: Mapping[str, object]) -> dict[str, object]:
    return {
        "project": PROJECT_NAME,
        "project_id": PROJECT_ID,
        "task_name": CONTROLLER_NAME,
        "parent_task_id": PARENT_TASK_ID,
        "entry_point": ENTRY_POINT,
        "source_sha256": prepared["source_record"]["sha256"],
        "deployment_seal_sha256": prepared["deployment_seal_sha256"],
        "fileserver_output_uri": FILES_SERVER_URI,
        "services_queue": SERVICES_QUEUE,
        "services_queue_id": SERVICES_QUEUE_ID,
        "service_docker_image": SERVICE_DOCKER_IMAGE,
        "service_docker_args": SERVICE_DOCKER_ARGS,
        "service_requirements": list(SERVICE_REQUIREMENTS),
        "runtime_parameters": _normalized_parameters(RUNTIME_PARAMETERS),
    }


def _base_receipt(
    *,
    prepared: Mapping[str, object],
    mode: str,
    status: str,
    remote_state_changed: bool,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "receipt_type": (
            "resilient_v2x_formal_candidate_evaluation_queue_deployment"
        ),
        "generated_at_utc": _utc_timestamp(),
        "mode": mode,
        "status": status,
        "remote_state_changed": remote_state_changed,
        "fixed_bindings": _fixed_bindings(prepared),
        "source": dict(prepared["source_record"]),
        "standalone_import_smoke": dict(prepared["standalone_import_smoke"]),
        "deployment_contract": dict(prepared["deployment_contract"]),
    }


def _dry_run_receipt() -> dict[str, object]:
    prepared = _prepare()
    receipt = _base_receipt(
        prepared=prepared,
        mode="dry_run",
        status="planned",
        remote_state_changed=False,
    )
    receipt.update(
        {
            "parameters": _normalized_parameters(prepared["parameters"]),
            "execution_contract": {
                "dry_run_performs_remote_calls": False,
                "remote_mutation_requires_execute_and_exact_token": True,
                "controller_shell_count": 1,
                "exact_name_source_and_deployment_seal_duplicate_guard": True,
                "existing_exact_created_shell_is_resumable": True,
                "existing_exact_queued_or_completed_task_is_reused": True,
                "same_name_contract_drift_fails_closed": True,
                "remote_controller_execute_remotely": False,
            },
        }
    )
    return _seal(receipt)


def _execute_receipt(task_class: object) -> dict[str, object]:
    prepared = _prepare()
    journal: dict[str, object] = {}
    try:
        result = _deploy(task_class, prepared=prepared, journal=journal)
    except Exception as error:
        changed = bool(
            journal.get("create_attempted")
            or journal.get("enqueue_attempted_task_id")
        )
        receipt = _base_receipt(
            prepared=prepared,
            mode="execute",
            status="failed_closed",
            remote_state_changed=changed,
        )
        receipt.update(
            {
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "partial_journal": journal,
            }
        )
        failure = DeploymentError(str(error))
        setattr(failure, "sealed_receipt", _seal(receipt))
        raise failure from error
    receipt = _base_receipt(
        prepared=prepared,
        mode="execute",
        status="deployed",
        remote_state_changed=result["disposition"] != "reused_exact_deployment",
    )
    receipt.update(result)
    return _seal(receipt)


def _verify_receipt(task_class: object, *, task_id: str) -> dict[str, object]:
    prepared = _prepare()
    parent = _validate_parent(task_class)
    authority = _validate_task(
        task_class,
        task_id=_task_id(task_id, "verification task"),
        prepared=prepared,
        expected_statuses=("created", "queued", "in_progress", "completed"),
    )
    receipt = _base_receipt(
        prepared=prepared,
        mode="verify",
        status="verified",
        remote_state_changed=False,
    )
    receipt.update({"parent": parent, "task_authority": authority})
    return _seal(receipt)


def _default_receipt_path(*, mode: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return ARTIFACT_DIR / f"{mode}-receipt-{timestamp}.json"


def _write_new_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            destination.unlink(missing_ok=True)
        finally:
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly permit remote shell creation or enqueue recovery.",
    )
    parser.add_argument(
        "--execute-token",
        default="",
        help="Exact deployment permission token; valid only with --execute.",
    )
    parser.add_argument(
        "--verify-task-id",
        default=None,
        help="Read-only authoritative verification of one deployed task ID.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Write-once sealed receipt path (defaults under artifacts/).",
    )
    parser.add_argument(
        "--evaluation-recovery-receipt",
        type=Path,
        default=None,
        help=(
            "Sealed exact-ID 26+2 recovery receipt. Required before --execute; "
            "optional in local-only plan mode for preflight validation."
        ),
    )
    parser.add_argument(
        "--controller-recovery-receipt",
        type=Path,
        default=None,
        help=(
            "Sealed exact-ID b6f0 controller recovery receipt. Required before "
            "--execute; optional in local-only plan mode for preflight validation."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.execute and args.verify_task_id is not None:
        parser.error("--execute and --verify-task-id are mutually exclusive")
    if not args.execute and args.execute_token:
        parser.error("--execute-token requires --execute")
    if args.execute and args.execute_token != DEPLOY_EXECUTE_TOKEN:
        raise DeploymentError(
            f"exact deployment token required: {DEPLOY_EXECUTE_TOKEN}"
        )
    if args.execute and args.evaluation_recovery_receipt is None:
        raise DeploymentError(
            "--evaluation-recovery-receipt is required with --execute"
        )
    if args.execute and args.controller_recovery_receipt is None:
        raise DeploymentError(
            "--controller-recovery-receipt is required with --execute"
        )
    mode = "execute" if args.execute else "verify" if args.verify_task_id else "plan"
    receipt_path = args.receipt or _default_receipt_path(mode=mode)
    recovery_binding = (
        _validate_evaluation_recovery_receipt(args.evaluation_recovery_receipt)
        if args.evaluation_recovery_receipt is not None
        else None
    )
    controller_recovery_binding = (
        _validate_controller_recovery_receipt(args.controller_recovery_receipt)
        if args.controller_recovery_receipt is not None
        else None
    )
    if recovery_binding is not None and controller_recovery_binding is not None:
        _validate_recovery_cross_binding(
            recovery_binding, controller_recovery_binding
        )
    if (
        args.evaluation_recovery_receipt is not None
        and receipt_path.resolve() == args.evaluation_recovery_receipt.resolve()
    ):
        raise DeploymentError("deployment receipt must not overwrite recovery evidence")
    if (
        args.controller_recovery_receipt is not None
        and receipt_path.resolve() == args.controller_recovery_receipt.resolve()
    ):
        raise DeploymentError(
            "deployment receipt must not overwrite controller recovery evidence"
        )
    if (
        args.evaluation_recovery_receipt is not None
        and args.controller_recovery_receipt is not None
        and args.evaluation_recovery_receipt.resolve()
        == args.controller_recovery_receipt.resolve()
    ):
        raise DeploymentError("evaluation and controller recovery receipts must differ")
    if not args.execute and args.verify_task_id is None:
        receipt = _dry_run_receipt()
        if recovery_binding is not None:
            receipt = _bind_evaluation_recovery(receipt, recovery_binding)
        if controller_recovery_binding is not None:
            receipt = _bind_controller_recovery(
                receipt, controller_recovery_binding
            )
        _write_new_receipt(receipt_path, receipt)
        print(json.dumps({"receipt": str(receipt_path), **receipt}, sort_keys=True))
        return 0
    try:
        from clearml import Task
    except ImportError as error:  # pragma: no cover - deployment environment only
        raise DeploymentError("remote mode requires the ClearML client") from error
    if args.verify_task_id is not None:
        receipt = _verify_receipt(Task, task_id=args.verify_task_id)
        if recovery_binding is not None:
            receipt = _bind_evaluation_recovery(receipt, recovery_binding)
        if controller_recovery_binding is not None:
            receipt = _bind_controller_recovery(
                receipt, controller_recovery_binding
            )
    else:
        try:
            receipt = _execute_receipt(Task)
        except DeploymentError as error:
            failed = getattr(error, "sealed_receipt", None)
            if isinstance(failed, Mapping):
                if recovery_binding is None:  # defensive: execute preflight must bind it
                    raise DeploymentError(
                        "evaluation recovery receipt binding is absent after preflight"
                    ) from error
                failed = _bind_evaluation_recovery(failed, recovery_binding)
                if controller_recovery_binding is None:
                    raise DeploymentError(
                        "controller recovery receipt binding is absent after preflight"
                    ) from error
                failed = _bind_controller_recovery(
                    failed, controller_recovery_binding
                )
                _write_new_receipt(receipt_path, failed)
            raise
        if recovery_binding is None:  # defensive: execute preflight must bind it
            raise DeploymentError(
                "evaluation recovery receipt binding is absent after preflight"
            )
        receipt = _bind_evaluation_recovery(receipt, recovery_binding)
        if controller_recovery_binding is None:
            raise DeploymentError(
                "controller recovery receipt binding is absent after preflight"
            )
        receipt = _bind_controller_recovery(receipt, controller_recovery_binding)
    _write_new_receipt(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DEPLOY_EXECUTE_TOKEN",
    "DeploymentError",
    "_deploy",
    "_dry_run_receipt",
    "_execute_receipt",
    "_prepare",
    "_verify_receipt",
    "main",
)
