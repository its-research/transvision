#!/usr/bin/env python3
"""Plan or explicitly deploy a sealed ClearML P->W->L->A->S successor chain.

The default mode is local-only.  ``--execute`` is the sole switch that permits
remote task creation and enqueueing.  Execution deliberately creates five empty
task shells first, then renders and installs the latest local producer sources
after all dynamic task IDs are known.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

try:
    from tools.resilient_v2x import (
        amend_clearml_formal_evaluation_plan as plan_amendment,
    )
    from tools.resilient_v2x import (
        recover_clearml_formal_evaluation_attempts as evaluation_recovery,
    )
except ModuleNotFoundError:  # direct execution from tools/resilient_v2x
    import amend_clearml_formal_evaluation_plan as plan_amendment
    import recover_clearml_formal_evaluation_attempts as evaluation_recovery


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts/resilient_v2x/formal-successor-chain"
PROJECT_NAME = "ResilientV2X/Training"
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
AMENDED_SUCCESSOR_EXECUTE_TOKEN = "DEPLOY_AMENDED_FORMAL_W_L_A_S"
ORIGINAL_QUEUE_SUCCESSOR_EXECUTE_TOKEN = "DEPLOY_ORIGINAL_QUEUE_FORMAL_W_L_A_S"
CANDIDATE_CONTROLLER_TASK_ID = "b6f0fbab32a5478183a45b3ca833fc01"

TRAINING_CONTROLLER_TASK_ID = "1011e98e10f64c428880af1d4b1d542b"
EVALUATION_TEMPLATE_TASK_ID = "8b77a3674dfe405388aae39ef82d06ef"
EVALUATION_SCRIPT_SHA256 = (
    "5d9584f1212bc021068e2a948bdf151aed97240c9994d5b994249316692f5066"
)
SOURCE_DATASET_ID = "4f7fac0078a4419a907fec6ff9e306c8"
SOURCE_ARCHIVE_SHA256 = (
    "655f33d9b684c26ff4ff577d5c5f3175d4fc4eac40426eb55820e7a8aa8bc40d"
)
TRAINING_DATASET_ID = "7c59fabb9da949e6b3c94c732f000975"
TRAINING_PREDECESSOR_TASK_ID = "f041d43e48c14ba4a4562281860d13f6"
CONTROLLER_ENTRY_POINT = "clearml_5090_training_controller.py"
CANDIDATE_RELEASE_BLOCKERS = (
    {
        "label": "E1",
        "training_task_id": "f0c3082f3aa34a81805903e0ffdc8610",
        "task_name": (
            "ResilientV2X fastlane E1 support_residual_linear [d6df8cc0ce59]"
        ),
        "subject": "support_residual_linear",
    },
    {
        "label": "E2",
        "training_task_id": "969c8fce6d24446299561772b3955274",
        "task_name": ("ResilientV2X fastlane E2 no_reliability_linear [d6df8cc0ce59]"),
        "subject": "no_reliability_linear",
    },
    {
        "label": "E3",
        "training_task_id": "dc037315c0684c3d854a2fd7c19a2a2f",
        "task_name": (
            "ResilientV2X fastlane E3 support_residual_no_reliability [d6df8cc0ce59]"
        ),
        "subject": "support_residual_no_reliability",
    },
    {
        "label": "P0",
        "training_task_id": "8883c51ced4f4951a45edbaefe6342d4",
        "task_name": "ResilientV2X round2 P0 [ebd8421be6fc]",
        "subject": "support_residual_no_reliability_linear",
    },
    {
        "label": "P2",
        "training_task_id": "f5d3820b4cdf416183c8f1fee566abe3",
        "task_name": "ResilientV2X round2 P2 bbox2.5 [4ced839db7fb]",
        "subject": "support_residual_no_reliability_linear_bbox25",
    },
)
CANDIDATE_RELEASE_BLOCKERS_JSON = json.dumps(
    CANDIDATE_RELEASE_BLOCKERS,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
)
AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID = "dbc05b28fbd044bf89edc3872742843a"
AUTHORITATIVE_EVALUATION_PLAN_ARTIFACT = "formal_1337_evaluation_plan"
AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256 = (
    "c5d14aa8021d06609c7a7e9a5401f4b5a0a417601fa8c95163f083eeb397e0f3"
)

COMPLETED_PROVENANCE_TASK_ID = "7e244a711751469b8cfdb25d77b05269"
COMPLETED_PROVENANCE_SOURCE_SHA256 = (
    "0bcc0b9ce383e5f2cfa35e87959caff9e388a988423e74797c7127a560eb007a"
)
COMPLETED_PROVENANCE_REQUIREMENTS_SHA256 = (
    "8940ef20ddc5401add52c2c8e7bf2b66146f58966b7442530a0d1409f7e3efd8"
)
COMPLETED_PROVENANCE_TAGS = frozenset(
    {
        "DAIR-CAUSAL-1337-v1",
        "ResilientV2X-suite",
        "cpu-controller",
        "formal-training-provenance-equivalence",
    }
)
COMPLETED_PROVENANCE_ARTIFACT = {
    "key": "formal_1337_training_provenance_equivalence",
    "type": "dict",
    "mode": "output",
    "uri": (
        "http://10.100.34.118:8081/ResilientV2X/Training/"
        "ResilientV2X%20formal%20training%20provenance%20successor."
        "7e244a711751469b8cfdb25d77b05269/artifacts/"
        "formal_1337_training_provenance_equivalence/"
        "formal_1337_training_provenance_equivalence.json"
    ),
    "hash": "06257a7d1bead5399ad7dc39772b270b0d70af411c73e4f21357158d378f2fee",
    "content_size": 139308,
    "timestamp": 1786508068,
    "display_data": [],
    "type_data_sha256": (
        "c5142c930f8320c565fc983a9a8bad89db6278cc2b6ed746fae5e9f0e003cb86"
    ),
    "type_data_content_type": "application/json",
    "type_data_preview_bytes": 65598,
}

ROLE_ORDER = ("P", "W", "L", "A", "S")
SUCCESSOR_ROLE_ORDER = ("W", "L", "A", "S")
ROLE_DEFINITIONS = {
    "P": {
        "entry_point": "clearml_formal_training_provenance.py",
        "task_name": "ResilientV2X formal training provenance successor",
    },
    "W": {
        "entry_point": "clearml_1337_dependency_watcher.py",
        "task_name": "ResilientV2X formal 1337 dependency watcher successor",
    },
    "L": {
        "entry_point": "clearml_1337_leaderboard.py",
        "task_name": "ResilientV2X formal 1337 leaderboard successor",
    },
    "A": {
        "entry_point": "clearml_formal_comparability_audit.py",
        "task_name": "ResilientV2X formal comparability audit successor",
    },
    "S": {
        "entry_point": "clearml_formal_candidate_selector.py",
        "task_name": "ResilientV2X formal candidate selector successor",
    },
}
ROLE_OUTPUT_ARTIFACTS = {
    "P": ("formal_1337_training_provenance_equivalence",),
    "W": ("formal_1337_evaluation_plan",),
    "L": ("formal_1337_leaderboard",),
    "A": ("formal_1337_comparability_audit",),
    "S": ("final_selector_formal_inputs", "formal_candidate_selection"),
}
SOURCE_PATHS = {
    role: ROOT / "tools/resilient_v2x" / str(definition["entry_point"])
    for role, definition in ROLE_DEFINITIONS.items()
}
LOCAL_MODULE_DIR = ROOT / "tools/resilient_v2x"
# S is expected to migrate between these pure policy helpers.  Embedding both
# keeps the deployed selector self-contained across that transition, while the
# recursive import scanner below covers future local producer dependencies.
REQUIRED_EMBEDDED_MODULES = {
    "P": (),
    "W": (),
    "L": (),
    "A": (),
    "S": ("tools.resilient_v2x.sota_gate", "tools.resilient_v2x.sota_selector"),
}
DRY_RUN_IDS = {role: f"{index:032x}" for index, role in enumerate(ROLE_ORDER, start=1)}
SUCCESSOR_DRY_RUN_IDS = {
    "P": COMPLETED_PROVENANCE_TASK_ID,
    **{
        role: f"{index:032x}"
        for index, role in enumerate(SUCCESSOR_ROLE_ORDER, start=0x201)
    },
}
SHA256_NAMES = frozenset(
    {
        "CONTROLLER_SCRIPT_SHA256",
        "WATCHER_SCRIPT_SHA256",
        "LEADERBOARD_SCRIPT_SHA256",
        "AUDIT_SCRIPT_SHA256",
        "TRAINING_PROVENANCE_SCRIPT_SHA256",
    }
)


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


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


def _validate_plan_amendment_receipt(path: Path) -> dict[str, object]:
    try:
        return plan_amendment.validate_amendment_receipt(path)
    except plan_amendment.AmendmentError as error:
        raise DeploymentError(f"plan amendment receipt is invalid: {error}") from error


def _validate_runtime_recovery_receipt(path: Path) -> dict[str, object]:
    try:
        from tools.resilient_v2x import (
            recover_clearml_exact_eval_chain_runtime_failure as runtime_recovery,
        )
    except ModuleNotFoundError:  # direct execution from tools/resilient_v2x
        import recover_clearml_exact_eval_chain_runtime_failure as runtime_recovery

    try:
        return runtime_recovery.validate_runtime_recovery_receipt(path)
    except runtime_recovery.RuntimeRecoveryError as error:
        raise DeploymentError(f"runtime recovery receipt is invalid: {error}") from error


def _validate_original_queue_runtime_recovery_receipt(
    path: Path,
) -> dict[str, object]:
    try:
        from tools.resilient_v2x import (
            recover_clearml_original_queue_runtime_failure as original_recovery,
        )
    except ModuleNotFoundError:  # direct execution from tools/resilient_v2x
        import recover_clearml_original_queue_runtime_failure as original_recovery

    try:
        return original_recovery.validate_ffnet_recovery_receipt(path)
    except original_recovery.OriginalQueueRecoveryError as error:
        raise DeploymentError(
            f"original-queue runtime recovery receipt is invalid: {error}"
        ) from error


def _bind_original_queue_runtime_recovery(
    receipt: Mapping[str, object], binding: Mapping[str, object]
) -> dict[str, object]:
    value = dict(receipt)
    value.pop("seal_sha256", None)
    value["original_queue_runtime_recovery"] = dict(binding)
    return _seal(value)


def _bind_runtime_recovery(
    receipt: Mapping[str, object], binding: Mapping[str, object]
) -> dict[str, object]:
    value = dict(receipt)
    value.pop("seal_sha256", None)
    value["exact_eval_chain_runtime_recovery"] = dict(binding)
    return _seal(value)


def _bind_plan_amendment(
    receipt: Mapping[str, object], binding: Mapping[str, object]
) -> dict[str, object]:
    value = dict(receipt)
    value.pop("seal_sha256", None)
    value["evaluation_plan_amendment_receipt"] = dict(binding)
    return _seal(value)


def _validate_amended_runtime_pair(
    amendment_binding: Mapping[str, object],
    runtime_recovery_binding: Mapping[str, object],
) -> None:
    if (
        runtime_recovery_binding.get("ffnet_created_unqueued") is not True
        or runtime_recovery_binding.get("candidate_services") is not True
        or runtime_recovery_binding.get("same_ids_only") is not True
        or runtime_recovery_binding.get("replacement_tasks_created") is not False
        or runtime_recovery_binding.get("coformer_mutated") is not False
        or runtime_recovery_binding.get("exact_evaluation_task_count") != 28
        or runtime_recovery_binding.get("target_task_ids")
        != {
            "FFNet": plan_amendment.FFNET_EVALUATION_TASK_ID,
            "C": CANDIDATE_CONTROLLER_TASK_ID,
        }
        or runtime_recovery_binding.get(
            "formal_plan_amendment_receipt_seal_sha256"
        )
        != amendment_binding["receipt_seal_sha256"]
        or runtime_recovery_binding.get("formal_plan_amendment_producer_task_id")
        != amendment_binding["producer_task_id"]
        or runtime_recovery_binding.get("revised_formal_plan_seal_sha256")
        != amendment_binding["revised_plan_seal_sha256"]
        or runtime_recovery_binding.get("exact_evaluation_task_ids_sha256")
        != amendment_binding["evaluation_task_ids_sha256"]
    ):
        raise DeploymentError("runtime recovery and plan amendment bindings drifted")


def _bind_evaluation_recovery(
    receipt: Mapping[str, object], binding: Mapping[str, object]
) -> dict[str, object]:
    value = dict(receipt)
    value.pop("seal_sha256", None)
    value["evaluation_recovery_receipt"] = dict(binding)
    return _seal(value)


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _task_id(value: object, context: str) -> str:
    result = str(value or "")
    if not _is_lower_hex(result, 32):
        raise DeploymentError(f"{context} must be a lowercase 32-hex ClearML ID")
    return result


def _sha256(value: object, context: str) -> str:
    result = str(value or "")
    if not _is_lower_hex(result, 64):
        raise DeploymentError(f"{context} must be a lowercase SHA-256")
    return result


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _default_receipt_path(*, execute: bool, recovery: bool = False) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    kind = (
        "recovery-deployment-receipt"
        if recovery
        else "deployment-receipt"
        if execute
        else "deployment-plan"
    )
    return ARTIFACT_DIR / f"{kind}-{timestamp}.json"


def _write_new_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o644)
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


def _read_source(path: Path, *, context: str) -> str:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DeploymentError(f"{context} source cannot be resolved") from error
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise DeploymentError(f"{context} source escapes the repository") from error
    if path.is_symlink() or not resolved.is_file():
        raise DeploymentError(f"{context} source must be a regular non-symlink file")
    try:
        source = resolved.read_text(encoding="utf-8")
        compile(source, str(resolved), "exec")
    except (OSError, UnicodeError, SyntaxError) as error:
        raise DeploymentError(f"{context} source is not valid UTF-8 Python") from error
    if not source:
        raise DeploymentError(f"{context} source is empty")
    return source


def _read_base_sources() -> dict[str, str]:
    return {
        role: _read_source(SOURCE_PATHS[role], context=f"role {role}")
        for role in ROLE_ORDER
    }


def _canonical_local_module_name(value: str) -> str:
    prefix = "tools.resilient_v2x."
    candidate = value if value.startswith(prefix) else f"{prefix}{value}"
    relative = candidate[len(prefix) :]
    if not relative or any(
        not component.isidentifier() for component in relative.split(".")
    ):
        raise DeploymentError(f"invalid repository-local module name {value!r}")
    return candidate


def _local_module_path(module_name: str) -> Path | None:
    canonical = _canonical_local_module_name(module_name)
    relative = canonical.removeprefix("tools.resilient_v2x.")
    path = LOCAL_MODULE_DIR.joinpath(*relative.split(".")).with_suffix(".py")
    try:
        resolved = path.resolve()
        resolved.relative_to(LOCAL_MODULE_DIR.resolve())
    except (OSError, ValueError) as error:
        raise DeploymentError(
            "local module path escapes tools/resilient_v2x"
        ) from error
    return path if path.is_file() and not path.is_symlink() else None


def _repository_local_imports(
    source: str, *, current_module: str | None = None
) -> tuple[str, ...]:
    """Return exact file-backed tools.resilient_v2x imports in one source."""

    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise DeploymentError("cannot scan imports in invalid Python source") from error
    observed: set[str] = set()

    def add_if_local(name: str) -> None:
        try:
            canonical = _canonical_local_module_name(name)
        except DeploymentError:
            return
        if _local_module_path(canonical) is not None:
            observed.add(canonical)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith("tools.resilient_v2x."):
                    parts = name.split(".")
                    for end in range(len(parts), 2, -1):
                        candidate = ".".join(parts[:end])
                        if _local_module_path(candidate) is not None:
                            observed.add(candidate)
                            break
                    else:
                        raise DeploymentError(
                            f"repository-local import {name!r} has no file-backed module"
                        )
                elif "." not in name:
                    add_if_local(name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                if current_module is None:
                    raise DeploymentError(
                        "relative repository-local import lacks a module context"
                    )
                package_parts = current_module.split(".")[: -node.level]
                if not package_parts:
                    raise DeploymentError("relative local import escapes its package")
                module = (
                    ".".join((*package_parts, module))
                    if module
                    else ".".join(package_parts)
                )
            if module == "tools.resilient_v2x":
                for alias in node.names:
                    if alias.name == "*":
                        raise DeploymentError(
                            "wildcard import from tools.resilient_v2x is not bundleable"
                        )
                    candidate = f"{module}.{alias.name}"
                    if _local_module_path(candidate) is None:
                        raise DeploymentError(
                            f"repository-local import {candidate!r} is not file-backed"
                        )
                    observed.add(candidate)
            elif module.startswith("tools.resilient_v2x."):
                parts = module.split(".")
                for end in range(len(parts), 2, -1):
                    candidate = ".".join(parts[:end])
                    if _local_module_path(candidate) is not None:
                        observed.add(candidate)
                        break
                else:
                    raise DeploymentError(
                        f"repository-local import {module!r} has no file-backed module"
                    )
            elif module and "." not in module:
                add_if_local(module)
    return tuple(sorted(observed))


def _embedded_module_sources(*, role: str, producer_source: str) -> dict[str, str]:
    if role not in ROLE_ORDER:
        raise DeploymentError(f"unknown producer role {role!r}")
    roots = set(REQUIRED_EMBEDDED_MODULES[role])
    roots.update(_repository_local_imports(producer_source))
    ordered: dict[str, str] = {}
    active: set[str] = set()

    def visit(raw_name: str) -> None:
        name = _canonical_local_module_name(raw_name)
        if name in ordered:
            return
        if name in active:
            raise DeploymentError(f"cyclic repository-local import at {name}")
        path = _local_module_path(name)
        if path is None:
            raise DeploymentError(f"embedded module {name!r} is not a regular file")
        active.add(name)
        source = _read_source(path, context=f"embedded module {name}")
        for dependency in _repository_local_imports(source, current_module=name):
            visit(dependency)
        active.remove(name)
        ordered[name] = source

    for name in sorted(roots):
        visit(name)
    return ordered


def _embedded_module_record(module_name: str, source: str) -> dict[str, object]:
    path = _local_module_path(module_name)
    if path is None:  # pragma: no cover - source collector owns this invariant
        raise DeploymentError(f"embedded module {module_name!r} disappeared")
    compressed = zlib.compress(source.encode("utf-8"), level=9)
    payload = base64.b85encode(compressed)
    return {
        "module_name": module_name,
        "path": str(path.relative_to(ROOT)),
        "base_bytes": len(source.encode("utf-8")),
        "base_sha256": _sha256_text(source),
        "compressed_bytes": len(compressed),
        "base85_payload_bytes": len(payload),
        "encoding": "utf8+zlib9+base85",
    }


def _bundle_prelude(modules: Mapping[str, str]) -> str:
    if not modules:
        return ""
    installs: list[str] = []
    for module_name, source in modules.items():
        payload = base64.b85encode(
            zlib.compress(source.encode("utf-8"), level=9)
        ).decode("ascii")
        installs.append(
            "_rv2x_install_embedded_module("
            f"{module_name!r}, {payload!r}, {_sha256_text(source)!r})"
        )
    return (
        "\n# BEGIN RESILIENTV2X SEALED STANDALONE MODULE BUNDLE\n"
        "import base64 as _rv2x_base64\n"
        "import hashlib as _rv2x_hashlib\n"
        "import sys as _rv2x_sys\n"
        "import types as _rv2x_types\n"
        "import zlib as _rv2x_zlib\n"
        "\n"
        "def _rv2x_ensure_embedded_package(_rv2x_name):\n"
        "    _rv2x_parent = None\n"
        '    _rv2x_full = ""\n'
        '    for _rv2x_part in _rv2x_name.split("."):\n'
        '        _rv2x_full = (_rv2x_full + "." + _rv2x_part).strip(".")\n'
        "        _rv2x_module = _rv2x_sys.modules.get(_rv2x_full)\n"
        "        if _rv2x_module is None:\n"
        "            _rv2x_module = _rv2x_types.ModuleType(_rv2x_full)\n"
        "            _rv2x_module.__package__ = _rv2x_full\n"
        "            _rv2x_module.__path__ = []\n"
        "            _rv2x_module.__resilient_v2x_embedded_package__ = True\n"
        "            _rv2x_sys.modules[_rv2x_full] = _rv2x_module\n"
        '        elif not getattr(_rv2x_module, "__resilient_v2x_embedded_package__", False):\n'
        '            raise RuntimeError("unsealed package collides with embedded bundle: " + _rv2x_full)\n'
        "        if _rv2x_parent is not None:\n"
        "            setattr(_rv2x_parent, _rv2x_part, _rv2x_module)\n"
        "        _rv2x_parent = _rv2x_module\n"
        "    return _rv2x_parent\n"
        "\n"
        "def _rv2x_install_embedded_module(_rv2x_name, _rv2x_payload, _rv2x_sha256):\n"
        '    _rv2x_package_name, _rv2x_short = _rv2x_name.rsplit(".", 1)\n'
        "    _rv2x_package = _rv2x_ensure_embedded_package(_rv2x_package_name)\n"
        "    _rv2x_existing = _rv2x_sys.modules.get(_rv2x_name)\n"
        "    if _rv2x_existing is not None:\n"
        '        if getattr(_rv2x_existing, "__resilient_v2x_embedded_sha256__", None) != _rv2x_sha256:\n'
        '            raise RuntimeError("unsealed module collides with embedded bundle: " + _rv2x_name)\n'
        "        return _rv2x_existing\n"
        '    _rv2x_source = _rv2x_zlib.decompress(_rv2x_base64.b85decode(_rv2x_payload.encode("ascii"))).decode("utf-8")\n'
        '    if _rv2x_hashlib.sha256(_rv2x_source.encode("utf-8")).hexdigest() != _rv2x_sha256:\n'
        '        raise RuntimeError("embedded module SHA-256 mismatch: " + _rv2x_name)\n'
        "    _rv2x_module = _rv2x_types.ModuleType(_rv2x_name)\n"
        '    _rv2x_module.__file__ = "<resilient-v2x-embedded:" + _rv2x_name + ">"\n'
        "    _rv2x_module.__package__ = _rv2x_package_name\n"
        "    _rv2x_module.__resilient_v2x_embedded_sha256__ = _rv2x_sha256\n"
        "    _rv2x_sys.modules[_rv2x_name] = _rv2x_module\n"
        "    _rv2x_sys.modules[_rv2x_short] = _rv2x_module\n"
        "    setattr(_rv2x_package, _rv2x_short, _rv2x_module)\n"
        '    exec(compile(_rv2x_source, _rv2x_module.__file__, "exec"), _rv2x_module.__dict__)\n'
        "    return _rv2x_module\n"
        "\n"
        + "\n".join(installs)
        + "\n# END RESILIENTV2X SEALED STANDALONE MODULE BUNDLE\n"
    )


def _inject_after_future_imports(source: str, prelude: str) -> str:
    if not prelude:
        return source
    tree = ast.parse(source)
    insertion_line = 0
    body = list(tree.body)
    index = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and type(body[0].value.value) is str
    ):
        insertion_line = int(body[0].end_lineno or body[0].lineno)
        index = 1
    while index < len(body):
        node = body[index]
        if not isinstance(node, ast.ImportFrom) or node.module != "__future__":
            break
        insertion_line = int(node.end_lineno or node.lineno)
        index += 1
    lines = source.splitlines(keepends=True)
    offset = sum(len(line) for line in lines[:insertion_line])
    bundled = source[:offset] + prelude + source[offset:]
    compile(bundled, "<standalone-bundled-producer>", "exec")
    return bundled


def _bundle_role_source(
    *, role: str, rendered_source: str
) -> tuple[str, dict[str, str], list[dict[str, object]]]:
    dependencies = _embedded_module_sources(role=role, producer_source=rendered_source)
    records = [
        _embedded_module_record(name, source) for name, source in dependencies.items()
    ]
    bundled = _inject_after_future_imports(
        rendered_source, _bundle_prelude(dependencies)
    )
    return bundled, dependencies, records


_ISOLATED_SMOKE_DRIVER = r"""
import hashlib
import importlib.util
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1]).resolve(strict=True)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
repository_root = pathlib.Path(manifest["repository_root"]).resolve(strict=True)
for entry in sys.path:
    if not entry:
        continue
    candidate = pathlib.Path(entry).resolve()
    if candidate == repository_root or repository_root in candidate.parents:
        raise RuntimeError("isolated smoke sys.path contains the repository")

results = {}
for item in manifest["roles"]:
    for name in item["purge_module_names"]:
        sys.modules.pop(name, None)
    path = pathlib.Path(item["path"]).resolve(strict=True)
    source = path.read_text(encoding="utf-8")
    observed_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if observed_sha != item["producer_sha256"]:
        raise RuntimeError("smoke producer source SHA-256 drifted")
    runtime_name = "_resilient_v2x_standalone_smoke_" + item["role"]
    spec = importlib.util.spec_from_file_location(runtime_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot build standalone smoke import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[runtime_name] = module
    spec.loader.exec_module(module)
    verified = []
    for embedded in item["embedded_modules"]:
        name = embedded["module_name"]
        dependency = sys.modules.get(name)
        if dependency is None:
            raise RuntimeError("embedded dependency was not installed: " + name)
        if getattr(dependency, "__file__", None) != (
            "<resilient-v2x-embedded:" + name + ">"
        ):
            raise RuntimeError("dependency came from outside the bundle: " + name)
        if getattr(
            dependency, "__resilient_v2x_embedded_sha256__", None
        ) != embedded["base_sha256"]:
            raise RuntimeError("embedded dependency identity drifted: " + name)
        verified.append(name)
    results[item["role"]] = {
        "producer_sha256": observed_sha,
        "embedded_modules_verified": verified,
    }
print("RESILIENT_V2X_STANDALONE_SMOKE=" + json.dumps(results, sort_keys=True))
"""


def _isolated_import_smoke(
    bundled_sources: Mapping[str, str],
    embedded_records: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    timeout_seconds: float = 120.0,
) -> dict[str, object]:
    if set(bundled_sources) != set(ROLE_ORDER) or set(embedded_records) != set(
        ROLE_ORDER
    ):
        raise DeploymentError("standalone smoke role inventory is incomplete")
    if (
        type(timeout_seconds) not in {int, float}
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise DeploymentError("standalone smoke timeout must be finite and positive")
    with tempfile.TemporaryDirectory(prefix="resilient-v2x-standalone-smoke-") as raw:
        directory = Path(raw)
        roles: list[dict[str, object]] = []
        all_module_names = sorted(
            {
                str(record["module_name"])
                for records in embedded_records.values()
                for record in records
            }
        )
        purge_names = [
            "tools",
            "tools.resilient_v2x",
            *(name.rsplit(".", 1)[-1] for name in all_module_names),
            *all_module_names,
        ]
        for role in ROLE_ORDER:
            path = directory / str(ROLE_DEFINITIONS[role]["entry_point"])
            path.write_text(bundled_sources[role], encoding="utf-8")
            roles.append(
                {
                    "role": role,
                    "path": str(path),
                    "producer_sha256": _sha256_text(bundled_sources[role]),
                    "embedded_modules": list(embedded_records[role]),
                    "purge_module_names": purge_names,
                }
            )
        manifest = {
            "repository_root": str(ROOT),
            "roles": roles,
        }
        manifest_path = directory / "smoke-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    _ISOLATED_SMOKE_DRIVER,
                    str(manifest_path),
                ],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise DeploymentError("standalone isolated import smoke failed") from error
    marker = "RESILIENT_V2X_STANDALONE_SMOKE="
    marked = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    if completed.returncode != 0 or len(marked) != 1:
        details = (completed.stderr or completed.stdout).strip()[-2000:]
        raise DeploymentError(f"standalone isolated import smoke failed: {details}")
    try:
        observed = json.loads(marked[0][len(marker) :])
    except (json.JSONDecodeError, TypeError) as error:
        raise DeploymentError("standalone smoke result is invalid JSON") from error
    if not isinstance(observed, Mapping) or set(observed) != set(ROLE_ORDER):
        raise DeploymentError("standalone smoke result role inventory drifted")
    expected = {
        role: {
            "producer_sha256": _sha256_text(bundled_sources[role]),
            "embedded_modules_verified": [
                str(record["module_name"]) for record in embedded_records[role]
            ],
        }
        for role in ROLE_ORDER
    }
    if _canonical_json(observed) != _canonical_json(expected):
        raise DeploymentError("standalone smoke result identity drifted")
    return {
        "result": "pass",
        "isolation": "python_-I_without_repository_on_sys_path",
        "interpreter": str(Path(sys.executable).resolve()),
        "roles": expected,
    }


def _assignment_value(source: str, name: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:  # pragma: no cover - guarded source loading
        raise DeploymentError("producer source is not valid Python") from error
    assignments = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            isinstance(node.target, ast.Name)
            if isinstance(node, ast.AnnAssign)
            else len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        )
        and (node.target.id if isinstance(node, ast.AnnAssign) else node.targets[0].id)
        == name
    ]
    if len(assignments) != 1:
        raise DeploymentError(f"source must assign {name} exactly once at module scope")
    value = assignments[0].value
    if not isinstance(value, ast.Constant) or type(value.value) is not str:
        raise DeploymentError(f"source assignment {name} must be a string literal")
    return value.value


def _replace_string_assignment(source: str, name: str, value: str) -> str:
    if name in SHA256_NAMES:
        _sha256(value, name)
    elif name.endswith("TASK_ID"):
        _task_id(value, name)
    tree = ast.parse(source)
    assignments = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            isinstance(node.target, ast.Name)
            if isinstance(node, ast.AnnAssign)
            else len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        )
        and (node.target.id if isinstance(node, ast.AnnAssign) else node.targets[0].id)
        == name
    ]
    if len(assignments) != 1:
        raise DeploymentError(f"source must assign {name} exactly once at module scope")
    expression = assignments[0].value
    if (
        not isinstance(expression, ast.Constant)
        or type(expression.value) is not str
        or expression.end_lineno is None
        or expression.end_col_offset is None
    ):
        raise DeploymentError(f"source assignment {name} must be a string literal")
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    start = offsets[expression.lineno - 1] + expression.col_offset
    end = offsets[expression.end_lineno - 1] + expression.end_col_offset
    rendered = source[:start] + repr(value) + source[end:]
    if _assignment_value(rendered, name) != value:
        raise DeploymentError(f"failed to render exact assignment {name}")
    compile(rendered, f"<{name}-rendered>", "exec")
    return rendered


def _render_sources_with_bundle_evidence(
    base_sources: Mapping[str, str], task_ids: Mapping[str, str]
) -> tuple[
    dict[str, str],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, list[dict[str, object]]],
]:
    if set(base_sources) != set(ROLE_ORDER) or set(task_ids) != set(ROLE_ORDER):
        raise DeploymentError("source or task-ID role inventory is incomplete")
    normalized_ids = {
        role: _task_id(task_ids[role], f"role {role} task") for role in ROLE_ORDER
    }
    if len(set(normalized_ids.values())) != len(ROLE_ORDER):
        raise DeploymentError("successor task IDs must be unique")
    if TRAINING_CONTROLLER_TASK_ID in normalized_ids.values():
        raise DeploymentError("successor task aliases the training controller")

    rendered = {role: str(base_sources[role]) for role in ROLE_ORDER}
    replacements: dict[str, dict[str, str]] = {role: {} for role in ROLE_ORDER}
    embedded_sources: dict[str, dict[str, str]] = {}
    embedded_records: dict[str, list[dict[str, object]]] = {}

    for role in ("P", "W", "L"):
        (
            rendered[role],
            embedded_sources[role],
            embedded_records[role],
        ) = _bundle_role_source(role=role, rendered_source=rendered[role])
    p_sha = _sha256_text(rendered["P"])
    w_sha = _sha256_text(rendered["W"])
    l_sha = _sha256_text(rendered["L"])
    a_values = {
        "CONTROLLER_SCRIPT_SHA256": _sha256_text(
            _read_source(
                ROOT / "tools/resilient_v2x" / CONTROLLER_ENTRY_POINT,
                context="training controller",
            )
        ),
        "TRAINING_PROVENANCE_SCRIPT_SHA256": p_sha,
        "WATCHER_SCRIPT_SHA256": w_sha,
        "LEADERBOARD_SCRIPT_SHA256": l_sha,
    }
    for name, value in a_values.items():
        rendered["A"] = _replace_string_assignment(rendered["A"], name, value)
        replacements["A"][name] = value
    (
        rendered["A"],
        embedded_sources["A"],
        embedded_records["A"],
    ) = _bundle_role_source(role="A", rendered_source=rendered["A"])
    a_sha = _sha256_text(rendered["A"])

    s_values = {
        "DEFAULT_AUDIT_TASK_ID": normalized_ids["A"],
        "DEFAULT_LEADERBOARD_TASK_ID": normalized_ids["L"],
        "TRAINING_CONTROLLER_TASK_ID": TRAINING_CONTROLLER_TASK_ID,
        "TRAINING_PROVENANCE_TASK_ID": normalized_ids["P"],
        "WATCHER_TASK_ID": normalized_ids["W"],
        "WATCHER_SCRIPT_SHA256": w_sha,
        "LEADERBOARD_SCRIPT_SHA256": l_sha,
        "AUDIT_SCRIPT_SHA256": a_sha,
    }
    for name, value in s_values.items():
        rendered["S"] = _replace_string_assignment(rendered["S"], name, value)
        replacements["S"][name] = value
    (
        rendered["S"],
        embedded_sources["S"],
        embedded_records["S"],
    ) = _bundle_role_source(role="S", rendered_source=rendered["S"])

    for role in ROLE_ORDER:
        compile(rendered[role], str(SOURCE_PATHS[role]), "exec")
    return rendered, replacements, embedded_sources, embedded_records


def _render_sources(
    base_sources: Mapping[str, str], task_ids: Mapping[str, str]
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    rendered, replacements, _, _ = _render_sources_with_bundle_evidence(
        base_sources, task_ids
    )
    return rendered, replacements


def _source_records(
    base_sources: Mapping[str, str],
    rendered_sources: Mapping[str, str],
    replacements: Mapping[str, Mapping[str, str]],
    embedded_records: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, dict[str, object]]:
    if (
        set(base_sources) != set(ROLE_ORDER)
        or set(rendered_sources) != set(ROLE_ORDER)
        or set(replacements) != set(ROLE_ORDER)
        or set(embedded_records) != set(ROLE_ORDER)
    ):
        raise DeploymentError("source record role inventory is incomplete")
    return {
        role: {
            "path": str(SOURCE_PATHS[role].relative_to(ROOT)),
            "entry_point": ROLE_DEFINITIONS[role]["entry_point"],
            "base_bytes": len(base_sources[role].encode("utf-8")),
            "base_sha256": _sha256_text(base_sources[role]),
            "deployed_bytes": len(rendered_sources[role].encode("utf-8")),
            "deployed_sha256": _sha256_text(rendered_sources[role]),
            "constant_replacements": dict(replacements[role]),
            "output_artifacts": list(ROLE_OUTPUT_ARTIFACTS[role]),
            "standalone_bundle": {
                "embedded_module_count": len(embedded_records[role]),
                "embedded_modules": [dict(item) for item in embedded_records[role]],
            },
        }
        for role in ROLE_ORDER
    }


def _parent_ids(task_ids: Mapping[str, str]) -> dict[str, str]:
    return {
        "P": TRAINING_CONTROLLER_TASK_ID,
        "W": task_ids["P"],
        "L": task_ids["W"],
        "A": task_ids["L"],
        "S": task_ids["A"],
    }


def _parameters(
    task_ids: Mapping[str, str],
    source_sha256: Mapping[str, str],
    *,
    authoritative_recovery: bool = False,
    amendment_binding: Mapping[str, object] | None = None,
    runtime_recovery_binding: Mapping[str, object] | None = None,
    original_queue_recovery_binding: Mapping[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    if original_queue_recovery_binding is not None and (
        amendment_binding is not None or runtime_recovery_binding is not None
    ):
        raise DeploymentError(
            "original-queue recovery is mutually exclusive with amendment recovery"
        )
    result = {
        "P": {
            "Args/training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
            "Args/poll_seconds": 60.0,
            "Args/timeout_hours": 720.0,
        },
        "W": {
            "Args/training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
            "Args/evaluation_template_task_id": EVALUATION_TEMPLATE_TASK_ID,
            "Args/training_provenance_task_id": task_ids["P"],
            "Args/expected_training_provenance_script_sha256": source_sha256["P"],
            "Args/evaluation_worker_queues": (
                "GPU4-A100,GPU4-A100,GPU4-V100,GPU4-5090"
            ),
            "Args/candidate_release_blockers_json": (CANDIDATE_RELEASE_BLOCKERS_JSON),
            "Args/expected_eval_script_sha256": EVALUATION_SCRIPT_SHA256,
            "Args/expected_source_dataset_id": SOURCE_DATASET_ID,
            "Args/expected_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "Args/expected_training_dataset_id": TRAINING_DATASET_ID,
            "Args/expected_predecessor_task_id": TRAINING_PREDECESSOR_TASK_ID,
            "Args/poll_seconds": 60.0,
            "Args/timeout_hours": 720.0,
        },
        "L": {
            "Args/watcher_task_id": task_ids["W"],
            "Args/training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
            "Args/poll_seconds": 60.0,
            "Args/timeout_hours": 720.0,
        },
        "A": {
            "Args/training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
            "Args/training_provenance_task_id": task_ids["P"],
            "Args/watcher_task_id": task_ids["W"],
            "Args/leaderboard_task_id": task_ids["L"],
            "Args/poll_seconds": 60.0,
            "Args/timeout_hours": 720.0,
        },
        "S": {
            "Args/audit_task_id": task_ids["A"],
            "Args/leaderboard_task_id": task_ids["L"],
            "Args/poll_seconds": 60.0,
            "Args/timeout_hours": 720.0,
        },
    }
    if authoritative_recovery:
        plan_producer_task_id = AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID
        plan_seal_sha256 = AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256
        if amendment_binding is not None:
            plan_producer_task_id = str(amendment_binding["producer_task_id"])
            plan_seal_sha256 = str(amendment_binding["revised_plan_seal_sha256"])
        result["W"].update(
            {
                "Args/authoritative_evaluation_plan_producer_task_id": (
                    plan_producer_task_id
                ),
                "Args/authoritative_evaluation_plan_seal_sha256": (
                    plan_seal_sha256
                ),
            }
        )
    if amendment_binding is not None:
        if not authoritative_recovery:
            raise DeploymentError(
                "plan amendment requires authoritative exact-ID recovery mode"
            )
        common = {
            "Args/evaluation_plan_amendment_producer_task_id": (
                amendment_binding["producer_task_id"]
            ),
            "Args/evaluation_plan_amendment_receipt_seal_sha256": (
                amendment_binding["receipt_seal_sha256"]
            ),
            "Args/evaluation_plan_amendment_revised_plan_seal_sha256": (
                amendment_binding["revised_plan_seal_sha256"]
            ),
            "Args/evaluation_plan_amendment_evidence_seal_sha256": (
                amendment_binding["amendment_evidence_seal_sha256"]
            ),
            "Args/evaluation_plan_amendment_worker_evidence_seal_sha256": (
                amendment_binding["worker_stats_evidence_seal_sha256"]
            ),
            "Args/evaluation_plan_amendment_task_ids_sha256": (
                amendment_binding["evaluation_task_ids_sha256"]
            ),
        }
        for role in SUCCESSOR_ROLE_ORDER:
            result[role].update(common)
        result["W"].update(
            {
                "Args/authoritative_evaluation_plan_amendment_receipt_seal_sha256": (
                    amendment_binding["receipt_seal_sha256"]
                ),
                "Args/authoritative_evaluation_plan_producer_source_sha256": (
                    amendment_binding["producer_source_sha256"]
                ),
                "Args/authoritative_evaluation_plan_amendment_evidence_seal_sha256": (
                    amendment_binding["amendment_evidence_seal_sha256"]
                ),
                "Args/authoritative_evaluation_plan_worker_evidence_seal_sha256": (
                    amendment_binding["worker_stats_evidence_seal_sha256"]
                ),
                "Args/authoritative_evaluation_plan_task_ids_sha256": (
                    amendment_binding["evaluation_task_ids_sha256"]
                ),
            }
        )
    if runtime_recovery_binding is not None:
        if amendment_binding is None:
            raise DeploymentError(
                "runtime recovery requires the same plan amendment binding"
            )
        if (
            runtime_recovery_binding[
                "formal_plan_amendment_receipt_seal_sha256"
            ]
            != amendment_binding["receipt_seal_sha256"]
            or runtime_recovery_binding["formal_plan_amendment_producer_task_id"]
            != amendment_binding["producer_task_id"]
            or runtime_recovery_binding["revised_formal_plan_seal_sha256"]
            != amendment_binding["revised_plan_seal_sha256"]
            or runtime_recovery_binding["exact_evaluation_task_ids_sha256"]
            != amendment_binding["evaluation_task_ids_sha256"]
        ):
            raise DeploymentError("runtime recovery amendment authority drifted")
        ffnet = runtime_recovery_binding["ffnet_authority"]
        candidate = runtime_recovery_binding["candidate_authority"]
        common = {
            "Args/exact_eval_runtime_recovery_receipt_seal_sha256": (
                runtime_recovery_binding["receipt_seal_sha256"]
            ),
            "Args/exact_eval_runtime_recovery_attempt_seal_sha256": (
                runtime_recovery_binding["attempt_receipt_seal_sha256"]
            ),
            "Args/exact_eval_runtime_ffnet_source_sha256": ffnet[
                "source_sha256"
            ],
            "Args/exact_eval_runtime_ffnet_parameters_sha256": ffnet[
                "parameters_sha256"
            ],
            "Args/exact_eval_runtime_candidate_source_sha256": candidate[
                "source_sha256"
            ],
            "Args/exact_eval_runtime_candidate_parameters_sha256": candidate[
                "parameters_sha256"
            ],
        }
        for role in SUCCESSOR_ROLE_ORDER:
            result[role].update(common)
    if original_queue_recovery_binding is not None:
        if not authoritative_recovery:
            raise DeploymentError(
                "original-queue recovery requires authoritative exact-ID mode"
            )
        if (
            original_queue_recovery_binding.get("ffnet_task_id")
            != "144397bfa9c242bc9a92a1279922558b"
            or original_queue_recovery_binding.get("ffnet_planned_queue")
            != "GPU4-V100"
            or original_queue_recovery_binding.get("ffnet_created_unqueued")
            is not True
            or original_queue_recovery_binding.get("coformer_task_id")
            != "d8fac86325e047e5aa25f5ce899902b6"
            or original_queue_recovery_binding.get("coformer_completed") is not True
            or original_queue_recovery_binding.get("same_ids_only") is not True
            or original_queue_recovery_binding.get("replacement_tasks_created")
            is not False
        ):
            raise DeploymentError("original-queue recovery binding drifted")
        common = {
            "Args/original_queue_runtime_recovery_receipt_path": (
                original_queue_recovery_binding["path"]
            ),
            "Args/original_queue_runtime_recovery_receipt_seal_sha256": (
                original_queue_recovery_binding["receipt_seal_sha256"]
            ),
            "Args/original_queue_runtime_recovery_attempt_seal_sha256": (
                original_queue_recovery_binding["attempt_receipt_seal_sha256"]
            ),
            "Args/original_queue_runtime_ffnet_source_sha256": (
                original_queue_recovery_binding["ffnet_source_sha256"]
            ),
            "Args/original_queue_runtime_ffnet_parameters_sha256": (
                original_queue_recovery_binding["ffnet_parameters_sha256"]
            ),
            "Args/original_queue_runtime_ffnet_task_id": (
                original_queue_recovery_binding["ffnet_task_id"]
            ),
            "Args/original_queue_runtime_ffnet_planned_queue": (
                original_queue_recovery_binding["ffnet_planned_queue"]
            ),
            "Args/original_queue_runtime_coformer_task_id": (
                original_queue_recovery_binding["coformer_task_id"]
            ),
            "Args/original_queue_runtime_coformer_artifacts_sha256": (
                original_queue_recovery_binding[
                    "coformer_artifact_inventory_sha256"
                ]
            ),
        }
        for role in SUCCESSOR_ROLE_ORDER:
            result[role].update(common)
    return result


def _normalized_status(task: object) -> str:
    value = getattr(task, "status", None)
    if value is None:
        getter = getattr(task, "get_status", None)
        value = getter() if callable(getter) else None
    value = getattr(value, "value", value)
    return str(value or "").rsplit(".", 1)[-1].lower()


def _task_script(task: object) -> dict[str, object]:
    script = getattr(getattr(task, "data", None), "script", None)
    to_dict = getattr(script, "to_dict", None)
    value = to_dict() if callable(to_dict) else None
    if not isinstance(value, Mapping):
        if isinstance(script, Mapping):
            value = script
        else:
            getter = getattr(task, "get_script", None)
            value = getter() if callable(getter) else None
    if not isinstance(value, Mapping):
        raise DeploymentError("task script metadata is unavailable")
    return dict(value)


def _script_source(task: object, *, entry_point: str, context: str) -> str:
    script = _task_script(task)
    if str(script.get("repository") or "") != "":
        raise DeploymentError(f"{context} is not a standalone producer")
    if str(script.get("working_dir") or "") != ".":
        raise DeploymentError(f"{context} working directory drifted")
    if str(script.get("entry_point") or "") != entry_point:
        raise DeploymentError(f"{context} entry point drifted")
    source = script.get("diff")
    if type(source) is not str or not source:
        raise DeploymentError(f"{context} standalone source is empty")
    return source


def _task_parameters(task: object, *, context: str) -> dict[str, object]:
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
    snapshot_id = _task_id(getattr(snapshot, "id", ""), f"{context} record")
    if snapshot_id != task_id:
        raise DeploymentError(f"{context} authoritative record identity mismatch")
    try:
        setattr(task, "_data", snapshot)
    except Exception as error:
        raise DeploymentError(
            f"{context} cannot install authoritative record"
        ) from error
    return task


def _task_parent(task: object) -> str:
    return str(getattr(getattr(task, "data", None), "parent", None) or "")


def _output_destination(task: object) -> str:
    return str(
        getattr(
            getattr(getattr(task, "data", None), "output", None), "destination", None
        )
        or ""
    )


def _raw_requirements(task: object) -> dict[str, object]:
    raw = _task_script(task).get("requirements")
    if not isinstance(raw, Mapping):
        raise DeploymentError("task requirement metadata is unavailable")
    return {str(key): value for key, value in raw.items()}


def _requirements(task: object) -> dict[str, str]:
    raw = _raw_requirements(task)
    return {str(key): str(value or "") for key, value in raw.items()}


def _task_docker(task: object) -> str:
    getter = getattr(task, "get_base_docker", None)
    if not callable(getter):
        raise DeploymentError("task cannot expose its base Docker command")
    value = getter()
    if type(value) is not str or not value.strip():
        raise DeploymentError("task base Docker command is empty")
    return " ".join(value.split())


def _project_name(task: object) -> str:
    getter = getattr(task, "get_project_name", None)
    if not callable(getter):
        raise DeploymentError("task cannot expose its project name")
    return str(getter() or "")


def _artifact_names(task: object) -> tuple[str, ...]:
    value = getattr(task, "artifacts", None)
    if not isinstance(value, Mapping):
        raise DeploymentError("task artifact inventory is unavailable")
    return tuple(sorted(str(key) for key in value))


def _task_tag_values(task: object, *, field: str, context: str) -> tuple[str, ...]:
    value = getattr(getattr(task, "data", None), field, None)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DeploymentError(f"{context} {field} inventory is unavailable")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise DeploymentError(f"{context} {field} contains an empty value")
    if len(set(result)) != len(result):
        raise DeploymentError(f"{context} {field} contains duplicate values")
    return result


def _task_configuration_objects(task: object, *, context: str) -> dict[str, object]:
    getter = getattr(task, "get_configuration_objects", None)
    if not callable(getter):
        raise DeploymentError(f"{context} cannot expose configuration objects")
    value = getter()
    if not isinstance(value, Mapping):
        raise DeploymentError(f"{context} configuration objects are invalid")
    return {str(key): item for key, item in value.items()}


def _artifact_metadata_records(
    task: object, *, context: str
) -> tuple[dict[str, object], ...]:
    execution = getattr(getattr(task, "data", None), "execution", None)
    artifacts = getattr(execution, "artifacts", None)
    if not isinstance(artifacts, Sequence) or isinstance(
        artifacts, (str, bytes, bytearray)
    ):
        raise DeploymentError(f"{context} authoritative artifact records unavailable")
    expected_keys = {
        "key",
        "type",
        "mode",
        "uri",
        "hash",
        "content_size",
        "timestamp",
        "display_data",
        "type_data",
    }
    records: list[dict[str, object]] = []
    for index, artifact in enumerate(artifacts):
        to_dict = getattr(artifact, "to_dict", None)
        value = to_dict() if callable(to_dict) else artifact
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise DeploymentError(
                f"{context} artifact record {index} key inventory drifted"
            )
        type_data = value.get("type_data")
        if not isinstance(type_data, Mapping) or set(type_data) != {
            "content_type",
            "preview",
        }:
            raise DeploymentError(
                f"{context} artifact record {index} type_data drifted"
            )
        preview = type_data.get("preview")
        if type(preview) is not str:
            raise DeploymentError(
                f"{context} artifact record {index} preview is invalid"
            )
        if (
            type(value.get("content_size")) is not int
            or type(value.get("timestamp")) is not int
        ):
            raise DeploymentError(
                f"{context} artifact record {index} numeric metadata drifted"
            )
        display_data = value.get("display_data")
        if not isinstance(display_data, list):
            raise DeploymentError(
                f"{context} artifact record {index} display_data drifted"
            )
        records.append(
            {
                "key": str(value.get("key") or ""),
                "type": str(value.get("type") or ""),
                "mode": str(value.get("mode") or ""),
                "uri": str(value.get("uri") or ""),
                "hash": str(value.get("hash") or ""),
                "content_size": value["content_size"],
                "timestamp": value["timestamp"],
                "display_data": list(display_data),
                "type_data_sha256": _content_sha256(dict(type_data)),
                "type_data_content_type": str(type_data.get("content_type") or ""),
                "type_data_preview_bytes": len(preview.encode("utf-8")),
            }
        )
    return tuple(records)


def _validate_completed_provenance_task_id(task_id: str) -> str:
    normalized = _task_id(task_id, "completed provenance task")
    if normalized != COMPLETED_PROVENANCE_TASK_ID:
        raise DeploymentError(
            "completed provenance task ID is not the sealed approved P evidence"
        )
    return normalized


def _completed_provenance_expected_contract() -> dict[str, object]:
    return {
        "task_id": COMPLETED_PROVENANCE_TASK_ID,
        "status": "completed",
        "parent_task_id": TRAINING_CONTROLLER_TASK_ID,
        "project": PROJECT_NAME,
        "task_name": ROLE_DEFINITIONS["P"]["task_name"],
        "entry_point": ROLE_DEFINITIONS["P"]["entry_point"],
        "script_sha256": COMPLETED_PROVENANCE_SOURCE_SHA256,
        "requirements_sha256": COMPLETED_PROVENANCE_REQUIREMENTS_SHA256,
        "tags": sorted(COMPLETED_PROVENANCE_TAGS),
        "system_tags": [],
        "configuration_objects": {},
        "parameters": {
            "Args/poll_seconds": "60.0",
            "Args/timeout_hours": "720.0",
            "Args/training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
        },
        "artifact": dict(COMPLETED_PROVENANCE_ARTIFACT),
        "output_uri": FILES_SERVER_URI,
    }


def _validate_completed_provenance(
    task_class: object, *, task_id: str, local_source: str
) -> dict[str, object]:
    task_id = _validate_completed_provenance_task_id(task_id)
    context = "completed provenance task"
    if _sha256_text(local_source) != COMPLETED_PROVENANCE_SOURCE_SHA256:
        raise DeploymentError("local completed provenance producer SHA-256 drifted")
    task = _authoritative_task(task_class, task_id, context=context)
    if _normalized_status(task) != "completed":
        raise DeploymentError(f"{context} is not completed")
    if _task_parent(task) != TRAINING_CONTROLLER_TASK_ID:
        raise DeploymentError(f"{context} parent drifted")
    if _project_name(task) != PROJECT_NAME:
        raise DeploymentError(f"{context} project drifted")
    if str(getattr(task, "name", "") or "") != ROLE_DEFINITIONS["P"]["task_name"]:
        raise DeploymentError(f"{context} name drifted")
    task_type = str(getattr(task, "task_type", "") or "").rsplit(".", 1)[-1]
    if task_type.lower() != "controller":
        raise DeploymentError(f"{context} type drifted")
    if _output_destination(task) != FILES_SERVER_URI:
        raise DeploymentError(f"{context} output destination drifted")
    observed_source = _script_source(
        task,
        entry_point=str(ROLE_DEFINITIONS["P"]["entry_point"]),
        context=context,
    )
    if (
        observed_source != local_source
        or _sha256_text(observed_source) != COMPLETED_PROVENANCE_SOURCE_SHA256
    ):
        raise DeploymentError(f"{context} source bytes drifted")
    if str(_task_script(task).get("binary") or "") != "python":
        raise DeploymentError(f"{context} binary drifted")
    requirements = _raw_requirements(task)
    if _content_sha256(requirements) != COMPLETED_PROVENANCE_REQUIREMENTS_SHA256:
        raise DeploymentError(f"{context} package requirements drifted")
    if requirements.get("org_pip") != SERVICE_REQUIREMENTS[0]:
        raise DeploymentError(f"{context} original package requirement drifted")
    pip_requirements = requirements.get("pip")
    if (
        not isinstance(pip_requirements, Sequence)
        or isinstance(pip_requirements, (str, bytes, bytearray))
        or list(pip_requirements).count(SERVICE_REQUIREMENTS[0]) != 1
    ):
        raise DeploymentError(f"{context} captured package inventory drifted")
    expected_docker = " ".join(f"{SERVICE_DOCKER_IMAGE} {SERVICE_DOCKER_ARGS}".split())
    if _task_docker(task) != expected_docker:
        raise DeploymentError(f"{context} Docker command drifted")
    expected_parameters = _completed_provenance_expected_contract()["parameters"]
    observed_parameters = _task_parameters(task, context=context)
    if (
        not isinstance(expected_parameters, Mapping)
        or set(observed_parameters) != set(expected_parameters)
        or any(
            not _parameter_equal(observed_parameters[key], value)
            for key, value in expected_parameters.items()
        )
    ):
        raise DeploymentError(f"{context} exact parameter contract drifted")
    tags = _task_tag_values(task, field="tags", context=context)
    if set(tags) != COMPLETED_PROVENANCE_TAGS:
        raise DeploymentError(f"{context} formal tag set drifted")
    system_tags = _task_tag_values(task, field="system_tags", context=context)
    if system_tags:
        raise DeploymentError(f"{context} system tag set drifted")
    configurations = _task_configuration_objects(task, context=context)
    if configurations:
        raise DeploymentError(f"{context} configuration objects drifted")
    expected_artifact_name = str(COMPLETED_PROVENANCE_ARTIFACT["key"])
    if _artifact_names(task) != (expected_artifact_name,):
        raise DeploymentError(f"{context} artifact inventory drifted")
    artifact_records = _artifact_metadata_records(task, context=context)
    if len(artifact_records) != 1 or _canonical_json(
        artifact_records[0]
    ) != _canonical_json(COMPLETED_PROVENANCE_ARTIFACT):
        raise DeploymentError(f"{context} sealed artifact metadata drifted")
    result = _completed_provenance_expected_contract()
    result["parameters_sha256"] = _content_sha256(expected_parameters)
    return result


def _validate_reference_task(
    task_class: object,
    *,
    task_id: str,
    entry_point: str,
    script_sha256: str,
    context: str,
) -> dict[str, object]:
    task = _authoritative_task(task_class, task_id, context=context)
    if _normalized_status(task) != "completed":
        raise DeploymentError(f"{context} is not completed")
    source = _script_source(task, entry_point=entry_point, context=context)
    observed_sha = _sha256_text(source)
    if observed_sha != script_sha256:
        raise DeploymentError(f"{context} producer SHA-256 mismatch")
    return {
        "task_id": task_id,
        "status": "completed",
        "entry_point": entry_point,
        "script_sha256": observed_sha,
    }


def _normalized_expected_parameters(parameters: Mapping[str, object]) -> dict[str, str]:
    return {str(key): str(value) for key, value in parameters.items()}


def _validate_configured_task(
    task_class: object,
    *,
    role: str,
    task_id: str,
    parent_id: str,
    source: str,
    parameters: Mapping[str, object],
    expected_statuses: Sequence[str],
) -> dict[str, object]:
    context = f"role {role} task"
    task = _authoritative_task(task_class, task_id, context=context)
    status = _normalized_status(task)
    if status not in expected_statuses:
        raise DeploymentError(
            f"{context} status {status!r} is outside {tuple(expected_statuses)!r}"
        )
    if _task_parent(task) != parent_id:
        raise DeploymentError(f"{context} parent drifted")
    if _project_name(task) != PROJECT_NAME:
        raise DeploymentError(f"{context} project drifted")
    if str(getattr(task, "name", "") or "") != ROLE_DEFINITIONS[role]["task_name"]:
        raise DeploymentError(f"{context} name drifted")
    task_type = str(getattr(task, "task_type", "") or "").rsplit(".", 1)[-1].lower()
    if task_type != "controller":
        raise DeploymentError(f"{context} type drifted")
    if _output_destination(task) != FILES_SERVER_URI:
        raise DeploymentError(f"{context} output destination drifted")
    observed_source = _script_source(
        task,
        entry_point=str(ROLE_DEFINITIONS[role]["entry_point"]),
        context=context,
    )
    if observed_source != source or _sha256_text(observed_source) != _sha256_text(
        source
    ):
        raise DeploymentError(f"{context} source bytes drifted")
    script = _task_script(task)
    if str(script.get("binary") or "") != "python":
        raise DeploymentError(f"{context} binary drifted")
    if _requirements(task) != {"pip": "\n".join(SERVICE_REQUIREMENTS)}:
        raise DeploymentError(f"{context} package requirements drifted")
    expected_docker = " ".join(f"{SERVICE_DOCKER_IMAGE} {SERVICE_DOCKER_ARGS}".split())
    if _task_docker(task) != expected_docker:
        raise DeploymentError(f"{context} Docker command drifted")
    observed_parameters = _task_parameters(task, context=context)
    if set(observed_parameters) != set(parameters) or any(
        not _parameter_equal(observed_parameters[key], value)
        for key, value in parameters.items()
    ):
        raise DeploymentError(f"{context} exact parameter contract drifted")
    if expected_statuses == ("created",) and _artifact_names(task):
        raise DeploymentError(f"{context} shell unexpectedly contains artifacts")
    queue: str | None = None
    if status != "created":
        queue_getter = getattr(task, "get_executed_queue", None)
        if not callable(queue_getter):
            raise DeploymentError(f"{context} cannot expose its execution queue")
        queue = queue_getter(return_name=True)
        if queue != SERVICES_QUEUE:
            raise DeploymentError(f"{context} execution queue drifted")
        queue_id = queue_getter(return_name=False)
        if queue_id != SERVICES_QUEUE_ID:
            raise DeploymentError(f"{context} execution queue ID drifted")
    return {
        "task_id": task_id,
        "status": status,
        "parent_task_id": parent_id,
        "entry_point": ROLE_DEFINITIONS[role]["entry_point"],
        "script_sha256": _sha256_text(source),
        "parameters": _normalized_expected_parameters(parameters),
        "parameters_sha256": _content_sha256(
            _normalized_expected_parameters(parameters)
        ),
        "output_uri": FILES_SERVER_URI,
        "queue": queue,
    }


def _assert_sources_unchanged(
    base_sources: Mapping[str, str],
    embedded_sources: Mapping[str, Mapping[str, str]],
) -> None:
    current = _read_base_sources()
    for role in ROLE_ORDER:
        if current[role] != base_sources[role]:
            raise DeploymentError(
                f"role {role} local producer changed after deployment rendering"
            )
    for role in ROLE_ORDER:
        for module_name, expected_source in embedded_sources[role].items():
            path = _local_module_path(module_name)
            if path is None:
                raise DeploymentError(
                    f"role {role} embedded module {module_name} disappeared"
                )
            observed = _read_source(
                path, context=f"role {role} embedded module {module_name}"
            )
            if observed != expected_source:
                raise DeploymentError(
                    f"role {role} embedded module {module_name} changed after rendering"
                )


def _configure_shell(
    task: object,
    *,
    role: str,
    parent_id: str,
    source: str,
    parameters: Mapping[str, object],
) -> None:
    task.set_script(
        repository="",
        branch="",
        commit="",
        diff=source,
        working_dir=".",
        entry_point=str(ROLE_DEFINITIONS[role]["entry_point"]),
    )
    task.set_packages(list(SERVICE_REQUIREMENTS))
    task.set_base_docker(
        docker_image=SERVICE_DOCKER_IMAGE,
        docker_arguments=SERVICE_DOCKER_ARGS,
    )
    task.set_parent(parent_id)
    task.output_uri = FILES_SERVER_URI
    task.set_parameters(dict(parameters))
    flushed = task.flush(wait_for_uploads=True)
    if flushed is False:
        raise DeploymentError(f"role {role} shell flush was not confirmed")


def _create_shell(task_class: object, *, role: str) -> object:
    task_types = getattr(task_class, "TaskTypes", None)
    controller_type = getattr(task_types, "controller", "controller")
    creator = getattr(task_class, "create", None)
    if not callable(creator):
        raise DeploymentError("ClearML Task class cannot create task shells")
    task = creator(
        project_name=PROJECT_NAME,
        task_name=str(ROLE_DEFINITIONS[role]["task_name"]),
        task_type=controller_type,
        script=None,
        packages=None,
        docker=None,
        docker_args=None,
        add_task_init_call=False,
        binary="python",
        detect_repository=False,
    )
    _task_id(getattr(task, "id", ""), f"role {role} shell")
    if _normalized_status(task) != "created":
        raise DeploymentError(f"role {role} shell was not created")
    return task


def _enqueue(
    task_class: object,
    task: object,
    *,
    role: str,
    task_id: str,
    sleeper: Callable[[float], None],
) -> str:
    enqueuer = getattr(task_class, "enqueue", None)
    if not callable(enqueuer):
        raise DeploymentError("ClearML Task class cannot enqueue tasks")
    response = enqueuer(task=task, queue_name=SERVICES_QUEUE)
    if response is None or response is False:
        raise DeploymentError(f"ClearML did not confirm role {role} enqueue")
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            observed = _authoritative_task(
                task_class, task_id, context=f"role {role} queued task"
            )
            status = _normalized_status(observed)
            if status in {"queued", "in_progress", "completed"}:
                queue_getter = getattr(observed, "get_executed_queue", None)
                if not callable(queue_getter):
                    raise DeploymentError(
                        f"role {role} cannot expose its execution queue"
                    )
                if queue_getter(return_name=True) != SERVICES_QUEUE:
                    raise DeploymentError(f"role {role} execution queue drifted")
                if queue_getter(return_name=False) != SERVICES_QUEUE_ID:
                    raise DeploymentError(f"role {role} execution queue ID drifted")
                return status
            last_error = DeploymentError(
                f"role {role} did not reach queued status after enqueue"
            )
        except Exception as error:  # bounded eventual-consistency retry
            last_error = error
        if attempt != 4:
            sleeper(1.0)
    raise DeploymentError(
        f"role {role} enqueue authority readback failed"
    ) from last_error


def _deploy_chain(
    task_class: object,
    *,
    sleeper: Callable[[float], None] = time.sleep,
    journal: dict[str, object] | None = None,
) -> dict[str, object]:
    journal = {} if journal is None else journal
    controller_source = _read_source(
        ROOT / "tools/resilient_v2x" / CONTROLLER_ENTRY_POINT,
        context="training controller",
    )
    controller = _validate_reference_task(
        task_class,
        task_id=TRAINING_CONTROLLER_TASK_ID,
        entry_point=CONTROLLER_ENTRY_POINT,
        script_sha256=_sha256_text(controller_source),
        context="training controller",
    )
    template = _validate_reference_task(
        task_class,
        task_id=EVALUATION_TEMPLATE_TASK_ID,
        entry_point="clearml_5090_bootstrap.py",
        script_sha256=EVALUATION_SCRIPT_SHA256,
        context="evaluation template",
    )
    journal["references"] = {
        "training_controller": controller,
        "evaluation_template": template,
    }

    shells: dict[str, object] = {}
    task_ids: dict[str, str] = {}
    journal["task_ids"] = task_ids
    for role in ROLE_ORDER:
        shell = _create_shell(task_class, role=role)
        shells[role] = shell
        task_ids[role] = _task_id(getattr(shell, "id", ""), f"role {role} shell")
    if len(set(task_ids.values())) != len(ROLE_ORDER):
        raise DeploymentError("ClearML returned duplicate successor shell IDs")
    if TRAINING_CONTROLLER_TASK_ID in task_ids.values():
        raise DeploymentError("ClearML successor shell aliases the controller")
    journal["task_ids"] = dict(task_ids)

    # This read intentionally happens only after every shell ID exists.
    base_sources = _read_base_sources()
    (
        rendered,
        replacements,
        embedded_sources,
        embedded_records,
    ) = _render_sources_with_bundle_evidence(base_sources, task_ids)
    smoke = _isolated_import_smoke(rendered, embedded_records)
    records = _source_records(base_sources, rendered, replacements, embedded_records)
    source_hashes = {role: str(records[role]["deployed_sha256"]) for role in ROLE_ORDER}
    parents = _parent_ids(task_ids)
    parameters = _parameters(task_ids, source_hashes)
    journal["sources"] = records
    journal["standalone_import_smoke"] = smoke
    journal["parents"] = parents
    journal["parameters"] = {
        role: _normalized_expected_parameters(parameters[role]) for role in ROLE_ORDER
    }

    for role in ROLE_ORDER:
        _configure_shell(
            shells[role],
            role=role,
            parent_id=parents[role],
            source=rendered[role],
            parameters=parameters[role],
        )

    # A local producer edit after rendering invalidates the entire staging set.
    _assert_sources_unchanged(base_sources, embedded_sources)
    pre_enqueue = {
        role: _validate_configured_task(
            task_class,
            role=role,
            task_id=task_ids[role],
            parent_id=parents[role],
            source=rendered[role],
            parameters=parameters[role],
            expected_statuses=("created",),
        )
        for role in ROLE_ORDER
    }
    # Recheck every shell is still created immediately before the first enqueue.
    for role in ROLE_ORDER:
        task = _authoritative_task(
            task_class, task_ids[role], context=f"role {role} final created check"
        )
        if _normalized_status(task) != "created":
            raise DeploymentError(
                "all five successor tasks must remain created before enqueue begins"
            )
    journal["pre_enqueue_authority"] = pre_enqueue

    enqueued: list[str] = []
    enqueue_status: dict[str, str] = {}
    for role in ROLE_ORDER:
        enqueue_status[role] = _enqueue(
            task_class,
            shells[role],
            role=role,
            task_id=task_ids[role],
            sleeper=sleeper,
        )
        enqueued.append(role)
        journal["enqueued_roles"] = list(enqueued)
        journal["enqueue_status"] = dict(enqueue_status)

    queued = {
        role: _validate_configured_task(
            task_class,
            role=role,
            task_id=task_ids[role],
            parent_id=parents[role],
            source=rendered[role],
            parameters=parameters[role],
            expected_statuses=("queued", "in_progress", "completed"),
        )
        for role in ROLE_ORDER
    }
    return {
        "references": journal["references"],
        "task_ids": task_ids,
        "parents": parents,
        "sources": records,
        "standalone_import_smoke": smoke,
        "parameters": journal["parameters"],
        "pre_enqueue_authority": pre_enqueue,
        "queued_authority": queued,
        "enqueue_order": list(ROLE_ORDER),
    }


def _deploy_successors_from_completed_provenance(
    task_class: object,
    *,
    completed_provenance_task_id: str = COMPLETED_PROVENANCE_TASK_ID,
    amendment_binding: Mapping[str, object] | None = None,
    runtime_recovery_binding: Mapping[str, object] | None = None,
    original_queue_recovery_binding: Mapping[str, object] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    journal: dict[str, object] | None = None,
) -> dict[str, object]:
    """Attach a fresh W->L->A->S chain to the sealed completed P evidence."""

    journal = {} if journal is None else journal
    completed_provenance_task_id = _validate_completed_provenance_task_id(
        completed_provenance_task_id
    )
    provenance_source = _read_source(
        SOURCE_PATHS["P"], context="completed provenance local producer"
    )
    controller_source = _read_source(
        ROOT / "tools/resilient_v2x" / CONTROLLER_ENTRY_POINT,
        context="training controller",
    )
    references = {
        "training_controller": _validate_reference_task(
            task_class,
            task_id=TRAINING_CONTROLLER_TASK_ID,
            entry_point=CONTROLLER_ENTRY_POINT,
            script_sha256=_sha256_text(controller_source),
            context="training controller",
        ),
        "evaluation_template": _validate_reference_task(
            task_class,
            task_id=EVALUATION_TEMPLATE_TASK_ID,
            entry_point="clearml_5090_bootstrap.py",
            script_sha256=EVALUATION_SCRIPT_SHA256,
            context="evaluation template",
        ),
    }
    provenance_precreate = _validate_completed_provenance(
        task_class,
        task_id=completed_provenance_task_id,
        local_source=provenance_source,
    )
    journal["references"] = references
    journal["completed_provenance_precreate"] = provenance_precreate
    journal["created_roles"] = []
    journal["created_task_ids"] = {}

    shells: dict[str, object] = {}
    task_ids = {"P": completed_provenance_task_id}
    for role in SUCCESSOR_ROLE_ORDER:
        shell = _create_shell(task_class, role=role)
        task_id = _task_id(getattr(shell, "id", ""), f"role {role} shell")
        if task_id in task_ids.values() or task_id == TRAINING_CONTROLLER_TASK_ID:
            raise DeploymentError("ClearML returned a duplicate successor shell ID")
        shells[role] = shell
        task_ids[role] = task_id
        journal["created_roles"] = [*journal["created_roles"], role]
        journal["created_task_ids"] = {
            role_name: task_ids[role_name]
            for role_name in SUCCESSOR_ROLE_ORDER
            if role_name in task_ids
        }
    journal["task_ids"] = dict(task_ids)

    # Read W/L/A/S only after all four new IDs exist.  This makes a watcher
    # compatibility fix made immediately before execution part of the sealed
    # deployed bytes and of every downstream producer pin.
    base_sources = _read_base_sources()
    if (
        base_sources["P"] != provenance_source
        or _sha256_text(base_sources["P"]) != COMPLETED_PROVENANCE_SOURCE_SHA256
    ):
        raise DeploymentError(
            "completed provenance local source drifted after validation"
        )
    (
        rendered,
        replacements,
        embedded_sources,
        embedded_records,
    ) = _render_sources_with_bundle_evidence(base_sources, task_ids)
    smoke = _isolated_import_smoke(rendered, embedded_records)
    records = _source_records(base_sources, rendered, replacements, embedded_records)
    source_hashes = {role: str(records[role]["deployed_sha256"]) for role in ROLE_ORDER}
    if source_hashes["P"] != COMPLETED_PROVENANCE_SOURCE_SHA256:
        raise DeploymentError("rendered completed provenance source identity drifted")
    all_parents = _parent_ids(task_ids)
    all_parameters = _parameters(
        task_ids,
        source_hashes,
        authoritative_recovery=True,
        amendment_binding=amendment_binding,
        runtime_recovery_binding=runtime_recovery_binding,
        original_queue_recovery_binding=original_queue_recovery_binding,
    )
    parents = {role: all_parents[role] for role in SUCCESSOR_ROLE_ORDER}
    parameters = {role: all_parameters[role] for role in SUCCESSOR_ROLE_ORDER}
    journal["sources"] = records
    journal["standalone_import_smoke"] = smoke
    journal["parents"] = parents
    journal["parameters"] = {
        role: _normalized_expected_parameters(parameters[role])
        for role in SUCCESSOR_ROLE_ORDER
    }

    for role in SUCCESSOR_ROLE_ORDER:
        _configure_shell(
            shells[role],
            role=role,
            parent_id=parents[role],
            source=rendered[role],
            parameters=parameters[role],
        )

    _assert_sources_unchanged(base_sources, embedded_sources)
    provenance_pre_enqueue = _validate_completed_provenance(
        task_class,
        task_id=completed_provenance_task_id,
        local_source=provenance_source,
    )
    if _canonical_json(provenance_pre_enqueue) != _canonical_json(provenance_precreate):
        raise DeploymentError("completed provenance authority drifted during staging")
    journal["completed_provenance_pre_enqueue"] = provenance_pre_enqueue
    pre_enqueue = {
        role: _validate_configured_task(
            task_class,
            role=role,
            task_id=task_ids[role],
            parent_id=parents[role],
            source=rendered[role],
            parameters=parameters[role],
            expected_statuses=("created",),
        )
        for role in SUCCESSOR_ROLE_ORDER
    }
    for role in SUCCESSOR_ROLE_ORDER:
        task = _authoritative_task(
            task_class, task_ids[role], context=f"role {role} final created check"
        )
        if _normalized_status(task) != "created":
            raise DeploymentError(
                "all four successor tasks must remain created before enqueue begins"
            )
    _assert_sources_unchanged(base_sources, embedded_sources)
    journal["pre_enqueue_authority"] = pre_enqueue

    enqueued: list[str] = []
    enqueue_status: dict[str, str] = {}
    for role in SUCCESSOR_ROLE_ORDER:
        enqueue_status[role] = _enqueue(
            task_class,
            shells[role],
            role=role,
            task_id=task_ids[role],
            sleeper=sleeper,
        )
        enqueued.append(role)
        journal["enqueued_roles"] = list(enqueued)
        journal["enqueue_status"] = dict(enqueue_status)
    queued = {
        role: _validate_configured_task(
            task_class,
            role=role,
            task_id=task_ids[role],
            parent_id=parents[role],
            source=rendered[role],
            parameters=parameters[role],
            expected_statuses=("queued", "in_progress", "completed"),
        )
        for role in SUCCESSOR_ROLE_ORDER
    }
    return {
        "references": references,
        "completed_provenance": provenance_pre_enqueue,
        "created_roles": list(SUCCESSOR_ROLE_ORDER),
        "created_task_ids": {role: task_ids[role] for role in SUCCESSOR_ROLE_ORDER},
        "task_ids": task_ids,
        "parents": parents,
        "sources": records,
        "standalone_import_smoke": smoke,
        "parameters": journal["parameters"],
        "pre_enqueue_authority": pre_enqueue,
        "queued_authority": queued,
        "enqueue_order": list(SUCCESSOR_ROLE_ORDER),
    }


def _fixed_bindings() -> dict[str, object]:
    return {
        "project": PROJECT_NAME,
        "training_controller_task_id": TRAINING_CONTROLLER_TASK_ID,
        "evaluation_template_task_id": EVALUATION_TEMPLATE_TASK_ID,
        "fileserver_output_uri": FILES_SERVER_URI,
        "services_queue": SERVICES_QUEUE,
        "services_queue_id": SERVICES_QUEUE_ID,
        "service_docker_image": SERVICE_DOCKER_IMAGE,
        "service_docker_args": SERVICE_DOCKER_ARGS,
        "service_requirements": list(SERVICE_REQUIREMENTS),
        "enqueue_order": list(ROLE_ORDER),
        "role_output_artifacts": {
            role: list(ROLE_OUTPUT_ARTIFACTS[role]) for role in ROLE_ORDER
        },
        "parent_policy": "P<-1011, W<-P, L<-W, A<-L, S<-A",
    }


def _completed_successor_fixed_bindings(
    amendment_binding: Mapping[str, object] | None = None,
    runtime_recovery_binding: Mapping[str, object] | None = None,
    original_queue_recovery_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result = _fixed_bindings()
    result.update(
        {
            "completed_provenance_task_id": COMPLETED_PROVENANCE_TASK_ID,
            "completed_provenance_script_sha256": (COMPLETED_PROVENANCE_SOURCE_SHA256),
            "candidate_release_blockers": [
                dict(item) for item in CANDIDATE_RELEASE_BLOCKERS
            ],
            "authoritative_evaluation_plan_producer_task_id": (
                AUTHORITATIVE_EVALUATION_PLAN_PRODUCER_TASK_ID
            ),
            "authoritative_evaluation_plan_artifact": (
                AUTHORITATIVE_EVALUATION_PLAN_ARTIFACT
            ),
            "authoritative_evaluation_plan_seal_sha256": (
                AUTHORITATIVE_EVALUATION_PLAN_SEAL_SHA256
            ),
            "authoritative_evaluation_task_policy": (
                "reuse_exact_ids_no_clone_replacements"
            ),
            "enqueue_order": list(SUCCESSOR_ROLE_ORDER),
            "parent_policy": "W<-completed-P, L<-W2, A<-L2, S<-A2",
        }
    )
    if amendment_binding is not None:
        result.update(
            {
                "authoritative_evaluation_plan_producer_task_id": (
                    amendment_binding["producer_task_id"]
                ),
                "authoritative_evaluation_plan_seal_sha256": (
                    amendment_binding["revised_plan_seal_sha256"]
                ),
                "evaluation_plan_amendment": dict(amendment_binding),
                "amendment_policy": (
                    "only_ffnet_queue_gpu4_v100_to_gpu4_a100_exact_28_ids"
                ),
            }
        )
    if runtime_recovery_binding is not None:
        result["exact_eval_chain_runtime_recovery"] = dict(
            runtime_recovery_binding
        )
        result["runtime_recovery_policy"] = (
            "ffnet_created_unqueued_candidate_services_same_28_ids_no_replacements"
        )
    if original_queue_recovery_binding is not None:
        if amendment_binding is not None or runtime_recovery_binding is not None:
            raise DeploymentError(
                "original-queue recovery cannot be combined with an amendment"
            )
        result["original_queue_runtime_recovery"] = dict(
            original_queue_recovery_binding
        )
        result["runtime_recovery_policy"] = (
            "ffnet_created_unqueued_original_gpu4_v100_no_plan_amendment"
        )
    return result


def _base_receipt(
    *,
    mode: str,
    status: str,
    remote_state_changed: bool,
    fixed_bindings: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "receipt_type": "resilient_v2x_formal_successor_chain_deployment",
        "generated_at_utc": _utc_timestamp(),
        "mode": mode,
        "status": status,
        "remote_state_changed": remote_state_changed,
        "fixed_bindings": dict(fixed_bindings or _fixed_bindings()),
    }


def _strict_json_load(path: Path, *, context: str) -> dict[str, object]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DeploymentError(f"{context} cannot be resolved") from error
    if path.is_symlink() or not resolved.is_file():
        raise DeploymentError(f"{context} must be a regular non-symlink file")
    if resolved.stat().st_size > 16 * 1024 * 1024:
        raise DeploymentError(f"{context} is unexpectedly large")

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DeploymentError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise DeploymentError(f"{context} contains non-finite constant {value}")

    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DeploymentError(f"{context} is not strict JSON") from error
    if not isinstance(value, dict):
        raise DeploymentError(f"{context} must contain one JSON object")
    return value


def _require_exact_mapping(
    value: object, *, keys: set[str], context: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DeploymentError(f"{context} must be an object")
    result = dict(value)
    if set(result) != keys:
        raise DeploymentError(f"{context} key inventory drifted")
    return result


def _validate_failed_receipt_fixed_bindings(value: object) -> None:
    if not isinstance(value, Mapping):
        raise DeploymentError("failed receipt fixed bindings must be an object")
    observed = dict(value)
    expected = _fixed_bindings()
    observed_queue = observed.pop("services_queue", None)
    observed_queue_id = observed.pop("services_queue_id", None)
    expected.pop("services_queue")
    expected.pop("services_queue_id")
    if _canonical_json(observed) != _canonical_json(expected):
        raise DeploymentError("failed receipt fixed bindings drifted")
    if observed_queue == "clearml-services":
        if observed_queue_id not in {None, ""}:
            raise DeploymentError("legacy failed queue must not invent a queue ID")
    elif observed_queue == SERVICES_QUEUE:
        if observed_queue_id != SERVICES_QUEUE_ID:
            raise DeploymentError("failed receipt services queue ID drifted")
    else:
        raise DeploymentError("failed receipt services queue name is unsupported")


def _load_failed_receipt(path: Path) -> tuple[dict[str, object], Path]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DeploymentError("failed deployment receipt cannot be resolved") from error
    receipt = _strict_json_load(path, context="failed deployment receipt")
    expected_keys = {
        "schema_version",
        "receipt_type",
        "generated_at_utc",
        "mode",
        "status",
        "remote_state_changed",
        "fixed_bindings",
        "failure",
        "partial_journal",
        "seal_sha256",
    }
    if set(receipt) != expected_keys:
        raise DeploymentError("failed deployment receipt key inventory drifted")
    observed_seal = _sha256(receipt.get("seal_sha256"), "failed receipt seal")
    if _seal(receipt)["seal_sha256"] != observed_seal:
        raise DeploymentError("failed deployment receipt seal mismatch")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("receipt_type")
        != "resilient_v2x_formal_successor_chain_deployment"
        or receipt.get("mode") != "execute"
        or receipt.get("status") != "failed_closed"
        or receipt.get("remote_state_changed") is not True
        or type(receipt.get("generated_at_utc")) is not str
        or not str(receipt["generated_at_utc"])
    ):
        raise DeploymentError("failed deployment receipt identity is invalid")
    failure = _require_exact_mapping(
        receipt.get("failure"), keys={"type", "message"}, context="failure"
    )
    if any(type(failure[key]) is not str or not failure[key] for key in failure):
        raise DeploymentError("failed deployment receipt failure is invalid")
    _validate_failed_receipt_fixed_bindings(receipt.get("fixed_bindings"))
    return receipt, resolved


def _resume_preflight(
    task_class: object,
    failed_receipt: Mapping[str, object],
) -> dict[str, object]:
    journal = _require_exact_mapping(
        failed_receipt.get("partial_journal"),
        keys={
            "references",
            "task_ids",
            "sources",
            "standalone_import_smoke",
            "parents",
            "parameters",
            "pre_enqueue_authority",
        },
        context="failed receipt partial journal",
    )
    # A confirmed enqueue is never resumable.  Receipts with either key present
    # are rejected above even if an attacker writes an empty-looking value.
    if "enqueued_roles" in journal or "enqueue_status" in journal:
        raise DeploymentError("failed receipt already contains enqueue state")
    task_id_values = _require_exact_mapping(
        journal.get("task_ids"), keys=set(ROLE_ORDER), context="successor task IDs"
    )
    task_ids = {
        role: _task_id(task_id_values[role], f"resume role {role} task")
        for role in ROLE_ORDER
    }
    if len(set(task_ids.values())) != len(ROLE_ORDER):
        raise DeploymentError("resume successor task IDs are not unique")
    if TRAINING_CONTROLLER_TASK_ID in task_ids.values():
        raise DeploymentError("resume successor task aliases the controller")

    base_sources = _read_base_sources()
    (
        rendered,
        replacements,
        embedded_sources,
        embedded_records,
    ) = _render_sources_with_bundle_evidence(base_sources, task_ids)
    smoke = _isolated_import_smoke(rendered, embedded_records)
    records = _source_records(base_sources, rendered, replacements, embedded_records)
    source_hashes = {role: str(records[role]["deployed_sha256"]) for role in ROLE_ORDER}
    parents = _parent_ids(task_ids)
    parameters = _parameters(task_ids, source_hashes)
    normalized_parameters = {
        role: _normalized_expected_parameters(parameters[role]) for role in ROLE_ORDER
    }
    for name, expected in {
        "task_ids": task_ids,
        "sources": records,
        "standalone_import_smoke": smoke,
        "parents": parents,
        "parameters": normalized_parameters,
    }.items():
        if _canonical_json(journal.get(name)) != _canonical_json(expected):
            raise DeploymentError(f"failed receipt partial journal {name} drifted")

    controller_source = _read_source(
        ROOT / "tools/resilient_v2x" / CONTROLLER_ENTRY_POINT,
        context="training controller",
    )
    references = {
        "training_controller": _validate_reference_task(
            task_class,
            task_id=TRAINING_CONTROLLER_TASK_ID,
            entry_point=CONTROLLER_ENTRY_POINT,
            script_sha256=_sha256_text(controller_source),
            context="training controller",
        ),
        "evaluation_template": _validate_reference_task(
            task_class,
            task_id=EVALUATION_TEMPLATE_TASK_ID,
            entry_point="clearml_5090_bootstrap.py",
            script_sha256=EVALUATION_SCRIPT_SHA256,
            context="evaluation template",
        ),
    }
    if _canonical_json(journal.get("references")) != _canonical_json(references):
        raise DeploymentError("failed receipt partial journal references drifted")

    pre_enqueue = {
        role: _validate_configured_task(
            task_class,
            role=role,
            task_id=task_ids[role],
            parent_id=parents[role],
            source=rendered[role],
            parameters=parameters[role],
            expected_statuses=("created",),
        )
        for role in ROLE_ORDER
    }
    if _canonical_json(journal.get("pre_enqueue_authority")) != _canonical_json(
        pre_enqueue
    ):
        raise DeploymentError(
            "failed receipt partial journal pre-enqueue authority drifted"
        )
    _assert_sources_unchanged(base_sources, embedded_sources)

    # Final all-at-once authority barrier: no enqueue begins unless every reused
    # shell is still exactly created after all receipt and source validation.
    shells: dict[str, object] = {}
    for role in ROLE_ORDER:
        task = _authoritative_task(
            task_class, task_ids[role], context=f"resume role {role} final check"
        )
        if _normalized_status(task) != "created":
            raise DeploymentError(
                "all five resumed successor tasks must remain created"
            )
        shells[role] = task
    _assert_sources_unchanged(base_sources, embedded_sources)
    return {
        "references": references,
        "task_ids": task_ids,
        "parents": parents,
        "sources": records,
        "standalone_import_smoke": smoke,
        "parameters": normalized_parameters,
        "raw_parameters": parameters,
        "rendered_sources": rendered,
        "pre_enqueue_authority": pre_enqueue,
        "shells": shells,
    }


def _resume_failed_deployment(
    task_class: object,
    *,
    failed_receipt_path: Path,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    failed_receipt, resolved_path = _load_failed_receipt(failed_receipt_path)
    source_receipt_seal = _sha256(
        failed_receipt.get("seal_sha256"), "failed receipt seal"
    )
    recovery_journal: dict[str, object] = {
        "source_receipt_path": str(resolved_path),
        "source_receipt_seal_sha256": source_receipt_seal,
        "shell_creation_performed": False,
        "enqueued_roles": [],
    }
    try:
        state = _resume_preflight(task_class, failed_receipt)
        recovery_journal["reused_task_ids"] = dict(state["task_ids"])
        recovery_journal["preflight"] = {
            "parents": state["parents"],
            "sources": state["sources"],
            "parameters": state["parameters"],
            "pre_enqueue_authority": state["pre_enqueue_authority"],
            "standalone_import_smoke": state["standalone_import_smoke"],
        }
        enqueue_status: dict[str, str] = {}
        for role in ROLE_ORDER:
            enqueue_status[role] = _enqueue(
                task_class,
                state["shells"][role],
                role=role,
                task_id=state["task_ids"][role],
                sleeper=sleeper,
            )
            recovery_journal["enqueued_roles"].append(role)
            recovery_journal["enqueue_status"] = dict(enqueue_status)
        queued = {
            role: _validate_configured_task(
                task_class,
                role=role,
                task_id=state["task_ids"][role],
                parent_id=state["parents"][role],
                source=state["rendered_sources"][role],
                parameters=state["raw_parameters"][role],
                expected_statuses=("queued", "in_progress", "completed"),
            )
            for role in ROLE_ORDER
        }
    except Exception as error:
        receipt = _base_receipt(
            mode="recovery_execute",
            status="failed_closed",
            remote_state_changed=bool(recovery_journal["enqueued_roles"]),
        )
        receipt.update(
            {
                "failure": {"type": type(error).__name__, "message": str(error)},
                "recovery_journal": recovery_journal,
            }
        )
        failure = DeploymentError(str(error))
        setattr(failure, "sealed_receipt", _seal(receipt))
        raise failure from error

    receipt = _base_receipt(
        mode="recovery_execute", status="deployed", remote_state_changed=True
    )
    receipt.update(
        {
            "recovery": {
                "source_failed_receipt_path": str(resolved_path),
                "source_failed_receipt_seal_sha256": source_receipt_seal,
                "shell_creation_performed": False,
                "reused_task_ids": dict(state["task_ids"]),
                "enqueue_order": list(ROLE_ORDER),
                "queue_name": SERVICES_QUEUE,
                "queue_id": SERVICES_QUEUE_ID,
            },
            "references": state["references"],
            "task_ids": state["task_ids"],
            "parents": state["parents"],
            "sources": state["sources"],
            "standalone_import_smoke": state["standalone_import_smoke"],
            "parameters": state["parameters"],
            "pre_enqueue_authority": state["pre_enqueue_authority"],
            "queued_authority": queued,
            "enqueue_order": list(ROLE_ORDER),
        }
    )
    return _seal(receipt)


def _dry_run_receipt() -> dict[str, object]:
    base_sources = _read_base_sources()
    (
        rendered,
        replacements,
        _,
        embedded_records,
    ) = _render_sources_with_bundle_evidence(base_sources, DRY_RUN_IDS)
    smoke = _isolated_import_smoke(rendered, embedded_records)
    records = _source_records(base_sources, rendered, replacements, embedded_records)
    hashes = {role: str(records[role]["deployed_sha256"]) for role in ROLE_ORDER}
    receipt = _base_receipt(
        mode="dry_run", status="planned", remote_state_changed=False
    )
    receipt.update(
        {
            "placeholder_task_ids": dict(DRY_RUN_IDS),
            "placeholder_ids_are_not_remote_tasks": True,
            "parents": _parent_ids(DRY_RUN_IDS),
            "sources": records,
            "standalone_import_smoke": smoke,
            "parameters": {
                role: _normalized_expected_parameters(values)
                for role, values in _parameters(DRY_RUN_IDS, hashes).items()
            },
            "execution_contract": {
                "shells_created_before_source_read": True,
                "all_five_created_and_authoritatively_verified_before_enqueue": True,
                "execute_requires_explicit_flag": "--execute",
                "dry_run_performs_remote_calls": False,
                "standalone_sources_are_import_smoked_without_repository": True,
            },
        }
    )
    return _seal(receipt)


def _completed_successor_dry_run_receipt(
    *,
    completed_provenance_task_id: str = COMPLETED_PROVENANCE_TASK_ID,
    amendment_binding: Mapping[str, object] | None = None,
    runtime_recovery_binding: Mapping[str, object] | None = None,
    original_queue_recovery_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    completed_provenance_task_id = _validate_completed_provenance_task_id(
        completed_provenance_task_id
    )
    task_ids = dict(SUCCESSOR_DRY_RUN_IDS)
    task_ids["P"] = completed_provenance_task_id
    base_sources = _read_base_sources()
    if _sha256_text(base_sources["P"]) != COMPLETED_PROVENANCE_SOURCE_SHA256:
        raise DeploymentError("local completed provenance producer SHA-256 drifted")
    (
        rendered,
        replacements,
        _,
        embedded_records,
    ) = _render_sources_with_bundle_evidence(base_sources, task_ids)
    smoke = _isolated_import_smoke(rendered, embedded_records)
    records = _source_records(base_sources, rendered, replacements, embedded_records)
    hashes = {role: str(records[role]["deployed_sha256"]) for role in ROLE_ORDER}
    all_parents = _parent_ids(task_ids)
    all_parameters = _parameters(
        task_ids,
        hashes,
        authoritative_recovery=True,
        amendment_binding=amendment_binding,
        runtime_recovery_binding=runtime_recovery_binding,
        original_queue_recovery_binding=original_queue_recovery_binding,
    )
    receipt = _base_receipt(
        mode="completed_provenance_successor_dry_run",
        status="planned",
        remote_state_changed=False,
        fixed_bindings=_completed_successor_fixed_bindings(
            amendment_binding,
            runtime_recovery_binding,
            original_queue_recovery_binding,
        ),
    )
    receipt.update(
        {
            "completed_provenance_expected_contract": (
                _completed_provenance_expected_contract()
            ),
            "dependency_task_ids": task_ids,
            "placeholder_task_ids": {
                role: task_ids[role] for role in SUCCESSOR_ROLE_ORDER
            },
            "placeholder_ids_are_not_remote_tasks": True,
            "created_roles": list(SUCCESSOR_ROLE_ORDER),
            "parents": {role: all_parents[role] for role in SUCCESSOR_ROLE_ORDER},
            "sources": records,
            "standalone_import_smoke": smoke,
            "parameters": {
                role: _normalized_expected_parameters(all_parameters[role])
                for role in SUCCESSOR_ROLE_ORDER
            },
            "execution_contract": {
                "completed_provenance_authority_validation": (
                    "deferred_until_explicit_execute"
                ),
                "only_four_successor_shells_are_created": True,
                "all_four_shells_created_before_latest_source_read": True,
                "all_four_created_and_authoritatively_verified_before_enqueue": True,
                "completed_provenance_revalidated_before_enqueue": True,
                "enqueue_order": list(SUCCESSOR_ROLE_ORDER),
                "execute_requires_explicit_flag": "--execute",
                "dry_run_performs_remote_calls": False,
                "standalone_sources_are_import_smoked_without_repository": True,
            },
        }
    )
    return _seal(receipt)


def _execute_receipt(task_class: object) -> dict[str, object]:
    journal: dict[str, object] = {}
    try:
        result = _deploy_chain(task_class, journal=journal)
    except Exception as error:
        receipt = _base_receipt(
            mode="execute",
            status="failed_closed",
            remote_state_changed=bool(journal.get("task_ids")),
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
        mode="execute", status="deployed", remote_state_changed=True
    )
    receipt.update(result)
    return _seal(receipt)


def _execute_completed_successor_receipt(
    task_class: object,
    *,
    completed_provenance_task_id: str = COMPLETED_PROVENANCE_TASK_ID,
    amendment_binding: Mapping[str, object] | None = None,
    runtime_recovery_binding: Mapping[str, object] | None = None,
    original_queue_recovery_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    journal: dict[str, object] = {}
    bindings = _completed_successor_fixed_bindings(
        amendment_binding,
        runtime_recovery_binding,
        original_queue_recovery_binding,
    )
    try:
        result = _deploy_successors_from_completed_provenance(
            task_class,
            completed_provenance_task_id=completed_provenance_task_id,
            amendment_binding=amendment_binding,
            runtime_recovery_binding=runtime_recovery_binding,
            original_queue_recovery_binding=original_queue_recovery_binding,
            journal=journal,
        )
    except Exception as error:
        receipt = _base_receipt(
            mode="completed_provenance_successor_execute",
            status="failed_closed",
            remote_state_changed=bool(journal.get("created_task_ids")),
            fixed_bindings=bindings,
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
        mode="completed_provenance_successor_execute",
        status="deployed",
        remote_state_changed=True,
        fixed_bindings=bindings,
    )
    receipt.update(result)
    return _seal(receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly permit remote task creation and enqueueing.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Write-once sealed receipt path (defaults under artifacts/).",
    )
    parser.add_argument(
        "--resume-failed-receipt",
        type=Path,
        default=None,
        help=(
            "With --execute, reuse the exact five still-created shells in a "
            "sealed failed receipt; never creates replacement shells."
        ),
    )
    parser.add_argument(
        "--from-completed-provenance-task-id",
        type=str,
        default=None,
        help=(
            "Reuse the sealed completed P task and create only fresh W2/L2/A2/S2 "
            "successors. Without --execute this remains a local-only plan."
        ),
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
        "--evaluation-plan-amendment-receipt",
        type=Path,
        default=None,
        help="Executed sealed FFNet queue-only amendment receipt.",
    )
    parser.add_argument(
        "--runtime-recovery-receipt",
        type=Path,
        default=None,
        help="Executed exact FFNet/candidate runtime recovery receipt.",
    )
    parser.add_argument(
        "--original-queue-runtime-recovery-receipt",
        type=Path,
        default=None,
        help=(
            "Executed exact FFNet-only recovery receipt that retains GPU4-V100; "
            "mutually exclusive with all amendment inputs."
        ),
    )
    parser.add_argument(
        "--execute-token",
        default="",
        help="Exact token required only for amended successor execution.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.resume_failed_receipt is not None and not args.execute:
        parser.error("--resume-failed-receipt requires --execute")
    if (
        args.resume_failed_receipt is not None
        and args.from_completed_provenance_task_id is not None
    ):
        parser.error(
            "--resume-failed-receipt and --from-completed-provenance-task-id "
            "are mutually exclusive"
        )
    if args.execute and args.evaluation_recovery_receipt is None:
        raise DeploymentError(
            "--evaluation-recovery-receipt is required with --execute"
        )
    amended_requested = any(
        value is not None
        for value in (
            args.evaluation_plan_amendment_receipt,
            args.runtime_recovery_receipt,
        )
    )
    original_queue_requested = (
        args.original_queue_runtime_recovery_receipt is not None
    )
    if amended_requested and original_queue_requested:
        raise DeploymentError(
            "original-queue and amended runtime recoveries are mutually exclusive"
        )
    if original_queue_requested and (
        args.from_completed_provenance_task_id is None
        or args.evaluation_recovery_receipt is None
    ):
        raise DeploymentError(
            "original-queue deployment requires completed P and evaluation "
            "recovery receipts together"
        )
    if amended_requested and (
        args.from_completed_provenance_task_id is None
        or args.evaluation_plan_amendment_receipt is None
        or args.runtime_recovery_receipt is None
        or args.evaluation_recovery_receipt is None
    ):
        raise DeploymentError(
            "amended deployment requires completed P, evaluation recovery, "
            "plan amendment, and runtime recovery receipts together"
        )
    if args.execute and amended_requested:
        if args.execute_token != AMENDED_SUCCESSOR_EXECUTE_TOKEN:
            raise DeploymentError(
                f"exact execute token required: {AMENDED_SUCCESSOR_EXECUTE_TOKEN}"
            )
    elif args.execute and original_queue_requested:
        if args.execute_token != ORIGINAL_QUEUE_SUCCESSOR_EXECUTE_TOKEN:
            raise DeploymentError(
                "exact execute token required: "
                f"{ORIGINAL_QUEUE_SUCCESSOR_EXECUTE_TOKEN}"
            )
    elif args.execute_token:
        raise DeploymentError(
            "--execute-token is only valid for amended or original-recovery execute"
        )
    receipt_path = args.receipt or _default_receipt_path(
        execute=bool(args.execute),
        recovery=args.resume_failed_receipt is not None,
    )
    recovery_binding = (
        _validate_evaluation_recovery_receipt(args.evaluation_recovery_receipt)
        if args.evaluation_recovery_receipt is not None
        else None
    )
    # Both validators run before importing ClearML for execution and therefore
    # before any successor shell can be created.
    amendment_binding = (
        _validate_plan_amendment_receipt(args.evaluation_plan_amendment_receipt)
        if args.evaluation_plan_amendment_receipt is not None
        else None
    )
    runtime_recovery_binding = (
        _validate_runtime_recovery_receipt(args.runtime_recovery_receipt)
        if args.runtime_recovery_receipt is not None
        else None
    )
    original_queue_recovery_binding = (
        _validate_original_queue_runtime_recovery_receipt(
            args.original_queue_runtime_recovery_receipt
        )
        if args.original_queue_runtime_recovery_receipt is not None
        else None
    )
    if amendment_binding is not None and runtime_recovery_binding is not None:
        _validate_amended_runtime_pair(amendment_binding, runtime_recovery_binding)
    if (
        args.evaluation_recovery_receipt is not None
        and receipt_path.resolve() == args.evaluation_recovery_receipt.resolve()
    ):
        raise DeploymentError("deployment receipt must not overwrite recovery evidence")
    if (
        args.original_queue_runtime_recovery_receipt is not None
        and receipt_path.resolve()
        == args.original_queue_runtime_recovery_receipt.resolve()
    ):
        raise DeploymentError(
            "deployment receipt must not overwrite original runtime recovery evidence"
        )
    if not args.execute:
        if args.from_completed_provenance_task_id is not None:
            receipt = _completed_successor_dry_run_receipt(
                completed_provenance_task_id=(args.from_completed_provenance_task_id),
                amendment_binding=amendment_binding,
                runtime_recovery_binding=runtime_recovery_binding,
                original_queue_recovery_binding=(
                    original_queue_recovery_binding
                ),
            )
        else:
            receipt = _dry_run_receipt()
        if recovery_binding is not None:
            receipt = _bind_evaluation_recovery(receipt, recovery_binding)
        if amendment_binding is not None:
            receipt = _bind_plan_amendment(receipt, amendment_binding)
        if runtime_recovery_binding is not None:
            receipt = _bind_runtime_recovery(receipt, runtime_recovery_binding)
        if original_queue_recovery_binding is not None:
            receipt = _bind_original_queue_runtime_recovery(
                receipt, original_queue_recovery_binding
            )
        _write_new_receipt(receipt_path, receipt)
        print(json.dumps({"receipt": str(receipt_path), **receipt}, sort_keys=True))
        return 0

    try:
        from clearml import Task
    except ImportError as error:  # pragma: no cover - deployment environment only
        raise DeploymentError("--execute requires the ClearML client") from error
    try:
        if args.resume_failed_receipt is not None:
            if receipt_path.resolve() == args.resume_failed_receipt.resolve():
                raise DeploymentError(
                    "recovery receipt must not overwrite its failed source receipt"
                )
            receipt = _resume_failed_deployment(
                Task, failed_receipt_path=args.resume_failed_receipt
            )
        elif args.from_completed_provenance_task_id is not None:
            receipt = _execute_completed_successor_receipt(
                Task,
                completed_provenance_task_id=(args.from_completed_provenance_task_id),
                amendment_binding=amendment_binding,
                runtime_recovery_binding=runtime_recovery_binding,
                original_queue_recovery_binding=(
                    original_queue_recovery_binding
                ),
            )
        else:
            receipt = _execute_receipt(Task)
    except DeploymentError as error:
        receipt = getattr(error, "sealed_receipt", None)
        if isinstance(receipt, Mapping):
            if recovery_binding is None:  # defensive: execute preflight must bind it
                raise DeploymentError(
                    "evaluation recovery receipt binding is absent after preflight"
                ) from error
            receipt = _bind_evaluation_recovery(receipt, recovery_binding)
            if amendment_binding is not None:
                receipt = _bind_plan_amendment(receipt, amendment_binding)
            if runtime_recovery_binding is not None:
                receipt = _bind_runtime_recovery(receipt, runtime_recovery_binding)
            if original_queue_recovery_binding is not None:
                receipt = _bind_original_queue_runtime_recovery(
                    receipt, original_queue_recovery_binding
                )
            _write_new_receipt(receipt_path, receipt)
        raise
    if recovery_binding is None:  # defensive: execute preflight must bind it
        raise DeploymentError(
            "evaluation recovery receipt binding is absent after preflight"
        )
    receipt = _bind_evaluation_recovery(receipt, recovery_binding)
    if amendment_binding is not None:
        receipt = _bind_plan_amendment(receipt, amendment_binding)
    if runtime_recovery_binding is not None:
        receipt = _bind_runtime_recovery(receipt, runtime_recovery_binding)
    if original_queue_recovery_binding is not None:
        receipt = _bind_original_queue_runtime_recovery(
            receipt, original_queue_recovery_binding
        )
    _write_new_receipt(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "COMPLETED_PROVENANCE_TASK_ID",
    "DeploymentError",
    "ORIGINAL_QUEUE_SUCCESSOR_EXECUTE_TOKEN",
    "ROLE_ORDER",
    "SUCCESSOR_ROLE_ORDER",
    "_completed_successor_dry_run_receipt",
    "_deploy_chain",
    "_deploy_successors_from_completed_provenance",
    "_dry_run_receipt",
    "_execute_completed_successor_receipt",
    "_execute_receipt",
    "_render_sources",
    "_resume_failed_deployment",
    "main",
)
