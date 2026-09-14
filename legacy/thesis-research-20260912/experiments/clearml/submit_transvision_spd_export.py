#!/usr/bin/env python3
"""Validate a TransVision SPD fixed-detection export request without submitting it.

This module is deliberately a local preflight only.  It imports no ClearML SDK,
performs no network access, creates no task, and never sets the explicit dataset
access acknowledgement.  The current v1 contract remains terminally blocked
until a reviewed production backend, runtime, and successor contract exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "experiments/clearml/contracts/transvision-spd-detection-export-v1.json"
)
TEMPLATE_RELATIVE_PATH = Path(
    "experiments/clearml/configs/transvision_spd_detection_export.json"
)
FIXED_DETECTION_RELATIVE_PATH = Path(
    "experiments/clearml/protocols/fixed-detections-v1.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "839f6daac240d1b818534606f55396115667d60c9374d62f54775688c37c0967"
)
EXPECTED_FIXED_DETECTION_SHA256 = (
    "4da0606aefbf1697bbd4f856c9445f175e0ee1e091c32afc6be13da5da866022"
)
CONTRACT_ID = "RTPV2X-TRANSVISION-SPD-DETECTION-EXPORT-v1"
TRANSVISION_REPOSITORY_URL = "https://github.com/its-research/transvision.git"
ACKNOWLEDGEMENT_VARIABLE = "V2XSEQ_ACCESS_ACKNOWLEDGED"
ACKNOWLEDGEMENT_VALUE = "1"
SAFE_QUEUE = "GPU4-A100"

SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
OCI_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
SAFE_IMAGE = re.compile(r"[A-Za-z0-9./_-]+@sha256:[0-9a-f]{64}")
MAX_JSON_BYTES = 256 * 1024
MAX_CONTROL_BYTES = 32 * 1024 * 1024

TOP_LEVEL_FIELDS = {
    "schema_version",
    "config_kind",
    "contract",
    "scientific_claim_allowed",
    "execution_enabled",
    "task_creation_allowed",
    "execution_status",
    "clearml",
    "data_access",
    "rights",
    "code",
    "detector",
    "container",
    "mounts",
    "output",
    "runtime",
}
CONTRACT_BINDING_FIELDS = {"path", "sha256"}
CLEARML_FIELDS = {
    "project",
    "queue",
    "authenticated_live_preflight_required",
    "authenticated_live_preflight_implemented",
    "require_task_id",
}
DATA_ACCESS_FIELDS = {
    "dataset",
    "acknowledgement_environment_variable",
    "acknowledgement_required_value",
    "receipt_path",
    "receipt_sha256",
    "release_identity_manifest_path",
    "release_identity_manifest_sha256",
    "split_manifest_path",
    "split_manifest_sha256",
}
RIGHTS_FIELDS = {
    "subject",
    "authorization_basis",
    "execute_allowed",
    "modify_allowed",
    "redistribute_allowed",
    "receipt_path",
    "receipt_sha256",
}
CODE_FIELDS = {
    "thesis_commit",
    "transvision_repository_url",
    "transvision_root",
    "transvision_commit",
    "clean_worktrees_required",
    "source_tree_seal_path",
    "source_tree_seal_sha256",
    "source_tree_sha256",
}
DETECTOR_FIELDS = {
    "model_id",
    "config_path",
    "config_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "environment_manifest_path",
    "environment_manifest_sha256",
    "backend_status",
    "runtime_status",
}
CONTAINER_FIELDS = {"image_reference", "final_oci_digest"}
MOUNT_FIELDS = {
    "dataset_root",
    "dataset_read_only_required",
    "checkpoint_root",
    "checkpoint_read_only_required",
    "writable_output_separate_required",
}
OUTPUT_FIELDS = {
    "contract_path",
    "contract_sha256",
    "diagnostic_only",
    "h1_support_allowed",
    "formal_fixed_detection_input_allowed",
    "score_is_reliability",
    "ground_truth_forbidden",
    "evaluation_forbidden",
    "registry_write_allowed",
}
RUNTIME_FIELDS = {
    "diagnostic_assertion_only",
    "require_queue",
    "require_queue_accepting_tasks",
    "minimum_online_workers",
    "require_visible_gpu_count",
    "require_device_name_substring",
}
OBSERVATION_FIELDS = {
    "queue_name",
    "queue_accepting_tasks",
    "online_worker_count",
    "visible_gpu_count",
    "device_name",
    "active_same_name_task_count",
}
RIGHTS_SUBJECTS = {
    "transvision_source",
    "detector_config",
    "detector_checkpoint",
}
MODEL_CONFIG_PATHS = {
    "coformernet_controlled_adaptation": (
        "configs/resilient_v2x/baselines/coformernet.py"
    ),
    "resilient_v2x": "configs/resilient_v2x/dair_resilient_v2x.py",
}


class ExportPreflightError(RuntimeError):
    """A stable, non-sensitive reason why local preflight is blocked."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise ExportPreflightError(code)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        _fail("control_asset_unreadable")
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _is_commit(value: object) -> bool:
    return isinstance(value, str) and COMMIT.fullmatch(value) is not None


def _object(value: object, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(code)
    return value


def _exact_fields(value: dict[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constants are forbidden")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON keys are forbidden")
        result[key] = value
    return result


def _load_json(path: Path, *, maximum_bytes: int, code: str) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            _fail(code)
        size = path.stat().st_size
        if size <= 0 or size > maximum_bytes:
            _fail(code)
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, ValueError):
        _fail(code)
    return _object(payload, code)


def _repository_contract(root: Path) -> dict[str, Any]:
    path = root / CONTRACT_RELATIVE_PATH
    if _sha256_file(path) != EXPECTED_CONTRACT_SHA256:
        _fail("repository_contract_hash_mismatch")
    contract = _load_json(
        path,
        maximum_bytes=MAX_JSON_BYTES,
        code="repository_contract_invalid",
    )
    required_fields = {
        "schema_version",
        "contract_id",
        "status",
        "scientific_claim_allowed",
        "task_creation_allowed",
        "scope",
        "claim_boundary",
        "submission_boundary",
        "resolved_config_policy",
        "authorization_requirements",
        "provenance_requirements",
        "mount_requirements",
        "live_observation_requirements",
        "failure_policy",
    }
    _exact_fields(contract, required_fields, "repository_contract_fields_invalid")
    if (
        contract["schema_version"] != 1
        or contract["contract_id"] != CONTRACT_ID
        or contract["scientific_claim_allowed"] is not False
        or contract["task_creation_allowed"] is not False
    ):
        _fail("repository_contract_boundary_invalid")
    scope = _object(contract["scope"], "repository_contract_scope_invalid")
    if (
        scope.get("output_contract_path")
        != FIXED_DETECTION_RELATIVE_PATH.as_posix()
        or scope.get("output_contract_sha256")
        != EXPECTED_FIXED_DETECTION_SHA256
        or scope.get("ground_truth_visible_to_exporter") is not False
        or scope.get("registry_write_allowed") is not False
    ):
        _fail("repository_contract_scope_invalid")
    claims = _object(
        contract["claim_boundary"], "repository_contract_claim_boundary_invalid"
    )
    if (
        claims.get("diagnostic_only") is not True
        or claims.get("h1_support_allowed") is not False
        or claims.get("formal_fixed_detection_input_allowed") is not False
        or claims.get("score_is_reliability") is not False
        or "v2" not in str(claims.get("successor_policy", ""))
    ):
        _fail("repository_contract_claim_boundary_invalid")
    boundary = _object(
        contract["submission_boundary"],
        "repository_contract_submission_boundary_invalid",
    )
    if (
        boundary.get("network_access_allowed_by_preflight") is not False
        or boundary.get("clearml_sdk_import_allowed_by_preflight") is not False
        or boundary.get("task_creation_implemented") is not False
        or boundary.get("task_enqueue_implemented") is not False
        or boundary.get("production_backend_status") != "unresolved"
        or boundary.get("production_runtime_status") != "unresolved"
    ):
        _fail("repository_contract_submission_boundary_invalid")
    return contract


def _validate_common_config(document: dict[str, Any]) -> None:
    _exact_fields(document, TOP_LEVEL_FIELDS, "config_fields_invalid")
    if document["schema_version"] != 1:
        _fail("config_schema_invalid")
    binding = _object(document["contract"], "contract_binding_invalid")
    _exact_fields(binding, CONTRACT_BINDING_FIELDS, "contract_binding_invalid")
    if (
        binding["path"] != CONTRACT_RELATIVE_PATH.as_posix()
        or binding["sha256"] != EXPECTED_CONTRACT_SHA256
    ):
        _fail("contract_binding_invalid")
    if document["scientific_claim_allowed"] is not False:
        _fail("scientific_claim_boundary_invalid")

    clearml = _object(document["clearml"], "clearml_config_invalid")
    _exact_fields(clearml, CLEARML_FIELDS, "clearml_config_invalid")
    if clearml != {
        "project": "Thesis/RTP-V2X",
        "queue": SAFE_QUEUE,
        "authenticated_live_preflight_required": True,
        "authenticated_live_preflight_implemented": False,
        "require_task_id": True,
    }:
        _fail("clearml_config_invalid")

    data_access = _object(document["data_access"], "data_access_fields_invalid")
    _exact_fields(data_access, DATA_ACCESS_FIELDS, "data_access_fields_invalid")
    if (
        data_access["dataset"] != "V2X-Seq-SPD"
        or data_access["acknowledgement_environment_variable"]
        != ACKNOWLEDGEMENT_VARIABLE
        or data_access["acknowledgement_required_value"] != ACKNOWLEDGEMENT_VALUE
    ):
        _fail("data_access_contract_invalid")

    rights = document["rights"]
    if not isinstance(rights, list) or len(rights) != len(RIGHTS_SUBJECTS):
        _fail("rights_entries_invalid")
    subjects: set[str] = set()
    for entry_value in rights:
        entry = _object(entry_value, "rights_entry_invalid")
        _exact_fields(entry, RIGHTS_FIELDS, "rights_entry_invalid")
        subject = entry["subject"]
        if not isinstance(subject, str) or subject not in RIGHTS_SUBJECTS:
            _fail("rights_subject_invalid")
        subjects.add(subject)
        for field in ("execute_allowed", "modify_allowed", "redistribute_allowed"):
            if not isinstance(entry[field], bool):
                _fail("rights_status_invalid")
    if subjects != RIGHTS_SUBJECTS:
        _fail("rights_subject_invalid")

    code = _object(document["code"], "code_fields_invalid")
    _exact_fields(code, CODE_FIELDS, "code_fields_invalid")
    if (
        code["transvision_repository_url"] != TRANSVISION_REPOSITORY_URL
        or code["clean_worktrees_required"] is not True
    ):
        _fail("code_policy_invalid")

    detector = _object(document["detector"], "detector_fields_invalid")
    _exact_fields(detector, DETECTOR_FIELDS, "detector_fields_invalid")
    container = _object(document["container"], "container_fields_invalid")
    _exact_fields(container, CONTAINER_FIELDS, "container_fields_invalid")

    mounts = _object(document["mounts"], "mount_fields_invalid")
    _exact_fields(mounts, MOUNT_FIELDS, "mount_fields_invalid")
    if (
        mounts["dataset_read_only_required"] is not True
        or mounts["checkpoint_read_only_required"] is not True
        or mounts["writable_output_separate_required"] is not True
    ):
        _fail("mount_policy_invalid")

    output = _object(document["output"], "output_fields_invalid")
    _exact_fields(output, OUTPUT_FIELDS, "output_fields_invalid")
    if output != {
        "contract_path": FIXED_DETECTION_RELATIVE_PATH.as_posix(),
        "contract_sha256": EXPECTED_FIXED_DETECTION_SHA256,
        "diagnostic_only": True,
        "h1_support_allowed": False,
        "formal_fixed_detection_input_allowed": False,
        "score_is_reliability": False,
        "ground_truth_forbidden": True,
        "evaluation_forbidden": True,
        "registry_write_allowed": False,
    }:
        _fail("fixed_detection_contract_binding_invalid")

    runtime = _object(document["runtime"], "runtime_fields_invalid")
    _exact_fields(runtime, RUNTIME_FIELDS, "runtime_fields_invalid")
    if runtime != {
        "diagnostic_assertion_only": True,
        "require_queue": SAFE_QUEUE,
        "require_queue_accepting_tasks": True,
        "minimum_online_workers": 1,
        "require_visible_gpu_count": 1,
        "require_device_name_substring": "A100",
    }:
        _fail("runtime_policy_invalid")


def validate_repository_template(root: Path = REPOSITORY_ROOT) -> dict[str, object]:
    """Validate the frozen repository template without resolving external input."""
    contract = _repository_contract(root)
    fixed_path = root / FIXED_DETECTION_RELATIVE_PATH
    if _sha256_file(fixed_path) != EXPECTED_FIXED_DETECTION_SHA256:
        _fail("fixed_detection_contract_hash_mismatch")
    template = _load_json(
        root / TEMPLATE_RELATIVE_PATH,
        maximum_bytes=MAX_JSON_BYTES,
        code="repository_template_invalid",
    )
    _validate_common_config(template)
    if (
        template["config_kind"]
        != "transvision_spd_detection_export_pending_template"
        or template["execution_enabled"] is not False
        or template["task_creation_allowed"] is not False
        or not isinstance(template["execution_status"], str)
        or not template["execution_status"].startswith("disabled_pending_")
    ):
        _fail("repository_template_must_remain_disabled")
    detector = _object(template["detector"], "repository_template_invalid")
    if (
        detector["backend_status"] != "unresolved"
        or detector["runtime_status"] != "unresolved"
    ):
        _fail("repository_template_backend_boundary_invalid")
    boundary = _object(
        contract["submission_boundary"],
        "repository_contract_submission_boundary_invalid",
    )
    return {
        "contract_id": CONTRACT_ID,
        "diagnostic_only": True,
        "execution_enabled": False,
        "formal_fixed_detection_input_allowed": False,
        "h1_support_allowed": False,
        "authenticated_live_preflight_implemented": False,
        "network_access_performed": False,
        "preflight": "template_valid",
        "production_backend_status": boundary["production_backend_status"],
        "production_runtime_status": boundary["production_runtime_status"],
        "scientific_claim_allowed": False,
        "score_is_reliability": False,
        "submission_performed": False,
        "task_creation_allowed": False,
    }


def filesystem_is_read_only(path: Path) -> bool:
    """Return whether *path* is on a filesystem mounted read-only."""
    try:
        return bool(os.statvfs(path).f_flag & os.ST_RDONLY)
    except OSError:
        _fail("mount_read_only_state_unavailable")


def _external_config_path(
    path: Path,
    *,
    contracts_root: Path,
    repository_root: Path,
    read_only_probe: Callable[[Path], bool],
) -> tuple[Path, Path]:
    try:
        if contracts_root.is_symlink() or not contracts_root.is_dir():
            _fail("external_contracts_root_invalid")
        root = contracts_root.resolve(strict=True)
        repository = repository_root.resolve(strict=True)
        candidate = path.resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        _fail("external_resolved_config_required")
    try:
        candidate.relative_to(repository)
    except ValueError:
        pass
    else:
        _fail("external_resolved_config_required")
    if path.is_symlink() or not path.is_file():
        _fail("external_resolved_config_required")
    if not read_only_probe(root):
        _fail("external_contracts_root_not_read_only")
    return candidate, root


def _relative_control_asset(
    root: Path,
    relative_value: object,
    expected_sha256: object,
    *,
    code: str,
) -> Path:
    if not isinstance(relative_value, str) or not _is_sha256(expected_sha256):
        _fail(code)
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        _fail(code)
    path = root.joinpath(*relative.parts)
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            _fail(code)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        _fail(code)
    if path.is_symlink() or not path.is_file():
        _fail(code)
    try:
        if path.stat().st_size <= 0 or path.stat().st_size > MAX_CONTROL_BYTES:
            _fail(code)
    except OSError:
        _fail(code)
    if _sha256_file(path) != expected_sha256:
        _fail(code)
    return resolved


def _absolute_directory(value: object, *, code: str) -> Path:
    if not isinstance(value, str):
        _fail(code)
    path = Path(value)
    try:
        if not path.is_absolute() or path.is_symlink() or not path.is_dir():
            _fail(code)
        return path.resolve(strict=True)
    except (OSError, RuntimeError):
        _fail(code)


def _relative_mounted_asset(
    root: Path,
    relative_value: object,
    expected_sha256: object,
    *,
    code: str,
) -> Path:
    if not isinstance(relative_value, str) or not _is_sha256(expected_sha256):
        _fail(code)
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        _fail(code)
    path = root.joinpath(*relative.parts)
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            _fail(code)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        _fail(code)
    if path.is_symlink() or not path.is_file():
        _fail(code)
    if _sha256_file(path) != expected_sha256:
        _fail(code)
    return resolved


def _git_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG_PARAMETERS",
    ):
        environment.pop(name, None)
    for name in tuple(environment):
        if re.fullmatch(r"GIT_CONFIG_(KEY|VALUE)_\d+", name):
            environment.pop(name, None)
    environment["GIT_CONFIG_COUNT"] = "0"
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _git_output(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                f"core.hooksPath={os.devnull}",
                "-C",
                str(root),
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        _fail("git_identity_unverifiable")
    return result.stdout.strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                f"core.hooksPath={os.devnull}",
                "-C",
                str(root),
                *arguments,
            ],
            check=True,
            capture_output=True,
            timeout=20,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        _fail("git_identity_unverifiable")
    return result.stdout


def _verify_clean_exact_commit(root: Path, expected: object, *, code: str) -> None:
    if not _is_commit(expected):
        _fail(code)
    if _git_output(root, "rev-parse", "HEAD") != expected:
        _fail(code)
    tree = _tracked_source_tree(root)
    index = _tracked_index(root)
    if index != tree:
        _fail(code)
    if any(
        row
        for row in _git_bytes(
            root, "ls-files", "-z", "--others", "--exclude-standard"
        ).split(b"\0")
    ):
        _fail(code)
    object_format = _git_output(root, "rev-parse", "--show-object-format")
    if object_format not in {"sha1", "sha256"}:
        _fail(code)
    for relative_path, (git_mode, object_id) in tree.items():
        observed_mode, observed_id = _raw_worktree_blob_identity(
            root,
            relative_path,
            object_format=object_format,
            code=code,
        )
        if (observed_mode, observed_id) != (git_mode, object_id):
            _fail(code)


def _canonical_repository_url(value: str) -> str | None:
    if value == TRANSVISION_REPOSITORY_URL:
        return TRANSVISION_REPOSITORY_URL
    if value == "git@github.com:its-research/transvision.git":
        return TRANSVISION_REPOSITORY_URL
    if value == "ssh://git@github.com/its-research/transvision.git":
        return TRANSVISION_REPOSITORY_URL
    return None


def _verify_transvision_remote(root: Path) -> None:
    urls = [
        row.strip()
        for row in _git_output(root, "remote", "get-url", "--all", "origin").splitlines()
        if row.strip()
    ]
    if not urls or any(_canonical_repository_url(url) is None for url in urls):
        _fail("transvision_remote_identity_mismatch")


def _tracked_source_tree(root: Path) -> dict[str, tuple[str, str]]:
    raw = _git_bytes(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    observed: dict[str, tuple[str, str]] = {}
    for row in raw.split(b"\0"):
        if not row:
            continue
        try:
            metadata, path_bytes = row.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            path_value = path_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            _fail("source_tree_seal_invalid")
        relative = PurePosixPath(path_value)
        if (
            object_type != "blob"
            or mode not in {"100644", "100755"}
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", object_id) is None
            or relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or relative.as_posix() != path_value
            or path_value in observed
        ):
            _fail("source_tree_seal_invalid")
        observed[path_value] = (mode, object_id)
    if not observed:
        _fail("source_tree_seal_invalid")
    return observed


def _tracked_index(root: Path) -> dict[str, tuple[str, str]]:
    raw = _git_bytes(root, "ls-files", "-z", "--stage")
    observed: dict[str, tuple[str, str]] = {}
    for row in raw.split(b"\0"):
        if not row:
            continue
        try:
            metadata, path_bytes = row.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split(" ")
            path_value = path_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            _fail("git_identity_unverifiable")
        if (
            stage != "0"
            or mode not in {"100644", "100755"}
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", object_id) is None
            or path_value in observed
        ):
            _fail("git_identity_unverifiable")
        observed[path_value] = (mode, object_id)
    return observed


def _raw_worktree_blob_identity(
    root: Path,
    relative_value: str,
    *,
    object_format: str,
    code: str,
) -> tuple[str, str]:
    relative = PurePosixPath(relative_value)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.as_posix() != relative_value
    ):
        _fail(code)
    path = root.joinpath(*relative.parts)
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            _fail(code)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        _fail(code)
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                _fail(code)
            hasher = hashlib.new(object_format)
            hasher.update(f"blob {before.st_size}\0".encode("ascii"))
            observed_size = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                observed_size += len(chunk)
                hasher.update(chunk)
            after = os.fstat(stream.fileno())
    except (OSError, ValueError):
        _fail(code)
    if (
        observed_size != before.st_size
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
        or before.st_dev != after.st_dev
    ):
        _fail(code)
    git_mode = "100755" if before.st_mode & stat.S_IXUSR else "100644"
    return git_mode, hasher.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        _fail("source_tree_seal_invalid")
    return (rendered + "\n").encode("utf-8")


def _validate_source_tree_seal(
    path: Path,
    *,
    source_root: Path,
    expected_commit: str,
    expected_payload_sha256: str,
) -> None:
    seal = _load_json(
        path,
        maximum_bytes=MAX_CONTROL_BYTES,
        code="source_tree_seal_invalid",
    )
    fields = {"schema_version", "seal_kind", "git_commit", "entries", "manifest_sha256"}
    _exact_fields(seal, fields, "source_tree_seal_invalid")
    if (
        seal["schema_version"] != 1
        or seal["seal_kind"] != "source_tree"
        or seal["git_commit"] != expected_commit
        or seal["manifest_sha256"] != expected_payload_sha256
    ):
        _fail("source_tree_seal_invalid")
    payload = dict(seal)
    del payload["manifest_sha256"]
    if hashlib.sha256(_canonical_json_bytes(payload)).hexdigest() != (
        expected_payload_sha256
    ):
        _fail("source_tree_seal_invalid")
    entries = seal["entries"]
    if not isinstance(entries, list) or not entries:
        _fail("source_tree_seal_invalid")
    tracked_tree = _tracked_source_tree(source_root)
    if len(entries) != len(tracked_tree):
        _fail("source_tree_seal_invalid")
    observed_paths: set[str] = set()
    for entry_value in entries:
        entry = _object(entry_value, "source_tree_seal_invalid")
        _exact_fields(
            entry,
            {"path", "size", "sha256", "git_mode", "git_blob"},
            "source_tree_seal_invalid",
        )
        relative_value = entry["path"]
        if not isinstance(relative_value, str) or not _is_sha256(entry["sha256"]):
            _fail("source_tree_seal_invalid")
        relative = PurePosixPath(relative_value)
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or relative.as_posix() != relative_value
            or relative_value in observed_paths
        ):
            _fail("source_tree_seal_invalid")
        observed_paths.add(relative_value)
        source = _relative_mounted_asset(
            source_root,
            relative_value,
            entry["sha256"],
            code="source_tree_seal_invalid",
        )
        size = entry["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _fail("source_tree_seal_invalid")
        if source.stat().st_size != size:
            _fail("source_tree_seal_invalid")
        if entry["git_mode"] not in {"100644", "100755"}:
            _fail("source_tree_seal_invalid")
        git_blob = entry["git_blob"]
        if (
            not isinstance(git_blob, str)
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", git_blob) is None
        ):
            _fail("source_tree_seal_invalid")
        if tracked_tree.get(relative_value) != (entry["git_mode"], git_blob):
            _fail("source_tree_seal_invalid")
    if observed_paths != set(tracked_tree):
        _fail("source_tree_seal_invalid")


def _validate_rights(
    rights_value: object,
    *,
    contracts_root: Path,
) -> None:
    rights = rights_value if isinstance(rights_value, list) else []
    observed: set[str] = set()
    for entry_value in rights:
        entry = _object(entry_value, "rights_entry_invalid")
        subject = entry["subject"]
        observed.add(subject)
        if entry["authorization_basis"] not in {
            "public_license",
            "private_authorization",
        }:
            _fail("rights_authorization_unresolved")
        if entry["execute_allowed"] is not True:
            _fail("rights_execute_not_authorized")
        if subject in {"transvision_source", "detector_config"}:
            if entry["modify_allowed"] is not True:
                _fail("rights_modify_not_authorized")
        _relative_control_asset(
            contracts_root,
            entry["receipt_path"],
            entry["receipt_sha256"],
            code="rights_receipt_invalid",
        )
    if observed != RIGHTS_SUBJECTS:
        _fail("rights_subject_invalid")


def _validate_diagnostic_observation(
    observation_value: Mapping[str, object],
    runtime: dict[str, Any],
) -> None:
    """Validate a caller assertion without treating it as ClearML evidence."""

    if runtime.get("diagnostic_assertion_only") is not True:
        _fail("runtime_policy_invalid")
    observation = dict(observation_value)
    _exact_fields(observation, OBSERVATION_FIELDS, "live_observation_invalid")
    if observation["queue_name"] != runtime["require_queue"]:
        _fail("a100_queue_mismatch")
    if observation["queue_accepting_tasks"] is not True:
        _fail("a100_queue_not_accepting_tasks")
    workers = observation["online_worker_count"]
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < runtime["minimum_online_workers"]
    ):
        _fail("a100_worker_unavailable")
    visible = observation["visible_gpu_count"]
    if (
        isinstance(visible, bool)
        or not isinstance(visible, int)
        or visible != runtime["require_visible_gpu_count"]
    ):
        _fail("a100_visible_gpu_count_invalid")
    device = observation["device_name"]
    if (
        not isinstance(device, str)
        or runtime["require_device_name_substring"] not in device
    ):
        _fail("a100_device_required")
    duplicates = observation["active_same_name_task_count"]
    if isinstance(duplicates, bool) or not isinstance(duplicates, int) or duplicates != 0:
        _fail("active_same_name_task_exists")


def preflight_resolved_config(
    config_path: Path,
    *,
    expected_config_sha256: str,
    contracts_root: Path,
    observation: Mapping[str, object],
    environment: Mapping[str, str] | None = None,
    repository_root: Path = REPOSITORY_ROOT,
    read_only_probe: Callable[[Path], bool] = filesystem_is_read_only,
) -> None:
    """Validate all local gates, then stop at the unresolved v1 backend boundary.

    A successful return is intentionally impossible for this contract version.
    The function raises :class:`ExportPreflightError` before any task can exist.
    """
    contract = _repository_contract(repository_root)
    if _sha256_file(repository_root / FIXED_DETECTION_RELATIVE_PATH) != (
        EXPECTED_FIXED_DETECTION_SHA256
    ):
        _fail("fixed_detection_contract_hash_mismatch")
    config, resolved_root = _external_config_path(
        config_path,
        contracts_root=contracts_root,
        repository_root=repository_root,
        read_only_probe=read_only_probe,
    )
    if not _is_sha256(expected_config_sha256):
        _fail("resolved_config_expected_hash_invalid")
    if _sha256_file(config) != expected_config_sha256:
        _fail("resolved_config_hash_mismatch")
    document = _load_json(
        config,
        maximum_bytes=MAX_JSON_BYTES,
        code="resolved_config_invalid",
    )
    _validate_common_config(document)
    if (
        document["config_kind"]
        != "transvision_spd_detection_export_resolved"
        or document["execution_enabled"] is not False
        or document["task_creation_allowed"] is not False
        or document["execution_status"]
        != "resolved_for_local_preflight_only_terminally_blocked"
    ):
        _fail("external_resolved_config_required")

    values = os.environ if environment is None else environment
    if values.get(ACKNOWLEDGEMENT_VARIABLE) != ACKNOWLEDGEMENT_VALUE:
        _fail("explicit_data_access_acknowledgement_required")

    data_access = _object(document["data_access"], "data_access_fields_invalid")
    for path_field, hash_field, code in (
        ("receipt_path", "receipt_sha256", "data_access_receipt_invalid"),
        (
            "release_identity_manifest_path",
            "release_identity_manifest_sha256",
            "release_identity_manifest_invalid",
        ),
        ("split_manifest_path", "split_manifest_sha256", "split_manifest_invalid"),
    ):
        _relative_control_asset(
            resolved_root,
            data_access[path_field],
            data_access[hash_field],
            code=code,
        )
    _validate_rights(document["rights"], contracts_root=resolved_root)

    code = _object(document["code"], "code_fields_invalid")
    source_root = _absolute_directory(
        code["transvision_root"], code="transvision_root_invalid"
    )
    _verify_clean_exact_commit(
        repository_root,
        code["thesis_commit"],
        code="thesis_commit_or_worktree_mismatch",
    )
    _verify_clean_exact_commit(
        source_root,
        code["transvision_commit"],
        code="transvision_commit_or_worktree_mismatch",
    )
    _verify_transvision_remote(source_root)
    if not _is_sha256(code["source_tree_sha256"]):
        _fail("source_tree_hash_invalid")
    source_seal_path = _relative_control_asset(
        resolved_root,
        code["source_tree_seal_path"],
        code["source_tree_seal_sha256"],
        code="source_tree_seal_invalid",
    )
    _validate_source_tree_seal(
        source_seal_path,
        source_root=source_root,
        expected_commit=code["transvision_commit"],
        expected_payload_sha256=code["source_tree_sha256"],
    )

    mounts = _object(document["mounts"], "mount_fields_invalid")
    dataset_root = _absolute_directory(
        mounts["dataset_root"], code="dataset_mount_invalid"
    )
    checkpoint_root = _absolute_directory(
        mounts["checkpoint_root"], code="checkpoint_mount_invalid"
    )
    if dataset_root == checkpoint_root:
        _fail("dataset_and_checkpoint_mounts_must_be_separate")
    if not read_only_probe(dataset_root):
        _fail("dataset_mount_not_read_only")
    if not read_only_probe(checkpoint_root):
        _fail("checkpoint_mount_not_read_only")

    detector = _object(document["detector"], "detector_fields_invalid")
    model_id = detector["model_id"]
    if model_id not in MODEL_CONFIG_PATHS:
        _fail("detector_model_invalid")
    if detector["config_path"] != MODEL_CONFIG_PATHS[model_id]:
        _fail("detector_config_path_invalid")
    _relative_mounted_asset(
        source_root,
        detector["config_path"],
        detector["config_sha256"],
        code="detector_config_hash_mismatch",
    )
    _relative_mounted_asset(
        checkpoint_root,
        detector["checkpoint_path"],
        detector["checkpoint_sha256"],
        code="detector_checkpoint_hash_mismatch",
    )
    _relative_control_asset(
        resolved_root,
        detector["environment_manifest_path"],
        detector["environment_manifest_sha256"],
        code="environment_manifest_invalid",
    )

    container = _object(document["container"], "container_fields_invalid")
    digest = container["final_oci_digest"]
    image = container["image_reference"]
    if not isinstance(digest, str) or OCI_DIGEST.fullmatch(digest) is None:
        _fail("final_oci_digest_invalid")
    if (
        not isinstance(image, str)
        or SAFE_IMAGE.fullmatch(image) is None
        or not image.endswith("@" + digest)
    ):
        _fail("digest_pinned_oci_image_required")

    runtime = _object(document["runtime"], "runtime_fields_invalid")
    _validate_diagnostic_observation(observation, runtime)

    boundary = _object(
        contract["submission_boundary"],
        "repository_contract_submission_boundary_invalid",
    )
    if (
        detector["backend_status"] != "unresolved"
        or detector["runtime_status"] != "unresolved"
        or boundary["production_backend_status"] != "unresolved"
        or boundary["production_runtime_status"] != "unresolved"
    ):
        _fail("backend_runtime_contract_drift")
    _fail("production_backend_and_runtime_unresolved")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-template")
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--expected-config-sha256", required=True)
    preflight.add_argument("--contracts-root", type=Path, required=True)
    preflight.add_argument("--observed-queue", required=True)
    preflight.add_argument("--queue-accepting-tasks", action="store_true")
    preflight.add_argument("--online-workers", type=int, required=True)
    preflight.add_argument("--visible-gpus", type=int, required=True)
    preflight.add_argument("--device-name", required=True)
    preflight.add_argument("--active-same-name-tasks", type=int, required=True)
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        if args.command == "check-template":
            report = validate_repository_template()
            print(json.dumps(report, sort_keys=True))
            return 0
        observation = {
            "queue_name": args.observed_queue,
            "queue_accepting_tasks": args.queue_accepting_tasks,
            "online_worker_count": args.online_workers,
            "visible_gpu_count": args.visible_gpus,
            "device_name": args.device_name,
            "active_same_name_task_count": args.active_same_name_tasks,
        }
        preflight_resolved_config(
            args.config,
            expected_config_sha256=args.expected_config_sha256,
            contracts_root=args.contracts_root,
            observation=observation,
        )
    except ExportPreflightError as exc:
        print(
            json.dumps(
                {
                    "network_access_performed": False,
                    "preflight": "blocked",
                    "reason": exc.code,
                    "submission_performed": False,
                    "task_created": False,
                },
                sort_keys=True,
            )
        )
        return 2
    raise AssertionError("v1 preflight must never reach task creation")


if __name__ == "__main__":
    raise SystemExit(main())
