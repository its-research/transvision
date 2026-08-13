#!/usr/bin/env python3
"""Prepare and explicitly launch the sealed round2 P0 candidate on A100.

The local prepare command extends the immutable E1/E2/E3 source package by
exactly one configuration file. Remote-writing commands require distinct exact
tokens and never modify the existing E1/E2/E3 Dataset, template, or tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterator

try:
    from tools.resilient_v2x import clearml_5090_training_controller as training
    from tools.resilient_v2x import clearml_sota_candidate_controller as candidate
    from tools.resilient_v2x import clearml_sota_fastlane_launcher as fastlane
    from tools.resilient_v2x import package_clearml_source as source_packager
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from tools.resilient_v2x import clearml_5090_training_controller as training
    from tools.resilient_v2x import clearml_sota_candidate_controller as candidate
    from tools.resilient_v2x import clearml_sota_fastlane_launcher as fastlane
    from tools.resilient_v2x import package_clearml_source as source_packager


ROOT = Path(__file__).resolve().parents[2]
BASE_SOURCE_DIR = ROOT / "artifacts/resilient_v2x/sota_candidate_source_e1_e2_e3"
BASE_SOURCE_INVENTORY = BASE_SOURCE_DIR / "source-inventory.json"
BASE_SOURCE_ARCHIVE = (
    BASE_SOURCE_DIR / candidate.CANDIDATE_SOURCE_PACKAGE["archive_name"]
)
OUTPUT_DIR = ROOT / "artifacts/resilient_v2x/sota_round2_p0"
PACKAGE_MANIFEST_NAME = "round2-p0-source-package.json"
DEPLOYMENT_PLAN_NAME = "round2-p0-deployment-plan.json"

P0_EXPERIMENT = "support_residual_no_reliability_linear"
P0_CONFIG = (
    "configs/resilient_v2x/improvements/support_residual_no_reliability_linear.py"
)
P0_IDENTITY = "dair_improvement_support_residual_no_reliability_linear"
SOURCE_PROJECT = "ResilientV2X/Source"
TRAINING_PROJECT = "ResilientV2X/Training"
TRAINING_PROJECT_ID = "6e43f972e5ea4cee901a7c8855fce8cd"
SOURCE_TRANSITION_ARTIFACT = "sota_round2_p0_source_transition"
LAUNCH_RECEIPT_ARTIFACT = "sota_round2_p0_launch_receipt"
SOURCE_DATASET_PREFIX = "ResilientV2X sealed SOTA round2 P0 source"
TEMPLATE_PREFIX = "ResilientV2X round2 P0 bootstrap template"
TASK_PREFIX = "ResilientV2X round2 P0"
WORKER_QUEUE = "GPU4-A100"
WORKER_QUEUE_ID = "9350f33af13a448da8339eb7bea52fdf"

GPU_COUNT = 4
BATCH_SIZE_PER_GPU = 2
GLOBAL_BATCH_SIZE = 8
MAX_EPOCHS = 50
VAL_INTERVAL = 10
TRAINING_SEED = 20_250_218
PRECISION = "FP32"

UPLOAD_TOKEN = "UPLOAD_EXACT_ROUND2_P0_SOURCE"
TEMPLATE_TOKEN = "CREATE_EXACT_ROUND2_P0_TEMPLATE"
LAUNCH_TOKEN = "LAUNCH_EXACT_ROUND2_P0_A100"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_P0_SPEC_ANCHOR = """    ExperimentSpec(
        "linear_no_distillation",
"""
_P0_SPEC_PATCH = """    ExperimentSpec(
        "support_residual_no_reliability_linear",
        "sota_candidate",
        "configs/resilient_v2x/improvements/support_residual_no_reliability_linear.py",
        True,
    ),
    ExperimentSpec(
        "linear_no_distillation",
"""
_P0_NESTED_ANCHOR = """        "support_residual_no_reliability",
        "resilient_v2x",
"""
_P0_NESTED_PATCH = """        "support_residual_no_reliability",
        "support_residual_no_reliability_linear",
        "resilient_v2x",
"""


@dataclass(frozen=True)
class Round2DeploymentProfile:
    """Candidate-specific bindings for the shared sealed deployment engine."""

    label: str
    output_dir: Path
    package_manifest_name: str
    deployment_plan_name: str
    package_type: str
    experiment: str
    config: str
    identity: str
    base_source_dir: Path
    base_source_inventory: Path
    base_source_archive: Path
    base_source_dataset_id: str
    base_source_package: Mapping[str, object]
    provenance: Mapping[str, object] | None
    early_gate_policy: Mapping[str, object] | None
    source_transition_artifact: str
    launch_receipt_artifact: str
    source_dataset_prefix: str
    source_dataset_version_suffix: str
    source_dataset_tags: tuple[str, ...]
    template_prefix: str
    task_prefix: str
    worker_queue: str
    worker_queue_id: str
    gpu_count: int
    batch_size_per_gpu: int
    global_batch_size: int
    max_epochs: int
    val_interval: int
    training_seed: int
    precision: str
    upload_token: str
    template_token: str
    launch_token: str
    spec_anchor: str
    spec_patch: str
    nested_anchor: str
    nested_patch: str
    bootstrap_parent: "Round2DeploymentProfile | None"
    transition_type: str
    plan_type: str
    receipt_type: str
    template_status_message: str
    transition_description: str
    invariants: tuple[str, ...]
    duplicate_guard_description: str
    duplicate_guard_stages: tuple[str, ...] | None
    relative_script: Path
    cli_description: str
    modified_source_paths: tuple[str, ...] = ()
    forbidden_source_paths: tuple[str, ...] = ()
    bootstrap_extra_patches: tuple[tuple[str, str], ...] = ()
    bootstrap_required_markers: tuple[str, ...] = ()
    bootstrap_forbidden_markers: tuple[str, ...] = ()
    execution_contract: Mapping[str, object] | None = None
    training_launch_enabled: bool = True
    task_parameter_bindings: Mapping[str, object] | None = None
    strict_enqueue_acknowledgement: bool = False
    enqueue_acknowledgement_artifact: str | None = None


P0_PROFILE = Round2DeploymentProfile(
    label="round2 P0",
    output_dir=OUTPUT_DIR,
    package_manifest_name=PACKAGE_MANIFEST_NAME,
    deployment_plan_name=DEPLOYMENT_PLAN_NAME,
    package_type="resilient_v2x_sota_round2_p0_source",
    experiment=P0_EXPERIMENT,
    config=P0_CONFIG,
    identity=P0_IDENTITY,
    base_source_dir=BASE_SOURCE_DIR,
    base_source_inventory=BASE_SOURCE_INVENTORY,
    base_source_archive=BASE_SOURCE_ARCHIVE,
    base_source_dataset_id=fastlane.SOURCE_DATASET_ID,
    base_source_package=dict(candidate.CANDIDATE_SOURCE_PACKAGE),
    provenance=None,
    early_gate_policy=None,
    source_transition_artifact=SOURCE_TRANSITION_ARTIFACT,
    launch_receipt_artifact=LAUNCH_RECEIPT_ARTIFACT,
    source_dataset_prefix=SOURCE_DATASET_PREFIX,
    source_dataset_version_suffix="round2-p0-v1",
    source_dataset_tags=(
        "ResilientV2X",
        "source",
        "sota-candidates",
        "round2-p0",
        "sealed",
    ),
    template_prefix=TEMPLATE_PREFIX,
    task_prefix=TASK_PREFIX,
    worker_queue=WORKER_QUEUE,
    worker_queue_id=WORKER_QUEUE_ID,
    gpu_count=GPU_COUNT,
    batch_size_per_gpu=BATCH_SIZE_PER_GPU,
    global_batch_size=GLOBAL_BATCH_SIZE,
    max_epochs=MAX_EPOCHS,
    val_interval=VAL_INTERVAL,
    training_seed=TRAINING_SEED,
    precision=PRECISION,
    upload_token=UPLOAD_TOKEN,
    template_token=TEMPLATE_TOKEN,
    launch_token=LAUNCH_TOKEN,
    spec_anchor=_P0_SPEC_ANCHOR,
    spec_patch=_P0_SPEC_PATCH,
    nested_anchor=_P0_NESTED_ANCHOR,
    nested_patch=_P0_NESTED_PATCH,
    bootstrap_parent=None,
    transition_type="exact_additive_sota_round2_p0_source_revision",
    plan_type="resilient_v2x_sota_round2_p0_a100_v1",
    receipt_type="resilient_v2x_sota_round2_p0_launch_v1",
    template_status_message="sealed round2 P0 bootstrap template",
    transition_description="exact one-config additive round2 P0 source",
    invariants=(
        "source transition adds only the P0 overlay to sealed E1/E2/E3 source",
        "existing E1/E2/E3 Dataset, template, controller, and tasks are untouched",
        "P0 preserves teacher, distillation, seed, FP32, 50 epochs, and val/10",
        "one task is enqueued to GPU4-A100 only after all receipts validate",
    ),
    duplicate_guard_description=(
        "exact task name plus predecessor parent preflight immediately before clone"
    ),
    duplicate_guard_stages=None,
    relative_script=Path(__file__).relative_to(ROOT),
    cli_description=__doc__ or "",
)

_ACTIVE_PROFILE: ContextVar[Round2DeploymentProfile] = ContextVar(
    "resilient_v2x_round2_deployment_profile", default=P0_PROFILE
)


def _profile() -> Round2DeploymentProfile:
    return _ACTIVE_PROFILE.get()


@contextmanager
def use_deployment_profile(
    profile: Round2DeploymentProfile,
) -> Iterator[None]:
    """Run the shared engine with an explicit, context-local candidate profile."""

    token = _ACTIVE_PROFILE.set(profile)
    try:
        yield
    finally:
        _ACTIVE_PROFILE.reset(token)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sealed(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = hashlib.sha256(
        _canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def _require_seal(payload: Mapping[str, object], *, context: str) -> None:
    observed = payload.get("seal_sha256")
    if not isinstance(observed, str) or _SHA256.fullmatch(observed) is None:
        raise RuntimeError(f"{context} has no valid seal")
    if observed != _sealed(payload)["seal_sha256"]:
        raise RuntimeError(f"{context} seal mismatch")


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def _require_token(observed: str, expected: str) -> None:
    if observed != expected:
        raise PermissionError(f"exact execute token required: {expected}")


def _clearml_json_artifact_receipt(
    payload: Mapping[str, object],
) -> tuple[str, int]:
    serialized = json.dumps(payload, sort_keys=True, indent=4).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest(), len(serialized)


def _require_transition_artifact(
    task: object, transition: Mapping[str, object]
) -> None:
    profile = _profile()
    _require_seal(transition, context=f"{profile.label} source transition")
    expected_sha256, expected_bytes = _clearml_json_artifact_receipt(transition)
    record = fastlane._artifact_record(
        task,
        profile.source_transition_artifact,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )
    observed = fastlane._artifact_preview_payload(
        record, name=profile.source_transition_artifact
    )
    if observed != transition:
        raise RuntimeError(f"{profile.label} source transition artifact drifted")


def _require_launch_receipt_artifact(
    task: object, receipt: Mapping[str, object]
) -> None:
    profile = _profile()
    _require_sealed_json_artifact(
        task,
        profile.launch_receipt_artifact,
        receipt,
        context=f"{profile.label} launch receipt",
    )


def _require_sealed_json_artifact(
    task: object,
    artifact_name: str,
    payload: Mapping[str, object],
    *,
    context: str,
) -> None:
    _require_seal(payload, context=context)
    expected_sha256, expected_bytes = _clearml_json_artifact_receipt(payload)
    record = fastlane._artifact_record(
        task,
        artifact_name,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )
    observed = fastlane._artifact_preview_payload(record, name=artifact_name)
    if observed != payload:
        raise RuntimeError(f"{context} artifact drifted")


def _resolve_live_queue_id(queue_name: str) -> str:
    """Resolve one exact ClearML queue name to one authoritative queue ID."""

    from clearml.backend_api.session.client import APIClient

    response = APIClient().queues.get_all(
        name=f"^{re.escape(queue_name)}$",
        only_fields=["id", "name"],
        page_size=2,
    )
    queues = getattr(response, "queues", None)
    if not isinstance(queues, (list, tuple)):
        raise RuntimeError("ClearML queue lookup returned an invalid response")
    exact = [item for item in queues if getattr(item, "name", None) == queue_name]
    if len(exact) != 1:
        raise RuntimeError(
            f"ClearML queue lookup expected one exact {queue_name!r} match, "
            f"got {len(exact)}"
        )
    return training._clearml_id(
        getattr(exact[0], "id", None),
        f"ClearML queue {queue_name}",
    )


def _post_enqueue_state(task: object) -> dict[str, object]:
    profile = _profile()
    status = fastlane._status(task)
    if status not in {"queued", "in_progress", "completed"}:
        raise RuntimeError(
            f"{profile.label} post-enqueue status is invalid: {status!r}"
        )
    queue_id = fastlane._execution_queue(task)
    if queue_id != profile.worker_queue_id:
        raise RuntimeError(
            f"{profile.label} enqueue queue ID drifted: "
            f"expected {profile.worker_queue_id}, got {queue_id}"
        )
    last_worker_raw = getattr(getattr(task, "data", None), "last_worker", None)
    if last_worker_raw in (None, ""):
        last_worker = None
    elif type(last_worker_raw) is str and last_worker_raw.strip():
        last_worker = last_worker_raw.strip()
    else:
        raise RuntimeError(f"{profile.label} post-enqueue last_worker is invalid")
    if status == "queued":
        if last_worker is not None:
            raise RuntimeError(
                f"{profile.label} queued task has a premature worker assignment"
            )
        assignment_state = "awaiting_worker_assignment"
    else:
        if last_worker is None:
            raise RuntimeError(
                f"{profile.label} active task has no auditable worker assignment"
            )
        assignment_state = "worker_assigned"
    return {
        "status": status,
        "execution_queue_id": queue_id,
        "execution_queue_name": profile.worker_queue,
        "last_worker": last_worker,
        "worker_assignment_state": assignment_state,
    }


def _require_bootstrap_script(
    task: object, *, expected_sha256: str, context: str
) -> str:
    script = fastlane._script(task)
    if script.get("entry_point") != fastlane.TEMPLATE_ENTRY_POINT:
        raise RuntimeError(f"{context} entry point drifted")
    diff = str(script.get("diff") or "")
    if hashlib.sha256(diff.encode("utf-8")).hexdigest() != expected_sha256:
        raise RuntimeError(f"{context} script SHA-256 drifted")
    return diff


def _validated_fixed_base(task_class: object) -> tuple[object, str, str]:
    base, patched = fastlane._validate_fixed_inputs(task_class)
    if (
        _task_project_name(base) != TRAINING_PROJECT
        or _task_project_id(base) != TRAINING_PROJECT_ID
    ):
        raise RuntimeError("immutable E1/E2/E3 template project binding drifted")
    raw = _require_bootstrap_script(
        base,
        expected_sha256=fastlane.TEMPLATE_SCRIPT_SHA256,
        context="immutable E1/E2/E3 bootstrap template",
    )
    if hashlib.sha256(patched.encode("utf-8")).hexdigest() != (
        fastlane.PATCHED_TEMPLATE_SCRIPT_SHA256
    ):
        raise RuntimeError("immutable E1/E2/E3 patched bootstrap script drifted")
    return base, raw, patched


def _task_project_name(task: object) -> str:
    getter = getattr(task, "get_project_name", None)
    if callable(getter):
        value = getter()
    else:
        value = getattr(task, "project", "")
        if callable(value):
            value = value()
    return str(value or "")


def _task_project_id(task: object) -> str:
    data = getattr(task, "data", None)
    return str(getattr(data, "project", "") or "")


def _clone_task(
    task_class: object,
    *,
    source_task: object,
    name: str,
    parent: str,
) -> object:
    return task_class.clone(
        source_task=source_task,
        name=name,
        parent=parent,
        project=TRAINING_PROJECT_ID,
    )


def _validate_round2_template(
    task: object,
    *,
    expected_name: str,
    expected_status: str,
    expected_parameters: Mapping[str, object],
) -> None:
    profile = _profile()
    fastlane._task_id(task, context=f"{profile.label} template")
    if (
        fastlane._status(task) != expected_status
        or getattr(task, "name", "") != expected_name
        or fastlane._task_parent(task) != fastlane.MAIN_CONTROLLER_TASK_ID
        or _task_project_name(task) != TRAINING_PROJECT
        or _task_project_id(task) != TRAINING_PROJECT_ID
    ):
        raise RuntimeError(f"{profile.label} template identity drifted")
    _require_bootstrap_script(
        task,
        expected_sha256=fastlane.TEMPLATE_SCRIPT_SHA256,
        context=f"{profile.label} template",
    )
    fastlane._require_parameters_exact(
        fastlane._parameters(task),
        expected_parameters,
        context=f"{profile.label} template",
    )


def _validate_training_clone(
    task: object,
    *,
    expected_name: str,
    patched_diff: str,
    expected_parameters: Mapping[str, object],
) -> dict[str, object]:
    profile = _profile()
    task_id = fastlane._task_id(task, context=f"{profile.label} clone")
    observed_name = str(getattr(task, "name", "") or "")
    observed_parent = fastlane._task_parent(task)
    observed_project = _task_project_name(task)
    observed_project_id = _task_project_id(task)
    observed_queue_id = fastlane._execution_queue(task)
    if (
        fastlane._status(task) != "created"
        or observed_name != expected_name
        or observed_parent != fastlane.PREDECESSOR_TASK_ID
        or observed_project != TRAINING_PROJECT
        or observed_project_id != TRAINING_PROJECT_ID
        or observed_queue_id is not None
    ):
        raise RuntimeError(f"{profile.label} clone identity drifted")
    patched_sha256 = hashlib.sha256(patched_diff.encode("utf-8")).hexdigest()
    observed_diff = _require_bootstrap_script(
        task,
        expected_sha256=patched_sha256,
        context=f"{profile.label} clone",
    )
    if observed_diff != patched_diff:
        raise RuntimeError(f"{profile.label} clone script payload drifted")
    fastlane._require_parameters_exact(
        fastlane._parameters(task),
        expected_parameters,
        context=f"{profile.label} clone",
    )
    return {
        "task_id": task_id,
        "task_name": observed_name,
        "parent_task_id": observed_parent,
        "project": observed_project,
        "project_id": observed_project_id,
        "entry_point": fastlane.TEMPLATE_ENTRY_POINT,
        "script_sha256": patched_sha256,
        "pre_enqueue_queue_id": observed_queue_id,
        "planned_queue": profile.worker_queue,
        "planned_queue_id": profile.worker_queue_id,
    }


def _upload_verify_and_enqueue(
    task_class: object,
    task: object,
    receipt: Mapping[str, object],
    *,
    queue: str,
) -> dict[str, object] | None:
    profile = _profile()
    _require_seal(receipt, context=f"{profile.label} launch receipt")
    resolved_queue_id: str | None = None
    if profile.strict_enqueue_acknowledgement:
        resolved_queue_id = _resolve_live_queue_id(queue)
        if resolved_queue_id != profile.worker_queue_id:
            raise RuntimeError(
                f"{profile.label} queue name-to-ID preflight drifted: "
                f"expected {profile.worker_queue_id}, got {resolved_queue_id}"
            )
    if not task.upload_artifact(
        profile.launch_receipt_artifact,
        artifact_object=dict(receipt),
        wait_on_upload=True,
    ):
        raise RuntimeError(f"failed to upload {profile.label} launch receipt")
    flushed = task.flush(wait_for_uploads=True)
    if flushed not in {None, True}:
        raise RuntimeError(f"{profile.label} launch receipt flush was not acknowledged")
    fastlane._reload(task)
    _require_launch_receipt_artifact(task, receipt)
    _require_unique_current_clone(task_class, task)
    if not profile.strict_enqueue_acknowledgement:
        fastlane._enqueue(task_class, task, queue=queue)
        observed_queue_id = fastlane._execution_queue(task)
        if observed_queue_id != profile.worker_queue_id:
            raise RuntimeError(
                f"{profile.label} enqueue queue ID drifted: "
                f"expected {profile.worker_queue_id}, got {observed_queue_id}"
            )
        return None
    assert resolved_queue_id is not None
    response = task_class.enqueue(
        task=task,
        queue_id=resolved_queue_id,
        force=False,
    )
    if isinstance(response, Mapping):
        acknowledged = response.get("queued") == 1 and response.get("updated") == 1
    else:
        acknowledged = (
            getattr(response, "queued", None) == 1
            and getattr(response, "updated", None) == 1
        )
    if not acknowledged:
        raise RuntimeError(f"{profile.label} enqueue was not exactly acknowledged")
    fastlane._reload(task)
    state = _post_enqueue_state(task)
    artifact_name = profile.enqueue_acknowledgement_artifact
    assert isinstance(artifact_name, str)
    acknowledgement = _sealed(
        {
            "schema_version": 1,
            "acknowledgement_type": (
                "resilient_v2x_round2_post_enqueue_acknowledgement"
            ),
            "task_id": fastlane._task_id(
                task, context=f"{profile.label} enqueued task"
            ),
            "launch_receipt_seal_sha256": receipt["seal_sha256"],
            "resolved_queue_name": queue,
            "resolved_queue_id": resolved_queue_id,
            "single_executor": True,
            "post_enqueue_state": state,
        }
    )
    if not task.upload_artifact(
        artifact_name,
        artifact_object=dict(acknowledgement),
        wait_on_upload=True,
    ):
        raise RuntimeError(f"failed to upload {profile.label} enqueue acknowledgement")
    flushed = task.flush(wait_for_uploads=True)
    if flushed not in {None, True}:
        raise RuntimeError(
            f"{profile.label} enqueue acknowledgement flush was not acknowledged"
        )
    fastlane._reload(task)
    _require_sealed_json_artifact(
        task,
        artifact_name,
        acknowledgement,
        context=f"{profile.label} enqueue acknowledgement",
    )
    final_state = _post_enqueue_state(task)
    if final_state["status"] != state["status"]:
        if not (
            state["status"] == "queued"
            and final_state["status"] in {"in_progress", "completed"}
        ):
            raise RuntimeError(
                f"{profile.label} post-enqueue status changed non-monotonically"
            )
    return acknowledgement


def _require_unique_current_clone(task_class: object, task: object) -> None:
    """Fail closed unless the post-receipt query sees only this exact clone."""

    profile = _profile()
    task_id = fastlane._task_id(task, context=f"{profile.label} current clone")
    task_name = str(getattr(task, "name", "") or "")
    parent = fastlane._task_parent(task)
    if not task_name or parent != fastlane.PREDECESSOR_TASK_ID:
        raise RuntimeError(f"{profile.label} current clone name/parent drifted")
    matches = task_class.get_tasks(
        task_name=f"^{re.escape(task_name)}$",
        task_filter={"parent": parent},
    )
    exact: list[object] = []
    for observed in matches or ():
        if (
            str(getattr(observed, "name", "") or "") == task_name
            and fastlane._task_parent(observed) == parent
        ):
            exact.append(observed)
    if len(exact) != 1:
        raise RuntimeError(
            f"{profile.label} post-receipt duplicate guard expected exactly one "
            f"exact name+parent match, got {len(exact)}"
        )
    observed_id = fastlane._task_id(
        exact[0], context=f"{profile.label} post-receipt duplicate match"
    )
    if observed_id != task_id:
        raise RuntimeError(
            f"{profile.label} post-receipt duplicate guard matched a different task ID"
        )


def _base_inventory() -> dict[str, object]:
    profile = _profile()
    package = profile.base_source_package
    if (
        not profile.base_source_archive.is_file()
        or profile.base_source_archive.is_symlink()
        or profile.base_source_archive.stat().st_size != package["archive_bytes"]
        or _sha256_file(profile.base_source_archive) != package["archive_sha256"]
    ):
        raise RuntimeError(f"immutable {profile.label} base source archive drifted")
    if (
        not profile.base_source_inventory.is_file()
        or profile.base_source_inventory.is_symlink()
        or profile.base_source_inventory.stat().st_size != package["inventory_bytes"]
        or _sha256_file(profile.base_source_inventory) != package["inventory_sha256"]
    ):
        raise RuntimeError(f"immutable {profile.label} base source inventory drifted")
    inventory = _load_json(profile.base_source_inventory)
    for key in ("tree_sha256", "file_count", "source_bytes"):
        if inventory.get(key) != package[key]:
            raise RuntimeError(f"immutable base source inventory drifted: {key}")
    files = inventory.get("files")
    if not isinstance(files, list) or len(files) != package["file_count"]:
        raise RuntimeError("immutable base source file list drifted")
    return inventory


def _extract_base(destination: Path, entries: list[dict[str, object]]) -> None:
    import zstandard

    profile = _profile()
    source_packager._verify_archive(profile.base_source_archive, entries)
    with profile.base_source_archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for index, member in enumerate(archive):
                    if index >= len(entries) or member.name != entries[index]["path"]:
                        raise RuntimeError("base source extraction order drifted")
                    pure = PurePosixPath(member.name)
                    if pure.is_absolute() or ".." in pure.parts or not member.isfile():
                        raise RuntimeError("unsafe base source archive member")
                    payload = archive.extractfile(member)
                    if payload is None:
                        raise RuntimeError("base source archive member has no payload")
                    target = destination.joinpath(*pure.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as output:
                        shutil.copyfileobj(payload, output, length=8 * 1024 * 1024)
                    target.chmod(int(entries[index]["mode"]))


def _normalized_source_paths(values: Sequence[str], *, context: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if type(value) is not str or not value:
            raise ValueError(f"{context} must contain non-empty strings")
        pure = PurePosixPath(value)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != value
            or value.endswith("/")
        ):
            raise ValueError(f"{context} contains an unsafe path: {value!r}")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{context} contains duplicate paths")
    return tuple(sorted(normalized))


def _local_source_entry(relative_path: str, *, mode: int) -> dict[str, object]:
    profile = _profile()
    path = ROOT.joinpath(*PurePosixPath(relative_path).parts)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(
            f"{profile.label} source member is missing or not a regular file: "
            f"{relative_path}"
        )
    return {
        "path": relative_path,
        "mode": mode,
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _expected_source_layout(
    base: Mapping[str, object],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    tuple[str, ...],
]:
    """Resolve the exact base-preserving source delta for the active profile."""

    profile = _profile()
    raw_files = base.get("files")
    if not isinstance(raw_files, list) or not all(
        isinstance(item, Mapping) for item in raw_files
    ):
        raise RuntimeError("immutable base source inventory contains invalid entries")
    base_entries = [dict(item) for item in raw_files]
    base_by_path: dict[str, dict[str, object]] = {}
    for entry in base_entries:
        path = entry.get("path")
        if type(path) is not str or not path or path in base_by_path:
            raise RuntimeError("immutable base source inventory paths are invalid")
        base_by_path[path] = entry

    candidate_path = _normalized_source_paths(
        (profile.config,), context=f"{profile.label} candidate config"
    )[0]
    modified_paths = _normalized_source_paths(
        profile.modified_source_paths,
        context=f"{profile.label} modified source paths",
    )
    forbidden_paths = _normalized_source_paths(
        profile.forbidden_source_paths,
        context=f"{profile.label} forbidden source paths",
    )
    if candidate_path in modified_paths or candidate_path in forbidden_paths:
        raise RuntimeError(f"{profile.label} source policy paths overlap")
    if set(modified_paths) & set(forbidden_paths):
        raise RuntimeError(f"{profile.label} modified and forbidden paths overlap")
    if candidate_path in base_by_path:
        raise RuntimeError(
            f"{profile.label} config unexpectedly exists in immutable base source"
        )
    forbidden_in_base = sorted(set(forbidden_paths) & set(base_by_path))
    if forbidden_in_base:
        raise RuntimeError(
            f"{profile.label} forbidden paths exist in immutable base source: "
            f"{forbidden_in_base}"
        )

    modified_entries: list[dict[str, object]] = []
    target_by_path = dict(base_by_path)
    for relative_path in modified_paths:
        base_entry = base_by_path.get(relative_path)
        if base_entry is None:
            raise RuntimeError(
                f"{profile.label} modified path is absent from immutable base: "
                f"{relative_path}"
            )
        mode = base_entry.get("mode")
        if type(mode) is not int:
            raise RuntimeError(
                f"{profile.label} base mode is invalid for {relative_path}"
            )
        target_entry = _local_source_entry(relative_path, mode=mode)
        if target_entry == base_entry:
            raise RuntimeError(
                f"{profile.label} declared modified path is unchanged: {relative_path}"
            )
        modified_entries.append(target_entry)
        target_by_path[relative_path] = target_entry

    candidate_entry = _local_source_entry(candidate_path, mode=0o644)
    target_by_path[candidate_path] = candidate_entry
    expected_paths = set(base_by_path) | {candidate_path}
    if set(target_by_path) != expected_paths:
        raise RuntimeError(f"{profile.label} target source path set drifted")
    leaked = sorted(set(target_by_path) & set(forbidden_paths))
    if leaked:
        raise RuntimeError(
            f"{profile.label} forbidden paths leaked into target source: {leaked}"
        )
    entries = sorted(target_by_path.values(), key=lambda item: str(item["path"]))
    modified_entries.sort(key=lambda item: str(item["path"]))
    return entries, modified_entries, candidate_entry, forbidden_paths


def _prepare_source(output_dir: Path) -> dict[str, object]:
    profile = _profile()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite {profile.label} package: {output_dir}"
        )
    base = _base_inventory()
    raw_entries = base["files"]
    assert isinstance(raw_entries, list)
    base_entries = [dict(item) for item in raw_entries if isinstance(item, Mapping)]
    if len(base_entries) != len(raw_entries):
        raise RuntimeError("base source inventory contains a non-object entry")
    entries, modified_entries, candidate_entry, forbidden_paths = (
        _expected_source_layout(base)
    )
    local_delta_entries = [*modified_entries, candidate_entry]
    tree_sha256 = source_packager._tree_sha256(entries)
    archive_name = f"resilient-v2x-source-{tree_sha256[:12]}.tar.zst"

    temporary_prefix = f"transvision-{profile.label.lower().replace(' ', '-')}-"
    with tempfile.TemporaryDirectory(prefix=temporary_prefix) as temporary:
        work = Path(temporary)
        staging = work / "source"
        staging.mkdir()
        _extract_base(staging, base_entries)
        for entry in local_delta_entries:
            relative_path = str(entry["path"])
            source = ROOT.joinpath(*PurePosixPath(relative_path).parts)
            staged = staging.joinpath(*PurePosixPath(relative_path).parts)
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, staged)
            staged.chmod(int(entry["mode"]))

        first = work / f".{archive_name}.first"
        second = work / f".{archive_name}.second"
        source_packager._write_archive(staging, entries, first)
        source_packager._write_archive(staging, entries, second)
        if first.stat().st_size != second.stat().st_size or _sha256_file(
            first
        ) != _sha256_file(second):
            raise RuntimeError("round2 source archive is not byte deterministic")
        source_packager._verify_archive(first, entries)
        source_packager._verify_archive(second, entries)
        for entry in local_delta_entries:
            relative_path = str(entry["path"])
            source = ROOT.joinpath(*PurePosixPath(relative_path).parts)
            if (
                source.stat().st_size != entry["size"]
                or _sha256_file(source) != entry["sha256"]
            ):
                raise RuntimeError(
                    f"{profile.label} source member changed while preparing: "
                    f"{relative_path}"
                )

        inventory = dict(base)
        inventory.update(
            {
                "file_count": len(entries),
                "source_bytes": sum(int(entry["size"]) for entry in entries),
                "tree_sha256": tree_sha256,
                "files": entries,
            }
        )
        inventory_bytes = (
            json.dumps(
                inventory,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        output_dir.mkdir(parents=True)
        archive_path = output_dir / archive_name
        first.replace(archive_path)
        second.unlink()
        inventory_path = output_dir / "source-inventory.json"
        inventory_path.write_bytes(inventory_bytes)

    package = {
        "archive_name": archive_name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "tree_sha256": tree_sha256,
        "inventory_sha256": _sha256_file(inventory_path),
        "inventory_bytes": inventory_path.stat().st_size,
        "file_count": len(entries),
        "source_bytes": inventory["source_bytes"],
    }
    manifest_payload: dict[str, object] = {
        "schema_version": 1,
        "package_type": profile.package_type,
        "base_source_dataset_id": profile.base_source_dataset_id,
        "base_source_package": dict(profile.base_source_package),
        "target_source_package": package,
        "delta": {
            "modified_files": modified_entries,
            "removed_files": [],
            "added_files": [candidate_entry],
        },
        "candidate": {
            "experiment": profile.experiment,
            "config": profile.config,
            "identity": profile.identity,
        },
    }
    if profile.provenance is not None:
        manifest_payload["provenance"] = dict(profile.provenance)
    if forbidden_paths:
        manifest_payload["excluded_paths"] = list(forbidden_paths)
    manifest = _sealed(manifest_payload)
    manifest_path = output_dir / profile.package_manifest_name
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _verify_package(output_dir: Path) -> dict[str, object]:
    profile = _profile()
    delta_label = "P0" if profile is P0_PROFILE else profile.label
    manifest = _load_json(output_dir / profile.package_manifest_name)
    _require_seal(manifest, context=f"{profile.label} source package manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("package_type") != profile.package_type
    ):
        raise RuntimeError(f"{profile.label} source manifest type drifted")
    if manifest.get("base_source_dataset_id") != profile.base_source_dataset_id:
        raise RuntimeError("round2 base source Dataset binding drifted")
    if manifest.get("base_source_package") != dict(profile.base_source_package):
        raise RuntimeError("round2 base source package binding drifted")
    target = manifest.get("target_source_package")
    delta = manifest.get("delta")
    if not isinstance(target, Mapping) or not isinstance(delta, Mapping):
        raise RuntimeError("round2 source manifest structure drifted")
    expected_candidate = {
        "experiment": profile.experiment,
        "config": profile.config,
        "identity": profile.identity,
    }
    if manifest.get("candidate") != expected_candidate:
        raise RuntimeError(f"{profile.label} candidate identity drifted")
    expected_provenance = (
        dict(profile.provenance) if profile.provenance is not None else None
    )
    if manifest.get("provenance") != expected_provenance:
        raise RuntimeError(f"{profile.label} provenance drifted")

    base = _base_inventory()
    (
        expected_entries,
        expected_modified_entries,
        expected_candidate_entry,
        forbidden_paths,
    ) = _expected_source_layout(base)
    expected_excluded_paths = list(forbidden_paths) if forbidden_paths else None
    if manifest.get("excluded_paths") != expected_excluded_paths:
        raise RuntimeError(f"{profile.label} excluded source paths drifted")
    modifications = delta.get("modified_files")
    additions = delta.get("added_files")
    exact_delta = (
        isinstance(modifications, list)
        and all(isinstance(item, Mapping) for item in modifications)
        and [dict(item) for item in modifications] == expected_modified_entries
        and delta.get("removed_files") == []
        and isinstance(additions, list)
        and len(additions) == 1
        and isinstance(additions[0], Mapping)
        and dict(additions[0]) == expected_candidate_entry
    )
    if not exact_delta:
        if not profile.modified_source_paths and not profile.forbidden_source_paths:
            raise RuntimeError("round2 source is not an exact one-file additive delta")
        raise RuntimeError(
            f"{profile.label} source is not the exact declared modified/additive delta"
        )
    expected_file_count = len(expected_entries)
    expected_source_bytes = sum(int(item["size"]) for item in expected_entries)
    expected_tree_sha256 = source_packager._tree_sha256(expected_entries)
    expected_inventory = dict(base)
    expected_inventory.update(
        {
            "file_count": expected_file_count,
            "source_bytes": expected_source_bytes,
            "tree_sha256": expected_tree_sha256,
            "files": expected_entries,
        }
    )
    if (
        target.get("file_count") != expected_file_count
        or target.get("source_bytes") != expected_source_bytes
        or target.get("tree_sha256") != expected_tree_sha256
        or target.get("archive_name")
        != f"resilient-v2x-source-{expected_tree_sha256[:12]}.tar.zst"
    ):
        raise RuntimeError(f"round2 target source is not exact base plus {delta_label}")

    archive = output_dir / str(target["archive_name"])
    inventory_path = output_dir / "source-inventory.json"
    observed = {
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256_file(archive),
        "inventory_bytes": inventory_path.stat().st_size,
        "inventory_sha256": _sha256_file(inventory_path),
    }
    for key, value in observed.items():
        if target.get(key) != value:
            raise RuntimeError(f"round2 package drifted: {key}")
    inventory = _load_json(inventory_path)
    if inventory != expected_inventory:
        raise RuntimeError(f"round2 inventory is not exact base plus {delta_label}")
    for key in ("tree_sha256", "file_count", "source_bytes"):
        if inventory.get(key) != target.get(key):
            raise RuntimeError(f"round2 source inventory drifted: {key}")
    files = inventory.get("files")
    if (
        not isinstance(files, list)
        or len(files) != expected_file_count
        or not all(isinstance(item, Mapping) for item in files)
        or [dict(item) for item in files] != expected_entries
    ):
        raise RuntimeError("round2 source inventory entry list drifted")
    if source_packager._tree_sha256(expected_entries) != target["tree_sha256"]:
        raise RuntimeError("round2 source tree SHA-256 recomputation drifted")
    source_packager._verify_archive(archive, expected_entries)
    return manifest


def _transition(
    manifest: Mapping[str, object], *, dataset_id: str
) -> dict[str, object]:
    profile = _profile()
    target = manifest["target_source_package"]
    delta = manifest["delta"]
    assert isinstance(target, Mapping)
    assert isinstance(delta, Mapping)
    modified_files = delta.get("modified_files")
    added_files = delta.get("added_files")
    removed_files = delta.get("removed_files")
    if not all(
        isinstance(value, list)
        for value in (modified_files, added_files, removed_files)
    ):
        raise RuntimeError(f"{profile.label} transition delta lists are invalid")
    assert isinstance(modified_files, list)
    assert isinstance(added_files, list)
    assert isinstance(removed_files, list)
    base_file_count = int(profile.base_source_package["file_count"])
    modified_file_count = len(modified_files)
    added_file_count = len(added_files)
    removed_file_count = len(removed_files)
    payload: dict[str, object] = {
        "schema_version": 1,
        "transition_type": profile.transition_type,
        "base_source": {
            "dataset_id": profile.base_source_dataset_id,
            **dict(profile.base_source_package),
        },
        "target_source": {"dataset_id": dataset_id, **dict(target)},
        "inventory_delta": {
            "base_file_count": base_file_count,
            "target_file_count": target["file_count"],
            "unchanged_file_count": (
                base_file_count - modified_file_count - removed_file_count
            ),
            "modified_file_count": modified_file_count,
            "added_file_count": added_file_count,
            "removed_file_count": removed_file_count,
            **dict(delta),
        },
        "candidate": dict(manifest["candidate"]),
        "protocol": {
            "global_batch_size": profile.global_batch_size,
            "gpu_count": profile.gpu_count,
            "batch_size_per_gpu": profile.batch_size_per_gpu,
            "max_epochs": profile.max_epochs,
            "val_interval": profile.val_interval,
            "training_seed": profile.training_seed,
            "training_overlay_protocol_seed": profile.training_seed,
            "precision": profile.precision,
        },
    }
    if profile.provenance is not None:
        payload["provenance"] = dict(profile.provenance)
    if profile.early_gate_policy is not None:
        payload["early_gate_policy"] = dict(profile.early_gate_policy)
    return _sealed(payload)


def _expected_template_parameters(
    manifest: Mapping[str, object], *, dataset_id: str
) -> dict[str, object]:
    profile = _profile()
    package = manifest["target_source_package"]
    assert isinstance(package, Mapping)
    parameters = dict(fastlane.EXPECTED_TEMPLATE_PARAMETERS)
    parameters.update(
        {
            "Args/source_dataset_id": dataset_id,
            "Args/source_archive_name": package["archive_name"],
            "Args/source_archive_bytes": package["archive_bytes"],
            "Args/source_archive_sha256": package["archive_sha256"],
        }
    )
    if profile.task_parameter_bindings is not None:
        if not isinstance(profile.task_parameter_bindings, Mapping):
            raise RuntimeError(f"{profile.label} task parameter bindings are invalid")
        for key, value in profile.task_parameter_bindings.items():
            if type(key) is not str or not key.startswith("Args/") or len(key) <= 5:
                raise RuntimeError(
                    f"{profile.label} task parameter binding key is invalid: {key!r}"
                )
            if value is None or isinstance(value, (dict, list, tuple, set)):
                raise RuntimeError(
                    f"{profile.label} task parameter binding value is invalid: {key}"
                )
        parameters.update(dict(profile.task_parameter_bindings))
    return parameters


def _downloaded_source_root(manifest: Mapping[str, object], *, dataset_id: str) -> Path:
    profile = _profile()
    from clearml import Dataset

    package = manifest["target_source_package"]
    assert isinstance(package, Mapping)
    dataset = Dataset.get(dataset_id=dataset_id)
    local = Path(dataset.get_local_copy()).resolve(strict=True)
    members = sorted(
        path.relative_to(local).as_posix()
        for path in local.rglob("*")
        if path.is_file()
    )
    expected = sorted(
        [
            f"source/{package['archive_name']}",
            "source/source-inventory.json",
        ]
    )
    if members != expected:
        raise RuntimeError(
            f"{profile.label} source Dataset inventory mismatch: {members}"
        )
    source_root = local / "source"
    archive = source_root / str(package["archive_name"])
    inventory = source_root / "source-inventory.json"
    if (
        archive.stat().st_size == package["archive_bytes"]
        and _sha256_file(archive) == package["archive_sha256"]
        and inventory.stat().st_size == package["inventory_bytes"]
        and _sha256_file(inventory) == package["inventory_sha256"]
    ):
        return source_root
    raise RuntimeError(
        f"{profile.label} source Dataset does not match its sealed package"
    )


def _patch_template(raw_diff: str) -> str:
    profile = _profile()
    if profile.bootstrap_parent is None:
        patched = fastlane._patched_template_diff(raw_diff)
    else:
        with use_deployment_profile(profile.bootstrap_parent):
            patched = _patch_template(raw_diff)
    if patched.count(profile.spec_anchor) != 1:
        raise RuntimeError(f"{profile.label} experiment-spec anchor is ambiguous")
    if patched.count(profile.nested_anchor) != 1:
        raise RuntimeError(f"{profile.label} nested-runner anchor is ambiguous")
    patched = patched.replace(profile.spec_anchor, profile.spec_patch, 1)
    patched = patched.replace(profile.nested_anchor, profile.nested_patch, 1)
    for index, patch in enumerate(profile.bootstrap_extra_patches):
        if (
            not isinstance(patch, tuple)
            or len(patch) != 2
            or not all(type(value) is str and value for value in patch)
        ):
            raise RuntimeError(
                f"{profile.label} bootstrap extra patch {index} is invalid"
            )
        anchor, replacement = patch
        if patched.count(anchor) != 1:
            raise RuntimeError(
                f"{profile.label} bootstrap extra patch {index} anchor is ambiguous"
            )
        patched = patched.replace(anchor, replacement, 1)
    if patched.count(f'"{profile.experiment}"') < 2:
        raise RuntimeError(f"{profile.label} identity was not installed in bootstrap")
    if patched.count(f'"{profile.config}"') != 1:
        raise RuntimeError(f"{profile.label} config path is ambiguous in bootstrap")
    for marker in profile.bootstrap_required_markers:
        if type(marker) is not str or not marker or marker not in patched:
            raise RuntimeError(
                f"{profile.label} required bootstrap marker is missing: {marker!r}"
            )
    for marker in profile.bootstrap_forbidden_markers:
        if type(marker) is not str or not marker or marker in patched:
            raise RuntimeError(
                f"{profile.label} forbidden bootstrap marker is present: {marker!r}"
            )
    return patched


def _task_parameters(
    manifest: Mapping[str, object], *, dataset_id: str
) -> dict[str, object]:
    profile = _profile()
    parameters = _expected_template_parameters(manifest, dataset_id=dataset_id)
    parameters.update(
        {
            "Args/stage": "all",
            "Args/experiment_from_task": profile.experiment,
            "Args/predecessor_task_id": fastlane.PREDECESSOR_TASK_ID,
            "Args/teacher_task_id": fastlane.TEACHER_TASK_ID,
            "Args/teacher_model_id": fastlane.TEACHER_MODEL_ID,
            "Args/teacher_checkpoint_sha256": fastlane.TEACHER_CHECKPOINT_SHA256,
            "Args/allow_failed_teacher_task": False,
            "Args/training_seed": profile.training_seed,
            "Args/gpus": profile.gpu_count,
            "Args/max_epochs": profile.max_epochs,
            "Args/amp": False,
        }
    )
    return parameters


def _static_plan(manifest: Mapping[str, object]) -> dict[str, object]:
    profile = _profile()
    if type(profile.training_launch_enabled) is not bool:
        raise RuntimeError(f"{profile.label} training launch flag is invalid")
    if profile.execution_contract is None and not profile.training_launch_enabled:
        raise RuntimeError(
            f"{profile.label} disables training without an execution contract"
        )
    if profile.execution_contract is not None:
        _require_seal(
            profile.execution_contract,
            context=f"{profile.label} execution contract",
        )
        contract_bindings = profile.execution_contract.get("task_parameter_bindings")
        if contract_bindings is not None and contract_bindings != (
            None
            if profile.task_parameter_bindings is None
            else dict(profile.task_parameter_bindings)
        ):
            raise RuntimeError(
                f"{profile.label} execution/task parameter bindings drifted"
            )
    if type(profile.strict_enqueue_acknowledgement) is not bool:
        raise RuntimeError(f"{profile.label} strict enqueue flag is invalid")
    if profile.strict_enqueue_acknowledgement:
        artifact_name = profile.enqueue_acknowledgement_artifact
        if type(artifact_name) is not str or not artifact_name.strip():
            raise RuntimeError(
                f"{profile.label} strict enqueue acknowledgement artifact is invalid"
            )
    elif profile.enqueue_acknowledgement_artifact is not None:
        raise RuntimeError(
            f"{profile.label} declares an enqueue acknowledgement artifact "
            "without strict enqueue acknowledgement"
        )
    target = manifest["target_source_package"]
    assert isinstance(target, Mapping)
    payload: dict[str, object] = {
        "schema_version": 1,
        "plan_type": profile.plan_type,
        "source_package_manifest_seal_sha256": manifest["seal_sha256"],
        "source_tree_sha256": target["tree_sha256"],
        "candidate": dict(manifest["candidate"]),
        "bindings": {
            "base_source_dataset_id": profile.base_source_dataset_id,
            "base_template_task_id": fastlane.TEMPLATE_TASK_ID,
            "predecessor_task_id": fastlane.PREDECESSOR_TASK_ID,
            "teacher_task_id": fastlane.TEACHER_TASK_ID,
            "teacher_model_id": fastlane.TEACHER_MODEL_ID,
            "teacher_checkpoint_sha256": fastlane.TEACHER_CHECKPOINT_SHA256,
            "training_dataset_id": fastlane.TRAINING_DATASET_ID,
            "training_project": TRAINING_PROJECT,
            "training_project_id": TRAINING_PROJECT_ID,
        },
        "protocol": {
            "worker_queue": profile.worker_queue,
            "worker_queue_id": profile.worker_queue_id,
            "global_batch_size": profile.global_batch_size,
            "gpu_count": profile.gpu_count,
            "batch_size_per_gpu": profile.batch_size_per_gpu,
            "max_epochs": profile.max_epochs,
            "val_interval": profile.val_interval,
            "training_seed": profile.training_seed,
            "precision": profile.precision,
        },
        "execution_policy": {
            "executor_count": 1,
            "duplicate_guard": profile.duplicate_guard_description,
            "concurrent_launchers_supported": False,
        },
    }
    if profile.provenance is not None:
        payload["provenance"] = dict(profile.provenance)
    if profile.early_gate_policy is not None:
        payload["early_gate_policy"] = dict(profile.early_gate_policy)
    if profile.execution_contract is not None:
        payload["execution_contract"] = dict(profile.execution_contract)
        execution_policy = payload["execution_policy"]
        assert isinstance(execution_policy, dict)
        execution_policy["training_launch_enabled"] = profile.training_launch_enabled
    if profile.task_parameter_bindings is not None:
        payload["task_parameter_bindings"] = dict(profile.task_parameter_bindings)
    if profile.strict_enqueue_acknowledgement:
        execution_policy = payload["execution_policy"]
        assert isinstance(execution_policy, dict)
        execution_policy.update(
            {
                "queue_name_to_exact_id_preflight": True,
                "post_enqueue_reload_required": True,
                "post_enqueue_worker_assignment_validation": True,
                "sealed_enqueue_acknowledgement_artifact": (
                    profile.enqueue_acknowledgement_artifact
                ),
            }
        )
    if profile.duplicate_guard_stages is not None:
        execution_policy = payload["execution_policy"]
        assert isinstance(execution_policy, dict)
        execution_policy["duplicate_guard_stages"] = list(
            profile.duplicate_guard_stages
        )
    return _sealed(payload)


def _deployment_plan(
    manifest: Mapping[str, object], *, output_dir: Path
) -> dict[str, object]:
    profile = _profile()
    plan = _static_plan(manifest)
    relative_script = profile.relative_script
    explicit_queue = "" if profile is P0_PROFILE else f"--queue {profile.worker_queue} "
    commands = {
        "1_upload_source": (
            f"python {relative_script} upload-source --output-dir {output_dir} "
            f"--execute-token {profile.upload_token}"
        ),
        "2_create_template": (
            f"python {relative_script} create-template --output-dir {output_dir} "
            "--source-dataset-id <DATASET_ID> "
            f"--execute-token {profile.template_token}"
        ),
    }
    if profile.training_launch_enabled:
        commands["3_launch_a100"] = (
            f"python {relative_script} launch --output-dir {output_dir} "
            "--source-dataset-id <DATASET_ID> --template-task-id <TEMPLATE_ID> "
            f"{explicit_queue}"
            f"--execute-token {profile.launch_token}"
        )
    if profile.execution_contract is not None:
        command_index = 4 if profile.training_launch_enabled else 3
        commands[f"{command_index}_show_execution_contract"] = (
            f"python {relative_script} show-execution-contract "
            f"--output-dir {output_dir}"
        )
    if profile.early_gate_policy is not None:
        command_index = (
            5
            if profile.execution_contract is not None
            and profile.training_launch_enabled
            else 4
        )
        commands[f"{command_index}_evaluate_e10_gate_local"] = (
            f"python {relative_script} evaluate-e10-gate "
            f"--plan-seal-sha256 {plan['seal_sha256']} "
            "--p0-task-id <P0_TASK_ID> --p0-bev-ap70 <P0_BEV_AP70> "
            "--p0-3d-ap70 <P0_3D_AP70> --p2-task-id <P2_TASK_ID> "
            "--p2-bev-ap70 <P2_BEV_AP70> --p2-3d-ap70 <P2_3D_AP70>"
        )
    return _sealed(
        {
            **plan,
            "static_plan_seal_sha256": plan["seal_sha256"],
            "remote_state_changed": False,
            "commands": commands,
            "invariants": list(profile.invariants),
        }
    )


def _verify_deployment_plan(
    output_dir: Path, manifest: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    profile = _profile()
    path = output_dir / profile.deployment_plan_name
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"sealed {profile.label} deployment plan is missing")
    observed = _load_json(path)
    _require_seal(observed, context=f"persisted {profile.label} deployment plan")
    expected = _deployment_plan(manifest, output_dir=output_dir)
    if observed != expected:
        raise RuntimeError(f"persisted {profile.label} deployment plan drifted")
    static = _static_plan(manifest)
    if observed.get("static_plan_seal_sha256") != static["seal_sha256"]:
        raise RuntimeError(f"persisted {profile.label} static plan binding drifted")
    return observed, static


def _prepare(args: argparse.Namespace) -> int:
    profile = _profile()
    manifest_path = args.output_dir / profile.package_manifest_name
    manifest = (
        _verify_package(args.output_dir)
        if manifest_path.is_file() and not manifest_path.is_symlink()
        else _prepare_source(args.output_dir)
    )
    deployment = _deployment_plan(manifest, output_dir=args.output_dir)
    _require_seal(deployment, context=f"{profile.label} deployment plan")
    plan_path = args.output_dir / profile.deployment_plan_name
    plan_path.write_text(
        json.dumps(deployment, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    _require_seal(
        _load_json(plan_path), context=f"persisted {profile.label} deployment plan"
    )
    _verify_deployment_plan(args.output_dir, manifest)
    print(json.dumps({"deployment_plan": str(plan_path), **deployment}, sort_keys=True))
    return 0


def _upload_source(args: argparse.Namespace) -> int:
    profile = _profile()
    _require_token(args.execute_token, profile.upload_token)
    manifest = _verify_package(args.output_dir)
    _verify_deployment_plan(args.output_dir, manifest)
    package = manifest["target_source_package"]
    assert isinstance(package, Mapping)
    from clearml import Dataset

    dataset_name = f"{profile.source_dataset_prefix} {str(package['tree_sha256'])[:12]}"
    dataset_version = (
        f"{str(package['tree_sha256'])[:12]}-{profile.source_dataset_version_suffix}"
    )
    dataset = Dataset.create(
        dataset_project=SOURCE_PROJECT,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_tags=list(profile.source_dataset_tags),
        output_uri=training.FILES_SERVER_URI,
        description=_canonical_json(
            {
                "base_source_dataset_id": profile.base_source_dataset_id,
                "transition": profile.transition_description,
                **(
                    {"provenance": dict(profile.provenance)}
                    if profile.provenance is not None
                    else {}
                ),
                "source_package_manifest_seal_sha256": manifest["seal_sha256"],
                "target_source_package": dict(package),
            }
        ),
    )
    dataset.add_files(
        args.output_dir,
        wildcard=[str(package["archive_name"]), "source-inventory.json"],
        local_base_folder=str(args.output_dir),
        dataset_path="source",
        recursive=False,
        verbose=True,
        max_workers=2,
    )
    dataset.upload(show_progress=True, verbose=True, preview=False, max_workers=2)
    if not dataset.finalize(verbose=True, raise_on_error=True, auto_upload=False):
        raise RuntimeError("round2 source Dataset did not finalize")
    _downloaded_source_root(manifest, dataset_id=dataset.id)
    print(
        json.dumps(
            {
                "source_dataset_id": dataset.id,
                "source_dataset_name": dataset_name,
                "source_dataset_version": dataset_version,
            },
            sort_keys=True,
        )
    )
    return 0


def _create_template(args: argparse.Namespace) -> int:
    profile = _profile()
    _require_token(args.execute_token, profile.template_token)
    manifest = _verify_package(args.output_dir)
    _verify_deployment_plan(args.output_dir, manifest)
    dataset_id = training._clearml_id(
        args.source_dataset_id, f"{profile.label} source Dataset"
    )
    _downloaded_source_root(manifest, dataset_id=dataset_id)
    transition = _transition(manifest, dataset_id=dataset_id)
    from clearml import Task

    base, _raw_diff, _patched_diff = _validated_fixed_base(Task)
    name = f"{profile.template_prefix} [{str(transition['seal_sha256'])[:12]}]"
    template = _clone_task(
        Task,
        source_task=base,
        name=name,
        parent=fastlane.MAIN_CONTROLLER_TASK_ID,
    )
    expected_parameters = _expected_template_parameters(manifest, dataset_id=dataset_id)
    template.set_parameters(expected_parameters)
    template.output_uri = training.FILES_SERVER_URI
    _validate_round2_template(
        template,
        expected_name=name,
        expected_status="created",
        expected_parameters=expected_parameters,
    )
    if not template.upload_artifact(
        profile.source_transition_artifact,
        artifact_object=transition,
        wait_on_upload=True,
    ):
        raise RuntimeError(f"failed to upload {profile.label} source transition")
    template.flush(wait_for_uploads=True)
    template.mark_completed(status_message=profile.template_status_message, force=True)
    template.reload()
    _validate_round2_template(
        template,
        expected_name=name,
        expected_status="completed",
        expected_parameters=expected_parameters,
    )
    _require_transition_artifact(template, transition)
    print(
        json.dumps(
            {
                "template_task_id": template.id,
                "template_name": name,
                "template_script_sha256": fastlane.TEMPLATE_SCRIPT_SHA256,
                "source_transition_seal_sha256": transition["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _launch(args: argparse.Namespace) -> int:
    profile = _profile()
    if not profile.training_launch_enabled:
        raise PermissionError(
            f"{profile.label} training launch is disabled; use the sealed "
            "checkpoint-reuse evaluation contract"
        )
    _require_token(args.execute_token, profile.launch_token)
    if args.queue != profile.worker_queue:
        raise ValueError(
            f"{profile.label} queue must be exactly {profile.worker_queue}"
        )
    manifest = _verify_package(args.output_dir)
    deployment, plan = _verify_deployment_plan(args.output_dir, manifest)
    dataset_id = training._clearml_id(
        args.source_dataset_id, f"{profile.label} source Dataset"
    )
    template_id = training._clearml_id(
        args.template_task_id, f"{profile.label} template"
    )
    _downloaded_source_root(manifest, dataset_id=dataset_id)
    transition = _transition(manifest, dataset_id=dataset_id)
    from clearml import Task

    _base, _base_raw_diff, _base_patched_diff = _validated_fixed_base(Task)
    template = Task.get_task(task_id=template_id)
    expected_name = f"{profile.template_prefix} [{str(transition['seal_sha256'])[:12]}]"
    expected_template_parameters = _expected_template_parameters(
        manifest, dataset_id=dataset_id
    )
    _validate_round2_template(
        template,
        expected_name=expected_name,
        expected_status="completed",
        expected_parameters=expected_template_parameters,
    )
    raw_diff = _require_bootstrap_script(
        template,
        expected_sha256=fastlane.TEMPLATE_SCRIPT_SHA256,
        context=f"{profile.label} template",
    )
    patched_diff = _patch_template(raw_diff)
    patched_sha256 = hashlib.sha256(patched_diff.encode("utf-8")).hexdigest()

    _require_transition_artifact(template, transition)

    name = f"{profile.task_prefix} [{str(plan['seal_sha256'])[:12]}]"
    existing = Task.get_tasks(
        task_name=f"^{re.escape(name)}$",
        task_filter={"parent": fastlane.PREDECESSOR_TASK_ID},
    )
    if any(getattr(task, "name", "") == name for task in existing or ()):
        raise RuntimeError(f"refusing duplicate {profile.label} task: {name}")
    task = _clone_task(
        Task,
        source_task=template,
        name=name,
        parent=fastlane.PREDECESSOR_TASK_ID,
    )
    fastlane._edit_script(
        fastlane._task_id(task, context=f"{profile.label} clone"), patched_diff
    )
    parameters = _task_parameters(manifest, dataset_id=dataset_id)
    if task.set_parameters(parameters) is False:
        raise RuntimeError(f"{profile.label} clone rejected sealed parameters")
    task.output_uri = training.FILES_SERVER_URI
    task.reload()
    observed_clone = _validate_training_clone(
        task,
        expected_name=name,
        patched_diff=patched_diff,
        expected_parameters=parameters,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    receipt_payload: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": profile.receipt_type,
        "created_at": created_at,
        "plan_seal_sha256": plan["seal_sha256"],
        "deployment_plan_seal_sha256": deployment["seal_sha256"],
        "source_package_manifest_seal_sha256": manifest["seal_sha256"],
        "source_dataset_id": dataset_id,
        "source_transition_seal_sha256": transition["seal_sha256"],
        "template_task_id": template_id,
        "template_script_sha256": fastlane.TEMPLATE_SCRIPT_SHA256,
        "patched_script_sha256": patched_sha256,
        "task_id": observed_clone["task_id"],
        "task_name": observed_clone["task_name"],
        "parent_task_id": observed_clone["parent_task_id"],
        "project": observed_clone["project"],
        "project_id": observed_clone["project_id"],
        "entry_point": observed_clone["entry_point"],
        "pre_enqueue_queue_id": observed_clone["pre_enqueue_queue_id"],
        "planned_queue": observed_clone["planned_queue"],
        "planned_queue_id": observed_clone["planned_queue_id"],
        "parameters_sha256": hashlib.sha256(
            _canonical_json(parameters).encode("utf-8")
        ).hexdigest(),
        "teacher_task_id": fastlane.TEACHER_TASK_ID,
        "teacher_model_id": fastlane.TEACHER_MODEL_ID,
        "teacher_checkpoint_sha256": fastlane.TEACHER_CHECKPOINT_SHA256,
        "protocol": dict(plan["protocol"]),
        "execution_policy": dict(plan["execution_policy"]),
        "receipt_upload_state": "before_enqueue",
    }
    if profile.early_gate_policy is not None:
        early_gate_policy = plan.get("early_gate_policy")
        if early_gate_policy != dict(profile.early_gate_policy):
            raise RuntimeError(f"{profile.label} early-gate plan binding drifted")
        assert isinstance(early_gate_policy, Mapping)
        _require_seal(
            early_gate_policy, context=f"{profile.label} E10 early-gate policy"
        )
        receipt_payload["early_gate_policy"] = dict(early_gate_policy)
    if profile.execution_contract is not None:
        execution_contract = plan.get("execution_contract")
        if execution_contract != dict(profile.execution_contract):
            raise RuntimeError(f"{profile.label} execution-contract binding drifted")
        assert isinstance(execution_contract, Mapping)
        _require_seal(
            execution_contract,
            context=f"{profile.label} execution contract",
        )
        receipt_payload["execution_contract"] = dict(execution_contract)
    if profile.task_parameter_bindings is not None:
        if parameters | dict(profile.task_parameter_bindings) != parameters:
            raise RuntimeError(f"{profile.label} task parameter bindings drifted")
        receipt_payload["task_parameter_bindings"] = dict(
            profile.task_parameter_bindings
        )
    receipt = _sealed(receipt_payload)
    acknowledgement = _upload_verify_and_enqueue(
        Task,
        task,
        receipt,
        queue=profile.worker_queue,
    )
    output = {
        "task_id": task.id,
        "task_name": name,
        "status": fastlane._status(task),
        "queue": profile.worker_queue,
        "source_dataset_id": dataset_id,
        "template_task_id": template_id,
        "plan_seal_sha256": plan["seal_sha256"],
        "launch_receipt_seal_sha256": receipt["seal_sha256"],
        "patched_script_sha256": patched_sha256,
    }
    if acknowledgement is not None:
        output["enqueue_acknowledgement_seal_sha256"] = acknowledgement["seal_sha256"]
    print(
        json.dumps(
            output,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    profile = _profile()
    parser = argparse.ArgumentParser(description=profile.cli_description)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, default=profile.output_dir)
    prepare.set_defaults(handler=_prepare)

    upload = subparsers.add_parser("upload-source")
    upload.add_argument("--output-dir", type=Path, default=profile.output_dir)
    upload.add_argument("--execute-token", default="")
    upload.set_defaults(handler=_upload_source)

    template = subparsers.add_parser("create-template")
    template.add_argument("--output-dir", type=Path, default=profile.output_dir)
    template.add_argument("--source-dataset-id", required=True)
    template.add_argument("--execute-token", default="")
    template.set_defaults(handler=_create_template)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--output-dir", type=Path, default=profile.output_dir)
    launch.add_argument("--source-dataset-id", required=True)
    launch.add_argument("--template-task-id", required=True)
    launch.add_argument("--queue", default=profile.worker_queue)
    launch.add_argument("--execute-token", default="")
    launch.set_defaults(handler=_launch)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir = args.output_dir.resolve(strict=False)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
