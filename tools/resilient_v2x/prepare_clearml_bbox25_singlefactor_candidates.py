#!/usr/bin/env python3
"""Prepare and gate the mutually-exclusive single-factor bbox2.5 candidates.

Both candidates are independent, one-file additive descendants of the sealed
round2 P2 source package.  This module never edits P0/P1/P2 launchers or their
artifacts.  A training launch additionally requires two local sealed receipts:

* an E10 metric trigger selecting exactly one candidate; and
* a full-model build/initialization/strict-load audit produced inside the exact
  four-A100 training runtime.

The E2+student-bbox2.5 candidate is preferred.  Either single-factor candidate
is a manual fallback and cannot launch until the preferred task is terminal.
Only one fallback can ever pass the group guard.  Group state is checked before
clone and again after receipt readback immediately before enqueue, closing the
normal clone/enqueue race window.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from tools.resilient_v2x import clearml_5090_bootstrap as bootstrap
    from tools.resilient_v2x import clearml_train as source_runner
    from tools.resilient_v2x import prepare_clearml_sota_round2_p0 as shared
    from tools.resilient_v2x import prepare_clearml_sota_round2_p2 as p2
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    _root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_root))
    from tools.resilient_v2x import clearml_5090_bootstrap as bootstrap
    from tools.resilient_v2x import clearml_train as source_runner
    from tools.resilient_v2x import prepare_clearml_sota_round2_p0 as shared
    from tools.resilient_v2x import prepare_clearml_sota_round2_p2 as p2


ROOT = Path(__file__).resolve().parents[2]
P2_OUTPUT_DIR = ROOT / "artifacts/resilient_v2x/sota_round2_p2"
P2_SOURCE_DATASET_ID = "858238049cad4d13918384bb8faec630"
P2_TEMPLATE_TASK_ID = "ac35b8f7403c41e1a8c50137abded09b"
P2_TRAINING_TASK_ID = "f5d3820b4cdf416183c8f1fee566abe3"
P0_TRAINING_TASK_ID = p2.P0_TRAINING_TASK_ID

OUTPUT_ROOT = ROOT / "artifacts/resilient_v2x/sota_bbox25_singlefactor"
MUTUAL_EXCLUSION_GROUP = "bbox25_single_factor_followup"
PRIMARY_KEY = "no_reliability_linear_bbox25"
SECONDARY_KEY = "no_reliability_bbox25"
BACKUP_KEY = "support_residual_bbox25"

TRIGGER_RECEIPT_NAME = "selection-trigger-receipt.json"
MODEL_PREFLIGHT_RECEIPT_NAME = "model-preflight-receipt.json"
SOURCE_INVENTORY_NAME = "source-inventory.json"
MAX_RECEIPT_BYTES = 1024 * 1024

MIN_P2_3D_GAIN_AP = 0.3
MIN_P2_BEV_GAIN_AP = -0.2
TERMINAL_PRIMARY_STATUSES = frozenset({"completed", "failed", "stopped", "closed"})

TRIGGER_POLICY = shared._sealed(
    {
        "schema_version": 1,
        "policy_type": "resilient_v2x_bbox25_singlefactor_trigger_v1",
        "metric_contract": {
            "dataset_slice": "clean-1789",
            "epoch": 10,
            "iou_threshold": 0.70,
            "metrics": ["BEV AP70", "3D AP70"],
            "p0_task_id": P0_TRAINING_TASK_ID,
            "p2_task_id": P2_TRAINING_TASK_ID,
        },
        "bbox_signal": {
            "minimum_p2_minus_p0_3d_ap70": MIN_P2_3D_GAIN_AP,
            "minimum_p2_minus_p0_bev_ap70": MIN_P2_BEV_GAIN_AP,
            "boolean_rule": ("p2_3d_minus_p0 >= 0.3 AND p2_bev_minus_p0 >= -0.2"),
        },
        "priority_tiers": [[PRIMARY_KEY], [SECONDARY_KEY, BACKUP_KEY]],
        "selection": {
            PRIMARY_KEY: "authorized only when bbox_signal is true",
            SECONDARY_KEY: (
                "authorized only when bbox_signal is true and the preferred "
                "E2+bbox2.5 task is terminal with an explicit fallback reason"
            ),
            BACKUP_KEY: (
                "authorized only when bbox_signal is true and the preferred "
                "E2+bbox2.5 task is terminal with an explicit fallback reason"
            ),
        },
        "automatic_enqueue_count": 0,
        "simultaneous_or_parallel_launch_forbidden": True,
    }
)

MODEL_PREFLIGHT_POLICY = shared._sealed(
    {
        "schema_version": 1,
        "policy_type": "resilient_v2x_bbox25_full_model_preflight_v1",
        "required_checks": [
            "sealed target source inventory matches the extracted source",
            "exact four-A100 training runtime and native bundle bindings match",
            "base and candidate resolve and build as complete registry models",
            "both complete models finish MMEngine init_weights",
            "candidate strict-loads the complete base state with no incompatibility",
            "complete pre/post state hashes are equal",
            "student bbox loss is 2.5 and nested frozen-teacher bbox loss is 2.0",
        ],
        "runtime": {
            "python": [3, 12],
            "torch": bootstrap.EXPECTED_TORCH,
            "torch_cuda": bootstrap.EXPECTED_TORCH_CUDA,
            "gpu_count": 4,
            "capability": [8, 0],
            "precision": "FP32",
            "base_image_manifest_digest": bootstrap.BASE_IMAGE_AMD64_MANIFEST_DIGEST,
            "base_image_config_digest": bootstrap.BASE_IMAGE_CONFIG_DIGEST,
            "native_build_task_id": bootstrap.BUILD_TASK_ID,
            "build_manifest_sha256": shared.fastlane.EXPECTED_TEMPLATE_PARAMETERS[
                "Args/build_manifest_sha256"
            ],
            "native_bundle_bytes": shared.fastlane.EXPECTED_TEMPLATE_PARAMETERS[
                "Args/native_bundle_bytes"
            ],
            "native_bundle_sha256": shared.fastlane.EXPECTED_TEMPLATE_PARAMETERS[
                "Args/native_bundle_sha256"
            ],
            "packages": dict(bootstrap.EXPECTED_PACKAGES),
        },
    }
)


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    priority: str
    experiment: str
    identity: str
    config: str
    base_experiment: str
    base_config: str
    peer_identity: str
    output_dir: Path
    manifest_name: str
    plan_name: str
    source_dataset_prefix: str
    template_prefix: str
    task_prefix: str
    upload_token: str
    template_token: str
    launch_token: str


CANDIDATE_SPECS = {
    PRIMARY_KEY: CandidateSpec(
        key=PRIMARY_KEY,
        priority="preferred",
        experiment=PRIMARY_KEY,
        identity="dair_improvement_no_reliability_linear_bbox25",
        config=("configs/resilient_v2x/improvements/no_reliability_linear_bbox25.py"),
        base_experiment="dair_improvement_no_reliability_linear",
        base_config=("configs/resilient_v2x/improvements/no_reliability_linear.py"),
        peer_identity="dair_improvement_no_reliability_bbox25",
        output_dir=OUTPUT_ROOT / PRIMARY_KEY,
        manifest_name="no-reliability-linear-bbox25-source-package.json",
        plan_name="no-reliability-linear-bbox25-deployment-plan.json",
        source_dataset_prefix=(
            "ResilientV2X sealed no-reliability linear-PTF bbox2.5 source"
        ),
        template_prefix=(
            "ResilientV2X no-reliability linear-PTF bbox2.5 bootstrap template"
        ),
        task_prefix=("ResilientV2X bbox25 preferred no-reliability linear-PTF"),
        upload_token="UPLOAD_EXACT_NO_RELIABILITY_LINEAR_BBOX25_SOURCE",
        template_token="CREATE_EXACT_NO_RELIABILITY_LINEAR_BBOX25_TEMPLATE",
        launch_token="LAUNCH_EXACT_NO_RELIABILITY_LINEAR_BBOX25_A100",
    ),
    SECONDARY_KEY: CandidateSpec(
        key=SECONDARY_KEY,
        priority="fallback",
        experiment=SECONDARY_KEY,
        identity="dair_improvement_no_reliability_bbox25",
        config="configs/resilient_v2x/improvements/no_reliability_bbox25.py",
        base_experiment="dair_ablation_no_reliability",
        base_config="configs/resilient_v2x/ablations/no_reliability.py",
        peer_identity="dair_improvement_support_residual_bbox25",
        output_dir=OUTPUT_ROOT / SECONDARY_KEY,
        manifest_name="no-reliability-bbox25-source-package.json",
        plan_name="no-reliability-bbox25-deployment-plan.json",
        source_dataset_prefix="ResilientV2X sealed no-reliability bbox2.5 source",
        template_prefix="ResilientV2X no-reliability bbox2.5 bootstrap template",
        task_prefix="ResilientV2X bbox25 single-factor no-reliability",
        upload_token="UPLOAD_EXACT_NO_RELIABILITY_BBOX25_SOURCE",
        template_token="CREATE_EXACT_NO_RELIABILITY_BBOX25_TEMPLATE",
        launch_token="LAUNCH_EXACT_NO_RELIABILITY_BBOX25_A100",
    ),
    BACKUP_KEY: CandidateSpec(
        key=BACKUP_KEY,
        priority="fallback",
        experiment=BACKUP_KEY,
        identity="dair_improvement_support_residual_bbox25",
        config="configs/resilient_v2x/improvements/support_residual_bbox25.py",
        base_experiment="dair_improvement_support_residual",
        base_config="configs/resilient_v2x/improvements/support_residual.py",
        peer_identity="dair_improvement_no_reliability_bbox25",
        output_dir=OUTPUT_ROOT / BACKUP_KEY,
        manifest_name="support-residual-bbox25-source-package.json",
        plan_name="support-residual-bbox25-deployment-plan.json",
        source_dataset_prefix="ResilientV2X sealed support-residual bbox2.5 source",
        template_prefix="ResilientV2X support-residual bbox2.5 bootstrap template",
        task_prefix="ResilientV2X bbox25 single-factor support-residual",
        upload_token="UPLOAD_EXACT_SUPPORT_RESIDUAL_BBOX25_SOURCE",
        template_token="CREATE_EXACT_SUPPORT_RESIDUAL_BBOX25_TEMPLATE",
        launch_token="LAUNCH_EXACT_SUPPORT_RESIDUAL_BBOX25_A100",
    ),
}

_SPEC_ANCHOR = """    ExperimentSpec(
        "linear_no_distillation",
"""
_NESTED_ANCHOR = """        "support_residual_no_reliability_linear_bbox25",
        "resilient_v2x",
"""


def _verified_p2_manifest() -> dict[str, object]:
    with shared.use_deployment_profile(p2.P2_PROFILE):
        return shared._verify_package(P2_OUTPUT_DIR)


def _execution_contract(spec: CandidateSpec) -> dict[str, object]:
    return shared._sealed(
        {
            "schema_version": 1,
            "contract_type": "resilient_v2x_bbox25_singlefactor_launch_v1",
            "candidate_key": spec.key,
            "candidate_priority": spec.priority,
            "candidate_identity": spec.identity,
            "peer_identity": spec.peer_identity,
            "mutual_exclusion_group": MUTUAL_EXCLUSION_GROUP,
            "trigger_policy": dict(TRIGGER_POLICY),
            "model_preflight_policy": dict(MODEL_PREFLIGHT_POLICY),
            "pre_enqueue_order": [
                "verify candidate package and deployment seals",
                "verify selected-candidate trigger receipt",
                "verify exact-runtime complete-model preflight receipt",
                "query the complete mutual-exclusion group before clone",
                "clone and verify task identity",
                "upload and force-readback the augmented launch receipt",
                "query the complete mutual-exclusion group immediately before enqueue",
                "enqueue exactly one task to GPU4-A100",
            ],
            "automatic_enqueue": False,
            "concurrent_launchers_supported": False,
        }
    )


def _build_profile(spec: CandidateSpec) -> shared.Round2DeploymentProfile:
    parent_manifest = _verified_p2_manifest()
    parent_package = parent_manifest["target_source_package"]
    parent_candidate = parent_manifest["candidate"]
    assert isinstance(parent_package, Mapping)
    assert isinstance(parent_candidate, Mapping)
    spec_patch = f'''    ExperimentSpec(
        "{spec.experiment}",
        "sota_candidate",
        "{spec.config}",
        True,
    ),
{_SPEC_ANCHOR}'''
    nested_patch = (
        '        "support_residual_no_reliability_linear_bbox25",\n'
        f'        "{spec.experiment}",\n'
        '        "resilient_v2x",\n'
    )
    provenance = {
        "derivation": f"{spec.key} exact additive descendant of sealed P2 source",
        "source_parent_candidate": dict(parent_candidate),
        "scientific_base_experiment": spec.base_experiment,
        "scientific_base_config": spec.base_config,
        "parent_source_dataset_id": P2_SOURCE_DATASET_ID,
        "parent_source_package_manifest_seal_sha256": parent_manifest["seal_sha256"],
        "parent_source_tree_sha256": parent_package["tree_sha256"],
        "parent_template_task_id": P2_TEMPLATE_TASK_ID,
        "parent_training_task_id": P2_TRAINING_TASK_ID,
        "p0_training_task_id": P0_TRAINING_TASK_ID,
        "mutual_exclusion_group": MUTUAL_EXCLUSION_GROUP,
        "candidate_priority": spec.priority,
    }
    return shared.Round2DeploymentProfile(
        label=f"bbox25 single-factor {spec.key}",
        output_dir=spec.output_dir,
        package_manifest_name=spec.manifest_name,
        deployment_plan_name=spec.plan_name,
        package_type=f"resilient_v2x_{spec.key}_sealed_source",
        experiment=spec.experiment,
        config=spec.config,
        identity=spec.identity,
        base_source_dir=P2_OUTPUT_DIR,
        base_source_inventory=P2_OUTPUT_DIR / SOURCE_INVENTORY_NAME,
        base_source_archive=P2_OUTPUT_DIR / str(parent_package["archive_name"]),
        base_source_dataset_id=P2_SOURCE_DATASET_ID,
        base_source_package=dict(parent_package),
        provenance=provenance,
        early_gate_policy=None,
        source_transition_artifact=f"{spec.key}_source_transition",
        launch_receipt_artifact=f"{spec.key}_launch_receipt",
        source_dataset_prefix=spec.source_dataset_prefix,
        source_dataset_version_suffix=f"{spec.key}-v1",
        source_dataset_tags=(
            "ResilientV2X",
            "source",
            "sota-candidates",
            "bbox25-singlefactor",
            spec.priority,
            spec.key,
            "derived-from-p2",
            "sealed",
        ),
        template_prefix=spec.template_prefix,
        task_prefix=spec.task_prefix,
        worker_queue=p2.WORKER_QUEUE,
        worker_queue_id=p2.WORKER_QUEUE_ID,
        gpu_count=p2.GPU_COUNT,
        batch_size_per_gpu=p2.BATCH_SIZE_PER_GPU,
        global_batch_size=p2.GLOBAL_BATCH_SIZE,
        max_epochs=p2.MAX_EPOCHS,
        val_interval=p2.VAL_INTERVAL,
        training_seed=p2.TRAINING_SEED,
        precision=p2.PRECISION,
        upload_token=spec.upload_token,
        template_token=spec.template_token,
        launch_token=spec.launch_token,
        spec_anchor=_SPEC_ANCHOR,
        spec_patch=spec_patch,
        nested_anchor=_NESTED_ANCHOR,
        nested_patch=nested_patch,
        bootstrap_parent=p2.P2_PROFILE,
        transition_type=f"exact_additive_{spec.key}_from_sealed_p2",
        plan_type=f"resilient_v2x_{spec.key}_sealed_a100_v1",
        receipt_type=f"resilient_v2x_{spec.key}_launch_v1",
        template_status_message=f"sealed {spec.key} bootstrap template",
        transition_description=(
            f"exact one-config additive {spec.key} source derived from sealed P2"
        ),
        invariants=(
            "source transition adds only the selected single-factor bbox2.5 config",
            "existing P0/P1/P2 packages, Datasets, templates, tasks and seals are untouched",
            "seed 20250218, FP32, global batch 8, 50 epochs and val/10 remain fixed",
            "a sealed metric trigger selects exactly one candidate",
            "full training-runtime model build and strict-load must pass before clone",
            "the mutual-exclusion group is checked before clone and before enqueue",
        ),
        duplicate_guard_description=(
            "group-wide exact parent/name/status guard before clone plus own clone "
            "guard and group-wide guard after receipt readback immediately before enqueue"
        ),
        duplicate_guard_stages=(
            "pre_clone_requires_authorized_group_state",
            "post_receipt_requires_authorized_group_state_and_current_clone_id",
        ),
        relative_script=Path(__file__).relative_to(ROOT),
        cli_description=__doc__ or "",
        execution_contract=_execution_contract(spec),
    )


PROFILES = {key: _build_profile(spec) for key, spec in CANDIDATE_SPECS.items()}
_ORIGINAL_DEPLOYMENT_PLAN = shared._deployment_plan


def _profile_for(key: str) -> shared.Round2DeploymentProfile:
    try:
        return PROFILES[key]
    except KeyError as error:
        raise ValueError(f"unknown bbox25 candidate: {key!r}") from error


def _spec_for(key: str) -> CandidateSpec:
    try:
        return CANDIDATE_SPECS[key]
    except KeyError as error:
        raise ValueError(f"unknown bbox25 candidate: {key!r}") from error


def _custom_deployment_plan(
    manifest: Mapping[str, object], *, output_dir: Path
) -> dict[str, object]:
    """Add the two mandatory local-gate commands without changing shared code."""

    profile = shared._profile()
    base = _ORIGINAL_DEPLOYMENT_PLAN(manifest, output_dir=output_dir)
    spec = _spec_for(profile.experiment)
    static_seal = str(base["static_plan_seal_sha256"])
    script = profile.relative_script
    common = f"--candidate {spec.key} --output-dir {output_dir}"
    trigger_args = (
        "--p0-bev-ap70 <P0_E10_BEV_AP70> --p0-3d-ap70 <P0_E10_3D_AP70> "
        "--p2-bev-ap70 <P2_E10_BEV_AP70> --p2-3d-ap70 <P2_E10_3D_AP70>"
    )
    if spec.priority != "preferred":
        trigger_args += (
            " --primary-task-id <PRIMARY_TASK_ID> "
            "--primary-status <PRIMARY_TERMINAL_STATUS> "
            "--fallback-reason <EXPLICIT_FALLBACK_REASON>"
        )
    payload = dict(base)
    payload.pop("seal_sha256", None)
    payload["commands"] = {
        "1_upload_source": (
            f"python {script} upload-source {common} "
            f"--execute-token {spec.upload_token}"
        ),
        "2_create_template": (
            f"python {script} create-template {common} "
            "--source-dataset-id <DATASET_ID> "
            f"--execute-token {spec.template_token}"
        ),
        "3_model_preflight_in_exact_a100_runtime": (
            f"python {script} model-preflight {common} "
            "--source-root <EXTRACTED_TRAINING_SOURCE_ROOT> "
            "--teacher-checkpoint <PINNED_TEACHER_CHECKPOINT> "
            "--resnet-checkpoint <PINNED_RESNET50_CHECKPOINT>"
        ),
        "4_build_selection_trigger_local": (
            f"python {script} build-trigger {common} {trigger_args}"
        ),
        "5_launch_selected_a100": (
            f"python {script} launch {common} "
            "--source-dataset-id <DATASET_ID> --template-task-id <TEMPLATE_ID> "
            f"--queue {profile.worker_queue} "
            f"--trigger-receipt {output_dir / TRIGGER_RECEIPT_NAME} "
            f"--model-preflight-receipt {output_dir / MODEL_PREFLIGHT_RECEIPT_NAME} "
            f"--execute-token {spec.launch_token}"
        ),
        "6_show_execution_contract": (
            f"python {script} show-contract {common} "
            f"--static-plan-seal-sha256 {static_seal}"
        ),
    }
    return shared._sealed(payload)


@contextlib.contextmanager
def _candidate_engine(key: str) -> Iterator[shared.Round2DeploymentProfile]:
    """Use one candidate profile and restore the shared engine unconditionally."""

    profile = _profile_for(key)
    previous = shared._deployment_plan
    shared._deployment_plan = _custom_deployment_plan
    try:
        with shared.use_deployment_profile(profile):
            yield profile
    finally:
        shared._deployment_plan = previous


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_metric(value: float, *, name: str) -> float:
    observed = float(value)
    if not math.isfinite(observed) or observed < 0.0 or observed > 100.0:
        raise ValueError(f"{name} must be finite and in [0, 100]")
    return observed


def _normalized_status(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").rsplit(".", 1)[-1].lower()


def _read_receipt(path: Path, *, context: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} must be a regular file")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_RECEIPT_BYTES:
        raise RuntimeError(f"{context} has an invalid size")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must contain a JSON object")
    shared._require_seal(value, context=context)
    return value


def _write_new_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    shared._require_seal(receipt, context=f"new receipt {path.name}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _candidate_bindings(
    manifest: Mapping[str, object],
    deployment: Mapping[str, object],
    static_plan: Mapping[str, object],
) -> dict[str, object]:
    package = manifest.get("target_source_package")
    if not isinstance(package, Mapping):
        raise RuntimeError("candidate package has no target source package")
    return {
        "source_package_manifest_seal_sha256": manifest["seal_sha256"],
        "source_tree_sha256": package["tree_sha256"],
        "deployment_plan_seal_sha256": deployment["seal_sha256"],
        "static_plan_seal_sha256": static_plan["seal_sha256"],
    }


def _expected_task_name(spec: CandidateSpec, static_plan: Mapping[str, object]) -> str:
    return f"{spec.task_prefix} [{str(static_plan['seal_sha256'])[:12]}]"


def build_trigger_receipt(
    *,
    key: str,
    bindings: Mapping[str, object],
    static_plan: Mapping[str, object],
    p0_bev_ap70: float,
    p0_3d_ap70: float,
    p2_bev_ap70: float,
    p2_3d_ap70: float,
    primary_task_id: str = "",
    primary_task_name: str = "",
    primary_status: str = "",
    fallback_reason: str = "",
    created_at: str | None = None,
) -> dict[str, object]:
    """Create a sealed decision; creating it never enqueues either candidate."""

    spec = _spec_for(key)
    values = {
        "p0_bev_ap70": _validated_metric(p0_bev_ap70, name="P0 BEV AP70"),
        "p0_3d_ap70": _validated_metric(p0_3d_ap70, name="P0 3D AP70"),
        "p2_bev_ap70": _validated_metric(p2_bev_ap70, name="P2 BEV AP70"),
        "p2_3d_ap70": _validated_metric(p2_3d_ap70, name="P2 3D AP70"),
    }
    delta_bev = values["p2_bev_ap70"] - values["p0_bev_ap70"]
    delta_3d = values["p2_3d_ap70"] - values["p0_3d_ap70"]
    bbox_signal = delta_bev >= MIN_P2_BEV_GAIN_AP and delta_3d >= MIN_P2_3D_GAIN_AP

    fallback: dict[str, object] | None = None
    if spec.priority == "preferred":
        if primary_task_id or primary_task_name or primary_status or fallback_reason:
            raise ValueError("preferred trigger cannot contain fallback fields")
        authorized = bbox_signal
    else:
        primary_id = shared.training._clearml_id(primary_task_id, "primary bbox25 task")
        primary_name = primary_task_name.strip()
        status = _normalized_status(primary_status)
        reason = fallback_reason.strip()
        if re.fullmatch(
            _group_regex(), primary_name
        ) is None or not primary_name.startswith(
            CANDIDATE_SPECS[PRIMARY_KEY].task_prefix + " ["
        ):
            raise ValueError("fallback trigger requires the sealed preferred task name")
        if status not in TERMINAL_PRIMARY_STATUSES:
            raise ValueError("fallback trigger requires a terminal preferred status")
        if not reason:
            raise ValueError("fallback trigger requires an explicit fallback reason")
        fallback = {
            "primary_task_id": primary_id,
            "primary_task_name": primary_name,
            "primary_terminal_status": status,
            "reason": reason,
        }
        authorized = bbox_signal

    return shared._sealed(
        {
            "schema_version": 1,
            "receipt_type": "resilient_v2x_bbox25_selection_trigger_v1",
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "policy": dict(TRIGGER_POLICY),
            "bindings": dict(bindings),
            "selected_candidate": {
                "key": spec.key,
                "priority": spec.priority,
                "experiment": spec.experiment,
                "identity": spec.identity,
                "config": spec.config,
                "peer_identity": spec.peer_identity,
                "expected_task_name": _expected_task_name(spec, static_plan),
            },
            "observations": {
                "p0": {
                    "task_id": P0_TRAINING_TASK_ID,
                    "epoch": 10,
                    "bev_ap70": values["p0_bev_ap70"],
                    "3d_ap70": values["p0_3d_ap70"],
                },
                "p2": {
                    "task_id": P2_TRAINING_TASK_ID,
                    "epoch": 10,
                    "bev_ap70": values["p2_bev_ap70"],
                    "3d_ap70": values["p2_3d_ap70"],
                },
                "p2_minus_p0": {"bev_ap70": delta_bev, "3d_ap70": delta_3d},
            },
            "fallback": fallback,
            "decision": {
                "bbox_signal": bbox_signal,
                "authorized": authorized,
                "automatic_enqueue": False,
                "selected_count": 1,
            },
        }
    )


def _inventory_entry(
    inventory: Mapping[str, object], relative: str
) -> dict[str, object]:
    raw = inventory.get("files")
    if not isinstance(raw, list):
        raise RuntimeError("source inventory has no file list")
    matches = [
        item
        for item in raw
        if isinstance(item, Mapping) and item.get("path") == relative
    ]
    if len(matches) != 1:
        raise RuntimeError(f"source inventory expected one entry for {relative}")
    return dict(matches[0])


def _verify_extracted_source(
    source_root: Path, inventory: Mapping[str, object]
) -> dict[str, object]:
    source_root = source_root.resolve(strict=True)
    raw = inventory.get("files")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("source inventory file list is invalid")
    compatibility_modified: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise RuntimeError("source inventory contains a non-object entry")
        relative = str(item.get("path") or "")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise RuntimeError(f"unsafe source inventory path: {relative!r}")
        path = source_root.joinpath(*pure.parts)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"extracted source member is missing: {relative}")
        observed_size = path.stat().st_size
        observed_sha = _sha256_file(path)
        expected_size = item.get("size")
        expected_sha = item.get("sha256")
        if observed_size == expected_size and observed_sha == expected_sha:
            continue
        if relative != "tools/resilient_v2x/clearml_train.py":
            raise RuntimeError(f"extracted source member drifted: {relative}")
        patched = bootstrap.CLEARML_TRAIN_METRICS_COMPATIBILITY_IDENTITIES.get(
            str(expected_sha)
        )
        allowed = {str(expected_sha), str(patched or "")}
        if observed_sha not in allowed:
            raise RuntimeError("extracted source training runner identity drifted")
        compatibility_modified.append(
            {
                "path": relative,
                "from_sha256": str(expected_sha),
                "to_sha256": observed_sha,
            }
        )
    return {
        "checked_file_count": len(raw),
        "inventory_tree_sha256": inventory.get("tree_sha256"),
        "mismatches": [],
        "compatibility_modified_paths": compatibility_modified,
    }


def _require_pinned_file(path: Path, *, expected_sha256: str, context: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} must be a regular file")
    resolved = path.resolve(strict=True)
    if resolved.stat().st_size <= 0 or _sha256_file(resolved) != expected_sha256:
        raise RuntimeError(f"{context} SHA-256 drifted")
    return resolved


def _runtime_audit() -> dict[str, object]:
    import torch

    runtime = {
        "python": list(sys.version_info[:2]),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "capabilities": [
            list(torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        ],
        "torch_arch_list": list(torch.cuda.get_arch_list()),
        "packages": {
            name: importlib.metadata.version(name)
            for name in bootstrap.EXPECTED_PACKAGES
        },
        "base_image_manifest_digest": os.environ.get(
            "RESILIENT_V2X_CONTAINER_IMAGE_DIGEST"
        ),
        "base_image_config_digest": os.environ.get(
            "RESILIENT_V2X_5090_BASE_IMAGE_CONFIG_DIGEST"
        ),
        "native_build_task_id": os.environ.get("RESILIENT_V2X_5090_BUILD_TASK_ID"),
        "build_manifest_sha256": os.environ.get(
            "RESILIENT_V2X_5090_BUILD_MANIFEST_SHA256"
        ),
        "native_bundle_bytes": shared.fastlane.EXPECTED_TEMPLATE_PARAMETERS[
            "Args/native_bundle_bytes"
        ],
        "native_bundle_sha256": os.environ.get(
            "RESILIENT_V2X_5090_NATIVE_BUNDLE_SHA256"
        ),
        "precision": "FP32",
    }
    expected = MODEL_PREFLIGHT_POLICY["runtime"]
    assert isinstance(expected, Mapping)
    exact_fields = {
        "python": expected["python"],
        "torch": expected["torch"],
        "torch_cuda": expected["torch_cuda"],
        "cuda_available": True,
        "gpu_count": 4,
        "capabilities": [[8, 0], [8, 0], [8, 0], [8, 0]],
        "packages": expected["packages"],
        "base_image_manifest_digest": expected["base_image_manifest_digest"],
        "base_image_config_digest": expected["base_image_config_digest"],
        "native_build_task_id": expected["native_build_task_id"],
        "build_manifest_sha256": expected["build_manifest_sha256"],
        "native_bundle_bytes": expected["native_bundle_bytes"],
        "native_bundle_sha256": expected["native_bundle_sha256"],
        "precision": "FP32",
    }
    for field, value in exact_fields.items():
        if runtime.get(field) != value:
            raise RuntimeError(
                f"model preflight runtime {field} mismatch: expected {value!r}, "
                f"got {runtime.get(field)!r}"
            )
    if "sm_120" not in runtime["torch_arch_list"]:
        raise RuntimeError("model preflight Torch build lacks sealed sm_120 support")
    return runtime


@contextlib.contextmanager
def _model_environment(
    *, source_root: Path, teacher_checkpoint: Path, resnet_checkpoint: Path
) -> Iterator[None]:
    previous_path = list(sys.path)
    keys = (
        "RESILIENT_V2X_TEACHER_CHECKPOINT",
        "RESILIENT_V2X_COMMON_INIT_CHECKPOINT",
        "RESILIENT_V2X_COMMON_INIT_SHA256",
        "RESILIENT_V2X_RESNET50_CHECKPOINT",
    )
    previous_env = {key: os.environ.get(key) for key in keys}
    teacher_sha = _sha256_file(teacher_checkpoint)
    for name in tuple(sys.modules):
        if name == "transvision" or name.startswith("transvision."):
            del sys.modules[name]
    sys.path.insert(0, str(source_root))
    os.environ.update(
        {
            "RESILIENT_V2X_TEACHER_CHECKPOINT": str(teacher_checkpoint),
            "RESILIENT_V2X_COMMON_INIT_CHECKPOINT": str(teacher_checkpoint),
            "RESILIENT_V2X_COMMON_INIT_SHA256": teacher_sha,
            "RESILIENT_V2X_RESNET50_CHECKPOINT": str(resnet_checkpoint),
        }
    )
    try:
        yield
    finally:
        sys.path[:] = previous_path
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _state_schema_sha256(state: Mapping[str, Any]) -> str:
    schema = [
        [key, str(value.dtype), list(value.shape)]
        for key, value in sorted(state.items())
    ]
    return hashlib.sha256(shared._canonical_json(schema).encode("utf-8")).hexdigest()


def _full_model_audit(
    *,
    spec: CandidateSpec,
    source_root: Path,
    teacher_checkpoint: Path,
    resnet_checkpoint: Path,
) -> dict[str, object]:
    import torch

    with _model_environment(
        source_root=source_root,
        teacher_checkpoint=teacher_checkpoint,
        resnet_checkpoint=resnet_checkpoint,
    ):
        from mmengine.config import Config
        from mmdet3d.registry import MODELS
        from transvision.models.common_teacher_initialization import (
            tensor_mapping_sha256,
        )
        from transvision.register import register_resilient_v2x_modules

        register_resilient_v2x_modules()
        base_cfg = Config.fromfile(str(source_root / spec.base_config))
        candidate_cfg = Config.fromfile(str(source_root / spec.config))
        torch.manual_seed(shared.TRAINING_SEED)
        base_model = MODELS.build(base_cfg.model)
        base_model.init_weights()
        torch.manual_seed(shared.TRAINING_SEED)
        candidate_model = MODELS.build(candidate_cfg.model)
        candidate_model.init_weights()

        base_state = base_model.state_dict()
        candidate_state_before = candidate_model.state_dict()
        if tuple(base_state) != tuple(candidate_state_before):
            raise RuntimeError("complete base/candidate state key order differs")
        if _state_schema_sha256(base_state) != _state_schema_sha256(
            candidate_state_before
        ):
            raise RuntimeError("complete base/candidate state schema differs")
        source_hash = tensor_mapping_sha256(base_state)
        incompatible = candidate_model.load_state_dict(base_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("strict complete-model load returned incompatibilities")
        target_state = candidate_model.state_dict()
        target_hash = tensor_mapping_sha256(target_state)
        if source_hash != target_hash:
            raise RuntimeError("strict complete-model load changed tensor values")

        student_weight = float(candidate_model.bbox_head.loss_bbox.loss_weight)
        base_weight = float(base_model.bbox_head.loss_bbox.loss_weight)
        teacher = getattr(candidate_model, "teacher", None)
        nested = getattr(teacher, "teacher", None)
        if nested is None:
            raise RuntimeError("candidate full model has no nested frozen teacher")
        teacher_weight = float(nested.bbox_head.loss_bbox.loss_weight)
        if (base_weight, student_weight, teacher_weight) != (2.0, 2.5, 2.0):
            raise RuntimeError("bbox loss weights violate the single-factor contract")

        loaded_paths = {}
        for name in (
            "transvision",
            "transvision.register",
            "transvision.models.detectors.resilient_v2x",
        ):
            module = sys.modules.get(name)
            path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
            path.relative_to(source_root)
            loaded_paths[name] = path.relative_to(source_root).as_posix()

        return {
            "base_full_model_build_passed": True,
            "candidate_full_model_build_passed": True,
            "base_init_weights_passed": True,
            "candidate_init_weights_passed": True,
            "strict_load_passed": True,
            "strict_missing_keys": [],
            "strict_unexpected_keys": [],
            "state_key_count": len(base_state),
            "state_schema_sha256": _state_schema_sha256(base_state),
            "source_state_sha256": source_hash,
            "target_state_sha256": target_hash,
            "complete_state_equal_after_strict_load": True,
            "base_student_bbox_loss_weight": base_weight,
            "candidate_student_bbox_loss_weight": student_weight,
            "candidate_nested_teacher_bbox_loss_weight": teacher_weight,
            "base_model_type": type(base_model).__name__,
            "candidate_model_type": type(candidate_model).__name__,
            "loaded_source_modules": loaded_paths,
        }


def _build_model_preflight_receipt(
    *,
    spec: CandidateSpec,
    bindings: Mapping[str, object],
    manifest: Mapping[str, object],
    inventory: Mapping[str, object],
    source_root: Path,
    teacher_checkpoint: Path,
    resnet_checkpoint: Path,
    created_at: str | None = None,
) -> dict[str, object]:
    runtime = _runtime_audit()
    source_audit = _verify_extracted_source(source_root, inventory)
    model_audit = _full_model_audit(
        spec=spec,
        source_root=source_root,
        teacher_checkpoint=teacher_checkpoint,
        resnet_checkpoint=resnet_checkpoint,
    )
    config_entry = _inventory_entry(inventory, spec.config)
    base_entry = _inventory_entry(inventory, spec.base_config)
    return shared._sealed(
        {
            "schema_version": 1,
            "receipt_type": "resilient_v2x_bbox25_full_model_preflight_v1",
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "policy": dict(MODEL_PREFLIGHT_POLICY),
            "bindings": dict(bindings),
            "candidate": dict(manifest["candidate"]),
            "config": {
                "candidate_path": spec.config,
                "candidate_sha256": config_entry["sha256"],
                "base_path": spec.base_config,
                "base_sha256": base_entry["sha256"],
            },
            "checkpoints": {
                "teacher_sha256": _sha256_file(teacher_checkpoint),
                "resnet50_sha256": _sha256_file(resnet_checkpoint),
            },
            "runtime": runtime,
            "source_audit": source_audit,
            "model_audit": model_audit,
            "authorized": True,
        }
    )


def _require_trigger_receipt(
    receipt: Mapping[str, object],
    *,
    spec: CandidateSpec,
    bindings: Mapping[str, object],
) -> None:
    shared._require_seal(receipt, context=f"{spec.key} trigger receipt")
    if receipt.get("receipt_type") != "resilient_v2x_bbox25_selection_trigger_v1":
        raise RuntimeError("bbox25 trigger receipt type drifted")
    if receipt.get("policy") != dict(TRIGGER_POLICY):
        raise RuntimeError("bbox25 trigger policy drifted")
    if receipt.get("bindings") != dict(bindings):
        raise RuntimeError("bbox25 trigger package/plan bindings drifted")
    selected = receipt.get("selected_candidate")
    if not isinstance(selected, Mapping) or selected.get("key") != spec.key:
        raise RuntimeError("bbox25 trigger selected a different candidate")
    if (
        selected.get("identity") != spec.identity
        or selected.get("config") != spec.config
    ):
        raise RuntimeError("bbox25 trigger candidate identity drifted")
    decision = receipt.get("decision")
    if not isinstance(decision, Mapping) or decision.get("authorized") is not True:
        raise RuntimeError("bbox25 trigger did not authorize launch")
    if decision.get("bbox_signal") is not True or decision.get("selected_count") != 1:
        raise RuntimeError("bbox25 trigger decision drifted")
    if decision.get("automatic_enqueue") is not False:
        raise RuntimeError("bbox25 trigger cannot authorize automatic enqueue")
    fallback = receipt.get("fallback")
    if spec.priority == "preferred" and fallback is not None:
        raise RuntimeError(
            "preferred bbox25 trigger unexpectedly contains fallback state"
        )
    if spec.priority != "preferred":
        if not isinstance(fallback, Mapping):
            raise RuntimeError("backup bbox25 trigger has no primary terminal receipt")
        if _normalized_status(fallback.get("primary_terminal_status")) not in (
            TERMINAL_PRIMARY_STATUSES
        ):
            raise RuntimeError("backup bbox25 trigger primary status is not terminal")
        if not str(fallback.get("reason") or "").strip():
            raise RuntimeError("backup bbox25 trigger has no fallback reason")


def _require_model_preflight_receipt(
    receipt: Mapping[str, object],
    *,
    spec: CandidateSpec,
    manifest: Mapping[str, object],
    bindings: Mapping[str, object],
    inventory: Mapping[str, object],
) -> None:
    shared._require_seal(receipt, context=f"{spec.key} model preflight receipt")
    if receipt.get("receipt_type") != "resilient_v2x_bbox25_full_model_preflight_v1":
        raise RuntimeError("bbox25 model preflight receipt type drifted")
    if receipt.get("policy") != dict(MODEL_PREFLIGHT_POLICY):
        raise RuntimeError("bbox25 model preflight policy drifted")
    if receipt.get("bindings") != dict(bindings):
        raise RuntimeError("bbox25 model preflight bindings drifted")
    if receipt.get("candidate") != dict(manifest["candidate"]):
        raise RuntimeError("bbox25 model preflight candidate drifted")
    if receipt.get("authorized") is not True:
        raise RuntimeError("bbox25 model preflight is not authorized")
    expected_config = {
        "candidate_path": spec.config,
        "candidate_sha256": _inventory_entry(inventory, spec.config)["sha256"],
        "base_path": spec.base_config,
        "base_sha256": _inventory_entry(inventory, spec.base_config)["sha256"],
    }
    if receipt.get("config") != expected_config:
        raise RuntimeError("bbox25 model preflight config binding drifted")
    checkpoints = receipt.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or checkpoints != {
        "teacher_sha256": shared.fastlane.TEACHER_CHECKPOINT_SHA256,
        "resnet50_sha256": source_runner.EXPECTED_RESNET_SHA256,
    }:
        raise RuntimeError("bbox25 model preflight checkpoint binding drifted")
    runtime = receipt.get("runtime")
    expected_runtime = MODEL_PREFLIGHT_POLICY["runtime"]
    if not isinstance(runtime, Mapping) or not isinstance(expected_runtime, Mapping):
        raise RuntimeError("bbox25 model preflight runtime is invalid")
    exact_runtime = {
        "python": expected_runtime["python"],
        "torch": expected_runtime["torch"],
        "torch_cuda": expected_runtime["torch_cuda"],
        "cuda_available": True,
        "gpu_count": 4,
        "capabilities": [[8, 0], [8, 0], [8, 0], [8, 0]],
        "packages": expected_runtime["packages"],
        "base_image_manifest_digest": expected_runtime["base_image_manifest_digest"],
        "base_image_config_digest": expected_runtime["base_image_config_digest"],
        "native_build_task_id": expected_runtime["native_build_task_id"],
        "build_manifest_sha256": expected_runtime["build_manifest_sha256"],
        "native_bundle_bytes": expected_runtime["native_bundle_bytes"],
        "native_bundle_sha256": expected_runtime["native_bundle_sha256"],
        "precision": "FP32",
    }
    for field, value in exact_runtime.items():
        if runtime.get(field) != value:
            raise RuntimeError(f"bbox25 model preflight runtime {field} drifted")
    arch = runtime.get("torch_arch_list")
    if not isinstance(arch, list) or "sm_120" not in arch:
        raise RuntimeError("bbox25 model preflight Torch architecture drifted")
    source_audit = receipt.get("source_audit")
    raw_files = inventory.get("files")
    if (
        not isinstance(source_audit, Mapping)
        or not isinstance(raw_files, list)
        or source_audit.get("checked_file_count") != len(raw_files)
        or source_audit.get("inventory_tree_sha256") != inventory.get("tree_sha256")
        or source_audit.get("mismatches") != []
    ):
        raise RuntimeError("bbox25 model preflight source audit drifted")
    model = receipt.get("model_audit")
    required_true = (
        "base_full_model_build_passed",
        "candidate_full_model_build_passed",
        "base_init_weights_passed",
        "candidate_init_weights_passed",
        "strict_load_passed",
        "complete_state_equal_after_strict_load",
    )
    if not isinstance(model, Mapping) or any(
        model.get(key) is not True for key in required_true
    ):
        raise RuntimeError("bbox25 full-model build/strict-load gate did not pass")
    if (
        model.get("strict_missing_keys") != []
        or model.get("strict_unexpected_keys") != []
    ):
        raise RuntimeError("bbox25 strict-load incompatibilities are non-empty")
    if (
        not isinstance(model.get("state_key_count"), int)
        or model["state_key_count"] <= 0
    ):
        raise RuntimeError("bbox25 full-model state is empty")
    for field in ("state_schema_sha256", "source_state_sha256", "target_state_sha256"):
        if (
            not isinstance(model.get(field), str)
            or shared._SHA256.fullmatch(model[field]) is None
        ):
            raise RuntimeError(f"bbox25 model preflight {field} is invalid")
    if model.get("source_state_sha256") != model.get("target_state_sha256"):
        raise RuntimeError("bbox25 strict-load full-state hashes differ")
    if (
        model.get("base_student_bbox_loss_weight"),
        model.get("candidate_student_bbox_loss_weight"),
        model.get("candidate_nested_teacher_bbox_loss_weight"),
    ) != (2.0, 2.5, 2.0):
        raise RuntimeError("bbox25 model preflight loss-weight contract drifted")


def _group_regex() -> str:
    prefixes = [re.escape(spec.task_prefix) for spec in CANDIDATE_SPECS.values()]
    return rf"^(?:{'|'.join(prefixes)}) \[[0-9a-f]{{12}}\]$"


def _group_tasks(task_class: object) -> list[object]:
    tasks = task_class.get_tasks(
        task_name=_group_regex(),
        task_filter={"parent": shared.fastlane.PREDECESSOR_TASK_ID},
    )
    exact = []
    for task in tasks or ():
        name = str(getattr(task, "name", "") or "")
        if re.fullmatch(_group_regex(), name) and shared.fastlane._task_parent(
            task
        ) == (shared.fastlane.PREDECESSOR_TASK_ID):
            exact.append(task)
    return exact


def _require_group_state(
    task_class: object,
    *,
    spec: CandidateSpec,
    trigger: Mapping[str, object],
    current_task: object | None,
) -> None:
    tasks = _group_tasks(task_class)
    selected = trigger["selected_candidate"]
    assert isinstance(selected, Mapping)
    expected_current_name = str(selected["expected_task_name"])
    current_id = (
        shared.fastlane._task_id(current_task, context="bbox25 current clone")
        if current_task is not None
        else None
    )
    current_matches = [
        task
        for task in tasks
        if str(getattr(task, "name", "") or "") == expected_current_name
    ]
    expected_current_count = 1 if current_task is not None else 0
    if len(current_matches) != expected_current_count:
        raise RuntimeError(
            "bbox25 mutual-exclusion guard observed an invalid selected-task count"
        )
    if current_task is not None:
        observed_id = shared.fastlane._task_id(
            current_matches[0], context="bbox25 group current clone"
        )
        if observed_id != current_id:
            raise RuntimeError("bbox25 mutual-exclusion guard matched another clone ID")

    allowed_ids = {current_id} if current_id is not None else set()
    if spec.priority != "preferred":
        fallback = trigger.get("fallback")
        assert isinstance(fallback, Mapping)
        primary_id = str(fallback["primary_task_id"])
        primary_name = str(fallback["primary_task_name"])
        primary = [
            task
            for task in tasks
            if shared.fastlane._task_id(task, context="bbox25 primary group task")
            == primary_id
            and str(getattr(task, "name", "") or "") == primary_name
        ]
        if len(primary) != 1:
            raise RuntimeError(
                "backup bbox25 launch requires exactly one referenced primary task"
            )
        live_status = shared.fastlane._status(primary[0])
        expected_status = str(fallback["primary_terminal_status"])
        if (
            live_status != expected_status
            or live_status not in TERMINAL_PRIMARY_STATUSES
        ):
            raise RuntimeError("backup bbox25 primary task terminal status drifted")
        allowed_ids.add(primary_id)

    observed_ids = {
        shared.fastlane._task_id(task, context="bbox25 mutual-exclusion task")
        for task in tasks
    }
    if observed_ids != allowed_ids:
        raise RuntimeError(
            "bbox25 mutual-exclusion group contains an unauthorized task; "
            "refusing clone/enqueue"
        )


def _launch_with_gates(
    args: argparse.Namespace,
    *,
    spec: CandidateSpec,
    trigger: dict[str, object],
    preflight: dict[str, object],
) -> int:
    from clearml import Task

    _require_group_state(Task, spec=spec, trigger=trigger, current_task=None)
    original_upload = shared._upload_verify_and_enqueue
    original_unique = shared._require_unique_current_clone
    calls = {"upload": 0, "unique": 0}

    def gated_upload(
        task_class: object,
        task: object,
        receipt: Mapping[str, object],
        *,
        queue: str,
    ) -> None:
        calls["upload"] += 1
        payload = dict(receipt)
        payload.pop("seal_sha256", None)
        payload.update(
            {
                "selection_trigger_receipt": trigger,
                "model_preflight_receipt": preflight,
                "mutual_exclusion_group": MUTUAL_EXCLUSION_GROUP,
                "candidate_priority": spec.priority,
            }
        )
        augmented = shared._sealed(payload)
        if not isinstance(receipt, dict):
            raise RuntimeError("shared launch receipt is not mutable")
        receipt.clear()
        receipt.update(augmented)
        original_upload(task_class, task, receipt, queue=queue)

    def gated_unique(task_class: object, task: object) -> None:
        calls["unique"] += 1
        original_unique(task_class, task)
        _require_group_state(
            task_class,
            spec=spec,
            trigger=trigger,
            current_task=task,
        )

    shared._upload_verify_and_enqueue = gated_upload
    shared._require_unique_current_clone = gated_unique
    try:
        result = shared._launch(args)
    finally:
        shared._upload_verify_and_enqueue = original_upload
        shared._require_unique_current_clone = original_unique
    if calls != {"upload": 1, "unique": 1}:
        raise RuntimeError(f"bbox25 gated launch call sequence drifted: {calls}")
    return result


def _load_launch_gates(
    args: argparse.Namespace, *, spec: CandidateSpec
) -> tuple[dict[str, object], dict[str, object]]:
    manifest = shared._verify_package(args.output_dir)
    deployment, static = shared._verify_deployment_plan(args.output_dir, manifest)
    bindings = _candidate_bindings(manifest, deployment, static)
    inventory = shared._load_json(args.output_dir / SOURCE_INVENTORY_NAME)
    trigger = _read_receipt(args.trigger_receipt, context="bbox25 trigger receipt")
    preflight = _read_receipt(
        args.model_preflight_receipt, context="bbox25 model preflight receipt"
    )
    _require_trigger_receipt(trigger, spec=spec, bindings=bindings)
    _require_model_preflight_receipt(
        preflight,
        spec=spec,
        manifest=manifest,
        bindings=bindings,
        inventory=inventory,
    )
    return trigger, preflight


def _candidate_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--candidate", choices=tuple(CANDIDATE_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def _parse_delegated(arguments: Sequence[str]) -> tuple[str, list[str]]:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--candidate", choices=tuple(CANDIDATE_SPECS), required=True)
    selected, remaining = selector.parse_known_args(arguments)
    return str(selected.candidate), remaining


def _resolved_output(spec: CandidateSpec, value: Path | None) -> Path:
    return (value or spec.output_dir).resolve(strict=False)


def _local_preferred_task_name() -> str:
    spec = CANDIDATE_SPECS[PRIMARY_KEY]
    with _candidate_engine(PRIMARY_KEY):
        manifest = shared._verify_package(spec.output_dir)
        _deployment, static = shared._verify_deployment_plan(spec.output_dir, manifest)
    return _expected_task_name(spec, static)


def _build_trigger_command(arguments: Sequence[str]) -> int:
    parser = _candidate_parser("Build a local sealed bbox25 selection trigger.")
    parser.add_argument("--p0-bev-ap70", type=float, required=True)
    parser.add_argument("--p0-3d-ap70", type=float, required=True)
    parser.add_argument("--p2-bev-ap70", type=float, required=True)
    parser.add_argument("--p2-3d-ap70", type=float, required=True)
    parser.add_argument("--primary-task-id", default="")
    parser.add_argument("--primary-status", default="")
    parser.add_argument("--fallback-reason", default="")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(arguments)
    spec = _spec_for(args.candidate)
    args.output_dir = _resolved_output(spec, args.output_dir)
    primary_task_name = (
        "" if spec.priority == "preferred" else _local_preferred_task_name()
    )
    with _candidate_engine(spec.key):
        manifest = shared._verify_package(args.output_dir)
        deployment, static = shared._verify_deployment_plan(args.output_dir, manifest)
        bindings = _candidate_bindings(manifest, deployment, static)
        receipt = build_trigger_receipt(
            key=spec.key,
            bindings=bindings,
            static_plan=static,
            p0_bev_ap70=args.p0_bev_ap70,
            p0_3d_ap70=args.p0_3d_ap70,
            p2_bev_ap70=args.p2_bev_ap70,
            p2_3d_ap70=args.p2_3d_ap70,
            primary_task_id=args.primary_task_id,
            primary_task_name=primary_task_name,
            primary_status=args.primary_status,
            fallback_reason=args.fallback_reason,
        )
    path = (args.receipt or (args.output_dir / TRIGGER_RECEIPT_NAME)).resolve(
        strict=False
    )
    _write_new_receipt(path, receipt)
    print(json.dumps({"receipt": str(path), **receipt}, sort_keys=True))
    return 0


def _model_preflight_command(arguments: Sequence[str]) -> int:
    parser = _candidate_parser("Run the exact-runtime full-model bbox25 preflight.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--resnet-checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(arguments)
    spec = _spec_for(args.candidate)
    args.output_dir = _resolved_output(spec, args.output_dir)
    source_root = args.source_root.resolve(strict=True)
    teacher = _require_pinned_file(
        args.teacher_checkpoint,
        expected_sha256=shared.fastlane.TEACHER_CHECKPOINT_SHA256,
        context="clean-teacher checkpoint",
    )
    resnet = _require_pinned_file(
        args.resnet_checkpoint,
        expected_sha256=source_runner.EXPECTED_RESNET_SHA256,
        context="ResNet-50 checkpoint",
    )
    with _candidate_engine(spec.key):
        manifest = shared._verify_package(args.output_dir)
        deployment, static = shared._verify_deployment_plan(args.output_dir, manifest)
        bindings = _candidate_bindings(manifest, deployment, static)
        inventory = shared._load_json(args.output_dir / SOURCE_INVENTORY_NAME)
        receipt = _build_model_preflight_receipt(
            spec=spec,
            bindings=bindings,
            manifest=manifest,
            inventory=inventory,
            source_root=source_root,
            teacher_checkpoint=teacher,
            resnet_checkpoint=resnet,
        )
    path = (args.receipt or (args.output_dir / MODEL_PREFLIGHT_RECEIPT_NAME)).resolve(
        strict=False
    )
    _write_new_receipt(path, receipt)
    print(json.dumps({"receipt": str(path), **receipt}, sort_keys=True))
    return 0


def _launch_command(arguments: Sequence[str]) -> int:
    parser = _candidate_parser("Launch one fully gated bbox25 candidate.")
    parser.add_argument("--source-dataset-id", required=True)
    parser.add_argument("--template-task-id", required=True)
    parser.add_argument("--queue", default=p2.WORKER_QUEUE)
    parser.add_argument("--trigger-receipt", type=Path, required=True)
    parser.add_argument("--model-preflight-receipt", type=Path, required=True)
    parser.add_argument("--execute-token", default="")
    args = parser.parse_args(arguments)
    spec = _spec_for(args.candidate)
    args.output_dir = _resolved_output(spec, args.output_dir)
    args.trigger_receipt = args.trigger_receipt.resolve(strict=False)
    args.model_preflight_receipt = args.model_preflight_receipt.resolve(strict=False)
    with _candidate_engine(spec.key):
        shared._require_token(args.execute_token, spec.launch_token)
        trigger, preflight = _load_launch_gates(args, spec=spec)
        return _launch_with_gates(
            args,
            spec=spec,
            trigger=trigger,
            preflight=preflight,
        )


def _show_contract_command(arguments: Sequence[str]) -> int:
    parser = _candidate_parser("Show and verify the candidate execution contract.")
    parser.add_argument("--static-plan-seal-sha256", default="")
    args = parser.parse_args(arguments)
    spec = _spec_for(args.candidate)
    args.output_dir = _resolved_output(spec, args.output_dir)
    with _candidate_engine(spec.key):
        manifest = shared._verify_package(args.output_dir)
        deployment, static = shared._verify_deployment_plan(args.output_dir, manifest)
    if (
        args.static_plan_seal_sha256
        and args.static_plan_seal_sha256 != static["seal_sha256"]
    ):
        raise RuntimeError("requested static plan seal does not match")
    print(
        json.dumps(
            {
                "candidate": spec.key,
                "deployment_plan_seal_sha256": deployment["seal_sha256"],
                "static_plan_seal_sha256": static["seal_sha256"],
                "execution_contract": static["execution_contract"],
            },
            sort_keys=True,
        )
    )
    return 0


def _prepare_all() -> int:
    for key, spec in CANDIDATE_SPECS.items():
        with _candidate_engine(key):
            shared.main(["prepare", "--output-dir", str(spec.output_dir)])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        raise SystemExit("a command is required")
    command, rest = arguments[0], arguments[1:]
    if command == "prepare-all":
        if rest:
            raise SystemExit("prepare-all takes no arguments")
        return _prepare_all()
    if command == "build-trigger":
        return _build_trigger_command(rest)
    if command == "model-preflight":
        return _model_preflight_command(rest)
    if command == "launch":
        return _launch_command(rest)
    if command == "show-contract":
        return _show_contract_command(rest)
    if command in {"prepare", "upload-source", "create-template"}:
        key, delegated = _parse_delegated(rest)
        spec = _spec_for(key)
        if "--output-dir" not in delegated:
            delegated.extend(["--output-dir", str(spec.output_dir)])
        with _candidate_engine(key):
            return shared.main([command, *delegated])
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
