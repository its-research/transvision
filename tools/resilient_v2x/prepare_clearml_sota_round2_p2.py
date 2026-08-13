#!/usr/bin/env python3
"""Prepare the sealed round2 P2 bbox2.5 candidate derived from P0.

The local ``prepare`` command extends the immutable sealed P0 source package by
exactly the P2 configuration overlay. Remote-writing commands require distinct
exact tokens and never modify the P0 Dataset, package, template, task, launcher,
or artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

try:
    from tools.resilient_v2x import prepare_clearml_sota_round2_p0 as shared
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from tools.resilient_v2x import prepare_clearml_sota_round2_p0 as shared


ROOT = Path(__file__).resolve().parents[2]
P0_OUTPUT_DIR = ROOT / "artifacts/resilient_v2x/sota_round2_p0"
P0_PACKAGE_MANIFEST = P0_OUTPUT_DIR / shared.PACKAGE_MANIFEST_NAME
P0_SOURCE_DATASET_ID = "ca2ef9dd8a984df6b05693fb02e89f34"
P0_TEMPLATE_TASK_ID = "13f87c0fd45d4621b3c93c0ff88702d8"
P0_TRAINING_TASK_ID = "8883c51ced4f4951a45edbaefe6342d4"

OUTPUT_DIR = ROOT / "artifacts/resilient_v2x/sota_round2_p2"
PACKAGE_MANIFEST_NAME = "round2-p2-source-package.json"
DEPLOYMENT_PLAN_NAME = "round2-p2-deployment-plan.json"

P2_EXPERIMENT = "support_residual_no_reliability_linear_bbox25"
P2_CONFIG = (
    "configs/resilient_v2x/improvements/"
    "support_residual_no_reliability_linear_bbox25.py"
)
P2_IDENTITY = "dair_improvement_support_residual_no_reliability_linear_bbox25"

SOURCE_TRANSITION_ARTIFACT = "sota_round2_p2_source_transition"
LAUNCH_RECEIPT_ARTIFACT = "sota_round2_p2_launch_receipt"
SOURCE_DATASET_PREFIX = "ResilientV2X sealed SOTA round2 P2 bbox2.5 source"
TEMPLATE_PREFIX = "ResilientV2X round2 P2 bbox2.5 bootstrap template"
TASK_PREFIX = "ResilientV2X round2 P2 bbox2.5"

WORKER_QUEUE = shared.WORKER_QUEUE
WORKER_QUEUE_ID = shared.WORKER_QUEUE_ID
GPU_COUNT = shared.GPU_COUNT
BATCH_SIZE_PER_GPU = shared.BATCH_SIZE_PER_GPU
GLOBAL_BATCH_SIZE = shared.GLOBAL_BATCH_SIZE
MAX_EPOCHS = shared.MAX_EPOCHS
VAL_INTERVAL = shared.VAL_INTERVAL
TRAINING_SEED = shared.TRAINING_SEED
PRECISION = shared.PRECISION

UPLOAD_TOKEN = "UPLOAD_EXACT_ROUND2_P2_SOURCE"
TEMPLATE_TOKEN = "CREATE_EXACT_ROUND2_P2_TEMPLATE"
LAUNCH_TOKEN = "LAUNCH_EXACT_ROUND2_P2_A100"

RESILIENT_REFERENCE_TASK_ID = "54d28bc513794051810fd383140ae96e"
RESILIENT_E10_BEV_AP70 = 60.6345
RESILIENT_E10_3D_AP70 = 31.5586
EARLY_GATE_SIGNIFICANT_MARGIN_AP = 0.5
EARLY_GATE_POLICY = shared._sealed(
    {
        "schema_version": 1,
        "policy_type": "resilient_v2x_round2_p2_e10_early_gate_v1",
        "decision_epoch": 10,
        "mode": "operator_gated_with_sealed_outcome_receipt",
        "default_action": "continue_to_epoch_50",
        "authorized_stop_action": "stop_after_epoch_10_validation",
        "declared_training_protocol": {
            "max_epochs": MAX_EPOCHS,
            "val_interval": VAL_INTERVAL,
            "protocol_unchanged_by_gate": True,
        },
        "metric_contract": {
            "dataset_slice": "clean-1789",
            "iou_threshold": 0.70,
            "metrics": ["BEV AP70", "3D AP70"],
            "epoch": 10,
        },
        "resilient_e10_reference": {
            "task_id": RESILIENT_REFERENCE_TASK_ID,
            "epoch": 10,
            "bev_ap70": RESILIENT_E10_BEV_AP70,
            "3d_ap70": RESILIENT_E10_3D_AP70,
        },
        "significant_margin_ap": EARLY_GATE_SIGNIFICANT_MARGIN_AP,
        "stop_authorization": {
            "both_p0_and_p2_significantly_below_resilient": (
                "for each of P0 and P2: BEV AP70 <= reference - 0.5 and "
                "3D AP70 <= reference - 0.5"
            ),
            "p2_significantly_regresses_from_p0": (
                "P2 BEV AP70 <= P0 - 0.5 and P2 3D AP70 <= P0 - 0.5"
            ),
            "runtime_regression": (
                "non-finite loss/metric or another explicitly evidenced fatal "
                "training regression"
            ),
            "boolean_rule": (
                "both_candidates_below_reference OR p2_regresses_from_p0 OR "
                "runtime_regression"
            ),
        },
        "outcome_receipt_requirements": {
            "p0_task_id": True,
            "p2_task_id": True,
            "p0_and_p2_clean_1789_e10_metrics": True,
            "policy_seal_binding": True,
            "plan_seal_binding": True,
            "decision_and_boolean_subconditions": True,
            "created_at_utc": True,
        },
    }
)

_P2_SPEC_ANCHOR = shared._P0_SPEC_ANCHOR
_P2_SPEC_PATCH = """    ExperimentSpec(
        "support_residual_no_reliability_linear_bbox25",
        "sota_candidate",
        "configs/resilient_v2x/improvements/support_residual_no_reliability_linear_bbox25.py",
        True,
    ),
    ExperimentSpec(
        "linear_no_distillation",
"""
_P2_NESTED_ANCHOR = """        "support_residual_no_reliability_linear",
        "resilient_v2x",
"""
_P2_NESTED_PATCH = """        "support_residual_no_reliability_linear",
        "support_residual_no_reliability_linear_bbox25",
        "resilient_v2x",
"""


def _verified_p0_manifest() -> dict[str, object]:
    """Verify the local immutable P0 package before accepting it as P2's parent."""

    with shared.use_deployment_profile(shared.P0_PROFILE):
        return shared._verify_package(P0_OUTPUT_DIR)


def _build_profile() -> shared.Round2DeploymentProfile:
    p0_manifest = _verified_p0_manifest()
    p0_package = p0_manifest["target_source_package"]
    p0_candidate = p0_manifest["candidate"]
    assert isinstance(p0_package, dict)
    assert isinstance(p0_candidate, dict)
    provenance = {
        "derivation": "P2 derived from P0",
        "parent_candidate": dict(p0_candidate),
        "parent_source_dataset_id": P0_SOURCE_DATASET_ID,
        "parent_source_package_manifest_seal_sha256": p0_manifest["seal_sha256"],
        "parent_source_tree_sha256": p0_package["tree_sha256"],
        "parent_template_task_id": P0_TEMPLATE_TASK_ID,
        "parent_training_task_id": P0_TRAINING_TASK_ID,
    }
    return shared.Round2DeploymentProfile(
        label="round2 P2",
        output_dir=OUTPUT_DIR,
        package_manifest_name=PACKAGE_MANIFEST_NAME,
        deployment_plan_name=DEPLOYMENT_PLAN_NAME,
        package_type="resilient_v2x_sota_round2_p2_source",
        experiment=P2_EXPERIMENT,
        config=P2_CONFIG,
        identity=P2_IDENTITY,
        base_source_dir=P0_OUTPUT_DIR,
        base_source_inventory=P0_OUTPUT_DIR / "source-inventory.json",
        base_source_archive=P0_OUTPUT_DIR / str(p0_package["archive_name"]),
        base_source_dataset_id=P0_SOURCE_DATASET_ID,
        base_source_package=dict(p0_package),
        provenance=provenance,
        early_gate_policy=EARLY_GATE_POLICY,
        source_transition_artifact=SOURCE_TRANSITION_ARTIFACT,
        launch_receipt_artifact=LAUNCH_RECEIPT_ARTIFACT,
        source_dataset_prefix=SOURCE_DATASET_PREFIX,
        source_dataset_version_suffix="round2-p2-v1",
        source_dataset_tags=(
            "ResilientV2X",
            "source",
            "sota-candidates",
            "round2-p2",
            "bbox25",
            "derived-from-p0",
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
        spec_anchor=_P2_SPEC_ANCHOR,
        spec_patch=_P2_SPEC_PATCH,
        nested_anchor=_P2_NESTED_ANCHOR,
        nested_patch=_P2_NESTED_PATCH,
        bootstrap_parent=shared.P0_PROFILE,
        transition_type="exact_additive_sota_round2_p2_from_p0_source_revision",
        plan_type="resilient_v2x_sota_round2_p2_bbox25_a100_v1",
        receipt_type="resilient_v2x_sota_round2_p2_bbox25_launch_v1",
        template_status_message="sealed round2 P2 bbox2.5 bootstrap template",
        transition_description=(
            "P2 derived from P0; exact one-config bbox2.5 additive source"
        ),
        invariants=(
            (
                "P2 derived from P0: source transition adds only the bbox2.5 "
                "overlay to sealed P0 source"
            ),
            (
                "existing P0 Dataset, source package, template, task, launcher, "
                "and artifacts are untouched"
            ),
            (
                "P2 preserves the P0 training Dataset, frozen teacher, seed, "
                "FP32, global batch 8, 50 epochs, and val/10"
            ),
            (
                "one task is enqueued to GPU4-A100 only after transition, "
                "template, launch receipt, duplicate, and queue-ID checks pass"
            ),
        ),
        duplicate_guard_description=(
            "exact task name plus predecessor parent before clone, followed by "
            "forced launch-receipt readback and a unique exact-name+parent "
            "current-clone ID guard immediately before enqueue"
        ),
        duplicate_guard_stages=(
            "pre_clone_requires_zero_exact_name_plus_parent_matches",
            (
                "post_receipt_readback_pre_enqueue_requires_one_exact_name_plus_"
                "parent_match_with_current_clone_id"
            ),
        ),
        relative_script=Path(__file__).relative_to(ROOT),
        cli_description=__doc__ or "",
    )


P2_PROFILE = _build_profile()


def _validated_metric(value: float, *, name: str) -> float:
    observed = float(value)
    if not math.isfinite(observed) or observed < 0.0 or observed > 100.0:
        raise ValueError(f"{name} must be finite and in [0, 100]")
    return observed


def _validated_sha256(value: str, *, name: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def build_e10_early_gate_receipt(
    *,
    plan_seal_sha256: str,
    p0_task_id: str,
    p0_bev_ap70: float,
    p0_3d_ap70: float,
    p2_task_id: str,
    p2_bev_ap70: float,
    p2_3d_ap70: float,
    runtime_regression: str = "",
    created_at: str | None = None,
) -> dict[str, object]:
    """Build a sealed local E10 gate outcome; this function never stops a task."""

    plan_seal_sha256 = _validated_sha256(plan_seal_sha256, name="plan_seal_sha256")
    p0_task_id = shared.training._clearml_id(p0_task_id, "P0 task")
    p2_task_id = shared.training._clearml_id(p2_task_id, "P2 task")
    p0_bev_ap70 = _validated_metric(p0_bev_ap70, name="P0 BEV AP70")
    p0_3d_ap70 = _validated_metric(p0_3d_ap70, name="P0 3D AP70")
    p2_bev_ap70 = _validated_metric(p2_bev_ap70, name="P2 BEV AP70")
    p2_3d_ap70 = _validated_metric(p2_3d_ap70, name="P2 3D AP70")
    runtime_regression = runtime_regression.strip()
    margin = EARLY_GATE_SIGNIFICANT_MARGIN_AP
    p0_below = (
        p0_bev_ap70 <= RESILIENT_E10_BEV_AP70 - margin
        and p0_3d_ap70 <= RESILIENT_E10_3D_AP70 - margin
    )
    p2_below = (
        p2_bev_ap70 <= RESILIENT_E10_BEV_AP70 - margin
        and p2_3d_ap70 <= RESILIENT_E10_3D_AP70 - margin
    )
    both_below = p0_below and p2_below
    p2_regresses = (
        p2_bev_ap70 <= p0_bev_ap70 - margin and p2_3d_ap70 <= p0_3d_ap70 - margin
    )
    runtime_regression_observed = bool(runtime_regression)
    stop_authorized = both_below or p2_regresses or runtime_regression_observed
    return shared._sealed(
        {
            "schema_version": 1,
            "receipt_type": "resilient_v2x_round2_p2_e10_early_gate_outcome_v1",
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "policy_seal_sha256": EARLY_GATE_POLICY["seal_sha256"],
            "plan_seal_sha256": plan_seal_sha256,
            "metric_contract": dict(EARLY_GATE_POLICY["metric_contract"]),
            "resilient_e10_reference": dict(
                EARLY_GATE_POLICY["resilient_e10_reference"]
            ),
            "observations": {
                "p0": {
                    "task_id": p0_task_id,
                    "epoch": 10,
                    "bev_ap70": p0_bev_ap70,
                    "3d_ap70": p0_3d_ap70,
                },
                "p2": {
                    "task_id": p2_task_id,
                    "epoch": 10,
                    "bev_ap70": p2_bev_ap70,
                    "3d_ap70": p2_3d_ap70,
                },
                "runtime_regression": runtime_regression or None,
            },
            "decision": {
                "p0_significantly_below_resilient": p0_below,
                "p2_significantly_below_resilient": p2_below,
                "both_candidates_below_reference": both_below,
                "p2_regresses_from_p0": p2_regresses,
                "runtime_regression": runtime_regression_observed,
                "stop_authorized": stop_authorized,
                "action": (
                    "stop_after_epoch_10_validation"
                    if stop_authorized
                    else "continue_to_epoch_50"
                ),
                "declared_training_protocol_changed": False,
            },
        }
    )


def _evaluate_e10_gate(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Build a local sealed P2 E10 early-gate outcome receipt."
    )
    parser.add_argument("--plan-seal-sha256", required=True)
    parser.add_argument("--p0-task-id", required=True)
    parser.add_argument("--p0-bev-ap70", type=float, required=True)
    parser.add_argument("--p0-3d-ap70", type=float, required=True)
    parser.add_argument("--p2-task-id", required=True)
    parser.add_argument("--p2-bev-ap70", type=float, required=True)
    parser.add_argument("--p2-3d-ap70", type=float, required=True)
    parser.add_argument("--runtime-regression", default="")
    args = parser.parse_args(argv)
    receipt = build_e10_early_gate_receipt(
        plan_seal_sha256=args.plan_seal_sha256,
        p0_task_id=args.p0_task_id,
        p0_bev_ap70=args.p0_bev_ap70,
        p0_3d_ap70=args.p0_3d_ap70,
        p2_task_id=args.p2_task_id,
        p2_bev_ap70=args.p2_bev_ap70,
        p2_3d_ap70=args.p2_3d_ap70,
        runtime_regression=args.runtime_regression,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "evaluate-e10-gate":
        return _evaluate_e10_gate(arguments[1:])
    with shared.use_deployment_profile(P2_PROFILE):
        return shared.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
