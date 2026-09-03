#!/usr/bin/env python3
"""Build the sealed EventTrack-V2X confirmatory experiment plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.experiment import (  # noqa: E402
    CONFIRMATORY_SCHEDULER_IDS_V1,
    ExperimentPlanV1,
    RunConfigBindingV1,
)


TRAINING_SEEDS = (1337, 2027, 3407)
NETWORK_SEEDS = tuple(range(1001, 1011))
BYTE_BUDGETS = (16000, 32000, 64000, 128000, 256000)
CONDITIONS = tuple(f"C{index}" for index in range(10))
SCHEDULERS = CONFIRMATORY_SCHEDULER_IDS_V1
METHODS = (
    "vehicle-only-ab3dmot",
    "vehicle-only-simpletrack",
    "vehicle-only-immortaltracker",
    "naive-async-late-fusion",
    "constant-velocity-compensation",
    "fixed-lag-oosm-controlled",
    "eventtrack-v2x",
    "synchronous-oracle",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--method-config-sha256", required=True)
    parser.add_argument("--run-config-bindings-file", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--primary-cohort-sha256", required=True)
    parser.add_argument("--primary-frame-contract-sha256", required=True)
    parser.add_argument("--primary-dataset-release-receipt-sha256", required=True)
    parser.add_argument("--development-dataset-release-receipt-sha256", required=True)
    parser.add_argument("--development-fold-manifest-sha256", required=True)
    parser.add_argument("--primary-sequence-ids-file", type=Path, required=True)
    parser.add_argument("--detector-cache-sha256", required=True)
    parser.add_argument("--network-trace-manifest-sha256", required=True)
    parser.add_argument("--wire-accounting-config-sha256", required=True)
    parser.add_argument("--heldout-c9-trace-receipt-sha256", required=True)
    parser.add_argument("--c9-trace-ids-file", type=Path, required=True)
    parser.add_argument("--evaluator-contract-sha256", required=True)
    parser.add_argument("--baseline-qualification-registry-sha256", required=True)
    parser.add_argument("--candidate-selection-registry-sha256", required=True)
    parser.add_argument("--preregistration-provider-id", required=True)
    parser.add_argument("--preregistration-public-key-sha256", required=True)
    parser.add_argument("--verification-provider-id", required=True)
    parser.add_argument("--verification-public-key-sha256", required=True)
    parser.add_argument(
        "--qualified-baseline-id", action="append", required=True
    )
    parser.add_argument("--strongest-qualified-baseline-id", required=True)
    parser.add_argument(
        "--strongest-scheduler-baseline-id",
        choices=tuple(item for item in SCHEDULERS if item != "marginal_voi"),
        required=True,
    )
    parser.add_argument(
        "--scheduler-token-bucket-burst-seconds", type=float, default=1.0
    )
    parser.add_argument("--griffin-dataset-id", default="griffin-25m")
    parser.add_argument("--griffin-dataset-manifest-sha256", required=True)
    parser.add_argument("--griffin-split", default="val")
    parser.add_argument("--griffin-cohort-sha256", required=True)
    parser.add_argument("--griffin-frame-contract-sha256", required=True)
    parser.add_argument("--griffin-dataset-release-receipt-sha256", required=True)
    parser.add_argument("--griffin-sequence-ids-file", type=Path, required=True)
    parser.add_argument("--griffin-detector-cache-sha256", required=True)
    parser.add_argument("--griffin-evaluator-contract-sha256", required=True)
    parser.add_argument("--griffin-strongest-baseline-id", required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    return parser


def _sequence_ids(path: Path, name: str) -> tuple[str, ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {name} JSON") from exc
    if not isinstance(value, list) or not all(type(item) is str for item in value):
        raise ValueError(f"{name} must contain one JSON array of sequence IDs")
    return tuple(value)


def _run_config_bindings(path: Path) -> tuple[RunConfigBindingV1, ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read run config bindings JSON") from exc
    if not isinstance(value, list):
        raise ValueError("run config bindings must contain one JSON array")
    return tuple(RunConfigBindingV1.from_mapping(item) for item in value)


def build(args: argparse.Namespace) -> ExperimentPlanV1:
    return ExperimentPlanV1(
        plan_id="eventtrack-v2x-confirmatory-v1",
        source_tree_sha256=args.source_tree_sha256,
        method_config_sha256=args.method_config_sha256,
        run_config_bindings=_run_config_bindings(args.run_config_bindings_file),
        dataset_id=args.dataset_id,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        split_name=args.split,
        split_sha256=args.split_sha256,
        primary_cohort_sha256=args.primary_cohort_sha256,
        primary_frame_contract_sha256=args.primary_frame_contract_sha256,
        primary_dataset_release_receipt_sha256=(
            args.primary_dataset_release_receipt_sha256
        ),
        development_dataset_release_receipt_sha256=(
            args.development_dataset_release_receipt_sha256
        ),
        development_fold_manifest_sha256=args.development_fold_manifest_sha256,
        primary_sequence_ids=_sequence_ids(
            args.primary_sequence_ids_file, "primary sequence IDs"
        ),
        detector_cache_sha256=args.detector_cache_sha256,
        network_trace_manifest_sha256=args.network_trace_manifest_sha256,
        wire_accounting_config_sha256=args.wire_accounting_config_sha256,
        heldout_c9_trace_receipt_sha256=(
            args.heldout_c9_trace_receipt_sha256
        ),
        c9_trace_ids=_sequence_ids(args.c9_trace_ids_file, "C9 trace IDs"),
        evaluator_contract_sha256=args.evaluator_contract_sha256,
        baseline_qualification_registry_sha256=(
            args.baseline_qualification_registry_sha256
        ),
        candidate_selection_registry_sha256=(
            args.candidate_selection_registry_sha256
        ),
        preregistration_provider_id=args.preregistration_provider_id,
        preregistration_public_key_sha256=(
            args.preregistration_public_key_sha256
        ),
        verification_provider_id=args.verification_provider_id,
        verification_public_key_sha256=args.verification_public_key_sha256,
        qualified_baseline_ids=tuple(sorted(args.qualified_baseline_id)),
        strongest_qualified_baseline_id=args.strongest_qualified_baseline_id,
        scheduler_ids=SCHEDULERS,
        candidate_scheduler_id="marginal_voi",
        qualified_scheduler_baseline_ids=tuple(
            item for item in SCHEDULERS if item not in {"full_send", "marginal_voi"}
        ),
        strongest_scheduler_baseline_id=args.strongest_scheduler_baseline_id,
        scheduler_token_bucket_burst_seconds=(
            args.scheduler_token_bucket_burst_seconds
        ),
        griffin_dataset_id=args.griffin_dataset_id,
        griffin_dataset_manifest_sha256=args.griffin_dataset_manifest_sha256,
        griffin_split_name=args.griffin_split,
        griffin_cohort_sha256=args.griffin_cohort_sha256,
        griffin_frame_contract_sha256=args.griffin_frame_contract_sha256,
        griffin_dataset_release_receipt_sha256=(
            args.griffin_dataset_release_receipt_sha256
        ),
        griffin_sequence_ids=_sequence_ids(
            args.griffin_sequence_ids_file, "Griffin sequence IDs"
        ),
        griffin_detector_cache_sha256=args.griffin_detector_cache_sha256,
        griffin_evaluator_contract_sha256=(
            args.griffin_evaluator_contract_sha256
        ),
        griffin_strongest_baseline_id=args.griffin_strongest_baseline_id,
        frequency_hz=10.0,
        class_names=("car",),
        method_ids=METHODS,
        network_condition_ids=CONDITIONS,
        training_seeds=TRAINING_SEEDS,
        network_seeds=NETWORK_SEEDS,
        byte_budgets_per_second=BYTE_BUDGETS,
        primary_budget_bytes_per_second=64000,
        primary_metric="robust-assa-at-64k",
        decision_deadline_ms=100.0,
        statistical_alpha=0.05,
        statistical_resamples=10000,
        statistical_random_seed=1337,
        claim_scope="same-detector robust cooperative association",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(plan.canonical_bytes())
    print(plan.content_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
