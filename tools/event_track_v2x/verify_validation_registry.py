#!/usr/bin/env python3
"""Verify and summarize a canonical EventTrack-V2X validation registry."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.experiment import (  # noqa: E402
    ExperimentPlanV1,
    decode_plan,
)
from transvision.models.event_track_v2x.measured_trace import (  # noqa: E402
    MeasuredTraceReceiptV1,
    decode_measured_trace_receipt,
)
from transvision.models.event_track_v2x.network_disturbance import (  # noqa: E402
    ConditionInputManifestV1,
    decode_condition_input_manifest_inventory_v1,
)
from transvision.models.event_track_v2x.validation_registry import (  # noqa: E402
    ValidationResultRegistryV1,
    ValidationSummariesV1,
    decode_validation_result_registry,
    derive_validation_summaries_v1,
    validate_validation_result_external_inputs_v1,
)
from transvision.models.event_track_v2x.wire import (  # noqa: E402
    canonical_json_bytes,
)


def _summary_document(
    plan: ExperimentPlanV1,
    registry: ValidationResultRegistryV1,
    summaries: ValidationSummariesV1,
    measured_trace_receipt: MeasuredTraceReceiptV1,
    condition_input_manifests: dict[str, ConditionInputManifestV1],
) -> dict[str, object]:
    return {
        "baseline_summaries": [
            {
                "clean_amota": item.clean_amota,
                "clean_hota": item.clean_hota,
                "failure_count": item.failure_count,
                "method_id": item.method_id,
                "robust_assa_at_64k": item.robust_assa_at_64k,
            }
            for item in summaries.baselines
        ],
        "cell_count": len(registry.cells),
        "condition_input_manifest_count": len(condition_input_manifests),
        "experiment_plan_sha256": plan.content_sha256,
        "kind": "validation_registry_verification_v1",
        "measured_trace_receipt_sha256": (measured_trace_receipt.content_sha256),
        "registry_sha256": registry.digest(),
        "scheduler_summaries": [
            {
                "actual_byte_auc": item.actual_byte_auc,
                "failure_count": item.failure_count,
                "robust_assa_at_64k": item.robust_assa_at_64k,
                "scheduler_id": item.scheduler_id,
            }
            for item in summaries.schedulers
        ],
        "schema_version": 1,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--measured-trace-receipt", type=Path, required=True)
    parser.add_argument(
        "--condition-input-manifest-inventory", type=Path, required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = decode_plan(args.plan.read_bytes())
    registry = decode_validation_result_registry(args.registry.read_bytes())
    measured_trace_receipt = decode_measured_trace_receipt(
        args.measured_trace_receipt.read_bytes()
    )
    condition_input_manifests = decode_condition_input_manifest_inventory_v1(
        args.condition_input_manifest_inventory.read_bytes()
    )
    validate_validation_result_external_inputs_v1(
        registry,
        plan,
        measured_trace_receipt=measured_trace_receipt,
        condition_input_manifests=condition_input_manifests,
    )
    summaries = derive_validation_summaries_v1(registry)
    output = canonical_json_bytes(
        _summary_document(
            plan,
            registry,
            summaries,
            measured_trace_receipt,
            condition_input_manifests,
        )
    )
    sys.stdout.buffer.write(output + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
