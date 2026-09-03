#!/usr/bin/env python3
"""Verify a confirmatory registry and derive canonical publication metrics."""

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
from transvision.models.event_track_v2x.publication_gate import (  # noqa: E402
    PublicationMetricsV1,
)
from transvision.models.event_track_v2x.results_registry import (  # noqa: E402
    RunResultRegistryV1,
    decode_run_result_registry,
    derive_publication_metrics_v1,
    validate_run_result_external_inputs_v1,
)
from transvision.models.event_track_v2x.wire import (  # noqa: E402
    canonical_json_bytes,
)


def _summary_document(
    plan: ExperimentPlanV1,
    registry: RunResultRegistryV1,
    metrics: PublicationMetricsV1,
    measured_trace_receipt: MeasuredTraceReceiptV1,
    condition_input_manifests: dict[str, ConditionInputManifestV1],
) -> dict[str, object]:
    return {
        "cell_count": len(registry.cells),
        "experiment_plan_sha256": plan.content_sha256,
        "condition_input_manifest_count": len(condition_input_manifests),
        "kind": "run_result_registry_verification_v1",
        "publication_metrics_sha256": metrics.digest(),
        "measured_trace_receipt_sha256": measured_trace_receipt.content_sha256,
        "run_result_registry_sha256": registry.digest(),
        "schema_version": 1,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--measured-trace-receipt", type=Path, required=True)
    parser.add_argument(
        "--condition-input-manifest-inventory",
        type=Path,
        required=True,
    )
    parser.add_argument("--metrics-output", type=Path)
    return parser


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = decode_plan(args.plan.read_bytes())
    registry = decode_run_result_registry(args.registry.read_bytes())
    measured_trace_receipt = decode_measured_trace_receipt(
        args.measured_trace_receipt.read_bytes()
    )
    condition_input_manifests = decode_condition_input_manifest_inventory_v1(
        args.condition_input_manifest_inventory.read_bytes()
    )
    validate_run_result_external_inputs_v1(
        registry,
        plan,
        measured_trace_receipt=measured_trace_receipt,
        condition_input_manifests=condition_input_manifests,
    )
    metrics = derive_publication_metrics_v1(registry, plan)
    if args.metrics_output is not None:
        _write_new(args.metrics_output, metrics.canonical_bytes())
    output = canonical_json_bytes(
        _summary_document(
            plan,
            registry,
            metrics,
            measured_trace_receipt,
            condition_input_manifests,
        )
    )
    sys.stdout.buffer.write(output + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
