"""Canonical raw-result registry for confirmatory EventTrack-V2X claims.

The publication gate deliberately consumes compact per-sequence summaries.  This
module provides the missing, fail-closed bridge from individual experiment
cells to those summaries.  Four scenarios are preregistered rather than taking
an unnecessary Cartesian product of every method, scheduler, budget, and
metric:

``primary_robust``
    EventTrack and every qualified detector-locked baseline, at 64 kB/s, over
    C1--C9.
``primary_clean``
    EventTrack and the sealed strongest baseline, at 64 kB/s, over C0, for the
    AMOTA/HOTA non-inferiority result.
``primary_pareto``
    EventTrack with the candidate and sealed strongest scheduler, at all five
    byte budgets, over C1--C9.
``griffin_robust``
    EventTrack and the sealed Griffin baseline, at 64 kB/s, over C1--C9.

C1--C8 use every frozen synthetic network seed.  C9 uses exactly twenty sorted
held-out trace identifiers.  Algorithm failures are required cells and are
uniquely converted to zero during aggregation; missing cells are never treated
as failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re
from typing import Iterable, Iterator, Mapping

import numpy as np

from .experiment import GRIFFIN_25M_VAL_SEQUENCE_IDS_V1, ExperimentPlanV1
from .measured_trace import MeasuredTraceReceiptV1
from .network_disturbance import ConditionInputManifestV1
from .publication_gate import (
    BudgetedSequenceScoresV1,
    MethodScoresV1,
    NamedMetricScoresV1,
    PublicationMetricsV1,
    SequenceScoresV1,
)
from .wire import canonical_json_bytes


PRIMARY_ROBUST_SCENARIO_V1 = "primary_robust"
PRIMARY_CLEAN_SCENARIO_V1 = "primary_clean"
PRIMARY_PARETO_SCENARIO_V1 = "primary_pareto"
GRIFFIN_ROBUST_SCENARIO_V1 = "griffin_robust"
RUN_RESULT_SCENARIOS_V1 = (
    GRIFFIN_ROBUST_SCENARIO_V1,
    PRIMARY_CLEAN_SCENARIO_V1,
    PRIMARY_PARETO_SCENARIO_V1,
    PRIMARY_ROBUST_SCENARIO_V1,
)
RUN_RESULT_STATUS_SUCCESS_V1 = "success"
RUN_RESULT_STATUS_FAILURE_V1 = "algorithm_failure"
RUN_RESULT_STATUSES_V1 = (
    RUN_RESULT_STATUS_FAILURE_V1,
    RUN_RESULT_STATUS_SUCCESS_V1,
)
RESULT_METRICS_ARTIFACT_ROLE_V1 = "per_sequence_metrics"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


class RunResultRegistryError(ValueError):
    """Raised when raw result evidence is incomplete or not preregistered."""


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise RunResultRegistryError(f"{name} must be a canonical identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise RunResultRegistryError(f"{name} must be a lowercase SHA-256")
    return value


def _utc_timestamp(value: object, name: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise RunResultRegistryError(f"{name} must be canonical RFC3339 UTC")
    try:
        observed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RunResultRegistryError(
            f"{name} must be canonical RFC3339 UTC"
        ) from exc
    if observed.utcoffset() is None or observed.utcoffset().total_seconds() != 0:
        raise RunResultRegistryError(f"{name} must use UTC")
    if observed.isoformat().replace("+00:00", "Z") != value:
        raise RunResultRegistryError(f"{name} must be canonical RFC3339 UTC")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise RunResultRegistryError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunResultRegistryError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise RunResultRegistryError(f"{name} must be finite and non-negative")
    return result


def _strict_fields(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise RunResultRegistryError(f"{name} must be a string-keyed object")
    if set(value) != set(expected):
        raise RunResultRegistryError(f"{name} has missing or unknown fields")
    return value


def _metric_values(
    values: object,
) -> tuple[tuple[str, float], ...]:
    if not isinstance(values, (list, tuple)):
        raise RunResultRegistryError("metric_values must be an array")
    result: list[tuple[str, float]] = []
    for index, item in enumerate(values):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RunResultRegistryError(
                f"metric_values[{index}] must be a two-value array"
            )
        metric_id = _identifier(item[0], f"metric_values[{index}].metric_id")
        if isinstance(item[1], bool) or not isinstance(item[1], (int, float)):
            raise RunResultRegistryError(
                f"metric_values[{index}].value must be numeric"
            )
        metric_value = float(item[1])
        if not np.isfinite(metric_value):
            raise RunResultRegistryError(
                f"metric_values[{index}].value must be finite"
            )
        result.append((metric_id, metric_value))
    if [item[0] for item in result] != sorted({item[0] for item in result}):
        raise RunResultRegistryError(
            "metric_values must have unique, sorted metric identifiers"
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ExpectedRunSpecV1:
    """One preregistered semantic cell, without outcome evidence."""

    scenario_id: str
    dataset_domain: str
    method_id: str
    sequence_id: str
    condition_id: str
    training_seed: int
    network_seed: int | None
    c9_trace_id: str | None
    budget_bytes_per_second: int
    scheduler_id: str
    metric_ids: tuple[str, ...]

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.scenario_id,
            self.dataset_domain,
            self.method_id,
            self.sequence_id,
            self.condition_id,
            self.training_seed,
            self.network_seed,
            self.c9_trace_id,
            self.budget_bytes_per_second,
            self.scheduler_id,
        )


@dataclass(frozen=True, slots=True)
class RunResultCellV1:
    """One sequence-level raw result bound to its run evidence and trace."""

    experiment_plan_sha256: str
    detection_cache_sha256: str
    evaluator_contract_sha256: str
    attempt_id: str
    scenario_id: str
    dataset_domain: str
    method_id: str
    sequence_id: str
    condition_id: str
    training_seed: int
    network_seed: int | None
    c9_trace_id: str | None
    budget_bytes_per_second: int
    scheduler_id: str
    metric_values: tuple[tuple[str, float], ...]
    on_wire_bytes_total: int
    wire_measurement_duration_seconds: float
    on_wire_bytes_per_second: float
    status: str
    failure_code: str | None
    checkpoint_sha256: str
    evidence_bundle_sha256: str
    metrics_artifact_role: str
    metrics_artifact_sha256: str
    channel_outcome_trace_sha256: str
    network_trace_sha256: str
    condition_input_manifest_sha256: str | None
    wire_ledger_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_plan_sha256",
            _sha256(self.experiment_plan_sha256, "experiment_plan_sha256"),
        )
        for name in (
            "scenario_id",
            "dataset_domain",
            "method_id",
            "sequence_id",
            "condition_id",
            "scheduler_id",
            "status",
            "metrics_artifact_role",
            "attempt_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise RunResultRegistryError("training_seed must be a non-negative integer")
        if self.network_seed is not None and (
            type(self.network_seed) is not int or self.network_seed < 0
        ):
            raise RunResultRegistryError("network_seed must be non-negative or None")
        if self.c9_trace_id is not None:
            object.__setattr__(
                self, "c9_trace_id", _identifier(self.c9_trace_id, "c9_trace_id")
            )
        if self.condition_id == "C0":
            if self.network_seed is not None or self.c9_trace_id is not None:
                raise RunResultRegistryError(
                    "C0 clean cells must not carry a network seed or C9 trace"
                )
        elif self.condition_id == "C9":
            if self.network_seed is not None or self.c9_trace_id is None:
                raise RunResultRegistryError(
                    "C9 cells require one held-out trace and no network seed"
                )
        elif self.network_seed is None or self.c9_trace_id is not None:
            raise RunResultRegistryError(
                "synthetic pressure cells require one network seed and no C9 trace"
            )
        if self.condition_id in {"C6", "C7", "C8"}:
            if self.condition_input_manifest_sha256 is None:
                raise RunResultRegistryError(
                    "C6-C8 cells require condition_input_manifest_sha256"
                )
            object.__setattr__(
                self,
                "condition_input_manifest_sha256",
                _sha256(
                    self.condition_input_manifest_sha256,
                    "condition_input_manifest_sha256",
                ),
            )
        elif self.condition_input_manifest_sha256 is not None:
            raise RunResultRegistryError(
                "C0-C5 and C9 cells must not carry condition input manifest"
            )
        budget = _positive_int(
            self.budget_bytes_per_second, "budget_bytes_per_second"
        )
        if type(self.on_wire_bytes_total) is not int or self.on_wire_bytes_total < 0:
            raise RunResultRegistryError(
                "on_wire_bytes_total must be a non-negative integer"
            )
        duration = _finite_nonnegative(
            self.wire_measurement_duration_seconds,
            "wire_measurement_duration_seconds",
        )
        if duration <= 0.0:
            raise RunResultRegistryError(
                "wire_measurement_duration_seconds must be positive"
            )
        on_wire = _finite_nonnegative(
            self.on_wire_bytes_per_second, "on_wire_bytes_per_second"
        )
        derived_on_wire = float(self.on_wire_bytes_total) / duration
        if not np.isclose(on_wire, derived_on_wire, rtol=1e-12, atol=1e-12):
            raise RunResultRegistryError(
                "on-wire BPS does not equal ledger bytes divided by duration"
            )
        if on_wire > float(budget) + 1e-9:
            raise RunResultRegistryError(
                "on-wire rate exceeds the preregistered cell budget"
            )
        object.__setattr__(self, "metric_values", _metric_values(self.metric_values))
        object.__setattr__(self, "wire_measurement_duration_seconds", duration)
        object.__setattr__(self, "on_wire_bytes_per_second", on_wire)
        if self.status not in RUN_RESULT_STATUSES_V1:
            raise RunResultRegistryError("status is not a registered terminal status")
        if self.status == RUN_RESULT_STATUS_SUCCESS_V1:
            if not self.metric_values or self.failure_code is not None:
                raise RunResultRegistryError(
                    "successful cells require metrics and no failure_code"
                )
        else:
            if self.metric_values or self.failure_code is None:
                raise RunResultRegistryError(
                    "failed cells require an empty metric set and a failure_code"
                )
            object.__setattr__(
                self, "failure_code", _identifier(self.failure_code, "failure_code")
            )
        for name in (
            "detection_cache_sha256",
            "evaluator_contract_sha256",
            "checkpoint_sha256",
            "evidence_bundle_sha256",
            "metrics_artifact_sha256",
            "channel_outcome_trace_sha256",
            "network_trace_sha256",
            "wire_ledger_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.metrics_artifact_role != RESULT_METRICS_ARTIFACT_ROLE_V1:
            raise RunResultRegistryError(
                "metrics_artifact_role must be per_sequence_metrics"
            )

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.scenario_id,
            self.dataset_domain,
            self.method_id,
            self.sequence_id,
            self.condition_id,
            self.training_seed,
            self.network_seed,
            self.c9_trace_id,
            self.budget_bytes_per_second,
            self.scheduler_id,
        )

    @property
    def replicate_id(self) -> str:
        if self.condition_id == "C0":
            return "clean"
        if self.c9_trace_id is not None:
            return self.c9_trace_id
        assert self.network_seed is not None
        return f"seed:{self.network_seed}"

    @property
    def physical_run_key(self) -> tuple[object, ...]:
        """Run identity shared by scenario aliases of the same execution."""

        return self.key[1:]

    def metric_or_failure_zero(self, metric_id: str) -> float:
        if self.status == RUN_RESULT_STATUS_FAILURE_V1:
            return 0.0
        values = dict(self.metric_values)
        try:
            return values[metric_id]
        except KeyError as exc:  # pragma: no cover - registry parity checks first.
            raise RunResultRegistryError(f"missing metric {metric_id}") from exc

    def to_primitive(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "budget_bytes_per_second": self.budget_bytes_per_second,
            "c9_trace_id": self.c9_trace_id,
            "channel_outcome_trace_sha256": self.channel_outcome_trace_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "condition_id": self.condition_id,
            "condition_input_manifest_sha256": (
                self.condition_input_manifest_sha256
            ),
            "dataset_domain": self.dataset_domain,
            "detection_cache_sha256": self.detection_cache_sha256,
            "evidence_bundle_sha256": self.evidence_bundle_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "experiment_plan_sha256": self.experiment_plan_sha256,
            "failure_code": self.failure_code,
            "method_id": self.method_id,
            "metric_values": [list(item) for item in self.metric_values],
            "metrics_artifact_role": self.metrics_artifact_role,
            "metrics_artifact_sha256": self.metrics_artifact_sha256,
            "network_seed": self.network_seed,
            "network_trace_sha256": self.network_trace_sha256,
            "on_wire_bytes_total": self.on_wire_bytes_total,
            "on_wire_bytes_per_second": self.on_wire_bytes_per_second,
            "scenario_id": self.scenario_id,
            "scheduler_id": self.scheduler_id,
            "sequence_id": self.sequence_id,
            "status": self.status,
            "training_seed": self.training_seed,
            "wire_ledger_sha256": self.wire_ledger_sha256,
            "wire_measurement_duration_seconds": (
                self.wire_measurement_duration_seconds
            ),
        }

    @classmethod
    def from_mapping(cls, value: object) -> "RunResultCellV1":
        fields = frozenset(
            {
                "attempt_id",
                "budget_bytes_per_second",
                "c9_trace_id",
                "channel_outcome_trace_sha256",
                "checkpoint_sha256",
                "condition_id",
                "condition_input_manifest_sha256",
                "dataset_domain",
                "detection_cache_sha256",
                "evidence_bundle_sha256",
                "evaluator_contract_sha256",
                "experiment_plan_sha256",
                "failure_code",
                "method_id",
                "metric_values",
                "metrics_artifact_role",
                "metrics_artifact_sha256",
                "network_seed",
                "network_trace_sha256",
                "on_wire_bytes_total",
                "on_wire_bytes_per_second",
                "scenario_id",
                "scheduler_id",
                "sequence_id",
                "status",
                "training_seed",
                "wire_ledger_sha256",
                "wire_measurement_duration_seconds",
            }
        )
        item = _strict_fields(value, fields, cls.__name__)
        raw_metrics = item["metric_values"]
        if not isinstance(raw_metrics, list):
            raise RunResultRegistryError("metric_values must be an array")
        return cls(
            experiment_plan_sha256=item["experiment_plan_sha256"],
            detection_cache_sha256=item["detection_cache_sha256"],
            evaluator_contract_sha256=item["evaluator_contract_sha256"],
            attempt_id=item["attempt_id"],
            scenario_id=item["scenario_id"],
            dataset_domain=item["dataset_domain"],
            method_id=item["method_id"],
            sequence_id=item["sequence_id"],
            condition_id=item["condition_id"],
            condition_input_manifest_sha256=item[
                "condition_input_manifest_sha256"
            ],
            training_seed=item["training_seed"],
            network_seed=item["network_seed"],
            c9_trace_id=item["c9_trace_id"],
            checkpoint_sha256=item["checkpoint_sha256"],
            budget_bytes_per_second=item["budget_bytes_per_second"],
            scheduler_id=item["scheduler_id"],
            metric_values=tuple(tuple(metric) for metric in raw_metrics),
            on_wire_bytes_total=item["on_wire_bytes_total"],
            wire_measurement_duration_seconds=item[
                "wire_measurement_duration_seconds"
            ],
            on_wire_bytes_per_second=item["on_wire_bytes_per_second"],
            status=item["status"],
            failure_code=item["failure_code"],
            evidence_bundle_sha256=item["evidence_bundle_sha256"],
            metrics_artifact_role=item["metrics_artifact_role"],
            metrics_artifact_sha256=item["metrics_artifact_sha256"],
            channel_outcome_trace_sha256=item[
                "channel_outcome_trace_sha256"
            ],
            network_trace_sha256=item["network_trace_sha256"],
            wire_ledger_sha256=item["wire_ledger_sha256"],
        )


@dataclass(frozen=True, slots=True)
class RunResultRegistryV1:
    """Hash-sealed collection of all expected confirmatory result cells."""

    experiment_plan_sha256: str
    network_trace_manifest_sha256: str
    wire_accounting_config_sha256: str
    heldout_c9_trace_receipt_sha256: str
    attempt_manifest_sha256: str
    confirmatory_val_unsealed_at_utc: str
    execution_started_at_utc: str
    first_prediction_at_utc: str
    completed_at_utc: str
    c9_trace_ids: tuple[str, ...]
    cells: tuple[RunResultCellV1, ...]

    def __post_init__(self) -> None:
        for name in (
            "experiment_plan_sha256",
            "network_trace_manifest_sha256",
            "wire_accounting_config_sha256",
            "heldout_c9_trace_receipt_sha256",
            "attempt_manifest_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in (
            "confirmatory_val_unsealed_at_utc",
            "execution_started_at_utc",
            "first_prediction_at_utc",
            "completed_at_utc",
        ):
            object.__setattr__(
                self, name, _utc_timestamp(getattr(self, name), name)
            )
        unsealed = datetime.fromisoformat(
            self.confirmatory_val_unsealed_at_utc[:-1] + "+00:00"
        )
        started = datetime.fromisoformat(
            self.execution_started_at_utc[:-1] + "+00:00"
        )
        first_prediction = datetime.fromisoformat(
            self.first_prediction_at_utc[:-1] + "+00:00"
        )
        completed = datetime.fromisoformat(self.completed_at_utc[:-1] + "+00:00")
        if not unsealed < started <= first_prediction <= completed:
            raise RunResultRegistryError(
                "confirmatory chronology must satisfy unseal < start <= first prediction <= completion"
            )
        if not isinstance(self.c9_trace_ids, (list, tuple)):
            raise RunResultRegistryError("c9_trace_ids must be an array")
        traces = tuple(_identifier(item, "c9_trace_id") for item in self.c9_trace_ids)
        if len(traces) != 20 or traces != tuple(sorted(set(traces))):
            raise RunResultRegistryError(
                "c9_trace_ids must contain exactly 20 unique sorted identifiers"
            )
        object.__setattr__(self, "c9_trace_ids", traces)
        if not isinstance(self.cells, (list, tuple)) or not self.cells:
            raise RunResultRegistryError("cells must be a non-empty array")
        if not all(isinstance(item, RunResultCellV1) for item in self.cells):
            raise RunResultRegistryError("cells must contain RunResultCellV1")
        cells = tuple(self.cells)
        keys = [item.key for item in cells]
        if len(set(keys)) != len(keys):
            raise RunResultRegistryError("duplicate result cell key")
        if keys != sorted(keys, key=_sortable_key):
            raise RunResultRegistryError("result cells must be sorted by semantic key")
        if any(
            item.experiment_plan_sha256 != self.experiment_plan_sha256
            for item in cells
        ):
            raise RunResultRegistryError("cell has stale experiment plan binding")
        object.__setattr__(self, "cells", cells)

    def payload(self) -> dict[str, object]:
        return {
            "attempt_manifest_sha256": self.attempt_manifest_sha256,
            "c9_trace_ids": list(self.c9_trace_ids),
            "cells": [item.to_primitive() for item in self.cells],
            "completed_at_utc": self.completed_at_utc,
            "execution_started_at_utc": self.execution_started_at_utc,
            "experiment_plan_sha256": self.experiment_plan_sha256,
            "first_prediction_at_utc": self.first_prediction_at_utc,
            "heldout_c9_trace_receipt_sha256": (
                self.heldout_c9_trace_receipt_sha256
            ),
            "kind": "run_result_registry_v1",
            "network_trace_manifest_sha256": self.network_trace_manifest_sha256,
            "schema_version": 1,
            "confirmatory_val_unsealed_at_utc": self.confirmatory_val_unsealed_at_utc,
            "wire_accounting_config_sha256": self.wire_accounting_config_sha256,
        }

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.payload())).hexdigest()

    def sealed_document(self) -> dict[str, object]:
        return {**self.payload(), "content_sha256": self.content_sha256}

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.sealed_document())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> "RunResultRegistryV1":
        fields = frozenset(
            {
                "attempt_manifest_sha256",
                "c9_trace_ids",
                "cells",
                "completed_at_utc",
                "content_sha256",
                "execution_started_at_utc",
                "experiment_plan_sha256",
                "first_prediction_at_utc",
                "heldout_c9_trace_receipt_sha256",
                "kind",
                "network_trace_manifest_sha256",
                "schema_version",
                "confirmatory_val_unsealed_at_utc",
                "wire_accounting_config_sha256",
            }
        )
        item = _strict_fields(value, fields, cls.__name__)
        if item["kind"] != "run_result_registry_v1" or item["schema_version"] != 1:
            raise RunResultRegistryError("unsupported RunResultRegistryV1 header")
        observed = _sha256(item["content_sha256"], "content_sha256")
        payload = {key: item[key] for key in fields if key != "content_sha256"}
        expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if observed != expected:
            raise RunResultRegistryError("registry content SHA-256 mismatch")
        raw_cells = item["cells"]
        if not isinstance(raw_cells, list):
            raise RunResultRegistryError("cells must be an array")
        return cls(
            experiment_plan_sha256=item["experiment_plan_sha256"],
            attempt_manifest_sha256=item["attempt_manifest_sha256"],
            confirmatory_val_unsealed_at_utc=item[
                "confirmatory_val_unsealed_at_utc"
            ],
            execution_started_at_utc=item["execution_started_at_utc"],
            first_prediction_at_utc=item["first_prediction_at_utc"],
            completed_at_utc=item["completed_at_utc"],
            network_trace_manifest_sha256=item["network_trace_manifest_sha256"],
            wire_accounting_config_sha256=item[
                "wire_accounting_config_sha256"
            ],
            heldout_c9_trace_receipt_sha256=item[
                "heldout_c9_trace_receipt_sha256"
            ],
            c9_trace_ids=tuple(item["c9_trace_ids"]),  # type: ignore[arg-type]
            cells=tuple(RunResultCellV1.from_mapping(cell) for cell in raw_cells),
        )


def _sortable_key(key: tuple[object, ...]) -> tuple[str, ...]:
    return tuple("" if item is None else f"{type(item).__name__}:{item}" for item in key)


def _method_scheduler(plan: ExperimentPlanV1, method_id: str) -> str:
    if method_id == "eventtrack-v2x":
        return plan.candidate_scheduler_id
    return plan.strongest_scheduler_baseline_id


def _replicates(
    plan: ExperimentPlanV1,
    condition_id: str,
    c9_trace_ids: tuple[str, ...],
) -> Iterator[tuple[int | None, str | None]]:
    if condition_id == "C9":
        yield from ((None, trace_id) for trace_id in c9_trace_ids)
        return
    if condition_id == "C0":
        yield None, None
        return
    yield from ((network_seed, None) for network_seed in plan.network_seeds)


def iter_expected_run_specs_v1(
    plan: ExperimentPlanV1,
    c9_trace_ids: Iterable[str],
) -> Iterator[ExpectedRunSpecV1]:
    """Yield every and only preregistered confirmatory raw-result cell."""

    if not isinstance(plan, ExperimentPlanV1):
        raise TypeError("plan must be ExperimentPlanV1")
    _validate_confirmatory_plan_bindings_v1(plan)
    traces = tuple(c9_trace_ids)
    if len(traces) != 20 or traces != tuple(sorted(set(traces))):
        raise RunResultRegistryError(
            "c9_trace_ids must contain exactly 20 unique sorted identifiers"
        )
    plan_traces = _plan_c9_trace_ids(plan)
    if plan_traces is not None and traces != plan_traces:
        raise RunResultRegistryError("C9 trace IDs do not match ExperimentPlanV1")

    robust_conditions = tuple(
        condition for condition in plan.network_condition_ids if condition != "C0"
    )
    robust_methods = ("eventtrack-v2x", *plan.qualified_baseline_ids)
    for method_id in sorted(robust_methods):
        for sequence_id in plan.primary_sequence_ids:
            for condition_id in robust_conditions:
                for training_seed in plan.training_seeds:
                    for network_seed, trace_id in _replicates(
                        plan, condition_id, traces
                    ):
                        yield ExpectedRunSpecV1(
                            scenario_id=PRIMARY_ROBUST_SCENARIO_V1,
                            dataset_domain="primary",
                            method_id=method_id,
                            sequence_id=sequence_id,
                            condition_id=condition_id,
                            training_seed=training_seed,
                            network_seed=network_seed,
                            c9_trace_id=trace_id,
                            budget_bytes_per_second=(
                                plan.primary_budget_bytes_per_second
                            ),
                            scheduler_id=_method_scheduler(plan, method_id),
                            metric_ids=("AssA",),
                        )

    for method_id in sorted(
        ("eventtrack-v2x", plan.strongest_qualified_baseline_id)
    ):
        for sequence_id in plan.primary_sequence_ids:
            for training_seed in plan.training_seeds:
                network_seed, trace_id = next(_replicates(plan, "C0", traces))
                yield ExpectedRunSpecV1(
                    scenario_id=PRIMARY_CLEAN_SCENARIO_V1,
                    dataset_domain="primary",
                    method_id=method_id,
                    sequence_id=sequence_id,
                    condition_id="C0",
                    training_seed=training_seed,
                    network_seed=network_seed,
                    c9_trace_id=trace_id,
                    budget_bytes_per_second=plan.primary_budget_bytes_per_second,
                    scheduler_id=_method_scheduler(plan, method_id),
                    metric_ids=("AMOTA", "HOTA"),
                )

    for scheduler_id in sorted(
        (plan.candidate_scheduler_id, plan.strongest_scheduler_baseline_id)
    ):
        for sequence_id in plan.primary_sequence_ids:
            for budget in plan.byte_budgets_per_second:
                for condition_id in robust_conditions:
                    for training_seed in plan.training_seeds:
                        for network_seed, trace_id in _replicates(
                            plan, condition_id, traces
                        ):
                            yield ExpectedRunSpecV1(
                                scenario_id=PRIMARY_PARETO_SCENARIO_V1,
                                dataset_domain="primary",
                                method_id="eventtrack-v2x",
                                sequence_id=sequence_id,
                                condition_id=condition_id,
                                training_seed=training_seed,
                                network_seed=network_seed,
                                c9_trace_id=trace_id,
                                budget_bytes_per_second=budget,
                                scheduler_id=scheduler_id,
                                metric_ids=("AssA",),
                            )

    for method_id in sorted(
        ("eventtrack-v2x", plan.griffin_strongest_baseline_id)
    ):
        for sequence_id in plan.griffin_sequence_ids:
            for condition_id in robust_conditions:
                for training_seed in plan.training_seeds:
                    for network_seed, trace_id in _replicates(
                        plan, condition_id, traces
                    ):
                        yield ExpectedRunSpecV1(
                            scenario_id=GRIFFIN_ROBUST_SCENARIO_V1,
                            dataset_domain="griffin",
                            method_id=method_id,
                            sequence_id=sequence_id,
                            condition_id=condition_id,
                            training_seed=training_seed,
                            network_seed=network_seed,
                            c9_trace_id=trace_id,
                            budget_bytes_per_second=(
                                plan.primary_budget_bytes_per_second
                            ),
                            scheduler_id=_method_scheduler(plan, method_id),
                            metric_ids=("AssA",),
                        )


def _plan_c9_trace_ids(plan: ExperimentPlanV1) -> tuple[str, ...] | None:
    value = getattr(plan, "c9_trace_ids", None)
    if value is None:
        return None
    return tuple(value)


def _validate_confirmatory_plan_bindings_v1(plan: ExperimentPlanV1) -> None:
    if plan.griffin_sequence_ids != GRIFFIN_25M_VAL_SEQUENCE_IDS_V1:
        raise RunResultRegistryError(
            "griffin_sequence_ids must equal the official Griffin-25m val cohort"
        )
    if (
        plan.griffin_strongest_baseline_id
        != plan.strongest_qualified_baseline_id
    ):
        raise RunResultRegistryError(
            "Griffin strongest baseline must equal the frozen strongest qualified "
            "baseline"
        )


def _scenario_agnostic_payload_v1(
    cell: RunResultCellV1,
) -> dict[str, object]:
    payload = cell.to_primitive()
    del payload["scenario_id"]
    return payload


def _validate_primary_duplicate_semantics_v1(
    registry: RunResultRegistryV1,
    plan: ExperimentPlanV1,
) -> None:
    robust_cells = {
        cell.key[1:]: cell
        for cell in registry.cells
        if cell.scenario_id == PRIMARY_ROBUST_SCENARIO_V1
        and cell.dataset_domain == "primary"
        and cell.method_id == "eventtrack-v2x"
        and cell.scheduler_id == plan.candidate_scheduler_id
        and cell.budget_bytes_per_second
        == plan.primary_budget_bytes_per_second
    }
    pareto_cells = {
        cell.key[1:]: cell
        for cell in registry.cells
        if cell.scenario_id == PRIMARY_PARETO_SCENARIO_V1
        and cell.dataset_domain == "primary"
        and cell.method_id == "eventtrack-v2x"
        and cell.scheduler_id == plan.candidate_scheduler_id
        and cell.budget_bytes_per_second
        == plan.primary_budget_bytes_per_second
    }
    if robust_cells.keys() != pareto_cells.keys():
        raise RunResultRegistryError(
            "EventTrack 64k robust/Pareto duplicate cell sets differ"
        )
    for key, robust in robust_cells.items():
        pareto = pareto_cells[key]
        if _scenario_agnostic_payload_v1(robust) != _scenario_agnostic_payload_v1(
            pareto
        ):
            raise RunResultRegistryError(
                "EventTrack 64k robust/Pareto duplicate cells differ outside "
                f"scenario_id for {key}"
            )


def validate_run_result_registry_v1(
    registry: RunResultRegistryV1,
    plan: ExperimentPlanV1,
) -> None:
    """Prove exact plan, matrix, metric, trace, and budget parity."""

    if not isinstance(registry, RunResultRegistryV1):
        raise TypeError("registry must be RunResultRegistryV1")
    if not isinstance(plan, ExperimentPlanV1):
        raise TypeError("plan must be ExperimentPlanV1")
    _validate_confirmatory_plan_bindings_v1(plan)
    if registry.experiment_plan_sha256 != plan.content_sha256:
        raise RunResultRegistryError("registry has stale experiment plan binding")
    if registry.network_trace_manifest_sha256 != plan.network_trace_manifest_sha256:
        raise RunResultRegistryError("network trace manifest does not match plan")
    if registry.wire_accounting_config_sha256 != plan.wire_accounting_config_sha256:
        raise RunResultRegistryError("wire accounting config does not match plan")
    if (
        registry.heldout_c9_trace_receipt_sha256
        != plan.heldout_c9_trace_receipt_sha256
    ):
        raise RunResultRegistryError("held-out C9 trace receipt does not match plan")
    plan_trace_ids = _plan_c9_trace_ids(plan)
    if plan_trace_ids is not None and registry.c9_trace_ids != plan_trace_ids:
        raise RunResultRegistryError("C9 trace IDs do not match ExperimentPlanV1")

    for cell in registry.cells:
        if cell.dataset_domain == "primary":
            expected_detection_cache = plan.detector_cache_sha256
            expected_evaluator_contract = plan.evaluator_contract_sha256
        elif cell.dataset_domain == "griffin":
            expected_detection_cache = plan.griffin_detector_cache_sha256
            expected_evaluator_contract = plan.griffin_evaluator_contract_sha256
        else:  # pragma: no cover - exact matrix validation rejects this too.
            raise RunResultRegistryError(
                f"unsupported result dataset domain: {cell.dataset_domain}"
            )
        if cell.detection_cache_sha256 != expected_detection_cache:
            raise RunResultRegistryError(
                f"detection cache does not match plan for {cell.dataset_domain}"
            )
        if cell.evaluator_contract_sha256 != expected_evaluator_contract:
            raise RunResultRegistryError(
                f"evaluator contract does not match plan for {cell.dataset_domain}"
            )
        expected_checkpoint = plan.run_checkpoint_sha256(
            cell.dataset_domain,
            cell.method_id,
            cell.scheduler_id,
            cell.training_seed,
        )
        if cell.checkpoint_sha256 != expected_checkpoint:
            raise RunResultRegistryError(
                "result checkpoint does not match the seed-specific frozen refit"
            )
        if cell.condition_id == "C9":
            if cell.c9_trace_id not in registry.c9_trace_ids:
                raise RunResultRegistryError(
                    f"unregistered C9 trace ID: {cell.c9_trace_id}"
                )
        elif cell.c9_trace_id is not None:
            raise RunResultRegistryError("only C9 cells may use held-out trace IDs")

    expected: dict[tuple[object, ...], tuple[str, ...]] = {
        spec.key: spec.metric_ids
        for spec in iter_expected_run_specs_v1(plan, registry.c9_trace_ids)
    }
    actual = {cell.key: cell for cell in registry.cells}
    missing = sorted(set(expected) - set(actual), key=_sortable_key)
    extra = sorted(set(actual) - set(expected), key=_sortable_key)
    if missing:
        raise RunResultRegistryError(
            f"missing result cells ({len(missing)}): {missing[:1]}"
        )
    if extra:
        raise RunResultRegistryError(
            f"unexpected result cells ({len(extra)}): {extra[:1]}"
        )
    for key, metric_ids in expected.items():
        cell = actual[key]
        observed_metric_ids = tuple(metric_id for metric_id, _ in cell.metric_values)
        if cell.status == RUN_RESULT_STATUS_SUCCESS_V1:
            if observed_metric_ids != metric_ids:
                raise RunResultRegistryError(
                    f"metric set does not match scenario for cell {key}"
                )
            if any(not 0.0 <= value <= 1.0 for _, value in cell.metric_values):
                raise RunResultRegistryError(
                    f"tracking metric is outside [0, 1] for cell {key}"
                )
        elif observed_metric_ids:
            raise RunResultRegistryError("failed cells cannot carry metric values")

    _validate_primary_duplicate_semantics_v1(registry, plan)

    trace_digests: dict[tuple[object, ...], str] = {}
    bundle_metrics: dict[str, str] = {}
    for cell in registry.cells:
        trace_key = (
            cell.dataset_domain,
            cell.sequence_id,
            cell.condition_id,
            cell.replicate_id,
        )
        previous = trace_digests.setdefault(
            trace_key, cell.channel_outcome_trace_sha256
        )
        if previous != cell.channel_outcome_trace_sha256:
            raise RunResultRegistryError(
                f"methods did not reuse one exogenous channel trace for {trace_key}"
            )
        previous_metrics = bundle_metrics.setdefault(
            cell.evidence_bundle_sha256, cell.metrics_artifact_sha256
        )
        if previous_metrics != cell.metrics_artifact_sha256:
            raise RunResultRegistryError(
                "one evidence bundle digest refers to multiple metrics artifacts"
            )


def validate_run_result_external_inputs_v1(
    registry: RunResultRegistryV1,
    plan: ExperimentPlanV1,
    *,
    measured_trace_receipt: MeasuredTraceReceiptV1,
    condition_input_manifests: Mapping[str, ConditionInputManifestV1],
) -> None:
    """Open and bind every C6--C9 external-input artifact used by formal runs.

    The structural registry validator deliberately accepts content digests.  This
    companion validator opens the referenced objects, joins C9 trace IDs to the
    measured packet receipt, and prevents one attempt or disturbance manifest
    from being aliased across distinct physical runs.  The only permitted
    repeated semantic cells are the preregistered robust/Pareto views of the
    same EventTrack 64 kB/s physical run.
    """

    validate_run_result_registry_v1(registry, plan)
    if not isinstance(measured_trace_receipt, MeasuredTraceReceiptV1):
        raise TypeError("measured_trace_receipt must be MeasuredTraceReceiptV1")
    if (
        measured_trace_receipt.content_sha256
        != plan.heldout_c9_trace_receipt_sha256
        or measured_trace_receipt.held_out_trace_ids != plan.c9_trace_ids
    ):
        raise RunResultRegistryError(
            "measured C9 trace receipt content or held-out IDs do not match plan"
        )
    if not isinstance(condition_input_manifests, Mapping):
        raise TypeError("condition_input_manifests must be a mapping")
    inventory: dict[str, ConditionInputManifestV1] = {}
    for raw_digest, manifest in condition_input_manifests.items():
        digest = _sha256(raw_digest, "condition input manifest inventory key")
        if not isinstance(manifest, ConditionInputManifestV1):
            raise TypeError(
                "condition_input_manifests values must be ConditionInputManifestV1"
            )
        if manifest.content_sha256 != digest:
            raise RunResultRegistryError(
                "condition input manifest inventory key does not match content"
            )
        inventory[digest] = manifest

    held_out_packets = dict(
        measured_trace_receipt.held_out_packet_metadata_sha256s
    )
    referenced_manifests: set[str] = set()
    attempt_to_physical_key: dict[str, tuple[object, ...]] = {}
    manifest_to_physical_key: dict[str, tuple[object, ...]] = {}
    physical_key_to_manifest: dict[tuple[object, ...], str] = {}
    for cell in registry.cells:
        physical_key = (
            cell.dataset_domain,
            cell.method_id,
            cell.scheduler_id,
            cell.sequence_id,
            cell.condition_id,
            cell.training_seed,
            cell.network_seed,
            cell.c9_trace_id,
            cell.budget_bytes_per_second,
        )
        previous_physical_key = attempt_to_physical_key.setdefault(
            cell.attempt_id, physical_key
        )
        if previous_physical_key != physical_key:
            raise RunResultRegistryError(
                "one attempt_id is aliased across distinct physical runs"
            )
        if cell.condition_id == "C9":
            assert cell.c9_trace_id is not None
            if (
                cell.channel_outcome_trace_sha256
                != held_out_packets.get(cell.c9_trace_id)
            ):
                raise RunResultRegistryError(
                    "C9 cell is not bound to held-out packet metadata"
                )
        if cell.condition_id not in {"C6", "C7", "C8"}:
            continue
        digest = cell.condition_input_manifest_sha256
        assert digest is not None
        manifest = inventory.get(digest)
        if manifest is None:
            raise RunResultRegistryError(
                "C6-C8 cell references a missing condition input manifest"
            )
        previous_physical_key = manifest_to_physical_key.setdefault(
            digest, physical_key
        )
        if previous_physical_key != physical_key:
            raise RunResultRegistryError(
                "one condition input manifest is aliased across distinct physical runs"
            )
        previous_manifest = physical_key_to_manifest.setdefault(physical_key, digest)
        if previous_manifest != digest:
            raise RunResultRegistryError(
                "one physical run refers to multiple condition input manifests"
            )
        referenced_manifests.add(digest)
        expected_config_sha256 = plan.run_config_sha256(
            cell.dataset_domain,
            cell.method_id,
            cell.scheduler_id,
        )
        observed = (
            manifest.run_id,
            manifest.condition_id.value,
            manifest.network_trace_sha256,
            manifest.run_config_sha256,
            manifest.detection_cache_sha256,
        )
        expected = (
            cell.attempt_id,
            cell.condition_id,
            cell.network_trace_sha256,
            expected_config_sha256,
            cell.detection_cache_sha256,
        )
        if observed != expected:
            raise RunResultRegistryError(
                "condition input manifest does not match its physical run"
            )
    if frozenset(inventory) != frozenset(referenced_manifests):
        raise RunResultRegistryError(
            "condition input manifest inventory has missing or extra artifacts"
        )


def _select_cells(
    registry: RunResultRegistryV1,
    *,
    scenario_id: str,
    method_id: str | None = None,
    scheduler_id: str | None = None,
    budget: int | None = None,
) -> tuple[RunResultCellV1, ...]:
    return tuple(
        cell
        for cell in registry.cells
        if cell.scenario_id == scenario_id
        and (method_id is None or cell.method_id == method_id)
        and (scheduler_id is None or cell.scheduler_id == scheduler_id)
        and (budget is None or cell.budget_bytes_per_second == budget)
    )


def _robust_scores(
    cells: tuple[RunResultCellV1, ...],
    *,
    sequence_ids: tuple[str, ...],
    condition_ids: tuple[str, ...],
    metric_id: str,
) -> SequenceScoresV1:
    grouped: dict[tuple[str, str], list[float]] = {}
    for cell in cells:
        grouped.setdefault((cell.sequence_id, cell.condition_id), []).append(
            cell.metric_or_failure_zero(metric_id)
        )
    rows: list[tuple[str, float]] = []
    for sequence_id in sequence_ids:
        condition_values = [
            float(np.mean(grouped[(sequence_id, condition_id)]))
            for condition_id in condition_ids
        ]
        rows.append((sequence_id, float(np.mean(condition_values))))
    return SequenceScoresV1(tuple(rows))


def _clean_scores(
    cells: tuple[RunResultCellV1, ...],
    *,
    sequence_ids: tuple[str, ...],
    metric_id: str,
) -> SequenceScoresV1:
    grouped: dict[str, list[float]] = {}
    for cell in cells:
        grouped.setdefault(cell.sequence_id, []).append(
            cell.metric_or_failure_zero(metric_id)
        )
    return SequenceScoresV1(
        tuple(
            (sequence_id, float(np.mean(grouped[sequence_id])))
            for sequence_id in sequence_ids
        )
    )


def _robust_on_wire_bps(
    cells: tuple[RunResultCellV1, ...],
    *,
    sequence_ids: tuple[str, ...],
    condition_ids: tuple[str, ...],
) -> dict[str, float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for cell in cells:
        grouped.setdefault((cell.sequence_id, cell.condition_id), []).append(
            cell.on_wire_bytes_per_second
        )
    return {
        sequence_id: float(
            np.mean(
                [
                    np.mean(grouped[(sequence_id, condition_id)])
                    for condition_id in condition_ids
                ]
            )
        )
        for sequence_id in sequence_ids
    }


def derive_publication_metrics_v1(
    registry: RunResultRegistryV1,
    plan: ExperimentPlanV1,
) -> PublicationMetricsV1:
    """Uniquely derive gate input from complete raw cells; failures score zero."""

    validate_run_result_registry_v1(registry, plan)
    robust_conditions = tuple(
        condition for condition in plan.network_condition_ids if condition != "C0"
    )
    robust_eventtrack = _robust_scores(
        _select_cells(
            registry,
            scenario_id=PRIMARY_ROBUST_SCENARIO_V1,
            method_id="eventtrack-v2x",
        ),
        sequence_ids=plan.primary_sequence_ids,
        condition_ids=robust_conditions,
        metric_id="AssA",
    )
    robust_baselines = tuple(
        MethodScoresV1(
            method_id,
            _robust_scores(
                _select_cells(
                    registry,
                    scenario_id=PRIMARY_ROBUST_SCENARIO_V1,
                    method_id=method_id,
                ),
                sequence_ids=plan.primary_sequence_ids,
                condition_ids=robust_conditions,
                metric_id="AssA",
            ),
        )
        for method_id in plan.qualified_baseline_ids
    )

    clean_candidate_cells = _select_cells(
        registry,
        scenario_id=PRIMARY_CLEAN_SCENARIO_V1,
        method_id="eventtrack-v2x",
    )
    clean_baseline_cells = _select_cells(
        registry,
        scenario_id=PRIMARY_CLEAN_SCENARIO_V1,
        method_id=plan.strongest_qualified_baseline_id,
    )
    clean_eventtrack = tuple(
        NamedMetricScoresV1(
            metric_id,
            _clean_scores(
                clean_candidate_cells,
                sequence_ids=plan.primary_sequence_ids,
                metric_id=metric_id,
            ),
        )
        for metric_id in ("AMOTA", "HOTA")
    )
    clean_baseline = tuple(
        NamedMetricScoresV1(
            metric_id,
            _clean_scores(
                clean_baseline_cells,
                sequence_ids=plan.primary_sequence_ids,
                metric_id=metric_id,
            ),
        )
        for metric_id in ("AMOTA", "HOTA")
    )

    def pareto_scores(scheduler_id: str) -> BudgetedSequenceScoresV1:
        selected_by_budget = {
            budget: _select_cells(
                registry,
                scenario_id=PRIMARY_PARETO_SCENARIO_V1,
                scheduler_id=scheduler_id,
                budget=budget,
            )
            for budget in plan.byte_budgets_per_second
        }
        by_budget = {
            budget: _robust_scores(
                selected_by_budget[budget],
                sequence_ids=plan.primary_sequence_ids,
                condition_ids=robust_conditions,
                metric_id="AssA",
            ).as_mapping()
            for budget in plan.byte_budgets_per_second
        }
        wire_by_budget = {
            budget: _robust_on_wire_bps(
                selected_by_budget[budget],
                sequence_ids=plan.primary_sequence_ids,
                condition_ids=robust_conditions,
            )
            for budget in plan.byte_budgets_per_second
        }
        rows = [
            (
                sequence_id,
                tuple(
                    by_budget[budget][sequence_id]
                    for budget in plan.byte_budgets_per_second
                ),
            )
            for sequence_id in plan.primary_sequence_ids
        ]
        wire_rows = [
            (
                sequence_id,
                tuple(
                    wire_by_budget[budget][sequence_id]
                    for budget in plan.byte_budgets_per_second
                ),
            )
            for sequence_id in plan.primary_sequence_ids
        ]
        return BudgetedSequenceScoresV1(
            scheduler_id=scheduler_id,
            byte_budgets_per_second=plan.byte_budgets_per_second,
            sequence_scores=tuple(rows),
            sequence_on_wire_bps=tuple(wire_rows),
        )

    griffin_eventtrack = _robust_scores(
        _select_cells(
            registry,
            scenario_id=GRIFFIN_ROBUST_SCENARIO_V1,
            method_id="eventtrack-v2x",
        ),
        sequence_ids=plan.griffin_sequence_ids,
        condition_ids=robust_conditions,
        metric_id="AssA",
    )
    griffin_baseline = _robust_scores(
        _select_cells(
            registry,
            scenario_id=GRIFFIN_ROBUST_SCENARIO_V1,
            method_id=plan.griffin_strongest_baseline_id,
        ),
        sequence_ids=plan.griffin_sequence_ids,
        condition_ids=robust_conditions,
        metric_id="AssA",
    )
    return PublicationMetricsV1(
        experiment_plan_sha256=plan.content_sha256,
        candidate_method_id="eventtrack-v2x",
        robust_metric_id=plan.primary_metric,
        robust_eventtrack=robust_eventtrack,
        robust_baselines=robust_baselines,
        clean_eventtrack=clean_eventtrack,
        clean_strongest_baseline_id=plan.strongest_qualified_baseline_id,
        clean_strongest_baseline=clean_baseline,
        pareto_metric_id="AssA",
        pareto_eventtrack=pareto_scores(plan.candidate_scheduler_id),
        pareto_strongest_baseline=pareto_scores(
            plan.strongest_scheduler_baseline_id
        ),
        griffin_metric_id=plan.primary_metric,
        griffin_eventtrack=griffin_eventtrack,
        griffin_strongest_baseline_id=plan.griffin_strongest_baseline_id,
        griffin_strongest_baseline=griffin_baseline,
    )


def decode_run_result_registry(data: bytes) -> RunResultRegistryV1:
    """Decode canonical registry bytes and reject duplicate JSON keys."""

    if type(data) is not bytes:
        raise TypeError("registry data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in items:
            if key in result:
                raise RunResultRegistryError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                RunResultRegistryError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunResultRegistryError("invalid RunResultRegistryV1 JSON") from exc
    registry = RunResultRegistryV1.from_mapping(value)
    if registry.canonical_bytes() != data:
        raise RunResultRegistryError("RunResultRegistryV1 JSON is not canonical")
    return registry


__all__ = [
    "ExpectedRunSpecV1",
    "GRIFFIN_ROBUST_SCENARIO_V1",
    "PRIMARY_CLEAN_SCENARIO_V1",
    "PRIMARY_PARETO_SCENARIO_V1",
    "PRIMARY_ROBUST_SCENARIO_V1",
    "RESULT_METRICS_ARTIFACT_ROLE_V1",
    "RUN_RESULT_SCENARIOS_V1",
    "RUN_RESULT_STATUSES_V1",
    "RUN_RESULT_STATUS_FAILURE_V1",
    "RUN_RESULT_STATUS_SUCCESS_V1",
    "RunResultCellV1",
    "RunResultRegistryError",
    "RunResultRegistryV1",
    "decode_run_result_registry",
    "derive_publication_metrics_v1",
    "iter_expected_run_specs_v1",
    "validate_run_result_external_inputs_v1",
    "validate_run_result_registry_v1",
]
