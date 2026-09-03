"""Canonical Table-IV ablation evidence for EventTrack-V2X.

The contract freezes the full EventTrack reference and nine planned ablations
over the 21-sequence V2X-Seq-SPD confirmatory validation cohort.  It is
supplement/Table-IV evidence only and can never enter the primary SOTA
publication gate.  The module records evidence; it does not execute trackers
or manufacture experimental results.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import zip_longest
import json
import re
from typing import Iterable, Iterator, Mapping

import numpy as np

from .experiment import (
    CONFIRMATORY_NETWORK_SEEDS_V1,
    CONFIRMATORY_TRAINING_SEEDS_V1,
    ExperimentPlanV1,
)
from .measured_trace import MeasuredTraceReceiptV1
from .network_disturbance import ConditionInputManifestV1
from .supplemental_registry import (
    CALIBRATION_METRIC_IDS_V1,
    SYSTEM_METRIC_IDS_V1,
    TRACKING_METRIC_IDS_V1,
    _metric_domain_violation_v1,
)
from .wire import canonical_json_bytes


ABLATION_VARIANT_IDS_V1 = tuple(
    sorted(
        (
            "full_eventtrack",
            "no_oosm",
            "point_timestamp",
            "no_mixture",
            "no_lineage",
            "correlated_as_independent",
            "no_ci",
            "top1_hard_association",
            "no_identity_entropy",
            "confidence_top_k_instead_of_voi",
        )
    )
)
FULL_EVENTTRACK_VARIANT_ID_V1 = "full_eventtrack"
ABLATION_BUDGET_BYTES_PER_SECOND_V1 = 64_000
DISTURBED_INPUT_CONDITION_IDS_V1 = ("C6", "C7", "C8")
REQUIRED_ABLATION_METRIC_IDS_V1 = tuple(
    sorted(
        TRACKING_METRIC_IDS_V1
        + CALIBRATION_METRIC_IDS_V1
        + SYSTEM_METRIC_IDS_V1
    )
)
ABLATION_STATUS_SUCCESS_V1 = "success"
ABLATION_STATUS_FAILURE_V1 = "algorithm_failure"
ABLATION_STATUSES_V1 = (
    ABLATION_STATUS_FAILURE_V1,
    ABLATION_STATUS_SUCCESS_V1,
)
ABLATION_FAILURE_SEMANTICS_V1 = "all-required-metrics-zero"
ABLATION_METRICS_ARTIFACT_ROLE_V1 = "per_sequence_metrics"
ABLATION_PAPER_SCOPE_V1 = "supplement-table-iv-only"
FORMAL_ABLATION_CELL_COUNT_V1 = 63_630
FORMAL_PRIMARY_SEQUENCE_COUNT_V1 = 21
FORMAL_C9_TRACE_COUNT_V1 = 20

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


class AblationRegistryError(ValueError):
    """Raised when ablation evidence is incomplete or non-canonical."""


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise AblationRegistryError(f"{name} must be a canonical identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise AblationRegistryError(f"{name} must be a lowercase SHA-256")
    return value


def _sorted_unique_identifiers(values: object, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise AblationRegistryError(f"{name} must be an array")
    result = tuple(_identifier(value, f"{name} item") for value in values)
    if not result or result != tuple(sorted(set(result))):
        raise AblationRegistryError(f"{name} must be non-empty, unique, sorted")
    return result


def _sorted_unique_positive_ints(values: object, name: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise AblationRegistryError(f"{name} must be an array")
    result = tuple(values)
    if (
        not result
        or any(type(value) is not int or value <= 0 for value in result)
        or result != tuple(sorted(set(result)))
    ):
        raise AblationRegistryError(
            f"{name} must contain positive, unique, sorted integers"
        )
    return result


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AblationRegistryError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise AblationRegistryError(f"{name} must be finite and positive")
    return result


def _strict_fields(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise AblationRegistryError(f"{name} must be a string-keyed object")
    if frozenset(value) != expected:
        raise AblationRegistryError(f"{name} has missing or unknown fields")
    return value


def _sortable_key(key: tuple[object, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for item in key:
        if item is None:
            result.append("")
        elif type(item) is int:
            result.append(f"int:{item:020d}")
        else:
            result.append(f"{type(item).__name__}:{item}")
    return tuple(result)


def _metric_values(values: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(values, (list, tuple)):
        raise AblationRegistryError("metric_values must be an array")
    result: list[tuple[str, float]] = []
    for index, item in enumerate(values):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise AblationRegistryError(
                f"metric_values[{index}] must be a two-value array"
            )
        metric_id = _identifier(item[0], f"metric_values[{index}].metric_id")
        if isinstance(item[1], bool) or not isinstance(item[1], (int, float)):
            raise AblationRegistryError(
                f"metric_values[{index}].value must be numeric"
            )
        metric_value = float(item[1])
        if not np.isfinite(metric_value):
            raise AblationRegistryError(
                f"metric_values[{index}].value must be finite"
            )
        result.append((metric_id, metric_value))
    if tuple(metric_id for metric_id, _ in result) != tuple(
        sorted({metric_id for metric_id, _ in result})
    ):
        raise AblationRegistryError(
            "metric_values must have unique, sorted metric identifiers"
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class AblationCohortV1:
    """Expected dimensions; reduced values are only for structural tests."""

    sequence_ids: tuple[str, ...]
    training_seeds: tuple[int, ...]
    network_seeds: tuple[int, ...]
    c9_trace_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence_ids",
            _sorted_unique_identifiers(self.sequence_ids, "sequence_ids"),
        )
        object.__setattr__(
            self,
            "training_seeds",
            _sorted_unique_positive_ints(self.training_seeds, "training_seeds"),
        )
        object.__setattr__(
            self,
            "network_seeds",
            _sorted_unique_positive_ints(self.network_seeds, "network_seeds"),
        )
        object.__setattr__(
            self,
            "c9_trace_ids",
            _sorted_unique_identifiers(self.c9_trace_ids, "c9_trace_ids"),
        )


@dataclass(frozen=True, slots=True)
class AblationConfigBindingV1:
    """Frozen parameter artifact for one reference or ablation variant."""

    variant_id: str
    config_sha256: str

    _FIELDS = frozenset({"config_sha256", "variant_id"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant_id", _identifier(self.variant_id, "variant_id"))
        if self.variant_id not in ABLATION_VARIANT_IDS_V1:
            raise AblationRegistryError("variant_id is not preregistered")
        object.__setattr__(
            self, "config_sha256", _sha256(self.config_sha256, "config_sha256")
        )

    def to_primitive(self) -> dict[str, str]:
        return {
            "config_sha256": self.config_sha256,
            "variant_id": self.variant_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "AblationConfigBindingV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        return cls(
            variant_id=item["variant_id"],
            config_sha256=item["config_sha256"],
        )


def ablation_config_manifest_sha256_v1(
    bindings: Iterable[AblationConfigBindingV1],
) -> str:
    """Hash a canonical complete variant-config manifest."""

    values = tuple(bindings)
    if not all(isinstance(item, AblationConfigBindingV1) for item in values):
        raise TypeError("bindings must contain AblationConfigBindingV1")
    variant_ids = tuple(item.variant_id for item in values)
    if variant_ids != tuple(sorted(set(variant_ids))):
        raise AblationRegistryError("config bindings must be unique and sorted")
    return hashlib.sha256(
        canonical_json_bytes([item.to_primitive() for item in values])
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ExpectedAblationRunSpecV1:
    """One expected semantic cell without outcome evidence."""

    variant_id: str
    sequence_id: str
    condition_id: str
    training_seed: int
    network_seed: int | None
    c9_trace_id: str | None

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.variant_id,
            self.sequence_id,
            self.condition_id,
            self.training_seed,
            self.network_seed,
            self.c9_trace_id,
            ABLATION_BUDGET_BYTES_PER_SECOND_V1,
        )


def _replicates(
    cohort: AblationCohortV1, condition_id: str
) -> Iterator[tuple[int | None, str | None]]:
    if condition_id == "C0":
        yield None, None
    elif condition_id == "C9":
        yield from ((None, trace_id) for trace_id in cohort.c9_trace_ids)
    else:
        yield from ((seed, None) for seed in cohort.network_seeds)


def _iter_expected_ablation_specs_for_cohort(
    cohort: AblationCohortV1,
) -> Iterator[ExpectedAblationRunSpecV1]:
    if not isinstance(cohort, AblationCohortV1):
        raise TypeError("cohort must be AblationCohortV1")
    for variant_id in ABLATION_VARIANT_IDS_V1:
        for sequence_id in cohort.sequence_ids:
            for condition_id in tuple(f"C{index}" for index in range(10)):
                for training_seed in cohort.training_seeds:
                    for network_seed, trace_id in _replicates(cohort, condition_id):
                        yield ExpectedAblationRunSpecV1(
                            variant_id=variant_id,
                            sequence_id=sequence_id,
                            condition_id=condition_id,
                            training_seed=training_seed,
                            network_seed=network_seed,
                            c9_trace_id=trace_id,
                        )


def formal_ablation_cohort_v1(plan: ExperimentPlanV1) -> AblationCohortV1:
    """Resolve official dimensions and reject any weakened experiment plan."""

    if not isinstance(plan, ExperimentPlanV1):
        raise TypeError("plan must be ExperimentPlanV1")
    if (
        len(plan.primary_sequence_ids) != FORMAL_PRIMARY_SEQUENCE_COUNT_V1
        or plan.primary_sequence_ids != tuple(sorted(set(plan.primary_sequence_ids)))
    ):
        raise AblationRegistryError(
            "formal ablations require exactly 21 unique sorted confirmatory "
            "validation sequences"
        )
    if plan.training_seeds != CONFIRMATORY_TRAINING_SEEDS_V1:
        raise AblationRegistryError("formal ablations require official training seeds")
    if plan.network_seeds != CONFIRMATORY_NETWORK_SEEDS_V1:
        raise AblationRegistryError("formal ablations require official network seeds")
    if (
        len(plan.c9_trace_ids) != FORMAL_C9_TRACE_COUNT_V1
        or plan.c9_trace_ids != tuple(sorted(set(plan.c9_trace_ids)))
    ):
        raise AblationRegistryError(
            "formal ablations require exactly twenty held-out C9 traces"
        )
    if plan.network_condition_ids != tuple(f"C{index}" for index in range(10)):
        raise AblationRegistryError("formal ablations require C0 through C9")
    if plan.primary_budget_bytes_per_second != (
        ABLATION_BUDGET_BYTES_PER_SECOND_V1
    ):
        raise AblationRegistryError("formal ablations require the 64 kB/s budget")
    return AblationCohortV1(
        sequence_ids=plan.primary_sequence_ids,
        training_seeds=plan.training_seeds,
        network_seeds=plan.network_seeds,
        c9_trace_ids=plan.c9_trace_ids,
    )


def iter_expected_paper_ablation_specs_v1(
    plan: ExperimentPlanV1,
) -> Iterator[ExpectedAblationRunSpecV1]:
    """Stream all 63,630 formal Table-IV semantic cells."""

    yield from _iter_expected_ablation_specs_for_cohort(
        formal_ablation_cohort_v1(plan)
    )


@dataclass(frozen=True, slots=True)
class AblationResultCellV1:
    """One ablation result bound to inputs, prediction, metrics, and ledger."""

    experiment_plan_sha256: str
    variant_id: str
    sequence_id: str
    condition_id: str
    training_seed: int
    network_seed: int | None
    c9_trace_id: str | None
    budget_bytes_per_second: int
    status: str
    failure_code: str | None
    metric_values: tuple[tuple[str, float], ...]
    on_wire_bytes_total: int
    measurement_duration_seconds: float
    attempt_id: str
    variant_config_sha256: str
    detection_cache_sha256: str
    network_trace_sha256: str
    channel_outcome_trace_sha256: str
    disturbed_input_manifest_sha256: str | None
    prediction_archive_sha256: str
    evaluator_contract_sha256: str
    metrics_artifact_role: str
    metrics_artifact_sha256: str
    evidence_bundle_sha256: str
    wire_ledger_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_plan_sha256",
            _sha256(self.experiment_plan_sha256, "experiment_plan_sha256"),
        )
        for name in (
            "variant_id",
            "sequence_id",
            "condition_id",
            "status",
            "attempt_id",
            "metrics_artifact_role",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.variant_id not in ABLATION_VARIANT_IDS_V1:
            raise AblationRegistryError("variant_id is not preregistered")
        if self.condition_id not in tuple(f"C{index}" for index in range(10)):
            raise AblationRegistryError("condition_id must be C0 through C9")
        if type(self.training_seed) is not int or self.training_seed <= 0:
            raise AblationRegistryError("training_seed must be a positive integer")
        if self.network_seed is not None and (
            type(self.network_seed) is not int or self.network_seed <= 0
        ):
            raise AblationRegistryError("network_seed must be positive or None")
        if self.c9_trace_id is not None:
            object.__setattr__(
                self, "c9_trace_id", _identifier(self.c9_trace_id, "c9_trace_id")
            )
        if self.condition_id == "C0":
            if self.network_seed is not None or self.c9_trace_id is not None:
                raise AblationRegistryError("C0 cannot carry a pressure replicate")
        elif self.condition_id == "C9":
            if self.network_seed is not None or self.c9_trace_id is None:
                raise AblationRegistryError("C9 requires one trace and no seed")
        elif self.network_seed is None or self.c9_trace_id is not None:
            raise AblationRegistryError(
                "C1-C8 require one network seed and no C9 trace"
            )
        if self.budget_bytes_per_second != ABLATION_BUDGET_BYTES_PER_SECOND_V1:
            raise AblationRegistryError("all ablation cells must use 64 kB/s")
        if self.condition_id in DISTURBED_INPUT_CONDITION_IDS_V1:
            if self.disturbed_input_manifest_sha256 is None:
                raise AblationRegistryError(
                    "C6-C8 require a disturbed tracker-input manifest"
                )
            object.__setattr__(
                self,
                "disturbed_input_manifest_sha256",
                _sha256(
                    self.disturbed_input_manifest_sha256,
                    "disturbed_input_manifest_sha256",
                ),
            )
        elif self.disturbed_input_manifest_sha256 is not None:
            raise AblationRegistryError(
                "only C6-C8 may carry a disturbed tracker-input manifest"
            )
        metrics = _metric_values(self.metric_values)
        object.__setattr__(self, "metric_values", metrics)
        metric_violation = _metric_domain_violation_v1(metrics)
        if metric_violation is not None:
            raise AblationRegistryError(
                f"metric semantic domain violation: {metric_violation}"
            )
        if self.status not in ABLATION_STATUSES_V1:
            raise AblationRegistryError("status is not a terminal status")
        if self.status == ABLATION_STATUS_SUCCESS_V1:
            if tuple(metric_id for metric_id, _ in metrics) != (
                REQUIRED_ABLATION_METRIC_IDS_V1
            ) or self.failure_code is not None:
                raise AblationRegistryError(
                    "success requires every exact ablation metric and no failure_code"
                )
        else:
            if metrics or self.failure_code is None:
                raise AblationRegistryError(
                    "algorithm failure requires no metrics and one failure_code"
                )
            object.__setattr__(
                self, "failure_code", _identifier(self.failure_code, "failure_code")
            )
        if type(self.on_wire_bytes_total) is not int or self.on_wire_bytes_total < 0:
            raise AblationRegistryError(
                "on_wire_bytes_total must be a non-negative integer"
            )
        duration = _finite_positive(
            self.measurement_duration_seconds, "measurement_duration_seconds"
        )
        object.__setattr__(self, "measurement_duration_seconds", duration)
        if self.on_wire_bytes_total / duration > (
            float(ABLATION_BUDGET_BYTES_PER_SECOND_V1) + 1e-9
        ):
            raise AblationRegistryError("on-wire rate exceeds 64 kB/s")
        for name in (
            "variant_config_sha256",
            "detection_cache_sha256",
            "network_trace_sha256",
            "channel_outcome_trace_sha256",
            "prediction_archive_sha256",
            "evaluator_contract_sha256",
            "metrics_artifact_sha256",
            "evidence_bundle_sha256",
            "wire_ledger_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.metrics_artifact_role != ABLATION_METRICS_ARTIFACT_ROLE_V1:
            raise AblationRegistryError(
                "metrics_artifact_role must be per_sequence_metrics"
            )

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.variant_id,
            self.sequence_id,
            self.condition_id,
            self.training_seed,
            self.network_seed,
            self.c9_trace_id,
            self.budget_bytes_per_second,
        )

    def metric_or_failure_zero(self, metric_id: str) -> float:
        if metric_id not in REQUIRED_ABLATION_METRIC_IDS_V1:
            raise AblationRegistryError("metric_id is not required")
        if self.status == ABLATION_STATUS_FAILURE_V1:
            return 0.0
        return dict(self.metric_values)[metric_id]

    def to_primitive(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "budget_bytes_per_second": self.budget_bytes_per_second,
            "c9_trace_id": self.c9_trace_id,
            "channel_outcome_trace_sha256": self.channel_outcome_trace_sha256,
            "condition_id": self.condition_id,
            "detection_cache_sha256": self.detection_cache_sha256,
            "disturbed_input_manifest_sha256": (
                self.disturbed_input_manifest_sha256
            ),
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "evidence_bundle_sha256": self.evidence_bundle_sha256,
            "experiment_plan_sha256": self.experiment_plan_sha256,
            "failure_code": self.failure_code,
            "measurement_duration_seconds": self.measurement_duration_seconds,
            "metric_values": [list(item) for item in self.metric_values],
            "metrics_artifact_role": self.metrics_artifact_role,
            "metrics_artifact_sha256": self.metrics_artifact_sha256,
            "network_seed": self.network_seed,
            "network_trace_sha256": self.network_trace_sha256,
            "on_wire_bytes_total": self.on_wire_bytes_total,
            "prediction_archive_sha256": self.prediction_archive_sha256,
            "sequence_id": self.sequence_id,
            "status": self.status,
            "training_seed": self.training_seed,
            "variant_config_sha256": self.variant_config_sha256,
            "variant_id": self.variant_id,
            "wire_ledger_sha256": self.wire_ledger_sha256,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "AblationResultCellV1":
        fields = frozenset(
            {
                "attempt_id",
                "budget_bytes_per_second",
                "c9_trace_id",
                "channel_outcome_trace_sha256",
                "condition_id",
                "detection_cache_sha256",
                "disturbed_input_manifest_sha256",
                "evaluator_contract_sha256",
                "evidence_bundle_sha256",
                "experiment_plan_sha256",
                "failure_code",
                "measurement_duration_seconds",
                "metric_values",
                "metrics_artifact_role",
                "metrics_artifact_sha256",
                "network_seed",
                "network_trace_sha256",
                "on_wire_bytes_total",
                "prediction_archive_sha256",
                "sequence_id",
                "status",
                "training_seed",
                "variant_config_sha256",
                "variant_id",
                "wire_ledger_sha256",
            }
        )
        item = _strict_fields(value, fields, cls.__name__)
        raw_metrics = item["metric_values"]
        if not isinstance(raw_metrics, list):
            raise AblationRegistryError("metric_values must be an array")
        return cls(
            experiment_plan_sha256=item["experiment_plan_sha256"],
            variant_id=item["variant_id"],
            sequence_id=item["sequence_id"],
            condition_id=item["condition_id"],
            training_seed=item["training_seed"],
            network_seed=item["network_seed"],
            c9_trace_id=item["c9_trace_id"],
            budget_bytes_per_second=item["budget_bytes_per_second"],
            status=item["status"],
            failure_code=item["failure_code"],
            metric_values=tuple(tuple(metric) for metric in raw_metrics),
            on_wire_bytes_total=item["on_wire_bytes_total"],
            measurement_duration_seconds=item["measurement_duration_seconds"],
            attempt_id=item["attempt_id"],
            variant_config_sha256=item["variant_config_sha256"],
            detection_cache_sha256=item["detection_cache_sha256"],
            network_trace_sha256=item["network_trace_sha256"],
            channel_outcome_trace_sha256=item[
                "channel_outcome_trace_sha256"
            ],
            disturbed_input_manifest_sha256=item[
                "disturbed_input_manifest_sha256"
            ],
            prediction_archive_sha256=item["prediction_archive_sha256"],
            evaluator_contract_sha256=item["evaluator_contract_sha256"],
            metrics_artifact_role=item["metrics_artifact_role"],
            metrics_artifact_sha256=item["metrics_artifact_sha256"],
            evidence_bundle_sha256=item["evidence_bundle_sha256"],
            wire_ledger_sha256=item["wire_ledger_sha256"],
        )


@dataclass(frozen=True, slots=True)
class AblationResultRegistryV1:
    """Hash-sealed Table-IV registry, explicitly outside the primary gate."""

    experiment_plan_sha256: str
    dataset_manifest_sha256: str
    split_sha256: str
    cohort_sha256: str
    frame_contract_sha256: str
    detection_cache_sha256: str
    evaluator_contract_sha256: str
    network_trace_manifest_sha256: str
    wire_accounting_config_sha256: str
    heldout_c9_trace_receipt_sha256: str
    sequence_ids: tuple[str, ...]
    c9_trace_ids: tuple[str, ...]
    config_bindings: tuple[AblationConfigBindingV1, ...]
    config_manifest_sha256: str
    paper_scope: str
    contributes_to_primary_sota_gate: bool
    failure_semantics: str
    cells: tuple[AblationResultCellV1, ...]

    def __post_init__(self) -> None:
        for name in (
            "experiment_plan_sha256",
            "dataset_manifest_sha256",
            "split_sha256",
            "cohort_sha256",
            "frame_contract_sha256",
            "detection_cache_sha256",
            "evaluator_contract_sha256",
            "network_trace_manifest_sha256",
            "wire_accounting_config_sha256",
            "heldout_c9_trace_receipt_sha256",
            "config_manifest_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(
            self,
            "sequence_ids",
            _sorted_unique_identifiers(self.sequence_ids, "sequence_ids"),
        )
        object.__setattr__(
            self,
            "c9_trace_ids",
            _sorted_unique_identifiers(self.c9_trace_ids, "c9_trace_ids"),
        )
        if not isinstance(self.config_bindings, (list, tuple)):
            raise AblationRegistryError("config_bindings must be an array")
        bindings = tuple(self.config_bindings)
        if not bindings or not all(
            isinstance(item, AblationConfigBindingV1) for item in bindings
        ):
            raise AblationRegistryError(
                "config_bindings must contain AblationConfigBindingV1"
            )
        variant_ids = tuple(item.variant_id for item in bindings)
        if variant_ids != tuple(sorted(set(variant_ids))):
            raise AblationRegistryError("config_bindings must be unique and sorted")
        if variant_ids != ABLATION_VARIANT_IDS_V1:
            raise AblationRegistryError("config_bindings must cover all ten variants")
        if len({item.config_sha256 for item in bindings}) != len(bindings):
            raise AblationRegistryError(
                "all ten ablation variants require distinct config SHA-256 values"
            )
        if ablation_config_manifest_sha256_v1(bindings) != self.config_manifest_sha256:
            raise AblationRegistryError("config manifest SHA-256 mismatch")
        object.__setattr__(self, "config_bindings", bindings)
        if self.paper_scope != ABLATION_PAPER_SCOPE_V1:
            raise AblationRegistryError("paper_scope must be supplement-table-iv-only")
        if self.contributes_to_primary_sota_gate is not False:
            raise AblationRegistryError(
                "ablation registry cannot contribute to the primary SOTA gate"
            )
        if self.failure_semantics != ABLATION_FAILURE_SEMANTICS_V1:
            raise AblationRegistryError("failure semantics must be zero scoring")
        if not isinstance(self.cells, (list, tuple)) or not self.cells:
            raise AblationRegistryError("cells must be a non-empty array")
        cells = tuple(self.cells)
        if not all(isinstance(item, AblationResultCellV1) for item in cells):
            raise AblationRegistryError("cells must contain AblationResultCellV1")
        keys = tuple(item.key for item in cells)
        if keys != tuple(sorted(set(keys), key=_sortable_key)):
            raise AblationRegistryError("cells must be unique and sorted by key")
        attempt_ids = tuple(item.attempt_id for item in cells)
        if len(attempt_ids) != len(set(attempt_ids)):
            raise AblationRegistryError(
                "each ablation cell must bind one unique physical attempt"
            )
        configs = {item.variant_id: item.config_sha256 for item in bindings}
        for cell in cells:
            if cell.experiment_plan_sha256 != self.experiment_plan_sha256:
                raise AblationRegistryError("cell has stale plan binding")
            if cell.variant_config_sha256 != configs[cell.variant_id]:
                raise AblationRegistryError("cell has wrong variant config binding")
            if cell.detection_cache_sha256 != self.detection_cache_sha256:
                raise AblationRegistryError("cell has wrong detector-cache binding")
            if cell.evaluator_contract_sha256 != self.evaluator_contract_sha256:
                raise AblationRegistryError("cell has wrong evaluator binding")
        object.__setattr__(self, "cells", cells)

    def payload(self) -> dict[str, object]:
        return {
            "c9_trace_ids": list(self.c9_trace_ids),
            "cells": [item.to_primitive() for item in self.cells],
            "cohort_sha256": self.cohort_sha256,
            "config_bindings": [item.to_primitive() for item in self.config_bindings],
            "config_manifest_sha256": self.config_manifest_sha256,
            "contributes_to_primary_sota_gate": self.contributes_to_primary_sota_gate,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "detection_cache_sha256": self.detection_cache_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "experiment_plan_sha256": self.experiment_plan_sha256,
            "failure_semantics": self.failure_semantics,
            "frame_contract_sha256": self.frame_contract_sha256,
            "heldout_c9_trace_receipt_sha256": self.heldout_c9_trace_receipt_sha256,
            "kind": "ablation_result_registry_v1",
            "network_trace_manifest_sha256": self.network_trace_manifest_sha256,
            "paper_scope": self.paper_scope,
            "schema_version": 1,
            "sequence_ids": list(self.sequence_ids),
            "split_sha256": self.split_sha256,
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
    def from_mapping(cls, value: object) -> "AblationResultRegistryV1":
        fields = frozenset(
            {
                "c9_trace_ids",
                "cells",
                "cohort_sha256",
                "config_bindings",
                "config_manifest_sha256",
                "content_sha256",
                "contributes_to_primary_sota_gate",
                "dataset_manifest_sha256",
                "detection_cache_sha256",
                "evaluator_contract_sha256",
                "experiment_plan_sha256",
                "failure_semantics",
                "frame_contract_sha256",
                "heldout_c9_trace_receipt_sha256",
                "kind",
                "network_trace_manifest_sha256",
                "paper_scope",
                "schema_version",
                "sequence_ids",
                "split_sha256",
                "wire_accounting_config_sha256",
            }
        )
        item = _strict_fields(value, fields, cls.__name__)
        if item["kind"] != "ablation_result_registry_v1" or item[
            "schema_version"
        ] != 1:
            raise AblationRegistryError("unsupported ablation registry header")
        observed = _sha256(item["content_sha256"], "content_sha256")
        payload = {key: item[key] for key in fields if key != "content_sha256"}
        expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if observed != expected:
            raise AblationRegistryError("registry content SHA-256 mismatch")
        raw_bindings = item["config_bindings"]
        raw_cells = item["cells"]
        if not isinstance(raw_bindings, list) or not isinstance(raw_cells, list):
            raise AblationRegistryError("bindings and cells must be arrays")
        return cls(
            experiment_plan_sha256=item["experiment_plan_sha256"],
            dataset_manifest_sha256=item["dataset_manifest_sha256"],
            split_sha256=item["split_sha256"],
            cohort_sha256=item["cohort_sha256"],
            frame_contract_sha256=item["frame_contract_sha256"],
            detection_cache_sha256=item["detection_cache_sha256"],
            evaluator_contract_sha256=item["evaluator_contract_sha256"],
            network_trace_manifest_sha256=item["network_trace_manifest_sha256"],
            wire_accounting_config_sha256=item["wire_accounting_config_sha256"],
            heldout_c9_trace_receipt_sha256=item[
                "heldout_c9_trace_receipt_sha256"
            ],
            sequence_ids=tuple(item["sequence_ids"]),  # type: ignore[arg-type]
            c9_trace_ids=tuple(item["c9_trace_ids"]),  # type: ignore[arg-type]
            config_bindings=tuple(
                AblationConfigBindingV1.from_mapping(binding)
                for binding in raw_bindings
            ),
            config_manifest_sha256=item["config_manifest_sha256"],
            paper_scope=item["paper_scope"],
            contributes_to_primary_sota_gate=item[
                "contributes_to_primary_sota_gate"
            ],
            failure_semantics=item["failure_semantics"],
            cells=tuple(AblationResultCellV1.from_mapping(cell) for cell in raw_cells),
        )


def _validate_registry_against_cohort(
    registry: AblationResultRegistryV1,
    cohort: AblationCohortV1,
) -> None:
    if registry.sequence_ids != cohort.sequence_ids:
        raise AblationRegistryError("registry sequence cohort is not exact")
    if registry.c9_trace_ids != cohort.c9_trace_ids:
        raise AblationRegistryError("registry C9 trace cohort is not exact")
    sentinel = object()
    for index, (expected, observed) in enumerate(
        zip_longest(
            _iter_expected_ablation_specs_for_cohort(cohort),
            registry.cells,
            fillvalue=sentinel,
        )
    ):
        if expected is sentinel:
            raise AblationRegistryError(f"unexpected extra cell at index {index}")
        if observed is sentinel:
            raise AblationRegistryError(f"missing expected cell at index {index}")
        assert isinstance(expected, ExpectedAblationRunSpecV1)
        assert isinstance(observed, AblationResultCellV1)
        if observed.key != expected.key:
            raise AblationRegistryError(
                f"cell key mismatch at index {index}: {observed.key!r} != {expected.key!r}"
            )


def _validate_measured_trace_receipt_against_plan_v1(
    plan: ExperimentPlanV1,
    receipt: MeasuredTraceReceiptV1,
) -> None:
    if not isinstance(receipt, MeasuredTraceReceiptV1):
        raise TypeError("measured_trace_receipt must be MeasuredTraceReceiptV1")
    if (
        receipt.content_sha256 != plan.heldout_c9_trace_receipt_sha256
        or receipt.held_out_trace_ids != plan.c9_trace_ids
    ):
        raise AblationRegistryError(
            "measured C9 trace receipt content or held-out IDs do not match plan"
        )


def _validated_condition_input_manifest_inventory_v1(
    manifests: Mapping[str, ConditionInputManifestV1],
) -> dict[str, ConditionInputManifestV1]:
    if not isinstance(manifests, Mapping):
        raise TypeError("condition_input_manifests must be a mapping")
    result: dict[str, ConditionInputManifestV1] = {}
    for raw_digest, manifest in manifests.items():
        digest = _sha256(raw_digest, "condition input manifest inventory key")
        if not isinstance(manifest, ConditionInputManifestV1):
            raise TypeError(
                "condition_input_manifests values must be ConditionInputManifestV1"
            )
        if manifest.content_sha256 != digest:
            raise AblationRegistryError(
                "condition input manifest inventory key does not match content"
            )
        result[digest] = manifest
    return result


def _validate_external_input_bindings_v1(
    registry: AblationResultRegistryV1,
    receipt: MeasuredTraceReceiptV1,
    condition_input_manifests: Mapping[str, ConditionInputManifestV1],
) -> None:
    """Bind each disturbed-input run and measured replay to raw evidence."""

    if not isinstance(receipt, MeasuredTraceReceiptV1):
        raise TypeError("measured_trace_receipt must be MeasuredTraceReceiptV1")
    inventory = _validated_condition_input_manifest_inventory_v1(
        condition_input_manifests
    )
    held_out_packets = dict(receipt.held_out_packet_metadata_sha256s)
    referenced_manifests: set[str] = set()
    for cell in registry.cells:
        if cell.condition_id == "C9":
            assert cell.c9_trace_id is not None
            if (
                cell.channel_outcome_trace_sha256
                != held_out_packets.get(cell.c9_trace_id)
            ):
                raise AblationRegistryError(
                    "C9 cell is not bound to held-out packet metadata"
                )
        if cell.condition_id not in DISTURBED_INPUT_CONDITION_IDS_V1:
            continue
        digest = cell.disturbed_input_manifest_sha256
        assert digest is not None
        if digest in referenced_manifests:
            raise AblationRegistryError(
                "condition input manifest cannot be reused across semantic cells"
            )
        manifest = inventory.get(digest)
        if manifest is None:
            raise AblationRegistryError(
                "C6-C8 cell references a missing condition input manifest"
            )
        referenced_manifests.add(digest)
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
            cell.variant_config_sha256,
            cell.detection_cache_sha256,
        )
        if observed != expected:
            raise AblationRegistryError(
                "condition input manifest does not match its physical run"
            )
    if frozenset(inventory) != frozenset(referenced_manifests):
        raise AblationRegistryError(
            "condition input manifest inventory has missing or extra artifacts"
        )


def validate_ablation_result_registry_v1(
    registry: AblationResultRegistryV1,
    plan: ExperimentPlanV1,
    *,
    measured_trace_receipt: MeasuredTraceReceiptV1,
    condition_input_manifests: Mapping[str, ConditionInputManifestV1],
) -> None:
    """Validate all formal Table-IV cells and their immutable plan bindings."""

    if not isinstance(registry, AblationResultRegistryV1):
        raise TypeError("registry must be AblationResultRegistryV1")
    if not isinstance(plan, ExperimentPlanV1):
        raise TypeError("plan must be ExperimentPlanV1")
    _validate_measured_trace_receipt_against_plan_v1(
        plan, measured_trace_receipt
    )
    cohort = formal_ablation_cohort_v1(plan)
    expected_headers = {
        "experiment_plan_sha256": plan.content_sha256,
        "dataset_manifest_sha256": plan.dataset_manifest_sha256,
        "split_sha256": plan.split_sha256,
        "cohort_sha256": plan.primary_cohort_sha256,
        "frame_contract_sha256": plan.primary_frame_contract_sha256,
        "detection_cache_sha256": plan.detector_cache_sha256,
        "evaluator_contract_sha256": plan.evaluator_contract_sha256,
        "network_trace_manifest_sha256": plan.network_trace_manifest_sha256,
        "wire_accounting_config_sha256": plan.wire_accounting_config_sha256,
        "heldout_c9_trace_receipt_sha256": plan.heldout_c9_trace_receipt_sha256,
    }
    for name, expected in expected_headers.items():
        if getattr(registry, name) != expected:
            raise AblationRegistryError(f"registry has stale {name}")
    full_config = next(
        item.config_sha256
        for item in registry.config_bindings
        if item.variant_id == FULL_EVENTTRACK_VARIANT_ID_V1
    )
    if full_config != plan.run_config_sha256(
        "primary", "eventtrack-v2x", "marginal_voi"
    ):
        raise AblationRegistryError(
            "full_eventtrack config must equal the frozen primary method config"
        )
    _validate_registry_against_cohort(registry, cohort)
    if len(registry.cells) != FORMAL_ABLATION_CELL_COUNT_V1:
        raise AblationRegistryError("formal ablation cell count mismatch")
    _validate_external_input_bindings_v1(
        registry, measured_trace_receipt, condition_input_manifests
    )


def decode_ablation_result_registry(data: bytes) -> AblationResultRegistryV1:
    """Decode canonical bytes and reject duplicate JSON keys."""

    if type(data) is not bytes:
        raise TypeError("registry data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise AblationRegistryError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                AblationRegistryError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AblationRegistryError("invalid AblationResultRegistryV1 JSON") from exc
    registry = AblationResultRegistryV1.from_mapping(value)
    if registry.canonical_bytes() != data:
        raise AblationRegistryError(
            "AblationResultRegistryV1 JSON is not canonical"
        )
    return registry


__all__ = [
    "ABLATION_BUDGET_BYTES_PER_SECOND_V1",
    "ABLATION_FAILURE_SEMANTICS_V1",
    "ABLATION_METRICS_ARTIFACT_ROLE_V1",
    "ABLATION_PAPER_SCOPE_V1",
    "ABLATION_STATUS_FAILURE_V1",
    "ABLATION_STATUS_SUCCESS_V1",
    "ABLATION_VARIANT_IDS_V1",
    "AblationConfigBindingV1",
    "AblationRegistryError",
    "AblationResultCellV1",
    "AblationResultRegistryV1",
    "DISTURBED_INPUT_CONDITION_IDS_V1",
    "ExpectedAblationRunSpecV1",
    "FORMAL_ABLATION_CELL_COUNT_V1",
    "FULL_EVENTTRACK_VARIANT_ID_V1",
    "REQUIRED_ABLATION_METRIC_IDS_V1",
    "ablation_config_manifest_sha256_v1",
    "decode_ablation_result_registry",
    "formal_ablation_cohort_v1",
    "iter_expected_paper_ablation_specs_v1",
    "validate_ablation_result_registry_v1",
]
