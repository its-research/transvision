"""Fail-closed raw validation registry for qualification decisions.

The qualification document is a compact decision record.  This module is its
raw, independently hash-bound source: it requires every preregistered
V2X-Seq-SPD train-split OOF cell before deriving tracker and scheduler summaries.
Algorithm failures remain in the matrix and contribute zero to aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable, Iterable, Iterator, Mapping

import numpy as np

from .experiment import (
    CONFIRMATORY_BYTE_BUDGETS_V1,
    CONFIRMATORY_NETWORK_SEEDS_V1,
    CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1,
    CONFIRMATORY_QUALIFIED_SCHEDULER_BASELINE_IDS_V1,
    CONFIRMATORY_TRAINING_SEEDS_V1,
    ExperimentPlanV1,
)
from .development_split import (
    DEVELOPMENT_FOLD_COUNT_V1,
    DEVELOPMENT_SEQUENCE_COUNT_V1,
    DEVELOPMENT_SPLIT_NAME_V1,
    DevelopmentSplitManifestV1,
)
from .measured_trace import MeasuredTraceReceiptV1, TraceRole
from .network_disturbance import ConditionInputManifestV1
from .wire import canonical_json_bytes


VALIDATION_DATASET_ID_V1 = "v2x-seq-spd"
VALIDATION_SPLIT_NAME_V1 = DEVELOPMENT_SPLIT_NAME_V1
VALIDATION_FREQUENCY_HZ_V1 = 10.0
VALIDATION_CLASS_NAMES_V1 = ("car",)
VALIDATION_CONDITION_IDS_V1 = tuple(f"C{index}" for index in range(10))
VALIDATION_SYNTHETIC_PRESSURE_CONDITION_IDS_V1 = tuple(
    f"C{index}" for index in range(1, 9)
)
VALIDATION_ROBUST_CONDITION_IDS_V1 = tuple(f"C{index}" for index in range(1, 10))
VALIDATION_PRIMARY_BUDGET_V1 = 64_000
VALIDATION_BASELINE_FAMILY_V1 = "baseline"
VALIDATION_SCHEDULER_FAMILY_V1 = "scheduler"
VALIDATION_STATUS_SUCCESS_V1 = "success"
VALIDATION_STATUS_FAILURE_V1 = "algorithm_failure"
VALIDATION_RESULT_STATUSES_V1 = (
    VALIDATION_STATUS_FAILURE_V1,
    VALIDATION_STATUS_SUCCESS_V1,
)
VALIDATION_METRICS_ARTIFACT_ROLE_V1 = "per_sequence_metrics"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


class ValidationRegistryError(ValueError):
    """Raised when validation evidence is malformed, incomplete, or stale."""


def _strict_fields(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise ValidationRegistryError(f"{name} must be a string-keyed object")
    if frozenset(value) != expected:
        missing = sorted(expected - frozenset(value))
        unknown = sorted(frozenset(value) - expected)
        raise ValidationRegistryError(
            f"{name} fields do not match schema; missing={missing}, unknown={unknown}"
        )
    return value


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise ValidationRegistryError(f"{name} must be a canonical identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValidationRegistryError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValidationRegistryError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationRegistryError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValidationRegistryError(f"{name} must be finite and non-negative")
    return result


def _metric_values(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValidationRegistryError("metric_values must be an array")
    result: list[tuple[str, float]] = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValidationRegistryError(
                f"metric_values[{index}] must be a two-value array"
            )
        metric_id = _identifier(item[0], f"metric_values[{index}].metric_id")
        metric_value = _finite_nonnegative(item[1], f"metric_values[{index}].value")
        if metric_value > 1.0:
            raise ValidationRegistryError("tracking metrics must be in [0, 1]")
        result.append((metric_id, metric_value))
    if [item[0] for item in result] != sorted({item[0] for item in result}):
        raise ValidationRegistryError(
            "metric_values must have unique, sorted metric identifiers"
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ExpectedValidationSpecV1:
    """One required semantic validation cell, without its outcome."""

    qualification_family: str
    method_id: str
    scheduler_id: str
    sequence_id: str
    development_fold_id: int
    condition_id: str
    training_seed: int
    network_seed: int | None
    c9_trace_id: str | None
    budget_bytes_per_second: int
    metric_ids: tuple[str, ...]

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.qualification_family,
            self.method_id,
            self.scheduler_id,
            self.sequence_id,
            self.development_fold_id,
            self.condition_id,
            self.training_seed,
            self.network_seed,
            self.c9_trace_id,
            self.budget_bytes_per_second,
        )


@dataclass(frozen=True, slots=True)
class ValidationResultCellV1:
    """One raw validation outcome and all provenance needed to audit it."""

    qualification_family: str
    method_id: str
    scheduler_id: str
    sequence_id: str
    development_fold_id: int
    condition_id: str
    training_seed: int
    network_seed: int | None
    c9_trace_id: str | None
    budget_bytes_per_second: int
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
        for name in (
            "qualification_family",
            "method_id",
            "scheduler_id",
            "sequence_id",
            "condition_id",
            "status",
            "metrics_artifact_role",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.qualification_family not in {
            VALIDATION_BASELINE_FAMILY_V1,
            VALIDATION_SCHEDULER_FAMILY_V1,
        }:
            raise ValidationRegistryError("unknown qualification_family")
        if (
            type(self.development_fold_id) is not int
            or not 0 <= self.development_fold_id < DEVELOPMENT_FOLD_COUNT_V1
        ):
            raise ValidationRegistryError(
                "development_fold_id must be an integer in [0, 5)"
            )
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise ValidationRegistryError("training_seed must be non-negative")
        if self.condition_id not in VALIDATION_CONDITION_IDS_V1:
            raise ValidationRegistryError("condition_id is not registered")
        if self.c9_trace_id is not None:
            object.__setattr__(
                self, "c9_trace_id", _identifier(self.c9_trace_id, "c9_trace_id")
            )
        if self.condition_id == "C0":
            if self.network_seed is not None or self.c9_trace_id is not None:
                raise ValidationRegistryError(
                    "C0 cells must not carry network_seed or c9_trace_id"
                )
        elif self.condition_id == "C9":
            if self.network_seed is not None or self.c9_trace_id is None:
                raise ValidationRegistryError(
                    "C9 cells require c9_trace_id and no network_seed"
                )
        elif (
            type(self.network_seed) is not int
            or self.network_seed < 0
            or self.c9_trace_id is not None
        ):
            raise ValidationRegistryError(
                "C1-C8 cells require a non-negative network_seed and no c9_trace_id"
            )
        if self.condition_id in {"C6", "C7", "C8"}:
            if self.condition_input_manifest_sha256 is None:
                raise ValidationRegistryError(
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
            raise ValidationRegistryError(
                "C0-C5 and C9 cells must not carry condition input manifest"
            )
        budget = _positive_int(self.budget_bytes_per_second, "budget_bytes_per_second")
        if type(self.on_wire_bytes_total) is not int or self.on_wire_bytes_total < 0:
            raise ValidationRegistryError(
                "on_wire_bytes_total must be a non-negative integer"
            )
        duration = _finite_nonnegative(
            self.wire_measurement_duration_seconds,
            "wire_measurement_duration_seconds",
        )
        if duration <= 0.0:
            raise ValidationRegistryError(
                "wire_measurement_duration_seconds must be positive"
            )
        on_wire = _finite_nonnegative(
            self.on_wire_bytes_per_second, "on_wire_bytes_per_second"
        )
        derived = float(self.on_wire_bytes_total) / duration
        if not np.isclose(on_wire, derived, rtol=1e-12, atol=1e-12):
            raise ValidationRegistryError(
                "on-wire BPS does not equal bytes divided by duration"
            )
        if on_wire > float(budget) + 1e-9:
            raise ValidationRegistryError("on-wire rate exceeds cell budget")
        object.__setattr__(self, "metric_values", _metric_values(self.metric_values))
        object.__setattr__(self, "wire_measurement_duration_seconds", duration)
        object.__setattr__(self, "on_wire_bytes_per_second", on_wire)
        if self.status not in VALIDATION_RESULT_STATUSES_V1:
            raise ValidationRegistryError("status is not a registered terminal status")
        if self.status == VALIDATION_STATUS_SUCCESS_V1:
            if not self.metric_values or self.failure_code is not None:
                raise ValidationRegistryError(
                    "successful cells require metrics and no failure_code"
                )
        else:
            if self.metric_values or self.failure_code is None:
                raise ValidationRegistryError(
                    "failed cells require empty metrics and failure_code"
                )
            object.__setattr__(
                self, "failure_code", _identifier(self.failure_code, "failure_code")
            )
        for name in (
            "checkpoint_sha256",
            "evidence_bundle_sha256",
            "metrics_artifact_sha256",
            "channel_outcome_trace_sha256",
            "network_trace_sha256",
            "wire_ledger_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.metrics_artifact_role != VALIDATION_METRICS_ARTIFACT_ROLE_V1:
            raise ValidationRegistryError(
                "metrics_artifact_role must be per_sequence_metrics"
            )

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.qualification_family,
            self.method_id,
            self.scheduler_id,
            self.sequence_id,
            self.development_fold_id,
            self.condition_id,
            self.training_seed,
            self.network_seed,
            self.c9_trace_id,
            self.budget_bytes_per_second,
        )

    def metric_or_failure_zero(self, metric_id: str) -> float:
        if self.status == VALIDATION_STATUS_FAILURE_V1:
            return 0.0
        try:
            return dict(self.metric_values)[metric_id]
        except KeyError as exc:  # pragma: no cover - exact matrix checked first.
            raise ValidationRegistryError(f"missing metric {metric_id}") from exc

    def to_primitive(self) -> dict[str, object]:
        return {
            "budget_bytes_per_second": self.budget_bytes_per_second,
            "channel_outcome_trace_sha256": self.channel_outcome_trace_sha256,
            "condition_id": self.condition_id,
            "development_fold_id": self.development_fold_id,
            "condition_input_manifest_sha256": (self.condition_input_manifest_sha256),
            "c9_trace_id": self.c9_trace_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "evidence_bundle_sha256": self.evidence_bundle_sha256,
            "failure_code": self.failure_code,
            "method_id": self.method_id,
            "metric_values": [list(item) for item in self.metric_values],
            "metrics_artifact_role": self.metrics_artifact_role,
            "metrics_artifact_sha256": self.metrics_artifact_sha256,
            "network_seed": self.network_seed,
            "network_trace_sha256": self.network_trace_sha256,
            "on_wire_bytes_per_second": self.on_wire_bytes_per_second,
            "on_wire_bytes_total": self.on_wire_bytes_total,
            "qualification_family": self.qualification_family,
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
    def from_mapping(cls, value: object) -> "ValidationResultCellV1":
        fields = frozenset(
            {
                "budget_bytes_per_second",
                "channel_outcome_trace_sha256",
                "condition_id",
                "development_fold_id",
                "condition_input_manifest_sha256",
                "c9_trace_id",
                "checkpoint_sha256",
                "evidence_bundle_sha256",
                "failure_code",
                "method_id",
                "metric_values",
                "metrics_artifact_role",
                "metrics_artifact_sha256",
                "network_seed",
                "network_trace_sha256",
                "on_wire_bytes_per_second",
                "on_wire_bytes_total",
                "qualification_family",
                "scheduler_id",
                "sequence_id",
                "status",
                "training_seed",
                "wire_ledger_sha256",
                "wire_measurement_duration_seconds",
            }
        )
        item = _strict_fields(value, fields, cls.__name__)
        metrics = item["metric_values"]
        if not isinstance(metrics, list):
            raise ValidationRegistryError("metric_values must be an array")
        return cls(
            qualification_family=item["qualification_family"],  # type: ignore[arg-type]
            method_id=item["method_id"],  # type: ignore[arg-type]
            scheduler_id=item["scheduler_id"],  # type: ignore[arg-type]
            sequence_id=item["sequence_id"],  # type: ignore[arg-type]
            condition_id=item["condition_id"],  # type: ignore[arg-type]
            development_fold_id=item["development_fold_id"],  # type: ignore[arg-type]
            condition_input_manifest_sha256=item[  # type: ignore[arg-type]
                "condition_input_manifest_sha256"
            ],
            training_seed=item["training_seed"],  # type: ignore[arg-type]
            network_seed=item["network_seed"],  # type: ignore[arg-type]
            c9_trace_id=item["c9_trace_id"],  # type: ignore[arg-type]
            checkpoint_sha256=item["checkpoint_sha256"],  # type: ignore[arg-type]
            budget_bytes_per_second=item["budget_bytes_per_second"],  # type: ignore[arg-type]
            metric_values=tuple(tuple(metric) for metric in metrics),
            on_wire_bytes_total=item["on_wire_bytes_total"],  # type: ignore[arg-type]
            wire_measurement_duration_seconds=item[  # type: ignore[arg-type]
                "wire_measurement_duration_seconds"
            ],
            on_wire_bytes_per_second=item["on_wire_bytes_per_second"],  # type: ignore[arg-type]
            status=item["status"],  # type: ignore[arg-type]
            failure_code=item["failure_code"],  # type: ignore[arg-type]
            evidence_bundle_sha256=item["evidence_bundle_sha256"],  # type: ignore[arg-type]
            metrics_artifact_role=item["metrics_artifact_role"],  # type: ignore[arg-type]
            metrics_artifact_sha256=item["metrics_artifact_sha256"],  # type: ignore[arg-type]
            channel_outcome_trace_sha256=item["channel_outcome_trace_sha256"],  # type: ignore[arg-type]
            network_trace_sha256=item["network_trace_sha256"],  # type: ignore[arg-type]
            wire_ledger_sha256=item["wire_ledger_sha256"],  # type: ignore[arg-type]
        )


def _sortable_key(key: tuple[object, ...]) -> tuple[str, ...]:
    return tuple(
        "" if item is None else f"{type(item).__name__}:{item}" for item in key
    )


def iter_expected_validation_specs_v1(
    development_fold_manifest: DevelopmentSplitManifestV1,
    c9_setup_trace_ids: Iterable[str],
) -> Iterator[ExpectedValidationSpecV1]:
    """Yield the exact pooled OOF development matrix used for qualification."""

    if not isinstance(development_fold_manifest, DevelopmentSplitManifestV1):
        raise TypeError(
            "development_fold_manifest must be DevelopmentSplitManifestV1"
        )
    sequences = development_fold_manifest.sequence_ids
    setup_trace_ids = tuple(c9_setup_trace_ids)
    if len(setup_trace_ids) != 10 or setup_trace_ids != tuple(
        sorted(set(setup_trace_ids))
    ):
        raise ValidationRegistryError(
            "c9_setup_trace_ids must contain exactly 10 unique sorted identifiers"
        )
    setup_trace_ids = tuple(
        _identifier(item, "c9_setup_trace_id") for item in setup_trace_ids
    )
    for method_id in CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1:
        for sequence_id in sequences:
            development_fold_id = development_fold_manifest.fold_for_sequence(
                sequence_id
            )
            for training_seed in CONFIRMATORY_TRAINING_SEEDS_V1:
                yield ExpectedValidationSpecV1(
                    qualification_family=VALIDATION_BASELINE_FAMILY_V1,
                    method_id=method_id,
                    scheduler_id="qualification",
                    sequence_id=sequence_id,
                    development_fold_id=development_fold_id,
                    condition_id="C0",
                    training_seed=training_seed,
                    network_seed=None,
                    c9_trace_id=None,
                    budget_bytes_per_second=VALIDATION_PRIMARY_BUDGET_V1,
                    metric_ids=("AMOTA", "HOTA"),
                )
            for condition_id in VALIDATION_SYNTHETIC_PRESSURE_CONDITION_IDS_V1:
                for training_seed in CONFIRMATORY_TRAINING_SEEDS_V1:
                    for network_seed in CONFIRMATORY_NETWORK_SEEDS_V1:
                        yield ExpectedValidationSpecV1(
                            qualification_family=VALIDATION_BASELINE_FAMILY_V1,
                            method_id=method_id,
                            scheduler_id="qualification",
                            sequence_id=sequence_id,
                            development_fold_id=development_fold_id,
                            condition_id=condition_id,
                            training_seed=training_seed,
                            network_seed=network_seed,
                            c9_trace_id=None,
                            budget_bytes_per_second=VALIDATION_PRIMARY_BUDGET_V1,
                            metric_ids=("AssA",),
                        )
            for training_seed in CONFIRMATORY_TRAINING_SEEDS_V1:
                for c9_trace_id in setup_trace_ids:
                    yield ExpectedValidationSpecV1(
                        qualification_family=VALIDATION_BASELINE_FAMILY_V1,
                        method_id=method_id,
                        scheduler_id="qualification",
                        sequence_id=sequence_id,
                        development_fold_id=development_fold_id,
                        condition_id="C9",
                        training_seed=training_seed,
                        network_seed=None,
                        c9_trace_id=c9_trace_id,
                        budget_bytes_per_second=VALIDATION_PRIMARY_BUDGET_V1,
                        metric_ids=("AssA",),
                    )
    for scheduler_id in CONFIRMATORY_QUALIFIED_SCHEDULER_BASELINE_IDS_V1:
        for sequence_id in sequences:
            development_fold_id = development_fold_manifest.fold_for_sequence(
                sequence_id
            )
            for budget in CONFIRMATORY_BYTE_BUDGETS_V1:
                for condition_id in VALIDATION_SYNTHETIC_PRESSURE_CONDITION_IDS_V1:
                    for training_seed in CONFIRMATORY_TRAINING_SEEDS_V1:
                        for network_seed in CONFIRMATORY_NETWORK_SEEDS_V1:
                            yield ExpectedValidationSpecV1(
                                qualification_family=VALIDATION_SCHEDULER_FAMILY_V1,
                                method_id="eventtrack-v2x",
                                scheduler_id=scheduler_id,
                                sequence_id=sequence_id,
                                development_fold_id=development_fold_id,
                                condition_id=condition_id,
                                training_seed=training_seed,
                                network_seed=network_seed,
                                c9_trace_id=None,
                                budget_bytes_per_second=budget,
                                metric_ids=("AssA",),
                            )
                for training_seed in CONFIRMATORY_TRAINING_SEEDS_V1:
                    for c9_trace_id in setup_trace_ids:
                        yield ExpectedValidationSpecV1(
                            qualification_family=VALIDATION_SCHEDULER_FAMILY_V1,
                            method_id="eventtrack-v2x",
                            scheduler_id=scheduler_id,
                            sequence_id=sequence_id,
                            development_fold_id=development_fold_id,
                            condition_id="C9",
                            training_seed=training_seed,
                            network_seed=None,
                            c9_trace_id=c9_trace_id,
                            budget_bytes_per_second=budget,
                            metric_ids=("AssA",),
                        )


@dataclass(frozen=True, slots=True)
class BaselineValidationSummaryV1:
    method_id: str
    robust_assa_at_64k: float
    clean_amota: float
    clean_hota: float
    failure_count: int


@dataclass(frozen=True, slots=True)
class SchedulerValidationSummaryV1:
    scheduler_id: str
    robust_assa_at_64k: float
    actual_byte_auc: float
    failure_count: int


@dataclass(frozen=True, slots=True)
class ValidationSummariesV1:
    baselines: tuple[BaselineValidationSummaryV1, ...]
    schedulers: tuple[SchedulerValidationSummaryV1, ...]


@dataclass(frozen=True, slots=True)
class ValidationResultRegistryV1:
    """Hash-sealed exact raw matrix for pooled train-split OOF qualification."""

    source_tree_sha256: str
    dataset_id: str
    dataset_manifest_sha256: str
    split_name: str
    split_sha256: str
    cohort_sha256: str
    sequence_ids: tuple[str, ...]
    development_fold_manifest: DevelopmentSplitManifestV1
    detection_cache_sha256: str
    detection_frame_contract_sha256: str
    network_trace_manifest_sha256: str
    measured_trace_receipt_sha256: str
    c9_setup_trace_ids: tuple[str, ...]
    wire_accounting_config_sha256: str
    evaluator_contract_sha256: str
    frequency_hz: float
    class_names: tuple[str, ...]
    cells: tuple[ValidationResultCellV1, ...]

    def __post_init__(self) -> None:
        for name in (
            "source_tree_sha256",
            "dataset_manifest_sha256",
            "split_sha256",
            "cohort_sha256",
            "detection_cache_sha256",
            "detection_frame_contract_sha256",
            "network_trace_manifest_sha256",
            "measured_trace_receipt_sha256",
            "wire_accounting_config_sha256",
            "evaluator_contract_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(
            self, "dataset_id", _identifier(self.dataset_id, "dataset_id")
        )
        object.__setattr__(
            self, "split_name", _identifier(self.split_name, "split_name")
        )
        if self.dataset_id != VALIDATION_DATASET_ID_V1:
            raise ValidationRegistryError("dataset_id must be v2x-seq-spd")
        if self.split_name != VALIDATION_SPLIT_NAME_V1:
            raise ValidationRegistryError("development split_name must be train")
        if not isinstance(self.sequence_ids, (list, tuple)):
            raise ValidationRegistryError("sequence_ids must be an array")
        sequences = tuple(
            _identifier(item, "sequence_id") for item in self.sequence_ids
        )
        if (
            len(sequences) != DEVELOPMENT_SEQUENCE_COUNT_V1
            or sequences != tuple(sorted(set(sequences)))
        ):
            raise ValidationRegistryError(
                "sequence_ids must contain exactly 46 unique sorted identifiers"
            )
        object.__setattr__(self, "sequence_ids", sequences)
        if not isinstance(
            self.development_fold_manifest, DevelopmentSplitManifestV1
        ):
            raise TypeError(
                "development_fold_manifest must be DevelopmentSplitManifestV1"
            )
        manifest_identity = (
            self.development_fold_manifest.dataset_id,
            self.development_fold_manifest.split_name,
            self.development_fold_manifest.split_sha256,
            self.development_fold_manifest.sequence_ids,
        )
        registry_identity = (
            self.dataset_id,
            self.split_name,
            self.split_sha256,
            sequences,
        )
        if manifest_identity != registry_identity:
            raise ValidationRegistryError(
                "development fold manifest does not match the train cohort"
            )
        if not isinstance(self.c9_setup_trace_ids, (list, tuple)):
            raise ValidationRegistryError("c9_setup_trace_ids must be an array")
        setup_trace_ids = tuple(
            _identifier(item, "c9_setup_trace_id") for item in self.c9_setup_trace_ids
        )
        if len(setup_trace_ids) != 10 or setup_trace_ids != tuple(
            sorted(set(setup_trace_ids))
        ):
            raise ValidationRegistryError(
                "c9_setup_trace_ids must contain exactly 10 unique sorted identifiers"
            )
        object.__setattr__(self, "c9_setup_trace_ids", setup_trace_ids)
        if float(self.frequency_hz) != VALIDATION_FREQUENCY_HZ_V1:
            raise ValidationRegistryError("frequency_hz must equal 10")
        object.__setattr__(self, "frequency_hz", VALIDATION_FREQUENCY_HZ_V1)
        if not isinstance(self.class_names, (list, tuple)):
            raise ValidationRegistryError("class_names must be an array")
        classes = tuple(_identifier(item, "class_name") for item in self.class_names)
        if classes != VALIDATION_CLASS_NAMES_V1:
            raise ValidationRegistryError("class_names must equal ('car',)")
        object.__setattr__(self, "class_names", classes)
        if not isinstance(self.cells, (list, tuple)) or not self.cells:
            raise ValidationRegistryError("cells must be a non-empty array")
        if not all(isinstance(item, ValidationResultCellV1) for item in self.cells):
            raise ValidationRegistryError("cells must contain ValidationResultCellV1")
        cells = tuple(self.cells)
        for cell in cells:
            if cell.development_fold_id != self.development_fold_manifest.fold_for_sequence(
                cell.sequence_id
            ):
                raise ValidationRegistryError(
                    "validation cell development fold does not match manifest"
                )
        keys = [item.key for item in cells]
        if len(keys) != len(set(keys)):
            raise ValidationRegistryError("duplicate validation cell key")
        if keys != sorted(keys, key=_sortable_key):
            raise ValidationRegistryError(
                "validation cells must be sorted by semantic key"
            )
        object.__setattr__(self, "cells", cells)

    def payload(self) -> dict[str, object]:
        return {
            "cells": [cell.to_primitive() for cell in self.cells],
            "class_names": list(self.class_names),
            "c9_setup_trace_ids": list(self.c9_setup_trace_ids),
            "cohort_sha256": self.cohort_sha256,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "detection_cache_sha256": self.detection_cache_sha256,
            "detection_frame_contract_sha256": self.detection_frame_contract_sha256,
            "development_fold_manifest": (
                self.development_fold_manifest.sealed_document()
            ),
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "frequency_hz": self.frequency_hz,
            "kind": "validation_result_registry_v1",
            "network_trace_manifest_sha256": self.network_trace_manifest_sha256,
            "measured_trace_receipt_sha256": self.measured_trace_receipt_sha256,
            "schema_version": 1,
            "sequence_ids": list(self.sequence_ids),
            "source_tree_sha256": self.source_tree_sha256,
            "split_name": self.split_name,
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
    def from_mapping(cls, value: object) -> "ValidationResultRegistryV1":
        fields = frozenset(
            {
                "cells",
                "class_names",
                "c9_setup_trace_ids",
                "cohort_sha256",
                "content_sha256",
                "dataset_id",
                "dataset_manifest_sha256",
                "detection_cache_sha256",
                "detection_frame_contract_sha256",
                "development_fold_manifest",
                "evaluator_contract_sha256",
                "frequency_hz",
                "kind",
                "network_trace_manifest_sha256",
                "measured_trace_receipt_sha256",
                "schema_version",
                "sequence_ids",
                "source_tree_sha256",
                "split_name",
                "split_sha256",
                "wire_accounting_config_sha256",
            }
        )
        item = _strict_fields(value, fields, cls.__name__)
        if (
            item["kind"] != "validation_result_registry_v1"
            or item["schema_version"] != 1
        ):
            raise ValidationRegistryError("unsupported validation registry header")
        observed = _sha256(item["content_sha256"], "content_sha256")
        payload = {key: item[key] for key in fields if key != "content_sha256"}
        if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != observed:
            raise ValidationRegistryError(
                "validation registry content SHA-256 mismatch"
            )
        cells = item["cells"]
        if not isinstance(cells, list):
            raise ValidationRegistryError("cells must be an array")
        return cls(
            source_tree_sha256=item["source_tree_sha256"],  # type: ignore[arg-type]
            dataset_id=item["dataset_id"],  # type: ignore[arg-type]
            dataset_manifest_sha256=item["dataset_manifest_sha256"],  # type: ignore[arg-type]
            split_name=item["split_name"],  # type: ignore[arg-type]
            split_sha256=item["split_sha256"],  # type: ignore[arg-type]
            cohort_sha256=item["cohort_sha256"],  # type: ignore[arg-type]
            sequence_ids=tuple(item["sequence_ids"]),  # type: ignore[arg-type]
            development_fold_manifest=DevelopmentSplitManifestV1.from_mapping(
                item["development_fold_manifest"]
            ),
            detection_cache_sha256=item["detection_cache_sha256"],  # type: ignore[arg-type]
            detection_frame_contract_sha256=item["detection_frame_contract_sha256"],  # type: ignore[arg-type]
            network_trace_manifest_sha256=item["network_trace_manifest_sha256"],  # type: ignore[arg-type]
            measured_trace_receipt_sha256=item["measured_trace_receipt_sha256"],  # type: ignore[arg-type]
            c9_setup_trace_ids=tuple(item["c9_setup_trace_ids"]),  # type: ignore[arg-type]
            wire_accounting_config_sha256=item["wire_accounting_config_sha256"],  # type: ignore[arg-type]
            evaluator_contract_sha256=item["evaluator_contract_sha256"],  # type: ignore[arg-type]
            frequency_hz=item["frequency_hz"],  # type: ignore[arg-type]
            class_names=tuple(item["class_names"]),  # type: ignore[arg-type]
            cells=tuple(ValidationResultCellV1.from_mapping(cell) for cell in cells),
        )


def validate_validation_result_registry_v1(
    registry: ValidationResultRegistryV1,
) -> None:
    """Require exact matrix/metrics and reuse of each exogenous channel trace."""

    if not isinstance(registry, ValidationResultRegistryV1):
        raise TypeError("registry must be ValidationResultRegistryV1")
    expected = {
        spec.key: spec.metric_ids
        for spec in iter_expected_validation_specs_v1(
            registry.development_fold_manifest, registry.c9_setup_trace_ids
        )
    }
    actual = {cell.key: cell for cell in registry.cells}
    missing = sorted(set(expected) - set(actual), key=_sortable_key)
    extra = sorted(set(actual) - set(expected), key=_sortable_key)
    if missing:
        raise ValidationRegistryError(
            f"missing validation cells ({len(missing)}): {missing[:1]}"
        )
    if extra:
        raise ValidationRegistryError(
            f"unexpected validation cells ({len(extra)}): {extra[:1]}"
        )
    for key, metric_ids in expected.items():
        cell = actual[key]
        observed = tuple(metric_id for metric_id, _ in cell.metric_values)
        if cell.status == VALIDATION_STATUS_SUCCESS_V1 and observed != metric_ids:
            raise ValidationRegistryError(f"metric set does not match cell {key}")
        if cell.status == VALIDATION_STATUS_FAILURE_V1 and observed:
            raise ValidationRegistryError("failed cells cannot carry metrics")

    trace_hashes: dict[tuple[str, int, str, int | None, str | None], str] = {}
    evidence_metrics: dict[str, str] = {}
    checkpoint_by_oof_model: dict[tuple[str, str, str, int, int], str] = {}
    oof_model_by_checkpoint: dict[str, tuple[str, str, str, int, int]] = {}
    for cell in registry.cells:
        trace_key = (
            cell.sequence_id,
            cell.development_fold_id,
            cell.condition_id,
            cell.network_seed,
            cell.c9_trace_id,
        )
        previous = trace_hashes.setdefault(trace_key, cell.channel_outcome_trace_sha256)
        if previous != cell.channel_outcome_trace_sha256:
            raise ValidationRegistryError(
                f"exogenous channel trace not reused for {trace_key}"
            )
        previous_metric = evidence_metrics.setdefault(
            cell.evidence_bundle_sha256, cell.metrics_artifact_sha256
        )
        if previous_metric != cell.metrics_artifact_sha256:
            raise ValidationRegistryError(
                "one evidence bundle digest refers to multiple metrics artifacts"
            )
        oof_model = (
            cell.qualification_family,
            cell.method_id,
            cell.scheduler_id,
            cell.development_fold_id,
            cell.training_seed,
        )
        previous_checkpoint = checkpoint_by_oof_model.setdefault(
            oof_model, cell.checkpoint_sha256
        )
        if previous_checkpoint != cell.checkpoint_sha256:
            raise ValidationRegistryError(
                "one OOF model identity refers to multiple checkpoints"
            )
        previous_model = oof_model_by_checkpoint.setdefault(
            cell.checkpoint_sha256, oof_model
        )
        if previous_model != oof_model:
            raise ValidationRegistryError(
                "one checkpoint is reused across distinct OOF model identities"
            )


def _validation_run_config_sha256_v1(
    cell: ValidationResultCellV1,
    plan: ExperimentPlanV1,
) -> str:
    if cell.qualification_family == VALIDATION_BASELINE_FAMILY_V1:
        return plan.run_config_sha256(
            "primary",
            cell.method_id,
            plan.strongest_scheduler_baseline_id,
        )
    return plan.run_config_sha256(
        "primary",
        "eventtrack-v2x",
        cell.scheduler_id,
    )


def validation_physical_run_id_v1(
    cell: ValidationResultCellV1,
    *,
    run_config_sha256: str,
    detection_cache_sha256: str,
) -> str:
    """Derive one stable pre-output identifier for a validation physical run."""

    if not isinstance(cell, ValidationResultCellV1):
        raise TypeError("cell must be ValidationResultCellV1")
    payload = {
        "budget_bytes_per_second": cell.budget_bytes_per_second,
        "condition_id": cell.condition_id,
        "development_fold_id": cell.development_fold_id,
        "c9_trace_id": cell.c9_trace_id,
        "checkpoint_sha256": cell.checkpoint_sha256,
        "detection_cache_sha256": _sha256(
            detection_cache_sha256, "detection_cache_sha256"
        ),
        "kind": "validation_physical_run_id_v1",
        "method_id": cell.method_id,
        "network_seed": cell.network_seed,
        "network_trace_sha256": cell.network_trace_sha256,
        "qualification_family": cell.qualification_family,
        "run_config_sha256": _sha256(run_config_sha256, "run_config_sha256"),
        "scheduler_id": cell.scheduler_id,
        "schema_version": 1,
        "sequence_id": cell.sequence_id,
        "training_seed": cell.training_seed,
    }
    return "validation-" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _validated_condition_input_manifest_inventory_v1(
    manifests: Mapping[str, ConditionInputManifestV1],
) -> dict[str, ConditionInputManifestV1]:
    if not isinstance(manifests, Mapping):
        raise TypeError("condition_input_manifests must be a mapping")
    inventory: dict[str, ConditionInputManifestV1] = {}
    for raw_digest, manifest in manifests.items():
        digest = _sha256(raw_digest, "condition input manifest inventory key")
        if not isinstance(manifest, ConditionInputManifestV1):
            raise TypeError(
                "condition_input_manifests values must be ConditionInputManifestV1"
            )
        if manifest.content_sha256 != digest:
            raise ValidationRegistryError(
                "condition input manifest inventory key does not match content"
            )
        inventory[digest] = manifest
    return inventory


def validate_validation_result_external_inputs_v1(
    registry: ValidationResultRegistryV1,
    plan: ExperimentPlanV1,
    *,
    measured_trace_receipt: MeasuredTraceReceiptV1,
    condition_input_manifests: Mapping[str, ConditionInputManifestV1],
) -> None:
    """Open and bind every C6--C9 input used for validation qualification."""

    if not isinstance(plan, ExperimentPlanV1):
        raise TypeError("plan must be ExperimentPlanV1")
    validate_validation_result_registry_v1(registry)
    if not isinstance(measured_trace_receipt, MeasuredTraceReceiptV1):
        raise TypeError("measured_trace_receipt must be MeasuredTraceReceiptV1")
    if (
        measured_trace_receipt.content_sha256 != plan.heldout_c9_trace_receipt_sha256
        or measured_trace_receipt.held_out_trace_ids != plan.c9_trace_ids
    ):
        raise ValidationRegistryError(
            "measured C9 trace receipt content or held-out IDs do not match plan"
        )
    if (
        registry.measured_trace_receipt_sha256 != measured_trace_receipt.content_sha256
        or registry.c9_setup_trace_ids != measured_trace_receipt.setup_trace_ids
    ):
        raise ValidationRegistryError(
            "validation C9 receipt content or setup trace IDs do not match"
        )

    setup_packets = {
        segment.trace_id: segment.packet_metadata_sha256
        for segment in measured_trace_receipt.segments
        if segment.role is TraceRole.SETUP
    }
    inventory = _validated_condition_input_manifest_inventory_v1(
        condition_input_manifests
    )
    referenced_manifests: set[str] = set()
    run_id_to_physical_key: dict[str, tuple[object, ...]] = {}
    manifest_to_physical_key: dict[str, tuple[object, ...]] = {}
    for cell in registry.cells:
        if cell.condition_id == "C9":
            assert cell.c9_trace_id is not None
            if cell.channel_outcome_trace_sha256 != setup_packets.get(cell.c9_trace_id):
                raise ValidationRegistryError(
                    "validation C9 cell is not bound to setup packet metadata"
                )
        if cell.condition_id not in {"C6", "C7", "C8"}:
            continue
        digest = cell.condition_input_manifest_sha256
        assert digest is not None
        manifest = inventory.get(digest)
        if manifest is None:
            raise ValidationRegistryError(
                "C6-C8 validation cell references a missing condition input manifest"
            )
        physical_key = cell.key
        previous_physical_key = manifest_to_physical_key.setdefault(
            digest, physical_key
        )
        if previous_physical_key != physical_key:
            raise ValidationRegistryError(
                "one condition input manifest is aliased across validation runs"
            )
        expected_config_sha256 = _validation_run_config_sha256_v1(cell, plan)
        expected_run_id = validation_physical_run_id_v1(
            cell,
            run_config_sha256=expected_config_sha256,
            detection_cache_sha256=registry.detection_cache_sha256,
        )
        previous_physical_key = run_id_to_physical_key.setdefault(
            manifest.run_id, physical_key
        )
        if previous_physical_key != physical_key:
            raise ValidationRegistryError(
                "one condition input run_id is aliased across validation runs"
            )
        observed = (
            manifest.run_id,
            manifest.condition_id.value,
            manifest.network_trace_sha256,
            manifest.run_config_sha256,
            manifest.detection_cache_sha256,
        )
        expected = (
            expected_run_id,
            cell.condition_id,
            cell.network_trace_sha256,
            expected_config_sha256,
            registry.detection_cache_sha256,
        )
        if observed != expected:
            raise ValidationRegistryError(
                "condition input manifest does not match its validation physical run"
            )
        referenced_manifests.add(digest)
    if frozenset(inventory) != frozenset(referenced_manifests):
        raise ValidationRegistryError(
            "condition input manifest inventory has missing or extra artifacts"
        )


def _mean_metric(cells: Iterable[ValidationResultCellV1], metric_id: str) -> float:
    values = [cell.metric_or_failure_zero(metric_id) for cell in cells]
    if not values:
        raise ValidationRegistryError("cannot aggregate an empty cell set")
    return float(np.mean(values))


def _robust_condition_macro_mean(
    cells: Iterable[ValidationResultCellV1],
    value_getter: Callable[[ValidationResultCellV1], float],
) -> float:
    """Average C1--C9 equally after averaging each condition's 10 replicates.

    The exact matrix gives every condition 10 exogenous replicates per training
    seed (network seeds for C1--C8, setup traces for C9).  Explicit condition
    macro-averaging preserves the preregistered estimand even if this helper is
    later reused on a differently represented but still validated registry.
    """

    selected = tuple(cells)
    grouped = {
        condition_id: [
            float(value_getter(cell))
            for cell in selected
            if cell.condition_id == condition_id
        ]
        for condition_id in VALIDATION_ROBUST_CONDITION_IDS_V1
    }
    if any(not values for values in grouped.values()):
        raise ValidationRegistryError(
            "robust aggregation requires non-empty C1-C9 condition groups"
        )
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _robust_metric_macro_mean(
    cells: Iterable[ValidationResultCellV1], metric_id: str
) -> float:
    return _robust_condition_macro_mean(
        cells, lambda cell: cell.metric_or_failure_zero(metric_id)
    )


def _normalized_actual_byte_aucs_common_support(
    curves: Mapping[str, Iterable[tuple[float, float]]],
) -> dict[str, float]:
    """Integrate every scheduler on one shared actual-BPS support.

    Normalizing each curve on its own observed range makes AUC values
    incomparable when schedulers consume different wire rates.  The validation
    winner must therefore be derived from the intersection of all eligible
    schedulers' observed ranges, using the union of their interior knots.
    """

    if not curves:
        raise ValidationRegistryError("scheduler AUC requires at least one curve")
    fronts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scheduler_id, points in sorted(curves.items()):
        best_by_x: dict[float, float] = {}
        for x, y in points:
            best_by_x[x] = max(best_by_x.get(x, 0.0), y)
        ordered = sorted(best_by_x.items())
        if len(ordered) < 2:
            raise ValidationRegistryError(
                f"scheduler curve has no actual-BPS width: {scheduler_id}"
            )
        xs = np.asarray([item[0] for item in ordered], dtype=float)
        ys = np.maximum.accumulate(
            np.asarray([item[1] for item in ordered], dtype=float)
        )
        fronts[scheduler_id] = (xs, ys)

    lower = max(float(xs[0]) for xs, _ in fronts.values())
    upper = min(float(xs[-1]) for xs, _ in fronts.values())
    if upper <= lower:
        raise ValidationRegistryError(
            "scheduler curves have no common actual-BPS width"
        )
    grid = np.unique(
        np.concatenate(
            [
                np.asarray([lower, upper]),
                *[xs[(xs > lower) & (xs < upper)] for xs, _ in fronts.values()],
            ]
        )
    )
    width = upper - lower
    return {
        scheduler_id: float(np.trapezoid(np.interp(grid, xs, ys), grid) / width)
        for scheduler_id, (xs, ys) in fronts.items()
    }


def derive_validation_summaries_v1(
    registry: ValidationResultRegistryV1,
) -> ValidationSummariesV1:
    """Derive qualification inputs uniquely from complete raw evidence."""

    validate_validation_result_registry_v1(registry)
    baselines: list[BaselineValidationSummaryV1] = []
    for method_id in CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1:
        selected = [
            cell
            for cell in registry.cells
            if cell.qualification_family == VALIDATION_BASELINE_FAMILY_V1
            and cell.method_id == method_id
        ]
        robust = [
            cell
            for cell in selected
            if cell.condition_id in VALIDATION_ROBUST_CONDITION_IDS_V1
        ]
        clean = [cell for cell in selected if cell.condition_id == "C0"]
        baselines.append(
            BaselineValidationSummaryV1(
                method_id=method_id,
                robust_assa_at_64k=_robust_metric_macro_mean(robust, "AssA"),
                clean_amota=_mean_metric(clean, "AMOTA"),
                clean_hota=_mean_metric(clean, "HOTA"),
                failure_count=sum(
                    cell.status == VALIDATION_STATUS_FAILURE_V1 for cell in selected
                ),
            )
        )

    scheduler_cells = {
        scheduler_id: [
            cell
            for cell in registry.cells
            if cell.qualification_family == VALIDATION_SCHEDULER_FAMILY_V1
            and cell.scheduler_id == scheduler_id
        ]
        for scheduler_id in CONFIRMATORY_QUALIFIED_SCHEDULER_BASELINE_IDS_V1
    }
    scheduler_aucs_by_sequence: dict[str, list[float]] = {
        scheduler_id: []
        for scheduler_id in CONFIRMATORY_QUALIFIED_SCHEDULER_BASELINE_IDS_V1
    }
    for sequence_id in registry.sequence_ids:
        curves: dict[str, list[tuple[float, float]]] = {}
        for scheduler_id, selected in scheduler_cells.items():
            points: list[tuple[float, float]] = []
            for budget in CONFIRMATORY_BYTE_BUDGETS_V1:
                budget_cells = [
                    cell
                    for cell in selected
                    if cell.sequence_id == sequence_id
                    and cell.budget_bytes_per_second == budget
                ]
                points.append(
                    (
                        _robust_condition_macro_mean(
                            budget_cells,
                            lambda cell: cell.on_wire_bytes_per_second,
                        ),
                        _robust_metric_macro_mean(budget_cells, "AssA"),
                    )
                )
            curves[scheduler_id] = points
        common_aucs = _normalized_actual_byte_aucs_common_support(curves)
        for scheduler_id, auc in common_aucs.items():
            scheduler_aucs_by_sequence[scheduler_id].append(auc)

    schedulers: list[SchedulerValidationSummaryV1] = []
    for scheduler_id in CONFIRMATORY_QUALIFIED_SCHEDULER_BASELINE_IDS_V1:
        selected = scheduler_cells[scheduler_id]
        at_64k = [
            cell
            for cell in selected
            if cell.budget_bytes_per_second == VALIDATION_PRIMARY_BUDGET_V1
        ]
        schedulers.append(
            SchedulerValidationSummaryV1(
                scheduler_id=scheduler_id,
                robust_assa_at_64k=_robust_metric_macro_mean(at_64k, "AssA"),
                actual_byte_auc=float(
                    np.mean(scheduler_aucs_by_sequence[scheduler_id])
                ),
                failure_count=sum(
                    cell.status == VALIDATION_STATUS_FAILURE_V1 for cell in selected
                ),
            )
        )
    return ValidationSummariesV1(tuple(baselines), tuple(schedulers))


def decode_validation_result_registry(data: bytes) -> ValidationResultRegistryV1:
    """Decode canonical bytes and reject duplicate keys/non-finite JSON."""

    if type(data) is not bytes:
        raise TypeError("registry data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValidationRegistryError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValidationRegistryError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationRegistryError(
            "invalid ValidationResultRegistryV1 JSON"
        ) from exc
    registry = ValidationResultRegistryV1.from_mapping(value)
    if registry.canonical_bytes() != data:
        raise ValidationRegistryError(
            "ValidationResultRegistryV1 JSON is not canonical"
        )
    validate_validation_result_registry_v1(registry)
    return registry


__all__ = [
    "BaselineValidationSummaryV1",
    "ExpectedValidationSpecV1",
    "SchedulerValidationSummaryV1",
    "VALIDATION_BASELINE_FAMILY_V1",
    "VALIDATION_METRICS_ARTIFACT_ROLE_V1",
    "VALIDATION_SCHEDULER_FAMILY_V1",
    "VALIDATION_STATUS_FAILURE_V1",
    "VALIDATION_STATUS_SUCCESS_V1",
    "ValidationRegistryError",
    "ValidationResultCellV1",
    "ValidationResultRegistryV1",
    "ValidationSummariesV1",
    "decode_validation_result_registry",
    "derive_validation_summaries_v1",
    "iter_expected_validation_specs_v1",
    "validate_validation_result_external_inputs_v1",
    "validate_validation_result_registry_v1",
    "validation_physical_run_id_v1",
]
