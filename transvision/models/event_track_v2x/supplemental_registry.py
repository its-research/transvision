"""Fail-closed raw evidence registry for the paper's supplemental matrices.

This registry is intentionally separate from :mod:`results_registry`.  The
existing ``RunResultRegistryV1`` is the compact, preregistered subset consumed
by the primary SOTA publication gate.  ``PaperSupplementalRegistryV1`` records
only the substantially larger scheduler, full-send diagnostic, synchronous
oracle, and raw-sensor tables without silently widening that primary claim. It
does not claim to implement the separate 2 Hz, full-class, or ablation matrices.

The formal validator expands expected keys as a stream.  It therefore checks
all 208,404 semantic cells without first materialising another copy of the
matrix.  ``SupplementalCohortV1`` and the cohort iterator are also useful for
small contract tests, but only ``validate_paper_supplemental_registry_v1`` is a
publication-valid entry point; it requires the frozen 21-sequence V2X-Seq-SPD
confirmatory validation plan, official seeds, five budgets, and twenty held-out
C9 traces.
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
    CONFIRMATORY_BYTE_BUDGETS_V1,
    CONFIRMATORY_NETWORK_SEEDS_V1,
    CONFIRMATORY_TRAINING_SEEDS_V1,
    ExperimentPlanV1,
)
from .measured_trace import MeasuredTraceReceiptV1
from .network_disturbance import ConditionInputManifestV1
from .wire import canonical_json_bytes


SCHEDULER_FRONTIER_SCENARIO_V1 = "scheduler_frontier"
FULL_SEND_DIAGNOSTIC_SCENARIO_V1 = "full_send_diagnostic"
SYNCHRONOUS_ORACLE_SCENARIO_V1 = "synchronous_oracle_upper_bound"
RAW_SENSOR_SECONDARY_SCENARIO_V1 = "raw_sensor_secondary_table"
SUPPLEMENTAL_SCENARIO_IDS_V1 = tuple(
    sorted(
        (
            SCHEDULER_FRONTIER_SCENARIO_V1,
            FULL_SEND_DIAGNOSTIC_SCENARIO_V1,
            SYNCHRONOUS_ORACLE_SCENARIO_V1,
            RAW_SENSOR_SECONDARY_SCENARIO_V1,
        )
    )
)

CAMERA_DETECTOR_INPUT_V1 = "camera-detector-locked"
RAW_SENSOR_DETECTOR_INPUT_V1 = "raw-sensor"
CAPPED_SCHEDULER_IDS_V1 = (
    "confidence_top_k",
    "fifo_aoi",
    "marginal_voi",
    "periodic",
    "random",
)
RAW_SENSOR_METHOD_IDS_V1 = tuple(
    sorted(
        (
            "no-fusion",
            "late-fusion",
            "ff-tracking",
            "univ2x-track",
            "cooptrack-r50",
            "sparsecoop-r50",
            "eventtrack-v2x",
        )
    )
)
RAW_SENSOR_SCHEDULER_ID_V1 = "raw_sensor_pipeline"
SYNCHRONOUS_ORACLE_METHOD_ID_V1 = "synchronous-oracle"
SYNCHRONOUS_ORACLE_SCHEDULER_ID_V1 = "synchronous_oracle"

SUPPLEMENTAL_STATUS_SUCCESS_V1 = "success"
SUPPLEMENTAL_STATUS_FAILURE_V1 = "algorithm_failure"
SUPPLEMENTAL_STATUSES_V1 = (
    SUPPLEMENTAL_STATUS_FAILURE_V1,
    SUPPLEMENTAL_STATUS_SUCCESS_V1,
)
SUPPLEMENTAL_FAILURE_SEMANTICS_V1 = "all-required-metrics-zero"
SUPPLEMENTAL_METRICS_ARTIFACT_ROLE_V1 = "per_sequence_metrics"

TRACKING_METRIC_IDS_V1 = tuple(
    sorted(
        (
            "AMOTA",
            "AMOTP",
            "MOTA",
            "HOTA",
            "AssA",
            "DetA",
            "IDF1",
            "IDS",
            "Frag",
            "MT",
            "ML",
        )
    )
)
COMMUNICATION_METRIC_IDS_V1 = tuple(
    sorted(
        (
            "application_bps",
            "on_wire_bps",
            "on_time_useful_bps",
            "expired_rate",
            "retransmission_rate",
            "aoi_p50_ms",
            "aoi_p95_ms",
        )
    )
)
SYSTEM_METRIC_IDS_V1 = tuple(
    sorted(
        (
            "fps",
            "frame_latency_p50_ms",
            "frame_latency_p95_ms",
            "frame_latency_p99_ms",
            "peak_gpu_memory_bytes",
            "peak_cpu_memory_bytes",
            "scheduler_overhead_ms",
        )
    )
)
TRAFFIC_APPLICATION_METRIC_IDS_V1 = tuple(
    sorted(
        (
            "useful_track_coverage_deadline_100ms",
            "correct_id_recovery_time_seconds",
            "continuous_miss_duration_30m_seconds",
        )
    )
)
CALIBRATION_METRIC_IDS_V1 = tuple(
    sorted(
        (
            "state_nll",
            "existence_brier",
            "existence_ece",
            "nis",
            "nees",
            "coverage_50",
            "coverage_90",
            "coverage_95",
        )
    )
)
REQUIRED_SUPPLEMENTAL_METRIC_IDS_V1 = tuple(
    sorted(
        TRACKING_METRIC_IDS_V1
        + CALIBRATION_METRIC_IDS_V1
        + COMMUNICATION_METRIC_IDS_V1
        + SYSTEM_METRIC_IDS_V1
        + TRAFFIC_APPLICATION_METRIC_IDS_V1
    )
)
_UNIT_INTERVAL_METRIC_IDS_V1 = frozenset(
    {
        "AMOTA",
        "AssA",
        "DetA",
        "HOTA",
        "IDF1",
        "coverage_50",
        "coverage_90",
        "coverage_95",
        "existence_brier",
        "existence_ece",
        "expired_rate",
        "retransmission_rate",
        "useful_track_coverage_deadline_100ms",
    }
)
_NONNEGATIVE_METRIC_IDS_V1 = frozenset(
    {
        "AMOTP",
        "Frag",
        "IDS",
        "ML",
        "MT",
        "aoi_p50_ms",
        "aoi_p95_ms",
        "application_bps",
        "continuous_miss_duration_30m_seconds",
        "correct_id_recovery_time_seconds",
        "fps",
        "frame_latency_p50_ms",
        "frame_latency_p95_ms",
        "frame_latency_p99_ms",
        "nees",
        "nis",
        "on_time_useful_bps",
        "on_wire_bps",
        "peak_cpu_memory_bytes",
        "peak_gpu_memory_bytes",
        "scheduler_overhead_ms",
    }
)
_SPECIAL_METRIC_IDS_V1 = frozenset({"MOTA", "state_nll"})

FORMAL_PRIMARY_SEQUENCE_COUNT_V1 = 21
FORMAL_C9_TRACE_COUNT_V1 = 20
FORMAL_SUPPLEMENTAL_CELL_COUNT_V1 = 208_404
DISTURBED_INPUT_CONDITION_IDS_V1 = ("C6", "C7", "C8")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


class SupplementalRegistryError(ValueError):
    """Raised when supplemental result evidence is not exact and canonical."""


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise SupplementalRegistryError(f"{name} must be a canonical identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise SupplementalRegistryError(f"{name} must be a lowercase SHA-256")
    return value


def _sorted_unique_identifiers(values: object, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise SupplementalRegistryError(f"{name} must be an array")
    result = tuple(_identifier(value, f"{name} item") for value in values)
    if not result or result != tuple(sorted(set(result))):
        raise SupplementalRegistryError(f"{name} must be non-empty, unique, sorted")
    return result


def _sorted_unique_positive_ints(values: object, name: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise SupplementalRegistryError(f"{name} must be an array")
    result = tuple(values)
    if (
        not result
        or any(type(value) is not int or value <= 0 for value in result)
        or result != tuple(sorted(set(result)))
    ):
        raise SupplementalRegistryError(
            f"{name} must contain positive, unique, sorted integers"
        )
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SupplementalRegistryError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise SupplementalRegistryError(f"{name} must be finite and non-negative")
    return result


def _strict_fields(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise SupplementalRegistryError(f"{name} must be a string-keyed object")
    if frozenset(value) != expected:
        raise SupplementalRegistryError(f"{name} has missing or unknown fields")
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
        raise SupplementalRegistryError("metric_values must be an array")
    result: list[tuple[str, float]] = []
    for index, item in enumerate(values):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise SupplementalRegistryError(
                f"metric_values[{index}] must be a two-value array"
            )
        metric_id = _identifier(item[0], f"metric_values[{index}].metric_id")
        if isinstance(item[1], bool) or not isinstance(item[1], (int, float)):
            raise SupplementalRegistryError(
                f"metric_values[{index}].value must be numeric"
            )
        metric_value = float(item[1])
        if not np.isfinite(metric_value):
            raise SupplementalRegistryError(
                f"metric_values[{index}].value must be finite"
            )
        result.append((metric_id, metric_value))
    if tuple(item[0] for item in result) != tuple(
        sorted({item[0] for item in result})
    ):
        raise SupplementalRegistryError(
            "metric_values must have unique, sorted metric identifiers"
        )
    return tuple(result)


def _metric_domain_violation_v1(
    values: tuple[tuple[str, float], ...],
) -> str | None:
    """Return the first registered semantic-domain violation, if any."""

    registered = (
        _UNIT_INTERVAL_METRIC_IDS_V1
        | _NONNEGATIVE_METRIC_IDS_V1
        | _SPECIAL_METRIC_IDS_V1
    )
    for metric_id, value in values:
        if metric_id not in registered:
            return f"{metric_id} has no registered semantic domain"
        if metric_id in _UNIT_INTERVAL_METRIC_IDS_V1 and not 0.0 <= value <= 1.0:
            return f"{metric_id} must be in [0, 1]"
        if metric_id in _NONNEGATIVE_METRIC_IDS_V1 and value < 0.0:
            return f"{metric_id} must be non-negative"
        if metric_id == "MOTA" and value > 1.0:
            return "MOTA must be at most 1"
    return None


@dataclass(frozen=True, slots=True)
class SupplementalCohortV1:
    """Matrix dimensions; reduced cohorts are never publication-valid."""

    sequence_ids: tuple[str, ...]
    training_seeds: tuple[int, ...]
    network_seeds: tuple[int, ...]
    c9_trace_ids: tuple[str, ...]
    byte_budgets_per_second: tuple[int, ...]

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
        object.__setattr__(
            self,
            "byte_budgets_per_second",
            _sorted_unique_positive_ints(
                self.byte_budgets_per_second, "byte_budgets_per_second"
            ),
        )


@dataclass(frozen=True, slots=True)
class SupplementalConfigBindingV1:
    """Immutable config artifact for one detector/method/scheduler pipeline."""

    detector_input_id: str
    method_id: str
    scheduler_id: str
    config_sha256: str

    _FIELDS = frozenset(
        {"config_sha256", "detector_input_id", "method_id", "scheduler_id"}
    )

    def __post_init__(self) -> None:
        for name in ("detector_input_id", "method_id", "scheduler_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self, "config_sha256", _sha256(self.config_sha256, "config_sha256")
        )

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.detector_input_id, self.method_id, self.scheduler_id)

    def to_primitive(self) -> dict[str, str]:
        return {
            "config_sha256": self.config_sha256,
            "detector_input_id": self.detector_input_id,
            "method_id": self.method_id,
            "scheduler_id": self.scheduler_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "SupplementalConfigBindingV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        return cls(
            detector_input_id=item["detector_input_id"],
            method_id=item["method_id"],
            scheduler_id=item["scheduler_id"],
            config_sha256=item["config_sha256"],
        )


def required_supplemental_config_keys_v1() -> tuple[tuple[str, str, str], ...]:
    """Return exactly the fourteen supplemental config namespaces."""

    keys = {
        *(
            (CAMERA_DETECTOR_INPUT_V1, "eventtrack-v2x", scheduler_id)
            for scheduler_id in CAPPED_SCHEDULER_IDS_V1 + ("full_send",)
        ),
        (
            CAMERA_DETECTOR_INPUT_V1,
            SYNCHRONOUS_ORACLE_METHOD_ID_V1,
            SYNCHRONOUS_ORACLE_SCHEDULER_ID_V1,
        ),
        *(
            (RAW_SENSOR_DETECTOR_INPUT_V1, method_id, RAW_SENSOR_SCHEDULER_ID_V1)
            for method_id in RAW_SENSOR_METHOD_IDS_V1
        ),
    }
    return tuple(sorted(keys))


def supplemental_config_manifest_sha256_v1(
    bindings: Iterable[SupplementalConfigBindingV1],
) -> str:
    """Hash the canonical ordered config-binding manifest."""

    values = tuple(bindings)
    if not all(isinstance(item, SupplementalConfigBindingV1) for item in values):
        raise TypeError("bindings must contain SupplementalConfigBindingV1")
    keys = tuple(item.key for item in values)
    if keys != tuple(sorted(set(keys))):
        raise SupplementalRegistryError("config bindings must be unique and sorted")
    return hashlib.sha256(
        canonical_json_bytes([item.to_primitive() for item in values])
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ExpectedSupplementalRunSpecV1:
    """One exact supplemental semantic cell without outcome evidence."""

    scenario_id: str
    detector_input_id: str
    method_id: str
    scheduler_id: str
    sequence_id: str
    condition_id: str
    training_seed: int
    network_seed: int | None
    c9_trace_id: str | None
    nominal_budget_bytes_per_second: int | None
    table_ranking_eligible: bool

    @property
    def condition_input_manifest_required(self) -> bool:
        """Whether the physical run must bind realized C6--C8 inputs."""

        return self.condition_id in DISTURBED_INPUT_CONDITION_IDS_V1

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.scenario_id,
            self.detector_input_id,
            self.method_id,
            self.scheduler_id,
            self.sequence_id,
            self.condition_id,
            self.training_seed,
            self.network_seed,
            self.c9_trace_id,
            self.nominal_budget_bytes_per_second,
        )


def _replicates(
    cohort: SupplementalCohortV1, condition_id: str
) -> Iterator[tuple[int | None, str | None]]:
    if condition_id == "C0":
        yield None, None
    elif condition_id == "C9":
        yield from ((None, trace_id) for trace_id in cohort.c9_trace_ids)
    else:
        yield from ((seed, None) for seed in cohort.network_seeds)


def _iter_expected_supplemental_specs_for_cohort(
    cohort: SupplementalCohortV1,
) -> Iterator[ExpectedSupplementalRunSpecV1]:
    """Stream an exact matrix for structural tests or canaries.

    Reduced cohorts produced here are not acceptable to the formal paper
    validator.
    """

    if not isinstance(cohort, SupplementalCohortV1):
        raise TypeError("cohort must be SupplementalCohortV1")

    for scenario_id in SUPPLEMENTAL_SCENARIO_IDS_V1:
        if scenario_id == FULL_SEND_DIAGNOSTIC_SCENARIO_V1:
            for sequence_id in cohort.sequence_ids:
                for condition_id in tuple(f"C{index}" for index in range(1, 10)):
                    for training_seed in cohort.training_seeds:
                        for network_seed, trace_id in _replicates(cohort, condition_id):
                            yield ExpectedSupplementalRunSpecV1(
                                scenario_id=scenario_id,
                                detector_input_id=CAMERA_DETECTOR_INPUT_V1,
                                method_id="eventtrack-v2x",
                                scheduler_id="full_send",
                                sequence_id=sequence_id,
                                condition_id=condition_id,
                                training_seed=training_seed,
                                network_seed=network_seed,
                                c9_trace_id=trace_id,
                                nominal_budget_bytes_per_second=None,
                                table_ranking_eligible=False,
                            )
        elif scenario_id == RAW_SENSOR_SECONDARY_SCENARIO_V1:
            for method_id in RAW_SENSOR_METHOD_IDS_V1:
                for sequence_id in cohort.sequence_ids:
                    for condition_id in tuple(f"C{index}" for index in range(10)):
                        for training_seed in cohort.training_seeds:
                            for network_seed, trace_id in _replicates(
                                cohort, condition_id
                            ):
                                yield ExpectedSupplementalRunSpecV1(
                                    scenario_id=scenario_id,
                                    detector_input_id=RAW_SENSOR_DETECTOR_INPUT_V1,
                                    method_id=method_id,
                                    scheduler_id=RAW_SENSOR_SCHEDULER_ID_V1,
                                    sequence_id=sequence_id,
                                    condition_id=condition_id,
                                    training_seed=training_seed,
                                    network_seed=network_seed,
                                    c9_trace_id=trace_id,
                                    nominal_budget_bytes_per_second=64_000,
                                    table_ranking_eligible=True,
                                )
        elif scenario_id == SCHEDULER_FRONTIER_SCENARIO_V1:
            for scheduler_id in CAPPED_SCHEDULER_IDS_V1:
                for sequence_id in cohort.sequence_ids:
                    for condition_id in tuple(f"C{index}" for index in range(1, 10)):
                        for training_seed in cohort.training_seeds:
                            for network_seed, trace_id in _replicates(
                                cohort, condition_id
                            ):
                                for budget in cohort.byte_budgets_per_second:
                                    yield ExpectedSupplementalRunSpecV1(
                                        scenario_id=scenario_id,
                                        detector_input_id=CAMERA_DETECTOR_INPUT_V1,
                                        method_id="eventtrack-v2x",
                                        scheduler_id=scheduler_id,
                                        sequence_id=sequence_id,
                                        condition_id=condition_id,
                                        training_seed=training_seed,
                                        network_seed=network_seed,
                                        c9_trace_id=trace_id,
                                        nominal_budget_bytes_per_second=budget,
                                        table_ranking_eligible=True,
                                    )
        else:
            assert scenario_id == SYNCHRONOUS_ORACLE_SCENARIO_V1
            for sequence_id in cohort.sequence_ids:
                for training_seed in cohort.training_seeds:
                    yield ExpectedSupplementalRunSpecV1(
                        scenario_id=scenario_id,
                        detector_input_id=CAMERA_DETECTOR_INPUT_V1,
                        method_id=SYNCHRONOUS_ORACLE_METHOD_ID_V1,
                        scheduler_id=SYNCHRONOUS_ORACLE_SCHEDULER_ID_V1,
                        sequence_id=sequence_id,
                        condition_id="C0",
                        training_seed=training_seed,
                        network_seed=None,
                        c9_trace_id=None,
                        nominal_budget_bytes_per_second=None,
                        table_ranking_eligible=False,
                    )


def formal_supplemental_cohort_v1(plan: ExperimentPlanV1) -> SupplementalCohortV1:
    """Resolve the formal matrix dimensions and reject any weakened plan."""

    if not isinstance(plan, ExperimentPlanV1):
        raise TypeError("plan must be ExperimentPlanV1")
    if (
        len(plan.primary_sequence_ids) != FORMAL_PRIMARY_SEQUENCE_COUNT_V1
        or plan.primary_sequence_ids != tuple(sorted(set(plan.primary_sequence_ids)))
    ):
        raise SupplementalRegistryError(
            "formal supplement requires exactly 21 unique sorted V2X-Seq-SPD "
            "confirmatory validation sequences"
        )
    if plan.training_seeds != CONFIRMATORY_TRAINING_SEEDS_V1:
        raise SupplementalRegistryError("formal supplement requires official training seeds")
    if plan.network_seeds != CONFIRMATORY_NETWORK_SEEDS_V1:
        raise SupplementalRegistryError("formal supplement requires official network seeds")
    if plan.byte_budgets_per_second != CONFIRMATORY_BYTE_BUDGETS_V1:
        raise SupplementalRegistryError("formal supplement requires all five byte budgets")
    if (
        len(plan.c9_trace_ids) != FORMAL_C9_TRACE_COUNT_V1
        or plan.c9_trace_ids != tuple(sorted(set(plan.c9_trace_ids)))
    ):
        raise SupplementalRegistryError(
            "formal supplement requires exactly twenty held-out C9 traces"
        )
    if plan.network_condition_ids != tuple(f"C{index}" for index in range(10)):
        raise SupplementalRegistryError("formal supplement requires C0 through C9")
    return SupplementalCohortV1(
        sequence_ids=plan.primary_sequence_ids,
        training_seeds=plan.training_seeds,
        network_seeds=plan.network_seeds,
        c9_trace_ids=plan.c9_trace_ids,
        byte_budgets_per_second=plan.byte_budgets_per_second,
    )


def iter_expected_paper_supplemental_specs_v1(
    plan: ExperimentPlanV1,
) -> Iterator[ExpectedSupplementalRunSpecV1]:
    """Stream all 208,404 cells in the four covered paper matrices."""

    yield from _iter_expected_supplemental_specs_for_cohort(
        formal_supplemental_cohort_v1(plan)
    )


@dataclass(frozen=True, slots=True)
class SupplementalResultCellV1:
    """One sequence/replicate result with complete evidence bindings."""

    experiment_plan_sha256: str
    scenario_id: str
    detector_input_id: str
    method_id: str
    scheduler_id: str
    sequence_id: str
    condition_id: str
    training_seed: int
    network_seed: int | None
    c9_trace_id: str | None
    nominal_budget_bytes_per_second: int | None
    table_ranking_eligible: bool
    status: str
    failure_code: str | None
    metric_values: tuple[tuple[str, float], ...]
    application_bytes_total: int
    on_wire_bytes_total: int
    measurement_duration_seconds: float
    attempt_id: str
    detection_cache_sha256: str
    config_sha256: str
    prediction_archive_sha256: str
    evidence_bundle_sha256: str
    metrics_artifact_role: str
    metrics_artifact_sha256: str
    evaluator_contract_sha256: str
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
            "detector_input_id",
            "method_id",
            "scheduler_id",
            "sequence_id",
            "condition_id",
            "status",
            "attempt_id",
            "metrics_artifact_role",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.scenario_id not in SUPPLEMENTAL_SCENARIO_IDS_V1:
            raise SupplementalRegistryError("scenario_id is not registered")
        if self.condition_id not in tuple(f"C{index}" for index in range(10)):
            raise SupplementalRegistryError("condition_id must be C0 through C9")
        if type(self.training_seed) is not int or self.training_seed <= 0:
            raise SupplementalRegistryError("training_seed must be a positive integer")
        if self.network_seed is not None and (
            type(self.network_seed) is not int or self.network_seed <= 0
        ):
            raise SupplementalRegistryError("network_seed must be positive or None")
        if self.c9_trace_id is not None:
            object.__setattr__(
                self, "c9_trace_id", _identifier(self.c9_trace_id, "c9_trace_id")
            )
        if self.condition_id == "C0":
            if self.network_seed is not None or self.c9_trace_id is not None:
                raise SupplementalRegistryError("C0 cannot carry a pressure replicate")
        elif self.condition_id == "C9":
            if self.network_seed is not None or self.c9_trace_id is None:
                raise SupplementalRegistryError("C9 requires one trace and no seed")
        elif self.network_seed is None or self.c9_trace_id is not None:
            raise SupplementalRegistryError(
                "C1-C8 require one network seed and no C9 trace"
            )
        if self.condition_id in DISTURBED_INPUT_CONDITION_IDS_V1:
            if self.condition_input_manifest_sha256 is None:
                raise SupplementalRegistryError(
                    "C6-C8 require condition_input_manifest_sha256"
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
            raise SupplementalRegistryError(
                "C0-C5 and C9 cannot carry condition_input_manifest_sha256"
            )
        if type(self.table_ranking_eligible) is not bool:
            raise SupplementalRegistryError("table_ranking_eligible must be boolean")
        self._validate_scenario_semantics()
        metrics = _metric_values(self.metric_values)
        object.__setattr__(self, "metric_values", metrics)
        metric_violation = _metric_domain_violation_v1(metrics)
        if metric_violation is not None:
            raise SupplementalRegistryError(
                f"metric semantic domain violation: {metric_violation}"
            )
        if self.status not in SUPPLEMENTAL_STATUSES_V1:
            raise SupplementalRegistryError("status is not a terminal status")
        if self.status == SUPPLEMENTAL_STATUS_SUCCESS_V1:
            if tuple(metric_id for metric_id, _ in metrics) != (
                REQUIRED_SUPPLEMENTAL_METRIC_IDS_V1
            ) or self.failure_code is not None:
                raise SupplementalRegistryError(
                    "success requires every exact supplemental metric and no failure_code"
                )
        else:
            if metrics or self.failure_code is None:
                raise SupplementalRegistryError(
                    "algorithm failure requires no metrics and one failure_code"
                )
            object.__setattr__(
                self, "failure_code", _identifier(self.failure_code, "failure_code")
            )
        for name in ("application_bytes_total", "on_wire_bytes_total"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SupplementalRegistryError(
                    f"{name} must be a non-negative integer"
                )
        if self.application_bytes_total > self.on_wire_bytes_total:
            raise SupplementalRegistryError(
                "application bytes cannot exceed measured on-wire bytes"
            )
        duration = _finite_nonnegative(
            self.measurement_duration_seconds, "measurement_duration_seconds"
        )
        if duration <= 0.0:
            raise SupplementalRegistryError("measurement duration must be positive")
        object.__setattr__(self, "measurement_duration_seconds", duration)
        if self.status == SUPPLEMENTAL_STATUS_SUCCESS_V1:
            values = dict(metrics)
            expected_application_bps = self.application_bytes_total / duration
            expected_on_wire_bps = self.on_wire_bytes_total / duration
            if not np.isclose(
                values["application_bps"], expected_application_bps, rtol=1e-12, atol=1e-12
            ) or not np.isclose(
                values["on_wire_bps"], expected_on_wire_bps, rtol=1e-12, atol=1e-12
            ):
                raise SupplementalRegistryError(
                    "reported application/on-wire BPS must equal byte ledgers over duration"
                )
        if self.nominal_budget_bytes_per_second is not None:
            if (
                type(self.nominal_budget_bytes_per_second) is not int
                or self.nominal_budget_bytes_per_second <= 0
            ):
                raise SupplementalRegistryError("nominal budget must be positive or None")
            if (
                self.on_wire_bytes_total / duration
                > float(self.nominal_budget_bytes_per_second) + 1e-9
            ):
                raise SupplementalRegistryError("on-wire rate exceeds nominal budget")
        for name in (
            "detection_cache_sha256",
            "config_sha256",
            "prediction_archive_sha256",
            "evidence_bundle_sha256",
            "metrics_artifact_sha256",
            "evaluator_contract_sha256",
            "channel_outcome_trace_sha256",
            "network_trace_sha256",
            "wire_ledger_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.metrics_artifact_role != SUPPLEMENTAL_METRICS_ARTIFACT_ROLE_V1:
            raise SupplementalRegistryError(
                "metrics_artifact_role must be per_sequence_metrics"
            )

    def _validate_scenario_semantics(self) -> None:
        budget = self.nominal_budget_bytes_per_second
        if self.scenario_id == FULL_SEND_DIAGNOSTIC_SCENARIO_V1:
            valid = (
                self.detector_input_id == CAMERA_DETECTOR_INPUT_V1
                and self.method_id == "eventtrack-v2x"
                and self.scheduler_id == "full_send"
                and self.condition_id != "C0"
                and budget is None
                and not self.table_ranking_eligible
            )
        elif self.scenario_id == SYNCHRONOUS_ORACLE_SCENARIO_V1:
            valid = (
                self.detector_input_id == CAMERA_DETECTOR_INPUT_V1
                and self.method_id == SYNCHRONOUS_ORACLE_METHOD_ID_V1
                and self.scheduler_id == SYNCHRONOUS_ORACLE_SCHEDULER_ID_V1
                and self.condition_id == "C0"
                and budget is None
                and not self.table_ranking_eligible
            )
        elif self.scenario_id == SCHEDULER_FRONTIER_SCENARIO_V1:
            valid = (
                self.detector_input_id == CAMERA_DETECTOR_INPUT_V1
                and self.method_id == "eventtrack-v2x"
                and self.scheduler_id in CAPPED_SCHEDULER_IDS_V1
                and self.condition_id != "C0"
                and type(budget) is int
                and self.table_ranking_eligible
            )
        else:
            valid = (
                self.detector_input_id == RAW_SENSOR_DETECTOR_INPUT_V1
                and self.method_id in RAW_SENSOR_METHOD_IDS_V1
                and self.scheduler_id == RAW_SENSOR_SCHEDULER_ID_V1
                and type(budget) is int
                and self.table_ranking_eligible
            )
        if not valid:
            raise SupplementalRegistryError(
                "cell violates scenario budget, ranking, or pipeline semantics"
            )

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.scenario_id,
            self.detector_input_id,
            self.method_id,
            self.scheduler_id,
            self.sequence_id,
            self.condition_id,
            self.training_seed,
            self.network_seed,
            self.c9_trace_id,
            self.nominal_budget_bytes_per_second,
        )

    def metric_or_failure_zero(self, metric_id: str) -> float:
        """Return a successful metric or the preregistered failure value zero."""

        if metric_id not in REQUIRED_SUPPLEMENTAL_METRIC_IDS_V1:
            raise SupplementalRegistryError("metric_id is not required by this registry")
        if self.status == SUPPLEMENTAL_STATUS_FAILURE_V1:
            return 0.0
        return dict(self.metric_values)[metric_id]

    def to_primitive(self) -> dict[str, object]:
        return {
            "application_bytes_total": self.application_bytes_total,
            "attempt_id": self.attempt_id,
            "c9_trace_id": self.c9_trace_id,
            "channel_outcome_trace_sha256": self.channel_outcome_trace_sha256,
            "condition_id": self.condition_id,
            "condition_input_manifest_sha256": (
                self.condition_input_manifest_sha256
            ),
            "config_sha256": self.config_sha256,
            "detection_cache_sha256": self.detection_cache_sha256,
            "detector_input_id": self.detector_input_id,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "evidence_bundle_sha256": self.evidence_bundle_sha256,
            "experiment_plan_sha256": self.experiment_plan_sha256,
            "failure_code": self.failure_code,
            "measurement_duration_seconds": self.measurement_duration_seconds,
            "method_id": self.method_id,
            "metric_values": [list(item) for item in self.metric_values],
            "metrics_artifact_role": self.metrics_artifact_role,
            "metrics_artifact_sha256": self.metrics_artifact_sha256,
            "network_seed": self.network_seed,
            "network_trace_sha256": self.network_trace_sha256,
            "nominal_budget_bytes_per_second": self.nominal_budget_bytes_per_second,
            "on_wire_bytes_total": self.on_wire_bytes_total,
            "prediction_archive_sha256": self.prediction_archive_sha256,
            "scenario_id": self.scenario_id,
            "scheduler_id": self.scheduler_id,
            "sequence_id": self.sequence_id,
            "status": self.status,
            "table_ranking_eligible": self.table_ranking_eligible,
            "training_seed": self.training_seed,
            "wire_ledger_sha256": self.wire_ledger_sha256,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "SupplementalResultCellV1":
        fields = frozenset(
            {
                "application_bytes_total",
                "attempt_id",
                "c9_trace_id",
                "channel_outcome_trace_sha256",
                "condition_id",
                "condition_input_manifest_sha256",
                "config_sha256",
                "detection_cache_sha256",
                "detector_input_id",
                "evaluator_contract_sha256",
                "evidence_bundle_sha256",
                "experiment_plan_sha256",
                "failure_code",
                "measurement_duration_seconds",
                "method_id",
                "metric_values",
                "metrics_artifact_role",
                "metrics_artifact_sha256",
                "network_seed",
                "network_trace_sha256",
                "nominal_budget_bytes_per_second",
                "on_wire_bytes_total",
                "prediction_archive_sha256",
                "scenario_id",
                "scheduler_id",
                "sequence_id",
                "status",
                "table_ranking_eligible",
                "training_seed",
                "wire_ledger_sha256",
            }
        )
        item = _strict_fields(value, fields, cls.__name__)
        raw_metrics = item["metric_values"]
        if not isinstance(raw_metrics, list):
            raise SupplementalRegistryError("metric_values must be an array")
        return cls(
            experiment_plan_sha256=item["experiment_plan_sha256"],
            scenario_id=item["scenario_id"],
            detector_input_id=item["detector_input_id"],
            method_id=item["method_id"],
            scheduler_id=item["scheduler_id"],
            sequence_id=item["sequence_id"],
            condition_id=item["condition_id"],
            training_seed=item["training_seed"],
            network_seed=item["network_seed"],
            c9_trace_id=item["c9_trace_id"],
            nominal_budget_bytes_per_second=item[
                "nominal_budget_bytes_per_second"
            ],
            table_ranking_eligible=item["table_ranking_eligible"],
            status=item["status"],
            failure_code=item["failure_code"],
            metric_values=tuple(tuple(metric) for metric in raw_metrics),
            application_bytes_total=item["application_bytes_total"],
            on_wire_bytes_total=item["on_wire_bytes_total"],
            measurement_duration_seconds=item["measurement_duration_seconds"],
            attempt_id=item["attempt_id"],
            detection_cache_sha256=item["detection_cache_sha256"],
            config_sha256=item["config_sha256"],
            prediction_archive_sha256=item["prediction_archive_sha256"],
            evidence_bundle_sha256=item["evidence_bundle_sha256"],
            metrics_artifact_role=item["metrics_artifact_role"],
            metrics_artifact_sha256=item["metrics_artifact_sha256"],
            evaluator_contract_sha256=item["evaluator_contract_sha256"],
            channel_outcome_trace_sha256=item[
                "channel_outcome_trace_sha256"
            ],
            network_trace_sha256=item["network_trace_sha256"],
            condition_input_manifest_sha256=item[
                "condition_input_manifest_sha256"
            ],
            wire_ledger_sha256=item["wire_ledger_sha256"],
        )


@dataclass(frozen=True, slots=True)
class PaperSupplementalRegistryV1:
    """Canonical supplement-only evidence; never a primary SOTA-gate input."""

    experiment_plan_sha256: str
    dataset_manifest_sha256: str
    split_sha256: str
    cohort_sha256: str
    frame_contract_sha256: str
    primary_detector_cache_sha256: str
    raw_sensor_detector_cache_sha256: str
    evaluator_contract_sha256: str
    network_trace_manifest_sha256: str
    wire_accounting_config_sha256: str
    heldout_c9_trace_receipt_sha256: str
    sequence_ids: tuple[str, ...]
    c9_trace_ids: tuple[str, ...]
    config_bindings: tuple[SupplementalConfigBindingV1, ...]
    config_manifest_sha256: str
    contributes_to_primary_sota_gate: bool
    failure_semantics: str
    cells: tuple[SupplementalResultCellV1, ...]

    def __post_init__(self) -> None:
        for name in (
            "experiment_plan_sha256",
            "dataset_manifest_sha256",
            "split_sha256",
            "cohort_sha256",
            "frame_contract_sha256",
            "primary_detector_cache_sha256",
            "raw_sensor_detector_cache_sha256",
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
            raise SupplementalRegistryError("config_bindings must be an array")
        bindings = tuple(self.config_bindings)
        if not bindings or not all(
            isinstance(item, SupplementalConfigBindingV1) for item in bindings
        ):
            raise SupplementalRegistryError(
                "config_bindings must contain SupplementalConfigBindingV1"
            )
        if tuple(item.key for item in bindings) != tuple(
            sorted({item.key for item in bindings})
        ):
            raise SupplementalRegistryError("config_bindings must be unique and sorted")
        if supplemental_config_manifest_sha256_v1(bindings) != (
            self.config_manifest_sha256
        ):
            raise SupplementalRegistryError("config manifest SHA-256 mismatch")
        object.__setattr__(self, "config_bindings", bindings)
        if self.contributes_to_primary_sota_gate is not False:
            raise SupplementalRegistryError(
                "supplemental registry cannot contribute to the primary SOTA gate"
            )
        if self.failure_semantics != SUPPLEMENTAL_FAILURE_SEMANTICS_V1:
            raise SupplementalRegistryError("failure semantics must be zero scoring")
        if not isinstance(self.cells, (list, tuple)) or not self.cells:
            raise SupplementalRegistryError("cells must be a non-empty array")
        cells = tuple(self.cells)
        if not all(isinstance(item, SupplementalResultCellV1) for item in cells):
            raise SupplementalRegistryError(
                "cells must contain SupplementalResultCellV1"
            )
        keys = tuple(item.key for item in cells)
        if keys != tuple(sorted(set(keys), key=_sortable_key)):
            raise SupplementalRegistryError("cells must be unique and sorted by key")
        attempt_ids = tuple(item.attempt_id for item in cells)
        if len(attempt_ids) != len(set(attempt_ids)):
            raise SupplementalRegistryError(
                "each supplemental cell must bind one unique physical attempt"
            )
        binding_map = {item.key: item.config_sha256 for item in bindings}
        for cell in cells:
            if cell.experiment_plan_sha256 != self.experiment_plan_sha256:
                raise SupplementalRegistryError("cell has stale experiment plan binding")
            expected_cache = (
                self.primary_detector_cache_sha256
                if cell.detector_input_id == CAMERA_DETECTOR_INPUT_V1
                else self.raw_sensor_detector_cache_sha256
            )
            if cell.detection_cache_sha256 != expected_cache:
                raise SupplementalRegistryError("cell has wrong detector cache binding")
            if cell.evaluator_contract_sha256 != self.evaluator_contract_sha256:
                raise SupplementalRegistryError("cell has wrong evaluator binding")
            try:
                expected_config = binding_map[
                    (cell.detector_input_id, cell.method_id, cell.scheduler_id)
                ]
            except KeyError as exc:
                raise SupplementalRegistryError(
                    "cell pipeline lacks a frozen config binding"
                ) from exc
            if cell.config_sha256 != expected_config:
                raise SupplementalRegistryError("cell has wrong config binding")
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
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "experiment_plan_sha256": self.experiment_plan_sha256,
            "failure_semantics": self.failure_semantics,
            "frame_contract_sha256": self.frame_contract_sha256,
            "heldout_c9_trace_receipt_sha256": self.heldout_c9_trace_receipt_sha256,
            "kind": "paper_supplemental_registry_v1",
            "network_trace_manifest_sha256": self.network_trace_manifest_sha256,
            "primary_detector_cache_sha256": self.primary_detector_cache_sha256,
            "raw_sensor_detector_cache_sha256": self.raw_sensor_detector_cache_sha256,
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
    def from_mapping(cls, value: object) -> "PaperSupplementalRegistryV1":
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
                "evaluator_contract_sha256",
                "experiment_plan_sha256",
                "failure_semantics",
                "frame_contract_sha256",
                "heldout_c9_trace_receipt_sha256",
                "kind",
                "network_trace_manifest_sha256",
                "primary_detector_cache_sha256",
                "raw_sensor_detector_cache_sha256",
                "schema_version",
                "sequence_ids",
                "split_sha256",
                "wire_accounting_config_sha256",
            }
        )
        item = _strict_fields(value, fields, cls.__name__)
        if item["kind"] != "paper_supplemental_registry_v1" or item[
            "schema_version"
        ] != 1:
            raise SupplementalRegistryError("unsupported supplemental registry header")
        observed = _sha256(item["content_sha256"], "content_sha256")
        payload = {key: item[key] for key in fields if key != "content_sha256"}
        expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if observed != expected:
            raise SupplementalRegistryError("registry content SHA-256 mismatch")
        raw_bindings = item["config_bindings"]
        raw_cells = item["cells"]
        if not isinstance(raw_bindings, list) or not isinstance(raw_cells, list):
            raise SupplementalRegistryError("bindings and cells must be arrays")
        return cls(
            experiment_plan_sha256=item["experiment_plan_sha256"],
            dataset_manifest_sha256=item["dataset_manifest_sha256"],
            split_sha256=item["split_sha256"],
            cohort_sha256=item["cohort_sha256"],
            frame_contract_sha256=item["frame_contract_sha256"],
            primary_detector_cache_sha256=item["primary_detector_cache_sha256"],
            raw_sensor_detector_cache_sha256=item[
                "raw_sensor_detector_cache_sha256"
            ],
            evaluator_contract_sha256=item["evaluator_contract_sha256"],
            network_trace_manifest_sha256=item["network_trace_manifest_sha256"],
            wire_accounting_config_sha256=item["wire_accounting_config_sha256"],
            heldout_c9_trace_receipt_sha256=item[
                "heldout_c9_trace_receipt_sha256"
            ],
            sequence_ids=tuple(item["sequence_ids"]),  # type: ignore[arg-type]
            c9_trace_ids=tuple(item["c9_trace_ids"]),  # type: ignore[arg-type]
            config_bindings=tuple(
                SupplementalConfigBindingV1.from_mapping(binding)
                for binding in raw_bindings
            ),
            config_manifest_sha256=item["config_manifest_sha256"],
            contributes_to_primary_sota_gate=item[
                "contributes_to_primary_sota_gate"
            ],
            failure_semantics=item["failure_semantics"],
            cells=tuple(SupplementalResultCellV1.from_mapping(cell) for cell in raw_cells),
        )


def _validate_registry_against_cohort(
    registry: PaperSupplementalRegistryV1,
    cohort: SupplementalCohortV1,
) -> None:
    if registry.sequence_ids != cohort.sequence_ids:
        raise SupplementalRegistryError("registry sequence cohort is not exact")
    if registry.c9_trace_ids != cohort.c9_trace_ids:
        raise SupplementalRegistryError("registry C9 trace cohort is not exact")
    if tuple(item.key for item in registry.config_bindings) != (
        required_supplemental_config_keys_v1()
    ):
        raise SupplementalRegistryError("config bindings do not cover the exact matrix")
    sentinel = object()
    expected_stream = _iter_expected_supplemental_specs_for_cohort(cohort)
    for index, (expected, observed) in enumerate(
        zip_longest(expected_stream, registry.cells, fillvalue=sentinel)
    ):
        if expected is sentinel:
            raise SupplementalRegistryError(f"unexpected extra cell at index {index}")
        if observed is sentinel:
            raise SupplementalRegistryError(f"missing expected cell at index {index}")
        assert isinstance(expected, ExpectedSupplementalRunSpecV1)
        assert isinstance(observed, SupplementalResultCellV1)
        if observed.key != expected.key:
            raise SupplementalRegistryError(
                f"cell key mismatch at index {index}: {observed.key!r} != {expected.key!r}"
            )
        if observed.table_ranking_eligible != expected.table_ranking_eligible:
            raise SupplementalRegistryError(
                f"ranking eligibility mismatch at index {index}"
            )
        if (
            observed.condition_input_manifest_sha256 is not None
        ) is not expected.condition_input_manifest_required:
            raise SupplementalRegistryError(
                f"condition-input manifest requirement mismatch at index {index}"
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
        raise SupplementalRegistryError(
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
            raise SupplementalRegistryError(
                "condition input manifest inventory key does not match content"
            )
        result[digest] = manifest
    return result


def _validate_external_input_bindings_v1(
    registry: PaperSupplementalRegistryV1,
    receipt: MeasuredTraceReceiptV1,
    condition_input_manifests: Mapping[str, ConditionInputManifestV1],
) -> None:
    """Bind every C6--C8 run and C9 replay to its raw input artifact."""

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
            # channel_outcome_trace is the exogenous packet artifact.  The
            # network trace remains a run-specific scheduler/outcome artifact.
            if (
                cell.channel_outcome_trace_sha256
                != held_out_packets.get(cell.c9_trace_id)
            ):
                raise SupplementalRegistryError(
                    "C9 cell is not bound to held-out packet metadata"
                )
        if cell.condition_id not in DISTURBED_INPUT_CONDITION_IDS_V1:
            continue
        digest = cell.condition_input_manifest_sha256
        assert digest is not None
        if digest in referenced_manifests:
            raise SupplementalRegistryError(
                "condition input manifest cannot be reused across semantic cells"
            )
        manifest = inventory.get(digest)
        if manifest is None:
            raise SupplementalRegistryError(
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
            cell.config_sha256,
            cell.detection_cache_sha256,
        )
        if observed != expected:
            raise SupplementalRegistryError(
                "condition input manifest does not match its physical run"
            )
    if frozenset(inventory) != frozenset(referenced_manifests):
        raise SupplementalRegistryError(
            "condition input manifest inventory has missing or extra artifacts"
        )


def validate_paper_supplemental_registry_v1(
    registry: PaperSupplementalRegistryV1,
    plan: ExperimentPlanV1,
    *,
    raw_sensor_detector_cache_sha256: str,
    measured_trace_receipt: MeasuredTraceReceiptV1,
    condition_input_manifests: Mapping[str, ConditionInputManifestV1],
) -> None:
    """Validate the full formal supplement and its frozen plan identities.

    This deliberately returns no primary publication metrics.  The supplement
    is not an input to the SOTA gate, while ``full_send`` and the synchronous
    oracle are additionally excluded from all table rankings.
    """

    if not isinstance(registry, PaperSupplementalRegistryV1):
        raise TypeError("registry must be PaperSupplementalRegistryV1")
    if not isinstance(plan, ExperimentPlanV1):
        raise TypeError("plan must be ExperimentPlanV1")
    raw_cache = _sha256(
        raw_sensor_detector_cache_sha256, "raw_sensor_detector_cache_sha256"
    )
    _validate_measured_trace_receipt_against_plan_v1(
        plan, measured_trace_receipt
    )
    cohort = formal_supplemental_cohort_v1(plan)
    expected_headers = {
        "experiment_plan_sha256": plan.content_sha256,
        "dataset_manifest_sha256": plan.dataset_manifest_sha256,
        "split_sha256": plan.split_sha256,
        "cohort_sha256": plan.primary_cohort_sha256,
        "frame_contract_sha256": plan.primary_frame_contract_sha256,
        "primary_detector_cache_sha256": plan.detector_cache_sha256,
        "raw_sensor_detector_cache_sha256": raw_cache,
        "evaluator_contract_sha256": plan.evaluator_contract_sha256,
        "network_trace_manifest_sha256": plan.network_trace_manifest_sha256,
        "wire_accounting_config_sha256": plan.wire_accounting_config_sha256,
        "heldout_c9_trace_receipt_sha256": plan.heldout_c9_trace_receipt_sha256,
    }
    for name, expected in expected_headers.items():
        if getattr(registry, name) != expected:
            raise SupplementalRegistryError(f"registry has stale {name}")
    supplemental_configs = {
        item.key: item.config_sha256 for item in registry.config_bindings
    }
    for scheduler_id in CAPPED_SCHEDULER_IDS_V1 + ("full_send",):
        key = (CAMERA_DETECTOR_INPUT_V1, "eventtrack-v2x", scheduler_id)
        if supplemental_configs[key] != plan.run_config_sha256(
            "primary", "eventtrack-v2x", scheduler_id
        ):
            raise SupplementalRegistryError(
                "camera EventTrack scheduler config does not match plan"
            )
    # Raw-sensor pipelines and the synchronous oracle have no matching
    # ExperimentPlanV1 physical namespace; their registry bindings remain
    # self-sealed and are deliberately not represented as plan-aligned here.
    _validate_registry_against_cohort(registry, cohort)
    if len(registry.cells) != FORMAL_SUPPLEMENTAL_CELL_COUNT_V1:
        raise SupplementalRegistryError("formal supplemental cell count mismatch")
    _validate_external_input_bindings_v1(
        registry, measured_trace_receipt, condition_input_manifests
    )


def decode_paper_supplemental_registry(
    data: bytes,
) -> PaperSupplementalRegistryV1:
    """Decode canonical bytes while rejecting duplicate JSON keys."""

    if type(data) is not bytes:
        raise TypeError("registry data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise SupplementalRegistryError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                SupplementalRegistryError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupplementalRegistryError(
            "invalid PaperSupplementalRegistryV1 JSON"
        ) from exc
    registry = PaperSupplementalRegistryV1.from_mapping(value)
    if registry.canonical_bytes() != data:
        raise SupplementalRegistryError(
            "PaperSupplementalRegistryV1 JSON is not canonical"
        )
    return registry


__all__ = [
    "CALIBRATION_METRIC_IDS_V1",
    "CAMERA_DETECTOR_INPUT_V1",
    "CAPPED_SCHEDULER_IDS_V1",
    "COMMUNICATION_METRIC_IDS_V1",
    "ExpectedSupplementalRunSpecV1",
    "FORMAL_SUPPLEMENTAL_CELL_COUNT_V1",
    "FULL_SEND_DIAGNOSTIC_SCENARIO_V1",
    "PaperSupplementalRegistryV1",
    "RAW_SENSOR_DETECTOR_INPUT_V1",
    "RAW_SENSOR_METHOD_IDS_V1",
    "RAW_SENSOR_SECONDARY_SCENARIO_V1",
    "REQUIRED_SUPPLEMENTAL_METRIC_IDS_V1",
    "SCHEDULER_FRONTIER_SCENARIO_V1",
    "SUPPLEMENTAL_FAILURE_SEMANTICS_V1",
    "SUPPLEMENTAL_METRICS_ARTIFACT_ROLE_V1",
    "SUPPLEMENTAL_STATUS_FAILURE_V1",
    "SUPPLEMENTAL_STATUS_SUCCESS_V1",
    "SYSTEM_METRIC_IDS_V1",
    "SYNCHRONOUS_ORACLE_SCENARIO_V1",
    "SupplementalCohortV1",
    "SupplementalConfigBindingV1",
    "SupplementalRegistryError",
    "SupplementalResultCellV1",
    "TRACKING_METRIC_IDS_V1",
    "TRAFFIC_APPLICATION_METRIC_IDS_V1",
    "decode_paper_supplemental_registry",
    "formal_supplemental_cohort_v1",
    "iter_expected_paper_supplemental_specs_v1",
    "required_supplemental_config_keys_v1",
    "supplemental_config_manifest_sha256_v1",
    "validate_paper_supplemental_registry_v1",
]
