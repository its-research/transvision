"""Canonical train-split OOF qualification registry for EventTrack-V2X.

This module seals the development decisions that must exist before the
confirmatory validation split is opened.  It deliberately carries the complete
``EvidenceBundleV1`` documents as well as their digests, so a registry cannot
silently point at configuration, prediction, or metric artifacts different
from the evidence it claims to qualify.

The deterministic selection rules are part of the schema:

* tracker backend: maximum Robust-AssA@64k, then clean HOTA, then clean
  AMOTA, then the lexicographically smallest method identifier;
* strongest qualified baseline: the same rule over all six baselines;
* strongest non-VoI scheduler: maximum Pareto AUC, then
  Robust-AssA@64k, then the lexicographically smallest scheduler identifier.

No trusted timestamp is invented here.  A detached external receipt signs the
finished experiment plan, which in turn binds this registry's digest.  Keeping
that receipt outside this object avoids a circular receipt/registry hash.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping

import numpy as np

from .contracts import EvidenceBundleV1
from .experiment import (
    CONFIRMATORY_NETWORK_SEEDS_V1,
    CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1,
    CONFIRMATORY_QUALIFIED_SCHEDULER_BASELINE_IDS_V1,
    CONFIRMATORY_TRAINING_SEEDS_V1,
)
from .validation_registry import (
    ValidationResultRegistryV1,
    derive_validation_summaries_v1,
)
from .wire import canonical_json_bytes


_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")

OFFICIAL_TRACKER_METHOD_TO_BACKEND_V1 = {
    "vehicle-only-ab3dmot": "ab3dmot",
    "vehicle-only-immortaltracker": "immortaltracker",
    "vehicle-only-simpletrack": "simpletrack",
}
CONTROL_BASELINE_IDS_V1 = tuple(
    sorted(
        set(CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1)
        - set(OFFICIAL_TRACKER_METHOD_TO_BACKEND_V1)
    )
)
QUALIFIED_SCHEDULER_BASELINE_IDS_V1 = tuple(
    scheduler_id
    for scheduler_id in CONFIRMATORY_QUALIFIED_SCHEDULER_BASELINE_IDS_V1
)

BACKEND_SELECTION_RULE_V1 = (
    "max-robust-assa-at-64k_then-clean-hota_then-clean-amota_"
    "then-lexicographic-method-id"
)
BASELINE_SELECTION_RULE_V1 = BACKEND_SELECTION_RULE_V1
SCHEDULER_SELECTION_RULE_V1 = (
    "max-pareto-auc_then-robust-assa-at-64k_"
    "then-lexicographic-scheduler-id"
)


class QualificationRegistryError(ValueError):
    """Raised when validation qualification evidence is incomplete or stale."""


def _strict_fields(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise QualificationRegistryError(f"{name} must be a string-keyed object")
    if frozenset(value) != expected:
        missing = sorted(expected - frozenset(value))
        unknown = sorted(frozenset(value) - expected)
        raise QualificationRegistryError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unknown={unknown}"
        )
    return value


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise QualificationRegistryError(f"{name} must be a canonical identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise QualificationRegistryError(f"{name} must be a lowercase SHA-256")
    return value


def _zero(value: object, name: str) -> int:
    if type(value) is not int or value != 0:
        raise QualificationRegistryError(f"{name} must equal zero")
    return value


def _passed(value: object, name: str) -> bool:
    if type(value) is not bool or not value:
        raise QualificationRegistryError(f"{name} must be true")
    return value


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationRegistryError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise QualificationRegistryError(f"{name} must be finite and in [0, 1]")
    return result


def _positive_ints(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise QualificationRegistryError(f"{name} must be an array")
    result: list[int] = []
    for item in value:
        if type(item) is not int or item <= 0:
            raise QualificationRegistryError(
                f"{name} items must be positive integers"
            )
        result.append(item)
    if not result or len(result) != len(set(result)):
        raise QualificationRegistryError(f"{name} must be non-empty and unique")
    return tuple(result)


def _canonical_ids(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise QualificationRegistryError(f"{name} must be an array")
    result = tuple(_identifier(item, f"{name} item") for item in value)
    if not result or len(result) != len(set(result)):
        raise QualificationRegistryError(f"{name} must be non-empty and unique")
    if result != tuple(sorted(result)):
        raise QualificationRegistryError(f"{name} must be lexicographically sorted")
    return result


def _validate_evidence_artifacts(
    *,
    evidence: EvidenceBundleV1,
    evidence_bundle_sha256: str,
    config_sha256: str,
    predictions_sha256: str,
    metrics_sha256: str,
) -> None:
    if evidence.digest() != evidence_bundle_sha256:
        raise QualificationRegistryError("EvidenceBundleV1 digest mismatch")
    expected = {
        "tracker_config": config_sha256,
        "predictions": predictions_sha256,
        "per_sequence_metrics": metrics_sha256,
    }
    mismatched = [
        role
        for role, sha256 in expected.items()
        if getattr(evidence, role).sha256 != sha256
    ]
    if mismatched:
        raise QualificationRegistryError(
            f"qualification artifact/EvidenceBundleV1 mismatch: {mismatched}"
        )


@dataclass(frozen=True, slots=True)
class BaselineQualificationEntryV1:
    """One fully successful validation qualification for a tracking baseline."""

    method_id: str
    tracker_backend_id: str
    qualification_kind: str
    backend_source_sha256: str
    config_sha256: str
    predictions_sha256: str
    metrics_sha256: str
    evidence_bundle_sha256: str
    evidence_bundle: EvidenceBundleV1
    failure_count: int
    robust_assa_at_64k: float
    clean_amota: float
    clean_hota: float
    passed: bool

    _FIELDS = frozenset(
        {
            "backend_source_sha256",
            "clean_amota",
            "clean_hota",
            "config_sha256",
            "evidence_bundle",
            "evidence_bundle_sha256",
            "failure_count",
            "method_id",
            "metrics_sha256",
            "passed",
            "predictions_sha256",
            "qualification_kind",
            "robust_assa_at_64k",
            "tracker_backend_id",
        }
    )

    def __post_init__(self) -> None:
        for name in ("method_id", "tracker_backend_id", "qualification_kind"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "backend_source_sha256",
            "config_sha256",
            "predictions_sha256",
            "metrics_sha256",
            "evidence_bundle_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if not isinstance(self.evidence_bundle, EvidenceBundleV1):
            raise TypeError("evidence_bundle must be EvidenceBundleV1")
        object.__setattr__(
            self, "failure_count", _zero(self.failure_count, "failure_count")
        )
        object.__setattr__(self, "passed", _passed(self.passed, "passed"))
        for name in ("robust_assa_at_64k", "clean_amota", "clean_hota"):
            object.__setattr__(
                self, name, _unit_interval(getattr(self, name), name)
            )
        _validate_evidence_artifacts(
            evidence=self.evidence_bundle,
            evidence_bundle_sha256=self.evidence_bundle_sha256,
            config_sha256=self.config_sha256,
            predictions_sha256=self.predictions_sha256,
            metrics_sha256=self.metrics_sha256,
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "backend_source_sha256": self.backend_source_sha256,
            "clean_amota": self.clean_amota,
            "clean_hota": self.clean_hota,
            "config_sha256": self.config_sha256,
            "evidence_bundle": self.evidence_bundle.to_primitive(),
            "evidence_bundle_sha256": self.evidence_bundle_sha256,
            "failure_count": self.failure_count,
            "method_id": self.method_id,
            "metrics_sha256": self.metrics_sha256,
            "passed": self.passed,
            "predictions_sha256": self.predictions_sha256,
            "qualification_kind": self.qualification_kind,
            "robust_assa_at_64k": self.robust_assa_at_64k,
            "tracker_backend_id": self.tracker_backend_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "BaselineQualificationEntryV1":
        result = _strict_fields(value, cls._FIELDS, cls.__name__)
        return cls(
            method_id=result["method_id"],
            tracker_backend_id=result["tracker_backend_id"],
            qualification_kind=result["qualification_kind"],
            backend_source_sha256=result["backend_source_sha256"],
            config_sha256=result["config_sha256"],
            predictions_sha256=result["predictions_sha256"],
            metrics_sha256=result["metrics_sha256"],
            evidence_bundle_sha256=result["evidence_bundle_sha256"],
            evidence_bundle=EvidenceBundleV1.from_mapping(
                result["evidence_bundle"]
            ),
            failure_count=result["failure_count"],
            robust_assa_at_64k=result["robust_assa_at_64k"],
            clean_amota=result["clean_amota"],
            clean_hota=result["clean_hota"],
            passed=result["passed"],
        )


@dataclass(frozen=True, slots=True)
class SchedulerQualificationEntryV1:
    """One fully successful validation qualification for a non-VoI scheduler."""

    scheduler_id: str
    config_sha256: str
    predictions_sha256: str
    metrics_sha256: str
    evidence_bundle_sha256: str
    evidence_bundle: EvidenceBundleV1
    failure_count: int
    robust_assa_at_64k: float
    pareto_auc: float
    passed: bool

    _FIELDS = frozenset(
        {
            "config_sha256",
            "evidence_bundle",
            "evidence_bundle_sha256",
            "failure_count",
            "metrics_sha256",
            "pareto_auc",
            "passed",
            "predictions_sha256",
            "robust_assa_at_64k",
            "scheduler_id",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scheduler_id", _identifier(self.scheduler_id, "scheduler_id")
        )
        for name in (
            "config_sha256",
            "predictions_sha256",
            "metrics_sha256",
            "evidence_bundle_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if not isinstance(self.evidence_bundle, EvidenceBundleV1):
            raise TypeError("evidence_bundle must be EvidenceBundleV1")
        object.__setattr__(
            self, "failure_count", _zero(self.failure_count, "failure_count")
        )
        object.__setattr__(self, "passed", _passed(self.passed, "passed"))
        for name in ("robust_assa_at_64k", "pareto_auc"):
            object.__setattr__(
                self, name, _unit_interval(getattr(self, name), name)
            )
        _validate_evidence_artifacts(
            evidence=self.evidence_bundle,
            evidence_bundle_sha256=self.evidence_bundle_sha256,
            config_sha256=self.config_sha256,
            predictions_sha256=self.predictions_sha256,
            metrics_sha256=self.metrics_sha256,
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "config_sha256": self.config_sha256,
            "evidence_bundle": self.evidence_bundle.to_primitive(),
            "evidence_bundle_sha256": self.evidence_bundle_sha256,
            "failure_count": self.failure_count,
            "metrics_sha256": self.metrics_sha256,
            "pareto_auc": self.pareto_auc,
            "passed": self.passed,
            "predictions_sha256": self.predictions_sha256,
            "robust_assa_at_64k": self.robust_assa_at_64k,
            "scheduler_id": self.scheduler_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "SchedulerQualificationEntryV1":
        result = _strict_fields(value, cls._FIELDS, cls.__name__)
        return cls(
            scheduler_id=result["scheduler_id"],
            config_sha256=result["config_sha256"],
            predictions_sha256=result["predictions_sha256"],
            metrics_sha256=result["metrics_sha256"],
            evidence_bundle_sha256=result["evidence_bundle_sha256"],
            evidence_bundle=EvidenceBundleV1.from_mapping(
                result["evidence_bundle"]
            ),
            failure_count=result["failure_count"],
            robust_assa_at_64k=result["robust_assa_at_64k"],
            pareto_auc=result["pareto_auc"],
            passed=result["passed"],
        )


def _winner(
    entries: tuple[BaselineQualificationEntryV1, ...],
) -> BaselineQualificationEntryV1:
    best_score = max(
        (item.robust_assa_at_64k, item.clean_hota, item.clean_amota)
        for item in entries
    )
    return min(
        (
            item
            for item in entries
            if (item.robust_assa_at_64k, item.clean_hota, item.clean_amota)
            == best_score
        ),
        key=lambda item: item.method_id,
    )


def _scheduler_winner(
    entries: tuple[SchedulerQualificationEntryV1, ...],
) -> SchedulerQualificationEntryV1:
    best_score = max(
        (item.pareto_auc, item.robust_assa_at_64k) for item in entries
    )
    return min(
        (
            item
            for item in entries
            if (item.pareto_auc, item.robust_assa_at_64k) == best_score
        ),
        key=lambda item: item.scheduler_id,
    )


@dataclass(frozen=True, slots=True)
class BaselineQualificationRegistryV1:
    """Frozen validation qualification for baselines and scheduler controls."""

    registry_id: str
    source_tree_sha256: str
    dataset_id: str
    dataset_manifest_sha256: str
    split_name: str
    split_sha256: str
    cohort_sha256: str
    sequence_ids: tuple[str, ...]
    development_fold_manifest_sha256: str
    frequency_hz: float
    class_names: tuple[str, ...]
    detection_cache_sha256: str
    detection_frame_contract_sha256: str
    network_trace_manifest_sha256: str
    evaluator_contract_sha256: str
    training_seeds: tuple[int, ...]
    network_seeds: tuple[int, ...]
    primary_budget_bytes_per_second: int
    validation_result_registry_sha256: str
    baseline_entries: tuple[BaselineQualificationEntryV1, ...]
    selected_tracker_backend_id: str
    strongest_qualified_baseline_id: str
    scheduler_entries: tuple[SchedulerQualificationEntryV1, ...]
    strongest_scheduler_baseline_id: str
    backend_selection_rule: str = BACKEND_SELECTION_RULE_V1
    baseline_selection_rule: str = BASELINE_SELECTION_RULE_V1
    scheduler_selection_rule: str = SCHEDULER_SELECTION_RULE_V1

    def __post_init__(self) -> None:
        for name in (
            "registry_id",
            "dataset_id",
            "split_name",
            "selected_tracker_backend_id",
            "strongest_qualified_baseline_id",
            "strongest_scheduler_baseline_id",
            "backend_selection_rule",
            "baseline_selection_rule",
            "scheduler_selection_rule",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "source_tree_sha256",
            "dataset_manifest_sha256",
            "split_sha256",
            "cohort_sha256",
            "development_fold_manifest_sha256",
            "detection_cache_sha256",
            "detection_frame_contract_sha256",
            "network_trace_manifest_sha256",
            "evaluator_contract_sha256",
            "validation_result_registry_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        sequence_ids = _canonical_ids(self.sequence_ids, "sequence_ids")
        object.__setattr__(self, "sequence_ids", sequence_ids)
        class_names = _canonical_ids(self.class_names, "class_names")
        object.__setattr__(self, "class_names", class_names)
        training_seeds = _positive_ints(self.training_seeds, "training_seeds")
        network_seeds = _positive_ints(self.network_seeds, "network_seeds")
        object.__setattr__(self, "training_seeds", training_seeds)
        object.__setattr__(self, "network_seeds", network_seeds)

        if self.registry_id != "eventtrack-v2x-development-qualification-v1":
            raise QualificationRegistryError("unexpected qualification registry_id")
        if self.dataset_id != "v2x-seq-spd" or self.split_name != "train":
            raise QualificationRegistryError(
                "qualification requires V2X-Seq-SPD train OOF"
            )
        if len(sequence_ids) != 46:
            raise QualificationRegistryError(
                "V2X-Seq-SPD train qualification requires exactly 46 sequences"
            )
        if isinstance(self.frequency_hz, bool) or not isinstance(
            self.frequency_hz, (int, float)
        ):
            raise QualificationRegistryError("frequency_hz must be numeric")
        frequency = float(self.frequency_hz)
        if not np.isfinite(frequency) or frequency != 10.0:
            raise QualificationRegistryError("qualification frequency_hz must equal 10")
        object.__setattr__(self, "frequency_hz", frequency)
        if class_names != ("car",):
            raise QualificationRegistryError("class_names must equal ('car',)")
        if training_seeds != CONFIRMATORY_TRAINING_SEEDS_V1:
            raise QualificationRegistryError(
                "training_seeds must equal 1337, 2027, and 3407"
            )
        if network_seeds != CONFIRMATORY_NETWORK_SEEDS_V1:
            raise QualificationRegistryError(
                "network_seeds must equal 1001 through 1010"
            )
        if (
            type(self.primary_budget_bytes_per_second) is not int
            or self.primary_budget_bytes_per_second != 64_000
        ):
            raise QualificationRegistryError(
                "primary_budget_bytes_per_second must equal 64000"
            )
        if self.backend_selection_rule != BACKEND_SELECTION_RULE_V1:
            raise QualificationRegistryError("backend selection rule is not frozen V1")
        if self.baseline_selection_rule != BASELINE_SELECTION_RULE_V1:
            raise QualificationRegistryError("baseline selection rule is not frozen V1")
        if self.scheduler_selection_rule != SCHEDULER_SELECTION_RULE_V1:
            raise QualificationRegistryError("scheduler selection rule is not frozen V1")

        baselines = self._validate_baselines(self.baseline_entries)
        schedulers = self._validate_schedulers(self.scheduler_entries)
        object.__setattr__(self, "baseline_entries", baselines)
        object.__setattr__(self, "scheduler_entries", schedulers)
        self._validate_evidence_identity((*baselines, *schedulers))

    def _validate_baselines(
        self, value: object
    ) -> tuple[BaselineQualificationEntryV1, ...]:
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, BaselineQualificationEntryV1) for item in value
        ):
            raise TypeError(
                "baseline_entries must contain BaselineQualificationEntryV1"
            )
        result = tuple(value)
        method_ids = tuple(item.method_id for item in result)
        if method_ids != CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1:
            raise QualificationRegistryError(
                "baseline_entries must contain the exact six qualified baselines "
                "in canonical order"
            )
        official = tuple(
            item
            for item in result
            if item.method_id in OFFICIAL_TRACKER_METHOD_TO_BACKEND_V1
        )
        for item in official:
            expected_backend = OFFICIAL_TRACKER_METHOD_TO_BACKEND_V1[item.method_id]
            if (
                item.qualification_kind != "official-backend"
                or item.tracker_backend_id != expected_backend
            ):
                raise QualificationRegistryError(
                    f"{item.method_id} does not qualify its official backend"
                )
        selected = _winner(official)
        if self.selected_tracker_backend_id != selected.tracker_backend_id:
            raise QualificationRegistryError(
                "selected_tracker_backend_id violates the frozen development rule"
            )
        for item in result:
            if item.method_id not in CONTROL_BASELINE_IDS_V1:
                continue
            if (
                item.qualification_kind != "shared-selected-backend"
                or item.tracker_backend_id != selected.tracker_backend_id
                or item.backend_source_sha256 != selected.backend_source_sha256
            ):
                raise QualificationRegistryError(
                    f"{item.method_id} must use the selected common tracker backend"
                )
        strongest = _winner(result)
        if self.strongest_qualified_baseline_id != strongest.method_id:
            raise QualificationRegistryError(
                "strongest_qualified_baseline_id violates the frozen development rule"
            )
        return result

    def _validate_schedulers(
        self, value: object
    ) -> tuple[SchedulerQualificationEntryV1, ...]:
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, SchedulerQualificationEntryV1) for item in value
        ):
            raise TypeError(
                "scheduler_entries must contain SchedulerQualificationEntryV1"
            )
        result = tuple(value)
        scheduler_ids = tuple(item.scheduler_id for item in result)
        if scheduler_ids != QUALIFIED_SCHEDULER_BASELINE_IDS_V1:
            raise QualificationRegistryError(
                "scheduler_entries must contain the exact four capped non-VoI controls "
                "in canonical order"
            )
        strongest = _scheduler_winner(result)
        if self.strongest_scheduler_baseline_id != strongest.scheduler_id:
            raise QualificationRegistryError(
                "strongest_scheduler_baseline_id violates the frozen development rule"
            )
        return result

    def _validate_evidence_identity(
        self,
        entries: tuple[
            BaselineQualificationEntryV1 | SchedulerQualificationEntryV1, ...
        ],
    ) -> None:
        expected = {
            "source": self.source_tree_sha256,
            "dataset": self.dataset_manifest_sha256,
            "detection_cache": self.detection_cache_sha256,
            "network_trace": self.network_trace_manifest_sha256,
            "evaluator_contract": self.evaluator_contract_sha256,
        }
        digests: list[str] = []
        run_ids: list[str] = []
        for item in entries:
            evidence = item.evidence_bundle
            mismatched = [
                role
                for role, sha256 in expected.items()
                if getattr(evidence, role).sha256 != sha256
            ]
            if mismatched:
                raise QualificationRegistryError(
                    "registry/EvidenceBundleV1 artifact mismatch for "
                    f"{evidence.run_id}: {mismatched}"
                )
            digests.append(item.evidence_bundle_sha256)
            run_ids.append(_identifier(evidence.run_id, "evidence run_id"))
        if len(digests) != len(set(digests)) or len(run_ids) != len(set(run_ids)):
            raise QualificationRegistryError(
                "each qualification must bind a distinct EvidenceBundleV1 run"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "backend_selection_rule": self.backend_selection_rule,
            "baseline_entries": [
                item.to_primitive() for item in self.baseline_entries
            ],
            "baseline_selection_rule": self.baseline_selection_rule,
            "class_names": list(self.class_names),
            "cohort_sha256": self.cohort_sha256,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "detection_cache_sha256": self.detection_cache_sha256,
            "detection_frame_contract_sha256": (
                self.detection_frame_contract_sha256
            ),
            "development_fold_manifest_sha256": (
                self.development_fold_manifest_sha256
            ),
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "frequency_hz": self.frequency_hz,
            "network_seeds": list(self.network_seeds),
            "network_trace_manifest_sha256": self.network_trace_manifest_sha256,
            "primary_budget_bytes_per_second": (
                self.primary_budget_bytes_per_second
            ),
            "validation_result_registry_sha256": (
                self.validation_result_registry_sha256
            ),
            "registry_id": self.registry_id,
            "scheduler_entries": [
                item.to_primitive() for item in self.scheduler_entries
            ],
            "scheduler_selection_rule": self.scheduler_selection_rule,
            "schema_version": 1,
            "selected_tracker_backend_id": self.selected_tracker_backend_id,
            "sequence_ids": list(self.sequence_ids),
            "source_tree_sha256": self.source_tree_sha256,
            "split_name": self.split_name,
            "split_sha256": self.split_sha256,
            "strongest_qualified_baseline_id": (
                self.strongest_qualified_baseline_id
            ),
            "strongest_scheduler_baseline_id": (
                self.strongest_scheduler_baseline_id
            ),
            "training_seeds": list(self.training_seeds),
        }

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.payload())).hexdigest()

    def sealed_document(self) -> dict[str, Any]:
        return {**self.payload(), "content_sha256": self.content_sha256}

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.sealed_document())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


_REGISTRY_PAYLOAD_FIELDS = frozenset(
    {
        "backend_selection_rule",
        "baseline_entries",
        "baseline_selection_rule",
        "class_names",
        "cohort_sha256",
        "dataset_id",
        "dataset_manifest_sha256",
        "detection_cache_sha256",
        "detection_frame_contract_sha256",
        "development_fold_manifest_sha256",
        "evaluator_contract_sha256",
        "frequency_hz",
        "network_seeds",
        "network_trace_manifest_sha256",
        "primary_budget_bytes_per_second",
        "registry_id",
        "scheduler_entries",
        "scheduler_selection_rule",
        "schema_version",
        "selected_tracker_backend_id",
        "sequence_ids",
        "source_tree_sha256",
        "split_name",
        "split_sha256",
        "strongest_qualified_baseline_id",
        "strongest_scheduler_baseline_id",
        "training_seeds",
        "validation_result_registry_sha256",
    }
)


def qualification_registry_from_document(
    value: object,
) -> BaselineQualificationRegistryV1:
    """Validate an exact sealed qualification-registry document."""

    result = _strict_fields(
        value,
        _REGISTRY_PAYLOAD_FIELDS | {"content_sha256"},
        "BaselineQualificationRegistryV1",
    )
    if result["schema_version"] != 1:
        raise QualificationRegistryError(
            "unsupported qualification registry schema version"
        )
    observed = _sha256(result["content_sha256"], "content_sha256")
    raw_payload = {name: result[name] for name in _REGISTRY_PAYLOAD_FIELDS}
    expected = hashlib.sha256(canonical_json_bytes(raw_payload)).hexdigest()
    if observed != expected:
        raise QualificationRegistryError(
            "qualification registry content SHA-256 mismatch"
        )
    raw_baselines = result["baseline_entries"]
    raw_schedulers = result["scheduler_entries"]
    if not isinstance(raw_baselines, list) or not isinstance(raw_schedulers, list):
        raise QualificationRegistryError(
            "baseline_entries and scheduler_entries must be arrays"
        )
    registry = BaselineQualificationRegistryV1(
        registry_id=result["registry_id"],
        source_tree_sha256=result["source_tree_sha256"],
        dataset_id=result["dataset_id"],
        dataset_manifest_sha256=result["dataset_manifest_sha256"],
        split_name=result["split_name"],
        split_sha256=result["split_sha256"],
        cohort_sha256=result["cohort_sha256"],
        sequence_ids=result["sequence_ids"],
        development_fold_manifest_sha256=result[
            "development_fold_manifest_sha256"
        ],
        frequency_hz=result["frequency_hz"],
        class_names=result["class_names"],
        detection_cache_sha256=result["detection_cache_sha256"],
        detection_frame_contract_sha256=result[
            "detection_frame_contract_sha256"
        ],
        network_trace_manifest_sha256=result[
            "network_trace_manifest_sha256"
        ],
        evaluator_contract_sha256=result["evaluator_contract_sha256"],
        training_seeds=result["training_seeds"],
        network_seeds=result["network_seeds"],
        primary_budget_bytes_per_second=result[
            "primary_budget_bytes_per_second"
        ],
        validation_result_registry_sha256=result[
            "validation_result_registry_sha256"
        ],
        baseline_entries=tuple(
            BaselineQualificationEntryV1.from_mapping(item)
            for item in raw_baselines
        ),
        selected_tracker_backend_id=result["selected_tracker_backend_id"],
        strongest_qualified_baseline_id=result[
            "strongest_qualified_baseline_id"
        ],
        scheduler_entries=tuple(
            SchedulerQualificationEntryV1.from_mapping(item)
            for item in raw_schedulers
        ),
        strongest_scheduler_baseline_id=result[
            "strongest_scheduler_baseline_id"
        ],
        backend_selection_rule=result["backend_selection_rule"],
        baseline_selection_rule=result["baseline_selection_rule"],
        scheduler_selection_rule=result["scheduler_selection_rule"],
    )
    if observed != registry.content_sha256:
        raise QualificationRegistryError(
            "qualification registry canonical payload mismatch"
        )
    return registry


def decode_qualification_registry(data: bytes) -> BaselineQualificationRegistryV1:
    """Decode canonical UTF-8 JSON, rejecting duplicates and non-finite values."""

    if type(data) is not bytes:
        raise TypeError("qualification registry data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in items:
            if key in result:
                raise QualificationRegistryError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                QualificationRegistryError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationRegistryError(
            "invalid qualification registry JSON"
        ) from exc
    registry = qualification_registry_from_document(value)
    if registry.canonical_bytes() != data:
        raise QualificationRegistryError(
            "qualification registry JSON is not canonical"
        )
    return registry


def validate_qualification_against_raw_registry_v1(
    qualification: BaselineQualificationRegistryV1,
    raw_registry: ValidationResultRegistryV1,
) -> None:
    """Recompute every pooled OOF summary before trusting the decision record."""

    if not isinstance(qualification, BaselineQualificationRegistryV1):
        raise TypeError("qualification must be BaselineQualificationRegistryV1")
    if not isinstance(raw_registry, ValidationResultRegistryV1):
        raise TypeError("raw_registry must be ValidationResultRegistryV1")
    if raw_registry.digest() != qualification.validation_result_registry_sha256:
        raise QualificationRegistryError(
            "validation raw-result registry digest mismatch"
        )
    expected_identity = (
        qualification.source_tree_sha256,
        qualification.dataset_id,
        qualification.dataset_manifest_sha256,
        qualification.split_name,
        qualification.split_sha256,
        qualification.cohort_sha256,
        qualification.sequence_ids,
        qualification.development_fold_manifest_sha256,
        qualification.detection_cache_sha256,
        qualification.detection_frame_contract_sha256,
        qualification.network_trace_manifest_sha256,
        qualification.evaluator_contract_sha256,
        qualification.frequency_hz,
        qualification.class_names,
    )
    observed_identity = (
        raw_registry.source_tree_sha256,
        raw_registry.dataset_id,
        raw_registry.dataset_manifest_sha256,
        raw_registry.split_name,
        raw_registry.split_sha256,
        raw_registry.cohort_sha256,
        raw_registry.sequence_ids,
        raw_registry.development_fold_manifest.content_sha256,
        raw_registry.detection_cache_sha256,
        raw_registry.detection_frame_contract_sha256,
        raw_registry.network_trace_manifest_sha256,
        raw_registry.evaluator_contract_sha256,
        raw_registry.frequency_hz,
        raw_registry.class_names,
    )
    if observed_identity != expected_identity:
        raise QualificationRegistryError(
            "validation raw-result registry identity mismatch"
        )

    summaries = derive_validation_summaries_v1(raw_registry)
    baseline_summaries = {item.method_id: item for item in summaries.baselines}
    for entry in qualification.baseline_entries:
        summary = baseline_summaries[entry.method_id]
        if entry.failure_count != summary.failure_count or not all(
            np.isclose(left, right, rtol=0.0, atol=1e-15)
            for left, right in (
                (entry.robust_assa_at_64k, summary.robust_assa_at_64k),
                (entry.clean_amota, summary.clean_amota),
                (entry.clean_hota, summary.clean_hota),
            )
        ):
            raise QualificationRegistryError(
                "baseline summary was not derived from raw OOF cells: "
                f"{entry.method_id}"
            )
    scheduler_summaries = {
        item.scheduler_id: item for item in summaries.schedulers
    }
    for entry in qualification.scheduler_entries:
        summary = scheduler_summaries[entry.scheduler_id]
        if entry.failure_count != summary.failure_count or not all(
            np.isclose(left, right, rtol=0.0, atol=1e-15)
            for left, right in (
                (entry.robust_assa_at_64k, summary.robust_assa_at_64k),
                (entry.pareto_auc, summary.actual_byte_auc),
            )
        ):
            raise QualificationRegistryError(
                "scheduler summary was not derived from raw OOF cells: "
                f"{entry.scheduler_id}"
            )


__all__ = [
    "BACKEND_SELECTION_RULE_V1",
    "BASELINE_SELECTION_RULE_V1",
    "CONTROL_BASELINE_IDS_V1",
    "OFFICIAL_TRACKER_METHOD_TO_BACKEND_V1",
    "QUALIFIED_SCHEDULER_BASELINE_IDS_V1",
    "SCHEDULER_SELECTION_RULE_V1",
    "BaselineQualificationEntryV1",
    "BaselineQualificationRegistryV1",
    "QualificationRegistryError",
    "SchedulerQualificationEntryV1",
    "decode_qualification_registry",
    "qualification_registry_from_document",
    "validate_qualification_against_raw_registry_v1",
]
