"""Sealed train-split OOF hyperparameter selection for EventTrack-V2X.

This contract sits before ``ExperimentPlanV1`` in the evidence graph.  It binds
the immutable inputs needed to reproduce pooled OOF development, the complete frozen
``4 x 4 x 3 x 3`` search grid, and the one configuration selected by the
pre-registered rule.  It intentionally does not refer to the final experiment
plan, avoiding a plan/selection digest cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import re
from typing import Any, Mapping

import numpy as np

from .wire import canonical_json_bytes


_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")

CANDIDATE_FIXED_LAG_SECONDS_V1 = (0.2, 0.5, 1.0, 2.0)
CANDIDATE_MIXTURE_COMPONENTS_V1 = (1, 3, 5, 9)
CANDIDATE_TOP_H_V1 = (1, 3, 5)
CANDIDATE_GATE_CONFIDENCE_V1 = (0.90, 0.95, 0.99)
CANDIDATE_TRAINING_SEEDS_V1 = (1337, 2027, 3407)
CANDIDATE_NETWORK_SEEDS_V1 = tuple(range(1001, 1011))
CANDIDATE_PRIMARY_BUDGET_BPS_V1 = 64_000
CANDIDATE_CLEAN_NONINFERIORITY_MARGIN_V1 = -0.01
CANDIDATE_SELECTION_RULE_V1 = (
    "eligible-clean-noninferiority-and-64k_"
    "then-max-robust-assa_then-min-actual-bps_"
    "then-min-p95-latency_then-canonical-hyperparameter-tuple"
)

CANDIDATE_COMBINATIONS_V1 = tuple(
    itertools.product(
        CANDIDATE_FIXED_LAG_SECONDS_V1,
        CANDIDATE_MIXTURE_COMPONENTS_V1,
        CANDIDATE_TOP_H_V1,
        CANDIDATE_GATE_CONFIDENCE_V1,
    )
)


class CandidateSelectionRegistryError(ValueError):
    """Raised when candidate-selection evidence is incomplete or inconsistent."""


def _strict_fields(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise CandidateSelectionRegistryError(
            f"{name} must be a string-keyed object"
        )
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise CandidateSelectionRegistryError(
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
        raise CandidateSelectionRegistryError(
            f"{name} must be a canonical identifier"
        )
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise CandidateSelectionRegistryError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateSelectionRegistryError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise CandidateSelectionRegistryError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _unit_interval(value: object, name: str) -> float:
    result = _number(value, name)
    if not 0.0 <= result <= 1.0:
        raise CandidateSelectionRegistryError(f"{name} must be in [0, 1]")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = _number(value, name)
    if result < 0.0:
        raise CandidateSelectionRegistryError(f"{name} must be non-negative")
    return result


def _canonical_ids(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise CandidateSelectionRegistryError(f"{name} must be an array")
    result = tuple(_identifier(item, f"{name} item") for item in value)
    if not result or len(result) != len(set(result)):
        raise CandidateSelectionRegistryError(
            f"{name} must be non-empty and unique"
        )
    if result != tuple(sorted(result)):
        raise CandidateSelectionRegistryError(
            f"{name} must be lexicographically sorted"
        )
    return result


def _positive_ints(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise CandidateSelectionRegistryError(f"{name} must be an array")
    result: list[int] = []
    for item in value:
        if type(item) is not int or item <= 0:
            raise CandidateSelectionRegistryError(
                f"{name} items must be positive integers"
            )
        result.append(item)
    if not result or len(result) != len(set(result)):
        raise CandidateSelectionRegistryError(
            f"{name} must be non-empty and unique"
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CandidateSelectionEntryV1:
    """Metrics and evidence for exactly one frozen hyperparameter tuple."""

    fixed_lag_seconds: float
    mixture_components: int
    top_h: int
    gate_confidence: float
    config_sha256: str
    clean_amota: float
    clean_hota: float
    robust_assa_at_64k: float
    actual_bytes_per_second: float
    p95_latency_ms: float
    oof_checkpoint_inventory_sha256: str
    evidence_sha256: str

    _FIELDS = frozenset(
        {
            "actual_bytes_per_second",
            "clean_amota",
            "clean_hota",
            "config_sha256",
            "evidence_sha256",
            "fixed_lag_seconds",
            "gate_confidence",
            "mixture_components",
            "oof_checkpoint_inventory_sha256",
            "p95_latency_ms",
            "robust_assa_at_64k",
            "top_h",
        }
    )

    def __post_init__(self) -> None:
        fixed_lag = _number(self.fixed_lag_seconds, "fixed_lag_seconds")
        if fixed_lag not in CANDIDATE_FIXED_LAG_SECONDS_V1:
            raise CandidateSelectionRegistryError(
                "fixed_lag_seconds is outside the frozen V1 grid"
            )
        object.__setattr__(self, "fixed_lag_seconds", fixed_lag)
        if (
            type(self.mixture_components) is not int
            or self.mixture_components not in CANDIDATE_MIXTURE_COMPONENTS_V1
        ):
            raise CandidateSelectionRegistryError(
                "mixture_components is outside the frozen V1 grid"
            )
        if type(self.top_h) is not int or self.top_h not in CANDIDATE_TOP_H_V1:
            raise CandidateSelectionRegistryError(
                "top_h is outside the frozen V1 grid"
            )
        gate_confidence = _number(self.gate_confidence, "gate_confidence")
        if gate_confidence not in CANDIDATE_GATE_CONFIDENCE_V1:
            raise CandidateSelectionRegistryError(
                "gate_confidence is outside the frozen V1 grid"
            )
        object.__setattr__(self, "gate_confidence", gate_confidence)
        for name in (
            "config_sha256",
            "evidence_sha256",
            "oof_checkpoint_inventory_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in ("clean_amota", "clean_hota", "robust_assa_at_64k"):
            object.__setattr__(
                self, name, _unit_interval(getattr(self, name), name)
            )
        for name in ("actual_bytes_per_second", "p95_latency_ms"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))

    @property
    def combination(self) -> tuple[float, int, int, float]:
        return (
            self.fixed_lag_seconds,
            self.mixture_components,
            self.top_h,
            self.gate_confidence,
        )

    def eligible(
        self,
        *,
        reference_clean_amota: float,
        reference_clean_hota: float,
        clean_noninferiority_margin: float,
        primary_budget_bytes_per_second: int,
    ) -> bool:
        """Apply the frozen clean-link and on-wire budget gate."""

        return (
            self.clean_amota
            >= reference_clean_amota + clean_noninferiority_margin
            and self.clean_hota
            >= reference_clean_hota + clean_noninferiority_margin
            and self.actual_bytes_per_second
            <= float(primary_budget_bytes_per_second)
        )

    def to_primitive(self) -> dict[str, object]:
        return {
            "actual_bytes_per_second": self.actual_bytes_per_second,
            "clean_amota": self.clean_amota,
            "clean_hota": self.clean_hota,
            "config_sha256": self.config_sha256,
            "evidence_sha256": self.evidence_sha256,
            "fixed_lag_seconds": self.fixed_lag_seconds,
            "gate_confidence": self.gate_confidence,
            "mixture_components": self.mixture_components,
            "oof_checkpoint_inventory_sha256": (
                self.oof_checkpoint_inventory_sha256
            ),
            "p95_latency_ms": self.p95_latency_ms,
            "robust_assa_at_64k": self.robust_assa_at_64k,
            "top_h": self.top_h,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "CandidateSelectionEntryV1":
        result = _strict_fields(value, cls._FIELDS, "CandidateSelectionEntryV1")
        return cls(
            fixed_lag_seconds=result["fixed_lag_seconds"],
            mixture_components=result["mixture_components"],
            top_h=result["top_h"],
            gate_confidence=result["gate_confidence"],
            config_sha256=result["config_sha256"],
            clean_amota=result["clean_amota"],
            clean_hota=result["clean_hota"],
            robust_assa_at_64k=result["robust_assa_at_64k"],
            actual_bytes_per_second=result["actual_bytes_per_second"],
            p95_latency_ms=result["p95_latency_ms"],
            oof_checkpoint_inventory_sha256=result[
                "oof_checkpoint_inventory_sha256"
            ],
            evidence_sha256=result["evidence_sha256"],
        )


@dataclass(frozen=True, slots=True)
class CandidateSelectionRegistryV1:
    """Complete, canonical, fail-closed validation selection registry."""

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
    wire_accounting_config_sha256: str
    evaluator_contract_sha256: str
    validation_result_registry_sha256: str
    baseline_qualification_registry_sha256: str
    strongest_baseline_method_id: str
    reference_clean_amota: float
    reference_clean_hota: float
    method_id: str
    scheduler_id: str
    training_seeds: tuple[int, ...]
    network_seeds: tuple[int, ...]
    primary_budget_bytes_per_second: int
    clean_noninferiority_margin: float
    entries: tuple[CandidateSelectionEntryV1, ...]
    selected_method_config_sha256: str
    selection_rule: str = CANDIDATE_SELECTION_RULE_V1

    def __post_init__(self) -> None:
        for name in (
            "registry_id",
            "dataset_id",
            "split_name",
            "strongest_baseline_method_id",
            "method_id",
            "scheduler_id",
            "selection_rule",
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
            "wire_accounting_config_sha256",
            "evaluator_contract_sha256",
            "validation_result_registry_sha256",
            "baseline_qualification_registry_sha256",
            "selected_method_config_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))

        sequence_ids = _canonical_ids(self.sequence_ids, "sequence_ids")
        class_names = _canonical_ids(self.class_names, "class_names")
        training_seeds = _positive_ints(self.training_seeds, "training_seeds")
        network_seeds = _positive_ints(self.network_seeds, "network_seeds")
        object.__setattr__(self, "sequence_ids", sequence_ids)
        object.__setattr__(self, "class_names", class_names)
        object.__setattr__(self, "training_seeds", training_seeds)
        object.__setattr__(self, "network_seeds", network_seeds)

        if self.registry_id != "eventtrack-v2x-candidate-selection-v1":
            raise CandidateSelectionRegistryError(
                "unexpected candidate-selection registry_id"
            )
        if self.dataset_id != "v2x-seq-spd" or self.split_name != "train":
            raise CandidateSelectionRegistryError(
                "candidate selection requires V2X-Seq-SPD train OOF"
            )
        if len(sequence_ids) != 46:
            raise CandidateSelectionRegistryError(
                "candidate selection requires exactly 46 development sequences"
            )
        frequency = _number(self.frequency_hz, "frequency_hz")
        if frequency != 10.0:
            raise CandidateSelectionRegistryError("frequency_hz must equal 10")
        object.__setattr__(self, "frequency_hz", frequency)
        if class_names != ("car",):
            raise CandidateSelectionRegistryError("class_names must equal ('car',)")
        if self.method_id != "eventtrack-v2x":
            raise CandidateSelectionRegistryError(
                "candidate selection method_id must equal eventtrack-v2x"
            )
        if self.scheduler_id != "marginal_voi":
            raise CandidateSelectionRegistryError(
                "candidate selection scheduler_id must equal marginal_voi"
            )
        if training_seeds != CANDIDATE_TRAINING_SEEDS_V1:
            raise CandidateSelectionRegistryError(
                "training_seeds must equal 1337, 2027, and 3407"
            )
        if network_seeds != CANDIDATE_NETWORK_SEEDS_V1:
            raise CandidateSelectionRegistryError(
                "network_seeds must equal 1001 through 1010"
            )
        if (
            type(self.primary_budget_bytes_per_second) is not int
            or self.primary_budget_bytes_per_second
            != CANDIDATE_PRIMARY_BUDGET_BPS_V1
        ):
            raise CandidateSelectionRegistryError(
                "primary_budget_bytes_per_second must equal 64000"
            )
        margin = _number(
            self.clean_noninferiority_margin, "clean_noninferiority_margin"
        )
        if margin != CANDIDATE_CLEAN_NONINFERIORITY_MARGIN_V1:
            raise CandidateSelectionRegistryError(
                "clean_noninferiority_margin must equal -0.01"
            )
        object.__setattr__(self, "clean_noninferiority_margin", margin)
        for name in ("reference_clean_amota", "reference_clean_hota"):
            object.__setattr__(
                self, name, _unit_interval(getattr(self, name), name)
            )
        if self.selection_rule != CANDIDATE_SELECTION_RULE_V1:
            raise CandidateSelectionRegistryError(
                "candidate selection rule is not frozen V1"
            )

        entries = self._validate_entries(self.entries)
        object.__setattr__(self, "entries", entries)
        winner = self._winner(entries)
        if self.selected_method_config_sha256 != winner.config_sha256:
            raise CandidateSelectionRegistryError(
                "selected_method_config_sha256 violates the frozen development rule"
            )

    def _validate_entries(
        self, value: object
    ) -> tuple[CandidateSelectionEntryV1, ...]:
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, CandidateSelectionEntryV1) for item in value
        ):
            raise TypeError(
                "entries must contain CandidateSelectionEntryV1 objects"
            )
        result = tuple(value)
        combinations = tuple(item.combination for item in result)
        if combinations != CANDIDATE_COMBINATIONS_V1:
            raise CandidateSelectionRegistryError(
                "entries must contain the exact 144 frozen hyperparameter "
                "combinations in canonical order"
            )
        config_digests = tuple(item.config_sha256 for item in result)
        evidence_digests = tuple(item.evidence_sha256 for item in result)
        if len(set(config_digests)) != len(config_digests):
            raise CandidateSelectionRegistryError(
                "every hyperparameter combination must have a distinct config SHA-256"
            )
        if len(set(evidence_digests)) != len(evidence_digests):
            raise CandidateSelectionRegistryError(
                "every hyperparameter combination must have distinct evidence"
            )
        checkpoint_inventories = tuple(
            item.oof_checkpoint_inventory_sha256 for item in result
        )
        if len(set(checkpoint_inventories)) != len(checkpoint_inventories):
            raise CandidateSelectionRegistryError(
                "every hyperparameter combination must bind a distinct OOF "
                "checkpoint inventory"
            )
        return result

    def _winner(
        self, entries: tuple[CandidateSelectionEntryV1, ...]
    ) -> CandidateSelectionEntryV1:
        eligible = tuple(
            item
            for item in entries
            if item.eligible(
                reference_clean_amota=self.reference_clean_amota,
                reference_clean_hota=self.reference_clean_hota,
                clean_noninferiority_margin=self.clean_noninferiority_margin,
                primary_budget_bytes_per_second=(
                    self.primary_budget_bytes_per_second
                ),
            )
        )
        if not eligible:
            raise CandidateSelectionRegistryError(
                "no candidate satisfies clean non-inferiority and the 64k budget"
            )
        return min(
            eligible,
            key=lambda item: (
                -item.robust_assa_at_64k,
                item.actual_bytes_per_second,
                item.p95_latency_ms,
                item.combination,
            ),
        )

    @property
    def selected_entry(self) -> CandidateSelectionEntryV1:
        return next(
            item
            for item in self.entries
            if item.config_sha256 == self.selected_method_config_sha256
        )

    def payload(self) -> dict[str, object]:
        return {
            "baseline_qualification_registry_sha256": (
                self.baseline_qualification_registry_sha256
            ),
            "class_names": list(self.class_names),
            "clean_noninferiority_margin": self.clean_noninferiority_margin,
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
            "entries": [item.to_primitive() for item in self.entries],
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "frequency_hz": self.frequency_hz,
            "method_id": self.method_id,
            "network_seeds": list(self.network_seeds),
            "network_trace_manifest_sha256": self.network_trace_manifest_sha256,
            "primary_budget_bytes_per_second": (
                self.primary_budget_bytes_per_second
            ),
            "reference_clean_amota": self.reference_clean_amota,
            "reference_clean_hota": self.reference_clean_hota,
            "registry_id": self.registry_id,
            "scheduler_id": self.scheduler_id,
            "schema_version": 1,
            "selected_method_config_sha256": (
                self.selected_method_config_sha256
            ),
            "selection_rule": self.selection_rule,
            "sequence_ids": list(self.sequence_ids),
            "source_tree_sha256": self.source_tree_sha256,
            "split_name": self.split_name,
            "split_sha256": self.split_sha256,
            "strongest_baseline_method_id": self.strongest_baseline_method_id,
            "training_seeds": list(self.training_seeds),
            "validation_result_registry_sha256": (
                self.validation_result_registry_sha256
            ),
            "wire_accounting_config_sha256": (
                self.wire_accounting_config_sha256
            ),
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


_REGISTRY_PAYLOAD_FIELDS = frozenset(
    {
        "baseline_qualification_registry_sha256",
        "class_names",
        "clean_noninferiority_margin",
        "cohort_sha256",
        "dataset_id",
        "dataset_manifest_sha256",
        "detection_cache_sha256",
        "detection_frame_contract_sha256",
        "development_fold_manifest_sha256",
        "entries",
        "evaluator_contract_sha256",
        "frequency_hz",
        "method_id",
        "network_seeds",
        "network_trace_manifest_sha256",
        "primary_budget_bytes_per_second",
        "reference_clean_amota",
        "reference_clean_hota",
        "registry_id",
        "scheduler_id",
        "schema_version",
        "selected_method_config_sha256",
        "selection_rule",
        "sequence_ids",
        "source_tree_sha256",
        "split_name",
        "split_sha256",
        "strongest_baseline_method_id",
        "training_seeds",
        "validation_result_registry_sha256",
        "wire_accounting_config_sha256",
    }
)


def candidate_selection_registry_from_document(
    value: object,
) -> CandidateSelectionRegistryV1:
    """Validate an exact sealed ``CandidateSelectionRegistryV1`` document."""

    result = _strict_fields(
        value,
        _REGISTRY_PAYLOAD_FIELDS | {"content_sha256"},
        "CandidateSelectionRegistryV1",
    )
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise CandidateSelectionRegistryError(
            "unsupported candidate-selection registry schema version"
        )
    observed = _sha256(result["content_sha256"], "content_sha256")
    raw_payload = {name: result[name] for name in _REGISTRY_PAYLOAD_FIELDS}
    expected = hashlib.sha256(canonical_json_bytes(raw_payload)).hexdigest()
    if observed != expected:
        raise CandidateSelectionRegistryError(
            "candidate-selection registry content SHA-256 mismatch"
        )
    raw_entries = result["entries"]
    if not isinstance(raw_entries, list):
        raise CandidateSelectionRegistryError("entries must be an array")
    registry = CandidateSelectionRegistryV1(
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
        wire_accounting_config_sha256=result[
            "wire_accounting_config_sha256"
        ],
        evaluator_contract_sha256=result["evaluator_contract_sha256"],
        validation_result_registry_sha256=result[
            "validation_result_registry_sha256"
        ],
        baseline_qualification_registry_sha256=result[
            "baseline_qualification_registry_sha256"
        ],
        strongest_baseline_method_id=result["strongest_baseline_method_id"],
        reference_clean_amota=result["reference_clean_amota"],
        reference_clean_hota=result["reference_clean_hota"],
        method_id=result["method_id"],
        scheduler_id=result["scheduler_id"],
        training_seeds=result["training_seeds"],
        network_seeds=result["network_seeds"],
        primary_budget_bytes_per_second=result[
            "primary_budget_bytes_per_second"
        ],
        clean_noninferiority_margin=result["clean_noninferiority_margin"],
        entries=tuple(
            CandidateSelectionEntryV1.from_mapping(item) for item in raw_entries
        ),
        selected_method_config_sha256=result[
            "selected_method_config_sha256"
        ],
        selection_rule=result["selection_rule"],
    )
    if observed != registry.content_sha256:
        raise CandidateSelectionRegistryError(
            "candidate-selection registry canonical payload mismatch"
        )
    return registry


def decode_candidate_selection_registry(
    data: bytes,
) -> CandidateSelectionRegistryV1:
    """Decode canonical UTF-8 JSON, rejecting duplicates and non-finite values."""

    if type(data) is not bytes:
        raise TypeError("candidate-selection registry data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise CandidateSelectionRegistryError(
                    f"duplicate JSON key: {key}"
                )
            result[key] = value
        return result

    def invalid_constant(value: str) -> object:
        raise CandidateSelectionRegistryError(
            f"non-finite JSON number is forbidden: {value}"
        )

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateSelectionRegistryError(
            "invalid candidate-selection registry JSON"
        ) from exc
    registry = candidate_selection_registry_from_document(value)
    if registry.canonical_bytes() != data:
        raise CandidateSelectionRegistryError(
            "candidate-selection registry is not canonical JSON"
        )
    return registry


__all__ = [
    "CANDIDATE_CLEAN_NONINFERIORITY_MARGIN_V1",
    "CANDIDATE_COMBINATIONS_V1",
    "CANDIDATE_FIXED_LAG_SECONDS_V1",
    "CANDIDATE_GATE_CONFIDENCE_V1",
    "CANDIDATE_MIXTURE_COMPONENTS_V1",
    "CANDIDATE_NETWORK_SEEDS_V1",
    "CANDIDATE_PRIMARY_BUDGET_BPS_V1",
    "CANDIDATE_SELECTION_RULE_V1",
    "CANDIDATE_TOP_H_V1",
    "CANDIDATE_TRAINING_SEEDS_V1",
    "CandidateSelectionEntryV1",
    "CandidateSelectionRegistryError",
    "CandidateSelectionRegistryV1",
    "candidate_selection_registry_from_document",
    "decode_candidate_selection_registry",
]
