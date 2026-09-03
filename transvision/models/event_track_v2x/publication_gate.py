"""Fail-closed preregistered gate for EventTrack-V2X SOTA wording.

The gate consumes one canonical metrics artifact. It does not accept loose
score dictionaries, cohort registries, baseline registries, or statistical
settings from its caller; those identities are sealed in the plan and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping

import numpy as np

from .contracts import ArtifactDigestV1, EvidenceBundleV1
from .experiment import ExperimentPlanV1
from .statistics import (
    PairedBootstrapResultV1,
    PairedPermutationResultV1,
    holm_correction_v1,
    paired_sequence_bca_v1,
    paired_sequence_permutation_v1,
)
from .wire import canonical_json_bytes


REQUIRED_INVARIANTS_V1 = (
    "budget_compliance",
    "cohort_identity",
    "deterministic_replay",
    "duplicate_message_idempotence",
    "evaluator_parity",
    "immutable_commit_prefix",
    "no_future_information",
    "no_nan_or_inf",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise ValueError(f"{name} must be a canonical identifier")
    return value


def _strict_fields(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise ValueError(f"{name} must be a string-keyed object")
    if set(value) != set(expected):
        raise ValueError(f"{name} has missing or unknown fields")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class SequenceScoresV1:
    """One metric value for every sequence in a canonical ordered cohort."""

    sequence_scores: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_scores, (list, tuple)):
            raise TypeError("sequence_scores must be an array")
        scores: list[tuple[str, float]] = []
        for index, item in enumerate(self.sequence_scores):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("sequence score entries must be two-value arrays")
            scores.append(
                (
                    _identifier(item[0], f"sequence_scores[{index}].sequence_id"),
                    _finite(item[1], f"sequence_scores[{index}].value"),
                )
            )
        if not scores or len({item[0] for item in scores}) != len(scores):
            raise ValueError("sequence_scores must be non-empty and unique")
        if scores != sorted(scores, key=lambda item: item[0]):
            raise ValueError("sequence_scores must be sorted by sequence_id")
        object.__setattr__(self, "sequence_scores", tuple(scores))

    @property
    def sequence_ids(self) -> tuple[str, ...]:
        return tuple(sequence_id for sequence_id, _ in self.sequence_scores)

    def as_mapping(self) -> dict[str, float]:
        return dict(self.sequence_scores)

    def to_primitive(self) -> list[list[str | float]]:
        return [[sequence_id, value] for sequence_id, value in self.sequence_scores]

    @classmethod
    def from_primitive(cls, value: object) -> "SequenceScoresV1":
        if not isinstance(value, list):
            raise ValueError("sequence score series must be an array")
        return cls(tuple(tuple(item) for item in value))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class MethodScoresV1:
    method_id: str
    scores: SequenceScoresV1

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", _identifier(self.method_id, "method_id"))
        if not isinstance(self.scores, SequenceScoresV1):
            raise TypeError("scores must be SequenceScoresV1")

    def to_primitive(self) -> dict[str, object]:
        return {"method_id": self.method_id, "scores": self.scores.to_primitive()}

    @classmethod
    def from_mapping(cls, value: object) -> "MethodScoresV1":
        result = _strict_fields(
            value, frozenset({"method_id", "scores"}), cls.__name__
        )
        return cls(
            method_id=result["method_id"],
            scores=SequenceScoresV1.from_primitive(result["scores"]),
        )


@dataclass(frozen=True, slots=True)
class NamedMetricScoresV1:
    metric_id: str
    scores: SequenceScoresV1

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_id", _identifier(self.metric_id, "metric_id"))
        if not isinstance(self.scores, SequenceScoresV1):
            raise TypeError("scores must be SequenceScoresV1")

    def to_primitive(self) -> dict[str, object]:
        return {"metric_id": self.metric_id, "scores": self.scores.to_primitive()}

    @classmethod
    def from_mapping(cls, value: object) -> "NamedMetricScoresV1":
        result = _strict_fields(
            value, frozenset({"metric_id", "scores"}), cls.__name__
        )
        return cls(
            metric_id=result["metric_id"],
            scores=SequenceScoresV1.from_primitive(result["scores"]),
        )


@dataclass(frozen=True, slots=True)
class BudgetedSequenceScoresV1:
    """Per-sequence accuracy at every sealed on-wire byte budget."""

    scheduler_id: str
    byte_budgets_per_second: tuple[int, ...]
    sequence_scores: tuple[tuple[str, tuple[float, ...]], ...]
    sequence_on_wire_bps: tuple[tuple[str, tuple[float, ...]], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scheduler_id", _identifier(self.scheduler_id, "scheduler_id")
        )
        if not isinstance(self.byte_budgets_per_second, (list, tuple)):
            raise TypeError("byte_budgets_per_second must be an array")
        budgets: list[int] = []
        for value in self.byte_budgets_per_second:
            if type(value) is not int or value <= 0:
                raise ValueError("byte budgets must be positive integers")
            budgets.append(value)
        if len(budgets) < 2 or budgets != sorted(set(budgets)):
            raise ValueError("byte budgets must be unique and strictly increasing")
        object.__setattr__(self, "byte_budgets_per_second", tuple(budgets))
        if not isinstance(self.sequence_scores, (list, tuple)):
            raise TypeError("budgeted sequence_scores must be an array")
        rows: list[tuple[str, tuple[float, ...]]] = []
        for index, item in enumerate(self.sequence_scores):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("budgeted score entries must be two-value arrays")
            if not isinstance(item[1], (list, tuple)):
                raise TypeError("budgeted score values must be an array")
            values = tuple(
                _finite(score, f"sequence_scores[{index}].values")
                for score in item[1]
            )
            if len(values) != len(budgets):
                raise ValueError("every budgeted score row must cover every budget")
            rows.append(
                (
                    _identifier(item[0], f"sequence_scores[{index}].sequence_id"),
                    values,
                )
            )
        if not rows or len({item[0] for item in rows}) != len(rows):
            raise ValueError("budgeted sequence scores must be non-empty and unique")
        if rows != sorted(rows, key=lambda item: item[0]):
            raise ValueError("budgeted sequence scores must be sorted by sequence_id")
        object.__setattr__(self, "sequence_scores", tuple(rows))
        if not isinstance(self.sequence_on_wire_bps, (list, tuple)):
            raise TypeError("sequence_on_wire_bps must be an array")
        wire_rows: list[tuple[str, tuple[float, ...]]] = []
        for index, item in enumerate(self.sequence_on_wire_bps):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("on-wire rows must be two-value arrays")
            if not isinstance(item[1], (list, tuple)):
                raise TypeError("on-wire values must be an array")
            values = tuple(
                _finite(value, f"sequence_on_wire_bps[{index}].values")
                for value in item[1]
            )
            if len(values) != len(budgets):
                raise ValueError("every on-wire row must cover every budget")
            if any(value < 0.0 for value in values):
                raise ValueError("actual on-wire BPS must be non-negative")
            if any(value > budget + 1e-9 for value, budget in zip(values, budgets)):
                raise ValueError("actual on-wire BPS exceeds its nominal budget")
            wire_rows.append(
                (
                    _identifier(item[0], f"sequence_on_wire_bps[{index}].sequence_id"),
                    values,
                )
            )
        if tuple(item[0] for item in wire_rows) != tuple(
            item[0] for item in rows
        ):
            raise ValueError("score and on-wire rows must use the same sequence cohort")
        object.__setattr__(self, "sequence_on_wire_bps", tuple(wire_rows))

    @property
    def sequence_ids(self) -> tuple[str, ...]:
        return tuple(sequence_id for sequence_id, _ in self.sequence_scores)

    def auc_by_sequence(self) -> dict[str, float]:
        scores = dict(self.sequence_scores)
        result: dict[str, float] = {}
        for sequence_id, actual_bps in self.sequence_on_wire_bps:
            x, y = _pareto_front(actual_bps, scores[sequence_id])
            width = float(x[-1] - x[0])
            if width <= 0.0:
                raise ValueError("actual on-wire Pareto curve has no positive width")
            result[sequence_id] = float(np.trapezoid(y, x) / width)
        return result

    def to_primitive(self) -> dict[str, object]:
        return {
            "byte_budgets_per_second": list(self.byte_budgets_per_second),
            "scheduler_id": self.scheduler_id,
            "sequence_on_wire_bps": [
                [sequence_id, list(values)]
                for sequence_id, values in self.sequence_on_wire_bps
            ],
            "sequence_scores": [
                [sequence_id, list(values)]
                for sequence_id, values in self.sequence_scores
            ],
        }

    @classmethod
    def from_mapping(cls, value: object) -> "BudgetedSequenceScoresV1":
        result = _strict_fields(
            value,
            frozenset(
                {
                    "byte_budgets_per_second",
                    "scheduler_id",
                    "sequence_on_wire_bps",
                    "sequence_scores",
                }
            ),
            cls.__name__,
        )
        raw_rows = result["sequence_scores"]
        if not isinstance(raw_rows, list):
            raise ValueError("budgeted sequence_scores must be an array")
        raw_wire_rows = result["sequence_on_wire_bps"]
        if not isinstance(raw_wire_rows, list):
            raise ValueError("sequence_on_wire_bps must be an array")
        return cls(
            scheduler_id=result["scheduler_id"],
            byte_budgets_per_second=result["byte_budgets_per_second"],
            sequence_scores=tuple(
                (item[0], tuple(item[1])) for item in raw_rows  # type: ignore[index]
            ),
            sequence_on_wire_bps=tuple(
                (item[0], tuple(item[1]))  # type: ignore[index]
                for item in raw_wire_rows
            ),
        )


def _pareto_front(
    actual_bps: tuple[float, ...], scores: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the non-dominated upper envelope in actual-byte coordinates."""

    by_x: dict[float, float] = {}
    for x_value, score in zip(actual_bps, scores):
        by_x[x_value] = max(by_x.get(x_value, -float("inf")), score)
    ordered = sorted(by_x.items())
    x = np.asarray([item[0] for item in ordered], dtype=np.float64)
    y = np.maximum.accumulate(
        np.asarray([item[1] for item in ordered], dtype=np.float64)
    )
    return x, y


def paired_pareto_auc_by_sequence_v1(
    treatment: BudgetedSequenceScoresV1,
    control: BudgetedSequenceScoresV1,
) -> tuple[dict[str, float], dict[str, float]]:
    """Integrate both curves on the same actual-BPS support per sequence."""

    if treatment.sequence_ids != control.sequence_ids:
        raise ValueError("paired Pareto curves must use the same sequence cohort")
    treatment_scores = dict(treatment.sequence_scores)
    treatment_wire = dict(treatment.sequence_on_wire_bps)
    control_scores = dict(control.sequence_scores)
    control_wire = dict(control.sequence_on_wire_bps)
    treatment_auc: dict[str, float] = {}
    control_auc: dict[str, float] = {}
    for sequence_id in treatment.sequence_ids:
        tx, ty = _pareto_front(
            treatment_wire[sequence_id], treatment_scores[sequence_id]
        )
        cx, cy = _pareto_front(control_wire[sequence_id], control_scores[sequence_id])
        lower = max(float(tx[0]), float(cx[0]))
        upper = min(float(tx[-1]), float(cx[-1]))
        if upper <= lower:
            raise ValueError(
                f"paired Pareto curves have no common actual-BPS width: {sequence_id}"
            )
        grid = np.unique(
            np.concatenate(
                (
                    np.asarray([lower, upper]),
                    tx[(tx > lower) & (tx < upper)],
                    cx[(cx > lower) & (cx < upper)],
                )
            )
        )
        width = upper - lower
        treatment_auc[sequence_id] = float(
            np.trapezoid(np.interp(grid, tx, ty), grid) / width
        )
        control_auc[sequence_id] = float(
            np.trapezoid(np.interp(grid, cx, cy), grid) / width
        )
    return treatment_auc, control_auc


@dataclass(frozen=True, slots=True)
class PublicationMetricsV1:
    """Canonical per-sequence evidence consumed by the publication gate."""

    experiment_plan_sha256: str
    candidate_method_id: str
    robust_metric_id: str
    robust_eventtrack: SequenceScoresV1
    robust_baselines: tuple[MethodScoresV1, ...]
    clean_eventtrack: tuple[NamedMetricScoresV1, ...]
    clean_strongest_baseline_id: str
    clean_strongest_baseline: tuple[NamedMetricScoresV1, ...]
    pareto_metric_id: str
    pareto_eventtrack: BudgetedSequenceScoresV1
    pareto_strongest_baseline: BudgetedSequenceScoresV1
    griffin_metric_id: str
    griffin_eventtrack: SequenceScoresV1
    griffin_strongest_baseline_id: str
    griffin_strongest_baseline: SequenceScoresV1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_plan_sha256",
            _sha256(self.experiment_plan_sha256, "experiment_plan_sha256"),
        )
        for name in (
            "candidate_method_id",
            "robust_metric_id",
            "clean_strongest_baseline_id",
            "pareto_metric_id",
            "griffin_metric_id",
            "griffin_strongest_baseline_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "robust_eventtrack",
            "griffin_eventtrack",
            "griffin_strongest_baseline",
        ):
            if not isinstance(getattr(self, name), SequenceScoresV1):
                raise TypeError(f"{name} must be SequenceScoresV1")
        for name, item_type, identifier_name in (
            ("robust_baselines", MethodScoresV1, "method_id"),
            ("clean_eventtrack", NamedMetricScoresV1, "metric_id"),
            ("clean_strongest_baseline", NamedMetricScoresV1, "metric_id"),
        ):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values or not all(
                isinstance(item, item_type) for item in values
            ):
                raise TypeError(
                    f"{name} must be a non-empty array of {item_type.__name__}"
                )
            identifiers = [getattr(item, identifier_name) for item in values]
            if identifiers != sorted(set(identifiers)):
                raise ValueError(f"{name} must have unique, sorted identifiers")
            object.__setattr__(self, name, tuple(values))
        for name in ("pareto_eventtrack", "pareto_strongest_baseline"):
            if not isinstance(getattr(self, name), BudgetedSequenceScoresV1):
                raise TypeError(f"{name} must be BudgetedSequenceScoresV1")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "candidate_method_id": self.candidate_method_id,
            "clean_eventtrack": [item.to_primitive() for item in self.clean_eventtrack],
            "clean_strongest_baseline": [
                item.to_primitive() for item in self.clean_strongest_baseline
            ],
            "clean_strongest_baseline_id": self.clean_strongest_baseline_id,
            "experiment_plan_sha256": self.experiment_plan_sha256,
            "griffin_eventtrack": self.griffin_eventtrack.to_primitive(),
            "griffin_metric_id": self.griffin_metric_id,
            "griffin_strongest_baseline": (
                self.griffin_strongest_baseline.to_primitive()
            ),
            "griffin_strongest_baseline_id": self.griffin_strongest_baseline_id,
            "kind": "publication_metrics_v1",
            "pareto_eventtrack": self.pareto_eventtrack.to_primitive(),
            "pareto_metric_id": self.pareto_metric_id,
            "pareto_strongest_baseline": (
                self.pareto_strongest_baseline.to_primitive()
            ),
            "robust_baselines": [item.to_primitive() for item in self.robust_baselines],
            "robust_eventtrack": self.robust_eventtrack.to_primitive(),
            "robust_metric_id": self.robust_metric_id,
            "schema_version": 1,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_primitive())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> "PublicationMetricsV1":
        fields = frozenset(
            {
                "candidate_method_id",
                "clean_eventtrack",
                "clean_strongest_baseline",
                "clean_strongest_baseline_id",
                "experiment_plan_sha256",
                "griffin_eventtrack",
                "griffin_metric_id",
                "griffin_strongest_baseline",
                "griffin_strongest_baseline_id",
                "kind",
                "pareto_eventtrack",
                "pareto_metric_id",
                "pareto_strongest_baseline",
                "robust_baselines",
                "robust_eventtrack",
                "robust_metric_id",
                "schema_version",
            }
        )
        result = _strict_fields(value, fields, cls.__name__)
        if result["kind"] != "publication_metrics_v1" or result["schema_version"] != 1:
            raise ValueError("unsupported PublicationMetricsV1 schema header")
        for name in (
            "robust_baselines",
            "clean_eventtrack",
            "clean_strongest_baseline",
        ):
            if not isinstance(result[name], list):
                raise ValueError(f"{name} must be an array")
        return cls(
            experiment_plan_sha256=result["experiment_plan_sha256"],
            candidate_method_id=result["candidate_method_id"],
            robust_metric_id=result["robust_metric_id"],
            robust_eventtrack=SequenceScoresV1.from_primitive(
                result["robust_eventtrack"]
            ),
            robust_baselines=tuple(
                MethodScoresV1.from_mapping(item)
                for item in result["robust_baselines"]  # type: ignore[union-attr]
            ),
            clean_eventtrack=tuple(
                NamedMetricScoresV1.from_mapping(item)
                for item in result["clean_eventtrack"]  # type: ignore[union-attr]
            ),
            clean_strongest_baseline_id=result["clean_strongest_baseline_id"],
            clean_strongest_baseline=tuple(
                NamedMetricScoresV1.from_mapping(item)
                for item in result["clean_strongest_baseline"]  # type: ignore[union-attr]
            ),
            pareto_metric_id=result["pareto_metric_id"],
            pareto_eventtrack=BudgetedSequenceScoresV1.from_mapping(
                result["pareto_eventtrack"]
            ),
            pareto_strongest_baseline=BudgetedSequenceScoresV1.from_mapping(
                result["pareto_strongest_baseline"]
            ),
            griffin_metric_id=result["griffin_metric_id"],
            griffin_eventtrack=SequenceScoresV1.from_primitive(
                result["griffin_eventtrack"]
            ),
            griffin_strongest_baseline_id=result[
                "griffin_strongest_baseline_id"
            ],
            griffin_strongest_baseline=SequenceScoresV1.from_primitive(
                result["griffin_strongest_baseline"]
            ),
        )


def decode_publication_metrics(data: bytes) -> PublicationMetricsV1:
    """Decode canonical metrics bytes and reject duplicate JSON keys."""

    if type(data) is not bytes:
        raise TypeError("publication metrics data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid PublicationMetricsV1 JSON") from exc
    metrics = PublicationMetricsV1.from_mapping(value)
    if metrics.canonical_bytes() != data:
        raise ValueError("PublicationMetricsV1 JSON is not canonical")
    return metrics


@dataclass(frozen=True, slots=True)
class InvariantEvidenceV1:
    invariant_id: str
    passed: bool
    experiment_plan_sha256: str
    evidence_bundle_sha256: str
    artifact_role: str
    artifact: ArtifactDigestV1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "invariant_id", _identifier(self.invariant_id, "invariant_id")
        )
        if type(self.passed) is not bool:
            raise TypeError("invariant passed must be bool")
        for name in ("experiment_plan_sha256", "evidence_bundle_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(
            self, "artifact_role", _identifier(self.artifact_role, "artifact_role")
        )
        if not isinstance(self.artifact, ArtifactDigestV1):
            raise TypeError("invariant artifact must be ArtifactDigestV1")


@dataclass(frozen=True, slots=True)
class ColdCacheReceiptV1:
    """Independent cold-read receipt for every artifact in EvidenceBundleV1."""

    receipt_id: str
    verified: bool
    experiment_plan_sha256: str
    evidence_bundle_sha256: str
    verified_artifacts: tuple[tuple[str, ArtifactDigestV1], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_id", _identifier(self.receipt_id, "receipt_id"))
        if type(self.verified) is not bool:
            raise TypeError("cold-cache verified must be bool")
        for name in ("experiment_plan_sha256", "evidence_bundle_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if not isinstance(self.verified_artifacts, (list, tuple)):
            raise TypeError("verified_artifacts must be an array")
        artifacts: list[tuple[str, ArtifactDigestV1]] = []
        for item in self.verified_artifacts:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("verified_artifacts entries must be two-value arrays")
            role = _identifier(item[0], "verified artifact role")
            if not isinstance(item[1], ArtifactDigestV1):
                raise TypeError("verified artifact must be ArtifactDigestV1")
            artifacts.append((role, item[1]))
        if not artifacts or [item[0] for item in artifacts] != sorted(
            {item[0] for item in artifacts}
        ):
            raise ValueError("verified_artifacts must have unique, sorted roles")
        object.__setattr__(self, "verified_artifacts", tuple(artifacts))

    def to_primitive(self) -> dict[str, object]:
        return {
            "evidence_bundle_sha256": self.evidence_bundle_sha256,
            "experiment_plan_sha256": self.experiment_plan_sha256,
            "receipt_id": self.receipt_id,
            "verified": self.verified,
            "verified_artifacts": [
                [role, artifact.to_primitive()]
                for role, artifact in self.verified_artifacts
            ],
        }

    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_primitive())).hexdigest()


@dataclass(frozen=True, slots=True)
class ComparisonGateV1:
    comparison: str
    minimum_effect: float
    bootstrap: PairedBootstrapResultV1
    permutation: PairedPermutationResultV1
    holm_adjusted_p_value: float
    passed: bool


@dataclass(frozen=True, slots=True)
class NonInferiorityGateV1:
    metric: str
    margin: float
    bootstrap: PairedBootstrapResultV1
    passed: bool


@dataclass(frozen=True, slots=True)
class PublicationGateResultV1:
    experiment_plan_sha256: str
    evidence_bundle_sha256: str
    metrics_document_sha256: str
    robust_comparisons: tuple[ComparisonGateV1, ...]
    clean_noninferiority: tuple[NonInferiorityGateV1, ...]
    pareto_comparison: ComparisonGateV1
    griffin_comparison: ComparisonGateV1
    invariants: tuple[InvariantEvidenceV1, ...]
    cold_cache_receipt_sha256: str | None
    cold_cache_verified: bool
    sota_wording_allowed: bool
    failed_gates: tuple[str, ...]


def _comparison_inputs(
    treatment: Mapping[str, float],
    control: Mapping[str, float],
    *,
    plan: ExperimentPlanV1,
) -> tuple[PairedBootstrapResultV1, PairedPermutationResultV1]:
    bootstrap = paired_sequence_bca_v1(
        treatment,
        control,
        confidence_level=1.0 - plan.statistical_alpha,
        resamples=plan.statistical_resamples,
        seed=plan.statistical_random_seed,
    )
    permutation = paired_sequence_permutation_v1(
        treatment,
        control,
        alternative="greater",
        permutations=plan.statistical_resamples,
        seed=plan.statistical_random_seed,
    )
    return bootstrap, permutation


def _named_metrics(
    values: tuple[NamedMetricScoresV1, ...],
) -> dict[str, SequenceScoresV1]:
    return {item.metric_id: item.scores for item in values}


def _require_cohort(
    scores: SequenceScoresV1 | BudgetedSequenceScoresV1,
    expected: tuple[str, ...],
    name: str,
) -> None:
    if scores.sequence_ids != expected:
        missing = sorted(set(expected) - set(scores.sequence_ids))
        extra = sorted(set(scores.sequence_ids) - set(expected))
        raise ValueError(
            f"{name} sequence cohort mismatch; missing={missing}, extra={extra}"
        )


def _validate_plan_evidence_parity(
    plan: ExperimentPlanV1, evidence: EvidenceBundleV1
) -> None:
    expected = {
        "source": plan.source_tree_sha256,
        "dataset": plan.dataset_manifest_sha256,
        "detection_cache": plan.detector_cache_sha256,
        "network_trace": plan.network_trace_manifest_sha256,
        "tracker_config": plan.method_config_sha256,
        "evaluator_contract": plan.evaluator_contract_sha256,
    }
    mismatched = [
        role
        for role, sha256 in expected.items()
        if getattr(evidence, role).sha256 != sha256
    ]
    if mismatched:
        raise ValueError(
            f"ExperimentPlanV1/EvidenceBundleV1 artifact mismatch: {mismatched}"
        )


def _validate_bound_invariants(
    invariants: tuple[InvariantEvidenceV1, ...],
    *,
    plan: ExperimentPlanV1,
    evidence: EvidenceBundleV1,
) -> tuple[InvariantEvidenceV1, ...]:
    if not isinstance(invariants, (list, tuple)) or not all(
        isinstance(item, InvariantEvidenceV1) for item in invariants
    ):
        raise TypeError("invariants must be InvariantEvidenceV1 objects")
    ordered = tuple(sorted(invariants, key=lambda item: item.invariant_id))
    if tuple(item.invariant_id for item in ordered) != REQUIRED_INVARIANTS_V1:
        raise ValueError("invariant set does not match preregistration")
    evidence_sha256 = evidence.digest()
    for item in ordered:
        if (
            item.experiment_plan_sha256 != plan.content_sha256
            or item.evidence_bundle_sha256 != evidence_sha256
        ):
            raise ValueError(
                f"invariant {item.invariant_id} has stale plan/evidence binding"
            )
        if item.artifact_role not in EvidenceBundleV1._ARTIFACT_FIELDS:
            raise ValueError(f"invariant {item.invariant_id} has unknown artifact role")
        if item.artifact != getattr(evidence, item.artifact_role):
            raise ValueError(f"invariant {item.invariant_id} artifact parity failed")
    return ordered


def _validate_cold_cache_receipt(
    receipt: ColdCacheReceiptV1 | None,
    *,
    plan: ExperimentPlanV1,
    evidence: EvidenceBundleV1,
) -> bool:
    if receipt is None:
        return False
    if not isinstance(receipt, ColdCacheReceiptV1):
        raise TypeError("cold_cache_receipt must be ColdCacheReceiptV1 or None")
    if (
        receipt.experiment_plan_sha256 != plan.content_sha256
        or receipt.evidence_bundle_sha256 != evidence.digest()
    ):
        raise ValueError("cold-cache receipt has stale plan/evidence binding")
    observed = dict(receipt.verified_artifacts)
    if set(observed) != set(EvidenceBundleV1._ARTIFACT_FIELDS):
        raise ValueError("cold-cache receipt does not cover every evidence artifact")
    for role in EvidenceBundleV1._ARTIFACT_FIELDS:
        if observed[role] != getattr(evidence, role):
            raise ValueError(f"cold-cache artifact parity failed for {role}")
    return receipt.verified


def _evaluate_publication_gate_v1(
    *,
    experiment_plan: ExperimentPlanV1,
    evidence_bundle: EvidenceBundleV1,
    publication_metrics: PublicationMetricsV1,
    invariants: tuple[InvariantEvidenceV1, ...],
    cold_cache_receipt: ColdCacheReceiptV1 | None,
    provenance_verified: bool,
) -> PublicationGateResultV1:
    """Evaluate preregistered arithmetic after explicit provenance handling."""

    if not isinstance(experiment_plan, ExperimentPlanV1):
        raise TypeError("experiment_plan must be ExperimentPlanV1")
    if not isinstance(evidence_bundle, EvidenceBundleV1):
        raise TypeError("evidence_bundle must be EvidenceBundleV1")
    if not isinstance(publication_metrics, PublicationMetricsV1):
        raise TypeError("publication_metrics must be PublicationMetricsV1")
    _validate_plan_evidence_parity(experiment_plan, evidence_bundle)
    if publication_metrics.experiment_plan_sha256 != experiment_plan.content_sha256:
        raise ValueError("PublicationMetricsV1 is not bound to ExperimentPlanV1")
    if publication_metrics.digest() != evidence_bundle.per_sequence_metrics.sha256:
        raise ValueError("PublicationMetricsV1 bytes are not bound by EvidenceBundleV1")
    if publication_metrics.candidate_method_id != "eventtrack-v2x":
        raise ValueError("unexpected publication candidate method")
    if publication_metrics.robust_metric_id != experiment_plan.primary_metric:
        raise ValueError("robust metric does not match ExperimentPlanV1")

    robust_baselines = {
        item.method_id: item.scores for item in publication_metrics.robust_baselines
    }
    if tuple(sorted(robust_baselines)) != tuple(
        sorted(experiment_plan.qualified_baseline_ids)
    ):
        raise ValueError("robust baselines do not match the sealed qualified registry")
    _require_cohort(
        publication_metrics.robust_eventtrack,
        experiment_plan.primary_sequence_ids,
        "robust EventTrack",
    )
    for baseline, scores in robust_baselines.items():
        _require_cohort(scores, experiment_plan.primary_sequence_ids, baseline)

    clean_eventtrack = _named_metrics(publication_metrics.clean_eventtrack)
    clean_baseline = _named_metrics(publication_metrics.clean_strongest_baseline)
    if set(clean_eventtrack) != {"AMOTA", "HOTA"} or set(clean_baseline) != {
        "AMOTA",
        "HOTA",
    }:
        raise ValueError("clean non-inferiority requires exactly AMOTA and HOTA")
    if (
        publication_metrics.clean_strongest_baseline_id
        != experiment_plan.strongest_qualified_baseline_id
    ):
        raise ValueError("clean strongest baseline does not match ExperimentPlanV1")
    for metric_id in ("AMOTA", "HOTA"):
        _require_cohort(
            clean_eventtrack[metric_id],
            experiment_plan.primary_sequence_ids,
            f"clean {metric_id} EventTrack",
        )
        _require_cohort(
            clean_baseline[metric_id],
            experiment_plan.primary_sequence_ids,
            f"clean {metric_id} baseline",
        )

    if publication_metrics.pareto_metric_id != "AssA":
        raise ValueError("Pareto metric must be AssA")
    if (
        publication_metrics.pareto_eventtrack.scheduler_id
        != experiment_plan.candidate_scheduler_id
        or publication_metrics.pareto_strongest_baseline.scheduler_id
        != experiment_plan.strongest_scheduler_baseline_id
    ):
        raise ValueError("Pareto scheduler identities do not match ExperimentPlanV1")
    for name, scores in (
        ("Pareto EventTrack", publication_metrics.pareto_eventtrack),
        ("Pareto strongest baseline", publication_metrics.pareto_strongest_baseline),
    ):
        if scores.byte_budgets_per_second != experiment_plan.byte_budgets_per_second:
            raise ValueError(f"{name} does not cover the five sealed byte budgets")
        _require_cohort(scores, experiment_plan.primary_sequence_ids, name)

    if publication_metrics.griffin_metric_id != experiment_plan.primary_metric:
        raise ValueError("Griffin metric does not match ExperimentPlanV1")
    if (
        publication_metrics.griffin_strongest_baseline_id
        != experiment_plan.griffin_strongest_baseline_id
    ):
        raise ValueError("Griffin strongest baseline does not match ExperimentPlanV1")
    _require_cohort(
        publication_metrics.griffin_eventtrack,
        experiment_plan.griffin_sequence_ids,
        "Griffin EventTrack",
    )
    _require_cohort(
        publication_metrics.griffin_strongest_baseline,
        experiment_plan.griffin_sequence_ids,
        "Griffin strongest baseline",
    )

    bound_invariants = _validate_bound_invariants(
        invariants, plan=experiment_plan, evidence=evidence_bundle
    )
    cold_cache_verified = _validate_cold_cache_receipt(
        cold_cache_receipt, plan=experiment_plan, evidence=evidence_bundle
    )

    eventtrack_robust = publication_metrics.robust_eventtrack.as_mapping()
    robust_raw_p: dict[str, float] = {}
    robust_intermediate: dict[
        str, tuple[PairedBootstrapResultV1, PairedPermutationResultV1]
    ] = {}
    for baseline in sorted(robust_baselines):
        bootstrap, permutation = _comparison_inputs(
            eventtrack_robust,
            robust_baselines[baseline].as_mapping(),
            plan=experiment_plan,
        )
        robust_intermediate[baseline] = (bootstrap, permutation)
        robust_raw_p[baseline] = permutation.p_value
    corrected = {
        item.comparison: item
        for item in holm_correction_v1(
            robust_raw_p, alpha=experiment_plan.statistical_alpha
        )
    }
    robust_gates = tuple(
        ComparisonGateV1(
            comparison=f"Robust-AssA@64k vs {baseline}",
            minimum_effect=0.01,
            bootstrap=robust_intermediate[baseline][0],
            permutation=robust_intermediate[baseline][1],
            holm_adjusted_p_value=corrected[baseline].adjusted_p_value,
            passed=(
                robust_intermediate[baseline][0].estimate >= 0.01 - 1e-12
                and robust_intermediate[baseline][0].confidence_lower > 0.0
                and corrected[baseline].reject
            ),
        )
        for baseline in sorted(robust_baselines)
    )

    clean_gates = tuple(
        NonInferiorityGateV1(
            metric=metric,
            margin=-0.01,
            bootstrap=(
                interval := paired_sequence_bca_v1(
                    clean_eventtrack[metric].as_mapping(),
                    clean_baseline[metric].as_mapping(),
                    confidence_level=1.0 - experiment_plan.statistical_alpha,
                    resamples=experiment_plan.statistical_resamples,
                    seed=experiment_plan.statistical_random_seed,
                )
            ),
            passed=interval.confidence_lower >= -0.01,
        )
        for metric in ("AMOTA", "HOTA")
    )

    eventtrack_pareto_auc, baseline_pareto_auc = (
        paired_pareto_auc_by_sequence_v1(
            publication_metrics.pareto_eventtrack,
            publication_metrics.pareto_strongest_baseline,
        )
    )
    pareto_bootstrap, pareto_permutation = _comparison_inputs(
        eventtrack_pareto_auc,
        baseline_pareto_auc,
        plan=experiment_plan,
    )
    pareto_gate = ComparisonGateV1(
        comparison="Pareto AUC vs strongest scheduler",
        minimum_effect=0.0,
        bootstrap=pareto_bootstrap,
        permutation=pareto_permutation,
        holm_adjusted_p_value=pareto_permutation.p_value,
        passed=(
            pareto_bootstrap.confidence_lower > 0.0
            and pareto_permutation.p_value < experiment_plan.statistical_alpha
        ),
    )
    griffin_bootstrap, griffin_permutation = _comparison_inputs(
        publication_metrics.griffin_eventtrack.as_mapping(),
        publication_metrics.griffin_strongest_baseline.as_mapping(),
        plan=experiment_plan,
    )
    griffin_gate = ComparisonGateV1(
        comparison="Griffin external validation",
        minimum_effect=0.0,
        bootstrap=griffin_bootstrap,
        permutation=griffin_permutation,
        holm_adjusted_p_value=griffin_permutation.p_value,
        passed=(
            griffin_bootstrap.confidence_lower > 0.0
            and griffin_permutation.p_value < experiment_plan.statistical_alpha
        ),
    )

    failed: list[str] = []
    failed.extend(item.comparison for item in robust_gates if not item.passed)
    failed.extend(
        f"clean {item.metric} non-inferiority"
        for item in clean_gates
        if not item.passed
    )
    if not pareto_gate.passed:
        failed.append(pareto_gate.comparison)
    if not griffin_gate.passed:
        failed.append(griffin_gate.comparison)
    failed.extend(item.invariant_id for item in bound_invariants if not item.passed)
    if not cold_cache_verified:
        failed.append("cold-cache evidence readback")
    if not provenance_verified:
        failed.append("trusted raw-result provenance")
    return PublicationGateResultV1(
        experiment_plan_sha256=experiment_plan.content_sha256,
        evidence_bundle_sha256=evidence_bundle.digest(),
        metrics_document_sha256=publication_metrics.digest(),
        robust_comparisons=robust_gates,
        clean_noninferiority=clean_gates,
        pareto_comparison=pareto_gate,
        griffin_comparison=griffin_gate,
        invariants=bound_invariants,
        cold_cache_receipt_sha256=(
            None if cold_cache_receipt is None else cold_cache_receipt.digest()
        ),
        cold_cache_verified=cold_cache_verified,
        sota_wording_allowed=not failed,
        failed_gates=tuple(failed),
    )


def evaluate_publication_gate_v1(
    *,
    experiment_plan: ExperimentPlanV1,
    evidence_bundle: EvidenceBundleV1,
    publication_metrics: PublicationMetricsV1,
    invariants: tuple[InvariantEvidenceV1, ...],
    cold_cache_receipt: ColdCacheReceiptV1 | None,
) -> PublicationGateResultV1:
    """Run the trusted-input arithmetic checker, never grant SOTA wording.

    Direct objects can be internally hash-consistent while still containing
    selected seeds, invented scores, or caller-set booleans.  The scientific
    entry point in :mod:`scientific_gate` first verifies the raw run registry
    and signed external receipts, then calls the private arithmetic routine.
    """

    return _evaluate_publication_gate_v1(
        experiment_plan=experiment_plan,
        evidence_bundle=evidence_bundle,
        publication_metrics=publication_metrics,
        invariants=invariants,
        cold_cache_receipt=cold_cache_receipt,
        provenance_verified=False,
    )


__all__ = [
    "BudgetedSequenceScoresV1",
    "ColdCacheReceiptV1",
    "ComparisonGateV1",
    "InvariantEvidenceV1",
    "MethodScoresV1",
    "NamedMetricScoresV1",
    "NonInferiorityGateV1",
    "PublicationGateResultV1",
    "PublicationMetricsV1",
    "REQUIRED_INVARIANTS_V1",
    "SequenceScoresV1",
    "decode_publication_metrics",
    "evaluate_publication_gate_v1",
    "paired_pareto_auc_by_sequence_v1",
]
