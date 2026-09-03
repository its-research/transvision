"""Recomputable independent-verification report for EventTrack-V2X.

This report contains no caller-controlled success flags apart from the exact
per-invariant results required by the preregistration.  Its evaluator, wire,
cold-cache, and C9 inventories are canonical functions of the evidence they
claim to verify.  A detached trusted-verifier receipt can therefore sign the
report digest without reducing verification to a collection of booleans.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

import numpy as np

from .candidate_selection import CandidateSelectionRegistryV1
from .contracts import EvidenceBundleV1
from .dataset_release import DatasetReleaseReceiptV1
from .experiment import ExperimentPlanV1
from .measured_trace import MeasuredTraceReceiptV1
from .publication_gate import REQUIRED_INVARIANTS_V1
from .results_registry import RunResultRegistryV1
from .validation_registry import ValidationResultRegistryV1
from .wire import canonical_json_bytes


_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


class VerificationReportError(ValueError):
    """Raised when an independent report is incomplete, stale, or tampered."""


def _strict_fields(
    value: object, expected: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise VerificationReportError(f"{name} must be a string-keyed object")
    if frozenset(value) != expected:
        missing = sorted(expected - frozenset(value))
        unknown = sorted(frozenset(value) - expected)
        raise VerificationReportError(
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
        raise VerificationReportError(f"{name} must be a canonical identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise VerificationReportError(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise VerificationReportError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class InvariantVerificationV1:
    """One invariant result bound to an artifact in the aggregate bundle."""

    invariant_id: str
    passed: bool
    artifact_role: str
    artifact_sha256: str

    _FIELDS = frozenset(
        {"artifact_role", "artifact_sha256", "invariant_id", "passed"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "invariant_id", _identifier(self.invariant_id, "invariant_id")
        )
        object.__setattr__(
            self, "artifact_role", _identifier(self.artifact_role, "artifact_role")
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(self.artifact_sha256, "artifact_sha256"),
        )
        if self.passed is not True:
            raise VerificationReportError("every required invariant must pass")

    def to_primitive(self) -> dict[str, object]:
        return {
            "artifact_role": self.artifact_role,
            "artifact_sha256": self.artifact_sha256,
            "invariant_id": self.invariant_id,
            "passed": self.passed,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "InvariantVerificationV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        return cls(
            invariant_id=item["invariant_id"],  # type: ignore[arg-type]
            passed=item["passed"],  # type: ignore[arg-type]
            artifact_role=item["artifact_role"],  # type: ignore[arg-type]
            artifact_sha256=item["artifact_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class VerificationInventorySummaryV1:
    """Count and digest of every cell participating in one recomputation."""

    cell_count: int
    inventory_sha256: str

    _FIELDS = frozenset({"cell_count", "inventory_sha256"})

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cell_count", _nonnegative_int(self.cell_count, "cell_count")
        )
        object.__setattr__(
            self,
            "inventory_sha256",
            _sha256(self.inventory_sha256, "inventory_sha256"),
        )

    def to_primitive(self) -> dict[str, object]:
        return {
            "cell_count": self.cell_count,
            "inventory_sha256": self.inventory_sha256,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "VerificationInventorySummaryV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        return cls(
            cell_count=item["cell_count"],  # type: ignore[arg-type]
            inventory_sha256=item["inventory_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ColdCacheArtifactRecordV1:
    """Observed cold-cache digest and size for one EvidenceBundle role."""

    artifact_role: str
    sha256: str
    byte_size: int

    _FIELDS = frozenset({"artifact_role", "byte_size", "sha256"})

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_role", _identifier(self.artifact_role, "artifact_role")
        )
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))
        object.__setattr__(
            self, "byte_size", _nonnegative_int(self.byte_size, "byte_size")
        )

    def to_primitive(self) -> dict[str, object]:
        return {
            "artifact_role": self.artifact_role,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "ColdCacheArtifactRecordV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        return cls(
            artifact_role=item["artifact_role"],  # type: ignore[arg-type]
            sha256=item["sha256"],  # type: ignore[arg-type]
            byte_size=item["byte_size"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class C9PacketMetadataRecordV1:
    """Exact trace-id to packet-metadata binding from the C9 receipt."""

    trace_id: str
    packet_metadata_sha256: str

    _FIELDS = frozenset({"packet_metadata_sha256", "trace_id"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_id", _identifier(self.trace_id, "trace_id"))
        object.__setattr__(
            self,
            "packet_metadata_sha256",
            _sha256(self.packet_metadata_sha256, "packet_metadata_sha256"),
        )

    def to_primitive(self) -> dict[str, str]:
        return {
            "packet_metadata_sha256": self.packet_metadata_sha256,
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "C9PacketMetadataRecordV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        return cls(
            trace_id=item["trace_id"],  # type: ignore[arg-type]
            packet_metadata_sha256=item["packet_metadata_sha256"],  # type: ignore[arg-type]
        )


def _inventory_sha256(items: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for item in items:
        if not first:
            digest.update(b",")
        digest.update(canonical_json_bytes(item))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def evaluator_recomputation_inventory_v1(
    run_registry: RunResultRegistryV1,
    validation_registry: ValidationResultRegistryV1,
) -> VerificationInventorySummaryV1:
    """Hash metric/status semantics for every run and validation cell."""

    if not isinstance(run_registry, RunResultRegistryV1):
        raise TypeError("run_registry must be RunResultRegistryV1")
    if not isinstance(validation_registry, ValidationResultRegistryV1):
        raise TypeError("validation_registry must be ValidationResultRegistryV1")
    def records() -> Iterable[dict[str, object]]:
        for registry_kind, cells, cache_sha256, evaluator_sha256 in (
            ("run", run_registry.cells, None, None),
            (
                "validation",
                validation_registry.cells,
                validation_registry.detection_cache_sha256,
                validation_registry.evaluator_contract_sha256,
            ),
        ):
            for cell in cells:
                yield {
                    "detection_cache_sha256": getattr(
                        cell, "detection_cache_sha256", cache_sha256
                    ),
                    "condition_input_manifest_sha256": (
                        cell.condition_input_manifest_sha256
                    ),
                    "evidence_bundle_sha256": cell.evidence_bundle_sha256,
                    "evaluator_contract_sha256": getattr(
                        cell, "evaluator_contract_sha256", evaluator_sha256
                    ),
                    "failure_code": cell.failure_code,
                    "key": list(cell.key),
                    "metric_values": [list(item) for item in cell.metric_values],
                    "metrics_artifact_role": cell.metrics_artifact_role,
                    "metrics_artifact_sha256": cell.metrics_artifact_sha256,
                    "registry_kind": registry_kind,
                    "status": cell.status,
                }

    return VerificationInventorySummaryV1(
        cell_count=len(run_registry.cells) + len(validation_registry.cells),
        inventory_sha256=_inventory_sha256(records()),
    )


def wire_ledger_replay_inventory_v1(
    run_registry: RunResultRegistryV1,
    validation_registry: ValidationResultRegistryV1,
) -> VerificationInventorySummaryV1:
    """Hash byte-accounting semantics for every run and validation cell."""

    if not isinstance(run_registry, RunResultRegistryV1):
        raise TypeError("run_registry must be RunResultRegistryV1")
    if not isinstance(validation_registry, ValidationResultRegistryV1):
        raise TypeError("validation_registry must be ValidationResultRegistryV1")
    def records() -> Iterable[dict[str, object]]:
        for registry_kind, cells in (
            ("run", run_registry.cells),
            ("validation", validation_registry.cells),
        ):
            for cell in cells:
                derived_bps = (
                    float(cell.on_wire_bytes_total)
                    / cell.wire_measurement_duration_seconds
                )
                if not np.isclose(
                    derived_bps,
                    cell.on_wire_bytes_per_second,
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise VerificationReportError(
                        "wire cell BPS does not equal bytes divided by duration"
                    )
                if derived_bps > float(cell.budget_bytes_per_second) + 1e-9:
                    raise VerificationReportError("wire cell exceeds its budget")
                yield {
                    "budget_bytes_per_second": cell.budget_bytes_per_second,
                    "channel_outcome_trace_sha256": (
                        cell.channel_outcome_trace_sha256
                    ),
                    "condition_input_manifest_sha256": (
                        cell.condition_input_manifest_sha256
                    ),
                    "key": list(cell.key),
                    "network_trace_sha256": cell.network_trace_sha256,
                    "on_wire_bytes_per_second": cell.on_wire_bytes_per_second,
                    "on_wire_bytes_total": cell.on_wire_bytes_total,
                    "registry_kind": registry_kind,
                    "wire_ledger_sha256": cell.wire_ledger_sha256,
                    "wire_measurement_duration_seconds": (
                        cell.wire_measurement_duration_seconds
                    ),
                }

    return VerificationInventorySummaryV1(
        cell_count=len(run_registry.cells) + len(validation_registry.cells),
        inventory_sha256=_inventory_sha256(records()),
    )


def candidate_selection_inventory_v1(
    registry: CandidateSelectionRegistryV1,
) -> VerificationInventorySummaryV1:
    """Hash all 144 claimed candidate configs, evidence, and summary metrics.

    This inventory binds what an independent verifier received; it does not by
    itself prove that the opaque evidence digests were cold-read or that their
    predictions were replayed through the evaluator.  Those operations remain
    guarded by the scientific-gate capability lock.
    """

    if not isinstance(registry, CandidateSelectionRegistryV1):
        raise TypeError("registry must be CandidateSelectionRegistryV1")

    def records() -> Iterable[dict[str, object]]:
        for entry in registry.entries:
            yield {
                "combination": list(entry.combination),
                "config_sha256": entry.config_sha256,
                "evidence_sha256": entry.evidence_sha256,
                "metrics": {
                    "actual_bytes_per_second": entry.actual_bytes_per_second,
                    "clean_amota": entry.clean_amota,
                    "clean_hota": entry.clean_hota,
                    "p95_latency_ms": entry.p95_latency_ms,
                    "robust_assa_at_64k": entry.robust_assa_at_64k,
                },
            }

    return VerificationInventorySummaryV1(
        cell_count=len(registry.entries),
        inventory_sha256=_inventory_sha256(records()),
    )


def _cold_cache_inventory_sha256(
    records: Iterable[ColdCacheArtifactRecordV1],
) -> str:
    return _inventory_sha256(item.to_primitive() for item in records)


def _c9_inventory_sha256(records: Iterable[C9PacketMetadataRecordV1]) -> str:
    return _inventory_sha256(item.to_primitive() for item in records)


@dataclass(frozen=True, slots=True)
class IndependentVerificationReportV1:
    """Canonical, recomputable output of the independent verifier."""

    experiment_plan_sha256: str
    run_result_registry_sha256: str
    validation_result_registry_sha256: str
    candidate_selection_registry_content_sha256: str
    aggregate_evidence_bundle_sha256: str
    development_dataset_release_receipt_sha256: str
    primary_dataset_release_receipt_sha256: str
    griffin_dataset_release_receipt_sha256: str
    measured_trace_receipt_sha256: str
    measured_trace_supports_clock_claims: bool
    invariants: tuple[InvariantVerificationV1, ...]
    evaluator_recomputation: VerificationInventorySummaryV1
    wire_ledger_replay: VerificationInventorySummaryV1
    candidate_selection_inventory: VerificationInventorySummaryV1
    cold_cache_inventory: tuple[ColdCacheArtifactRecordV1, ...]
    cold_cache_inventory_sha256: str
    c9_packet_metadata_inventory: tuple[C9PacketMetadataRecordV1, ...]
    c9_packet_metadata_inventory_sha256: str

    _FIELDS = frozenset(
        {
            "aggregate_evidence_bundle_sha256",
            "candidate_selection_inventory",
            "candidate_selection_registry_content_sha256",
            "c9_packet_metadata_inventory",
            "c9_packet_metadata_inventory_sha256",
            "cold_cache_inventory",
            "cold_cache_inventory_sha256",
            "content_sha256",
            "evaluator_recomputation",
            "experiment_plan_sha256",
            "development_dataset_release_receipt_sha256",
            "griffin_dataset_release_receipt_sha256",
            "invariants",
            "kind",
            "measured_trace_receipt_sha256",
            "measured_trace_supports_clock_claims",
            "primary_dataset_release_receipt_sha256",
            "run_result_registry_sha256",
            "schema_version",
            "validation_result_registry_sha256",
            "wire_ledger_replay",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "experiment_plan_sha256",
            "run_result_registry_sha256",
            "validation_result_registry_sha256",
            "candidate_selection_registry_content_sha256",
            "aggregate_evidence_bundle_sha256",
            "development_dataset_release_receipt_sha256",
            "primary_dataset_release_receipt_sha256",
            "griffin_dataset_release_receipt_sha256",
            "measured_trace_receipt_sha256",
            "cold_cache_inventory_sha256",
            "c9_packet_metadata_inventory_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if type(self.measured_trace_supports_clock_claims) is not bool:
            raise TypeError("measured_trace_supports_clock_claims must be bool")
        if self.measured_trace_supports_clock_claims:
            raise VerificationReportError(
                "V1 clock claims require content-validated measured packet "
                "artifacts; receipt metadata alone is insufficient"
            )
        if not isinstance(self.invariants, (list, tuple)) or not all(
            isinstance(item, InvariantVerificationV1) for item in self.invariants
        ):
            raise TypeError("invariants must contain InvariantVerificationV1")
        invariants = tuple(self.invariants)
        if tuple(item.invariant_id for item in invariants) != REQUIRED_INVARIANTS_V1:
            raise VerificationReportError(
                "report must contain every required invariant in canonical order"
            )
        object.__setattr__(self, "invariants", invariants)
        for name in (
            "evaluator_recomputation",
            "wire_ledger_replay",
            "candidate_selection_inventory",
        ):
            if not isinstance(getattr(self, name), VerificationInventorySummaryV1):
                raise TypeError(f"{name} must be VerificationInventorySummaryV1")
        if not isinstance(self.cold_cache_inventory, (list, tuple)) or not all(
            isinstance(item, ColdCacheArtifactRecordV1)
            for item in self.cold_cache_inventory
        ):
            raise TypeError(
                "cold_cache_inventory must contain ColdCacheArtifactRecordV1"
            )
        cold = tuple(self.cold_cache_inventory)
        if tuple(item.artifact_role for item in cold) != tuple(
            sorted(EvidenceBundleV1._ARTIFACT_FIELDS)
        ):
            raise VerificationReportError(
                "cold-cache inventory must contain every evidence artifact role"
            )
        if _cold_cache_inventory_sha256(cold) != self.cold_cache_inventory_sha256:
            raise VerificationReportError("cold-cache inventory hash mismatch")
        object.__setattr__(self, "cold_cache_inventory", cold)
        if not isinstance(self.c9_packet_metadata_inventory, (list, tuple)) or not all(
            isinstance(item, C9PacketMetadataRecordV1)
            for item in self.c9_packet_metadata_inventory
        ):
            raise TypeError(
                "c9_packet_metadata_inventory must contain C9PacketMetadataRecordV1"
            )
        c9 = tuple(self.c9_packet_metadata_inventory)
        trace_ids = tuple(item.trace_id for item in c9)
        if len(c9) != 20 or trace_ids != tuple(sorted(set(trace_ids))):
            raise VerificationReportError(
                "C9 packet inventory must contain 20 unique sorted held-out traces"
            )
        if _c9_inventory_sha256(c9) != self.c9_packet_metadata_inventory_sha256:
            raise VerificationReportError("C9 packet inventory hash mismatch")
        object.__setattr__(self, "c9_packet_metadata_inventory", c9)

    def payload(self) -> dict[str, Any]:
        return {
            "aggregate_evidence_bundle_sha256": (
                self.aggregate_evidence_bundle_sha256
            ),
            "candidate_selection_inventory": (
                self.candidate_selection_inventory.to_primitive()
            ),
            "candidate_selection_registry_content_sha256": (
                self.candidate_selection_registry_content_sha256
            ),
            "c9_packet_metadata_inventory": [
                item.to_primitive() for item in self.c9_packet_metadata_inventory
            ],
            "c9_packet_metadata_inventory_sha256": (
                self.c9_packet_metadata_inventory_sha256
            ),
            "cold_cache_inventory": [
                item.to_primitive() for item in self.cold_cache_inventory
            ],
            "cold_cache_inventory_sha256": self.cold_cache_inventory_sha256,
            "evaluator_recomputation": self.evaluator_recomputation.to_primitive(),
            "experiment_plan_sha256": self.experiment_plan_sha256,
            "development_dataset_release_receipt_sha256": (
                self.development_dataset_release_receipt_sha256
            ),
            "griffin_dataset_release_receipt_sha256": (
                self.griffin_dataset_release_receipt_sha256
            ),
            "invariants": [item.to_primitive() for item in self.invariants],
            "kind": "independent_verification_report_v1",
            "measured_trace_receipt_sha256": self.measured_trace_receipt_sha256,
            "measured_trace_supports_clock_claims": (
                self.measured_trace_supports_clock_claims
            ),
            "primary_dataset_release_receipt_sha256": (
                self.primary_dataset_release_receipt_sha256
            ),
            "run_result_registry_sha256": self.run_result_registry_sha256,
            "schema_version": 1,
            "validation_result_registry_sha256": (
                self.validation_result_registry_sha256
            ),
            "wire_ledger_replay": self.wire_ledger_replay.to_primitive(),
        }

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.payload())).hexdigest()

    def sealed_document(self) -> dict[str, Any]:
        return {**self.payload(), "content_sha256": self.content_sha256}

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.sealed_document())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> "IndependentVerificationReportV1":
        item = _strict_fields(value, cls._FIELDS, cls.__name__)
        if (
            item["kind"] != "independent_verification_report_v1"
            or item["schema_version"] != 1
        ):
            raise VerificationReportError("unsupported verification report schema")
        observed = _sha256(item["content_sha256"], "content_sha256")
        payload = {key: item[key] for key in cls._FIELDS if key != "content_sha256"}
        if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != observed:
            raise VerificationReportError("verification report content hash mismatch")
        invariants = item["invariants"]
        cold = item["cold_cache_inventory"]
        c9 = item["c9_packet_metadata_inventory"]
        if not all(isinstance(value, list) for value in (invariants, cold, c9)):
            raise VerificationReportError("report inventories must be arrays")
        return cls(
            experiment_plan_sha256=item["experiment_plan_sha256"],  # type: ignore[arg-type]
            run_result_registry_sha256=item["run_result_registry_sha256"],  # type: ignore[arg-type]
            validation_result_registry_sha256=item["validation_result_registry_sha256"],  # type: ignore[arg-type]
            candidate_selection_registry_content_sha256=item["candidate_selection_registry_content_sha256"],  # type: ignore[arg-type]
            aggregate_evidence_bundle_sha256=item["aggregate_evidence_bundle_sha256"],  # type: ignore[arg-type]
            development_dataset_release_receipt_sha256=item[
                "development_dataset_release_receipt_sha256"
            ],  # type: ignore[arg-type]
            primary_dataset_release_receipt_sha256=item["primary_dataset_release_receipt_sha256"],  # type: ignore[arg-type]
            griffin_dataset_release_receipt_sha256=item["griffin_dataset_release_receipt_sha256"],  # type: ignore[arg-type]
            measured_trace_receipt_sha256=item["measured_trace_receipt_sha256"],  # type: ignore[arg-type]
            measured_trace_supports_clock_claims=item["measured_trace_supports_clock_claims"],  # type: ignore[arg-type]
            invariants=tuple(InvariantVerificationV1.from_mapping(value) for value in invariants),
            evaluator_recomputation=VerificationInventorySummaryV1.from_mapping(
                item["evaluator_recomputation"]
            ),
            wire_ledger_replay=VerificationInventorySummaryV1.from_mapping(
                item["wire_ledger_replay"]
            ),
            candidate_selection_inventory=VerificationInventorySummaryV1.from_mapping(
                item["candidate_selection_inventory"]
            ),
            cold_cache_inventory=tuple(
                ColdCacheArtifactRecordV1.from_mapping(value) for value in cold
            ),
            cold_cache_inventory_sha256=item["cold_cache_inventory_sha256"],  # type: ignore[arg-type]
            c9_packet_metadata_inventory=tuple(
                C9PacketMetadataRecordV1.from_mapping(value) for value in c9
            ),
            c9_packet_metadata_inventory_sha256=item["c9_packet_metadata_inventory_sha256"],  # type: ignore[arg-type]
        )


def build_independent_verification_report_v1(
    *,
    experiment_plan: ExperimentPlanV1,
    run_result_registry: RunResultRegistryV1,
    validation_result_registry: ValidationResultRegistryV1,
    candidate_selection_registry: CandidateSelectionRegistryV1,
    aggregate_evidence_bundle: EvidenceBundleV1,
    development_dataset_receipt: DatasetReleaseReceiptV1,
    primary_dataset_receipt: DatasetReleaseReceiptV1,
    griffin_dataset_receipt: DatasetReleaseReceiptV1,
    measured_trace_receipt: MeasuredTraceReceiptV1,
    invariants: Iterable[InvariantVerificationV1],
    cold_cache_inventory: Iterable[ColdCacheArtifactRecordV1],
) -> IndependentVerificationReportV1:
    """Build a report, deriving every raw-cell and C9 inventory.

    V1 receives only the measured-trace receipt and its artifact hashes, not the
    measured packet artifacts themselves.  A receipt assertion cannot prove
    that every packet record used a trusted, synchronized clock.  Clock-claim
    capability therefore remains false until a future schema binds validated
    packet-artifact contents into both construction and external validation.
    """

    cold_records = tuple(cold_cache_inventory)
    c9_records = tuple(
        C9PacketMetadataRecordV1(
            trace_id=trace_id,
            packet_metadata_sha256=packet_metadata_sha256,
        )
        for trace_id, packet_metadata_sha256 in (
            measured_trace_receipt.held_out_packet_metadata_sha256s
        )
    )
    report = IndependentVerificationReportV1(
        experiment_plan_sha256=experiment_plan.content_sha256,
        run_result_registry_sha256=run_result_registry.digest(),
        validation_result_registry_sha256=validation_result_registry.digest(),
        candidate_selection_registry_content_sha256=(
            candidate_selection_registry.content_sha256
        ),
        aggregate_evidence_bundle_sha256=aggregate_evidence_bundle.digest(),
        development_dataset_release_receipt_sha256=(
            development_dataset_receipt.content_sha256
        ),
        primary_dataset_release_receipt_sha256=(
            primary_dataset_receipt.content_sha256
        ),
        griffin_dataset_release_receipt_sha256=(
            griffin_dataset_receipt.content_sha256
        ),
        measured_trace_receipt_sha256=measured_trace_receipt.content_sha256,
        measured_trace_supports_clock_claims=False,
        invariants=tuple(invariants),
        evaluator_recomputation=evaluator_recomputation_inventory_v1(
            run_result_registry, validation_result_registry
        ),
        wire_ledger_replay=wire_ledger_replay_inventory_v1(
            run_result_registry, validation_result_registry
        ),
        candidate_selection_inventory=candidate_selection_inventory_v1(
            candidate_selection_registry
        ),
        cold_cache_inventory=cold_records,
        cold_cache_inventory_sha256=_cold_cache_inventory_sha256(
            cold_records
        ),
        c9_packet_metadata_inventory=c9_records,
        c9_packet_metadata_inventory_sha256=_c9_inventory_sha256(c9_records),
    )
    validate_independent_verification_report_v1(
        report,
        experiment_plan=experiment_plan,
        run_result_registry=run_result_registry,
        validation_result_registry=validation_result_registry,
        candidate_selection_registry=candidate_selection_registry,
        aggregate_evidence_bundle=aggregate_evidence_bundle,
        development_dataset_receipt=development_dataset_receipt,
        primary_dataset_receipt=primary_dataset_receipt,
        griffin_dataset_receipt=griffin_dataset_receipt,
        measured_trace_receipt=measured_trace_receipt,
    )
    return report


def validate_independent_verification_report_v1(
    report: IndependentVerificationReportV1,
    *,
    experiment_plan: ExperimentPlanV1,
    run_result_registry: RunResultRegistryV1,
    validation_result_registry: ValidationResultRegistryV1,
    candidate_selection_registry: CandidateSelectionRegistryV1,
    aggregate_evidence_bundle: EvidenceBundleV1,
    development_dataset_receipt: DatasetReleaseReceiptV1,
    primary_dataset_receipt: DatasetReleaseReceiptV1,
    griffin_dataset_receipt: DatasetReleaseReceiptV1,
    measured_trace_receipt: MeasuredTraceReceiptV1,
) -> None:
    """Recompute every report binding and inventory from supplied evidence."""

    if not isinstance(report, IndependentVerificationReportV1):
        raise TypeError("report must be IndependentVerificationReportV1")
    _validate_measured_trace_clock_claims(report, measured_trace_receipt)
    expected_bindings = (
        experiment_plan.content_sha256,
        run_result_registry.digest(),
        validation_result_registry.digest(),
        candidate_selection_registry.content_sha256,
        aggregate_evidence_bundle.digest(),
        development_dataset_receipt.content_sha256,
        primary_dataset_receipt.content_sha256,
        griffin_dataset_receipt.content_sha256,
        measured_trace_receipt.content_sha256,
    )
    observed_bindings = (
        report.experiment_plan_sha256,
        report.run_result_registry_sha256,
        report.validation_result_registry_sha256,
        report.candidate_selection_registry_content_sha256,
        report.aggregate_evidence_bundle_sha256,
        report.development_dataset_release_receipt_sha256,
        report.primary_dataset_release_receipt_sha256,
        report.griffin_dataset_release_receipt_sha256,
        report.measured_trace_receipt_sha256,
    )
    if observed_bindings != expected_bindings:
        raise VerificationReportError("verification report has stale evidence bindings")
    if run_result_registry.experiment_plan_sha256 != experiment_plan.content_sha256:
        raise VerificationReportError("run registry is not bound to experiment plan")
    if (
        candidate_selection_registry.digest()
        != experiment_plan.candidate_selection_registry_sha256
    ):
        raise VerificationReportError(
            "candidate-selection registry is not bound to experiment plan"
        )
    if (
        development_dataset_receipt.content_sha256
        != experiment_plan.development_dataset_release_receipt_sha256
        or primary_dataset_receipt.content_sha256
        != experiment_plan.primary_dataset_release_receipt_sha256
        or griffin_dataset_receipt.content_sha256
        != experiment_plan.griffin_dataset_release_receipt_sha256
        or measured_trace_receipt.content_sha256
        != experiment_plan.heldout_c9_trace_receipt_sha256
    ):
        raise VerificationReportError("dataset or C9 receipt is not bound to plan")
    if (
        validation_result_registry.source_tree_sha256
        != experiment_plan.source_tree_sha256
        or validation_result_registry.dataset_manifest_sha256
        != experiment_plan.dataset_manifest_sha256
        or validation_result_registry.network_trace_manifest_sha256
        != experiment_plan.network_trace_manifest_sha256
        or validation_result_registry.wire_accounting_config_sha256
        != experiment_plan.wire_accounting_config_sha256
        or validation_result_registry.development_fold_manifest.content_sha256
        != experiment_plan.development_fold_manifest_sha256
    ):
        raise VerificationReportError("validation registry is stale relative to plan")
    expected_evaluator = evaluator_recomputation_inventory_v1(
        run_result_registry, validation_result_registry
    )
    if report.evaluator_recomputation != expected_evaluator:
        raise VerificationReportError("evaluator recomputation inventory mismatch")
    expected_wire = wire_ledger_replay_inventory_v1(
        run_result_registry, validation_result_registry
    )
    if report.wire_ledger_replay != expected_wire:
        raise VerificationReportError("wire-ledger replay inventory mismatch")
    expected_candidates = candidate_selection_inventory_v1(
        candidate_selection_registry
    )
    if report.candidate_selection_inventory != expected_candidates:
        raise VerificationReportError(
            "candidate-selection inventory mismatch"
        )
    expected_artifacts = tuple(
        (
            role,
            getattr(aggregate_evidence_bundle, role).sha256,
            getattr(aggregate_evidence_bundle, role).byte_size,
        )
        for role in sorted(EvidenceBundleV1._ARTIFACT_FIELDS)
    )
    observed_artifacts = tuple(
        (item.artifact_role, item.sha256, item.byte_size)
        for item in report.cold_cache_inventory
    )
    if observed_artifacts != expected_artifacts:
        raise VerificationReportError("cold-cache inventory does not match evidence")
    for invariant in report.invariants:
        if invariant.artifact_role not in EvidenceBundleV1._ARTIFACT_FIELDS:
            raise VerificationReportError("invariant artifact role is not in evidence")
        if (
            invariant.artifact_sha256
            != getattr(aggregate_evidence_bundle, invariant.artifact_role).sha256
        ):
            raise VerificationReportError("invariant artifact digest mismatch")
    expected_c9 = measured_trace_receipt.held_out_packet_metadata_sha256s
    observed_c9 = tuple(
        (item.trace_id, item.packet_metadata_sha256)
        for item in report.c9_packet_metadata_inventory
    )
    if observed_c9 != expected_c9:
        raise VerificationReportError("C9 packet inventory does not match receipt")


def _validate_measured_trace_clock_claims(
    report: IndependentVerificationReportV1,
    receipt: MeasuredTraceReceiptV1,
) -> None:
    """Keep V1 fail-closed without content-validated packet artifacts."""

    if not isinstance(receipt, MeasuredTraceReceiptV1):
        raise TypeError("receipt must be MeasuredTraceReceiptV1")
    if report.measured_trace_supports_clock_claims:
        raise VerificationReportError(
            "measured-trace clock-claim capability requires content-validated "
            "measured packet artifacts"
        )


def decode_independent_verification_report(
    data: bytes,
) -> IndependentVerificationReportV1:
    """Decode exact canonical JSON and reject duplicate keys/non-finite values."""

    if type(data) is not bytes:
        raise TypeError("verification report data must be bytes")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise VerificationReportError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                VerificationReportError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationReportError("invalid verification report JSON") from exc
    report = IndependentVerificationReportV1.from_mapping(value)
    if report.canonical_bytes != data:
        raise VerificationReportError("verification report JSON is not canonical")
    return report


__all__ = [
    "C9PacketMetadataRecordV1",
    "ColdCacheArtifactRecordV1",
    "IndependentVerificationReportV1",
    "InvariantVerificationV1",
    "VerificationInventorySummaryV1",
    "VerificationReportError",
    "build_independent_verification_report_v1",
    "candidate_selection_inventory_v1",
    "decode_independent_verification_report",
    "evaluator_recomputation_inventory_v1",
    "validate_independent_verification_report_v1",
    "wire_ledger_replay_inventory_v1",
]
