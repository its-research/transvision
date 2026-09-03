"""Closed scientific publication gate for EventTrack-V2X.

Unlike the low-level arithmetic checker, this entry point cannot consume loose
score dictionaries or caller-set verification booleans.  It derives metrics
from the exact raw run matrix and requires two independently trusted Ed25519
receipts: one proving pre-confirmatory plan registration and one proving cold-cache,
evaluator, invariant, and wire-ledger verification.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
from typing import Mapping

from .attempt_manifest import (
    ConfirmatoryAttemptManifestV1,
    validate_confirmatory_attempt_manifest_v1,
)
from .candidate_selection import CandidateSelectionRegistryV1
from .contracts import EvidenceBundleV1
from .dataset_release import DatasetReleaseReceiptV1
from .experiment import ExperimentPlanV1
from .measured_trace import MeasuredTraceReceiptV1
from .network_disturbance import ConditionInputManifestV1
from .preregistration import (
    PreregistrationReceiptV1,
    verify_preregistration_receipt_v1,
)
from .publication_gate import (
    ColdCacheReceiptV1,
    InvariantEvidenceV1,
    PublicationGateResultV1,
    _evaluate_publication_gate_v1,
)
from .qualification import (
    BaselineQualificationRegistryV1,
    validate_qualification_against_raw_registry_v1,
)
from .results_registry import (
    RunResultRegistryV1,
    derive_publication_metrics_v1,
    validate_run_result_external_inputs_v1,
)
from .verification_receipt import (
    TrustedVerificationReceiptV1,
    run_evidence_inventory_sha256,
    verify_trusted_verification_receipt_v1,
)
from .verification_report import (
    IndependentVerificationReportV1,
    validate_independent_verification_report_v1,
)
from .validation_registry import (
    ValidationResultRegistryV1,
    validate_validation_result_external_inputs_v1,
)
from .wire import canonical_json_bytes


class ScientificGateError(ValueError):
    """Raised when provenance is incomplete, stale, or semantically inconsistent."""


REQUIRED_SCIENTIFIC_GATE_CAPABILITIES_V1 = frozenset(
    {
        "artifact-byte-cold-read-and-evaluator-replay",
        "authenticated-attempt-log-completeness",
        "candidate-oof-raw-cell-replay",
        "checkpoint-bytes-loaded-per-attempt",
        "governance-root-trust-chain",
    }
)
# This is a source-controlled capability lock, not a caller argument.  A capability
# may be added only with its implementation, independent tests, and governance
# review.  Keeping it empty prevents internally consistent invented evidence from
# authorizing SOTA wording while those external mechanisms are absent.
IMPLEMENTED_SCIENTIFIC_GATE_CAPABILITIES_V1: frozenset[str] = frozenset()


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _require_scientific_gate_capabilities_v1() -> None:
    missing = tuple(
        sorted(
            REQUIRED_SCIENTIFIC_GATE_CAPABILITIES_V1
            - IMPLEMENTED_SCIENTIFIC_GATE_CAPABILITIES_V1
        )
    )
    if missing:
        raise ScientificGateError(
            f"scientific publication gate is capability-locked; missing={list(missing)}"
        )


def _validate_independent_trust_anchors(
    preregistration_receipt: PreregistrationReceiptV1,
    verification_receipt: TrustedVerificationReceiptV1,
) -> None:
    """Reject one actor or one key acting as both timestamp and verifier."""

    if preregistration_receipt.provider_id == verification_receipt.provider_id:
        raise ScientificGateError(
            "preregistration registry and publication verifier must be distinct providers"
        )
    if (
        preregistration_receipt.public_key_sha256
        == verification_receipt.public_key_sha256
    ):
        raise ScientificGateError(
            "preregistration registry and publication verifier must use distinct keys"
        )


def _validate_frozen_trust_policy(
    plan: ExperimentPlanV1,
    preregistration_receipt: PreregistrationReceiptV1,
    verification_receipt: TrustedVerificationReceiptV1,
) -> None:
    expected = (
        plan.preregistration_provider_id,
        plan.preregistration_public_key_sha256,
        plan.verification_provider_id,
        plan.verification_public_key_sha256,
    )
    observed = (
        preregistration_receipt.provider_id,
        preregistration_receipt.public_key_sha256,
        verification_receipt.provider_id,
        verification_receipt.public_key_sha256,
    )
    if observed != expected:
        raise ScientificGateError(
            "receipts do not match the trust policy frozen before val unsealing"
        )


def _validate_c9_trace_bindings(
    registry: RunResultRegistryV1,
    receipt: MeasuredTraceReceiptV1,
) -> None:
    """Bind each C9 result to the held-out packet-metadata artifact it replayed."""

    expected = dict(receipt.held_out_packet_metadata_sha256s)
    for cell in registry.cells:
        if cell.condition_id != "C9":
            continue
        assert cell.c9_trace_id is not None
        if cell.channel_outcome_trace_sha256 != expected.get(cell.c9_trace_id):
            raise ScientificGateError(
                "C9 result is not bound to its held-out packet metadata"
            )


def _validate_validation_c9_setup_bindings(
    registry: ValidationResultRegistryV1,
    receipt: MeasuredTraceReceiptV1,
) -> None:
    """Bind validation C9 cells exclusively to the receipt's 10 setup traces."""

    if (
        registry.measured_trace_receipt_sha256 != receipt.content_sha256
        or registry.c9_setup_trace_ids != receipt.setup_trace_ids
    ):
        raise ScientificGateError(
            "validation C9 setup trace receipt or trace IDs do not match"
        )
    setup_ids = frozenset(receipt.setup_trace_ids)
    expected = {
        segment.trace_id: segment.packet_metadata_sha256
        for segment in receipt.segments
        if segment.trace_id in setup_ids
    }
    for cell in registry.cells:
        if cell.condition_id != "C9":
            continue
        assert cell.c9_trace_id is not None
        if cell.channel_outcome_trace_sha256 != expected.get(cell.c9_trace_id):
            raise ScientificGateError(
                "validation C9 result is not bound to setup packet metadata"
            )


def _validate_independent_condition_input_inventories(
    confirmatory_manifests: Mapping[str, ConditionInputManifestV1],
    development_manifests: Mapping[str, ConditionInputManifestV1],
) -> None:
    """Keep development disturbances distinct from sealed confirmatory runs."""

    if not isinstance(confirmatory_manifests, Mapping):
        raise TypeError("confirmatory_condition_input_manifests must be a mapping")
    if not isinstance(development_manifests, Mapping):
        raise TypeError("development_condition_input_manifests must be a mapping")
    if confirmatory_manifests is development_manifests or set(
        confirmatory_manifests
    ).intersection(
        development_manifests
    ):
        raise ScientificGateError(
            "development and confirmatory condition input inventories must be independent"
        )


def _validate_execution_chronology(
    preregistration_receipt: PreregistrationReceiptV1,
    registry: RunResultRegistryV1,
    verification_receipt: TrustedVerificationReceiptV1,
) -> None:
    if _utc(registry.confirmatory_val_unsealed_at_utc) <= _utc(
        preregistration_receipt.registered_at_utc
    ):
        raise ScientificGateError(
            "confirmatory val data must remain sealed until after plan registration"
        )
    if _utc(verification_receipt.verified_at_utc) <= _utc(registry.completed_at_utc):
        raise ScientificGateError(
            "independent verification must occur after confirmatory completion"
        )


def _validate_dataset_receipt(
    receipt: DatasetReleaseReceiptV1,
    *,
    dataset_id: str,
    split_name: str,
    dataset_manifest_sha256: str,
    split_sha256: str | None,
    cohort_sha256: str,
    frame_contract_sha256: str,
    sequence_ids: tuple[str, ...],
    expected_receipt_sha256: str,
) -> None:
    if not isinstance(receipt, DatasetReleaseReceiptV1):
        raise TypeError("dataset receipt must be DatasetReleaseReceiptV1")
    expected = (
        dataset_id,
        split_name,
        dataset_manifest_sha256,
        cohort_sha256,
        frame_contract_sha256,
        sequence_ids,
        expected_receipt_sha256,
    )
    observed = (
        receipt.dataset_id,
        receipt.split_name,
        receipt.dataset_manifest_sha256,
        receipt.cohort_sha256,
        receipt.frame_contract_sha256,
        receipt.sequence_ids,
        receipt.content_sha256,
    )
    if observed != expected:
        raise ScientificGateError("dataset release receipt does not match the plan")
    if split_sha256 is not None and receipt.split_sha256 != split_sha256:
        raise ScientificGateError("dataset release split hash does not match the plan")


def _validate_independent_spd_sequence_cohorts(
    development_receipt: DatasetReleaseReceiptV1,
    confirmatory_receipt: DatasetReleaseReceiptV1,
) -> None:
    """Reject train/val sequence leakage even when both receipts are well formed."""

    if not set(development_receipt.sequence_ids).isdisjoint(
        confirmatory_receipt.sequence_ids
    ):
        raise ScientificGateError(
            "development train and confirmatory val sequence cohorts overlap"
        )


def _validate_qualification(
    registry: BaselineQualificationRegistryV1,
    raw_registry: ValidationResultRegistryV1,
    plan: ExperimentPlanV1,
) -> None:
    if not isinstance(registry, BaselineQualificationRegistryV1):
        raise TypeError(
            "qualification_registry must be BaselineQualificationRegistryV1"
        )
    if not isinstance(raw_registry, ValidationResultRegistryV1):
        raise TypeError("validation_result_registry must be ValidationResultRegistryV1")
    validate_qualification_against_raw_registry_v1(registry, raw_registry)
    if registry.digest() != plan.baseline_qualification_registry_sha256:
        raise ScientificGateError("qualification registry digest does not match plan")
    if (
        registry.source_tree_sha256 != plan.source_tree_sha256
        or registry.dataset_manifest_sha256 != plan.dataset_manifest_sha256
        or registry.network_trace_manifest_sha256 != plan.network_trace_manifest_sha256
        or registry.strongest_qualified_baseline_id
        != plan.strongest_qualified_baseline_id
        or registry.strongest_scheduler_baseline_id
        != plan.strongest_scheduler_baseline_id
        or raw_registry.wire_accounting_config_sha256
        != plan.wire_accounting_config_sha256
        or raw_registry.measured_trace_receipt_sha256
        != plan.heldout_c9_trace_receipt_sha256
        or registry.development_fold_manifest_sha256
        != plan.development_fold_manifest_sha256
        or raw_registry.development_fold_manifest.content_sha256
        != plan.development_fold_manifest_sha256
    ):
        raise ScientificGateError("validation qualification is stale or mismatched")
    baseline_configs = {
        entry.method_id: entry.config_sha256 for entry in registry.baseline_entries
    }
    for method_id in plan.qualified_baseline_ids:
        if (
            plan.run_config_sha256(
                "primary", method_id, plan.strongest_scheduler_baseline_id
            )
            != baseline_configs[method_id]
        ):
            raise ScientificGateError(
                "primary baseline run config was not frozen by validation"
            )
    scheduler_configs = {
        entry.scheduler_id: entry.config_sha256 for entry in registry.scheduler_entries
    }
    if (
        plan.run_config_sha256(
            "primary", "eventtrack-v2x", plan.strongest_scheduler_baseline_id
        )
        != scheduler_configs[plan.strongest_scheduler_baseline_id]
    ):
        raise ScientificGateError(
            "strongest scheduler run config was not frozen by validation"
        )
    if (
        plan.run_config_sha256("primary", "eventtrack-v2x", plan.candidate_scheduler_id)
        != plan.method_config_sha256
    ):
        raise ScientificGateError(
            "candidate EventTrack run config does not match the frozen method config"
        )


def _validate_candidate_selection(
    registry: CandidateSelectionRegistryV1,
    qualification_registry: BaselineQualificationRegistryV1,
    raw_registry: ValidationResultRegistryV1,
    plan: ExperimentPlanV1,
) -> None:
    """Bind the complete train OOF grid and its unique winner into the plan."""

    if not isinstance(registry, CandidateSelectionRegistryV1):
        raise TypeError(
            "candidate_selection_registry must be CandidateSelectionRegistryV1"
        )
    strongest = next(
        entry
        for entry in qualification_registry.baseline_entries
        if entry.method_id == qualification_registry.strongest_qualified_baseline_id
    )
    observed = (
        registry.digest(),
        registry.source_tree_sha256,
        registry.dataset_id,
        registry.dataset_manifest_sha256,
        registry.split_name,
        registry.split_sha256,
        registry.cohort_sha256,
        registry.sequence_ids,
        registry.development_fold_manifest_sha256,
        registry.frequency_hz,
        registry.class_names,
        registry.detection_cache_sha256,
        registry.detection_frame_contract_sha256,
        registry.network_trace_manifest_sha256,
        registry.wire_accounting_config_sha256,
        registry.evaluator_contract_sha256,
        registry.validation_result_registry_sha256,
        registry.baseline_qualification_registry_sha256,
        registry.strongest_baseline_method_id,
        registry.reference_clean_amota,
        registry.reference_clean_hota,
        registry.method_id,
        registry.scheduler_id,
        registry.training_seeds,
        registry.network_seeds,
        registry.primary_budget_bytes_per_second,
        registry.selected_method_config_sha256,
    )
    expected = (
        plan.candidate_selection_registry_sha256,
        plan.source_tree_sha256,
        raw_registry.dataset_id,
        raw_registry.dataset_manifest_sha256,
        raw_registry.split_name,
        raw_registry.split_sha256,
        raw_registry.cohort_sha256,
        raw_registry.sequence_ids,
        raw_registry.development_fold_manifest.content_sha256,
        raw_registry.frequency_hz,
        raw_registry.class_names,
        raw_registry.detection_cache_sha256,
        raw_registry.detection_frame_contract_sha256,
        raw_registry.network_trace_manifest_sha256,
        raw_registry.wire_accounting_config_sha256,
        raw_registry.evaluator_contract_sha256,
        raw_registry.digest(),
        qualification_registry.digest(),
        qualification_registry.strongest_qualified_baseline_id,
        strongest.clean_amota,
        strongest.clean_hota,
        "eventtrack-v2x",
        plan.candidate_scheduler_id,
        plan.training_seeds,
        plan.network_seeds,
        plan.primary_budget_bytes_per_second,
        plan.method_config_sha256,
    )
    if observed != expected:
        raise ScientificGateError(
            "candidate selection registry is stale, incomplete, or mismatched"
        )


def evaluate_scientific_publication_gate_v1(
    *,
    experiment_plan: ExperimentPlanV1,
    run_result_registry: RunResultRegistryV1,
    attempt_manifest: ConfirmatoryAttemptManifestV1,
    aggregate_evidence_bundle: EvidenceBundleV1,
    qualification_registry: BaselineQualificationRegistryV1,
    validation_result_registry: ValidationResultRegistryV1,
    candidate_selection_registry: CandidateSelectionRegistryV1,
    development_dataset_receipt: DatasetReleaseReceiptV1,
    primary_dataset_receipt: DatasetReleaseReceiptV1,
    griffin_dataset_receipt: DatasetReleaseReceiptV1,
    measured_trace_receipt: MeasuredTraceReceiptV1,
    confirmatory_condition_input_manifests: Mapping[str, ConditionInputManifestV1],
    development_condition_input_manifests: Mapping[str, ConditionInputManifestV1],
    preregistration_receipt: PreregistrationReceiptV1,
    verification_receipt: TrustedVerificationReceiptV1,
    independent_verification_report: IndependentVerificationReportV1,
) -> PublicationGateResultV1:
    """Validate all evidence identities, derive metrics, then evaluate claims."""

    if not isinstance(experiment_plan, ExperimentPlanV1):
        raise TypeError("experiment_plan must be ExperimentPlanV1")
    if not isinstance(run_result_registry, RunResultRegistryV1):
        raise TypeError("run_result_registry must be RunResultRegistryV1")
    if not isinstance(aggregate_evidence_bundle, EvidenceBundleV1):
        raise TypeError("aggregate_evidence_bundle must be EvidenceBundleV1")
    _validate_independent_condition_input_inventories(
        confirmatory_condition_input_manifests,
        development_condition_input_manifests,
    )
    validate_confirmatory_attempt_manifest_v1(
        attempt_manifest,
        experiment_plan=experiment_plan,
        run_result_registry=run_result_registry,
    )
    if (
        attempt_manifest.execution_environment_sha256
        != aggregate_evidence_bundle.environment.sha256
    ):
        raise ScientificGateError(
            "attempt execution environment does not match aggregate evidence"
        )
    _validate_dataset_receipt(
        development_dataset_receipt,
        dataset_id=validation_result_registry.dataset_id,
        split_name=validation_result_registry.split_name,
        dataset_manifest_sha256=validation_result_registry.dataset_manifest_sha256,
        split_sha256=validation_result_registry.split_sha256,
        cohort_sha256=validation_result_registry.cohort_sha256,
        frame_contract_sha256=(
            validation_result_registry.detection_frame_contract_sha256
        ),
        sequence_ids=validation_result_registry.sequence_ids,
        expected_receipt_sha256=(
            experiment_plan.development_dataset_release_receipt_sha256
        ),
    )
    _validate_dataset_receipt(
        primary_dataset_receipt,
        dataset_id=experiment_plan.dataset_id,
        split_name=experiment_plan.split_name,
        dataset_manifest_sha256=experiment_plan.dataset_manifest_sha256,
        split_sha256=experiment_plan.split_sha256,
        cohort_sha256=experiment_plan.primary_cohort_sha256,
        frame_contract_sha256=experiment_plan.primary_frame_contract_sha256,
        sequence_ids=experiment_plan.primary_sequence_ids,
        expected_receipt_sha256=(
            experiment_plan.primary_dataset_release_receipt_sha256
        ),
    )
    _validate_independent_spd_sequence_cohorts(
        development_dataset_receipt,
        primary_dataset_receipt,
    )
    _validate_dataset_receipt(
        griffin_dataset_receipt,
        dataset_id=experiment_plan.griffin_dataset_id,
        split_name=experiment_plan.griffin_split_name,
        dataset_manifest_sha256=experiment_plan.griffin_dataset_manifest_sha256,
        split_sha256=None,
        cohort_sha256=experiment_plan.griffin_cohort_sha256,
        frame_contract_sha256=experiment_plan.griffin_frame_contract_sha256,
        sequence_ids=experiment_plan.griffin_sequence_ids,
        expected_receipt_sha256=(
            experiment_plan.griffin_dataset_release_receipt_sha256
        ),
    )
    if not isinstance(measured_trace_receipt, MeasuredTraceReceiptV1):
        raise TypeError("measured_trace_receipt must be MeasuredTraceReceiptV1")
    if (
        measured_trace_receipt.content_sha256
        != experiment_plan.heldout_c9_trace_receipt_sha256
        or measured_trace_receipt.held_out_trace_ids != experiment_plan.c9_trace_ids
    ):
        raise ScientificGateError("measured C9 trace receipt does not match plan")
    validate_validation_result_external_inputs_v1(
        validation_result_registry,
        experiment_plan,
        measured_trace_receipt=measured_trace_receipt,
        condition_input_manifests=development_condition_input_manifests,
    )
    _validate_qualification(
        qualification_registry,
        validation_result_registry,
        experiment_plan,
    )
    _validate_candidate_selection(
        candidate_selection_registry,
        qualification_registry,
        validation_result_registry,
        experiment_plan,
    )
    _validate_frozen_trust_policy(
        experiment_plan,
        preregistration_receipt,
        verification_receipt,
    )
    verify_preregistration_receipt_v1(
        preregistration_receipt,
        experiment_plan=experiment_plan,
        trusted_public_key_sha256=(experiment_plan.preregistration_public_key_sha256),
    )
    verify_trusted_verification_receipt_v1(
        verification_receipt,
        trusted_public_key_sha256=experiment_plan.verification_public_key_sha256,
    )
    _validate_independent_trust_anchors(preregistration_receipt, verification_receipt)
    _validate_c9_trace_bindings(run_result_registry, measured_trace_receipt)
    validate_run_result_external_inputs_v1(
        run_result_registry,
        experiment_plan,
        measured_trace_receipt=measured_trace_receipt,
        condition_input_manifests=confirmatory_condition_input_manifests,
    )
    validate_independent_verification_report_v1(
        independent_verification_report,
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
    _validate_execution_chronology(
        preregistration_receipt,
        run_result_registry,
        verification_receipt,
    )
    if _utc(verification_receipt.verified_at_utc) < _utc(
        attempt_manifest.attempt_inventory_closed_at_utc
    ):
        raise ScientificGateError(
            "independent verification predates attempt-inventory closure"
        )

    metrics = derive_publication_metrics_v1(run_result_registry, experiment_plan)
    expected_verification = (
        experiment_plan.content_sha256,
        run_result_registry.digest(),
        aggregate_evidence_bundle.digest(),
        qualification_registry.digest(),
        validation_result_registry.digest(),
        development_dataset_receipt.content_sha256,
        primary_dataset_receipt.content_sha256,
        griffin_dataset_receipt.content_sha256,
        measured_trace_receipt.content_sha256,
        run_evidence_inventory_sha256(run_result_registry),
        attempt_manifest.digest(),
        independent_verification_report.digest(),
    )
    observed_verification = (
        verification_receipt.experiment_plan_sha256,
        verification_receipt.run_result_registry_sha256,
        verification_receipt.aggregate_evidence_bundle_sha256,
        verification_receipt.qualification_registry_sha256,
        verification_receipt.validation_result_registry_sha256,
        verification_receipt.development_dataset_release_receipt_sha256,
        verification_receipt.primary_dataset_release_receipt_sha256,
        verification_receipt.griffin_dataset_release_receipt_sha256,
        verification_receipt.measured_trace_receipt_sha256,
        verification_receipt.run_evidence_inventory_sha256,
        verification_receipt.attempt_manifest_sha256,
        verification_receipt.independent_verification_report_sha256,
    )
    if observed_verification != expected_verification:
        raise ScientificGateError(
            "trusted verifier receipt has stale artifact bindings"
        )
    if metrics.digest() != aggregate_evidence_bundle.per_sequence_metrics.sha256:
        raise ScientificGateError(
            "derived metrics are not bound by aggregate EvidenceBundleV1"
        )

    invariant_inventory_sha256 = hashlib.sha256(
        canonical_json_bytes(
            [item.to_primitive() for item in independent_verification_report.invariants]
        )
    ).hexdigest()
    expected_report_components = (
        invariant_inventory_sha256,
        independent_verification_report.evaluator_recomputation.inventory_sha256,
        independent_verification_report.wire_ledger_replay.inventory_sha256,
        independent_verification_report.cold_cache_inventory_sha256,
    )
    observed_report_components = (
        verification_receipt.invariant_report_sha256,
        verification_receipt.evaluator_recompute_report_sha256,
        verification_receipt.wire_ledger_replay_report_sha256,
        verification_receipt.cold_cache_inventory_sha256,
    )
    if observed_report_components != expected_report_components:
        raise ScientificGateError(
            "trusted verifier receipt does not bind parsed report components"
        )

    evidence_sha256 = aggregate_evidence_bundle.digest()
    invariants = tuple(
        InvariantEvidenceV1(
            invariant_id=item.invariant_id,
            passed=item.passed,
            experiment_plan_sha256=experiment_plan.content_sha256,
            evidence_bundle_sha256=evidence_sha256,
            artifact_role=item.artifact_role,
            artifact=getattr(aggregate_evidence_bundle, item.artifact_role),
        )
        for item in independent_verification_report.invariants
    )
    cold_receipt = ColdCacheReceiptV1(
        receipt_id=verification_receipt.log_entry_id,
        verified=True,
        experiment_plan_sha256=experiment_plan.content_sha256,
        evidence_bundle_sha256=evidence_sha256,
        verified_artifacts=tuple(
            (
                item.artifact_role,
                getattr(aggregate_evidence_bundle, item.artifact_role),
            )
            for item in independent_verification_report.cold_cache_inventory
        ),
    )
    _require_scientific_gate_capabilities_v1()
    return _evaluate_publication_gate_v1(
        experiment_plan=experiment_plan,
        evidence_bundle=aggregate_evidence_bundle,
        publication_metrics=metrics,
        invariants=invariants,
        cold_cache_receipt=cold_receipt,
        provenance_verified=True,
    )


__all__ = [
    "IMPLEMENTED_SCIENTIFIC_GATE_CAPABILITIES_V1",
    "REQUIRED_SCIENTIFIC_GATE_CAPABILITIES_V1",
    "ScientificGateError",
    "_validate_candidate_selection",
    "_validate_independent_condition_input_inventories",
    "_validate_independent_spd_sequence_cohorts",
    "_validate_validation_c9_setup_bindings",
    "evaluate_scientific_publication_gate_v1",
]
