from dataclasses import replace
import hashlib
import json

import pytest

from transvision.models.event_track_v2x.candidate_selection import (
    CANDIDATE_COMBINATIONS_V1,
    CandidateSelectionEntryV1,
    CandidateSelectionRegistryV1,
)
from transvision.models.event_track_v2x.contracts import (
    ArtifactDigestV1,
    EvidenceBundleV1,
)
from transvision.models.event_track_v2x.dataset_release import (
    OFFICIAL_RELEASE_STATUS_V1,
    DatasetReleaseReceiptV1,
)
from transvision.models.event_track_v2x.development_split import (
    build_development_split_manifest_v1,
)
from transvision.models.event_track_v2x.experiment import (
    CONFIRMATORY_METHOD_IDS_V1,
    CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1,
    GRIFFIN_25M_VAL_SEQUENCE_IDS_V1,
    ExperimentPlanV1,
    RunConfigBindingV1,
    confirmatory_run_config_keys_v1,
)
from transvision.models.event_track_v2x.measured_trace import (
    REQUIRED_PACKET_FIELDS_V1,
    ClockReference,
    MeasuredTraceReceiptV1,
    MeasuredTraceSegmentV1,
    TraceRole,
)
from transvision.models.event_track_v2x.publication_gate import (
    REQUIRED_INVARIANTS_V1,
)
from transvision.models.event_track_v2x.results_registry import (
    RESULT_METRICS_ARTIFACT_ROLE_V1,
    RUN_RESULT_STATUS_SUCCESS_V1,
    RunResultCellV1,
    RunResultRegistryV1,
)
from transvision.models.event_track_v2x.scientific_gate import (
    ScientificGateError,
    _validate_validation_c9_setup_bindings,
)
from transvision.models.event_track_v2x.validation_registry import (
    VALIDATION_BASELINE_FAMILY_V1,
    VALIDATION_METRICS_ARTIFACT_ROLE_V1,
    VALIDATION_STATUS_SUCCESS_V1,
    ValidationResultCellV1,
    ValidationResultRegistryV1,
)
from transvision.models.event_track_v2x.verification_report import (
    ColdCacheArtifactRecordV1,
    InvariantVerificationV1,
    VerificationInventorySummaryV1,
    VerificationReportError,
    _validate_measured_trace_clock_claims,
    build_independent_verification_report_v1,
    candidate_selection_inventory_v1,
    decode_independent_verification_report,
    evaluator_recomputation_inventory_v1,
    validate_independent_verification_report_v1,
    wire_ledger_replay_inventory_v1,
)
from transvision.models.event_track_v2x.wire import canonical_json_bytes


PRIMARY_SEQUENCES = tuple(f"sequence-{index:02d}" for index in range(21))
DEVELOPMENT_SEQUENCES = tuple(
    f"development-sequence-{index:02d}" for index in range(46)
)
GRIFFIN_SEQUENCES = GRIFFIN_25M_VAL_SEQUENCE_IDS_V1
SETUP_TRACE_IDS = tuple(f"setup-trace-{index:02d}" for index in range(10))


def _artifact(role: str, index: int) -> ArtifactDigestV1:
    return ArtifactDigestV1(
        uri=f"clearml://verified/{role}",
        sha256=f"{index:x}" * 64,
        byte_size=index,
    )


@pytest.fixture(scope="module")
def evidence() -> EvidenceBundleV1:
    return EvidenceBundleV1(
        run_id="aggregate-run",
        **{
            role: _artifact(role, index)
            for index, role in enumerate(EvidenceBundleV1._ARTIFACT_FIELDS, start=1)
        },
    )


def _dataset_receipt(
    *, dataset_id: str, split_name: str, sequence_ids: tuple[str, ...], digit: str
) -> DatasetReleaseReceiptV1:
    return DatasetReleaseReceiptV1(
        receipt_id=f"{dataset_id}-{split_name}-receipt",
        dataset_id=dataset_id,
        split_name=split_name,
        dataset_manifest_sha256=digit * 64,
        official_inventory_sha256="a" * 64,
        license_evidence_sha256="b" * 64,
        split_sha256="c" * 64,
        cohort_sha256="d" * 64,
        frame_contract_sha256="e" * 64,
        sequence_ids=sequence_ids,
        clearml_dataset_id=f"{dataset_id}-clearml",
        clearml_project_id="eventtrack-v2x",
        clearml_version="v1",
        cold_cache_verification_sha256="f" * 64,
        release_identity_status=OFFICIAL_RELEASE_STATUS_V1,
        license_use_authorized=True,
        source_bytes_verified=True,
        cold_cache_verified=True,
        scientific_claims_allowed=True,
    )


@pytest.fixture(scope="module")
def primary_receipt() -> DatasetReleaseReceiptV1:
    return _dataset_receipt(
        dataset_id="v2x-seq-spd",
        split_name="val",
        sequence_ids=PRIMARY_SEQUENCES,
        digit="2",
    )


@pytest.fixture(scope="module")
def development_receipt() -> DatasetReleaseReceiptV1:
    return _dataset_receipt(
        dataset_id="v2x-seq-spd",
        split_name="train",
        sequence_ids=DEVELOPMENT_SEQUENCES,
        digit="2",
    )


@pytest.fixture(scope="module")
def griffin_receipt() -> DatasetReleaseReceiptV1:
    return _dataset_receipt(
        dataset_id="griffin-25m",
        split_name="val",
        sequence_ids=GRIFFIN_SEQUENCES,
        digit="9",
    )


@pytest.fixture(scope="module")
def measured_receipt() -> MeasuredTraceReceiptV1:
    segments = []
    for index in range(30):
        held_out = index >= 10
        trace_id = (
            f"heldout-trace-{index - 10:02d}"
            if held_out
            else f"setup-trace-{index:02d}"
        )
        segments.append(
            MeasuredTraceSegmentV1(
                trace_id=trace_id,
                role=TraceRole.HELD_OUT if held_out else TraceRole.SETUP,
                date_cohort_id=f"date-{index % 3}",
                road_coverage_type=f"road-{index % 3}",
                duration_seconds=600,
                packet_count=100 + index,
                packet_metadata_sha256=f"{index + 1:064x}",
                packet_fields=REQUIRED_PACKET_FIELDS_V1,
                clock_reference=ClockReference.PTP,
            )
        )
    return MeasuredTraceReceiptV1(
        receipt_id="c9-receipt",
        segments=tuple(segments),
        collection_protocol_sha256="7" * 64,
    )


@pytest.fixture(scope="module")
def plan(
    development_receipt: DatasetReleaseReceiptV1,
    primary_receipt: DatasetReleaseReceiptV1,
    griffin_receipt: DatasetReleaseReceiptV1,
    measured_receipt: MeasuredTraceReceiptV1,
    candidate_selection_registry: CandidateSelectionRegistryV1,
) -> ExperimentPlanV1:
    run_configs = tuple(
        RunConfigBindingV1(
            dataset_domain=domain,
            method_id=method,
            scheduler_id=scheduler,
            config_sha256="2" * 64,
            checkpoint_sha256s=tuple(
                hashlib.sha256(
                    f"{domain}:{method}:{scheduler}:{seed}".encode()
                ).hexdigest()
                for seed in (1337, 2027, 3407)
            ),
        )
        for domain, method, scheduler in confirmatory_run_config_keys_v1(
            strongest_baseline_id="vehicle-only-ab3dmot",
            strongest_scheduler_id="confidence_top_k",
        )
    )
    return ExperimentPlanV1(
        plan_id="eventtrack-v2x-confirmatory-v1",
        source_tree_sha256="1" * 64,
        method_config_sha256="2" * 64,
        run_config_bindings=run_configs,
        dataset_id="v2x-seq-spd",
        dataset_manifest_sha256=primary_receipt.dataset_manifest_sha256,
        split_name="val",
        split_sha256=primary_receipt.split_sha256,
        primary_cohort_sha256=primary_receipt.cohort_sha256,
        primary_frame_contract_sha256=primary_receipt.frame_contract_sha256,
        primary_dataset_release_receipt_sha256=primary_receipt.content_sha256,
        development_dataset_release_receipt_sha256=(
            development_receipt.content_sha256
        ),
        development_fold_manifest_sha256=(
            candidate_selection_registry.development_fold_manifest_sha256
        ),
        primary_sequence_ids=PRIMARY_SEQUENCES,
        detector_cache_sha256="5" * 64,
        network_trace_manifest_sha256="6" * 64,
        wire_accounting_config_sha256="0" * 64,
        heldout_c9_trace_receipt_sha256=measured_receipt.content_sha256,
        c9_trace_ids=measured_receipt.held_out_trace_ids,
        evaluator_contract_sha256="8" * 64,
        baseline_qualification_registry_sha256="1" * 64,
        candidate_selection_registry_sha256=(
            candidate_selection_registry.digest()
        ),
        preregistration_provider_id="institutional-log",
        preregistration_public_key_sha256="2" * 64,
        verification_provider_id="independent-verifier",
        verification_public_key_sha256="3" * 64,
        qualified_baseline_ids=CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1,
        strongest_qualified_baseline_id="vehicle-only-ab3dmot",
        scheduler_ids=(
            "confidence_top_k",
            "fifo_aoi",
            "full_send",
            "marginal_voi",
            "periodic",
            "random",
        ),
        candidate_scheduler_id="marginal_voi",
        qualified_scheduler_baseline_ids=(
            "confidence_top_k",
            "fifo_aoi",
            "periodic",
            "random",
        ),
        strongest_scheduler_baseline_id="confidence_top_k",
        scheduler_token_bucket_burst_seconds=1.0,
        griffin_dataset_id="griffin-25m",
        griffin_dataset_manifest_sha256=griffin_receipt.dataset_manifest_sha256,
        griffin_split_name="val",
        griffin_cohort_sha256=griffin_receipt.cohort_sha256,
        griffin_frame_contract_sha256=griffin_receipt.frame_contract_sha256,
        griffin_dataset_release_receipt_sha256=griffin_receipt.content_sha256,
        griffin_sequence_ids=GRIFFIN_SEQUENCES,
        griffin_detector_cache_sha256="b" * 64,
        griffin_evaluator_contract_sha256="c" * 64,
        griffin_strongest_baseline_id="vehicle-only-ab3dmot",
        frequency_hz=10.0,
        class_names=("car",),
        method_ids=CONFIRMATORY_METHOD_IDS_V1,
        network_condition_ids=tuple(f"C{index}" for index in range(10)),
        training_seeds=(1337, 2027, 3407),
        network_seeds=tuple(range(1001, 1011)),
        byte_budgets_per_second=(16_000, 32_000, 64_000, 128_000, 256_000),
        primary_budget_bytes_per_second=64_000,
        primary_metric="robust-assa-at-64k",
        decision_deadline_ms=100.0,
        statistical_alpha=0.05,
        statistical_resamples=10_000,
        statistical_random_seed=1337,
        claim_scope="same-detector robust cooperative association",
    )


@pytest.fixture(scope="module")
def run_registry(plan: ExperimentPlanV1) -> RunResultRegistryV1:
    cell = RunResultCellV1(
        experiment_plan_sha256=plan.content_sha256,
        attempt_id="attempt-01",
        detection_cache_sha256=plan.detector_cache_sha256,
        evaluator_contract_sha256=plan.evaluator_contract_sha256,
        scenario_id="primary_clean",
        dataset_domain="primary",
        method_id="eventtrack-v2x",
        sequence_id=PRIMARY_SEQUENCES[0],
        condition_id="C0",
        training_seed=1337,
        network_seed=None,
        c9_trace_id=None,
        budget_bytes_per_second=64_000,
        scheduler_id="marginal_voi",
        metric_values=(("AMOTA", 0.7), ("HOTA", 0.69)),
        on_wire_bytes_total=32_000,
        wire_measurement_duration_seconds=1.0,
        on_wire_bytes_per_second=32_000.0,
        status=RUN_RESULT_STATUS_SUCCESS_V1,
        failure_code=None,
        checkpoint_sha256=plan.run_checkpoint_sha256(
            "primary", "eventtrack-v2x", "marginal_voi", 1337
        ),
        evidence_bundle_sha256="a" * 64,
        metrics_artifact_role=RESULT_METRICS_ARTIFACT_ROLE_V1,
        metrics_artifact_sha256="b" * 64,
        channel_outcome_trace_sha256="c" * 64,
        network_trace_sha256="d" * 64,
        condition_input_manifest_sha256=None,
        wire_ledger_sha256="e" * 64,
    )
    return RunResultRegistryV1(
        experiment_plan_sha256=plan.content_sha256,
        attempt_manifest_sha256="f" * 64,
        confirmatory_val_unsealed_at_utc="2026-09-02T00:00:00Z",
        execution_started_at_utc="2026-09-02T00:00:01Z",
        first_prediction_at_utc="2026-09-02T00:00:02Z",
        completed_at_utc="2026-09-02T00:01:00Z",
        network_trace_manifest_sha256=plan.network_trace_manifest_sha256,
        wire_accounting_config_sha256=plan.wire_accounting_config_sha256,
        heldout_c9_trace_receipt_sha256=plan.heldout_c9_trace_receipt_sha256,
        c9_trace_ids=plan.c9_trace_ids,
        cells=(cell,),
    )


@pytest.fixture(scope="module")
def validation_registry(
    development_receipt: DatasetReleaseReceiptV1,
    measured_receipt: MeasuredTraceReceiptV1,
) -> ValidationResultRegistryV1:
    fold_manifest = build_development_split_manifest_v1(
        DEVELOPMENT_SEQUENCES,
        split_sha256=development_receipt.split_sha256,
    )
    cell = ValidationResultCellV1(
        qualification_family=VALIDATION_BASELINE_FAMILY_V1,
        method_id="vehicle-only-ab3dmot",
        scheduler_id="qualification",
        sequence_id=DEVELOPMENT_SEQUENCES[0],
        development_fold_id=fold_manifest.fold_for_sequence(
            DEVELOPMENT_SEQUENCES[0]
        ),
        condition_id="C0",
        training_seed=1337,
        network_seed=None,
        c9_trace_id=None,
        budget_bytes_per_second=64_000,
        metric_values=(("AMOTA", 0.65), ("HOTA", 0.64)),
        on_wire_bytes_total=16_000,
        wire_measurement_duration_seconds=1.0,
        on_wire_bytes_per_second=16_000.0,
        status=VALIDATION_STATUS_SUCCESS_V1,
        failure_code=None,
        checkpoint_sha256=hashlib.sha256(b"development-oof-checkpoint").hexdigest(),
        evidence_bundle_sha256="1" * 64,
        metrics_artifact_role=VALIDATION_METRICS_ARTIFACT_ROLE_V1,
        metrics_artifact_sha256="2" * 64,
        channel_outcome_trace_sha256="3" * 64,
        network_trace_sha256="4" * 64,
        condition_input_manifest_sha256=None,
        wire_ledger_sha256="5" * 64,
    )
    return ValidationResultRegistryV1(
        source_tree_sha256="1" * 64,
        dataset_id="v2x-seq-spd",
        dataset_manifest_sha256=development_receipt.dataset_manifest_sha256,
        split_name="train",
        split_sha256=development_receipt.split_sha256,
        cohort_sha256=development_receipt.cohort_sha256,
        sequence_ids=DEVELOPMENT_SEQUENCES,
        development_fold_manifest=fold_manifest,
        detection_cache_sha256="c" * 64,
        detection_frame_contract_sha256=development_receipt.frame_contract_sha256,
        network_trace_manifest_sha256="6" * 64,
        measured_trace_receipt_sha256=measured_receipt.content_sha256,
        c9_setup_trace_ids=SETUP_TRACE_IDS,
        wire_accounting_config_sha256="0" * 64,
        evaluator_contract_sha256="e" * 64,
        frequency_hz=10.0,
        class_names=("car",),
        cells=(cell,),
    )


@pytest.fixture(scope="module")
def candidate_selection_registry(
    validation_registry: ValidationResultRegistryV1,
) -> CandidateSelectionRegistryV1:
    entries = tuple(
        CandidateSelectionEntryV1(
            fixed_lag_seconds=combination[0],
            mixture_components=combination[1],
            top_h=combination[2],
            gate_confidence=combination[3],
            config_sha256=(
                "2" * 64
                if index == 0
                else hashlib.sha256(
                    f"candidate-config-{index}".encode()
                ).hexdigest()
            ),
            clean_amota=0.65,
            clean_hota=0.64,
            robust_assa_at_64k=0.9 if index == 0 else 0.8,
            actual_bytes_per_second=63_000.0,
            p95_latency_ms=20.0 + index,
            oof_checkpoint_inventory_sha256=hashlib.sha256(
                f"candidate-oof-checkpoints-{index}".encode()
            ).hexdigest(),
            evidence_sha256=hashlib.sha256(
                f"candidate-evidence-{index}".encode()
            ).hexdigest(),
        )
        for index, combination in enumerate(CANDIDATE_COMBINATIONS_V1)
    )
    return CandidateSelectionRegistryV1(
        registry_id="eventtrack-v2x-candidate-selection-v1",
        source_tree_sha256=validation_registry.source_tree_sha256,
        dataset_id=validation_registry.dataset_id,
        dataset_manifest_sha256=validation_registry.dataset_manifest_sha256,
        split_name=validation_registry.split_name,
        split_sha256=validation_registry.split_sha256,
        cohort_sha256=validation_registry.cohort_sha256,
        sequence_ids=validation_registry.sequence_ids,
        development_fold_manifest_sha256=(
            validation_registry.development_fold_manifest.content_sha256
        ),
        frequency_hz=validation_registry.frequency_hz,
        class_names=validation_registry.class_names,
        detection_cache_sha256=validation_registry.detection_cache_sha256,
        detection_frame_contract_sha256=(
            validation_registry.detection_frame_contract_sha256
        ),
        network_trace_manifest_sha256=(
            validation_registry.network_trace_manifest_sha256
        ),
        wire_accounting_config_sha256=(
            validation_registry.wire_accounting_config_sha256
        ),
        evaluator_contract_sha256=(
            validation_registry.evaluator_contract_sha256
        ),
        validation_result_registry_sha256=validation_registry.digest(),
        baseline_qualification_registry_sha256="1" * 64,
        strongest_baseline_method_id="vehicle-only-ab3dmot",
        reference_clean_amota=0.65,
        reference_clean_hota=0.64,
        method_id="eventtrack-v2x",
        scheduler_id="marginal_voi",
        training_seeds=(1337, 2027, 3407),
        network_seeds=tuple(range(1001, 1011)),
        primary_budget_bytes_per_second=64_000,
        clean_noninferiority_margin=-0.01,
        entries=entries,
        selected_method_config_sha256="2" * 64,
    )


@pytest.fixture(scope="module")
def invariants(evidence: EvidenceBundleV1) -> tuple[InvariantVerificationV1, ...]:
    return tuple(
        InvariantVerificationV1(
            invariant_id=invariant_id,
            passed=True,
            artifact_role="logs",
            artifact_sha256=evidence.logs.sha256,
        )
        for invariant_id in REQUIRED_INVARIANTS_V1
    )


@pytest.fixture(scope="module")
def cold_inventory(
    evidence: EvidenceBundleV1,
) -> tuple[ColdCacheArtifactRecordV1, ...]:
    return tuple(
        ColdCacheArtifactRecordV1(
            artifact_role=role,
            sha256=getattr(evidence, role).sha256,
            byte_size=getattr(evidence, role).byte_size,
        )
        for role in sorted(EvidenceBundleV1._ARTIFACT_FIELDS)
    )


@pytest.fixture(scope="module")
def report(
    plan: ExperimentPlanV1,
    run_registry: RunResultRegistryV1,
    validation_registry: ValidationResultRegistryV1,
    candidate_selection_registry: CandidateSelectionRegistryV1,
    evidence: EvidenceBundleV1,
    development_receipt: DatasetReleaseReceiptV1,
    primary_receipt: DatasetReleaseReceiptV1,
    griffin_receipt: DatasetReleaseReceiptV1,
    measured_receipt: MeasuredTraceReceiptV1,
    invariants: tuple[InvariantVerificationV1, ...],
    cold_inventory: tuple[ColdCacheArtifactRecordV1, ...],
):
    return build_independent_verification_report_v1(
        experiment_plan=plan,
        run_result_registry=run_registry,
        validation_result_registry=validation_registry,
        candidate_selection_registry=candidate_selection_registry,
        aggregate_evidence_bundle=evidence,
        development_dataset_receipt=development_receipt,
        primary_dataset_receipt=primary_receipt,
        griffin_dataset_receipt=griffin_receipt,
        measured_trace_receipt=measured_receipt,
        invariants=invariants,
        cold_cache_inventory=(item for item in cold_inventory),
    )


def test_build_binds_every_artifact_and_recomputed_inventory(
    report,
    run_registry: RunResultRegistryV1,
    validation_registry: ValidationResultRegistryV1,
    candidate_selection_registry: CandidateSelectionRegistryV1,
) -> None:
    assert report.evaluator_recomputation.cell_count == 2
    assert report.wire_ledger_replay.cell_count == 2
    assert report.evaluator_recomputation == evaluator_recomputation_inventory_v1(
        run_registry, validation_registry
    )
    assert report.wire_ledger_replay == wire_ledger_replay_inventory_v1(
        run_registry, validation_registry
    )
    assert report.candidate_selection_registry_content_sha256 == (
        candidate_selection_registry.content_sha256
    )
    assert report.candidate_selection_inventory.cell_count == 144
    assert report.candidate_selection_inventory == (
        candidate_selection_inventory_v1(candidate_selection_registry)
    )
    assert len(report.cold_cache_inventory) == len(
        EvidenceBundleV1._ARTIFACT_FIELDS
    )
    assert len(report.c9_packet_metadata_inventory) == 20
    assert tuple(item.invariant_id for item in report.invariants) == (
        REQUIRED_INVARIANTS_V1
    )
    assert not report.measured_trace_supports_clock_claims


def test_candidate_inventory_binds_config_evidence_and_metrics(
    candidate_selection_registry: CandidateSelectionRegistryV1,
) -> None:
    original = candidate_selection_inventory_v1(candidate_selection_registry)
    assert original.cell_count == len(CANDIDATE_COMBINATIONS_V1) == 144

    variants = (
        replace(candidate_selection_registry.entries[1], config_sha256="f" * 64),
        replace(candidate_selection_registry.entries[1], evidence_sha256="e" * 64),
        replace(candidate_selection_registry.entries[1], robust_assa_at_64k=0.79),
    )
    for changed_entry in variants:
        changed_entries = list(candidate_selection_registry.entries)
        changed_entries[1] = changed_entry
        changed = replace(
            candidate_selection_registry, entries=tuple(changed_entries)
        )
        assert candidate_selection_inventory_v1(changed) != original


def test_verification_inventories_bind_condition_input_manifest(
    run_registry: RunResultRegistryV1,
    validation_registry: ValidationResultRegistryV1,
) -> None:
    c6_cell = replace(
        run_registry.cells[0],
        condition_id="C6",
        network_seed=1001,
        c9_trace_id=None,
        condition_input_manifest_sha256="a" * 64,
    )
    first = replace(run_registry, cells=(c6_cell,))
    second = replace(
        first,
        cells=(
            replace(c6_cell, condition_input_manifest_sha256="b" * 64),
        ),
    )
    assert evaluator_recomputation_inventory_v1(
        first, validation_registry
    ) != evaluator_recomputation_inventory_v1(second, validation_registry)
    assert wire_ledger_replay_inventory_v1(
        first, validation_registry
    ) != wire_ledger_replay_inventory_v1(second, validation_registry)


def test_measured_clock_claim_capability_requires_validated_packet_contents(
    report,
    measured_receipt: MeasuredTraceReceiptV1,
) -> None:
    assert measured_receipt.supports_clock_claims
    assert not report.measured_trace_supports_clock_claims
    _validate_measured_trace_clock_claims(report, measured_receipt)

    unsupported = replace(
        measured_receipt,
        segments=tuple(
            replace(segment, clock_reference=ClockReference.NONE)
            for segment in measured_receipt.segments
        ),
    )
    assert not unsupported.supports_clock_claims
    _validate_measured_trace_clock_claims(report, unsupported)
    with pytest.raises(VerificationReportError, match="content-validated"):
        replace(report, measured_trace_supports_clock_claims=True)
    with pytest.raises(TypeError, match="must be bool"):
        replace(report, measured_trace_supports_clock_claims=1)


def test_validation_c9_is_bound_to_receipt_setup_traces_only(
    validation_registry: ValidationResultRegistryV1,
    measured_receipt: MeasuredTraceReceiptV1,
) -> None:
    assert validation_registry.c9_setup_trace_ids == measured_receipt.setup_trace_ids
    assert len(validation_registry.c9_setup_trace_ids) == 10
    _validate_validation_c9_setup_bindings(
        validation_registry, measured_receipt
    )

    c9_cell = replace(
        validation_registry.cells[0],
        condition_id="C9",
        network_seed=None,
        c9_trace_id=SETUP_TRACE_IDS[0],
        metric_values=(("AssA", 0.6),),
        channel_outcome_trace_sha256=f"{1:064x}",
    )
    c9_registry = replace(validation_registry, cells=(c9_cell,))
    _validate_validation_c9_setup_bindings(c9_registry, measured_receipt)
    with pytest.raises(ScientificGateError, match="setup packet metadata"):
        _validate_validation_c9_setup_bindings(
            replace(
                c9_registry,
                cells=(
                    replace(c9_cell, channel_outcome_trace_sha256="f" * 64),
                ),
            ),
            measured_receipt,
        )

    with pytest.raises(ScientificGateError, match="receipt or trace IDs"):
        _validate_validation_c9_setup_bindings(
            replace(
                validation_registry,
                c9_setup_trace_ids=tuple(
                    f"wrong-setup-trace-{index:02d}" for index in range(10)
                ),
            ),
            measured_receipt,
        )
    with pytest.raises(ScientificGateError, match="receipt or trace IDs"):
        _validate_validation_c9_setup_bindings(
            replace(
                validation_registry,
                measured_trace_receipt_sha256="f" * 64,
            ),
            measured_receipt,
        )


def test_external_validation_rejects_stale_bindings_and_cell_semantics(
    report,
    plan: ExperimentPlanV1,
    run_registry: RunResultRegistryV1,
    validation_registry: ValidationResultRegistryV1,
    candidate_selection_registry: CandidateSelectionRegistryV1,
    evidence: EvidenceBundleV1,
    development_receipt: DatasetReleaseReceiptV1,
    primary_receipt: DatasetReleaseReceiptV1,
    griffin_receipt: DatasetReleaseReceiptV1,
    measured_receipt: MeasuredTraceReceiptV1,
) -> None:
    kwargs = dict(
        experiment_plan=plan,
        run_result_registry=run_registry,
        validation_result_registry=validation_registry,
        candidate_selection_registry=candidate_selection_registry,
        aggregate_evidence_bundle=evidence,
        development_dataset_receipt=development_receipt,
        primary_dataset_receipt=primary_receipt,
        griffin_dataset_receipt=griffin_receipt,
        measured_trace_receipt=measured_receipt,
    )
    validate_independent_verification_report_v1(report, **kwargs)
    with pytest.raises(VerificationReportError, match="stale evidence bindings"):
        validate_independent_verification_report_v1(
            replace(report, experiment_plan_sha256="f" * 64), **kwargs
        )
    changed_summary = VerificationInventorySummaryV1(
        cell_count=3,
        inventory_sha256=report.evaluator_recomputation.inventory_sha256,
    )
    with pytest.raises(VerificationReportError, match="evaluator recomputation"):
        validate_independent_verification_report_v1(
            replace(report, evaluator_recomputation=changed_summary), **kwargs
        )
    changed_candidates = VerificationInventorySummaryV1(
        cell_count=143,
        inventory_sha256=report.candidate_selection_inventory.inventory_sha256,
    )
    with pytest.raises(VerificationReportError, match="candidate-selection inventory"):
        validate_independent_verification_report_v1(
            replace(
                report,
                candidate_selection_inventory=changed_candidates,
            ),
            **kwargs,
        )


def test_missing_invariant_cold_artifact_and_c9_tamper_fail_closed(report) -> None:
    with pytest.raises(VerificationReportError, match="every required invariant"):
        replace(report, invariants=report.invariants[:-1])
    with pytest.raises(VerificationReportError, match="cold-cache inventory hash"):
        changed = replace(report.cold_cache_inventory[0], sha256="f" * 64)
        replace(
            report,
            cold_cache_inventory=(changed, *report.cold_cache_inventory[1:]),
        )
    with pytest.raises(VerificationReportError, match="C9 packet inventory hash"):
        changed_c9 = replace(
            report.c9_packet_metadata_inventory[0],
            packet_metadata_sha256="f" * 64,
        )
        replace(
            report,
            c9_packet_metadata_inventory=(
                changed_c9,
                *report.c9_packet_metadata_inventory[1:],
            ),
        )
    with pytest.raises(VerificationReportError, match="invariant must pass"):
        replace(report.invariants[0], passed=False)


def test_metric_and_wire_changes_change_semantic_inventory(
    run_registry: RunResultRegistryV1,
    validation_registry: ValidationResultRegistryV1,
) -> None:
    changed_metric_cell = replace(
        run_registry.cells[0], metric_values=(("AMOTA", 0.5), ("HOTA", 0.49))
    )
    changed_metric_registry = replace(run_registry, cells=(changed_metric_cell,))
    assert evaluator_recomputation_inventory_v1(
        changed_metric_registry, validation_registry
    ) != evaluator_recomputation_inventory_v1(run_registry, validation_registry)

    changed_wire_cell = replace(
        run_registry.cells[0],
        on_wire_bytes_total=31_000,
        on_wire_bytes_per_second=31_000.0,
    )
    changed_wire_registry = replace(run_registry, cells=(changed_wire_cell,))
    assert wire_ledger_replay_inventory_v1(
        changed_wire_registry, validation_registry
    ) != wire_ledger_replay_inventory_v1(run_registry, validation_registry)


def test_canonical_round_trip_and_tamper_rejection(report) -> None:
    decoded = decode_independent_verification_report(report.canonical_bytes)
    assert decoded == report
    assert decoded.content_sha256 == report.content_sha256
    assert decoded.digest() == report.digest()

    document = report.sealed_document()
    document["experiment_plan_sha256"] = "f" * 64
    with pytest.raises(VerificationReportError, match="content hash mismatch"):
        decode_independent_verification_report(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        )
    with pytest.raises(VerificationReportError, match="duplicate JSON key"):
        decode_independent_verification_report(b'{"kind":"x","kind":"y"}')
    with pytest.raises(VerificationReportError, match="not canonical"):
        decode_independent_verification_report(
            json.dumps(report.sealed_document(), separators=(", ", ": ")).encode()
        )

    forged_clock_payload = report.payload()
    forged_clock_payload["measured_trace_supports_clock_claims"] = True
    forged_clock_document = {
        **forged_clock_payload,
        "content_sha256": hashlib.sha256(
            canonical_json_bytes(forged_clock_payload)
        ).hexdigest(),
    }
    with pytest.raises(VerificationReportError, match="content-validated"):
        decode_independent_verification_report(
            canonical_json_bytes(forged_clock_document)
        )
