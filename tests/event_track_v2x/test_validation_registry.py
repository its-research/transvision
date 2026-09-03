from dataclasses import replace
import hashlib
import json

import pytest

import transvision.models.event_track_v2x.validation_registry as validation_registry_module
from transvision.models.event_track_v2x.candidate_selection import (
    CANDIDATE_COMBINATIONS_V1,
    CandidateSelectionEntryV1,
    CandidateSelectionRegistryV1,
)
from transvision.models.event_track_v2x.contracts import (
    ArtifactDigestV1,
    EvidenceBundleV1,
)
from transvision.models.event_track_v2x.development_split import (
    build_development_split_manifest_v1,
)
from transvision.models.event_track_v2x.experiment import (
    CONFIRMATORY_METHOD_IDS_V1,
    CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1,
    GRIFFIN_25M_VAL_SEQUENCE_IDS_V1,
    CONFIRMATORY_QUALIFIED_SCHEDULER_BASELINE_IDS_V1,
    ExperimentPlanV1,
    RunConfigBindingV1,
    confirmatory_run_config_keys_v1,
)
from transvision.models.event_track_v2x.measured_trace import (
    ClockReference,
    MeasuredTraceReceiptV1,
    MeasuredTraceSegmentV1,
    REQUIRED_PACKET_FIELDS_V1,
    TraceRole,
)
from transvision.models.event_track_v2x.network import (
    NetworkConditionId,
    condition_plan_v1,
)
from transvision.models.event_track_v2x.network_disturbance import (
    ConditionInputManifestV1,
)
from transvision.models.event_track_v2x.qualification import (
    OFFICIAL_TRACKER_METHOD_TO_BACKEND_V1,
    BaselineQualificationEntryV1,
    BaselineQualificationRegistryV1,
    QualificationRegistryError,
    SchedulerQualificationEntryV1,
    validate_qualification_against_raw_registry_v1,
)
from transvision.models.event_track_v2x.scientific_gate import (
    ScientificGateError,
    _validate_candidate_selection,
    _validate_qualification,
)
from transvision.models.event_track_v2x.validation_registry import (
    VALIDATION_BASELINE_FAMILY_V1,
    VALIDATION_METRICS_ARTIFACT_ROLE_V1,
    VALIDATION_SCHEDULER_FAMILY_V1,
    VALIDATION_STATUS_FAILURE_V1,
    VALIDATION_STATUS_SUCCESS_V1,
    ValidationRegistryError,
    ValidationResultCellV1,
    ValidationResultRegistryV1,
    _normalized_actual_byte_aucs_common_support,
    decode_validation_result_registry,
    derive_validation_summaries_v1,
    iter_expected_validation_specs_v1,
    validate_validation_result_external_inputs_v1,
    validate_validation_result_registry_v1,
    validation_physical_run_id_v1,
)


SEQUENCES = tuple(f"sequence-{index:02d}" for index in range(46))
SETUP_TRACE_IDS = tuple(f"setup-trace-{index:02d}" for index in range(10))
MEASURED_TRACE_RECEIPT_SHA256 = "0" * 64
FOLD_MANIFEST = build_development_split_manifest_v1(
    SEQUENCES,
    split_sha256="3" * 64,
)


def _digest(value: object) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _measured_trace_receipt() -> MeasuredTraceReceiptV1:
    segments = []
    for index in range(30):
        setup = index < 10
        trace_index = index if setup else index - 10
        segments.append(
            MeasuredTraceSegmentV1(
                trace_id=(
                    f"setup-trace-{trace_index:02d}"
                    if setup
                    else f"heldout-{trace_index:02d}"
                ),
                role=TraceRole.SETUP if setup else TraceRole.HELD_OUT,
                date_cohort_id=f"date-{index % 3}",
                road_coverage_type=f"coverage-{index % 3}",
                duration_seconds=600,
                packet_count=100 + index,
                packet_metadata_sha256=f"{index + 1:064x}",
                packet_fields=REQUIRED_PACKET_FIELDS_V1,
                clock_reference=ClockReference.PTP,
            )
        )
    return MeasuredTraceReceiptV1(
        receipt_id="validation-c9-receipt-v1",
        segments=tuple(segments),
        collection_protocol_sha256="a" * 64,
    )


def _sort_key(cell: ValidationResultCellV1) -> tuple[str, ...]:
    return tuple(
        "" if value is None else f"{type(value).__name__}:{value}" for value in cell.key
    )


def _metrics(spec) -> tuple[tuple[str, float], ...]:
    if spec.condition_id == "C0":
        return (("AMOTA", 0.70), ("HOTA", 0.69))
    if spec.qualification_family == VALIDATION_SCHEDULER_FAMILY_V1:
        value = 0.50 + spec.budget_bytes_per_second / 1_000_000.0
        return (("AssA", value),)
    return (("AssA", 0.60),)


@pytest.fixture(scope="module")
def cells() -> tuple[ValidationResultCellV1, ...]:
    result = []
    for spec in iter_expected_validation_specs_v1(FOLD_MANIFEST, SETUP_TRACE_IDS):
        trace_key = (
            spec.sequence_id,
            spec.condition_id,
            spec.network_seed,
            spec.c9_trace_id,
        )
        actual_bytes = spec.budget_bytes_per_second // 2
        result.append(
            ValidationResultCellV1(
                qualification_family=spec.qualification_family,
                method_id=spec.method_id,
                scheduler_id=spec.scheduler_id,
                sequence_id=spec.sequence_id,
                development_fold_id=spec.development_fold_id,
                condition_id=spec.condition_id,
                training_seed=spec.training_seed,
                network_seed=spec.network_seed,
                c9_trace_id=spec.c9_trace_id,
                budget_bytes_per_second=spec.budget_bytes_per_second,
                metric_values=_metrics(spec),
                on_wire_bytes_total=actual_bytes,
                wire_measurement_duration_seconds=1.0,
                on_wire_bytes_per_second=float(actual_bytes),
                status=VALIDATION_STATUS_SUCCESS_V1,
                failure_code=None,
                checkpoint_sha256=_digest(
                    (
                        spec.qualification_family,
                        spec.method_id,
                        spec.scheduler_id,
                        spec.development_fold_id,
                        spec.training_seed,
                        "checkpoint",
                    )
                ),
                evidence_bundle_sha256=_digest((spec.key, "bundle")),
                metrics_artifact_role=VALIDATION_METRICS_ARTIFACT_ROLE_V1,
                metrics_artifact_sha256=_digest((spec.key, "metrics")),
                channel_outcome_trace_sha256=_digest(trace_key),
                network_trace_sha256=_digest((spec.key, "network")),
                condition_input_manifest_sha256=(
                    _digest((spec.key, "condition-input"))
                    if spec.condition_id in {"C6", "C7", "C8"}
                    else None
                ),
                wire_ledger_sha256=_digest((spec.key, "wire")),
            )
        )
    return tuple(sorted(result, key=_sort_key))


@pytest.fixture(scope="module")
def registry(
    cells: tuple[ValidationResultCellV1, ...],
) -> ValidationResultRegistryV1:
    return ValidationResultRegistryV1(
        source_tree_sha256="1" * 64,
        dataset_id="v2x-seq-spd",
        dataset_manifest_sha256="2" * 64,
        split_name="train",
        split_sha256="3" * 64,
        cohort_sha256="4" * 64,
        sequence_ids=SEQUENCES,
        development_fold_manifest=FOLD_MANIFEST,
        detection_cache_sha256="5" * 64,
        detection_frame_contract_sha256="6" * 64,
        network_trace_manifest_sha256="7" * 64,
        measured_trace_receipt_sha256=MEASURED_TRACE_RECEIPT_SHA256,
        c9_setup_trace_ids=SETUP_TRACE_IDS,
        wire_accounting_config_sha256="8" * 64,
        evaluator_contract_sha256="9" * 64,
        frequency_hz=10.0,
        class_names=("car",),
        cells=cells,
    )


def _artifact(uri: str, sha256: str) -> ArtifactDigestV1:
    return ArtifactDigestV1(uri=uri, sha256=sha256, byte_size=1)


def _evidence(
    registry: ValidationResultRegistryV1,
    run_id: str,
    *,
    config_sha256: str,
    predictions_sha256: str,
    metrics_sha256: str,
) -> EvidenceBundleV1:
    return EvidenceBundleV1(
        run_id=run_id,
        source=_artifact("source.json", registry.source_tree_sha256),
        dataset=_artifact("dataset.json", registry.dataset_manifest_sha256),
        detection_cache=_artifact("cache.json", registry.detection_cache_sha256),
        network_trace=_artifact("trace.json", registry.network_trace_manifest_sha256),
        tracker_config=_artifact("config.json", config_sha256),
        evaluator_contract=_artifact(
            "evaluator.json", registry.evaluator_contract_sha256
        ),
        checkpoint=_artifact("checkpoint.json", _digest((run_id, "checkpoint"))),
        predictions=_artifact("predictions.json", predictions_sha256),
        per_sequence_metrics=_artifact("metrics.json", metrics_sha256),
        logs=_artifact("logs.json", _digest((run_id, "logs"))),
        environment=_artifact("environment.json", _digest((run_id, "environment"))),
    )


def _qualification(
    registry: ValidationResultRegistryV1,
) -> BaselineQualificationRegistryV1:
    summaries = derive_validation_summaries_v1(registry)
    baseline_summaries = {item.method_id: item for item in summaries.baselines}
    scheduler_summaries = {item.scheduler_id: item for item in summaries.schedulers}
    selected_backend_source = _digest("ab3dmot-upstream")
    baselines = []
    for method_id in CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1:
        backend = OFFICIAL_TRACKER_METHOD_TO_BACKEND_V1.get(method_id)
        backend_id = backend or "ab3dmot"
        backend_source = (
            _digest(f"{backend_id}-upstream")
            if backend is not None
            else selected_backend_source
        )
        config = _digest((method_id, "config"))
        predictions = _digest((method_id, "predictions"))
        metrics = _digest((method_id, "metrics"))
        evidence = _evidence(
            registry,
            f"baseline-{method_id}",
            config_sha256=config,
            predictions_sha256=predictions,
            metrics_sha256=metrics,
        )
        summary = baseline_summaries[method_id]
        baselines.append(
            BaselineQualificationEntryV1(
                method_id=method_id,
                tracker_backend_id=backend_id,
                qualification_kind=(
                    "official-backend"
                    if backend is not None
                    else "shared-selected-backend"
                ),
                backend_source_sha256=backend_source,
                config_sha256=config,
                predictions_sha256=predictions,
                metrics_sha256=metrics,
                evidence_bundle_sha256=evidence.digest(),
                evidence_bundle=evidence,
                failure_count=summary.failure_count,
                robust_assa_at_64k=summary.robust_assa_at_64k,
                clean_amota=summary.clean_amota,
                clean_hota=summary.clean_hota,
                passed=True,
            )
        )
    schedulers = []
    for scheduler_id in CONFIRMATORY_QUALIFIED_SCHEDULER_BASELINE_IDS_V1:
        config = _digest((scheduler_id, "config"))
        predictions = _digest((scheduler_id, "predictions"))
        metrics = _digest((scheduler_id, "metrics"))
        evidence = _evidence(
            registry,
            f"scheduler-{scheduler_id}",
            config_sha256=config,
            predictions_sha256=predictions,
            metrics_sha256=metrics,
        )
        summary = scheduler_summaries[scheduler_id]
        schedulers.append(
            SchedulerQualificationEntryV1(
                scheduler_id=scheduler_id,
                config_sha256=config,
                predictions_sha256=predictions,
                metrics_sha256=metrics,
                evidence_bundle_sha256=evidence.digest(),
                evidence_bundle=evidence,
                failure_count=summary.failure_count,
                robust_assa_at_64k=summary.robust_assa_at_64k,
                pareto_auc=summary.actual_byte_auc,
                passed=True,
            )
        )
    return BaselineQualificationRegistryV1(
        registry_id="eventtrack-v2x-development-qualification-v1",
        source_tree_sha256=registry.source_tree_sha256,
        dataset_id=registry.dataset_id,
        dataset_manifest_sha256=registry.dataset_manifest_sha256,
        split_name=registry.split_name,
        split_sha256=registry.split_sha256,
        cohort_sha256=registry.cohort_sha256,
        sequence_ids=registry.sequence_ids,
        development_fold_manifest_sha256=(
            registry.development_fold_manifest.content_sha256
        ),
        frequency_hz=registry.frequency_hz,
        class_names=registry.class_names,
        detection_cache_sha256=registry.detection_cache_sha256,
        detection_frame_contract_sha256=(registry.detection_frame_contract_sha256),
        network_trace_manifest_sha256=registry.network_trace_manifest_sha256,
        evaluator_contract_sha256=registry.evaluator_contract_sha256,
        training_seeds=(1337, 2027, 3407),
        network_seeds=tuple(range(1001, 1011)),
        primary_budget_bytes_per_second=64_000,
        validation_result_registry_sha256=registry.digest(),
        baseline_entries=tuple(baselines),
        selected_tracker_backend_id="ab3dmot",
        strongest_qualified_baseline_id="constant-velocity-compensation",
        scheduler_entries=tuple(schedulers),
        strongest_scheduler_baseline_id="confidence_top_k",
    )


def _plan(
    raw_registry: ValidationResultRegistryV1,
    qualification: BaselineQualificationRegistryV1,
) -> ExperimentPlanV1:
    baseline_configs = {
        entry.method_id: entry.config_sha256 for entry in qualification.baseline_entries
    }
    scheduler_configs = {
        entry.scheduler_id: entry.config_sha256
        for entry in qualification.scheduler_entries
    }

    def config_sha256(domain: str, method: str, scheduler: str) -> str:
        if domain == "primary" and method in baseline_configs:
            return baseline_configs[method]
        if (
            domain == "primary"
            and method == "eventtrack-v2x"
            and scheduler == qualification.strongest_scheduler_baseline_id
        ):
            return scheduler_configs[scheduler]
        if domain == "primary" and method == "eventtrack-v2x":
            return "a" * 64
        return _digest((domain, method, scheduler, "frozen-config"))

    run_configs = tuple(
        RunConfigBindingV1(
            dataset_domain=domain,
            method_id=method,
            scheduler_id=scheduler,
            config_sha256=config_sha256(domain, method, scheduler),
            checkpoint_sha256s=tuple(
                _digest((domain, method, scheduler, seed, "final-refit"))
                for seed in (1337, 2027, 3407)
            ),
        )
        for domain, method, scheduler in confirmatory_run_config_keys_v1(
            strongest_baseline_id=qualification.strongest_qualified_baseline_id,
            strongest_scheduler_id=qualification.strongest_scheduler_baseline_id,
        )
    )
    return ExperimentPlanV1(
        plan_id="eventtrack-v2x-confirmatory-v1",
        source_tree_sha256=raw_registry.source_tree_sha256,
        method_config_sha256="a" * 64,
        run_config_bindings=run_configs,
        dataset_id="v2x-seq-spd",
        dataset_manifest_sha256=raw_registry.dataset_manifest_sha256,
        split_name="val",
        split_sha256="b" * 64,
        primary_cohort_sha256="c" * 64,
        primary_frame_contract_sha256="d" * 64,
        primary_dataset_release_receipt_sha256="e" * 64,
        development_dataset_release_receipt_sha256="3" * 64,
        development_fold_manifest_sha256=(
            raw_registry.development_fold_manifest.content_sha256
        ),
        primary_sequence_ids=tuple(f"val-sequence-{index:02d}" for index in range(21)),
        detector_cache_sha256="f" * 64,
        network_trace_manifest_sha256=(raw_registry.network_trace_manifest_sha256),
        wire_accounting_config_sha256=(raw_registry.wire_accounting_config_sha256),
        heldout_c9_trace_receipt_sha256="0" * 64,
        c9_trace_ids=tuple(f"heldout-{index:02d}" for index in range(20)),
        evaluator_contract_sha256="1" * 64,
        baseline_qualification_registry_sha256=qualification.digest(),
        candidate_selection_registry_sha256="2" * 64,
        preregistration_provider_id="institutional-log",
        preregistration_public_key_sha256="8" * 64,
        verification_provider_id="independent-verifier",
        verification_public_key_sha256="9" * 64,
        qualified_baseline_ids=CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1,
        strongest_qualified_baseline_id="constant-velocity-compensation",
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
        griffin_dataset_manifest_sha256="2" * 64,
        griffin_split_name="val",
        griffin_cohort_sha256="3" * 64,
        griffin_frame_contract_sha256="4" * 64,
        griffin_dataset_release_receipt_sha256="5" * 64,
        griffin_sequence_ids=GRIFFIN_25M_VAL_SEQUENCE_IDS_V1,
        griffin_detector_cache_sha256="6" * 64,
        griffin_evaluator_contract_sha256="7" * 64,
        griffin_strongest_baseline_id="constant-velocity-compensation",
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


def _candidate_selection(
    raw_registry: ValidationResultRegistryV1,
    qualification: BaselineQualificationRegistryV1,
) -> CandidateSelectionRegistryV1:
    strongest = next(
        entry
        for entry in qualification.baseline_entries
        if entry.method_id == qualification.strongest_qualified_baseline_id
    )
    entries = tuple(
        CandidateSelectionEntryV1(
            fixed_lag_seconds=combination[0],
            mixture_components=combination[1],
            top_h=combination[2],
            gate_confidence=combination[3],
            config_sha256=(
                "a" * 64 if index == 0 else _digest(("candidate-config", index))
            ),
            clean_amota=strongest.clean_amota,
            clean_hota=strongest.clean_hota,
            robust_assa_at_64k=0.9 if index == 0 else 0.8,
            actual_bytes_per_second=63_000.0,
            p95_latency_ms=20.0 + index,
            oof_checkpoint_inventory_sha256=_digest(
                ("candidate-oof-checkpoint-inventory", index)
            ),
            evidence_sha256=_digest(("candidate-evidence", index)),
        )
        for index, combination in enumerate(CANDIDATE_COMBINATIONS_V1)
    )
    return CandidateSelectionRegistryV1(
        registry_id="eventtrack-v2x-candidate-selection-v1",
        source_tree_sha256=raw_registry.source_tree_sha256,
        dataset_id=raw_registry.dataset_id,
        dataset_manifest_sha256=raw_registry.dataset_manifest_sha256,
        split_name=raw_registry.split_name,
        split_sha256=raw_registry.split_sha256,
        cohort_sha256=raw_registry.cohort_sha256,
        sequence_ids=raw_registry.sequence_ids,
        development_fold_manifest_sha256=(
            raw_registry.development_fold_manifest.content_sha256
        ),
        frequency_hz=raw_registry.frequency_hz,
        class_names=raw_registry.class_names,
        detection_cache_sha256=raw_registry.detection_cache_sha256,
        detection_frame_contract_sha256=(raw_registry.detection_frame_contract_sha256),
        network_trace_manifest_sha256=(raw_registry.network_trace_manifest_sha256),
        wire_accounting_config_sha256=(raw_registry.wire_accounting_config_sha256),
        evaluator_contract_sha256=raw_registry.evaluator_contract_sha256,
        validation_result_registry_sha256=raw_registry.digest(),
        baseline_qualification_registry_sha256=qualification.digest(),
        strongest_baseline_method_id=(qualification.strongest_qualified_baseline_id),
        reference_clean_amota=strongest.clean_amota,
        reference_clean_hota=strongest.clean_hota,
        method_id="eventtrack-v2x",
        scheduler_id="marginal_voi",
        training_seeds=(1337, 2027, 3407),
        network_seeds=tuple(range(1001, 1011)),
        primary_budget_bytes_per_second=64_000,
        clean_noninferiority_margin=-0.01,
        entries=entries,
        selected_method_config_sha256="a" * 64,
    )


def test_exact_matrix_derives_only_raw_evidence_summaries(
    registry: ValidationResultRegistryV1,
) -> None:
    assert len(registry.cells) == 323_748
    baseline_replicates = [
        cell
        for cell in registry.cells
        if cell.qualification_family == VALIDATION_BASELINE_FAMILY_V1
        and cell.method_id == CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1[0]
        and cell.sequence_id == SEQUENCES[0]
    ]
    scheduler_replicates = [
        cell
        for cell in registry.cells
        if cell.qualification_family == VALIDATION_SCHEDULER_FAMILY_V1
        and cell.scheduler_id == CONFIRMATORY_QUALIFIED_SCHEDULER_BASELINE_IDS_V1[0]
        and cell.sequence_id == SEQUENCES[0]
        and cell.budget_bytes_per_second == 64_000
    ]
    for condition_id in tuple(f"C{index}" for index in range(1, 10)):
        assert (
            sum(cell.condition_id == condition_id for cell in baseline_replicates) == 30
        )
        assert (
            sum(cell.condition_id == condition_id for cell in scheduler_replicates)
            == 30
        )
    assert all(
        cell.network_seed is None and cell.c9_trace_id in SETUP_TRACE_IDS
        for cell in registry.cells
        if cell.condition_id == "C9"
    )
    assert all(
        cell.network_seed is not None and cell.c9_trace_id is None
        for cell in registry.cells
        if cell.condition_id in tuple(f"C{index}" for index in range(1, 9))
    )
    summaries = derive_validation_summaries_v1(registry)
    assert tuple(item.method_id for item in summaries.baselines) == (
        CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1
    )
    assert tuple(item.scheduler_id for item in summaries.schedulers) == (
        CONFIRMATORY_QUALIFIED_SCHEDULER_BASELINE_IDS_V1
    )
    baseline = summaries.baselines[0]
    assert baseline.robust_assa_at_64k == pytest.approx(0.60)
    assert baseline.clean_amota == pytest.approx(0.70)
    assert baseline.clean_hota == pytest.approx(0.69)
    scheduler = summaries.schedulers[0]
    assert scheduler.robust_assa_at_64k == pytest.approx(0.564)
    assert scheduler.actual_byte_auc == pytest.approx(0.636)
    assert baseline.failure_count == scheduler.failure_count == 0


def test_robust_summary_macro_averages_c1_through_c9_equally(
    registry: ValidationResultRegistryV1,
) -> None:
    method_id = CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1[0]
    changed = tuple(
        replace(cell, metric_values=(("AssA", 0.0),))
        if cell.qualification_family == VALIDATION_BASELINE_FAMILY_V1
        and cell.method_id == method_id
        and cell.condition_id == "C9"
        else cell
        for cell in registry.cells
    )
    summary = derive_validation_summaries_v1(
        replace(registry, cells=changed)
    ).baselines[0]
    assert summary.robust_assa_at_64k == pytest.approx(0.60 * 8.0 / 9.0)


def test_scheduler_auc_uses_one_common_actual_bps_support() -> None:
    aucs = _normalized_actual_byte_aucs_common_support(
        {
            "fifo_aoi": ((10.0, 0.0), (20.0, 1.0), (30.0, 1.0)),
            "confidence_top_k": (
                (15.0, 0.0),
                (25.0, 0.0),
                (35.0, 1.0),
            ),
        }
    )

    assert aucs["fifo_aoi"] == pytest.approx(11.0 / 12.0)
    assert aucs["confidence_top_k"] == pytest.approx(1.0 / 12.0)

    with pytest.raises(ValidationRegistryError, match="no common actual-BPS width"):
        _normalized_actual_byte_aucs_common_support(
            {
                "fifo_aoi": ((1.0, 0.0), (2.0, 1.0)),
                "confidence_top_k": ((3.0, 0.0), (4.0, 1.0)),
            }
        )


def test_compact_qualification_must_equal_raw_val_derivation(
    registry: ValidationResultRegistryV1,
) -> None:
    qualification = _qualification(registry)
    validate_qualification_against_raw_registry_v1(qualification, registry)

    changed = list(qualification.baseline_entries)
    changed[0] = replace(
        changed[0], robust_assa_at_64k=changed[0].robust_assa_at_64k + 0.001
    )
    tampered = replace(qualification, baseline_entries=tuple(changed))
    with pytest.raises(QualificationRegistryError, match="not derived from raw"):
        validate_qualification_against_raw_registry_v1(tampered, registry)


def test_scientific_gate_qualification_stage_requires_raw_registry(
    registry: ValidationResultRegistryV1,
) -> None:
    qualification = _qualification(registry)
    plan = _plan(registry, qualification)
    _validate_qualification(qualification, registry, plan)

    changed_bindings = tuple(
        replace(binding, config_sha256="0" * 64)
        if binding.key
        == (
            "primary",
            qualification.strongest_qualified_baseline_id,
            qualification.strongest_scheduler_baseline_id,
        )
        else binding
        for binding in plan.run_config_bindings
    )
    with pytest.raises(ScientificGateError, match="baseline run config"):
        _validate_qualification(
            qualification,
            registry,
            replace(plan, run_config_bindings=changed_bindings),
        )

    stale = replace(registry, wire_accounting_config_sha256="f" * 64)
    stale_qualification = replace(
        qualification, validation_result_registry_sha256=stale.digest()
    )
    stale_plan = _plan(registry, stale_qualification)
    with pytest.raises(ScientificGateError, match="stale or mismatched"):
        _validate_qualification(stale_qualification, stale, stale_plan)


def test_scientific_gate_binds_candidate_grid_and_selected_config(
    registry: ValidationResultRegistryV1,
) -> None:
    qualification = _qualification(registry)
    candidate = _candidate_selection(registry, qualification)
    plan = replace(
        _plan(registry, qualification),
        candidate_selection_registry_sha256=candidate.digest(),
    )

    _validate_candidate_selection(candidate, qualification, registry, plan)

    stale = replace(
        candidate,
        reference_clean_amota=candidate.reference_clean_amota - 0.001,
    )
    stale_plan = replace(plan, candidate_selection_registry_sha256=stale.digest())
    with pytest.raises(ScientificGateError, match="stale, incomplete, or mismatched"):
        _validate_candidate_selection(stale, qualification, registry, stale_plan)


def test_failure_is_retained_counted_and_scores_zero(
    registry: ValidationResultRegistryV1,
    cells: tuple[ValidationResultCellV1, ...],
) -> None:
    target = next(
        index
        for index, cell in enumerate(cells)
        if cell.qualification_family == VALIDATION_BASELINE_FAMILY_V1
        and cell.method_id == CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1[0]
        and cell.condition_id == "C1"
    )
    changed = list(cells)
    changed[target] = replace(
        changed[target],
        metric_values=(),
        status=VALIDATION_STATUS_FAILURE_V1,
        failure_code="algorithm-crash",
    )
    changed_registry = replace(registry, cells=tuple(changed))
    summary = derive_validation_summaries_v1(changed_registry).baselines[0]
    assert summary.failure_count == 1
    assert summary.robust_assa_at_64k == pytest.approx(0.60 * 12419.0 / 12420.0)


def test_missing_extra_duplicate_and_non_reused_trace_fail_closed(
    registry: ValidationResultRegistryV1,
    cells: tuple[ValidationResultCellV1, ...],
) -> None:
    missing = replace(registry, cells=cells[:-1])
    with pytest.raises(ValidationRegistryError, match="missing validation cells"):
        validate_validation_result_registry_v1(missing)

    with pytest.raises(ValidationRegistryError, match="duplicate validation cell"):
        replace(registry, cells=tuple(sorted((*cells, cells[0]), key=_sort_key)))

    target = next(
        index
        for index, cell in enumerate(cells)
        if cell.qualification_family == VALIDATION_SCHEDULER_FAMILY_V1
        and cell.condition_id == "C1"
    )
    changed = list(cells)
    changed[target] = replace(changed[target], channel_outcome_trace_sha256="a" * 64)
    with pytest.raises(ValidationRegistryError, match="trace not reused"):
        validate_validation_result_registry_v1(replace(registry, cells=tuple(changed)))

    c9_target = next(
        index
        for index, cell in enumerate(cells)
        if cell.qualification_family == VALIDATION_SCHEDULER_FAMILY_V1
        and cell.condition_id == "C9"
    )
    changed = list(cells)
    changed[c9_target] = replace(
        changed[c9_target], channel_outcome_trace_sha256="b" * 64
    )
    with pytest.raises(ValidationRegistryError, match="trace not reused"):
        validate_validation_result_registry_v1(replace(registry, cells=tuple(changed)))


def test_c9_setup_trace_header_and_cell_ids_fail_closed(
    registry: ValidationResultRegistryV1,
    cells: tuple[ValidationResultCellV1, ...],
) -> None:
    with pytest.raises(ValidationRegistryError, match="exactly 10"):
        replace(registry, c9_setup_trace_ids=SETUP_TRACE_IDS[:-1])
    with pytest.raises(ValidationRegistryError, match="exactly 10"):
        replace(
            registry,
            c9_setup_trace_ids=(*SETUP_TRACE_IDS, "setup-trace-10"),
        )
    with pytest.raises(ValidationRegistryError, match="unique sorted"):
        replace(registry, c9_setup_trace_ids=tuple(reversed(SETUP_TRACE_IDS)))

    target = next(
        index for index, cell in enumerate(cells) if cell.condition_id == "C9"
    )
    changed = list(cells)
    changed[target] = replace(changed[target], c9_trace_id="setup-trace-99")
    wrong_trace_registry = replace(
        registry, cells=tuple(sorted(changed, key=_sort_key))
    )
    with pytest.raises(ValidationRegistryError, match="missing validation cells"):
        validate_validation_result_registry_v1(wrong_trace_registry)


def _external_validation_sample(
    registry: ValidationResultRegistryV1,
    plan: ExperimentPlanV1,
    receipt: MeasuredTraceReceiptV1,
) -> tuple[
    ValidationResultRegistryV1,
    dict[str, ConditionInputManifestV1],
]:
    selectors = (
        lambda cell: (
            cell.qualification_family == VALIDATION_BASELINE_FAMILY_V1
            and cell.condition_id == "C6"
            and cell.training_seed == 1337
            and cell.network_seed == 1001
        ),
        lambda cell: (
            cell.qualification_family == VALIDATION_BASELINE_FAMILY_V1
            and cell.condition_id == "C6"
            and cell.training_seed == 2027
            and cell.network_seed == 1001
        ),
        lambda cell: (
            cell.qualification_family == VALIDATION_SCHEDULER_FAMILY_V1
            and cell.condition_id == "C7"
            and cell.training_seed == 1337
            and cell.network_seed == 1001
            and cell.budget_bytes_per_second == 16_000
        ),
        lambda cell: (
            cell.qualification_family == VALIDATION_SCHEDULER_FAMILY_V1
            and cell.condition_id == "C8"
            and cell.training_seed == 1337
            and cell.network_seed == 1001
            and cell.budget_bytes_per_second == 16_000
        ),
        lambda cell: (
            cell.qualification_family == VALIDATION_BASELINE_FAMILY_V1
            and cell.condition_id == "C9"
            and cell.training_seed == 1337
            and cell.c9_trace_id == SETUP_TRACE_IDS[0]
        ),
    )
    selected = [
        next(cell for cell in registry.cells if selector(cell))
        for selector in selectors
    ]
    setup_packets = {
        segment.trace_id: segment.packet_metadata_sha256
        for segment in receipt.segments
        if segment.role is TraceRole.SETUP
    }
    manifests: dict[str, ConditionInputManifestV1] = {}
    bound_cells = []
    for cell in selected:
        if cell.condition_id == "C9":
            assert cell.c9_trace_id is not None
            bound_cells.append(
                replace(
                    cell,
                    channel_outcome_trace_sha256=setup_packets[cell.c9_trace_id],
                )
            )
            continue
        config_sha256 = plan.run_config_sha256(
            "primary",
            (
                cell.method_id
                if cell.qualification_family == VALIDATION_BASELINE_FAMILY_V1
                else "eventtrack-v2x"
            ),
            (
                plan.strongest_scheduler_baseline_id
                if cell.qualification_family == VALIDATION_BASELINE_FAMILY_V1
                else cell.scheduler_id
            ),
        )
        manifest = ConditionInputManifestV1(
            run_id=validation_physical_run_id_v1(
                cell,
                run_config_sha256=config_sha256,
                detection_cache_sha256=registry.detection_cache_sha256,
            ),
            condition_id=NetworkConditionId(cell.condition_id),
            network_trace_sha256=cell.network_trace_sha256,
            condition_plan_sha256=condition_plan_v1(
                NetworkConditionId(cell.condition_id)
            ).content_sha256,
            run_config_sha256=config_sha256,
            detection_cache_sha256=registry.detection_cache_sha256,
            evidence_sha256s=(_digest((cell.key, "condition-evidence")),),
        )
        manifests[manifest.content_sha256] = manifest
        bound_cells.append(
            replace(
                cell,
                condition_input_manifest_sha256=manifest.content_sha256,
            )
        )
    return (
        replace(
            registry,
            measured_trace_receipt_sha256=receipt.content_sha256,
            cells=tuple(sorted(bound_cells, key=_sort_key)),
        ),
        manifests,
    )


def test_external_validation_binds_setup_receipt_configs_and_manifests(
    registry: ValidationResultRegistryV1,
    monkeypatch,
) -> None:
    receipt = _measured_trace_receipt()
    qualification = _qualification(registry)
    plan = replace(
        _plan(registry, qualification),
        heldout_c9_trace_receipt_sha256=receipt.content_sha256,
        c9_trace_ids=receipt.held_out_trace_ids,
    )
    bound_registry, manifests = _external_validation_sample(registry, plan, receipt)
    monkeypatch.setattr(
        validation_registry_module,
        "validate_validation_result_registry_v1",
        lambda value: None,
    )

    validate_validation_result_external_inputs_v1(
        bound_registry,
        plan,
        measured_trace_receipt=receipt,
        condition_input_manifests=manifests,
    )
    baseline = next(
        cell
        for cell in bound_registry.cells
        if cell.qualification_family == VALIDATION_BASELINE_FAMILY_V1
        and cell.condition_id == "C6"
    )
    scheduler = next(
        cell
        for cell in bound_registry.cells
        if cell.qualification_family == VALIDATION_SCHEDULER_FAMILY_V1
    )
    assert manifests[baseline.condition_input_manifest_sha256].run_config_sha256 == (
        plan.run_config_sha256(
            "primary",
            baseline.method_id,
            plan.strongest_scheduler_baseline_id,
        )
    )
    assert manifests[scheduler.condition_input_manifest_sha256].run_config_sha256 == (
        plan.run_config_sha256("primary", "eventtrack-v2x", scheduler.scheduler_id)
    )


def test_external_validation_rejects_c9_and_manifest_aliases(
    registry: ValidationResultRegistryV1,
    monkeypatch,
) -> None:
    receipt = _measured_trace_receipt()
    plan = replace(
        _plan(registry, _qualification(registry)),
        heldout_c9_trace_receipt_sha256=receipt.content_sha256,
        c9_trace_ids=receipt.held_out_trace_ids,
    )
    bound_registry, manifests = _external_validation_sample(registry, plan, receipt)
    monkeypatch.setattr(
        validation_registry_module,
        "validate_validation_result_registry_v1",
        lambda value: None,
    )

    c9_index = next(
        index
        for index, cell in enumerate(bound_registry.cells)
        if cell.condition_id == "C9"
    )
    changed = list(bound_registry.cells)
    changed[c9_index] = replace(
        changed[c9_index], channel_outcome_trace_sha256="f" * 64
    )
    with pytest.raises(ValidationRegistryError, match="setup packet metadata"):
        validate_validation_result_external_inputs_v1(
            replace(bound_registry, cells=tuple(changed)),
            plan,
            measured_trace_receipt=receipt,
            condition_input_manifests=manifests,
        )

    with pytest.raises(ValidationRegistryError, match="receipt content or setup"):
        validate_validation_result_external_inputs_v1(
            replace(
                bound_registry,
                measured_trace_receipt_sha256="f" * 64,
            ),
            plan,
            measured_trace_receipt=receipt,
            condition_input_manifests=manifests,
        )

    c6_indices = [
        index
        for index, cell in enumerate(bound_registry.cells)
        if cell.condition_id == "C6"
    ]
    changed = list(bound_registry.cells)
    changed[c6_indices[1]] = replace(
        changed[c6_indices[1]],
        condition_input_manifest_sha256=(
            changed[c6_indices[0]].condition_input_manifest_sha256
        ),
    )
    with pytest.raises(ValidationRegistryError, match="aliased"):
        validate_validation_result_external_inputs_v1(
            replace(bound_registry, cells=tuple(changed)),
            plan,
            measured_trace_receipt=receipt,
            condition_input_manifests=manifests,
        )


def test_external_validation_rejects_wrong_run_and_inventory_set(
    registry: ValidationResultRegistryV1,
    monkeypatch,
) -> None:
    receipt = _measured_trace_receipt()
    plan = replace(
        _plan(registry, _qualification(registry)),
        heldout_c9_trace_receipt_sha256=receipt.content_sha256,
        c9_trace_ids=receipt.held_out_trace_ids,
    )
    bound_registry, manifests = _external_validation_sample(registry, plan, receipt)
    monkeypatch.setattr(
        validation_registry_module,
        "validate_validation_result_registry_v1",
        lambda value: None,
    )
    target_index = next(
        index
        for index, cell in enumerate(bound_registry.cells)
        if cell.condition_id == "C7"
    )
    target = bound_registry.cells[target_index]
    assert target.condition_input_manifest_sha256 is not None
    original = manifests[target.condition_input_manifest_sha256]
    wrong = replace(original, run_id="validation-wrong-run")
    wrong_inventory = dict(manifests)
    del wrong_inventory[original.content_sha256]
    wrong_inventory[wrong.content_sha256] = wrong
    changed = list(bound_registry.cells)
    changed[target_index] = replace(
        target, condition_input_manifest_sha256=wrong.content_sha256
    )
    with pytest.raises(ValidationRegistryError, match="physical run"):
        validate_validation_result_external_inputs_v1(
            replace(bound_registry, cells=tuple(changed)),
            plan,
            measured_trace_receipt=receipt,
            condition_input_manifests=wrong_inventory,
        )

    wrong_config = replace(original, run_config_sha256="0" * 64)
    wrong_config_inventory = dict(manifests)
    del wrong_config_inventory[original.content_sha256]
    wrong_config_inventory[wrong_config.content_sha256] = wrong_config
    changed = list(bound_registry.cells)
    changed[target_index] = replace(
        target,
        condition_input_manifest_sha256=wrong_config.content_sha256,
    )
    with pytest.raises(ValidationRegistryError, match="physical run"):
        validate_validation_result_external_inputs_v1(
            replace(bound_registry, cells=tuple(changed)),
            plan,
            measured_trace_receipt=receipt,
            condition_input_manifests=wrong_config_inventory,
        )

    missing = dict(manifests)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValidationRegistryError, match="missing condition"):
        validate_validation_result_external_inputs_v1(
            bound_registry,
            plan,
            measured_trace_receipt=receipt,
            condition_input_manifests=missing,
        )

    extra = replace(original, run_id="validation-unreferenced-run")
    with pytest.raises(ValidationRegistryError, match="missing or extra"):
        validate_validation_result_external_inputs_v1(
            bound_registry,
            plan,
            measured_trace_receipt=receipt,
            condition_input_manifests={
                **manifests,
                extra.content_sha256: extra,
            },
        )


def test_budget_and_schema_constraints_fail_closed(
    cells: tuple[ValidationResultCellV1, ...],
) -> None:
    with pytest.raises(ValidationRegistryError, match="exceeds cell budget"):
        replace(
            cells[0],
            on_wire_bytes_total=64_001,
            on_wire_bytes_per_second=64_001.0,
        )
    with pytest.raises(ValidationRegistryError, match="C0 cells"):
        replace(cells[0], network_seed=1001)
    c9 = next(cell for cell in cells if cell.condition_id == "C9")
    with pytest.raises(ValidationRegistryError, match="C9 cells"):
        replace(c9, network_seed=1001)
    c1 = next(cell for cell in cells if cell.condition_id == "C1")
    with pytest.raises(ValidationRegistryError, match="C1-C8 cells"):
        replace(c1, c9_trace_id=SETUP_TRACE_IDS[0])
    c6 = next(cell for cell in cells if cell.condition_id == "C6")
    assert ValidationResultCellV1.from_mapping(c6.to_primitive()) == c6
    with pytest.raises(ValidationRegistryError, match="C6-C8 cells require"):
        replace(c6, condition_input_manifest_sha256=None)
    with pytest.raises(ValidationRegistryError, match="must not carry"):
        replace(c1, condition_input_manifest_sha256="a" * 64)
    with pytest.raises(ValidationRegistryError, match="must not carry"):
        replace(c9, condition_input_manifest_sha256="a" * 64)
    with pytest.raises(ValidationRegistryError, match="metrics must be in"):
        replace(cells[0], metric_values=(("AMOTA", 1.1), ("HOTA", 0.5)))


def test_canonical_round_trip_hash_and_tampering_fail_closed(
    registry: ValidationResultRegistryV1,
) -> None:
    encoded = registry.canonical_bytes()
    decoded = decode_validation_result_registry(encoded)
    assert decoded.content_sha256 == registry.content_sha256
    assert decoded.digest() == registry.digest()

    document = registry.sealed_document()
    document["source_tree_sha256"] = "f" * 64
    with pytest.raises(ValidationRegistryError, match="content SHA-256 mismatch"):
        ValidationResultRegistryV1.from_mapping(document)

    duplicate = b'{"kind":"validation_result_registry_v1","kind":"x"}'
    with pytest.raises(ValidationRegistryError, match="duplicate JSON key"):
        decode_validation_result_registry(duplicate)

    noncanonical = json.dumps(
        registry.sealed_document(), separators=(", ", ": ")
    ).encode()
    with pytest.raises(ValidationRegistryError, match="not canonical"):
        decode_validation_result_registry(noncanonical)

    incomplete = replace(registry, cells=registry.cells[:-1])
    with pytest.raises(ValidationRegistryError, match="missing validation cells"):
        decode_validation_result_registry(incomplete.canonical_bytes())
