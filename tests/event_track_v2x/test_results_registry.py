from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from transvision.models.event_track_v2x import results_registry as results_module
from transvision.models.event_track_v2x.experiment import (
    CONFIRMATORY_METHOD_IDS_V1,
    CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1,
    GRIFFIN_25M_VAL_SEQUENCE_IDS_V1,
    ExperimentPlanError,
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
from transvision.models.event_track_v2x.network import (
    NetworkConditionId,
    condition_plan_v1,
)
from transvision.models.event_track_v2x.network_disturbance import (
    ConditionInputManifestV1,
)
from transvision.models.event_track_v2x.results_registry import (
    PRIMARY_CLEAN_SCENARIO_V1,
    PRIMARY_PARETO_SCENARIO_V1,
    PRIMARY_ROBUST_SCENARIO_V1,
    RESULT_METRICS_ARTIFACT_ROLE_V1,
    RUN_RESULT_STATUS_FAILURE_V1,
    RUN_RESULT_STATUS_SUCCESS_V1,
    RunResultCellV1,
    RunResultRegistryError,
    RunResultRegistryV1,
    decode_run_result_registry,
    derive_publication_metrics_v1,
    iter_expected_run_specs_v1,
    validate_run_result_external_inputs_v1,
)


PRIMARY_SEQUENCES = tuple(f"sequence-{index:02d}" for index in range(21))
GRIFFIN_SEQUENCES = GRIFFIN_25M_VAL_SEQUENCE_IDS_V1
C9_TRACE_IDS = tuple(f"heldout-trace-{index:02d}" for index in range(20))


def _digest(value: object) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _measured_receipt() -> MeasuredTraceReceiptV1:
    segments = []
    for index in range(30):
        held_out = index >= 10
        trace_index = index - 10 if held_out else index
        segments.append(
            MeasuredTraceSegmentV1(
                trace_id=(
                    f"heldout-trace-{trace_index:02d}"
                    if held_out
                    else f"setup-trace-{trace_index:02d}"
                ),
                role=TraceRole.HELD_OUT if held_out else TraceRole.SETUP,
                date_cohort_id=f"date-{index % 3}",
                road_coverage_type=f"coverage-{index % 3}",
                duration_seconds=600,
                packet_count=100 + index,
                packet_metadata_sha256=_digest(("packet-metadata", index)),
                packet_fields=REQUIRED_PACKET_FIELDS_V1,
                clock_reference=ClockReference.PTP,
            )
        )
    return MeasuredTraceReceiptV1(
        receipt_id="measured-c9-receipt-v1",
        segments=tuple(segments),
        collection_protocol_sha256="b" * 64,
    )


def _sort_key(cell: RunResultCellV1) -> tuple[str, ...]:
    return tuple(
        "" if value is None else f"{type(value).__name__}:{value}"
        for value in cell.key
    )


@pytest.fixture(scope="module")
def plan() -> ExperimentPlanV1:
    run_configs = tuple(
        RunConfigBindingV1(
            dataset_domain=domain,
            method_id=method,
            scheduler_id=scheduler,
            config_sha256=_digest((domain, method, scheduler)),
            checkpoint_sha256s=tuple(
                _digest((domain, method, scheduler, seed, "final-refit"))
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
        dataset_manifest_sha256="3" * 64,
        split_name="val",
        split_sha256="4" * 64,
        primary_cohort_sha256="d" * 64,
        primary_frame_contract_sha256="e" * 64,
        primary_dataset_release_receipt_sha256="f" * 64,
        development_dataset_release_receipt_sha256="a" * 64,
        development_fold_manifest_sha256="b" * 64,
        primary_sequence_ids=PRIMARY_SEQUENCES,
        detector_cache_sha256="5" * 64,
        network_trace_manifest_sha256="6" * 64,
        wire_accounting_config_sha256="0" * 64,
        heldout_c9_trace_receipt_sha256="7" * 64,
        c9_trace_ids=C9_TRACE_IDS,
        evaluator_contract_sha256="8" * 64,
        baseline_qualification_registry_sha256="1" * 64,
        candidate_selection_registry_sha256="9" * 64,
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
        griffin_dataset_manifest_sha256="9" * 64,
        griffin_split_name="val",
        griffin_cohort_sha256="a" * 64,
        griffin_frame_contract_sha256="3" * 64,
        griffin_dataset_release_receipt_sha256="4" * 64,
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
        byte_budgets_per_second=(16000, 32000, 64000, 128000, 256000),
        primary_budget_bytes_per_second=64000,
        primary_metric="robust-assa-at-64k",
        decision_deadline_ms=100.0,
        statistical_alpha=0.05,
        statistical_resamples=10000,
        statistical_random_seed=1337,
        claim_scope="same-detector robust cooperative association",
    )


def _metrics_for_spec(spec) -> tuple[tuple[str, float], ...]:
    if spec.scenario_id == PRIMARY_CLEAN_SCENARIO_V1:
        if spec.method_id == "eventtrack-v2x":
            return (("AMOTA", 0.70), ("HOTA", 0.69))
        return (("AMOTA", 0.705), ("HOTA", 0.695))
    if spec.scenario_id == PRIMARY_PARETO_SCENARIO_V1:
        value = 0.70 if spec.scheduler_id == "marginal_voi" else 0.65
        return (("AssA", value),)
    value = 0.70 if spec.method_id == "eventtrack-v2x" else 0.65
    return (("AssA", value),)


@pytest.fixture(scope="module")
def cells(plan: ExperimentPlanV1) -> tuple[RunResultCellV1, ...]:
    result = []
    for spec in iter_expected_run_specs_v1(plan, C9_TRACE_IDS):
        replicate = (
            spec.c9_trace_id
            if spec.c9_trace_id is not None
            else "clean"
            if spec.network_seed is None
            else f"seed:{spec.network_seed}"
        )
        trace_key = (
            spec.dataset_domain,
            spec.sequence_id,
            spec.condition_id,
            replicate,
        )
        result.append(
            RunResultCellV1(
                experiment_plan_sha256=plan.content_sha256,
                attempt_id=f"attempt-{_digest(spec.key[1:])[:24]}",
                detection_cache_sha256=(
                    plan.detector_cache_sha256
                    if spec.dataset_domain == "primary"
                    else plan.griffin_detector_cache_sha256
                ),
                evaluator_contract_sha256=(
                    plan.evaluator_contract_sha256
                    if spec.dataset_domain == "primary"
                    else plan.griffin_evaluator_contract_sha256
                ),
                scenario_id=spec.scenario_id,
                dataset_domain=spec.dataset_domain,
                method_id=spec.method_id,
                sequence_id=spec.sequence_id,
                condition_id=spec.condition_id,
                training_seed=spec.training_seed,
                network_seed=spec.network_seed,
                c9_trace_id=spec.c9_trace_id,
                budget_bytes_per_second=spec.budget_bytes_per_second,
                scheduler_id=spec.scheduler_id,
                metric_values=_metrics_for_spec(spec),
                on_wire_bytes_total=spec.budget_bytes_per_second - 1,
                wire_measurement_duration_seconds=1.0,
                on_wire_bytes_per_second=spec.budget_bytes_per_second - 1.0,
                status=RUN_RESULT_STATUS_SUCCESS_V1,
                failure_code=None,
                checkpoint_sha256=plan.run_checkpoint_sha256(
                    spec.dataset_domain,
                    spec.method_id,
                    spec.scheduler_id,
                    spec.training_seed,
                ),
                evidence_bundle_sha256="d" * 64,
                metrics_artifact_role=RESULT_METRICS_ARTIFACT_ROLE_V1,
                metrics_artifact_sha256="e" * 64,
                channel_outcome_trace_sha256=_digest(trace_key),
                network_trace_sha256=_digest(
                    (
                        trace_key,
                        spec.method_id,
                        spec.scheduler_id,
                        spec.budget_bytes_per_second,
                    )
                ),
                condition_input_manifest_sha256=(
                    _digest(("condition-input", spec.key[1:]))
                    if spec.condition_id in {"C6", "C7", "C8"}
                    else None
                ),
                wire_ledger_sha256="f" * 64,
            )
        )
    return tuple(sorted(result, key=_sort_key))


@pytest.fixture(scope="module")
def registry(
    plan: ExperimentPlanV1,
    cells: tuple[RunResultCellV1, ...],
) -> RunResultRegistryV1:
    return RunResultRegistryV1(
        experiment_plan_sha256=plan.content_sha256,
        attempt_manifest_sha256=_digest("attempt-manifest"),
        confirmatory_val_unsealed_at_utc="2026-09-02T08:01:00Z",
        execution_started_at_utc="2026-09-02T08:02:00Z",
        first_prediction_at_utc="2026-09-02T08:03:00Z",
        completed_at_utc="2026-09-02T10:00:00Z",
        network_trace_manifest_sha256=plan.network_trace_manifest_sha256,
        wire_accounting_config_sha256=plan.wire_accounting_config_sha256,
        heldout_c9_trace_receipt_sha256=plan.heldout_c9_trace_receipt_sha256,
        c9_trace_ids=C9_TRACE_IDS,
        cells=cells,
    )


def test_complete_registry_uniquely_derives_publication_metrics(
    plan: ExperimentPlanV1,
    registry: RunResultRegistryV1,
) -> None:
    assert len(registry.cells) == 113_226
    assert len({cell.physical_run_key for cell in registry.cells}) == 106_926
    metrics = derive_publication_metrics_v1(registry, plan)
    assert metrics.experiment_plan_sha256 == plan.content_sha256
    assert metrics.robust_eventtrack.as_mapping()["sequence-00"] == pytest.approx(
        0.70
    )
    assert tuple(item.method_id for item in metrics.robust_baselines) == (
        CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1
    )
    assert metrics.clean_eventtrack[0].scores.as_mapping()[
        "sequence-00"
    ] == pytest.approx(0.70)
    assert metrics.pareto_eventtrack.byte_budgets_per_second == (
        16000,
        32000,
        64000,
        128000,
        256000,
    )
    assert metrics.griffin_eventtrack.as_mapping()[
        "scene-0000-Town03-000"
    ] == pytest.approx(0.70)


def test_algorithm_failure_is_present_and_scores_zero_after_seed_then_condition_macro(
    plan: ExperimentPlanV1,
    cells: tuple[RunResultCellV1, ...],
) -> None:
    selected = next(
        index
        for index, cell in enumerate(cells)
        if cell.scenario_id == PRIMARY_ROBUST_SCENARIO_V1
        and cell.method_id == "eventtrack-v2x"
        and cell.sequence_id == "sequence-00"
        and cell.condition_id == "C1"
    )
    changed = list(cells)
    selected_key = changed[selected].key[1:]
    for index, cell in enumerate(changed):
        if (
            cell.scenario_id
            in {PRIMARY_ROBUST_SCENARIO_V1, PRIMARY_PARETO_SCENARIO_V1}
            and cell.key[1:] == selected_key
        ):
            changed[index] = replace(
                cell,
                metric_values=(),
                status=RUN_RESULT_STATUS_FAILURE_V1,
                failure_code="algorithm-crash",
            )
    registry = registry_from_cells(plan, tuple(changed))
    metrics = derive_publication_metrics_v1(registry, plan)
    assert metrics.robust_eventtrack.as_mapping()["sequence-00"] == pytest.approx(
        0.70 * 269.0 / 270.0
    )


def test_missing_cell_wrong_trace_and_inconsistent_trace_hash_fail_closed(
    plan: ExperimentPlanV1,
    cells: tuple[RunResultCellV1, ...],
) -> None:
    missing = registry_from_cells(plan, cells[:-1])
    with pytest.raises(RunResultRegistryError, match="missing result cells"):
        derive_publication_metrics_v1(missing, plan)

    c9_index = next(
        index for index, cell in enumerate(cells) if cell.condition_id == "C9"
    )
    wrong_trace_cells = list(cells)
    wrong_trace_cells[c9_index] = replace(
        wrong_trace_cells[c9_index], c9_trace_id="heldout-trace-unknown"
    )
    wrong_trace_cells.sort(key=_sort_key)
    wrong_trace = replace(registry_from_cells(plan, tuple(wrong_trace_cells)))
    with pytest.raises(RunResultRegistryError, match="unregistered C9"):
        derive_publication_metrics_v1(wrong_trace, plan)

    repeated_trace_index = next(
        index
        for index, cell in enumerate(cells)
        if cell.scenario_id == PRIMARY_PARETO_SCENARIO_V1
    )
    inconsistent_cells = list(cells)
    inconsistent_cells[repeated_trace_index] = replace(
        inconsistent_cells[repeated_trace_index],
        channel_outcome_trace_sha256="f" * 64,
    )
    inconsistent = registry_from_cells(plan, tuple(inconsistent_cells))
    with pytest.raises(RunResultRegistryError, match="exogenous channel trace"):
        derive_publication_metrics_v1(inconsistent, plan)


def registry_from_cells(
    plan: ExperimentPlanV1,
    cells: tuple[RunResultCellV1, ...],
) -> RunResultRegistryV1:
    return RunResultRegistryV1(
        experiment_plan_sha256=plan.content_sha256,
        attempt_manifest_sha256=_digest("attempt-manifest"),
        confirmatory_val_unsealed_at_utc="2026-09-02T08:01:00Z",
        execution_started_at_utc="2026-09-02T08:02:00Z",
        first_prediction_at_utc="2026-09-02T08:03:00Z",
        completed_at_utc="2026-09-02T10:00:00Z",
        network_trace_manifest_sha256=plan.network_trace_manifest_sha256,
        wire_accounting_config_sha256=plan.wire_accounting_config_sha256,
        heldout_c9_trace_receipt_sha256=plan.heldout_c9_trace_receipt_sha256,
        c9_trace_ids=C9_TRACE_IDS,
        cells=cells,
    )


def test_over_budget_duplicate_and_noncanonical_registry_are_rejected(
    plan: ExperimentPlanV1,
    cells: tuple[RunResultCellV1, ...],
) -> None:
    with pytest.raises(RunResultRegistryError, match="exceeds"):
        replace(
            cells[0],
            on_wire_bytes_total=cells[0].budget_bytes_per_second + 1,
            on_wire_bytes_per_second=cells[0].budget_bytes_per_second + 1.0,
        )
    with pytest.raises(RunResultRegistryError, match="does not equal ledger"):
        replace(cells[0], on_wire_bytes_per_second=1.0)
    c6 = next(cell for cell in cells if cell.condition_id == "C6")
    assert RunResultCellV1.from_mapping(c6.to_primitive()) == c6
    with pytest.raises(RunResultRegistryError, match="C6-C8 cells require"):
        replace(c6, condition_input_manifest_sha256=None)
    c5 = next(cell for cell in cells if cell.condition_id == "C5")
    with pytest.raises(RunResultRegistryError, match="must not carry"):
        replace(c5, condition_input_manifest_sha256="a" * 64)
    c9 = next(cell for cell in cells if cell.condition_id == "C9")
    with pytest.raises(RunResultRegistryError, match="must not carry"):
        replace(c9, condition_input_manifest_sha256="a" * 64)
    with pytest.raises(RunResultRegistryError, match="duplicate"):
        registry_from_cells(plan, (cells[0], cells[0]))

    tiny = registry_from_cells(plan, (cells[0],))
    assert decode_run_result_registry(tiny.canonical_bytes()) == tiny
    noncanonical = json.dumps(tiny.sealed_document()).encode()
    with pytest.raises(RunResultRegistryError, match="not canonical"):
        decode_run_result_registry(noncanonical)


def test_registry_requires_exact_sorted_twenty_c9_traces(
    plan: ExperimentPlanV1,
    cells: tuple[RunResultCellV1, ...],
) -> None:
    with pytest.raises(RunResultRegistryError, match="exactly 20"):
        RunResultRegistryV1(
            experiment_plan_sha256=plan.content_sha256,
            attempt_manifest_sha256=_digest("attempt-manifest"),
            confirmatory_val_unsealed_at_utc="2026-09-02T08:01:00Z",
            execution_started_at_utc="2026-09-02T08:02:00Z",
            first_prediction_at_utc="2026-09-02T08:03:00Z",
            completed_at_utc="2026-09-02T10:00:00Z",
            network_trace_manifest_sha256=plan.network_trace_manifest_sha256,
            wire_accounting_config_sha256=plan.wire_accounting_config_sha256,
            heldout_c9_trace_receipt_sha256=plan.heldout_c9_trace_receipt_sha256,
            c9_trace_ids=C9_TRACE_IDS[:-1],
            cells=(cells[0],),
        )


def test_external_validator_opens_c6_manifest_and_c9_receipt(
    monkeypatch,
    plan: ExperimentPlanV1,
    cells: tuple[RunResultCellV1, ...],
) -> None:
    receipt = _measured_receipt()
    bound_plan = replace(
        plan,
        heldout_c9_trace_receipt_sha256=receipt.content_sha256,
    )
    c6 = next(cell for cell in cells if cell.condition_id == "C6")
    c9 = next(cell for cell in cells if cell.condition_id == "C9")
    condition = NetworkConditionId.C6
    manifest = ConditionInputManifestV1(
        run_id=c6.attempt_id,
        condition_id=condition,
        network_trace_sha256=c6.network_trace_sha256,
        condition_plan_sha256=condition_plan_v1(condition).content_sha256,
        run_config_sha256=bound_plan.run_config_sha256(
            c6.dataset_domain,
            c6.method_id,
            c6.scheduler_id,
        ),
        detection_cache_sha256=c6.detection_cache_sha256,
        evidence_sha256s=(_digest(("condition-evidence", c6.attempt_id)),),
    )
    held_out_packets = dict(receipt.held_out_packet_metadata_sha256s)
    bound_cells = (
        replace(c6, condition_input_manifest_sha256=manifest.content_sha256),
        replace(
            c9,
            channel_outcome_trace_sha256=held_out_packets[c9.c9_trace_id],
        ),
    )
    registry_like = SimpleNamespace(cells=bound_cells)
    monkeypatch.setattr(
        results_module,
        "validate_run_result_registry_v1",
        lambda value, value_plan: None,
    )

    validate_run_result_external_inputs_v1(
        registry_like,  # type: ignore[arg-type]
        bound_plan,
        measured_trace_receipt=receipt,
        condition_input_manifests={manifest.content_sha256: manifest},
    )

    forged = SimpleNamespace(
        cells=(bound_cells[0], replace(bound_cells[1], channel_outcome_trace_sha256="0" * 64))
    )
    with pytest.raises(RunResultRegistryError, match="held-out packet metadata"):
        validate_run_result_external_inputs_v1(
            forged,  # type: ignore[arg-type]
            bound_plan,
            measured_trace_receipt=receipt,
            condition_input_manifests={manifest.content_sha256: manifest},
        )


def test_duplicate_eventtrack_64k_cells_must_match_outside_scenario_id(
    plan: ExperimentPlanV1,
    cells: tuple[RunResultCellV1, ...],
) -> None:
    selected = next(
        index
        for index, cell in enumerate(cells)
        if cell.scenario_id == PRIMARY_PARETO_SCENARIO_V1
        and cell.method_id == "eventtrack-v2x"
        and cell.scheduler_id == plan.candidate_scheduler_id
        and cell.budget_bytes_per_second
        == plan.primary_budget_bytes_per_second
    )
    changed = list(cells)
    changed[selected] = replace(changed[selected], metric_values=(("AssA", 0.69),))

    with pytest.raises(RunResultRegistryError, match="duplicate cells differ"):
        derive_publication_metrics_v1(
            registry_from_cells(plan, tuple(changed)), plan
        )


@pytest.mark.parametrize(
    ("dataset_domain", "field_name", "error"),
    (
        ("primary", "detection_cache_sha256", "detection cache"),
        ("primary", "evaluator_contract_sha256", "evaluator contract"),
        ("griffin", "detection_cache_sha256", "detection cache"),
        ("griffin", "evaluator_contract_sha256", "evaluator contract"),
    ),
)
def test_each_cell_binds_domain_specific_detection_and_evaluator_contracts(
    plan: ExperimentPlanV1,
    cells: tuple[RunResultCellV1, ...],
    dataset_domain: str,
    field_name: str,
    error: str,
) -> None:
    selected = next(
        index
        for index, cell in enumerate(cells)
        if cell.dataset_domain == dataset_domain
    )
    changed = list(cells)
    changed[selected] = replace(changed[selected], **{field_name: "0" * 64})

    with pytest.raises(RunResultRegistryError, match=error):
        derive_publication_metrics_v1(
            registry_from_cells(plan, tuple(changed)), plan
        )


def test_griffin_cohort_and_baseline_are_exactly_frozen(
    plan: ExperimentPlanV1,
) -> None:
    with pytest.raises(ExperimentPlanError, match="exact official Griffin-25m val"):
        replace(
            plan,
            griffin_sequence_ids=GRIFFIN_25M_VAL_SEQUENCE_IDS_V1[:-1],
        )

    with pytest.raises(ExperimentPlanError, match="reuse the strongest baseline"):
        replace(
            plan,
            griffin_strongest_baseline_id="vehicle-only-simpletrack",
        )


def test_registry_seals_confirmatory_attempt_chronology(
    registry: RunResultRegistryV1,
) -> None:
    assert decode_run_result_registry(registry.canonical_bytes()) == registry
    with pytest.raises(RunResultRegistryError, match="chronology"):
        replace(
            registry,
            execution_started_at_utc="2026-09-02T08:04:00Z",
            first_prediction_at_utc="2026-09-02T08:03:00Z",
        )
    with pytest.raises(RunResultRegistryError, match="canonical RFC3339"):
        replace(registry, completed_at_utc="2026-09-02 10:00:00Z")
