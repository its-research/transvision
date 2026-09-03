from dataclasses import replace
from collections import Counter
import hashlib
import json

import pytest

from transvision.models.event_track_v2x import supplemental_registry as supplemental_module
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
from transvision.models.event_track_v2x.network import (
    NetworkConditionId,
    condition_plan_v1,
)
from transvision.models.event_track_v2x.network_disturbance import (
    ConditionInputManifestV1,
)
from transvision.models.event_track_v2x.supplemental_registry import (
    FORMAL_SUPPLEMENTAL_CELL_COUNT_V1,
    FULL_SEND_DIAGNOSTIC_SCENARIO_V1,
    RAW_SENSOR_METHOD_IDS_V1,
    RAW_SENSOR_SECONDARY_SCENARIO_V1,
    REQUIRED_SUPPLEMENTAL_METRIC_IDS_V1,
    SCHEDULER_FRONTIER_SCENARIO_V1,
    SUPPLEMENTAL_FAILURE_SEMANTICS_V1,
    SUPPLEMENTAL_METRICS_ARTIFACT_ROLE_V1,
    SUPPLEMENTAL_STATUS_FAILURE_V1,
    SUPPLEMENTAL_STATUS_SUCCESS_V1,
    SYNCHRONOUS_ORACLE_SCENARIO_V1,
    PaperSupplementalRegistryV1,
    SupplementalCohortV1,
    SupplementalConfigBindingV1,
    SupplementalRegistryError,
    SupplementalResultCellV1,
    _iter_expected_supplemental_specs_for_cohort,
    _validate_external_input_bindings_v1,
    _validate_registry_against_cohort,
    decode_paper_supplemental_registry,
    iter_expected_paper_supplemental_specs_v1,
    required_supplemental_config_keys_v1,
    supplemental_config_manifest_sha256_v1,
    validate_paper_supplemental_registry_v1,
)


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


def _bind_external_inputs(
    registry: PaperSupplementalRegistryV1,
    receipt: MeasuredTraceReceiptV1,
) -> tuple[
    PaperSupplementalRegistryV1,
    dict[str, ConditionInputManifestV1],
]:
    packets = dict(receipt.held_out_packet_metadata_sha256s)
    manifests: dict[str, ConditionInputManifestV1] = {}
    cells = []
    for cell in registry.cells:
        changes: dict[str, object] = {}
        if cell.condition_id == "C9":
            assert cell.c9_trace_id is not None
            changes["channel_outcome_trace_sha256"] = packets[cell.c9_trace_id]
        if cell.condition_id in {"C6", "C7", "C8"}:
            condition = NetworkConditionId(cell.condition_id)
            manifest = ConditionInputManifestV1(
                run_id=cell.attempt_id,
                condition_id=condition,
                network_trace_sha256=cell.network_trace_sha256,
                condition_plan_sha256=condition_plan_v1(condition).content_sha256,
                run_config_sha256=cell.config_sha256,
                detection_cache_sha256=cell.detection_cache_sha256,
                evidence_sha256s=(_digest(("input-evidence", cell.key)),),
            )
            manifests[manifest.content_sha256] = manifest
            changes["condition_input_manifest_sha256"] = manifest.content_sha256
        cells.append(replace(cell, **changes))
    return replace(registry, cells=tuple(cells)), manifests


@pytest.fixture(scope="module")
def cohort() -> SupplementalCohortV1:
    return SupplementalCohortV1(
        sequence_ids=("sequence-00",),
        training_seeds=(1337,),
        network_seeds=(1001,),
        c9_trace_ids=("heldout-trace-00",),
        byte_budgets_per_second=(16_000, 32_000),
    )


@pytest.fixture(scope="module")
def config_bindings() -> tuple[SupplementalConfigBindingV1, ...]:
    return tuple(
        SupplementalConfigBindingV1(*key, _digest(key))
        for key in required_supplemental_config_keys_v1()
    )


def _metrics() -> tuple[tuple[str, float], ...]:
    values = {metric_id: 0.5 for metric_id in REQUIRED_SUPPLEMENTAL_METRIC_IDS_V1}
    values["application_bps"] = 100.0
    values["on_wire_bps"] = 120.0
    return tuple(sorted(values.items()))


def _set_metric(
    metrics: tuple[tuple[str, float], ...], metric_id: str, value: float
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (observed_id, value if observed_id == metric_id else observed_value)
        for observed_id, observed_value in metrics
    )


@pytest.fixture(scope="module")
def cells(
    cohort: SupplementalCohortV1,
    config_bindings: tuple[SupplementalConfigBindingV1, ...],
) -> tuple[SupplementalResultCellV1, ...]:
    configs = {item.key: item.config_sha256 for item in config_bindings}
    result = []
    for index, spec in enumerate(
        _iter_expected_supplemental_specs_for_cohort(cohort)
    ):
        cache = "1" * 64 if spec.detector_input_id.startswith("camera") else "2" * 64
        result.append(
            SupplementalResultCellV1(
                experiment_plan_sha256="0" * 64,
                scenario_id=spec.scenario_id,
                detector_input_id=spec.detector_input_id,
                method_id=spec.method_id,
                scheduler_id=spec.scheduler_id,
                sequence_id=spec.sequence_id,
                condition_id=spec.condition_id,
                training_seed=spec.training_seed,
                network_seed=spec.network_seed,
                c9_trace_id=spec.c9_trace_id,
                nominal_budget_bytes_per_second=(
                    spec.nominal_budget_bytes_per_second
                ),
                table_ranking_eligible=spec.table_ranking_eligible,
                status=SUPPLEMENTAL_STATUS_SUCCESS_V1,
                failure_code=None,
                metric_values=_metrics(),
                application_bytes_total=1_000,
                on_wire_bytes_total=1_200,
                measurement_duration_seconds=10.0,
                attempt_id=f"attempt-{index:04d}",
                detection_cache_sha256=cache,
                config_sha256=configs[
                    (spec.detector_input_id, spec.method_id, spec.scheduler_id)
                ],
                prediction_archive_sha256=_digest(("prediction", index)),
                evidence_bundle_sha256=_digest(("evidence", index)),
                metrics_artifact_role=SUPPLEMENTAL_METRICS_ARTIFACT_ROLE_V1,
                metrics_artifact_sha256=_digest(("metrics", index)),
                evaluator_contract_sha256="3" * 64,
                channel_outcome_trace_sha256=_digest(("channel", index)),
                network_trace_sha256=_digest(("network", index)),
                condition_input_manifest_sha256=(
                    _digest(("condition-input", index))
                    if spec.condition_input_manifest_required
                    else None
                ),
                wire_ledger_sha256=_digest(("wire", index)),
            )
        )
    return tuple(result)


@pytest.fixture(scope="module")
def registry(
    cohort: SupplementalCohortV1,
    config_bindings: tuple[SupplementalConfigBindingV1, ...],
    cells: tuple[SupplementalResultCellV1, ...],
) -> PaperSupplementalRegistryV1:
    return PaperSupplementalRegistryV1(
        experiment_plan_sha256="0" * 64,
        dataset_manifest_sha256="4" * 64,
        split_sha256="5" * 64,
        cohort_sha256="6" * 64,
        frame_contract_sha256="7" * 64,
        primary_detector_cache_sha256="1" * 64,
        raw_sensor_detector_cache_sha256="2" * 64,
        evaluator_contract_sha256="3" * 64,
        network_trace_manifest_sha256="8" * 64,
        wire_accounting_config_sha256="9" * 64,
        heldout_c9_trace_receipt_sha256="a" * 64,
        sequence_ids=cohort.sequence_ids,
        c9_trace_ids=cohort.c9_trace_ids,
        config_bindings=config_bindings,
        config_manifest_sha256=supplemental_config_manifest_sha256_v1(
            config_bindings
        ),
        contributes_to_primary_sota_gate=False,
        failure_semantics=SUPPLEMENTAL_FAILURE_SEMANTICS_V1,
        cells=cells,
    )


def _formal_plan() -> ExperimentPlanV1:
    sequences = tuple(f"sequence-{index:02d}" for index in range(21))
    traces = tuple(f"heldout-trace-{index:02d}" for index in range(20))
    configs = tuple(
        RunConfigBindingV1(
            domain,
            method,
            scheduler,
            _digest((domain, method, scheduler)),
            tuple(
                _digest(("checkpoint", domain, method, scheduler, seed))
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
        run_config_bindings=configs,
        dataset_id="v2x-seq-spd",
        dataset_manifest_sha256="3" * 64,
        split_name="val",
        split_sha256="4" * 64,
        primary_cohort_sha256="d" * 64,
        primary_frame_contract_sha256="e" * 64,
        primary_dataset_release_receipt_sha256="f" * 64,
        development_dataset_release_receipt_sha256="a" * 64,
        development_fold_manifest_sha256="b" * 64,
        primary_sequence_ids=sequences,
        detector_cache_sha256="5" * 64,
        network_trace_manifest_sha256="6" * 64,
        wire_accounting_config_sha256="0" * 64,
        heldout_c9_trace_receipt_sha256="7" * 64,
        c9_trace_ids=traces,
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
        griffin_sequence_ids=GRIFFIN_25M_VAL_SEQUENCE_IDS_V1,
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


def test_small_cohort_exactly_covers_four_supplemental_matrices(
    cohort: SupplementalCohortV1,
) -> None:
    specs = tuple(_iter_expected_supplemental_specs_for_cohort(cohort))
    counts = {
        scenario: sum(item.scenario_id == scenario for item in specs)
        for scenario in (
            FULL_SEND_DIAGNOSTIC_SCENARIO_V1,
            RAW_SENSOR_SECONDARY_SCENARIO_V1,
            SCHEDULER_FRONTIER_SCENARIO_V1,
            SYNCHRONOUS_ORACLE_SCENARIO_V1,
        )
    }
    assert counts == {
        FULL_SEND_DIAGNOSTIC_SCENARIO_V1: 9,
        RAW_SENSOR_SECONDARY_SCENARIO_V1: 70,
        SCHEDULER_FRONTIER_SCENARIO_V1: 90,
        SYNCHRONOUS_ORACLE_SCENARIO_V1: 1,
    }
    full_send = [
        item for item in specs if item.scenario_id == FULL_SEND_DIAGNOSTIC_SCENARIO_V1
    ]
    oracle = [
        item for item in specs if item.scenario_id == SYNCHRONOUS_ORACLE_SCENARIO_V1
    ]
    raw = [
        item for item in specs if item.scenario_id == RAW_SENSOR_SECONDARY_SCENARIO_V1
    ]
    assert all(
        item.nominal_budget_bytes_per_second is None
        and not item.table_ranking_eligible
        for item in full_send + oracle
    )
    assert {item.method_id for item in raw} == set(RAW_SENSOR_METHOD_IDS_V1)
    assert {item.condition_id for item in raw} == {
        f"C{index}" for index in range(10)
    }


def test_formal_plan_stream_has_exact_preregistered_cell_count() -> None:
    pressure_replicates = 8 * 10 + 20
    independent_formula = (
        5 * 5 * 21 * 3 * pressure_replicates
        + 21 * 3 * pressure_replicates
        + 21 * 3
        + len(RAW_SENSOR_METHOD_IDS_V1) * 21 * 3 * (1 + pressure_replicates)
    )
    counts = Counter(
        spec.scenario_id
        for spec in iter_expected_paper_supplemental_specs_v1(_formal_plan())
    )
    assert counts == {
        SCHEDULER_FRONTIER_SCENARIO_V1: 157_500,
        FULL_SEND_DIAGNOSTIC_SCENARIO_V1: 6_300,
        SYNCHRONOUS_ORACLE_SCENARIO_V1: 63,
        RAW_SENSOR_SECONDARY_SCENARIO_V1: 44_541,
    }
    assert independent_formula == FORMAL_SUPPLEMENTAL_CELL_COUNT_V1 == 208_404


def test_registry_validates_exact_cells_and_round_trips_canonically(
    registry: PaperSupplementalRegistryV1,
    cohort: SupplementalCohortV1,
) -> None:
    _validate_registry_against_cohort(registry, cohort)
    decoded = decode_paper_supplemental_registry(registry.canonical_bytes())
    assert decoded == registry
    assert decoded.digest() == registry.digest()
    assert decoded.contributes_to_primary_sota_gate is False


def test_strict_decoder_rejects_duplicate_and_unknown_keys(
    registry: PaperSupplementalRegistryV1,
) -> None:
    duplicate = registry.canonical_bytes().replace(
        b'{"c9_trace_ids":', b'{"c9_trace_ids":[],"c9_trace_ids":', 1
    )
    with pytest.raises(SupplementalRegistryError, match="duplicate JSON key"):
        decode_paper_supplemental_registry(duplicate)

    document = registry.sealed_document()
    document["unknown"] = True
    with pytest.raises(SupplementalRegistryError, match="missing or unknown"):
        PaperSupplementalRegistryV1.from_mapping(document)


def test_exact_validator_rejects_missing_extra_and_duplicate_cells(
    registry: PaperSupplementalRegistryV1,
    cohort: SupplementalCohortV1,
) -> None:
    missing = replace(registry, cells=registry.cells[:-1])
    with pytest.raises(SupplementalRegistryError, match="missing expected cell"):
        _validate_registry_against_cohort(missing, cohort)

    extra_cell = replace(
        registry.cells[-1],
        sequence_id="sequence-extra",
        attempt_id="attempt-extra",
    )
    extra = replace(registry, cells=registry.cells + (extra_cell,))
    with pytest.raises(SupplementalRegistryError, match="unexpected extra cell"):
        _validate_registry_against_cohort(extra, cohort)

    with pytest.raises(SupplementalRegistryError, match="unique and sorted"):
        replace(registry, cells=registry.cells + (registry.cells[-1],))


def test_success_requires_complete_metrics_and_consistent_byte_ledgers(
    cells: tuple[SupplementalResultCellV1, ...],
) -> None:
    cell = cells[0]
    with pytest.raises(SupplementalRegistryError, match="every exact"):
        replace(cell, metric_values=cell.metric_values[:-1])
    with pytest.raises(SupplementalRegistryError, match="byte ledgers"):
        replace(cell, application_bytes_total=999)


@pytest.mark.parametrize(
    "metric_id",
    (
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
    ),
)
def test_unit_interval_metric_domains_are_strict(
    cells: tuple[SupplementalResultCellV1, ...], metric_id: str
) -> None:
    with pytest.raises(SupplementalRegistryError, match="semantic domain"):
        replace(
            cells[0],
            metric_values=_set_metric(cells[0].metric_values, metric_id, 1.01),
        )


@pytest.mark.parametrize(
    "metric_id",
    (
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
    ),
)
def test_nonnegative_metric_domains_are_strict(
    cells: tuple[SupplementalResultCellV1, ...], metric_id: str
) -> None:
    with pytest.raises(SupplementalRegistryError, match="semantic domain"):
        replace(
            cells[0],
            metric_values=_set_metric(cells[0].metric_values, metric_id, -0.01),
        )


def test_mota_may_be_negative_but_not_above_one_and_state_nll_is_finite_only(
    cells: tuple[SupplementalResultCellV1, ...],
) -> None:
    cell = cells[0]
    with pytest.raises(SupplementalRegistryError, match="semantic domain"):
        replace(cell, metric_values=_set_metric(cell.metric_values, "MOTA", 1.01))
    negative = replace(
        cell,
        metric_values=_set_metric(
            _set_metric(cell.metric_values, "MOTA", -3.0), "state_nll", -10.0
        ),
    )
    assert dict(negative.metric_values)["MOTA"] == -3.0
    with pytest.raises(SupplementalRegistryError, match="must be finite"):
        replace(
            cell,
            metric_values=_set_metric(cell.metric_values, "state_nll", float("nan")),
        )


def test_algorithm_failure_has_no_metrics_and_scores_zero(
    cells: tuple[SupplementalResultCellV1, ...],
) -> None:
    failed = replace(
        cells[0],
        status=SUPPLEMENTAL_STATUS_FAILURE_V1,
        failure_code="tracker-crash",
        metric_values=(),
    )
    assert failed.metric_or_failure_zero("AssA") == 0.0
    with pytest.raises(SupplementalRegistryError, match="requires no metrics"):
        replace(failed, metric_values=_metrics())


def test_full_send_and_oracle_cannot_be_ranked_or_given_nominal_budget(
    cells: tuple[SupplementalResultCellV1, ...],
) -> None:
    full_send = next(
        cell
        for cell in cells
        if cell.scenario_id == FULL_SEND_DIAGNOSTIC_SCENARIO_V1
    )
    oracle = next(
        cell for cell in cells if cell.scenario_id == SYNCHRONOUS_ORACLE_SCENARIO_V1
    )
    for cell in (full_send, oracle):
        with pytest.raises(SupplementalRegistryError, match="scenario budget"):
            replace(cell, table_ranking_eligible=True)
        with pytest.raises(SupplementalRegistryError, match="scenario budget"):
            replace(cell, nominal_budget_bytes_per_second=64_000)


def test_config_manifest_and_cell_config_are_fail_closed(
    registry: PaperSupplementalRegistryV1,
) -> None:
    with pytest.raises(SupplementalRegistryError, match="config manifest"):
        replace(registry, config_manifest_sha256="f" * 64)
    with pytest.raises(SupplementalRegistryError, match="wrong config"):
        replace(
            registry,
            cells=(replace(registry.cells[0], config_sha256="f" * 64),)
            + registry.cells[1:],
        )


def test_canonical_content_hash_detects_tampering(
    registry: PaperSupplementalRegistryV1,
) -> None:
    document = json.loads(registry.canonical_bytes())
    document["cells"][0]["attempt_id"] = "attempt-tampered"
    with pytest.raises(SupplementalRegistryError, match="content SHA-256 mismatch"):
        PaperSupplementalRegistryV1.from_mapping(document)


def test_c6_c8_require_condition_manifest_and_other_conditions_forbid_it(
    cells: tuple[SupplementalResultCellV1, ...],
) -> None:
    disturbed = next(cell for cell in cells if cell.condition_id == "C6")
    clean = next(cell for cell in cells if cell.condition_id == "C0")
    with pytest.raises(SupplementalRegistryError, match="C6-C8 require"):
        replace(disturbed, condition_input_manifest_sha256=None)
    with pytest.raises(SupplementalRegistryError, match="C0-C5 and C9"):
        replace(clean, condition_input_manifest_sha256="f" * 64)


def test_external_input_binding_checks_c9_and_exact_manifest_inventory(
    registry: PaperSupplementalRegistryV1,
) -> None:
    receipt = _measured_receipt()
    bound, manifests = _bind_external_inputs(registry, receipt)
    _validate_external_input_bindings_v1(bound, receipt, manifests)

    c9_index = next(
        index for index, cell in enumerate(bound.cells) if cell.condition_id == "C9"
    )
    stale_c9 = list(bound.cells)
    stale_c9[c9_index] = replace(
        stale_c9[c9_index], channel_outcome_trace_sha256="f" * 64
    )
    with pytest.raises(SupplementalRegistryError, match="held-out packet"):
        _validate_external_input_bindings_v1(
            replace(bound, cells=tuple(stale_c9)), receipt, manifests
        )

    extra_manifest = replace(
        next(iter(manifests.values())), run_id="extra-attempt"
    )
    with pytest.raises(SupplementalRegistryError, match="missing or extra"):
        _validate_external_input_bindings_v1(
            bound,
            receipt,
            {**manifests, extra_manifest.content_sha256: extra_manifest},
        )


@pytest.mark.parametrize(
    "mismatch",
    ("run_id", "condition_id", "network_trace", "run_config", "detection_cache"),
)
def test_condition_manifest_five_tuple_must_match_physical_run(
    registry: PaperSupplementalRegistryV1,
    mismatch: str,
) -> None:
    receipt = _measured_receipt()
    bound, manifests = _bind_external_inputs(registry, receipt)
    index, cell = next(
        (index, cell)
        for index, cell in enumerate(bound.cells)
        if cell.condition_id == "C6"
    )
    assert cell.condition_input_manifest_sha256 is not None
    original = manifests[cell.condition_input_manifest_sha256]
    if mismatch == "condition_id":
        condition = NetworkConditionId.C7
        changed = replace(
            original,
            condition_id=condition,
            condition_plan_sha256=condition_plan_v1(condition).content_sha256,
        )
    else:
        field = {
            "run_id": "run_id",
            "network_trace": "network_trace_sha256",
            "run_config": "run_config_sha256",
            "detection_cache": "detection_cache_sha256",
        }[mismatch]
        changed = replace(
            original,
            **{field: "other-attempt" if field == "run_id" else "f" * 64},
        )
    cells = list(bound.cells)
    cells[index] = replace(
        cell, condition_input_manifest_sha256=changed.content_sha256
    )
    changed_inventory = {
        digest: manifest
        for digest, manifest in manifests.items()
        if digest != original.content_sha256
    }
    changed_inventory[changed.content_sha256] = changed
    with pytest.raises(SupplementalRegistryError, match="physical run"):
        _validate_external_input_bindings_v1(
            replace(bound, cells=tuple(cells)), receipt, changed_inventory
        )


def test_attempt_and_manifest_reuse_across_semantic_cells_is_forbidden(
    registry: PaperSupplementalRegistryV1,
) -> None:
    with pytest.raises(SupplementalRegistryError, match="unique physical attempt"):
        replace(
            registry,
            cells=(
                registry.cells[0],
                replace(registry.cells[1], attempt_id=registry.cells[0].attempt_id),
                *registry.cells[2:],
            ),
        )

    receipt = _measured_receipt()
    bound, manifests = _bind_external_inputs(registry, receipt)
    disturbed = [
        (index, cell)
        for index, cell in enumerate(bound.cells)
        if cell.condition_id in {"C6", "C7", "C8"}
    ]
    first_index, first = disturbed[0]
    second_index, second = disturbed[1]
    reused = list(bound.cells)
    reused[second_index] = replace(
        second,
        condition_input_manifest_sha256=(
            first.condition_input_manifest_sha256
        ),
    )
    with pytest.raises(SupplementalRegistryError, match="cannot be reused"):
        _validate_external_input_bindings_v1(
            replace(bound, cells=tuple(reused)), receipt, manifests
        )
    assert first_index != second_index


def test_formal_validator_wires_receipt_and_manifest_inventory(
    monkeypatch,
    registry: PaperSupplementalRegistryV1,
    cohort: SupplementalCohortV1,
) -> None:
    receipt = _measured_receipt()
    plan = replace(
        _formal_plan(),
        heldout_c9_trace_receipt_sha256=receipt.content_sha256,
    )
    formal_bindings = tuple(
        replace(
            binding,
            config_sha256=plan.run_config_sha256(
                "primary", "eventtrack-v2x", binding.scheduler_id
            ),
        )
        if (
            binding.detector_input_id == "camera-detector-locked"
            and binding.method_id == "eventtrack-v2x"
        )
        else binding
        for binding in registry.config_bindings
    )
    formal_configs = {
        binding.key: binding.config_sha256 for binding in formal_bindings
    }
    aligned_cells = tuple(
        replace(
            cell,
            experiment_plan_sha256=plan.content_sha256,
            detection_cache_sha256=(
                plan.detector_cache_sha256
                if cell.detector_input_id.startswith("camera")
                else registry.raw_sensor_detector_cache_sha256
            ),
            evaluator_contract_sha256=plan.evaluator_contract_sha256,
            config_sha256=formal_configs[
                (cell.detector_input_id, cell.method_id, cell.scheduler_id)
            ],
        )
        for cell in registry.cells
    )
    formal_unbound = replace(
        registry,
        experiment_plan_sha256=plan.content_sha256,
        dataset_manifest_sha256=plan.dataset_manifest_sha256,
        split_sha256=plan.split_sha256,
        cohort_sha256=plan.primary_cohort_sha256,
        frame_contract_sha256=plan.primary_frame_contract_sha256,
        primary_detector_cache_sha256=plan.detector_cache_sha256,
        evaluator_contract_sha256=plan.evaluator_contract_sha256,
        network_trace_manifest_sha256=plan.network_trace_manifest_sha256,
        wire_accounting_config_sha256=plan.wire_accounting_config_sha256,
        heldout_c9_trace_receipt_sha256=receipt.content_sha256,
        config_bindings=formal_bindings,
        config_manifest_sha256=supplemental_config_manifest_sha256_v1(
            formal_bindings
        ),
        cells=aligned_cells,
    )
    formal_registry, manifests = _bind_external_inputs(formal_unbound, receipt)
    monkeypatch.setattr(
        supplemental_module,
        "formal_supplemental_cohort_v1",
        lambda _plan: cohort,
    )
    monkeypatch.setattr(
        supplemental_module,
        "FORMAL_SUPPLEMENTAL_CELL_COUNT_V1",
        len(formal_registry.cells),
    )
    validate_paper_supplemental_registry_v1(
        formal_registry,
        plan,
        raw_sensor_detector_cache_sha256=(
            formal_registry.raw_sensor_detector_cache_sha256
        ),
        measured_trace_receipt=receipt,
        condition_input_manifests=manifests,
    )

    with pytest.raises(SupplementalRegistryError, match="receipt content"):
        validate_paper_supplemental_registry_v1(
            formal_registry,
            replace(plan, heldout_c9_trace_receipt_sha256="f" * 64),
            raw_sensor_detector_cache_sha256=(
                formal_registry.raw_sensor_detector_cache_sha256
            ),
            measured_trace_receipt=receipt,
            condition_input_manifests=manifests,
        )
    wrong_ids = tuple(f"other-trace-{index:02d}" for index in range(20))
    with pytest.raises(SupplementalRegistryError, match="held-out IDs"):
        validate_paper_supplemental_registry_v1(
            formal_registry,
            replace(plan, c9_trace_ids=wrong_ids),
            raw_sensor_detector_cache_sha256=(
                formal_registry.raw_sensor_detector_cache_sha256
            ),
            measured_trace_receipt=receipt,
            condition_input_manifests=manifests,
        )

    drift_bindings = tuple(
        replace(binding, config_sha256="f" * 64)
        if (
            binding.detector_input_id == "camera-detector-locked"
            and binding.method_id == "eventtrack-v2x"
            and binding.scheduler_id == "periodic"
        )
        else binding
        for binding in formal_registry.config_bindings
    )
    drift_cells = tuple(
        replace(cell, config_sha256="f" * 64)
        if (
            cell.detector_input_id == "camera-detector-locked"
            and cell.method_id == "eventtrack-v2x"
            and cell.scheduler_id == "periodic"
        )
        else cell
        for cell in formal_registry.cells
    )
    drift_registry = replace(
        formal_registry,
        config_bindings=drift_bindings,
        config_manifest_sha256=supplemental_config_manifest_sha256_v1(
            drift_bindings
        ),
        cells=drift_cells,
    )
    with pytest.raises(SupplementalRegistryError, match="does not match plan"):
        validate_paper_supplemental_registry_v1(
            drift_registry,
            plan,
            raw_sensor_detector_cache_sha256=(
                drift_registry.raw_sensor_detector_cache_sha256
            ),
            measured_trace_receipt=receipt,
            condition_input_manifests=manifests,
        )
