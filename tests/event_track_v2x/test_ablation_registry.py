from collections import Counter
from dataclasses import replace
import hashlib
import json

import pytest

from transvision.models.event_track_v2x import ablation_registry as ablation_module
from transvision.models.event_track_v2x.ablation_registry import (
    ABLATION_BUDGET_BYTES_PER_SECOND_V1,
    ABLATION_FAILURE_SEMANTICS_V1,
    ABLATION_METRICS_ARTIFACT_ROLE_V1,
    ABLATION_PAPER_SCOPE_V1,
    ABLATION_STATUS_FAILURE_V1,
    ABLATION_STATUS_SUCCESS_V1,
    ABLATION_VARIANT_IDS_V1,
    DISTURBED_INPUT_CONDITION_IDS_V1,
    FORMAL_ABLATION_CELL_COUNT_V1,
    REQUIRED_ABLATION_METRIC_IDS_V1,
    AblationCohortV1,
    AblationConfigBindingV1,
    AblationRegistryError,
    AblationResultCellV1,
    AblationResultRegistryV1,
    _iter_expected_ablation_specs_for_cohort,
    _validate_external_input_bindings_v1,
    _validate_registry_against_cohort,
    ablation_config_manifest_sha256_v1,
    decode_ablation_result_registry,
    iter_expected_paper_ablation_specs_v1,
    validate_ablation_result_registry_v1,
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
from transvision.models.event_track_v2x.network import (
    NetworkConditionId,
    condition_plan_v1,
)
from transvision.models.event_track_v2x.network_disturbance import (
    ConditionInputManifestV1,
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
    registry: AblationResultRegistryV1,
    receipt: MeasuredTraceReceiptV1,
) -> tuple[AblationResultRegistryV1, dict[str, ConditionInputManifestV1]]:
    packets = dict(receipt.held_out_packet_metadata_sha256s)
    manifests: dict[str, ConditionInputManifestV1] = {}
    cells = []
    for cell in registry.cells:
        changes: dict[str, object] = {}
        if cell.condition_id == "C9":
            assert cell.c9_trace_id is not None
            changes["channel_outcome_trace_sha256"] = packets[cell.c9_trace_id]
        if cell.condition_id in DISTURBED_INPUT_CONDITION_IDS_V1:
            condition = NetworkConditionId(cell.condition_id)
            manifest = ConditionInputManifestV1(
                run_id=cell.attempt_id,
                condition_id=condition,
                network_trace_sha256=cell.network_trace_sha256,
                condition_plan_sha256=condition_plan_v1(condition).content_sha256,
                run_config_sha256=cell.variant_config_sha256,
                detection_cache_sha256=cell.detection_cache_sha256,
                evidence_sha256s=(_digest(("input-evidence", cell.key)),),
            )
            manifests[manifest.content_sha256] = manifest
            changes["disturbed_input_manifest_sha256"] = manifest.content_sha256
        cells.append(replace(cell, **changes))
    return replace(registry, cells=tuple(cells)), manifests


@pytest.fixture(scope="module")
def cohort() -> AblationCohortV1:
    return AblationCohortV1(
        sequence_ids=("sequence-00",),
        training_seeds=(1337,),
        network_seeds=(1001,),
        c9_trace_ids=("heldout-trace-00",),
    )


@pytest.fixture(scope="module")
def config_bindings() -> tuple[AblationConfigBindingV1, ...]:
    return tuple(
        AblationConfigBindingV1(variant_id, _digest(("config", variant_id)))
        for variant_id in ABLATION_VARIANT_IDS_V1
    )


def _metrics() -> tuple[tuple[str, float], ...]:
    return tuple((metric_id, 0.5) for metric_id in REQUIRED_ABLATION_METRIC_IDS_V1)


def _set_metric(
    metrics: tuple[tuple[str, float], ...], metric_id: str, value: float
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (observed_id, value if observed_id == metric_id else observed_value)
        for observed_id, observed_value in metrics
    )


@pytest.fixture(scope="module")
def cells(
    cohort: AblationCohortV1,
    config_bindings: tuple[AblationConfigBindingV1, ...],
) -> tuple[AblationResultCellV1, ...]:
    configs = {item.variant_id: item.config_sha256 for item in config_bindings}
    result = []
    for index, spec in enumerate(_iter_expected_ablation_specs_for_cohort(cohort)):
        input_manifest = (
            _digest(("disturbed-input", index))
            if spec.condition_id in DISTURBED_INPUT_CONDITION_IDS_V1
            else None
        )
        result.append(
            AblationResultCellV1(
                experiment_plan_sha256="0" * 64,
                variant_id=spec.variant_id,
                sequence_id=spec.sequence_id,
                condition_id=spec.condition_id,
                training_seed=spec.training_seed,
                network_seed=spec.network_seed,
                c9_trace_id=spec.c9_trace_id,
                budget_bytes_per_second=ABLATION_BUDGET_BYTES_PER_SECOND_V1,
                status=ABLATION_STATUS_SUCCESS_V1,
                failure_code=None,
                metric_values=_metrics(),
                on_wire_bytes_total=1_200,
                measurement_duration_seconds=10.0,
                attempt_id=f"attempt-{index:03d}",
                variant_config_sha256=configs[spec.variant_id],
                detection_cache_sha256="1" * 64,
                network_trace_sha256=_digest(("network", index)),
                channel_outcome_trace_sha256=_digest(("channel", index)),
                disturbed_input_manifest_sha256=input_manifest,
                prediction_archive_sha256=_digest(("prediction", index)),
                evaluator_contract_sha256="2" * 64,
                metrics_artifact_role=ABLATION_METRICS_ARTIFACT_ROLE_V1,
                metrics_artifact_sha256=_digest(("metrics", index)),
                evidence_bundle_sha256=_digest(("evidence", index)),
                wire_ledger_sha256=_digest(("wire", index)),
            )
        )
    return tuple(result)


@pytest.fixture(scope="module")
def registry(
    cohort: AblationCohortV1,
    config_bindings: tuple[AblationConfigBindingV1, ...],
    cells: tuple[AblationResultCellV1, ...],
) -> AblationResultRegistryV1:
    return AblationResultRegistryV1(
        experiment_plan_sha256="0" * 64,
        dataset_manifest_sha256="3" * 64,
        split_sha256="4" * 64,
        cohort_sha256="5" * 64,
        frame_contract_sha256="6" * 64,
        detection_cache_sha256="1" * 64,
        evaluator_contract_sha256="2" * 64,
        network_trace_manifest_sha256="7" * 64,
        wire_accounting_config_sha256="8" * 64,
        heldout_c9_trace_receipt_sha256="9" * 64,
        sequence_ids=cohort.sequence_ids,
        c9_trace_ids=cohort.c9_trace_ids,
        config_bindings=config_bindings,
        config_manifest_sha256=ablation_config_manifest_sha256_v1(
            config_bindings
        ),
        paper_scope=ABLATION_PAPER_SCOPE_V1,
        contributes_to_primary_sota_gate=False,
        failure_semantics=ABLATION_FAILURE_SEMANTICS_V1,
        cells=cells,
    )


def _formal_plan() -> ExperimentPlanV1:
    sequences = tuple(f"sequence-{index:02d}" for index in range(21))
    traces = tuple(f"heldout-trace-{index:02d}" for index in range(20))
    run_configs = tuple(
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


def test_exact_variant_set_and_small_cohort_matrix(cohort: AblationCohortV1) -> None:
    assert set(ABLATION_VARIANT_IDS_V1) == {
        "full_eventtrack",
        "no_oosm",
        "point_timestamp",
        "no_mixture",
        "no_lineage",
        "correlated_as_independent",
        "no_ci",
        "top1_hard_association",
        "no_identity_entropy",
        "confidence_top_k_instead_of_voi",
    }
    specs = tuple(_iter_expected_ablation_specs_for_cohort(cohort))
    assert len(specs) == 100
    assert Counter(item.variant_id for item in specs) == {
        variant_id: 10 for variant_id in ABLATION_VARIANT_IDS_V1
    }


def test_formal_iterator_matches_independent_formula() -> None:
    pressure_replicates = 1 + 8 * 10 + 20
    independent_formula = 21 * 3 * pressure_replicates * 10
    assert independent_formula == FORMAL_ABLATION_CELL_COUNT_V1 == 63_630
    assert (
        sum(1 for _ in iter_expected_paper_ablation_specs_v1(_formal_plan()))
        == independent_formula
    )


def test_registry_round_trips_and_is_outside_primary_gate(
    registry: AblationResultRegistryV1,
    cohort: AblationCohortV1,
) -> None:
    _validate_registry_against_cohort(registry, cohort)
    decoded = decode_ablation_result_registry(registry.canonical_bytes())
    assert decoded == registry
    assert decoded.digest() == registry.digest()
    assert decoded.paper_scope == "supplement-table-iv-only"
    assert decoded.contributes_to_primary_sota_gate is False


def test_strict_decoder_rejects_duplicate_unknown_and_tampered_data(
    registry: AblationResultRegistryV1,
) -> None:
    duplicate = registry.canonical_bytes().replace(
        b'{"c9_trace_ids":', b'{"c9_trace_ids":[],"c9_trace_ids":', 1
    )
    with pytest.raises(AblationRegistryError, match="duplicate JSON key"):
        decode_ablation_result_registry(duplicate)

    document = registry.sealed_document()
    document["unknown"] = True
    with pytest.raises(AblationRegistryError, match="missing or unknown"):
        AblationResultRegistryV1.from_mapping(document)

    document = json.loads(registry.canonical_bytes())
    document["cells"][0]["attempt_id"] = "tampered"
    with pytest.raises(AblationRegistryError, match="content SHA-256 mismatch"):
        AblationResultRegistryV1.from_mapping(document)


def test_missing_extra_and_duplicate_cells_fail_closed(
    registry: AblationResultRegistryV1,
    cohort: AblationCohortV1,
) -> None:
    missing = replace(registry, cells=registry.cells[:-1])
    with pytest.raises(AblationRegistryError, match="missing expected cell"):
        _validate_registry_against_cohort(missing, cohort)

    extra_cell = replace(
        registry.cells[-1],
        sequence_id="sequence-extra",
        attempt_id="attempt-extra",
    )
    extra = replace(registry, cells=registry.cells + (extra_cell,))
    with pytest.raises(AblationRegistryError, match="unexpected extra cell"):
        _validate_registry_against_cohort(extra, cohort)

    with pytest.raises(AblationRegistryError, match="unique and sorted"):
        replace(registry, cells=registry.cells + (registry.cells[-1],))


def test_c6_c8_require_disturbed_input_manifest_and_other_conditions_forbid_it(
    cells: tuple[AblationResultCellV1, ...],
) -> None:
    disturbed = next(cell for cell in cells if cell.condition_id == "C6")
    clean = next(cell for cell in cells if cell.condition_id == "C0")
    with pytest.raises(AblationRegistryError, match="require a disturbed"):
        replace(disturbed, disturbed_input_manifest_sha256=None)
    with pytest.raises(AblationRegistryError, match="only C6-C8"):
        replace(clean, disturbed_input_manifest_sha256="f" * 64)


def test_success_metrics_failure_zero_and_64k_budget_are_strict(
    cells: tuple[AblationResultCellV1, ...],
) -> None:
    cell = cells[0]
    with pytest.raises(AblationRegistryError, match="every exact"):
        replace(cell, metric_values=cell.metric_values[:-1])
    failed = replace(
        cell,
        status=ABLATION_STATUS_FAILURE_V1,
        failure_code="tracker-crash",
        metric_values=(),
    )
    assert failed.metric_or_failure_zero("AssA") == 0.0
    with pytest.raises(AblationRegistryError, match="64 kB/s"):
        replace(cell, budget_bytes_per_second=32_000)
    with pytest.raises(AblationRegistryError, match="exceeds 64"):
        replace(cell, on_wire_bytes_total=640_001, measurement_duration_seconds=10.0)


@pytest.mark.parametrize("metric_id", ("AMOTA", "HOTA", "coverage_95"))
def test_ablation_unit_interval_metrics_are_strict(
    cells: tuple[AblationResultCellV1, ...], metric_id: str
) -> None:
    with pytest.raises(AblationRegistryError, match="semantic domain"):
        replace(
            cells[0],
            metric_values=_set_metric(cells[0].metric_values, metric_id, 1.01),
        )


@pytest.mark.parametrize(
    "metric_id", ("AMOTP", "IDS", "nees", "frame_latency_p95_ms")
)
def test_ablation_nonnegative_metrics_are_strict(
    cells: tuple[AblationResultCellV1, ...], metric_id: str
) -> None:
    with pytest.raises(AblationRegistryError, match="semantic domain"):
        replace(
            cells[0],
            metric_values=_set_metric(cells[0].metric_values, metric_id, -0.01),
        )


def test_config_cache_and_evaluator_bindings_are_fail_closed(
    registry: AblationResultRegistryV1,
) -> None:
    with pytest.raises(AblationRegistryError, match="config manifest"):
        replace(registry, config_manifest_sha256="f" * 64)
    with pytest.raises(AblationRegistryError, match="wrong variant config"):
        replace(
            registry,
            cells=(replace(registry.cells[0], variant_config_sha256="f" * 64),)
            + registry.cells[1:],
        )
    with pytest.raises(AblationRegistryError, match="wrong detector-cache"):
        replace(
            registry,
            cells=(replace(registry.cells[0], detection_cache_sha256="f" * 64),)
            + registry.cells[1:],
        )

    duplicate_config = replace(
        registry.config_bindings[1],
        config_sha256=registry.config_bindings[0].config_sha256,
    )
    duplicate_bindings = (
        registry.config_bindings[0],
        duplicate_config,
        *registry.config_bindings[2:],
    )
    with pytest.raises(AblationRegistryError, match="distinct config"):
        replace(
            registry,
            config_bindings=duplicate_bindings,
            config_manifest_sha256=ablation_config_manifest_sha256_v1(
                duplicate_bindings
            ),
        )


def test_external_input_binding_checks_c9_and_exact_manifest_inventory(
    registry: AblationResultRegistryV1,
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
    with pytest.raises(AblationRegistryError, match="held-out packet"):
        _validate_external_input_bindings_v1(
            replace(bound, cells=tuple(stale_c9)), receipt, manifests
        )

    extra_manifest = replace(
        next(iter(manifests.values())), run_id="extra-attempt"
    )
    with pytest.raises(AblationRegistryError, match="missing or extra"):
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
    registry: AblationResultRegistryV1,
    mismatch: str,
) -> None:
    receipt = _measured_receipt()
    bound, manifests = _bind_external_inputs(registry, receipt)
    index, cell = next(
        (index, cell)
        for index, cell in enumerate(bound.cells)
        if cell.condition_id == "C6"
    )
    assert cell.disturbed_input_manifest_sha256 is not None
    original = manifests[cell.disturbed_input_manifest_sha256]
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
        cell, disturbed_input_manifest_sha256=changed.content_sha256
    )
    changed_inventory = {
        digest: manifest
        for digest, manifest in manifests.items()
        if digest != original.content_sha256
    }
    changed_inventory[changed.content_sha256] = changed
    with pytest.raises(AblationRegistryError, match="physical run"):
        _validate_external_input_bindings_v1(
            replace(bound, cells=tuple(cells)), receipt, changed_inventory
        )


def test_attempt_and_manifest_reuse_across_semantic_cells_is_forbidden(
    registry: AblationResultRegistryV1,
) -> None:
    with pytest.raises(AblationRegistryError, match="unique physical attempt"):
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
        if cell.condition_id in DISTURBED_INPUT_CONDITION_IDS_V1
    ]
    first_index, first = disturbed[0]
    second_index, second = disturbed[1]
    reused = list(bound.cells)
    reused[second_index] = replace(
        second,
        disturbed_input_manifest_sha256=(
            first.disturbed_input_manifest_sha256
        ),
    )
    with pytest.raises(AblationRegistryError, match="cannot be reused"):
        _validate_external_input_bindings_v1(
            replace(bound, cells=tuple(reused)), receipt, manifests
        )
    assert first_index != second_index


def test_formal_validator_wires_receipt_and_manifest_inventory(
    monkeypatch,
    registry: AblationResultRegistryV1,
    cohort: AblationCohortV1,
) -> None:
    receipt = _measured_receipt()
    plan = replace(
        _formal_plan(),
        heldout_c9_trace_receipt_sha256=receipt.content_sha256,
    )
    full_config = plan.run_config_sha256(
        "primary", "eventtrack-v2x", "marginal_voi"
    )
    bindings = tuple(
        replace(item, config_sha256=full_config)
        if item.variant_id == "full_eventtrack"
        else item
        for item in registry.config_bindings
    )
    aligned_cells = tuple(
        replace(
            cell,
            experiment_plan_sha256=plan.content_sha256,
            variant_config_sha256=(
                full_config
                if cell.variant_id == "full_eventtrack"
                else cell.variant_config_sha256
            ),
            detection_cache_sha256=plan.detector_cache_sha256,
            evaluator_contract_sha256=plan.evaluator_contract_sha256,
        )
        for cell in registry.cells
    )
    formal_unbound = replace(
        registry,
        config_bindings=bindings,
        config_manifest_sha256=ablation_config_manifest_sha256_v1(bindings),
        cells=aligned_cells,
        experiment_plan_sha256=plan.content_sha256,
        dataset_manifest_sha256=plan.dataset_manifest_sha256,
        split_sha256=plan.split_sha256,
        cohort_sha256=plan.primary_cohort_sha256,
        frame_contract_sha256=plan.primary_frame_contract_sha256,
        detection_cache_sha256=plan.detector_cache_sha256,
        evaluator_contract_sha256=plan.evaluator_contract_sha256,
        network_trace_manifest_sha256=plan.network_trace_manifest_sha256,
        wire_accounting_config_sha256=plan.wire_accounting_config_sha256,
        heldout_c9_trace_receipt_sha256=receipt.content_sha256,
    )
    formal_registry, manifests = _bind_external_inputs(formal_unbound, receipt)
    monkeypatch.setattr(
        ablation_module,
        "formal_ablation_cohort_v1",
        lambda _plan: cohort,
    )
    monkeypatch.setattr(
        ablation_module,
        "FORMAL_ABLATION_CELL_COUNT_V1",
        len(formal_registry.cells),
    )
    validate_ablation_result_registry_v1(
        formal_registry,
        plan,
        measured_trace_receipt=receipt,
        condition_input_manifests=manifests,
    )
    with pytest.raises(AblationRegistryError, match="receipt content"):
        validate_ablation_result_registry_v1(
            formal_registry,
            replace(plan, heldout_c9_trace_receipt_sha256="f" * 64),
            measured_trace_receipt=receipt,
            condition_input_manifests=manifests,
        )
    wrong_ids = tuple(f"other-trace-{index:02d}" for index in range(20))
    with pytest.raises(AblationRegistryError, match="held-out IDs"):
        validate_ablation_result_registry_v1(
            formal_registry,
            replace(plan, c9_trace_ids=wrong_ids),
            measured_trace_receipt=receipt,
            condition_input_manifests=manifests,
        )
