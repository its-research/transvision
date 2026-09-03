from dataclasses import replace
import hashlib

import pytest

from transvision.models.event_track_v2x.contracts import (
    ArtifactDigestV1,
    EvidenceBundleV1,
)
from transvision.models.event_track_v2x.experiment import (
    CONFIRMATORY_METHOD_IDS_V1,
    CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1,
    GRIFFIN_25M_VAL_SEQUENCE_IDS_V1,
    ExperimentPlanV1,
    RunConfigBindingV1,
    confirmatory_run_config_keys_v1,
)
from transvision.models.event_track_v2x.publication_gate import (
    BudgetedSequenceScoresV1,
    ColdCacheReceiptV1,
    InvariantEvidenceV1,
    MethodScoresV1,
    NamedMetricScoresV1,
    PublicationMetricsV1,
    REQUIRED_INVARIANTS_V1,
    SequenceScoresV1,
    decode_publication_metrics,
    evaluate_publication_gate_v1,
    paired_pareto_auc_by_sequence_v1,
)


PRIMARY_SEQUENCES = tuple(f"sequence-{index:02d}" for index in range(21))
GRIFFIN_SEQUENCES = GRIFFIN_25M_VAL_SEQUENCE_IDS_V1
BUDGETS = (16000, 32000, 64000, 128000, 256000)


def _scores(
    sequence_ids: tuple[str, ...], value: float
) -> SequenceScoresV1:
    return SequenceScoresV1(
        tuple(
            (sequence_id, value + index * 1e-5)
            for index, sequence_id in enumerate(sequence_ids)
        )
    )


def _budgeted(
    sequence_ids: tuple[str, ...], scheduler_id: str, value: float
) -> BudgetedSequenceScoresV1:
    utilization = 0.90 if scheduler_id == "marginal_voi" else 0.80
    return BudgetedSequenceScoresV1(
        scheduler_id=scheduler_id,
        byte_budgets_per_second=BUDGETS,
        sequence_scores=tuple(
            (
                sequence_id,
                tuple(value + budget_index * 0.01 for budget_index in range(5)),
            )
            for sequence_id in sequence_ids
        ),
        sequence_on_wire_bps=tuple(
            (
                sequence_id,
                tuple(budget * utilization for budget in BUDGETS),
            )
            for sequence_id in sequence_ids
        ),
    )


def _plan() -> ExperimentPlanV1:
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
        dataset_manifest_sha256="3" * 64,
        split_name="val",
        split_sha256="4" * 64,
        primary_cohort_sha256="8" * 64,
        primary_frame_contract_sha256="9" * 64,
        primary_dataset_release_receipt_sha256="0" * 64,
        development_dataset_release_receipt_sha256="e" * 64,
        development_fold_manifest_sha256="f" * 64,
        primary_sequence_ids=PRIMARY_SEQUENCES,
        detector_cache_sha256="5" * 64,
        network_trace_manifest_sha256="6" * 64,
        wire_accounting_config_sha256="f" * 64,
        heldout_c9_trace_receipt_sha256="a" * 64,
        c9_trace_ids=tuple(f"heldout-{index:02d}" for index in range(20)),
        evaluator_contract_sha256="7" * 64,
        baseline_qualification_registry_sha256="8" * 64,
        candidate_selection_registry_sha256="a" * 64,
        preregistration_provider_id="institutional-log",
        preregistration_public_key_sha256="9" * 64,
        verification_provider_id="independent-verifier",
        verification_public_key_sha256="a" * 64,
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
        griffin_dataset_manifest_sha256="b" * 64,
        griffin_split_name="val",
        griffin_cohort_sha256="c" * 64,
        griffin_frame_contract_sha256="f" * 64,
        griffin_dataset_release_receipt_sha256="0" * 64,
        griffin_sequence_ids=GRIFFIN_SEQUENCES,
        griffin_detector_cache_sha256="d" * 64,
        griffin_evaluator_contract_sha256="e" * 64,
        griffin_strongest_baseline_id="vehicle-only-ab3dmot",
        frequency_hz=10.0,
        class_names=("car",),
        method_ids=CONFIRMATORY_METHOD_IDS_V1,
        network_condition_ids=tuple(f"C{index}" for index in range(10)),
        training_seeds=(1337, 2027, 3407),
        network_seeds=tuple(range(1001, 1011)),
        byte_budgets_per_second=BUDGETS,
        primary_budget_bytes_per_second=64000,
        primary_metric="robust-assa-at-64k",
        decision_deadline_ms=100.0,
        statistical_alpha=0.05,
        statistical_resamples=10000,
        statistical_random_seed=1337,
        claim_scope="same-detector robust cooperative association",
    )


def _metrics(plan: ExperimentPlanV1) -> PublicationMetricsV1:
    return PublicationMetricsV1(
        experiment_plan_sha256=plan.content_sha256,
        candidate_method_id="eventtrack-v2x",
        robust_metric_id="robust-assa-at-64k",
        robust_eventtrack=_scores(PRIMARY_SEQUENCES, 0.70),
        robust_baselines=tuple(
            MethodScoresV1(
                baseline,
                _scores(
                    PRIMARY_SEQUENCES,
                    0.68
                    if baseline == "vehicle-only-ab3dmot"
                    else 0.66 - index * 0.005,
                ),
            )
            for index, baseline in enumerate(
                CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1
            )
        ),
        clean_eventtrack=(
            NamedMetricScoresV1("AMOTA", _scores(PRIMARY_SEQUENCES, 0.72)),
            NamedMetricScoresV1("HOTA", _scores(PRIMARY_SEQUENCES, 0.71)),
        ),
        clean_strongest_baseline_id="vehicle-only-ab3dmot",
        clean_strongest_baseline=(
            NamedMetricScoresV1("AMOTA", _scores(PRIMARY_SEQUENCES, 0.725)),
            NamedMetricScoresV1("HOTA", _scores(PRIMARY_SEQUENCES, 0.715)),
        ),
        pareto_metric_id="AssA",
        pareto_eventtrack=_budgeted(
            PRIMARY_SEQUENCES, "marginal_voi", 0.80
        ),
        pareto_strongest_baseline=_budgeted(
            PRIMARY_SEQUENCES, "confidence_top_k", 0.76
        ),
        griffin_metric_id="robust-assa-at-64k",
        griffin_eventtrack=_scores(GRIFFIN_SEQUENCES, 0.65),
        griffin_strongest_baseline_id="vehicle-only-ab3dmot",
        griffin_strongest_baseline=_scores(GRIFFIN_SEQUENCES, 0.62),
    )


def _evidence(plan: ExperimentPlanV1, metrics: PublicationMetricsV1) -> EvidenceBundleV1:
    hashes = {
        "source": plan.source_tree_sha256,
        "dataset": plan.dataset_manifest_sha256,
        "detection_cache": plan.detector_cache_sha256,
        "network_trace": plan.network_trace_manifest_sha256,
        "tracker_config": plan.method_config_sha256,
        "evaluator_contract": plan.evaluator_contract_sha256,
        "per_sequence_metrics": metrics.digest(),
    }
    artifacts = {
        name: ArtifactDigestV1(
            f"artifacts/{name}", hashes.get(name, "8" * 64), 1
        )
        for name in EvidenceBundleV1._ARTIFACT_FIELDS
    }
    return EvidenceBundleV1(run_id="aggregate-result", **artifacts)


def _invariants(
    plan: ExperimentPlanV1,
    evidence: EvidenceBundleV1,
    *,
    failed: str | None = None,
) -> tuple[InvariantEvidenceV1, ...]:
    return tuple(
        InvariantEvidenceV1(
            invariant_id=name,
            passed=name != failed,
            experiment_plan_sha256=plan.content_sha256,
            evidence_bundle_sha256=evidence.digest(),
            artifact_role="logs",
            artifact=evidence.logs,
        )
        for name in REQUIRED_INVARIANTS_V1
    )


def _receipt(
    plan: ExperimentPlanV1,
    evidence: EvidenceBundleV1,
    *,
    verified: bool = True,
) -> ColdCacheReceiptV1:
    return ColdCacheReceiptV1(
        receipt_id="cold-cache-readback-01",
        verified=verified,
        experiment_plan_sha256=plan.content_sha256,
        evidence_bundle_sha256=evidence.digest(),
        verified_artifacts=tuple(
            (role, getattr(evidence, role))
            for role in sorted(EvidenceBundleV1._ARTIFACT_FIELDS)
        ),
    )


def _gate(
    *,
    plan: ExperimentPlanV1 | None = None,
    metrics: PublicationMetricsV1 | None = None,
    evidence: EvidenceBundleV1 | None = None,
    invariants: tuple[InvariantEvidenceV1, ...] | None = None,
    cold_cache_receipt: ColdCacheReceiptV1 | None | object = ...,
):
    selected_plan = _plan() if plan is None else plan
    selected_metrics = _metrics(selected_plan) if metrics is None else metrics
    selected_evidence = (
        _evidence(selected_plan, selected_metrics) if evidence is None else evidence
    )
    selected_invariants = (
        _invariants(selected_plan, selected_evidence)
        if invariants is None
        else invariants
    )
    selected_receipt = (
        _receipt(selected_plan, selected_evidence)
        if cold_cache_receipt is ...
        else cold_cache_receipt
    )
    return evaluate_publication_gate_v1(
        experiment_plan=selected_plan,
        evidence_bundle=selected_evidence,
        publication_metrics=selected_metrics,
        invariants=selected_invariants,
        cold_cache_receipt=selected_receipt,
    )


def test_direct_metrics_checker_never_grants_scientific_sota_wording() -> None:
    plan = _plan()
    metrics = _metrics(plan)
    assert decode_publication_metrics(metrics.canonical_bytes()) == metrics
    result = _gate(plan=plan, metrics=metrics)
    assert not result.sota_wording_allowed
    assert result.failed_gates == ("trusted raw-result provenance",)
    assert all(item.bootstrap.resamples == 10000 for item in result.robust_comparisons)
    assert all(item.bootstrap.seed == 1337 for item in result.robust_comparisons)


def test_gate_fails_for_small_effect_invariant_or_cold_cache() -> None:
    plan = _plan()
    metrics = replace(
        _metrics(plan), robust_eventtrack=_scores(PRIMARY_SEQUENCES, 0.685)
    )
    evidence = _evidence(plan, metrics)
    result = _gate(
        plan=plan,
        metrics=metrics,
        evidence=evidence,
        invariants=_invariants(plan, evidence, failed="evaluator_parity"),
        cold_cache_receipt=None,
    )
    assert not result.sota_wording_allowed
    assert any(
        "vehicle-only-ab3dmot" in failure for failure in result.failed_gates
    )
    assert "evaluator_parity" in result.failed_gates
    assert "cold-cache evidence readback" in result.failed_gates


def test_gate_rejects_deleted_sequence_or_baseline() -> None:
    plan = _plan()
    truncated = SequenceScoresV1(
        _scores(PRIMARY_SEQUENCES, 0.70).sequence_scores[:-1]
    )
    with pytest.raises(ValueError, match="sequence cohort mismatch"):
        _gate(metrics=replace(_metrics(plan), robust_eventtrack=truncated))

    with pytest.raises(ValueError, match="qualified registry"):
        _gate(
            metrics=replace(
                _metrics(plan),
                robust_baselines=_metrics(plan).robust_baselines[:-1],
            )
        )


def test_gate_rejects_missing_pareto_budget_and_wrong_scheduler() -> None:
    plan = _plan()
    invalid_curve = BudgetedSequenceScoresV1(
        scheduler_id="marginal_voi",
        byte_budgets_per_second=BUDGETS[:-1],
        sequence_scores=tuple(
            (sequence_id, (0.8, 0.81, 0.82, 0.83))
            for sequence_id in PRIMARY_SEQUENCES
        ),
        sequence_on_wire_bps=tuple(
            (
                sequence_id,
                tuple(float(budget - 1) for budget in BUDGETS[:-1]),
            )
            for sequence_id in PRIMARY_SEQUENCES
        ),
    )
    with pytest.raises(ValueError, match="five sealed"):
        _gate(metrics=replace(_metrics(plan), pareto_eventtrack=invalid_curve))

    wrong_scheduler = replace(
        _metrics(plan).pareto_strongest_baseline, scheduler_id="fifo_aoi"
    )
    with pytest.raises(ValueError, match="scheduler identities"):
        _gate(
            metrics=replace(
                _metrics(plan), pareto_strongest_baseline=wrong_scheduler
            )
        )


def test_gate_rejects_unbound_metrics_and_plan_evidence_artifacts() -> None:
    plan = _plan()
    original = _metrics(plan)
    evidence = _evidence(plan, original)
    changed = replace(
        original, griffin_eventtrack=_scores(GRIFFIN_SEQUENCES, 0.66)
    )
    with pytest.raises(ValueError, match="not bound by EvidenceBundleV1"):
        _gate(plan=plan, metrics=changed, evidence=evidence)

    bad_artifacts = {
        name: getattr(evidence, name) for name in EvidenceBundleV1._ARTIFACT_FIELDS
    }
    bad_artifacts["dataset"] = ArtifactDigestV1("artifacts/dataset", "f" * 64, 1)
    mismatched = EvidenceBundleV1(run_id="aggregate-result", **bad_artifacts)
    with pytest.raises(ValueError, match="artifact mismatch"):
        _gate(plan=plan, metrics=original, evidence=mismatched)


def test_gate_rejects_stale_invariant_and_cold_cache_bindings() -> None:
    plan = _plan()
    metrics = _metrics(plan)
    evidence = _evidence(plan, metrics)
    stale_invariants = list(_invariants(plan, evidence))
    stale_invariants[0] = replace(
        stale_invariants[0], evidence_bundle_sha256="f" * 64
    )
    with pytest.raises(ValueError, match="stale plan/evidence"):
        _gate(
            plan=plan,
            metrics=metrics,
            evidence=evidence,
            invariants=tuple(stale_invariants),
        )

    stale_receipt = replace(_receipt(plan, evidence), evidence_bundle_sha256="f" * 64)
    with pytest.raises(ValueError, match="stale plan/evidence"):
        _gate(
            plan=plan,
            metrics=metrics,
            evidence=evidence,
            cold_cache_receipt=stale_receipt,
        )


def test_gate_has_no_caller_override_for_statistical_settings() -> None:
    plan = _plan()
    metrics = _metrics(plan)
    evidence = _evidence(plan, metrics)
    with pytest.raises(TypeError, match="unexpected keyword"):
        evaluate_publication_gate_v1(
            experiment_plan=plan,
            evidence_bundle=evidence,
            publication_metrics=metrics,
            invariants=_invariants(plan, evidence),
            cold_cache_receipt=_receipt(plan, evidence),
            resamples=10,
        )


def test_paired_pareto_auc_uses_common_actual_byte_support() -> None:
    treatment = BudgetedSequenceScoresV1(
        scheduler_id="marginal_voi",
        byte_budgets_per_second=BUDGETS,
        sequence_scores=(("sequence-00", (0.50, 0.60, 0.70, 0.80, 0.90)),),
        sequence_on_wire_bps=(("sequence-00", (10.0, 20.0, 30.0, 40.0, 50.0)),),
    )
    control = BudgetedSequenceScoresV1(
        scheduler_id="confidence_top_k",
        byte_budgets_per_second=BUDGETS,
        sequence_scores=(("sequence-00", (0.40, 0.50, 0.60, 0.70, 0.80)),),
        sequence_on_wire_bps=(("sequence-00", (12.0, 22.0, 32.0, 42.0, 52.0)),),
    )

    treatment_auc, control_auc = paired_pareto_auc_by_sequence_v1(
        treatment, control
    )
    assert treatment_auc["sequence-00"] > control_auc["sequence-00"]


def test_pareto_contract_rejects_budget_overrun_and_disjoint_actual_support() -> None:
    with pytest.raises(ValueError, match="exceeds its nominal budget"):
        BudgetedSequenceScoresV1(
            scheduler_id="marginal_voi",
            byte_budgets_per_second=BUDGETS,
            sequence_scores=(("sequence-00", (0.5,) * 5),),
            sequence_on_wire_bps=(
                ("sequence-00", (16_001.0, 20_000.0, 30_000.0, 40_000.0, 50_000.0)),
            ),
        )

    treatment = BudgetedSequenceScoresV1(
        scheduler_id="marginal_voi",
        byte_budgets_per_second=BUDGETS,
        sequence_scores=(("sequence-00", (0.5,) * 5),),
        sequence_on_wire_bps=(("sequence-00", (1.0, 2.0, 3.0, 4.0, 5.0)),),
    )
    control = BudgetedSequenceScoresV1(
        scheduler_id="confidence_top_k",
        byte_budgets_per_second=BUDGETS,
        sequence_scores=(("sequence-00", (0.4,) * 5),),
        sequence_on_wire_bps=(("sequence-00", (6.0, 7.0, 8.0, 9.0, 10.0)),),
    )
    with pytest.raises(ValueError, match="no common actual-BPS width"):
        paired_pareto_auc_by_sequence_v1(treatment, control)


def test_budgeted_metrics_decoder_rejects_non_array_wire_rows() -> None:
    value = _budgeted(PRIMARY_SEQUENCES, "marginal_voi", 0.8).to_primitive()
    value["sequence_on_wire_bps"] = "not-an-array"
    with pytest.raises(ValueError, match="sequence_on_wire_bps must be an array"):
        BudgetedSequenceScoresV1.from_mapping(value)
