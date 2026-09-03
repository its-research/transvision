from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from transvision.models.event_track_v2x.attempt_manifest import (
    ATTEMPT_STATUS_INFRASTRUCTURE_FAILURE_V1,
    ATTEMPT_STATUS_SUCCESS_V1,
    AttemptManifestError,
    ConfirmatoryAttemptManifestV1,
    ConfirmatoryAttemptRecordV1,
    decode_confirmatory_attempt_manifest,
    physical_result_cell_sha256_v1,
    validate_confirmatory_attempt_manifest_v1,
)
from transvision.models.event_track_v2x.results_registry import (
    RESULT_METRICS_ARTIFACT_ROLE_V1,
    RUN_RESULT_STATUS_SUCCESS_V1,
    RunResultCellV1,
    RunResultRegistryV1,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _registry(*, plan_sha256: str, manifest_sha256: str) -> RunResultRegistryV1:
    cell = RunResultCellV1(
        experiment_plan_sha256=plan_sha256,
        detection_cache_sha256=_sha("cache"),
        evaluator_contract_sha256=_sha("evaluator"),
        attempt_id="attempt-02",
        scenario_id="primary_clean",
        dataset_domain="primary",
        method_id="eventtrack-v2x",
        sequence_id="sequence-00",
        condition_id="C0",
        training_seed=1337,
        network_seed=None,
        c9_trace_id=None,
        budget_bytes_per_second=64_000,
        scheduler_id="marginal_voi",
        metric_values=(("HOTA", 0.5),),
        on_wire_bytes_total=1,
        wire_measurement_duration_seconds=1.0,
        on_wire_bytes_per_second=1.0,
        status=RUN_RESULT_STATUS_SUCCESS_V1,
        failure_code=None,
        checkpoint_sha256=_sha("final-refit-1337"),
        evidence_bundle_sha256=_sha("bundle"),
        metrics_artifact_role=RESULT_METRICS_ARTIFACT_ROLE_V1,
        metrics_artifact_sha256=_sha("metrics"),
        channel_outcome_trace_sha256=_sha("channel"),
        network_trace_sha256=_sha("network"),
        condition_input_manifest_sha256=None,
        wire_ledger_sha256=_sha("wire"),
    )
    return RunResultRegistryV1(
        experiment_plan_sha256=plan_sha256,
        attempt_manifest_sha256=manifest_sha256,
        confirmatory_val_unsealed_at_utc="2026-09-02T08:01:00Z",
        execution_started_at_utc="2026-09-02T08:02:00Z",
        first_prediction_at_utc="2026-09-02T08:05:00Z",
        completed_at_utc="2026-09-02T10:00:00Z",
        network_trace_manifest_sha256=_sha("traces"),
        wire_accounting_config_sha256=_sha("wire-config"),
        heldout_c9_trace_receipt_sha256=_sha("c9"),
        c9_trace_ids=tuple(f"trace-{index:02d}" for index in range(20)),
        cells=(cell,),
    )


def _attempt(
    cell: RunResultCellV1,
    *,
    attempt_index: int,
    attempt_id: str,
    rerun_of_attempt_id: str | None,
    status: str,
) -> ConfirmatoryAttemptRecordV1:
    failed = status == ATTEMPT_STATUS_INFRASTRUCTURE_FAILURE_V1
    return ConfirmatoryAttemptRecordV1(
        dataset_domain=cell.dataset_domain,
        method_id=cell.method_id,
        sequence_id=cell.sequence_id,
        condition_id=cell.condition_id,
        training_seed=cell.training_seed,
        network_seed=cell.network_seed,
        c9_trace_id=cell.c9_trace_id,
        budget_bytes_per_second=cell.budget_bytes_per_second,
        scheduler_id=cell.scheduler_id,
        attempt_index=attempt_index,
        attempt_id=attempt_id,
        rerun_of_attempt_id=rerun_of_attempt_id,
        job_id=f"job-{attempt_index:02d}",
        started_at_utc=(
            "2026-09-02T08:02:00Z" if failed else "2026-09-02T08:04:00Z"
        ),
        first_prediction_at_utc=(
            None if failed else "2026-09-02T08:05:00Z"
        ),
        completed_at_utc=(
            "2026-09-02T08:03:00Z" if failed else "2026-09-02T10:00:00Z"
        ),
        terminal_status=status,
        prediction_emitted=not failed,
        parameters_sha256=_sha("method-config"),
        checkpoint_sha256=cell.checkpoint_sha256,
        environment_sha256=_sha("environment"),
        result_cell_sha256=(
            None if failed else physical_result_cell_sha256_v1(cell)
        ),
        evidence_bundle_sha256=(None if failed else cell.evidence_bundle_sha256),
        incident_evidence_sha256=(_sha("incident") if failed else None),
    )


def _bound_manifest_and_registry():
    plan_sha256 = _sha("plan")
    draft = _registry(plan_sha256=plan_sha256, manifest_sha256="0" * 64)
    cell = draft.cells[0]
    attempts = (
        _attempt(
            cell,
            attempt_index=1,
            attempt_id="attempt-01",
            rerun_of_attempt_id=None,
            status=ATTEMPT_STATUS_INFRASTRUCTURE_FAILURE_V1,
        ),
        _attempt(
            cell,
            attempt_index=2,
            attempt_id="attempt-02",
            rerun_of_attempt_id="attempt-01",
            status=ATTEMPT_STATUS_SUCCESS_V1,
        ),
    )
    manifest = ConfirmatoryAttemptManifestV1(
        experiment_plan_sha256=plan_sha256,
        confirmatory_val_unsealed_at_utc=(
            draft.confirmatory_val_unsealed_at_utc
        ),
        attempt_inventory_closed_at_utc="2026-09-02T10:01:00Z",
        orchestrator_log_sha256=_sha("orchestrator-log"),
        execution_environment_sha256=_sha("environment"),
        attempts=attempts,
    )
    registry = replace(draft, attempt_manifest_sha256=manifest.digest())
    plan = SimpleNamespace(
        content_sha256=plan_sha256,
        run_config_sha256=lambda dataset_domain, method_id, scheduler_id: _sha(
            "method-config"
        ),
        run_checkpoint_sha256=(
            lambda dataset_domain, method_id, scheduler_id, training_seed: _sha(
                f"final-refit-{training_seed}"
            )
        ),
    )
    return plan, manifest, registry


def test_attempt_manifest_round_trip_and_exact_result_binding() -> None:
    plan, manifest, registry = _bound_manifest_and_registry()
    validate_confirmatory_attempt_manifest_v1(
        manifest, experiment_plan=plan, run_result_registry=registry
    )
    assert decode_confirmatory_attempt_manifest(manifest.canonical_bytes) == manifest


def test_only_pre_prediction_infrastructure_failure_is_rerunnable() -> None:
    _, manifest, _ = _bound_manifest_and_registry()
    with pytest.raises(AttemptManifestError, match="must emit no prediction"):
        replace(
            manifest.attempts[0],
            prediction_emitted=True,
            first_prediction_at_utc="2026-09-02T08:02:30Z",
        )
    with pytest.raises(AttemptManifestError, match="only a pre-prediction"):
        replace(
            manifest,
            attempts=(
                replace(
                    manifest.attempts[0],
                    terminal_status=ATTEMPT_STATUS_SUCCESS_V1,
                    prediction_emitted=True,
                    first_prediction_at_utc="2026-09-02T08:02:30Z",
                    result_cell_sha256=_sha("result"),
                    evidence_bundle_sha256=_sha("bundle"),
                    incident_evidence_sha256=None,
                ),
                manifest.attempts[1],
            ),
        )


def test_manifest_detects_missing_run_and_result_attempt_mismatch() -> None:
    plan, manifest, registry = _bound_manifest_and_registry()
    with pytest.raises(AttemptManifestError, match="terminal attempt"):
        validate_confirmatory_attempt_manifest_v1(
            manifest,
            experiment_plan=plan,
            run_result_registry=replace(
                registry,
                cells=(replace(registry.cells[0], attempt_id="attempt-unknown"),),
            ),
        )


def test_attempt_parameters_bind_to_domain_method_and_scheduler_config() -> None:
    plan, manifest, registry = _bound_manifest_and_registry()
    wrong = replace(
        manifest,
        attempts=tuple(
            replace(item, parameters_sha256=_sha("wrong-config"))
            for item in manifest.attempts
        ),
    )
    registry = replace(registry, attempt_manifest_sha256=wrong.digest())
    with pytest.raises(AttemptManifestError, match="parameters"):
        validate_confirmatory_attempt_manifest_v1(
            wrong, experiment_plan=plan, run_result_registry=registry
        )
    wrong_checkpoint = replace(
        manifest,
        attempts=tuple(
            replace(item, checkpoint_sha256=_sha("wrong-checkpoint"))
            for item in manifest.attempts
        ),
    )
    with pytest.raises(AttemptManifestError, match="checkpoint"):
        validate_confirmatory_attempt_manifest_v1(
            wrong_checkpoint,
            experiment_plan=plan,
            run_result_registry=replace(
                registry,
                attempt_manifest_sha256=wrong_checkpoint.digest(),
            ),
        )
    changed_manifest = replace(
        manifest,
        attempts=(
            manifest.attempts[0],
            replace(manifest.attempts[1], result_cell_sha256=_sha("wrong")),
        ),
    )
    with pytest.raises(AttemptManifestError, match="result digest"):
        validate_confirmatory_attempt_manifest_v1(
            changed_manifest,
            experiment_plan=plan,
            run_result_registry=replace(
                registry, attempt_manifest_sha256=changed_manifest.digest()
            ),
        )


def test_attempt_manifest_rejects_noncanonical_json() -> None:
    data = json.dumps({"kind": "confirmatory_attempt_manifest_v1"}).encode()
    with pytest.raises(AttemptManifestError):
        decode_confirmatory_attempt_manifest(data)
