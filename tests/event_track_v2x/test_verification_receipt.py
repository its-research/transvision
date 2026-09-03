from dataclasses import replace
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from transvision.models.event_track_v2x.publication_gate import (
    REQUIRED_INVARIANTS_V1,
)
from transvision.models.event_track_v2x.results_registry import (
    RESULT_METRICS_ARTIFACT_ROLE_V1,
    RUN_RESULT_STATUS_SUCCESS_V1,
    RunResultCellV1,
    RunResultRegistryV1,
)
from transvision.models.event_track_v2x.verification_receipt import (
    VerificationReceiptError,
    decode_trusted_verification_receipt,
    run_evidence_inventory_sha256,
    sign_trusted_verification_receipt_v1,
    verify_trusted_verification_receipt_v1,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _registry(*, metric_value: float = 0.5) -> RunResultRegistryV1:
    plan_sha = _sha("plan")
    cell = RunResultCellV1(
        experiment_plan_sha256=plan_sha,
        attempt_id="attempt-01",
        detection_cache_sha256=_sha("detection-cache"),
        evaluator_contract_sha256=_sha("evaluator-contract"),
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
        metric_values=(("HOTA", metric_value),),
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
        experiment_plan_sha256=plan_sha,
        attempt_manifest_sha256=_sha("attempt-manifest"),
        confirmatory_val_unsealed_at_utc="2026-09-02T08:01:00Z",
        execution_started_at_utc="2026-09-02T08:02:00Z",
        first_prediction_at_utc="2026-09-02T08:03:00Z",
        completed_at_utc="2026-09-02T10:00:00Z",
        network_trace_manifest_sha256=_sha("trace-manifest"),
        wire_accounting_config_sha256=_sha("wire-config"),
        heldout_c9_trace_receipt_sha256=_sha("c9-receipt"),
        c9_trace_ids=tuple(f"trace-{index:02d}" for index in range(20)),
        cells=(cell,),
    )


def _receipt(registry: RunResultRegistryV1):
    private_key = Ed25519PrivateKey.generate()
    receipt = sign_trusted_verification_receipt_v1(
        private_key=private_key,
        provider_id="independent-verifier",
        log_entry_id="verification-0001",
        verified_at_utc="2026-09-02T12:00:00Z",
        experiment_plan_sha256=registry.experiment_plan_sha256,
        run_result_registry_sha256=registry.digest(),
        aggregate_evidence_bundle_sha256=_sha("aggregate-bundle"),
        qualification_registry_sha256=_sha("qualification"),
        validation_result_registry_sha256=_sha("validation-registry"),
        development_dataset_release_receipt_sha256=_sha(
            "development-release"
        ),
        primary_dataset_release_receipt_sha256=_sha("primary-release"),
        griffin_dataset_release_receipt_sha256=_sha("griffin-release"),
        measured_trace_receipt_sha256=_sha("measured-trace"),
        run_evidence_inventory_sha256=run_evidence_inventory_sha256(registry),
        attempt_manifest_sha256=registry.attempt_manifest_sha256,
        independent_verification_report_sha256=_sha("verification-report"),
        invariant_report_sha256=_sha("invariants"),
        evaluator_recompute_report_sha256=_sha("evaluator"),
        wire_ledger_replay_report_sha256=_sha("wire-replay"),
        cold_cache_inventory_sha256=_sha("cold-cache"),
        invariant_ids=REQUIRED_INVARIANTS_V1,
        cold_cache_verified=True,
        dataset_release_identity_and_license_verified=True,
        measured_trace_semantics_verified=True,
        validation_evidence_semantics_verified=True,
        run_evidence_semantics_verified=True,
        execution_chronology_verified=True,
        evaluator_outputs_recomputed=True,
        wire_ledgers_replayed=True,
        scientific_claims_allowed=True,
    )
    return receipt


def test_signed_verification_receipt_round_trips_and_uses_trust_anchor() -> None:
    registry = _registry()
    receipt = _receipt(registry)
    decoded = decode_trusted_verification_receipt(receipt.canonical_bytes)

    assert decoded == receipt
    verify_trusted_verification_receipt_v1(
        decoded, trusted_public_key_sha256=receipt.public_key_sha256
    )
    with pytest.raises(VerificationReceiptError, match="not the trusted key"):
        verify_trusted_verification_receipt_v1(
            decoded, trusted_public_key_sha256="0" * 64
        )


def test_signed_payload_tampering_is_rejected() -> None:
    receipt = _receipt(_registry())
    tampered = replace(receipt, invariant_report_sha256=_sha("tampered"))

    with pytest.raises(VerificationReceiptError, match="invalid.*signature"):
        verify_trusted_verification_receipt_v1(
            tampered, trusted_public_key_sha256=receipt.public_key_sha256
        )


def test_run_evidence_inventory_binds_cell_semantics() -> None:
    original = _registry(metric_value=0.5)
    changed = _registry(metric_value=0.6)

    # Metric bytes are represented by the metrics artifact digest.  Merely changing
    # an unbound in-memory score demonstrates why the independent verifier must
    # re-hash and parse that artifact before signing the inventory.
    assert run_evidence_inventory_sha256(original) == run_evidence_inventory_sha256(
        changed
    )
    changed_cell = replace(
        changed.cells[0], metrics_artifact_sha256=_sha("changed-metrics")
    )
    changed_registry = replace(changed, cells=(changed_cell,))
    assert run_evidence_inventory_sha256(
        original
    ) != run_evidence_inventory_sha256(changed_registry)


def test_receipt_requires_every_invariant_and_positive_attestation() -> None:
    receipt = _receipt(_registry())
    with pytest.raises(VerificationReceiptError, match="every preregistered"):
        replace(receipt, invariant_ids=REQUIRED_INVARIANTS_V1[:-1])
    with pytest.raises(VerificationReceiptError, match="must be true"):
        replace(receipt, evaluator_outputs_recomputed=False)


def test_receipt_rejects_invalid_attempt_manifest_digest() -> None:
    receipt = _receipt(_registry())

    with pytest.raises(VerificationReceiptError, match="attempt_manifest_sha256"):
        replace(receipt, attempt_manifest_sha256="not-a-sha256")
