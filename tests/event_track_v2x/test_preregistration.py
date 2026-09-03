from dataclasses import replace
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from transvision.models.event_track_v2x.experiment import (
    CONFIRMATORY_METHOD_IDS_V1,
    CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1,
    GRIFFIN_25M_VAL_SEQUENCE_IDS_V1,
    ExperimentPlanV1,
    RunConfigBindingV1,
    confirmatory_run_config_keys_v1,
)
from transvision.models.event_track_v2x.preregistration import (
    PreregistrationError,
    decode_preregistration_receipt,
    sign_preregistration_receipt_v1,
    verify_preregistration_receipt_v1,
)


def plan() -> ExperimentPlanV1:
    run_configs = tuple(
        RunConfigBindingV1(
            dataset_domain=domain,
            method_id=method,
            scheduler_id=scheduler,
            config_sha256="f" * 64,
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
        source_tree_sha256="e" * 64,
        method_config_sha256="f" * 64,
        run_config_bindings=run_configs,
        dataset_id="v2x-seq-spd",
        dataset_manifest_sha256="a" * 64,
        split_name="val",
        split_sha256="b" * 64,
        primary_cohort_sha256="7" * 64,
        primary_frame_contract_sha256="8" * 64,
        primary_dataset_release_receipt_sha256="9" * 64,
        development_dataset_release_receipt_sha256="4" * 64,
        development_fold_manifest_sha256="5" * 64,
        primary_sequence_ids=tuple(
            f"sequence-{index:02d}" for index in range(21)
        ),
        detector_cache_sha256="c" * 64,
        network_trace_manifest_sha256="1" * 64,
        wire_accounting_config_sha256="0" * 64,
        heldout_c9_trace_receipt_sha256="2" * 64,
        c9_trace_ids=tuple(f"heldout-{index:02d}" for index in range(20)),
        evaluator_contract_sha256="d" * 64,
        baseline_qualification_registry_sha256="1" * 64,
        candidate_selection_registry_sha256="2" * 64,
        preregistration_provider_id="institutional-log",
        preregistration_public_key_sha256="6" * 64,
        verification_provider_id="independent-verifier",
        verification_public_key_sha256="7" * 64,
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
        griffin_dataset_manifest_sha256="3" * 64,
        griffin_split_name="val",
        griffin_cohort_sha256="4" * 64,
        griffin_frame_contract_sha256="7" * 64,
        griffin_dataset_release_receipt_sha256="8" * 64,
        griffin_sequence_ids=GRIFFIN_25M_VAL_SEQUENCE_IDS_V1,
        griffin_detector_cache_sha256="5" * 64,
        griffin_evaluator_contract_sha256="6" * 64,
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


def _key_digest(key: Ed25519PrivateKey) -> str:
    raw = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()


def test_preregistration_receipt_round_trip_and_trusted_signature() -> None:
    key = Ed25519PrivateKey.generate()
    receipt = sign_preregistration_receipt_v1(
        experiment_plan=plan(),
        provider_id="institutional-transparency-log",
        log_entry_id="entry-0001",
        registered_at_utc="2026-09-02T08:00:00Z",
        private_key=key,
    )
    decoded = decode_preregistration_receipt(receipt.canonical_bytes)
    verify_preregistration_receipt_v1(
        decoded,
        experiment_plan=plan(),
        trusted_public_key_sha256=_key_digest(key),
    )


def test_preregistration_rejects_untrusted_or_tampered_receipt() -> None:
    key = Ed25519PrivateKey.generate()
    receipt = sign_preregistration_receipt_v1(
        experiment_plan=plan(),
        provider_id="institutional-transparency-log",
        log_entry_id="entry-0001",
        registered_at_utc="2026-09-02T08:00:00Z",
        private_key=key,
    )
    with pytest.raises(PreregistrationError, match="trusted key"):
        verify_preregistration_receipt_v1(
            receipt,
            experiment_plan=plan(),
            trusted_public_key_sha256="f" * 64,
        )
    tampered = replace(receipt, log_entry_id="entry-0002")
    with pytest.raises(PreregistrationError, match="signature"):
        verify_preregistration_receipt_v1(
            tampered,
            experiment_plan=plan(),
            trusted_public_key_sha256=_key_digest(key),
        )
