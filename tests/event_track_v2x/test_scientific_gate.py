from types import SimpleNamespace

import pytest

from transvision.models.event_track_v2x.scientific_gate import (
    ScientificGateError,
    _require_scientific_gate_capabilities_v1,
    _validate_c9_trace_bindings,
    _validate_execution_chronology,
    _validate_frozen_trust_policy,
    _validate_independent_condition_input_inventories,
    _validate_independent_spd_sequence_cohorts,
    _validate_independent_trust_anchors,
)


def test_scientific_gate_is_source_locked_until_external_verifiers_exist() -> None:
    with pytest.raises(ScientificGateError, match="capability-locked") as exc:
        _require_scientific_gate_capabilities_v1()
    message = str(exc.value)
    assert "governance-root-trust-chain" in message
    assert "artifact-byte-cold-read-and-evaluator-replay" in message
    assert "authenticated-attempt-log-completeness" in message
    assert "candidate-oof-raw-cell-replay" in message
    assert "checkpoint-bytes-loaded-per-attempt" in message


def test_scientific_gate_requires_distinct_frozen_roles_and_keys() -> None:
    preregistration = SimpleNamespace(
        provider_id="institutional-log", public_key_sha256="1" * 64
    )
    verifier = SimpleNamespace(
        provider_id="independent-verifier", public_key_sha256="2" * 64
    )
    _validate_independent_trust_anchors(preregistration, verifier)

    with pytest.raises(ScientificGateError, match="distinct providers"):
        _validate_independent_trust_anchors(
            preregistration,
            SimpleNamespace(
                provider_id="institutional-log", public_key_sha256="2" * 64
            ),
        )

    plan = SimpleNamespace(
        preregistration_provider_id="institutional-log",
        preregistration_public_key_sha256="1" * 64,
        verification_provider_id="independent-verifier",
        verification_public_key_sha256="2" * 64,
    )
    _validate_frozen_trust_policy(plan, preregistration, verifier)
    with pytest.raises(ScientificGateError, match="frozen before val"):
        _validate_frozen_trust_policy(
            plan,
            preregistration,
            SimpleNamespace(
                provider_id="independent-verifier", public_key_sha256="3" * 64
            ),
        )
    with pytest.raises(ScientificGateError, match="distinct keys"):
        _validate_independent_trust_anchors(
            preregistration,
            SimpleNamespace(
                provider_id="independent-verifier", public_key_sha256="1" * 64
            ),
        )


def test_scientific_gate_binds_each_c9_cell_to_packet_metadata() -> None:
    receipt = SimpleNamespace(
        held_out_packet_metadata_sha256s=(("heldout-00", "a" * 64),)
    )
    good = SimpleNamespace(
        condition_id="C9",
        c9_trace_id="heldout-00",
        channel_outcome_trace_sha256="a" * 64,
    )
    _validate_c9_trace_bindings(SimpleNamespace(cells=(good,)), receipt)

    bad = SimpleNamespace(
        condition_id="C9",
        c9_trace_id="heldout-00",
        channel_outcome_trace_sha256="b" * 64,
    )
    with pytest.raises(ScientificGateError, match="packet metadata"):
        _validate_c9_trace_bindings(SimpleNamespace(cells=(bad,)), receipt)


def test_scientific_gate_keeps_development_and_confirmatory_inputs_independent() -> None:
    confirmatory_inventory = {"a" * 64: object()}
    development_inventory = {"b" * 64: object()}
    _validate_independent_condition_input_inventories(
        confirmatory_inventory, development_inventory
    )

    with pytest.raises(ScientificGateError, match="must be independent"):
        _validate_independent_condition_input_inventories(
            confirmatory_inventory, confirmatory_inventory
        )
    with pytest.raises(ScientificGateError, match="must be independent"):
        _validate_independent_condition_input_inventories(
            confirmatory_inventory,
            {"a" * 64: object(), "b" * 64: object()},
        )

    development_receipt = SimpleNamespace(sequence_ids=("train-00", "train-01"))
    confirmatory_receipt = SimpleNamespace(sequence_ids=("val-00", "val-01"))
    _validate_independent_spd_sequence_cohorts(
        development_receipt, confirmatory_receipt
    )
    with pytest.raises(ScientificGateError, match="sequence cohorts overlap"):
        _validate_independent_spd_sequence_cohorts(
            development_receipt,
            SimpleNamespace(sequence_ids=("train-01", "val-00")),
        )


def test_scientific_gate_orders_registration_unseal_run_and_verification() -> None:
    preregistration = SimpleNamespace(registered_at_utc="2026-09-02T08:00:00Z")
    registry = SimpleNamespace(
        confirmatory_val_unsealed_at_utc="2026-09-02T08:01:00Z",
        completed_at_utc="2026-09-02T10:00:00Z",
    )
    verification = SimpleNamespace(verified_at_utc="2026-09-02T10:01:00Z")
    _validate_execution_chronology(preregistration, registry, verification)

    with pytest.raises(ScientificGateError, match="remain sealed"):
        _validate_execution_chronology(
            preregistration,
            SimpleNamespace(
                confirmatory_val_unsealed_at_utc="2026-09-02T07:59:00Z",
                completed_at_utc="2026-09-02T10:00:00Z",
            ),
            verification,
        )
    with pytest.raises(ScientificGateError, match="after confirmatory completion"):
        _validate_execution_chronology(
            preregistration,
            registry,
            SimpleNamespace(verified_at_utc="2026-09-02T09:59:00Z"),
        )
