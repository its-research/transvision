from dataclasses import replace
import hashlib
import json

import pytest

from transvision.models.event_track_v2x.candidate_selection import (
    CANDIDATE_CLEAN_NONINFERIORITY_MARGIN_V1,
    CANDIDATE_COMBINATIONS_V1,
    CANDIDATE_NETWORK_SEEDS_V1,
    CANDIDATE_PRIMARY_BUDGET_BPS_V1,
    CANDIDATE_TRAINING_SEEDS_V1,
    CandidateSelectionEntryV1,
    CandidateSelectionRegistryError,
    CandidateSelectionRegistryV1,
    candidate_selection_registry_from_document,
    decode_candidate_selection_registry,
)
from transvision.models.event_track_v2x.wire import canonical_json_bytes


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _entry(
    index: int,
    combination: tuple[float, int, int, float],
) -> CandidateSelectionEntryV1:
    lag, mixture, top_h, gate = combination
    values: dict[str, float] = {
        "clean_amota": 0.80,
        "clean_hota": 0.81,
        "robust_assa_at_64k": 0.70,
        "actual_bytes_per_second": 63_000.0,
        "p95_latency_ms": 25.0,
    }
    # These candidates exercise every eligibility and tie-break stage.
    if index == 0:
        values.update(
            robust_assa_at_64k=1.0,
            actual_bytes_per_second=64_000.1,
        )
    elif index == 1:
        values.update(clean_amota=0.789, robust_assa_at_64k=0.99)
    elif index in {2, 3}:
        values.update(
            robust_assa_at_64k=0.90,
            actual_bytes_per_second=62_000.0,
            p95_latency_ms=10.0,
        )
    elif index == 4:
        values.update(
            robust_assa_at_64k=0.90,
            actual_bytes_per_second=62_000.0,
            p95_latency_ms=11.0,
        )
    elif index == 5:
        values.update(
            robust_assa_at_64k=0.90,
            actual_bytes_per_second=63_000.0,
            p95_latency_ms=1.0,
        )
    return CandidateSelectionEntryV1(
        fixed_lag_seconds=lag,
        mixture_components=mixture,
        top_h=top_h,
        gate_confidence=gate,
        config_sha256=_sha(f"config-{index}"),
        oof_checkpoint_inventory_sha256=_sha(
            f"oof-checkpoint-inventory-{index}"
        ),
        evidence_sha256=_sha(f"evidence-{index}"),
        **values,
    )


def _entries() -> tuple[CandidateSelectionEntryV1, ...]:
    return tuple(
        _entry(index, combination)
        for index, combination in enumerate(CANDIDATE_COMBINATIONS_V1)
    )


def _registry(
    *, entries: tuple[CandidateSelectionEntryV1, ...] | None = None
) -> CandidateSelectionRegistryV1:
    result_entries = _entries() if entries is None else entries
    return CandidateSelectionRegistryV1(
        registry_id="eventtrack-v2x-candidate-selection-v1",
        source_tree_sha256=_sha("source"),
        dataset_id="v2x-seq-spd",
        dataset_manifest_sha256=_sha("dataset-manifest"),
        split_name="train",
        split_sha256=_sha("train-split"),
        cohort_sha256=_sha("train-cohort"),
        sequence_ids=tuple(f"sequence-{index:02d}" for index in range(46)),
        development_fold_manifest_sha256=_sha("development-fold-manifest"),
        frequency_hz=10.0,
        class_names=("car",),
        detection_cache_sha256=_sha("detection-cache"),
        detection_frame_contract_sha256=_sha("frame-contract"),
        network_trace_manifest_sha256=_sha("trace-manifest"),
        wire_accounting_config_sha256=_sha("wire-accounting"),
        evaluator_contract_sha256=_sha("evaluator-contract"),
        validation_result_registry_sha256=_sha("validation-raw-registry"),
        baseline_qualification_registry_sha256=_sha(
            "baseline-qualification-registry"
        ),
        strongest_baseline_method_id="fixed-lag-oosm-controlled",
        reference_clean_amota=0.80,
        reference_clean_hota=0.81,
        method_id="eventtrack-v2x",
        scheduler_id="marginal_voi",
        training_seeds=CANDIDATE_TRAINING_SEEDS_V1,
        network_seeds=CANDIDATE_NETWORK_SEEDS_V1,
        primary_budget_bytes_per_second=CANDIDATE_PRIMARY_BUDGET_BPS_V1,
        clean_noninferiority_margin=(
            CANDIDATE_CLEAN_NONINFERIORITY_MARGIN_V1
        ),
        entries=result_entries,
        selected_method_config_sha256=_sha("config-2"),
    )


def _reseal(document: dict[str, object]) -> dict[str, object]:
    payload = {key: value for key, value in document.items() if key != "content_sha256"}
    document["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return document


def test_registry_covers_exact_grid_selects_by_frozen_rule_and_round_trips() -> None:
    expected = _registry()
    assert len(expected.entries) == 4 * 4 * 3 * 3 == 144
    assert tuple(item.combination for item in expected.entries) == (
        CANDIDATE_COMBINATIONS_V1
    )
    assert expected.entries[0].robust_assa_at_64k == 1.0
    assert not expected.entries[0].eligible(
        reference_clean_amota=expected.reference_clean_amota,
        reference_clean_hota=expected.reference_clean_hota,
        clean_noninferiority_margin=expected.clean_noninferiority_margin,
        primary_budget_bytes_per_second=(
            expected.primary_budget_bytes_per_second
        ),
    )
    assert not expected.entries[1].eligible(
        reference_clean_amota=expected.reference_clean_amota,
        reference_clean_hota=expected.reference_clean_hota,
        clean_noninferiority_margin=expected.clean_noninferiority_margin,
        primary_budget_bytes_per_second=(
            expected.primary_budget_bytes_per_second
        ),
    )
    assert expected.selected_entry.combination == CANDIDATE_COMBINATIONS_V1[2]
    assert expected.selected_method_config_sha256 == _sha("config-2")

    raw = expected.canonical_bytes()
    observed = decode_candidate_selection_registry(raw)
    assert observed == expected
    assert observed.content_sha256 == expected.content_sha256
    assert observed.digest() == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "reordered"])
def test_registry_rejects_missing_extra_duplicate_or_noncanonical_grid(
    mutation: str,
) -> None:
    entries = _entries()
    if mutation == "missing":
        changed = entries[:-1]
    elif mutation == "extra":
        changed = (*entries, entries[-1])
    elif mutation == "duplicate":
        changed = (*entries[:-1], entries[0])
    else:
        changed = (entries[1], entries[0], *entries[2:])
    with pytest.raises(CandidateSelectionRegistryError, match="exact 144"):
        _registry(entries=changed)


def test_registry_rejects_duplicate_config_evidence_or_checkpoint_binding() -> None:
    entries = _entries()
    duplicate_config = replace(
        entries[1], config_sha256=entries[0].config_sha256
    )
    with pytest.raises(CandidateSelectionRegistryError, match="distinct config"):
        _registry(entries=(entries[0], duplicate_config, *entries[2:]))

    duplicate_evidence = replace(
        entries[1], evidence_sha256=entries[0].evidence_sha256
    )
    with pytest.raises(CandidateSelectionRegistryError, match="distinct evidence"):
        _registry(entries=(entries[0], duplicate_evidence, *entries[2:]))

    duplicate_checkpoint = replace(
        entries[1],
        oof_checkpoint_inventory_sha256=(
            entries[0].oof_checkpoint_inventory_sha256
        ),
    )
    with pytest.raises(CandidateSelectionRegistryError, match="checkpoint inventory"):
        _registry(entries=(entries[0], duplicate_checkpoint, *entries[2:]))


def test_registry_rejects_wrong_selected_method_config() -> None:
    registry = _registry()
    with pytest.raises(CandidateSelectionRegistryError, match="frozen development rule"):
        replace(
            registry,
            selected_method_config_sha256=registry.entries[3].config_sha256,
        )


def test_registry_fails_closed_when_no_candidate_is_eligible() -> None:
    entries = tuple(
        replace(item, clean_amota=0.0, clean_hota=0.0) for item in _entries()
    )
    with pytest.raises(CandidateSelectionRegistryError, match="no candidate"):
        _registry(entries=entries)


def test_clean_noninferiority_boundary_is_inclusive() -> None:
    entry = replace(_entries()[10], clean_amota=0.79, clean_hota=0.80)
    assert entry.eligible(
        reference_clean_amota=0.80,
        reference_clean_hota=0.81,
        clean_noninferiority_margin=-0.01,
        primary_budget_bytes_per_second=64_000,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("validation_result_registry_sha256", "0" * 63, "SHA-256"),
        ("cohort_sha256", "A" * 64, "SHA-256"),
        ("primary_budget_bytes_per_second", 64_001, "64000"),
        ("clean_noninferiority_margin", -0.02, "-0.01"),
        ("method_id", "candidate", "eventtrack-v2x"),
        ("scheduler_id", "confidence_top_k", "marginal_voi"),
    ],
)
def test_registry_rejects_stale_or_nonfrozen_prerequisites(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(CandidateSelectionRegistryError, match=message):
        replace(_registry(), **{field: value})


def test_entry_rejects_out_of_grid_nonfinite_and_invalid_metrics() -> None:
    entry = _entries()[2]
    with pytest.raises(CandidateSelectionRegistryError, match="frozen V1 grid"):
        replace(entry, fixed_lag_seconds=0.3)
    with pytest.raises(CandidateSelectionRegistryError, match="finite"):
        replace(entry, p95_latency_ms=float("inf"))
    with pytest.raises(CandidateSelectionRegistryError, match=r"\[0, 1\]"):
        replace(entry, robust_assa_at_64k=1.1)
    with pytest.raises(CandidateSelectionRegistryError, match="non-negative"):
        replace(entry, actual_bytes_per_second=-1.0)


def test_document_rejects_tampering_unknown_fields_and_noncanonical_json() -> None:
    registry = _registry()
    document = registry.sealed_document()
    document["cohort_sha256"] = _sha("tampered-cohort")
    with pytest.raises(CandidateSelectionRegistryError, match="content SHA-256"):
        candidate_selection_registry_from_document(document)

    unknown = registry.sealed_document()
    unknown["future_field"] = True
    _reseal(unknown)
    with pytest.raises(CandidateSelectionRegistryError, match="unknown"):
        candidate_selection_registry_from_document(unknown)

    pretty = json.dumps(
        registry.sealed_document(), sort_keys=True, indent=2
    ).encode("utf-8")
    with pytest.raises(CandidateSelectionRegistryError, match="not canonical JSON"):
        decode_candidate_selection_registry(pretty)


def test_decoder_rejects_duplicate_keys_nonfinite_and_nonbytes() -> None:
    with pytest.raises(CandidateSelectionRegistryError, match="duplicate JSON key"):
        decode_candidate_selection_registry(
            b'{"registry_id":"one","registry_id":"two"}'
        )
    with pytest.raises(CandidateSelectionRegistryError, match="non-finite"):
        decode_candidate_selection_registry(b'{"value":NaN}')
    with pytest.raises(TypeError, match="must be bytes"):
        decode_candidate_selection_registry("{}")  # type: ignore[arg-type]

    document = _registry().sealed_document()
    document["schema_version"] = True
    _reseal(document)
    with pytest.raises(CandidateSelectionRegistryError, match="schema version"):
        candidate_selection_registry_from_document(document)
