from dataclasses import replace
import hashlib
import json

import pytest

from transvision.models.event_track_v2x.contracts import (
    ArtifactDigestV1,
    EvidenceBundleV1,
)
from transvision.models.event_track_v2x.experiment import (
    CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1,
)
from transvision.models.event_track_v2x.qualification import (
    OFFICIAL_TRACKER_METHOD_TO_BACKEND_V1,
    QUALIFIED_SCHEDULER_BASELINE_IDS_V1,
    BaselineQualificationEntryV1,
    BaselineQualificationRegistryV1,
    QualificationRegistryError,
    SchedulerQualificationEntryV1,
    decode_qualification_registry,
    qualification_registry_from_document,
)
from transvision.models.event_track_v2x.wire import canonical_json_bytes


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


SOURCE_SHA = _sha("source")
DATASET_SHA = _sha("dataset-manifest")
CACHE_SHA = _sha("detection-cache")
TRACE_SHA = _sha("network-trace-manifest")
EVALUATOR_SHA = _sha("evaluator-contract")


def _artifact(uri: str, sha256: str) -> ArtifactDigestV1:
    return ArtifactDigestV1(uri=uri, sha256=sha256, byte_size=1)


def _evidence(
    run_id: str, *, config_sha256: str, predictions_sha256: str, metrics_sha256: str
) -> EvidenceBundleV1:
    return EvidenceBundleV1(
        run_id=run_id,
        source=_artifact("source.json", SOURCE_SHA),
        dataset=_artifact("dataset.json", DATASET_SHA),
        detection_cache=_artifact("cache.json", CACHE_SHA),
        network_trace=_artifact("trace.json", TRACE_SHA),
        tracker_config=_artifact("config.json", config_sha256),
        evaluator_contract=_artifact("evaluator.json", EVALUATOR_SHA),
        checkpoint=_artifact("checkpoint.json", _sha(f"{run_id}-checkpoint")),
        predictions=_artifact("predictions.json", predictions_sha256),
        per_sequence_metrics=_artifact("metrics.json", metrics_sha256),
        logs=_artifact("logs.json", _sha(f"{run_id}-logs")),
        environment=_artifact("environment.json", _sha(f"{run_id}-environment")),
    )


_BASELINE_SCORES = {
    "constant-velocity-compensation": (0.71, 0.72, 0.73),
    "fixed-lag-oosm-controlled": (0.73, 0.74, 0.75),
    "naive-async-late-fusion": (0.69, 0.70, 0.71),
    "vehicle-only-ab3dmot": (0.70, 0.71, 0.72),
    "vehicle-only-immortaltracker": (0.69, 0.73, 0.74),
    "vehicle-only-simpletrack": (0.68, 0.75, 0.76),
}


def _baseline(method_id: str) -> BaselineQualificationEntryV1:
    official_backend = OFFICIAL_TRACKER_METHOD_TO_BACKEND_V1.get(method_id)
    tracker_backend_id = official_backend or "ab3dmot"
    qualification_kind = (
        "official-backend"
        if official_backend is not None
        else "shared-selected-backend"
    )
    backend_source_sha256 = _sha(f"{tracker_backend_id}-upstream")
    config_sha256 = _sha(f"{method_id}-config")
    predictions_sha256 = _sha(f"{method_id}-predictions")
    metrics_sha256 = _sha(f"{method_id}-metrics")
    evidence = _evidence(
        f"baseline-{method_id}",
        config_sha256=config_sha256,
        predictions_sha256=predictions_sha256,
        metrics_sha256=metrics_sha256,
    )
    robust, amota, hota = _BASELINE_SCORES[method_id]
    return BaselineQualificationEntryV1(
        method_id=method_id,
        tracker_backend_id=tracker_backend_id,
        qualification_kind=qualification_kind,
        backend_source_sha256=backend_source_sha256,
        config_sha256=config_sha256,
        predictions_sha256=predictions_sha256,
        metrics_sha256=metrics_sha256,
        evidence_bundle_sha256=evidence.digest(),
        evidence_bundle=evidence,
        failure_count=0,
        robust_assa_at_64k=robust,
        clean_amota=amota,
        clean_hota=hota,
        passed=True,
    )


_SCHEDULER_SCORES = {
    "confidence_top_k": (0.69, 0.70),
    "fifo_aoi": (0.68, 0.68),
    "full_send": (0.70, 0.69),
    "periodic": (0.67, 0.67),
    "random": (0.65, 0.65),
}


def _scheduler(scheduler_id: str) -> SchedulerQualificationEntryV1:
    config_sha256 = _sha(f"{scheduler_id}-config")
    predictions_sha256 = _sha(f"{scheduler_id}-predictions")
    metrics_sha256 = _sha(f"{scheduler_id}-metrics")
    evidence = _evidence(
        f"scheduler-{scheduler_id}",
        config_sha256=config_sha256,
        predictions_sha256=predictions_sha256,
        metrics_sha256=metrics_sha256,
    )
    robust, auc = _SCHEDULER_SCORES[scheduler_id]
    return SchedulerQualificationEntryV1(
        scheduler_id=scheduler_id,
        config_sha256=config_sha256,
        predictions_sha256=predictions_sha256,
        metrics_sha256=metrics_sha256,
        evidence_bundle_sha256=evidence.digest(),
        evidence_bundle=evidence,
        failure_count=0,
        robust_assa_at_64k=robust,
        pareto_auc=auc,
        passed=True,
    )


def _registry() -> BaselineQualificationRegistryV1:
    return BaselineQualificationRegistryV1(
        registry_id="eventtrack-v2x-development-qualification-v1",
        source_tree_sha256=SOURCE_SHA,
        dataset_id="v2x-seq-spd",
        dataset_manifest_sha256=DATASET_SHA,
        split_name="train",
        split_sha256=_sha("train-split"),
        cohort_sha256=_sha("train-cohort"),
        sequence_ids=tuple(f"sequence-{index:02d}" for index in range(46)),
        development_fold_manifest_sha256=_sha("development-fold-manifest"),
        frequency_hz=10.0,
        class_names=("car",),
        detection_cache_sha256=CACHE_SHA,
        detection_frame_contract_sha256=_sha("frame-contract"),
        network_trace_manifest_sha256=TRACE_SHA,
        evaluator_contract_sha256=EVALUATOR_SHA,
        training_seeds=(1337, 2027, 3407),
        network_seeds=tuple(range(1001, 1011)),
        primary_budget_bytes_per_second=64_000,
        validation_result_registry_sha256=_sha("validation-result-registry"),
        baseline_entries=tuple(
            _baseline(method_id)
            for method_id in CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1
        ),
        selected_tracker_backend_id="ab3dmot",
        strongest_qualified_baseline_id="fixed-lag-oosm-controlled",
        scheduler_entries=tuple(
            _scheduler(scheduler_id)
            for scheduler_id in QUALIFIED_SCHEDULER_BASELINE_IDS_V1
        ),
        strongest_scheduler_baseline_id="confidence_top_k",
    )


def _reseal(document: dict[str, object]) -> dict[str, object]:
    payload = {key: value for key, value in document.items() if key != "content_sha256"}
    document["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return document


def test_registry_is_canonical_digest_bound_and_round_trips() -> None:
    expected = _registry()
    raw = expected.canonical_bytes()
    observed = decode_qualification_registry(raw)
    assert observed == expected
    assert observed.content_sha256 == expected.content_sha256
    assert observed.digest() == hashlib.sha256(raw).hexdigest()
    assert observed.selected_tracker_backend_id == "ab3dmot"
    assert observed.strongest_qualified_baseline_id == (
        "fixed-lag-oosm-controlled"
    )
    assert observed.strongest_scheduler_baseline_id == "confidence_top_k"


def test_registry_rejects_missing_duplicate_or_unknown_baselines() -> None:
    registry = _registry()
    with pytest.raises(QualificationRegistryError, match="exact six"):
        replace(registry, baseline_entries=registry.baseline_entries[:-1])

    duplicated = (*registry.baseline_entries[:-1], registry.baseline_entries[0])
    with pytest.raises(QualificationRegistryError, match="exact six"):
        replace(registry, baseline_entries=duplicated)

    unknown = replace(registry.baseline_entries[0], method_id="unregistered")
    with pytest.raises(QualificationRegistryError, match="exact six"):
        replace(registry, baseline_entries=(unknown, *registry.baseline_entries[1:]))


def test_registry_rejects_wrong_official_or_shared_backend() -> None:
    registry = _registry()
    ab3_index = next(
        index
        for index, item in enumerate(registry.baseline_entries)
        if item.method_id == "vehicle-only-ab3dmot"
    )
    official = replace(
        registry.baseline_entries[ab3_index], tracker_backend_id="simpletrack"
    )
    entries = list(registry.baseline_entries)
    entries[ab3_index] = official
    with pytest.raises(QualificationRegistryError, match="official backend"):
        replace(registry, baseline_entries=tuple(entries))

    control = replace(registry.baseline_entries[0], tracker_backend_id="simpletrack")
    with pytest.raises(QualificationRegistryError, match="common tracker backend"):
        replace(
            registry,
            baseline_entries=(control, *registry.baseline_entries[1:]),
        )


def test_only_all_passed_zero_failure_entries_can_be_registered() -> None:
    baseline = _registry().baseline_entries[0]
    with pytest.raises(QualificationRegistryError, match="must equal zero"):
        replace(baseline, failure_count=1)
    with pytest.raises(QualificationRegistryError, match="must be true"):
        replace(baseline, passed=False)

    scheduler = _registry().scheduler_entries[0]
    with pytest.raises(QualificationRegistryError, match="must equal zero"):
        replace(scheduler, failure_count=2)
    with pytest.raises(QualificationRegistryError, match="must be true"):
        replace(scheduler, passed=False)


def test_frozen_development_winner_rules_reject_cherry_picked_names() -> None:
    registry = _registry()
    with pytest.raises(QualificationRegistryError, match="selected_tracker"):
        replace(registry, selected_tracker_backend_id="simpletrack")
    with pytest.raises(QualificationRegistryError, match="strongest_qualified"):
        replace(
            registry,
            strongest_qualified_baseline_id="vehicle-only-ab3dmot",
        )
    with pytest.raises(QualificationRegistryError, match="strongest_scheduler"):
        replace(registry, strongest_scheduler_baseline_id="full_send")


def test_registry_requires_all_four_capped_non_voi_schedulers_without_duplicates() -> None:
    registry = _registry()
    with pytest.raises(QualificationRegistryError, match="exact four"):
        replace(registry, scheduler_entries=registry.scheduler_entries[:-1])
    duplicated = (*registry.scheduler_entries[:-1], registry.scheduler_entries[0])
    with pytest.raises(QualificationRegistryError, match="exact four"):
        replace(registry, scheduler_entries=duplicated)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_id", "other", "V2X-Seq-SPD train"),
        ("split_name", "val", "V2X-Seq-SPD train"),
        ("frequency_hz", 2.0, "frequency"),
        ("class_names", ("truck",), "class_names"),
        ("training_seeds", (1,), "training_seeds"),
        ("network_seeds", (1,), "network_seeds"),
    ],
)
def test_registry_rejects_nondevelopment_protocol(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(QualificationRegistryError, match=message):
        replace(_registry(), **{field: value})

    with pytest.raises(QualificationRegistryError, match="46 sequences"):
        replace(_registry(), sequence_ids=_registry().sequence_ids[:-1])
    with pytest.raises(QualificationRegistryError, match="sorted"):
        replace(_registry(), sequence_ids=tuple(reversed(_registry().sequence_ids)))


def test_entry_rejects_evidence_digest_and_artifact_mismatch() -> None:
    entry = _registry().baseline_entries[0]
    with pytest.raises(QualificationRegistryError, match="digest mismatch"):
        replace(entry, evidence_bundle_sha256=_sha("wrong-evidence"))
    with pytest.raises(QualificationRegistryError, match="artifact/Evidence"):
        replace(entry, config_sha256=_sha("wrong-config"))

    wrong_source = replace(
        entry.evidence_bundle,
        source=_artifact("source.json", _sha("wrong-source")),
    )
    changed = replace(
        entry,
        evidence_bundle=wrong_source,
        evidence_bundle_sha256=wrong_source.digest(),
    )
    with pytest.raises(QualificationRegistryError, match="registry/Evidence"):
        replace(
            _registry(),
            baseline_entries=(changed, *_registry().baseline_entries[1:]),
        )


def test_tampering_unknown_fields_duplicates_and_noncanonical_json_are_rejected() -> None:
    registry = _registry()
    document = registry.sealed_document()
    document["frequency_hz"] = 2.0
    with pytest.raises(QualificationRegistryError, match="content SHA-256"):
        qualification_registry_from_document(document)

    document = registry.sealed_document()
    document["unexpected"] = True
    with pytest.raises(QualificationRegistryError, match="fields do not match"):
        qualification_registry_from_document(document)

    document = registry.sealed_document()
    baseline = document["baseline_entries"][0]
    baseline["config_sha256"] = _sha("resealed-wrong-config")
    with pytest.raises(QualificationRegistryError, match="artifact/Evidence"):
        qualification_registry_from_document(_reseal(document))

    noncanonical = json.dumps(registry.sealed_document()).encode("utf-8")
    with pytest.raises(QualificationRegistryError, match="not canonical"):
        decode_qualification_registry(noncanonical)

    raw = registry.canonical_bytes()
    duplicate = b'{"content_sha256":"' + b"0" * 64 + b'",' + raw[1:]
    with pytest.raises(QualificationRegistryError, match="duplicate JSON key"):
        decode_qualification_registry(duplicate)
