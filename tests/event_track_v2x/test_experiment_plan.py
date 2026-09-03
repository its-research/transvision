from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from transvision.models.event_track_v2x.experiment import (
    CONFIRMATORY_METHOD_IDS_V1,
    CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1,
    GRIFFIN_25M_VAL_SEQUENCE_IDS_V1,
    ExperimentPlanError,
    ExperimentPlanV1,
    RunConfigBindingV1,
    confirmatory_run_config_keys_v1,
    decode_plan,
    plan_from_document,
)
from tools.event_track_v2x.build_experiment_plan import build


def _run_config_bindings(digit: str = "f") -> tuple[RunConfigBindingV1, ...]:
    return tuple(
        RunConfigBindingV1(
            dataset_domain=domain,
            method_id=method,
            scheduler_id=scheduler,
            config_sha256=digit * 64,
            checkpoint_sha256s=tuple(
                hashlib.sha256(
                    f"{digit}:{domain}:{method}:{scheduler}:{seed}".encode()
                ).hexdigest()
                for seed in (1337, 2027, 3407)
            ),
        )
        for domain, method, scheduler in confirmatory_run_config_keys_v1(
            strongest_baseline_id="vehicle-only-ab3dmot",
            strongest_scheduler_id="confidence_top_k",
        )
    )


def plan() -> ExperimentPlanV1:
    return ExperimentPlanV1(
        plan_id="eventtrack-v2x-confirmatory-v1",
        source_tree_sha256="e" * 64,
        method_config_sha256="f" * 64,
        run_config_bindings=_run_config_bindings(),
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


def test_experiment_plan_is_canonical_sealed_and_round_trips() -> None:
    expected = plan()
    data = expected.canonical_bytes()
    observed = decode_plan(data)
    assert observed == expected
    assert observed.content_sha256 == expected.content_sha256
    assert json.loads(data)["schema_version"] == 1


def test_experiment_plan_rejects_tampering_unknown_fields_and_noncanonical_json() -> None:
    document = plan().sealed_document()
    document["frequency_hz"] = 2.0
    with pytest.raises(ExperimentPlanError, match="SHA-256 mismatch"):
        plan_from_document(document)

    document = plan().sealed_document()
    document["unexpected"] = True
    with pytest.raises(ExperimentPlanError, match="missing or unknown"):
        plan_from_document(document)

    noncanonical = json.dumps(plan().sealed_document()).encode("utf-8")
    with pytest.raises(ExperimentPlanError, match="not canonical"):
        decode_plan(noncanonical)


def test_experiment_plan_rejects_invalid_primary_budget_and_duplicate_methods() -> None:
    values = plan().payload()
    values.pop("schema_version")
    values["primary_budget_bytes_per_second"] = 999
    with pytest.raises(ExperimentPlanError, match="primary budget"):
        ExperimentPlanV1(**values)

    values = plan().payload()
    values.pop("schema_version")
    values["method_ids"] = ["same", "same"]
    with pytest.raises(ExperimentPlanError, match="unique"):
        ExperimentPlanV1(**values)


def test_experiment_plan_rejects_a_separately_selected_griffin_baseline() -> None:
    values = plan().payload()
    values.pop("schema_version")
    values["griffin_strongest_baseline_id"] = "vehicle-only-simpletrack"
    with pytest.raises(ExperimentPlanError, match="reuse the strongest baseline"):
        ExperimentPlanV1(**values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("verification_provider_id", "institutional-log", "providers"),
        ("verification_public_key_sha256", "6" * 64, "public keys"),
    ),
)
def test_experiment_plan_freezes_distinct_trust_anchors(
    field: str, value: str, message: str
) -> None:
    values = plan().payload()
    values.pop("schema_version")
    values[field] = value
    with pytest.raises(ExperimentPlanError, match=message):
        ExperimentPlanV1(**values)


def test_experiment_plan_requires_exact_per_run_config_map() -> None:
    values = plan().payload()
    values.pop("schema_version")
    values["run_config_bindings"] = values["run_config_bindings"][:-1]
    with pytest.raises(ExperimentPlanError, match="exactly cover"):
        ExperimentPlanV1(**values)

    values = plan().payload()
    values.pop("schema_version")
    values["run_config_bindings"].append(values["run_config_bindings"][0])
    with pytest.raises(ExperimentPlanError, match="unique"):
        ExperimentPlanV1(**values)

def test_experiment_plan_rejects_unsealed_statistics_budgets_and_cohorts() -> None:
    values = plan().payload()
    values.pop("schema_version")
    values["statistical_resamples"] = 9999
    with pytest.raises(ExperimentPlanError, match="10000"):
        ExperimentPlanV1(**values)

    values = plan().payload()
    values.pop("schema_version")
    values["byte_budgets_per_second"] = [16000, 64000, 256000]
    with pytest.raises(ExperimentPlanError, match="five preregistered"):
        ExperimentPlanV1(**values)

    values = plan().payload()
    values.pop("schema_version")
    values["primary_sequence_ids"] = list(reversed(plan().primary_sequence_ids))
    with pytest.raises(ExperimentPlanError, match="sorted"):
        ExperimentPlanV1(**values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split_name", "train", "val"),
        ("frequency_hz", 2.0, "frequency"),
        ("class_names", ["banana"], "class_names"),
        ("network_condition_ids", ["C0"], "C0 through C9"),
        ("training_seeds", [1], "training seeds"),
        ("network_seeds", [1], "network seeds"),
    ],
)
def test_confirmatory_plan_rejects_validation_or_toy_protocol(
    field: str, value: object, message: str
) -> None:
    values = plan().payload()
    values.pop("schema_version")
    values[field] = value
    with pytest.raises(ExperimentPlanError, match=message):
        ExperimentPlanV1(**values)

    values = plan().payload()
    values.pop("schema_version")
    values["primary_sequence_ids"] = list(plan().primary_sequence_ids[:-1])
    with pytest.raises(ExperimentPlanError, match="21 sequences"):
        ExperimentPlanV1(**values)


def test_build_tool_binds_both_frozen_cohorts_and_receipts(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    griffin = tmp_path / "griffin.json"
    c9 = tmp_path / "c9.json"
    run_configs = tmp_path / "run-configs.json"
    primary.write_text(
        json.dumps([f"sequence-{index:02d}" for index in range(21)]),
        encoding="utf-8",
    )
    griffin.write_text(
        json.dumps(list(GRIFFIN_25M_VAL_SEQUENCE_IDS_V1)), encoding="utf-8"
    )
    c9.write_text(
        json.dumps([f"heldout-{index:02d}" for index in range(20)]),
        encoding="utf-8",
    )
    run_configs.write_text(
        json.dumps([item.to_primitive() for item in _run_config_bindings("2")]),
        encoding="utf-8",
    )
    built = build(
        Namespace(
            source_tree_sha256="1" * 64,
            method_config_sha256="2" * 64,
            run_config_bindings_file=run_configs,
            dataset_id="v2x-seq-spd",
            dataset_manifest_sha256="3" * 64,
            split="val",
            split_sha256="4" * 64,
            primary_cohort_sha256="d" * 64,
            primary_frame_contract_sha256="e" * 64,
            primary_dataset_release_receipt_sha256="f" * 64,
            development_dataset_release_receipt_sha256="1" * 64,
            development_fold_manifest_sha256="2" * 64,
            primary_sequence_ids_file=primary,
            detector_cache_sha256="5" * 64,
            network_trace_manifest_sha256="6" * 64,
            wire_accounting_config_sha256="0" * 64,
            heldout_c9_trace_receipt_sha256="7" * 64,
            c9_trace_ids_file=c9,
            evaluator_contract_sha256="8" * 64,
            baseline_qualification_registry_sha256="d" * 64,
            candidate_selection_registry_sha256="e" * 64,
            preregistration_provider_id="institutional-log",
            preregistration_public_key_sha256="6" * 64,
            verification_provider_id="independent-verifier",
            verification_public_key_sha256="7" * 64,
            qualified_baseline_id=list(
                CONFIRMATORY_QUALIFIED_BASELINE_IDS_V1
            ),
            strongest_qualified_baseline_id="vehicle-only-ab3dmot",
            strongest_scheduler_baseline_id="confidence_top_k",
            scheduler_token_bucket_burst_seconds=1.0,
            griffin_dataset_id="griffin-25m",
            griffin_dataset_manifest_sha256="9" * 64,
            griffin_split="val",
            griffin_cohort_sha256="a" * 64,
            griffin_frame_contract_sha256="d" * 64,
            griffin_dataset_release_receipt_sha256="e" * 64,
            griffin_sequence_ids_file=griffin,
            griffin_detector_cache_sha256="b" * 64,
            griffin_evaluator_contract_sha256="c" * 64,
            griffin_strongest_baseline_id="vehicle-only-ab3dmot",
        )
    )
    assert len(built.primary_sequence_ids) == 21
    assert built.griffin_sequence_ids == GRIFFIN_25M_VAL_SEQUENCE_IDS_V1
    assert built.heldout_c9_trace_receipt_sha256 == "7" * 64
    assert built.statistical_resamples == 10000
    assert built.run_config_sha256(
        "primary", "eventtrack-v2x", "marginal_voi"
    ) == "2" * 64
    assert built.run_checkpoint_sha256(
        "primary", "eventtrack-v2x", "marginal_voi", 1337
    ) == _run_config_bindings("2")[
        tuple(item.key for item in _run_config_bindings("2")).index(
            ("primary", "eventtrack-v2x", "marginal_voi")
        )
    ].checkpoint_sha256s[0]
