from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/formal_multiseed_plan.py"
SELECTOR_MODULE_PATH = ROOT / "tools/resilient_v2x/clearml_formal_candidate_selector.py"
PROVENANCE_MODULE_PATH = (
    ROOT / "tools/resilient_v2x/clearml_formal_training_provenance.py"
)
SOURCE_D_SCRIPT_SHA256 = "1" * 64
SOURCE_D_EQUIVALENCE_SHA256 = "2" * 64
AUDIT_SEAL_SHA256 = "3" * 64
SELECTOR_TASK_ID = "9" * 32
TRAINING_PROVENANCE_TASK_ID = "c" * 32
TRAINING_PROGRESS_ARTIFACT_SEAL_SHA256 = "4" * 64
TRAINING_PROVENANCE_ARTIFACT_SEAL_SHA256 = "5" * 64
LEGACY_TRAINING_SCRIPT_SHA256 = (
    "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
)
CANONICAL_TRAINING_SCRIPT_SHA256 = (
    "5d9584f1212bc021068e2a948bdf151aed97240c9994d5b994249316692f5066"
)
TRAINING_SCRIPT_EQUIVALENCE = {
    "legacy_script_sha256": LEGACY_TRAINING_SCRIPT_SHA256,
    "canonical_script_sha256": CANONICAL_TRAINING_SCRIPT_SHA256,
    "legacy_script_subjects": ["support_residual", "no_distillation"],
    "only_difference": "NESTED_TEACHER_EXPERIMENTS membership",
    "runtime_usage_closure": [
        "expect_nested_teacher_keyword",
        "expected_nested_teacher_contract_field",
    ],
}
AUDIT_CHAIN = {
    "training_provenance_task_id": TRAINING_PROVENANCE_TASK_ID,
    "training_progress_seal_sha256": TRAINING_PROGRESS_ARTIFACT_SEAL_SHA256,
    "training_provenance_seal_sha256": (TRAINING_PROVENANCE_ARTIFACT_SEAL_SHA256),
    "training_script_equivalence": TRAINING_SCRIPT_EQUIVALENCE,
    "evaluation_script_sha256": CANONICAL_TRAINING_SCRIPT_SHA256,
}


def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _source_revision_binding() -> dict[str, object]:
    provenance = _load_path(
        "formal_training_provenance_for_multiseed_fixture",
        PROVENANCE_MODULE_PATH,
    )
    equivalence = provenance._source_revision_equivalence()
    subject_map = provenance._source_revision_subject_map()
    evaluation_tree = provenance.SOURCE_TREE_SHA256
    return {
        "source_revision_equivalence": equivalence,
        "source_revision_equivalence_seal_sha256": equivalence["seal_sha256"],
        "source_revision_subject_map": subject_map,
        "source_revision_subject_map_seal_sha256": subject_map["seal_sha256"],
        "evaluation_source_revision_tree_sha256": evaluation_tree,
        "evaluation_source_revision": equivalence["source_revisions"][evaluation_tree],
    }


@pytest.fixture
def module():
    return _load_path("formal_multiseed_plan_fixture", MODULE_PATH)


@pytest.fixture
def selector_artifact():
    selector = _load_path("formal_candidate_selector_fixture", SELECTOR_MODULE_PATH)
    bases = {
        "resilient_v2x": 90.0,
        "support_residual": 89.0,
        "linear_no_distillation": 88.0,
        "no_distillation_peak_lr_3e4": 87.0,
        **{
            baseline: 50.0 + index
            for index, baseline in enumerate(selector.BASELINE_SUBJECTS)
        },
    }
    runs = {
        subject: {
            (delay, condition): {
                selector.LEADERSHIP_METRIC: (
                    base
                    - delay / 100.0
                    - {"Full": 0.0, "L-Fail": 2.0, "C-Fail": 1.0}[condition]
                )
            }
            for delay in selector.DELAYS_MS
            for condition in selector.CONDITIONS
        }
        for subject, base in bases.items()
    }
    audit_chain = copy.deepcopy(AUDIT_CHAIN)
    audit_chain.update(_source_revision_binding())
    return selector.build_selection(
        audit_task_id="a" * 32,
        leaderboard_task_id="b" * 32,
        audit_seal=AUDIT_SEAL_SHA256,
        leaderboard_seal="8" * 64,
        audit_chain=audit_chain,
        runs_by_subject=runs,
    )


def _build(
    module,
    selector_artifact,
    *,
    selected_candidate="resilient_v2x",
    selector_task_id=SELECTOR_TASK_ID,
    selector_seal=None,
    tied_candidates=None,
    audit_chain=AUDIT_CHAIN,
):
    return module.build_plan(
        source_d_script_sha256=SOURCE_D_SCRIPT_SHA256,
        source_d_equivalence_sha256=SOURCE_D_EQUIVALENCE_SHA256,
        formal_audit_seal_sha256=AUDIT_SEAL_SHA256,
        training_provenance_task_id=audit_chain["training_provenance_task_id"],
        training_progress_artifact_seal_sha256=audit_chain[
            "training_progress_seal_sha256"
        ],
        training_provenance_artifact_seal_sha256=audit_chain[
            "training_provenance_seal_sha256"
        ],
        training_script_equivalence=audit_chain["training_script_equivalence"],
        evaluation_script_sha256=audit_chain["evaluation_script_sha256"],
        formal_selector_seal_sha256=(
            selector_artifact["seal_sha256"] if selector_seal is None else selector_seal
        ),
        formal_selector_task_id=selector_task_id,
        formal_selector_artifact=selector_artifact,
        selected_candidate=selected_candidate,
        tied_candidates=tied_candidates,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value.pop("seal_sha256", None)
    value["seal_sha256"] = _canonical_sha256(value)
    return value


def _set_path(value: object, path: tuple[object, ...], replacement: object) -> None:
    target = value
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]


@pytest.fixture
def plan(module, selector_artifact):
    return _build(module, selector_artifact)


def test_initial_plan_is_exact_45_train_45_eval_540_conditions(module, plan) -> None:
    assert plan["schema_version"] == 2
    assert plan["plan_type"] == module.PLAN_TYPE
    assert plan["protocol_id"] == "DAIR-CAUSAL-1337-v1"
    assert plan["baseline_subjects"] == list(module.BASELINE_SUBJECTS)
    assert plan["baseline_count"] == 14
    assert plan["subject_count"] == 15
    assert plan["planned_candidates"] == ["resilient_v2x"]
    assert plan["candidate_count"] == 1
    assert plan["seed_count"] == 3
    assert plan["baseline_training_task_count"] == 42
    assert plan["baseline_evaluation_task_count"] == 42
    assert plan["candidate_training_task_count"] == 3
    assert plan["candidate_evaluation_task_count"] == 3
    assert plan["training_task_count"] == 45
    assert plan["evaluation_task_count"] == 45
    assert plan["total_evaluation_condition_count"] == 540
    assert len(plan["training_tasks"]) == 45
    assert len(plan["evaluation_tasks"]) == 45
    assert len(plan["selection_history"]) == 1
    assert module.validate_plan(plan) == plan


def test_initial_manifest_and_fallback_policy_cannot_claim_execution(
    module, plan
) -> None:
    delta = plan["append_delta"]
    assert delta == {
        "manifest_type": "initial_round_only",
        "round": 1,
        "predecessor_plan_seal_sha256": None,
        "training_task_keys": [task["task_key"] for task in plan["training_tasks"]],
        "evaluation_task_keys": [task["task_key"] for task in plan["evaluation_tasks"]],
        "training_task_count": 45,
        "evaluation_task_count": 45,
        "evaluation_condition_count": 540,
    }
    policy = plan["fallback_execution_policy"]
    assert policy["implementation_status"] == module.FALLBACK_IMPLEMENTATION_STATUS
    assert policy["implementation_status"] == (
        "disabled_until_verified_clearml_gate_producer"
    )
    assert policy["public_append_api"] == "unconditionally_fail_closed"
    assert policy["cli_append_command"] == "not_exposed"
    assert (
        policy["ordered_candidate_intent"]
        == plan["formal_selector_provenance"]["artifact"]["ranked_candidates"]
    )
    assert "append_policy" not in plan
    assert plan["execution_policy"]["fallback_action"] == (
        module.FALLBACK_IMPLEMENTATION_STATUS
    )
    assert "later_round_action" not in plan["execution_policy"]


def test_baselines_appear_exactly_once_per_seed_and_task_keys_are_unique(
    module, plan
) -> None:
    for collection in ("training_tasks", "evaluation_tasks"):
        tasks = plan[collection]
        keys = [task["task_key"] for task in tasks]
        assert len(keys) == len(set(keys)) == 45
        baseline_counts = Counter(
            task["subject"]
            for task in tasks
            if task["kind"] == "external_controlled_baseline"
        )
        assert baseline_counts == Counter(
            {subject: 3 for subject in module.BASELINE_SUBJECTS}
        )
        assert Counter(
            task["subject"] for task in tasks if task["kind"] == "selected_candidate"
        ) == Counter({"resilient_v2x": 3})


def test_every_task_has_fixed_training_and_12_condition_contract(module, plan) -> None:
    condition_ids = [
        f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        for delay in module.DELAYS_MS
        for condition in module.CONDITIONS
    ]
    training_keys = set()
    for index, task in enumerate(plan["training_tasks"], start=1):
        assert task["index"] == index
        training_keys.add(task["task_key"])
        assert task["gpus"] == 4
        assert task["batch_size_per_gpu"] == 2
        assert task["max_epochs"] == 50
        assert task["precision"] == "FP32"
        assert task["global_batch_size"] == 8
        assert task["val_interval"] == 10
        assert task["checkpoint_policy"] == "epoch_50_final_only"
    for index, task in enumerate(plan["evaluation_tasks"], start=1):
        assert task["index"] == index
        assert task["training_task_key"] in training_keys
        assert task["protocol_id"] == module.PROTOCOL_ID
        assert task["checkpoint_policy"] == "epoch_50_final_only"
        assert task["run_count"] == 12
        assert task["condition_ids"] == condition_ids


def test_each_seed_has_one_gpu_model_and_a100_has_two_balanced_slots(plan) -> None:
    expected_models = {1: "A100", 2: "V100", 3: "RTX5090"}
    expected_queues = {1: "GPU4-A100", 2: "GPU4-V100", 3: "GPU4-5090"}
    for seed_index, model in expected_models.items():
        tasks = [
            task
            for collection in ("training_tasks", "evaluation_tasks")
            for task in plan[collection]
            if task["seed_index"] == seed_index
        ]
        assert len(tasks) == 30
        assert {task["gpu_model"] for task in tasks} == {model}
        assert {task["worker_queue"] for task in tasks} == {expected_queues[seed_index]}
    a100_training = [task for task in plan["training_tasks"] if task["seed_index"] == 1]
    assert Counter(task["queue_pool_slot"] for task in a100_training) == Counter(
        {1: 8, 2: 7}
    )


def test_multiseed_gate_requires_all_36_conditions_and_all_9_aggregates(
    plan,
) -> None:
    gate = plan["multiseed_gate_contract"]
    assert gate["required_condition_wins_per_seed"] == 12
    assert gate["required_condition_wins_total"] == 36
    assert gate["required_seed_pass_count"] == 3
    assert gate["required_aggregate_wins_per_seed"] == 3
    assert gate["required_aggregate_wins_total"] == 9
    assert gate["comparison_epsilon"] == 0.0
    assert gate["tie_policy"] == "any_equal_value_or_zero_margin_is_failure"


def test_source_d_and_provisional_selector_provenance_are_bound(
    plan, selector_artifact
) -> None:
    assert plan["source_d_identity"] == {
        "script_sha256": SOURCE_D_SCRIPT_SHA256,
        "equivalence_artifact_sha256": SOURCE_D_EQUIVALENCE_SHA256,
    }
    assert plan["training_provenance_task_id"] == TRAINING_PROVENANCE_TASK_ID
    assert plan["training_progress_artifact_seal_sha256"] == (
        TRAINING_PROGRESS_ARTIFACT_SEAL_SHA256
    )
    assert plan["training_provenance_artifact_seal_sha256"] == (
        TRAINING_PROVENANCE_ARTIFACT_SEAL_SHA256
    )
    assert plan["training_script_equivalence"] == TRAINING_SCRIPT_EQUIVALENCE
    assert plan["evaluation_script_sha256"] == CANONICAL_TRAINING_SCRIPT_SHA256
    provenance = plan["formal_selector_provenance"]
    assert provenance["task_id"] == SELECTOR_TASK_ID
    assert provenance["artifact_seal_sha256"] == selector_artifact["seal_sha256"]
    assert provenance["selection_is_final"] is False
    assert provenance["plan_role"] == (
        "provisional_candidate_order_only_not_final_claim"
    )
    assert provenance["artifact"]["selection_is_final"] is False
    assert provenance["artifact"]["requires_multiseed_confirmation"] is True
    assert (
        provenance["artifact"]["training_provenance_task_id"]
        == (plan["training_provenance_task_id"])
    )
    assert (
        provenance["artifact"]["training_progress_seal_sha256"]
        == (plan["training_progress_artifact_seal_sha256"])
    )
    assert (
        provenance["artifact"]["training_provenance_seal_sha256"]
        == (plan["training_provenance_artifact_seal_sha256"])
    )
    assert (
        provenance["artifact"]["training_script_equivalence"]
        == plan["training_script_equivalence"]
    )
    assert (
        provenance["artifact"]["evaluation_script_sha256"]
        == plan["evaluation_script_sha256"]
    )
    assert plan["selection_history"] == [
        {
            "round": 1,
            "selector_rank": 1,
            "formal_selector_seal_sha256": selector_artifact["seal_sha256"],
            "selected_candidate": "resilient_v2x",
            "trigger": "initial_formal_selection",
            "previous_candidate": None,
            "previous_outcome": None,
            "tie_status": "resolved_total_order",
            "failure_evidence": None,
        }
    ]


def test_build_deep_freezes_all_nested_selector_input(
    module, selector_artifact
) -> None:
    audit_chain = copy.deepcopy(AUDIT_CHAIN)
    built = _build(module, selector_artifact, audit_chain=audit_chain)
    snapshot = copy.deepcopy(built)
    selector_artifact["ranked_candidates"].reverse()
    selector_artifact["candidate_results"][0]["condition_comparisons"][0][
        "margin"
    ] = -999.0
    selector_artifact["candidate_results"][0]["ranking_values"][
        "mean_12_margin"
    ] = -999.0
    selector_artifact["live_snapshot_recheck"]["training_tasks"] = "mutated"
    selector_artifact["training_script_equivalence"]["only_difference"] = "mutated"
    audit_chain["training_script_equivalence"]["runtime_usage_closure"].reverse()
    assert built == snapshot
    assert module.validate_plan(built) == built


def test_validate_plan_returns_a_deeply_independent_normalized_plan(
    module, plan
) -> None:
    supplied = copy.deepcopy(plan)
    normalized = module.validate_plan(supplied)
    supplied["formal_selector_provenance"]["artifact"]["ranked_candidates"].reverse()
    supplied["selection_history"][0]["selected_candidate"] = "support_residual"
    supplied["training_script_equivalence"]["legacy_script_subjects"].reverse()
    supplied["training_tasks"][0]["worker_queue"] = "mutated"
    assert normalized == plan
    assert module.validate_plan(normalized) == normalized


@pytest.mark.parametrize(
    "field",
    [
        "training_provenance_task_id",
        "training_progress_seal_sha256",
        "training_provenance_seal_sha256",
        "training_script_equivalence",
        "evaluation_script_sha256",
        "source_revision_equivalence",
        "source_revision_equivalence_seal_sha256",
        "source_revision_subject_map",
        "source_revision_subject_map_seal_sha256",
        "evaluation_source_revision_tree_sha256",
        "evaluation_source_revision",
    ],
)
def test_selector_audit_chain_missing_fields_fail_closed(
    module, selector_artifact, field
) -> None:
    tampered = copy.deepcopy(selector_artifact)
    tampered.pop(field)
    _reseal(tampered)
    with pytest.raises(ValueError, match="keys mismatch"):
        _build(module, tampered)


def test_selector_rejects_top_level_artifact_seal_name_as_extra_field(
    module, selector_artifact
) -> None:
    tampered = copy.deepcopy(selector_artifact)
    tampered["training_progress_artifact_seal_sha256"] = tampered[
        "training_progress_seal_sha256"
    ]
    _reseal(tampered)
    with pytest.raises(ValueError, match="keys mismatch"):
        _build(module, tampered)


def test_selector_progress_and_provenance_seal_swap_fails_cross_binding(
    module, selector_artifact
) -> None:
    tampered = copy.deepcopy(selector_artifact)
    (
        tampered["training_progress_seal_sha256"],
        tampered["training_provenance_seal_sha256"],
    ) = (
        tampered["training_provenance_seal_sha256"],
        tampered["training_progress_seal_sha256"],
    )
    _reseal(tampered)
    with pytest.raises(ValueError, match="training progress seal mismatch"):
        _build(module, tampered)


def test_selector_dependency_inventory_requires_five_unique_tasks(
    module, selector_artifact
) -> None:
    tampered = copy.deepcopy(selector_artifact)
    duplicate_id = tampered["watcher_task_id"]
    tampered["training_provenance_task_id"] = duplicate_id
    audit_chain = copy.deepcopy(AUDIT_CHAIN)
    audit_chain["training_provenance_task_id"] = duplicate_id
    _reseal(tampered)
    with pytest.raises(ValueError, match="dependency provenance is not unique"):
        _build(module, tampered, audit_chain=audit_chain)


@pytest.mark.parametrize(
    "attack",
    [
        "swap_script_hashes",
        "reverse_legacy_subjects",
        "change_only_difference",
        "reverse_runtime_usage_closure",
    ],
)
def test_training_script_equivalence_semantics_fail_closed_when_copies_agree(
    module, selector_artifact, attack
) -> None:
    equivalence = copy.deepcopy(TRAINING_SCRIPT_EQUIVALENCE)
    if attack == "swap_script_hashes":
        (
            equivalence["legacy_script_sha256"],
            equivalence["canonical_script_sha256"],
        ) = (
            equivalence["canonical_script_sha256"],
            equivalence["legacy_script_sha256"],
        )
    elif attack == "reverse_legacy_subjects":
        equivalence["legacy_script_subjects"].reverse()
    elif attack == "change_only_difference":
        equivalence["only_difference"] = "arbitrary source drift"
    else:
        equivalence["runtime_usage_closure"].reverse()

    tampered = copy.deepcopy(selector_artifact)
    tampered["training_script_equivalence"] = copy.deepcopy(equivalence)
    audit_chain = copy.deepcopy(AUDIT_CHAIN)
    audit_chain["training_script_equivalence"] = equivalence
    _reseal(tampered)
    with pytest.raises(ValueError, match="semantic contract mismatch"):
        _build(module, tampered, audit_chain=audit_chain)


def test_evaluation_script_must_equal_canonical_training_script(
    module, selector_artifact
) -> None:
    tampered = copy.deepcopy(selector_artifact)
    tampered["evaluation_script_sha256"] = LEGACY_TRAINING_SCRIPT_SHA256
    audit_chain = copy.deepcopy(AUDIT_CHAIN)
    audit_chain["evaluation_script_sha256"] = LEGACY_TRAINING_SCRIPT_SHA256
    _reseal(tampered)
    with pytest.raises(ValueError, match="evaluation script semantic contract"):
        _build(module, tampered, audit_chain=audit_chain)


def test_plan_top_level_schema_requires_exact_a_copy_field_names(module, plan) -> None:
    missing = copy.deepcopy(plan)
    missing.pop("training_progress_artifact_seal_sha256")
    _reseal(missing)
    with pytest.raises(ValueError, match="keys mismatch"):
        module.validate_plan(missing)

    extra = copy.deepcopy(plan)
    extra["training_progress_seal_sha256"] = TRAINING_PROGRESS_ARTIFACT_SEAL_SHA256
    _reseal(extra)
    with pytest.raises(ValueError, match="keys mismatch"):
        module.validate_plan(extra)


def test_schema_v1_plan_fails_closed_without_compatibility(module, plan) -> None:
    old = copy.deepcopy(plan)
    old["schema_version"] = 1
    _reseal(old)
    with pytest.raises(ValueError, match="schema_version must be exactly 2"):
        module.validate_plan(old)

    old_names = copy.deepcopy(plan)
    old_names["schema_version"] = 1
    old_names["training_progress_seal_sha256"] = old_names.pop(
        "training_progress_artifact_seal_sha256"
    )
    old_names["training_provenance_seal_sha256"] = old_names.pop(
        "training_provenance_artifact_seal_sha256"
    )
    _reseal(old_names)
    with pytest.raises(ValueError, match="keys mismatch"):
        module.validate_plan(old_names)


def test_build_plan_self_validates_before_returning(
    module, selector_artifact, monkeypatch
) -> None:
    original = module.validate_plan
    calls = []

    def recording_validate(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(module, "validate_plan", recording_validate)
    built = _build(module, selector_artifact)
    assert len(calls) == 1
    assert built == original(built)


def test_selector_seal_ranking_and_snapshot_scope_fail_closed(
    module, selector_artifact
) -> None:
    plain_tamper = copy.deepcopy(selector_artifact)
    plain_tamper["ranked_candidates"][0:2] = reversed(
        plain_tamper["ranked_candidates"][0:2]
    )
    with pytest.raises(ValueError, match="artifact seal SHA-256 mismatch"):
        _build(module, plain_tamper, selector_seal=selector_artifact["seal_sha256"])

    _reseal(plain_tamper)
    with pytest.raises(ValueError, match="ranking/selection is inconsistent"):
        _build(module, plain_tamper)

    snapshot_tamper = copy.deepcopy(selector_artifact)
    snapshot_tamper["live_snapshot_recheck"]["training_tasks"] = "untrusted"
    _reseal(snapshot_tamper)
    with pytest.raises(ValueError, match="live_snapshot_recheck mismatch"):
        _build(module, snapshot_tamper)


def test_clearml_ids_reject_ints_even_when_their_digits_look_valid(
    module, selector_artifact
) -> None:
    with pytest.raises(ValueError, match="lowercase 32-hex ClearML ID"):
        _build(module, selector_artifact, selector_task_id=int("9" * 32))

    dependency_tamper = copy.deepcopy(selector_artifact)
    dependency_tamper["audit_task_id"] = int("a" * 32, 16)
    _reseal(dependency_tamper)
    with pytest.raises(ValueError, match="lowercase 32-hex ClearML ID"):
        _build(module, dependency_tamper)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("selection_is_final",), 0),
        (("selected_candidate_gate_passed",), 1),
        (("requires_checkpoint_byte_audit",), 1),
        (("requires_multiseed_confirmation",), 1),
        (("candidate_results", 0, "gate_passed"), 1),
        (("candidate_results", 0, "strictly_leads_all_conditions"), 1),
        (
            (
                "candidate_results",
                0,
                "condition_comparisons",
                0,
                "strictly_leads",
            ),
            1,
        ),
    ],
)
def test_selector_boolean_fields_reject_zero_one_aliases(
    module, selector_artifact, path, replacement
) -> None:
    tampered = copy.deepcopy(selector_artifact)
    _set_path(tampered, path, replacement)
    _reseal(tampered)
    with pytest.raises(ValueError, match="must be exactly"):
        _build(module, tampered)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), True),
        (("unsupported_sample_count",), False),
        (("delays_ms", 0), False),
        (("candidate_results", 0, "fixed_order_index"), False),
        (
            (
                "candidate_results",
                0,
                "ranking_values",
                "fixed_candidate_order",
            ),
            False,
        ),
        (("candidate_results", 0, "performance_rank"), True),
        (
            (
                "candidate_results",
                0,
                "condition_comparisons",
                0,
                "delay_ms",
            ),
            False,
        ),
    ],
)
def test_selector_integer_fields_reject_boolean_aliases(
    module, selector_artifact, path, replacement
) -> None:
    tampered = copy.deepcopy(selector_artifact)
    _set_path(tampered, path, replacement)
    _reseal(tampered)
    with pytest.raises(ValueError):
        _build(module, tampered)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("candidate_count",), True, "candidate_count"),
        (("training_task_count",), 45.0, "training_task_count"),
        (("selection_history", 0, "round"), True, "history round"),
        (
            ("formal_selector_provenance", "selection_is_final"),
            0,
            "selection_is_final",
        ),
        (
            ("execution_policy", "actual_worker_gpu_model_audit_required"),
            1,
            "deterministic contract",
        ),
    ],
)
def test_resealed_plan_rejects_bool_int_and_float_aliases(
    module, plan, path, replacement, message
) -> None:
    tampered = copy.deepcopy(plan)
    _set_path(tampered, path, replacement)
    _reseal(tampered)
    with pytest.raises(ValueError, match=message):
        module.validate_plan(tampered)


def test_validate_accepts_only_one_initial_history_record(module, plan) -> None:
    extra_history = copy.deepcopy(plan)
    extra_history["selection_history"].append(
        copy.deepcopy(extra_history["selection_history"][0])
    )
    _reseal(extra_history)
    with pytest.raises(module.TrustedGateRequiredError, match="exactly the initial"):
        module.validate_plan(extra_history)

    fallback_evidence = copy.deepcopy(plan)
    fallback_evidence["selection_history"][0]["failure_evidence"] = {
        "self_sealed": True
    }
    _reseal(fallback_evidence)
    with pytest.raises(module.TrustedGateRequiredError, match="cannot contain"):
        module.validate_plan(fallback_evidence)


@pytest.mark.parametrize("supplied_plan", [{}, {"seal_sha256": 7}, None])
def test_every_append_attempt_unconditionally_requires_a_trusted_gate(
    module, supplied_plan
) -> None:
    with pytest.raises(
        module.TrustedGateRequiredError,
        match="verified ClearML gate producer",
    ):
        module.append_candidate(
            supplied_plan,
            failed_candidate=7,
            next_candidate=False,
            formal_selector_seal_sha256=object(),
            failed_multiseed_gate_task_id=123,
            failed_multiseed_gate_artifact={"self_sealed": True},
            tied_candidates=["untrusted"],
        )


def test_obsolete_private_gate_artifact_append_helpers_are_absent(module) -> None:
    for name in (
        "_candidate_task_records",
        "_candidate_task_keys",
        "_validated_multiseed_gate_artifact",
        "_failure_evidence",
        "_validated_failure_evidence",
    ):
        assert not hasattr(module, name)

    history = [
        {
            "selected_candidate": "resilient_v2x",
        },
        {
            "selected_candidate": "support_residual",
        },
    ]
    with pytest.raises(module.TrustedGateRequiredError, match="initial-only"):
        module._task_records(history)


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        ("weak_feature_distillation", "alias"),
        ("linear_weak_feature_distillation", "alias"),
        ("teacher_init_linear_no_distillation", "alias"),
        ("ffnet", "external baseline"),
        ("future_magic_model", "not a known formal candidate"),
    ],
)
def test_build_rejects_alias_baseline_and_unknown_candidates(
    module, selector_artifact, candidate, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _build(module, selector_artifact, selected_candidate=candidate)


def test_build_rejects_unresolved_or_malformed_ties(module, selector_artifact) -> None:
    with pytest.raises(ValueError, match="unresolved candidate ties"):
        _build(
            module,
            selector_artifact,
            tied_candidates=["resilient_v2x", "support_residual"],
        )
    with pytest.raises(ValueError, match="must be a sequence"):
        _build(module, selector_artifact, tied_candidates="resilient_v2x")


def test_seal_and_plan_are_deterministic(module, selector_artifact) -> None:
    first = _build(module, selector_artifact)
    second = _build(module, copy.deepcopy(selector_artifact))
    assert first == second
    payload = copy.deepcopy(first)
    observed = payload.pop("seal_sha256")
    assert observed == _canonical_sha256(payload)


def test_cli_generates_and_validates_but_does_not_expose_append(
    module, selector_artifact, tmp_path
) -> None:
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(json.dumps(selector_artifact), encoding="utf-8")
    equivalence_path = tmp_path / "training_script_equivalence.json"
    equivalence_path.write_text(
        json.dumps(TRAINING_SCRIPT_EQUIVALENCE), encoding="utf-8"
    )
    generated = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "generate",
            "--source-d-script-sha256",
            SOURCE_D_SCRIPT_SHA256,
            "--source-d-equivalence-sha256",
            SOURCE_D_EQUIVALENCE_SHA256,
            "--formal-audit-seal-sha256",
            AUDIT_SEAL_SHA256,
            "--training-provenance-task-id",
            TRAINING_PROVENANCE_TASK_ID,
            "--training-progress-artifact-seal-sha256",
            TRAINING_PROGRESS_ARTIFACT_SEAL_SHA256,
            "--training-provenance-artifact-seal-sha256",
            TRAINING_PROVENANCE_ARTIFACT_SEAL_SHA256,
            "--training-script-equivalence",
            str(equivalence_path),
            "--evaluation-script-sha256",
            CANONICAL_TRAINING_SCRIPT_SHA256,
            "--formal-selector-seal-sha256",
            selector_artifact["seal_sha256"],
            "--formal-selector-task-id",
            SELECTOR_TASK_ID,
            "--formal-selector-artifact",
            str(selector_path),
            "--selected-candidate",
            "resilient_v2x",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    generated_plan = json.loads(generated.stdout)
    assert module.validate_plan(generated_plan) == generated_plan

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(generated_plan), encoding="utf-8")
    validated = subprocess.run(
        [sys.executable, str(MODULE_PATH), "validate", str(plan_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(validated.stdout) == generated_plan

    rejected = subprocess.run(
        [sys.executable, str(MODULE_PATH), "append", str(plan_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "invalid choice" in rejected.stderr
    assert "{generate,validate}" in rejected.stderr


def test_module_is_local_only_and_has_no_clearml_import() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "import clearml" not in source
    assert "from clearml" not in source
