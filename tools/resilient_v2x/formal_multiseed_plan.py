#!/usr/bin/env python3
"""Generate and validate the sealed formal ResilientV2X multi-seed plan.

This module is deliberately local-only: it describes planned work but never
creates, clones, enqueues, or mutates a ClearML task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path


PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
PLAN_TYPE = "resilient_v2x_formal_multiseed_plan"
TRAINING_OVERLAY_PROTOCOL_SEED = 20_250_218
MAX_EPOCHS = 50
PRECISION = "FP32"
GPU_COUNT = 4
BATCH_SIZE_PER_GPU = 2
GLOBAL_BATCH_SIZE = 8
VAL_INTERVAL = 10
CHECKPOINT_POLICY = "epoch_50_final_only"
SAMPLE_COUNT = 1_337
GROUND_TRUTH_COUNT = 11_330
UNSUPPORTED_SAMPLE_COUNT = 0
DELAYS_MS = (0, 100, 200, 300)
CONDITIONS = ("Full", "L-Fail", "C-Fail")
RUN_COUNT_PER_EVALUATION = len(DELAYS_MS) * len(CONDITIONS)
AP_METRIC_KEYS = (
    "resilient_v2x/car_bev_ap_r40_0.50",
    "resilient_v2x/car_bev_ap_r40_0.70",
    "resilient_v2x/car_3d_ap_r40_0.50",
    "resilient_v2x/car_3d_ap_r40_0.70",
)
LEADERSHIP_METRIC = "resilient_v2x/car_bev_ap_r40_0.70"
FORMAL_SELECTOR_ARTIFACT = "formal_candidate_selection"
LEGACY_TRAINING_SCRIPT_SHA256 = (
    "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
)
CANONICAL_TRAINING_SCRIPT_SHA256 = (
    "5d9584f1212bc021068e2a948bdf151aed97240c9994d5b994249316692f5066"
)
EVALUATION_SCRIPT_SHA256 = CANONICAL_TRAINING_SCRIPT_SHA256
LEGACY_SCRIPT_SUBJECTS = ("support_residual", "no_distillation")
TRAINING_SCRIPT_ONLY_DIFFERENCE = "NESTED_TEACHER_EXPERIMENTS membership"
TRAINING_SCRIPT_RUNTIME_USAGE_CLOSURE = (
    "expect_nested_teacher_keyword",
    "expected_nested_teacher_contract_field",
)
OLD_SOURCE_DATASET_ID = "4f7fac0078a4419a907fec6ff9e306c8"
OLD_SOURCE_ARCHIVE_NAME = "resilient-v2x-source-5c984ad49b52.tar.zst"
OLD_SOURCE_ARCHIVE_BYTES = 1_222_481
OLD_SOURCE_ARCHIVE_SHA256 = (
    "655f33d9b684c26ff4ff577d5c5f3175d4fc4eac40426eb55820e7a8aa8bc40d"
)
OLD_SOURCE_TREE_SHA256 = (
    "5c984ad49b5232d7f6d053fb641895283477efcbf2de40b36d9b3f3c6f8e28b6"
)
OLD_SOURCE_INVENTORY_BYTES = 101_195
OLD_SOURCE_INVENTORY_SHA256 = (
    "bed72cd86438f2ba932edda14052e3cd3d9589ec201d09d18a315e18a2f7cff2"
)
OLD_SOURCE_FILE_COUNT = 631
OLD_SOURCE_BYTES = 8_926_106
NEW_SOURCE_DATASET_ID = "351feedbbe81481fa31f1e9ae11a3f4e"
NEW_SOURCE_ARCHIVE_NAME = "resilient-v2x-source-ad511d88b731.tar.zst"
NEW_SOURCE_ARCHIVE_BYTES = 1_222_492
NEW_SOURCE_ARCHIVE_SHA256 = (
    "b94a01c2acf2cc456fe9729f7c40e990e6d44b65e789c6fed11989a673f4f6da"
)
NEW_SOURCE_TREE_SHA256 = (
    "ad511d88b731cb45ef2defb873712bdb2a325c648634b66c349fe1c1459510e4"
)
NEW_SOURCE_INVENTORY_BYTES = 101_195
NEW_SOURCE_INVENTORY_SHA256 = (
    "39b1a42af65ad5df935945bd0a4eeac6e7f6e1cfdd1cc608f3dd8a708e9c5ca0"
)
NEW_SOURCE_FILE_COUNT = 631
NEW_SOURCE_BYTES = 8_926_102
SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256 = (
    "29de9700cac66f9998be643e85a8ec646c04ec17fddbb6207bc1438e9e73941b"
)
SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256 = (
    "c3170f4a88b080f9cc7267053f354640dd260687cb2c35a6f7ffed73d69f4154"
)
SOURCE_REVISION_CERTIFICATE_BY_TREE = {
    OLD_SOURCE_TREE_SHA256: {
        "dataset_id": OLD_SOURCE_DATASET_ID,
        "tree_sha256": OLD_SOURCE_TREE_SHA256,
        "file_count": OLD_SOURCE_FILE_COUNT,
        "source_bytes": OLD_SOURCE_BYTES,
        "archive": {
            "name": OLD_SOURCE_ARCHIVE_NAME,
            "size_bytes": OLD_SOURCE_ARCHIVE_BYTES,
            "sha256": OLD_SOURCE_ARCHIVE_SHA256,
        },
        "inventory": {
            "name": "source-inventory.json",
            "size_bytes": OLD_SOURCE_INVENTORY_BYTES,
            "sha256": OLD_SOURCE_INVENTORY_SHA256,
        },
    },
    NEW_SOURCE_TREE_SHA256: {
        "dataset_id": NEW_SOURCE_DATASET_ID,
        "tree_sha256": NEW_SOURCE_TREE_SHA256,
        "file_count": NEW_SOURCE_FILE_COUNT,
        "source_bytes": NEW_SOURCE_BYTES,
        "archive": {
            "name": NEW_SOURCE_ARCHIVE_NAME,
            "size_bytes": NEW_SOURCE_ARCHIVE_BYTES,
            "sha256": NEW_SOURCE_ARCHIVE_SHA256,
        },
        "inventory": {
            "name": "source-inventory.json",
            "size_bytes": NEW_SOURCE_INVENTORY_BYTES,
            "sha256": NEW_SOURCE_INVENTORY_SHA256,
        },
    },
}
FALLBACK_IMPLEMENTATION_STATUS = "disabled_until_verified_clearml_gate_producer"
RANKING_KEY = (
    "gate_passed",
    "min_condition_margin",
    "worst_12_margin",
    "mean_12_margin",
    "clean_0ms_margin",
    "fixed_candidate_order",
)
RANKING_VALUE_KEYS = RANKING_KEY[1:]

BASELINE_SUBJECTS = (
    "coformernet",
    "ffnet",
    "bevfusion",
    "v2x_vit",
    "cobevt",
    "ego_only",
    "fcooper",
    "attfuse",
    "v2vnet",
    "when2com",
    "where2comm",
    "late_fusion",
    "disconet",
    "how2comm",
)
CANDIDATE_ORDER = (
    "resilient_v2x",
    "support_residual",
    "linear_no_distillation",
    "no_distillation_peak_lr_3e4",
)
CANDIDATE_ALIASES = {
    "weak_feature_distillation": "resilient_v2x",
    "linear_weak_feature_distillation": "ptf_linear",
    "teacher_init_linear_no_distillation": "linear_no_distillation",
}

SEED_DERIVATION_DOMAIN = "resilient-v2x/formal-multiseed-training-seed/v1"
SEED_DERIVATION_ALGORITHM = (
    "sha256(UTF-8 input), take the first 32 bits as unsigned big-endian"
)
SEED_INPUT_TEMPLATE = "{domain_separator}|protocol_id={protocol_id}|derived_slot={slot}"
SEED_COLLISION_POLICY = "fail_closed_no_retry"
SHA256_HEX_LENGTH = 64

_TOP_LEVEL_KEYS = {
    "schema_version",
    "plan_type",
    "protocol_id",
    "source_d_identity",
    "formal_audit_seal_sha256",
    "training_provenance_task_id",
    "training_progress_artifact_seal_sha256",
    "training_provenance_artifact_seal_sha256",
    "training_script_equivalence",
    "evaluation_script_sha256",
    "initial_formal_selector_seal_sha256",
    "formal_selector_provenance",
    "initial_selected_candidate",
    "baseline_subjects",
    "baseline_count",
    "subject_order",
    "subject_count",
    "candidate_allowlist",
    "candidate_aliases_rejected",
    "planned_candidates",
    "candidate_count",
    "selection_history",
    "selection_tie_policy",
    "seed_derivation",
    "seeds",
    "seed_count",
    "training_contract",
    "evaluation_contract",
    "multiseed_gate_contract",
    "training_tasks",
    "evaluation_tasks",
    "baseline_training_task_count",
    "baseline_evaluation_task_count",
    "candidate_training_task_count",
    "candidate_evaluation_task_count",
    "training_task_count",
    "evaluation_task_count",
    "total_evaluation_condition_count",
    "fallback_execution_policy",
    "append_delta",
    "execution_policy",
    "seal_sha256",
}


class TrustedGateRequiredError(RuntimeError):
    """Raised because no independently trusted fallback gate exists yet."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _frozen_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    try:
        frozen = json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} must be canonical JSON") from error
    if type(frozen) is not dict:
        raise ValueError(f"{context} must be a JSON object")
    return frozen


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    result = _frozen_mapping(value, context="sealed payload")
    result.pop("seal_sha256", None)
    result["seal_sha256"] = _content_sha256(result)
    return result


def _sha256(value: object, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _clearml_id(value: object, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase 32-hex ClearML ID")
    return value


def _exact_int(value: object, expected: int, *, context: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{context} must be exactly {expected}")
    return value


def _exact_bool(value: object, expected: bool, *, context: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{context} must be exactly {expected!r}")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], *, context: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{context} keys mismatch; missing={missing!r}, extra={extra!r}"
        )


def _validate_training_script_equivalence(
    value: object, *, context: str
) -> dict[str, object]:
    equivalence = _frozen_mapping(value, context=context)
    _require_exact_keys(
        equivalence,
        {
            "legacy_script_sha256",
            "canonical_script_sha256",
            "legacy_script_subjects",
            "only_difference",
            "runtime_usage_closure",
        },
        context=context,
    )
    legacy_sha = _sha256(
        equivalence.get("legacy_script_sha256"),
        f"{context} legacy script",
    )
    canonical_sha = _sha256(
        equivalence.get("canonical_script_sha256"),
        f"{context} canonical script",
    )
    if (
        legacy_sha != LEGACY_TRAINING_SCRIPT_SHA256
        or canonical_sha != CANONICAL_TRAINING_SCRIPT_SHA256
        or legacy_sha == canonical_sha
        or equivalence.get("legacy_script_subjects") != list(LEGACY_SCRIPT_SUBJECTS)
        or equivalence.get("only_difference") != TRAINING_SCRIPT_ONLY_DIFFERENCE
        or equivalence.get("runtime_usage_closure")
        != list(TRAINING_SCRIPT_RUNTIME_USAGE_CLOSURE)
    ):
        raise ValueError(f"{context} semantic contract mismatch")
    return equivalence


def _validate_source_provenance_binding(
    value: Mapping[str, object], *, context: str
) -> dict[str, object]:
    expected_keys = {
        "source_revision_equivalence",
        "source_revision_equivalence_seal_sha256",
        "source_revision_subject_map",
        "source_revision_subject_map_seal_sha256",
        "evaluation_source_revision_tree_sha256",
        "evaluation_source_revision",
    }
    _require_exact_keys(value, expected_keys, context=context)
    equivalence = _frozen_mapping(
        value.get("source_revision_equivalence"),
        context=f"{context} source revision equivalence",
    )
    _require_exact_keys(
        equivalence,
        {
            "schema_version",
            "contract",
            "passed",
            "old_tree_sha256",
            "new_tree_sha256",
            "file_count",
            "path_set_equal",
            "mode_map_equal",
            "byte_identical_file_count",
            "changed_file_count",
            "changed_files",
            "source_revisions",
            "seal_sha256",
        },
        context=f"{context} source revision equivalence",
    )
    equivalence_seal = _sha256(
        equivalence.get("seal_sha256"),
        f"{context} source revision equivalence seal",
    )
    root_equivalence_seal = _sha256(
        value.get("source_revision_equivalence_seal_sha256"),
        f"{context} source revision equivalence root seal",
    )
    if (
        _sealed(equivalence)["seal_sha256"] != equivalence_seal
        or equivalence_seal != root_equivalence_seal
        or equivalence_seal != SOURCE_REVISION_EQUIVALENCE_SEAL_SHA256
        or equivalence.get("schema_version") != 1
        or equivalence.get("contract")
        != "sealed-source-inventory-tuple-to-list-only-v1"
        or equivalence.get("passed") is not True
        or equivalence.get("old_tree_sha256") != OLD_SOURCE_TREE_SHA256
        or equivalence.get("new_tree_sha256") != NEW_SOURCE_TREE_SHA256
        or equivalence.get("file_count") != OLD_SOURCE_FILE_COUNT
        or equivalence.get("path_set_equal") is not True
        or equivalence.get("mode_map_equal") is not True
        or equivalence.get("byte_identical_file_count") != 627
        or equivalence.get("changed_file_count") != 4
        or not isinstance(equivalence.get("changed_files"), list)
        or len(equivalence["changed_files"]) != 4
        or equivalence.get("source_revisions")
        != SOURCE_REVISION_CERTIFICATE_BY_TREE
    ):
        raise ValueError(f"{context} source revision equivalence mismatch")

    subject_map = _frozen_mapping(
        value.get("source_revision_subject_map"),
        context=f"{context} source revision subject map",
    )
    _require_exact_keys(
        subject_map,
        {
            "schema_version",
            "contract",
            "subject_order",
            "source_revision_by_subject",
            "revision_subjects",
            "revision_subject_counts",
            "evaluation_source_revision_tree_sha256",
            "seal_sha256",
        },
        context=f"{context} source revision subject map",
    )
    subject_map_seal = _sha256(
        subject_map.get("seal_sha256"),
        f"{context} source revision subject map seal",
    )
    root_subject_map_seal = _sha256(
        value.get("source_revision_subject_map_seal_sha256"),
        f"{context} source revision subject map root seal",
    )
    subject_order = subject_map.get("subject_order")
    revision_by_subject = subject_map.get("source_revision_by_subject")
    revision_subjects = subject_map.get("revision_subjects")
    if (
        _sealed(subject_map)["seal_sha256"] != subject_map_seal
        or subject_map_seal != root_subject_map_seal
        or subject_map_seal != SOURCE_REVISION_SUBJECT_MAP_SEAL_SHA256
        or subject_map.get("schema_version") != 1
        or subject_map.get("contract")
        != "formal-26-source-revision-subject-map-v1"
        or not isinstance(subject_order, list)
        or len(subject_order) != 26
        or any(type(subject) is not str or not subject for subject in subject_order)
        or len(set(subject_order)) != 26
        or not isinstance(revision_by_subject, Mapping)
        or set(revision_by_subject) != set(subject_order)
        or any(
            type(tree_sha256) is not str
            for tree_sha256 in revision_by_subject.values()
        )
        or set(revision_by_subject.values())
        != {OLD_SOURCE_TREE_SHA256, NEW_SOURCE_TREE_SHA256}
        or not isinstance(revision_subjects, Mapping)
        or set(revision_subjects) != {OLD_SOURCE_TREE_SHA256, NEW_SOURCE_TREE_SHA256}
        or subject_map.get("revision_subject_counts")
        != {OLD_SOURCE_TREE_SHA256: 21, NEW_SOURCE_TREE_SHA256: 5}
        or subject_map.get("evaluation_source_revision_tree_sha256")
        != OLD_SOURCE_TREE_SHA256
    ):
        raise ValueError(f"{context} source revision subject map mismatch")
    for tree_sha256, expected_count in (
        (OLD_SOURCE_TREE_SHA256, 21),
        (NEW_SOURCE_TREE_SHA256, 5),
    ):
        subjects = revision_subjects[tree_sha256]
        if (
            not isinstance(subjects, list)
            or len(subjects) != expected_count
            or any(type(subject) is not str or not subject for subject in subjects)
            or len(set(subjects)) != expected_count
            or any(revision_by_subject.get(subject) != tree_sha256 for subject in subjects)
        ):
            raise ValueError(f"{context} source revision subject inventory mismatch")

    evaluation_tree = _sha256(
        value.get("evaluation_source_revision_tree_sha256"),
        f"{context} evaluation source revision tree",
    )
    evaluation_revision = value.get("evaluation_source_revision")
    if (
        evaluation_tree != OLD_SOURCE_TREE_SHA256
        or evaluation_revision != SOURCE_REVISION_CERTIFICATE_BY_TREE[evaluation_tree]
    ):
        raise ValueError(f"{context} evaluation source revision mismatch")
    return {
        "source_revision_equivalence": equivalence,
        "source_revision_equivalence_seal_sha256": equivalence_seal,
        "source_revision_subject_map": subject_map,
        "source_revision_subject_map_seal_sha256": subject_map_seal,
        "evaluation_source_revision_tree_sha256": evaluation_tree,
        "evaluation_source_revision": dict(evaluation_revision),
    }


def _candidate(value: object, *, context: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{context} must be a non-empty canonical candidate name")
    if value in CANDIDATE_ALIASES:
        canonical = CANDIDATE_ALIASES[value]
        raise ValueError(
            f"{context} {value!r} is an alias for {canonical!r}; aliases are forbidden"
        )
    if value in BASELINE_SUBJECTS:
        raise ValueError(
            f"{context} {value!r} is an external baseline, not a candidate"
        )
    if value not in CANDIDATE_ORDER:
        raise ValueError(f"{context} {value!r} is not a known formal candidate")
    return value


def _finite_number(value: object, *, context: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"{context} must be a finite number")
    return float(value)


def _validate_best_baselines(value: Mapping[str, object], *, context: str) -> None:
    subjects = value.get("best_baseline_subjects")
    if (
        not isinstance(subjects, list)
        or not subjects
        or len(subjects) != len(set(subjects))
        or any(subject not in BASELINE_SUBJECTS for subject in subjects)
        or subjects != [subject for subject in BASELINE_SUBJECTS if subject in subjects]
        or value.get("best_baseline_subject") != subjects[0]
    ):
        raise ValueError(f"{context} best-baseline tie set is invalid")


def _validate_selector_comparison(
    value: object,
    *,
    context: str,
    condition: tuple[int, str] | None,
) -> float:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} is not an object")
    common_keys = {
        "candidate_value",
        "best_baseline_value",
        "best_baseline_subject",
        "best_baseline_subjects",
        "margin",
        "strictly_leads",
    }
    expected_keys = common_keys | (
        {"condition_id", "delay_ms", "condition"} if condition is not None else set()
    )
    _require_exact_keys(value, expected_keys, context=context)
    if condition is not None:
        delay, condition_name = condition
        condition_id = f"delay_{delay:03d}_{condition_name.lower().replace('-', '_')}"
        _exact_int(value.get("delay_ms"), delay, context=f"{context} delay_ms")
        if (
            value.get("condition") != condition_name
            or value.get("condition_id") != condition_id
        ):
            raise ValueError(f"{context} condition identity mismatch")
    _validate_best_baselines(value, context=context)
    candidate_value = _finite_number(
        value.get("candidate_value"), context=f"{context} candidate value"
    )
    baseline_value = _finite_number(
        value.get("best_baseline_value"), context=f"{context} baseline value"
    )
    margin = _finite_number(value.get("margin"), context=f"{context} margin")
    if margin != candidate_value - baseline_value:
        raise ValueError(f"{context} margin mismatch")
    _exact_bool(
        value.get("strictly_leads"),
        margin > 0.0,
        context=f"{context} strict-lead flag",
    )
    return margin


def _validate_candidate_result(
    value: object, *, subject: str
) -> tuple[dict[str, object], tuple[int, float, float, float, float, int]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"selector candidate result {subject} is not an object")
    _require_exact_keys(
        value,
        {
            "subject",
            "fixed_order_index",
            "conditions_won",
            "conditions_total",
            "conditions_won_fraction",
            "strictly_leads_all_conditions",
            "condition_comparisons",
            "aggregate_comparisons",
            "strictly_leads_all_aggregate",
            "gate_passed",
            "ranking_values",
            "performance_rank",
            "selection_priority_rank",
            "rank",
        },
        context=f"selector candidate result {subject}",
    )
    fixed_index = CANDIDATE_ORDER.index(subject)
    if value.get("subject") != subject:
        raise ValueError(f"selector candidate result {subject} identity mismatch")
    _exact_int(
        value.get("fixed_order_index"),
        fixed_index,
        context=f"selector candidate result {subject} fixed_order_index",
    )
    raw_conditions = value.get("condition_comparisons")
    if not isinstance(raw_conditions, list) or len(raw_conditions) != 12:
        raise ValueError(
            f"selector candidate result {subject} condition count mismatch"
        )
    condition_margins = [
        _validate_selector_comparison(
            comparison,
            context=f"selector candidate {subject} condition {index}",
            condition=pair,
        )
        for index, (comparison, pair) in enumerate(
            zip(
                raw_conditions,
                ((delay, condition) for delay in DELAYS_MS for condition in CONDITIONS),
                strict=True,
            ),
            start=1,
        )
    ]
    won = sum(margin > 0.0 for margin in condition_margins)
    _exact_int(
        value.get("conditions_won"),
        won,
        context=f"selector candidate result {subject} conditions_won",
    )
    _exact_int(
        value.get("conditions_total"),
        12,
        context=f"selector candidate result {subject} conditions_total",
    )
    _exact_bool(
        value.get("strictly_leads_all_conditions"),
        won == 12,
        context=f"selector candidate result {subject} all-condition flag",
    )
    if value.get("conditions_won_fraction") != f"{won}/12":
        raise ValueError(f"selector candidate result {subject} win count mismatch")
    raw_aggregates = value.get("aggregate_comparisons")
    dimensions = ("clean_0ms", "mean_12", "worst_12")
    if not isinstance(raw_aggregates, Mapping) or set(raw_aggregates) != set(
        dimensions
    ):
        raise ValueError(f"selector candidate result {subject} aggregates mismatch")
    aggregate_margins = {
        dimension: _validate_selector_comparison(
            raw_aggregates[dimension],
            context=f"selector candidate {subject} aggregate {dimension}",
            condition=None,
        )
        for dimension in dimensions
    }
    all_aggregate = all(margin > 0.0 for margin in aggregate_margins.values())
    gate_passed = won == 12 and all_aggregate
    _exact_bool(
        value.get("strictly_leads_all_aggregate"),
        all_aggregate,
        context=f"selector candidate result {subject} all-aggregate flag",
    )
    _exact_bool(
        value.get("gate_passed"),
        gate_passed,
        context=f"selector candidate result {subject} gate flag",
    )
    ranking = value.get("ranking_values")
    if not isinstance(ranking, Mapping) or set(ranking) != set(RANKING_VALUE_KEYS):
        raise ValueError(f"selector candidate result {subject} ranking values mismatch")
    expected_ranking: dict[str, float | int] = {
        "min_condition_margin": min(condition_margins),
        "worst_12_margin": aggregate_margins["worst_12"],
        "mean_12_margin": aggregate_margins["mean_12"],
        "clean_0ms_margin": aggregate_margins["clean_0ms"],
        "fixed_candidate_order": fixed_index,
    }
    for key in RANKING_VALUE_KEYS[:-1]:
        observed = _finite_number(
            ranking.get(key),
            context=f"selector candidate result {subject} ranking {key}",
        )
        if observed != expected_ranking[key]:
            raise ValueError(f"selector candidate result {subject} ranking drifted")
    _exact_int(
        ranking.get("fixed_candidate_order"),
        fixed_index,
        context=f"selector candidate result {subject} fixed candidate order",
    )
    for field in ("performance_rank", "selection_priority_rank", "rank"):
        rank = value.get(field)
        if type(rank) is not int or not 1 <= rank <= len(CANDIDATE_ORDER):
            raise ValueError(f"selector candidate result {subject} {field} is invalid")
    sort_key = (
        0 if gate_passed else 1,
        -expected_ranking["min_condition_margin"],
        -expected_ranking["worst_12_margin"],
        -expected_ranking["mean_12_margin"],
        -expected_ranking["clean_0ms_margin"],
        fixed_index,
    )
    return _frozen_mapping(
        value, context=f"selector candidate result {subject}"
    ), sort_key


def _validate_selector_artifact(
    value: object,
    *,
    selector_task_id: str,
    expected_seal: str,
    expected_audit_seal: str,
    expected_training_provenance_task_id: str,
    expected_training_progress_seal: str,
    expected_training_provenance_seal: str,
    expected_training_script_equivalence: Mapping[str, object],
    expected_evaluation_script_sha256: str,
    selected_candidate: str,
) -> dict[str, object]:
    artifact = _frozen_mapping(value, context="formal selector artifact")
    _require_exact_keys(
        artifact,
        {
            "schema_version",
            "document_type",
            "status",
            "selection_claim",
            "selection_stage",
            "selection_is_final",
            "claim_scope",
            "live_snapshot_recheck",
            "model_checkpoint_byte_verification_complete",
            "requires_checkpoint_byte_audit",
            "requires_multiseed_confirmation",
            "architecture_revision_required",
            "audit_task_id",
            "leaderboard_task_id",
            "training_controller_task_id",
            "training_provenance_task_id",
            "watcher_task_id",
            "training_progress_seal_sha256",
            "training_provenance_seal_sha256",
            "source_revision_equivalence",
            "source_revision_equivalence_seal_sha256",
            "source_revision_subject_map",
            "source_revision_subject_map_seal_sha256",
            "evaluation_source_revision_tree_sha256",
            "evaluation_source_revision",
            "training_script_equivalence",
            "evaluation_script_sha256",
            "audit_seal_sha256",
            "leaderboard_seal_sha256",
            "protocol_id",
            "sample_count",
            "ground_truth_count",
            "unsupported_sample_count",
            "delays_ms",
            "conditions",
            "run_count_per_subject",
            "selection_metric",
            "baseline_subjects",
            "baseline_count",
            "candidate_subjects",
            "candidate_count",
            "gate",
            "ranking_key",
            "ranking_direction",
            "performance_ranked_candidates",
            "selection_priority_key",
            "selection_priority_direction",
            "primary_retention_policy",
            "ranked_candidates_semantics",
            "ranked_candidates",
            "selected_candidate",
            "selected_candidate_gate_passed",
            "candidate_results",
            "seal_sha256",
        },
        context="formal selector artifact",
    )
    observed_seal = _sha256(artifact.get("seal_sha256"), "formal selector seal")
    if (
        observed_seal != expected_seal
        or _sealed(artifact)["seal_sha256"] != observed_seal
    ):
        raise ValueError("formal selector artifact seal SHA-256 mismatch")
    expected_values = {
        "document_type": "resilient_v2x_formal_candidate_selection",
        "status": "selected",
        "selection_claim": "provisional_single_seed_1337_bev70_strict_leader",
        "selection_stage": "single_seed_candidate_screening",
        "claim_scope": "metrics_and_clearml_metadata_only",
        "live_snapshot_recheck": {
            "audit_and_leaderboard": ("status_parent_script_parameters_and_artifact"),
            "evaluation_tasks": (
                "status_parent_script_parameters_input_model_and_metrics"
            ),
            "training_tasks": "sealed_comparability_audit_snapshot_only",
        },
        "audit_seal_sha256": expected_audit_seal,
        "training_script_equivalence": dict(expected_training_script_equivalence),
        "evaluation_script_sha256": expected_evaluation_script_sha256,
        "protocol_id": PROTOCOL_ID,
        "conditions": list(CONDITIONS),
        "selection_metric": LEADERSHIP_METRIC,
        "baseline_subjects": list(BASELINE_SUBJECTS),
        "candidate_subjects": list(CANDIDATE_ORDER),
        "ranking_key": list(RANKING_KEY),
        "ranking_direction": (
            "gate_passed_first_then_descending_margins_then_ascending_fixed_order"
        ),
        "selection_priority_key": [
            "primary_retained_when_gate_passes",
            "performance_rank",
        ],
        "selection_priority_direction": (
            "primary_first_if_gate_passes_else_performance_rank"
        ),
        "primary_retention_policy": "resilient_v2x_first_when_its_gate_passes",
        "ranked_candidates_semantics": "selection_priority_rank",
        "selected_candidate": selected_candidate,
    }
    for key, expected in expected_values.items():
        if artifact.get(key) != expected:
            raise ValueError(f"formal selector artifact {key} mismatch")
    expected_ints = {
        "schema_version": 3,
        "sample_count": SAMPLE_COUNT,
        "ground_truth_count": GROUND_TRUTH_COUNT,
        "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        "run_count_per_subject": RUN_COUNT_PER_EVALUATION,
        "baseline_count": len(BASELINE_SUBJECTS),
        "candidate_count": len(CANDIDATE_ORDER),
    }
    for key, expected in expected_ints.items():
        _exact_int(
            artifact.get(key), expected, context=f"formal selector artifact {key}"
        )
    expected_bools = {
        "selection_is_final": False,
        "model_checkpoint_byte_verification_complete": False,
        "requires_checkpoint_byte_audit": True,
        "requires_multiseed_confirmation": True,
        "architecture_revision_required": False,
        "selected_candidate_gate_passed": True,
    }
    for key, expected in expected_bools.items():
        _exact_bool(
            artifact.get(key), expected, context=f"formal selector artifact {key}"
        )
    raw_delays = artifact.get("delays_ms")
    if not isinstance(raw_delays, list) or len(raw_delays) != len(DELAYS_MS):
        raise ValueError("formal selector artifact delays_ms mismatch")
    for index, (observed, expected) in enumerate(
        zip(raw_delays, DELAYS_MS, strict=True), start=1
    ):
        _exact_int(
            observed,
            expected,
            context=f"formal selector artifact delay {index}",
        )
    observed_provenance_task_id = _clearml_id(
        artifact.get("training_provenance_task_id"),
        "formal selector training provenance task",
    )
    if observed_provenance_task_id != expected_training_provenance_task_id:
        raise ValueError("formal selector training provenance task mismatch")
    observed_progress_seal = _sha256(
        artifact.get("training_progress_seal_sha256"),
        "formal selector training progress seal",
    )
    if observed_progress_seal != expected_training_progress_seal:
        raise ValueError("formal selector training progress seal mismatch")
    observed_provenance_seal = _sha256(
        artifact.get("training_provenance_seal_sha256"),
        "formal selector training provenance seal",
    )
    if observed_provenance_seal != expected_training_provenance_seal:
        raise ValueError("formal selector training provenance seal mismatch")
    _validate_source_provenance_binding(
        {
            key: artifact[key]
            for key in (
                "source_revision_equivalence",
                "source_revision_equivalence_seal_sha256",
                "source_revision_subject_map",
                "source_revision_subject_map_seal_sha256",
                "evaluation_source_revision_tree_sha256",
                "evaluation_source_revision",
            )
        },
        context="formal selector",
    )
    observed_equivalence = _validate_training_script_equivalence(
        artifact.get("training_script_equivalence"),
        context="formal selector training script equivalence",
    )
    if _canonical_json(observed_equivalence) != _canonical_json(
        expected_training_script_equivalence
    ):
        raise ValueError("formal selector training script equivalence mismatch")
    observed_evaluation_sha = _sha256(
        artifact.get("evaluation_script_sha256"),
        "formal selector evaluation script",
    )
    if observed_evaluation_sha != expected_evaluation_script_sha256:
        raise ValueError("formal selector evaluation script mismatch")
    dependency_ids = {
        _clearml_id(artifact.get(key), f"formal selector {key}")
        for key in (
            "audit_task_id",
            "leaderboard_task_id",
            "training_controller_task_id",
            "training_provenance_task_id",
            "watcher_task_id",
        )
    }
    if len(dependency_ids) != 5 or selector_task_id in dependency_ids:
        raise ValueError("formal selector task/dependency provenance is not unique")
    _sha256(artifact.get("leaderboard_seal_sha256"), "formal leaderboard seal")
    expected_gate = {
        "conditions_required": "12/12_strictly_leading",
        "aggregate_dimensions_required": ["clean_0ms", "mean_12", "worst_12"],
        "aggregate_comparison": "strictly_greater_than_best_of_14_baselines",
    }
    if artifact.get("gate") != expected_gate:
        raise ValueError("formal selector artifact gate mismatch")
    ranked = artifact.get("ranked_candidates")
    if (
        not isinstance(ranked, list)
        or len(ranked) != len(CANDIDATE_ORDER)
        or set(ranked) != set(CANDIDATE_ORDER)
        or len(set(ranked)) != len(CANDIDATE_ORDER)
    ):
        raise ValueError("formal selector ranked candidates are not a permutation")
    raw_results = artifact.get("candidate_results")
    if not isinstance(raw_results, list) or len(raw_results) != len(CANDIDATE_ORDER):
        raise ValueError("formal selector candidate result count mismatch")
    results: dict[str, dict[str, object]] = {}
    sort_keys: dict[str, tuple[int, float, float, float, float, int]] = {}
    for raw, subject in zip(raw_results, CANDIDATE_ORDER, strict=True):
        result, sort_key = _validate_candidate_result(raw, subject=subject)
        results[subject] = result
        sort_keys[subject] = sort_key
    margin_ranking = sorted(CANDIDATE_ORDER, key=sort_keys.__getitem__)
    primary = CANDIDATE_ORDER[0]
    expected_ranking = (
        [primary] + [subject for subject in margin_ranking if subject != primary]
        if results[primary]["gate_passed"] is True
        else margin_ranking
    )
    passing = [
        subject for subject in expected_ranking if results[subject]["gate_passed"]
    ]
    expected_selected = (
        primary
        if results[primary]["gate_passed"] is True
        else passing[0]
        if passing
        else None
    )
    if (
        artifact.get("performance_ranked_candidates") != margin_ranking
        or ranked != expected_ranking
        or expected_selected != selected_candidate
    ):
        raise ValueError("formal selector ranking/selection is inconsistent")
    performance_rank = {
        subject: rank for rank, subject in enumerate(margin_ranking, start=1)
    }
    selection_rank = {subject: rank for rank, subject in enumerate(ranked, start=1)}
    for subject in CANDIDATE_ORDER:
        if (
            results[subject]["performance_rank"] != performance_rank[subject]
            or results[subject]["selection_priority_rank"] != selection_rank[subject]
            or results[subject]["rank"] != selection_rank[subject]
        ):
            raise ValueError("formal selector candidate rank fields are inconsistent")
    return artifact


def _selector_provenance(
    *,
    selector_task_id: object,
    selector_artifact: object,
    selector_seal: object,
    audit_seal: str,
    training_provenance_task_id: str,
    training_progress_artifact_seal_sha256: str,
    training_provenance_artifact_seal_sha256: str,
    training_script_equivalence: Mapping[str, object],
    evaluation_script_sha256: str,
    selected_candidate: str,
) -> dict[str, object]:
    task_id = _clearml_id(selector_task_id, "formal selector task")
    seal = _sha256(selector_seal, "formal selector seal")
    artifact = _validate_selector_artifact(
        selector_artifact,
        selector_task_id=task_id,
        expected_seal=seal,
        expected_audit_seal=audit_seal,
        expected_training_provenance_task_id=training_provenance_task_id,
        expected_training_progress_seal=(training_progress_artifact_seal_sha256),
        expected_training_provenance_seal=(training_provenance_artifact_seal_sha256),
        expected_training_script_equivalence=training_script_equivalence,
        expected_evaluation_script_sha256=evaluation_script_sha256,
        selected_candidate=selected_candidate,
    )
    return {
        "task_id": task_id,
        "artifact_name": FORMAL_SELECTOR_ARTIFACT,
        "artifact_seal_sha256": seal,
        "selection_stage": "single_seed_candidate_screening",
        "selection_is_final": False,
        "plan_role": "provisional_candidate_order_only_not_final_claim",
        "artifact": artifact,
    }


def _reject_unresolved_ties(values: Sequence[str] | None) -> None:
    if values is None:
        return
    if isinstance(values, (str, bytes)):
        raise ValueError("tied_candidates must be a sequence of canonical names")
    normalized = [_candidate(value, context="tied candidate") for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError("tied_candidates contains a duplicate candidate")
    if normalized:
        raise ValueError(
            "unresolved candidate ties are forbidden; the formal selector must "
            "supply one total-order winner"
        )


def _derived_seed_record(slot: int) -> dict[str, object]:
    seed_input = SEED_INPUT_TEMPLATE.format(
        domain_separator=SEED_DERIVATION_DOMAIN,
        protocol_id=PROTOCOL_ID,
        slot=slot,
    )
    digest = hashlib.sha256(seed_input.encode("utf-8")).hexdigest()
    unsigned_prefix32 = int(digest[:8], 16)
    return {
        "seed_index": slot,
        "input_utf8": seed_input,
        "sha256": digest,
        "unsigned_prefix32": unsigned_prefix32,
        "training_seed": unsigned_prefix32,
    }


def _seed_derivation() -> dict[str, object]:
    derived = [_derived_seed_record(slot) for slot in (2, 3)]
    seeds = [
        TRAINING_OVERLAY_PROTOCOL_SEED,
        *(int(record["training_seed"]) for record in derived),
    ]
    if len(set(seeds)) != 3 or any(seed <= 0 for seed in seeds):
        raise RuntimeError("fixed multi-seed derivation collided or produced zero")
    return {
        "algorithm": SEED_DERIVATION_ALGORITHM,
        "domain_separator": SEED_DERIVATION_DOMAIN,
        "protocol_id": PROTOCOL_ID,
        "input_template": SEED_INPUT_TEMPLATE,
        "anchor_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
        "anchor_seed_origin": "existing_formal_training_seed",
        "derived_seed_records": derived,
        "ordered_training_seeds": seeds,
        "portable_seed_range": "unsigned_32_bit_matching_source_d_seed_contract",
        "collision_policy": SEED_COLLISION_POLICY,
    }


def _seed_specs() -> list[dict[str, object]]:
    training_seeds = _seed_derivation()["ordered_training_seeds"]
    if not isinstance(training_seeds, list):
        raise RuntimeError("internal seed derivation is invalid")
    return [
        {
            "seed_index": 1,
            "seed_role": "formal_anchor",
            "training_seed": training_seeds[0],
            "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "gpu_model": "A100",
            "queue_pool": [
                {"logical_slot": 1, "queue": "GPU4-A100", "gpu_model": "A100"},
                {"logical_slot": 2, "queue": "GPU4-A100", "gpu_model": "A100"},
            ],
            "load_balance_policy": (
                "round_robin_by_global_subject_order_across_two_a100_workers"
            ),
            "queue_execution_semantics": (
                "two_homogeneous_a100_workers_share_GPU4-A100_queue_capacity_2"
            ),
            "actual_worker_identity_audit_required": True,
        },
        {
            "seed_index": 2,
            "seed_role": "protocol_sha256_derived",
            "training_seed": training_seeds[1],
            "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "gpu_model": "V100",
            "queue_pool": [
                {"logical_slot": 1, "queue": "GPU4-V100", "gpu_model": "V100"}
            ],
            "load_balance_policy": "single_homogeneous_gpu_queue",
            "queue_execution_semantics": "one_v100_worker_queue",
            "actual_worker_identity_audit_required": True,
        },
        {
            "seed_index": 3,
            "seed_role": "protocol_sha256_derived",
            "training_seed": training_seeds[2],
            "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
            "gpu_model": "RTX5090",
            "queue_pool": [
                {
                    "logical_slot": 1,
                    "queue": "GPU4-5090",
                    "gpu_model": "RTX5090",
                }
            ],
            "load_balance_policy": "single_homogeneous_gpu_queue",
            "queue_execution_semantics": "one_rtx5090_worker_queue",
            "actual_worker_identity_audit_required": True,
        },
    ]


def _condition_ids() -> list[str]:
    return [
        f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        for delay in DELAYS_MS
        for condition in CONDITIONS
    ]


def _multiseed_gate_contract() -> dict[str, object]:
    return {
        "contract_id": "formal_1337_multiseed_strict_lead_v1",
        "candidate_pool": list(CANDIDATE_ORDER),
        "comparison_pool": list(BASELINE_SUBJECTS),
        "leadership_metric": LEADERSHIP_METRIC,
        "reported_metric_keys": list(AP_METRIC_KEYS),
        "seed_count": 3,
        "condition_ids": _condition_ids(),
        "condition_count_per_seed": RUN_COUNT_PER_EVALUATION,
        "condition_comparison": (
            "candidate_strictly_greater_than_best_of_14_baselines_within_same_seed"
        ),
        "required_condition_wins_per_seed": RUN_COUNT_PER_EVALUATION,
        "required_condition_wins_total": 3 * RUN_COUNT_PER_EVALUATION,
        "required_seed_pass_count": 3,
        "per_seed_aggregate_dimensions": ["clean_0ms", "mean_12", "worst_12"],
        "aggregate_comparison": (
            "candidate_strictly_greater_than_best_of_14_baselines_within_same_seed"
        ),
        "required_aggregate_wins_per_seed": 3,
        "required_aggregate_wins_total": 9,
        "comparison_epsilon": 0.0,
        "tie_policy": "any_equal_value_or_zero_margin_is_failure",
        "pass_rule": (
            "all_36_condition_comparisons_and_all_9_per_seed_aggregate_"
            "comparisons_must_be_strict_wins"
        ),
        "failure_rule": "logical_negation_of_pass_rule",
        "outcome_values": ["passed", "failed"],
    }


def _queue_assignment(
    seed: Mapping[str, object], *, global_subject_index: int
) -> tuple[str, int]:
    queue_pool = seed["queue_pool"]
    if not isinstance(queue_pool, list) or not queue_pool:
        raise RuntimeError("internal queue pool is invalid")
    selected = queue_pool[(global_subject_index - 1) % len(queue_pool)]
    if not isinstance(selected, Mapping):
        raise RuntimeError("internal queue assignment is invalid")
    return str(selected["queue"]), int(selected["logical_slot"])


def _selection_record(
    *,
    candidate: str,
    selector_seal: str,
) -> dict[str, object]:
    return {
        "round": 1,
        "selector_rank": 1,
        "formal_selector_seal_sha256": selector_seal,
        "selected_candidate": candidate,
        "trigger": "initial_formal_selection",
        "previous_candidate": None,
        "previous_outcome": None,
        "tie_status": "resolved_total_order",
        "failure_evidence": None,
    }


def _task_records(
    selection_history: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(selection_history) != 1:
        raise TrustedGateRequiredError(
            "task generation is initial-only until a trusted gate producer exists"
        )
    seeds = _seed_specs()
    candidate = _candidate(
        selection_history[0].get("selected_candidate"),
        context="initial task-generation candidate",
    )
    global_subject_order = [*BASELINE_SUBJECTS, candidate]
    global_index = {
        subject: index for index, subject in enumerate(global_subject_order, start=1)
    }
    training: list[dict[str, object]] = []
    evaluation: list[dict[str, object]] = []
    round_index = 1
    for seed in seeds:
        seed_index = int(seed["seed_index"])
        for subject in global_subject_order:
            queue, queue_slot = _queue_assignment(
                seed,
                global_subject_index=global_index[subject],
            )
            task_key = f"train-r{round_index:02d}-s{seed_index:02d}-{subject}"
            training.append(
                {
                    "index": len(training) + 1,
                    "task_key": task_key,
                    "round": round_index,
                    "subject": subject,
                    "kind": (
                        "external_controlled_baseline"
                        if subject in BASELINE_SUBJECTS
                        else "selected_candidate"
                    ),
                    "seed_index": seed_index,
                    "training_seed": seed["training_seed"],
                    "training_overlay_protocol_seed": (TRAINING_OVERLAY_PROTOCOL_SEED),
                    "gpu_model": seed["gpu_model"],
                    "worker_queue": queue,
                    "queue_pool_slot": queue_slot,
                    "gpus": GPU_COUNT,
                    "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
                    "max_epochs": MAX_EPOCHS,
                    "precision": PRECISION,
                    "global_batch_size": GLOBAL_BATCH_SIZE,
                    "val_interval": VAL_INTERVAL,
                    "checkpoint_policy": CHECKPOINT_POLICY,
                }
            )
            evaluation.append(
                {
                    "index": len(evaluation) + 1,
                    "task_key": (
                        f"eval-r{round_index:02d}-s{seed_index:02d}-{subject}"
                    ),
                    "training_task_key": task_key,
                    "round": round_index,
                    "subject": subject,
                    "kind": (
                        "external_controlled_baseline"
                        if subject in BASELINE_SUBJECTS
                        else "selected_candidate"
                    ),
                    "seed_index": seed_index,
                    "training_seed": seed["training_seed"],
                    "training_overlay_protocol_seed": (TRAINING_OVERLAY_PROTOCOL_SEED),
                    "gpu_model": seed["gpu_model"],
                    "worker_queue": queue,
                    "queue_pool_slot": queue_slot,
                    "checkpoint_policy": CHECKPOINT_POLICY,
                    "protocol_id": PROTOCOL_ID,
                    "run_count": RUN_COUNT_PER_EVALUATION,
                    "condition_ids": _condition_ids(),
                }
            )
    return training, evaluation


def _build_payload(
    *,
    source_d_script_sha256: str,
    source_d_equivalence_sha256: str,
    formal_audit_seal_sha256: str,
    training_provenance_task_id: str,
    training_progress_artifact_seal_sha256: str,
    training_provenance_artifact_seal_sha256: str,
    training_script_equivalence: Mapping[str, object],
    evaluation_script_sha256: str,
    formal_selector_provenance: Mapping[str, object],
    selection_history: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(selection_history) != 1:
        raise TrustedGateRequiredError(
            "formal multi-seed plans are initial-only until a verified "
            "ClearML gate producer and execution manifest exist"
        )
    frozen_provenance = _frozen_mapping(
        formal_selector_provenance, context="formal selector provenance"
    )
    frozen_script_equivalence = _validate_training_script_equivalence(
        training_script_equivalence,
        context="formal multi-seed training script equivalence",
    )
    frozen_history = [
        _frozen_mapping(selection_history[0], context="initial selection history")
    ]
    training, evaluation = _task_records(frozen_history)
    planned_candidates = [str(frozen_history[0]["selected_candidate"])]
    subject_order = [*BASELINE_SUBJECTS, *planned_candidates]
    candidate_task_count = 3
    baseline_task_count = len(BASELINE_SUBJECTS) * 3
    delta_training = [str(item["task_key"]) for item in training]
    delta_evaluation = [str(item["task_key"]) for item in evaluation]
    selector_artifact = frozen_provenance.get("artifact")
    if not isinstance(selector_artifact, Mapping):
        raise ValueError("formal selector provenance artifact is missing")
    ranked_candidates = selector_artifact.get("ranked_candidates")
    if not isinstance(ranked_candidates, list):
        raise ValueError("formal selector ranked candidates are missing")
    return _sealed(
        {
            "schema_version": 2,
            "plan_type": PLAN_TYPE,
            "protocol_id": PROTOCOL_ID,
            "source_d_identity": {
                "script_sha256": source_d_script_sha256,
                "equivalence_artifact_sha256": source_d_equivalence_sha256,
            },
            "formal_audit_seal_sha256": formal_audit_seal_sha256,
            "training_provenance_task_id": training_provenance_task_id,
            "training_progress_artifact_seal_sha256": (
                training_progress_artifact_seal_sha256
            ),
            "training_provenance_artifact_seal_sha256": (
                training_provenance_artifact_seal_sha256
            ),
            "training_script_equivalence": frozen_script_equivalence,
            "evaluation_script_sha256": evaluation_script_sha256,
            "initial_formal_selector_seal_sha256": frozen_history[0][
                "formal_selector_seal_sha256"
            ],
            "formal_selector_provenance": frozen_provenance,
            "initial_selected_candidate": planned_candidates[0],
            "baseline_subjects": list(BASELINE_SUBJECTS),
            "baseline_count": len(BASELINE_SUBJECTS),
            "subject_order": subject_order,
            "subject_count": len(subject_order),
            "candidate_allowlist": list(CANDIDATE_ORDER),
            "candidate_aliases_rejected": dict(CANDIDATE_ALIASES),
            "planned_candidates": planned_candidates,
            "candidate_count": 1,
            "selection_history": frozen_history,
            "selection_tie_policy": (
                "formal_selector_must_resolve_to_one_total_order_no_unresolved_ties"
            ),
            "seed_derivation": _seed_derivation(),
            "seeds": _seed_specs(),
            "seed_count": 3,
            "training_contract": {
                "gpus": GPU_COUNT,
                "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
                "max_epochs": MAX_EPOCHS,
                "precision": PRECISION,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "val_interval": VAL_INTERVAL,
                "checkpoint_policy": CHECKPOINT_POLICY,
                "training_overlay_protocol_seed": TRAINING_OVERLAY_PROTOCOL_SEED,
                "hardware_comparability": "same_seed_all_subjects_same_gpu_model",
            },
            "evaluation_contract": {
                "protocol_id": PROTOCOL_ID,
                "sample_count": SAMPLE_COUNT,
                "ground_truth_count": GROUND_TRUTH_COUNT,
                "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
                "delays_ms": list(DELAYS_MS),
                "conditions": list(CONDITIONS),
                "condition_ids": _condition_ids(),
                "metric_keys": list(AP_METRIC_KEYS),
                "run_count_per_evaluation": RUN_COUNT_PER_EVALUATION,
                "checkpoint_policy": CHECKPOINT_POLICY,
                "hardware_comparability": "same_seed_all_subjects_same_gpu_model",
            },
            "multiseed_gate_contract": _multiseed_gate_contract(),
            "training_tasks": training,
            "evaluation_tasks": evaluation,
            "baseline_training_task_count": baseline_task_count,
            "baseline_evaluation_task_count": baseline_task_count,
            "candidate_training_task_count": candidate_task_count,
            "candidate_evaluation_task_count": candidate_task_count,
            "training_task_count": len(training),
            "evaluation_task_count": len(evaluation),
            "total_evaluation_condition_count": (
                len(evaluation) * RUN_COUNT_PER_EVALUATION
            ),
            "fallback_execution_policy": {
                "implementation_status": FALLBACK_IMPLEMENTATION_STATUS,
                "public_append_api": "unconditionally_fail_closed",
                "cli_append_command": "not_exposed",
                "ordered_candidate_intent": list(ranked_candidates),
                "required_before_enable": [
                    "independently_verified_clearml_gate_producer",
                    "sealed_execution_manifest_with_server_readback_provenance",
                    "adversarial_contract_review",
                ],
                "baseline_reuse_intent": (
                    "never_repeat_the_42_initial_baseline_train_or_eval_tasks"
                ),
            },
            "append_delta": {
                "manifest_type": "initial_round_only",
                "round": 1,
                "predecessor_plan_seal_sha256": None,
                "training_task_keys": delta_training,
                "evaluation_task_keys": delta_evaluation,
                "training_task_count": len(delta_training),
                "evaluation_task_count": len(delta_evaluation),
                "evaluation_condition_count": (
                    len(delta_evaluation) * RUN_COUNT_PER_EVALUATION
                ),
            },
            "execution_policy": {
                "round_1_action": "consume_initial_append_delta_once",
                "fallback_action": FALLBACK_IMPLEMENTATION_STATUS,
                "existing_task_action": (
                    "reuse_existing_by_immutable_task_key_never_clone_or_enqueue_again"
                ),
                "task_key_idempotency": "globally_unique_immutable",
                "full_task_lists_role": "sealed_initial_registry_only",
                "actual_worker_gpu_model_audit_required": True,
                "a100_worker_slot_balance_audit_required": True,
            },
        }
    )


def build_plan(
    *,
    source_d_script_sha256: str,
    source_d_equivalence_sha256: str,
    formal_audit_seal_sha256: str,
    training_provenance_task_id: str,
    training_progress_artifact_seal_sha256: str,
    training_provenance_artifact_seal_sha256: str,
    training_script_equivalence: Mapping[str, object],
    evaluation_script_sha256: str,
    formal_selector_seal_sha256: str,
    formal_selector_task_id: str,
    formal_selector_artifact: Mapping[str, object],
    selected_candidate: str,
    tied_candidates: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build the initial 14-baseline plus one-candidate, three-seed plan."""

    _reject_unresolved_ties(tied_candidates)
    candidate = _candidate(selected_candidate, context="selected candidate")
    source_script = _sha256(source_d_script_sha256, "source-D script")
    equivalence = _sha256(source_d_equivalence_sha256, "source-D equivalence")
    audit_seal = _sha256(formal_audit_seal_sha256, "formal audit seal")
    provenance_task_id = _clearml_id(
        training_provenance_task_id, "formal training provenance task"
    )
    progress_seal = _sha256(
        training_progress_artifact_seal_sha256,
        "formal training progress artifact seal",
    )
    provenance_seal = _sha256(
        training_provenance_artifact_seal_sha256,
        "formal training provenance artifact seal",
    )
    script_equivalence = _validate_training_script_equivalence(
        training_script_equivalence,
        context="formal training script equivalence",
    )
    evaluation_sha = _sha256(evaluation_script_sha256, "formal evaluation script")
    if evaluation_sha != EVALUATION_SCRIPT_SHA256:
        raise ValueError("formal evaluation script semantic contract mismatch")
    selector_seal = _sha256(formal_selector_seal_sha256, "formal selector seal")
    selector_provenance = _selector_provenance(
        selector_task_id=formal_selector_task_id,
        selector_artifact=formal_selector_artifact,
        selector_seal=selector_seal,
        audit_seal=audit_seal,
        training_provenance_task_id=provenance_task_id,
        training_progress_artifact_seal_sha256=progress_seal,
        training_provenance_artifact_seal_sha256=provenance_seal,
        training_script_equivalence=script_equivalence,
        evaluation_script_sha256=evaluation_sha,
        selected_candidate=candidate,
    )
    history = [
        _selection_record(
            candidate=candidate,
            selector_seal=selector_seal,
        )
    ]
    plan = _build_payload(
        source_d_script_sha256=source_script,
        source_d_equivalence_sha256=equivalence,
        formal_audit_seal_sha256=audit_seal,
        training_provenance_task_id=provenance_task_id,
        training_progress_artifact_seal_sha256=progress_seal,
        training_provenance_artifact_seal_sha256=provenance_seal,
        training_script_equivalence=script_equivalence,
        evaluation_script_sha256=evaluation_sha,
        formal_selector_provenance=selector_provenance,
        selection_history=history,
    )
    return validate_plan(plan)


def _validated_history(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != 1:
        raise TrustedGateRequiredError(
            "formal multi-seed selection history must contain exactly the "
            "initial candidate; fallback execution is disabled"
        )
    expected_keys = {
        "round",
        "selector_rank",
        "formal_selector_seal_sha256",
        "selected_candidate",
        "trigger",
        "previous_candidate",
        "previous_outcome",
        "tie_status",
        "failure_evidence",
    }
    raw = _frozen_mapping(value[0], context="initial selection history")
    _require_exact_keys(raw, expected_keys, context="initial selection history")
    _exact_int(raw.get("round"), 1, context="initial selection history round")
    _exact_int(
        raw.get("selector_rank"),
        1,
        context="initial selection history selector rank",
    )
    candidate = _candidate(
        raw.get("selected_candidate"), context="initial selection candidate"
    )
    selector_seal = _sha256(
        raw.get("formal_selector_seal_sha256"), "initial selection selector seal"
    )
    if raw.get("failure_evidence") is not None:
        raise TrustedGateRequiredError(
            "initial-only selection history cannot contain fallback evidence"
        )
    expected = _selection_record(
        candidate=candidate,
        selector_seal=selector_seal,
    )
    if _canonical_json(raw) != _canonical_json(expected):
        raise ValueError("initial selection history contract mismatch")
    return [expected]


def validate_plan(value: Mapping[str, object]) -> dict[str, object]:
    """Validate a plan and return its normalized sealed mapping."""

    frozen = _frozen_mapping(value, context="formal multi-seed plan")
    _require_exact_keys(frozen, _TOP_LEVEL_KEYS, context="formal multi-seed plan")
    observed_seal = _sha256(frozen.get("seal_sha256"), "formal multi-seed plan seal")
    if _sealed(frozen)["seal_sha256"] != observed_seal:
        raise ValueError("formal multi-seed plan seal SHA-256 mismatch")
    source_identity = frozen.get("source_d_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("source-D identity must be an object")
    _require_exact_keys(
        source_identity,
        {"script_sha256", "equivalence_artifact_sha256"},
        context="source-D identity",
    )
    source_script = _sha256(source_identity.get("script_sha256"), "source-D script")
    equivalence = _sha256(
        source_identity.get("equivalence_artifact_sha256"),
        "source-D equivalence artifact",
    )
    audit_seal = _sha256(frozen.get("formal_audit_seal_sha256"), "formal audit seal")
    provenance_task_id = _clearml_id(
        frozen.get("training_provenance_task_id"),
        "formal training provenance task",
    )
    progress_seal = _sha256(
        frozen.get("training_progress_artifact_seal_sha256"),
        "formal training progress artifact seal",
    )
    provenance_seal = _sha256(
        frozen.get("training_provenance_artifact_seal_sha256"),
        "formal training provenance artifact seal",
    )
    script_equivalence = _validate_training_script_equivalence(
        frozen.get("training_script_equivalence"),
        context="formal training script equivalence",
    )
    evaluation_sha = _sha256(
        frozen.get("evaluation_script_sha256"),
        "formal evaluation script",
    )
    if evaluation_sha != EVALUATION_SCRIPT_SHA256:
        raise ValueError("formal evaluation script semantic contract mismatch")
    initial_candidate = _candidate(
        frozen.get("initial_selected_candidate"), context="initial selected candidate"
    )
    raw_provenance = frozen.get("formal_selector_provenance")
    if not isinstance(raw_provenance, Mapping):
        raise ValueError("formal selector provenance must be an object")
    _require_exact_keys(
        raw_provenance,
        {
            "task_id",
            "artifact_name",
            "artifact_seal_sha256",
            "selection_stage",
            "selection_is_final",
            "plan_role",
            "artifact",
        },
        context="formal selector provenance",
    )
    if raw_provenance.get("artifact_name") != FORMAL_SELECTOR_ARTIFACT:
        raise ValueError("formal selector artifact name mismatch")
    _exact_bool(
        raw_provenance.get("selection_is_final"),
        False,
        context="formal selector provenance selection_is_final",
    )
    selector_provenance = _selector_provenance(
        selector_task_id=raw_provenance.get("task_id"),
        selector_artifact=raw_provenance.get("artifact"),
        selector_seal=raw_provenance.get("artifact_seal_sha256"),
        audit_seal=audit_seal,
        training_provenance_task_id=provenance_task_id,
        training_progress_artifact_seal_sha256=progress_seal,
        training_provenance_artifact_seal_sha256=provenance_seal,
        training_script_equivalence=script_equivalence,
        evaluation_script_sha256=evaluation_sha,
        selected_candidate=initial_candidate,
    )
    if _canonical_json(raw_provenance) != _canonical_json(selector_provenance):
        raise ValueError("formal selector provenance drifted")
    history = _validated_history(frozen.get("selection_history"))
    selector_artifact = selector_provenance["artifact"]
    if not isinstance(selector_artifact, Mapping):
        raise RuntimeError("validated selector artifact disappeared")
    ranked_candidates = selector_artifact["ranked_candidates"]
    if not isinstance(ranked_candidates, list):
        raise ValueError("formal selector ranking is missing")
    selector_seal = _sha256(
        selector_provenance["artifact_seal_sha256"], "formal selector seal"
    )
    if (
        history[0]["selected_candidate"] != ranked_candidates[0]
        or history[0]["formal_selector_seal_sha256"] != selector_seal
    ):
        raise ValueError(
            "initial selection does not follow the sealed selector ranking"
        )
    expected_counts = {
        "schema_version": 2,
        "baseline_count": 14,
        "subject_count": 15,
        "candidate_count": 1,
        "seed_count": 3,
        "baseline_training_task_count": 42,
        "baseline_evaluation_task_count": 42,
        "candidate_training_task_count": 3,
        "candidate_evaluation_task_count": 3,
        "training_task_count": 45,
        "evaluation_task_count": 45,
        "total_evaluation_condition_count": 540,
    }
    for key, expected_count in expected_counts.items():
        _exact_int(
            frozen.get(key),
            expected_count,
            context=f"formal multi-seed plan {key}",
        )
    expected = _build_payload(
        source_d_script_sha256=source_script,
        source_d_equivalence_sha256=equivalence,
        formal_audit_seal_sha256=audit_seal,
        training_provenance_task_id=provenance_task_id,
        training_progress_artifact_seal_sha256=progress_seal,
        training_provenance_artifact_seal_sha256=provenance_seal,
        training_script_equivalence=script_equivalence,
        evaluation_script_sha256=evaluation_sha,
        formal_selector_provenance=selector_provenance,
        selection_history=history,
    )
    if _canonical_json(frozen) != _canonical_json(expected):
        raise ValueError("formal multi-seed plan violates the deterministic contract")
    return expected


def append_candidate(
    plan: Mapping[str, object],
    *,
    failed_candidate: str,
    next_candidate: str,
    formal_selector_seal_sha256: str,
    failed_multiseed_gate_task_id: str,
    failed_multiseed_gate_artifact: Mapping[str, object],
    tied_candidates: Sequence[str] | None = None,
) -> dict[str, object]:
    """Fail closed until a trusted independent gate producer exists."""

    _ = (
        plan,
        failed_candidate,
        next_candidate,
        formal_selector_seal_sha256,
        failed_multiseed_gate_task_id,
        failed_multiseed_gate_artifact,
        tied_candidates,
    )
    raise TrustedGateRequiredError(
        "fallback append is disabled until a verified ClearML gate producer "
        "and sealed execution manifest exist"
    )


def _read_plan(path: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read plan JSON {path!r}") from error
    if not isinstance(value, Mapping):
        raise ValueError("plan JSON must contain one object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate the initial plan")
    generate.add_argument("--source-d-script-sha256", required=True)
    generate.add_argument("--source-d-equivalence-sha256", required=True)
    generate.add_argument("--formal-audit-seal-sha256", required=True)
    generate.add_argument("--training-provenance-task-id", required=True)
    generate.add_argument("--training-progress-artifact-seal-sha256", required=True)
    generate.add_argument("--training-provenance-artifact-seal-sha256", required=True)
    generate.add_argument("--training-script-equivalence", required=True)
    generate.add_argument("--evaluation-script-sha256", required=True)
    generate.add_argument("--formal-selector-seal-sha256", required=True)
    generate.add_argument("--formal-selector-task-id", required=True)
    generate.add_argument("--formal-selector-artifact", required=True)
    generate.add_argument("--selected-candidate", required=True)
    validate = commands.add_parser("validate", help="validate an existing plan")
    validate.add_argument("plan")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "generate":
        plan = build_plan(
            source_d_script_sha256=args.source_d_script_sha256,
            source_d_equivalence_sha256=args.source_d_equivalence_sha256,
            formal_audit_seal_sha256=args.formal_audit_seal_sha256,
            training_provenance_task_id=args.training_provenance_task_id,
            training_progress_artifact_seal_sha256=(
                args.training_progress_artifact_seal_sha256
            ),
            training_provenance_artifact_seal_sha256=(
                args.training_provenance_artifact_seal_sha256
            ),
            training_script_equivalence=_read_plan(args.training_script_equivalence),
            evaluation_script_sha256=args.evaluation_script_sha256,
            formal_selector_seal_sha256=args.formal_selector_seal_sha256,
            formal_selector_task_id=args.formal_selector_task_id,
            formal_selector_artifact=_read_plan(args.formal_selector_artifact),
            selected_candidate=args.selected_candidate,
        )
    else:
        plan = validate_plan(_read_plan(args.plan))
    print(_canonical_json(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "BASELINE_SUBJECTS",
    "CANONICAL_TRAINING_SCRIPT_SHA256",
    "CANDIDATE_ALIASES",
    "CANDIDATE_ORDER",
    "EVALUATION_SCRIPT_SHA256",
    "FALLBACK_IMPLEMENTATION_STATUS",
    "LEGACY_TRAINING_SCRIPT_SHA256",
    "PROTOCOL_ID",
    "TRAINING_OVERLAY_PROTOCOL_SEED",
    "TrustedGateRequiredError",
    "append_candidate",
    "build_plan",
    "validate_plan",
)
