#!/usr/bin/env python3
"""Extend the single-seed SOTA selector with the P1 zero-shot screening row.

The canonical selector and candidate-extension leaderboard are consumed
read-only.  A trained eligible candidate may be selected from one successful
run.  P1 checkpoint reuse can only trigger independent 50-epoch training and
formal evaluation; it can never enter the final trained-method table directly.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from allegroai import Task
except ImportError:
    try:
        from clearml import Task
    except ImportError:
        Task = None  # type: ignore[assignment]

from tools.resilient_v2x import (  # noqa: E402
    clearml_1337_candidate_extension_leaderboard as ext_lb,
)
from tools.resilient_v2x import clearml_formal_candidate_selector as formal  # noqa: E402
from tools.resilient_v2x import sota_gate  # noqa: E402
from tools.resilient_v2x import (  # noqa: E402
    clearml_p1_zero_shot_candidate_extension as extension,
)


DEFAULT_PROJECT = "ResilientV2X/Training"
DEFAULT_CANONICAL_SELECTOR_TASK_ID = "b8876e88bb494985a900e3d627439776"
CANONICAL_SELECTION_ARTIFACT = "formal_candidate_selection"
ROUND2_SELECTION_ARTIFACT = "formal_candidate_selection_round2"
ROUND2_DOCUMENT_TYPE = "resilient_v2x_formal_candidate_selection_round2"
ROUND2_CANDIDATE_ORDER = sota_gate.CANDIDATE_ORDER


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-selector-task-id",
        default=DEFAULT_CANONICAL_SELECTOR_TASK_ID,
    )
    parser.add_argument("--candidate-extension-leaderboard-task-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=168.0)
    return parser


def _require_bool(value: object, *, context: str) -> bool:
    if type(value) is not bool:
        raise extension.P1CheckpointReuseError(f"{context} must be boolean")
    return value


def _finite_float(value: object, *, context: str) -> float:
    if type(value) not in {int, float}:
        raise extension.P1CheckpointReuseError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise extension.P1CheckpointReuseError(f"{context} must be finite")
    return result


def _validate_candidate_result(
    value: Mapping[str, object],
    *,
    subject: str,
    fixed_order_index: int,
) -> dict[str, object]:
    if subject not in sota_gate.CANDIDATE_BY_SUBJECT:
        raise extension.P1CheckpointReuseError(
            f"candidate {subject!r} is outside the fixed SOTA registry"
        )
    if value.get("subject") != subject:
        raise extension.P1CheckpointReuseError(
            f"candidate result subject mismatch for {subject}"
        )
    if value.get("fixed_order_index") != fixed_order_index:
        raise extension.P1CheckpointReuseError(
            f"candidate result fixed order mismatch for {subject}"
        )
    spec = sota_gate.CANDIDATE_BY_SUBJECT[subject]
    expected_metadata = {
        "candidate_label": spec.label,
        "evidence_class": spec.evidence_class,
        "weights_retrained": spec.weights_retrained,
        "eligible_for_final_paper_selection": (spec.eligible_for_final_paper_selection),
        "per_condition_lead_required": False,
    }
    for key, expected in expected_metadata.items():
        if value.get(key) != expected:
            raise extension.P1CheckpointReuseError(
                f"candidate {subject} metadata {key!r} mismatch"
            )

    comparisons = value.get("condition_comparisons")
    aggregate = value.get("aggregate_comparisons")
    ranking = value.get("ranking_values")
    if (
        not isinstance(comparisons, list)
        or len(comparisons) != sota_gate.RUN_COUNT
        or not isinstance(aggregate, Mapping)
        or set(aggregate) != {"full_0ms", "mean_12", "worst_12"}
        or not isinstance(ranking, Mapping)
        or set(ranking) != set(sota_gate.RANKING_KEY[1:])
    ):
        raise extension.P1CheckpointReuseError(
            f"candidate result comparison contract is invalid for {subject}"
        )
    expected_pairs = (
        (delay, condition)
        for delay in formal.DELAYS_MS
        for condition in formal.CONDITIONS
    )
    won = 0
    for index, (item, pair) in enumerate(
        zip(comparisons, expected_pairs, strict=True),
        start=1,
    ):
        if not isinstance(item, Mapping):
            raise extension.P1CheckpointReuseError(
                f"candidate {subject} condition {index} is invalid"
            )
        delay, condition = pair
        condition_id = f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        if (
            item.get("condition_id") != condition_id
            or item.get("delay_ms") != delay
            or item.get("condition") != condition
        ):
            raise extension.P1CheckpointReuseError(
                f"candidate {subject} condition order drifted"
            )
        candidate = _finite_float(
            item.get("candidate_value"), context=f"{subject} candidate value"
        )
        baseline = _finite_float(
            item.get("best_baseline_value"), context=f"{subject} baseline value"
        )
        margin = _finite_float(item.get("margin"), context=f"{subject} margin")
        if not math.isclose(margin, candidate - baseline, rel_tol=0.0, abs_tol=1e-12):
            raise extension.P1CheckpointReuseError(
                f"candidate {subject} condition margin is inconsistent"
            )
        strictly_leads = _require_bool(
            item.get("strictly_leads"), context=f"{subject} strictly leads"
        )
        if strictly_leads is not (margin > 0.0):
            raise extension.P1CheckpointReuseError(
                f"candidate {subject} strict-lead flag is inconsistent"
            )
        won += int(strictly_leads)
    if (
        value.get("conditions_won") != won
        or value.get("conditions_total") != sota_gate.RUN_COUNT
        or value.get("conditions_won_fraction") != f"{won}/{sota_gate.RUN_COUNT}"
        or value.get("strictly_leads_all_conditions")
        is not (won == sota_gate.RUN_COUNT)
    ):
        raise extension.P1CheckpointReuseError(
            f"candidate {subject} condition gate is inconsistent"
        )
    aggregate_margins: dict[str, float] = {}
    aggregate_passes = True
    for dimension in ("full_0ms", "mean_12", "worst_12"):
        item = aggregate[dimension]
        if not isinstance(item, Mapping):
            raise extension.P1CheckpointReuseError(
                f"candidate {subject} aggregate {dimension} is invalid"
            )
        candidate = _finite_float(
            item.get("candidate_value"), context=f"{subject} {dimension} candidate"
        )
        baseline = _finite_float(
            item.get("best_baseline_value"), context=f"{subject} {dimension} baseline"
        )
        margin = _finite_float(
            item.get("margin"), context=f"{subject} {dimension} margin"
        )
        if not math.isclose(margin, candidate - baseline, rel_tol=0.0, abs_tol=1e-12):
            raise extension.P1CheckpointReuseError(
                f"candidate {subject} aggregate margin is inconsistent"
            )
        leads = _require_bool(
            item.get("strictly_leads"), context=f"{subject} {dimension} leads"
        )
        if leads is not (margin > 0.0):
            raise extension.P1CheckpointReuseError(
                f"candidate {subject} aggregate lead flag is inconsistent"
            )
        expected_pass = (
            margin >= -sota_gate.FULL_0MS_MAX_DEFICIT
            if dimension == "full_0ms"
            else margin > 0.0
        )
        expected_comparison = (
            "greater_than_or_equal_to_best_minus_0.5"
            if dimension == "full_0ms"
            else "strictly_greater_than_best"
        )
        passes = _require_bool(
            item.get("passes_gate"), context=f"{subject} {dimension} gate"
        )
        if (
            passes is not expected_pass
            or item.get("gate_comparison") != expected_comparison
        ):
            raise extension.P1CheckpointReuseError(
                f"candidate {subject} aggregate gate {dimension} is inconsistent"
            )
        aggregate_passes = aggregate_passes and passes
        aggregate_margins[dimension] = margin
    if value.get("gate_passed") is not aggregate_passes:
        raise extension.P1CheckpointReuseError(
            f"candidate {subject} overall gate is inconsistent"
        )
    expected_ranking = {
        "worst_12_margin": aggregate_margins["worst_12"],
        "mean_12_margin": aggregate_margins["mean_12"],
        "full_0ms_margin": aggregate_margins["full_0ms"],
        "fixed_candidate_order": fixed_order_index,
    }
    for key, expected in expected_ranking.items():
        if key == "fixed_candidate_order":
            if ranking.get(key) != expected:
                raise extension.P1CheckpointReuseError(
                    f"candidate {subject} ranking order is invalid"
                )
        elif not math.isclose(
            _finite_float(ranking.get(key), context=f"{subject} ranking {key}"),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise extension.P1CheckpointReuseError(
                f"candidate {subject} ranking {key} is inconsistent"
            )
    return dict(value)


def validate_canonical_selection(
    value: Mapping[str, object],
    *,
    canonical_leaderboard_task_id: str,
    canonical_leaderboard_seal: str,
) -> tuple[dict[str, dict[str, object]], str, str | None]:
    seal = formal._require_seal(value, context="canonical candidate selection")
    expected = {
        "schema_version": 4,
        "document_type": "resilient_v2x_formal_candidate_selection",
        "selection_stage": "single_seed_1337_sota_selection",
        "leaderboard_task_id": canonical_leaderboard_task_id,
        "leaderboard_seal_sha256": canonical_leaderboard_seal,
        "protocol_id": extension.PROTOCOL_ID,
        "sample_count": extension.SAMPLE_COUNT,
        "candidate_subjects": list(formal.CANDIDATE_ORDER),
        "candidate_count": len(formal.CANDIDATE_ORDER),
    }
    for key, expected_value in expected.items():
        if formal._exact_json_equal(value.get(key), expected_value) is not True:
            raise extension.P1CheckpointReuseError(
                f"canonical candidate selection {key!r} mismatch"
            )
    if any(str(key).endswith("seed_confirmation") for key in value):
        raise extension.P1CheckpointReuseError(
            "canonical candidate selection retains obsolete repeated-seed policy"
        )
    fingerprint = value.get("protocol_evidence_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise extension.P1CheckpointReuseError(
            "canonical candidate selection evidence fingerprint is missing"
        )
    try:
        sota_gate.validate_evidence_fingerprint(
            fingerprint, context="canonical candidate selection"
        )
    except ValueError as error:
        raise extension.P1CheckpointReuseError(str(error)) from error
    raw_results = value.get("candidate_results")
    if not isinstance(raw_results, list) or len(raw_results) != len(
        formal.CANDIDATE_ORDER
    ):
        raise extension.P1CheckpointReuseError(
            "canonical candidate result count mismatch"
        )
    results: dict[str, dict[str, object]] = {}
    for index, (raw, subject) in enumerate(
        zip(raw_results, formal.CANDIDATE_ORDER, strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise extension.P1CheckpointReuseError(
                f"canonical candidate result {subject} is invalid"
            )
        results[subject] = _validate_candidate_result(
            raw,
            subject=subject,
            fixed_order_index=index,
        )
    expected_selected = next(
        (
            subject
            for subject in formal.CANDIDATE_ORDER
            if results[subject]["gate_passed"] is True
        ),
        None,
    )
    selected = value.get("selected_candidate")
    if selected != expected_selected:
        raise extension.P1CheckpointReuseError(
            "canonical selected candidate is inconsistent with its gate"
        )
    expected_final = selected is not None
    if (
        value.get("selection_is_final") is not expected_final
        or value.get("selected_candidate_gate_passed") is not expected_final
        or value.get("architecture_revision_required") is expected_final
        or value.get("recommended_action")
        != ("select_single_seed_winner" if expected_final else "revise_architecture")
    ):
        raise extension.P1CheckpointReuseError(
            "canonical single-seed selection state is inconsistent"
        )
    return results, seal, selected


def _validate_candidate_row(
    value: Mapping[str, object],
) -> dict[tuple[int, str], dict[str, float]]:
    expected = {
        "subject": extension.ZERO_SHOT_CANDIDATE_IDENTITY,
        "display_subject_alias": extension.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS,
        "identity_alias_mapping_seal_sha256": (
            extension.identity_alias_mapping()["seal_sha256"]
        ),
        "kind": "diagnostic_candidate_extension",
        "evidence_class": extension.ZERO_SHOT_EVIDENCE_CLASS,
        "weights_retrained": False,
        "optimization_origin": extension.OPTIMIZATION_ORIGIN,
        "eligible_for_trained_26_method_table": False,
        "config_subject": extension.P1_CONFIG_SUBJECT,
        "checkpoint_subject": extension.P0_CONFIG_SUBJECT,
        "source_task_id": extension.P0_TASK_ID,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise extension.P1CheckpointReuseError(
                f"candidate-extension row {key!r} mismatch"
            )
    extension._task_id(value.get("source_model_id"), context="P0 source model")
    extension._task_id(value.get("evaluation_task_id"), context="P1 evaluation")
    extension._positive_int(
        value.get("source_checkpoint_size_bytes"), context="P0 checkpoint bytes"
    )
    for key in (
        "source_checkpoint_sha256",
        "checkpoint_reuse_binding_seal_sha256",
        "checkpoint_reuse_equivalence_seal_sha256",
        "evaluation_receipt_seal_sha256",
        "metrics_sha256",
    ):
        extension._sha256(value.get(key), context=f"candidate-extension {key}")
    metrics = value.get("metrics")
    runs = value.get("runs")
    if not isinstance(metrics, Mapping) or set(metrics) != set(formal.AP_METRIC_KEYS):
        raise extension.P1CheckpointReuseError(
            "candidate-extension metric summaries are invalid"
        )
    if not isinstance(runs, list) or len(runs) != 12:
        raise extension.P1CheckpointReuseError(
            "candidate-extension run matrix is invalid"
        )
    normalized: dict[tuple[int, str], dict[str, float]] = {}
    expected_pairs = (
        (delay, condition)
        for delay in formal.DELAYS_MS
        for condition in formal.CONDITIONS
    )
    for index, (run, pair) in enumerate(zip(runs, expected_pairs, strict=True), 1):
        if not isinstance(run, Mapping):
            raise extension.P1CheckpointReuseError(
                f"candidate-extension run {index} is invalid"
            )
        extension._require_exact_keys(
            run,
            {"condition_id", "delay_ms", "condition", "metrics"},
            context=f"candidate-extension run {index}",
        )
        delay, condition = pair
        condition_id = f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
        if (
            run.get("condition_id") != condition_id
            or run.get("delay_ms") != delay
            or run.get("condition") != condition
        ):
            raise extension.P1CheckpointReuseError(
                f"candidate-extension run {index} order drifted"
            )
        raw_metrics = run.get("metrics")
        if not isinstance(raw_metrics, Mapping) or set(raw_metrics) != set(
            formal.AP_METRIC_KEYS
        ):
            raise extension.P1CheckpointReuseError(
                f"candidate-extension run {index} metrics are invalid"
            )
        parsed: dict[str, float] = {}
        for key in formal.AP_METRIC_KEYS:
            metric = _finite_float(
                raw_metrics.get(key), context=f"candidate-extension run {index} {key}"
            )
            if not 0.0 <= metric <= 100.0:
                raise extension.P1CheckpointReuseError(
                    f"candidate-extension run {index} {key} is outside [0, 100]"
                )
            parsed[key] = metric
        normalized[pair] = parsed
    expected_summaries = {
        key: formal._metric_summary(normalized, key) for key in formal.AP_METRIC_KEYS
    }
    if formal._exact_json_equal(metrics, expected_summaries) is not True:
        raise extension.P1CheckpointReuseError(
            "candidate-extension metric summaries do not match its runs"
        )
    return normalized


def validate_extension_leaderboard(
    value: Mapping[str, object],
) -> tuple[dict[str, object], str, str, str]:
    seal = extension._require_seal(value, context="candidate-extension leaderboard")
    extension._require_exact_keys(
        value,
        {
            "schema_version",
            "leaderboard_type",
            "protocol_id",
            "sample_count",
            "ground_truth_count",
            "unsupported_sample_count",
            "delays_ms",
            "conditions",
            "run_count_per_subject",
            "protocol_evidence_fingerprint",
            "canonical_26",
            "extension_subject_order",
            "extension_subject_count",
            "identity_alias_mapping",
            "baseline_subjects",
            "metric_keys",
            "candidate",
            "candidate_screening",
            "canonical_result_count_revalidated",
            "conclusion_scope",
            "seal_sha256",
        },
        context="candidate-extension leaderboard",
    )
    expected = {
        "schema_version": 2,
        "leaderboard_type": ext_lb.EXTENSION_LEADERBOARD_TYPE,
        "protocol_id": extension.PROTOCOL_ID,
        "sample_count": extension.SAMPLE_COUNT,
        "ground_truth_count": extension.GROUND_TRUTH_COUNT,
        "unsupported_sample_count": 0,
        "delays_ms": list(extension.DELAYS_MS),
        "conditions": list(extension.CONDITIONS),
        "run_count_per_subject": extension.RUN_COUNT,
        "extension_subject_order": [extension.ZERO_SHOT_CANDIDATE_IDENTITY],
        "extension_subject_count": 1,
        "baseline_subjects": list(formal.BASELINE_SUBJECTS),
        "metric_keys": list(formal.AP_METRIC_KEYS),
        "canonical_result_count_revalidated": len(formal.SUBJECT_ORDER),
    }
    for key, expected_value in expected.items():
        if formal._exact_json_equal(value.get(key), expected_value) is not True:
            raise extension.P1CheckpointReuseError(
                f"candidate-extension leaderboard {key!r} mismatch"
            )
    fingerprint = value.get("protocol_evidence_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise extension.P1CheckpointReuseError(
            "candidate-extension evidence fingerprint is missing"
        )
    try:
        sota_gate.validate_evidence_fingerprint(
            fingerprint, context="candidate-extension leaderboard"
        )
    except ValueError as error:
        raise extension.P1CheckpointReuseError(str(error)) from error
    canonical = value.get("canonical_26")
    alias_mapping = value.get("identity_alias_mapping")
    candidate = value.get("candidate")
    screening = value.get("candidate_screening")
    if not all(
        isinstance(item, Mapping)
        for item in (canonical, alias_mapping, candidate, screening)
    ):
        raise extension.P1CheckpointReuseError(
            "candidate-extension leaderboard nested contracts are invalid"
        )
    assert isinstance(canonical, Mapping)
    assert isinstance(alias_mapping, Mapping)
    assert isinstance(candidate, Mapping)
    assert isinstance(screening, Mapping)
    if (
        canonical.get("artifact") != ext_lb.CANONICAL_LEADERBOARD_ARTIFACT
        or canonical.get("subject_order") != list(formal.SUBJECT_ORDER)
        or canonical.get("subject_count") != len(formal.SUBJECT_ORDER)
        or canonical.get("mutated") is not False
    ):
        raise extension.P1CheckpointReuseError(
            "candidate-extension canonical-26 reference drifted"
        )
    canonical_task_id = extension._task_id(
        canonical.get("leaderboard_task_id"), context="canonical leaderboard"
    )
    canonical_seal = extension._sha256(
        canonical.get("seal_sha256"), context="canonical leaderboard"
    )
    extension._require_exact_keys(
        alias_mapping,
        {
            "artifact",
            "seal_sha256",
            "canonical_identity",
            "display_subject_alias",
            "mapping_cardinality",
            "alias_is_candidate_identity",
        },
        context="candidate-extension identity alias mapping",
    )
    expected_alias_seal = extension.identity_alias_mapping()["seal_sha256"]
    if (
        alias_mapping.get("artifact") != extension.IDENTITY_ALIAS_MAPPING_ARTIFACT
        or alias_mapping.get("seal_sha256") != expected_alias_seal
        or alias_mapping.get("canonical_identity")
        != extension.ZERO_SHOT_CANDIDATE_IDENTITY
        or alias_mapping.get("display_subject_alias")
        != extension.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS
        or alias_mapping.get("mapping_cardinality") != "one_to_one"
        or alias_mapping.get("alias_is_candidate_identity") is not False
        or candidate.get("identity_alias_mapping_seal_sha256") != expected_alias_seal
    ):
        raise extension.P1CheckpointReuseError(
            "candidate-extension identity alias mapping drifted"
        )
    candidate_runs = _validate_candidate_row(candidate)
    validated_screening = _validate_candidate_result(
        screening,
        subject=extension.ZERO_SHOT_CANDIDATE_IDENTITY,
        fixed_order_index=sota_gate.CANDIDATE_ORDER.index(
            extension.ZERO_SHOT_CANDIDATE_IDENTITY
        ),
    )
    comparisons = validated_screening["condition_comparisons"]
    assert isinstance(comparisons, list)
    for item in comparisons:
        assert isinstance(item, Mapping)
        pair = (int(item["delay_ms"]), str(item["condition"]))
        expected_value = candidate_runs[pair][formal.LEADERSHIP_METRIC]
        if not math.isclose(
            float(item["candidate_value"]),
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise extension.P1CheckpointReuseError(
                "candidate screening values do not match extension runs"
            )
    summaries = candidate["metrics"]
    assert isinstance(summaries, Mapping)
    aggregate = validated_screening["aggregate_comparisons"]
    assert isinstance(aggregate, Mapping)
    for dimension in ("full_0ms", "mean_12", "worst_12"):
        summary = summaries[formal.LEADERSHIP_METRIC]
        assert isinstance(summary, Mapping)
        expected_value = (
            summary[dimension]["value"]
            if dimension == "worst_12"
            else summary["clean_0ms"]
            if dimension == "full_0ms"
            else summary[dimension]
        )
        item = aggregate[dimension]
        assert isinstance(item, Mapping)
        if not math.isclose(
            float(item["candidate_value"]),
            float(expected_value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise extension.P1CheckpointReuseError(
                "candidate screening aggregate does not match extension runs"
            )
    return validated_screening, seal, canonical_task_id, canonical_seal


def _margin_sort_key(
    result: Mapping[str, object],
) -> tuple[int, float, float, float, int]:
    try:
        return sota_gate.ranking_sort_key(result)
    except ValueError as error:
        raise extension.P1CheckpointReuseError(str(error)) from error


def build_round2_selection(
    *,
    canonical_selector_task_id: str,
    canonical_selection: Mapping[str, object],
    extension_leaderboard_task_id: str,
    extension_leaderboard: Mapping[str, object],
) -> dict[str, object]:
    selector_task_id = extension._task_id(
        canonical_selector_task_id,
        context="canonical selector task",
    )
    extension_task_id = extension._task_id(
        extension_leaderboard_task_id,
        context="candidate-extension leaderboard task",
    )
    zero_shot, extension_seal, canonical_lb_task_id, canonical_lb_seal = (
        validate_extension_leaderboard(extension_leaderboard)
    )
    trained, canonical_selection_seal, original_selected = validate_canonical_selection(
        canonical_selection,
        canonical_leaderboard_task_id=canonical_lb_task_id,
        canonical_leaderboard_seal=canonical_lb_seal,
    )
    available = {
        **{subject: dict(trained[subject]) for subject in formal.CANDIDATE_ORDER},
        extension.ZERO_SHOT_CANDIDATE_IDENTITY: dict(zero_shot),
    }
    ranked = sorted(available.values(), key=_margin_sort_key)
    passing = [item for item in ranked if item.get("gate_passed") is True]
    eligible_passing = [
        item
        for item in passing
        if item.get("eligible_for_final_paper_selection") is True
    ]
    screening_winner = str(passing[0]["subject"]) if passing else None
    trained_winner = str(eligible_passing[0]["subject"]) if eligible_passing else None
    zero_shot_winner = screening_winner == extension.ZERO_SHOT_CANDIDATE_IDENTITY
    selection_is_final = trained_winner is not None and not zero_shot_winner
    selected_candidate = trained_winner if selection_is_final else None
    pending = [
        subject for subject in ROUND2_CANDIDATE_ORDER if subject not in available
    ]
    rank_by_subject = {
        str(item["subject"]): index for index, item in enumerate(ranked, start=1)
    }
    enriched: list[dict[str, object]] = []
    for subject in ROUND2_CANDIDATE_ORDER:
        spec = sota_gate.CANDIDATE_BY_SUBJECT[subject]
        if subject in available:
            row = dict(available[subject])
            row["evidence_status"] = "validated"
            row["performance_rank"] = rank_by_subject[subject]
            if subject == extension.ZERO_SHOT_CANDIDATE_IDENTITY:
                row["display_subject_alias"] = extension.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS
                row["identity_alias_mapping_seal_sha256"] = (
                    extension.identity_alias_mapping()["seal_sha256"]
                )
        else:
            row = {
                "subject": subject,
                "candidate_label": spec.label,
                "fixed_order_index": sota_gate.CANDIDATE_ORDER.index(subject),
                "evidence_status": "pending",
                "evidence_class": spec.evidence_class,
                "weights_retrained": spec.weights_retrained,
                "eligible_for_final_paper_selection": (
                    spec.eligible_for_final_paper_selection
                ),
                "gate_passed": None,
                "ranking_values": None,
            }
        enriched.append(row)

    if selection_is_final:
        status = "selected"
        action = "select_single_seed_winner"
    elif zero_shot_winner:
        status = "p1_independent_training_required"
        action = "train_p1_50e_then_evaluate"
    elif pending:
        status = "awaiting_candidate_evidence"
        action = "complete_pending_candidate_evidence"
    else:
        status = "architecture_revision_required"
        action = "revise_architecture"
    return extension._sealed(
        {
            "schema_version": 2,
            "document_type": ROUND2_DOCUMENT_TYPE,
            "status": status,
            "selection_claim": (
                "single_seed_1337_bev70_sota_gate_winner"
                if selection_is_final
                else None
            ),
            "selection_stage": "single_seed_1337_sota_selection",
            "selection_is_final": selection_is_final,
            "requires_p1_independent_training": zero_shot_winner,
            "checkpoint_reuse_can_enter_trained_table": False,
            "canonical_selector": {
                "task_id": selector_task_id,
                "artifact": CANONICAL_SELECTION_ARTIFACT,
                "seal_sha256": canonical_selection_seal,
                "selected_candidate": original_selected,
                "mutated": False,
            },
            "candidate_extension_leaderboard": {
                "task_id": extension_task_id,
                "artifact": ext_lb.EXTENSION_LEADERBOARD_ARTIFACT,
                "seal_sha256": extension_seal,
            },
            "candidate_identity_alias_mapping": {
                "artifact": extension.IDENTITY_ALIAS_MAPPING_ARTIFACT,
                "seal_sha256": extension.identity_alias_mapping()["seal_sha256"],
                "canonical_identity": extension.ZERO_SHOT_CANDIDATE_IDENTITY,
                "display_subject_alias": extension.ZERO_SHOT_DISPLAY_SUBJECT_ALIAS,
                "mapping_cardinality": "one_to_one",
                "alias_is_candidate_identity": False,
            },
            "canonical_leaderboard": {
                "task_id": canonical_lb_task_id,
                "seal_sha256": canonical_lb_seal,
                "subject_count": len(formal.SUBJECT_ORDER),
                "mutated": False,
            },
            "protocol_id": extension.PROTOCOL_ID,
            "sample_count": extension.SAMPLE_COUNT,
            "protocol_evidence_fingerprint": sota_gate.evidence_fingerprint(),
            "training_seed": extension.TRAINING_SEED,
            "training_overlay_protocol_seed": (
                extension.TRAINING_OVERLAY_PROTOCOL_SEED
            ),
            "baseline_subjects": list(formal.BASELINE_SUBJECTS),
            "candidate_registry": sota_gate.candidate_registry(),
            "candidate_subjects": list(ROUND2_CANDIDATE_ORDER),
            "candidate_count": len(ROUND2_CANDIDATE_ORDER),
            "evaluated_candidate_count": len(available),
            "pending_candidate_subjects": pending,
            "pending_candidate_count": len(pending),
            "gate": {
                "metric": sota_gate.LEADERSHIP_METRIC,
                "full_0ms": "candidate >= best controlled baseline - 0.5",
                "mean_12": "candidate > best controlled baseline",
                "worst_12": "candidate > best controlled baseline",
                "per_condition_lead_required": False,
            },
            "ranking_key": list(sota_gate.RANKING_KEY),
            "performance_ranked_candidates": [str(item["subject"]) for item in ranked],
            "screening_selected_candidate": screening_winner,
            "screening_selected_candidate_gate_passed": (screening_winner is not None),
            "screening_selected_candidate_eligible_for_final_claim": (
                screening_winner is not None and not zero_shot_winner
            ),
            "selected_trained_candidate": trained_winner,
            "selected_candidate": selected_candidate,
            "recommended_action": action,
            "candidate_results": enriched,
            "claim_scope": (
                "single-seed final selection; zero-shot evidence is screening-only"
            ),
        }
    )


def _wait_for_completed(
    dependencies: Sequence[tuple[str, object]],
    *,
    deadline: float,
    poll_seconds: float,
    monotonic: Callable[[], float],
    sleeper: Callable[[float], None],
) -> None:
    while True:
        pending = False
        for context, task in dependencies:
            extension._reload(task, context=context)
            status = extension._status(task)
            if status == "completed":
                continue
            if status in extension.FAILED_STATUSES:
                raise extension.P1CheckpointReuseError(f"{context} ended as {status!r}")
            if status not in extension.WAITING_STATUSES:
                raise extension.P1CheckpointReuseError(
                    f"{context} has unexpected status {status!r}"
                )
            pending = True
        if not pending:
            return
        if monotonic() >= deadline:
            raise TimeoutError("timed out waiting for round2 selector dependencies")
        sleeper(poll_seconds)


def _publish(task: object, payload: Mapping[str, object]) -> None:
    artifacts = getattr(task, "artifacts", None)
    if isinstance(artifacts, Mapping) and ROUND2_SELECTION_ARTIFACT in artifacts:
        observed = extension._artifact_mapping(
            task,
            ROUND2_SELECTION_ARTIFACT,
            context="round2 selector task",
        )
        if extension._canonical_json(observed) != extension._canonical_json(payload):
            raise extension.P1CheckpointReuseError(
                "existing round2 selection artifact drifted"
            )
        return
    uploader = getattr(task, "upload_artifact", None)
    if not callable(uploader) or not uploader(
        ROUND2_SELECTION_ARTIFACT,
        artifact_object=dict(payload),
        wait_on_upload=True,
    ):
        raise extension.P1CheckpointReuseError(
            "failed to publish round2 candidate selection"
        )
    flusher = getattr(task, "flush", None)
    if callable(flusher):
        flusher(wait_for_uploads=True)
    extension._reload(task, context="round2 selector task")
    observed = extension._artifact_mapping(
        task,
        ROUND2_SELECTION_ARTIFACT,
        context="round2 selector task",
    )
    if extension._canonical_json(observed) != extension._canonical_json(payload):
        raise extension.P1CheckpointReuseError(
            "round2 candidate selection upload readback drifted"
        )


def run(
    args: argparse.Namespace,
    *,
    task_class: object = Task,
    output_task: object | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    if task_class is None:
        raise extension.P1CheckpointReuseError("ClearML runtime is unavailable")
    if (
        type(args.poll_seconds) not in {int, float}
        or not math.isfinite(float(args.poll_seconds))
        or args.poll_seconds <= 0
        or type(args.timeout_hours) not in {int, float}
        or not math.isfinite(float(args.timeout_hours))
        or args.timeout_hours <= 0
    ):
        raise ValueError("poll interval and timeout must be finite and positive")
    selector_id = extension._task_id(
        args.canonical_selector_task_id,
        context="canonical selector task",
    )
    extension_id = extension._task_id(
        args.candidate_extension_leaderboard_task_id,
        context="candidate-extension leaderboard task",
    )
    if selector_id == extension_id:
        raise ValueError("canonical selector and extension leaderboard must differ")
    if output_task is None:
        getter = getattr(task_class, "current_task", None)
        output_task = getter() if callable(getter) else None
    if output_task is None:
        raise extension.P1CheckpointReuseError(
            "round2 selector requires a current ClearML task"
        )
    canonical_task = task_class.get_task(task_id=selector_id)
    extension_task = task_class.get_task(task_id=extension_id)
    deadline = monotonic() + float(args.timeout_hours) * 3600.0
    _wait_for_completed(
        (
            ("canonical candidate selector", canonical_task),
            ("candidate-extension leaderboard", extension_task),
        ),
        deadline=deadline,
        poll_seconds=float(args.poll_seconds),
        monotonic=monotonic,
        sleeper=sleeper,
    )
    canonical_payload = extension._artifact_mapping(
        canonical_task,
        CANONICAL_SELECTION_ARTIFACT,
        context="canonical selector task",
    )
    extension_payload = extension._artifact_mapping(
        extension_task,
        ext_lb.EXTENSION_LEADERBOARD_ARTIFACT,
        context="candidate-extension leaderboard task",
    )
    payload = build_round2_selection(
        canonical_selector_task_id=selector_id,
        canonical_selection=canonical_payload,
        extension_leaderboard_task_id=extension_id,
        extension_leaderboard=extension_payload,
    )

    # Re-read both immutable inputs immediately before publication.
    extension._reload(canonical_task, context="canonical selector task")
    extension._reload(extension_task, context="candidate-extension leaderboard task")
    if (
        extension._status(canonical_task) != "completed"
        or extension._status(extension_task) != "completed"
    ):
        raise extension.P1CheckpointReuseError(
            "round2 selector dependency status changed before publication"
        )
    canonical_final = extension._artifact_mapping(
        canonical_task,
        CANONICAL_SELECTION_ARTIFACT,
        context="canonical selector task",
    )
    extension_final = extension._artifact_mapping(
        extension_task,
        ext_lb.EXTENSION_LEADERBOARD_ARTIFACT,
        context="candidate-extension leaderboard task",
    )
    if extension._canonical_json(canonical_final) != extension._canonical_json(
        canonical_payload
    ) or extension._canonical_json(extension_final) != extension._canonical_json(
        extension_payload
    ):
        raise extension.P1CheckpointReuseError(
            "round2 selector dependency artifact changed before publication"
        )
    _publish(output_task, payload)
    return payload


def main() -> int:
    args = _parser().parse_args()
    if Task is None:
        raise extension.P1CheckpointReuseError("ClearML runtime is unavailable")
    task = Task.init(
        project_name=DEFAULT_PROJECT,
        task_name="ResilientV2X formal candidate selector round2",
        reuse_last_task_id=False,
        output_uri=extension.FILES_SERVER_URI,
    )
    run(args, output_task=task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ROUND2_CANDIDATE_ORDER",
    "ROUND2_SELECTION_ARTIFACT",
    "build_round2_selection",
    "run",
    "validate_canonical_selection",
    "validate_extension_leaderboard",
)
