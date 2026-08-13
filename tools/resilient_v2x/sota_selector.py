#!/usr/bin/env python3
"""Evidence-bound selection over the fixed ResilientV2X SOTA registry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

try:
    from tools.resilient_v2x import sota_gate
except ModuleNotFoundError:  # pragma: no cover - standalone execution
    import sota_gate  # type: ignore[no-redef]


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"selection is outside the canonical JSON domain: {error}"
        ) from error


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("seal_sha256", None)
    result["seal_sha256"] = hashlib.sha256(
        _canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def _validate_inputs(
    *,
    candidate_runs_by_subject: Mapping[
        str, Mapping[tuple[int, str], Mapping[str, float]]
    ],
    baseline_runs_by_subject: Mapping[
        str, Mapping[tuple[int, str], Mapping[str, float]]
    ],
    evidence_fingerprints_by_subject: Mapping[str, Mapping[str, object]],
) -> None:
    if tuple(baseline_runs_by_subject) != sota_gate.BASELINE_SUBJECTS:
        raise ValueError("SOTA selector baseline run inventory drifted")
    unknown = set(candidate_runs_by_subject) - set(sota_gate.CANDIDATE_ORDER)
    if unknown:
        raise ValueError(f"SOTA selector has unknown candidates: {sorted(unknown)!r}")
    expected_evidence = set(sota_gate.BASELINE_SUBJECTS) | set(
        candidate_runs_by_subject
    )
    if set(evidence_fingerprints_by_subject) != expected_evidence:
        missing = sorted(expected_evidence - set(evidence_fingerprints_by_subject))
        extra = sorted(set(evidence_fingerprints_by_subject) - expected_evidence)
        raise ValueError(
            "SOTA selector evidence inventory mismatch; "
            f"missing={missing!r}, extra={extra!r}"
        )
    for subject in (
        *sota_gate.BASELINE_SUBJECTS,
        *(
            candidate
            for candidate in sota_gate.CANDIDATE_ORDER
            if candidate in candidate_runs_by_subject
        ),
    ):
        sota_gate.validate_evidence_fingerprint(
            evidence_fingerprints_by_subject[subject], context=subject
        )


def _pending_row(subject: str) -> dict[str, object]:
    spec = sota_gate.CANDIDATE_BY_SUBJECT[subject]
    return {
        "subject": subject,
        "candidate_label": spec.label,
        "fixed_order_index": sota_gate.CANDIDATE_ORDER.index(subject),
        "evidence_status": "pending",
        "evidence_class": spec.evidence_class,
        "weights_retrained": spec.weights_retrained,
        "eligible_for_final_paper_selection": (spec.eligible_for_final_paper_selection),
        "gate_passed": None,
        "ranking_values": None,
    }


def build_selection(
    *,
    candidate_runs_by_subject: Mapping[
        str, Mapping[tuple[int, str], Mapping[str, float]]
    ],
    baseline_runs_by_subject: Mapping[
        str, Mapping[tuple[int, str], Mapping[str, float]]
    ],
    evidence_fingerprints_by_subject: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Build a sealed selection; missing candidates stay explicitly pending."""

    _validate_inputs(
        candidate_runs_by_subject=candidate_runs_by_subject,
        baseline_runs_by_subject=baseline_runs_by_subject,
        evidence_fingerprints_by_subject=evidence_fingerprints_by_subject,
    )
    evaluated = {
        subject: {
            **sota_gate.evaluate_candidate(
                subject,
                candidate_runs_by_subject[subject],
                baseline_runs_by_subject,
            ),
            "evidence_status": "validated",
            "evidence_fingerprint_sha256": evidence_fingerprints_by_subject[subject][
                "fingerprint_sha256"
            ],
        }
        for subject in sota_gate.CANDIDATE_ORDER
        if subject in candidate_runs_by_subject
    }
    ranked = sorted(evaluated.values(), key=sota_gate.ranking_sort_key)
    passing = [row for row in ranked if row["gate_passed"] is True]
    eligible_passing = [
        row for row in passing if row["eligible_for_final_paper_selection"] is True
    ]
    screening_winner = str(passing[0]["subject"]) if passing else None
    trained_winner = str(eligible_passing[0]["subject"]) if eligible_passing else None
    zero_shot_winner = screening_winner == sota_gate.P1_ZERO_SHOT_SUBJECT
    pending = [
        subject for subject in sota_gate.CANDIDATE_ORDER if subject not in evaluated
    ]
    rows = [
        dict(evaluated[subject]) if subject in evaluated else _pending_row(subject)
        for subject in sota_gate.CANDIDATE_ORDER
    ]

    selection_is_final = trained_winner is not None and not zero_shot_winner
    selected_candidate = trained_winner if selection_is_final else None
    if selection_is_final:
        status = "selected"
    elif zero_shot_winner:
        status = "p1_independent_training_required"
    elif pending:
        status = "awaiting_candidate_evidence"
    else:
        status = "architecture_revision_required"
    if zero_shot_winner:
        action = "train_p1_50e_then_evaluate"
    elif trained_winner is not None:
        action = "select_single_seed_winner"
    elif pending:
        action = "complete_pending_candidate_evidence"
    else:
        action = "revise_architecture"

    return _sealed(
        {
            "schema_version": 1,
            "document_type": "resilient_v2x_sota_candidate_selection",
            "status": status,
            "selection_stage": "single_seed_1337_sota_screening",
            "selection_is_final": selection_is_final,
            "protocol_evidence_fingerprint": sota_gate.evidence_fingerprint(),
            "baseline_subjects": list(sota_gate.BASELINE_SUBJECTS),
            "baseline_display_names": dict(sota_gate.BASELINE_DISPLAY_NAMES),
            "candidate_registry": sota_gate.candidate_registry(),
            "candidate_subjects": list(sota_gate.CANDIDATE_ORDER),
            "candidate_count": len(sota_gate.CANDIDATE_ORDER),
            "evaluated_candidate_count": len(evaluated),
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
            "performance_ranked_candidates": [str(row["subject"]) for row in ranked],
            "screening_selected_candidate": screening_winner,
            "screening_selected_candidate_eligible_for_final_claim": (
                screening_winner is not None
                and screening_winner != sota_gate.P1_ZERO_SHOT_SUBJECT
            ),
            "selected_trained_candidate": trained_winner,
            "selected_candidate": selected_candidate,
            "requires_p1_independent_training": zero_shot_winner,
            "checkpoint_reuse_can_enter_trained_table": False,
            "recommended_action": action,
            "candidate_results": rows,
        }
    )


__all__ = ("build_selection",)
