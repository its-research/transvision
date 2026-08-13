from __future__ import annotations

import copy

import pytest

from tools.resilient_v2x import sota_gate, sota_selector


def _runs(values: float | list[float]) -> dict[tuple[int, str], dict[str, float]]:
    ordered = [values] * sota_gate.RUN_COUNT if isinstance(values, float) else values
    return {
        (delay, condition): {sota_gate.LEADERSHIP_METRIC: float(value)}
        for (delay, condition), value in zip(
            (
                (delay, condition)
                for delay in sota_gate.DELAYS_MS
                for condition in sota_gate.CONDITIONS
            ),
            ordered,
            strict=True,
        )
    }


def _baselines():
    values = [40.0] * sota_gate.RUN_COUNT
    values[0] = 50.0
    return {subject: _runs(values) for subject in sota_gate.BASELINE_SUBJECTS}


def _evidence(*candidates: str):
    return {
        subject: copy.deepcopy(sota_gate.evidence_fingerprint())
        for subject in (*sota_gate.BASELINE_SUBJECTS, *candidates)
    }


def _passing(full: float, other: float, worst: float | None = None):
    values = [other] * sota_gate.RUN_COUNT
    values[0] = full
    if worst is not None:
        values[-1] = worst
    return _runs(values)


def test_missing_candidate_evidence_is_explicitly_pending() -> None:
    subject = "resilient_v2x"
    payload = sota_selector.build_selection(
        candidate_runs_by_subject={subject: _passing(49.5, 45.0)},
        baseline_runs_by_subject=_baselines(),
        evidence_fingerprints_by_subject=_evidence(subject),
    )
    assert payload["candidate_subjects"] == list(sota_gate.CANDIDATE_ORDER)
    assert payload["candidate_count"] == 7
    assert payload["evaluated_candidate_count"] == 1
    assert payload["pending_candidate_count"] == 6
    assert payload["screening_selected_candidate"] == subject
    assert payload["selected_trained_candidate"] == subject
    assert payload["selected_candidate"] == subject
    assert payload["selection_is_final"] is True
    assert payload["recommended_action"] == "select_single_seed_winner"
    pending = [
        row
        for row in payload["candidate_results"]
        if row["evidence_status"] == "pending"
    ]
    assert len(pending) == 6
    assert all(row["gate_passed"] is None for row in pending)


def test_multiple_passers_follow_worst_mean_full_and_fixed_order() -> None:
    e1 = "support_residual_linear"
    e2 = "no_reliability_linear"
    e3 = "support_residual_no_reliability"
    candidates = {
        "resilient_v2x": _passing(60.0, 50.0, 41.0),
        e1: _passing(60.0, 50.0, 42.0),
        e2: _passing(60.0, 51.0, 42.0),
        e3: _passing(61.0, 51.0, 42.0),
        "support_residual_no_reliability_linear": _passing(61.0, 51.0, 42.0),
    }
    payload = sota_selector.build_selection(
        candidate_runs_by_subject=candidates,
        baseline_runs_by_subject=_baselines(),
        evidence_fingerprints_by_subject=_evidence(*candidates),
    )
    assert payload["performance_ranked_candidates"] == [
        e3,
        "support_residual_no_reliability_linear",
        e2,
        e1,
        "resilient_v2x",
    ]
    assert payload["screening_selected_candidate"] == e3
    assert payload["selected_trained_candidate"] == e3
    assert payload["selected_candidate"] == e3
    assert payload["selection_is_final"] is True


def test_p1_zero_shot_can_win_screening_but_never_trained_selection() -> None:
    p1 = sota_gate.P1_ZERO_SHOT_SUBJECT
    main = "resilient_v2x"
    payload = sota_selector.build_selection(
        candidate_runs_by_subject={
            main: _passing(51.0, 45.0),
            p1: _passing(60.0, 55.0),
        },
        baseline_runs_by_subject=_baselines(),
        evidence_fingerprints_by_subject=_evidence(main, p1),
    )
    assert payload["screening_selected_candidate"] == p1
    assert payload["screening_selected_candidate_eligible_for_final_claim"] is False
    assert payload["selected_trained_candidate"] == main
    assert payload["selected_candidate"] is None
    assert payload["selection_is_final"] is False
    assert payload["requires_p1_independent_training"] is True
    assert payload["recommended_action"] == "train_p1_50e_then_evaluate"
    p1_row = next(row for row in payload["candidate_results"] if row["subject"] == p1)
    assert p1_row["eligible_for_final_paper_selection"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_id", "wrong"),
        ("sample_count", 1336),
        ("manifest_content_sha256", "0" * 64),
        ("overlay_index_content_sha256", "1" * 64),
        ("sample_ids_sha256", "2" * 64),
        ("fingerprint_sha256", "3" * 64),
    ],
)
def test_any_available_subject_evidence_drift_is_terminal(
    field: str, value: object
) -> None:
    main = "resilient_v2x"
    evidence = _evidence(main)
    evidence[main][field] = value
    with pytest.raises(ValueError, match="evidence fingerprint mismatch"):
        sota_selector.build_selection(
            candidate_runs_by_subject={main: _passing(51.0, 45.0)},
            baseline_runs_by_subject=_baselines(),
            evidence_fingerprints_by_subject=evidence,
        )


def test_missing_or_extra_evidence_is_terminal_not_pending() -> None:
    main = "resilient_v2x"
    evidence = _evidence(main)
    evidence.pop("ffnet")
    with pytest.raises(ValueError, match="evidence inventory mismatch"):
        sota_selector.build_selection(
            candidate_runs_by_subject={main: _passing(51.0, 45.0)},
            baseline_runs_by_subject=_baselines(),
            evidence_fingerprints_by_subject=evidence,
        )


def test_unknown_candidate_is_terminal() -> None:
    with pytest.raises(ValueError, match="unknown candidates"):
        sota_selector.build_selection(
            candidate_runs_by_subject={"unknown": _passing(60.0, 55.0)},
            baseline_runs_by_subject=_baselines(),
            evidence_fingerprints_by_subject=_evidence(),
        )
