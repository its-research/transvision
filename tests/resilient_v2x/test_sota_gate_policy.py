from __future__ import annotations

import copy

import pytest

from tools.resilient_v2x import sota_gate


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


def _baselines(value: float = 50.0):
    return {subject: _runs(value) for subject in sota_gate.BASELINE_SUBJECTS}


def _full_strong_baselines():
    values = [50.0, *([40.0] * (sota_gate.RUN_COUNT - 1))]
    return {subject: _runs(values) for subject in sota_gate.BASELINE_SUBJECTS}


def test_registry_and_controlled_baseline_pool_are_fixed() -> None:
    assert sota_gate.BASELINE_SUBJECTS == (
        "ffnet",
        "coformernet",
        "v2x_vit",
        "cobevt",
        "bevfusion",
    )
    assert [item["label"] for item in sota_gate.candidate_registry()] == [
        "formal_main",
        "E1",
        "E2",
        "E3",
        "P0",
        "P2",
        "P1_zero_shot",
    ]
    assert sota_gate.CANDIDATE_ORDER == (
        "resilient_v2x",
        "support_residual_linear",
        "no_reliability_linear",
        "support_residual_no_reliability",
        "support_residual_no_reliability_linear",
        "support_residual_no_reliability_linear_bbox25",
        sota_gate.P1_ZERO_SHOT_SUBJECT,
    )
    p1 = sota_gate.candidate_registry()[-1]
    assert p1["weights_retrained"] is False
    assert p1["eligible_for_final_paper_selection"] is False


def test_gate_does_not_require_leading_each_condition() -> None:
    values = [49.5, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 50.1]
    result = sota_gate.evaluate_candidate(
        "resilient_v2x", _runs(values), _full_strong_baselines()
    )
    assert result["conditions_won_fraction"] == "11/12"
    assert result["strictly_leads_all_conditions"] is False
    assert result["per_condition_lead_required"] is False
    assert result["gate_passed"] is True
    assert result["aggregate_comparisons"]["full_0ms"]["passes_gate"] is True


@pytest.mark.parametrize("full", [49.5, 50.0, 55.0])
def test_full_0ms_allows_exactly_half_point_deficit(full: float) -> None:
    values = [55.0] * sota_gate.RUN_COUNT
    values[0] = full
    result = sota_gate.evaluate_candidate(
        "resilient_v2x", _runs(values), _full_strong_baselines()
    )
    assert result["gate_passed"] is True


def test_full_0ms_rejects_more_than_half_point_deficit() -> None:
    values = [55.0] * sota_gate.RUN_COUNT
    values[0] = 49.499999
    result = sota_gate.evaluate_candidate(
        "resilient_v2x", _runs(values), _full_strong_baselines()
    )
    assert result["gate_passed"] is False


@pytest.mark.parametrize("failure", ["mean", "worst"])
def test_mean_and_worst_must_strictly_exceed_best_baseline(failure: str) -> None:
    if failure == "mean":
        values = [50.0] * sota_gate.RUN_COUNT
    else:
        values = [51.0] * sota_gate.RUN_COUNT
        values[-1] = 50.0
    result = sota_gate.evaluate_candidate("resilient_v2x", _runs(values), _baselines())
    assert result["gate_passed"] is False
    assert result["aggregate_comparisons"][f"{failure}_12"]["passes_gate"] is False


def test_ranking_uses_worst_then_mean_then_full_then_registry_order() -> None:
    def row(subject: str, worst: float, mean: float, full: float):
        return {
            "gate_passed": True,
            "ranking_values": {
                "worst_12_margin": worst,
                "mean_12_margin": mean,
                "full_0ms_margin": full,
                "fixed_candidate_order": sota_gate.CANDIDATE_ORDER.index(subject),
            },
            "subject": subject,
        }

    results = [
        row("resilient_v2x", 1.0, 99.0, 99.0),
        row("support_residual_linear", 2.0, 1.0, 99.0),
        row("no_reliability_linear", 2.0, 2.0, 1.0),
        row("support_residual_no_reliability", 2.0, 2.0, 2.0),
        row("support_residual_no_reliability_linear", 2.0, 2.0, 2.0),
    ]
    assert [
        row["subject"] for row in sorted(results, key=sota_gate.ranking_sort_key)
    ] == [
        "support_residual_no_reliability",
        "support_residual_no_reliability_linear",
        "no_reliability_linear",
        "support_residual_linear",
        "resilient_v2x",
    ]


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
def test_protocol_count_hash_or_evidence_drift_fails_closed(
    field: str, value: object
) -> None:
    fingerprint = copy.deepcopy(sota_gate.evidence_fingerprint())
    fingerprint[field] = value
    with pytest.raises(ValueError, match="evidence fingerprint mismatch"):
        sota_gate.validate_evidence_fingerprint(fingerprint, context="candidate")


def test_missing_evidence_field_fails_closed() -> None:
    fingerprint = sota_gate.evidence_fingerprint()
    fingerprint.pop("sample_ids_sha256")
    with pytest.raises(ValueError, match="evidence keys mismatch"):
        sota_gate.validate_evidence_fingerprint(fingerprint, context="candidate")


def test_incomplete_condition_evidence_fails_closed() -> None:
    candidate = _runs(51.0)
    candidate.pop((300, "C-Fail"))
    with pytest.raises(ValueError, match="condition set mismatch"):
        sota_gate.evaluate_candidate("resilient_v2x", candidate, _baselines())


def test_baseline_pool_mismatch_fails_closed() -> None:
    baselines = _baselines()
    baselines["ego_only"] = _runs(100.0)
    with pytest.raises(ValueError, match="fixed five-method order"):
        sota_gate.evaluate_candidate("resilient_v2x", _runs(51.0), baselines)
