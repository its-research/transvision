#!/usr/bin/env python3
"""Pure, fail-closed SOTA screening contract for ResilientV2X.

The ClearML selectors deliberately delegate all score arithmetic and candidate
ordering to this module.  Keeping the policy independent from remote task I/O
makes it possible to test the scientific gate without mocking ClearML.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass


PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
SAMPLE_COUNT = 1_337
GROUND_TRUTH_COUNT = 11_330
UNSUPPORTED_SAMPLE_COUNT = 0
MANIFEST_CONTENT_SHA256 = (
    "715ac6f7a14225e20327eed0650c55abdc0cb98431830164e84545238099645d"
)
OVERLAY_INDEX_CONTENT_SHA256 = (
    "77bd4585dbb02901f862b8da6aa208a504674b824a3d55cf15005aacbeeeaaff"
)
SAMPLE_IDS_SHA256 = "a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a"
DELAYS_MS = (0, 100, 200, 300)
CONDITIONS = ("Full", "L-Fail", "C-Fail")
RUN_COUNT = len(DELAYS_MS) * len(CONDITIONS)
LEADERSHIP_METRIC = "resilient_v2x/car_bev_ap_r40_0.70"

# These are the five controlled adaptations used by the paper's SOTA claim.
# Other controlled methods may remain on the descriptive leaderboard, but they
# are intentionally outside this gate's comparison pool.
BASELINE_SUBJECTS = (
    "ffnet",
    "coformernet",
    "v2x_vit",
    "cobevt",
    "bevfusion",
)
BASELINE_DISPLAY_NAMES = {
    "ffnet": "FFNet (controlled adaptation)",
    "coformernet": "CoFormerNet (controlled adaptation)",
    "v2x_vit": "V2X-ViT (controlled adaptation)",
    "cobevt": "CoBEVT (controlled adaptation)",
    "bevfusion": "BEVFusion (controlled adaptation)",
}

P1_ZERO_SHOT_SUBJECT = (
    "dair_improvement_reliability_gated_residual_zero_shot_p0_epoch50_final"
)


@dataclass(frozen=True)
class CandidateSpec:
    subject: str
    label: str
    evidence_class: str
    weights_retrained: bool
    eligible_for_final_paper_selection: bool


CANDIDATE_SPECS = (
    CandidateSpec(
        "resilient_v2x",
        "formal_main",
        "formal_trained_50e_candidate",
        True,
        True,
    ),
    CandidateSpec(
        "support_residual_linear",
        "E1",
        "formal_trained_50e_candidate",
        True,
        True,
    ),
    CandidateSpec(
        "no_reliability_linear",
        "E2",
        "formal_trained_50e_candidate",
        True,
        True,
    ),
    CandidateSpec(
        "support_residual_no_reliability",
        "E3",
        "formal_trained_50e_candidate",
        True,
        True,
    ),
    CandidateSpec(
        "support_residual_no_reliability_linear",
        "P0",
        "formal_trained_50e_candidate",
        True,
        True,
    ),
    CandidateSpec(
        "support_residual_no_reliability_linear_bbox25",
        "P2",
        "formal_trained_50e_candidate",
        True,
        True,
    ),
    CandidateSpec(
        P1_ZERO_SHOT_SUBJECT,
        "P1_zero_shot",
        "inference_only_checkpoint_reuse",
        False,
        False,
    ),
)
CANDIDATE_ORDER = tuple(spec.subject for spec in CANDIDATE_SPECS)
CANDIDATE_BY_SUBJECT = {spec.subject: spec for spec in CANDIDATE_SPECS}

FULL_0MS_MAX_DEFICIT = 0.5
RANKING_KEY = (
    "gate_passed",
    "worst_12_margin",
    "mean_12_margin",
    "full_0ms_margin",
    "fixed_candidate_order",
)


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
            f"value is outside the canonical JSON domain: {error}"
        ) from error


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def evidence_fingerprint() -> dict[str, object]:
    """Return the exact evaluation identity required by the SOTA gate."""

    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "sample_count": SAMPLE_COUNT,
        "ground_truth_count": GROUND_TRUTH_COUNT,
        "unsupported_sample_count": UNSUPPORTED_SAMPLE_COUNT,
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "overlay_index_content_sha256": OVERLAY_INDEX_CONTENT_SHA256,
        "sample_ids_sha256": SAMPLE_IDS_SHA256,
        "delays_ms": list(DELAYS_MS),
        "conditions": list(CONDITIONS),
        "run_count": RUN_COUNT,
        "leadership_metric": LEADERSHIP_METRIC,
    }
    payload["fingerprint_sha256"] = _content_sha256(payload)
    return payload


def validate_evidence_fingerprint(
    value: Mapping[str, object], *, context: str
) -> dict[str, object]:
    """Reject partial, stale, or mixed-protocol evidence."""

    expected = evidence_fingerprint()
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        raise ValueError(
            f"{context} evidence keys mismatch; missing={missing!r}, extra={extra!r}"
        )
    if _canonical_json(dict(value)) != _canonical_json(expected):
        raise ValueError(f"{context} evidence fingerprint mismatch")
    return expected


def candidate_registry() -> list[dict[str, object]]:
    """Materialize the fixed candidate order as canonical JSON values."""

    return [
        {**asdict(spec), "fixed_order_index": index}
        for index, spec in enumerate(CANDIDATE_SPECS)
    ]


def _finite_metric(value: object, *, context: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _validate_runs(
    runs: Mapping[tuple[int, str], Mapping[str, float]], *, context: str
) -> dict[tuple[int, str], float]:
    expected = {(delay, condition) for delay in DELAYS_MS for condition in CONDITIONS}
    if set(runs) != expected:
        missing = sorted(expected - set(runs))
        extra = sorted(set(runs) - expected)
        raise ValueError(
            f"{context} condition set mismatch; missing={missing!r}, extra={extra!r}"
        )
    normalized: dict[tuple[int, str], float] = {}
    for key in ((delay, condition) for delay in DELAYS_MS for condition in CONDITIONS):
        metrics = runs[key]
        if not isinstance(metrics, Mapping) or LEADERSHIP_METRIC not in metrics:
            raise ValueError(f"{context} {key!r} lacks {LEADERSHIP_METRIC}")
        normalized[key] = _finite_metric(
            metrics[LEADERSHIP_METRIC], context=f"{context} {key!r} metric"
        )
    return normalized


def _summary(values: Mapping[tuple[int, str], float], dimension: str) -> float:
    ordered = [
        values[(delay, condition)] for delay in DELAYS_MS for condition in CONDITIONS
    ]
    if dimension == "full_0ms":
        return values[(0, "Full")]
    if dimension == "mean_12":
        return math.fsum(ordered) / len(ordered)
    if dimension == "worst_12":
        return min(ordered)
    raise ValueError(f"unsupported aggregate dimension {dimension!r}")


def _best_baseline(values: Mapping[str, float]) -> tuple[float, list[str]]:
    if tuple(values) != BASELINE_SUBJECTS:
        raise ValueError("SOTA baseline order or membership drifted")
    best = max(values.values())
    subjects = [subject for subject in BASELINE_SUBJECTS if values[subject] == best]
    return best, subjects


def evaluate_candidate(
    subject: str,
    candidate_runs: Mapping[tuple[int, str], Mapping[str, float]],
    baseline_runs: Mapping[str, Mapping[tuple[int, str], Mapping[str, float]]],
) -> dict[str, object]:
    """Evaluate one candidate under the paper's three-part BEV AP@0.7 gate."""

    if subject not in CANDIDATE_BY_SUBJECT:
        raise ValueError(f"candidate {subject!r} is outside the fixed registry")
    if tuple(baseline_runs) != BASELINE_SUBJECTS:
        raise ValueError(
            "SOTA baseline evidence must match the fixed five-method order"
        )
    candidate = _validate_runs(candidate_runs, context=f"candidate {subject}")
    baselines = {
        baseline: _validate_runs(
            baseline_runs[baseline], context=f"baseline {baseline}"
        )
        for baseline in BASELINE_SUBJECTS
    }

    comparisons: list[dict[str, object]] = []
    for delay in DELAYS_MS:
        for condition in CONDITIONS:
            baseline_values = {
                baseline: baselines[baseline][(delay, condition)]
                for baseline in BASELINE_SUBJECTS
            }
            best_value, best_subjects = _best_baseline(baseline_values)
            candidate_value = candidate[(delay, condition)]
            margin = candidate_value - best_value
            comparisons.append(
                {
                    "condition_id": (
                        f"delay_{delay:03d}_{condition.lower().replace('-', '_')}"
                    ),
                    "delay_ms": delay,
                    "condition": condition,
                    "candidate_value": candidate_value,
                    "best_baseline_value": best_value,
                    "best_baseline_subject": best_subjects[0],
                    "best_baseline_subjects": best_subjects,
                    "margin": margin,
                    "strictly_leads": margin > 0.0,
                }
            )

    aggregate: dict[str, dict[str, object]] = {}
    for dimension in ("full_0ms", "mean_12", "worst_12"):
        baseline_values = {
            baseline: _summary(baselines[baseline], dimension)
            for baseline in BASELINE_SUBJECTS
        }
        best_value, best_subjects = _best_baseline(baseline_values)
        candidate_value = _summary(candidate, dimension)
        margin = candidate_value - best_value
        if dimension == "full_0ms":
            passes_gate = margin >= -FULL_0MS_MAX_DEFICIT
            comparison = "greater_than_or_equal_to_best_minus_0.5"
        else:
            passes_gate = margin > 0.0
            comparison = "strictly_greater_than_best"
        aggregate[dimension] = {
            "candidate_value": candidate_value,
            "best_baseline_value": best_value,
            "best_baseline_subject": best_subjects[0],
            "best_baseline_subjects": best_subjects,
            "margin": margin,
            "strictly_leads": margin > 0.0,
            "gate_comparison": comparison,
            "passes_gate": passes_gate,
        }

    won = sum(bool(item["strictly_leads"]) for item in comparisons)
    gate_passed = all(bool(item["passes_gate"]) for item in aggregate.values())
    spec = CANDIDATE_BY_SUBJECT[subject]
    fixed_order = CANDIDATE_ORDER.index(subject)
    return {
        "subject": subject,
        "candidate_label": spec.label,
        "fixed_order_index": fixed_order,
        "evidence_class": spec.evidence_class,
        "weights_retrained": spec.weights_retrained,
        "eligible_for_final_paper_selection": (spec.eligible_for_final_paper_selection),
        "conditions_won": won,
        "conditions_total": RUN_COUNT,
        "conditions_won_fraction": f"{won}/{RUN_COUNT}",
        "strictly_leads_all_conditions": won == RUN_COUNT,
        "per_condition_lead_required": False,
        "condition_comparisons": comparisons,
        "aggregate_comparisons": aggregate,
        "gate_passed": gate_passed,
        "ranking_values": {
            "worst_12_margin": aggregate["worst_12"]["margin"],
            "mean_12_margin": aggregate["mean_12"]["margin"],
            "full_0ms_margin": aggregate["full_0ms"]["margin"],
            "fixed_candidate_order": fixed_order,
        },
    }


def ranking_sort_key(
    result: Mapping[str, object],
) -> tuple[int, float, float, float, int]:
    """Gate first, then worst/mean/Full margins, then fixed registry order."""

    if type(result.get("gate_passed")) is not bool:
        raise ValueError("candidate gate status must be boolean")
    ranking = result.get("ranking_values")
    if not isinstance(ranking, Mapping) or set(ranking) != set(RANKING_KEY[1:]):
        raise ValueError("candidate ranking values do not match the SOTA contract")
    worst = _finite_metric(ranking["worst_12_margin"], context="worst margin")
    mean = _finite_metric(ranking["mean_12_margin"], context="mean margin")
    full = _finite_metric(ranking["full_0ms_margin"], context="Full 0ms margin")
    order = ranking["fixed_candidate_order"]
    if type(order) is not int or not 0 <= order < len(CANDIDATE_ORDER):
        raise ValueError("fixed candidate order is invalid")
    return (
        0 if result["gate_passed"] is True else 1,
        -worst,
        -mean,
        -full,
        order,
    )


__all__ = (
    "BASELINE_DISPLAY_NAMES",
    "BASELINE_SUBJECTS",
    "CANDIDATE_BY_SUBJECT",
    "CANDIDATE_ORDER",
    "CANDIDATE_SPECS",
    "CONDITIONS",
    "DELAYS_MS",
    "FULL_0MS_MAX_DEFICIT",
    "GROUND_TRUTH_COUNT",
    "LEADERSHIP_METRIC",
    "MANIFEST_CONTENT_SHA256",
    "OVERLAY_INDEX_CONTENT_SHA256",
    "P1_ZERO_SHOT_SUBJECT",
    "PROTOCOL_ID",
    "RANKING_KEY",
    "RUN_COUNT",
    "SAMPLE_COUNT",
    "SAMPLE_IDS_SHA256",
    "UNSUPPORTED_SAMPLE_COUNT",
    "candidate_registry",
    "evaluate_candidate",
    "evidence_fingerprint",
    "ranking_sort_key",
    "validate_evidence_fingerprint",
)
