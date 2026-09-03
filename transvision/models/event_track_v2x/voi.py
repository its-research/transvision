"""Ground-truth-free analytic VoI proxy for EventTrack-V2X.

This module is an executable reference scorer.  It is neither a learned VoI
model nor evidence of state-of-the-art performance.  The scorer compares a
current tracker prediction with a causal counterfactual prediction obtained if
one message were absorbed on time.  It accepts no ground-truth feature mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any

import numpy as np

from .contracts import TrackingPredictionV1
from .scheduler import CausalScoreEvidenceV1
from .schema import WireMessage
from .wire import canonical_json_bytes, wire_digest


VOI_SCORER_CONFIG_SCHEMA_V1 = "eventtrack-v2x.voi-scorer-config.v1"
ON_TIME_PROBABILITY_SCHEMA_V1 = "eventtrack-v2x.on-time-probability.v1"
TRACK_RISK_BREAKDOWN_SCHEMA_V1 = "eventtrack-v2x.track-risk-breakdown.v1"
COUNTERFACTUAL_VOI_SCORE_SCHEMA_V1 = "eventtrack-v2x.counterfactual-voi-score.v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _probability(value: float, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _sha256(value: str, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _content_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class VoiScorerConfigV1:
    """Frozen natural-log entropy weights for the analytic proxy."""

    state_entropy_weight: float = 1.0
    existence_entropy_weight: float = 1.0
    identity_entropy_weight: float = 1.0
    schema_version: str = VOI_SCORER_CONFIG_SCHEMA_V1

    def __post_init__(self) -> None:
        for name in (
            "state_entropy_weight",
            "existence_entropy_weight",
            "identity_entropy_weight",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )
        if not any(
            (
                self.state_entropy_weight,
                self.existence_entropy_weight,
                self.identity_entropy_weight,
            )
        ):
            raise ValueError("at least one VoI risk weight must be positive")
        if self.schema_version != VOI_SCORER_CONFIG_SCHEMA_V1:
            raise ValueError("unsupported VoI scorer config schema")

    def to_dict(self) -> dict[str, float | str]:
        return {
            "existence_entropy_weight": self.existence_entropy_weight,
            "identity_entropy_weight": self.identity_entropy_weight,
            "schema_version": self.schema_version,
            "state_entropy_weight": self.state_entropy_weight,
        }

    @property
    def content_sha256(self) -> str:
        return _content_sha256(self.to_dict())


DEFAULT_VOI_SCORER_CONFIG_V1 = VoiScorerConfigV1()
IDENTITY_ENTROPY_ABLATION_CONFIG_V1 = VoiScorerConfigV1(
    identity_entropy_weight=0.0
)


@dataclass(frozen=True, slots=True)
class FrozenOnTimeProbabilityV1:
    """Externally estimated deadline probability frozen before scheduling."""

    probability: float
    as_of_network_time: float
    network_transmit_time: float
    message_deadline: float
    channel_model_sha256: str
    ground_truth_free: bool = True
    schema_version: str = ON_TIME_PROBABILITY_SCHEMA_V1

    def __post_init__(self) -> None:
        probability = _probability(self.probability, "probability")
        as_of = _finite(self.as_of_network_time, "as_of_network_time")
        transmit = _finite(self.network_transmit_time, "network_transmit_time")
        deadline = _finite(self.message_deadline, "message_deadline")
        if as_of > transmit:
            raise ValueError("as_of_network_time cannot be in the future")
        if transmit > deadline:
            raise ValueError("network_transmit_time cannot exceed message_deadline")
        if self.ground_truth_free is not True:
            raise ValueError("on-time probability must be ground-truth-free")
        if self.schema_version != ON_TIME_PROBABILITY_SCHEMA_V1:
            raise ValueError("unsupported on-time probability schema")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "as_of_network_time", as_of)
        object.__setattr__(self, "network_transmit_time", transmit)
        object.__setattr__(self, "message_deadline", deadline)
        object.__setattr__(
            self,
            "channel_model_sha256",
            _sha256(self.channel_model_sha256, "channel_model_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of_network_time": self.as_of_network_time,
            "channel_model_sha256": self.channel_model_sha256,
            "ground_truth_free": self.ground_truth_free,
            "message_deadline": self.message_deadline,
            "network_transmit_time": self.network_transmit_time,
            "probability": self.probability,
            "schema_version": self.schema_version,
        }

    @property
    def content_sha256(self) -> str:
        return _content_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class TrackRiskBreakdownV1:
    """Entropy risk components for one tracker prediction, measured in nats."""

    prediction_sha256: str
    scorer_config_sha256: str
    state_gaussian_entropy: float
    existence_bernoulli_entropy: float
    identity_categorical_entropy: float
    weighted_state_risk: float
    weighted_existence_risk: float
    weighted_identity_risk: float
    total_risk: float
    schema_version: str = TRACK_RISK_BREAKDOWN_SCHEMA_V1

    def __post_init__(self) -> None:
        for name in ("prediction_sha256", "scorer_config_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in (
            "state_gaussian_entropy",
            "existence_bernoulli_entropy",
            "identity_categorical_entropy",
            "weighted_state_risk",
            "weighted_existence_risk",
            "weighted_identity_risk",
            "total_risk",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.existence_bernoulli_entropy < 0.0:
            raise ValueError("existence entropy must be non-negative")
        if self.identity_categorical_entropy < 0.0:
            raise ValueError("identity entropy must be non-negative")
        weighted_total = (
            self.weighted_state_risk
            + self.weighted_existence_risk
            + self.weighted_identity_risk
        )
        if not np.isclose(self.total_risk, weighted_total, rtol=1e-12, atol=1e-12):
            raise ValueError("total_risk must equal the weighted component sum")
        if self.schema_version != TRACK_RISK_BREAKDOWN_SCHEMA_V1:
            raise ValueError("unsupported track risk breakdown schema")

    def to_dict(self) -> dict[str, float | str]:
        return {
            "existence_bernoulli_entropy": self.existence_bernoulli_entropy,
            "identity_categorical_entropy": self.identity_categorical_entropy,
            "prediction_sha256": self.prediction_sha256,
            "schema_version": self.schema_version,
            "scorer_config_sha256": self.scorer_config_sha256,
            "state_gaussian_entropy": self.state_gaussian_entropy,
            "total_risk": self.total_risk,
            "weighted_existence_risk": self.weighted_existence_risk,
            "weighted_identity_risk": self.weighted_identity_risk,
            "weighted_state_risk": self.weighted_state_risk,
        }

    @property
    def content_sha256(self) -> str:
        return _content_sha256(self.to_dict())


def _entropy(probabilities: tuple[float, ...]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )


def _validated_identity_probabilities(
    prediction: TrackingPredictionV1,
) -> tuple[float, ...]:
    identities = prediction.identity_hypotheses
    if not identities:
        raise ValueError("identity hypotheses must not be empty")
    if len({item.identity_id for item in identities}) != len(identities):
        raise ValueError("identity hypotheses must use unique labels")
    probabilities = tuple(float(item.probability) for item in identities) + (
        float(prediction.other_identity_probability),
    )
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("identity probabilities must be finite and in [0, 1]")
    if not np.isclose(sum(probabilities), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Top-H identity mass plus other mass must equal one")
    return probabilities


def track_risk_breakdown_v1(
    prediction: TrackingPredictionV1,
    config: VoiScorerConfigV1 = DEFAULT_VOI_SCORER_CONFIG_V1,
) -> TrackRiskBreakdownV1:
    """Compute the analytic uncertainty risk for one typed prediction."""

    if not isinstance(prediction, TrackingPredictionV1):
        raise TypeError("prediction must be TrackingPredictionV1; mappings are forbidden")
    if not isinstance(config, VoiScorerConfigV1):
        raise TypeError("config must be VoiScorerConfigV1")
    covariance = np.asarray(prediction.covariance, dtype=np.float64)
    if covariance.shape != (9, 9) or not np.all(np.isfinite(covariance)):
        raise ValueError("prediction covariance must be a finite 9x9 matrix")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("prediction covariance must be symmetric")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("prediction covariance must be positive definite") from exc
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
    dimension = covariance.shape[0]
    state_entropy = 0.5 * (
        dimension * (1.0 + math.log(2.0 * math.pi)) + log_determinant
    )
    existence = _probability(
        prediction.existence_probability,
        "existence_probability",
    )
    existence_entropy = _entropy((existence, 1.0 - existence))
    identity_entropy = _entropy(_validated_identity_probabilities(prediction))
    weighted_state = config.state_entropy_weight * state_entropy
    weighted_existence = config.existence_entropy_weight * existence_entropy
    weighted_identity = config.identity_entropy_weight * identity_entropy
    return TrackRiskBreakdownV1(
        prediction_sha256=prediction.digest(),
        scorer_config_sha256=config.content_sha256,
        state_gaussian_entropy=state_entropy,
        existence_bernoulli_entropy=existence_entropy,
        identity_categorical_entropy=identity_entropy,
        weighted_state_risk=weighted_state,
        weighted_existence_risk=weighted_existence,
        weighted_identity_risk=weighted_identity,
        total_risk=weighted_state + weighted_existence + weighted_identity,
    )


def _evidence_to_dict(evidence: CausalScoreEvidenceV1) -> dict[str, Any]:
    return {
        "age_seconds": evidence.age_seconds,
        "as_of_network_time": evidence.as_of_network_time,
        "channel_model_sha256": evidence.channel_model_sha256,
        "confidence": evidence.confidence,
        "candidate_deadline": evidence.candidate_deadline,
        "candidate_message_sha256": evidence.candidate_message_sha256,
        "candidate_network_transmit_time": (
            evidence.candidate_network_transmit_time
        ),
        "ground_truth_free": evidence.ground_truth_free,
        "on_time_estimate_sha256": evidence.on_time_estimate_sha256,
        "on_time_probability": evidence.on_time_probability,
        "risk_reduction": evidence.risk_reduction,
        "scorer_config_sha256": evidence.scorer_config_sha256,
        "tracker_state_sha256": evidence.tracker_state_sha256,
    }


@dataclass(frozen=True, slots=True)
class CounterfactualVoiScoreV1:
    """Auditable score plus the scheduler-facing causal evidence."""

    before: TrackRiskBreakdownV1
    counterfactual_after: TrackRiskBreakdownV1
    raw_risk_delta: float
    clipped_risk_reduction: float
    before_prediction_sha256: str
    counterfactual_after_prediction_sha256: str
    tracker_state_sha256: str
    on_time_estimate_sha256: str
    causal_score_evidence: CausalScoreEvidenceV1
    schema_version: str = COUNTERFACTUAL_VOI_SCORE_SCHEMA_V1

    def __post_init__(self) -> None:
        if not isinstance(self.before, TrackRiskBreakdownV1) or not isinstance(
            self.counterfactual_after,
            TrackRiskBreakdownV1,
        ):
            raise TypeError("before and counterfactual_after must be risk breakdowns")
        raw_delta = _finite(self.raw_risk_delta, "raw_risk_delta")
        expected_raw_delta = self.before.total_risk - self.counterfactual_after.total_risk
        if not np.isclose(raw_delta, expected_raw_delta, rtol=1e-12, atol=1e-12):
            raise ValueError("raw_risk_delta must equal before risk minus after risk")
        clipped = _nonnegative(
            self.clipped_risk_reduction,
            "clipped_risk_reduction",
        )
        if clipped != max(0.0, raw_delta):
            raise ValueError("clipped_risk_reduction must equal max(0, raw delta)")
        for name in (
            "before_prediction_sha256",
            "counterfactual_after_prediction_sha256",
            "tracker_state_sha256",
            "on_time_estimate_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.before.prediction_sha256 != self.before_prediction_sha256:
            raise ValueError("before prediction hash disagrees with risk breakdown")
        if (
            self.counterfactual_after.prediction_sha256
            != self.counterfactual_after_prediction_sha256
        ):
            raise ValueError("after prediction hash disagrees with risk breakdown")
        if (
            self.before.scorer_config_sha256
            != self.counterfactual_after.scorer_config_sha256
        ):
            raise ValueError("before and after must use the same scorer config")
        if not isinstance(self.causal_score_evidence, CausalScoreEvidenceV1):
            raise TypeError("causal_score_evidence must be CausalScoreEvidenceV1")
        evidence = self.causal_score_evidence
        if evidence.risk_reduction != clipped:
            raise ValueError("causal evidence risk reduction disagrees with score")
        if evidence.tracker_state_sha256 != self.tracker_state_sha256:
            raise ValueError("causal evidence tracker hash disagrees with score")
        if evidence.scorer_config_sha256 != self.before.scorer_config_sha256:
            raise ValueError("causal evidence config hash disagrees with score")
        if evidence.on_time_estimate_sha256 != self.on_time_estimate_sha256:
            raise ValueError("causal evidence on-time estimate hash disagrees with score")
        if self.schema_version != COUNTERFACTUAL_VOI_SCORE_SCHEMA_V1:
            raise ValueError("unsupported counterfactual VoI score schema")
        object.__setattr__(self, "raw_risk_delta", raw_delta)
        object.__setattr__(self, "clipped_risk_reduction", clipped)

    @property
    def negative_delta_clipped(self) -> bool:
        return self.raw_risk_delta < 0.0

    @property
    def expected_value(self) -> float:
        return (
            self.clipped_risk_reduction
            * self.causal_score_evidence.on_time_probability
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before.to_dict(),
            "before_prediction_sha256": self.before_prediction_sha256,
            "causal_score_evidence": _evidence_to_dict(self.causal_score_evidence),
            "clipped_risk_reduction": self.clipped_risk_reduction,
            "counterfactual_after": self.counterfactual_after.to_dict(),
            "counterfactual_after_prediction_sha256": (
                self.counterfactual_after_prediction_sha256
            ),
            "expected_value": self.expected_value,
            "negative_delta_clipped": self.negative_delta_clipped,
            "on_time_estimate_sha256": self.on_time_estimate_sha256,
            "raw_risk_delta": self.raw_risk_delta,
            "schema_version": self.schema_version,
            "tracker_state_sha256": self.tracker_state_sha256,
        }

    @property
    def content_sha256(self) -> str:
        return _content_sha256(self.to_dict())


def _require_comparable_predictions(
    before: TrackingPredictionV1,
    after: TrackingPredictionV1,
) -> None:
    fields = ("sequence_id", "frame_id", "track_id", "class_label", "decision_time")
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        raise ValueError("before and after must describe the same track decision")
    if before.committed or after.committed:
        raise ValueError("VoI scoring cannot rewrite committed predictions")


def score_counterfactual_voi_v1(
    before: TrackingPredictionV1,
    counterfactual_after: TrackingPredictionV1,
    candidate_message: WireMessage,
    on_time_estimate: FrozenOnTimeProbabilityV1,
    config: VoiScorerConfigV1 = DEFAULT_VOI_SCORER_CONFIG_V1,
) -> CounterfactualVoiScoreV1:
    """Score a causal before/after counterfactual without ground truth."""

    if not isinstance(before, TrackingPredictionV1) or not isinstance(
        counterfactual_after,
        TrackingPredictionV1,
    ):
        raise TypeError(
            "before and counterfactual_after must be TrackingPredictionV1; "
            "feature mappings, including GT fields, are forbidden"
        )
    if not isinstance(on_time_estimate, FrozenOnTimeProbabilityV1):
        raise TypeError("on_time_estimate must be FrozenOnTimeProbabilityV1")
    if not isinstance(candidate_message, WireMessage):
        raise TypeError("candidate_message must be WireMessage")
    if not isinstance(config, VoiScorerConfigV1):
        raise TypeError("config must be VoiScorerConfigV1")
    if candidate_message.deadline != on_time_estimate.message_deadline:
        raise ValueError("candidate message deadline disagrees with on-time estimate")
    _require_comparable_predictions(before, counterfactual_after)
    as_of = on_time_estimate.as_of_network_time
    if before.decision_time > as_of or counterfactual_after.decision_time > as_of:
        raise ValueError("prediction state is from the future of as_of_network_time")
    before_risk = track_risk_breakdown_v1(before, config)
    after_risk = track_risk_breakdown_v1(counterfactual_after, config)
    raw_delta = before_risk.total_risk - after_risk.total_risk
    clipped = max(0.0, raw_delta)
    before_sha256 = before.digest()
    after_sha256 = counterfactual_after.digest()
    tracker_state_sha256 = _content_sha256(
        {
            "before_prediction_sha256": before_sha256,
            "counterfactual_after_prediction_sha256": after_sha256,
        }
    )
    evidence = CausalScoreEvidenceV1(
        as_of_network_time=as_of,
        risk_reduction=clipped,
        on_time_probability=on_time_estimate.probability,
        confidence=counterfactual_after.existence_probability,
        age_seconds=as_of - before.event_time,
        tracker_state_sha256=tracker_state_sha256,
        channel_model_sha256=on_time_estimate.channel_model_sha256,
        scorer_config_sha256=config.content_sha256,
        on_time_estimate_sha256=on_time_estimate.content_sha256,
        candidate_message_sha256=wire_digest(candidate_message),
        candidate_network_transmit_time=on_time_estimate.network_transmit_time,
        candidate_deadline=on_time_estimate.message_deadline,
        ground_truth_free=True,
    )
    return CounterfactualVoiScoreV1(
        before=before_risk,
        counterfactual_after=after_risk,
        raw_risk_delta=raw_delta,
        clipped_risk_reduction=clipped,
        before_prediction_sha256=before_sha256,
        counterfactual_after_prediction_sha256=after_sha256,
        tracker_state_sha256=tracker_state_sha256,
        on_time_estimate_sha256=on_time_estimate.content_sha256,
        causal_score_evidence=evidence,
    )


__all__ = [
    "COUNTERFACTUAL_VOI_SCORE_SCHEMA_V1",
    "CounterfactualVoiScoreV1",
    "DEFAULT_VOI_SCORER_CONFIG_V1",
    "FrozenOnTimeProbabilityV1",
    "IDENTITY_ENTROPY_ABLATION_CONFIG_V1",
    "ON_TIME_PROBABILITY_SCHEMA_V1",
    "TRACK_RISK_BREAKDOWN_SCHEMA_V1",
    "TrackRiskBreakdownV1",
    "VOI_SCORER_CONFIG_SCHEMA_V1",
    "VoiScorerConfigV1",
    "score_counterfactual_voi_v1",
    "track_risk_breakdown_v1",
]
