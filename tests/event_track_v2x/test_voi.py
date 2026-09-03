import math

import numpy as np
import pytest

from transvision.models.event_track_v2x.contracts import (
    IdentityHypothesisV1,
    TrackingPredictionV1,
)
from transvision.models.event_track_v2x.scheduler import ScheduleCandidate
from transvision.models.event_track_v2x.schema import (
    IndependentIncrement,
    Lineage,
    LocalTimestamps,
    WireMessage,
)
from transvision.models.event_track_v2x.voi import (
    DEFAULT_VOI_SCORER_CONFIG_V1,
    IDENTITY_ENTROPY_ABLATION_CONFIG_V1,
    FrozenOnTimeProbabilityV1,
    VoiScorerConfigV1,
    score_counterfactual_voi_v1,
    track_risk_breakdown_v1,
)


def _prediction(
    probabilities: tuple[float, ...] = (0.5, 0.5),
    *,
    other_probability: float = 0.0,
    covariance: np.ndarray | None = None,
    existence_probability: float = 0.9,
    decision_time: float = 2.0,
    committed: bool = False,
) -> TrackingPredictionV1:
    return TrackingPredictionV1(
        sequence_id="sequence-1",
        frame_id="frame-20",
        track_id="track-1",
        class_label="car",
        event_time=1.0,
        arrival_time=1.5,
        decision_time=decision_time,
        mean=np.zeros(9),
        covariance=np.eye(9) if covariance is None else covariance,
        existence_probability=existence_probability,
        identity_hypotheses=tuple(
            IdentityHypothesisV1(f"identity-{index}", probability)
            for index, probability in enumerate(probabilities)
        ),
        other_identity_probability=other_probability,
        lineage=Lineage(("factor-1",), True),
        committed=committed,
    )


def _estimate(
    probability: float = 1.0,
    *,
    as_of_network_time: float = 2.0,
    network_transmit_time: float = 2.1,
    message_deadline: float = 3.0,
) -> FrozenOnTimeProbabilityV1:
    return FrozenOnTimeProbabilityV1(
        probability=probability,
        as_of_network_time=as_of_network_time,
        network_transmit_time=network_transmit_time,
        message_deadline=message_deadline,
        channel_model_sha256="c" * 64,
    )


def _message(
    *,
    message_id: str = "message-1",
    transmitted: float = 2.1,
    deadline: float = 3.0,
) -> WireMessage:
    return WireMessage(
        message_id=message_id,
        source="vehicle-1",
        sequence=1,
        timestamps=LocalTimestamps(1.0, 1.0, 2.0, transmitted),
        deadline=deadline,
        coordinate_frame="world",
        ttl=2.0,
        payload=IndependentIncrement(
            target_id="track-1",
            measurement=np.zeros(1),
            measurement_matrix=np.ones((1, 9)),
            measurement_covariance=np.eye(1),
            lineage=Lineage(("candidate-factor",), True),
        ),
    )


def test_track_risk_uses_gaussian_bernoulli_and_top_h_other_entropy() -> None:
    prediction = _prediction(
        (0.5, 0.25),
        other_probability=0.25,
        existence_probability=0.5,
    )
    risk = track_risk_breakdown_v1(prediction)

    expected_state = 0.5 * 9 * (1.0 + math.log(2.0 * math.pi))
    expected_identity = -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25))
    assert risk.state_gaussian_entropy == pytest.approx(expected_state)
    assert risk.existence_bernoulli_entropy == pytest.approx(math.log(2.0))
    assert risk.identity_categorical_entropy == pytest.approx(expected_identity)
    assert risk.total_risk == pytest.approx(
        expected_state + math.log(2.0) + expected_identity
    )
    assert len(risk.content_sha256) == 64


def test_reducing_identity_ambiguity_increases_voi() -> None:
    before = _prediction((0.5, 0.5))
    moderate_after = _prediction((0.8, 0.2))
    strong_after = _prediction((0.99, 0.01))

    moderate = score_counterfactual_voi_v1(
        before, moderate_after, _message(), _estimate()
    )
    strong = score_counterfactual_voi_v1(before, strong_after, _message(), _estimate())

    assert strong.clipped_risk_reduction > moderate.clipped_risk_reduction > 0.0
    assert strong.causal_score_evidence.ground_truth_free
    assert (
        strong.causal_score_evidence.scorer_config_sha256
        == DEFAULT_VOI_SCORER_CONFIG_V1.content_sha256
    )
    assert strong.tracker_state_sha256 not in (
        strong.before_prediction_sha256,
        strong.counterfactual_after_prediction_sha256,
    )


def test_identical_counterfactual_has_zero_value() -> None:
    prediction = _prediction((0.7, 0.3))
    result = score_counterfactual_voi_v1(
        prediction, prediction, _message(), _estimate(0.8)
    )

    assert result.raw_risk_delta == pytest.approx(0.0)
    assert result.clipped_risk_reduction == pytest.approx(0.0)
    assert result.expected_value == pytest.approx(0.0)
    assert not result.negative_delta_clipped


def test_identity_entropy_ablation_sets_identity_gain_to_zero() -> None:
    before = _prediction((0.5, 0.5))
    after = _prediction((0.99, 0.01))

    result = score_counterfactual_voi_v1(
        before,
        after,
        _message(),
        _estimate(),
        IDENTITY_ENTROPY_ABLATION_CONFIG_V1,
    )

    assert IDENTITY_ENTROPY_ABLATION_CONFIG_V1.identity_entropy_weight == 0.0
    assert result.before.weighted_identity_risk == 0.0
    assert result.counterfactual_after.weighted_identity_risk == 0.0
    assert result.clipped_risk_reduction == pytest.approx(0.0)


def test_deadline_probability_scales_expected_value_and_is_hashed() -> None:
    before = _prediction((0.5, 0.5))
    after = _prediction((0.99, 0.01))
    certain_estimate = _estimate(1.0)
    unlikely_estimate = _estimate(0.25)

    certain = score_counterfactual_voi_v1(
        before, after, _message(), certain_estimate
    )
    unlikely = score_counterfactual_voi_v1(
        before, after, _message(), unlikely_estimate
    )

    assert unlikely.clipped_risk_reduction == pytest.approx(
        certain.clipped_risk_reduction
    )
    assert unlikely.expected_value == pytest.approx(0.25 * certain.expected_value)
    assert unlikely.causal_score_evidence.on_time_probability == 0.25
    assert unlikely.on_time_estimate_sha256 == unlikely_estimate.content_sha256
    assert unlikely.on_time_estimate_sha256 != certain.on_time_estimate_sha256


def test_negative_raw_delta_is_recorded_before_clipping() -> None:
    before = _prediction((0.99, 0.01))
    riskier_after = _prediction((0.5, 0.5))

    result = score_counterfactual_voi_v1(
        before, riskier_after, _message(), _estimate()
    )

    assert result.raw_risk_delta < 0.0
    assert result.clipped_risk_reduction == 0.0
    assert result.causal_score_evidence.risk_reduction == 0.0
    assert result.negative_delta_clipped


def test_hashes_and_scores_are_deterministic() -> None:
    before = _prediction((0.6, 0.4), covariance=2.0 * np.eye(9))
    after = _prediction((0.9, 0.1), covariance=0.5 * np.eye(9))
    estimate = _estimate(0.75)
    config = VoiScorerConfigV1(2.0, 0.5, 3.0)

    first = score_counterfactual_voi_v1(before, after, _message(), estimate, config)
    second = score_counterfactual_voi_v1(before, after, _message(), estimate, config)

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.content_sha256 == second.content_sha256
    assert first.causal_score_evidence.channel_model_sha256 == "c" * 64
    assert first.causal_score_evidence.scorer_config_sha256 == config.content_sha256
    assert config.content_sha256 != DEFAULT_VOI_SCORER_CONFIG_V1.content_sha256


def test_future_ground_truth_and_invalid_uncertainty_fail_closed() -> None:
    before = _prediction()
    after = _prediction((0.9, 0.1))
    with pytest.raises(ValueError, match="future"):
        score_counterfactual_voi_v1(
            before,
            after,
            _message(),
            _estimate(as_of_network_time=1.9),
        )
    with pytest.raises(ValueError, match="ground-truth-free"):
        FrozenOnTimeProbabilityV1(
            probability=0.5,
            as_of_network_time=2.0,
            network_transmit_time=2.1,
            message_deadline=3.0,
            channel_model_sha256="c" * 64,
            ground_truth_free=False,
        )
    gt_mapping = before.to_primitive() | {"ground_truth_track_id": "gt-1"}
    with pytest.raises(TypeError, match="GT fields"):
        score_counterfactual_voi_v1(  # type: ignore[arg-type]
            gt_mapping, after, _message(), _estimate()
        )

    bad_covariance = _prediction()
    object.__setattr__(bad_covariance, "covariance", np.diag([0.0] + [1.0] * 8))
    with pytest.raises(ValueError, match="positive definite"):
        track_risk_breakdown_v1(bad_covariance)

    bad_identity_mass = _prediction()
    object.__setattr__(
        bad_identity_mass,
        "identity_hypotheses",
        (IdentityHypothesisV1("identity-0", 0.8),),
    )
    object.__setattr__(bad_identity_mass, "other_identity_probability", 0.0)
    with pytest.raises(ValueError, match="mass"):
        track_risk_breakdown_v1(bad_identity_mass)

    with pytest.raises(ValueError, match="future"):
        _estimate(as_of_network_time=2.2, network_transmit_time=2.1)


def test_voi_evidence_is_bound_to_candidate_message_time_and_deadline() -> None:
    before = _prediction((0.5, 0.5))
    after = _prediction((0.9, 0.1))
    message = _message()
    estimate = _estimate()

    result = score_counterfactual_voi_v1(before, after, message, estimate)
    candidate = ScheduleCandidate(
        message=message,
        network_transmit_time=estimate.network_transmit_time,
        score_evidence=result.causal_score_evidence,
    )
    assert candidate.message is message
    assert (
        result.causal_score_evidence.on_time_estimate_sha256
        == estimate.content_sha256
    )

    with pytest.raises(ValueError, match="another message"):
        ScheduleCandidate(
            message=_message(message_id="message-2"),
            network_transmit_time=estimate.network_transmit_time,
            score_evidence=result.causal_score_evidence,
        )
    with pytest.raises(ValueError, match="another transmit time"):
        ScheduleCandidate(
            message=message,
            network_transmit_time=2.2,
            score_evidence=result.causal_score_evidence,
        )
    with pytest.raises(ValueError, match="deadline"):
        score_counterfactual_voi_v1(
            before,
            after,
            _message(deadline=3.1),
            estimate,
        )
