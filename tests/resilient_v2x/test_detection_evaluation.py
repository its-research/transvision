from __future__ import annotations

import numpy as np
import pytest

from transvision.evaluation.resilient_v2x_detection import (
    DetectionEvaluationError,
    DetectionSample,
    box_iou,
    evaluate_car_ap,
    filter_detection_sample_to_range,
)


BOX = np.array([0.0, 0.0, 1.0, 4.0, 2.0, 2.0, 0.0])


def _sample(sample_id: str, predicted_boxes, scores) -> DetectionSample:
    predicted_boxes = np.asarray(predicted_boxes, dtype=np.float64).reshape(-1, 7)
    return DetectionSample(
        sample_id=sample_id,
        predicted_boxes=predicted_boxes,
        predicted_scores=np.asarray(scores),
        predicted_labels=np.zeros(len(scores), dtype=np.int64),
        ground_truth_boxes=BOX.reshape(1, 7),
        ground_truth_labels=np.zeros(1, dtype=np.int64),
    )


def test_rotated_bev_and_3d_iou_have_exact_identity_and_disjoint_values() -> None:
    shifted = BOX.copy()
    shifted[0] = 10.0

    assert box_iou(BOX, BOX, "bev") == pytest.approx(1.0)
    assert box_iou(BOX, BOX, "3d") == pytest.approx(1.0)
    assert box_iou(BOX, shifted, "bev") == 0.0
    assert box_iou(BOX, shifted, "3d") == 0.0


def test_3d_iou_uses_lidar_bottom_height_convention() -> None:
    first = np.array([0.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0])
    second = np.array([0.0, 0.0, 1.5, 4.0, 2.0, 1.0, 0.0])

    assert box_iou(first, second, "bev") == pytest.approx(1.0)
    assert box_iou(first, second, "3d") == pytest.approx(0.2)


def test_ap_r40_is_perfect_for_one_exact_prediction_per_sample() -> None:
    metrics = evaluate_car_ap(
        [_sample("a", [BOX], [0.9]), _sample("b", [BOX], [0.8])]
    )

    assert metrics["car_bev_ap_r40_0.50"] == pytest.approx(100.0)
    assert metrics["car_bev_ap_r40_0.70"] == pytest.approx(100.0)
    assert metrics["car_3d_ap_r40_0.50"] == pytest.approx(100.0)
    assert metrics["car_3d_ap_r40_0.70"] == pytest.approx(100.0)


def test_high_score_false_positive_reduces_ap_and_matching_is_one_to_one() -> None:
    false_box = BOX.copy()
    false_box[0] = 20.0
    metrics = evaluate_car_ap(
        [_sample("a", [false_box, BOX, BOX], [0.99, 0.9, 0.8])]
    )

    assert 0.0 < metrics["car_bev_ap_r40_0.50"] < 100.0
    assert metrics["car_prediction_count"] == 3.0


def test_ap_rejects_zero_car_ground_truth() -> None:
    sample = DetectionSample(
        sample_id="empty",
        predicted_boxes=np.empty((0, 7)),
        predicted_scores=np.empty(0),
        predicted_labels=np.empty(0, dtype=np.int64),
        ground_truth_boxes=np.empty((0, 7)),
        ground_truth_labels=np.empty(0, dtype=np.int64),
    )

    with pytest.raises(DetectionEvaluationError, match="without Car"):
        evaluate_car_ap([sample])


def test_range_filter_uses_box_center_and_half_open_boundaries() -> None:
    inside = BOX.copy()
    outside = BOX.copy()
    outside[0] = 80.0
    sample = _sample("range", [inside, outside], [0.9, 0.8])

    filtered = filter_detection_sample_to_range(
        sample,
        [0.0, -40.0, -3.0, 80.0, 40.0, 3.0],
    )

    assert filtered.predicted_boxes.shape == (1, 7)
    assert filtered.predicted_scores.tolist() == [0.9]
    assert filtered.ground_truth_boxes.shape == (1, 7)
