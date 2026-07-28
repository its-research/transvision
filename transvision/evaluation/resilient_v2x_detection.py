from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


class DetectionEvaluationError(ValueError):
    """Raised when prediction evidence is malformed or not evaluable."""


def _boxes(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 7:
        raise DetectionEvaluationError(f"{name} must have shape [N,>=7]")
    array = array[:, :7]
    if not np.isfinite(array).all():
        raise DetectionEvaluationError(f"{name} must contain finite values")
    if array.size and (array[:, 3:6] <= 0).any():
        raise DetectionEvaluationError(f"{name} dimensions must be positive")
    return array


def _vector(value: object, name: str, length: int, dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != (length,):
        raise DetectionEvaluationError(f"{name} must have shape [{length}]")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise DetectionEvaluationError(f"{name} must contain finite values")
    return array


@dataclass(frozen=True)
class DetectionSample:
    sample_id: str
    predicted_boxes: np.ndarray
    predicted_scores: np.ndarray
    predicted_labels: np.ndarray
    ground_truth_boxes: np.ndarray
    ground_truth_labels: np.ndarray

    def __post_init__(self) -> None:
        if (
            type(self.sample_id) is not str
            or not self.sample_id
            or self.sample_id != self.sample_id.strip()
        ):
            raise DetectionEvaluationError("sample_id must be trimmed and non-empty")
        predicted_boxes = _boxes(self.predicted_boxes, "predicted_boxes")
        ground_truth_boxes = _boxes(
            self.ground_truth_boxes,
            "ground_truth_boxes",
        )
        predicted_scores = _vector(
            self.predicted_scores,
            "predicted_scores",
            predicted_boxes.shape[0],
            np.float64,
        )
        if predicted_scores.size and (
            (predicted_scores < 0).any() or (predicted_scores > 1).any()
        ):
            raise DetectionEvaluationError("predicted_scores must be in [0,1]")
        predicted_labels = _vector(
            self.predicted_labels,
            "predicted_labels",
            predicted_boxes.shape[0],
            np.int64,
        )
        ground_truth_labels = _vector(
            self.ground_truth_labels,
            "ground_truth_labels",
            ground_truth_boxes.shape[0],
            np.int64,
        )
        object.__setattr__(self, "predicted_boxes", predicted_boxes)
        object.__setattr__(self, "predicted_scores", predicted_scores)
        object.__setattr__(self, "predicted_labels", predicted_labels)
        object.__setattr__(self, "ground_truth_boxes", ground_truth_boxes)
        object.__setattr__(self, "ground_truth_labels", ground_truth_labels)


def _bev_corners(box: np.ndarray) -> np.ndarray:
    x, y, _, length, width, _, yaw = box
    local = np.array(
        (
            (length / 2.0, width / 2.0),
            (-length / 2.0, width / 2.0),
            (-length / 2.0, -width / 2.0),
            (length / 2.0, -width / 2.0),
        ),
        dtype=np.float64,
    )
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    return local @ rotation.T + np.array((x, y))


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _inside(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    return _cross(end - start, point - start) >= -1e-10


def _line_intersection(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> np.ndarray:
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    denominator = _cross(first_direction, second_direction)
    if abs(denominator) <= 1e-12:
        return (first_end + second_start) / 2.0
    distance = second_start - first_start
    scale = _cross(distance, second_direction) / denominator
    return first_start + scale * first_direction


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = [point for point in subject]
    for edge_index in range(len(clip)):
        start = clip[edge_index]
        end = clip[(edge_index + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = _inside(current, start, end)
            previous_inside = _inside(previous, start, end)
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, start, end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, start, end))
            previous = current
    if not output:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


def _polygon_area(polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def rotated_bev_intersection(first: np.ndarray, second: np.ndarray) -> float:
    first_box = _boxes(np.asarray(first).reshape(1, -1), "first")[0]
    second_box = _boxes(np.asarray(second).reshape(1, -1), "second")[0]
    intersection = _clip_polygon(
        _bev_corners(first_box),
        _bev_corners(second_box),
    )
    return _polygon_area(intersection)


def box_iou(
    first: np.ndarray,
    second: np.ndarray,
    mode: Literal["bev", "3d"],
) -> float:
    if mode not in ("bev", "3d"):
        raise DetectionEvaluationError("mode must be 'bev' or '3d'")
    first_box = _boxes(np.asarray(first).reshape(1, -1), "first")[0]
    second_box = _boxes(np.asarray(second).reshape(1, -1), "second")[0]
    bev_intersection = rotated_bev_intersection(first_box, second_box)
    first_bev = float(first_box[3] * first_box[4])
    second_bev = float(second_box[3] * second_box[4])
    if mode == "bev":
        union = first_bev + second_bev - bev_intersection
        return bev_intersection / union if union > 0 else 0.0
    # LiDARInstance3DBoxes stores z at the bottom face (origin z=0), matching
    # the DAIR manifest's explicit z_bottom field.
    first_bottom = first_box[2]
    first_top = first_box[2] + first_box[5]
    second_bottom = second_box[2]
    second_top = second_box[2] + second_box[5]
    height_intersection = max(
        0.0,
        min(first_top, second_top) - max(first_bottom, second_bottom),
    )
    intersection = bev_intersection * height_intersection
    first_volume = first_bev * float(first_box[5])
    second_volume = second_bev * float(second_box[5])
    union = first_volume + second_volume - intersection
    return intersection / union if union > 0 else 0.0


def _validated_point_cloud_range(value: object) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise DetectionEvaluationError(
            "point_cloud_range must contain six finite values"
        )
    if (array[3:] <= array[:3]).any():
        raise DetectionEvaluationError("point_cloud_range maxima must exceed minima")
    return tuple(float(item) for item in array)


def filter_detection_sample_to_range(
    sample: DetectionSample,
    point_cloud_range: tuple[float, ...] | list[float],
) -> DetectionSample:
    """Filter predictions and targets by LiDAR box center in one fixed range."""

    if not isinstance(sample, DetectionSample):
        raise DetectionEvaluationError("sample must be a DetectionSample")
    limits = _validated_point_cloud_range(point_cloud_range)

    def keep(boxes: np.ndarray) -> np.ndarray:
        if not boxes.shape[0]:
            return np.empty(0, dtype=bool)
        centers = boxes[:, :3].copy()
        centers[:, 2] += boxes[:, 5] / 2.0
        lower = np.asarray(limits[:3])
        upper = np.asarray(limits[3:])
        return ((centers >= lower) & (centers < upper)).all(axis=1)

    prediction_mask = keep(sample.predicted_boxes)
    target_mask = keep(sample.ground_truth_boxes)
    return DetectionSample(
        sample_id=sample.sample_id,
        predicted_boxes=sample.predicted_boxes[prediction_mask],
        predicted_scores=sample.predicted_scores[prediction_mask],
        predicted_labels=sample.predicted_labels[prediction_mask],
        ground_truth_boxes=sample.ground_truth_boxes[target_mask],
        ground_truth_labels=sample.ground_truth_labels[target_mask],
    )


def _ap_r40(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.shape != precision.shape or recall.ndim != 1:
        raise DetectionEvaluationError("recall and precision must be vectors")
    if not recall.size:
        return 0.0
    envelope = precision.copy()
    for index in range(envelope.size - 2, -1, -1):
        envelope[index] = max(envelope[index], envelope[index + 1])
    samples = np.linspace(0.0, 1.0, 41, dtype=np.float64)[1:]
    values = [
        float(envelope[np.flatnonzero(recall >= target)[0]])
        if np.any(recall >= target)
        else 0.0
        for target in samples
    ]
    return 100.0 * float(np.mean(values))


def evaluate_car_ap(
    samples: tuple[DetectionSample, ...] | list[DetectionSample],
    *,
    iou_thresholds: tuple[float, ...] = (0.5, 0.7),
    max_detections: int = 100,
    point_cloud_range: tuple[float, ...] | list[float] | None = None,
) -> dict[str, float]:
    if not isinstance(samples, (tuple, list)) or not samples:
        raise DetectionEvaluationError("samples must be a non-empty sequence")
    if any(not isinstance(sample, DetectionSample) for sample in samples):
        raise DetectionEvaluationError("samples contain an invalid record")
    if point_cloud_range is not None:
        samples = [
            filter_detection_sample_to_range(sample, point_cloud_range)
            for sample in samples
        ]
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise DetectionEvaluationError("sample IDs must be unique")
    if type(max_detections) is not int or max_detections <= 0:
        raise DetectionEvaluationError("max_detections must be positive")
    for threshold in iou_thresholds:
        if not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
            raise DetectionEvaluationError("IoU thresholds must be in (0,1]")

    ground_truth = {
        sample.sample_id: sample.ground_truth_boxes[sample.ground_truth_labels == 0]
        for sample in samples
    }
    gt_count = sum(value.shape[0] for value in ground_truth.values())
    if gt_count == 0:
        raise DetectionEvaluationError("Car AP is undefined without Car ground truth")

    predictions: list[tuple[float, str, int, np.ndarray]] = []
    for sample in samples:
        valid = np.flatnonzero(sample.predicted_labels == 0)
        order = valid[np.argsort(-sample.predicted_scores[valid], kind="stable")][
            :max_detections
        ]
        predictions.extend(
            (
                float(sample.predicted_scores[index]),
                sample.sample_id,
                int(index),
                sample.predicted_boxes[index],
            )
            for index in order
        )
    predictions.sort(key=lambda item: (-item[0], item[1], item[2]))

    metrics: dict[str, float] = {
        "sample_count": float(len(samples)),
        "car_ground_truth_count": float(gt_count),
        "car_prediction_count": float(len(predictions)),
    }
    for mode in ("bev", "3d"):
        for threshold in iou_thresholds:
            matched = {
                sample_id: np.zeros(boxes.shape[0], dtype=bool)
                for sample_id, boxes in ground_truth.items()
            }
            true_positive = np.zeros(len(predictions), dtype=np.float64)
            false_positive = np.zeros(len(predictions), dtype=np.float64)
            for prediction_index, (_, sample_id, _, predicted_box) in enumerate(
                predictions
            ):
                boxes = ground_truth[sample_id]
                if boxes.shape[0] == 0:
                    false_positive[prediction_index] = 1.0
                    continue
                overlaps = np.asarray(
                    [box_iou(predicted_box, target, mode) for target in boxes]
                )
                overlaps[matched[sample_id]] = -1.0
                target_index = int(np.argmax(overlaps))
                if overlaps[target_index] >= threshold:
                    matched[sample_id][target_index] = True
                    true_positive[prediction_index] = 1.0
                else:
                    false_positive[prediction_index] = 1.0
            cumulative_tp = np.cumsum(true_positive)
            cumulative_fp = np.cumsum(false_positive)
            recall = cumulative_tp / gt_count
            precision = cumulative_tp / np.maximum(
                cumulative_tp + cumulative_fp,
                np.finfo(np.float64).eps,
            )
            metrics[f"car_{mode}_ap_r40_{threshold:.2f}"] = _ap_r40(
                recall,
                precision,
            )
    return metrics


__all__ = (
    "DetectionEvaluationError",
    "DetectionSample",
    "rotated_bev_intersection",
    "box_iou",
    "filter_detection_sample_to_range",
    "evaluate_car_ap",
)
