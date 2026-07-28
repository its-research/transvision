from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from mmdet3d.registry import METRICS
from mmengine.evaluator import BaseMetric

from transvision.evaluation.resilient_v2x_detection import (
    DetectionSample,
    evaluate_car_ap,
    filter_detection_sample_to_range,
)
from transvision.evaluation.resilient_v2x_evidence import (
    seal_document,
    write_document,
)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _box_tensor(value):
    return value.tensor if hasattr(value, "tensor") else value


def _plain_diagnostic(value: object) -> dict[str, object] | None:
    branches = getattr(value, "branches", None)
    routing = getattr(value, "routing", None)
    if not isinstance(branches, tuple) or routing is None:
        return None

    def enum_value(item: object) -> object:
        return getattr(item, "value", item)

    branch_values = []
    for branch in branches:
        branch_values.append(
            {
                "agent": enum_value(getattr(branch, "agent", None)),
                "modality": enum_value(getattr(branch, "modality", None)),
                "supported": bool(getattr(branch, "supported", False)),
                "source_tick": getattr(branch, "source_tick", None),
                "source_tau_ms": getattr(branch, "source_tau_ms", None),
                "horizon": getattr(branch, "horizon", None),
                "observed": bool(getattr(branch, "observed", False)),
                "propagated": bool(getattr(branch, "propagated", False)),
                "gamma": getattr(branch, "gamma", None),
                "reliability": float(getattr(branch, "reliability", 0.0)),
                "ptf_queried": bool(getattr(branch, "ptf_queried", False)),
                "reason": enum_value(getattr(branch, "reason", None)),
            }
        )

    def tensor_list(item: object) -> object:
        if item is None:
            return None
        return _numpy(item).tolist()

    return {
        "method": getattr(value, "method", None),
        "overall_supported": any(item["supported"] for item in branch_values),
        "branches": branch_values,
        "routing": {
            "expert_support": tensor_list(
                getattr(routing, "expert_support", None)
            ),
            "weights": tensor_list(getattr(routing, "weights", None)),
            "not_applicable_reason": getattr(
                routing,
                "not_applicable_reason",
                None,
            ),
        },
    }


@METRICS.register_module()
class ResilientV2XMetric(BaseMetric):
    """Evaluate the predictions produced by the current Runner exactly once."""

    default_prefix = "resilient_v2x"

    def __init__(
        self,
        iou_thresholds: Sequence[float] = (0.5, 0.7),
        max_detections: int = 100,
        point_cloud_range: Sequence[float] | None = None,
        prediction_output: str | None = None,
        collect_device: str = "cpu",
        prefix: str | None = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thresholds = tuple(float(value) for value in iou_thresholds)
        self.max_detections = max_detections
        self.point_cloud_range = (
            None
            if point_cloud_range is None
            else tuple(float(value) for value in point_cloud_range)
        )
        self.prediction_output = prediction_output

    def process(self, data_batch: object, data_samples: Sequence[object]) -> None:
        for data_sample in data_samples:
            metainfo = getattr(data_sample, "metainfo", {})
            if not isinstance(metainfo, Mapping):
                raise ValueError("data sample metainfo must be a mapping")
            sample_id = metainfo.get("sample_id", metainfo.get("sample_idx"))
            predictions = getattr(data_sample, "pred_instances_3d", None)
            targets = getattr(data_sample, "gt_instances_3d", None)
            if predictions is None or targets is None:
                raise ValueError("prediction and ground truth instances are required")
            predicted_boxes = _box_tensor(predictions.bboxes_3d)
            target_boxes = _box_tensor(targets.bboxes_3d)
            self.results.append(
                {
                    "sample_id": sample_id,
                    "predicted_boxes": _numpy(predicted_boxes),
                    "predicted_scores": _numpy(predictions.scores_3d),
                    "predicted_labels": _numpy(predictions.labels_3d),
                    "ground_truth_boxes": _numpy(target_boxes),
                    "ground_truth_labels": _numpy(targets.labels_3d),
                    "diagnostic": _plain_diagnostic(
                        metainfo.get("resilient_v2x_diagnostics")
                    ),
                }
            )

    def compute_metrics(self, results: list[dict[str, object]]) -> dict[str, float]:
        samples = [
            DetectionSample(
                **{key: value for key, value in result.items() if key != "diagnostic"}
            )
            for result in results
        ]
        evaluated = (
            samples
            if self.point_cloud_range is None
            else [
                filter_detection_sample_to_range(sample, self.point_cloud_range)
                for sample in samples
            ]
        )
        metrics = evaluate_car_ap(
            evaluated,
            iou_thresholds=self.iou_thresholds,
            max_detections=self.max_detections,
        )
        diagnostics = [result.get("diagnostic") for result in results]
        if all(isinstance(value, Mapping) for value in diagnostics):
            metrics["unsupported_sample_count"] = float(
                sum(not bool(value["overall_supported"]) for value in diagnostics)
            )
        if self.prediction_output is not None:
            records = []
            for sample, result in zip(evaluated, results):
                records.append(
                    {
                        "sample_id": sample.sample_id,
                        "predicted_boxes_lidar_bottom_center": sample.predicted_boxes.tolist(),
                        "predicted_scores": sample.predicted_scores.tolist(),
                        "predicted_labels": sample.predicted_labels.tolist(),
                        "ground_truth_boxes_lidar_bottom_center": sample.ground_truth_boxes.tolist(),
                        "ground_truth_labels": sample.ground_truth_labels.tolist(),
                        "diagnostic": result.get("diagnostic"),
                    }
                )
            document = seal_document(
                "resilient_v2x_predictions",
                {
                    "coordinate_convention": (
                        "[x,y,z_bottom,length,width,height,yaw] in current ego LiDAR"
                    ),
                    "point_cloud_range": (
                        None
                        if self.point_cloud_range is None
                        else list(self.point_cloud_range)
                    ),
                    "iou_thresholds": list(self.iou_thresholds),
                    "max_detections": self.max_detections,
                    "sample_count": len(records),
                    "samples": records,
                },
            )
            write_document(self.prediction_output, document)
        return metrics


__all__ = ("ResilientV2XMetric",)
