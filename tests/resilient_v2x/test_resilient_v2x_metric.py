from __future__ import annotations

import json

import pytest
import torch
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmengine.evaluator import Evaluator
from mmengine.structures import InstanceData

from transvision.evaluation.metrics.resilient_v2x_metric import (
    ResilientV2XMetric,
)


def test_metric_accepts_mmengine_dictionary_conversion() -> None:
    box = torch.tensor([[4.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0]])
    sample = Det3DDataSample()
    sample.set_metainfo({"sample_id": "sample-0"})

    targets = InstanceData()
    targets.bboxes_3d = LiDARInstance3DBoxes(box.clone(), box_dim=7)
    targets.labels_3d = torch.zeros(1, dtype=torch.long)
    sample.gt_instances_3d = targets

    predictions = InstanceData()
    predictions.bboxes_3d = LiDARInstance3DBoxes(box.clone(), box_dim=7)
    predictions.scores_3d = torch.tensor([0.9])
    predictions.labels_3d = torch.zeros(1, dtype=torch.long)
    sample.pred_instances_3d = predictions

    metric = ResilientV2XMetric()
    evaluator = Evaluator(metric)
    evaluator.process(data_samples=[sample], data_batch={})

    assert metric.results[0]["sample_id"] == "sample-0"
    metrics = evaluator.evaluate(size=1)
    assert metrics["resilient_v2x/car_3d_ap_r40_0.70"] == 100.0
    assert metrics["resilient_v2x/diagnostic_bev_match_050_count"] == 1.0
    assert (
        metrics["resilient_v2x/diagnostic_bev_match_050_abs_z_error_p50"] == 0.0
    )


def test_metric_preserves_controlled_baseline_diagnostic_in_evidence(
    tmp_path,
) -> None:
    box = torch.tensor([[4.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0]])
    sample = Det3DDataSample()
    sample.set_metainfo(
        {
            "sample_id": "sample-baseline",
            "controlled_baseline_diagnostics": {
                "schema_version": 1,
                "sample_id": "sample-baseline",
                "method": "ffnet",
                "branch_order": (
                    "lidar_ego",
                    "lidar_rsu",
                    "camera_ego",
                    "camera_rsu",
                ),
                "support": {
                    "lidar_ego": True,
                    "lidar_rsu": True,
                    "camera_ego": False,
                    "camera_rsu": True,
                },
                "age_intervals": {
                    "lidar_ego": 0.0,
                    "lidar_rsu": 2.0,
                    "camera_ego": None,
                    "camera_rsu": 1.0,
                },
            },
        }
    )
    targets = InstanceData()
    targets.bboxes_3d = LiDARInstance3DBoxes(box.clone(), box_dim=7)
    targets.labels_3d = torch.zeros(1, dtype=torch.long)
    sample.gt_instances_3d = targets
    predictions = InstanceData()
    predictions.bboxes_3d = LiDARInstance3DBoxes(box.clone(), box_dim=7)
    predictions.scores_3d = torch.tensor([0.9])
    predictions.labels_3d = torch.zeros(1, dtype=torch.long)
    sample.pred_instances_3d = predictions
    output = tmp_path / "predictions.json"
    metric = ResilientV2XMetric(prediction_output=str(output))

    metric.process({}, [sample])
    diagnostic = metric.results[0]["diagnostic"]
    assert diagnostic["diagnostic_type"] == "controlled_baseline"
    assert diagnostic["method"] == "ffnet"
    assert diagnostic["overall_supported"] is True
    assert diagnostic["support"]["camera_ego"] is False
    assert diagnostic["age_intervals"]["camera_ego"] is None
    metrics = metric.compute_metrics(metric.results)
    assert metrics["unsupported_sample_count"] == 0.0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["samples"][0]["diagnostic"] == diagnostic


def test_metric_rejects_mismatched_controlled_baseline_sample_id() -> None:
    sample = Det3DDataSample()
    sample.set_metainfo(
        {
            "sample_id": "outer",
            "controlled_baseline_diagnostics": {
                "schema_version": 1,
                "sample_id": "inner",
                "method": "cobevt",
                "branch_order": (
                    "lidar_ego",
                    "lidar_rsu",
                    "camera_ego",
                    "camera_rsu",
                ),
                "support": {
                    "lidar_ego": True,
                    "lidar_rsu": False,
                    "camera_ego": False,
                    "camera_rsu": False,
                },
                "age_intervals": {
                    "lidar_ego": 0.0,
                    "lidar_rsu": None,
                    "camera_ego": None,
                    "camera_rsu": None,
                },
            },
        }
    )
    targets = InstanceData()
    targets.bboxes_3d = LiDARInstance3DBoxes(torch.zeros(0, 7), box_dim=7)
    targets.labels_3d = torch.zeros(0, dtype=torch.long)
    sample.gt_instances_3d = targets
    predictions = InstanceData()
    predictions.bboxes_3d = LiDARInstance3DBoxes(torch.zeros(0, 7), box_dim=7)
    predictions.scores_3d = torch.zeros(0)
    predictions.labels_3d = torch.zeros(0, dtype=torch.long)
    sample.pred_instances_3d = predictions

    with pytest.raises(ValueError, match="sample_id does not match"):
        ResilientV2XMetric().process({}, [sample])
