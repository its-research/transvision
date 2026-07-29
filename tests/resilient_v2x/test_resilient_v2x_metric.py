from __future__ import annotations

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
