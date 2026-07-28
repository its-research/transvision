from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmengine.model import BaseDataPreprocessor
from mmengine.structures import InstanceData
from torch import Tensor

from transvision.dataset.resilient_v2x_runtime import (
    ResolvedTemporalSample,
    collate_resilient_samples,
)


@MODELS.register_module()
class ResilientV2XDataPreprocessor(BaseDataPreprocessor):
    """Move a causal sparse batch to device and normalize RGB inputs."""

    def __init__(
        self,
        mean: Sequence[float] = (123.675, 116.28, 103.53),
        std: Sequence[float] = (58.395, 57.12, 57.375),
        non_blocking: bool = False,
    ) -> None:
        super().__init__(non_blocking=non_blocking)
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must contain three RGB values")
        if any(float(value) <= 0 for value in std):
            raise ValueError("image standard deviations must be positive")
        self.register_buffer(
            "mean",
            torch.tensor(tuple(float(value) for value in mean)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(tuple(float(value) for value in std)).view(1, 3, 1, 1),
            persistent=False,
        )

    def _normalize_inputs(self, inputs: dict[str, object]) -> dict[str, object]:
        images = inputs.get("camera_images")
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError("camera_images must be a rank-4 tensor")
        images = images.float()
        inputs["camera_images"] = (images - self.mean) / self.std
        nested = inputs.get("teacher_clean")
        if nested is not None:
            if not isinstance(nested, dict):
                raise ValueError("teacher_clean must be a dictionary")
            inputs["teacher_clean"] = self._normalize_inputs(nested)
        return inputs

    @staticmethod
    def _data_samples(inputs: Mapping[str, object]) -> list[Det3DDataSample]:
        resolved = inputs.get("resolved")
        gt_boxes = inputs.get("gt_bboxes_3d")
        gt_labels = inputs.get("gt_labels_3d")
        if (
            not isinstance(resolved, tuple)
            or not isinstance(gt_boxes, tuple)
            or not isinstance(gt_labels, tuple)
            or not (len(resolved) == len(gt_boxes) == len(gt_labels))
        ):
            raise ValueError("resolved metadata and GT tuples must share batch size")
        data_samples: list[Det3DDataSample] = []
        for metadata, boxes, labels in zip(resolved, gt_boxes, gt_labels):
            if not isinstance(metadata, ResolvedTemporalSample):
                raise ValueError("resolved tuple contains an invalid record")
            if not isinstance(boxes, Tensor) or not isinstance(labels, Tensor):
                raise ValueError("ground truth values must be tensors")
            sample = Det3DDataSample()
            sample.set_metainfo(
                {
                    "sample_idx": metadata.sample.sample_id,
                    "sample_id": metadata.sample.sample_id,
                    "sequence_id": metadata.sample.sequence_id,
                    "split": metadata.sample.split,
                    "n_t": metadata.sample.n_t,
                    "tau_t_ms": metadata.sample.tau_t_ms,
                    "epoch": metadata.epoch,
                    "augmentation_seed": metadata.augmentation_seed,
                    "box_type_3d": LiDARInstance3DBoxes,
                }
            )
            instances = InstanceData()
            instances.bboxes_3d = LiDARInstance3DBoxes(boxes, box_dim=7)
            instances.labels_3d = labels
            sample.gt_instances_3d = instances
            data_samples.append(sample)
        return data_samples

    def forward(
        self,
        data: Mapping[str, object] | Sequence[Mapping[str, object]],
        training: bool = False,
    ) -> dict[str, object]:
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, Mapping)):
            inputs = collate_resilient_samples(data)
        elif isinstance(data, Mapping):
            inputs = dict(data)
        else:
            raise TypeError("data must be a collated mapping or sample sequence")
        inputs = self.cast_data(inputs)
        if not isinstance(inputs, dict):
            raise RuntimeError("cast_data must preserve the input mapping")
        inputs = self._normalize_inputs(inputs)
        data_samples = self._data_samples(inputs)
        inputs.pop("gt_bboxes_3d", None)
        inputs.pop("gt_labels_3d", None)
        return {"inputs": inputs, "data_samples": data_samples}


__all__ = ("ResilientV2XDataPreprocessor",)
