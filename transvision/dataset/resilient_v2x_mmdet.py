from __future__ import annotations

from mmdet3d.registry import DATASETS, DATA_SAMPLERS
from mmengine.registry import FUNCTIONS

from .resilient_v2x_runtime import (
    ResilientTemporalDataset,
    EpochIndexSampler,
    collate_resilient_samples,
)


if DATASETS.get("ResilientTemporalDataset") is None:
    DATASETS.register_module(module=ResilientTemporalDataset)
if FUNCTIONS.get("collate_resilient_samples") is None:
    FUNCTIONS.register_module(module=collate_resilient_samples)
if DATA_SAMPLERS.get("EpochIndexSampler") is None:
    DATA_SAMPLERS.register_module(module=EpochIndexSampler)


__all__ = (
    "ResilientTemporalDataset",
    "collate_resilient_samples",
    "EpochIndexSampler",
)
