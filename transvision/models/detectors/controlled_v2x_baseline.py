from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from torch import Tensor, nn

from transvision.models.resilient_v2x import (
    BEVGridSpec,
    CONTROLLED_BRANCH_KEYS,
    ControlledBaselineInputBatch,
    ControlledBaselineInputSelector,
    ResilientBatchSelections,
    scatter_sparse_bev_history,
)


def _build_required(config: object, name: str) -> nn.Module:
    if isinstance(config, nn.Module):
        return config
    if not isinstance(config, Mapping):
        raise ValueError(f"{name} must be a module config or nn.Module")
    module = MODELS.build(dict(config))
    if not isinstance(module, nn.Module):
        raise RuntimeError(f"{name} did not build an nn.Module")
    return module


@MODELS.register_module()
class ControlledCooperativeBaselineNet(Base3DDetector):
    """Controlled cooperative baseline on the strict causal data contract.

    This detector intentionally contains no trajectory repair, dynamic expert
    routing, DER logic, or teacher path. It shares the project encoders and
    detection-head interface, then delegates only four-branch fusion to the
    named controlled baseline implementation.
    """

    channels = 256

    def __init__(
        self,
        grid_spec: Mapping[str, object] | BEVGridSpec,
        lidar_encoder: Mapping[str, object] | nn.Module,
        camera_encoder: Mapping[str, object] | nn.Module,
        bbox_head: Mapping[str, object] | nn.Module,
        baseline_name: str,
        detection_projection: Mapping[str, object] | nn.Module | None = None,
        baseline_cfg: Mapping[str, object] | None = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        **kwargs: object,
    ) -> None:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise ValueError(f"unexpected controlled baseline options: {unexpected}")
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if isinstance(grid_spec, Mapping):
            self.grid_spec = BEVGridSpec(**dict(grid_spec))
        elif isinstance(grid_spec, BEVGridSpec):
            self.grid_spec = grid_spec
        else:
            raise ValueError("grid_spec must be a mapping or BEVGridSpec")
        if (
            type(baseline_name) is not str
            or not baseline_name
            or baseline_name != baseline_name.strip()
        ):
            raise ValueError("baseline_name must be a trimmed non-empty string")
        if baseline_cfg is None:
            normalized_baseline_cfg: dict[str, object] = {}
        elif isinstance(baseline_cfg, Mapping):
            normalized_baseline_cfg = dict(baseline_cfg)
        else:
            raise ValueError("baseline_cfg must be a mapping")
        if "channels" in normalized_baseline_cfg:
            raise ValueError("baseline_cfg must not override the shared channels")

        self.baseline_name = baseline_name
        self.lidar_encoder = _build_required(lidar_encoder, "lidar_encoder")
        self.camera_encoder = _build_required(camera_encoder, "camera_encoder")
        self.bbox_head = _build_required(bbox_head, "bbox_head")
        self.detection_projection = (
            None
            if detection_projection is None
            else _build_required(detection_projection, "detection_projection")
        )
        self.input_selector = ControlledBaselineInputSelector(
            grid_spec=self.grid_spec,
            channels=self.channels,
        )

        # Imported lazily so model registration stays independent of optional
        # baseline implementations until this detector is actually configured.
        from transvision.models.resilient_v2x.baselines import (
            build_controlled_baseline_fusion,
        )

        fusion = build_controlled_baseline_fusion(
            baseline_name,
            channels=self.channels,
            **normalized_baseline_cfg,
        )
        if not isinstance(fusion, nn.Module):
            raise RuntimeError(
                "build_controlled_baseline_fusion must return an nn.Module"
            )
        self.fusion = fusion

    @property
    def with_bbox_head(self) -> bool:
        return self.bbox_head is not None

    def _encode_histories(
        self,
        inputs: Mapping[str, object],
    ) -> tuple[Tensor, Tensor]:
        availability = inputs.get("availability")
        if (
            not isinstance(availability, Tensor)
            or availability.ndim != 4
            or availability.shape[1:] != (2, 2, 4)
            or availability.dtype is not torch.bool
        ):
            raise ValueError("availability must be boolean [B,2,2,4]")
        batch = availability.shape[0]
        if batch <= 0:
            raise ValueError("availability batch must be positive")

        precomputed_lidar = inputs.get("lidar_history_features")
        precomputed_camera = inputs.get("camera_history_features")
        if precomputed_lidar is not None or precomputed_camera is not None:
            if not isinstance(precomputed_lidar, Tensor) or not isinstance(
                precomputed_camera,
                Tensor,
            ):
                raise ValueError("both precomputed modality histories are required")
            return precomputed_lidar, precomputed_camera

        lidar_points = inputs.get("lidar_points")
        lidar_owner = inputs.get("lidar_owner")
        if not isinstance(lidar_points, (tuple, list)) or not isinstance(
            lidar_owner,
            Tensor,
        ):
            raise ValueError("sparse LiDAR payloads and owner tensor are required")
        lidar_encoded = self.lidar_encoder(lidar_points)
        if not isinstance(lidar_encoded, Tensor):
            raise RuntimeError("lidar_encoder must return a tensor")
        lidar_history = scatter_sparse_bev_history(
            lidar_encoded,
            lidar_owner,
            availability[:, 0],
            channels=self.channels,
        )

        camera_images = inputs.get("camera_images")
        camera_owner = inputs.get("camera_owner")
        camera_intrinsics = inputs.get("camera_intrinsics")
        camera_to_agent = inputs.get("camera_agent_from_sensor")
        if not all(
            isinstance(value, Tensor)
            for value in (
                camera_images,
                camera_owner,
                camera_intrinsics,
                camera_to_agent,
            )
        ):
            raise ValueError("sparse camera tensors are required")
        assert isinstance(camera_images, Tensor)
        assert isinstance(camera_intrinsics, Tensor)
        assert isinstance(camera_to_agent, Tensor)
        camera_encoded = self.camera_encoder(
            camera_images,
            camera_intrinsics,
            camera_to_agent,
        )
        if not isinstance(camera_encoded, Tensor):
            raise RuntimeError("camera_encoder must return a tensor")
        assert isinstance(camera_owner, Tensor)
        camera_history = scatter_sparse_bev_history(
            camera_encoded,
            camera_owner,
            availability[:, 1],
            channels=self.channels,
        )
        if lidar_history.shape[0] != batch or camera_history.shape[0] != batch:
            raise RuntimeError("encoded history batch mismatch")
        return lidar_history, camera_history

    def select_baseline_inputs(
        self,
        batch_inputs_dict: Mapping[str, object],
    ) -> ControlledBaselineInputBatch:
        lidar_history, camera_history = self._encode_histories(batch_inputs_dict)
        source_to_target = batch_inputs_dict.get("source_to_target")
        availability = batch_inputs_dict.get("availability")
        selections = batch_inputs_dict.get("selections")
        if not isinstance(source_to_target, Tensor):
            raise ValueError("source_to_target tensor is required")
        if not isinstance(availability, Tensor):
            raise ValueError("availability tensor is required")
        if not isinstance(selections, ResilientBatchSelections):
            raise ValueError("ResilientBatchSelections are required")
        return self.input_selector(
            lidar_history=lidar_history,
            camera_history=camera_history,
            source_to_target=source_to_target,
            availability=availability,
            selections=selections,
        )

    def extract_controlled_feature(
        self,
        batch_inputs_dict: Mapping[str, object],
    ) -> tuple[Tensor, ControlledBaselineInputBatch]:
        selected = self.select_baseline_inputs(batch_inputs_dict)
        fused = self.fusion(
            branches=selected.branches,
            support=selected.support,
            ages=selected.ages,
        )
        expected = (
            selected.branches.shape[0],
            self.channels,
            self.grid_spec.height,
            self.grid_spec.width,
        )
        if (
            not isinstance(fused, Tensor)
            or tuple(fused.shape) != expected
            or not fused.is_floating_point()
        ):
            raise RuntimeError(
                "controlled baseline fusion must return floating [B,256,H,W]"
            )
        if (
            fused.device != selected.branches.device
            or fused.dtype != selected.branches.dtype
        ):
            raise RuntimeError(
                "controlled baseline fusion output must preserve dtype and device"
            )
        if not torch.isfinite(fused.detach()).all().item():
            raise RuntimeError(
                "controlled baseline fusion output must contain only finite values"
            )
        return fused, selected

    def _head_feature(self, fused: Tensor) -> Tensor:
        projected = (
            fused
            if self.detection_projection is None
            else self.detection_projection(fused)
        )
        if (
            not isinstance(projected, Tensor)
            or projected.ndim != 4
            or projected.shape[0] != fused.shape[0]
            or projected.shape[2:] != fused.shape[2:]
        ):
            raise RuntimeError(
                "detection projection must preserve batch and spatial shape"
            )
        return projected

    def extract_feat(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_input_metas: Sequence[Mapping[str, object]] | None = None,
        **kwargs: object,
    ) -> list[Tensor]:
        fused, _ = self.extract_controlled_feature(batch_inputs_dict)
        return [self._head_feature(fused)]

    def _forward(
        self,
        batch_inputs: Mapping[str, object],
        batch_data_samples: OptSampleList = None,
        **kwargs: object,
    ) -> object:
        fused, _ = self.extract_controlled_feature(batch_inputs)
        return self.bbox_head([self._head_feature(fused)])

    def loss(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: list[Det3DDataSample],
        **kwargs: object,
    ) -> dict[str, Tensor]:
        fused, _ = self.extract_controlled_feature(batch_inputs_dict)
        return dict(
            self.bbox_head.loss([self._head_feature(fused)], batch_data_samples)
        )

    def predict(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: list[Det3DDataSample],
        **kwargs: object,
    ) -> list[Det3DDataSample]:
        fused, selected = self.extract_controlled_feature(batch_inputs_dict)
        outputs = self.bbox_head.predict(
            [self._head_feature(fused)], batch_data_samples
        )
        results = self.add_pred_to_datasample(batch_data_samples, outputs)
        if len(results) != selected.branches.shape[0]:
            raise RuntimeError("prediction count does not match the baseline batch")
        support_rows = selected.support.detach().cpu().tolist()
        age_rows = selected.ages.detach().float().cpu().tolist()
        for index, sample in enumerate(results):
            support_by_branch = {
                key: bool(value)
                for key, value in zip(
                    CONTROLLED_BRANCH_KEYS,
                    support_rows[index],
                )
            }
            age_by_branch = {
                key: (float(age) if support_by_branch[key] else None)
                for key, age in zip(
                    CONTROLLED_BRANCH_KEYS,
                    age_rows[index],
                )
            }
            sample.set_metainfo(
                {
                    "controlled_baseline_diagnostics": {
                        "schema_version": 1,
                        "sample_id": selected.sample_ids[index],
                        "method": self.baseline_name,
                        "branch_order": CONTROLLED_BRANCH_KEYS,
                        "support": support_by_branch,
                        "age_intervals": age_by_branch,
                    }
                }
            )
        return results


__all__ = ("ControlledCooperativeBaselineNet",)
