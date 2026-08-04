from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from torch import Tensor, nn
from torch.nn import functional as F

from transvision.models.resilient_v2x import (
    BEVGridSpec,
    CONTROLLED_BRANCH_KEYS,
    FrozenTeacher,
    ResilientBatchSelections,
    ResilientFeatureBatch,
    ResilientV2XFeatureFusion,
    assert_teacher_frozen,
    bev_xy_to_yx,
    head_distillation_losses,
    scatter_sparse_bev_history,
)
from transvision.models.voxel import Voxelization


def _build_optional(config: object) -> nn.Module | None:
    if config is None:
        return None
    if isinstance(config, nn.Module):
        return config
    if not isinstance(config, Mapping):
        raise ValueError("module config must be a mapping or nn.Module")
    return MODELS.build(dict(config))


def _first_tensor(value: object, name: str) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], Tensor):
        return value[0]
    raise RuntimeError(f"{name} must produce a tensor or non-empty tensor sequence")


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters(), None)
    if parameter is None:
        buffer = next(module.buffers(), None)
        if buffer is None:
            return torch.device("cpu"), torch.float32
        return buffer.device, buffer.dtype
    return parameter.device, parameter.dtype


def _autocast_feature_output(value: Tensor) -> Tensor:
    """Normalize shared-encoder outputs to the active CUDA AMP dtype."""

    if (
        value.device.type == "cuda"
        and torch.is_autocast_enabled()
        and value.dtype is torch.float32
    ):
        return value.to(dtype=torch.get_autocast_gpu_dtype())
    return value


def _initialize_nested_modules(modules: Sequence[nn.Module | None]) -> None:
    """Propagate MMEngine initialization through shared nn.Module wrappers."""

    seen: set[int] = set()
    for module in modules:
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        initializer = getattr(module, "init_weights", None)
        if callable(initializer) and not bool(getattr(module, "is_init", False)):
            initializer()


@MODELS.register_module()
class SharedPointPillarsBEVEncoder(nn.Module):
    """One shared PointPillars/SECOND encoder for every agent and tick."""

    def __init__(
        self,
        voxelize_cfg: Mapping[str, object],
        middle_encoder: Mapping[str, object] | nn.Module,
        backbone: Mapping[str, object] | nn.Module,
        neck: Mapping[str, object] | nn.Module,
        output_height: int,
        output_width: int,
        voxel_encoder: Mapping[str, object] | nn.Module | None = None,
        output_projection: Mapping[str, object] | nn.Module | None = None,
        voxelize_reduce: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(voxelize_cfg, Mapping):
            raise ValueError("voxelize_cfg must be a mapping")
        if type(voxelize_reduce) is not bool:
            raise ValueError("voxelize_reduce must be boolean")
        if type(output_height) is not int or output_height <= 0:
            raise ValueError("output_height must be positive")
        if type(output_width) is not int or output_width <= 0:
            raise ValueError("output_width must be positive")
        self.voxel_layer = Voxelization(**dict(voxelize_cfg))
        self.voxelize_reduce = voxelize_reduce
        self.voxel_encoder = _build_optional(voxel_encoder)
        self.middle_encoder = _build_optional(middle_encoder)
        self.backbone = _build_optional(backbone)
        self.neck = _build_optional(neck)
        self.output_projection = _build_optional(output_projection)
        if self.middle_encoder is None or self.backbone is None or self.neck is None:
            raise ValueError("middle_encoder, backbone, and neck are required")
        self.output_height = output_height
        self.output_width = output_width

    def init_weights(self) -> None:
        _initialize_nested_modules(
            (
                self.voxel_encoder,
                self.middle_encoder,
                self.backbone,
                self.neck,
                self.output_projection,
            )
        )

    @torch.no_grad()
    def _voxelize(
        self,
        points: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        features: list[Tensor] = []
        coordinates: list[Tensor] = []
        sizes: list[Tensor] = []
        for batch_index, point_tensor in enumerate(points):
            if not isinstance(point_tensor, Tensor) or point_tensor.ndim != 2:
                raise ValueError("every point payload must be a rank-2 tensor")
            result = self.voxel_layer(point_tensor.float())
            if len(result) == 3:
                feature, coordinate, size = result
                sizes.append(size)
            elif len(result) == 2:
                feature, coordinate = result
            else:
                raise RuntimeError("voxelizer returned an unsupported tuple")
            features.append(feature)
            coordinates.append(
                F.pad(coordinate, (1, 0), mode="constant", value=batch_index)
            )
        if not features:
            raise ValueError("PointPillars encoder requires at least one payload")
        feature = torch.cat(features, dim=0)
        coordinate = torch.cat(coordinates, dim=0)
        size_tensor = torch.cat(sizes, dim=0) if sizes else None
        return feature, coordinate, size_tensor

    def forward(self, points: Sequence[Tensor]) -> Tensor:
        if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
            raise ValueError("points must be a sequence")
        if not points:
            device, dtype = _module_device_dtype(self)
            return torch.empty(
                0,
                256,
                self.output_height,
                self.output_width,
                device=device,
                dtype=dtype,
            )
        feature, coordinate, sizes = self._voxelize(points)
        if self.voxel_encoder is not None:
            if sizes is None:
                feature = self.voxel_encoder(feature, coordinate)
            else:
                feature = self.voxel_encoder(feature, sizes, coordinate)
        elif sizes is not None and self.voxelize_reduce:
            feature = feature.sum(dim=1) / sizes.type_as(feature).clamp_min(1).view(
                -1, 1
            )
            feature = feature.contiguous()
        batch_size = len(points)
        encoded = self.middle_encoder(feature, coordinate, batch_size)
        encoded = self.backbone(encoded)
        encoded = self.neck(encoded)
        encoded = _first_tensor(encoded, "PointPillars neck")
        if self.output_projection is not None:
            encoded = self.output_projection(encoded)
            encoded = _first_tensor(encoded, "PointPillars output projection")
        if encoded.shape != (
            batch_size,
            256,
            self.output_height,
            self.output_width,
        ):
            raise RuntimeError(
                "PointPillars encoder must output [K,256,grid_height,grid_width]"
            )
        return _autocast_feature_output(encoded)


@MODELS.register_module()
class SharedResNetLSSBEVEncoder(nn.Module):
    """One shared ResNet50 + LSS encoder for every camera packet."""

    def __init__(
        self,
        image_backbone: Mapping[str, object] | nn.Module,
        image_neck: Mapping[str, object] | nn.Module,
        view_transform: Mapping[str, object] | nn.Module,
        output_height: int,
        output_width: int,
        bev_backbone: Mapping[str, object] | nn.Module | None = None,
        bev_neck: Mapping[str, object] | nn.Module | None = None,
        output_projection: Mapping[str, object] | nn.Module | None = None,
        view_transform_output_order: Literal["xy", "yx"] = "xy",
    ) -> None:
        super().__init__()
        self.image_backbone = _build_optional(image_backbone)
        self.image_neck = _build_optional(image_neck)
        self.view_transform = _build_optional(view_transform)
        self.bev_backbone = _build_optional(bev_backbone)
        self.bev_neck = _build_optional(bev_neck)
        self.output_projection = _build_optional(output_projection)
        if (
            self.image_backbone is None
            or self.image_neck is None
            or self.view_transform is None
        ):
            raise ValueError("image backbone, neck, and view transform are required")
        if type(output_height) is not int or output_height <= 0:
            raise ValueError("output_height must be positive")
        if type(output_width) is not int or output_width <= 0:
            raise ValueError("output_width must be positive")
        if view_transform_output_order not in ("xy", "yx"):
            raise ValueError("view_transform_output_order must be 'xy' or 'yx'")
        self.output_height = output_height
        self.output_width = output_width
        self.view_transform_output_order = view_transform_output_order

    def init_weights(self) -> None:
        _initialize_nested_modules(
            (
                self.image_backbone,
                self.image_neck,
                self.view_transform,
                self.bev_backbone,
                self.bev_neck,
                self.output_projection,
            )
        )

    def forward(
        self,
        images: Tensor,
        intrinsics: Tensor,
        camera_to_agent: Tensor,
    ) -> Tensor:
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError("images must have shape [K,3,H,W]")
        count = images.shape[0]
        if count == 0:
            device, dtype = _module_device_dtype(self)
            return torch.empty(
                0,
                256,
                self.output_height,
                self.output_width,
                device=device,
                dtype=dtype,
            )
        if images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("images must be floating RGB tensors")
        if intrinsics.shape != (count, 3, 3):
            raise ValueError("intrinsics must have shape [K,3,3]")
        if camera_to_agent.shape != (count, 4, 4):
            raise ValueError("camera_to_agent must have shape [K,4,4]")
        if (
            intrinsics.device != images.device
            or camera_to_agent.device != images.device
            or intrinsics.dtype != images.dtype
            or camera_to_agent.dtype != images.dtype
        ):
            raise ValueError("camera tensors must share dtype and device")

        image_features = self.image_backbone(images)
        image_features = self.image_neck(image_features)
        image_features = _first_tensor(image_features, "image neck")
        image_features = image_features.view(
            count,
            1,
            image_features.shape[1],
            image_features.shape[2],
            image_features.shape[3],
        )

        intrinsic4 = (
            torch.eye(4, device=images.device, dtype=images.dtype)
            .view(
                1,
                1,
                4,
                4,
            )
            .repeat(count, 1, 1, 1)
        )
        intrinsic4[:, 0, :3, :3] = intrinsics
        camera_to_agent = camera_to_agent.view(count, 1, 4, 4)
        agent_to_camera = torch.linalg.inv(camera_to_agent)
        lidar_to_image = intrinsic4 @ agent_to_camera
        identity = (
            torch.eye(
                4,
                device=images.device,
                dtype=images.dtype,
            )
            .view(1, 1, 4, 4)
            .repeat(count, 1, 1, 1)
        )
        points = [images.new_empty(0, 4) for _ in range(count)]
        metas = [{} for _ in range(count)]
        bev = self.view_transform(
            image_features,
            points,
            lidar_to_image,
            intrinsic4,
            camera_to_agent,
            identity,
            identity[:, 0],
            metas,
        )
        # This repository's LSSTransform passes nx[0] (X) as the pooling H
        # dimension and nx[1] (Y) as W, so its tensor is [B,C,X,Y].  Every
        # causal alignment/PTF operation and PointPillars uses [B,C,Y,X].
        if self.view_transform_output_order == "xy":
            bev = bev_xy_to_yx(bev)
        if self.bev_backbone is not None:
            bev = self.bev_backbone(bev)
        if self.bev_neck is not None:
            bev = self.bev_neck(bev)
        bev = _first_tensor(bev, "camera BEV encoder")
        if self.output_projection is not None:
            bev = self.output_projection(bev)
            bev = _first_tensor(bev, "camera output projection")
        if bev.shape != (
            count,
            256,
            self.output_height,
            self.output_width,
        ):
            raise RuntimeError(
                "camera encoder must output [K,256,grid_height,grid_width]"
            )
        return _autocast_feature_output(bev)


@MODELS.register_module()
class VehiclePointPillarsPretrainNet(Base3DDetector):
    """Pretrain the shared LiDAR encoder and Car head on current ego scans."""

    def __init__(
        self,
        lidar_encoder: Mapping[str, object] | nn.Module,
        bbox_head: Mapping[str, object] | nn.Module,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        **kwargs: object,
    ) -> None:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise ValueError(f"unexpected vehicle pretraining options: {unexpected}")
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.lidar_encoder = _build_optional(lidar_encoder)
        self.bbox_head = _build_optional(bbox_head)
        if self.lidar_encoder is None or self.bbox_head is None:
            raise ValueError("lidar_encoder and bbox_head are required")

    @property
    def with_bbox_head(self) -> bool:
        return self.bbox_head is not None

    def _vehicle_feature(self, inputs: Mapping[str, object]) -> Tensor:
        points = inputs.get("lidar_points")
        owners = inputs.get("lidar_owner")
        availability = inputs.get("availability")
        if not isinstance(points, (tuple, list)) or not isinstance(owners, Tensor):
            raise ValueError("sparse LiDAR payloads and owner tensor are required")
        if owners.ndim != 2 or owners.shape[1] != 3 or owners.dtype != torch.long:
            raise ValueError("lidar_owner must be int64 [K,3]")
        if len(points) != owners.shape[0]:
            raise ValueError("lidar payload and owner counts must match")
        if not isinstance(availability, Tensor) or availability.ndim != 4:
            raise ValueError("availability must define the batch dimension")
        batch_size = int(availability.shape[0])
        mask = (owners[:, 1] == 0) & (owners[:, 2] == 0)
        selected = torch.nonzero(mask, as_tuple=False).flatten()
        if selected.numel() != batch_size:
            raise RuntimeError(
                "vehicle pretraining requires exactly one current ego LiDAR "
                "payload per sample"
            )
        batch_indices = owners[selected, 0]
        order = torch.argsort(batch_indices, stable=True)
        selected = selected[order]
        batch_indices = batch_indices[order]
        expected = torch.arange(batch_size, device=owners.device)
        if not torch.equal(batch_indices, expected):
            raise RuntimeError(
                "current ego LiDAR payloads must cover every batch sample exactly once"
            )
        selected_points = tuple(points[index] for index in selected.cpu().tolist())
        feature = self.lidar_encoder(selected_points)
        if not isinstance(feature, Tensor) or feature.shape[0] != batch_size:
            raise RuntimeError("lidar_encoder returned an invalid vehicle batch")
        return feature

    def extract_feat(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_input_metas: Sequence[Mapping[str, object]] | None = None,
        **kwargs: object,
    ) -> list[Tensor]:
        return [self._vehicle_feature(batch_inputs_dict)]

    def _forward(
        self,
        batch_inputs: Mapping[str, object],
        batch_data_samples: OptSampleList = None,
        **kwargs: object,
    ) -> object:
        return self.bbox_head(self.extract_feat(batch_inputs))

    def loss(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: list[Det3DDataSample],
        **kwargs: object,
    ) -> dict[str, Tensor]:
        return dict(
            self.bbox_head.loss(
                self.extract_feat(batch_inputs_dict),
                batch_data_samples,
            )
        )

    def predict(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: list[Det3DDataSample],
        **kwargs: object,
    ) -> list[Det3DDataSample]:
        outputs = self.bbox_head.predict(
            self.extract_feat(batch_inputs_dict),
            batch_data_samples,
        )
        results = self.add_pred_to_datasample(batch_data_samples, outputs)
        for sample in results:
            sample_id = sample.metainfo.get(
                "sample_id",
                sample.metainfo.get("sample_idx"),
            )
            sample.set_metainfo(
                {
                    "controlled_baseline_diagnostics": {
                        "schema_version": 1,
                        "sample_id": sample_id,
                        "method": "vehicle_pointpillars_pretrain",
                        "branch_order": CONTROLLED_BRANCH_KEYS,
                        "support": {
                            key: key == "lidar_ego"
                            for key in CONTROLLED_BRANCH_KEYS
                        },
                        "age_intervals": {
                            key: (0.0 if key == "lidar_ego" else None)
                            for key in CONTROLLED_BRANCH_KEYS
                        },
                    }
                }
            )
        return results


def _nested_tensor(value: object, path: Sequence[str | int]) -> Tensor:
    current = value
    for item in path:
        if isinstance(item, int):
            if not isinstance(current, (list, tuple)):
                raise RuntimeError("distillation logit path expects a sequence")
            current = current[item]
        else:
            if not isinstance(current, Mapping):
                raise RuntimeError("distillation logit path expects a mapping")
            current = current[item]
    if not isinstance(current, Tensor) or current.ndim != 4:
        raise RuntimeError("distillation logit path must resolve to rank-4 Tensor")
    return current


@MODELS.register_module()
class ResilientV2XNet(Base3DDetector):
    """Causal, multimodal, teacher-student ResilientV2X detector."""

    def __init__(
        self,
        grid_spec: Mapping[str, object],
        lidar_encoder: Mapping[str, object] | nn.Module,
        camera_encoder: Mapping[str, object] | nn.Module,
        bbox_head: Mapping[str, object] | nn.Module,
        data_preprocessor: OptConfigType = None,
        ptf_mode: Literal["nonlinear", "linear", "none"] = "nonlinear",
        routing_mode: Literal["dynamic", "static", "uniform", "concat"] = "dynamic",
        use_reliability: bool = True,
        use_delay_metadata: bool = True,
        delta_t_ms: int = 100,
        teacher: Mapping[str, object] | nn.Module | None = None,
        teacher_checkpoint: str | None = None,
        distillation: Mapping[str, object] | None = None,
        init_cfg: OptMultiConfig = None,
        **kwargs,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if not isinstance(grid_spec, Mapping):
            raise ValueError("grid_spec must be a mapping")
        self.grid_spec = BEVGridSpec(**dict(grid_spec))
        self.lidar_encoder = _build_optional(lidar_encoder)
        self.camera_encoder = _build_optional(camera_encoder)
        self.bbox_head = _build_optional(bbox_head)
        if (
            self.lidar_encoder is None
            or self.camera_encoder is None
            or self.bbox_head is None
        ):
            raise ValueError("both encoders and bbox_head are required")
        self.resilient_fusion = ResilientV2XFeatureFusion(
            grid_spec=self.grid_spec,
            ptf_mode=ptf_mode,
            routing_mode=routing_mode,
            use_reliability=use_reliability,
            use_delay_metadata=use_delay_metadata,
            delta_t_ms=delta_t_ms,
        )

        self.teacher: FrozenTeacher | None = None
        self.distillation_cfg: dict[str, object] | None = None
        if teacher is not None:
            teacher_model = _build_optional(teacher)
            if not isinstance(teacher_model, ResilientV2XNet):
                raise ValueError("teacher must build another ResilientV2XNet")
            if type(teacher_checkpoint) is not str or not teacher_checkpoint:
                raise ValueError(
                    "teacher_checkpoint is required to avoid a random frozen teacher"
                )
            from mmengine.runner import load_checkpoint

            load_checkpoint(
                teacher_model,
                teacher_checkpoint,
                map_location="cpu",
                strict=True,
            )
            self.teacher = FrozenTeacher(teacher_model)
            if not isinstance(distillation, Mapping):
                raise ValueError("distillation config is required with teacher")
            required = {
                "temperature",
                "lambda_feature",
                "lambda_logit",
                "head_type",
                "logit_path",
            }
            if frozenset(distillation) != required:
                raise ValueError("distillation config fields mismatch")
            path = distillation["logit_path"]
            if not isinstance(path, (list, tuple)) or any(
                not isinstance(item, (str, int)) for item in path
            ):
                raise ValueError(
                    "distillation logit_path must be a string/int sequence"
                )
            self.distillation_cfg = dict(distillation)
        elif distillation is not None or teacher_checkpoint is not None:
            raise ValueError(
                "distillation config and teacher_checkpoint require a teacher"
            )

    @property
    def with_bbox_head(self) -> bool:
        return self.bbox_head is not None

    def train(self, mode: bool = True) -> "ResilientV2XNet":
        super().train(mode)
        if self.teacher is not None:
            self.teacher.train(False)
            assert_teacher_frozen(self.teacher.teacher)
        return self

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
        lidar_history = scatter_sparse_bev_history(
            lidar_encoded,
            lidar_owner,
            availability[:, 0],
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
        camera_encoded = self.camera_encoder(
            camera_images,
            camera_intrinsics,
            camera_to_agent,
        )
        camera_history = scatter_sparse_bev_history(
            camera_encoded,
            camera_owner,
            availability[:, 1],
        )
        if lidar_history.shape[0] != batch or camera_history.shape[0] != batch:
            raise RuntimeError("encoded history batch mismatch")
        return lidar_history, camera_history

    def extract_resilient_feature(
        self,
        batch_inputs_dict: Mapping[str, object],
    ) -> ResilientFeatureBatch:
        lidar_history, camera_history = self._encode_histories(batch_inputs_dict)
        transforms = batch_inputs_dict.get("source_to_target")
        availability = batch_inputs_dict.get("availability")
        selections = batch_inputs_dict.get("selections")
        if (
            not isinstance(transforms, Tensor)
            or transforms.ndim != 6
            or transforms.shape[1:] != (2, 2, 4, 4, 4)
        ):
            raise ValueError("source_to_target must have shape [B,2,2,4,4,4]")
        if not isinstance(availability, Tensor):
            raise ValueError("availability tensor is required")
        if not isinstance(selections, ResilientBatchSelections):
            raise ValueError("ResilientBatchSelections are required")
        return self.resilient_fusion(
            lidar_history=lidar_history,
            camera_history=camera_history,
            lidar_source_to_target=transforms[:, 0],
            camera_source_to_target=transforms[:, 1],
            lidar_availability=availability[:, 0],
            camera_availability=availability[:, 1],
            selections=selections,
        )

    def extract_feat(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_input_metas: Sequence[Mapping[str, object]] | None = None,
        **kwargs,
    ) -> list[Tensor]:
        return [self.extract_resilient_feature(batch_inputs_dict).fused]

    def _forward(
        self,
        batch_inputs: Mapping[str, object],
        batch_data_samples: OptSampleList = None,
        **kwargs,
    ) -> object:
        feature = self.extract_resilient_feature(batch_inputs)
        return self.bbox_head([feature.fused])

    def predict(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: list[Det3DDataSample],
        **kwargs,
    ) -> list[Det3DDataSample]:
        feature = self.extract_resilient_feature(batch_inputs_dict)
        outputs = self.bbox_head.predict(
            [feature.fused],
            batch_data_samples,
        )
        results = self.add_pred_to_datasample(batch_data_samples, outputs)
        for sample, diagnostic in zip(results, feature.diagnostics):
            sample.set_metainfo({"resilient_v2x_diagnostics": diagnostic})
        return results

    def loss(
        self,
        batch_inputs_dict: Mapping[str, object],
        batch_data_samples: list[Det3DDataSample],
        **kwargs,
    ) -> dict[str, Tensor]:
        teacher_feature: ResilientFeatureBatch | None = None
        teacher_logits: Tensor | None = None
        if self.teacher is not None:
            if self.distillation_cfg is None:
                raise RuntimeError("teacher is configured without distillation")
            clean_inputs = batch_inputs_dict.get("teacher_clean")
            if not isinstance(clean_inputs, Mapping):
                raise RuntimeError(
                    "training with a teacher requires teacher_clean inputs"
                )
            path = self.distillation_cfg["logit_path"]
            if not isinstance(path, (list, tuple)):
                raise RuntimeError("distillation logit_path is invalid")
            assert_teacher_frozen(self.teacher.teacher)
            # Run the frozen path before constructing the student's autograd
            # graph.  This is mathematically identical and avoids overlapping
            # teacher activations with dense multi-frame student activations.
            with torch.no_grad():
                teacher_feature = self.teacher.teacher.extract_resilient_feature(
                    clean_inputs
                )
                teacher_raw = self.teacher.teacher.bbox_head([teacher_feature.fused])
                teacher_logits = _nested_tensor(teacher_raw, path)

        student = self.extract_resilient_feature(batch_inputs_dict)
        losses = dict(self.bbox_head.loss([student.fused], batch_data_samples))
        if self.teacher is None:
            return losses
        if (
            self.distillation_cfg is None
            or teacher_feature is None
            or teacher_logits is None
        ):
            raise RuntimeError("teacher distillation state is incomplete")
        student_raw = self.bbox_head([student.fused])
        path = self.distillation_cfg["logit_path"]
        if not isinstance(path, (list, tuple)):
            raise RuntimeError("distillation logit_path is invalid")
        student_logits = _nested_tensor(student_raw, path)
        distilled = head_distillation_losses(
            teacher_feature=teacher_feature.fused,
            student_feature=student.fused,
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            temperature=float(self.distillation_cfg["temperature"]),
            lambda_feature=float(self.distillation_cfg["lambda_feature"]),
            lambda_logit=float(self.distillation_cfg["lambda_logit"]),
            valid_sample_mask=student.overall_support,
            head_type=str(self.distillation_cfg["head_type"]),
        )
        losses["loss_distillation"] = distilled.total
        losses["distillation_feature"] = distilled.feature.detach()
        losses["distillation_logit"] = distilled.logit.detach()
        return losses


__all__ = (
    "SharedPointPillarsBEVEncoder",
    "SharedResNetLSSBEVEncoder",
    "VehiclePointPillarsPretrainNet",
    "ResilientV2XNet",
)
