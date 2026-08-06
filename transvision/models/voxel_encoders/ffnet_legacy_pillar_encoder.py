"""FFNet v0.17-compatible PointPillars feature decoration."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from mmdet3d.models.voxel_encoders.utils import PFNLayer, get_paddings_indicator
from mmdet3d.registry import MODELS
from torch import Tensor, nn


@MODELS.register_module()
class FFNetLegacyPillarFeatureNet(nn.Module):
    """Reproduce the 9-channel PFN input used by FFNet's mmdet3d v0.17.

    FFNet decorates each four-channel point with three cluster offsets and
    two XY pillar-center offsets. MMDetection3D 1.3 adds a third Z
    pillar-center offset, producing ten channels and making the released
    ``[64, 9]`` PFN weights incompatible.
    """

    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: Tuple[int, ...] = (64,),
        with_distance: bool = False,
        with_cluster_center: bool = True,
        with_voxel_center: bool = True,
        voxel_size: Tuple[float, ...] = (0.2, 0.2, 4.0),
        point_cloud_range: Tuple[float, ...] = (
            0.0,
            -40.0,
            -3.0,
            70.4,
            40.0,
            1.0,
        ),
        norm_cfg: Optional[dict] = None,
        mode: str = "max",
        legacy: bool = True,
    ) -> None:
        super().__init__()
        if not feat_channels:
            raise ValueError("feat_channels must be non-empty")
        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.legacy = legacy
        decorated_channels = int(in_channels)
        if with_cluster_center:
            decorated_channels += 3
        if with_voxel_center:
            decorated_channels += 2
        if with_distance:
            decorated_channels += 1
        self.in_channels = decorated_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center

        channels = [decorated_channels, *feat_channels]
        layers = []
        for index, (input_channels, output_channels) in enumerate(
            zip(channels[:-1], channels[1:])
        ):
            layers.append(
                PFNLayer(
                    input_channels,
                    output_channels,
                    norm_cfg=norm_cfg,
                    last_layer=index == len(channels) - 2,
                    mode=mode,
                )
            )
        self.pfn_layers = nn.ModuleList(layers)

        self.vx = float(voxel_size[0])
        self.vy = float(voxel_size[1])
        self.x_offset = self.vx / 2 + float(point_cloud_range[0])
        self.y_offset = self.vy / 2 + float(point_cloud_range[1])
        self.point_cloud_range = point_cloud_range

    def forward(
        self,
        features: Tensor,
        num_points: Tensor,
        coors: Tensor,
        *args: object,
        **kwargs: object,
    ) -> Tensor:
        features_ls = [features]
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True
            ) / num_points.type_as(features).view(-1, 1, 1)
            features_ls.append(features[:, :, :3] - points_mean)

        if self._with_voxel_center:
            dtype = features.dtype
            if not self.legacy:
                center = torch.zeros_like(features[:, :, :2])
                center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx
                    + self.x_offset
                )
                center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy
                    + self.y_offset
                )
            else:
                # Preserve the released model's legacy in-place view behavior.
                center = features[:, :, :2]
                center[:, :, 0] = center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx
                    + self.x_offset
                )
                center[:, :, 1] = center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy
                    + self.y_offset
                )
            features_ls.append(center)

        if self._with_distance:
            features_ls.append(torch.norm(features[:, :, :3], 2, 2, keepdim=True))

        decorated = torch.cat(features_ls, dim=-1)
        voxel_count = decorated.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        decorated *= torch.unsqueeze(mask, -1).type_as(decorated)
        for layer in self.pfn_layers:
            decorated = layer(decorated, num_points)
        return decorated.squeeze(1)
