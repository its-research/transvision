from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from torch import Tensor, nn
from torch.nn import functional as F

from transvision.models.voxel import Voxelization
from .utils import PixelWeightedFusion, ReduceInfTC


class PerceptionTrajectoryField(nn.Module):

    def __init__(self, in_channels: int, history_steps: int = 3, hidden_channels: int = 64) -> None:
        super().__init__()
        self.history_steps = history_steps
        self.flow_encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, 2, kernel_size=1),
        )

    def forward(self, history_feats: Tensor) -> Tensor:
        if history_feats.ndim != 5:
            raise ValueError('history_feats should be a 5D tensor [B, T, C, H, W]')
        history_feats = history_feats.transpose(1, 2)
        flow = self.flow_encoder(history_feats)
        return flow.squeeze(2)

    def warp(self, features: Tensor, flow: Tensor, delay_scale: Tensor) -> Tensor:
        batch_size, _, height, width = features.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=features.device, dtype=features.dtype),
            torch.linspace(-1.0, 1.0, width, device=features.device, dtype=features.dtype),
            indexing='ij',
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        flow_x = flow[:, 0] / ((width - 1) / 2.0)
        flow_y = flow[:, 1] / ((height - 1) / 2.0)
        flow_grid = torch.stack((flow_x, flow_y), dim=-1)
        delay_scale = delay_scale.view(batch_size, 1, 1, 1)
        warped_grid = base_grid + flow_grid * delay_scale
        return F.grid_sample(features, warped_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    def align(self, delayed_feat: Tensor, flow: Tensor, latency_ms: Tensor) -> Tensor:
        delay_scale = latency_ms / torch.clamp(latency_ms.max(), min=1.0)
        return self.warp(delayed_feat, flow, delay_scale)

    def reconstruct(self, last_valid_feat: Tensor, flow: Tensor) -> Tensor:
        delay_scale = torch.ones(last_valid_feat.size(0), device=last_valid_feat.device, dtype=last_valid_feat.dtype)
        return self.warp(last_valid_feat, flow, delay_scale)


class DynamicExpertRouting(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        self.lidar_expert = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.camera_expert = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.synergy_expert = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gating = nn.Sequential(
            nn.Linear(in_channels * 3 + 3, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 3),
        )

    def forward(
        self,
        lidar_feat: Tensor,
        camera_feat: Tensor,
        fusion_feat: Tensor,
        modality_mask: Tensor,
        latency_ms: Tensor,
    ) -> Tensor:
        lidar_tokens = F.adaptive_avg_pool2d(lidar_feat, (1, 1)).flatten(1)
        camera_tokens = F.adaptive_avg_pool2d(camera_feat, (1, 1)).flatten(1)
        fusion_tokens = F.adaptive_avg_pool2d(fusion_feat, (1, 1)).flatten(1)
        gating_inputs = torch.cat(
            [lidar_tokens, camera_tokens, fusion_tokens, modality_mask, latency_ms.view(-1, 1)],
            dim=1,
        )
        logits = self.gating(gating_inputs)
        mask = torch.stack([modality_mask[:, 0], modality_mask[:, 1], torch.ones_like(modality_mask[:, 0])], dim=1)
        logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
        weights = F.softmax(logits, dim=1).view(-1, 3, 1, 1, 1)

        expert_outputs = torch.stack(
            [self.lidar_expert(lidar_feat), self.camera_expert(camera_feat), self.synergy_expert(fusion_feat)],
            dim=1,
        )
        fused = (expert_outputs * weights).sum(dim=1)
        return fused


@MODELS.register_module()
class ResilientV2XNet(Base3DDetector):

    def __init__(
        self,
        mode: str = 'fusion',
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        ptf_cfg: Optional[dict] = None,
        der_cfg: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.mode = mode
        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)
        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
        self.img_backbone = MODELS.build(img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        self.fusion_layer = MODELS.build(fusion_layer) if fusion_layer is not None else None
        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)
        self.bbox_head = MODELS.build(bbox_head)

        if 'fusion' in self.mode:
            self.inf_pts_voxel_layer = Voxelization(**voxelize_cfg)
            self.inf_pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
            self.inf_pts_middle_encoder = MODELS.build(pts_middle_encoder)
            self.inf_pts_backbone = MODELS.build(pts_backbone)
            self.inf_pts_neck = MODELS.build(pts_neck)
            self.fusion_weighted = PixelWeightedFusion(512)
            self.encoder = ReduceInfTC(1024)

        ptf_cfg = ptf_cfg or {}
        der_cfg = der_cfg or {}
        self.ptf = PerceptionTrajectoryField(**ptf_cfg)
        self.der = DynamicExpertRouting(**der_cfg)

        self.init_weights()

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        batch_size, num_cams, channels, height, width = x.size()
        x = x.view(batch_size * num_cams, channels, height, width).contiguous()
        x = self.img_backbone(x)
        x = self.img_neck(x)
        if not isinstance(x, torch.Tensor):
            x = x[0]

        bn, channels, height, width = x.size()
        x = x.view(batch_size, int(bn / batch_size), channels, height, width)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return x

    def extract_pts_feat(self, batch_inputs_dict, points_view='vehicle') -> torch.Tensor:
        if points_view == 'vehicle':
            points = batch_inputs_dict['points']
            with torch.autocast('cuda', enabled=False):
                points = [point.float() for point in points]
                feats, coords, sizes = self.voxelize(points)
                batch_size = coords[-1, 0] + 1
            x = self.pts_middle_encoder(feats, coords, batch_size)
            return x
        if points_view == 'infrastructure':
            points = batch_inputs_dict['infrastructure_points']
            with torch.autocast('cuda', enabled=False):
                points = [point.float() for point in points]
                feats, coords, sizes = self.inf_voxelize(points)
                batch_size = coords[-1, 0] + 1
            x = self.inf_pts_middle_encoder(feats, coords, batch_size)
            return x
        raise ValueError(f'Unknown points_view: {points_view}')

    def predict(
        self,
        batch_inputs_dict: Dict[str, Optional[Tensor]],
        batch_data_samples: List[Det3DDataSample],
        **kwargs,
    ) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        outputs = self.bbox_head.predict(feats, batch_input_metas)
        return self.add_pred_to_datasample(batch_data_samples, outputs)

    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for meta in batch_input_metas:
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.asarray(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))

            img_feature = self.extract_img_feat(
                imgs,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                batch_input_metas,
            )
            features.append(img_feature)

        pts_feature = self.extract_pts_feat(batch_inputs_dict, points_view='vehicle')
        features.append(pts_feature)

        if self.fusion_layer is not None:
            veh_x = self.fusion_layer(features)
        else:
            veh_x = features[-1]

        veh_x = self.pts_backbone(veh_x)
        veh_x = self.pts_neck(veh_x)
        if self.mode == 'veh_only':
            return veh_x

        inf_pts_feature = self.extract_pts_feat(batch_inputs_dict, points_view='infrastructure')
        inf_x = self.inf_pts_backbone(inf_pts_feature)
        inf_x = self.inf_pts_neck(inf_x)
        inf_x[0] = self.encoder(inf_x[0])

        history_feats = batch_inputs_dict.get('infrastructure_history', None)
        if history_feats is None:
            history_feats = inf_x[0].unsqueeze(1).repeat(1, self.ptf.history_steps, 1, 1, 1)
        flow = self.ptf(history_feats)
        latency_ms = torch.tensor(
            [meta.get('v2x_latency_ms', 0.0) for meta in batch_input_metas],
            device=inf_x[0].device,
            dtype=inf_x[0].dtype,
        )
        aligned_inf = self.ptf.align(inf_x[0], flow, latency_ms)

        modality_mask = torch.tensor(
            [meta.get('modality_mask', [1.0, 1.0]) for meta in batch_input_metas],
            device=inf_x[0].device,
            dtype=inf_x[0].dtype,
        )
        lidar_feat = veh_x[0]
        camera_feat = veh_x[0] if len(features) == 1 else features[0]
        fusion_feat = self.fusion_weighted(torch.cat([veh_x[0], aligned_inf], dim=1))
        fused = self.der(lidar_feat, camera_feat, fusion_feat, modality_mask, latency_ms)
        return [fused]

    def loss(
        self,
        batch_inputs_dict: Dict[str, Optional[Tensor]],
        batch_data_samples: List[Det3DDataSample],
        **kwargs,
    ) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        losses = self.bbox_head.loss(feats, batch_data_samples)
        return losses

    def add_pred_to_datasample(
        self,
        data_samples: List[Det3DDataSample],
        results: List[Det3DDataSample],
    ) -> List[Det3DDataSample]:
        for data_sample, pred_instances in zip(data_samples, results):
            data_sample.pred_instances_3d = pred_instances
        return data_samples

    @torch.no_grad()
    def voxelize(self, points: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        if self.voxelize_reduce:
            voxels = voxels.sum(dim=1, keepdim=False) / num_points.type_as(voxels).view(-1, 1)
            voxels = voxels.contiguous()
        return voxels, coors_batch, num_points

    @torch.no_grad()
    def inf_voxelize(self, points: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.inf_pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        if self.voxelize_reduce:
            voxels = voxels.sum(dim=1, keepdim=False) / num_points.type_as(voxels).view(-1, 1)
            voxels = voxels.contiguous()
        return voxels, coors_batch, num_points
