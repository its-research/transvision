# Copyright (c) DAIR-V2X (AIR). All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
# from mmcv.runner import force_fp32
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from transvision.v2x_utils.visual import save_feature_map


class ReduceInfTC(nn.Module):

    def __init__(self, channel):
        super(ReduceInfTC, self).__init__()
        self.conv1_2 = nn.Conv2d(channel // 2, channel // 4, kernel_size=3, stride=2, padding=0)
        self.bn1_2 = nn.BatchNorm2d(channel // 4, track_running_stats=True)
        self.conv1_3 = nn.Conv2d(channel // 4, channel // 8, kernel_size=3, stride=2, padding=0)
        self.bn1_3 = nn.BatchNorm2d(channel // 8, track_running_stats=True)
        self.conv1_4 = nn.Conv2d(channel // 8, channel // 64, kernel_size=3, stride=2, padding=1)
        self.bn1_4 = nn.BatchNorm2d(channel // 64, track_running_stats=True)

        self.deconv2_1 = nn.ConvTranspose2d(channel // 64, channel // 8, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = nn.BatchNorm2d(channel // 8, track_running_stats=True)
        self.deconv2_2 = nn.ConvTranspose2d(channel // 8, channel // 4, kernel_size=3, stride=2, padding=0)
        self.bn2_2 = nn.BatchNorm2d(channel // 4, track_running_stats=True)
        self.deconv2_3 = nn.ConvTranspose2d(
            channel // 4,
            channel // 2,
            kernel_size=3,
            stride=2,
            padding=0,
            output_padding=1,
        )
        self.bn2_3 = nn.BatchNorm2d(channel // 2, track_running_stats=True)

    def forward(self, x):
        # outputsize = x.shape
        # out = F.relu(self.bn1_1(self.conv1_1(x)))
        out = F.relu(self.bn1_2(self.conv1_2(x)))
        out = F.relu(self.bn1_3(self.conv1_3(out)))
        out = F.relu(self.bn1_4(self.conv1_4(out)))

        out = F.relu(self.bn2_1(self.deconv2_1(out)))
        out = F.relu(self.bn2_2(self.deconv2_2(out)))
        x_1 = F.relu(self.bn2_3(self.deconv2_3(out)))

        # x_1 = F.relu(self.bn2_4(self.deconv2_4(out)))
        return x_1


class PixelWeightedFusion(nn.Module):

    def __init__(self, channel):
        super(PixelWeightedFusion, self).__init__()
        self.conv1_1 = nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        return x_1


@MODELS.register_module()
class V2XVoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(
        self,
        voxel_encoder: ConfigType,
        middle_encoder: ConfigType,
        backbone: ConfigType,
        neck: OptConfigType = None,
        bbox_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        mode: str = 'fusion',
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )

        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)

        self.inf_backbone = MODELS.build(backbone)
        if neck is not None:
            self.inf_neck = MODELS.build(neck)
        self.inf_voxel_encoder = MODELS.build(voxel_encoder)
        self.inf_middle_encoder = MODELS.build(middle_encoder)

        # TODO: channel configuration
        self.fusion_weighted = PixelWeightedFusion(384)
        self.encoder = ReduceInfTC(768)

        self.mode = mode

    def generate_matrix(self, theta, x0, y0):
        import numpy as np

        c = theta[0][0]
        s = theta[1][0]
        matrix = np.zeros((3, 3))
        matrix[0, 0] = c
        matrix[0, 1] = -s
        matrix[1, 0] = s
        matrix[1, 1] = c
        matrix[0, 2] = -c * x0 + s * y0 + x0
        matrix[1, 2] = -c * y0 - s * x0 + y0
        matrix[2, 2] = 1
        return matrix

    def extract_feat(self, batch_inputs_dict: Dict[str, Tensor], points_view='vehicle') -> Union[Tuple[torch.Tensor], Dict[str, Tensor]]:
        """Extract features from points."""
        if points_view == 'vehicle':
            voxel_dict = batch_inputs_dict['voxels']
            voxel_features = self.voxel_encoder(voxel_dict['voxels'], voxel_dict['num_points'], voxel_dict['coors'])
            batch_size = voxel_dict['coors'][-1, 0].item() + 1
            veh_x = self.middle_encoder(voxel_features, voxel_dict['coors'], batch_size)
            veh_x = self.backbone(veh_x)
            if self.with_neck:
                veh_x = self.neck(veh_x)
            return veh_x

        elif points_view == 'infrastructure':
            inf_voxel_dict = batch_inputs_dict['infrastructure_voxels']
            inf_voxel_features = self.voxel_encoder(
                inf_voxel_dict['voxels'],
                inf_voxel_dict['num_points'],
                inf_voxel_dict['coors'],
            )
            inf_batch_size = inf_voxel_dict['coors'][-1, 0].item() + 1
            inf_x = self.inf_middle_encoder(inf_voxel_features, inf_voxel_dict['coors'], inf_batch_size)
            inf_x = self.inf_backbone(inf_x)
            if self.with_neck:
                inf_x = self.inf_neck(inf_x)

            inf_x[0] = self.encoder(inf_x[0])
            return inf_x
        else:
            raise Exception('Points View is Error: {}'.format(points_view))

    def feature_fusion(self, veh_x, inf_x, img_metas, mode='fusion'):
        """Method II: Based on affine transformation."""

        if mode not in ['fusion', 'inf_only', 'veh_only']:
            raise Exception('Mode is Error: {}'.format(mode))

        if mode == 'veh_only':
            return veh_x

        wrap_feats_ii = []
        point_cloud_range = [0, -46.08, -3, 92.16, 46.08, 1]

        for ii in range(len(veh_x[0])):
            inf_feature = inf_x[0][ii:ii + 1]
            veh_feature = veh_x[0][ii:ii + 1]

            # Affine transformation module.
            # The affine transformation is implemented with the affine_grid function supported in Pytorch.
            # We ignore the rotation around the x-y plane.
            # theta_rot = [[cos(-theta), sin(-theta), 0.0], [cos(-theta), sin(-theta), 0.0]], theta is in the lidar coordinate.
            # according to the relationship between lidar coordinate system and input coordinate system.
            # First, the coordinates of features and the real world are different in units,
            # so the rotation/translation matrix of the real world needs to be mapped to the feature world.
            # Secondly, there are also some constraints when using the F.affine_grid() function for feature translation and rotation.
            # You can refer to Pytorch's introduction for details
            # range: [-1, 1].
            # Moving right and down is negative.
            # This transformation is employed to convert a feature from the infrastructure-side
            # feature coordinate system to the vehicle-side feature coordinate system.
            # The process encompasses various coordinate systems, including the infrastructure-side
            # feature coordinate system,
            # infrastructure-side LiDAR coordinate system, vehicle-side feature coordinate system, and vehicle-side LiDAR coordinate system.

            # Regarding the version concern you raised, we have been utilizing v0.17.1.
            # We recommend visualizing the transformed feature to assess the relative positional
            # relationship between the two versions.
            # With the help of visualization, you can make necessary adjustments to the transformation as needed.
            # 为简化问题，我们只考虑bev方向的rotation, 暂时忽略z方向的rotation, 取负是考虑Lidar坐标系的y方向与feature坐标系在H维度不一致，所以旋转方向取负。
            # 具体可以画下坐标系的图看看

            calib_inf2veh_rotation = img_metas[ii]['calib']['lidar_i2v']['rotation']
            calib_inf2veh_translation = img_metas[ii]['calib']['lidar_i2v']['translation']

            inf_pointcloud_range = point_cloud_range

            theta_rot = (
                torch.tensor([
                    [
                        calib_inf2veh_rotation[0][0],
                        -calib_inf2veh_rotation[0][1],
                        0.0,
                    ],
                    [
                        -calib_inf2veh_rotation[1][0],
                        calib_inf2veh_rotation[1][1],
                        0.0,
                    ],
                    [0, 0, 1],
                ]).type(dtype=torch.float).cuda(next(self.parameters()).device))
            theta_rot = (torch.FloatTensor(self.generate_matrix(theta_rot, -1, 0)).type(dtype=torch.float).cuda(next(self.parameters()).device))
            # Moving right and down is negative.
            x_trans = (-2 * calib_inf2veh_translation[0][0] / (inf_pointcloud_range[3] - inf_pointcloud_range[0]))
            y_trans = (-2 * calib_inf2veh_translation[1][0] / (inf_pointcloud_range[4] - inf_pointcloud_range[1]))
            theta_trans = (torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans], [0.0, 0.0, 1]]).type(dtype=torch.float).cuda(next(self.parameters()).device))
            theta_r_t = torch.mm(theta_rot, theta_trans, out=None)

            grid_r_t = F.affine_grid(
                theta_r_t[0:2].unsqueeze(0),
                size=torch.Size(veh_feature.shape),
                align_corners=False,
            )
            warp_feat_trans = F.grid_sample(inf_feature, grid_r_t, mode='bilinear', align_corners=False)
            wrap_feats_ii.append(warp_feat_trans)
            save_feature_map('work_dirs/inf_feature_map_1.2.0/inf_feature_map_{}_b.png'.format(ii), inf_feature)
            save_feature_map('work_dirs/inf_feature_map_1.2.0/inf_feature_map_{}_a.png'.format(ii), warp_feat_trans)
            save_feature_map('work_dirs/inf_feature_map_1.2.0/veh_feature_map_{}.png'.format(ii), veh_feature)

        wrap_feats = [torch.cat(wrap_feats_ii, dim=0)]

        if mode == 'inf_only':
            return wrap_feats

        veh_cat_feats = [torch.cat([veh_x[0], wrap_feats[0]], dim=1)]
        veh_cat_feats[0] = self.fusion_weighted(veh_cat_feats[0])

        return veh_cat_feats

    def extract_feats(
        self,
        batch_inputs_dict: Dict[str, Optional[Tensor]],
        batch_data_samples: List[Det3DDataSample],
    ):
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feat_veh = self.extract_feat(batch_inputs_dict, points_view='vehicle')
        feat_inf = self.extract_feat(batch_inputs_dict, points_view='infrastructure')
        feat_fused = self.feature_fusion(feat_veh, feat_inf, batch_input_metas, mode=self.mode)
        return feat_fused

    def _forward_train(
        self,
        batch_inputs_dict: Dict[str, Optional[Tensor]],
        batch_data_samples: List[Det3DDataSample],
    ):
        """Training forward function."""
        feat_fused = self.extract_feats(batch_inputs_dict, batch_data_samples)
        outs = self.bbox_head(feat_fused)
        return outs

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]], batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []

        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(data_sample.get('ignored_instances', None))

        outs = self._forward_train(batch_inputs_dict, batch_data_samples)
        loss_inputs = outs + (
            batch_gt_instances_3d,
            batch_input_metas,
            batch_gt_instances_ignore,
        )
        losses = self.bbox_head.loss_by_feat(*loss_inputs)

        return losses

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]], batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        feat_fused = self.extract_feats(batch_inputs_dict, batch_data_samples)
        bbox_results = self.bbox_head.predict(feat_fused, batch_data_samples)
        res = self.add_pred_to_datasample(batch_data_samples, bbox_results)

        return res
