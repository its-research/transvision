# Copyright (c) DAIR-V2X (AIR). All rights reserved.
import copy
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
# from mmdet3d.structures.det3d_data_sample import OptSampleList
# from mmcv.runner import force_fp32
from torch import Tensor, nn
from torch.nn import functional as F

from transvision.models.voxel import Voxelization
from .utils import AttentionMask, PixelWeightedFusion, ReduceInfTC


class FlowGenerator(nn.Module):

    def __init__(self, voxel_layer, voxel_encoder, middle_encoder, backbone, with_neck=False, neck=None):
        super(FlowGenerator, self).__init__()
        backbone_flow = copy.deepcopy(backbone)
        backbone_flow['in_channels'] = backbone['in_channels'] * 2
        self.inf_backbone = MODELS.build(backbone_flow)
        self.inf_with_neck = with_neck
        if neck is not None:
            self.inf_neck = MODELS.build(neck)
        self.inf_voxel_layer = Voxelization(**voxel_layer)
        self.inf_voxel_encoder = MODELS.build(voxel_encoder)
        self.inf_middle_encoder = MODELS.build(middle_encoder)
        self.pre_encoder = ReduceInfTC(768)
        self.with_attention_mask = False

    def forward(self, points_t_0, points_t_1):
        voxels_t_0, num_points_t_0, coors_t_0 = self.inf_voxelize(points_t_0)
        voxel_features_t_0 = self.inf_voxel_encoder(voxels_t_0, num_points_t_0, coors_t_0)
        batch_size_t_0 = coors_t_0[-1, 0].item() + 1
        feat_t_0 = self.inf_middle_encoder(voxel_features_t_0, coors_t_0, batch_size_t_0)

        voxels_t_1, num_points_t_1, coors_t_1 = self.inf_voxelize(points_t_1)
        voxel_features_t_1 = self.inf_voxel_encoder(voxels_t_1, num_points_t_1, coors_t_1)
        batch_size_t_1 = coors_t_1[-1, 0].item() + 1
        feat_t_1 = self.inf_middle_encoder(voxel_features_t_1, coors_t_1, batch_size_t_1)

        flow_pred = torch.cat([feat_t_0, feat_t_1], dim=1)
        flow_pred = self.inf_backbone(flow_pred)
        if self.inf_with_neck:
            flow_pred = self.inf_neck(flow_pred)

        if self.with_attention_mask:
            attention_mask = AttentionMask(feat_t_0, feat_t_1)
            flow_pred[0] = self.pre_encoder(flow_pred[0], attention_mask)
        else:
            flow_pred[0] = self.pre_encoder(flow_pred[0])
        return flow_pred

    @torch.no_grad()
    def inf_voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res = res.contiguous()
            res_voxels, res_coors, res_num_points = self.inf_voxel_layer(res)
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
        return voxels, num_points, coors_batch


@MODELS.register_module()
class FeatureFlowNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 data_preprocessor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(FeatureFlowNet, self).__init__(
            backbone=backbone, neck=neck, bbox_head=bbox_head, train_cfg=train_cfg, test_cfg=test_cfg, data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)

        self.inf_voxel_layer = Voxelization(**voxel_layer)
        self.inf_voxel_encoder = MODELS.build(voxel_encoder)
        self.inf_middle_encoder = MODELS.build(middle_encoder)
        self.inf_backbone = MODELS.build(backbone)
        if neck is not None:
            self.inf_neck = MODELS.build(neck)

        # TODO: channel configuration
        self.fusion_weighted = PixelWeightedFusion(384)
        self.fusion_training = False
        self.flow_training = True
        self.mse_loss = nn.MSELoss()
        self.flownet = FlowGenerator(voxel_layer, voxel_encoder, middle_encoder, backbone, with_neck=self.with_neck, neck=neck)
        self.encoder = ReduceInfTC(768)

        try:
            self.data_root = 'data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/ffnet'
            self.pretraind_checkpoint_path = train_cfg['pretrained_model']
        except:
            pass
            self.data_root = 'data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/ffnet'
            self.pretraind_checkpoint_path = test_cfg['pretrained_model']
        self.flownet_pretrained = False
        if 'test_mode' in test_cfg.keys():
            self.test_mode = test_cfg['test_mode']
        else:
            self.test_mode = 'FlowPred'
        self.init_weights()
        if self.pretraind_checkpoint_path != '':
            self.flownet_init()

    def extract_feat(self, points: Dict[str, Tensor], points_view='vehicle'):
        """Extract features from points."""
        if points_view == 'vehicle':
            voxels, num_points, coors = self.voxelize(points)
            voxel_features = self.voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0].item() + 1
            veh_x = self.middle_encoder(voxel_features, coors, batch_size)
            veh_x = self.backbone(veh_x)
            if self.with_neck:
                veh_x = self.neck(veh_x)
            return veh_x

        elif 'infrastructure' in points_view:
            inf_voxels, inf_num_points, inf_coors = self.inf_voxelize(points)
            inf_voxel_features = self.inf_voxel_encoder(inf_voxels, inf_num_points, inf_coors)
            inf_batch_size = inf_coors[-1, 0].item() + 1
            inf_x = self.inf_middle_encoder(inf_voxel_features, inf_coors, inf_batch_size)
            inf_x = self.inf_backbone(inf_x)
            if self.with_neck:
                inf_x = self.inf_neck(inf_x)

            inf_x[0] = self.encoder(inf_x[0])
            return inf_x
        else:
            raise Exception('Points View is Error: {}'.format(points_view))

    def generate_matrix(self, theta, x0, y0):
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

    def feature_fusion(self, veh_x, inf_x, img_metas, mode='fusion'):
        wrap_feats_ii = []
        point_cloud_range = [0, -46.08, -3, 92.16, 46.08, 1]
        for ii in range(len(veh_x[0])):
            inf_feature = inf_x[0][ii:ii + 1]
            veh_feature = veh_x[0][ii:ii + 1]

            calib_inf2veh_rotation = img_metas[ii]['calib']['lidar_i2v']['rotation']
            calib_inf2veh_translation = img_metas[ii]['calib']['lidar_i2v']['translation']

            inf_pointcloud_range = point_cloud_range

            theta_rot = (
                torch.tensor([[calib_inf2veh_rotation[0][0], -calib_inf2veh_rotation[0][1], 0.0], [-calib_inf2veh_rotation[1][0], calib_inf2veh_rotation[1][1], 0.0],
                              [0, 0, 1]]).type(dtype=torch.float).cuda(next(self.parameters()).device))
            theta_rot = torch.FloatTensor(self.generate_matrix(theta_rot, -1, 0)).type(dtype=torch.float).cuda(next(self.parameters()).device)

            x_trans = -2 * calib_inf2veh_translation[0][0] / (inf_pointcloud_range[3] - inf_pointcloud_range[0])
            y_trans = -2 * calib_inf2veh_translation[1][0] / (inf_pointcloud_range[4] - inf_pointcloud_range[1])
            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans], [0.0, 0.0, 1]]).type(dtype=torch.float).cuda(next(self.parameters()).device)
            theta_r_t = torch.mm(theta_rot, theta_trans, out=None)

            grid_r_t = F.affine_grid(theta_r_t[0:2].unsqueeze(0), size=torch.Size(veh_feature.shape), align_corners=False)
            warp_feat_trans = F.grid_sample(inf_feature, grid_r_t, mode='bilinear', align_corners=False)
            wrap_feats_ii.append(warp_feat_trans)

        wrap_feats = [torch.cat(wrap_feats_ii, dim=0)]

        if mode not in ['fusion', 'inf_only', 'veh_only']:
            raise Exception('Mode is Error: {}'.format(mode))
        if mode == 'inf_only':
            return wrap_feats
        elif mode == 'veh_only':
            return veh_x

        veh_cat_feats = [torch.cat([veh_x[0], wrap_feats[0]], dim=1)]
        veh_cat_feats[0] = self.fusion_weighted(veh_cat_feats[0])

        return veh_cat_feats

    def flownet_init(self):
        pretraind_checkpoint_path = self.pretraind_checkpoint_path
        flownet_pretrained = self.flownet_pretrained
        pretraind_checkpoint = torch.load(pretraind_checkpoint_path, map_location='cpu')['state_dict']
        pretraind_checkpoint_modify = {}

        checkpoint_source = 'single_infrastructure_side'
        for k, v in pretraind_checkpoint.items():
            if 'inf_' in k:
                checkpoint_source = 'v2x_voxelnet'
                break

        if checkpoint_source == 'single_infrastructure_side':
            for k, v in pretraind_checkpoint.items():
                pretraind_checkpoint_modify['inf_' + k] = v
                if flownet_pretrained:
                    pretraind_checkpoint_modify['flownet.inf_' + k] = v
        elif checkpoint_source == 'v2x_voxelnet':
            for k, v in pretraind_checkpoint.items():
                if 'inf_' in k and flownet_pretrained:
                    pretraind_checkpoint_modify['flownet.' + k] = v
                pretraind_checkpoint_modify[k] = v

        self.load_state_dict(pretraind_checkpoint_modify, strict=False)

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]], batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        losses = self._forward_train(batch_inputs_dict, batch_data_samples)
        return losses

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
        if self.fusion_training:
            feat_fused = self.extract_feats(batch_inputs_dict, batch_data_samples)
            outs = self.bbox_head(feat_fused)
            return outs
        img_metas = [item.metainfo for item in batch_data_samples]

        if self.flow_training:
            points_t_0_1 = []
            points_t_1_2 = []
            inf_points_t_0 = []
            inf_points_t_1 = []
            inf_points_t_2 = []

            for ii in range(len(batch_inputs_dict['points'])):
                inf_points_path_t_0 = os.path.join(img_metas[ii]['v2x_info']['infrastructure_pointcloud_bin_path_t_0'])
                tem_inf_points = torch.from_numpy(np.fromfile(inf_points_path_t_0, dtype=np.float32))
                tem_inf_points = torch.reshape(tem_inf_points, (-1, 4)).cuda(device=batch_inputs_dict['points'][0].device)
                inf_points_t_0.append(tem_inf_points)

                inf_points_path_t_1 = os.path.join(img_metas[ii]['v2x_info']['infrastructure_pointcloud_bin_path_t_1'])
                tem_inf_points = torch.from_numpy(np.fromfile(inf_points_path_t_1, dtype=np.float32))
                tem_inf_points = torch.reshape(tem_inf_points, (-1, 4)).cuda(device=batch_inputs_dict['points'][0].device)
                inf_points_t_1.append(tem_inf_points)

                inf_points_path_t_2 = os.path.join(img_metas[ii]['v2x_info']['infrastructure_pointcloud_bin_path_t_2'])
                tem_inf_points = torch.from_numpy(np.fromfile(inf_points_path_t_2, dtype=np.float32))
                tem_inf_points = torch.reshape(tem_inf_points, (-1, 4)).cuda(device=batch_inputs_dict['points'][0].device)
                inf_points_t_2.append(tem_inf_points)

                points_t_0_1.append(img_metas[ii]['v2x_info']['infrastructure_t_0_1'])
                points_t_1_2.append(img_metas[ii]['v2x_info']['infrastructure_t_1_2'])

            for ii in range(len(inf_points_t_0)):
                inf_points_t_0[ii][:, 3] = 255 * inf_points_t_0[ii][:, 3]
                inf_points_t_1[ii][:, 3] = 255 * inf_points_t_1[ii][:, 3]
                inf_points_t_2[ii][:, 3] = 255 * inf_points_t_2[ii][:, 3]

            feat_inf_t_1 = self.extract_feat(inf_points_t_1, points_view='infrastructure')
            feat_inf_t_2 = self.extract_feat(inf_points_t_2, points_view='infrastructure')
            for ii in range(len(feat_inf_t_1)):
                feat_inf_t_1[ii] = feat_inf_t_1[ii].detach()
                feat_inf_t_2[ii] = feat_inf_t_2[ii].detach()

            flow_pred = self.flownet(inf_points_t_0, inf_points_t_1)
            feat_inf_apprs = []
            for ii in range(len(flow_pred)):
                for bs in range(len(batch_inputs_dict['points'])):
                    # tem_feat_inf_t_1_before_max = feat_inf_t_1[ii][bs].mean()
                    tem_feat_inf_t_1_before_max = feat_inf_t_1[ii][bs].mean().detach()
                    feat_inf_t_1[ii][bs] = feat_inf_t_1[ii][bs] + flow_pred[ii][bs] / points_t_0_1[bs] * points_t_1_2[bs]
                    tem_feat_inf_t_1_after_max = feat_inf_t_1[ii][bs].mean().detach()
                    feat_inf_t_1[ii][bs] = feat_inf_t_1[ii][bs] / tem_feat_inf_t_1_after_max * tem_feat_inf_t_1_before_max
                feat_inf_apprs.append(feat_inf_t_1[ii])

            similarity = torch.cosine_similarity(torch.flatten(feat_inf_t_2[0], start_dim=1, end_dim=3), torch.flatten(feat_inf_apprs[0], start_dim=1, end_dim=3), dim=1)
            # print("The similarity is: ", similarity, points_t_1_2)

            label = torch.ones(len(batch_inputs_dict['points']), requires_grad=False).cuda(device=batch_inputs_dict['points'][0].device)
            if not self.fusion_training:
                losses = {}
            losses['similarity_loss'] = self.mse_loss(similarity, label)

        return losses

    def simple_test(self, batch_inputs_dict: Dict[str, Optional[Tensor]], batch_data_samples: List[Det3DDataSample]):

        feat_veh = self.extract_feat(batch_inputs_dict['points'], points_view='vehicle')
        # feat_inf = self.extract_feat(batch_inputs_dict, points_view='infrastructure')

        if self.test_mode not in ['FlowPred', 'OriginFeat', 'Async']:
            raise Exception('FlowNet Test Mode is Error: {}'.format(self.test_mode))

        if self.test_mode == 'OriginFeat':
            feat_inf = self.extract_feat(batch_inputs_dict, points_view='infrastructure')

        if self.test_mode == 'Async':

            feat_inf_t_1 = self.extract_feat(batch_inputs_dict, points_view='infrastructure_t1')
            feat_inf = feat_inf_t_1

        img_metas = [item.metainfo for item in batch_data_samples]

        if self.test_mode == 'FlowPred':
            points_t_0_1 = []
            points_t_1_2 = []
            inf_points_t_0 = []
            inf_points_t_1 = []
            inf_points_t_2 = []

            for ii in range(len(batch_inputs_dict['points'])):
                inf_points_path_t_0 = os.path.join(img_metas[ii]['v2x_info']['infrastructure_pointcloud_bin_path_t_0'])
                tem_inf_points = torch.from_numpy(np.fromfile(inf_points_path_t_0, dtype=np.float32))
                tem_inf_points = torch.reshape(tem_inf_points, (-1, 4)).cuda(device=batch_inputs_dict['points'][0].device)
                inf_points_t_0.append(tem_inf_points)

                inf_points_path_t_1 = os.path.join(img_metas[ii]['v2x_info']['infrastructure_pointcloud_bin_path_t_1'])
                tem_inf_points = torch.from_numpy(np.fromfile(inf_points_path_t_1, dtype=np.float32))
                tem_inf_points = torch.reshape(tem_inf_points, (-1, 4)).cuda(device=batch_inputs_dict['points'][0].device)
                inf_points_t_1.append(tem_inf_points)

                inf_points_path_t_2 = os.path.join(img_metas[ii]['v2x_info']['infrastructure_pointcloud_bin_path_t_2'])
                tem_inf_points = torch.from_numpy(np.fromfile(inf_points_path_t_2, dtype=np.float32))
                tem_inf_points = torch.reshape(tem_inf_points, (-1, 4)).cuda(device=batch_inputs_dict['points'][0].device)
                inf_points_t_2.append(tem_inf_points)

                points_t_0_1.append(img_metas[ii]['v2x_info']['infrastructure_t_0_1'])
                points_t_1_2.append(img_metas[ii]['v2x_info']['infrastructure_t_1_2'])

            for ii in range(len(inf_points_t_0)):
                inf_points_t_0[ii][:, 3] = 255 * inf_points_t_0[ii][:, 3]
                inf_points_t_1[ii][:, 3] = 255 * inf_points_t_1[ii][:, 3]
                inf_points_t_2[ii][:, 3] = 255 * inf_points_t_2[ii][:, 3]

            feat_inf_t_1 = self.extract_feat(inf_points_t_1, points_view='infrastructure')
            # feat_inf_t_2 = self.extract_feat(inf_points_t_2, img_metas, points_view='infrastructure')
            feat_inf_temp = self.extract_feat(inf_points_t_2, points_view='infrastructure')
            flow_pred = self.flownet(inf_points_t_0, inf_points_t_1)
            feat_inf_apprs = []
            for ii in range(len(flow_pred)):
                for bs in range(len(batch_inputs_dict['points'])):
                    tem_feat_inf_t_1_before_max = feat_inf_t_1[ii][bs].mean()
                    feat_inf_temp[ii][bs] = feat_inf_t_1[ii][bs] + flow_pred[ii][bs] / points_t_0_1[bs] * points_t_1_2[bs]
                    tem_feat_inf_t_1_after_max = feat_inf_temp[ii][bs].mean().detach()
                    feat_inf_temp[ii][bs] = feat_inf_temp[ii][bs] / tem_feat_inf_t_1_after_max * tem_feat_inf_t_1_before_max
                feat_inf_apprs.append(feat_inf_temp[ii])

            # similarity = torch.cosine_similarity(torch.flatten(feat_inf_t_2[0], start_dim=1, end_dim=3),
            #                                      torch.flatten(feat_inf_apprs[0], start_dim=1, end_dim=3), dim=1)

            feat_inf = feat_inf_apprs

        feat_fused = self.feature_fusion(feat_veh, feat_inf, img_metas)
        bbox_list = self.bbox_head.predict(feat_fused, batch_data_samples)

        return bbox_list

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]], batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        bbox_results = self.simple_test(batch_inputs_dict, batch_data_samples)
        res = self.add_pred_to_datasample(batch_data_samples, bbox_results)

        return res

    def aug_test(self):
        """Test function with augmentaiton."""
        return None

    @torch.no_grad()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res = res.contiguous()
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
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
        return voxels, num_points, coors_batch

    @torch.no_grad()
    def inf_voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res = res.contiguous()
            res_voxels, res_coors, res_num_points = self.inf_voxel_layer(res)
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
        return voxels, num_points, coors_batch
