from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from transvision.models.voxel import Voxelization
from .utils import PixelWeightedFusion, ReduceInfTC


@MODELS.register_module()
class CoFormerNet(Base3DDetector):

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

        if self.mode == 'fusion':
            self.inf_pts_voxel_layer = Voxelization(**voxelize_cfg)
            self.inf_pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
            self.inf_pts_middle_encoder = MODELS.build(pts_middle_encoder)
            self.inf_pts_backbone = MODELS.build(pts_backbone)
            self.inf_pts_neck = MODELS.build(pts_neck)

            self.fusion_weighted = PixelWeightedFusion(512)
            self.encoder = ReduceInfTC(1024)

        self.init_weights()

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

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post- processing.
        """
        pass

    def parse_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

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
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

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
        elif points_view == 'infrastructure':
            points = batch_inputs_dict['infrastructure_points']
            with torch.autocast('cuda', enabled=False):
                points = [point.float() for point in points]
                feats, coords, sizes = self.inf_voxelize(points)
                batch_size = coords[-1, 0] + 1
            x = self.pts_middle_encoder(feats, coords, batch_size)
            return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    @torch.no_grad()
    def inf_voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            res = res.contiguous()
            ret = self.inf_pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]], batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(imgs, deepcopy(points), lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix, batch_input_metas)
            features.append(img_feature)
        pts_feature = self.extract_pts_feat(batch_inputs_dict, points_view='vehicle')
        features.append(pts_feature)
        if self.fusion_layer is not None:
            veh_x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            veh_x = features[0]

        veh_x = self.pts_backbone(veh_x)
        veh_x = self.pts_neck(veh_x)
        if self.mode == 'veh_only':
            return veh_x

        inf_pts_feature = self.extract_pts_feat(batch_inputs_dict, points_view='infrastructure')
        inf_x = self.inf_pts_backbone(inf_pts_feature)
        inf_x = self.inf_pts_neck(inf_x)

        inf_x[0] = self.encoder(inf_x[0])

        wrap_feats_ii = []
        for ii in range(len(veh_x[0])):
            veh_feature = veh_x[0][ii:ii + 1]
            inf_feature = inf_x[0][ii:ii + 1]

            calib_inf2veh_rotation = batch_input_metas[ii]['calib']['lidar_i2v']['rotation']
            calib_inf2veh_translation = batch_input_metas[ii]['calib']['lidar_i2v']['translation']

            inf_pointcloud_range = [0, -46.08, -3, 92.16, 46.08, 1]

            theta_rot = (
                torch.tensor([[calib_inf2veh_rotation[0][0], -calib_inf2veh_rotation[0][1], 0.0], [-calib_inf2veh_rotation[1][0], calib_inf2veh_rotation[1][1], 0.0],
                              [0, 0, 1]]).type(dtype=torch.float).cuda(next(self.parameters()).device))
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

        inf_wrap_feats = [torch.cat(wrap_feats_ii, dim=0)]
        veh_cat_feats = [torch.cat([veh_x[0], inf_wrap_feats[0]], dim=1)]
        veh_cat_feats[0] = self.fusion_weighted(veh_cat_feats[0])

        return veh_cat_feats

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]], batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)

        return losses
