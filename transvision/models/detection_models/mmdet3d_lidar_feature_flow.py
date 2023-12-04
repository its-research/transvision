import logging
import os
from copy import deepcopy
from os import path as osp

import numpy as np
import torch
from mmdet3d.structures import CameraInstance3DBoxes, get_box_type
from mmengine.dataset import Compose, pseudo_collate
from mmengine.fileio import load

from transvision.models.base_model import BaseModel
from transvision.models.detection_models.utils import gen_pred_dict, get_box_info
from transvision.models.model_utils import init_model
from transvision.v2x_utils import mkdir

logger = logging.getLogger(__name__)


def inference_detector_feature_fusion(model, data_list, veh_bin, inf_bin, rotation, translation, vic_frame):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.
    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    # device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.test_dataloader.dataset.box_type_3d)
    v2x_info = {}
    data_root = 'data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.3.0_training/ffnet'

    v2x_info['infrastructure_idx_t_1'] = vic_frame['infrastructure_idx_t_1']
    v2x_info['infrastructure_pointcloud_bin_path_t_1'] = os.path.join(data_root, vic_frame['infrastructure_pointcloud_bin_path_t_1'])
    v2x_info['infrastructure_idx_t_0'] = vic_frame['infrastructure_idx_t_0']
    v2x_info['infrastructure_pointcloud_bin_path_t_0'] = os.path.join(data_root, vic_frame['infrastructure_pointcloud_bin_path_t_0'])
    v2x_info['infrastructure_t_0_1'] = vic_frame['infrastructure_t_0_1']
    v2x_info['infrastructure_idx_t_2'] = vic_frame['infrastructure_idx_t_2']
    v2x_info['infrastructure_pointcloud_bin_path_t_2'] = os.path.join(data_root, vic_frame['infrastructure_pointcloud_bin_path_t_2'])
    v2x_info['infrastructure_t_1_2'] = vic_frame['infrastructure_t_1_2']

    instances = data_list['instances']
    gt_bboxes_3d = []
    gt_labels_3d = []
    gt_bboxes_3d = np.array([item['bbox_3d'] for item in instances]).astype(np.float32)
    gt_labels_3d = np.array([item['bbox_label_3d'] for item in instances]).astype(np.int64)

    lidar2cam = np.array(data_list['images']['CAM2']['lidar2cam'])
    gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(box_mode_3d, np.linalg.inv(lidar2cam))

    data_ = dict(
        lidar_points=dict(lidar_path=veh_bin, inf_lidar_path=inf_bin),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        calib=dict(lidar_i2v=dict(rotation=rotation, translation=translation)),
        v2x_info=v2x_info,
        ann_info=dict(gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d),
    )

    data_ = test_pipeline(data_)

    data = []
    data.append(data_)
    collate_data = pseudo_collate(data)

    with torch.no_grad():
        result = model.test_step(collate_data)

    return result, data


class FeatureFlow(BaseModel):

    def add_arguments(parser):
        parser.add_argument('--inf-config-path', type=str, default='')
        parser.add_argument('--inf-model-path', type=str, default='')
        parser.add_argument('--veh-config-path', type=str, default='')
        parser.add_argument('--veh-model-path', type=str, default='')
        parser.add_argument('--no-comp', action='store_true')
        parser.add_argument('--overwrite-cache', action='store_true')

    def __init__(self, args, pipe):
        super().__init__()
        # self.model = LateFusionVeh(args)
        if osp.exists(args.output):
            import shutil

            shutil.rmtree(args.output)
        self.args = args
        self.pipe = pipe
        self.model = init_model(
            self.args.veh_config_path,
            self.args.veh_model_path,
            device=self.args.device,
        )
        # self.model.flownet_init()
        ann_root_path = self.model.cfg.test_dataloader.dataset.data_root
        ann_file = self.model.cfg.test_dataloader.dataset.ann_file
        self.data_list = load(os.path.join(ann_root_path, ann_file))['data_list']

        mkdir(args.output)
        mkdir(osp.join(args.output, 'inf'))
        mkdir(osp.join(args.output, 'veh'))
        mkdir(osp.join(args.output, 'inf', 'lidar'))
        mkdir(osp.join(args.output, 'veh', 'lidar'))
        mkdir(osp.join(args.output, 'inf', 'camera'))
        mkdir(osp.join(args.output, 'veh', 'camera'))
        mkdir(osp.join(args.output, 'result'))

    def forward(self, vic_frame, filt, idx, prev_inf_frame_func=None, *args):
        tmp_veh = vic_frame.veh_frame.point_cloud(data_format='file')
        tmp_inf = vic_frame.inf_frame.point_cloud(data_format='file')

        trans = vic_frame.transform('Infrastructure_lidar', 'Vehicle_lidar')
        rotation, translation = trans.get_rot_trans()

        result, _ = inference_detector_feature_fusion(self.model, self.data_list[idx], tmp_veh, tmp_inf, rotation, translation, vic_frame)
        box, box_ry, box_center, arrow_ends = get_box_info(result)

        remain = []
        if len(result[0].pred_instances_3d.bboxes_3d.tensor) != 0:
            for i in range(box.shape[0]):
                if filt(box[i]):
                    remain.append(i)
        if len(remain) >= 1:
            box = box[remain]
            box_center = box_center[remain]
            arrow_ends = arrow_ends[remain]
            scores_3d = result[0].pred_instances_3d.scores_3d.cpu().numpy()[remain]
            labels_3d = result[0].pred_instances_3d.labels_3d.cpu().numpy()[remain]
        else:
            box = np.zeros((1, 8, 3))
            box_center = np.zeros((1, 1, 3))
            arrow_ends = np.zeros((1, 1, 3))
            scores_3d = np.zeros((1))
            labels_3d = np.zeros((1))
        # Save results
        pred = gen_pred_dict(
            id,
            [],
            box,
            np.concatenate([box_center, arrow_ends], axis=1),
            np.array(1),
            scores_3d.tolist(),
            labels_3d.tolist(),
        )

        for ii in range(len(pred['labels_3d'])):
            pred['labels_3d'][ii] = 2
        self.pipe.send('boxes_3d', pred['boxes_3d'])
        self.pipe.send('labels_3d', pred['labels_3d'])
        self.pipe.send('scores_3d', pred['scores_3d'])

        return {
            'boxes_3d': np.array(pred['boxes_3d']),
            'labels_3d': np.array(pred['labels_3d']),
            'scores_3d': np.array(pred['scores_3d']),
        }
