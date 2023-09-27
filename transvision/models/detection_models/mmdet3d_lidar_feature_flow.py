import logging
import os
import sys
from copy import deepcopy
from os import path as osp

import numpy as np
import torch
from mmdet3d.structures import get_box_type
from mmengine.dataset import Compose, pseudo_collate

from transvision.models.base_model import BaseModel
from transvision.models.model_utils import init_model
from transvision.v2x_utils import get_arrow_end, mkdir

logger = logging.getLogger(__name__)


def get_box_info(result):
    for i in range(len(result[0].pred_instances_3d.bboxes_3d)):
        temp = result[0].pred_instances_3d.bboxes_3d.tensor[i][4].clone()
        result[0].pred_instances_3d.bboxes_3d.tensor[i][4] = result[0].pred_instances_3d.bboxes_3d.tensor[i][3]
        result[0].pred_instances_3d.bboxes_3d.tensor[i][3] = temp
        result[0].pred_instances_3d.bboxes_3d.tensor[i][6] = result[0].pred_instances_3d.bboxes_3d.tensor[i][6]
    if len(result[0].pred_instances_3d.bboxes_3d.tensor) == 0:
        box_lidar = np.zeros((1, 8, 3))
        box_ry = np.zeros(1)
    else:
        box_lidar = result[0].pred_instances_3d.bboxes_3d.corners.numpy()
        box_ry = result[0].pred_instances_3d.bboxes_3d.tensor[:, -1].numpy()
    box_centers_lidar = box_lidar.mean(axis=1)
    arrow_ends_lidar = get_arrow_end(box_centers_lidar, box_ry)
    return box_lidar, box_ry, box_centers_lidar, arrow_ends_lidar


def gen_pred_dict(id, timestamp, box, arrow, points, score, label):
    if len(label) == 0:
        score = [-2333]
        label = [-1]
    save_dict = {
        'info': id,
        'timestamp': timestamp,
        'boxes_3d': box.tolist(),
        'arrows': arrow.tolist(),
        'scores_3d': score,
        'labels_3d': label,
        'points': points.tolist(),
    }
    return save_dict


def inference_detector_feature_fusion(model, veh_bin, inf_bin, rotation, translation, vic_frame):
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

    data_ = dict(
        vehicle_pts_filename=veh_bin,
        infrastructure_pts_filename=inf_bin,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        ann_info=dict(axis_align_matrix=np.eye(4)),
        sweeps=[],
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[],
    )

    data_ = test_pipeline(data_)

    data = []
    data.append(data_)
    collate_data = pseudo_collate(data)

    collate_data['data_samples'][0].set_metainfo(dict(inf2veh=dict(rotation=rotation, translation=translation)))
    collate_data['data_samples'][0].set_metainfo(dict(infrastructure_idx_t_1=vic_frame['infrastructure_idx_t_1']))
    collate_data['data_samples'][0].set_metainfo(dict(infrastructure_pointcloud_bin_path_t_1=vic_frame['infrastructure_pointcloud_bin_path_t_1']))
    collate_data['data_samples'][0].set_metainfo(dict(infrastructure_idx_t_0=vic_frame['infrastructure_idx_t_0']))
    collate_data['data_samples'][0].set_metainfo(dict(infrastructure_pointcloud_bin_path_t_0=vic_frame['infrastructure_pointcloud_bin_path_t_0']))
    collate_data['data_samples'][0].set_metainfo(dict(infrastructure_t_0_1=vic_frame['infrastructure_t_0_1']))
    collate_data['data_samples'][0].set_metainfo(dict(infrastructure_idx_t_2=vic_frame['infrastructure_idx_t_2']))
    collate_data['data_samples'][0].set_metainfo(dict(infrastructure_pointcloud_bin_path_t_2=vic_frame['infrastructure_pointcloud_bin_path_t_2']))
    collate_data['data_samples'][0].set_metainfo(dict(infrastructure_t_1_2=vic_frame['infrastructure_t_1_2']))

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
        mkdir(args.output)
        mkdir(osp.join(args.output, 'inf'))
        mkdir(osp.join(args.output, 'veh'))
        mkdir(osp.join(args.output, 'inf', 'lidar'))
        mkdir(osp.join(args.output, 'veh', 'lidar'))
        mkdir(osp.join(args.output, 'inf', 'camera'))
        mkdir(osp.join(args.output, 'veh', 'camera'))
        mkdir(osp.join(args.output, 'result'))

    def forward(self, vic_frame, filt, prev_inf_frame_func=None, *args):
        tmp_veh = vic_frame.veh_frame.point_cloud(data_format='file')
        tmp_inf = vic_frame.inf_frame.point_cloud(data_format='file')

        trans = vic_frame.transform('Infrastructure_lidar', 'Vehicle_lidar')
        rotation, translation = trans.get_rot_trans()
        result, _ = inference_detector_feature_fusion(self.model, tmp_veh, tmp_inf, rotation, translation, vic_frame)
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
            result[0].pred_instances_3d.scores_3d = result[0].pred_instances_3d.scores_3d.numpy()[remain]
            result[0].pred_instances_3d.labels_3d = result[0].pred_instances_3d.labels_3d.numpy()[remain]
        else:
            box = np.zeros((1, 8, 3))
            box_center = np.zeros((1, 1, 3))
            arrow_ends = np.zeros((1, 1, 3))
            result[0].pred_instances_3d.labels_3d = np.zeros((1))
            result[0].pred_instances_3d.scores_3d = np.zeros((1))
        # Save results
        pred = gen_pred_dict(
            id,
            [],
            box,
            np.concatenate([box_center, arrow_ends], axis=1),
            np.array(1),
            result[0].pred_instances_3d.scores_3d.tolist(),
            result[0].pred_instances_3d.labels_3d.tolist(),
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


if __name__ == '__main__':
    sys.path.append('..')
    sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk('../') for name in dirs])
