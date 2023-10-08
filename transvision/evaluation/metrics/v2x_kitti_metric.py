# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import mmengine
import numpy as np
import torch
from mmdet3d.evaluation import KittiMetric
from mmdet3d.registry import METRICS
from mmdet3d.structures import Box3DMode, CameraInstance3DBoxes, LiDARInstance3DBoxes, points_cam2img


@METRICS.register_module()
class V2XKittiMetric(KittiMetric):

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM2',
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'V2X Kitti metric'
        super(V2XKittiMetric, self).__init__(
            ann_file=ann_file,
            metric=metric,
            pcd_limit_range=pcd_limit_range,
            prefix=prefix,
            pklfile_prefix=pklfile_prefix,
            default_cam_key=default_cam_key,
            format_only=format_only,
            submission_prefix=submission_prefix,
            collect_device=collect_device,
            backend_args=backend_args)

    def convert_annos_to_kitti_annos(self, data_infos: dict) -> List[dict]:

        data_annos = data_infos['data_list']

        for _, annos in enumerate(data_annos):
            for name in annos['kitti_annos']:
                annos['kitti_annos'][name] = np.array(annos['kitti_annos'][name])

        return data_annos

    def bbox2result_kitti(self,
                          net_outputs: List[dict],
                          sample_idx_list: List[int],
                          class_names: List[str],
                          pklfile_prefix: Optional[str] = None,
                          submission_prefix: Optional[str] = None) -> List[dict]:
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (List[dict]): List of dict storing the inferenced
                bounding boxes and scores.
            sample_idx_list (List[int]): List of input sample idx.
            class_names (List[str]): A list of class names.
            pklfile_prefix (str, optional): The prefix of pkl file.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submission file.
                Defaults to None.

        Returns:
            List[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmengine.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting 3D prediction to KITTI format')
        for idx, pred_dicts in enumerate(mmengine.track_iter_progress(net_outputs)):
            sample_idx = sample_idx_list[idx]
            info = self.data_infos[sample_idx]
            # Here default used 'CAM2' to compute metric. If you want to
            # use another camera, please modify it.
            image_shape = (info['image']['image_shape'][0], info['image']['image_shape'][1])
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'dimensions': [], 'location': [], 'rotation_y': [], 'score': []}
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']
                pred_box_type_3d = box_dict['pred_box_type_3d']

                for box, box_lidar, bbox, score, label in zip(box_preds, box_preds_lidar, box_2d_preds, scores, label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    if pred_box_type_3d == CameraInstance3DBoxes:
                        anno['alpha'].append(-np.arctan2(box[0], box[2]) + box[6])
                    elif pred_box_type_3d == LiDARInstance3DBoxes:
                        anno['alpha'].append(-np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(anno['name'][idx], anno['alpha'][idx], bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                                                               dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0], loc[idx][1], loc[idx][2],
                                                                               anno['rotation_y'][idx], anno['score'][idx]),
                            file=f)

            anno['sample_idx'] = np.array([sample_idx] * len(anno['score']), dtype=np.int64)

            det_annos.append(anno)

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            else:
                out = pklfile_prefix
            mmengine.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def convert_valid_bboxes(self, box_dict: dict, info: dict) -> dict:
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - bboxes_3d (:obj:`BaseInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (Tensor): Scores of boxes.
                - labels_3d (Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

            - bbox (np.ndarray): 2D bounding boxes.
            - box3d_camera (np.ndarray): 3D bounding boxes in
              camera coordinate.
            - box3d_lidar (np.ndarray): 3D bounding boxes in
              LiDAR coordinate.
            - scores (np.ndarray): Scores of boxes.
            - label_preds (np.ndarray): Class label predictions.
            - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['bboxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['sample_idx']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]), box3d_camera=np.zeros([0, 7]), box3d_lidar=np.zeros([0, 7]), scores=np.zeros([0]), label_preds=np.zeros([0, 4]), sample_idx=sample_idx)
        # Here default used 'CAM2' to compute metric. If you want to
        # use another camera, please modify it.
        # lidar2cam = np.array(info['images'][self.default_cam_key]['lidar2cam']).astype(np.float32)
        P2 = np.array(info['calib']['P2']).astype(np.float32)
        img_shape = (info['image']['image_shape'][0], info['image']['image_shape'][1])
        P2 = box_preds.tensor.new_tensor(P2)

        rect = np.array(info['calib']['R0_rect']).astype(np.float32)
        Trv2c = np.array(info['calib']['Tr_velo_to_cam']).astype(np.float32)
        lidar2cam = rect @ Trv2c

        # box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)

        if isinstance(box_preds, LiDARInstance3DBoxes):
            box_preds_camera = box_preds.convert_to(Box3DMode.CAM, lidar2cam)
            box_preds_lidar = box_preds
        elif isinstance(box_preds, CameraInstance3DBoxes):
            box_preds_camera = box_preds
            box_preds_lidar = box_preds.convert_to(Box3DMode.LIDAR, np.linalg.inv(lidar2cam))

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds_camera
        image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) & (box_2d_preds[:, 1] < image_shape[0]) & (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds_lidar
        if isinstance(box_preds, LiDARInstance3DBoxes):
            limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
            valid_pcd_inds = ((box_preds_lidar.center > limit_range[:3]) & (box_preds_lidar.center < limit_range[3:]))
            valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)
        else:
            valid_inds = valid_cam_inds

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                pred_box_type_3d=type(box_preds),
                box3d_camera=box_preds_camera[valid_inds].numpy(),
                box3d_lidar=box_preds_lidar[valid_inds].numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                pred_box_type_3d=type(box_preds),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0]),
                sample_idx=sample_idx)