# Copyright (c) DAIR-V2X (AIR). All rights reserved.
import copy
import json
import os
import pickle
from os import path as osp
from typing import List, Optional, Union

import numpy as np
from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes
from mmengine.fileio import dump, join_path, load

# TODO https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/det3d_dataset.py#L218


@DATASETS.register_module()
class V2XDataset(Det3DDataset):
    r"""DAIR-V2X Dataset.

    https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/customize_dataset.html

    """
    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car', 'Truck', 'Van', 'Person_sitting', 'Tram', 'Misc'),
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255)]
    }

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 data_prefix='velodyne',
                 pipeline=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 metainfo: Optional[dict] = None,
                 backend_args: Optional[dict] = None,
                 pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0]):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            metainfo=metainfo)
        self.backend_args = backend_args
        self.split = split
        self.root_split = os.path.join(self.data_root, split)
        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range
        self.data_prefix = data_prefix

    def load_data_list(self) -> List[dict]:
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        self._metainfo = annotations['metainfo']

        # metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        data_list = self.__load_v2x_annotations(raw_data_list)

        annotations['data_list'] = data_list
        dump(annotations, self.ann_file)

        return data_list

    def __my_read_json(self, path_json):
        with open(path_json, 'r') as load_f:
            my_json = json.load(load_f)
        return my_json

    def __box_convert_lidar2cam(self, location, dimension, rotation, calib_lidar2cam):
        location['z'] = location['z'] - dimension['h'] / 2
        extended_xyz = np.array([location['x'], location['y'], location['z'], 1])
        location_cam = extended_xyz @ calib_lidar2cam.T
        location_cam = location_cam[:3]

        dimension_cam = [dimension['w'], dimension['h'], dimension['l']]
        rotation_y = rotation

        # TODO: hard code by yuhb
        alpha = -10.0

        return location_cam, dimension_cam, rotation_y, alpha

    def __load_v2x_annotations(self, raw_data_list):
        """Load annotations from dair-v2x
        Args:
            dict_keys(['name', 'truncated', 'occluded',
            'alpha', 'bbox', 'dimensions', 'location',
            'rotation_y', 'score', 'index', 'group_ids',
            'difficulty', 'num_points_in_gt'])
        Returns:
            Dict
        """
        data_list = []
        for raw_data_info in raw_data_list:
            # data_info = self.parse_data_info(raw_data_info)
            data_info = raw_data_info
            anno_path = os.path.join(self.data_root, data_info['cooperative_label_w2v_path'])
            annos = self.__my_read_json(anno_path)
            kitti_annos = {}
            kitti_annos['name'] = []
            kitti_annos['occluded'] = []
            kitti_annos['truncated'] = []
            kitti_annos['dimensions'] = []
            kitti_annos['location'] = []
            kitti_annos['rotation_y'] = []
            kitti_annos['index'] = []
            kitti_annos['alpha'] = []
            kitti_annos['bbox'] = []

            calib_v_lidar2cam_filename = os.path.join(self.data_root, data_info['calib_v_lidar2cam_path'])
            calib_v_lidar2cam = self.__my_read_json(calib_v_lidar2cam_filename)
            calib_v_cam_intrinsic_filename = os.path.join(self.data_root, data_info['calib_v_cam_intrinsic_path'])
            calib_v_cam_intrinsic = self.__my_read_json(calib_v_cam_intrinsic_filename)
            rect = np.identity(4)
            Trv2c = np.identity(4)
            Trv2c[0:3, 0:3] = calib_v_lidar2cam['rotation']
            Trv2c[0:3, 3] = [calib_v_lidar2cam['translation'][0][0], calib_v_lidar2cam['translation'][1][0], calib_v_lidar2cam['translation'][2][0]]
            P2 = np.identity(4)
            P2[0:3, 0:3] = np.array(calib_v_cam_intrinsic['cam_K']).reshape(3, 3)
            data_info['calib'] = {}
            data_info['calib']['R0_rect'] = rect
            data_info['calib']['Tr_velo_to_cam'] = Trv2c
            data_info['calib']['P2'] = P2

            for idx, anno in enumerate(annos):
                location, dimensions, rotation_y, alpha = self.__box_convert_lidar2cam(anno['3d_location'], anno['3d_dimensions'], anno['rotation'], Trv2c)
                if dimensions[0] == 0.0:
                    continue
                anno['bbox_label'] = anno['type'].capitalize()
                kitti_annos['name'].append(anno['type'].capitalize())
                kitti_annos['dimensions'].append(dimensions)
                kitti_annos['location'].append(location)
                kitti_annos['rotation_y'].append(rotation_y)
                kitti_annos['alpha'].append(alpha)
                kitti_annos['index'].append(idx)
                """ TODO: Valid Bbox"""
                kitti_annos['occluded'].append(0)
                kitti_annos['truncated'].append(0)
                bbox = [0, 0, 100, 100]
                kitti_annos['bbox'].append(bbox)

            for name in kitti_annos:
                kitti_annos[name] = np.array(kitti_annos[name])

            data_info['annos'] = kitti_annos
            data_info['kitti_annos'] = kitti_annos
            data_info['sample_idx'] = data_info['vehicle_idx']

            data_list.append(data_info)

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        This method should return dict or list of dict. Each dict or list
        contains the data information of a training sample. If the protocol of
        the sample annotations is changed, this function can be overridden to
        update the parsing logic while keeping compatibility.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            list or list[dict]: Parsed annotation.
        """
        for prefix_key, prefix in self.data_prefix.items():
            assert prefix_key in raw_data_info, (f'raw_data_info: {raw_data_info} dose not contain prefix key'
                                                 f'{prefix_key}, please check your data_prefix.')
            raw_data_info[prefix_key] = join_path(prefix, raw_data_info[prefix_key])
        return raw_data_info

    def _get_pts_filename(self, idx):
        """Get point cloud filename according to the given index.

        Args:
            index (int): Index of the point cloud file to get.

        Returns:
            str: Name of the point cloud file.
        """
        pts_filename = osp.join(self.root_split, self.data_prefix, f'{idx:06d}.bin')
        return pts_filename

    def get_data_info(self, index: int) -> dict:
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str | None): Prefix of image files.
                - img_info (dict): Image info.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        if self.serialize_data:
            start_addr = 0 if index == 0 else self.data_address[index - 1].item()
            end_addr = self.data_address[index].item()
            bytes = memoryview(self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[index])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if index >= 0:
            data_info['sample_idx'] = index
        else:
            data_info['sample_idx'] = len(self) + index

        # data_info = self.data_list[index]
        sample_veh_idx = data_info['vehicle_idx']
        sample_inf_idx = data_info['infrastructure_idx']
        # inf_img_filename = os.path.join(self.data_root, info['infrastructure_image_path'])
        veh_img_filename = os.path.join(self.data_root, data_info['vehicle_image_path'])

        calib_inf2veh_filename = os.path.join(self.data_root, data_info['calib_lidar_i2v_path'])
        calib_inf2veh = self.__my_read_json(calib_inf2veh_filename)

        # TODO: consider use torch.Tensor only
        rect = data_info['calib']['R0_rect'].astype(np.float32)
        Trv2c = data_info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = data_info['calib']['P2'].astype(np.float32)
        lidar2img = P2 @ rect @ Trv2c

        inf_pts_filename = os.path.join(self.data_root, data_info['infrastructure_pointcloud_bin_path'])
        veh_pts_filename = os.path.join(self.data_root, data_info['vehicle_pointcloud_bin_path'])

        # For FlowNet
        if 'infrastructure_idx_t_0' in data_info.keys():
            infrastructure_pointcloud_bin_path_t_0 = data_info['infrastructure_pointcloud_bin_path_t_0']
            infrastructure_pointcloud_bin_path_t_1 = data_info['infrastructure_pointcloud_bin_path_t_1']
            infrastructure_pointcloud_bin_path_t_2 = data_info['infrastructure_pointcloud_bin_path_t_2']
            infrastructure_t_0_1 = data_info['infrastructure_t_0_1']
            infrastructure_t_1_2 = data_info['infrastructure_t_1_2']
        else:
            infrastructure_pointcloud_bin_path_t_0 = None
            infrastructure_pointcloud_bin_path_t_1 = None
            infrastructure_pointcloud_bin_path_t_2 = None
            infrastructure_t_0_1 = None
            infrastructure_t_1_2 = None

        input_dict = dict(
            sample_veh_idx=sample_veh_idx,
            sample_inf_idx=sample_inf_idx,
            infrastructure_pts_filename=inf_pts_filename,
            vehicle_pts_filename=veh_pts_filename,
            img_prefix=None,
            img_info=dict(filename=veh_img_filename),
            lidar2img=lidar2img,
            inf2veh=calib_inf2veh,
            infrastructure_pointcloud_bin_path_t_0=infrastructure_pointcloud_bin_path_t_0,
            infrastructure_pointcloud_bin_path_t_1=infrastructure_pointcloud_bin_path_t_1,
            infrastructure_pointcloud_bin_path_t_2=infrastructure_pointcloud_bin_path_t_2,
            infrastructure_t_0_1=infrastructure_t_0_1,
            infrastructure_t_1_2=infrastructure_t_1_2,
            sample_idx=data_info['sample_idx'])

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        if self.serialize_data:
            start_addr = 0 if index == 0 else self.data_address[index - 1].item()
            end_addr = self.data_address[index].item()
            bytes = memoryview(self.data_bytes[start_addr:end_addr])  # type: ignore
            info = pickle.loads(bytes)  # type: ignore
        else:
            info = copy.deepcopy(self.data_list[index])

        # info = self.data_list[index]
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

        # convert gt_bboxes_3d to velodyne coordinates
        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(self.box_mode_3d, np.linalg.inv(rect @ Trv2c))
        gt_bboxes = annos['bbox']

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]

        gt_labels = []
        for cat in gt_names:
            if cat in self.METAINFO['classes']:
                gt_labels.append(self.METAINFO['classes'].index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, bboxes=gt_bboxes, labels=gt_labels, gt_names=gt_names)
        return anns_results

    def drop_arrays_by_name(self, gt_names, used_classes):
        """Drop irrelevant ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be dropped.
        """
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def keep_arrays_by_name(self, gt_names, used_classes):
        """Keep useful ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be keeped.
        """
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def remove_dontcare(self, ann_info):
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos. The ``'DontCare'``
                annotations will be removed according to ann_file['name'].

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        relevant_annotation_indices = [i for i, x in enumerate(ann_info['name']) if x != 'DontCare']
        for key in ann_info.keys():
            img_filtered_annotations[key] = (ann_info[key][relevant_annotation_indices])
        return img_filtered_annotations
