import copy
import os
import pickle
from typing import Callable, List, Union

import numpy as np
from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes, limit_period
# from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.fileio import load

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
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 data_prefix: str = 'velodyne',
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = 'CAM2',
                 load_type: str = 'frame_based',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 **kwargs) -> None:

        self.pcd_limit_range = pcd_limit_range
        assert load_type in ('frame_based', 'mv_image_based', 'fov_image_based')
        self.load_type = load_type
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera')

    def __box_convert_lidar2cam(self, location, dimension, rotation, calib_lidar2cam):
        location['z'] = location['z'] - dimension['h'] / 2
        extended_xyz = np.array([location['x'], location['y'], location['z'], 1])
        location_cam = extended_xyz @ calib_lidar2cam.T
        location_cam = location_cam[:3]

        dimension_cam = [dimension['l'], dimension['h'], dimension['w']]
        rotation_y = rotation
        rotation_y = limit_period(rotation_y, period=np.pi * 2)

        # TODO: hard code by yuhb
        alpha = -10.0

        return location_cam, dimension_cam, rotation_y, alpha

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Load annotations from dair-v2x
        Args:
            dict_keys(['name', 'truncated', 'occluded',
            'alpha', 'bbox', 'dimensions', 'location',
            'rotation_y', 'score', 'index', 'group_ids',
            'difficulty', 'num_points_in_gt'])
        Returns:
            Dict
        """
        # data_info = self.parse_data_info(raw_data_info)
        data_info = raw_data_info
        anno_path = os.path.join(self.data_root, data_info['cooperative_label_w2v_path'])
        annos = load(anno_path)
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
        calib_v_lidar2cam = load(calib_v_lidar2cam_filename)
        calib_v_cam_intrinsic_filename = os.path.join(self.data_root, data_info['calib_v_cam_intrinsic_path'])
        calib_v_cam_intrinsic = load(calib_v_cam_intrinsic_filename)
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

        return data_info

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
        calib_inf2veh = load(calib_inf2veh_filename)

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
        # annos = self._remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']

        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

        # # convert gt_bboxes_3d to velodyne coordinates
        # if index == 728:
        #     print(gt_bboxes_3d[0])
        #     print(rect @ Trv2c)
        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(self.box_mode_3d, np.linalg.inv(rect @ Trv2c))
        # if index == 728:
        #     print(gt_bboxes_3d[0])
        #     exit()

        gt_bboxes = annos['bbox'].astype('float32')

        gt_labels = []
        gt_names = annos['name']
        for cat in gt_names:
            if cat in self.METAINFO['classes']:
                gt_labels.append(self.METAINFO['classes'].index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, bboxes=gt_bboxes, labels=gt_labels, gt_names=gt_names)
        return anns_results
