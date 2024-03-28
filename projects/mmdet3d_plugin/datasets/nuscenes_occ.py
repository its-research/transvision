import gzip
import os
import pickle

import mmcv
import numpy as np
import torch
from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..ray_iou.ego_pose_extractor import EgoPoseDataset
from .ray_metrics import generate_lidar_rays
from .ray_metrics import main as ray_based_miou
from .ray_metrics import process_one_sample


@DATASETS.register_module()
class NuSceneOcc(NuScenesDataset):
    r"""NuScenes Dataset.

    This dataset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_infos = self.load_annotations(self.ann_file)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        # self.train_split=data['train_split']
        # self.val_split=data['val_split']
        data_infos = data['infos'][::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'] if 'lidar_path' in info else '',
            sweeps=info['sweeps'] if 'sweeps' in info else [],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            timestamp=info['timestamp'] / 1e6,
        )
        if 'occ_path' in info:
            input_dict['occ_path'] = info['occ_path']
        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        ego2lidar = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation), inverse=True)
        input_dict['ego2lidar'] = ego2lidar
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                if 'LightwheelOcc' in self.version:
                    image_paths.append(os.path.join(self.data_root, cam_info['cam_path']))
                else:
                    image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                if 'LightwheelOcc' in self.version:
                    lidar2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2lidar_rotation']).rotation_matrix)
                else:
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])

                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = np.array(cam_info['cam_intrinsic'], dtype=np.float32)
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
            input_dict.update(dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # info = self.data_infos[index]
        # # filter out bbox containing no points
        # # if self.use_valid_flag:
        # #     mask = info['valid_flag']
        # # else:
        # #     mask = info['num_lidar_pts'] > 0
        # gt_bboxes_3d = info['gt_boxes'][mask]
        # gt_names_3d = info['gt_names'][mask]
        # gt_labels_3d = []
        # for cat in gt_names_3d:
        #     if cat in self.CLASSES:
        #         gt_labels_3d.append(self.CLASSES.index(cat))
        #     else:
        #         gt_labels_3d.append(-1)
        # gt_labels_3d = np.array(gt_labels_3d)

        # if self.with_velocity:
        #     gt_velocity = info['gt_velocity'][mask]
        #     nan_mask = np.isnan(gt_velocity[:, 0])
        #     gt_velocity[nan_mask] = [0.0, 0.0]
        #     gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # # the same as KITTI (0.5, 0.5, 0)
        # gt_bboxes_3d = LiDARInstance3DBoxes(
        #     gt_bboxes_3d,
        #     box_dim=gt_bboxes_3d.shape[-1],
        #     origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(gt_bboxes_3d=None, gt_labels_3d=None, gt_names=None)
        return anns_results

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def evaluate_miou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts = []
        flow_gts = []
        occ_preds = []
        flow_preds = []
        lidar_origins = []

        print('\nStarting Evaluation...')

        if 'LightwheelOcc' in self.version:
            ego_pose_dataset = EgoPoseDataset(self.data_infos, dataset_type='lightwheelocc')
        else:
            ego_pose_dataset = EgoPoseDataset(self.data_infos, dataset_type='openocc_v2')

        data_loader_kwargs = {
            'pin_memory': False,
            'shuffle': False,
            'batch_size': 1,
            'num_workers': 8,
        }

        data_loader = DataLoader(
            ego_pose_dataset,
            **data_loader_kwargs,
        )

        sample_tokens = [info['token'] for info in self.data_infos]

        for i, batch in tqdm(enumerate(data_loader), ncols=50):
            token = batch[0][0]
            output_origin = batch[1]

            data_id = sample_tokens.index(token)
            info = self.data_infos[data_id]

            occ_gt = np.load(info['occ_path'], allow_pickle=True)
            gt_semantics = occ_gt['semantics']
            gt_flow = occ_gt['flow']

            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            flow_gts.append(gt_flow)
            occ_preds.append(occ_results[data_id]['occ_results'].cpu().numpy())
            flow_preds.append(occ_results[data_id]['flow_results'].cpu().numpy())

        ray_based_miou(occ_preds, occ_gts, flow_preds, flow_gts, lidar_origins)

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        result_dict = {}

        if 'LightwheelOcc' in self.version:
            ego_pose_dataset = EgoPoseDataset(self.data_infos, dataset_type='lightwheelocc')
        else:
            ego_pose_dataset = EgoPoseDataset(self.data_infos, dataset_type='openocc_v2')

        data_loader_kwargs = {
            'pin_memory': False,
            'shuffle': False,
            'batch_size': 1,
            'num_workers': 8,
        }

        data_loader = DataLoader(
            ego_pose_dataset,
            **data_loader_kwargs,
        )

        sample_tokens = [info['token'] for info in self.data_infos]

        lidar_rays = generate_lidar_rays()
        lidar_rays = torch.from_numpy(lidar_rays)

        for index, batch in tqdm(enumerate(data_loader), ncols=50):
            token = batch[0][0]
            output_origin = batch[1]

            data_id = sample_tokens.index(token)

            occ_pred = occ_results[data_id]
            sem_pred = occ_pred['occ_results'].cpu().numpy()
            sem_pred = np.reshape(sem_pred, [200, 200, 16])

            flow_pred = occ_pred['flow_results'].cpu().numpy()
            flow_pred = np.reshape(flow_pred, [200, 200, 16, 2])

            pcd_pred = process_one_sample(sem_pred, lidar_rays, output_origin, flow_pred)

            pcd_cls = pcd_pred[:, 0].astype(np.int8)
            pcd_dist = pcd_pred[:, 1].astype(np.float16)
            pcd_flow = pcd_pred[:, 2:4].astype(np.float16)

            sample_dict = {'pcd_cls': pcd_cls, 'pcd_dist': pcd_dist, 'pcd_flow': pcd_flow}
            result_dict.update({token: sample_dict})

        final_submission_dict = {
            'method': 'occ',
            'team': 'The16thRoute',
            'authors': 'Bin Li',
            'e-mail': 'lbin@outlook.com',
            'institution / company': 'SEU',
            'country / region': 'China/Beijing',
            'results': result_dict
        }

        save_path = os.path.join(submission_prefix, 'submission.gz')
        with open(save_path, 'wb') as f:
            f.write(gzip.compress(pickle.dumps(final_submission_dict), mtime=0))

        print('\nFinished.')
