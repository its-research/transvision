# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import numpy as np
from mmdet3d.evaluation import KittiMetric
from mmdet3d.registry import METRICS


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
        if not self.format_only:
            cat2label = data_infos['metainfo']['categories']
            label2cat = dict((v, k) for (k, v) in cat2label.items())
            assert 'instances' in data_annos[0]
            for i, annos in enumerate(data_annos):
                if len(annos['instances']) == 0:
                    kitti_annos = {
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
                else:
                    kitti_annos = {'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'location': [], 'dimensions': [], 'rotation_y': [], 'score': []}
                    for instance in annos['instances']:
                        label = instance['bbox_label']
                        kitti_annos['name'].append(label2cat[label])
                        kitti_annos['truncated'].append(instance['truncated'])
                        kitti_annos['occluded'].append(instance['occluded'])
                        kitti_annos['alpha'].append(instance['alpha'])
                        kitti_annos['bbox'].append(instance['bbox'])
                        kitti_annos['location'].append(instance['bbox_3d'][:3])
                        kitti_annos['dimensions'].append(instance['bbox_3d'][3:6])
                        kitti_annos['rotation_y'].append(instance['bbox_3d'][6])
                        kitti_annos['score'].append(instance['score'])
                    for name in kitti_annos:
                        kitti_annos[name] = np.array(kitti_annos[name])
                data_annos[i]['kitti_annos'] = kitti_annos
        return data_annos
