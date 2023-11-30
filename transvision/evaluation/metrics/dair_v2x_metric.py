# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmdet3d.registry import METRICS
# from mmengine import load
from mmengine.evaluator import BaseMetric

from transvision.config import add_arguments
from transvision.dataset import SUPPROTED_DATASETS
from transvision.eval import eval_vic
from transvision.models import SUPPROTED_MODELS
from transvision.models.model_utils import Channel
from transvision.v2x_utils import Evaluator, range2box


@METRICS.register_module()
class DAIRV2XMetric(BaseMetric):
    """Kitti evaluation metric.

    Args:
        ann_file (str): Annotation file path.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes. Defaults to [0, -40, -3, 70.4, 40, 0.0].
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        pklfile_prefix (str, optional): The prefix of pkl files, including the
            file path and the prefix of filename, e.g., "a/b/prefix". If not
            specified, a temp file will be created. Defaults to None.
        default_cam_key (str): The default camera for lidar to camera
            conversion. By default, KITTI: 'CAM2', Waymo: 'CAM_FRONT'.
            Defaults to 'CAM2'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 ann_file: str,
                 veh_config_path: str,
                 work_dir: str,
                 split_data_path: str,
                 model: str,
                 input: str,
                 test_mode: str,
                 val_data_path: str = None,
                 pcd_limit_range: List[float] = [0, -39.68, -3, 100, 39.68, 1],
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM2',
                 data_root: str = '',
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'dair_metric'
        super(DAIRV2XMetric, self).__init__(collect_device=collect_device, prefix=prefix)
        self.pcd_limit_range = pcd_limit_range
        self.ann_file = ann_file
        self.pklfile_prefix = pklfile_prefix

        self.default_cam_key = default_cam_key
        self.backend_args = backend_args
        self.data_root = data_root
        self.veh_config_path = veh_config_path
        self.work_dir = work_dir
        self.split_data_path = split_data_path
        self.model = model
        self.input = input
        self.val_data_path = val_data_path
        self.test_mode = test_mode

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        # TODO: current inference is not compatible with mmdet3d
        parser = argparse.ArgumentParser(conflict_handler='resolve')
        add_arguments(parser)
        args, _ = parser.parse_known_args()

        args.veh_config_path = self.veh_config_path
        last_checkpoint = os.path.join(self.work_dir, 'last_checkpoint')
        # read first line from last_checkpoint
        with open(last_checkpoint, 'r') as f:
            last_checkpoint = f.readline()
        args.veh_model_path = last_checkpoint
        args.split_data_path = self.split_data_path
        args.pred_classes = ['car']
        args.model = self.model
        args.input = self.input
        args.test_mode = self.test_mode
        args.output = './cache'

        evaluator = Evaluator(args.pred_classes)
        extended_range = range2box(np.array([0, -39.68, -3, 100, 39.68, 1]))
        dataset = SUPPROTED_DATASETS['vic-sync'](args.input, args, split='val', sensortype='lidar', extended_range=extended_range, val_data_path=self.val_data_path)

        pipe = Channel()
        model = SUPPROTED_MODELS[args.model](args, pipe)
        # Patch for FFNet evaluation
        if args.model == 'feature_flow':
            model.model.data_root = args.input
            model.model.test_mode = args.test_mode
        metric_dict = eval_vic(args, dataset, model, evaluator, pipe)

        return metric_dict
