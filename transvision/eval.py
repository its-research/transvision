import argparse
import logging
import os.path as osp

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from transvision.config import add_arguments
from transvision.dataset import SUPPROTED_DATASETS
from transvision.dataset.dataset_utils import save_pkl
from transvision.models import SUPPROTED_MODELS
from transvision.models.model_utils import Channel
from transvision.v2x_utils import Evaluator, range2box

logger = logging.getLogger(__name__)


def eval_vic(args, dataset, model, evaluator, pipe):
    idx = -1
    for VICFrame, label, filt, bbox_3d_lwhr in tqdm(dataset):
        idx += 1

        try:
            veh_id = (dataset.data[idx][0]['vehicle_pointcloud_path'].split('/')[-1].replace('.pcd', ''))
        except Exception:
            veh_id = (VICFrame['vehicle_pointcloud_path'].split('/')[-1].replace('.pcd', ''))

        pred = model(
            VICFrame,
            filt,
            idx,
            None if not hasattr(dataset, 'prev_inf_frame') else dataset.prev_inf_frame,
        )
        # print(veh_id)
        # print(bbox_3d_lwhr)
        # perm_pred = [0, 4, 7, 3, 1, 5, 6, 2]
        # perm_pred = [3, 0, 4, 7, 2, 1, 5, 6]
        # perm_label = [3, 2, 1, 0, 7, 6, 5, 4]

        # pred['boxes_3d'][:, :, 2] -= 0.8
        # for p, l in zip(pred['boxes_3d'], label['boxes_3d']):
        #     print(p[perm_pred], l[perm_label])
        #     print('=====================')

        # exit()

        evaluator.add_frame(pred, label)
        pipe.flush()
        pred['label'] = label['boxes_3d']
        pred['veh_id'] = veh_id
        save_pkl(pred, osp.join(args.output, 'result', pred['veh_id'] + '.pkl'))

    results_3d = evaluator.print_ap('3d')
    results_bev = evaluator.print_ap('bev')
    print('Average Communication Cost = %.2lf Bytes' % (pipe.average_bytes()))

    table = []
    line = ['3d']
    results = {}
    for key in results_3d:
        line.append(results_3d[key])
        results[key] = results_3d[key]
    line.append(pipe.average_bytes())
    table.append(line)
    line = ['bev']
    for key in results_bev:
        line.append(results_bev[key])
        results[key] = results_bev[key]
    line.append(pipe.average_bytes())
    table.append(line)
    headers = ['Car', 'AP@0.30', '0.50', '0.70', 'AB(Byte)']
    print(tabulate(table, headers, tablefmt='pipe'))

    return results


def eval_single(args, dataset, model, evaluator):
    for frame, label, filt in tqdm(dataset):
        pred = model(frame, filt)
        if args.sensortype == 'camera':
            evaluator.add_frame(pred, label['camera'])
        elif args.sensortype == 'lidar':
            evaluator.add_frame(pred, label['lidar'])
        save_pkl(
            {'boxes_3d': label['lidar']['boxes_3d']},
            osp.join(args.output, 'result', frame.id['camera'] + '.pkl'),
        )

    evaluator.print_ap('3d')
    evaluator.print_ap('bev')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    add_arguments(parser)
    args, _ = parser.parse_known_args()
    # add model-specific arguments
    SUPPROTED_MODELS[args.model].add_arguments(parser)
    args = parser.parse_args()

    if args.quiet:
        level = logging.ERROR
    elif args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=level,
    )

    extended_range = range2box(np.array(args.extended_range))
    logger.info('loading dataset')

    dataset = SUPPROTED_DATASETS[args.dataset](
        args.input,
        args,
        split=args.split,
        sensortype=args.sensortype,
        extended_range=extended_range,
        val_data_path=args.val_data_path,
    )

    logger.info('loading evaluator')
    evaluator = Evaluator(args.pred_classes)

    logger.info('loading model')
    if args.eval_single:
        model = SUPPROTED_MODELS[args.model](args)
        eval_single(args, dataset, model, evaluator)
    else:
        pipe = Channel()
        model = SUPPROTED_MODELS[args.model](args, pipe)
        # Patch for FFNet evaluation
        if args.model == 'feature_flow':
            model.model.data_root = args.input
            model.model.test_mode = args.test_mode
        eval_vic(args, dataset, model, evaluator, pipe)
