import numpy as np

from transvision.v2x_utils import get_arrow_end


def get_box_info(result):
    if len(result[0].pred_instances_3d.bboxes_3d.tensor) == 0:
        box_lidar = np.zeros((1, 8, 3))
        box_ry = np.zeros(1)
    else:
        box_lidar = result[0].pred_instances_3d.bboxes_3d.corners.cpu().numpy()
        box_ry = result[0].pred_instances_3d.bboxes_3d.tensor[:, -1].cpu().numpy()

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
