import json
import pickle

import mmcv
import numpy as np
import yaml
from pypcd4 import PointCloud


def load_json(path):
    with open(path, mode='r') as f:
        data = json.load(f)
    return data


def load_yaml(path):
    with open(path, 'r') as f:
        data = yaml.load(f)
    return data


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(item, path):
    with open(path, 'wb') as f:
        pickle.dump(item, f)


def read_pcd(pcd_path):
    pcd = PointCloud.from_path(pcd_path)
    time = None
    pcd_np_points = np.asarray(
        pcd.numpy(("x", "y", "z", "intensity")),
        dtype=np.float32,
    ).copy()
    pcd_np_points[:, 3] /= 256.0
    pcd_np_points = pcd_np_points[
        np.isfinite(pcd_np_points).all(axis=1)
    ]
    return pcd_np_points, time


def read_jpg(jpg_path):
    image = mmcv.imread(jpg_path)
    return image
