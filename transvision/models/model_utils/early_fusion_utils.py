import numpy as np
from pypcd4 import Encoding, PointCloud


def read_pcd(path_pcd):
    return PointCloud.from_path(path_pcd)


def concatenate_pcd2bin(pc1, pc2, path_save):
    points1 = np.asarray(
        pc1.numpy(("x", "y", "z", "intensity")),
        dtype=np.float32,
    ).copy()
    points2 = np.asarray(
        pc2.numpy(("x", "y", "z", "intensity")),
        dtype=np.float32,
    ).copy()
    points1[:, 3] /= 255.0
    points = np.concatenate((points1, points2), axis=0)
    PointCloud.from_xyzi_points(points).save(
        path_save,
        encoding=Encoding.BINARY_COMPRESSED,
    )
