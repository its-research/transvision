import os

import cv2
import numpy as np


def featuremap_2_heatmap(feature_map):
    # print(type(feature_map))
    # assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:, 0, :, :] * 0
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


def save_feature_map(save_file, featuremap):
    heatmap = featuremap_2_heatmap(featuremap)
    heatmap = cv2.resize(heatmap, (288, 288))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 1.0
    cv2.imwrite(os.path.join(save_file), superimposed_img)
