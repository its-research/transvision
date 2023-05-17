# ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection

> [ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection](https://arxiv.org/abs/2106.01178)

## Introduction

We implement ImVoxelNet and provide the results and checkpoints on DAIR-V2X-V datasets with MMDetection3D.

## Results and models


| Modality |   Model   |  Car  |        |      | Pedestrain |        |      | Cyclist |        |      |                                          Download                                          |
| :--------: | :----------: | :-----: | :------: | :-----: | :----------: | :------: | :----: | :-------: | :------: | :----: | :-------------------------------------------------------------------------------------------: |
|          |            | Easy | Middle | Hard |    Easy    | Middle | Hard |  Easy  | Middle | Hard |                                                                                            |
|  Image  | ImVoxelNet | 56.86 | 39.74 | 33.00 |    9.09    |  9.09  | 9.09 |  10.48  |  9.09  | 9.09 | [model](https://drive.google.com/file/d/1HimYmBKhpnpU14UdKLykQJQrkSBDKGyf/view?usp=share_link) |

## Training & Evaluation

### Data Preparation

#### Download data and organise as follows

```
# For DAIR-V2X-V Dataset located at ${DAIR-V2X-V_DATASET_ROOT}
└─── single-vehicle-side
     ├───── image
     ├───── velodyne
     ├───── calib
     ├───── label
     └───── data_info.json        
```


#### Create Kitti-format data (Option for model training)

Data creation should be under the gpu environment.

In the end, the data and info files should be organized as follows
```
└─── single-vehicle-side             
     ├───── image
     ├───── velodyne
     ├───── calib
     ├───── label
     ├───── data_info.json
     ├───── ImageSets
     └───── training
        ├───── image_2
        ├───── velodyne
        ├───── label_2
        └───── calib
```

### Training & Evaluation

* Implementation Framework. We directly implement the benchmark with [mmdetection3d-1.1.0](https://github.com/open-mmlab/mmdetection3d/tree/v1.1.0).
* Training & Evaluation details.
  Before training the detectors, we should follow MMDetection3D to convert the "./data/DAIR-V2X/single-vehicle-side" into specific training format.
  We train the ImVoxelNet for 12 epochs.
  We evaluate the models on the valid part of DAIR-V2X-V.
  We set [0.5, 0.25, 0.25] as the IoU threshold for [Car, Pedestrain, Cyclist].
  Please refer [imvoxelnet_dair_v_3d_3class.py](./imvoxelnet_dair_v_3d_3class.py) for more evaluation details.
  We provide the evaluation results with 3D Average Precision.

## Citation

```latex
@inproceedings{rukhovich2022imvoxelnet,
  title={Imvoxelnet: Image to voxels projection for monocular and multi-view general-purpose 3d object detection},
  author={Rukhovich, Danila and Vorontsova, Anna and Konushin, Anton},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2397--2406},
  year={2022}
}
```
