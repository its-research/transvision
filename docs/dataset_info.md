# English | [简体中文](./README_zh-CN.md)

## DAIR-V2X

### Data Structure

```markdown
single-infrastructure-side              # DAIR-V2X-I Dataset
    ├── image
        ├── {id}.jpg
    ├── velodyne
        ├── {id}.pcd
    ├── calib
        ├── camera_intrinsic
            ├── {id}.json
        ├── virtuallidar_to_camera
            ├── {id}.json
    ├── label
        ├── camera                      # Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects in image based on image frame time
            ├── {id}.json
        ├── virtuallidar                # Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects in point cloud based on point cloud frame time
            ├── {id}.json
    ├── data_info.json                  # Relevant index information of the Infrastructure data
single-vehicle-side                     # DAIR-V2X-V
    ├── image
        ├── {id}.jpg
    ├── velodyne
        ├── {id}.pcd
    ├── calib
        ├── camera_intrinsic
            ├── {id}.json
        ├── lidar_to_camera
            ├── {id}.json
    ├── label
        ├── camera                      # Labeled data in Vehicle LiDAR Coordinate System fitting objects in image based on image frame time
            ├── {id}.json
        ├── lidar                       # Labeled data in Vehicle LiDAR Coordinate System fitting objects in point cloud based on point cloud frame time
            ├── {id}.json
    ├── data_info.json                  # Relevant index information of the Vehicle data
cooperative-vehicle-infrastructure      # DAIR-V2X-C
    ├── infrastructure-side             # DAIR-V2X-C-I
        ├── image
            ├── {id}.jpg
        ├── velodyne
            ├── {id}.pcd
        ├── calib
            ├── camera
                ├── {id}.json
            ├── virtuallidar_to_world
                ├── {id}.json
            ├── virtuallidar_to_camera
                ├── {id}.json
        ├── label
            ├── camera                  # Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects in image based on image frame time
                ├── {id}.json
            ├── virtuallidar            # Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects in point cloud based on point cloud frame time
                ├── {id}.json
        ├── data_info.json              # Relevant index information of Infrastructure data
    ├── vehicle-side                    # DAIR-V2X-C-V
        ├── image
            ├── {id}.jpg
        ├── velodyne
            ├── {id}.pcd
        ├── calib
            ├── camera_intrinsic
                ├── {id}.json
            ├── lidar_to_camera
                ├── {id}.json
            ├── lidar_to_novatel
                ├── {id}.json
            ├── novatel_to_world
                ├── {id}.json
        ├── label
            ├── camera                  # Labeled data in Vehicle LiDAR Coordinate System fitting objects in image based on image frame time
                ├── {id}.json
            ├── lidar                   # Labeled data in Vehicle LiDAR Coordinate System fitting objects in point cloud based on point cloud frame time
                ├── {id}.json
        ├── data_info.json              # Relevant index information of the Vehicle data
    ├── cooperative                     # Coopetative Files
        ├── label_world                 # Vehicle-Infrastructure Cooperative (VIC) Annotation files
            ├── {id}.json
        ├── data_info.json              # Relevant index information combined the Infrastructure data and the Vehicle data
```

______________________________________________________________________

### Introduction to data-info.json

#### single-infrastructure-side/data_info.json

```json
{
  "image_path",
  "pointcloud_path",
  "label_virtuallidar_path",
  "label_camera_path",
  "calib_virtuallidar_to_camera_path",
  "calib_camera_intrinsic_path"
}
```

#### single-vehicle-side/data_info.json

```json
{
  "image_path",
  "image_timestamp",
  "pointcloud_path",
  "pointcloud_timestamp",
  "label_lidar_path",
  "label_camera_path",
  "calib_lidar_to_camera_path",
  "calib_camera_intrinsic_path"
}
```

#### cooperative-vehicle-infrastructure/infrastructure-side/data_info.json

```json
{
  "image_path",
  "image_timestamp",
  "pointcloud_path",
  "pointcloud_timestamp",
  "label_lidar_path",
  "label_camera_path",
  "calib_virtuallidar_to_world_path",
  "calib_camera_intrinsic_path",
  "batch_id",
  "intersection_loc",
  "batch_start_id",
  "batch_end_id"
}
```

**Comment**：

- Infrastructure and vehicle frame with the same "batch_id" share the same segments.

#### cooperative-vehicle-infrastructure/vehicle-side/data_info.json

```json
{
  "image_path",
  "image_timestamp",
  "pointcloud_path",
  "pointcloud_timestamp",
  "label_lidar_path",
  "label_camera_path",
  "calib_lidar_to_camera_path",
  "calib_lidar_to_novatel_path",
  "calib_novatel_to_world_path",
  "calib_camera_intrinsic_path",
  "batch_id",
  "intersection_loc",
  "batch_start_id",
  "batch_end_id"
}
```

##### cooperative-vehicle-infrastructure/cooperative/data_info.json

```json
{
  "infrastructure_image_path",
  "infrastructure_pointcloud_path",
  "vehicle_image_path",
  "vehicle_pointcloud_path",
  "cooperative_label_path"
}
```

______________________________________________________________________

### Single-view Annotation File

```json
{
  "type": type,
  "truncated_state": truncated_state,
  "occluded_state": occluded_state,
  "2d_box": {
    "xmin": xmin,
    "ymin": ymin,
    "xmax": xmax,
    "ymax": ymax
  },
  "3d_dimensions": {
    "h": height,
    "w": width,
    "l": length
  },
  "3d_location": {
    "x": x,
    "y": y,
    "z": z
  },
  "rotation": rotation
}
```

Comment

- 10 object classes, including: Car, Truck, Van, Bus, Pedestrian, Cyclist, Tricyclist,
  Motorcyclist, Barrowlist, and TrafficCone.

______________________________________________________________________

### Cooperative Annotation File

```json
{
  "type": type,
  "world_8_points": 8 corners of 3d bounding box,
  "system_error_offset": {
    "delta_x": delta_x,
    "delta_y": delta_y,
  }
}
```

Comment

We only consider the four class \["Car", "Truck", "Van", "Bus"\] and generate 9311 annotation files.

______________________________________________________________________

### Statistics

DAIR-V2X is the first large-scale, multi-modality, multi-view dataset for Vehicle-Infrastructure Cooperative Autonomous Driving (VICAD), with 2D&3D object annotations. All data is captured from real scenarios.

- Totally 71254 LiDAR frames and 71254 Camera images:
  - DAIR-V2X Cooperative Dataset (DAIR-V2X-C): 38845 LiDAR frames, 38845 Camera images
  - DAIR-V2X Infrastructure Dataset (DAIR-V2X-I): 10084 LiDAR frames, 10084 Camera images
  - DAIR-V2X Vehicle Dataset (DAIR-V2X-V): 22325 LiDAR frames, 22325 Camera images

We split 50%, 20% and 30% of the dataset into a training set, validation set, and testing set separately. The training set and validation set is now available, and the testing set will be released along with the subsequent challenge activities.

______________________________________________________________________

### Citation

```txt
@inproceedings{yu2022dairv2x,
    title={DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection},
    author={Yu, Haibao and Luo, Yizhen and Shu, Mao and Huo, Yiyi and Yang, Zebang and Shi, Yifeng and Guo, Zhenglong and Li, Hanyu and Hu, Xing and Yuan, Jirui and Nie, Zaiqing},
    booktitle={IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
    month = jun,
    year={2022}
}
```

______________________________________________________________________

### Organizations

- Institute for AI Industry Research, Tsinghua University (AIR)
- Beijing High-level Autonomous Driving Demonstration Area
- Beijing Connected and Autonomous Vehicles Technology Co., Ltd
- Baidu Apollo
- Beijing Academy of Artificial Intelligence, BAAI

______________________________________________________________________

## Data Preparation

### DAIR-V2X Dataset

DAIR-V2X is the first large-scale and real-world vehicle-infrastructure cooperative 3D object detection dataset. This dataset includes the DAIR-V2X-C, which has the cooperative view.
We train and evaluate the models on DAIR-V2X dataset. For downloading DAIR-V2X dataset, please refer to the guidelines in [DAIR-V2X](https://thudair.baai.ac.cn/cooptest).

### flow_data_jsons/\*

We construct the frame pairs to generate the json files for FFNet training and evaluation.

- flow_data_info_train_2.json: frame pairs constructing from DAIR-V2X-C to simulate the different latency for training, including k=1,2
- flow_data_info_val_n.json: frame pairs constructing from DAIR-V2X-C to simulate the different latency for evaluation. n=1,2,3,4,5, corresponding to the 100ms, 200ms, 300, 400ms and 500ms latency, respectively.
- example_flow_data_info_train_2.json: frame pairs constructing from the example dataset to simulate the different latency for training, including k=1,2
- example_flow_data_info_val_n.json: frame pairs constructing from the example dataset to simulate the different latency for evaluation. n=1,2,3,4,5, corresponding to the 100ms to 500ms latency.

### split_datas

The json files are used for splitting the dataset into train/val/test parts.
Please refer to the [split_data](https://github.com/AIR-THU/DAIR-V2X/tree/main/data/split_datas) for the latest updates.

## Data Preprocess

We use the DAIR-V2X-C-Example to illustrate how we preprocess the dataset for our experiment. For the convenience of overseas users, we provide the original DAIR-V2X-Example dataset [here](https://drive.google.com/file/d/1bFwWGXa6rMDimKeu7yJazAkGYO8s4RSI/view?usp=sharing). We provide the preprocessed DAIR-V2X-C-Example dataset [here](https://drive.google.com/file/d/1y8bGwI63TEBkDEh2JU_gdV7uidthSnoe/view?usp=sharing).

```shell
# Preprocess the dair-v2x-c dataset
python ./data/dair-v2x/preprocess.py --source-root ./data/dair-v2x/DAIR-V2X-Examples/cooperative-vehicle-infrastructure
```

## Generate the Frame Pairs

We have provided the frame pair files in [flow_data_jsons](./flow_data_jsons).
You can generate your frame pairs with the provided [example script](./frame_pair_generation.py).

```shell
# Preprocess the dair-v2x-c dataset
python ./data/dair-v2x/frame_pair_generation.py --source-root ./data/dair-v2x/DAIR-V2X-Examples/cooperative-vehicle-infrastructure
```
