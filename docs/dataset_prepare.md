# Data Preparation

## FFNet flow data

### Data Structure

#### flow_data_jsons/\*

We construct the frame pairs to generate the json files for FFNet training and evaluation.

- flow_data_info_train_2.json: frame pairs constructing from DAIR-V2X-C to simulate the different latency for training, including k=1,2
- flow_data_info_val_n.json: frame pairs constructing from DAIR-V2X-C to simulate the different latency for evaluation. n=1,2,3,4,5, corresponding to the 100ms, 200ms, 300, 400ms and 500ms latency, respectively.
- example_flow_data_info_train_2.json: frame pairs constructing from the example dataset to simulate the different latency for training, including k=1,2
- example_flow_data_info_val_n.json: frame pairs constructing from the example dataset to simulate the different latency for evaluation. n=1,2,3,4,5, corresponding to the 100ms to 500ms latency.

#### split_datas

The json files are used for splitting the dataset into train/val/test parts.
Please refer to the [split_data](https://github.com/AIR-THU/DAIR-V2X/tree/main/data/split_datas) for the latest updates.

### Data Preprocess

We use the DAIR-V2X-C-Example to illustrate how we preprocess the dataset for our experiment. For the convenience of overseas users, we provide the original DAIR-V2X-Example dataset [here](https://drive.google.com/file/d/1bFwWGXa6rMDimKeu7yJazAkGYO8s4RSI/view?usp=sharing). We provide the preprocessed DAIR-V2X-C-Example dataset [here](https://drive.google.com/file/d/1y8bGwI63TEBkDEh2JU_gdV7uidthSnoe/view?usp=sharing).

```shell
# Preprocess the dair-v2x-c dataset
python ./data/preprocess.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure
```

Origin

```txt
-- cooperative
    -- label_world
        -- *.json
    -- data_info.json
-- infrastructure-side
    -- calib
        -- camera_intrinsic
            -- *.json
        -- virtuallidar_to_camera
            -- *.json
        -- virtuallidar_to_world
            -- *.json
    -- image
        -- *.jpg
    -- label
        -- camera
            -- *.json
        -- virtuallidar
            -- *.json
    -- velodyne
        -- *.pcd
    -- data_info.json
-- vehicle-side
    -- calib
        -- camera_intrinsic
            -- *.json
        -- lidar_to_camera
            -- *.json
        -- lidar_to_novatel
            -- *.json
        -- novatel_to_world
            -- *.json
    -- image
        -- *.jpg
    -- label
        -- camera
            -- *.json
        -- lidar
            -- *.json
    -- velodyne
        -- *.pcd
    -- data_info.json
```

#### Generate the Frame Pairs

We have provided the frame pair files in [flow_data_jsons](./flow_data_jsons).
You can generate your frame pairs with the provided [example script](./frame_pair_generation.py).

```shell
# Preprocess the dair-v2x-c dataset
python ./data/frame_pair_generation.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure
```

## Convert to Kitti

### DAIR-V2X-V

```shell
python tools/dataset_converters/dair2kitti.py --source-root ./data/DAIR-V2X/single-vehicle-side --target-root ./data/DAIR-V2X/single-vehicle-side --split-path ./data/split_datas/single-vehicle-split-data.json --label-type camera --sensor-view vehicle

python tools/create_data.py kitti --root-path data/DAIR-V2X/single-vehicle-side/ --out-dir data/DAIR-V2X/single-vehicle-side/ --extra-tag kitti
```

### DAIR-V2X-I

```shell
python tools/dataset_converters/dair2kitti.py --source-root ./data/DAIR-V2X/single-infrastructure-side --target-root ./data/DAIR-V2X/single-infrastructure-side --split-path ./data/split_datas/single-infrastructure-split-data.json --label-type camera --sensor-view infrastructure

python tools/create_data.py kitti --root-path data/DAIR-V2X/single-infrastructure-side --out-dir data/DAIR-V2X/single-infrastructure-side --extra-tag kitti
```

### DAIR-V2X-C

```shell
bash ./scripts/convert_dair_v2x_c.sh
```

05/22 18:27:38 - mmengine - INFO - The number of instances per category in the dataset:

```shell
+----------------+--------+
| category       | number |
+----------------+--------+
| Car            | 81393  |
+----------------+--------+
```
