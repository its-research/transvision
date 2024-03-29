# Benchmark

## FFNET-VIC3D

### PaperWithCodes

| Model              | Latency (ms) | 3D IoU=0.5 | 3D Iou=0.7 | bev IoU=0.5 | bev Iou=0.7 | AB (Byte) |
| :----------------- | :----------: | :--------: | :--------: | :---------: | :---------: | :-------: |
| FFNet              |      0       |   55.81    |   30.23    |    63.54    |    54.16    |  1.2×105  |
| FFNet              |     100      |   55.48    |   31.50    |    63.14    |    54.28    |  1.2×105  |
| FFNet              |     200      |   55.37    |   31.66    |    63.20    |    54.69    |  1.2×105  |
| FFNet(w/o pred)    |      0       |   55.81    |   30.23    |    63.54    |    54.16    |  6.2×104  |
| FFNet-V2(w/o pred) |      0       |   55.78    |   30.22    |    64.23    |    55.00    |  1.2×105  |
| FFNet(w/o pred)    |     200      |   50.27    |   27.57    |    57.93    |    48.16    |  6.2×104  |
| FFNet-V2(w/o pred) |     200      |   49.90    |   27.33    |    58.00    |    48.22    |  1.2×105  |

Example: evaluate `FFNET` on `DAIR-V2X-C` with 100ms latency:

```shell
# bash scripts/lidar_feature_flow.sh [YOUR_CUDA_DEVICE] [YOUR_FFNET_WORKDIR] [DELAY_K]
cd ${OpenDAIRV2X_root}/v2x
bash scripts/lidar_feature_flow.sh 0 1 'FlowPred'
```

### config_basemodel

(work with mmdet3d==0.17.1)

| origin | Car(AP@0.70) | fusion  | veh_only | inf_only | transvision | Car(AP@0.70) |  fusion(ms/s)   | veh_only | inf_only |
| :----: | :----------: | :-----: | :------: | :------: | :---------: | :----------: | :-------------: | :------: | :------: |
|        |     bev      | 34.4805 |          |          |             |     bev      | 40.5926/34.1077 | 32.9987  |  1.2987  |
|        |      3d      | 21.3489 |          |          |             |      3d      | 22.0547/22.4537 | 21.1024  |  1.2987  |
|        | Car(AP@0.50) | fusion  | veh_only | inf_only |             | Car(AP@0.50) |     fusion      | veh_only | inf_only |
|        |     bev      | 43.5100 |          |          |             |     bev      | 43.5164/42.5577 | 35.0143  |  6.5574  |
|        |      3d      | 41.1886 |          |          |             |      3d      | 40.8895/34.3691 | 33.9840  |  4.5455  |

work_dirs/ffnet-vic3d/basemodel/mmdet3d_0.17.1/20231024_131833.log

(work with mmdet3d==1.2.0)

| transvision | Car(AP@0.70) | fusion | veh_only | inf_only |
| :---------: | :----------: | :----: | :------: | :------: |
|             |     bev      |        |          |          |
|             |      3d      |        |          |          |
|             | Car(AP@0.50) | fusion | veh_only | inf_only |
|             |     bev      |        |          |          |
|             |      3d      |        |          |          |

### config_ffnet

bash scripts/lidar_feature_flow.sh 0 ./ 1 'FlowPred'

| Car | AP@0.30 | 0.50  | 0.70  |   AB(Byte)    |
| :-: | :-----: | :---: | :---: | :-----------: |
| bev |  66.09  | 63.84 | 54.41 | 1672.45 Bytes |
| 3d  |  64.39  | 56.23 | 32.02 |               |

### Downloaded Model

#### ffnet.pth

```shell
bash scripts/lidar_feature_flow.sh 0 1 'FlowPred'
```

1501 frames

| Car | AP@0.30 | 0.50  | 0.70  |   AB(Byte)    |
| :-: | :-----: | :---: | :---: | :-----------: |
| bev |  65.67  | 63.15 | 54.27 | 1681.18 Bytes |
| 3d  |  63.55  | 55.48 | 31.54 |               |

1.2.0 refer

| Car | AP@0.30 |    0.50 |    0.70 | AB(Byte) |
| :-- | ------: | ------: | ------: | -------: |
| bev | 69.6456 | 66.2805 | 54.2942 |  4544.41 |
| 3d  | 67.1332 | 56.2883 | 28.7926 |  4544.41 |

#### ffnet_without_prediction.pth

1789 frames

```shell
bash scripts/lidar_feature_fusion.sh 0 ./
```

0.17.0

| Car | AP@0.30 | 0.50  | 0.70  |   AB(Byte)    |
| :-: | :-----: | :---: | :---: | :-----------: |
| bev |  65.70  | 62.95 | 53.45 | 1760.50 Bytes |
| 3d  |  63.52  | 54.93 | 29.29 |               |

1.2.0 our best model

| Car | AP@0.30 |  0.50   |  0.70   | AB(Byte) |
| :-- | :-----: | :-----: | :-----: | :------: |
| bev | 66.1538 | 63.6624 | 54.3148 | 1746.78  |
| 3d  | 64.1446 | 55.9369 | 29.9052 | 1746.78  |

refer

conf=0.02

| Car | AP@0.30 |    0.50 |    0.70 | AB(Byte) |
| :-- | ------: | ------: | ------: | -------: |
| bev | 72.5936 | 68.4444 | 56.7335 |  8565.79 |
| 3d  | 69.5802 |  58.921 | 30.4349 |  8565.79 |

conf=0.05

| Car | AP@0.30 |    0.50 |    0.70 | AB(Byte) |
| :-- | ------: | ------: | ------: | -------: |
| bev | 72.0086 | 68.1078 | 56.5984 |   3797.6 |
| 3d  |  69.158 | 58.7409 | 30.4154 |   3797.6 |
