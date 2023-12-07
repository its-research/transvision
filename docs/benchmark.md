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

### Downloaded Model

```shell
bash scripts/lidar_feature_fusion.sh 0 ./ # 1789 frames
# bash scripts/lidar_feature_flow.sh [YOUR_CUDA_DEVICE] [YOUR_FFNET_WORKDIR] [DELAY_K]
bash scripts/lidar_feature_flow.sh 0 1 'FlowPred' # 1501 frames
bash scripts/lidar_feature_flow.sh 0 2 'FlowPred'
```

| Car          | 3DAP@0.30 | 0.50  | 0.70  | AP@0.30 | 0.50  | 0.70  |
| :----------- | :-------: | :---: | :---: | :-----: | :---: | :---: |
| 0.17.1-base  |   63.52   | 54.93 | 29.29 |  65.70  | 62.95 | 53.45 |
| 0.17.1-100ms |   63.55   | 55.48 | 31.54 |  65.67  | 63.15 | 54.27 |
| 1.3.0-base   |   67.26   | 58.47 | 30.97 |  69.39  | 67.08 | 58.08 |
| 1.3.0-100ms  |   69.91   | 60.97 | 33.97 |  71.81  | 69.13 | 58.65 |
| 1.3.0-200ms  |   69.35   | 60.63 | 34.14 |  71.22  | 68.60 | 58.29 |

conf=0.2

### Comparison

| Car         | 3DAP@0.30 | 0.50  | 0.70  | BEVAP@0.30 | 0.50  | 0.70  |
| :---------- | :-------: | :---: | :---: | :--------: | :---: | :---: |
| FFNet-B-V   |   58.70   | 51.60 | 29.99 |   59.87    | 56.62 | 49.15 |
| FFNet-B-F   |   63.55   | 55.48 | 31.54 |   65.67    | 63.15 | 54.27 |
| FFNet-100ms |   69.91   | 60.97 | 33.97 |   71.81    | 69.13 | 58.65 |
| FFNet-200ms |   69.35   | 60.63 | 34.14 |   71.22    | 68.60 | 58.29 |
| TF-L-V      |   64.91   | 56.40 | 34.69 |   66.21    | 62.08 | 52.48 |
| CoFormerNet |           |       |       |            |       |       |

conf=0.2

- FF-B-V: FFNet Basemodel veh-only(our re-implementation)
- FF-B-F: FFNet Basemodel fusion(our re-implementation)
- FFNet-100ms: FFNet with latency: 100ms(our re-implementation)
- FFNet-200ms: FFNet with latency: 200ms(our re-implementation)
- TF-L-V: Transfusion-L veh-only
