# Transvision

## CoFormerNet Results

| Car | Latency | 3D AP@0.50 | 3D AP@0.70 | BEV AP@0.50 | BEV AP@0.70 |
| :--- | :---: | ---: | ---: | ---: | ---: |
| FFNet-B-V | 0 ms | 51.60 | 29.99 | 56.62 | 49.15 |
| FFNet-B-F | 0 ms | 55.48 | 31.54 | 63.15 | 54.27 |
| FFNet | 0 ms | 55.81 | 30.23 | **63.54** | 54.16 |
| FFNet | 200 ms | 55.37 | 31.66 | 63.20 | **54.69** |
| FFNet (w/o pred) | 200 ms | 50.27 | 27.57 | 57.93 | 48.16 |
| TF-L-V | 0 ms | 56.40 | 34.69 | 62.08 | 52.48 |
| TF-L-F | 0 ms | **58.46** | **37.28** | 62.73 | 54.21 |
| CoFormerNet | sync | 55.41 | 36.15 | 60.82 | 51.35 |

`conf=0.2`

CoFormerNet 复现结果：ClearML [`f7ec1cf7add24170b67e2ed807567c10`](http://10.100.34.118:8080/projects/8fb6dbc7a09a4163961d4992f218ee26/experiments/f7ec1cf7add24170b67e2ed807567c10/output/log)，DAIR-V2X-C `vic-sync`，LiDAR-only，1783 个验证样本，评测范围 `[0,-46.08,-3,92.16,46.08,1]`。

- FF-B-V：FFNet Basemodel veh-only（re-implementation）
- FF-B-F：FFNet Basemodel fusion（re-implementation）
- TF-L-V：TransFusion-L veh-only

## ResilientV2X Results

受控证据：ClearML [`63b337e0e31c4a08a6ef0007321f653a`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/63b337e0e31c4a08a6ef0007321f653a/output/log)，seed `20250218`，dataset `f24a09190df8449fa0192fb3fda6c25a`，1337 个验证样本，10583 个 Car ground truth，`content_sha256=1403ce66639a47ab9f622e268a9438f9e3bc9146a7eaf5506adda058bfc7a7e2`。

受控协议：DAIR-V2X-C、v2 manifest、官方 validation split、Car-only、LiDAR + Camera、评测范围 `[0,-40,-3,80,40,1]`。L-Fail 和 C-Fail 使用 `E+R`。AP 与 PDR 单位均为 `%`。

| 标记 | 状态 |
| :--- | :--- |
| `—（待测）` | 尚未完成 |
| `数值‡` | 公开论文参考值，协议不同 |
| `数值§` | 根据公开 AP 数值计算 |
| `数值¶` | 受控协议单随机种子实测值 |
| `数值 ± 标准差` | 至少 3 个随机种子的聚合结果 |
| `N/A` | 该条件不适用 |

### 正常输入比较

主指标：BEV AP@0.7。

| 方法 | 模态 | 骨干网络 | BEV AP@0.7 |
| :--- | :---: | :--- | ---: |
| V2X-ViT-style | LiDAR + Camera | PointPillars + ResNet-50/LSS | —（待测） |
| CoBEVT-style | LiDAR + Camera | PointPillars + ResNet-50/LSS | —（待测） |
| CoFormerNet-style | LiDAR + Camera | PointPillars + ResNet-50/LSS | —（待测） |
| MIT-HAN BEVFusion-style | LiDAR + Camera | PointPillars + ResNet-50/LSS | —（待测） |
| FFNet-style | LiDAR + Camera | PointPillars + ResNet-50/LSS | —（待测） |
| Resilient V2X（seed `20250218`） | LiDAR + Camera | PointPillars + ResNet-50/LSS | 0.165¶ |

### RSU 时延比较

主指标：BEV AP@0.7。

| 方法 | 模态 | 0 ms | 100 ms | 200 ms | 300 ms | PDR |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| V2X-ViT-style | L + C | —（待测） | —（待测） | —（待测） | —（待测） | —（待测） |
| CoBEVT-style | L + C | —（待测） | —（待测） | —（待测） | —（待测） | —（待测） |
| CoFormerNet-style | L + C | —（待测） | —（待测） | —（待测） | —（待测） | —（待测） |
| MIT-HAN BEVFusion-style | L + C | —（待测） | —（待测） | —（待测） | —（待测） | —（待测） |
| FFNet-style | L + C | —（待测） | —（待测） | —（待测） | —（待测） | —（待测） |
| [FFNet](https://proceedings.neurips.cc/paper_files/paper/2023/file/6ca5d2665de83394f437dad0c3746907-Paper-Conference.pdf) | LiDAR | 54.16‡ | N/A | 54.69‡ | 52.44‡ | 3.2§ |
| [CoFormerNet](https://doi.org/10.3390/s24134101) | LiDAR | 54.59‡ | N/A | 54.65‡ | 53.29‡ | 2.4§ |
| Resilient V2X（seed `20250218`） | L + C | 0.165¶ | 0.135¶ | 0.148¶ | 0.158¶ | 4.5¶ |

### 单模态故障

结果格式：`BEV AP@0.5 / BEV AP@0.7（基于 AP@0.7 的 PDR）`。

| 方法 | Normal | L-Fail | C-Fail |
| :--- | :---: | :---: | :---: |
| V2X-ViT-style | —（待测） | —（待测） | —（待测） |
| CoBEVT-style | —（待测） | —（待测） | —（待测） |
| CoFormerNet-style | —（待测） | —（待测） | —（待测） |
| MIT-HAN BEVFusion-style | —（待测） | —（待测） | —（待测） |
| FFNet-style | —（待测） | —（待测） | —（待测） |
| Resilient V2X（seed `20250218`） | 2.794¶ / 0.165¶ | 2.768¶ / 0.130¶（↓21.5¶） | 2.416¶ / 0.135¶（↓18.1¶） |

### 模态故障与时延联合退化

主指标：BEV AP@0.7。

| 条件 | 0 ms | 100 ms | 200 ms | 300 ms |
| :--- | ---: | ---: | ---: | ---: |
| Full | 0.165¶ | 0.135¶ | 0.148¶ | 0.158¶ |
| L-Fail（E+R） | 0.130¶ | 0.132¶ | 0.130¶ | 0.000¶ |
| C-Fail（E+R） | 0.135¶ | 0.122¶ | 0.091¶ | 0.000¶ |

### 容量匹配消融

主指标：BEV AP@0.7。

| 变体 | Full | L-Fail | 300 ms |
| :--- | ---: | ---: | ---: |
| Full nonlinear PTF + DER | 0.165¶ | 0.130¶ | 0.158¶ |
| No PTF | —（待测） | —（待测） | —（待测） |
| Linear PTF | —（待测） | —（待测） | —（待测） |
| Static three-expert | —（待测） | —（待测） | —（待测） |
| Uniform gate | —（待测） | —（待测） | —（待测） |
| No reliability | —（待测） | —（待测） | —（待测） |
| No delay metadata | —（待测） | —（待测） | —（待测） |
| No distillation | —（待测） | —（待测） | —（待测） |
| Capacity-matched concat | —（待测） | —（待测） | —（待测） |

### 连续故障持续时间

主指标：0 ms、`E+R` 下的 BEV AP@0.7。

| 持续时间 | BEV AP@0.7 | 支持状态 |
| ---: | ---: | :--- |
| 1 帧 | —（待测） | 历史窗口内 |
| 2 帧 | —（待测） | 历史窗口内 |
| 3 帧 | —（待测） | 历史窗口内 |
| 4 帧及以上 | —（待测） | neutral/unsupported |

### 部署复杂度

| 模型 | 参数量（M） | FLOPs（G） | 峰值 GPU 显存（GB） | 端到端时延（ms） |
| :--- | ---: | ---: | ---: | ---: |
| Capacity-matched concat | —（待测） | —（待测） | —（待测） | —（待测） |
| Resilient V2X student | —（待测） | —（待测） | —（待测） | —（待测） |

### 三随机种子汇总

| 数据集 / 条件 | BEV AP@0.5 mean | std | BEV AP@0.7 mean | std | 3D AP@0.5 mean | std | 3D AP@0.7 mean | std | 完成种子数 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DAIR-V2X-C / Resilient V2X / Full / 0 ms | —（待测） | —（待测） | —（待测） | —（待测） | —（待测） | —（待测） | —（待测） | —（待测） | 1 / ≥3 |

### 补充结果

| 记录 | 指标 | 0 ms | 100 ms | 200 ms | 300 ms |
| :--- | :--- | ---: | ---: | ---: | ---: |
| CoFormerNet public | BEV AP@0.5 | 69.33‡ | N/A | 69.13‡ | 68.60‡ |
| FFNet controlled baseline | BEV AP@0.7 | N/A | —（待测） | N/A | N/A |
| CoFormerNet controlled baseline | BEV AP@0.7 | N/A | —（待测） | N/A | N/A |

| 条件 | Agent scope | Resilient V2X − CoFormerNet-style BEV AP@0.7 |
| :--- | :---: | ---: |
| L-Fail | E+R | —（待测） |
| C-Fail | E+R | —（待测） |

### E-only / R-only 诊断

| 故障 | Agent scope | 0 ms | 100 ms | 200 ms | 300 ms |
| :--- | :---: | ---: | ---: | ---: | ---: |
| L-Fail | E-only | —（待测） | —（待测） | —（待测） | —（待测） |
| L-Fail | R-only | —（待测） | —（待测） | —（待测） | —（待测） |
| C-Fail | E-only | —（待测） | —（待测） | —（待测） | —（待测） |
| C-Fail | R-only | —（待测） | —（待测） | —（待测） | —（待测） |

### V2XSet

| 轨道 | Full 0 ms | Full 100 ms | Full 200 ms | Full 300 ms | L-Fail | C-Fail |
| :--- | ---: | ---: | ---: | ---: | :---: | :---: |
| V2XSet-Standard / LiDAR-only | —（待测） | —（待测） | —（待测） | —（待测） | N/A | N/A |
| V2XSet-Pair / LiDAR + Camera | —（待测） | —（待测） | —（待测） | —（待测） | —（待测） | —（待测） |

## Published Reference Results

### DAIR-V2X LiDAR-only 3D Detection

来源：[How2comm, NeurIPS 2023, Table 1](https://papers.neurips.cc/paper_files/paper/2023/file/4f31327e046913c7238d5b671f5d820e-Paper-Conference.pdf)。设置：100 ms 传输时延，`0.2 m / 0.2°` 定位与航向噪声，通信量不超过 1 MB。

| 方法 | 3D AP@0.5‡ | 3D AP@0.7‡ |
| :--- | ---: | ---: |
| No Fusion | 50.03 | 43.57 |
| Late Fusion | 48.93 | 34.06 |
| When2com | 46.64 | 32.49 |
| F-Cooper | 49.77 | 35.21 |
| AttFuse | 50.86 | 38.30 |
| V2VNet | 52.18 | 38.62 |
| DiscoNet | 51.44 | 40.01 |
| V2X-ViT | 51.68 | 39.97 |
| CoBEVT | 56.08 | 41.45 |
| Where2comm | 59.34 | 43.53 |
| How2comm | 62.36 | 47.18 |

### Other Published Results

| 方法 | 数据集 / 输入 | 设置 | 公开结果‡ |
| :--- | :--- | :--- | :--- |
| [V2X-ViT](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990106.pdf) | V2XSet / LiDAR | perfect；noisy | AP@0.5/0.7 = 88.2/71.2；83.6/61.4 |
| [CoBEVT / OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD/tree/31ba16025da27ffe4e336f011290dfbc66f9a1f1#results-of-3d-detection-on-v2xset-lidar-track) | V2XSet / LiDAR | perfect；noisy | AP@0.5/0.7 = 84.9/66.0；81.1/54.3 |
| [MIT-HAN BEVFusion](https://github.com/mit-han-lab/bevfusion) | nuScenes val / 单车 L+C | camera；LiDAR；L+C | mAP/NDS = 35.56/41.21；64.68/69.28；68.52/71.38 |
| [ADLab BEVFusion](https://proceedings.neurips.cc/paper_files/paper/2022/file/43d2b7fbee8431f7cef0d0afed51c691-Paper-Conference.pdf) | nuScenes val / 单车 L+C | clean；缺 front camera；50% camera frame stuck | mAP/NDS = 67.9/71.0；65.9/70.7；66.2/70.3 |

## Reproduction

- [ResilientV2X 配置](configs/resilient_v2x/README.md)
- [复现实验指南](docs/resilient_v2x/reproduction.md)
- [论文—代码覆盖矩阵](docs/resilient_v2x/paper-coverage.md)
- [旧稿结果归档](docs/resilient_v2x/archive/old-draft-results.md)

## Reference

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.3.0)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X), legacy Transvision reference commit: `c65a55617f7d0a9b78dc9d107370c95bcac55dca`
- [FFNet-VIC3D](https://github.com/haibao-yu/FFNet-VIC3D), audited commit: `52164dfe00764c9a9925539e99689cf25b88eace`
