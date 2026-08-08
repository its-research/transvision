# Transvision

## 结果来源标签

| 标签 | 含义 |
| :--- | :--- |
| `论文原始结果` | 原论文表格或图中直接报告的数值；保持原指标、单位与协议 |
| `官方代码库结果` | 论文作者官方仓库报告的基准数值，不等同于论文原表 |
| `本仓 ClearML 复现` | 本仓任务日志可追溯的复现实测数值 |
| `本仓受控结果` | 本仓统一受控协议下的实测数值 |

## 本仓现有复现结果

| Car | Latency | 3D AP@0.50 | 3D AP@0.70 | BEV AP@0.50 | BEV AP@0.70 | 结果标签 |
| :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| FFNet-B-V | 0 ms | 51.60 | 29.99 | 56.62 | 49.15 | 官方代码库结果（TransVision benchmark） |
| FFNet-B-F | 0 ms | 55.48 | 31.54 | 63.15 | 54.27 | 官方代码库结果（TransVision benchmark） |
| FFNet-B-F（本次复现） | 0 ms | 65.16 | 39.60 | 71.27 | **62.44** | 本仓 ClearML 复现 |
| FFNet | 0 ms | 55.81 | 30.23 | **63.54** | 54.16 | 论文原始结果（FFNet Table 2） |
| FFNet | 200 ms | 55.37 | 31.66 | 63.20 | **54.69** | 论文原始结果（FFNet Table 2） |
| FFNet (w/o pred) | 200 ms | 50.27 | 27.57 | 57.93 | 48.16 | 论文原始结果（FFNet Table 2） |
| TF-L-V | 0 ms | 56.40 | 34.69 | 62.08 | 52.48 | 官方代码库结果（TransVision benchmark） |
| TF-L-F | 0 ms | **58.46** | **37.28** | 62.73 | 54.21 | 官方代码库结果（TransVision benchmark） |
| CoFormerNet | sync | 55.34 | 35.95 | 60.65 | 51.26 | 本仓 ClearML 复现 |

`conf=0.2`

FFNet-B-F 本次复现：ClearML [`9859bc7fbb694ca1b26f2a641711b4d2`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/9859bc7fbb694ca1b26f2a641711b4d2/output/log)，official 3-class complemented，40 epoch，final val（1789 samples）。

CoFormerNet 复现结果：ClearML [`2ea50800c8ff4bb3b0f87b4058d16ccc`](http://10.100.34.118:8080/projects/8fb6dbc7a09a4163961d4992f218ee26/experiments/2ea50800c8ff4bb3b0f87b4058d16ccc/output/log)（fusion formal eval），DAIR-V2X-C `vic-sync`，LiDAR-only，1789 个验证样本，评测范围 `[0,-46.08,-3,92.16,46.08,1]`。veh-only formal：3D@0.5/0.7 = 55.55/36.27，BEV@0.5/0.7 = 60.89/51.39。

- FF-B-V：FFNet Basemodel veh-only（re-implementation）
- FF-B-F：FFNet Basemodel fusion（re-implementation）
- TF-L-V：TransFusion-L veh-only

## ResilientV2X Results

受控证据：ClearML [`2992081bc95949f198e062c136810736`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/2992081bc95949f198e062c136810736/output/log)，seed `20250218`，student [`77afadda645f44748e1236eb91b5e664`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/77afadda645f44748e1236eb91b5e664/output/log)，dataset `fc242933c3ac43c2b47aaa3bd7f4a920`，1337 个验证样本，11330 个 Car ground truth，`content_sha256=5b3c9a9c77de823caafc7050317d99d2095ecfa5a005519d0d5d5f64da9af721`。

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
| Resilient V2X（seed `20250218`） | LiDAR + Camera | PointPillars + ResNet-50/LSS | 59.975¶ |

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
| Resilient V2X（seed `20250218`） | L + C | 59.975¶ | 57.936¶ | 57.868¶ | 57.829¶ | 3.6¶ |

### 单模态故障

结果格式：`BEV AP@0.5 / BEV AP@0.7（基于 AP@0.7 的 PDR）`。

| 方法 | Normal | L-Fail | C-Fail |
| :--- | :---: | :---: | :---: |
| V2X-ViT-style | —（待测） | —（待测） | —（待测） |
| CoBEVT-style | —（待测） | —（待测） | —（待测） |
| CoFormerNet-style | —（待测） | —（待测） | —（待测） |
| MIT-HAN BEVFusion-style | —（待测） | —（待测） | —（待测） |
| FFNet-style | —（待测） | —（待测） | —（待测） |
| Resilient V2X（seed `20250218`） | 68.955¶ / 59.975¶ | 69.319¶ / 48.915¶（↓18.4¶） | 68.940¶ / 60.005¶（↑0.1¶） |

### 模态故障与时延联合退化

主指标：BEV AP@0.7。

| 条件 | 0 ms | 100 ms | 200 ms | 300 ms |
| :--- | ---: | ---: | ---: | ---: |
| Full | 59.975¶ | 57.936¶ | 57.868¶ | 57.829¶ |
| L-Fail（E+R） | 48.915¶ | 49.018¶ | 49.106¶ | 44.636¶ |
| C-Fail（E+R） | 60.005¶ | 57.919¶ | 57.897¶ | 57.865¶ |

### 容量匹配消融

主指标：BEV AP@0.7。

| 变体 | Full | L-Fail | 300 ms |
| :--- | ---: | ---: | ---: |
| Full nonlinear PTF + DER | 59.975¶ | 48.915¶ | 57.829¶ |
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
| DAIR-V2X-C / Resilient V2X / Full / 0 ms | 68.955¶ | —（待测） | 59.975¶ | —（待测） | 65.372¶ | —（待测） | 35.142¶ | —（待测） | 1 / ≥3 |

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

## 论文与官方代码库原始结果

以下数值保持来源中的原始指标和单位，不跨数据集换算。`—` 表示来源未报告。

### 【论文原始结果】CoFormerNet（Sensors 2024）

来源：[CoFormerNet 原论文](https://doi.org/10.3390/s24134101)。

#### DAIR-V2X，原论文 Table 1

| 方法 | 融合类型 | 时延（ms） | 3D AP@0.5 | 3D AP@0.7 | BEV AP@0.5 | BEV AP@0.7 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| EMIFF（Camera） | Intermediate | 0 | 15.61 | — | 21.44 | — |
| PointPillars | Non-fusion | — | 48.06 | — | 52.24 | — |
| VoxelNet | Non-fusion | — | 52.40 | 34.69 | 58.08 | 49.18 |
| FFNet | Intermediate | 0 | 55.81 | 30.23 | 63.54 | 54.16 |
| TransIFF | Intermediate | 0 | 59.62 | 46.03 | — | — |
| CoFormerNet | Intermediate | 0 | 61.03 | 39.14 | 69.33 | 54.59 |
| Early Fusion | Early | 200 | 54.63 | 38.23 | 61.08 | 50.06 |
| Late Fusion | Late | 200 | 52.43 | 36.54 | 58.10 | 49.25 |
| FFNet | Intermediate | 200 | 55.37 | 31.66 | 63.20 | 54.69 |
| TransIFF | Intermediate | 200 | 53.47 | 37.21 | — | — |
| CoFormerNet | Intermediate | 200 | 60.97 | 38.97 | 69.13 | 54.65 |
| Early Fusion | Early | 300 | 51.37 | 37.25 | 58.28 | 49.81 |
| Late Fusion | Late | 300 | 51.35 | 36.24 | 56.89 | 48.79 |
| FFNet | Intermediate | 300 | 53.46 | 30.42 | 61.20 | 52.44 |
| TransIFF | Intermediate | 300 | 51.02 | 31.74 | — | — |
| CoFormerNet | Intermediate | 300 | 60.63 | 37.28 | 68.60 | 53.29 |

#### V2XSet，原论文 Table 2

| 方法 | 时延（ms） | 3D AP@0.5 | 3D AP@0.7 |
| :--- | ---: | ---: | ---: |
| V2X-ViT | 0 | 88.23 | 71.27 |
| FFNet | 0 | 89.47 | 72.66 |
| CoFormerNet | 0 | 90.23 | 72.95 |
| V2X-ViT | 200 | 83.61 | 61.49 |
| FFNet | 200 | 85.45 | 70.23 |
| CoFormerNet | 200 | 89.28 | 71.02 |
| V2X-ViT | 300 | 80.71 | 54.66 |
| FFNet | 300 | 83.31 | 57.93 |
| CoFormerNet | 300 | 84.22 | 59.01 |

#### 消融，原论文 Table 3

| TAM | SMCA | End2End | 时延（ms） | 3D AP@0.5 | 3D AP@0.7 | BEV AP@0.5 | BEV AP@0.7 |
| :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| × | × | × | 0 | 55.67 | 35.12 | 63.78 | 54.26 |
| ✓ | × | × | 0 | 58.25 | 36.27 | 67.84 | 54.31 |
| ✓ | ✓ | ✓ | 0 | 60.67 | 39.12 | 70.78 | 55.26 |
| × | × | × | 200 | 52.40 | 34.69 | 58.08 | 49.48 |
| × | ✓ | × | 200 | 55.40 | 34.44 | 63.14 | 52.32 |
| ✓ | × | × | 200 | 57.11 | 35.09 | 66.93 | 52.87 |
| ✓ | ✓ | × | 200 | 59.24 | 36.06 | 67.53 | 54.02 |
| ✓ | ✓ | ✓ | 200 | 60.97 | 38.97 | 69.13 | 54.65 |

#### 速度，原论文 Table 4

| 方法 | 推理时间 | 3D AP@0.5 |
| :--- | ---: | ---: |
| PointPillars | 31 ms | 48.06 |
| VoxelNet | 95 ms | 52.40 |
| TransIFF | 110 ms | 59.62 |
| FFNet | 101 ms | 55.81 |
| CoFormerNet | 122 ms | 61.03 |

### 【论文原始结果】FFNet（NeurIPS 2023）

来源：[FFNet 原论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/6ca5d2665de83394f437dad0c3746907-Paper-Conference.pdf)。DAIR-V2X、Car、范围 `[0,-39.12,100,39.12]`。

#### 融合方法比较，原论文 Table 1

| 方法 | 融合类型 | 时延（ms） | 3D AP@0.5 | 3D AP@0.7 | BEV AP@0.5 | BEV AP@0.7 | AB（Byte） |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| PointPillars | Non-fusion | — | 48.06 | — | 52.24 | — | 0 |
| AutoAlignV2 | Non-fusion | — | 50.32 | — | 53.88 | — | 0 |
| Early Fusion | Early | 200 | 54.63 | 38.23 | 61.08 | 50.06 | 1.4×10^6 |
| Late Fusion | Late | 200 | 52.43 | 36.54 | 58.10 | 49.25 | 5.1×10^2 |
| DiscoNet | Middle | 200 | 50.76 | 28.57 | 58.20 | 48.90 | 1.2×10^5 |
| V2VNet | Middle | 200 | 49.67 | 26.96 | 56.02 | 46.32 | 1.2×10^5 |
| FFNet | Middle | 200 | 55.37 | 31.66 | 63.20 | 54.69 | 1.2×10^5 |
| FFNet-C1 | Middle | 200 | 55.17 | 31.20 | 62.87 | 54.28 | 1.7×10^4 |
| Early Fusion | Early | 300 | 51.37 | 37.25 | 58.28 | 49.81 | 1.4×10^6 |
| Late Fusion | Late | 300 | 51.35 | 36.24 | 56.89 | 48.79 | 5.1×10^2 |
| DiscoNet | Middle | 300 | 49.03 | 27.39 | 55.81 | 47.28 | 1.2×10^5 |
| V2VNet | Middle | 300 | 48.51 | 27.00 | 55.81 | 46.32 | 1.2×10^5 |
| FFNet | Middle | 300 | 53.46 | 30.42 | 61.20 | 52.44 | 1.2×10^5 |
| FFNet-C1 | Middle | 300 | 54.10 | 29..87（原文排版） | 60.76 | 53.28 | 1.7×10^4 |

#### 特征预测消融，原论文 Table 2

| 方法 | 时延（ms） | 3D AP@0.5 | 3D AP@0.7 | BEV AP@0.5 | BEV AP@0.7 | AB（Byte） |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| FFNet | 0 | 55.81 | 30.23 | 63.54 | 54.16 | 1.2×10^5 |
| FFNet（without prediction） | 0 | 55.81 | 30.23 | 63.54 | 54.16 | 6.2×10^4 |
| FFNet-V2（without prediction） | 0 | 55.78 | 30.22 | 64.23 | 55.00 | 1.2×10^5 |
| FFNet | 200 | 55.37 | 31.66 | 63.20 | 54.69 | 1.2×10^5 |
| FFNet（without prediction） | 200 | 50.27 | 27.57 | 57.93 | 48.16 | 6.2×10^4 |
| FFNet-V2（without prediction） | 200 | 49.90 | 27.33 | 58.00 | 48.22 | 1.2×10^5 |

### 【论文原始结果】V2X-ViT（ECCV 2022）

来源：[V2X-ViT 原论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990106.pdf)。原文 AP 单位为 `0–1`；noisy 设置为 `0.2 m / 0.2° / 100 ms`。

#### V2XSet，原论文 Table 1

| 方法 | Perfect AP@0.5 | Perfect AP@0.7 | Noisy AP@0.5 | Noisy AP@0.7 |
| :--- | ---: | ---: | ---: | ---: |
| No Fusion | 0.606 | 0.402 | 0.606 | 0.402 |
| Late Fusion | 0.727 | 0.620 | 0.549 | 0.307 |
| Early Fusion | 0.819 | 0.710 | 0.720 | 0.384 |
| F-Cooper | 0.840 | 0.680 | 0.715 | 0.469 |
| OPV2V | 0.807 | 0.664 | 0.709 | 0.487 |
| V2VNet | 0.845 | 0.677 | 0.791 | 0.493 |
| DiscoNet | 0.844 | 0.695 | 0.798 | 0.541 |
| V2X-ViT | 0.882 | 0.712 | 0.836 | 0.614 |

#### 组件消融，原论文 Table 2

| Base | MSwin | SpAttn | HMSA | DPE | AP@0.5 | AP@0.7 |
| :---: | :---: | :---: | :---: | :---: | ---: | ---: |
| ✓ | × | × | × | × | 0.719 | 0.478 |
| ✓ | ✓ | × | × | × | 0.748 | 0.519 |
| ✓ | ✓ | ✓ | × | × | 0.786 | 0.548 |
| ✓ | ✓ | ✓ | ✓ | × | 0.823 | 0.601 |
| ✓ | ✓ | ✓ | ✓ | ✓ | 0.836 | 0.614 |

#### DPE 时延消融，原论文 Table 3

| 时延 | AP@0.7 without DPE | AP@0.7 with DPE |
| ---: | ---: | ---: |
| 100 ms | 0.639 | 0.650 |
| 200 ms | 0.558 | 0.572 |
| 300 ms | 0.496 | 0.514 |
| 400 ms | 0.458 | 0.478 |

#### 推理速度，原论文 Table 4

| 方法 | V100 推理时间 | Perfect AP@0.7 | Noisy AP@0.7 |
| :--- | ---: | ---: | ---: |
| V2X-ViT-S | 28 ms | 0.696 | 0.591 |
| V2X-ViT | 57 ms | 0.712 | 0.614 |

### 【论文原始结果】CoBEVT（CoRL 2022）

来源：[CoBEVT 原论文](https://proceedings.mlr.press/v205/xu23a/xu23a.pdf)。

#### OPV2V Camera Track，原论文 Table 1

| 方法 | Vehicle IoU | Drivable Area IoU | Lane IoU |
| :--- | ---: | ---: | ---: |
| No Fusion | 37.7 | 57.8 | 43.7 |
| Map Fusion | 45.1 | 60.0 | 44.1 |
| F-Cooper | 52.5 | 60.4 | 46.5 |
| AttFuse | 51.9 | 60.5 | 46.2 |
| V2VNet | 53.5 | 60.2 | 47.5 |
| DiscoNet | 52.9 | 60.7 | 45.8 |
| FuseBEVT | 59.0 | 62.1 | 49.2 |
| CoBEVT | 60.4 | 63.0 | 53.0 |

#### OPV2V LiDAR Track，原论文 Table 2

| 方法 | AP@0.7 | AP@0.7（64× 压缩） |
| :--- | ---: | ---: |
| No Fusion | 60.2 | 60.2 |
| Late Fusion | 78.1 | 78.1 |
| Early Fusion | 80.0 | — |
| F-Cooper | 79.0 | 78.8 |
| AttFuse | 81.5 | 81.0 |
| V2VNet | 82.2 | 81.4 |
| DiscoNet | 83.6 | 83.1 |
| FuseBEVT | 85.2 | 84.9 |

#### nuScenes 单车地图分割，原论文 Table 3

| 方法 | Vehicle IoU | 参数量（M） | FPS |
| :--- | ---: | ---: | ---: |
| VPN* | 29.3 | 4.0 | 31 |
| OFT | 30.1 | — | — |
| Lift-Splat | 32.1 | 14 | 25 |
| FIERY | 35.8 | 7 | 8 |
| CVT | 36.0 | 1.2 | 35 |
| SinBEVT | 37.1 | 1.6 | 35 |

#### 压缩，原论文 Table 4

| 压缩率 | 大小（KB） | Vehicle IoU |
| ---: | ---: | ---: |
| 0× | 524 | 60.4 |
| 8× | 66 | 60.1 |
| 16× | 33 | 58.9 |
| 32× | 16 | 56.2 |
| 64× | 8 | 54.8 |

#### FAX 组件消融，原论文 Table 5

| Local | Global | Vehicle IoU | Drivable Area IoU | Lane IoU |
| :---: | :---: | ---: | ---: | ---: |
| × | × | 52.6 | 57.9 | 42.0 |
| ✓ | × | 57.8 | 61.5 | 49.2 |
| × | ✓ | 57.9 | 60.8 | 48.6 |
| ✓ | ✓ | 60.4 | 63.0 | 53.0 |

### 【论文原始结果】MIT-HAN BEVFusion（ICRA 2023）

来源：[BEVFusion 原论文](https://arxiv.org/pdf/2205.13542)。

#### 3D 检测，原论文 Table I–II

| 数据集 / Split | 方法 | 模态 | mAP | NDS / mAPH | 其他 |
| :--- | :--- | :---: | ---: | ---: | :--- |
| nuScenes test | BEVFusion | C+L | 70.2 | 72.9 | 253.2 G MACs；119.2 ms |
| nuScenes val | BEVFusion | C+L | 68.5 | 71.4 | 253.2 G MACs；119.2 ms |
| Waymo test L1 | BEVFusion† | C+L | 85.7 | 84.4 | 3 frames |
| Waymo test L2 | BEVFusion† | C+L | 80.8 | 79.5 | 3 frames |

`†`：原论文使用 test-time augmentation。

#### nuScenes 地图分割，原论文 Table III

| 方法 | 模态 | Drivable | Ped. Cross. | Walkway | Stop Line | Carpark | Divider | Mean IoU |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BEVFusion | C | 81.7 | 54.8 | 58.4 | 47.4 | 50.7 | 46.4 | 56.6 |
| BEVFusion | C+L | 85.5 | 60.5 | 67.6 | 52.0 | 57.0 | 53.7 | 62.7 |

#### 天气与光照，原论文 Table IV

| 方法 | 模态 | Sunny mAP/mIoU | Rainy mAP/mIoU | Day mAP/mIoU | Night mAP/mIoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| BEVFusion | C | — / 59.0 | — / 50.5 | — / 57.4 | — / 30.8 |
| BEVFusion | C+L | 68.2 / 65.6 | 69.9 / 55.9 | 68.5 / 63.1 | 42.8 / 43.6 |

### 【官方代码库结果】MIT-HAN BEVFusion

来源：[MIT-HAN BEVFusion 官方仓库](https://github.com/mit-han-lab/bevfusion)。

| 数据集 / Split | 输入 | mAP | NDS / mIoU |
| :--- | :---: | ---: | ---: |
| nuScenes val | Camera | 35.56 | 41.21 NDS |
| nuScenes val | LiDAR | 64.68 | 69.28 NDS |
| nuScenes val | Camera + LiDAR | 68.52 | 71.38 NDS |
| nuScenes val map segmentation | Camera + LiDAR | — | 62.95 mIoU |
| nuScenes test | Camera + LiDAR | 70.23 | 72.88 NDS |
| nuScenes test，BEVFusion-e | Camera + LiDAR | 74.99 | 76.09 NDS |

### 【论文原始结果】ADLab BEVFusion（NeurIPS 2022）

来源：[ADLab BEVFusion 原论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/43d2b7fbee8431f7cef0d0afed51c691-Paper-Conference.pdf)。

#### 泛化能力，原论文 Table 1

每格为 `mAP / NDS`。

| 输入 | PointPillars | CenterPoint | TransFusion-L |
| :--- | :---: | :---: | :---: |
| Camera | 22.9 / 31.1 | 27.1 / 32.1 | 22.7 / 26.1 |
| LiDAR | 35.1 / 49.8 | 57.1 / 65.4 | 64.9 / 69.9 |
| Camera + LiDAR | 53.5 / 60.4 | 64.2 / 68.0 | 67.9 / 71.0 |

#### nuScenes 主结果，原论文 Table 2

| Split | 方法 | mAP | NDS |
| :--- | :--- | ---: | ---: |
| val | BEVFusion | 67.9 | 71.0 |
| val | BEVFusion* | 69.6 | 72.1 |
| test | BEVFusion | 69.2 | 71.8 |
| test | BEVFusion* | 71.3 | 73.3 |

`*`：原论文使用 BEV-space data augmentation 训练。

#### LiDAR 有限视场，原论文 Table 3

每格为 `LiDAR-only mAP/NDS → BEVFusion mAP/NDS`。

| LiDAR FOV | PointPillars | CenterPoint | TransFusion-L | TransFusion LC |
| :---: | :---: | :---: | :---: | :---: |
| ±π/2 | 12.4/37.1 → 36.8/45.8 | 23.6/48.0 → 45.5/54.9 | 27.8/50.5 → 46.4/55.8 | 31.1/49.2 |
| ±π/3 | 8.4/34.3 → 33.5/42.1 | 15.9/43.5 → 40.9/49.9 | 19.0/45.3 → 41.5/50.8 | 21.0/41.2 |

#### LiDAR 目标点丢失，原论文 Table 4

每格为 `LiDAR-only mAP/NDS → BEVFusion mAP/NDS`；增强训练行的 LiDAR-only 值未报告。

| Robust Aug. | PointPillars | CenterPoint | TransFusion-L | TransFusion LC |
| :---: | :---: | :---: | :---: | :---: |
| × | 12.7/36.6 → 34.3/49.1 | 31.3/50.7 → 40.2/54.3 | 34.6/53.6 → 40.8/56.0 | 38.1/55.4 |
| ✓ | — → 41.6/51.9 | — → 54.0/61.6 | — → 50.3/57.6 | 37.2/51.1 |

#### Camera 故障，原论文 Table 5

每格为 `mAP / NDS`。

| 方法 | Clean | Missing Front | Preserve Front Only | 50% Frames Stuck |
| :--- | :---: | :---: | :---: | :---: |
| DETR3D | 34.9 / 43.4 | 25.8 / 39.2 | 3.3 / 20.5 | 17.3 / 32.3 |
| PointAugmenting | 46.9 / 55.6 | 42.4 / 53.0 | 31.6 / 46.5 | 42.1 / 52.8 |
| MVX-Net | 61.0 / 66.1 | 47.8 / 59.4 | 17.5 / 41.7 | 48.3 / 58.8 |
| TransFusion | 66.9 / 70.9 | 65.3 / 70.1 | 64.4 / 69.3 | 65.9 / 70.2 |
| BEVFusion | 67.9 / 71.0 | 65.9 / 70.7 | 65.1 / 69.9 | 66.2 / 70.3 |

#### Camera Stream 消融，原论文 Table 6

| BE | ADP | Large Backbone | mAP | NDS |
| :---: | :---: | :---: | ---: | ---: |
| × | × | × | 13.9 | 24.5 |
| ✓ | × | × | 17.9 | 27.0 |
| ✓ | ✓ | × | 18.0 | 27.1 |
| ✓ | ✓ | ✓ | 22.9 | 31.1 |

#### Dynamic Fusion 消融，原论文 Table 7

| CSF | AFS | PointPillars mAP/NDS | CenterPoint mAP/NDS | TransFusion mAP/NDS |
| :---: | :---: | :---: | :---: | :---: |
| × | × | 35.1 / 49.8 | 57.1 / 65.4 | 64.9 / 69.9 |
| ✓ | × | 51.6 / 57.4 | 63.0 / 67.4 | 67.3 / 70.5 |
| ✓ | ✓ | 53.5 / 60.4 | 64.2 / 68.0 | 67.9 / 71.0 |

### 【论文原始结果】How2comm（NeurIPS 2023）

来源：[How2comm 原论文 Table 1](https://papers.neurips.cc/paper_files/paper/2023/file/4f31327e046913c7238d5b671f5d820e-Paper-Conference.pdf)。DAIR-V2X LiDAR-only；100 ms 传输时延；`0.2 m / 0.2°` 定位与航向噪声；通信量不超过 1 MB。

| 方法 | 3D AP@0.5 | 3D AP@0.7 |
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

### 【官方代码库结果】CoBEVT / OpenCOOD

来源：[OpenCOOD 官方结果](https://github.com/DerrickXuNu/OpenCOOD/tree/31ba16025da27ffe4e336f011290dfbc66f9a1f1#results-of-3d-detection-on-v2xset-lidar-track)。该表不是 CoBEVT 论文原表。

| 数据集 / 输入 | 设置 | AP@0.5 | AP@0.7 |
| :--- | :--- | ---: | ---: |
| V2XSet / LiDAR | Perfect | 84.9 | 66.0 |
| V2XSet / LiDAR | Noisy | 81.1 | 54.3 |

## Reproduction

- [ResilientV2X 配置](configs/resilient_v2x/README.md)
- [复现实验指南](docs/resilient_v2x/reproduction.md)
- [论文—代码覆盖矩阵](docs/resilient_v2x/paper-coverage.md)
- [旧稿结果归档](docs/resilient_v2x/archive/old-draft-results.md)

## Reference

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.3.0)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X), legacy Transvision reference commit: `c65a55617f7d0a9b78dc9d107370c95bcac55dca`
- [FFNet-VIC3D](https://github.com/haibao-yu/FFNet-VIC3D), audited commit: `52164dfe00764c9a9925539e99689cf25b88eace`
