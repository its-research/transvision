# Resilient V2X

**Temporally Valid Feature Repair and Reliability-Aware Routing for Collaborative 3D Detection**

Resilient V2X 面向车路协同三维检测中的两类联合退化：通信时延导致路侧特征过期，短时 LiDAR 或相机失效导致当前观测缺失。方法在决策时刻的信息可用性边界内选择最新有效历史特征，通过感知轨迹场（Perception Trajectory Field，PTF）完成时序修复，再由动态专家路由（Dynamic Expert Routing，DER）依据可用性、时延和可靠度融合 LiDAR、相机及跨模态专家。

本仓库维护模型、数据协议、训练、评测和证据封装实现。论文正文、结果注册表和配套材料位于相邻的独立论文仓库 `../ResilientV2X`。

## 论文摘要

现有车路协同感知方法通常分别处理通信异步和模态缺失。Resilient V2X 将两类退化放在同一决策时刻信息边界内：仅使用已经到达且元数据有效的观测；没有有效历史来源的分支输出零特征；有效历史特征先完成几何对齐，再由 PTF 预测残余运动并形成分支可靠度；DER 最后根据专家输出、模态可用性、分支来源时延、模态可靠度和路侧端点时延进行融合。

在 DAIR-V2X 固定验证子集和统一受控协议下，Resilient V2X 在全部 12 个“输入条件 × 通信时延”组合中取得最高的 BEV AP@0.7，并在完整输入及模态失效条件下保持稳定的严格指标表现。

## 主要贡献

1. 定义单车端、单路侧单元的协同三维检测问题，显式约束决策时刻可用信息、到达截止、故障掩码和无有效输入时的行为。
2. 以时域跨度为条件的 PTF 修复最新有效历史特征，并将轨迹置信度、来源时延和模态可用性纳入分支可靠度。
3. 以可靠度感知 DER 融合 LiDAR、相机和跨模态专家，并在统一协议下报告通信时延、模态失效、连续故障、消融、诊断和部署成本。

## 评测协议

论文结果采用 `DAIR-CAUSAL-1337-v1` 协议。所有受控方法共享样本、编码器、检测头、共享模块初始化、优化器、故障注入和评测器；各方法的融合模块不做参数量强制匹配，Resilient V2X 的完整训练配置包含完整输入教师模型蒸馏。

| 协议项 | 定义 |
| :--- | :--- |
| 数据集 | DAIR-V2X 车路协同数据 |
| 评测子集 | 1,337 个车端—路侧单元验证样本，11,330 个 Car 真值框 |
| 评测范围 | 当前车端坐标系，`[0, 80] × [-40, 40] × [-3, 1] m` |
| 名义附加路侧时延 | 无附加时延、100 ms、200 ms、300 ms |
| 输入条件 | `Full`、`L-Fail`、`C-Fail`；后两者同时移除车端与最新到达路侧端点的对应模态 |
| 评测网格 | 6 个方法 × 4 个时延 × 3 个输入条件 × 4 个指标，共 288 个受控结果单元 |
| 指标 | Car BEV/3D AP-R40，IoU 阈值为 0.5 和 0.7；主指标为 BEV AP@0.7 |
| 训练与检查点 | 固定种子 `20250218`；学生模型使用 epoch-50 最终检查点 |
| 证据状态 | 受控结果注册表状态为 `verified`，每个条件的 1,337 个样本均满足 `unsupported_sample_count=0` |

五个对比方法为 FFNet、CoFormerNet、V2X-ViT、CoBEVT 和 BEVFusion 的协议适配实现，不是其公开代码和原始配置的精确复现。以下结果仅支持统一协议内比较，不能与采用不同数据、模态、骨干网络或评测器的公开数值直接比较。

## 主要结果

### 完整输入、无附加时延

下表为 AP-R40（%）。粗体表示该列最高值。

| 方法 | BEV AP@0.5 | BEV AP@0.7 | 3D AP@0.5 | 3D AP@0.7 |
| :--- | ---: | ---: | ---: | ---: |
| FFNet-style | **71.94** | 59.24 | 66.70 | 37.51 |
| CoFormerNet-style | 69.96 | 58.70 | 64.41 | 35.03 |
| V2X-ViT-style | 69.91 | 58.67 | 64.41 | 34.19 |
| CoBEVT-style | 67.89 | 59.11 | 64.41 | 35.76 |
| BEVFusion-style | 70.38 | 59.75 | 65.07 | 37.80 |
| **Resilient V2X** | 70.93 | **62.36** | **67.63** | **40.57** |

### 通信时延与模态失效

每个单元格依次给出“Resilient V2X / 最强受控对比方法”的 BEV AP@0.7（%）。

| 名义附加路侧时延 | Full | L-Fail | C-Fail |
| ---: | ---: | ---: | ---: |
| 0 ms | **62.36** / 59.75 | **53.19** / 34.62 | **62.32** / 59.69 |
| 100 ms | **62.18** / 59.51 | **53.24** / 34.88 | **62.13** / 59.54 |
| 200 ms | **60.42** / 59.43 | **53.21** / 34.81 | **60.41** / 59.45 |
| 300 ms | **60.38** / 59.35 | **50.86** / 33.65 | **60.19** / 59.27 |

论文结果支持以下结论：

- Resilient V2X 在 12 个条件的 BEV AP@0.7 上均排名第一。
- 在 Full 条件下，Resilient V2X 在四个时延设置的 BEV AP@0.7 和 3D AP@0.7 上均排名第一；BEV AP@0.5 并非所有设置都领先。
- 在 L-Fail 和 C-Fail 条件下，Resilient V2X 在每个时延设置的四项指标上均排名第一。
- L-Fail 是更困难的故障条件。300 ms 时，受影响的 RSU 分支已无更早的受支持观测，BEV AP@0.7 仍为 50.86。

### 连续故障与部署成本

无附加时延下，LiDAR 连续失效会随持续时间显著退化，相机连续失效在本协议内变化较小。

| 连续失效时长 | LiDAR 失效 BEV AP@0.7 | 相机失效 BEV AP@0.7 |
| :--- | ---: | ---: |
| 1 帧 | 53.19 | 62.32 |
| 2 帧 | 11.57 | 62.35 |
| 3 帧 | 5.27 | 62.37 |

部署测量不包含教师模型和蒸馏分支；延迟为 RTX 5090 上 10 次预热后 100 次同步运行的中位数，范围从完成拼批的 `model.test_step` 输入开始，到输出解码后的预测结果结束。

| 模型 | 参数量 | `torch.profiler` 支持的 FLOPs | 峰值显存 | 模型路径延迟 |
| :--- | ---: | ---: | ---: | ---: |
| 容量匹配拼接对照 | 35.076 M | 128.744 G | 0.833 GB | 52.341 ms |
| Resilient V2X 学生模型 | 35.076 M | 128.745 G | 0.833 GB | 53.534 ms |

### 消融结论

- 零偏移 PTF 保留历史来源选择、几何对齐和时延衰减，但 L-Fail BEV AP@0.7 从 53.19 降至 32.81，是该组消融中最大的下降。
- 移除蒸馏后，Full 0 ms、L-Fail 0 ms 和 Full 300 ms 的 BEV AP@0.7 分别由 62.36、53.19、60.38 降至 60.34、50.84、58.15。
- 完整模型并非在每个次要设置中都最高：线性 PTF 和二值支持机制在部分 Full 条件取得更高单点结果，完整模型则在 L-Fail 条件保持优势。
- 可靠度与路由诊断显示，故障模态的平均可靠度会下降；这些诊断用于描述模型响应，不等同于概率校准或可靠度对路由的独立因果效应。

## 论文与代码对应关系

| 论文内容 | 实现入口 |
| :--- | :--- |
| Resilient V2X 检测器 | `transvision/models/detectors/resilient_v2x.py` |
| PTF、可靠度和 DER | `transvision/models/resilient_v2x/` |
| 时序 manifest、调度和运行时数据集 | `transvision/dataset/resilient_v2x_*` |
| 受控条件与统一评测 | `transvision/evaluation/resilient_v2x_*`、`tools/resilient_v2x/evaluate_controlled_baselines.py` |
| 主模型、消融和协议适配方法 | `configs/resilient_v2x/` |
| 数据准备与条件覆盖配置 | `tools/resilient_v2x/prepare_data.py`、`tools/resilient_v2x/build_overlays.py` |
| 训练 | `tools/train.py`、`configs/resilient_v2x/dair_clean_teacher.py`、`configs/resilient_v2x/dair_resilient_v2x.py` |
| 性能测量与证据封装 | `tools/resilient_v2x/profile.py`、`tools/resilient_v2x/build_evidence.py` |

## 复现入口

锁定环境基于 Linux/amd64、CUDA 11.8、Python 3.10、PyTorch 2.0.1、MMEngine 0.10.7、MMCV 2.1.0、MMDetection 3.2.0 和 MMDetection3D 1.3.0。推荐使用仓库提供的容器：

```bash
docker build \
  -f environments/resilient_v2x/Dockerfile \
  -t resilient-v2x:locked \
  .

docker run --rm --gpus all resilient-v2x:locked
```

在已安装开发依赖的环境中运行回归测试：

```bash
python -m pytest -q tests/resilient_v2x
```

原始 DAIR 数据、`artifacts/`、`models/` 和 `work_dirs/` 均为本地或外部制品，不包含在 Git 仓库中。数据准备、两阶段训练、12 条件评测、性能测量和证据封装分别由 `tools/resilient_v2x/` 下的对应命令完成。

## 文档

- [配置与训练入口](configs/resilient_v2x/README.md)
- [核心 API](docs/resilient_v2x/core-api.md)
- 论文正文、结果注册表、PDF、源码包和引用元数据：`../ResilientV2X`

## 上游项目

- [MMDetection3D v1.3.0](https://github.com/open-mmlab/mmdetection3d/tree/v1.3.0)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
- [FFNet-VIC3D](https://github.com/haibao-yu/FFNet-VIC3D)
