# Transvision: ResilientV2X

本仓库提供 ResilientV2X 的可执行实现，用于研究协同车路三维检测在通信时延、LiDAR 分支缺失和 camera 分支缺失条件下的响应。当前主路径覆盖 DAIR-V2X 数据准备、因果协议生成、两阶段训练、受控基线、统一评测、证据封装以及可审计的 ClearML 运行控制。

## 发布状态

| 交付面 | 状态 | 权威入口 |
| :--- | :--- | :--- |
| T-ITS initial submission | **READY** | [ResilientV2X 论文仓库](https://github.com/lbin/ResilientV2X) |
| Public reproducibility release | **NOT READY** | 公开模型、数据制品和永久链接完成归档后更新 |

论文仓库负责稿件、结果注册表和投稿工件；本仓库负责模型、训练、评测与实验运行时。论文中的数值以论文仓库的 `results/registry.json` 及其生成宏为准。

## 已实现能力

- 严格校验 DAIR-V2X release、split、时间戳、标定和 PCD，并生成自哈希 temporal manifest。
- 先选择因果可达端点，再施加 transport/fault overlay，统一生成 Full、LiDAR 分支缺失和 camera 分支缺失条件。
- 四分支 LiDAR-camera 表征、horizon-conditioned PTF、可靠性加权聚合和动态专家路由。
- clean teacher 与 degraded-input student 的两阶段训练及特征/检测头蒸馏。
- PTF、路由、可靠性、No temporal metadata、蒸馏和容量匹配消融。
- CoFormerNet、FFNet、V2X-ViT、CoBEVT、BEVFusion 等统一协议受控适配配置。
- Car BEV/3D AP-R40 评测、prediction evidence、复杂度与 model-path latency 记录。
- 运行计划、环境、配置、checkpoint、评测输出和摘要之间的 SHA-256 绑定。

## 仓库结构

| 目录 | 职责 |
| :--- | :--- |
| `transvision/models/resilient_v2x/` | 因果修复、PTF、路由、蒸馏与受控基线模块 |
| `transvision/models/detectors/resilient_v2x.py` | ResilientV2X 检测器与训练/推理集成 |
| `transvision/dataset/resilient_v2x_*` | manifest、调度、运行时数据集与 PCD 处理 |
| `transvision/evaluation/resilient_v2x_*` | 检测、证据与部署测量合同 |
| `configs/resilient_v2x/` | 主模型、条件、消融和受控基线配置 |
| `tools/resilient_v2x/` | 数据准备、训练评测、证据封装与 ClearML 编排 |
| `tests/resilient_v2x/` | 模型、协议、证据和运行控制回归测试 |
| `environments/resilient_v2x/` | 锁定环境、容器和依赖约束 |

## 快速验证

锁定环境定义位于 `environments/resilient_v2x/`。构建容器并执行环境检查：

```bash
docker build \
  -f environments/resilient_v2x/Dockerfile \
  -t resilient-v2x:locked \
  .

docker run --rm --gpus all resilient-v2x:locked
```

在已安装开发依赖的环境中运行 ResilientV2X 测试：

```bash
python -m pytest -q tests/resilient_v2x
```

P1 ClearML 控制器的专项测试：

```bash
python -m pytest -q \
  tests/resilient_v2x/test_clearml_p1_*multiseed*.py \
  tests/resilient_v2x/test_clearml_p1_*parallel*.py \
  tests/resilient_v2x/test_clearml_p1_5090_retirement.py
```

数据与协议入口可先通过帮助页核对参数：

```bash
python tools/resilient_v2x/prepare_data.py --help
python tools/resilient_v2x/build_overlays.py --help
python tools/resilient_v2x/train_controlled_baseline.py --help
python tools/resilient_v2x/evaluate_controlled_baselines.py --help
python tools/resilient_v2x/build_evidence.py --help
```

完整的数据路径、哈希绑定、teacher/student 训练和十二条件评测命令见[受控复现手册](docs/resilient_v2x/reproduction.md)。

## DAIR-CAUSAL-1337-v1

当前论文协议固定为：

- DAIR-V2X validation 的 1,337 个样本和 11,330 个 Car ground-truth boxes；
- `0/100/200/300 ms` 名义附加时延；
- `Full/L-Fail/C-Fail` 三种输入条件，共 12 个 E+R 评测格子；
- 固定训练种子 `20250218` 和固定 epoch-50 checkpoint；
- BEV AP@0.5、BEV AP@0.7、3D AP@0.5、3D AP@0.7 四项指标。

五个论文基线是在同一数据、输入接口、优化和评测协议下构建的受控适配。它们用于协议内比较；主方法的端到端训练配方还包含 clean-teacher 蒸馏，因此各配置的训练合同随证据一并记录。

## 证据解释范围

- 当前结果是单一固定训练种子和固定 checkpoint 下的协议内验证估计。
- validation split 同时参与模型选择、配置确认和结果报告，数值按该用途范围解释。
- `100–300 ms` 表示协议注入的名义附加时延；数据准备中的 camera 重相位是独立的时序近似。
- reliability 与 routing 诊断量化规定故障下的分支响应；概率校准和机制因果效应对应独立的验证问题。
- 主要证据覆盖单帧故障。连续故障诊断显示 LiDAR 中断随持续时间增加会明显退化。
- profiler 报告从已整理 batch 的 `model.test_step` 到 decoded predictions 的 model-path latency，并保留测量范围与算子覆盖说明。

## P1 运行控制

以下 7 个文件构成同目录、按字节摘要互锁的运行控制链：

- `clearml_p1_multiseed_executor.py`
- `clearml_p1_5090_retirement.py`
- `clearml_p1_a100_multiseed_executor.py`
- `clearml_p1_a100_parallel_v2_supervisor.py`
- `clearml_p1_a100_parallel_v3_cachefix.py`
- `clearml_p1_a100_parallel_v4_parameter_compat.py`
- `clearml_p1_a100_parallel_v5_evaluation_contract_compat.py`

这些文件均位于 `tools/resilient_v2x/`。最新入口为：

```bash
python tools/resilient_v2x/clearml_p1_a100_parallel_v5_evaluation_contract_compat.py \
  --pretty
```

默认入口只生成本地密封计划；`--preflight` 用于只读核验远端状态；写入路径需要显式执行模式和对应授权令牌。各前驱文件共同提供计划、pinset、journal、运行时和兼容性证据，迁移或发布时按完整文件集合校验 SHA-256。

## 文档

- [配置与训练入口](configs/resilient_v2x/README.md)
- [受控复现手册](docs/resilient_v2x/reproduction.md)
- [论文—代码覆盖矩阵](docs/resilient_v2x/paper-coverage.md)
- [核心 API](docs/resilient_v2x/core-api.md)

## 上游项目

- [MMDetection3D v1.3.0](https://github.com/open-mmlab/mmdetection3d/tree/v1.3.0)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
- [FFNet-VIC3D](https://github.com/haibao-yu/FFNet-VIC3D)
