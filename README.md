# Transvision

## News

## CoFormerNet

### Results

| Car             | Latency | 3DAP@0.50 |   0.70    | BEVAP@0.50 |   0.70    |
| :-------------- | :-----: | :-------: | :-------: | :--------: | :-------: |
| FFNet-B-V       |   0ms   |   51.60   |   29.99   |   56.62    |   49.15   |
| FFNet-B-F       |   0ms   |   55.48   |   31.54   |   63.15    |   54.27   |
| FFNet           |   0ms   |   55.81   |   30.23   | __63.54__  |   54.16   |
| FFNet           |  200ms  |   55.37   |   31.66   |   63.20    | __54.69__ |
| FFNet(w/o pred) |  200ms  |   50.27   |   27.57   |   57.93    |   48.16   |
| TF-L-V          |   0ms   |   56.40   |   34.69   |   62.08    |   52.48   |
| TF-L-F          |   0ms   | __58.46__ | __37.28__ |   62.73    |   54.21   |
| CoFormerNet     |  sync   |   55.41   |   36.15   |   60.82    |   51.35   |

conf=0.2

CoFormerNet 行来自已完成的 8×RTX5090、160 epoch ClearML 复现任务
[`f7ec1cf7add24170b67e2ed807567c10`](http://10.100.34.118:8080/projects/8fb6dbc7a09a4163961d4992f218ee26/experiments/f7ec1cf7add24170b67e2ed807567c10/output/log)：
DAIR-V2X-C `vic-sync`、LiDAR-only、1783 个验证样本、外部范围
`[0,-46.08,-3,92.16,46.08,1]`。它与下方 ResilientV2X 的 1337 样本、
LiDAR+Camera、因果时延/故障协议不兼容，不能用于受控排名。

- FF-B-V: FFNet Basemodel veh-only(our re-implementation)
- FF-B-F: FFNet Basemodel fusion(our re-implementation)
- TF-L-V: Transfusion-L veh-only

## ResilientV2X 论文实验结果表

本节完整列出论文实验章节要求的结果表。当前已有一个完整的 12 条件
单随机种子受控证据包通过内容哈希校验，但仍**没有满足提交条件的聚合
结果**：还缺至少两个独立随机种子，五个同协议适配基线及容量匹配消融也
尚未完成。因此，单种子数值以 `¶` 标记，只用于诊断和进度记录，不能写成
`均值 ± 标准差`，也不能支持“本文方法优于其他方法”的论文结论。固定
submission registry 尚未回填，仍为 `verified=0`。

本轮受控证据来自 4×RTX5090 ClearML 任务
[`63b337e0e31c4a08a6ef0007321f653a`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/63b337e0e31c4a08a6ef0007321f653a/output/log)，
随机种子 `20250218`，数据集 `f24a09190df8449fa0192fb3fda6c25a`，
1337 个验证样本、10583 个 Car ground truth；完整汇总
`content_sha256=1403ce66639a47ab9f622e268a9438f9e3bc9146a7eaf5506adda058bfc7a7e2`。
下表中的 `¶` 值均来自该汇总，而不是控制台中间日志。

**当前比较结论：尚未证明本文方法优于其他方法。** 同协议的五个受控适配
基线还没有完成；仅作数值诊断时，本文 0 ms Full 的 BEV AP@0.7 为
`0.165¶`，而协议不兼容的 CoFormerNet `vic-sync` 复现为 `51.35`。两者
不能作正式差值或排名，但当前数量级显然不支持领先主张，并提示应先排查
训练有效性、checkpoint 质量或评测几何的一致性。

上方原有的 CoFormerNet 表是 Transvision 项目的历史说明，不属于本节 ResilientV2X 受控实验的证据。

| 标记 | 含义 | 能否作为当前论文的受控结论 |
| :--- | :--- | :---: |
| `—（待测）` | 尚无合格测量；最终应填至少 3 个随机种子的均值 ± 标准差 | 否 |
| `数值†` | 旧论文草稿中保留的数值，没有配置、检查点或日志证据，协议未说明 | 否 |
| `数值‡` | 引用公开论文的参考数值，模态、骨干或评测协议与本论文不兼容 | 否 |
| `数值§` | 由同一行公开 AP 数值按本文 PDR 公式计算，并非来源论文直接报告 | 否 |
| `数值¶` | 同一受控协议下通过证据哈希校验的单随机种子 ClearML 实测值 | 否；需补齐 ≥3 seeds 和同协议基线 |
| `数值 ± 标准差` | 仅在协议兼容、至少 3 个随机种子且证据包验证通过后使用 | 是 |
| `N/A` | 该实验轨道不定义此条件，不能用待测值代替 | 不适用 |

除非表内另有说明，AP 与 PDR 的单位均为 `%`；受控轨道固定使用 v2
manifest、官方 validation split、同一个 1337-sample validation cohort、
Car-only 目标与 `[0,-40,-3,80,40,1]` 三维评测范围。受控主故障协议中的 L-Fail 和 C-Fail 均为 `E+R`，先进行因果时延端点选择，再在原始输入上
施加故障。PDR 按 `(正常值 - 退化值) / 正常值 × 100%` 计算，不能跨协议
或跨 agent scope 计算。

### 公开论文的外部对照（不可回填受控主表）

[How2comm 的 NeurIPS 2023 论文表 1](https://papers.neurips.cc/paper_files/paper/2023/file/4f31327e046913c7238d5b671f5d820e-Paper-Conference.pdf)
在统一的 LiDAR-only 3D 检测设置下给出了较完整的 DAIR-V2X
横向比较。该表默认包含 `100 ms` 传输时延、标准差为 `0.2 m /
0.2°` 的定位/航向噪声，并把通信量约束在 `1 MB`；因此下列数值只能
作为外部参考，不能填入本文的 LiDAR + Camera、BEV AP、0 ms 或
E+R 模态故障单元格。

| 方法 | 输入 | 外部设置 | 3D AP@0.5‡ | 3D AP@0.7‡ |
| :--- | :---: | :--- | ---: | ---: |
| No Fusion | LiDAR | How2comm / DAIR-V2X 默认噪声 | 50.03 | 43.57 |
| Late Fusion | LiDAR | How2comm / DAIR-V2X 默认噪声 | 48.93 | 34.06 |
| When2com | LiDAR | How2comm / DAIR-V2X 默认噪声 | 46.64 | 32.49 |
| F-Cooper | LiDAR | How2comm / DAIR-V2X 默认噪声 | 49.77 | 35.21 |
| AttFuse | LiDAR | How2comm / DAIR-V2X 默认噪声 | 50.86 | 38.30 |
| V2VNet | LiDAR | How2comm / DAIR-V2X 默认噪声 | 52.18 | 38.62 |
| DiscoNet | LiDAR | How2comm / DAIR-V2X 默认噪声 | 51.44 | 40.01 |
| V2X-ViT | LiDAR | How2comm / DAIR-V2X 默认噪声 | 51.68 | 39.97 |
| CoBEVT | LiDAR | How2comm / DAIR-V2X 默认噪声 | 56.08 | 41.45 |
| Where2comm | LiDAR | How2comm / DAIR-V2X 默认噪声 | 59.34 | 43.53 |
| How2comm | LiDAR | How2comm / DAIR-V2X 默认噪声 | 62.36 | 47.18 |

原论文/官方实现还提供以下数值，但数据集、任务或指标与本文主表不同：

| 方法与来源 | 数据集 / 输入 | 设置 | 公开结果‡ | 与主表的关键差异 |
| :--- | :--- | :--- | :--- | :--- |
| [V2X-ViT](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990106.pdf) | V2XSet / LiDAR | perfect；noisy | AP@0.5/0.7 = 88.2/71.2；83.6/61.4 | 模拟数据、LiDAR-only；noisy 含时延与位姿噪声 |
| [CoBEVT / OpenCOOD benchmark](https://github.com/DerrickXuNu/OpenCOOD/tree/31ba16025da27ffe4e336f011290dfbc66f9a1f1#results-of-3d-detection-on-v2xset-lidar-track) | V2XSet / LiDAR | perfect；noisy | AP@0.5/0.7 = 84.9/66.0；81.1/54.3 | 模拟数据、LiDAR-only；noisy 配方允许 200 或 300 ms async overhead；[CoBEVT 原论文](https://proceedings.mlr.press/v205/xu23a.html)主任务是多相机 BEV 分割 |
| [MIT-HAN BEVFusion](https://github.com/mit-han-lab/bevfusion) | nuScenes val / 单车 L+C | camera；LiDAR；L+C | mAP/NDS = 35.56/41.21；64.68/69.28；68.52/71.38 | 单车 nuScenes，使用 mAP/NDS，无协同时延 |
| [ADLab BEVFusion](https://proceedings.neurips.cc/paper_files/paper/2022/file/43d2b7fbee8431f7cef0d0afed51c691-Paper-Conference.pdf) | nuScenes val / 单车 L+C | clean；缺 front camera；50% camera frame stuck | mAP/NDS = 67.9/71.0；65.9/70.7；66.2/70.3 | 局部相机故障，不是 DAIR E+R 全模态失效 |

这里的两个 BEVFusion 是不同论文。本文的受控适配明确采用
MIT-HAN ICRA 2023 的 BEV 级多传感器融合思想；ADLab NeurIPS 2022
的鲁棒性数值只用于说明已有故障定义，不能与该实现混用。CoFormerNet
公开时延数值见表 2；其论文表 1 与表 3 的 0 ms 完整模型数值存在
内部不一致，本文固定引用包含完整 `0/200/300 ms` 序列的表 1。

### 受控适配实现状态

公开论文没有提供可直接运行于本文 DAIR-V2X-C、四分支 L+C、严格因果
故障/时延协议的完整实现，所以仓库为五个主对比方法加入了独立的受控
适配。它们统一经过 `ResilientTemporalDataset` 的因果端点选择和纯几何
对齐，固定分支顺序为 `L_E, L_R, C_E, C_R`，并共享 PointPillars、
ResNet-50/LSS、Anchor3DHead、训练 overlay、随机种子和 evaluator。适配
模型不包含本文方法的 PTF、repair、DER、teacher 或 distillation 路径。

| 方法 | 本仓融合实现 | 受控配置 | 方法特有结构 | 当前状态 |
| :--- | :--- | :--- | :--- | :--- |
| V2X-ViT-style | [`baselines.py`](transvision/models/resilient_v2x/baselines.py) | [`v2x_vit.py`](configs/resilient_v2x/baselines/v2x_vit.py) | heterogeneous branch attention、relative-age encoding、multi-scale window attention | 已实现并通过单测；正式训练待排队 |
| CoBEVT-style | [`baselines.py`](transvision/models/resilient_v2x/baselines.py) | [`cobevt.py`](configs/resilient_v2x/baselines/cobevt.py) | masked branch attention、factorized axial attention | 已实现并通过单测；正式训练待排队 |
| CoFormerNet-style | [`baselines.py`](transvision/models/resilient_v2x/baselines.py) | [`coformernet.py`](configs/resilient_v2x/baselines/coformernet.py) | local spatial cross-attention、relative-temporal cross-attention | 已实现并通过单测；正式训练待排队 |
| MIT-HAN BEVFusion-style | [`baselines.py`](transvision/models/resilient_v2x/baselines.py) | [`bevfusion.py`](configs/resilient_v2x/baselines/bevfusion.py) | support/age-conditioned cooperative ConvFuser | 已实现并通过单测；正式训练待排队 |
| FFNet-style | [`baselines.py`](transvision/models/resilient_v2x/baselines.py) | [`ffnet.py`](configs/resilient_v2x/baselines/ffnet.py) | delayed-RSU single-frame feature-flow/一阶残差预测、concat-conv fusion | 已实现并通过单测；正式训练待排队 |

这些名称均带 `-style`：它们是为了控制 dataset、modality、backbone、head
和故障/时延协议而实现的 L+C adaptations，不冒充原论文官方系统的
bit-exact reproduction。融合输入契约、detector 和配置说明见
[`baseline_inputs.py`](transvision/models/resilient_v2x/baseline_inputs.py)、
[`controlled_v2x_baseline.py`](transvision/models/detectors/controlled_v2x_baseline.py)
与[配置指南](configs/resilient_v2x/README.md)。
统一执行入口为
[`train_controlled_baseline.py`](tools/resilient_v2x/train_controlled_baseline.py)
和
[`evaluate_controlled_baselines.py`](tools/resilient_v2x/evaluate_controlled_baselines.py)；
两者都把数据、split、manifest 与 overlay 身份写入可审计计划，不能用一个
方法的 checkpoint 填另一个方法的单元格。

### 表 1：正常输入比较（`tab:normal_comparison`）

主指标：DAIR-V2X-C，BEV AP@0.7。

| 方法 | 来源 | 模态 | 骨干网络 | 协议 | BEV AP@0.7 |
| :--- | :--- | :---: | :--- | :--- | ---: |
| V2X-ViT-style | 本仓受控适配已实现 | LiDAR + Camera | PointPillars + ResNet-50/LSS | v2 共享协议；待训练 | —（待测） |
| CoBEVT-style | 本仓受控适配已实现 | LiDAR + Camera | PointPillars + ResNet-50/LSS | v2 共享协议；待训练 | —（待测） |
| CoFormerNet-style | 本仓受控适配已实现 | LiDAR + Camera | PointPillars + ResNet-50/LSS | v2 共享协议；待训练 | —（待测） |
| MIT-HAN BEVFusion-style | 本仓受控适配已实现 | LiDAR + Camera | PointPillars + ResNet-50/LSS | v2 共享协议；待训练 | —（待测） |
| FFNet-style | 本仓受控适配已实现 | LiDAR + Camera | PointPillars + ResNet-50/LSS | v2 单帧适配；待训练 | —（待测） |
| Resilient V2X（受控单种子） | ClearML `63b337e0...` | LiDAR + Camera | PointPillars + ResNet-50/LSS | v2 共享协议；seed `20250218` | 0.165¶ |
| Resilient V2X | 旧稿方法记录 | LiDAR + Camera | PointPillars + ResNet-50/LSS | 旧稿协议未说明 | 58.1† |

### 表 2：RSU 时延比较（`tab:latency`）

主指标：BEV AP@0.7。FFNet 与 CoFormerNet 是公开 LiDAR-only 参考，不与多模态受控结果排名。

| 方法 | 来源 | 模态 | 骨干网络 | 协议 | 0 ms | 200 ms | 300 ms | PDR |
| :--- | :--- | :---: | :--- | :--- | ---: | ---: | ---: | ---: |
| V2X-ViT-style | 本仓受控适配已实现 | L + C | PointPillars + ResNet-50/LSS | v2 共享协议 | —（待测） | —（待测） | —（待测） | —（待测） |
| CoBEVT-style | 本仓受控适配已实现 | L + C | PointPillars + ResNet-50/LSS | v2 共享协议 | —（待测） | —（待测） | —（待测） | —（待测） |
| CoFormerNet-style | 本仓受控适配已实现 | L + C | PointPillars + ResNet-50/LSS | v2 共享协议 | —（待测） | —（待测） | —（待测） | —（待测） |
| MIT-HAN BEVFusion-style | 本仓受控适配已实现 | L + C | PointPillars + ResNet-50/LSS | v2 共享协议 | —（待测） | —（待测） | —（待测） | —（待测） |
| FFNet-style | 本仓受控适配已实现 | L + C | PointPillars + ResNet-50/LSS | v2 单帧适配 | —（待测） | —（待测） | —（待测） | —（待测） |
| FFNet | [公开论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/6ca5d2665de83394f437dad0c3746907-Paper-Conference.pdf) | LiDAR | PFN + SECOND/FPN | DAIR val；Car；外部 XY ROI `[0,-39.12,100,39.12]`；“0 ms”仍含原始 `[-30,30] ms` 配对偏差，200/300 ms 由替换历史 RSU 帧模拟 | 54.16‡ | 54.69‡ | 52.44‡ | 3.2§ |
| CoFormerNet | [公开论文](https://doi.org/10.3390/s24134101) | LiDAR | VoxelNet | DAIR val；Car 口径含 bus/truck/van；外部 XY ROI `[0,-39.12,100,39.12]` | 54.59‡ | 54.65‡ | 53.29‡ | 2.4§ |
| Resilient V2X（受控单种子） | ClearML `63b337e0...` | L + C | PointPillars + ResNet-50/LSS | v2 共享协议；seed `20250218` | 0.165¶ | 0.148¶ | 0.158¶ | 4.5¶ |
| Resilient V2X | 旧稿方法记录 | LiDAR + Camera | PointPillars + ResNet-50/LSS | 旧稿协议未说明 | 58.1† | 57.7† | 57.3† | 1.4† |

论文主时延矩阵中的 100 ms 受控点见表 4；公开来源未报告兼容的 100 ms 多模态结果，不能插值补齐。

### 表 3：单模态故障（`tab:modality_missing`）

每个结果单元格格式为 `BEV AP@0.5 / BEV AP@0.7（基于 AP@0.7 的 PDR）`。计划中的受控 L-Fail/C-Fail 使用 `E+R`；旧稿的故障范围未知，不能追溯标成 `E+R`。

| 方法 | 来源 | 模态 | Normal | L-Fail | C-Fail |
| :--- | :--- | :---: | :---: | :---: | :---: |
| V2X-ViT-style | 本仓受控适配已实现 | L + C | —（待测） | —（待测） | —（待测） |
| CoBEVT-style | 本仓受控适配已实现 | L + C | —（待测） | —（待测） | —（待测） |
| CoFormerNet-style | 本仓受控适配已实现 | L + C | —（待测） | —（待测） | —（待测） |
| MIT-HAN BEVFusion-style | 本仓受控适配已实现 | L + C | —（待测） | —（待测） | —（待测） |
| FFNet-style | 本仓受控适配已实现 | L + C | —（待测） | —（待测） | —（待测） |
| Resilient V2X（受控单种子） | ClearML `63b337e0...` | L + C | 2.794¶ / 0.165¶ | 2.768¶ / 0.130¶（↓21.5¶） | 2.416¶ / 0.135¶（↓18.1¶） |
| Resilient V2X | 旧稿方法记录 | L + C | 71.5† / 58.1† | 45.8† / 28.6†（↓50.8†） | 68.7† / 54.8†（↓5.7†） |

### 表 4：模态故障与时延联合退化（`tab:joint_degradation`）

主指标：BEV AP@0.7。受控 L-Fail/C-Fail 均使用 `E+R`。`¶` 是同一
checkpoint、cohort 和 evaluator 下的单种子实测值；括号中的 `†` 只保留
旧稿锚点，不能用于验证或替代受控结果。

| 条件 | 0 ms | 100 ms | 200 ms | 300 ms |
| :--- | ---: | ---: | ---: | ---: |
| Full | 0.165¶（旧稿 58.1†） | 0.135¶ | 0.148¶（旧稿 57.7†） | 0.158¶（旧稿 57.3†） |
| L-Fail（受控定义为 E+R） | 0.130¶（旧稿 28.6†，scope 未知） | 0.132¶ | 0.130¶ | 0.000¶ |
| C-Fail（受控定义为 E+R） | 0.135¶（旧稿 54.8†，scope 未知） | 0.122¶ | 0.091¶ | 0.000¶ |

### 表 5：容量匹配消融（`tab:ablation`）

主指标：BEV AP@0.7。Panel A 中的 Full 与 300 ms 是完整输入条件，L-Fail 使用 0 ms、`E+R`；所有变体必须共享数据清单、检查点选择规则、随机种子和评测器。Panel B 的旧稿故障 scope 未知，不能追溯解释为 `E+R`。

#### Panel A：受控容量匹配消融

| 变体 | 唯一受控变化 | Full | L-Fail | 300 ms |
| :--- | :--- | ---: | ---: | ---: |
| Full nonlinear PTF + DER | 容量匹配的受控参考配置 | 0.165¶ | 0.130¶ | 0.158¶ |
| No PTF | 移除时序传播，保留容量匹配路由 | —（待测） | —（待测） | —（待测） |
| Linear PTF | 将非线性时域条件场替换为定义好的线性传播规则 | —（待测） | —（待测） | —（待测） |
| Static three-expert | 将动态路由替换为容量匹配的静态三专家融合 | —（待测） | —（待测） | —（待测） |
| Uniform gate | 将学习到的 DER 权重替换为均匀权重 | —（待测） | —（待测） | —（待测） |
| No reliability | 从路由输入中移除可靠性 | —（待测） | —（待测） | —（待测） |
| No delay metadata | 从路由输入中移除时延和特征年龄元数据 | —（待测） | —（待测） | —（待测） |
| No distillation | 不使用特征及输出分布一致性蒸馏训练学生模型 | —（待测） | —（待测） | —（待测） |

#### Panel B：旧稿模块消融记录（不保证容量匹配）

| Exp. | 旧稿描述 | Full | L-Fail | 300 ms |
| :---: | :--- | ---: | ---: | ---: |
| A | Concat fusion；无 PTF、无蒸馏 | 53.0† | 10.1† | 32.5† |
| B | DER；无 PTF、无蒸馏 | 54.7† | 24.3† | 33.2† |
| C | DER + 线性传播；无蒸馏 | 56.3† | 25.9† | 50.0† |
| D | DER + trajectory-field propagation；无蒸馏 | 57.0† | 27.5† | 55.0† |
| E | DER + trajectory-field propagation + consistency distillation | 58.1† | 28.6† | 57.3† |

### 表 6：训练故障概率敏感性（`tab:sensitivity_p`）

主指标：BEV AP@0.7。以下均为旧稿记录，尚无因果协议、容量匹配或多随机种子证据；不能据此宣称最佳概率。`p=0.3` 行复用旧稿方法主结果，而不是一条独立测量。

| `p_L = p_C` | Full | L-Fail | 300 ms |
| ---: | ---: | ---: | ---: |
| 0.0 | 57.4† | 24.6† | 53.1† |
| 0.1 | 57.7† | 26.5† | 54.8† |
| 0.3 | 58.1† | 28.6† | 57.3† |
| 0.5 | 56.9† | 28.8† | 56.9† |

### 表 7：连续故障与部署复杂度（`tab:duration_complexity`）

#### Panel A：连续故障持续时间

主指标：0 ms、`E+R` 下的 BEV AP@0.7。论文登记表中的 `k` 尚未由合格证据固定；正式结果必须逐个报告 `1, ..., k`，不能把整段折叠成一个虚构点。

| 持续时间 | BEV AP@0.7 | 解释 |
| :--- | ---: | :--- |
| `1, ..., k` 帧 | —（待测） | 最终应拆成每个持续时间一行 |
| `k+1` 及以上 | —（待测） | 无外推路径；报告为 neutral/unsupported，并保留实测输出 |

当前复现实现在受控覆盖中采用 `history_limit=3` 这一明确的 implementation choice，因此执行台账应分别保留以下结果；它不反向证明旧稿中的 `k`。固定 submission contract 当前仍只有上面的两个 symbolic duration ID，以下四行是尚未注册 result ID 的合同外执行计划；正式测量前必须先更新 registry 与 contract：

| 实现持续时间 | Agent scope | 0 ms BEV AP@0.7 | 支持状态 |
| ---: | :---: | ---: | :--- |
| 1 帧 | E+R | —（待测） | 历史窗口内 |
| 2 帧 | E+R | —（待测） | 历史窗口内 |
| 3 帧 | E+R | —（待测） | 历史窗口内 |
| 4 帧及以上 | E+R | —（待测） | neutral/unsupported |

#### Panel B：仅部署阶段的计算开销

冻结的训练期 teacher 不计入部署开销。两行必须在同一硬件、批大小、精度模式、预热与计时流程下测量。

| 模型 | 参数量（M） | FLOPs（G） | 峰值 GPU 显存（GB） | 端到端时延（ms） |
| :--- | ---: | ---: | ---: | ---: |
| Capacity-matched concat | —（待测） | —（待测） | —（待测） | —（待测） |
| Resilient V2X student | —（待测） | —（待测） | —（待测） | —（待测） |

### 受控结果的三随机种子汇总

这是主表成为可提交结果前必须补齐的核心统计。每个均值和标准差必须由同一数据清单、协议、模型定义与评测器下至少 3 个独立随机种子得到。

| 数据集 / 条件 | BEV AP@0.5 mean | std | BEV AP@0.7 mean | std | 3D AP@0.5 mean | std | 3D AP@0.7 mean | std | 合格完成种子数 | 证据状态 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| DAIR-V2X-C / Resilient V2X / Full / 0 ms | —（需 ≥3 seeds） | —（需 ≥3 seeds） | —（需 ≥3 seeds） | —（需 ≥3 seeds） | —（需 ≥3 seeds） | —（需 ≥3 seeds） | —（需 ≥3 seeds） | —（需 ≥3 seeds） | 1 / ≥3 | 单种子 12-condition 证据包已验证；不可计算 mean/std |

### 提交门禁补充结果

五个正式论文表文件覆盖 113 个唯一 result ID；固定 submission contract 还要求下列 15 个 ID。三随机种子的 8 个 mean/std 已在上一表逐项列出，其余 7 个如下。它们不能因为正式表版面没有显示就被省略。

#### 公开 AP@0.5 与 100 ms 受控基线

| 记录 | 指标 | 0 ms | 100 ms | 200 ms | 300 ms | 证据状态 |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| CoFormerNet public | BEV AP@0.5 | 69.33‡ | N/A | 69.13‡ | 68.60‡ | 公开 LiDAR-only 参考，协议不兼容 |
| FFNet controlled baseline | BEV AP@0.7 | N/A | —（待测） | N/A | N/A | 仅补充 100 ms 受控 ID |
| CoFormerNet controlled baseline | BEV AP@0.7 | N/A | —（待测） | N/A | N/A | 仅补充 100 ms 受控 ID |

#### 同协议方法差值

差值定义为 `Resilient V2X - adapted CoFormerNet` 的 BEV AP@0.7 百分点；只有两个操作数的 dataset、split、故障协议和 agent scope 完全一致时才允许计算。

| 条件 | Agent scope | AP@0.7 差值（百分点） | 证据状态 |
| :--- | :---: | ---: | :--- |
| L-Fail | 与受控主表一致的 E+R | —（待测） | 两个操作数均未完成受控测量 |
| C-Fail | 与受控主表一致的 E+R | —（待测） | 两个操作数均未完成受控测量 |

### E-only / R-only agent-scope 诊断（合同外计划）

论文正文要求这些诊断未来与表 3、表 4 的 `E+R` 主结果使用相同 split、checkpoint、evaluator、seeds 和 latency mapping，并使用独立 result ID、mask 和 evaluator 输出，不能替换主表单元格。但当前 registry 与 submission contract 尚未创建这 16 个诊断 ID；下表仅是合同外计划，不能被计作已注册的 pending 结果。

| 故障 | Agent scope | 0 ms | 100 ms | 200 ms | 300 ms |
| :--- | :---: | ---: | ---: | ---: | ---: |
| L-Fail | E-only | —（待测） | —（待测） | —（待测） | —（待测） |
| L-Fail | R-only | —（待测） | —（待测） | —（待测） | —（待测） |
| C-Fail | E-only | —（待测） | —（待测） | —（待测） | —（待测） |
| C-Fail | R-only | —（待测） | —（待测） | —（待测） | —（待测） |

### V2XSet 两条独立实验轨道（合同外计划）

V2XSet-Standard 与 V2XSet-Pair 不能共享 manifest、checkpoint、result ID 或表格标签，也不能跨轨道排名或求差。当前 registry 与 submission contract 没有注册任何 V2XSet result ID，因此下表只表达论文中的计划边界。

| 轨道 | 输入 / 拓扑 | Full 0 ms | Full 100 ms | Full 200 ms | Full 300 ms | L-Fail | C-Fail | 当前状态 |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: | :---: | :--- |
| V2XSet-Standard | 原生 multi-agent、LiDAR-only | —（待测：需 adapter） | —（补充待测：需 adapter） | —（待测：需 adapter） | —（待测：需 adapter） | N/A | N/A | 当前方法缺少经证据验证的 multi-agent adapter；不能声称复现数值 |
| V2XSet-Pair | 派生的一 Ego / 一 Infra、LiDAR + Camera | —（待测） | —（待测） | —（待测） | —（待测） | —（待测，E+R） | —（待测，E+R） | 独立训练与评测待完成 |

Standard 的 0/200/300 ms 是来源对齐点，100 ms 只有实测后才能作为补充点；该轨道没有 Camera 分支，因此 C-Fail 不定义。Pair 轨道还必须复用同一 target sample-ID manifest 覆盖 Full、E+R 以及四个 E-only/R-only 诊断。

### 结果填表门槛

只有同时满足以下条件，才能把 `—（待测）` 替换为 `均值 ± 标准差`：冻结配置和数据/时序 manifest；至少 3 个独立随机种子；保存 checkpoint 哈希、原始预测、评测或 profiling 输出、完整日志及软硬件环境；证据清单验证通过；比较双方的 dataset、split、metric、latency/fault protocol 与 agent scope 完全一致。当前训练的中间 loss、旧失败检查点及其 AP 均不满足这些门槛。

可执行协议与证据边界见 [复现实验指南](docs/resilient_v2x/reproduction.md) 和 [论文—代码覆盖矩阵](docs/resilient_v2x/paper-coverage.md)。

## Reference

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.3.0)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) legacy Transvision reference commit: c65a55617f7d0a9b78dc9d107370c95bcac55dca
- [FFNet-VIC3D](https://github.com/haibao-yu/FFNet-VIC3D) audited commit: 52164dfe00764c9a9925539e99689cf25b88eace（仓库未提供 LICENSE；本实现未复用其源码）
