# ResilientV2X 论文→代码覆盖矩阵

本文档逐项对照论文的 `03_problem.tex`、`04_method.tex` 和
`05_exp.tex`，说明当前仓库中的实现、测试和证据边界。这里的“已覆盖”
表示存在可执行代码与对应校验，不表示已经得到论文数值。

## 1. 状态定义

| 状态 | 含义 |
|---|---|
| A | 核心逻辑已实现，并有针对性单元测试或静态配置测试 |
| B | 已接入训练/评测配置，但尚未在当前工作区完成受控端到端训练与评测 |
| C | 只有离线适配或实验脚手架，不能视为完整数据集复现 |
| D | 论文明确保留为待测或待实现，当前没有可支持声明的证据 |

论文自身明确写明仓库中没有 verified result，实验表是协议和待测记录。
因此，下面的状态只回答“代码是否覆盖”，不升级任何 legacy 数值，也不替
checkpoint、日志、预测或 evaluator 输出生成结果。

## 2. 问题定义、时间协议与故障语义

| 论文条目 | 实现位置 | 验证位置 | 状态与边界 |
|---|---|---|---|
| 单 ego `E`、单 RSU `R`，LiDAR/Camera 四分支 | `transvision/models/resilient_v2x/contracts.py`；`transvision/models/resilient_v2x/fusion.py` | `tests/resilient_v2x/test_contracts.py`；`tests/resilient_v2x/test_feature_fusion.py` | A；不扩展到任意多智能体 |
| packet-specific arrival、最新已到达 RSU source、禁止未来帧 | `transvision/dataset/resilient_v2x_schedule.py`；`transvision/dataset/resilient_v2x_runtime.py` | `tests/resilient_v2x/test_schedule.py`；`tests/resilient_v2x/test_runtime_dataset.py` | A；空 arrival set 进入中性分支 |
| ego 与 RSU 各自的有界因果历史 | `transvision/dataset/resilient_v2x_runtime.py`；`transvision/models/resilient_v2x/causal_repair.py` | `tests/resilient_v2x/test_runtime_dataset.py`；`tests/resilient_v2x/test_causal_repair.py`；`tests/resilient_v2x/test_repair_pipeline.py` | A；`k=3`、`Δt=100 ms` 是本仓库固定实现剖面，不是论文已验证超参数 |
| 先做当前 ego 坐标系几何对齐，再由 PTF 预测残余运动 | `transvision/models/resilient_v2x/geometry.py`；`transvision/models/resilient_v2x/fusion.py` | `tests/resilient_v2x/test_geometry.py`；`tests/resilient_v2x/test_feature_fusion.py` | A；缺 pose/calibration 的分支不近似对齐 |
| 缺失、超历史范围、无效时间元数据时不外推，输出零特征/零可靠性 | `transvision/models/resilient_v2x/causal_repair.py`；`transvision/models/resilient_v2x/fusion.py` | `tests/resilient_v2x/test_causal_repair.py`；`tests/resilient_v2x/test_repair_pipeline.py` | A |
| fault 必须标明 agent、modality、source time；非零延迟先选因果端点再 mask | `transvision/dataset/resilient_v2x_schedule.py`；`tools/resilient_v2x/build_overlays.py` | `tests/resilient_v2x/test_schedule.py`；`tests/resilient_v2x/test_overlay_cli.py` | A；不能用统一 target-tick mask 替代 |
| 单帧及连续故障 `1..k` | `transvision/dataset/resilient_v2x_schedule.py`；`configs/resilient_v2x/conditions/causal_fault_diagnostic.py` | `tests/resilient_v2x/test_schedule.py`；`tests/resilient_v2x/test_configs.py` | A/B；生成与装载已覆盖，真实 duration 曲线仍待评测 |
| 所有条件复用同一 cohort | `tools/resilient_v2x/build_overlays.py` | `tests/resilient_v2x/test_overlay_cli.py` | A；cohort 记录样本 ID 摘要和排除原因 |

数据准备入口是 `tools/resilient_v2x/prepare_data.py`。它通过
`transvision/dataset/resilient_v2x_manifest.py` 和
`transvision/dataset/resilient_v2x_pcd.py` 校验 release inventory、split、
标定、时间与 PCD，并发布自哈希 manifest。对应测试是
`tests/resilient_v2x/test_manifest.py` 和
`tests/resilient_v2x/test_prepare_data.py`。

## 3. 方法模块

| 论文方法或公式 | 实现位置 | 验证位置 | 状态与边界 |
|---|---|---|---|
| modality 内共享编码器：PointPillars 与 ResNet-50+LSS | `transvision/models/detectors/resilient_v2x.py`；`configs/resilient_v2x/_base_/model.py` | `tests/resilient_v2x/test_feature_io.py`；`tests/resilient_v2x/test_configs.py` | B；真实 MMDetection3D/CUDA 栈上的合成输入前向和反向 smoke 已通过，但尚无官方数据上的受控完整训练记录 |
| 统一 `[B,C,Y,X]` BEV 约定与稀疏历史编码 | `transvision/models/resilient_v2x/feature_io.py`；`transvision/models/detectors/resilient_v2x.py` | `tests/resilient_v2x/test_feature_io.py`；`tests/resilient_v2x/test_geometry.py` | A；LSS 的 `[X,Y]` 输出在边界显式转置 |
| horizon-conditioned nonlinear PTF，直接预测累计 backward-sampling displacement 与置信图 | `transvision/models/resilient_v2x/ptf.py` | `tests/resilient_v2x/test_ptf.py`；`tests/resilient_v2x/test_repair_pipeline.py` | A |
| `h=0` 恒等、支持区间内 warp、同一 source/horizon 计算 decay 与 reliability | `transvision/models/resilient_v2x/causal_repair.py`；`transvision/models/resilient_v2x/fusion.py` | `tests/resilient_v2x/test_causal_repair.py`；`tests/resilient_v2x/test_repair_pipeline.py` | A |
| linear PTF：一阶场按 horizon 线性缩放 | `transvision/models/resilient_v2x/ptf.py`；`configs/resilient_v2x/ablations/ptf_linear.py` | `tests/resilient_v2x/test_ptf.py`；`tests/resilient_v2x/test_configs.py` | A |
| no-PTF 几何-only、容量匹配 | `transvision/models/resilient_v2x/fusion.py`；`configs/resilient_v2x/ablations/ptf_none.py` | `tests/resilient_v2x/test_feature_fusion.py`；`tests/resilient_v2x/test_configs.py` | A；匹配参数不参与前向 |
| endpoint-relative observed/propagated availability、source age、RSU delay、trajectory confidence 共同形成 routing metadata | `transvision/models/resilient_v2x/causal_repair.py`；`transvision/models/resilient_v2x/contracts.py` | `tests/resilient_v2x/test_causal_repair.py`；`tests/resilient_v2x/test_repair_pipeline.py`；`tests/resilient_v2x/test_contracts.py` | A；最新到达 RSU endpoint 可为 observed，但仍使用 `gamma × confidence`；只有真正当前 Ego reliability 为 1 |
| ego/RSU modality aggregation | `transvision/models/resilient_v2x/routing.py` 中 `ModalityAggregator` | `tests/resilient_v2x/test_routing.py` | A |
| LiDAR、Camera、Synergy 三专家 | `transvision/models/resilient_v2x/routing.py` 中 `DynamicExpertRouter` | `tests/resilient_v2x/test_routing.py` | A |
| DER descriptor 严格由 3 个 256 维 GAP、2 个 reliability、8 个 availability、4 个 age、1 个 delay 组成，共 783 维 | `transvision/models/resilient_v2x/routing.py`；`transvision/models/resilient_v2x/contracts.py` | `tests/resilient_v2x/test_routing.py`；`tests/resilient_v2x/test_contracts.py` | A；禁用 metadata 时保留维度并将相应字段中性化 |
| support-aware softmax gate；无可用专家时中性输出 | `transvision/models/resilient_v2x/routing.py` | `tests/resilient_v2x/test_routing.py` | A |
| 从四分支历史到检测 head 的完整 student 前向 | `transvision/models/resilient_v2x/fusion.py`；`transvision/models/detectors/resilient_v2x.py`；`transvision/models/data_preprocessors/resilient_v2x.py` | `tests/resilient_v2x/test_feature_fusion.py`；`tests/resilient_v2x/test_runtime_dataset.py`；`tests/resilient_v2x/test_configs.py` | B；fixture DAIR 数据准备→dataset→preprocessor→四分支编码→检测/NMS 的 CUDA smoke 已通过，尚未以官方数据和真实 checkpoint 完成受控端到端评测 |

## 4. 两阶段训练与蒸馏

| 论文训练条目 | 实现位置 | 验证位置 | 状态与边界 |
|---|---|---|---|
| clean full-input teacher 独立训练 | `configs/resilient_v2x/dair_clean_teacher.py` | `tests/resilient_v2x/test_configs.py` | B；配置存在，当前工作区没有可声明的训练 checkpoint |
| student 使用退化输入，teacher 使用同 sample 的 clean 输入 | `transvision/dataset/resilient_v2x_runtime.py`；`transvision/models/detectors/resilient_v2x.py` | `tests/resilient_v2x/test_runtime_dataset.py`；`tests/resilient_v2x/test_head_distillation.py` | A/B；数据/前向契约已测，完整训练仍待执行 |
| teacher strict checkpoint load、`eval`、`requires_grad=False`、`no_grad` | `transvision/models/resilient_v2x/distillation.py`；`transvision/models/detectors/resilient_v2x.py` | `tests/resilient_v2x/test_distillation.py`；`tests/resilient_v2x/test_head_distillation.py` | A |
| fused-feature MSE，按 feature 元素归一化 | `transvision/models/resilient_v2x/distillation.py` | `tests/resilient_v2x/test_distillation.py`；`tests/resilient_v2x/test_head_distillation.py` | A |
| multiclass categorical KL 与 sigmoid-head Bernoulli KL 两条合法路径 | `transvision/models/resilient_v2x/distillation.py` | `tests/resilient_v2x/test_distillation.py`；`tests/resilient_v2x/test_head_distillation.py` | A；当前 Anchor3DHead 配置选择 Bernoulli |
| logit distillation 只对匹配有效元素，且 `T²` 只乘一次 | `transvision/models/resilient_v2x/distillation.py`；`transvision/models/detectors/resilient_v2x.py` | `tests/resilient_v2x/test_distillation.py`；`tests/resilient_v2x/test_head_distillation.py` | A |
| 部署时只保留 student | `transvision/models/detectors/resilient_v2x.py`；`tools/resilient_v2x/profile.py` | `tests/resilient_v2x/test_head_distillation.py`；`tests/resilient_v2x/test_evidence_profile.py` | A/B；代码可排除 teacher，尚无正式部署测量 |

完整两阶段命令和 checkpoint 哈希要求见
[受控复现手册](reproduction.md)。

## 5. 主条件、诊断与消融

### 5.1 DAIR 主矩阵

4 个 Full 配置：

- `configs/resilient_v2x/conditions/global_delay_000_full.py`
- `configs/resilient_v2x/conditions/global_delay_100_full.py`
- `configs/resilient_v2x/conditions/global_delay_200_full.py`
- `configs/resilient_v2x/conditions/global_delay_300_full.py`

8 个 E+R 因果故障配置：

- `configs/resilient_v2x/conditions/causal_delay_000_l_fail.py`
- `configs/resilient_v2x/conditions/causal_delay_000_c_fail.py`
- `configs/resilient_v2x/conditions/causal_delay_100_l_fail.py`
- `configs/resilient_v2x/conditions/causal_delay_100_c_fail.py`
- `configs/resilient_v2x/conditions/causal_delay_200_l_fail.py`
- `configs/resilient_v2x/conditions/causal_delay_200_c_fail.py`
- `configs/resilient_v2x/conditions/causal_delay_300_l_fail.py`
- `configs/resilient_v2x/conditions/causal_delay_300_c_fail.py`

这 12 个配置由 `tests/resilient_v2x/test_configs.py` 校验 delay、condition、
agent scope、overlay/hash 外部输入和 evaluator 键。E-only、R-only、连续
duration 只属于 `configs/resilient_v2x/conditions/causal_fault_diagnostic.py`
诊断，不替代论文主矩阵的 E+R 单帧格子。

### 5.2 容量匹配消融

| 论文计划消融 | 配置 | 执行逻辑与测试 | 状态 |
|---|---|---|---|
| nonlinear PTF | `configs/resilient_v2x/dair_resilient_v2x.py` | `transvision/models/resilient_v2x/ptf.py`；`tests/resilient_v2x/test_ptf.py` | A/B |
| linear PTF | `configs/resilient_v2x/ablations/ptf_linear.py` | `tests/resilient_v2x/test_ptf.py`；`tests/resilient_v2x/test_configs.py` | A |
| no PTF | `configs/resilient_v2x/ablations/ptf_none.py` | `tests/resilient_v2x/test_feature_fusion.py`；`tests/resilient_v2x/test_configs.py` | A |
| static routing | `configs/resilient_v2x/ablations/router_static.py` | `tests/resilient_v2x/test_routing.py` | A |
| uniform routing | `configs/resilient_v2x/ablations/router_uniform.py` | `tests/resilient_v2x/test_routing.py` | A |
| no reliability | `configs/resilient_v2x/ablations/no_reliability.py` | `tests/resilient_v2x/test_routing.py` | A |
| no delay metadata | `configs/resilient_v2x/ablations/no_delay_metadata.py` | `tests/resilient_v2x/test_routing.py` | A |
| no distillation | `configs/resilient_v2x/ablations/no_distillation.py` | `tests/resilient_v2x/test_configs.py`；`tests/resilient_v2x/test_head_distillation.py` | A/B |
| capacity-matched concat deployment baseline | `configs/resilient_v2x/ablations/concat_capacity_matched.py` | `transvision/models/resilient_v2x/routing.py`；`tests/resilient_v2x/test_routing.py`；`tests/resilient_v2x/test_configs.py` | A；参数量严格匹配，gate 不执行 |

论文没有定义 concat baseline 的具体拓扑。本实现选择“masked
modality-expert concat projection”，并用 dormant capacity parameters 做精确
容量匹配；这是显式 implementation choice，不能写成论文指定结构。消融的
真实效果仍必须在同一 manifest、cohort、checkpoint 选择、seed 与 evaluator
下测量。

## 6. 检测指标、预测与证据

| 论文证据要求 | 实现位置 | 验证位置 | 状态与边界 |
|---|---|---|---|
| Car BEV/3D AP，IoU 0.5/0.7，R40 | `transvision/evaluation/resilient_v2x_detection.py`；`transvision/evaluation/metrics/resilient_v2x_metric.py` | `tests/resilient_v2x/test_detection_evaluation.py` | A；这是本仓库 evaluator 实现，不能假定与未记录 legacy evaluator 相同 |
| 同一推理保存 prediction evidence | `transvision/evaluation/metrics/resilient_v2x_metric.py` | `tests/resilient_v2x/test_detection_evaluation.py` | A；不能用第二次推理伪装成同一次评测预测 |
| condition、manifest、overlay、checkpoint、metrics、prediction、profile 哈希绑定 | `transvision/evaluation/resilient_v2x_evidence.py`；`tools/resilient_v2x/build_evidence.py` | `tests/resilient_v2x/test_evidence_profile.py` | A |
| trainable parameters | `transvision/evaluation/resilient_v2x_profile.py`；`tools/resilient_v2x/profile.py` | `tests/resilient_v2x/test_evidence_profile.py` | A/B；测量入口已测，正式数字仍待受控执行 |
| 声明 counting convention 与输入后测 FLOPs | `transvision/evaluation/resilient_v2x_profile.py`；`tools/resilient_v2x/profile.py` | `tests/resilient_v2x/test_evidence_profile.py` | A/B |
| peak GPU memory | `transvision/evaluation/resilient_v2x_profile.py`；`tools/resilient_v2x/profile.py` | `tests/resilient_v2x/test_evidence_profile.py` | A/B；必须来自 CUDA 受控运行 |
| 同步 model-path latency、warm-up、iterations、统计量 | `transvision/evaluation/resilient_v2x_profile.py`；`tools/resilient_v2x/profile.py` | `tests/resilient_v2x/test_evidence_profile.py` | A/B；从已 collate batch 的 `model.test_step` 到 decoded predictions，包含编码、对齐、PTF、DER、head 和后处理，但排除 dataloader I/O |
| 部署复杂度排除 frozen teacher | `tools/resilient_v2x/profile.py` 的 `--exclude-module` 与部署检查 | `tests/resilient_v2x/test_evidence_profile.py` | A/B |
| 锁定环境、CUDA/custom ops 检查和环境 manifest | `environments/resilient_v2x/Dockerfile`；`tools/resilient_v2x/check_environment.py`；`tools/resilient_v2x/capture_environment.py` | `tests/resilient_v2x/test_environment.py`；`tests/resilient_v2x/test_packaging.py` | A/B；只有 controlled 检查通过才可支持正式证据 |

`tests/resilient_v2x/test_evidence_profile.py` 验证的是证据结构、哈希关系和
测量逻辑，不是 GPU 测量值本身。开发环境中的跳过项或 CPU 输出不能升级成
论文复杂度结果。当前 latency 明确排除 sensing-tensor ingestion/dataloader
I/O，因此还不满足论文对完整 end-to-end latency 的最宽定义；必须将其标成
model-path latency，或另行增加并留证 I/O 计时。`torch.profiler` FLOP 只
覆盖它能识别的算子，自定义 CUDA op 等未支持算子可能漏计，报告必须保留
counting-convention 与 partial-coverage 说明。

## 7. 论文未给出的 implementation choices

论文方法章节故意保留 `k`、`Δt`、`α`、训练扰动概率、时延分布和多项训练
细节为符号或待执行协议。实验章节保存的 legacy implementation record 也
明确不是 verified evidence。当前配置为了让实验可执行而固定了以下选择：

- 时间剖面：`k=3`、`Δt=100 ms`、固定延迟映射 0/100/200/300 ms；
- 数据校验剖面：相邻采样间隔 50–150 ms、最大 capture skew 50 ms、
  固定官方 split 摘要与当前 Car class mapping；
- feature channels `C=256`，因此当前 routing descriptor 是 783 维；
  PointPillars 与 ResNet-50+LSS 是 planned instantiation，不是数值证据；
- PTF：`α=0.9`、projected channels 64、context channels 128、low-resolution
  displacement 上限 8 cells；
- 空间剖面：point-cloud range `[0,-40,-3,80,40,1]`、voxel
  `[0.4,0.4,4.0]`、BEV resolution 0.8 m、100×100 grid；
- camera：输入 256×704、depth bins `[1,81)` step 1、ImageNet mean/std；
- detection：Car anchor 尺寸/高度、assigner 阈值、NMS、score threshold 和
  max detections；
- distillation：Anchor3DHead 的 Bernoulli 路径、`T=2`、
  `lambda_feature=1`、`lambda_logit=1`；
- routing：PTF/aggregator/router 的隐藏宽度和具体卷积拓扑；
- runtime：AdamW、learning rate `1e-4`、weight decay `0.01`、50 epochs、
  500 iteration warm-up、gradient clip 35、单卡 batch 1、worker 数和
  seed `20250218`；
- checkpoint 的 best/epoch 选择规则，以及本仓库自定义 Car BEV/3D AP-R40
  evaluator、detection range 和后处理约定；
- 训练 overlay 的 seed、delay schedule、LiDAR/Camera fault probabilities
  是外部协议产物。复现手册中的 `0.2/0.2` 只是可追踪示例，不是论文默认值。

这些值分别记录在：

- `configs/resilient_v2x/_base_/model.py` 的
  `implementation_choices_model`；
- `configs/resilient_v2x/_base_/dataset.py` 的
  `implementation_choices_dataset`；
- `configs/resilient_v2x/_base_/runtime.py` 的
  `implementation_choices_runtime`；
- overlay index、merged config、环境 manifest 与最终 evidence bundle。

尤其要注意：论文 legacy record 曾保存过若干训练数值，但同时明确说明没有
配置、checkpoint 或日志验证它们。当前实现即使选择了相同数字，也只能称为
本次执行记录，不能倒推为已复现的论文默认设置。

## 8. V2XSet Pair 与 Standard 的边界

`tools/resilient_v2x/prepare_v2xset_pair.py` 和
`tools/resilient_v2x/V2XSET_PAIR_README.md` 实现 V2XSet-Pair 的严格序列级
离线清单适配边界。`tests/resilient_v2x/test_v2xset_pair.py` 验证 train 最早
合格 anchor 与最小 numeric AV、validation Standard fixed Ego、最近 Infra 与
numeric-ID tie break、整序列 pair 冻结，以及每个目标的全部 delay/history
因果端点。端点检查覆盖 LiDAR、四路 RGB、内参、Camera-to-LiDAR 刚体变换、
pose、70 m、payload 哈希和身份连续性；后续不合格帧只精确排除，不重选 pair。
输出还包含嵌套自哈希 target manifest，所有 Full/E+R 故障主条件和 0 ms
分支诊断必须引用同一 target hash/count。

`tools/resilient_v2x/export_v2xset_standard.py` 和
`tools/resilient_v2x/V2XSET_STANDARD_README.md` 实现 V2XSet-Standard 的
fail-closed source-loader trace 导出边界。`tests/resilient_v2x/test_v2xset_standard.py`
验证 train 每 epoch 随机 Ego 的 RNG/draw 证据、validation 每 scene 固定 Ego、
70 m 完整候选集、实际 agent 顺序与 `max_cav` 前缀截断、LiDAR payload，
以及逐 agent 生成时延、source/arrival 与因果 frame offset。100 ms 被显式标为
supplemental，不能冒充 source-aligned 条件。

当前边界必须明确：

- Pair 需要调用者先提供由官方 V2XSet parser 产生的 canonical、自哈希
  normalized inventory；本仓库不猜测或替代官方原始 schema；
- Pair 已实现 split/Ego/anchor/frozen-pair/causal-endpoint 和跨条件
  sample-ID/hash/count 协议，但尚未接到当前 DAIR runtime、训练配置、检测目标
  loader 和 evaluator，也没有 Pair checkpoint 或结果 evidence；
- 当前仓库没有 V2XSet/OpenCOOD 多智能体 loader；Standard exporter 只接受
  真实 source dataloader 已执行时产生的规范 trace，不解析数据、不重建选择、
  不运行模型或 evaluator；
- 当前 ResilientV2X 方法固定为单 Ego/单 RSU，Standard manifest 因而强制
  `status=N/A` 与 `multi_agent_adapter_implemented=false`，不能改写为 eligible；
- Standard 中 C-Fail、二智能体 E+R/R-only 标签并不成立，不能由 Pair
  结果替代；
- Pair 与 Standard 必须使用不同 manifest、checkpoint、result ID 和表格
  标签，不能跨 track 排名或计算差值。

因此当前状态是 Pair 离线序列适配=C、Standard source-trace 导出=C，
Standard 方法参评状态=N/A；两者都不能声称已复现任何 V2XSet 数值。

## 9. 当前可声明与不可声明

当前可以声明：

- 论文的单 ego/单 RSU 因果 repair-then-route 方法已形成可执行模块；
- DAIR manifest、overlay、12 主配置、持续故障诊断、两阶段训练配置、
  evaluator、prediction evidence、profile 与 evidence bundle 已接线；
- 核心公式、边界条件、配置矩阵和证据不变量有针对性测试。
- 在真实 MMDetection3D/CUDA 开发栈上，四分支原始输入、检测/NMS、loss、
  teacher-student 蒸馏及反向传播的集成 smoke 已通过；这只证明代码路径可执行，
  不是正式显存、速度、精度或论文复现证据。

当前不能声明：

- 论文任一 AP、PDR、FLOPs、显存、时延或消融差值已复现；
- 模型已在官方 split 上完成三 seed 受控训练；
- 当前 checkpoint 与论文作者模型等价；
- DAIR 主表或 duration 曲线已有有效数字；
- V2XSet-Pair 或 V2XSet-Standard 已完成端到端复现；
- 相对任何 baseline 的优越性已经得到证据支持。

从“代码覆盖”升级到“数值复现”至少需要：受控环境通过、官方数据和 split
哈希通过、冻结 cohort、记录全部 overlay、完成 clean teacher 与 student
训练、保存真实 checkpoint/metrics/predictions/profile、构建哈希闭合的
evidence bundle，并按论文协议运行至少三个已声明 seed。
