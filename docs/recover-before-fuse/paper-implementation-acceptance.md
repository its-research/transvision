# Recover Before Fuse 代码验收

隔离工作树：`/private/tmp/rbf-paper-implementation.bFckuK/transvision`，基于 `f8d7f54`。原工作树和运行中的实验未修改；不启动正式训练，不发布 ClearML，不提交或推送。

**整体仍为部分完成。** 下列软件证据不是论文收益或真实数据复现证据。本次增量提交范围、静态检查与复现命令见 [提交前说明](paper-commit-readiness.md)。

## 最终提交前验证

最终源码研究回归为 **3222 passed、6 skipped**；独立评价回归 **36 passed**，另补跑官方指标 golden case **1 passed**。11 个新增 CLI 在仓库外目录的 `--help` 冒烟通过；适用的静态、格式、文本与文档检查通过。6 项跳过中，论文评价和官方指标用独立环境补验，其余外部参考资产与 Open3D 检查未验证。

源码/配置/测试共 77 个变更文件的哈希在最终回归期间保持不变；加上 4 份说明文档和 1 份回执，共 82 个待提交文件。实际数量、回执哈希、环境版本及未完成项见 [提交回执](paper-commit-receipt.json)。这只表明本次增量可提交，不改变下方整体计划的未完成状态。

## 历史验证回执

| 检查                                   | 结果                                        | 原始回执                                                                            |
| -------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------- |
| EventTrackV2X 研究回归                 | 3207 passed，6 skipped                      | `/private/tmp/rbf-paper-implementation.bFckuK/research-regression-final.xml`        |
| 独立评价回归                           | 36 passed，0 failed，0 skipped              | `/private/tmp/rbf-paper-implementation.bFckuK/evaluator-regression-final.xml`       |
| 本轮资源/优先级/论文入口及直接依赖回归 | 292 passed，0 failed                        | `/private/tmp/rbf-paper-implementation.bFckuK/continuation-research.xml`            |
| 本轮独立评价与表图修复后回归           | 36 passed，0 failed                         | `/private/tmp/rbf-paper-implementation.bFckuK/continuation-evaluator-fixed.xml`     |
| 配对基线首次全量回归（失败保留）       | 3187 passed、3 failed、32 errors、6 skipped | `/private/tmp/rbf-paper-implementation.bFckuK/pair-integration-full-regression.xml` |
| 配对基线修复后直接依赖/历史审计        | 210 passed、1 skipped                       | `/private/tmp/rbf-paper-implementation.bFckuK/pair-source-compatibility-fixed.xml`  |
| 补丁检查                               | `git diff --check` 通过                     | 本轮只读核验                                                                        |

上表保留前期完整回归、直接依赖回归及失败记录；这些覆盖存在重复，不累加为独立测试数量。双数据格式集成与两进程 CPU/Gloo 测试包含在研究回归中。首次 torchrun localhost rendezvous 超时，以及本轮 Figure 4 夹具缺字段的失败回执 `continuation-evaluator.xml` 均保留；后者已修复并通过重跑。GPU/NCCL 未验证。

本轮已补资源扫描的源码、检查点种子、调度和评价证据绑定，以及训练内优先级最佳 epoch 的实际权重恢复；使用方式和边界见 [资源扫描说明](paper-resource-scan.md)。

配对基线本轮曾向封存模块加入可选钩子，导致源码哈希审计拒绝（不是数值差异）。现已精确撤回四个历史文件的改动，新适配层为实例绑定原函数的依赖，并验证复用相同 code object、不污染模块全局及快照不共享写。所有首次失败项已随直接依赖回归通过；随后提交前全量复核也已通过（3222 passed、6 skipped，`commit-readiness-research.xml`）。该回执早于最后的静态检查修复；最终源码以 `paper-commit-receipt.json` 绑定的独立回执为准，历史失败记录保留。该入口和限制见 [配对基线说明](paper-pair-baselines.md)。

## 要求—入口—验证—状态

模块位于 `transvision/models/event_track_v2x/`，工具位于 `tools/event_track_v2x/`，测试位于 `tests/event_track_v2x/`。

| 要求                                          | 实现入口                                                             | 测试                                             | 状态与限制                                                                                                                                        |
| --------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 全类别 score >= 0.05 / top64                  | `paper_protocol.py`、`forest_tracking.py`                            | `test_paper_contract.py`                         | 已实现；car 仅评价，旧 car-first 独立                                                                                                             |
| 训练/推断候选一致、split 与 GT 隔离           | `forest_training_data.py`、`persistent_cache_stream.py`              | contract、pipeline                               | SPD train/val；V2V4Real train/official_test；训练只用 train                                                                                       |
| 在线森林、分支独立、稳定 ID、重开、历史不可变 | `paper_runtime.py`、`persistent_*`                                   | 原持久化回归、end_to_end                         | 已贯通，缓存和旧序列化兼容                                                                                                                        |
| 合法动作、原始观测状态重建                    | `paper_decision.py`                                                  | contract                                         | 条件 root-Hamming，有独立穷举；MAP 仅保留集合 MAP，不宣称全局已证                                                                                 |
| 固定/上界/学习优先级                          | `allocation_policy.py`、`paper_runtime.py`                           | end_to_end                                       | 三策略真实运行、因子哈希一致                                                                                                                      |
| 恢复/过期/覆盖/资源审计                       | 原持久化审计、`paper_resources.py`                                   | 原审计回归、end_to_end                           | 多维成本，重叠计数不相加声称 FLOPs；磁盘不等于内存                                                                                                |
| 正式配置与入口                                | `run_paper.py`、`configs/event_track_v2x/paper/`                     | pipeline、end_to_end                             | 已有预检与重放；资产缺失不出正式成功回执                                                                                                          |
| 连续序列、训练内划分、三种子、冻结            | `prepare_paper_rows.py`、`train_paper_identity.py`                   | pipeline、end_to_end                             | 支持 1337/2027/3407；未运行真实全量训练                                                                                                           |
| 两类损失、历史前向、精确小模型 NLL            | `paper_calibration.py`、`paper_exact_loss.py`、`learned_identity.py` | pipeline                                         | 梯度已测；大组件明确为 surrogate                                                                                                                  |
| DDP                                           | `paper_ddp.py`                                                       | pipeline                                         | CPU/Gloo 已测，真实多卡未测                                                                                                                       |
| 教师、优先级训练                              | `train_paper_priority.py`                                            | end_to_end、`test_paper_priority_selection.py`   | 小集成贯通；训练内 holdout 最佳 epoch 选择和实际权重恢复已补齐；正式全集回执仍缺                                                                  |
| 独立存在校准                                  | `paper_calibration.py`                                               | pipeline                                         | train-only Platt、NLL/Brier/ECE；完整冻结编排未完成                                                                                               |
| native 点云到缓存                             | `paper_pointpillar.py`、`build_native_paper_cache.py`                | pipeline                                         | 数学与模拟管线已测；真实权重、锚框一致性、编译依赖未核验                                                                                          |
| native 身份映射                               | `prepare_paper_rows.py`                                              | pipeline、end_to_end                             | 接受独立标注 envelope；官方原始标注自动转换未实现                                                                                                 |
| 单端、几何、Top-K、不可恢复图、MHT、JPDA/PKF  | `paper_runtime.py`                                                   | end_to_end                                       | 九类内部后端已贯通模拟集成                                                                                                                        |
| learned+CI、M0–M4                             | `paper_pair_baselines.py`、`run_paper_pair_baselines.py`、pair 配置  | `test_paper_pair_baselines.py` 及原机制/审计回归 | clean-link 双格式入口、状态重开、CLI 和独立评价已贯通；复用封存算法，四个历史模块字节未改；任意异步流、真实配对模型资产和资源扫描自动调度仍未完成 |
| 历史关闭、优先级替换                          | paper 配置与 scorer                                                  | pipeline、end_to_end                             | 已实现实际前向/策略变化                                                                                                                           |
| 独立恢复关闭                                  | 当前只有 irreversible 对照                                           | 尚缺                                             | 未实现；仅关前沿展开仍可能被全合法动作解码恢复，不能据此宣称恢复关闭                                                                              |
| train 内资源扫描/选择/冻结                    | `scan_paper_resources.py`、`resource-scan.json`                      | `test_paper_resource_scan.py`                    | 已有 plan/run/freeze、独立进程测量、三种子约束选择及证据绑定；当前 CPU，未验真实全集或 GPU；不能按 K 声称等资源                                   |
| 独立评价、逐序列指标                          | `evaluate_paper.py`                                                  | `test_paper_evaluation.py`                       | car-only，nuScenes/TrackEval；native XY ROI 不冒称原生 corner ROI                                                                                 |
| 三种子均值/样本标准差、bootstrap              | `paper_reports.py`、`paper_protocol.py`                              | evaluation、pipeline                             | AMOTP 定义混排拒绝已补齐；macro-sequence 描述性 CI，不是官方 pooled 非线性指标重采样 CI                                                           |
| 校准/错误持续/恢复事件                        | `paper_calibration.py`、`paper_reports.py`                           | pipeline                                         | 保留删失与恢复证据；真实输入尚缺                                                                                                                  |
| Table 1–4 / Figure 1–5                        | `report_paper.py`、`plot_paper.py`                                   | evaluation                                       | 数据生成器已测，fixture 水印，无真实性能填值                                                                                                      |

## 公开基线

CoopTrack、SparseCoop、DMSTrack 已核实官方仓库并提供源码锁定的实际命令适配器；尚缺运行时、权重、原生特征及 split 核验，未完成复现。SparseCoop cleaned-label 2 Hz 只能列原协议表。详细来源、commit 和命令见 `configs/event_track_v2x/paper/public-baselines.json`。

Graph Lap-CoMOT、Long-SCOPE 的完整算法仍未实现；已读论文不等于代码完成。CoTrack 尚缺可核验完整算法或官方代码，不与同名点跟踪器混用。

## 环境和重现

研究 Python：`/private/tmp/eventtrack-v2-checks.bYZm1l/bin/python`，numpy 1.26.4、scipy 1.14.1、torch 2.9.1、pytest 8.4.2。

评价 Python：`/private/tmp/eventtrack-evaluator-v1.zT0Eh9/venv/bin/python`，独立提供 nuscenes、trackeval、pyquaternion、matplotlib。以上为本机环境路径，不是跨机器依赖锁。

研究回归：设置 `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` 和 `RBF_EVALUATOR_PYTHON` 为评价 Python，再用研究 Python 执行 `-m pytest -q -p no:cacheprovider --ignore=tests/event_track_v2x/test_train_inference_evaluator.py tests/event_track_v2x`。

独立评价回归：用评价 Python 执行 `-m pytest -q -p no:cacheprovider tests/event_track_v2x/test_train_inference_evaluator.py tests/event_track_v2x/test_paper_evaluation.py`。

真实资产依赖：官方 split 全量数据与哈希、recording 映射、主协议冻结检测器/特征/身份/优先级权重、train-only 校准、原生 PointPillar 锚框与编译依赖、全类别身份标注转换、GPU/NCCL 环境锁，以及公开基线资产。不得将 fixture 改标签充当正式数据或成功回执。
