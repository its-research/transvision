# MHT 与 JPDA／PKF：完整三种子比较入口

已实现 `tools/event_track_v2x/compare_mht_probabilistic_validation.py`。
它要求三个 scan-MHT K=4 运行和九个 JPDA／PKF 运行全部通过完整评价，才生成比较文件。
当前没有完整 MHT 指标，不能生成真实十二格结果表，也不能据此声称论文方法有效。

## 1. 本表回答什么

固定 SPD official val 的 21 序列、3,316 个输出帧，以及已完成四卡 A100 训练的
身份种子 1337、2027、3407，比较以下四种后端适配：

| 后端 | 每个种子的身份处理 | 连续状态处理 |
| --- | --- | --- |
| scan-MHT K=4 | 保留四条全局历史，永久剪枝，不能恢复已剪分支 | 每条历史独立维护 CI 状态 |
| JPDA-CI | 条件于已提交的单条身份历史 | 关联边缘概率下的 CI 更新，随后压缩 |
| JPDA-Kalman | 同上 | Kalman 更新，随后压缩 |
| PKF | 同上 | 概率状态更新，随后压缩 |

检测缓存、embedding、学习势、car-first-top64 选择、源到达约束、官方评价 ROI 和
指标引擎保持一致。只报告 car；SPD 的 Car／Truck／Van／Bus 映射为 car，不读取
SPD test 或 test_A。已使用的 val 不是未见确认集。

**这不是仅改变「恢复开关」的因果实验。** MHT 与当前 JPDA／PKF 对连续状态及
状态时刻的处理不同；共同配置中的运动参数相同不能消除这一差异。本表也未包含
完整可恢复方法、强单端后端或公开方法原样复现。

## 2. 完成和拒绝条件

先分别完成 [MHT 完整评价流程](mht-full-val-contract-20260914.md) 和
[JPDA／PKF 完整评价流程](spd-probabilistic-full-val-20260914.md)。比较入口会重新读取：

1. 两份冻结实验计划和全部十二份完整评价报告，不接受缺失、重复或失败格。
2. 各运行的最终回执、独立审计、预测、逐帧审计、计时和数据库证据。
3. 三个冻结检查点、全部共享推断源码，以及 MHT 专属源码和评价入口。
4. 原生指标、评价运行时、golden cases、GT 摘要及评价输入清单。

仅报告文件名或 `status=complete` 不足以通过。MHT 运行还须满足固定 K=4、四项
指派预算和完整官方日程；JPDA／PKF 沿用已有九格比较器的固定配置检查。
读取前后文件摘要必须一致。已有输出目录不得覆盖。

跨方法的原生评价协议允许 `kind` 方法标识不同，其余字段必须相同。
MHT 多记录 SciPy、SQLite 和指派二进制身份；其余推断运行时字段须与 JPDA／PKF
一致。各方法的原始关联因子必须逐种子相同，而不只是计划写着使用相同模型。

### 实际因子核验

每帧读取 `(sequence_id, event_id, factor_rows_sha256)`，按日程顺序进行规范 JSON
编码、添加换行并累积 SHA-256。这与 JPDA 审计的全流指纹定义一致。
MHT 同时按序列重算指纹和帧数，与独立 SQLite 身份账本审计逐项比较。
不能将若干序列摘要再次哈希后，冒充按帧累积的全流摘要。

测试覆盖漏帧、多帧、顺序改变、重复事件、非法摘要、摘要改动及序列审计不一致。
完整结果未齐时，不输出部分主表，也不选择较有利的种子替代缺失运行。

## 3. 输出与统计口径

输出为新目录下的 `comparison.json`，保留十二个运行的指标、资源记录、报告摘要和
完整输入证据。四种后端分别报告三种子的均值、样本标准差、最小值、最大值和逐种子值。
主指标包含 HOTA、AssA、DetA、IDF1、AMOTA、AMOTP_m、MOTA、FP、FN、IDS、Frag。

三种子均值为 `sum(x_s)/3`，样本标准差使用分母 `3-1`；标准差不是置信区间。
同一种子的 21 个序列不是 21 次独立训练重复。原生整体 HOTA／IDF1 不用序列值均值替代。
MHT 减去每种基线的差值同时保留逐种子整体指标和逐序列 HOTA／AssA／DetA／IDF1。
AMOTP_m 的单位为米，越低越好；FP／FN／IDS 使用 nuScenes 计数，不与 TrackEval
身份计数混合。

每个运行分别保留实际耗时、进程峰值 RSS、数据库字节数、步耗时与帧耗时的
p50／p95／p99／最大值，以及对应计时范围。**不平均分位数以构造尾时延，不把单进程
RSS 当作并发任务总峰值。** 这些是共享 CPU 上的观测，不证明同延迟、同内存或部署尾时延优势。

比较文件即使 `status=complete`，也只表示十二格适配比较齐全。它明确保持：

- `fair_resources_verified=false`，`deployment_tail_latency_verified=false`。
- `same_state_time_protocol_verified=false`，`recovery_only_causal_effect_verified=false`。
- `recoverable_method_included=false`，`strong_single_endpoint_controls_included=false`。
- `public_methods_reproduced=false`，`full_paper_comparison_completed=false`，`paper_eligible=false`。

## 4. 调用方式

在 canonical transvision 使用研究 Python 环境执行 `--help` 查看入口。
参数如下；每个报告参数是两个独立值：绝对路径与 SHA-256。

| 参数 | 数量 | 来源 |
| --- | ---: | --- |
| `--mht-campaign`、`--mht-campaign-sha256` | 各 1 | 冻结的 MHT `campaign.json` |
| `--probabilistic-campaign`、`--probabilistic-campaign-sha256` | 各 1 | 冻结的 JPDA／PKF preflight |
| `--mht-report REPORT SHA256` | 恰好 3 | 三个 K=4 完整原生评价报告 |
| `--probabilistic-report REPORT SHA256` | 恰好 9 | 三种子 × 三种 JPDA／PKF 后端的完整评价报告 |
| `--output DIRECTORY` | 1 | 尚不存在的普通目录 |

当前冻结 MHT 计划为暂存 val 根目录下 `mht-k4-campaign-v1/campaign.json`，摘要
`67ea52bf5008b5b99404dc345a6e447b411e0d578cf6a6d0e503335822b5aec0`。
JPDA／PKF 计划为同一根目录的 `campaign-preflight-v2.json`，摘要
`19c3861c52f9b7375fa39958db522a965c6728e25f46a1ae23576577787f7715`。
十二份报告尚未齐备，因此本记录不提供虚构的完整运行命令或结果文件。

## 5. 本次实际验证

150 项聚合、文件绑定、完整 MHT 审计及预检相关测试通过。新增比较器的测试采用
明确的合成报告对象；测试中的小型绑定对象不经过真实 SQLite 审计或原生指标引擎，
不能代替真实 3,316 帧结果。首轮测试曾因新增测试的 MHT 导入路径错误而在收集阶段失败；
修正为现有 `tools/event_track_v2x` 路径后，以下完整测试通过。

测试回执 `mht-comparison-tests-v1.xml` 位于现有 train 暂存根目录，SHA-256 为
`7e02c8ef4da55285209fbf312829012b4eaaa4d7dbb75543205b6b51bdad714d`。
比较器 SHA-256 为 `da29fb69655438df52978845916a27c3c7b5c277a8be9f0dccce86ade5fc0e57`。

另对现有真实 MHT 八帧前缀重算实际因子指纹，与同种子 JPDA-CI 前八帧相同：
`d946740ac9e8e5121a391f41b028e4ede17187d86f053223339b9a380cdd4ac1`。
该核验不读取 GT，范围仅为八帧，不证明完整 val 因子一致。
真实计划的 113 个共享源码、MHT 新增 3 个源码、6 个合同源码和 8 个输入文件也已复核一致。
新增基线读取器还重新验收了已完成的 2027 JPDA-CI 报告：11 项主指标、21 序列、
实际因子流和资源字段均通过绑定检查；没有重新训练或重新计算原生指标。
完整 MHT 运行尚未启动，不生成十二格比较结果。

参数训练继续只用至少四卡 A100。上述聚合、审计和冻结推断不做参数训练；不上传
GT、模型、完整预测或教师流，也不修改运行中的冻结源码。
