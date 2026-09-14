# 新进程 CPU 资源测量与输入一致性检查

新增 `run_resource_sweep_v2.py`，按预先给定的配置和顺序启动独立 Python 进程，
测量持久化跟踪后端的逐帧延迟、进程内存高水位和保留产物大小。正式入口要求
完整 SPD val、每个配置的三个完整 train 检查点，以及至少三次重复。

最初执行了本地合成夹具：九个后端、每个后端两次，共 18 个实际子进程。接入前沿完整类候选后扩展为 11 个后端，增加覆盖感知容量判断后为 13 个。后续增加三种排序上界实现，当前为 16 个后端、32 个实际子进程的重复检查。
资源重复只使用身份／优先级种子 1337，不是三个种子的正式对照。
尚未运行真实数据资源曲线，也没有证明同算力或论文性能优势。所有回执保留
`paper_eligible=False` 和 `physically_equal_resources_claimed=False`。

## 1. 支持的后端

| `backend` | 实现 |
| --- | --- |
| `monolithic` | 单体可恢复身份森林 |
| `component` | 确定性预算分配的可恢复分量 |
| `learned_component` | 学习式展开优先级的可恢复分量 |
| `component_completion` | 确定性分配，增加前沿完整类候选 |
| `learned_completion` | 学习式分配，增加前沿完整类候选 |
| `component_covered_completion` | 确定性分配，完整类候选采用覆盖感知容量判断 |
| `learned_covered_completion` | 学习式分配，完整类候选采用覆盖感知容量判断 |
| `node_beam` | 逐观测不可恢复 Top-K |
| `batch_beam` | 旧分量组合预选后进行批排序 |
| `joint_beam` | 保留的旧分量组合与新增批次联合排序 |
| `class_bound_joint_beam` | 联合排序，使用最大单身份类的行松弛上界 |
| `slot_bound_joint_beam` | 联合排序，使用源帧赋值的对偶覆盖上界 |
| `sparse_slot_bound_joint_beam` | 对同一源帧赋值放松作精确稀疏分解 |
| `jpda_ci` | 单历史 JPDA-CI |
| `jpda_kalman` | 单历史 JPDA-Kalman |
| `pkf` | 单历史解耦 PKF |

所有配置使用相同的正式 V2 接收、原始候选评分上下文和 car 选择规则。它们
不全具有相同的身份解码或状态时间协议，尤其不能把 JPDA／PKF 与森林的差异
归结为单独去掉恢复。适配边界见[概率基线说明](probabilistic-tracking-v2.md)。
这些本地实现也不替代经典 MHT 和公开论文方法的复现。

新增三种排名实现均不可恢复，质量／风险计算仍走原路径。它们是加强不可恢复基线的工程版本，不是三个新论文贡献。最新稀疏版本的 216 项集成回归包含上述 32 个新进程；完整真实 train 序列尚未通过，不能把夹具重复算成真实资源主表。理论、失败版本和回执见[源帧赋值上界](slot-assignment-ranking-bound-20260913.md)。

两种 `completion` 变体使用独立配置和匹配的优先级检查点，入口与结果验收均核对是否实际启用。它们已经进入测量入口，但其他后端尚未提供相同的前沿候选生成操作；直接比较不能隔离「恢复」与「候选生成」的贡献。变体边界见[前沿完整类候选](frontier-completion-20260913.md)。

两种 `covered_completion` 变体另用 `PersistentCoveredCompletionConfig` 和独立 schema，回执验收还核对 `coverage_aware_proposal_admission`。它们不改变原始势或质量公式，只修正容量准入；不能用原 `completion` 优先级检查点冒充。真实大分量风险仍未解决，见[当前动作风险诊断](decision-local-risk-20260913.md)。

## 2. 测量范围与单位

共同回放入口新增 `frame-timings.jsonl`，每行包含序列、帧、参考／决策时刻，
以及两个秒单位计时。计时使用 `time.monotonic`，分位数采用 NumPy 的线性插值。

| 字段 | 包含 | 不包含 |
| --- | --- | --- |
| `step_seconds` | 新缓存载荷读取、评分、推断、状态更新及数据库事件提交 | 初始化、源选择、外围审计与预测文件写入 |
| `frame_seconds` | 序列切换、源选择、step、外围审计与缓冲文件写入 | 本行计时记录写入、进度打印、最终关闭、全局启动 |
| `child_wall_seconds` | 启动等待段、子进程导入／校验／回放、父进程日志开关 | 父进程随后逐产物回读验证 |
| `process_peak_rss_bytes` | 子进程完成回放和源码复核后的进程内存高水位，单位字节 | 最后标记序列化、解释器退出、父进程内存、GPU 显存 |
| `database_bytes` | 最终关闭后的全部序列数据库文件大小 | 日志、预测、审计和检查点 |
| `retained_artifact_bytes` | 子进程结果目录中的全部普通文件逻辑大小 | 父进程日志、输入缓存、检查点、文件系统实际分配块 |

每次重复都启动新解释器，避免沿用同一 Python 进程此前运行的 RSS 高水位。
macOS 的 `ru_maxrss` 按字节读取，Linux 按 KiB 乘 1024；其他平台拒绝执行。
RSS 不是采样瞬时值，也不等于设备物理内存占用或容器整体内存。数据库大小
不是瞬时磁盘峰值；训练和检测成本均不在上述在线回放指标内。

只支持 CPU。每个子进程设置 BLAS／OpenMP 环境线程数为 1，并将 PyTorch 的
计算及 inter-op 线程数设置为 1。记录解释器摘要、运行库版本、主机、CPU 数量
和 PID；同一测量批次的运行环境必须一致。没有控制 CPU 睿频、热状态、操作
系统页缓存或其他进程，也没有证明独占计算节点。
本轮实际验证平台为 macOS arm64；Linux 分支尚未在计算节点执行验证。

计时中的文件写入是普通缓冲 I/O，不是额外 fsync 的耐久性基准。已有数据库
事务的同步策略不变。不得把上述帧延迟称为含检测器、实际网络和全部输出刷盘
的端到端传感器延迟。

## 3. 输入、重复和失败条件

正式入口复用完整官方 SPD val 日程检查：21 个序列、3316 次输出、7189 个源帧。
每个规范化的后端／配置组合必须提供 `1337 / 2027 / 3407` 三个完整 train
检查点；同一种子在不同后端使用完全相同的身份评分器。学习式分配还要求绑定
相同配置、身份模型和上游生产者的同种子优先级检查点。

配置按固定 `order_seed` 生成随机完整重复轮次：每轮每个配置恰好执行一次。
顺序在创建首个子进程前写入 `plan.json`，不根据已观察到的耗时改序。保留全部
帧和重复，不删除 warmup、慢帧或选择最快一次；也不清空操作系统页缓存。

每个子进程完成后，父进程重新读取并核对：

- 计划、预测、审计、逐帧计时、数据库和完成标记的摘要及配置绑定。
- 逐帧计时与审计的精确帧覆盖；从原始计时重新计算 p50、p95、p99、最大值。
- 相同种子跨方法的逐事件原始观测及行势摘要，不能仅凭模型名称判为相同输入。
- 相同配置重复运行的预测文件字节；计时和数据库文件不要求跨重复字节一致。

源码清单包含研究模型包、工具目录及入口依赖，不假定脏工作树等于某个 Git
提交。该清单是本地完整性证据，不是外部签名、第三方认证或全部运行库源码锁。

任意超时、子进程异常、源码／产物变更、输入差异或重复预测差异会停止后续
配置，并生成 `failure.json`。已经完成的逐次记录保留，但不生成整个对照完成
回执。超时是单个子进程的总运行上限，不是每帧实时截止期限或相同计算预算。
失败重跑必须使用新目录；当前没有自动续跑或跳过失败方法的功能。

## 4. 运行配置

所有字段都必须在正式实验前确定。JSON 顶层字段如下：

| 字段 | 内容 |
| --- | --- |
| `kind` | 固定为 `rbf_cpu_resource_sweep_v1` |
| `cache` | 含 `path` 和 `sha256`；摘要指向缓存 `manifest.json` |
| `schedule` | 含 `path` 和 `sha256`；摘要指向完整 SPD val 日程 |
| `jobs` | 1 至 256 个显式任务；每种后端／配置须覆盖三个种子 |
| `repetitions` | 正式入口为 3 至 100 |
| `order_seed` | 预先固定的整数，不用于选择性能最优的运行顺序 |
| `timeout_seconds` | 单子进程的正、有限超时值，包含启动和输入校验 |

每个任务包含唯一 `id`、上表中的 `backend`、`configuration`、`checkpoint`。
`checkpoint` 同样用 `path` 和 `sha256` 绑定 `checkpoint.json`。`configuration`
接受对应持久化配置类的字段；嵌套 `state` 和 JPDA 的 `inference` 使用各自配置
类的字段，缺省字段补为当前默认值后写入完整计划。无效字段或未使用的预算
选项会报错，不悄悄忽略。不同宽度或预算是不同配置，均须覆盖三个种子。

只有 `learned_component`、`learned_completion` 和 `learned_covered_completion` 额外提供 `allocation_checkpoint`，同样绑定目录与
清单摘要。完整真实检查点尚不存在时，不用夹具路径编造正式运行配置。

将 `RBF_RESOURCE_SPEC`、`RBF_RESOURCE_SPEC_SHA256` 和 `RBF_RESOURCE_OUTPUT`
设置为实际冻结的配置路径、文件摘要和新的结果目录后，在 transvision 根目录
运行：

```sh
python tools/event_track_v2x/run_resource_sweep_v2.py \
  --spec "$RBF_RESOURCE_SPEC" --spec-sha256 "$RBF_RESOURCE_SPEC_SHA256" \
  --output "$RBF_RESOURCE_OUTPUT"
```

命令没有读取 GT、评价指标、训练、检查点选择或 ClearML 发布功能。SPD val
仍是已见研究集，不称为未见确认集；SPD test/test_A 继续排除。本入口不能代替
V2V4Real 冻结后的官方 test，也不处理 GPU 的同步计时或显存预算。

## 5. 本轮证据

使用实际训练过的合成身份检查点，并实际拟合一个合成优先级检查点，九个
后端在新进程中各重复两次。运行结束后又由独立调用重新验证全部 18 份产物：
源观察／行势摘要均相同，重复预测一致，实际记录到 18 个不同 PID。夹具每次
只含 2 帧，不能用其 p99 或 RSS 为真实长序列资源优劣排序。

[保留的夹具总回执](../../work_dirs/recover-before-fuse/resource-sweep-fixtures-20260913/test_all_nine_backends_in_new_0/sweep/receipt.json)
绑定逐次预测、计时和数据库。测试曾故意改坏一份计时以检查篡改拒绝，随后
恢复原字节；最终独立回读再次通过。这些文件不外发。

最终相关回归为 **42 通过、0 失败、0 跳过，20.72 秒**，覆盖资源入口、共同 V2
回放、概率基线接收、实际夹具训练、固定 beam、分量接收和优先级训练。还包含
实际失败子进程、注入的超时及跨后端输入差异、夹具不能冒充正式训练的检查。
后两种注入测试不冒称真实计算超时或真实数据输入缺陷。

前沿候选扩展后，11 项资源入口测试通过，耗时 25.95 s。其中跨后端检查实际启动 22 个新进程，核对相同身份势、重复预测、逐帧计时和数据库绑定，并拒绝把旧分配器检查点用于新变体。该阶段使用两个分别拟合的夹具优先级模型，不是实际完整 train 优先级训练。JUnit 为 `/private/tmp/rbf-teacher-probe.PeVfiw/completion-resource-sweep-tests-20260913.xml`，SHA-256 为 `8688daa2e4ed652578e5a6daa9310ea8362a129c3966a614477b6e8e3a2393d2`。这 11 项与早期资源回归重叠，不能相加为独立测试数量。

覆盖感知扩展后，11 项资源测试通过，耗时 27.87 s；13 个后端各重复两次，共 26 个新进程。三个分别拟合的夹具优先级模型绑定各自变体，同一身份检查点产生相同行势；跨版本优先级混用被拒绝。JUnit 为 `/private/tmp/rbf-teacher-probe.PeVfiw/covered-resource-sweep-tests-20260913.xml`，SHA-256 为 `03ebd2421c4938bb3b71b94059a1af2aa7aa4127c19b3c9e1affbc83c7b72968`。本轮仍仅是夹具回归，不将叠加后的测试次数或后端数写成真实论文比较结果。

[最终 JUnit](../../work_dirs/recover-before-fuse/resource-sweep-regression-20260913-v3.xml)
与此前 20 项、41 项阶段回归重叠，数量不能相加。本轮未再次运行完整 EventTrack
套件；上一轮 1538 通过、1 跳过属于加入资源计时前的版本。

```text
transvision/models/event_track_v2x/resource_sweep.py
0f2cf1a15c063a97cff3cd955ff18cafaea52039d7541d0c6a68697d93ed415d
tools/event_track_v2x/run_resource_sweep_v2.py
14bb1378445a3957aa0ac8c3398fdc7438474c5873118e79ffcc333883cc857b
tools/event_track_v2x/run_persistent_forest_v2.py
e0b19631cd35e94857283347c85d90b8304c6846a2abf09b2b16dbde9e60912d
tests/event_track_v2x/test_resource_sweep.py
7c2092e703456ff0e0b30026e88e2079437cd547f716228ce6cf54e11de85e17
resource-sweep-regression-20260913-v3.xml
715024fbae84199de67807926b97a4f7d67674b83c168dbdeacf49713f5414e7
保留的夹具总回执
5cb3c61be0de1ce4df7e82436df247223dcec22f28b9c7a0bbf7550fa7bfa0cc
```

## 6. 尚未完成的真实实验

2026-09-13 只读核验了计算节点 `10.100.35.112`：既有完整 train 缓存清单
SHA256 仍为 `1137740ecdf2aca7372536998ac485585ca89f4792bf68e287fa75d2e07a6fa0`，
但当前目标路径未发现新的 `train_forest_identity.py`。本轮未上传源码或启动
远端任务，已请求研究源码同步与 car-only SPD 训练／val 实验的明确授权。

新身份模块和计算分配模块的完整真实训练、严格同资源性能曲线、独立强单端及
经典多假设基线、双数据集方法效果仍未完成。资源测量不等于预算已经匹配：
最终仍需固定候选配置、在相同资源约束下比较身份指标，并保留失败和超限结果。
所有研究改动保存在 transvision，论文构建工具保持原位。
