# SPD 完整 val：固定 JPDA／PKF 基线验证

本次将三个已固定的单历史基线推进到 SPD 官方完整 val，不改变检测器、身份模型、
关联解码器或阈值。种子 1337、2027 的六个完整运行已正常退出，各完成 21 序列、
3,316 帧，并通过逐帧数值、身份历史审计和原生评价。种子 3407 的三组正在运行。
九组汇总尚未完成。第 6、7 节记录完整运行证据；
此前的预检、启动修正和测试保留为历史。

## 1. 实验范围与冻结输入

方法为 JPDA-CI、JPDA-Kalman 和解耦 PKF，均采用已固定的 LBP 与 `joint-map`。
每个方法使用三个已完成四卡 A100 训练的身份检查点。共计划 9 个完整运行，
每个运行覆盖 21 个序列、3,316 个输出帧；7,189 个源缓存帧全部校验，未配对帧
不补入输出日程。先执行种子 1337 的三个方法，再执行 2027 和 3407，不按 val
结果选择后续种子。三个检查点的训练输入、配置及选择边界见
[四卡训练验收](clearml-ddp-training-20260913.md)。

| 输入 | SHA-256 |
| --- | --- |
| val 缓存清单 | `66c58025bd79ea674ff676f1df8d76e7812c8b1d2b62bc54e8de3ced0bc3aca8` |
| 无 GT 预测日程 | `2c8999ecbe2ab98bedf13ba2da4b22bd6167eca07b368a99db148abf36de982a` |
| seed 1337 检查点清单 | `70609d3a7bf330d0d56b92150d0f97db511dfe4545c07cc193e993d0b77a0e96` |
| seed 2027 检查点清单 | `0d0289fc153ea66b5e022c4cc1dd02f1d4f0e552599a9d4f01e924e5dda0a33f` |
| seed 3407 检查点清单 | `96d1db472bef678112de2187046c7d8002df35bfe0dfeb153c746948d59f5a36` |

缓存与日程的远端源目录为
`10.100.35.112:/home/lbin/Desktop/eventtrack-cooptrack-runtime-20260911/tracking-validation-20260912/`。
本地实验根为 `/private/tmp/rbf-spd-val-20260914.Dwpzln/`。复制方向是内部管理机到
本地私有目录。缓存及日程复制共 972,980,428 字节，14,380 个文件；该复制不含
test、GT 全文或已有完整预测流，也没有上传新源码。正式加载器已完整重算全部
缓存载荷并核对 7,189 帧、21 序列和三个检查点的冻结上游身份。

评价器所需的 GT 另行复制到 `/private/tmp/rbf-spd-val-gt-20260914.8iV7ZO/`，
两份文件共 21,737,047 字节，不进入推断参数或缓存目录。清单 SHA-256 为
`94675ac8585d893195b7e801b754f62ca4ee3524cf852e05e53121fe2cc4084a`，GT 流为
`94908e0010003b42a5ed2c35ccc894005fe8d40cadfe57653871bbd95e0a9a76`，均已与原封存
值一致。输入预检时没有计算新指标；第 6 节的正式评价另行记录。GT 未上传到 ClearML。

仅报告 SPD 官方三类映射中的 car，即 Car／Truck／Van／Bus 归入 car；不报告
pedestrian。这不等于 V2V4Real 的 raw Car-only 协议。SPD val 已参与研究，不能
称为未见确认集。本次不读取 SPD test／test_A，也不读取 V2V4Real test。

## 2. 入口与独立审计

`run_probabilistic_tracking_v2.py` 继续使用原有完整 val 日程、缓存和检查点校验。
本次增加以下结束条件，不修改教师或跟踪核心：

- 绑定全部模型模块、包初始化文件及直接调用的工具源码，不只绑定所选后端顶层文件。
- 绑定实际加载的缓存清单、日程、检查点清单和权重文件；开始与结束分别核验。
- 要求进程启动前固定哈希种子和各计算库线程环境，实际设置并记录 PyTorch 的
  intra-op／inter-op 线程数为 1；记录解释器摘要、版本、主机和进程 ID。
- 只有完整完成 21 个序列、3,316 帧且来源未变，才生成 `full-validation-receipt.json`。
  内层 `receipt.json` 或部分预测不能代替该回执。

新增 `audit_probabilistic_validation.py`，在独立进程中逐帧检查预测、审计、计时与
冻结日程一致，验证预测和审计的连续摘要链、car-only 输出、唯一输出 ID、
每个 scan 的匹配因子与概率边际、一对一锚点、求解器限额及状态更新计数。
同时重算延迟分位数并核验全部序列数据库摘要。

独立审计不读 GT，也不计算 HOTA 或 IDF1。LBP 收敛残差、边际归一化误差均不是
完整历史后验误差或真实身份风险上界。

## 3. 执行与完成判据

以下命令在 canonical transvision 中执行。环境是 CPU 冻结模型推断，不进行参数
更新；后续真实参数训练仍限于 ClearML A100、每任务至少四卡。

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  /private/tmp/eventtrack-v2-checks.bYZm1l/bin/python \
  tools/event_track_v2x/run_probabilistic_tracking_v2.py \
  --cache /private/tmp/rbf-spd-val-20260914.Dwpzln/detection-cache-v2 \
  --cache-sha256 66c58025bd79ea674ff676f1df8d76e7812c8b1d2b62bc54e8de3ced0bc3aca8 \
  --schedule /private/tmp/rbf-spd-val-20260914.Dwpzln/schedule.json \
  --schedule-sha256 2c8999ecbe2ab98bedf13ba2da4b22bd6167eca07b368a99db148abf36de982a \
  --checkpoint /private/tmp/rbf-a100-result-1337.besRjh/seed-1337 \
  --checkpoint-sha256 70609d3a7bf330d0d56b92150d0f97db511dfe4545c07cc193e993d0b77a0e96 \
  --device cpu --association lbp --anchor-decoder joint-map --update-rule jpda-ci \
  --output /private/tmp/rbf-spd-val-20260914.Dwpzln/seed-1337-jpda-ci-v2
```

另两个方法只将 `--update-rule` 改为 `jpda-kalman` 或 `pkf`，并使用对应的新输出
目录。其余种子沿用同一配置，检查点路径分别为
`/private/tmp/rbf-a100-result-2027.yOejRx/seed-2027` 和
`/private/tmp/rbf-a100-result-3407.txs5K6/seed-3407`，摘要使用第 1 节固定值。
已存在目录不得覆盖；发生失败时保留全部输出与失败证据，不跳过该方法或静默重试。

### 本次启动修正

第一次启动通过临时命令封装调用 Python 函数，将输出目录保留为字符串。
检查代码发现，最终使用 `/` 拼接回执文件名时需要 `Path`；继续运行将无法完成
回执写入。因此主动向本次三个已核对 PID 的进程发送 SIGINT，均确认以 130 退出，
而不是因日志暂时不更新而重启。

| 方法 | 中止前完整输出帧 | 原输出目录 |
| --- | ---: | --- |
| JPDA-CI | 48 | `seed-1337-jpda-ci-v1` |
| JPDA-Kalman | 46 | `seed-1337-jpda-kalman-v1` |
| PKF | 45 | `seed-1337-pkf-v1` |

三个 `failure.json` 记录的直接异常是 `KeyboardInterrupt`，不能称为已观察到
最终路径异常或算法失败。原日志和部分结果全部保留，未计算指标。第二次启动改为
上述公开 CLI，使 `argparse` 正确解析 `Path`，仅将前三组输出目录改为 `-v2`；
没有修改推断源码、数据、阈值或方法配置。当时其余六组尚未启动，计划目录仍为 `-v1`；
当前状态见第 6 节。

原始九组计划 `campaign-preflight-v1.json` 的摘要为
`e6ebf9bb853605e6f4c8b4f09b2efd5370e16973eb9aac5b3cacaaa5e2df83ae`。
记录此次中止原因和目录调整的 `campaign-preflight-v2.json` 摘要为
`19c3861c52f9b7375fa39958db522a965c6728e25f46a1ae23576577787f7715`。
两份计划均位于实验根，绑定同一组 113 个推断源码文件。

完整结束后，先计算该运行 `full-validation-receipt.json` 的 SHA-256，再执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /private/tmp/eventtrack-v2-checks.bYZm1l/bin/python \
  tools/event_track_v2x/audit_probabilistic_validation.py \
  --run <完整运行目录> --receipt-sha256 <实际回执摘要> --output <新的审计目录>
```

只有完整独立审计通过后，才使用单独的原生评价器环境评测 car。后续三方法比较
必须核对同一种子的实际因子流、输入、完整日程和环境，而不只比较配置文件。
三个种子的结果均需保留，不选择最佳种子。统计汇总和完整评价尚待实际执行。

### 原生评价与九组汇总

`evaluate_probabilistic_validation.py` 仅接受完整运行及其独立审计。它重新校验
运行、数据库、预测、检查点声明、实际算法记录与全部推断源码；随后在单独的
原生评价环境中读取固定 GT。部分运行、内容变化或审计对应另一运行时均拒绝计算。

```bash
PYTHONDONTWRITEBYTECODE=1 /private/tmp/eventtrack-evaluator-v1.zT0Eh9/venv/bin/python \
  tools/event_track_v2x/evaluate_probabilistic_validation.py \
  --run <完整运行目录> --receipt-sha256 <实际回执摘要> \
  --audit <独立审计目录>/audit.json --audit-sha256 <实际审计摘要> \
  --ground-truth /private/tmp/rbf-spd-val-gt-20260914.8iV7ZO \
  --output <新的原生指标目录>
```

实际输入先保留 `class_index == 0` 且 `raw_score >= 0.05` 的检测，再按分数和
索引排序，为每个源帧取前 64 个。旧 source-ablation 协议则先对所有类别取前
64 个，再计算 car 指标。两者的原生指标引擎和 ROI 相同，但检测筛选顺序不同。
因此新入口单独命名协议，不沿用旧包装器的输入声明，也不将这批结果与旧单端
结果直接宣称为同输入对照。此处澄清元数据，没有改变正在运行的推断或历史结果。

`compare_probabilistic_validation.py` 要求固定计划中的三个方法 × 三个种子全部
完成，并重新检查每份指标的运行来源、文件摘要、评价协议、GT、配置和环境。
同一种子的三个方法必须使用相同实际因子流；不同种子的因子流允许不同。

```bash
PYTHONDONTWRITEBYTECODE=1 /private/tmp/eventtrack-v2-checks.bYZm1l/bin/python \
  tools/event_track_v2x/compare_probabilistic_validation.py \
  --campaign /private/tmp/rbf-spd-val-20260914.Dwpzln/campaign-preflight-v2.json \
  --campaign-sha256 19c3861c52f9b7375fa39958db522a965c6728e25f46a1ae23576577787f7715 \
  --report <第一份指标目录>/report.json <实际摘要> \
  --report <其余八份各自的指标目录>/report.json <各自实际摘要> \
  --output <新的九组汇总目录>
```

上例中的 `--report` 必须实际重复九次，每次提供对应的路径与摘要，不能将
占位符直接运行。汇总包括 11 个指标的三个种子原值、均值、样本标准差及范围，
并给出相对 JPDA-CI 的逐种子和逐序列差值。标准差不是置信区间；缺组时不生成
完整比较，不删除失败方法，不选择最佳种子。完整硬身份历史不变性尚需另行审计，
不能仅凭因子流相同认定；同资源或仅恢复开关变化的因果结论也不在该汇总的范围内。

### 独立核对身份历史与状态变化

`audit_probabilistic_validation_identity.py` 接受同一个种子的三个完整运行，以及
各自已通过的独立数值审计。该入口不读取 GT、不重新推断，也不打开可写跟踪器。
SQLite 使用 `mode=ro` 和 `query_only`。每个序列逐帧核对以下内容：

- 数据库事件与已发布的预测、审计记录一致，末帧审计与数据库最终摘要一致。
- 从最终 `identity_anchors` 的观测前缀重算每一帧历史摘要，检查不可变根映射。
- 每个新观测恰好属于一个 scan；父节点先前已处理，代表父节点来自合法势函数边，
  同一身份根不能在相同源帧中重复占用，源输入不得晚于决策时刻到达。
- 根据原始观测成员、分数和时间，独立重算输出 ID、成员列表、新生时间、最近更新
  时间、分数及生命周期筛选，并与已发布记录逐项比较。

该审计报告三类独立指纹：条件 scan、历史身份映射、去掉均值与协方差的输出。
即使发现不同方法的身份映射确有差异，也会明确报告差异，不将「必须一致」作为
成功条件。只有结构或证据不合法时拒绝审计。报告状态为 `complete` 表示检查完成，
不自动表示身份不变量成立；必须另看三个一致性字段。

```bash
PYTHONDONTWRITEBYTECODE=1 /private/tmp/eventtrack-v2-checks.bYZm1l/bin/python \
  tools/event_track_v2x/audit_probabilistic_validation_identity.py \
  --run <JPDA-CI 运行目录> <完整回执摘要> <独立审计文件> <审计摘要> \
  --run <JPDA-Kalman 运行目录> <完整回执摘要> <独立审计文件> <审计摘要> \
  --run <PKF 运行目录> <完整回执摘要> <独立审计文件> <审计摘要> \
  --output <新的身份审计目录>
```

**实现特定的不变量。** 对按到达顺序处理的源 scan \(s\)，令 \(F_s\) 为固定的
原始关联因子，\(H_s\) 为已提交的硬身份历史，\(X_s^{(u)}\) 为更新器 \(u\) 的
连续状态。当前实现的关系为

\[
H_s=T(H_{s-1},F_s),\qquad
X_s^{(u)}=G_u(X_{s-1}^{(u)},H_{s-1},F_s,Z_s).
\]

这里 \(T\) 不读取已融合的 Gaussian 状态。若初始历史相同，原始因子、scan
顺序、约束、解码器及确定性并列处理相同，并且各运行均成功完成，则所有更新器的
\(H_s\) 相同。证明为对 scan 的归纳：相同 \(H_{s-1}\) 将原始父节点聚合成相同的
身份根、条件因子和合法配置，确定性解码给出相同新锚点，故 \(H_s\) 相同。

当前生命周期只依赖原始成员、分数和源时间，因此同一身份历史也产生相同非状态
输出。不过原生跟踪指标是 \(M(H,X;G)\)，其中 \(G\) 为 GT；改变 \(X\) 仍可能改变
ROI 内的预测集合和几何匹配，进而改变 HOTA、IDF1 或 IDS。若改用融合状态生成
关联因子、门控或生命周期，此不变量不再适用。它不是一般 JPDA／PKF 定理，也不是
可恢复方法的创新或身份正确性证明。完整 val 是否符合这些条件仍需实际审计。

## 4. 已验证内容与限制

四个相关测试文件合计 61 项通过，耗时 5.20 s。JUnit 为实验根下的
`validation-regression-v1.xml`，SHA-256 为
`b65a1aca1639ca60fffd189a41f774db01871377b781b2dda0cf49066883b570`。
早先的 22 项入口检查包含在本轮结果内，不重复相加。

补充原生评价与九组汇总后，入口、审计、评价、汇总四个测试文件共 95 项通过，
耗时 4.96 s。对应 `metric-integration-tests-v2.xml` 的 SHA-256 为
`51e58c0e0e732c78363a24eb8ba2543ee2b7d86573a423a4ea777cc985e6b819`。
这与上述回归集合有重叠，不将通过数相加。评价绑定和九组汇总的合成用例仅验证
软件约束，不是 3,316 帧真实指标或九次真实运行的结果。

当前原生评价包装器已在不导入 PyTorch 的原生环境完成预检，依赖校验通过。
`native-evaluator-wrapper-preflight-v2.json` 的 SHA-256 为
`90d958dd712ebd99a43399c37c9a701b65ffc69cb85cf04cb51c1de3b95d68ce`，
绑定评价器源码 `217de779b0700bb23c55a32f1d9f110c53737b0369e65fc6ad136045159c7ee4`
及协议 `9a3cf99c2c1ee87fe096c5df03c1aa2ff9199039fa8f76ba4fdd22356033bca8`。
前一版包装器预检保留为历史，不能代替当前源码的预检；该预检本身没有计算真实预测指标。

加入身份历史审计后的六个相关测试文件共 156 项通过，耗时 5.05 s，包含前述
95 项测试及旧身份指纹测试，不累计计数。新用例还覆盖数据库事件不一致、非法
代表父节点、观测晚到、生命周期记录变化、同种子三方法配置不一致和跨序列读取。
JUnit 为 `identity-metric-integration-tests-v2.xml`，SHA-256 为
`86417519c1c836f3b8932944d13283ee5c012c5fd1f7b30562882248a7428537`。

新审计的读取逻辑已用于三个完整 train `0001` 基线，每个运行 183 帧、4,671 个
观测。历史摘要、数据库末审计和原始输出成员重建均通过。条件 scan、身份历史和
非状态输出完全一致；相对 JPDA-CI，Kalman 与 PKF 各有 182 帧预测状态不同。
这与旧 train 审计结论一致，没有重新计算指标，也不能当作完整 val 或九组验收。
当前源码 SHA-256 为 `19f91ec872dd8927a23bfa9f1b4f9a0aeace0d37d670b782c6aa4846d3e83918`，
真实兼容性预检 `identity-real-train0001-preflight-v2.json` 的摘要为
`21efaf16c9a7d93c229ebcab9b3f6b15e0cd03d9cf42401bd0f10338deb33f46`。

独立审计器还重新读取已完成 train `0001` 的三个真实基线，各 183 帧、259 个
条件 scan，均通过事件、摘要链及数值检查；共同原始因子流摘要仍为
`f336a1a94ed42b4040d64de690d36631390b95ff5a381e27b19bcf91c3258a27`。
这不是新增 val 结果，也没有重跑推断或训练。

本次 CPU 实验与教师采集共享主机，不具备独占资源或重复测量条件。记录的耗时、
内存和数据库大小仅用于运行诊断，不能据此宣称同延迟、同内存或同 FLOPs 优势。
三种方法是本地协同适配，不是公开 JPDA／PKF 程序的端到端复现；它们与可恢复
后端的状态、时间处理不同，不能作为只改变恢复开关的因果消融。

论文仍需完整恢复方法、强单端和同资源多假设对照、第二真实数据集，以及真实
恢复事件的解释证据。通过本记录中的测试或开始 val 推断，都不等于这些要求完成。

## 5. 历史启动与本地保全

第二次启动的早期快照中，三个进程均已完成前 2,100 帧，进度位于 val 序列 `0063`；
各运行目标仍是全部 3,316 帧，不按该进度推断最终结果。三份实际 `plan.json`
均与九组预检计划的源码、模型、配置一致。启动核验回执为
`launch-verification-v2.json`，SHA-256 为
`554ef244027fcd2857d6af0ae3c8fda771242e576f97f63b2a99853f8084fd56`。

独立原生评价器已校验完整 GT 与预测日程逐帧一致，全部运行依赖版本及源码树
匹配原封存值，car 解析用例通过。ROI 内 car 标注数为 26,648，包含第 1 节约定
的官方映射；没有计算这三个新预测流的指标。预检回执 SHA-256 为
`3ddf7778d93340119fd8c0f9167e9095a3423d4925c1c052de7b8a63717b309a`。
运行中出现不可写字体缓存警告，进程最终退出码为 0；未修改系统字体或评价器源码。

本地保全目录为
[spd-probabilistic-full-val-20260914](../../work_dirs/recover-before-fuse/spd-probabilistic-full-val-20260914/manifest.json)。

| 文件 | 大小／内容 | SHA-256 |
| --- | --- | --- |
| `bound-sources-v1.tar.gz` | 445,531 字节，125 个源码或测试文件 | `c9ed3073b283c8dcaf16dfe9f4f370ce9bc28ac3ee0b124dd061b55e922a9db5` |
| `audits-v1.tar.gz` | 13,246 字节，14 个计划、测试或启动审计文件 | `c050a6448a9747b8c553895e96f69454d9a0f9576e587007dff41e81d797874c` |
| `manifest.json` | 两份归档的逐成员摘要与范围 | `c77aa1cc53a56a4b0ec835eb372a82b3d3c0a4b57224321bb65a0c3636d5864f` |
| `evaluator-preflight-v1.json` | 后追加的独立评价器预检，不在上述两份归档内 | `3ddf7778d93340119fd8c0f9167e9095a3423d4925c1c052de7b8a63717b309a` |

两份归档均已逐成员与原文件比较，未包含 GT 全文、模型权重、完整预测流或
数据库。它们保全本轮实现及启动证据，不是可独立运行的完整数据包；全量复核仍需
原缓存、检查点、运行输出及固定评价器环境。本次没有 ClearML 发布、Git 提交或推送。

后续新增的四个评价／汇总源码与测试文件单独保全在
[`metric-integration-v1/manifest.json`](../../work_dirs/recover-before-fuse/spd-probabilistic-full-val-20260914/metric-integration-v1/manifest.json)。
源码归档共 9,448 字节，SHA-256 为
`cb32f77426f2951083948fbea23ec1d00b7e00b906e190b9bcdcb4c594707aca`，
四个成员均与工作文件逐字节一致；同目录保留当前 95 项测试回执及原生包装器预检。
该新增归档同样不含 GT、权重、预测、数据库或内部资源快照，不覆盖前述历史归档。

身份历史审计的独立保全目录为
[`identity-integration-v1/manifest.json`](../../work_dirs/recover-before-fuse/spd-probabilistic-full-val-20260914/identity-integration-v1/manifest.json)。
其中源码归档为 10,456 字节，含新审计器、复用的历史指纹器及两个测试文件，四个
成员均已逐字节核对，SHA-256 为
`54fbc641ac9f45474c418e22da9546b483bb7834c89a19c049a685e9f3207c83`。
同目录保存 156 项软件测试回执。真实 train 兼容性详细回执保留在本地实验根，
此代码归档不包含其事件记录，也不包含 GT、权重、完整预测、数据库或内部资源快照。

## 6. 种子 1337 完整运行与身份审计

三个 CLI 进程均以退出码 0 正常结束，各覆盖完整 21 序列、3,316 帧，且生成
`full-validation-receipt.json`。不是从部分预测文件或进程状态推断完成。
独立逐帧审计检查 4,965 个条件 scan，三个实际因子流的共同摘要为
`e8ca0d90e220ea219958c5b2e6aa9425aefb5fbdd9c0d605f69055824d567930`。

以下文件均位于本地实验根 `/private/tmp/rbf-spd-val-20260914.Dwpzln/`，没有新发布操作。

| 方法 | 完整回执 SHA-256 | 独立 `audit.json` SHA-256 |
| --- | --- | --- |
| JPDA-CI | `173385770346deb8564ddde75ea6c0f225b118b5364f52b32eacbcb2b829942e` | `fbbad927ea1d820cc07058bc5088ef65878c1b8d9b736e7f2b3574f900cbd03c` |
| JPDA-Kalman | `6f7c4aaa5869955c124bca33ae3ff2e05ac937122a91d9d9d544736880ca6265` | `6f0f7cbf5dee5b5a911ed9d8715df25d35f768f0a6ad23c230b7669242840bf4` |
| PKF | `c0e402a47cbbd19e095d7046b6db1a24e836d21d126e3501632f84420dd238f1` | `099c924b677c2c863613971f7e546a67072741ce09f76bc1bba6aa6ca2b5e06c` |

运行目录为 `seed-1337-<方法>-v2`，独立审计目录为 `audit-seed-1337-<方法>-v1`；
路径中的方法值分别为 `jpda-ci`、`jpda-kalman` 和 `pkf`。

另行完成只读身份审计，覆盖全部 82,952 个观测，重建各帧身份前缀及发布成员。
全部序列的条件 scan、历史身份映射、非状态输出均相同。回执为
`identity-audit-seed-1337-v1/identity-audit.json`，SHA-256 为
`fced82d38e59f47b6cc23e19ed71b8fa8d579f54b9eb4e8d0c6b1d1288648587`。
该结果验证了第 3 节实现特定的不变量，不证明身份本身正确，也不证明恢复方法有效。

相对 CI，Kalman 和 PKF 各有 3,297 帧的状态载荷改变，但 ID、成员和其他非状态
输出均不变。每组共有 105,004 个 car 输出框，没有空输出帧。

### 原生指标与解释边界

三个原生评价进程均以退出码 0 完成。后续重新读取报告、原始指标、运行环境和
完整推断证据，绑定检查通过。独立使用 `2*IDTP/(2*IDTP+IDFP+IDFN)` 重算 IDF1，
使用 `1-(FN+FP+IDS)/GT` 重算 MOTA，均与保存结果一致。HOTA／AssA／DetA 使用
原生跨序列合并，不使用序列均值或汇总值乘积替代。另行重算三组各 19 个阈值的
曲线均值及逐阈值 `HOTA = sqrt(DetA * AssA)`，与原生结果的绝对差均不超过
`1e-12`；该逐阈值关系不意味着汇总 HOTA 等于汇总 AssA 与 DetA 乘积的平方根。

以下是固定种子 1337 的完整 val 结果，不是三种子均值。

| 指标 | JPDA-CI | JPDA-Kalman | PKF |
| --- | ---: | ---: | ---: |
| HOTA ↑ | 0.1775477838 | 0.1781801208 | 0.1767554141 |
| AssA ↑ | 0.2902290148 | 0.2969247987 | 0.2981020252 |
| DetA ↑ | 0.1110944573 | 0.1095495527 | 0.1076821363 |
| IDF1 ↑ | 0.1446986036 | 0.1435855847 | 0.1400306577 |
| AMOTA ↑ | 0.3066795058 | 0.3175648573 | 0.3083582963 |
| AMOTP（m）↓ | 1.3100895508 | 1.3162873449 | 1.3186849213 |
| MOTA ↑ | 0.2719153407 | 0.2751425998 | 0.2703392375 |
| IDS ↓ | 26 | 20 | 18 |
| Frag ↓ | 382 | 243 | 228 |
| FP ↓ | 3,204 | 3,411 | 3,603 |
| FN ↓ | 16,172 | 15,885 | 15,823 |

FP／FN／IDS／Frag 来自 nuScenes 引擎选定的标准 MOTA 工作点；IDF1 来自
TrackEval 的 BEV IoU 0.5 匹配，两套计数不可混用。TrackEval 的 IDTP 分别为
7,145／7,082／6,897，IDFN 为 19,503／19,566／19,751，IDFP 为
64,964／64,915／64,962。固定 ROI 中的 GT 数均为 26,648。

Kalman 的 HOTA 和 AMOTA 高于 CI，但 IDF1、DetA 和 AMOTP 更差；PKF 也没有
全面优于 CI。指标并非同向变化，不据此挑选最有利指标或改变后续实验。
虽然 IDS 分别为 26、20、18，三组内部身份历史仍完全相同；该差异涉及连续状态
对 ROI 筛选、几何匹配和原生评价的影响，不能解释为恢复了不同身份分支。

| 方法 | `report.json` SHA-256 | `metrics.json` SHA-256 |
| --- | --- | --- |
| JPDA-CI | `c97c15d4e93d6e1e3fc8d6618fc4b284509ad57966fca1c26a3bff3bd4946b23` | `cbe5a9603883270afb702aa1bf031b890943bfd4358e334acdb8c3777f4a873f` |
| JPDA-Kalman | `9c247751e8c723ccb5b2240e60bf0d7030f022b405b72fc130f5a60f35a955c8` | `960f67a200a56baa8e9b011609e17f0e48c25ebc00e81a1cd4785a5cefcda1b7` |
| PKF | `837a118aba83b4f30495d48515717d848a93c38db946a8f014a574c60d0f4498` | `2b999840bc52c1dbd737582428540e8c58734fd7ceaa9d4b967822cc8b5ab156` |

指标目录为 `metrics-seed-1337-<方法>-v1`，三组协议摘要均为
`9a3cf99c2c1ee87fe096c5df03c1aa2ff9199039fa8f76ba4fdd22356033bca8`。
读取指标不会改变已启动的 2027 配置。当前结果可以作为带限制的单种子适配基线，
不能作为恢复收益、同资源优势、与旧 source-ablation 同输入比较或论文主结论。

### 传感器可用性核对

从全部 7,189 个缓存元数据文件重新读取时间，按每个日程的
`max(box_reference_timestamp_us, source_image_timestamp_us) <= reference + 100000`
独立重算可用性，并与三份完整审计流逐帧核对。3,316 个事件中，1,643 个车端
指定源帧尚不可用，路端为 0；1,673 个事件的两端指定源帧都可用，记录完全一致。
这不是模拟丢包率，也不代表 1,643 帧没有任何历史车端信息；它是当前固定源时刻
和 100 ms 决策条件的输入约束。物理推断耗时没有纳入该决策截止时间。

三个运行耗时分别为 5,712.356 s、5,734.445 s、5,710.723 s；峰值 RSS 分别为
335,855,616、337,854,464、339,181,568 字节。它们与教师采集共享主机，只记录成本，
不构成同资源优势或部署时延保证。

### 当时的后续运行记录

种子 2027 的三个运行已使用公开 CLI 启动，
没有根据 1337 结果改动输入、配置或选择规则，其 `plan.json` 摘要如下：

| 方法 | 种子 2027 计划 SHA-256 |
| --- | --- |
| JPDA-CI | `f6ea8779aa57af01b4166b3a147719e43778916d038f36439ada625e754582e9` |
| JPDA-Kalman | `03c3c4c1d6e820a4b5fa7a3889daeb50f26245198a873dd5f1667b455dd7fee2` |
| PKF | `27e28a3592a403024a7fe2086faf7d31e88c1b7edd2a49093c8ce45c4054ca43` |

种子 3407 尚未启动。当前完成的是九组计划中的三次完整推断、独立审计和原生评价，
不是三种子完整比较、公开方法复现或最终论文验收。

同轮教师序列 `0008` 以退出码 0 完成 196 帧，完整叶节点与来源核验通过。
教师累计为 5 条完整序列、979 个事件；并未达到 46 序列、7,445 事件。
其 `receipt.json` 摘要为 `921bcd3186d915605e7c7cd3fdcdf297b3ba5f75d6263b59c000b28f4dbb4615`，
最终单序列回执摘要为 `ecb6a585e4b1de1fc809a1b63c460f43f90d1b8af2e4ec47c99724a689cd680d`。
按固定顺序启动 `0010`（124 帧），原 `0002` 继续运行。没有把教师试算时延混入
上述普通推断成本，也没有新增参数训练或上传。
`0010` 的启动计划摘要为
`bb001afa0d85b47b59645a8cfe348d3058685d2da56d7d0b4ca6e8a5402ae3be`。

## 7. 种子 2027 完整评价与种子 3407 接续

2027 的 JPDA-CI、JPDA-Kalman、PKF 三个原进程均以退出码 0 完成 3,316 帧、
21 个序列。独立逐帧审计检查 4,965 个条件扫描；三组实际因子流摘要均为
`3ca051d99087e8b0ada973f6ffaca50ca07eaddd75ebce99dd4090e5f611588a`。
不同训练种子的势函数不同，不要求该摘要与 1337 相同。

只读身份审计覆盖 82,952 个观测：三组的条件扫描、全部历史身份映射和非状态输出
均相同。每组输出 102,759 个 car 框；相对 CI，Kalman、PKF 各有 3,297 帧状态
载荷变化。该审计不使用 GT，不把相同身份解释为正确身份。

以下为 2027 的完整 val 原生结果，不是三种子均值。car 映射、50 m 严格 ROI、
冻结引擎和计数语义沿用第 6 节；三组原生评价进程均正常退出。

| 指标 | JPDA-CI | JPDA-Kalman | PKF |
| --- | ---: | ---: | ---: |
| HOTA ↑ | 0.1776369770 | 0.1787869604 | 0.1775441883 |
| AssA ↑ | 0.2851756827 | 0.2922428230 | 0.2954020854 |
| DetA ↑ | 0.1132126289 | 0.1122271293 | 0.1097444294 |
| IDF1 ↑ | 0.1445726299 | 0.1428952772 | 0.1401277811 |
| AMOTA ↑ | 0.3058023441 | 0.3167746293 | 0.3050740264 |
| AMOTP（m）↓ | 1.3114447792 | 1.3172486270 | 1.3437584153 |
| MOTA ↑ | 0.2683128190 | 0.2682377664 | 0.2601696187 |
| IDS ↓ | 38 | 33 | 18 |
| Frag ↓ | 402 | 283 | 217 |
| FP ↓ | 3,296 | 4,352 | 3,167 |
| FN ↓ | 16,164 | 15,115 | 16,530 |

三组报告的全部输入和输出绑定重新核对通过。按各自原生计数独立复算 IDF1、MOTA，
误差小于 `1e-12`；每组另外复算 19 个阈值的 HOTA 关系以及 HOTA／AssA／DetA
三个曲线均值，共 22 项，最大绝对误差为 `5.56e-17` 以下。没有以汇总 AssA 和
DetA 的乘积替代 HOTA，也没有混用 nuScenes 与 TrackEval 的 FP／FN。

Kalman 的 HOTA、AMOTA 高于 CI，但 IDF1 更低；PKF 也没有全面改善。
三组内部身份历史完全相同，因此不能将 IDS 或 HOTA 的变化称为分支恢复收益。
仍不主张同资源优势、公开方法原样复现或与旧 all-class-top64 缓存选择直接比较。

产物仍位于 `/private/tmp/rbf-spd-val-20260914.Dwpzln/`，未新上传。
方法目录中的 `<rule>` 分别为 `jpda-ci`、`jpda-kalman`、`pkf`。

| 方法 | `seed-2027-<rule>-v1/full-validation-receipt.json` | `audit-seed-2027-<rule>-v1/audit.json` |
| --- | --- | --- |
| JPDA-CI | `1c927355a797fc8892595125df6427b53f2e73619f787429cfb0cd8b28af75de` | `349de38ca86ad6e634af3e6c1479b7241f857f6a432f3b411720500bd48ee94f` |
| JPDA-Kalman | `be97f4132291ab8993ceba66fad9ea1ad03448c79bfcc23d1118d3879507cc8d` | `36a9a4458843096da35c32d1da64f43eed4fc662ac6161f151bfba2619479d63` |
| PKF | `2fafecb1004d7ce61e0034510fae9f84166880779f5e3c8259c145d3dfe78e85` | `7d224a555af3f6ff84c0d9b34517cbc8364448779b497ee417ede49aa47dabab` |

| 方法 | `metrics-seed-2027-<rule>-v1/report.json` | 同目录 `metrics.json` |
| --- | --- | --- |
| JPDA-CI | `3b75fb2720dc55a541d5e82226e319f9cd53a1f2c34f10c72d81841a3bb83ecf` | `e6ea2ae36da7f6dddb3c76e7ad5ec9e70a843bfa8232330d5d97686203013e2d` |
| JPDA-Kalman | `c0b669730c07cdcb4da9689e03e2557a065f06c7f948c103a508f1278c532b91` | `805b8ee5ad2f32c4fe5abd00f2f50a3165f69c3a6a4cd1d35d14e8a12e61b7e3` |
| PKF | `45218d86014b59c7fe1464f32bd9db1f870e53c2ef9fb40a2f48aca7eef1582d` | `9656f693e1ef8eb35cb7bc819ac08370980196cef01914b5b456c97e6063954b` |

`identity-audit-seed-2027-v1/identity-audit.json` 的 SHA-256 为
`e0f3923aa01e0cc71bba83f5351304600f58c055e79197fd62b5383cefff2c2f`。
协议摘要仍为 `9a3cf99c2c1ee87fe096c5df03c1aa2ff9199039fa8f76ba4fdd22356033bca8`。

已按预先固定的九组计划启动种子 3407 的三个完整重放，不根据 2027 指标调整配置。
启动前重新核对全部 113 个源码绑定、计划摘要和不存在的输出目录，三组均进入
逐帧推断。这里只计作运行中，不计入完整六组之外的已完成结果。

| 方法 | `seed-3407-<rule>-v1/plan.json` SHA-256 |
| --- | --- |
| JPDA-CI | `64528d980346b62fe9cb5296a468d34a97d62993a11e50d42366e22503022b72` |
| JPDA-Kalman | `0221bc7e31ec4a2cacb2d436335f1c84de2b67aa3f5f040bb3fc3bbcf1b27947` |
| PKF | `9559d5e344d0f7a8e705a4cb4c0b059008d9630ebe0678b2cf749b6bde385414` |

本轮只做冻结 CPU 推断、审计和评价；参数训练继续限定为至少四卡 A100。
没有重新训练已完成的三个身份种子，没有修改 ClearML 队列，也没有启动优先级参数训练。
