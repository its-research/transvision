# 未完成实验的代码准备

权威范围：`/Users/lbin/thesis/tracking/recover-before-fuse/experiments.md` 和
`method.md`。本文件记录代码准备，不是实验完成或论文指标验收。
不修改已经发布的 source-freezes，不重启现有任务，不提交或推送。

2026-10-05 后续代码补齐：恢复关闭的 teacher/learned 分配及独立选择轨迹检查、
bound 消融完整 CPU 验收入口、V2V4Real official_test 原始导出器、合并 vehicle GT/校准/缓存接口、原生评价接口、
公开基线真实输出收集均有新增代码。下文早期准备记录保留其当时范围；真实实验及
尚缺外部资产必须继续分别核对。最新 vehicle 决定回执为
`test/recover-before-fuse/receipts/rbf-v2v4real-official-benchmark-vehicle-protocol-decision-20261005.json`。
代码补齐明细与软件检查见 `code-gap-resolution-20261005.md`，汇总回执为
`test/recover-before-fuse/receipts/rbf-remaining-code-gap-resolution-20261005.json`。

## 可复查的代码清单

运行 `tools/event_track_v2x/prepare_remaining_experiments.py --output PATH`，
生成 create-once JSON：14 个实验族的源码路径、SHA-256、语法检查、依赖和具体门槛。
14 是整理用的实验族数量，不是论文实验总数。存在文件或语法通过不会设置实验 ready。
输出放入 `/Volumes/Data/test/recover-before-fuse/receipts/`。

## 本轮补齐的真实入口

`run_paper.py configuration --backend exclusive --allocation teacher` 生成显式
互斥身份森林配置；`--allocation learned` 生成同后端的学习式配置。
`replay` 从已绑定配置选择实际运行器，未知 backend 立即拒绝。
不含 backend 的旧配置继续进入旧运行器，防止改变历史实验语义。

`train_paper_priority.py export` 和 `run_paper.py replay` 的优先级 checkpoint
绑定使用同一个配置类型选择器，保留互斥森林全部 limits，避免误绑旧 covered 配置。
这些是新工作区代码；当前远端实验仍使用其既有冻结版本。

互斥后端的 `training_binding` 现额外冻结 35 个本地传递依赖的 SHA-256。
教师 export 检查这些字节与原回放 plan 一致；CPU fit、DDP 数据审计和 checkpoint
加载都会拒绝缺失或变更的互斥后端绑定。旧 covered 配置不自动升级为互斥配置。
SPD/V2V4Real 合成夹具的 teacher → export → fit → load → learned replay 已贯通，
相关接口测试 10 项通过，分布式/绑定测试 28 项通过；这不是完整真实教师训练验收。

森林执行优化另见 `forest-search-execution-optimization-20261004.md`。短序列 CPU
对照已通过；完整 195 事件、56,389 个状态的 CPU 候选已完成独立验收，复用了既有
串行参照。CUDA 派发、字节回读和独立验收入口已冻结，具体输入包上传仍待明确授权。
此单序列 CPU 结果不是完整队列、实际 CUDA 加速或显存 75–80% 的证据。

运行链：完整 train 教师回放 → 独立教师验收 → export → train holdout 选择/fit →
冻结三种子 checkpoint → learned 回放 → 独立评价。不能把模型风险界改善标签当作真实
身份风险，不能将 `full_official_train_trace=false` 改成 true 跳过全集验收。

## 后续入口与不能省略的工作

| 工作 | 现有入口 | 仍需补齐或验收 |
|---|---|---|
| M0–M4 | run_paper_pair_baselines / audit_mechanism_replay_v2 | M3 历史字节差异与完整传播链；不能放宽逐字节合同 |
| 精确小模型 | run_stage2_* | 复用已验收矩阵；核对尚缺的事件和资源比较 |
| 最终森林、Top-1/K | 已冻结派发器与 corrected receipt-row 验收器 | 全序列三种子独立验收；不能继承历史模型结果 |
| 互斥教师/学习优先级 | run_paper / train_paper_priority | 最终模型绑定的完整教师全集、冻结训练合同与真实回放 |
| 同资源基线 | scan_paper_resources / run_paper | 可选 CUDA 计量路径已准备，真实 GPU 数值接入及资源扫描未验收；MHT 全类别运行合同、完整成本对比仍缺 |
| SPD 已见 val | run_paper / evaluate_paper | 匹配模型、缓存、完整森林预测及独立指标 |
| V2V4Real official_test | run_paper / evaluate_paper | 按2026-10-05授权的官方划分、合并vehicle口径；会话映射不再阻塞，接续输入／类别校准／模型／评价依赖 |
| 公开方法 | run_public_baseline | 原生资产和协议；Long-SCOPE 完整实现缺失 |
| 指标与表图 | evaluate_paper / report_paper / plot_paper | 完整预测、GT envelope、全部种子及正确 cluster 统计 |
| 显存目标 | batched_row_context_scoring / rbf_gpu_replay_measurement | 无冲突 GPU 实测、数值独立验收、逐 UUID 75–80% |

独立 recovery-off 消融现已实现 bound、teacher 和 learned 三种分配路径；学习式
checkpoint 使用独立源码与配置绑定，不能复用未限制历史支持的旧优先级模型。
完整真实消融队列及同资源结果仍缺。只关闭前沿展开不阻止全合法动作解码恢复身份，
因此没有把现有 irreversible 对照直接重命名；候选实现及当前验收范围见下节。
以上未解决项必须保持显式未完成；本轮不能宣称“全部剩余实验代码已备齐”。

最终 Top1 seed3407 的原 GPU 任务、独立全产物读取、全序列搜索及条件选择器验收已完成。
`continue_final_top1_seed3407.py` 已封存为单次衔接器：绑定现有读取器的 PID、
启动时间与完整命令，只有确认其退出、完整字节/事件/因子回执已登记且原任务、
最终模型和源码重新核验通过后，才调用原冻结 Top1 CPU v2。观察失败继续等待，
不重启读取器、不重复 CPU、不重试失败。23 项检查通过；除种子和 Task ID 绑定外，
整个控制器 AST 与原 seed1337 版本一致。其完整 CPU 连续状态验收已由原控制器启动，
不得重复启动。三种子的字节/事件/因子/搜索/选择器索引为
`test/recover-before-fuse/receipts/rbf-final-refit-Top1-three-seed-byte-search-selector-independent-index-20261005.json`，
各覆盖 46 序列、7,445 事件，累计 677,744 个种子对应节点。该索引不代表完整连续状态、
完整在线方法或论文性能通过。

### 事件边界恢复能力消融候选（2026-10-05）

`run_paper.py configuration --backend recovery-off --allocation bound` 使用独立
`RecoveryOffTracker`。每次新事件仅允许延伸上次 commit 的显式身份类与实际输出类；
新组件合并使用各前驱的完整关联约束，不把逐节点允许根集合做笛卡尔组合。原始因子、
历史前缀与已提交输出保留在数据库中，但归档前缀不再授予当前搜索或动作解码权限。
前沿展开、直接前缀物化、全合法动作解码和重开数据库均使用这项限制。

同一事件内仍可探索新观测的合法后继；仍保留的身份类可在新证据下改变排序。
因此关闭的是跨事件恢复已删历史的能力，不是把输出直接改成 MAP，也不改变
检测、神经因子、运动模型或回放流程。回放函数 AST 与工作区 exclusive 入口一致；
与最终 GPU 发布入口相比仅增加进度日志上下文。配置仅新增独立 backend 名称和版本字段。
附加支持缓存纳入共享 LRU 上限，
限制元数据写入数据库、计入存储；序列化字节数不冒充实际峰值内存。

原模型的遗漏质量和 regret 保持保守上界 1；较紧的值另标为删减支持内的条件量。
残余区域显式声明与历史限制的交集，不把受限空间说成完整模型覆盖。
软件验收包含有限原始父路径穷举、跨组件合并、重评分、迟到输入、归档前缀绕过、
空消息、回滚和持久化，以及 SPD/V2V4Real 两种缓存接口的合成神经回放。

原始发布源码绑定候选现为
`test/recover-before-fuse/source-freezes/rbf-recovery-off-original-source-bound-candidate-v3-20261005`，
冻结 SHA256 为 `b778aee162f177066da207b70ecc807578be687373583a6b9748fa63b3aa9b70`。
准备器从已独立回读的原始源归档和最终任务 bootstrap 重建 126 个实际源文件，
其中 124 个逐字节保留，仅另外两个文件显式增加进度日志上下文；恢复消融代码另行新增。
候选不继承工作区的 `allocation_training.py` 和旧 `paper_runtime.py` 修改。
v3 独立目录的同一组 38 项软件检查通过；v2 漏带测试助手的失败包和日志保留。
这证明本地源码组装及上述软件范围。随后三份实际最终 checkpoint 已在该冻结源码下
严格加载，有限张量、模型摘要及七个评分源文件核验通过；同一缓存归档、manifest 与
每种子 46 序列、7,445 事件 schedule 已绑定，回执为
`test/recover-before-fuse/artifacts/rbf-recovery-off-final-checkpoint-input-contract-v1-20261005/acceptance.json`。
此检查复用既有缓存与全行神经数值验收，没有重做完整缓存构造或森林回放。

这个已发布冻结入口仅接受 bound 分配。2026-10-05 后续工作区新增 teacher/learned
实现及独立 checkpoint 绑定，未修改此历史冻结包。完整真实数据回放、独立全状态
验收和同资源性能比较均未完成。
三种子 GPU 生产器、去重派发器及独立字节/事件/因子读取器另行冻结于
`test/recover-before-fuse/source-freezes/rbf-final-refit-recovery-off-bound-GPU-v1-20261005`。
独立重建远端实际源码，128 个部署文件均与冻结候选逐字节一致；生产器实际工作函数
除 runtime 导入外 AST 相同，固定数值容差不变。派发前必须读取同目录
`source-qualification.json` 并重验源码、输入证明与准备合同；6 项检查覆盖缺失回执、
源码/输入/合同篡改及错误继承主实验验收的拒绝行为。已完成任务和旧失败不会因此重跑。
该准入只允许产生单独命名的 bound 消融候选，不能继承主任务完整森林、连续状态或
同资源性能验收；recovery-off 历史支持限制仍须进行全数据独立验收。

实际派发已经完成：seed2027 `dbbd57f63e824689afc9730a5953f09c` 使用 V100 八卡；
seed1337 `2a57ddaf3ea64a65b832a7025f3b66ad` 使用 A100 卡 4–7；seed3407
`178d14e808504f8cabff65223939c38a` 使用 A100 卡 0–3。2026-10-05 01:00 UTC
只读回查三项均 in_progress，远端源码和配置与冻结派发记录匹配；V100 已出现真实事件
进度，A100 处于输入准备阶段。三者均未完成产物验收，不能记为实验通过。派发回执为
`test/recover-before-fuse/receipts/rbf-final-refit-recovery-off-bound-GPU-dispatch-20261005.json`。
已配置逐事件、逐序列或输入字节 ETA，整个实验及独立验收 ETA 仍未知。

### 主回放 seed3407 原产物读取（2026-10-05）

原 GPU 任务 `daaefc04aa844cb3984f7b1237bc8c23` 已 completed，登记六项产物。
四个原始 rank 归档合计 15,614,589,058 字节，原读取器使用两个并发连接回读，保留
partial 及真实字节进度。独立衔接器冻结于
`test/recover-before-fuse/source-freezes/rbf-final-forest-seed3407-after-prefetch-continuation-v1-20261005`，
只观察已启动的读取器；进程确认退出、成功字节回执匹配、任务/配置/原始最终模型重验
后才调用原字节/事件/因子读取器及 CPU v4 全状态验收。工作区与冻结目录各 22 项
控制流检查通过；下载或验收失败均保留，不自动重试。不把 GPU completed 记为完整实验完成。

seed1337 原主任务 `4d76fa64a7e440a8a7396eda70cfc5e8` 随后也 completed，
四个原归档合计 17,357,910,918 字节；经进程去重后启动原读取器。仅替换种子、Task ID
和该读取器回执绑定的单次控制器冻结于
`test/recover-before-fuse/source-freezes/rbf-final-forest-seed1337-after-prefetch-continuation-v1-20261005`，
工作区与冻结包各 22 项相同控制检查通过。两条读取链均在运行；seed2027 损坏的 rank5
仍保留原失败与诊断，不重新下载同一坏文件、不替换登记哈希。

## SPD seen-val 完整森林输入准备（2026-10-05）

`prepare_rbf_seen_val_forest_inputs.py` 复用三种子已验收 V2 缓存、最终 checkpoint、
完整行上下文与 NN 输出。它从原始 3,316 条 schedule 生成森林事件，逐源核对信息时间、
首个到达时刻、帧哈希及原行事件，保留没有候选的事件；不从非空 NN 行反推 schedule。
每个种子覆盖 21 序列、7,189 缓存帧，1337/2027/3407 分别绑定
116,663 / 115,173 / 114,089 条已验收 NN 行。16 项事件边界和篡改检查通过。

`package_rbf_seen_val_forest_cache.py` 使用完整清单封装既有缓存，逐文件校验写入字节后
独立流式回读新归档，拒绝缺项、额外成员、链接、重复和字节变化。12 项封装检查通过。
三个本地归档均已完成，每个 14,379 成员，包括完整缓存 manifest；不重建缓存或重跑 NN。
源码分别冻结于 `test/recover-before-fuse/source-freezes/` 下的
`rbf-seen-val-forest-input-bridge-v1-20261005` 与
`rbf-seen-val-forest-cache-transport-v1-20261005`，同名 artifacts 目录保存逐种子回执。

`audit_rbf_seen_val_forest_inputs.py` 是单独的全 schedule/归档哈希复查入口，不导入桥接或
封装生产器。三种子复查已通过，索引为
`test/recover-before-fuse/receipts/rbf-seen-val-three-seed-forest-input-transport-review-20261005.json`。
输入范围保持 SPD 已见 val 的探索性 scheduled-snapshot，不能称为实测
网络到达历史。完整 train 森林接口验收、缓存上传及云端独立回读、冻结 val 森林生产器与
完整输出验收、学习式/同资源比较和独立评价仍待完成。这些本地输入包未创建 GPU 任务。

`prepare_rbf_seen_val_forest_runtime.py` 随后从已冻结最终 train 生产器生成独立命名的
`rbf-seen-val-bound-forest-GPU-producer-v1-20261005`。原森林模块、补丁、资源上限及
`1e-4` 因子对照容差不变；GPU worker 函数仅把 `PaperProtocol('spd','train')`
改为 `PaperProtocol('spd','val')`，主入口改为完整 21 序列/3,316 事件并增加输入门槛。
19 项源码 AST 保持及软件元数据篡改检查通过，检查日志和测试一起冻结。

远端入口要求同种子完整 train 主回放的 v4 独立验收、原任务 completed、原脚本/配置/
checkpoint/全部产物身份匹配，以及七项输入资产的完整云端字节回读。准备阶段不生成
这些尚缺的回执，三种子输入模板明确 `dispatch_ready=false`。该生产器承担确定性
bound 分配的探索性 val 基线；完整学习式方法仍需教师、优先级模型和对应独立验收。
输入发布和实际派发入口现已接入，见下节；val 全输出字节/事件/因子读取器也已准备，
完整连续状态、203 维因果上下文与森林语义验收的 val 适配已接入并单独冻结，见后文。
不能把源码冻结算作 val 回放完成，也不能把 bound 基线称为学习式完整方法。

### seen-val 输入发布与去重 GPU 派发

`publish_rbf_seen_val_forest_inputs.py` 默认只读。完整 train 主回放的 v4 独立验收
缺失时直接报告等待，不创建任务或上传。通过前置检查后，只允许发布已复查的缓存
归档、缓存 manifest、完整事件、输入绑定、归档回读回执、三种子复查索引及主回放
验收这七项资产。按内容/种子去重，上传每项后独立读取全部字节并核验 SHA-256；
失败或中断保留原任务和部分产物，不自动创建替代任务。

`submit_rbf_seen_val_bound_forest.py` 再次绑定实时主任务、七项发布资产、最终模型
及冻结生产器。沿用原森林配置、检查点、源码替换和资源上限，使用既有 val NN 输出；
train 输入的验收哈希明确放在继承的 train 来源记录中。派发身份不含卡数，换四卡/
八卡或 GPU 型号不会重复同一种子实验；L40S 不进入 GPU 候选，排队预约、正在运行
的其他任务及四卡/八卡物理交集均检查，创建后再核验一次才入队。

两个入口冻结于 `test/recover-before-fuse/source-freezes/` 下的
`rbf-seen-val-forest-publication-dispatch-v1-20261005`。22 项软件检查通过，包括
真实冻结远端准入函数的本地接口检查，以及上传中断/字节变化/主任务状态变化/
重复派发/物理 GPU 冲突拒绝。软件使用模拟 ClearML 传输，不是云端发布或 GPU
运行证据。seed2027 的真实入口只读检查仍等待完整主回放独立验收；其他种子的
主回放仍在运行，也不具备该前置验收。本轮没有上传
这些输入，也没有创建 seen-val 森林任务。显存目标仍须实际 UUID 运行测量。

### seen-val 全输出字节、事件和因子回读

`read_rbf_seen_val_forest_outputs.py` 与 `rbf_seen_val_forest_output_binding.py` 接入
实际 val 输入发布及最终模型来源合同。调用前后均核验实时任务、配置、产物和主回放
前置验收；只读尚未完成任务，失败产物另存，已接受回执不覆盖、不重复读取远端。
下载和稳定 float64 log-softmax 使用原冻结读取器，比较容差仍为 `1e-4`。

新入口使用已验收 val NN 文件的实际命名、逐行身份、权重及独立数值回执，绑定原始
完整 21 序列、3,316 事件；从完整 schedule 检查无新候选事件，而不是由非空预测
反推事件。检查协议为 SPD val、全类别 `.05/top64`、同一最终模型和森林源码。
它逐项核验 SQLite 提交与预测/audit、原始观测哈希、父候选及归一化势；解包目录
额外与归档成员逐字节比对，拒绝已存在解包标记掩盖文件变化、额外文件或链接。

37 项软件与既有输入接口检查通过，三种子已有 345,925 行 NN 输出仅作输入回读，
没有重跑网络或数值验收。源码冻结于
`test/recover-before-fuse/source-freezes/rbf-seen-val-bound-forest-output-reader-v1-20261005`。
实际 val 森林任务尚未派发，因此没有实际 val 森林输出通过这一检查。完整新鲜状态、
全部 203 维特征/因果上下文、恢复与合法动作、同资源性能及论文指标仍须分别验收。

### seen-val 完整森林 CPU 验收入口

`prepare_seen_val_forest_CPU.py` 从原冻结 v4 完整 train 验收器生成单独入口。
改变的是明确的 val 输入、21 序列/3,316 事件覆盖与探索性结论范围；保留原结构、
因果、残余质量、恢复来源、新鲜连续状态和合法动作检查调用。派生修改逐项记录，
可反向还原原驱动源码；数值容差保持 `1e-8`。

`rbf_seen_val_forest_cache203.py` 绑定已验收的真实 val V2 缓存、完整 schedule、
最终 checkpoint 和三种子 val NN 输出。序列时钟原点取全部原始帧，包含不可用帧；
消费 NPZ 时校验字节，以原冻结 float64 数学独立重建全部特征和首次到达父上下文。
train NN 证据单独记录，不继承为 val 森林完成。入口拒绝前缀、混种子、train 替代、
未完成云任务、缺失注册回执、变更容差及未经证明的测量网络到达历史声明。

完整验收开始前及结束后重新核验任务、源码、配置、注册产物与输入发布，每个序列
保存独立回执，最终回执记录这些文件的哈希，不创建远端实验。
43 项检查通过，包含三种子真实缓存接口及生成后入口导入；源码冻结于
`test/recover-before-fuse/source-freezes/rbf-seen-val-bound-forest-full-independent-CPU-v1-20261005`。
软件检查和真实缓存适配核对不代表真实 val 森林通过；实际运行仍须等待最终 train
主回放完整验收、val 输入发布、GPU 回放以及字节/事件/因子回读完成。

### seen-val 固定 K1 / K4 独立生产器

`prepare_seen_val_fixed_baselines.py` 分别从原冻结 Top-1 与 Top-K train 生产器派生，
保留各自固定宽度后端和原配置。worker 唯一变化是 `PaperProtocol('spd','train')`
变成 `PaperProtocol('spd','val')`；模型、全类别候选、时序监督来源、线程、搜索上限、
原生 GPU 架构检查、TF32 禁用及 `1e-4` 因子比较保持不变。没有借用主方法的可恢复
森林后端冒充固定 Top-K。

两份入口复用原 seen-val 七资产准入函数的原始字节，并增加各自完整 train CPU 前置
回执。完整主方法输入计划与固定基线计划分别保留，明确核对公共缓存、事件、神经因子、
检测/身份权重及独立 baseline 配置。CPU 前置回执须台账登记，覆盖 46 序列、7,445
事件，逐个绑定数据库、原始因果提交、203 维上下文和连续状态证明。运行前再核对
原基线任务的 completed、脚本、配置及全部注册产物；K4 回执不能用于 K1。

59 项软件检查通过，包括实际生成准入函数的远端接口模拟、两份原 worker AST 保持、
缺序列、混模型/宽度、GT、错误配置、放宽容差、伪继承及变化产物的拒绝测试。
首轮夹具遗漏四卡配置导致的失败日志保留；没有变更实验配置或数值容差。
源码冻结于 `test/recover-before-fuse/source-freezes/rbf-seen-val-fixed-K1-K4-GPU-producers-v1-20261005`。

实际完整 train CPU 验收及输入发布仍待完成。两个新变体的去重派发器现已接入，
专用 val 字节读取器及完整因果/203 维上下文/新鲜状态 CPU 适配也已准备；搜索、选择器和同资源验收仍需接入，不能使用 bound 方法的
验收器直接宣称固定基线通过。没有创建新 GPU 任务，也没有重跑已验收的缓存或行输出。

### seen-val 固定 K1 / K4 去重派发

`submit_seen_val_fixed_baselines.py` 要求主方法和对应 K/seed 的完整 train CPU、字节
回执，以及独立回读的七项 val 云端输入全部合格。复用已冻结生产器准入函数和公共
资源查询，创建与入队之间再次核验源码、前置回执和物理 GPU 绑定。仅接受四卡/八卡
GPU 队列，排除 L40S；K 与 seed 定义实验身份，换卡数不会创建重复实验。

创建前保存不可覆盖的意图记录，返回 Task ID 后立即登记。创建结果未知、配置失败、
前置证据或 GPU 占用变化均保留原尝试且不自动重试。默认只读；前置文件缺失时在
导入 ClearML 或查询远端之前返回具体缺项与 ETA 未知。

23 项检查通过，覆盖六种任务状态去重、四卡/八卡物理交集、任务创建结果未知、入队前
重新核验，以及实际冻结的 K1/K4 准入函数。封存入口为
`prepare_seen_val_fixed_dispatch.py`，目标目录是
`test/recover-before-fuse/source-freezes/rbf-seen-val-fixed-K1-K4-GPU-dispatch-v1-20261005`。
这些检查只证明派发控制和接口准备情况，不代表实际 val 基线、同资源指标或论文验收。

### seen-val 固定 K1 / K4 字节、事件和因子验收

`read_rbf_seen_val_fixed_outputs.py` 与 `rbf_seen_val_fixed_output_binding.py` 分别绑定
对应 K/seed 的派发记录、五项原始前置回执和两份生产器绑定产物。读取前后均重新核验
任务源码、配置、全部产物及已完成 train 前置证明。原固定宽度后端的源文件哈希从
已冻结源归档独立提取，不能接受主方法的 exclusive 后端或不同 K 的输出。

读取器通过 21 处可逆、逐项计数的变换从已冻结 seen-val 读取器派生，传输、归档
字节比对、SQLite 事件/行检查和 float64 log-softmax 计算保持原样。固定后端另查
不可恢复标记、固定宽度剪枝和保留类条件 Bayes 输出策略，保留全部 21 序列/3,316
原始事件，包括没有新增查询的事件。因子容差保持 `1e-4`，不重新执行已验收的 NN。

56 项软件检查通过，包括 K1/K4 模拟完整事件回读、历史失败保留、不重复下载、
输入回执变化、重新计算文件哈希后的错误上下文/到达/因子/后端拒绝检查。
封存入口为 `prepare_seen_val_fixed_output_reader.py`，目标目录为
`test/recover-before-fuse/source-freezes/rbf-seen-val-fixed-K1-K4-independent-output-reader-v1-20261005`。
当前尚无真实 val 基线任务或产物通过此检查；新鲜状态、完整因果/203 维上下文、
搜索剪枝、输出选择、同资源和指标仍须各自独立验收。

### seen-val 固定 K1 / K4 完整因果、上下文和新鲜状态 CPU 入口

`prepare_seen_val_fixed_CPU.py` 分别从原冻结 K1/K4 完整 train CPU 驱动生成
`accept_seen_val_fixed_K1_cohort.py` 与 `accept_seen_val_fixed_K4_cohort.py`。
每个驱动仅有 15 处可逆接口/范围变换，三项数值核验调用逐项保持原 AST。
两个固定后端原始 `physical_rows`、`expected_context`、`verify_database` 与已验证
验证集接口所用函数逐字节一致；新缓存适配器只有 4 处导入/范围变换，实际数值仍由
各自固定后端的冻结模块执行。全部 CPU 容差保持 `1e-8`。

入口要求 K/seed 专属、已入账的完整字节回执，再核验原任务 completed、源码、五项
前置证明、七项输入发布、全部产物字节、21 个序列数据库和完整事件身份。限定
21 序列、3,316 事件；禁止前缀和 train/main/其他种子验收继承。每序列独立保存
因果、缓存特征与状态证据，最终回执前重新核验远端身份并记录全部序列证明哈希。

50 项软件检查通过，包括三个种子两种 K 的真实缓存接口、原冻结数值函数一致性、
部分/混种子/混宽度/放宽容差/错误范围拒绝。准备目录是
`test/recover-before-fuse/source-freezes/rbf-seen-val-fixed-K1-K4-full-independent-CPU-v1-20261005`。
这里没有运行真实 val 森林全量 CPU 验收，也不包含完整搜索剪枝、条件选择器、
同资源性能或论文指标验收；这些仍有独立前置与后续任务。

### seen-val 固定 K1 / K4 搜索、剪枝与条件输出验收

`accept_seen_val_fixed_search_selector.py` 分别接入原冻结 K1/K4 的搜索剪枝 oracle
和 70 位 Decimal 条件输出 oracle；四份数值源码及原资格证明均按哈希核验，未修改
算法、归一化、比较容差或 tie 规则。入口分 `search` / `selector` 两步，覆盖全部
21 个序列、3,316 个事件，不以非空 query 过滤事件。

两步均要求已登记的同一 K/seed 验证集字节回执，以及当前 completed 任务、配置、
源码和全部输入/产物身份。selector 必须使用本入口的完整 search 回执与每序列
数据库绑定证明，逐项核对路径和哈希，不再执行已经通过的搜索。最终验收前再次
核对任务与证明，保留部分结果及失败；禁止把 train、其他种子或软件夹具当成真实 val。
日志提供事件级进度，异质搜索工作量不足以支持完成时间时将整组 ETA 标为未知。

软件检查覆盖真实文件上的两阶段衔接、零 query 序列、远端身份/字节/序列证明变化、
部分覆盖、混种子、容差放宽和路径越界拒绝。衔接测试模拟数值调用，不构成真实森林
实验验收；原数值函数从既有冻结目录直接加载，不重跑既有已验收实验。
封存目标是 `test/recover-before-fuse/source-freezes/rbf-seen-val-fixed-K1-K4-search-selector-v1-20261005`。
这些入口补齐后仍须实际派发验证集回放、逐项验收、完成同资源计量和独立指标评价。

## K4 seed3407 回读后的验收衔接（2026-10-05）

`continue_final_topk_seed3407.py` 已单独冻结并启动，观察原读取器 PID 97090。
它只在确认原进程退出、完整字节/因子回执入账、原任务仍为 completed 且源码、模型、
配置和全部注册产物匹配后，执行原冻结 v2 CPU 验收器一次。观察超时或错误继续等待，
PID 被复用、缺回执、已有验收或历史失败均拒绝自动衔接；不重启读取器。
13 项衔接检查通过，原数值算法和容差没有改变。启动衔接不表示 CPU 验收已启动或通过。
运行证据位于 `test/recover-before-fuse/artifacts/rbf-final-topk-seed3407-after-readback-continuation-v1-20261005`。

后续核验：原读取器已正常完成，K4 seed3407 的完整字节/事件/因子回执已经入账。
46 序列、7,445 事件、223,355 观测的独立搜索/剪枝及条件选择器也已通过，原 `1e-8`
容差未改；审计索引为
`test/recover-before-fuse/receipts/rbf-final-refit-K4-seed3407-search-selector-independent-review-20261004T205848289710Z.json`。
现有衔接进程已启动完整 CPU 验收（PID 73876），后续实时观察其状态，不再另起一份。
这两项组件通过尚未证明全部连续分支状态、完整基线或论文指标完成。

## 主模型 seed2027 与基线验收并行（2026-10-05）

`continue_final_forest_seed2027.py` 已单独冻结并启动，观察原预取 PID 66505。
只有原预取退出且全部已登记分片哈希通过，才检查原任务 completed、全部八个 rank
产物及配置，然后运行原冻结终态读取器；完整回读验收通过后再运行原冻结 v4 CPU
森林验收。预取锁在终态读取期间保持独占；已有结果、失败、进程或 PID 变化会阻止
重复执行。17 项衔接检查通过，原数值检查代码与容差没有修改，实际全量验收仍待执行。

K4 seed1337/2027、Top-1 seed2027 的独立搜索/剪枝与输出选择器验收均已使用原冻结
入口完成，三组分别覆盖全部 46 序列、7,445 事件。输入只依赖各任务已通过的完整
字节/事件/因子回执，故可与连续状态验收并行，无需重跑回放或等待后者完成。
三个搜索及三个选择器验收的固定容差均为 `atol=rtol=1e-8`；全部逐序列回执、原始
数据库身份、源码和台账哈希已复核，索引为
`test/recover-before-fuse/receipts/rbf-baseline-search-selector-independent-review-20261004T195413654702Z.json`。
这些是 6 个验收环节，不是 6 个完整实验；连续状态/203 维上下文、同资源成本和
论文指标仍须分别完成。seed3407 及其他未列入的组合不能继承此索引。

## 学习式展开顺序独立验收（2026-10-05）

`rbf_independent_learned_trajectory.py` 从不可变 SQL 中的原始观测、因子和前缀行重建
组件划分、合并、初始状态、历史/前沿提议、共享容量与每次实际执行操作。它复用已冻结
独立参考的合法根身份、遗漏质量和条件动作数值函数，不导入生产器模型或跟踪器。
18 维特征从重建状态独立计算，32 隐单元 tanh MLP 用补偿求和独立复算；固定
`atol=rtol=1e-8` 不变。该检查只接受最终回放的 `max_model_regret=1` 合同。

浮点分数接近并列时，独立补偿求和与生产器批量 float64 运算可能有末位差异。
检查器先独立核验全部输入特征及分数，再用已核验特征重新执行 float64 矩阵运算；
选择顺序和按组件 ID 处理完全并列必须精确一致，不使用排序容差。独立补偿求和的
排序差异单独逐项报告，不宣称特征逐字节一致或跨算术实现排序一致。

26 项真实冻结运行时的软件夹具及篡改检查通过，覆盖空到达、过期窗口、组件合并、
0/1/4/100 搜索预算、容量受限动作及错误候选/特征/分数/并列/前缀/权重拒绝。
生产器实际加载的模型文件按原始归档与冻结 bootstrap 替换清单逐字节核对。
`check_learned_search_trajectory.py` 是单库只读入口，输出单独新目录并保留失败证据。
该单库检查不覆盖最终动作最优性、连续状态、完整数据/模型资格及同资源成本。
完整集合入口的后续接入如下；实际优先级 checkpoint 与真实森林验收仍未完成，
不能凭软件检查开启论文门槛。

## 学习式回放完整 CPU 验收入口（2026-10-05）

`prepare_final_refit_learned_CPU.py` 从已冻结 v4 主回放验收器派生独立入口
`accept_final_refit_learned_cohort.py`，保留五项原始数值验收调用和全部软件资格门槛：
原始缓存 203 维特征/父上下文、互斥结构及恢复、因果历史、全部新鲜连续状态和
有界合法动作搜索。新增逐库学习式轨迹检查，并要求 46 个唯一序列、7,445 事件及
所有节点数在各验收器之间一致。容量不足仍须报告 undecided，不能冒充完整最优。

`rbf_final_refit_learned_forest_binding.py` 在执行前后实时核验 completed ClearML 任务、
注册产物、源码、完整教师及真实拟合/独立发布合同；再次绑定学习式字节回执、最终模型
和每个序列的优先级 checkpoint。旧输出、局部覆盖、错种子、改权重、缺 rank、变化的
任务或回执、放宽容差均拒绝。原生 timings/resources 文件重新核对哈希，但同资源性能
资格保持关闭。入口仅写新目录，不重启、不派发、不覆盖先前失败。

56 项有限软件检查通过；另用真实冻结学习式生产器的 7 事件夹具验证了四项原始森林
验收器可接收该接口。源哈希、单库浮点排序差异和固定容差都在输出中保留。
这是完整验收代码接入；真实教师全集、优先级 checkpoint、实际学习式全量回放、
完整集合执行、资源比较及论文性能仍未完成。

## 最终模型教师的主回放前置检查

`rbf_final_refit_teacher_prerequisites.py` 只读核验最终模型主回放资格。它绑定已冻结
v4 CPU 验收器、最终 checkpoint/NN 数值回执、实时 completed Task 和全部注册产物，
并逐项核对 46 个序列的结构、因果、全部新鲜分支状态、合法动作和 203 维原始上下文
验收。旧模型、固定 K 基线、混种子、局部覆盖、修改容差和容量受限决策冒充最优均拒绝。
已验收输入不会重新训练或重新执行数值验收。

此入口不会派发任务，也不代表教师标签已通过：容量教师运行时必须另行绑定到最终
模型，真实全集反事实 witness/标签/成本仍须独立验收，随后才能 export 和 fit。
现有 Linux 小样本 witness 与通用训练接口的软件通过不能替代这些步骤。

`prepare_rbf_final_refit_capacity_teacher.py` 已生成单独冻结的最终模型教师生产器。
它在原主回放生产器上只增加两个已验收字节的 witness 模块；三个森林核心替换文件
与 Linux witness 完全相同，真实事件 replay 函数 AST 不变，GPU worker 的数值和
全序列循环除运行器选择与 teacher 分配外保持不变。上线时先读取单独发布的最终主
回放前置证明，并实时核对主任务、模型、原始输入、部署限制和 Linux witness 任务产物。

生成器和前置检查合计 36 项软件检查通过。源码冻结位于
`test/recover-before-fuse/source-freezes/rbf-final-refit-capacity-witness-teacher-producer-v1-20261004`。
目前未上传、未派发、未运行真实教师全集。前置证明发布及去重派发入口见下文，
只能在对应最终主回放完成独立验收后执行。

## 最终模型教师的收集与独立验收入口（2026-10-05）

新增 `rbf_final_refit_teacher_binding.py`、`collect_final_refit_capacity_teacher.py`
与 `accept_final_refit_capacity_teacher.py`，冻结于
`test/recover-before-fuse/source-freezes/rbf-final-refit-teacher-collection-target-admission-v1-20261005`。
入口要求主回放 completed、46 个序列的完整独立验收和单独发布的前置证明；逐项绑定
最终 checkpoint、原始缓存、7,445 个真实事件、注册源码、全部 rank 产物和已验收 NN 因子。
4/8 卡均接受，但必须完整覆盖，TF32 仍禁用。

旧收集器在复制原生资源文件时覆盖了 `original` 计划变量，导致第二个序列读取失败。
用双序列软件夹具复现后，仅在新版本中修正，保留旧冻结代码。新入口不会补造空帧，
不会合并进程峰值冒充成本，也不会把字节校验当作标签验收。

完整标签入口使用原冻结的结构、因果、连续状态、合法动作、原始反事实 witness 和
18 维特征/有符号目标数值验收函数，另接已冻结的最终模型 203 维上下文绑定。
22 项软件检查通过；旧模型、局部覆盖、混种子/产物、改源、放宽容差与非有限因子被拒绝。
尚未运行真实教师全集，尚未导出或训练学习优先级；同资源成本、完整 Stage2、论文性能
继续保持未完成。

## 教师前置证明发布与 GPU 去重派发（2026-10-05）

`publish_final_refit_teacher_prerequisite.py` 默认只读检查，`--execute` 才创建数据任务。
发布载荷限于主任务 ID、种子、模型/验收哈希和 46 份序列证明哈希，不含源码、SQLite、
预测或原始数据。上传后逐字节独立读回并核对注册产物，才生成可供教师使用的发布回执。
中断或失败的发布保留任务和本地产物，不自动创建替代任务。

`submit_final_refit_capacity_teacher.py` 默认只显示候选，`--execute` 才派发。必须提供
匹配种子的主回放验收、字节回执和前置证明发布回执。配置/种子去重身份不含 GPU
型号和卡数；支持已有冻结生产器的 4/8 卡运行。保留原主回放的全部部署限制，新增教师
witness 与离线分配，不提高搜索上限。L40S 排除在 GPU 候选外，八卡/四卡 Worker 交集、
排队预约和正在运行的其他任务均在派发前检查；创建任务即记账，再核验占用后入队。
主回放和 Top-1/K 的既定缺项优先，失败教师不自动重启。

两项入口及依赖冻结于
`test/recover-before-fuse/source-freezes/rbf-final-refit-teacher-publication-dispatch-v1-20261005`。
15 项无网络软件检查覆盖数据字段白名单、独立回读、失败保留、去重与资源变化拒绝。
尚未发布真实前置证明或创建真实教师任务；显存利用率必须由实际 UUID 测量，派发
参数中的 75–80% 目标不构成达标证据。

## 最终模型优先级导出与训练衔接（2026-10-05）

旧 v7 导出器要求教师记录两个训练端模块的源码哈希，但原始教师没有执行这两个模块；
旧验收入口也只认识历史模型的回执，不能直接接最终模型教师。新版本明确区分教师执行
源码与训练端源码：全部共同的运行时字节必须一致，只有两个指定训练模块单独绑定。
原始模型、优化器、10 epoch 拟合、序列 holdout 选择和 checkpoint 格式逐字节保留。

`prepare_final_refit_priority_export.py` 已生成单独的 135 文件运行目录，冻结于
`test/recover-before-fuse/source-freezes/rbf-final-refit-priority-export-consumer-v1-20261005`。
只有导出入口的三处元数据调用和验收模块改变；18 项接口检查、所有文件哈希与本机
入口导入检查通过。本机导入不是新 Linux 运行时验收，也没有运行真实教师导出。

`run_final_refit_priority_after_targets.py` 是独立的只读溯源及 CPU 执行入口。它要求完整
教师标签验收、最终模型主回放及其产物、原始缓存与事件、前置证明全部匹配；实时核对
ClearML completed 状态、冻结任务源码、原始配置和全部注册产物后，才允许 export/fit。
支持主回放与教师分别使用已合格的 4/8 卡，不改变搜索上限；本入口不派发或重启任务。
16 项无网络检查覆盖终态、源码、模型、配置、产物和发布回执变化的拒绝，以及进度观察
异常不重启活进程。训练日志从实际完成 epoch 估计当前拟合耗时，导出进度不足则未知。

**仍缺新消费者的实际 Linux 运行时探针及独立验收**。执行入口拒绝复用旧 v7 运行时
回执，当前不能启动真实拟合。完整教师全集、真实 export、三种子拟合、独立 checkpoint
和 learned 回放也仍待完成。序列 holdout 不构成上游全流水线隔离；标签仍是模型内
决策改善，不能宣称真实身份风险或完整 Stage2 已完成。

## 新消费者 Linux CPU 探针（2026-10-05）

`prepare_final_priority_runtime_probe.py` 和 `control_final_priority_runtime_probe.py`
已冻结于 `test/recover-before-fuse/source-freezes/rbf-final-refit-priority-Linux-CPU-runtime-probe-v1-20261005`。
探针复用已验收的原始源码和离线 Linux 依赖，新增包为 66,772 字节、17 个源码/元数据
成员，不含数据集、SQLite、预测、权重或凭据。17 项软件检查及完整包本机重建通过；
重建后的 135 个源码文件和三个核心文件的替换前后哈希全部匹配。

入口先核对原始资产和离线 wheel 字节、按完整配置去重，再检查 L40S 空闲 CPU 任务槽。
默认 `dispatch` 仅检查；`dispatch --execute` 才创建任务，创建后立即记账，异常不重试。
`readback` 必须独立读回注册产物，并核对实际 Linux/Python/Torch/NumPy、CPU 线程、
CUDA 未初始化、完整源码导入和输入缺失拒绝，才生成新运行时验收回执。

本次具体源码发送至 ClearML 的动作在创建进程前被自动审批拒绝；理由是已有概括上传
授权没有明确覆盖此源码及指定目的地。已向本线程用户提出具体授权问题，尚未收到
答复；未创建 ClearML 探针任务，未上传。拒绝范围内不使用其他传输方式重试。
该问题与另一个待批准的 CUDA 原始输入包分别记录；已有远端实验及本机验收继续运行。

## 优先级导出和 checkpoint 的独立数值检查（2026-10-05）

`audit_final_refit_priority_checkpoint.py` 已准备。它绑定原最终模型主回放、完整教师标签
回执、新消费者源码和 Linux 运行时，并实时核对已完成上游任务及其全部注册产物。
随后逐条比较导出分片与已独立验收的原始 teacher audit，覆盖每个事件、分量候选组、
18 维特征及有符号目标，拒绝漏组、额外组、顺序/数值变化和混入其他模型输入。

检查器独立构造固定的 37/9 train 序列划分，核对 10 个 epoch 的组数与选择记录；
对每条特征用 NumPy float64 重算 18→32→1 tanh 网络，按组等权计算留出 MSE，要求
checkpoint 对应最小留出误差和最早的精确平局 epoch。固定绝对/相对容差均为 1e-8。
它不导入训练模型、优化器或教师特征函数；实际 teacher 特征/目标正确性由既有完整
独立标签回执绑定，此处检查序列化完整性与固定权重数值，不重新生成标签或重训。

此入口只出具 `local-numeric-audit.json`。云端拟合产物独立字节回读、完整优化器轨迹
复算、实际 learned 森林回放、资源计量及论文指标不会因此被标为完成。当前没有真实
优先级拟合产物，尚未运行该真实数据验收。

## 主回放已注册分片的提前字节回读（2026-10-05）

`prefetch_final_refit_rank_artifacts.py` 已单独冻结于
`test/recover-before-fuse/source-freezes/rbf-final-refit-registered-rank-byte-prefetch-v1-20261005`。
它绑定原始派发 recipe 和远端任务源码，仅对已经注册的 rank 归档使用原冻结读取器的
流式下载、断点位置和完整 SHA-256 校验。最多两个并行下载，不上传、不派发、不解包，
不写森林验收回执。元数据边界的六项本机检查通过；这不是实验完成证据。

seed2027 的 V100 主任务 `4e9019afec67453ba5a83e8401949991` 在本轮观察时仍运行，
已注册七个分片，总计 13,786,759,730 字节。提前回读进程为 PID 66505、工具会话
47086；后续必须实时确认句柄或进程，不能仅凭本记录推断存活。进度日志位于
`test/recover-before-fuse/logs/rbf-final-refit-seed2027-rank-byte-prefetch-v1-20261005.log`，
其中 ETA 只描述对应文件传输，整体任务和验收 ETA 仍未知。

归档直接进入原始终态读取器的预定缓存目录，已有完整文件须重新核对注册大小和哈希，
部分文件保留 `.partial`。完成前后均绑定远端注册元数据；元数据变化、下载或校验失败
均保存失败类型及部分文件，不覆盖实验历史。该进程退出前不得并发启动原始终态读取器；
退出后，任务仍须满足原来的终态、全部 rank、receipt、manifest、全事件和数值验收。
提前取得七个分片不构成第八个分片、任务完成或任何森林语义通过。

后续 2026-10-05 核验发现：任务已 completed，但预读取 PID 66505 和衔接器
PID 34012 均以失败退出。rank0/1/2/3/4/6 六个归档的注册哈希匹配；rank5 的
1,918,350,389 字节 `.partial` 大小完整，本机 SHA-256 为
`d030ea8dfa5f5f52dbe45d9ff8c8c195ae253747d57892a5165b049eb48f937f`，与注册值
`df07f956a8b56226fd7ada9dbccd329c6c51b415176c0afb13c3f4be8cf30dca` 不同。
原失败回执、控制器失败和坏样本保留。单次隔离诊断入口
`read_final_forest_rank5_and_rank7_diagnostic.py` 使用原冻结传输函数，只将 rank5
和先前未登记的 rank7 读取到新目录，不重启实验、不重新下载六个合格归档、不解包或
启动 CPU 验收。两项独立哈希都通过后仍须另行核验并接回原完整终态读取器；本步骤
不自动重试，也不能覆盖旧失败或标记完整森林已验收。

### 损坏分片隔离读取后的主模型验收衔接

`continue_final_forest_seed2027_after_diagnostic.py` 是单独命名的一次性本机衔接程序。
先核验原隔离读取进程的 PID、启动身份及已登记 started 回执；进程仍在运行或观察
失败时继续等待，不能把超时当作退出。只有进程确实退出、两项独立字节哈希都通过，
且原 completed 任务的完整十产物、脚本、配置与 recipe 仍匹配，才进入后续步骤。

在同一缓存锁内，只将 rank5、rank7 的合格隔离归档排他复制为新的完整文件，保留
原损坏 `.partial`、六个合格归档、诊断原件和两项旧失败。随后启动原冻结终态读取器，
由其重新核验全部归档、全事件及模型因子。该回执完整通过后才启动原 v4 CPU 全量
验收器，使用新输出目录；不启动新的 GPU 任务，也不重复下载六个已核对分片。

复制中断、诊断失败、远端身份变化或任一子进程失败均留下证据并停止，不自动重试。
默认只读；`--execute` 才开始等待和依赖衔接。子进程退出或控制器成功退出仍不能直接
计作论文实验完成，必须继续独立审阅实际回执。源码目标为
`test/recover-before-fuse/source-freezes/rbf-final-forest-seed2027-after-isolated-diagnostic-v1-20261005`。

## 最终模型学习式优先级 GPU 回放源码（2026-10-05）

`prepare_rbf_final_refit_learned_replay.py` 从原始最终模型主回放生产器生成单独候选，
已冻结于 `test/recover-before-fuse/source-freezes/rbf-final-refit-learned-priority-full-train-GPU-v1-20261005`。
保留全部原始森林核心、真实因果事件循环、全类别候选、限制、归一化因子比较和
TF32 禁用设置；仅接入冻结优先级策略、其来源核验及签名审计。主模型前向仍在 GPU，
冻结优先级 MLP 和森林搜索保持原执行方式；不能据此宣称搜索已移至 GPU 或显存达标。

候选要求完整本机导出/权重数值回执及单独发布的 checkpoint、权重、回执的独立云端
字节回读证明。派发计划必须绑定全部主回放和教师产物；运行前重新核对两项 completed
任务的源码、recipe、注册哈希、模型和部署限制，允许各自使用合格的 4/8 卡。
运行时使用原冻结 `load_priority` 核验完整 source map、因子实现、策略权重和缓存绑定。
每个 rank 记录实际策略签名和 checkpoint 哈希；全部 46 序列必须与导出验收一致。

本机 NumPy 1.26 / Torch 2.6 环境下 22 项软件检查通过，包括真实策略加载器、跨端
来源/产物变化拒绝、缺 rank、部分导出、混种子、错误权重、验证集选择及修改预算拒绝。
首轮软件夹具未填入运行卡数而失败的日志保留，随后明确测试主回放四卡、教师八卡。
这些检查没有运行真实教师、拟合或 learned GPU 回放，也不是 Linux/GPU 环境验收。

仍须完成实际三种子优先级拟合及独立数值验收、checkpoint 发布和去重派发入口，随后
对真实 learned 回放做完整优先级排序、森林状态、因果历史与总成本验收。
`rbf_final_refit_priority_checkpoint_publication_independent_bytes_v1` 是本候选要求的
发布回执接口；当前尚无真实发布回执，不得用软件夹具或手填元数据替代。

## 优先级 checkpoint 发布及独立云端字节回读（2026-10-05）

`publish_final_refit_priority_checkpoint.py` 已补齐。默认只做资格检查并显示具体载荷；
只有显式 `--execute` 才创建 ClearML 数据任务。资格检查要求台账已登记的完整本机
数值回执、其精确七项拟合输入哈希、完整教师验收、新 Linux 运行时及实时上游来源。
模型文件存在、训练任务结束或源码准备均不能绕过这些条件。

发布载荷严格为 checkpoint、权重和独立数值回执三项，不包含源码、原始数据、SQLite、
预测或教师训练向量。目的地为既有 ClearML 文件服务 `10.100.34.118:8081`，项目
`Thesis/Recover-Before-Fuse/Inference`。按种子、载荷及上游来源形成发布身份；创建前
记 intent，获得任务 ID 后立即记账，再上传和独立逐字节回读，全部匹配后才登记发布回执。
中断、未知创建结果、上传/回读失败保留原任务和部分文件；不自动补建替代任务。

22 项无网络检查通过，覆盖缺项、部分覆盖、变更数值容差、载荷越界、默认不上传、
完整三产物复用、上传及回读失败保留、远端哈希/终态和本机独立回读变化拒绝。
发布回执接口直接匹配已准备的 learned GPU 生产器；实际优先级拟合尚未完成，
当前未发布 checkpoint，也没有创建新的 ClearML 数据或实验任务。

最终使用独立冻结的
`test/recover-before-fuse/source-freezes/rbf-final-refit-priority-checkpoint-publication-v2-20261005`。
v2 额外绑定资格核验时实际读取的数值回执和权重哈希，拒绝核验后、上传前改变任何
三项载荷。此前 v1 源码和 19 项软件检查日志保留，未执行过真实发布；不能再使用 v1
进行实际上传。冻结依赖引用也在入口逐项校验。

## 学习式回放 GPU 去重派发入口（2026-10-05）

`submit_final_refit_learned_replay.py` 接入上面的 v2 发布器、最终教师/拟合来源检查和
已冻结的 learned GPU 生产器。默认只读核验；实际拟合、台账登记的数值回执、新 Linux
运行时和 checkpoint 独立云端字节回读有任何缺项时，不能创建实验任务。

去重身份不含 GPU 型号或执行卡数。按已支持的四卡/八卡选择物理无冲突的空闲 Worker，
L40S 排除在 GPU 候选外，并保留主回放和固定 Top-1/K 的既定派发优先级。创建前保存
命令、配置和 intent，获得 Task ID 后立即记账；入队前再次检查物理卡交集。创建结果
未知、配置失败、资源变化或既有失败任务都不能触发自动补建或重启。

10 项无网络检查覆盖真实原始 plan 的精确变化范围、显式默认不派发、跨卡数去重、
失败保留、L40S 排除和重叠八卡 Worker 占用拒绝。`75–80%` 仅记录为资源目标，不能
据此宣称实际显存达标。同资源计量、真实 learned 全量排序/森林验收和论文性能仍未完成。

该入口及 v2 发布器、物理 GPU 调度依赖、测试和日志另行冻结于
`test/recover-before-fuse/source-freezes/rbf-final-refit-learned-priority-GPU-dispatch-v1-20261005`。
冻结入口会重新核对生产器、发布器和依赖引用的源码哈希；本机准备与检查不创建真实
learned 任务，必须继续等待完整教师、拟合、数值验收和 checkpoint 发布证据。

本轮还检查了正在运行的 CPU 验收器：它们按序列连续执行，既不复用外部序列回执，
也不为并行写入加锁。因此不能在其输出目录中另起并行验收；现有进程和冻结代码保持
不变，没有重复计算未开始的序列或覆盖部分结果。

## 学习式回放独立产物入口（2026-10-05）

新增 `read_final_refit_learned_outputs.py` 和
`rbf_final_refit_learned_output_binding.py`，由
`prepare_final_refit_learned_readback.py` 从原冻结读取器逐项派生。
新入口再次核验完整教师、真实拟合、数值回执和独立 checkpoint 发布；要求所有 rank
与全部序列绑定同一优先级 checkpoint 和 policy signature，拒绝旧模型、混种子和
变更部署限制。每个序列还必须保留原生时序、输入 hash、成本文件和预测/audit 字节。

原下载、解包、log-softmax 函数的 AST 和完整观测因子检查段保持一致；NN 的绝对与
相对容差均保持 1e-4。产物读取前后重新核对任务、源码、配置、哈希与终态，读取目录
加排他锁，已有回执只验证复用。失败产物另存，不重启任务。42 项无网络软件检查通过，
包含 SQLite 行/父候选/非有限因子损坏、策略混用、成本文件缺失和路径越界等拒绝检查。

入口与依赖冻结于
`test/recover-before-fuse/source-freezes/rbf-final-refit-learned-priority-independent-output-reader-v1-20261005`。
此入口只验收完整字节、事件与固定模型因子；`learned_expansion_order_independently_verified`
保持 false。真实学习式展开顺序、分支状态、完整森林、同资源成本和论文指标仍须独立
验收，不能用此软件检查或未来字节回执替代。

## GPU 资源扫描计量接口（2026-10-05）

`scan_paper_resources.py` 增加可选的 `GPU_runtime` 文件及 SHA-256 绑定，GPU 计划使用
独立的 v2 标识；CPU 计划入口保留。运行合同必须明确 CUDA 索引、物理 UUID、预热事件数、
Torch/OMP/OpenBLAS 线程数、原生因果前向方式和禁用 TF32。执行时再次核对实际设备；
不限制 GPU 型号，L40S 仍只执行 CPU 工作。不自动选设备、调大 batch、占位分配显存或派发任务。

每个候选/种子使用新进程；预热只读实际到达 schedule 的前缀，写入单独的森林和日志，
正式回放重新构造森林并覆盖完整 schedule。记录预热/加载时间、正式回放墙钟、完整子进程
墙钟、实际设备内存采样和本进程张量/分配器峰值。资源选择检查每一项证据哈希、事件数量、
设备身份、预热合同、峰值、采样时间与吞吐的一致性，禁止跨设备或环境混合比较。
主机 RSS 是包含加载与预热的进程寿命峰值，设备总占用可能包含其他进程，两者都明确标注。
独立评价子进程移除外部 `PYTHONPATH/PYTHONHOME` 并禁用 user-site，避免研究环境的二进制包
污染独立评价环境；指标实现未改。

47 项软件检查通过，包含模拟 CUDA 计数器下的真实合成森林预热/回放、三种子扫描与独立
评价，以及原 CPU 新进程扫描回归；没有真实 GPU 计量实验。首轮测试的共享内存限制、旧
评价路径缺失和研究路径污染日志均保留。该工作树运行接口尚须与正式冻结 checkpoint
及原始源码完成数值接入验收，不能据此直接替代既有生产器。真实同资源实验还需明确预算、
设备隔离、完整输入覆盖及成本审计，`equal_resources_claimed`、正式性能与显存 75–80%
达标标志保持 false。文件与验证日志冻结于
`test/recover-before-fuse/source-freezes/rbf-paper-GPU-resource-scan-measurement-v2-20261005`。
v1 的冻结包测试发现遗漏合成数据辅助模块 `build_detection_cache_v2.py`，v1 和失败日志保留；
v2 补齐测试依赖，未更改计量或森林算法。

## Top1 seed1337 读取到全 CPU 验收衔接（2026-10-05）

最终 Top1 seed1337 任务 `237f30c74248439c91f07a7e868ca383` 已完成，五项云产物由
原冻结读取器独立读取。新增的单次衔接器只等待该进程，不重启或替代它；绑定 PID、
启动时刻、完整命令和原读取器 SHA，观察失败视为状态未知。只有进程确认退出、完整
byte/event/factor 回执存在且与台账哈希一致，再次通过原冻结 Top1 来源核验（最终模型、
46 序列/7445 事件、原配置与容差、实时 completed 任务和所有云产物）后才启动原 CPU
v2 验收器。重复进程、历史失败或已有验收目录均阻止启动。

`continue_final_top1_seed1337.py` 默认仅作只读预检，`--execute` 启动单次等待。
23 项软件检查通过，覆盖观察超时、PID 被复用、错误种子/任务/来源、缺失或变更台账、
不完整字节证明、历史失败和重复 CPU 验收。原数值检查器及读取器未改，原进程观察函数
AST 保持一致。源码已冻结在
`test/recover-before-fuse/source-freezes/rbf-final-top1-seed1337-after-readback-continuation-v1-20261005`。
启动或子进程退出码 0 都不自动宣称验收完成，仍须独立检查最终回执。

## 公开基线实际权重路径核对（2026-10-05）

`run_public_baseline.py` 补齐 DMSTrack 原生入口的双网络权重绑定。固定版本
`d3b9949499c8e68ea33060873bd1cb95b6d4d323` 的 `main_dkf.py` 通过对完整 ego 路径
执行 `replace('ego', cav_id)` 加载 ego 与 CAV 1，没有单独的远端 checkpoint 参数。
原适配器虽然检查计划中第二份文件的哈希，却未证明该文件就是原生程序实际加载的文件。
现在预检核对精确派生路径、两个实际文件的独立身份及哈希，在过程回执中记录两者，
进程结束后再次核对；即使进程返回 0，权重变化仍记为失败。未改变官方算法或命令。

16 项软件检查通过，包括从独立下载的固定版本官方源码提取路径表达式做对照，以及
未消费文件、父目录替换、缺失/变化/别名权重、进程运行期间变化的拒绝检查。
这只是原生调用准备，未执行 DMSTrack。按2026-10-05后续用户决定，V2V4Real采用官方划分及
合并vehicle，物理会话映射不再阻塞该基准；两份实际权重、原生BEV特征、运行时、输入与
合并类别仍须按实际使用情况核对，不能把不同指标定义混排。

同时重新核对 [Long-SCOPE 论文](https://arxiv.org/html/2604.09206v1) 与
[SparseCoop 作者的公开仓库目录](https://github.com/wang-jh18-SVM?tab=repositories)。
在此次检查范围内未找到可绑定的 Long-SCOPE 官方代码或权重，不能推断所有位置均未
发布，更不能用 SparseCoop 替代。原生公开方法基线仍保持未完成。

固定版本 SparseCoop `tools/test.py` 的 `--out` 仅触发写出，实际路径由配置/权重位置
和时间戳生成，未直接使用该参数的文件名。适配器现要求 `--output` 等于官方代码实际
派生的配置输出目录，在创建目录或进程前拒绝不一致；仍以新目录规则保留全部旧结果。
进程退出后只收集该目录内唯一原生时间戳子目录的 `results.pkl` 和伴生产物，拒绝缺失、
空文件、多次运行和链接逃逸，逐文件保存哈希。即使返回码为 0，收集不完整也记为失败。
不反序列化 pickle，也不将有文件等同于指标验收；官方源码和计算路径未改。

原 DMSTrack 检查与新增 SparseCoop 检查合计 26 项通过，其中包括从各自独立下载的
固定版本源码提取实际路径表达式做对照。模拟进程只用于验证收集和失败保存，不是原生
GPU 模型运行。真实权重、数据/训练资格、原生环境、数值及指标独立验收仍未完成；此轮
没有执行公开方法的原生推断或评价。
