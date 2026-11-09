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
设备选择遵循最新用户安排：2026-10-07 明确要求单机 GPU 工作在 10.100.35.112 执行，
覆盖此前 L40S 仅执行 CPU 的放置规则。实际设备须读回物理 UUID；原科学 batch、预算与
数值容差继续冻结，不能据放置变更改写既有 4/8 卡训练合同。

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


## M0–M4 原始产物与部分独立统计（2026-10-07 01:18 CST）

本次复用完整原封存产物，没有重启预测器或替换 M3 seed2027 的历史失败。
当前实际产物根目录为
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-all-thirteen-full-stream-provenance-20261006T170240089706Z`。

- 原注册归档的 78 个实际流均已逐字节展开并重新核对原 size/SHA；另读回
  34 份原执行计划、summary 和 receipt。原始 evaluator-only GT 的 3,316 帧
  JSONL 与 manifest 已按原 SHA 读回，不进入预测或身份训练。
- `verify_stage1_preserved_streams.py` 使用固定 SHA 的原输出链扫描器，M4 仅提取
  作者扫描函数 AST，未执行生产器导入。26 次执行、86,216 帧内部提交链均通过；
  所有执行的输入顺序一致，18 项 M2/M3/M4 与 M0 的配对/权重字节比较通过。
  M3 seed2027 的 predictions 和 diagnostics 首次/复跑仍不一致，完整阶段保持未通过。
  全扫描回执 SHA：`52d1a17a06b1edde9ecd90d8e420a6afe196fb6819475bc1912ad1e090bed6b0`。
- 从原 Linux 评价位置独立读回 13×4 份完整 report/metrics/runtime/golden-case，
  两个完整评价 plan/execution，以及三份实际 TrackEval 作者源码，共 59 项。
  每项原文件 size/SHA 与本地实际字节一致；第一版运输脚本因换行转义失败，失败目录
  保留，修正后使用另一个新目录 `metric-original-readback-v2`。
- `stage1_sequence_statistics.py` 从全 21 序列的 TP/FN/FP、关联加权分子及
  IDTP/IDFN/IDFP 独立重算完整官方合并曲线；13 个配置的最大误差均为 0。
  这核对了原指标的汇总算术，没有独立重跑逐帧匹配或 nuScenes 评价引擎。
  新统计入口的 10 项软件检查覆盖非宏平均、重复序列独立 ID、零 TP 尾部、计数
  损坏、固定完整序列抽样及确定性对照的未知样本标准差。
- 使用固定 seed1337、PCG64、10,000 次全序列配对 cluster bootstrap，所有种子
  与模型共享实际抽样矩阵；只报告 HOTA/AssA/DetA/IDF1。每次抽样重新合并充分统计，
  然后取种子均值。M1 只有一个确定性执行，样本标准差为 null；M3 原三种子值完整
  保留，但 strict parity 失败使其不进入通过复现要求的模型 bootstrap 比较。
  已见 SPD val 的 M4−M0 HOTA 差为 −0.035378664599607645，95% 序列区间
  [−0.04152818762470179, −0.029714222773461328]，作为负结果保留；不能推广为
  未见数据结论或恢复方法的收益。统计回执
  `independent-sequence-statistics/statistics-receipt.json` SHA：
  `c4115d1eacc5020e085956a305b774ef709af115f8f7a7cd7311af35a316a0a7`。

完整误配—融合—birth/kill—碎片因果链及真实错误持续时间、M3 seed2027 原因、
完整独立逐帧评价、nuScenes 全阈值 bootstrap 和其他数据集/主方法统计仍未完成。
现有 first-anchor 的 ID episode 不代表真实错误身份。E/K 与论文资格均保持 false。
截至 01:17:40 CST，两项最终模型绑定 teacher 和三项 recovery-off GPU 任务均
in_progress，尚无完成产物；V2V seed1337 已生成 8 个完整序列行 shard，整批独立行
验收仍等待候选 manifest，不能由 shard 数宣称训练输入或最终 checkpoint 通过。


### M0 原封存逐字节对照及 M3 数值路径补验（01:24 CST）

另从原始 `tracking-run-v1` 读回三种子的 predictions/association 六份完整字节，
加原 plan/summary，均匹配机制计划绑定的封存 SHA。M0 首次/复跑相对原封存的
12 项完整字节比较全部通过，回执位于上述产物根的
`original-sealed-M0-readback/byte-parity-receipt.json`。这补齐了实际原基线字节，
未以原 receipt 的 reported parity 代替读取。

在原 Linux 主机、原 NumPy 1.26.4 下，以全 SHA 绑定的原缓存帧
`vehicle-side/0003/000870`、23 个实际选中源行及原纯函数 AST 做新进程诊断：
默认 CPU dispatch 的全部 23×9 状态逐位等于首次执行；新诊断子进程单独禁用
AVX512F 后的全部状态逐位等于历史复跑。两者 23 份完整 covariance SHA 均与
原诊断一致，最大 yaw 差 4.440892098500626e-16。五个差异源 yaw 的两种数值
由 CPU dispatch 变化可控复现，不再只停留在 NumPy/libm 数值相近的观察。
完整诊断回执为 `M3-2027-full-source-frame-controlled-dispatch-receipt.json`。
没有修改生产进程 CPU feature、源缓存、历史流或容差；原进程切换计算路径的
具体触发状态仍未捕获，M3 seed2027 的 strict failure 与 E 未通过状态保持不变。


### 独立逐帧指标引擎重算（2026-10-07 01:40:51 CST）

`stage1_independent_evaluation.py` 单独从原封存预测及 GT 构造 car 50m 严格
ROI、原生帧/时间、序列作用域 ID、轨迹平均分数和旋转框相似度；没有导入原
`tracking_evaluation_v2` 适配器或生产器。独立 Sutherland–Hodgman 凸多边形
裁剪在共同局部原点计算 IoU，不调用 GEOS 或原 IoU/ROI helper；长度/宽度、
旋转、平移、包含/接触、类别、严格边界、重复 ID 和指标计数的 8 项软件检查通过。

独立构造的数据实际调用原未修改 TrackEval 1.0.0、nuScenes 1.2.0 和
motmetrics 1.4.0；完整作者 Python 源树及全部原版本在运行前后重查。
全部原 21 序列/3,316 帧和 prediction commit 逐帧核对；评价器仅读原 GT，
不进入模型、预测、参数选择或历史状态回写。40 个 nuScenes 阈值曲线、每序列
HOTA/Identity 完整结果和完整 cohort 指标均与原实际文件比较；计数零容差，
浮点容差在执行前固定为 1e-12，只排除 eval_time。失败输出会完整保留。

源码冻结于
`/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-stage1-independent-actual-metric-engines-20261006T173323145100Z`，
实际执行/读回目录为
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-independent-actual-metric-engines-20261006T173323145100Z`。
新科学评价源文件发布到原评价环境的独立新目录，并逐字节读回 SHA；没有包含
曾被拒绝的 CUDA branch-state 或 Linux priority 包，没有修改原生产或评价源码。
本地协调器 PID 78986 保持串行 13 个完整配置，CPU 每项固定 1 线程；不作为
同资源性能测量。当前已产生 3 份终态 outcome，其中
3 份通过。
M0 seed1337 全指标最大浮点差为 1.837419105754634e-13，计数全部一致，
实际完整引擎运行约 114.94 秒；其他配置须等待各自真实终态，不能继承。

整批 13 配置独立引擎验收、nuScenes 序列 bootstrap 和真实错误持续时间/传播链
仍未完成。M3 seed2027 的预测复跑失败继续单列；评价指标重算通过不会修复它。
E/K 与论文完成标志仍为 false。


### 全 13 配置实际指标及可观测事件链收口（2026-10-07 02:00 CST）

上述实际指标引擎协调器已终态，13/13 配置全部独立重算通过。每个配置完整覆盖
21 序列、3,316 帧；全部计数零误差，完整曲线与汇总指标最大浮点差
2.2026824808563106e-13，低于执行前冻结的 1e-12。整批回执
`rbf-stage1-independent-actual-metric-engines-20261006T173323145100Z/campaign-receipt.json`
SHA 为 `b6e3d8f6d4a913cd2110fe17fb571e8ee4354353d2770fb4bf53d41c058d9591`。
M3 seed2027 的实际指标重算通过不修复其严格输出复跑失败。

`stage1_identity_episode_oracle.py` 与 `audit_stage1_identity_episodes.py` 在独立几何
输入下重构原 TrackEval Identity 0.5 IoU 全序列整数指派目标，每项正重叠 owner
通过禁止该边后重新求解的整数目标差核定唯一性；同等最优解、零重叠和未指派
保持未知。逐帧只接受双向唯一的 >=0.5 BEV IoU 对应，不用首次出现的 GT 当作
任意身份锚点。未来 GT 只供封存预测之后的评价，不进入预测器或历史状态。

15 项相关检查通过后源码冻结并串行运行了全部 13 个真实配置。所有原生帧、
GT、诊断和输出 commitment 对齐，21 序列 IDTP/IDFN/IDFP 每个配置完全一致。
逐帧记录实际源配对组件、节点、birth/update/miss/kill、封存输出及片段变更引用。
道路残余节点只执行未匹配支持，其非零 matched_mass 是其他假设的关联质量，
单列为未执行的配对质量，不能算成该节点执行过的 CI 或错误融合质量。

完整事件链审计目录：
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-native-identity-episode-and-structural-chain-20261006T175705414418Z`。
13/13 审计通过的整批回执 SHA 为
`4ed67fc9ed7ea9fd2a6d3b31ca0e33aac577c27fb901642cb6a21285f869bb26`。
M0 seed1337 有 5,598 个明确正确、396 个在唯一全序列身份目标下错误、20,654 个
未知的 GT 帧观察，形成 205 个可观测错误片段，其中 204 个右侧删失。
只保存相邻真实帧间隔、观测跨度及明确正确的后边界；未知、缺帧、ROI 退出、
更换错误预测和序列结束均不外推。不能把删失跨度当作完整真实错误持续时间，
不能跨未知区间拼接，不声称隐藏时间连续或物理源身份已证明。

这些实际组件和生命周期引用证明结构联系，未识别反事实因果效果。观测到的
prediction identity 变更与官方 nuScenes Frag 分开保存。完整误差传播因果证据、
nuScenes 10,000 次序列重采样以及其余 A–K 项仍缺，E/K/论文均保持未完成。

新增 `stage1_nuscenes_statistics_oracle.py` 和
`stage1_nuscenes_sufficient_statistics.py` 为每个序列的全部真实轨迹均分截止值
调用未修改作者匹配器，保存离散预测子集的整数事件计数和距离和；仅在独立
合并这些充分统计后完整复现原 40 阈值及 headline 时准入。序列重采样会重新
构造整个 cohort 的 MATCH 分数、GT 总数和召回阈值，不平均序列 AMOTA，也
不把原冻结阈值直接用于新抽样。9 项软件检查覆盖重复序列、MATCH 与 SWITCH
区别、最高召回 tie、不可达到召回的惩罚和未知 FP/IDS/Frag。软件通过还不是
实际全部截止值缓存或重采样的完成证据。


### nuScenes 实际截止值缓存与配对重采样接续（2026-10-07 02:14 CST）

首轮新缓存入口在作者配置初始化之前构造 TrackingBox，在新进程中触发
`Unknown tracking_name car`，停止后续 12 个配置。失败目录
`rbf-stage1-nuscenes-all-local-score-subsets-20261006T180614343503Z`、源码冻结和
已发布源字节均保留。修正初始化顺序后使用新的目录和源码版本，没有覆盖旧源。
实际作者环境的新进程检查确认 TRACKING_NAMES 初始为空，原配置初始化后包含
car；纯构造检查不代表完整评价。

V2 实际协调器在
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-nuscenes-all-local-score-subsets-config-order-v2-20261006T180941649939Z`。
M0 seed1337 的 21 序列全部 1,207 个实际预测分数子集已调用未修改作者匹配器，
完整原始 40 阈值曲线与全部 15 个 headline 独立重构通过，最大差
3.3306690738754696e-16。缓存 SHA 为
`9b8a05757f0285d31b5332ebfea7c2cc37447df852a50ff897f88e5453dc2a8c`，
真实运行约 137.28 秒；本地 NumPy 2.4.1 从同一缓存重算全部原曲线及 headline
也通过同一固定容差。其余配置独立执行、独立准入，不继承首个结果。

`bootstrap_stage1_nuscenes.py` 使用已冻结的 seed1337/PCG64/10,000×21 全序列
抽样字节，每次抽样复制独立序列的 MATCH 分数、GT 总数和完整事件充分统计，
重新求全局召回阈值，并从每个序列的实际匹配子集缓存取相应结果。MOTAR 使用
作者 MATCH 计数；recall 和 MOTP 使用 MATCH+SWITCH。最高召回 MOTA tie、
不可达阈值 AMOTA/AMOTP 惩罚、空匹配的未知 FP/IDS/Frag 均保留原定义。
有任意不定义的抽样时不丢弃该样本，区间标为不可用并报告有效数量。
AMOTP 为中心距离米，越低越好，不能移用于 V2V 原生 IoU AMOTP。

共 12 项针对统计及准入的检查通过，来源冻结在
`rbf-stage1-nuscenes-complete-shared-bootstrap-after-admission-20261006T181425503427Z`。
本地协调器 PID 99705 只等待当前真实缓存协调器 PID/command 和准备 SHA；全部
13 缓存实际准入后才会自动启动完整配对重采样。死亡、PID 复用、超时、源改变或
依赖未准入均保存失败而不执行。M3 原全部抽样可保留供诊断，但 strict failure
使其没有合格推断区间；M1 只一个确定性执行，样本 SD 不造三次重复。
当前完整重采样尚未发生，软件检查和一个缓存不能代表 K 或论文完成。

13 配置实际指标与可观测结构链的完成证据已创建注册 Task
`65f453f6d90e492db355cb89be5ac530`。227 个成员，原始 3,771,295,125 字节，
压缩归档 668,694,389 字节、SHA
`8f99dc5d9c456c45783e8e0cde010ecf8b8469eced3fed3c9e0c170525692292`。
目前仍在上传，尚不能称为云端完成或独立读回通过。登记流程使用同一 Task、
一次上传和原 manifest 校验，保留所有部分文件，没有重复注册。

### 完整 nuScenes 配对统计与融合数值验收接续（2026-10-07 02:54 CST）

上节归档后来因未附加认证的文件服务器上传返回 HTTP 401，原进程失败退出。
原 SDK 仅将当前配置的文件服务器识别为需要认证，旧脚本硬编码的另一个地址
不在该列表。原失败、日志和冻结源全部保留；新恢复脚本使用 SDK 当前配置，
继续同一 Task `65f453f6d90e492db355cb89be5ac530`，没有创建重复 Task，也没有
改全局服务器或认证配置。4 个注册产物及全部 227 个实际成员已独立字节读回。
恢复证明在
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-completed-evidence-same-Task-configured-file-server-resume-20261006T182325894616Z/independent-readback-receipt.json`，
SHA `c77081ce9906197e695e62612ab58ae2b8ba7d0b12a12a15dd459553e9bee595`。

V2 的全部 13 个真实作者分数子集缓存现均完成独立准入，整批证明 SHA 为
`7302b4f020c9083d6845ac2d77e8b87f279d5784fb3e788e0d200f91cb496a41`。
依赖准入后的实际共享序列 bootstrap 已完成，原始数组为 13×10,000×15，
没有非有限值，SHA
`02cd4dd172176b74f94694f76bf062c90d3c20423a194d54e41d2d9eda799474`。
完整结果在
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-nuscenes-complete-shared-bootstrap-after-admission-20261006T181425503427Z/actual-full-bootstrap/bootstrap-receipt.json`，
SHA `3def79e22a108c8dbe275f433d332e616f66532e7736a162b65ba7b91592b176`。
15 个实际统计量为 GT、TP、MOTAR、MOTA、MOTP、FAF、MT、ML、FP、FN、IDS、
Frag、recall、AMOTA、AMOTP。每次抽样重新构造全局阈值，未采用序列指标宏平均。

新增 `verify_stage1_nuscenes_bootstrap_summary.py` 使用 Python fsum、排序统计量
和样本方差独立读回全部数组及来源，复算所有均值、SD 和合格区间。345 项
数值检查、105 个区间全部通过固定 1e-12，最大差 2.842170943040401e-14。
该读回验证统计产物，没有冒充再次执行匹配引擎。M3 三种子原始抽样保留，但
seed2027 的 strict failure 使 M3 没有合格推断区间；M1 的样本 SD 为 null。

M0 的 AMOTA 三种子均值为 0.33138042281101265，样本 SD 为
0.00796336540111582；M4 为 0.260394550768899，SD 为 0.00330958601709186。
M4−M0 的 95% 配对区间为 [−0.09639589576961337, −0.04199523249068433]。
M1、M2 相对 M0 的 AMOTA 区间均跨零。M4 的 Frag 改善与整体 AMOTA 下降同时
保留，不能用局部碎片减少支持整体质量优势。这里仅为 SPD 已见 val 的探索性
M0–M4 结果，不能替代恢复方法、V2V、同资源或因果优势结论。

全部统计、13 个实际作者缓存及独立输出读回已封存为 64 个成员，归档
11,553,665 字节，SHA
`4ebc72fc7b8ba5d0205eae302fca2a17079563642c4d9e1df9257e96a557063d`。
新 Task `b83879763f464873835e1a90f35a9da9` completed，3 个注册产物及全部
64 个成员均独立读回；证明 SHA
`d07efdf230563856fffe55c7f4f16ae0684049d5fbea08ac36f89c768bc7e24a`。
此统计归档引用原指标/结构链 Task，原指标和结构链大文件仍在先前独立封存包。

新增独立数学入口 `stage1_fusion_numeric_oracle.py` 与完整帧入口
`audit_stage1_fusion_numerics.py`，11 项检查通过。独立方程重算原始源状态的
legacy 轴/yaw、世界姿态和时间输运 Jacobian，保留完整 9×9 协方差交叉项；
CI 使用线性求解，混合保留假设间方差。原 sidecar 只记录完整协方差 SHA，
故严格白名单 AST 仅从已绑定原纯数学源恢复参考协方差；参考矩阵必须先逐字节
匹配原 SHA，才能接受独立方程比较。没有导入原模型、tracker、特征或缓存适配器。
均值/协方差 atol 固定 1e-8、rtol 1e-10，在实际全集执行前冻结，未放宽。

原始前两帧 smoke 通过后启动 13 配置完整批次，位置为
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-full-sealed-decision-conditional-fusion-numerics-20261006T184812250702Z`。
M0 seed1337 已完整通过 21 序列/3,316 帧、124,061 个源状态、165,345 个组件、
124,061 个节点、72,358 个时间匹配、103,423 个输出和 485,825 个精确参考协方差
preimage。完整协方差最大差 1.262634441445698e-11，均值最大差
2.284572531152662e-11，证明 SHA
`b3c0669d80b207596a78692287853ca18a6d261faa1afca72e174542dbae5c2c`。
M0 seed2027 也已完整通过，证明 SHA
`e9908fa5fe8d184c43193fd259750ae647e31bcb8acc98e68b244b14743c30fa`。
其余配置继续逐个执行，不继承前两个结果。

数值检查以原封存离散配对、生命周期和上一帧实际完整输出为条件，验证各步
实际融合和状态更新；它没有独立证明全局分配、从零数值回放或误差因果反事实。
已有结构链和全序列可观测身份持续时间仍保留未知、歧义与删失。完整 E、K 和
论文均未验收。5 个 teacher/recovery-off GPU Task 最新元数据均为 in_progress，
没有注册产物；V2V 三种子身份输入生产仍无最终 manifest。继续原进程，保留
2027 坏 rank5、所有历史失败和授权门槛；没有提交或推送工作树。

### 完整独立时序回放、负结果图与当前覆盖边界（2026-10-07 03:32 CST）

上述条件融合数值批次已完成全部 13 个配置，独立逐源、组件、节点、时间更新和
完整 9×9 协方差比较均通过；整批证明 SHA 为
`81ae9aa60e02a5567b9287cf750e7281cf6b445ffc852fe290959bfe2b4dc488`。

新增 `stage1_discrete_replay_oracle.py` 与 `verify_stage1_discrete_replay.py`。
独立最短增广路径求解器不调用 SciPy 分配，逐帧核验原始/对偶可行性、选中边
紧性和目标间隙。该入口从原始缓存及封存跨端配对备选出发，独立持有并传播
均值、完整协方差、分数、身份假设和生命周期；原离散选择和上一帧实际输出
只作比较目标，不参与独立决策，不在分配不符后回退到原选择。8 项有意义的软件
检查通过，含穷举小矩阵、贪心冲突反例和 birth/update/miss/prune；与已有独立
融合数学检查合计 15 项。一次检查的解析预期遗漏传播后的 x/v 交叉项，已修正
解析预期并保留检查历史，未改变实现或容差。

实际全集批次位于
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-full-independent-discrete-continuous-sealed-association-replay-20261006T190301130932Z`。
13/13 配置各自完整 21 序列/3,316 帧通过，整批 SHA 为
`832a96cc64cf3a9f50213ede96273c5ad03477983e2f5fac4843fae554df3e04`。
M0 seed1337 的 3,316 个分配证书、72,358 个时间匹配和 103,423 个完整输出
均通过；最大原始/对偶间隙 1.1368683772161603e-13，均值最大差
2.73656e-11，完整协方差最大差 1.05462e-11。固定 atol 1e-8、rtol 1e-10，
没有放宽。M3 seed2027 的新数学重算通过，但原 strict 字节失败仍为 false；
新数值容差验收不能替代原字节复现，也没有重新执行神经配对或误差因果反事实。

仅在这两批全部 26 个完整验收通过后，守卫协调器创建 Task
`b7194755f42d4e0bbc5c0241c4848aa1`，现在 completed。156 个成员及 3 个注册产物
全部独立读回；归档 95,758 字节、SHA
`e72294a9a3389979906bd6713b00fbc90eb4a1afbb60fd2bf48b4ed51f618489`。
独立证明位于
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-full-fusion-and-independent-discrete-replay-archive-after-admission-20261006T191303791484Z/independent-readback-receipt.json`，
SHA `9ef6f3e854a4066f9231010dbe2c4f1ad2ca33e7d5ee04f57b3ad626abf28a3a`。
原指标、结构链和统计大文件分别引用此前已读回 Task，未冒充本包中的成员。

`plot_stage1_qualified_paired_effects.py` 从固定 SHA 的两套完整统计读取均值与
95% 配对区间，并从全部 10,000 个实际抽样值再次核对 AMOTA/Frag 区间。
PNG、SVG、灰度 PNG、完整绘图数值及 raw 均值/样本 SD 位于
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-qualified-paired-diagnostic-figure-20261006T192833823241Z`。
9 个点/区间绑定全部核对，彩色和灰度 PNG 实际视觉检查通过；SVG 解析通过，
没有声称单独渲染过 SVG。M4−M0 的 AMOTA 为 −7.098587204211365 个百分点，
HOTA 为 −3.5378664599607645 个百分点，Frag 为 −59.666666666666686；三项
配对区间均不跨零。碎片减少与总体质量下降同时保留，不能称恢复独立收益。
图中明确 SPD 已见 val、条件序列不确定性、M1 单次无种子 SD，以及 M3 没有
合格区间。新图源使用隔离绘图环境，没有改变活科学进程的依赖。

图、精确数值、视觉检查与冻结源另外注册为 Task
`7eaf934e95434b938659dddedf44770a`，completed；全部 9 个成员独立读回，证明 SHA
`9f1d3f03f6642d819cdb09eaf4afd8e8f65f87842619ad915412f099ef8ecf58`。
完整 E/K 和论文验收仍为 false。

针对 M3 的固定真实源帧，另做 256 个同值 yaw/heading 缓冲区偏移组合，全部
得到 first 输出，replay 输出为 0、其他输出为 0。来源、原输入和值均保持不变，
没有改 CPU 特性或生产数学。该负结果只说明所检查的缓冲区对齐不足以触发
历史差异；历史环境触发原因仍未确认，原失败继续保留。

F 的已验共享原始输入实际只有 initial、deadline 前迟到消歧以及 12 个 compute
事件；3 个冻结压缩流均完整解码核对。既有重传、冲突 ID、未完整到达和过去决策
4 项协议边界仍有效，但不能自动算作交叉、遮挡、空侧、无允许边、预算耗尽、
错误 gate 和过期历史的完整机制结果。本次覆盖边界证明在
`/Volumes/Data/test/recover-before-fuse/receipts/rbf-stage2-current-raw-event-coverage-boundaries-20261006T191656551626Z.json`。
没有重跑已验矩阵，也没有把预算参数或过去事件拒绝改名为缺失场景。

J 新核查作者公开主页的 56 个自有仓库，以及 Long-SCOPE 的 CVPR2026 官方论文
页面；所查范围没有 Long-SCOPE 原生实现/权重链接。官方页面仅列论文 PDF 和
arXiv；这不能证明其他位置不存在资产。原始 API/HTML 均保留在
`rbf-Long-SCOPE-public-author-assets-refresh-20261006T190506604341Z`，原生复现未完成。

03:29 CST 的 5 个 teacher/recovery-off Task 均 in_progress、注册产物为空，
实际 console 单列保存。两个 teacher 和 recovery-off1337 有新增事件记录；
另外两个任务的所取 console 尾部未新增，不能用心跳冒充事件推进或整体百分比。
V2V seed1337 准备进程继续运行，已看到 24 个 NPZ 路径、最终 manifest 为 0；
先前仅 21 个封闭分片有完整 CRC 观察，尚无完整行/监督/模型资格。保留所有原
生产、独立检查和训练输入守卫进程，没有新 GPU 派发、提交或推送。

### 第二阶段原空侧复用与错误 gate／预算耗尽实证（2026-10-07 03:56 CST）

继续按完整 A–K 目标推进。上一轮为 progress，本轮没有缩小 F 的完成条件。
在共享 14-event 原始包之外，找到另一个历史空侧／无边实验。旧本地汇总仍写
queued，但当前原 Task `6dbdd050c5d847eebdd23b2e96d59eb9`、
`e2b2c439a80a4c698db1a9d3871114b4`、`67e47f68da43497cad368fcd639cd92c`
实际均 completed。9 个原始 case（两侧空、source0 空、无允许 parent 边，各三种子）
的 24 个注册资产已直接按原 URL 读回并核对 size/SHA，没有重跑原实验。

复用 SHA `91af36bb2baac61711ccd93283008d26fb9ffcc6a28bb1e5bb8360334c022218`
的原注册源码及原独立 oracle、SQLite/verifier；全部 18 个实际事件重算精确
合法历史、Z、零 regret、零遗漏质量，另独立重算全部 12 个完整 9D birth 输出
（包括 x/v 和 y/v 交叉协方差及分数存活项）。只对私有数据库副本重新请求已封存
初始事件，预测字节与全部逻辑表不变；原数据库不变。新检查器首次把 genesis
previous-commit 预期误写为 null，按原后端明确的 64 个零字符契约修正到新版本，
原失败、源与原输入均保留；不是原实验失败，也没有放宽提交链检查。
新的原产物准入在
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-original-empty-edge-readback-genesis-v2-20261006T194051501914Z/independent-admission-receipt.json`。
该证据覆盖原有限 CoveredCompletion 配置，不能直接继承为当前 exclusive/learned
全部对照的空侧资格，更不能当作错误真配对 gate 实验。

新增 `run_stage2_gate_and_budget_events.py` 和
`audit_stage2_gate_and_budget_events.py`。原始九维检测、时间、有限正势和允许边
在运行前冻结；固定真身份只供另一个独立检查进程读输出后评价，未传入 tracker。
两个新场景 × 3 个固定 fixture seed × K1/4/16，共 18 个配置、90 个实际事件：

- 错误 gate：先移除 c→b（其几何距离小于 c→a，也小于候选距离门限），仅允许
  c→a 与 birth；这是一项显式有限候选干预，不是在诊断真实缓存 gate 的 bug。
  9 个配置均选择错误身份 a，模型同动作集 regret 为 0，而固定评价身份损失为
  1/3，真身份动作不在模型支持中。有限模型的 bound 没有失效；零校准误差或
  真实身份保证不适用。5 个事件后仍不因额外计算恢复不存在的 b 选择。
- 实际预算耗尽：保留 c→a 和 c→b。每事件支持的 persistent 展开预算固定为 1，
  首事件及所有后续事件实际均花完 1 次；未提高预算。K1/seed1337 的真实模型
  regret 为 0.17499306249295501；遗漏质量从 0.9999999999999509 降至
  0.47502081252108574，但当前动作仍为未匹配 c，不能把前沿质量下降称为
  已恢复当前决策。这不是正式资源候选扫描或隔离成本测量。

首轮预算配置试图指定该 persistent 后端不使用的 finite-window `action_budget=0`，
在创建预算 tracker 之前被明确拒绝。原失败、冻结源和已完成的三个错误 gate
产物保留。新版本移除未使用参数，实际 persistent 展开预算仍为 1，实际 decoder
限制保持原值；没有提高失败 runtime cap。先前三个完整错误 gate 产物直接复用。

新候选与独立验收位于
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-new-gate-budget-supported-persistent-v2-20261006T194857251585Z`。
独立检查不导入生产模块；从每个封闭 SQLite 的原始行、允许边、前缀祖先和
实际 commit 重构合法 action、完整 posterior、精确 Z、遗漏质量和同动作集
regret。逐个前沿区域核对 prefix-minus-retained-leaves 的真实质量与上界，核对
覆盖质量 + 保留质量 = 1、条件组件 Z 乘积、所有完整 9D 状态和分数。原有独立
80 位 Decimal history oracle 直接复用，没有重跑已验的随机矩阵。新 225 个
九维预测状态全部核对，最大数值差为 0；atol/rtol 固定 1e-8，离散 bound 的
float 比较固定 1e-11。仍没有声称区间数值证书或真实分布校准已完成。

两组完整证据、原资产、独立源及两个历史失败已注册为 Task
`eeda4956a37b40f4aadca8266e99977f`，completed；89 个成员和全部 3 个注册产物
均独立字节读回。归档 540,376 字节，SHA
`096a2d0d31081634226622eb9a5c233e52e31fed5e54c4c68c9b83ecbbcd6f31`。
读回证明在
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-complete-empty-gate-budget-accepted-archive-20261006T195318177088Z/independent-readback-receipt.json`，
SHA `6e0d9e6e2376f382aff0f3b676dedfc5856505e414760bffceb8d76cbac50572`。
合计 27 个特定有限 case、108 个事件、237 个完整 9D 状态；这些计数不能加到
真实数据实验数或论文完成率。新原运行 backend snapshot 仅在本地保存并绑定
SHA，没有上传 runtime 包，更没有运输先前被拒绝的 GPU/Linux priority 包。

F 仍缺真实轨迹式交叉、遮挡后重现、明确不可恢复的过期历史，以及全部合法历史
与 learned/不可恢复/MHT 等对照的完整比较。先前四个协议边界、Bayes/精确矩阵
和违反前提的反例继续复用；不能将本次静态几何反例改名为车辆交叉或遮挡。
最新 03:52 CST 的两个 teacher 和三个 recovery-off Task 均 in_progress，仍无注册
产物；两个 teacher 有新事件记录，三个 recovery-off 所取尾部没有新事件，不能
用心跳当作实际事件推进。V2V 身份行生产和两个准入守卫继续存活，当前 26 个
NPZ 路径、最终 manifest 为 0；未派发未经完整行资格的训练。

### 第二阶段真实时序的交叉／缺检后重现实证（2026-10-07 04:12 CST）

本轮为 progress，完整 A–K 目标仍 active。新增源冻结的有限合成 CPU 事件：
两辆同外观、同速率 1.5m/s 的车在 y=-2/+2m 不同车道反向运动，信息时刻
1.0s 到 1.4s 的 x 排序发生交换；这不是物理接触／碰撞。缺检后重现样例在
1.2s 仅 A 有原始检测，B 在 1.4s 重新出现。它是遮挡的缺检代理，没有验证
物理遮挡原因；没有把缺失检测加入 existence likelihood，身份不确定性仍单列。
允许 parent 支持、严格正势、时间和全部预算在运行前冻结。fixture seed 不等同
训练模型 seed。生产器按到达顺序只 append 当时可见的检测，没有传入 evaluator
身份；新观察到达不得早于此前决策，首事件封存预测和 audit 再请求均逐字节不变。

新增 `run_stage2_crossing_occlusion_events.py`、
`audit_stage2_crossing_occlusion_events.py`、
`verify_stage2_causal_recovery_witnesses.py` 和
`plot_stage2_causal_event_traces.py`。2 类场景 × 3 个 fixture seed × K1/4/16，
18 个配置的全部 261 个事件、690 个完整 9D 输出状态已经独立验收。检测状态有
非零 x 速度及 x/v、y/v 协方差交叉项，全部 unequal-time 传播、CI、完整协方差
和分数重算最大绝对差为 2.220446049250313e-16。未改变固定 1e-8 atol/rtol
和 1e-11 离散质量比较门限。另完整核对 1224 个 residual region 的真实质量、
保留/覆盖分割、Z、同动作集 regret 和组件分解。全部 coarse class 另读回为 car；
该补充 class receipt 单独登记，不能冒充已经进入稍早的 57-member archive。

27 次真实新观察转移使用旧合格 80 位 Decimal 历史 oracle，独立从每个合法
suffix 的正势直接求条件 likelihood；不是用已归一化 posterior 比值伪装独立
likelihood 重算。核对 eta_y=Z_new/Z_old、每个旧历史更新后的 posterior、有限
kappa 和遗漏质量放大界；旧历史支持未删除。crossing/1337/K1 初始遗漏质量
4.538717908255e-5，新增同源互斥证据后，旧 active 集外质量达到
0.9999999958767218，kappa=5.343237290762981e12，eta_y=4.5387179269702915e-5。
该结果说明新证据可强烈放大旧遗漏质量，不是小先验遗漏足以保证真实身份。

15 个配置有 current-output recovery：初始 c→b 根身份既不在 active 集、也非
旧输出，只被 frontier 区域覆盖；后续选择的 predecessor-prefix 在上一决策时
尚未 discovered，随后进入当前 active 并把当前 c 身份由 a 改为 b。另一独立
检查器从不可变 SQLite prefix/member/discovery 行重构投影，核对实际旧 active、
旧 output、discovery 时间和当前 commit，不能仅继承 audit 中的 recovery 标志。

交叉场景 9 个配置均恢复；缺检后重现的 K4/K16 共 6 个配置恢复。
缺检后重现 K1 的 3 个种子在固定 3 次／事件展开上限和 12 个预声明 compute
事件后仍失败。seed1337 最终 roots 为 [a,b,a,a,e,f]，模型 regret 为
0.4999999948459861，固定身份损失为 0.5，遗漏质量为 0.9999999979383609。
没有追加 compute 脉冲、提高 K/预算/工作上限或放宽门限追求成功；这三个负结果
不是验收失败，完整模型和状态检查均通过，但恢复机制在这些有限预算下未成功。

全部候选/SQLite、独立验收、时序 witness、冻结源、计划和经过彩色／灰度视觉
检查的科学图注册为 Task `e4f556e73df343789a682c204cf1f6fe`（completed）。
57 个成员和 3 个实际注册产物全部独立 size/SHA 读回；归档 1,489,580 字节，SHA
`355f5e865393331493901d461856ef2ed04caf6a3198be66bc0cb5bb06601c65`。
读回 receipt SHA `672ca724470f7069e4090f6a20b40748fce7125777df7d794501ccf6078d7a2c`，位于
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-accepted-causal-event-archive-20261006T201013133286Z/independent-readback-receipt.json`。
实际计算和独立证明位于
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-causal-crossing-occlusion-20261006T200442443636Z`。
该科学图显示三个实际事件轨迹；横轴明确是事件 ordinal，不是执行延迟。SVG 已
检查 XML；实际视觉检查仅彩色及灰度 PNG，不声称独立 native SVG render 已验。
图与数值不证明 learned、真实数据或同资源收益；同展开上限不等同同资源。
运行时 backend 仅复用本地已有冻结源及哈希；未上传 runtime 模块或曾被拒绝的
CUDA/Linux priority 包，没有新 GPU 派发。

F 仍须补这些场景下的强基线、不可恢复对照和完整 learned 比较，以及明确的
不可恢复过期历史。当前 persistent 的 window_us 仅选择决策 loss interval，
不会删除旧历史；expired_output_only 也不是不可恢复的 history expiry。
旧有限窗口 CausalForestTracker 会在窗口结束后要求显式 handoff，尚不能凭
这个 reject 声称执行了合法 irreversible support handoff。下一步需冻结并核验
该边界合同，并对新时序输入补齐实际基线，而不是重跑已验的随机／Bayes矩阵。

最新 04:05 CST 核验中，两个 teacher 有新增实际事件记录；三个 recovery-off
所取 console 尾部未新增，均仍 in_progress 且注册产物为空。V2V identity row
原生产与两个资格守卫仍存活，最新看到 27 个 NPZ 路径、最终 manifest 为 0；
路径数不是封闭行、最终监督或新模型资格。保留原进程及脏树，不提交／推送。

### 第二阶段同输入不可恢复 beam 与全部分支状态（2026-10-07 04:29 CST）

本轮继续完整 A–K，为 progress。新增逐节点、整批完整扩展排序、整批 joint
排序三个不可恢复 beam 变体；它们是本地明确命名的 beam 实现，不是作者 MHT
复现，也不代表尚待决定的 Linux 正式 MHT 同资源合同已经准入。复用上一轮
18 个已验 RBF 配置，比较各相同 fixture seed、K1/4/16 与两个有限时序场景。
原三种子宽度组的 raw 检测、允许 parent 行、绝对 log 势和事件时刻逐字节相同，
提取为 6 个独立运行输入。生产输入只有 sequence/raw/rows/schedule，没有真身份、
RBF 输出或 audit。新 beam 的未使用 recoverable 展开参数保持默认；实际 beam
work cap 在运行前完整冻结，没有在失败后增大限制。不同搜索算法的工作计数
和同 K 都不等于同资源，未执行正式资源重复测量。

新增 `run_stage2_causal_beam_baselines.py`、
`audit_stage2_causal_beam_baselines.py`、
`audit_stage2_causal_branch_states.py` 和
`compare_stage2_causal_beam_recovery.py`。
54 个新的 beam 配置、全部 783 个事件和 1800 个完整九维预测独立验收通过。
独立检查不导入生产模型，用旧合格 Decimal oracle 完整枚举支持历史；逐节点
variant 在每个节点后筛选旧保留 root class 的合法扩展，batch/joint variant
对当前整批扩展排序。原始 parent 路径的同 root class 概率先求和，并独立从
最小 parent 和 root 索引重构递归 prefix SHA，验证全部排序和 tie-break。
还核对实际保留集、条件 Bayes 决策、完整原模型 Z/遗漏质量/regret、所有九维
输出、coarse class、允许边、完整 commit 链和封存首事件 byte parity。
该 corpus 只有一个已验证连通组件，没有新的 merge 场景；batch 与 joint 在
这些输入中相同，不能用本次结果概括它们的组件合并行为。

另一个独立检查器直接读取 18 个原 RBF 和 54 个新 beam 的封闭 SQLite，没有
重新运行已验 RBF。全部 72 个范围中，1224 个缓存 root-chain 状态（完整 mean9、
9×9 covariance、first/last/order/score）逐一从原始检测独立重算，最大数值差
2.220446049250313e-16，仍为固定 1e-8 atol/rtol。5597 个 active/output
branch-event 的完整预测字节 digest 由已数值验收缓存及显式传播重构；总共
17995 个完整 branch 九维预测核对，实际 output 分支也与原 commit 预测一致。
这些是多个前缀/分支的科学范围，不能将计数当作独立真实数据试验数。

三 fixture seed 的最终模型 regret 均值如下（四舍五入；具体逐种子值和事件
轨迹绑定在 qualified-same-input-comparison.json；fixture SD 不等于训练模型
seed 不确定性，未推导置信区间）：

| 有限场景 | K | RBF | 逐节点 beam | batch beam | joint beam |
|---|---:|---:|---:|---:|---:|
| 交叉 | 1 | 0 | 0.400000 | 0.400000 | 0.400000 |
| 交叉 | 4 | 0 | 0 | 0 | 0 |
| 交叉 | 16 | 0 | 0 | 0 | 0 |
| 缺检后重现 | 1 | 0.500000 | 0.333333 | 0.500000 | 0.500000 |
| 缺检后重现 | 4 | 0 | 0 | 0 | 0 |
| 缺检后重现 | 16 | 0 | 0 | 0 | 0 |

K1 的三个不可恢复变体均在初始事件永久剪去 c→b 根身份，之后 12 个额外计算
事件也不能改变 c 的身份；即使原始检测和已存档 SQL 前缀仍在磁盘，它们不再
是推断候选。交叉 K1 的 RBF 在 compute-2 恢复该分支并达到零 regret，这是
有限事件中保留历史支持的正例。宽 beam 在新的两车观察到达时就选对 c→b，
RBF 的 K4/K16 仍到 compute-2 才正确，因此没有把同宽 final regret=0 写成
RBF 更快／更省资源。缺检 K1 的 RBF 未恢复，且 regret 高于逐节点 beam；
batch/joint 同样未恢复。这些不利结果全部保留，不声称恢复收益一致为正。
24 行比较表来自完整 72 个已验配置、1044 个实际事件，仍没有 learned、
JPDA/PKF、正式 MHT、不可恢复时间过期 handoff 或同资源成本的完整资格。

新候选、完整排序/状态验收和比较位于
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-causal-same-input-irreversible-beams-20261006T202029922228Z`。
原 RBF 的 18 个 raw/SQLite 继续复用先前已字节读回的 Task
`e4f556e73df343789a682c204cf1f6fe`，在新 package 中只绑定原注册 archive reference，
没有重复原产物或运送本地 runtime 模块，更没有运输曾拒绝的 CUDA/Linux priority
包。新的完整归档注册状态以实际 independent-readback receipt 为准，不能用本段
准备说明代替云端读回。

最新实际 SDK Task console（04:24 CST）中两个 teacher 有新事件，三 recovery-off
尾部仍未新增，全部 in_progress 且 artifact 列表为空。SDK last_worker_report
也有当前心跳。只读 worker SSH 探测：A100 地址无已知 ED25519 host key，严格
校验拒绝连接；V100 地址连接超时。没有放宽主机校验、没有把探测失败解释为
实验终止或重启原 Task。V2V 行生产、原独立行检查和输入资格守卫继续存活；
目前 31 个 NPZ 路径、最终 manifest 为 0。守卫按每个种子独立接续输入准备，
不等待全部种子后才处理首个已验种子；准备输入不等于训练已派发／模型已合格。

04:30 CST 后新的 authoritative C 进展：seed1337 已生成最终 manifest 和 candidate。
完整 32 序列、14,210 源帧、7,105 paired frames、801,957 个原始节点，111,855
supervised rows；manifest SHA
`270a0080aa3fb0ad10910e9e5b56396887b78686ed1b45e5d4d07c29b8362431`，
candidate SHA `f9b35bc2511481adb975f73453c3f5fc10fdba75f5d389d0cff465a899baa40c`。
原 producer 转入 seed2027，原独立 checker 已真实启动 seed1337 的逐序列全量
验收；正在运行，不把候选或局部序列通过记作最终资格。

新增全 32 个封闭 NPZ 的监督覆盖诊断：801,957 节点中跨端 positive parent edge
为 0，时间正 parent edge 为 341,737；跨端已知负边 116,486，跨端 unknown 边
1,379,041，birth positive 225。这不是特征计算失败；旧适配器的“两端都为明确
nonmissing association ID 且相等”合同遗漏了作者原生的非对称身份映射。
全训练 source0 的 61,762 个 annotation association ID 均为 -1；source1 有
35,021 个明确 association ID。原生示例 scene
`testoutput_CAV_data_2022-03-15-09-54-40_0` / frame000001：source0 object1、ass=-1
和 source1 object7、ass=1 都映射到原生 ID1。旧 adapter 不建立这个合法跨端链接。
作者已绑定的本地 `fulltrain-native-export-source/official/.../base_postprocessor.py`
及本轮读到的[作者原生源码](https://github.com/ucla-mobility/V2V4Real/blob/main/opencood/data_utils/post_processor/base_postprocessor.py)
明确采用 ass_id 非 -1 时用 ass_id，否则用 object_id+100*cav_id。

完整监督覆盖 diagnostic 和单列 final-joint-training preflight gate 已保存登记，
见当前 progress checkpoint 中的对应 path/SHA。这些旧 partial rows 仍保留，原
numeric/support/partial-label checker 不修改、不重启；它们的局部资格没有被
重新改写，却不能被当成完整跨端联合身份训练资格。没有派发训练。下一步采用
独立新版本修正 target-only adapter：遵循作者原生映射，要求至少一端明确 assoc、
两个源各自 frame 内唯一、保留 track alias/conflict 的 unknown mask；复用原
prediction-only 特征、允许支持和已验 source labels，不能改写活 producer 的
源码或用额外物理会话映射阻断官方 benchmark。新 targets 必须完成独立全行
重算后，才进入新联合模型训练。

本轮新 beam 完整归档 Task `f783ae17d822473b81e493e817db3f8c` 已 completed，
133 个成员及全部 3 个注册产物均独立 size/SHA 读回。归档 2,481,239 字节，SHA
`5dc010c97180c95c3f230fb00c535e4306910f648760da5b804ff2151fca6f69`。
读回 receipt SHA `435c7ebfd65d9b52b15bbc2b63c56d36bd9bc09b53f14dac0b4e5fd8a07dff06`，位于
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-qualified-causal-beam-comparison-archive-20261006T202720712670Z/independent-readback-receipt.json`。
原 RBF raw/SQLite 依然通过旧 archive reference 复用，没有复制或重新实验。
完整 F、C 和论文收口仍未完成，所有不利结果和授权门槛保持。


### V2V 原生身份目标 v2：非对称关联修正（2026-10-07 04:45 CST）

旧 seed1337 的完整 32 序列、801957 节点已获得原合同下的特征/上下文/部分标签验收，但跨车正样本为 0，不能作为最终联合跨端身份模型训练资格。该原始验收、旧数据和运行中的其余种子准备/检查/本地输入打包进程全部保留；没有启动最终训练。

作者原生 ID 规则为 `ass_id if ass_id != -1 else object_id + 100*cav_id`。旧适配器要求双方都有相同非缺失 `ass_id`，漏掉了实际 CAV0 `object_id=1, ass_id=-1` 与 CAV1 `object_id=7, ass_id=1` 的合法 native ID=1 配对。新独立版本使用 paired frame 的 author-native mapped ID 相等、至少一端有显式关联、各端 key 唯一；两端仅 fallback 的巧合 ID 不自动建立跨车联系。原生对象轨迹及 54 条 alias/association-conflict 轨迹继续保守处理，冲突传播至整个连接分量；跨车合并后的同源同帧重复也置为未知，不声称额外物理连续性。

已执行全部 137134 标注、2035 个源轨迹的实际预检：生产端 union-find 与独立 directed-link 图闭包的组件分区、状态和 cross flags 完全一致；得到 34191 个 paired-frame 跨车链接，630 个 cross-linked components，127506 matched / 9628 ambiguous 标注。此预检不替代检测节点目标验收。

修正版三种子 campaign：`/Volumes/Data/test/recover-before-fuse/artifacts/v2v4real-native-identity-target-only-v2-20261006T204325976623Z`。新工具只重算 `positive`、`known`、`reason`；独立审计逐字节比较其余 15 个数组与已验旧数据，并从完整原始检测重新按分数顺序进行严格 `<2m` GT 匹配，再用独立标量 oracle 重算全部目标。每个种子都等待自己的旧完整数值/支持验收，不能继承另一个种子的资格。预检、计划和四个新工具已冻结；启动父进程 PID76539、单线程 CPU。目标尚在运行时不得声称完整 admission；已有 GPU 输入 guard 的旧 partial bundles 不能用于最终联合训练。

04:44 UTC 实际 ClearML console 更新：两个 teacher 都有新 replay events；recovery-off3407 从旧尾部126/197推进到151/197。recovery-off1337/2027仍显示相同旧尾部，虽然 task metadata/worker heartbeat 更新；这不能证明这些 schedule 已有新进展或完整产物。五个 task 均无最终 artifacts，保持活任务和分配不变。实际 console 原文与元数据已存入 `rbf-five-live-registered-task-console-progress-20261006T204428528023Z`。


本次后续实际结果（2026-10-07）：seed1337 的 native-v2 目标已完成并通过全部 32 序列、801957 节点的独立标量匹配/身份目标重算。验收 SHA=`ed7159f90acda47785e48335dbfabf884243b9de48c8e9c1cdece4f84032c61a`，跨车正边 78029、真实同源时序正边 330042；原特征/来源/时间/候选/上下文等其余 15 个数组 dtype、shape、原始 bytes 与原先已验输入完全一致。seed2027/3407 仍等待各自原始完整验收，没有复跑或重启前驱。

原始 28 fit / 4 holdout 分组的真实两类监督：fit 跨端 informative samples=35275、temporal=89696；holdout 分别 4383、17963。actual positive parent edges 为 fit 跨端70361/时序267938、holdout 跨端7668/时序62104。新训练合同检查同一个真实完整 manifest，并实际拒绝旧 partial proof、非逐字节复用 proof、fit 跨车正边为0及holdout跨车正边为0的反例。八个原训练/DDP/forward 实现定义与原版本 AST 完全一致，新的分组选 epoch/全train同seed refit 入口只强化 native-v2 资格与来源绑定，训练参数/选择规则不变。

`/Volumes/Data/test/recover-before-fuse/artifacts/v2v4real-native-identity-target-only-v2-20261006T204325976623Z/seed1337/GPU-inputs-native-v2` 的修正训练输入包已经完成本地 64 个 member 的全 SHA/size 读回：576072630 bytes，semantic identity=`2d181ceff727fafeb8763c964711992cdb4ce20bf7d3613e3bbef4c8f6a602d1`。训练-only 实际顶层导入闭包为27个源码文件，不包含 CUDA branch-state、priority 或 persistent inference 包。第一次源码枚举因 Torch 合成 `_classes.py` 的非文件 `__file__` 失败；失败记录保留，后续只枚举存在的真实源码文件，明确记录被排除的 `torch.ops`/`torch.classes` 合成条目。此源码导入检查不称 GPU 执行验收。

新训练包上传 task=`53b4fcaf1f014cbdaeb7dbc62aa35809`，实际注册 byte/member 云端读回尚待结果，不把上传过程称完成。当前20:52 UTC fleet没有合格空闲非L40S GPU4/8选择，5090 GPU0–7 已绑定活 task=`91e58cccc7e840f6b23e498dfef039bb`。sam3、两个teacher、三个recovery-off的既有卡分配保留，没有提前创建或 enqueue 最终训练。独立 native-v2 GPU消费者与全部最终行 NumPy checkpoint验收入口已准备；source/config/candidate SHA、actual UUID及既有 compute-process 相交检查仍须在实际更新前通过。


21:08 UTC 云端最终实际结果：同一个输入 Task `53b4fcaf1f014cbdaeb7dbc62aa35809` 的 4 个实际注册产物（input-bundle、local-byte-readback、bundle-plan、head-supervision-coverage）全部成功独立 HTTP GET，逐项 size/SHA 一致；576072630-byte tar 再次逐个校验全部64个 member。最终 cloud admission：`/Volumes/Data/test/recover-before-fuse/artifacts/v2v4real-native-identity-target-only-v2-20261006T204325976623Z/seed1337/cloud-input-publication/independent-readback-receipt.json`，SHA=`7feee5b6ad4549b05864ddc284652e0790aa5698b5acde540d648dbab58a90e3`。

随后实际调用冻结 native-v2 dispatcher 的 `--execute` 入口，重新核对原始4个注册资产、semantic recipe和实时physical worker交集；正常返回 `waiting_for_disjoint_idle_physical_GPU4_or_GPU8`、`training_dispatched=false`、ETA unknown。没有创建/enqueue新的训练 task，没有动既有七个 GPU worker 活绑定。此检查的 command、stdout/stderr SHA及完整返回已保存为 `/Volumes/Data/test/recover-before-fuse/artifacts/v2v4real-native-identity-target-only-v2-20261006T204325976623Z/seed1337/actual-native-v2-dispatch-check/receipt.json`。seed1337 完整本地及云端输入资格已闭合，实际训练/checkpoint资格尚未闭合；其余两种子继续自己的输入前驱。

### 有限同输入概率/MHT对照与真实历史过期反例（2026-10-07）

在六份已冻结交叉、缺观测/重现原始检测、因子、schedule 上，新执行 exact-scan
JPDA-CI、PKF 与本地 global-scan MHT K=1/4/16 共30个闭合运行。独立验收不导入
生产模型：穷举合法扫描赋值与完整历史，核对联合MAP及其等权最优动作、根类权重、
MHT逐源帧剪枝、提交链和完整9D输出；PKF另用展开测量与Joseph协方差推导，
JPDA保留未匹配分量及分量间均值方差。全部435事件、1104输出状态、5406保留分支
当前预测和300个SQLite根链状态通过原1e-8容差，最大绝对差1.6653345369377348e-15。
仅本轮零yaw夹具通过，不宣称一般圆周分布或原生公开方法复现。

新增30运行与旧18 RBF、54 node/batch/joint beam共102个已验同输入运行进行逐事件
对齐，源SHA与全部原始输入、时钟和完整模型Z一致。JPDA/PKF及MHT K1最终交叉
真实根损失为0.4、缺观测/重现为1/3；MHT K4/16为0。较宽MHT在新观测到达时已正确，
原RBF部分配置需额外计算事件才恢复；原遮挡RBF K1失败保持。该比较不声称恢复统一
优胜，相同K不是同资源，JPDA/PKF折叠状态协议与RBF的分支时序回放不同。
完整包Task `4ed55c6cff104d73afbcd9aa69b1e9f2` completed，84个成员及3个实际
注册产物全部独立HTTP size/SHA读回，receipt SHA
`093e1a9553cfe439b585a9ce239d903a2dd1f59bcd7f76de2f962f1eaf419e98`。
产物根：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-causal-same-input-exact-JPDA-PKF-local-MHT-20261006T212222378748Z`。
比较脚本v1假定旧RBF候选有schedule字段而失败；v1源码及失败记录保留，v2从原始
已提交事件重建时序，未重跑任何候选或改变验收容差。

另冻结单次明确的历史期限：初始reference=1100000us、保留150000us，到1250000us
新增双方都有的空观测tick。K4/16的两个方法/三夹具共24运行，其中12个过期事务在
提交事务内部仅留下原MAP身份前缀，实际删除此前保留的正确B历史推理支持；SQLite
归档保留只作审计。相同原始因子/状态模型/预算的12对保留历史控制组仍可推理该分支。
这不是输出年龄过滤、decision-window截断或初始K1剪枝，也不是一般滚动历史期限合同。
独立核对所有372事件、900输出9D状态、7173分支预测、423缓存状态及12次真实删除，
最大绝对差2.220446049250313e-16。控制组最终真实根损失全为0；过期组交叉0.4、
缺观测/重现2/3，保留反例，不增加预算清除失败。
完整包Task `8ecc1548e6e54a718a506fc3e813361e` completed，70个成员及全部3个实际
注册资产独立读回，receipt SHA
`90336cdacd6561cd0c1954f977ee84c611319e4e9a191efabde200957be5d628`。
产物根：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-fixed-deadline-identity-history-expiry-20261006T212750943638Z`。
该目录figures含PNG/PDF/SVG科学图、372个原始标量与独立绑定/PNG视觉检查回执；
图中范围来自因子扰动夹具，不是训练种子置信区间；图本身仅本地验收，未混入上述既有归档。

本轮21:26 UTC实际fleet仍无合格空闲GPU4/8，所有七个活worker绑定保留，native-v2
seed1337训练尚未派发。21:19 UTC实际console两个teacher及recovery-off3407有新事件，
五个任务尚无最终artifacts；recovery-off1337/2027旧尾部不能凭metadata heartbeat
算新事件。四个V2V输入/检查/guard父进程继续存活，2027/3407各等待自身完整资格。
完整C/F、学习式、正式Linux MHT、跨场景全对照、真实两数据集主表、隔离同资源成本
及论文收口均未完成。15实验族源码清单重新生成，但all_experiment_code_ready保持false。


### 2026-10-06T22:10 UTC：J作者权重、完整源码和原生val特征的实际进展

新公开资产根为 `/Volumes/Data/test/recover-before-fuse/artifacts/rbf-public-author-native-asset-acquisition-20261006T214405929601Z`。
CoopTrack固定作者提交29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399与HF固定发布91b46fd08c950572131c7c0f039b01bf543a4151，
实际取得veh/inf/coop三个tracking权重并匹配作者MD5、HF LFS完整size/SHA。
DMSTrack固定提交d3b9949499c8e68ea33060873bd1cb95b6d4d323，实际取得作者Drive发布的ego/1两个epoch3权重；
作者未公开其checksum，保留实际HTTP下载、本地独立再读及自身完整SHA的证据，不声称作者checksum比对。
五权重共1927197341字节；安全weights_only读取、全5744个checkpoint张量有限值及逐张量shape/dtype/SHA完成。
DMSTrack未改动作者ObservationCovarianceNet与get_config两个AST，作者fusion/multi_sensor配置下分别严格加载两端各34张量、
各111128190个state元素，缺失/额外key为空且加载后逐项精确相同；验证原生ego路径replace到1的两个实际独立文件。
这是权重/类资格，不是原生检测、tracking或指标复现。CoopTrack尚未加载其完整原生类。

独立新目录clone完整DMSTrack作者提交。全部9696个Git blob、640696256字节逐一与commit对象及SHA比较；
4个作者KITTI/nuScenes数据symlink只核对链接文字，未跟随访问目标。
最初审计器只允许普通文件的失败保留在source-readback-v1-failure.json；v2读取原生symlink blob，未修改作者源码。
作者公开no_fusion_keep_all.zip实际6324129277字节，完整SHA c71996a1f9c36dd3a78eed38664b6530ded82255cd4754a93379ea27eedb8196。
41903个ZIP条目中41895个普通文件、15095903855解压字节全部独立CRC/size/SHA并安全提取；
9序列、每CAV1993帧、两端60046作者检测行对应11958个必需feature/early_feature/4x4变换数组，全部shape、行数、有限值与变换可逆性通过。
该资格仍未证明原始official_test GT/ROI/类别/频率映射，也没有神经特征前向、tracking预测或独立指标。
作者train_no_fusion_keep_all.zip实际HEAD为19872259356字节，原下载PID14723继续运行；
独立PID19019绑定其准确源码/命令和完成字节回执后接续全成员与逐帧特征审计，不重建既有RBF缓存或前向。

CPU主机10.100.35.112已观察到原Python3.7.16。仅在新目录
`/home/lbin/Desktop/rbf-native-public-methods/DMSTrack-d3b9949-20261007` 创建独立作者CPU tracking依赖环境，
安装与原作者主要依赖版本对应，Torch1.12.0+cpu用于作者已发布特征；未修改既有conda/分析环境。
该准备不代表原生main_dkf入口已导入、原预测运行或完整V2V CUDA detector环境完成；
也不运输被拒绝的RBF priority/CUDA branch-state包。运行结果与pip完整lock仍待核验。

G的1337/3407七资产输入发布Task分别74cf59a567314e3ca58bbe8aebfe2118与7368470f2b34493cab8864039f492fd8，
各自复用已验完整train森林与既有seen-val缓存/原始schedule，无NN或缓存重建。
新transport-only发布器直接使用实际注册URL与配置fileserver 35端；原六个数学/输入资格函数AST保持。
发布、实际HTTP全字节读回、14379个cache成员独立size/SHA均需通过；当前不能以已创建Task或传输进度宣称输入/森林验收。
两个独立接续进程绑定各自正在运行的原publisher PID、Task ID、identity和源码，完整发布后自动核对七资产及全部成员。
没有新GPU inference Task，也未上传新的被拒绝运行包；seen-val消费者的旧固定34端路由仍须独立版本修正与资格绑定后才能派发。
22:10 UTC fleet仍七个活worker绑定、无合格空闲4/8卡；native-v2 seed1337训练保持关键路径优先。
完整A-K目标继续，完整C/F/J/G/H/I/K与论文收口未完成，历史坏rank5、严格失败和负质量结论均保留。


原生环境v1的step0/1完成、step2下载Open3D0.16.0的422.4MB wheel至406.9MB时
发生files.pythonhosted.org socket读取超时；完整远端failure、精确command与pip日志独立保存，
已有隔离环境/缓存、v1源码和失败均保留。v2仅将网络读取等待由默认15s绑定为120s、
关闭pip内部retry，固定依赖版本完全相同，不改变科学容差、work-cap或任何模型代码。
此修复使用原PyPI/PyTorch公共源，在同一新建的独立环境继续；没有重建/修改既有环境，
不涉及被拒绝的RBF执行包。v2仍待实际安装、pip check、lock与原生入口完整导入。


训练ZIP也实际完成19872259356字节完整下载与本地独立再读，SHA
73f473e5b9ef6c19b72ce21a8b8d1552697177518b77d206f515c796edc4a6a4。
接续父PID19019已真正启动train全成员/特征子审计PID24967，记录中的旧等待stage不能代替当前子进程证据。
两ZIP完整byte取得总26196388633字节，train成员审计尚未通过；不能把字节或val成员资格换成原生复现完成。


## 2026-10-07 原生 DMSTrack 完整输入与 CPU 接续资格

本次增加的是 J 的输入、映射和运行环境证据。整个 A–K 目标仍然 active；完整原生预测、独立指标、同资源比较与论文验收没有完成。

- 作者固定版本仍为 `d3b9949499c8e68ea33060873bd1cb95b6d4d323`。两个公开特征 ZIP 已完成完整字节 acquisition 和全成员 CRC、size、SHA、独立磁盘读回；不重复 detector 或 embedding forward。
- val：41,895 个普通文件、15,095,903,855 解压字节、11,958 个必需特征数组、9 序列、每 CAV 1,993 帧。train：149,339 个普通文件、45,836,011,078 解压字节、42,630 个必需特征数组、32 序列、每 CAV 7,105 帧；174,377 行原生检测与特征行数一致。训练特征审计已实际退出 0，没有用下载完成替代数组资格。
- 独立标量 float64 几何检查将全部 60,046 行原生 CSV 检测和 31,419 个原生 KITTI GT 对象与已验的作者 NPY 角点、score、ID、frame 和 row order 对齐。运行前冻结 8×float32 eps 的几何舍入界与 score 文本 0.500001e-6 界；最大几何差 1.268127441278466e-5 m，最大误差/固定界 0.3445150358875571；没有放宽容差。
- 进一步逐帧对齐已独立准入的官方 test 原始 YAML GT 投影：9 个作者 `0000`–`0008` 序列、1,993 帧、31,419 个对象，ID 集合完全一致；最大角点差 7.62939453125e-6 m，低于既有 GT 合同固定 1e-4 m。类别保持 merged vehicle，作者文件使用 `Car` 别名，排除原始 exact Pedestrian；复用既有 source-local ROI 与 final ego all-eight-corners XY ROI。作者 `val` 是本路径的官方 test 别名，仍保持标称 10 Hz、披露 train/test 重叠，不宣称统计独立。
- 新建独立 Python 3.7 CPU tracking 环境，Torch 1.12.0+cpu。V1 Open3D wheel 网络读取失败、V2 固定依赖安装成功但实际入口缺少 Shapely 的失败均保留。补齐作者 requirements 明确列出的 `shapely==2.0.0` 后，完整未改作者入口 `main_dkf.py --help` 实际退出 0，`pip check` 退出 0。完整环境 lock SHA：`f50729a5e8996dd7eda4c1780005e945a0a3b481d255ac6a2367ea5267fe7eb8`。这只资格化原生特征驱动的 CPU tracking 入口，不等价于完整 CUDA detector 环境。
- `run_public_baseline.py` 移除仅为 JSON/SHA utility 而触发的私有模型包导入，内联 utility AST 与原实现完全一致；改用作者 SETUP 指定的四个 PYTHONPATH 根目录。六个既有 adapter 函数在剔除这两类环境/导入变动后的 AST 完全不变；原生命令、checkpoint native ego-to-1 路径、同文件别名拒绝、源码/config/output guard 保持。现有 guard tests 51 passed、4 skipped；此数字不计入科学验收。
- CPU public Google Drive 连接超时已保留。接续使用已完整准入的 Mac public author 字节，经 SSH 传输，同步记录本地流 SHA 与独立 CPU 磁盘 SHA，不涉及曾拒绝的 RBF CUDA branch-state 或 Linux priority 包。只复用完整 Git 源码 clone，不覆盖作者 tracked config/checkpoint；NPY 解压在 clone 外，作者未跟踪的数据路径指向这些 exact bytes。
- 完整九序列原生 CPU inference 接续器 V2 已冻结并运行。它等待并核对完整 CPU input receipt 后才创建 native Task；随后在原生进程启动前检查全部 exact asset、原始源码、环境、协议，并独立重读全部 41,895 个 CPU feature 文件。执行作者固定 `num_epochs=0`、`use_multiple_nets`、`feature=fusion` 命令，force-GT-prediction 保持 false，不改作者算法。作者现有 diagnostic GT loss 与 evaluation checkpoint copies 保留；源码检查已单列，尚未执行完整预测的 empirical counterfactual GT invariance 检查。V1 等待器在没有 Task 或 prediction 时修正 ClearML entry_point metadata，旧源码、计划与 supersession receipt 均保留；输入传输和原始活回放/验收没有停止。

本批固定证据根目录：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-public-author-native-asset-acquisition-20261006T214405929601Z`；对应 source-freezes 同名。关键新增文件为 train `native-feature-input-receipt.json`、`native-detection-GT-independent-parity-v1/receipt.json`、`native-official-test-original-GT-mapping-v1/receipt.json`、`native-CPU-actual-entry-and-environment-lock-v3-readback.json` 与 `native-CPU-full-inference-continuation-v2-plan.json`。原生运行结束后仍须完整预测封存、cloud bytes 读回和独立指标重算，不能将过程退出 0 写成 J 完成。

同时保留 G seed1337/3407 七资产 publisher 与全成员 continuation。两个大 cache archive 正在 actual registered URL 独立 byte readback，尚无 complete seven-asset publication/member receipt。22:37 UTC 的物理 worker/task 交集查询仍无可准入空闲卡；两项 teacher 与 recovery-off3407 有新 console event，另两项 recovery-off 有 heartbeat、没有新的完整输出资产。C 的 native-v2 关键路径仍获得 idle GPU 第一优先级。主2027坏原 rank5 不第三次读取，各既有授权门槛保持。


### 2026-10-06T23:04:02.512674+00:00 — seen-val 原生注册 URL 执行路径

新增隔离冻结的传输修订版，按实际注册 URL 读取当前文件服务；未修改原森林算法、PATCHES/REPLACEMENTS、预算、checkpoint 或数值容差。其余八个执行器函数及模块 AST 保持一致。一次真实 completed Task 的 6864 字节注册产物完成 HTTP 读取与独立磁盘 SHA 校验，另有七项实际元数据/URL 拒绝检查；这仅是传输组件资格，不能代表大缓存、完整回放或论文验收。

执行器冻结：`/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-seen-val-bound-forest-native-registered-URL-GPU-v2-20261007/source-freeze.json`；调度冻结：`/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-seen-val-bound-forest-native-registered-URL-GPU-v2-20261007/dispatch-source-freeze.json`。调度器固定实际导入依赖，并要求七份云端资产和 14379 个缓存成员全部验收。原物理 GPU 交集、既有任务保留和核心任务优先检查继续有效；新增合格 native-v2 训练的首个空闲 GPU 优先门槛。只读实时检查显示 native-v2 seed1337 尚未调度。

两个特定接续进程 PID48729/48730 分别绑定 seed1337/3407 的现有完整成员审计进程 PID16902/16918。输入验收完成后才调用冻结调度器；容量等待间隔 300 秒，失败不自动重试，已创建任务原样保留。当前两者均在等待成员验收，尚未创建 GPU Task。此前拒绝的 CUDA branch-state 与 Linux priority 包未新增、未运输。

DMSTrack 已完成 train/val 全成员与检测对齐审计、全部9序列官方原始GT映射和原作者入口完整导入，CPU环境锁已冻结。两份权重在 CPU 机已完成独立读回，val特征ZIP仍在传输，原生预测尚未启动；GT反事实不变性、完整原生预测和独立指标仍未验收。不得将入口导入或组件资格写成公开方法完整复现。


### 2026-10-06T23:19:40.624341+00:00 — 两种子完整seen-val输入与公开SPD原生源码

SPD seen-val seed1337/3407分别完成实际7份注册资产与14379个完整缓存成员的独立读回，均覆盖21序列/7189源帧/3316事件。原上游completed主森林46序列/7445事件、checkpoint、Task脚本、参数及资产元数据和原冻结worker准入函数均实际核对通过。产物根为 `/Volumes/Data/test/recover-before-fuse/artifacts/rbf-seen-val-bound-forest-native-registered-URL-qualification-v2-20261007`；两个特定接续进程仍等待native-v2训练优先，不称为完整seen-val预测或G验收。

V2V native-v2调度另冻结源码导入路径及Task.create自动仓库捕获关闭；其余submit AST、原consumer、27模块训练包、模型/损失/预算、去重和物理交集准入保持不变。旧源码与未运行的首个绑定草案保留。实际新CLI完成输入与冻结依赖核查，当前仍返回waiting_for_disjoint_idle_physical_GPU4_or_GPU8，无新GPU Task。

完整作者CoopTrack/SparseCoop源码分别127/132个Git blob、2596767/4689101字节已逐一核对，固定提交与原注册版本一致，clone均为新隔离目录。另对完整原生数据转换/配置/标签修订源作静态绑定：46train/21val/28test；CoopTrack原生10Hz与SparseCoop作者2Hz命令单列，实际数据转换尚未运行。两个gen_example_data全部函数AST相同；车辆29/路侧146标签修订记录会被读取，而6963份cooperative记录的执行循环在源码中禁用。实际标签修订/去重后的数据必须另验，不能仅依据README“cleaned”文字宣称数据资格。

SparseCoop官方releases当前0条，3个issues及全部已有评论已读回；所查作者源未给出可绑定的训练checkpoint。官方README初始ResNet50下载完成102502400字节，SHA19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097，匹配官方文件名前缀并独立磁盘重读。PyTorch2.6拒绝旧tar weights_only读取的失败保留；v2按原生容器格式用受限Unpickler、CPU有界storage/stride读出完整267有限张量，无unsafe weights_only=False。该结果仅初始化backbone组件；训练后的SparseCoop模型、完整原生类加载、CUDA算子、2Hz数据转换、预测和独立指标仍缺。

CPU主机已有py38当前Torch2.4.1/torchvision0.19.1，缺mmcv/mmdet/mmdet3d/nuscenes等作者依赖，未修改该环境。新公开方法源码/组件根：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-CoopTrack-SparseCoop-complete-native-source-20261006T230625589928Z`。DMSTrack原有输入传输与预测接续保持，完整J/论文仍false。

### 2026-10-07 08:03 CST：公开原生模型严格加载与两种 val 标签合同实证

- 固定 mmdetection3d v0.17.1 提交 `f1107977dfd26155fc1f83779ee6535d2468f449` 的 775 个源码文件完整读回。主机默认 CUDA 12.8 与作者旧依赖不匹配；实际使用既有只读镜像 `sha256:85525aefed5d9a5d5f6d72d7853f1a9ba3744206c889c767a8dfac8202c1d2ab`，Python 3.8.20、Torch 1.9.1+cu111、mmcv-full 1.4.0、mmdet 2.14.0、mmsegmentation 0.14.1、mmdet3d 0.17.1、NumPy 1.19.5。原生模型代码 127 文件一致；11 个 mmdet3d 扩展和 mmcv 扩展成功导入并绑定二进制 SHA。镜像的缺失源码挂载、C++ ABI 导入失败与修正记录均保留；使用镜像已有库的 LD_LIBRARY_PATH，未修改共享环境或原作者代码。torchaudio 缺项和 GPU 前向资格仍未解决。
- 三个作者权重在原先安全 `weights_only=True` 读回基础上，导出不含 pickle 的值等价状态包。原始 Task `d8554b6b9fd347e1900cd66a68b63153` 完成四项输入发布，共 539865354 字节；CPU 主机从实际注册 URL 完整下载、独立重读，再向 vehicle/infrastructure/cooperative 三个完整原生类执行 strict load。全部 817/817/914，共 2548 个参数及缓冲张量逐项值、形状、dtype 和 SHA 精确一致。证据：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-CoopTrack-exact-mmdetection3d-native-source-20261006T232725831265Z/safe-author-weight-state-only/CPU-three-native-models-strict-load-and-all-tensor-readback.json`，SHA `ff13f996f818238fc558df9c504d056b804c1e5dbfe8871684ccc921848392c3`。两次 SDK 令牌接口错误及其原目录保留；修正仅涉及实际 Session.token 访问器与新隔离目录，不涉及科学计算或失败任务重试。
- 官方完整 SPD Drive 清单及 HEAD 已读回，但 train/val GET 返回 quota exceeded 页面，未将 HTML 当 ZIP。复用此前已保存的 `V2X-Seq-SPD.zip`，234358463 字节、SHA `10bbf7130114e693cda8c99c6f3e0f88949b2224be9725a268cd5ee3825c97c1`；全部 140659 个文件、774631042 字节 payload、57815 个 GT 标签完整 CRC/SHA/JSON 有限性读回。原始 release identity 的 `unverified-source-bytes` 仍披露，不能据相同字节数声称已与当前官方 Drive 文件逐字节比较。另取得 test 元数据 ZIP 并完整读回 34821 文件，明确不含 GT；未用于训练或选择。
- 新隔离相机 val 源已绑定全部 7189 个原 JPEG、原始校准、原始 GT 和作者 split，21 序列、3316 cooperative 配对。使用未改动的原作者 gen_example_data.py，分别完成 CoopTrack 10Hz 与 SparseCoop label-cleaned 2Hz 的 val 标签处理。独立窗口选择、逐行 metadata、token 删除/类型或 track_id 修订及首次重复 ID 保留规则严格通过：10Hz 为 3316 时刻/9948 标签文件；2Hz 为 675 时刻/2025 标签文件。2Hz 单列，不与统一输入机制表混用。10Hz 独立检查器的空 operation 列表遗漏失败保留并按作者既有 falsy operation 语义修正；未重跑或放宽原生成结果。准备器使用隔离 CPython3.14/tqdm4.66.5，仅做 JSON 整理；预测模型环境未改动。
- V2V seed1337 冻结绑定的首空闲 GPU 接续程序存活，按全项目物理交集再次检查后仍等待 disjoint GPU4/GPU8；不等待另外两个种子的输入资格，不重试科学失败 Task。G 两种子的既有接续及 DMSTrack 原输入传输/原生推理接续保留。五个原 teacher/recovery-off Task 在 23:50 UTC 仍运行、无完整产物；控制台的序列局部进度不能当全任务完成。
- 该轮新增证据属于输入与模型部件资格。仍缺原生 nuScenes annotations/temporal infos、GPU 前向及全部原生预测/独立指标、SparseCoop 实际训练权重、完整 C/G/H/I/J/K 与论文收口；不声称正式资源测量或整项 J 已验。组件索引：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-CoopTrack-exact-mmdetection3d-native-source-20261006T232725831265Z/original-native-runtime-weight-and-val-label-component-index.json`。


### 2026-10-07 08:45 CST：原生 SPD metadata 独立读回与 DMSTrack CPU 实际前向

- CoopTrack 与 SparseCoop 的完整作者源码、原生 val 标签与全部引用标定组成 metadata 专用包；40,179 个成员在 Mac 和 CPU 主机分别完整读回 SHA。未混入被拒绝的 RBF CUDA branch-state 或 Linux priority 包，未改作者算法。
- 在已核验的原生 Py3.8 / Torch1.9 Docker 环境完成 15 个作者 metadata 进程，分别保留 CoopTrack `--forecasting` 和 SparseCoop 无 forecasting、额外速度坐标更新的原脚本行为。六个 side/profile 均覆盖 21 个 val 序列：CoopTrack 每侧 3,316 帧，SparseCoop 每侧 675 帧。原始输入与作者源码转换前后逐文件 SHA 不变。
- 安全读取器与独立 oracle 完成逐帧删帧/时间链、原 GT 框/类别/ID、相机标定、nuScenes 引用及生成的全局中心/尺寸检查：221,635 个原 GT 框精确一致，255,832 个生成 annotation 的全局中心最大误差约 `9.095e-13`。固定 `1e-8` 容差未放宽。旧 pickle 编码解析失败和错误套用 CoopTrack 框约定到 SparseCoop 的 oracle 失败均保留；按两份实际作者源码分别核验 `w,l,h,-yaw-pi/2` 与 `l,w,h,yaw` 后通过。**34,197 个新增插值 GT、速度和未来字段尚未完整独立重算，预测端 GT counterfactual 未验收。**
- SparseCoop 按原作者清洗及 2Hz 选择规则准备完整 46 train / 21 val 的 metadata：每侧 2,196 帧、6,588 个标签文件独立核验通过；未使用 28 test 序列。存在 4,563 个缺失训练图片引用，原生训练 anchors、完整模型环境和实际训练权重仍未合格。不得用 val 替代 train 生成 anchors。
- DMSTrack 原始两份权重和完整验证 ZIP 已传至 CPU 主机；全部 41,895 个特征成员和 9,696 个作者源文件完整读回。原 Task `fe359b9a73934948aa6142786cfb075c` 在前向前因原 checkpoint `cuda:0` 标签不能由 CPU 入口直接加载而失败，Task、日志及原文件均保留。
- 新副本仅将 68 个 storage location 标签从 `cuda:0` 改为 `cpu`；全部 tensor payload 成员未变、68 个张量完整逐值 SHA 与原 safe-weights-only 证据一致，原生 Torch1.12 CPU 不加 map_location 的读取通过。新绑定/输出和执行标识冻结后，Task `02eec562fc3b492ca8b336a07d18443e` 已进入作者原算法 CPU 前向；08:43 核验到第 0004 序列 frame index 158。9 序列完整预测、输出归档、独立指标和 GT counterfactual 仍待收口。
- 保留 C seed1337 的第一组空闲物理 GPU 优先级和 G 两种子的已有等待进程。V2V seed2027 原行生产进程仍实际计算，29/32 个序列文件已写出，未冒充整体验收。五个原 GPU 实验仍 in_progress、尚无最终 artifacts。保留主森林 seed2027 坏 rank5、MHT 与被拒绝包的既有门槛；无提交或推送。J/G/完整论文验收仍 false。
- 本轮不可变进度 checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-native-two-author-metadata-and-DMSTrack-actual-forward-progress-20261007T004443610213Z.json`；前一轮 222 个引用全部重核验，新增 103 个引用，共 325 个。`/goal` 保持 active。


### 2026-10-07 09:16 CST：完整原生 GT 运动字段通过，DMSTrack 完整前向后保留指标差异

CoopTrack/SparseCoop 六个 val profile 的原始 calibration、全部 34,197 个作者插值 GT、255,832 个原生 forward-velocity annotation 已独立重算。SparseCoop 另有全部 40,943 个 nuScenes centered-velocity annotation；CoopTrack 全部 214,889 个对象的 13 步未来 GT 合计 2,793,557 步。统一原容差 `1e-8` 未放宽，最大原生 forward-velocity 差约 `3.19e-11`。原 vehicle pose 使用作者 quaternion trace 归一化语义，infrastructure 用独立 SVD polar oracle；先前误将 vehicle raw rotation 投影到 SVD 的失败保留。这些未来字段是 GT metadata，不构成预测的因果输入资格，也没有运行 CoopTrack/SparseCoop 原生预测。

完整 oracle 在 `/Volumes/Data/test/recover-before-fuse/artifacts/rbf-CoopTrack-exact-mmdetection3d-native-source-20261006T232725831265Z/original-native-val-metadata-conversion-inputs/all-author-native-interpolation-velocity-and-forecast-GT-independent-readback-v3.json`，新增组件索引为同目录根下 `original-native-runtime-weight-and-two-author-metadata-component-index-v3.json`。SparseCoop 缺实际 trained weights、完整原生环境和 4,563 个训练相机引用等边界继续保留。

DMSTrack 原 Task `02eec562fc3b492ca8b336a07d18443e` 已完成完整 9 序列／1,993 帧，原生进程及独立 controller 均退出 0，保留 CPU device-tag-only 权重兼容包及全部 68 个原始 state tensor 精确同值证据。作者日志报告 sAMOTA=0.9150411854995134、AMOTA=0.4351586619561412、AMOTP=0.579422843594428。完整预测为 42,769 行，GT 为 31,419 行；完整 1,993 帧均由固定 seqmap 明确保留，本次两角色实际空帧均为 0。独立导出的每一原生 metric-consumed 字段精确保留，原始 GT/source/weights 没有修改。

原生输出树共有 5,990 个 regular 文件、912,139,777 字节 payload，CPU archive 831,526,657 字节，SHA256 `2b5e6a8f88a6df2383703eacb0c6d59464c0f27bdbdc447f7d8c406d8f577509`；CPU 已将全部 archive member 与运行后输出树精确比对，Mac 运输、完整 member 复查和注册 cloud byte 读回仍在接续。不能以退出 0、Task completed 或 CPU archive 存在替代 J 接受。

现代 Python3.12/NumPy2.2.6 的独立完整评价得 sAMOTA=0.9176414341429158、AMOTA=0.43726725866513894、AMOTP=0.5793663281856124，与作者精确值有差异，故保留 failed-agreement receipt，不接受或排名这些指标。新的独立原环境读回使用原 CPU Python3.7/Torch1.12/NumPy1.21.6、完整原始 KITTI 文本字节、原 JIT 函数与固定 3D IoU≥0.25；没有修改数值函数、门限或容差。原生 AMOTP 是平均 3D IoU、越高越好。该进程正在运行，运行环境是否解释差异尚未结论化。

完整 GT 诊断反事实保持原模型、检测、原生 feature 和全部 9 序列，在独立目录置换 GT 身份并平移中心。大位移 x+37/z-19 触发作者原诊断 regression loss 的空 TensorList 错误，历史失败保留；没有删改该 loss 或扩 gate。当前新的固定 x+0.125/z-0.125 与一致 ID 双射用例已重新独立验证所有原 feature 字节，原进程正在完整前向。它是诊断用例，不能当作 baseline、training selection 或一般性的 GT 不依赖证明。早期 archive-member root 和 PYTHONPATH/importlib.metadata 的启动错误另留各版 failure/process receipts。

C 的原 CPU 控制器自然完成 seed2027 行验收；修正 native-v2 监督完整覆盖 789,401 行／32 train 序列、80,831 条跨端正边和 349,009 条时序正边，原全部 15 个 prediction/context 字段逐字节不变。独立实际 loss-mask 覆盖计数：fit 的 cross/temporal informative samples 分别 35,593/91,057；holdout 分别 4,207/17,949。原 partition、27 模块训练 consumer 和 scientific body 均未变。完整 GPU 输入本地 package/member 读回已通过，seed2027 输入发布 Task `5f8286da9de648e3a2441d0933288362` 正在接续云端字节验收，尚未创建该种子 GPU 训练 Task。

原 seed3407 全行 producer、其三个独立 CPU 依赖控制器、seed1337 冻结 first-idle 训练 watcher、G 两 seed 守卫和五个原 GPU replay/teacher Task 均保留。01:03 UTC 的五任务实际 metadata/console 仍全部 in_progress，均无注册产物；heartbeat 与局部序列结束不构成最终验收。首个空闲的物理 4/8 卡交集仍优先交给原已合格 native-v2 seed1337。主森林2027损坏 rank5 没有第三次读取，CUDA branch-state/Linux priority/MHT 原授权门槛未绕过。完整 C/G/I/J/K、论文与同资源测量继续未完成；无自动 commit/push。


后续同一原进程读回（2026-10-07 09:20 CST）：原环境独立评价已退出 0，完整精度 sAMOTA/AMOTA/AMOTP 与作者原日志完全相同，最大绝对差 0；全部 37 个 recall threshold header 与整行指标／计数文本逐项一致，固定验收容差 `1e-12` 未放宽。接受的是独立重放固定作者 engine 的指标读回，不宣称第二套指标实现。现代环境 failed-agreement 继续保留，尚没有以猜测指定其具体差异原因。独立接受文件为 DMSTrack 资产目录 `native-complete-prediction-exact-original-runtime-metrics-v3/all-original-native-metrics-independent-exact-agreement-acceptance.json`。完整 5,990 个原生输出 member 已在 CPU 与 Mac archive 两端逐字节读回；cloud byte 与 GT 反事实仍未完成，所以整体原生方法、J 与论文接受继续为 false。


### 2026-10-07 09:56 CST：完整 DMSTrack GT 扰动收口、C seed2027 云输入与 SparseCoop 67 序列原生元数据

完整 A–K 目标保持不变。DMSTrack 原始 CPU 推理 Task `02eec562fc3b492ca8b336a07d18443e` 已完成；固定 GT ID 双射及 x/z ±0.125 m 扰动后的完整原生推理和评价均退出 0，全部 9 序列、1993 帧的 3995 个预测、track 和 affinity 文件与原输出逐字节一致。诊断损失确有变化；这是一项固定干预的数值证据，不外推为任意 GT 干预下的一般不变性，也不把干预后的评价当作性能基线。完整原生指标在作者原 CPU 环境重新计算，与原日志的 37 个 recall 阈值及全部统计行完全相同；这是作者评价实现的独立回放，未宣称第二套独立指标实现。CPU 兼容权重仅改变 CUDA storage device 标签，保留原张量值；GPU 原论文运行环境与同资源成本仍未资格。

完整输出包已有 CPU/Mac 全字节验收，但此前上传登记在原 Task 已完成状态下遭 `tasks.add_or_update_artifacts` 400/110 拒绝，publisher 退出 1，两个完整输出 artifact key 均未登记。已只读确认原 Task 仍为 completed 且只有原三个元数据 artifacts；物理文件上传不构成 artifact 登记或云字节验收。保留失败，没有重开原 Task、重跑原推理或追加上传；遵守论文合同“原 GT 全文、模型和完整预测流不自动上传 ClearML”的边界。

C seed2027 原生 v2 输入 Task `5f8286da9de648e3a2441d0933288362` 的四个登记角色和完整 64 个 archive 成员均已独立读回，输入语义身份 `bcee857b8592b37c12801824765c176b0a690e3e90cb025781a510dfba6b4d0a`。原 27 模块训练 consumer 和 SHA 保持不变。新增冻结元数据接续仅在原 seed1337 首次 Task 准入之后才重新查询完整物理 GPU 交集并尝试 seed2027；当前仍在等待 seed1337，未创建 seed2027 训练 Task，模型/checkpoint 未资格。

SparseCoop 的 46 train +21 val 序列已执行作者原 `spd_to_uniad`、`spd_to_nuscenes` 和 `update_velocity_coordinate`，三侧共九个程序均退出 0。原生输出须保留作者冻结排除规则：车辆 train 1434、基础设施 train 1521、合作侧 train 1495，三侧 val 各 675；原输入各 2196 行不能冒充原生输出计数。首次验收误把未过滤输入数作为输出数，停止后保留；其后的基础设施侧因打包误用 label/lidar、漏入 label/virtuallidar 原标签而退出，亦保留。补充的 2196 个原标签逐字节验收，输入 metadata/标签内容和作者科学源码没有改动，车辆侧三个已完成程序直接复用。

新的专属 CPU 内存目录采用冻结 2 GiB 数据上限、4 核、8 GiB 容器内存及 8 GiB 磁盘预留；保留完整与失败产物，不作为正式同资源测量。独立语义读回覆盖 6475 个原生三侧帧记录、112712 个原 cleaned GT 对象、9121 个插值对象及 121833 个对象的更新前与更新后速度向量（各 121833 条）。帧/实例/JSON 外键、train/val 划分、原 GT 尺寸及全局中心、原 calibration、UUID、prev/next 与 centered-global-to-lidar 速度均通过；最大前速度误差 1.2361667245386343e-11，后速度 9.349188090368443e-12，固定容差 1e-8。完整输出/输入 archive 122061754 bytes、SHA `6b2c8519d4b2ee48c9785dbf5abb0fefa98b8d0d77c54b6f40635cce7c654f70`，89475 个成员共 432823237 payload bytes 已在本机全部独立读回。原 132 个作者源文件、89424 个输入成员和 51 个新生成成员在审计前后保持一致。

训练图像仍缺 4563 个引用；完整 SparseCoop 模型运行环境、实际 trained checkpoint、原 train-only anchors 和原生预测仍未资格。原 SPD release 身份仍为 unverified-source-bytes，GT 速度属于作者训练元数据，不替代因果检测速度。Full C/J/G/H/I/K 和论文收口仍未完成，保留所有其他活进程、历史失败和授权门槛，不提交或推送。新的不可变组件索引为 `/Volumes/Data/test/recover-before-fuse/artifacts/rbf-CoopTrack-exact-mmdetection3d-native-source-20261006T232725831265Z/original-native-runtime-weight-and-two-author-metadata-component-index-v4.json`；完整语义证据为 `/Volumes/Data/test/recover-before-fuse/artifacts/rbf-CoopTrack-exact-mmdetection3d-native-source-20261006T232725831265Z/original-native-SparseCoop-67-train-val-metadata-conversion-inputs/all-native-67-original-GT-independent-semantic-audit-v3/all-original-SparseCoop-67-native-full-train-val-GT-geometry-interpolation-and-velocity-independent-acceptance.json`。


### 2026-10-07 10:28 CST：单机 GPU 放置改为 35.112，实际 CUDA 兼容性通过

用户直接要求“需要单机跑GPU的在35.112上进行”，后续新增单机 GPU 工作按此安排执行。
严格 SSH 的实际读回表明 10.100.35.112 为一张 NVIDIA GeForce RTX 3060、12288 MiB，
物理 UUID `GPU-68c03b8c-acc3-421c-0e7a-8da0bb9ad3fa`，driver 595.84；查询时
没有 compute process、利用率 0%。此前 L40S 标签不能替代当前硬件事实，CPU-only 放置
规则已被新指令覆盖。既有活任务与原 4/8 卡训练 consumer/budget 保留。

既有只读镜像 `sha256:85525aefed5d9a5d5f6d72d7853f1a9ba3744206c889c767a8dfac8202c1d2ab`
在这张 GPU 上实际执行成功：Py3.8、Torch1.9.1+cu111、mmcv-full1.4.0、torchvision0.10.1+cu111。
Torch float64 CUDA backward 与解析梯度逐值精确一致；实际 MMCV CUDA rotated-box IoU
为 `[1, 0.3333333432674408, 0]`，torchvision CUDA NMS 保留索引 `[0, 2]`。
该结果确认旧原生环境的这些真实 CUDA 入口可运行，未接受完整模型前向或论文成本。
证据为 `original-public-native-35112-cached-image-GPU-qualification-20261007T022416410579Z/actual-CUDA-compatibility-readback.json`，
相对根目录 `/Volumes/Data/test/recover-before-fuse/artifacts/rbf-CoopTrack-exact-mmdetection3d-native-source-20261006T232725831265Z`。

SparseCoop 作者 Py3.8/Torch1.13.0+cu116/torchvision0.14.0+cu116/mmcv-full1.7.1 的
八项官方原生依赖已在 Mac 完成全字节 SHA 与全部 ZIP/deb payload 成员 CRC/SHA 读回。
其中 Torch wheel 为 1983011483 bytes，10736 成员、3690946742 expanded payload bytes；
独立 CUDA11.6 NVCC/cudart/driver-dev/CCCL 解包进入专属目录，不更改主机 CUDA12.8 或共享镜像。
首次 CPU 直取官方 torchvision 返回 HTTP403，原失败及已取得文件保留；新隔离 v2 仅通过
原 strict SSH/SCP 传输已在 Mac 合格的同一公开依赖字节，版本、原作者 132 文件和科学代码不变。
原生环境使用冻结 8 GiB 专属 RAM payload 上限、2 CPU/4 GiB 容器及 6 GiB 主机 RAM、
8 GiB 磁盘预留；不通过扩容失败 cap 或假 import 追求资格。

新的预执行冻结 GPU 接续 PID 71966 正在等待原 v2 环境完成真实 CUDA11.6 扩展编译与完整
plugin import；之后只在 35.112 空闲物理 GPU 上执行一次原作者 deformable aggregation 前向
与三类梯度核验。独立 CPU bilinear/grid_sample autograd 使用冻结 dyadic 输入，完整值按
`1e-8` 比较，原源码与扩展二进制在 GPU 执行前后核对 SHA。当前依赖未就绪，不能将
controller 存活或 GPU 空闲当作 SparseCoop 算子已通过，更不能当作完整原生预测通过。

本轮 checkpoint `/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-user-35112-GPU-and-original-native-runtime-progress-20261007T022750356203Z.json`
完整重核验前 490 引用并新增 38 个，共 528 个不可变引用；完整 A–K goal 保持 active。
主森林2027原坏 rank5、CUDA branch-state/Linux priority 包与正式 MHT 的既有门槛仍保留，
未借公开作者依赖的 GPU 执行绕过这些包的门槛；完整 C/J 与论文接受均为 false，无 commit/push。


### 2026-10-07 11:13 CST：35.112 原生全帧前向及 CUDA 数值验收

- 按用户最新要求，单机 GPU 工作使用 `10.100.35.112` 实际 RTX 3060 12 GiB；原多卡科学合同和既有活任务保持原安排。
- CoopTrack 原生 vehicle val Task `599375d38cde4c728567771e6a94cfdd` 已完成 21 序列 / 3316 帧全量预测及作者原生检测、跟踪评价，进程退出 0。3748 张输入图片、14 项元数据、实际 vehicle 权重均有完整字节核验，127 个作者源码文件未改动。预测 pickle 为 85,489,150 bytes，SHA256 `6a3790b8f00004f5ff3b1e03eff990759adf1f2cfa987445ba8a303b8aef17cf`。
- 纠正本次准备阶段对 `test_tracking_api.py` 缺失的错误判断：实机文件及原始入口完整存在。只取消尚未创建 ClearML Task 的新准入控制器，保留原 SCP；修正前置判断后使用作者原生入口，未修改科学实现。相关失败、取消和更正回执全部保留。
- 作者输出三类平均 AMOTA `0.10901749872064326`，car AMOTA `0.2341057840199639`。独立按全部原始阈值与 precision 数组重算 25 项汇总，最大误差 `2.220446049250313e-16`，容差 `1e-12`；此处只验汇总算术，尚未独立重算逐预测与 GT 匹配。pedestrian 无有效结果的原生 NaN/最坏值保持原样。11 个完整输出文件正在由 SCP PID `95368` 读回 Mac，并按远端登记 size/SHA 全量核验；不上传全预测至 ClearML。
- SparseCoop 原生环境首次 CUDA 编译因缺少 `cusparse.h` 失败并保留。补齐 NVIDIA CUDA 11.6 Update 2 三个官方开发包的原始头文件后，原作者扩展编译及完整插件 CPU 导入通过；原 132 个源码文件和既有安装环境所有文件 SHA 未变。旧、新 RAM 合计上限仍为 8 GiB，没有扩大失败 cap。
- 随后在 35.112 执行原生 CUDA 前向及 feature/location/weight 三组梯度检查，416 个标量经 Mac 再读回，均与独立 CPU bilinear/autograd 参考精确一致，最大误差 0，冻结容差仍为 `1e-8`。这只验算子和环境；SparseCoop 实际训练权重、缺失训练图片、anchors 与完整原生模型预测仍待完成。
- CoopTrack INF query 导出、cooperative 全量运行和独立逐预测评价仍待接续；614.72 秒原生前向耗时不作为正式同资源测量。全部 A–K 目标仍 active，不据本次进程完成或组件通过宣称 J 或论文完成；保留原 CUDA/Linux priority/MHT 门槛及 main2027 损坏 rank5，不第三次读取坏资产。不提交或推送脏工作树。


### 2026-10-07 11:19 CST：CoopTrack 全预测与原生检测 GT 匹配独立验收

- 原 SCP 完成，11 个原始输出文件合计 420,943,558 bytes 全部按远端登记 size/SHA 在 Mac 重读一致；未重新运行模型、未上传预测至 ClearML。
- 独立从原始 pickle 张量重建中心、尺寸、旋转、速度及全局坐标、距离过滤、类别/分数/track ID，覆盖全部 3316 帧、20,228 个原始跟踪框与 994,800 个原始检测框；作者格式化输出分别为 16,446 和 754,071 框。123 个空跟踪帧保留。位置/尺寸/旋转/速度最大绝对误差分别为 `4.547473508864641e-13 / 0 / 2.220446049250313e-16 / 8.881784197001252e-15`，均满足原冻结 `1e-8`。
- 第一版独立重算因未复现 `Box.rotate` 调用 `Quaternion.rotation_matrix` 时原地归一化变换四元数的语义，在 000871 帧旋转项产生 `3.6057680508072565e-08` 差异。保留失败，读取实际库源码确认原因；只修正独立验算器归一化，未改模型、原始输出或数值容差。v2 完整重新验算通过。
- 进一步只使用原生 JSON GT 表、原始格式化预测和独立 NumPy 匹配实现，从 50,731 条 GT 重建类别、LIDAR_TOP ego pose、严格距离及非零点数过滤；检测过滤后 car/bicycle/pedestrian 为 14,593/4,999/1,147。完整核验三类 × 四距离阈值的置信度总序、贪心最近邻一对一匹配、每条 101 点 recall/precision/confidence 曲线和 AP，全部最大误差 0，冻结容差 `1e-12`。每种匹配的决策 SHA 保留。
- 跟踪 GT 在作者评价中还经过轨迹处理，检测 GT 数量不能替代跟踪分母；独立 tracking assignment/AMOTA 和检测 TP error arrays 尚未通过，明确留待后续。全 J、INF/cooperative、完整 A–K 与论文尚未收口。
- 五个原 teacher/recovery-off Task 在 11:16 CST API 仍 in_progress，尚无注册输出；teacher 报告继续推进。原 C/G 控制器存活；seed3407 身份行生产子进程 PID13427 本轮实测 CPU 100%，保留原进程，没有因耗时重启。


### 2026-10-07 11:26 CST：检测 TP 误差与原生跟踪输入独立验收

- 上一轮为 progress；本轮沿同一 CoopTrack Task 的原始预测推进，没有重跑 GPU 模型。独立从全部原生 GT 重建逐匹配 translation/velocity/scale/orientation/attribute 误差；原生中心差分/单边差分速度、无 attribute 时的 NaN 与累计均值处理均保持原语义。
- 完整 12 组匹配下的五类 101 点 TP error arrays 共 6,060 标量及 15 项类别 TP 汇总全部通过，最大误差 `2.220446049250313e-16`，冻结容差仍 `1e-12`。结合先前全量 AP/PR/置信度曲线与汇总校验，原生 detection 数值链路已具备独立证据；不以此替代 tracking 验收。
- 从原生 JSON 表和预测独立重建 tracking 类别/距离/点数过滤、全轨迹平均分数及全部补帧插值，再与作者原函数实际准备的输入逐字段比较。GT 从 20,739 框补入 92 框到 20,831 框（car/bicycle/pedestrian 为 14,667/5,017/1,147）；预测从 16,446 框补入 18 框到 16,464 框（12,985/2,940/539）。全部 37,295 框、596,720 个数值标量通过；旋转最大误差 `2.220446049250313e-16`，其他字段最大误差 0，冻结容差 `1e-8`。
- 发现并保留作者评价的插值方向：右端权重为 `(right_timestamp - timestamp) / (right_timestamp - left_timestamp)`，随后用于右端盒，而非通常按时间向右递增的权重。92 个 GT 和 18 个预测补帧均非精确中点；相较常规时间权重，最大单坐标位置差分别约 19.57536 m 和 3.68725 m。原实现未改，具体序列/轨迹/时间戳例子已记录。该行为属于评价后处理，不宣称模型使用未来输入，也不能静默改成更好看的评价结果。
- 独立 checker 初版错误地把预测数值 track ID 转为字符串，失败保留，v2 原样保留 JSON identity 类型后全量通过。实际 worker 退出 0；接续控制器因进度条空格及末尾空行不能按“最后一行”解析 JSON，解析失败保留；对同一完整日志按唯一结果 kind 读回并绑定 SHA 恢复验收回执，没有重新执行科学检查。
- 尚待独立重算跟踪阈值、逐帧分配与 AMOTA/其他 tracking 指标；CoopTrack INF/cooperative 原生运行和全 A–K 目标继续。现有 GPU/CPU 任务、脏工作树、失败及既有执行授权门槛均保留，不自动提交或推送。


### 2026-10-07 11:32 CST：CoopTrack vehicle 原生全量数值链路收口

- 上一轮为 progress；本轮基于已独立核验的 GT/预测轨迹，另写匹配状态机与 SciPy 最小费用分配，对每个真实帧执行独立配对，并逐条与原生 MOTAccumulator 事件比较。独立距离矩阵与原生 sklearn 距离最大差异 0；未调用作者 TrackingEvaluation 的匹配或指标计算作为独立结果。
- 全部三类、无阈值起始回放和所有可达阈值共核验 70,429 次帧级分配、336,862 条 MATCH/SWITCH/MISS/FP 事件。car 无阈值 MATCH 分数 6,864 个、可达阈值 16；bicycle 1,384 个、可达阈值 8；pedestrian 68 个、无达到最低 0.1 recall 的阈值，保留原生最坏值和 NaN，不改阈值。
- 全部 2,040 个阈值/指标曲线标量通过，最大误差 `4.218847493575595e-15`，冻结容差 `1e-8`。独立重算包括 gt/tp/fp/fn/ids/frag、recall、MOTA/MOTAR/MOTP、MT/ML、FAF、TID/LGD；同时保留原生“阈值分数只计 MATCH”“MOTAR recall 只用 num_matches”口径。
- 从独立曲线按原生最高 MOTA、并列优先高召回阈值选取传统指标，并重算 AMOTA/AMOTP、类别汇总与 dataset.evaluate 输出，共 85 项检查通过，最大误差 `1.887379141862766e-15`，冻结容差 `1e-12`。全部 11 个原始输出再次完整 size/SHA 核对一致。
- 独立结果：三类平均 AMOTA `0.10901749872064326`；AMOTP `1.778267568629886` **米，越低越好**；FP 3224、FN 14158、IDS 51、Frag 241。原生 GT 字段为类别平均 `6943.666666666667`，不能当作 GT 总量；pedestrian 未达到最低召回仍按原生规则计入结果。
- 原生 TID/LGD 固定按 0.5 秒/帧换算，连同上一轮确认的反向时间权重插值一并披露；没有更改原生协议，也没有把这一时间口径当作 SPD 实际采样间隔。原生复现与修正/统一输入评价继续分开。
- Task `599375d38cde4c728567771e6a94cfdd` 的 vehicle 21 序列/3316 帧从原始输出到检测和跟踪指标的全量数值链路已独立验收，后续不重跑该合格模型运行。作者权重 release 身份未因数值一致而升级；INF query、cooperative 完整原生预测与正式同资源比较仍未完成，不能宣称完整 J 或论文验收。单机 GPU 继续使用 35.112；既有多卡任务、CPU 验收和所有明确授权门槛保持。


### 2026-10-07 11:45 CST：INF 全量图片准备与原生 query 管道字节资格

- 上一轮为 progress，vehicle 合格全量运行没有重跑。35.112 当前 GPU 空闲；新鲜观测 root 盘余约 8.88 GB，`/dev/shm` 余 6.36 GB，MemAvailable 约 16.2 GB，因此完整 INF query 不适合在远端累计保存。
- INF val 原始 3441 张图片在 Mac 全量重读并逐成员验收，payload 1,119,708,242 bytes；归档 1,125,867,520 bytes，SHA256 `aa1b044c194b883a4d08ea382730f8756d970981a57a08d479846db3e10acc2b`。原 SCP/CPU 准入控制器 PID11054 正在运输，完成后逐图核验、保留原归档与图片；owned 输入 RAM 上限在运行前冻结为 3 GiB，host RAM/磁盘保留 6/8 GiB。
- 使用原生 INF 配置和实际 INF 权重，817 个 state tensor 严格加载且逐位一致；原模型 `_generate_empty_tracks` 的 900 query、hist_len=4、fut_len=8 布局用于传输资格，不冒充真实图像前向。
- 空 query 与完整 900-query 经原始 `mmcv.dump` 写入普通 `.pkl` 和同名扩展 FIFO，输出字节完全一致；全部字段反序列化形状/dtype/字节一致，gzip level1 往返字节一致。空 query 原始 13,037 bytes；900-query 原始 33,387,235 bytes、gzip 4,360,958 bytes。按此 900-query 布局乘 3316 帧为 110,712,071,260 bytes，只用于容量估计，不宣称已产生全部真实 query。
- 初版预检误用 `.fifo` 扩展，被 mmcv 格式分派拒绝，失败保留。只对已确认的本次失败预检容器释放空 FIFO 读线程，使其按原错误退出；没有终止其他任务。v2 改为作者原生 `.pkl` 扩展并让测试读线程 daemon 化，完整原生字节资格通过；作者代码、权重、模型阈值和预算没有改变。
- 已保存后续执行合同：每帧原生 `.pkl` FIFO 接收完整关闭的 pickle 字节，压缩传至 Mac 并核对 size/SHA 后确认；下一 FIFO 的打开提供背压，不拆分原生 RNG 时序、不删字段、不截断 query。每 query 上限 128 MiB，远端非 query 输出上限 2 GiB，容器 6 GiB，Mac 留 1 TiB；本机实际余约 3.08 TB。此流程不作为正式同资源测量。
- 完整双向流控制器尚未实现/冻结；INF/cooperative 元数据 query token 覆盖仍需核对；图片远端验收仍待原传输完成。尚未启动 INF GPU 推理、尚未导出真实全帧 query，也没有新增完整 J 或论文结论。接续入口为 `/tmp/rbf-current-CoopTrack-full-native-INF-query-preparation`。

### 2026-10-07 12:02 CST：35.112 全量 INF 原生 GPU 与 query 流已启动

- INF 原始 3441 张图片在 35.112 全量 size/SHA 验收完成。INF val 21 序列/3316 帧及 cooperative 所需 3316 个唯一 INF token 覆盖完整，缺失为零；全部相机引用均在已验图片清单内。
- 双向 query 流控制器已冻结并启动，Task `b003115a4c054cf29efe76e057d73700`，本地控制器 PID17534；35.112 实机为 RTX3060，UUID `GPU-68c03b8c-acc3-421c-0e7a-8da0bb9ad3fa`。GPU 进程和实际 query 数据均有读回证据。使用同一次连续原生全量推理，原生 query 保存功能开启；通过 `.pkl` FIFO 保留作者序列化字节，逐帧传回 Mac、完整 SHA 确认后接续，无模型源码、权重、阈值或 RNG 时序改写。
- 12:02 的独立读回覆盖前 1518/3316 帧，原始 query 共 608,331,584 bytes；每份原始 pickle 和 gzip 运输副本均重新核对 size/SHA，token 与冻结清单的有序前缀完全一致。这是部分输出资格，完整 query、原生预测、独立指标和完整 J 均尚未验收；流式传输耗时不充当正式同资源测量。
- 原 vehicle 全量数值链路继续复用，不重跑。五个既有 teacher/recovery-off Task 在 11:54 API 读回仍为 in_progress；运行日志与心跳不升级为完整产物验收。既有活进程、4/8 卡配方、历史失败、授权门槛及 main2027 损坏 rank5 限制均保持。
- 新 checkpoint `/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-full-native-INF-GPU-query-stream-progress-20261007T040201202444Z.json`，SHA256 `6edc91daee3081d0f3c3adfda8e1c62ed1992c8e363ff5ca7db70b2074dee105`；此前 700 个引用重新校验，新增 23 个，共 723 个。A–K goal 保持 active，当前轮为 progress；无提交或推送。

### 2026-10-07 12:04 CST：原生 cooperative query 读取路径资格

- 上轮新增执行及独立字节证据，分类为 progress。本轮实机确认同一 INF CUDA PID4149966 仍在 RTX3060 上运行，原日志达到 1890/3316 帧；host MemAvailable 10,661,748,736 bytes、磁盘余 8,881,496,064 bytes，仍满足运行前冻结的 6/8 GiB 保留量。没有因观察等待重启任务。
- 在无 GPU、1 CPU/1 GiB 内存上限的独立原生容器，使用真实 query token `000902`（SHA256 `ab96cc10bf1ba4d04877ed6ffd33108071a4315f552436a9eb86eca5f5cf9181`）验证作者未修改的 `LoadInfInformation`：普通文件与 `.pkl` FIFO 的 `query_feats`、`query_embeds`、`cache_motion_feats`、`ref_pts`、`pred_boxes` 全部 dtype/shape/原始字段字节一致。前三项为 `[14,256]`，后两项为 `[14,3]`、`[14,10]`，均 float32。
- 此结果证明该真实 query 的原生读取兼容性，不代表全部 query、协同模型前向或完整 J 验收；后续仍须全量 query 资格、完整协同输入与权重绑定、GPU 准入和连续原生推理。预检日志及退出码 0 已保留，活 INF 和其他任务未中断。

### 2026-10-07 12:06 CST：cooperative 全量输入与实际模型参数资格

- 本轮完成新资格工作，分类 progress。原 INF 控制器 PID17534 经实际 ps 确认存活，未重启；其完整输出仍待完成。
- 在 35.112 的 CPU 原生环境，以 1 CPU、3 GiB 容器上限并预留 host RAM 6 GiB/磁盘 8 GiB，验证 cooperative 全部 3316 帧/21 序列。3316 个实际引用相机文件逐一重读并核对大小/SHA；全部 cooperative token 及所需 INF token 与冻结覆盖合同一致，14 个相关元数据文件绑定字节证据。
- 原生 cooperative 模型严格加载实际 `coop-author-state-only.npz`，914 个 state tensor 的名称、形状、dtype、原始内容逐位相符；权重包 189,041,766 bytes，SHA256 `6098345b4e04eab33ee49e93562686b4ea54dbf65aeb352795ead73f9dab1ea8`。资格容器退出 0，源码与实际权重 release 身份边界保持。
- 已从合格 INF 入口派生并冻结完整 cooperative GPU worker：原始配置、3316 帧单次连续原生 API、seed0/world1/workers0，读取原始 INF pickle；更改限于配置选择、实际 cooperative 权重、路径和输入核验，没有改写作者模型。语法编译通过；尚未实现对应供给端控制器，也未启动协同 GPU。待全部 INF query 完整验收和 GPU 空闲后才能准入；CPU 权重资格和入口编译不代表完整模型前向或 J 验收。

### 2026-10-07 12:12 CST：INF 全 query 到齐与完整 cooperative 接续

- 本轮实现并冻结原生 cooperative query 双向供给控制器，属于 progress。35.112 逐帧请求冻结 INF token，Mac 发送原始 pickle 的 gzip 运输副本；host 解压后核对完整大小/SHA，再通过 `.pkl` FIFO 写给未修改的作者 `LoadInfInformation`。当前帧缓冲上限 128 MiB、非 query 输出 2 GiB、owned RAM 3 GiB、容器 6 GiB、host RAM/磁盘保留 6/8 GiB；完整运行上限 86400 秒均在运行前冻结。原始全部 query 保留在 Mac，未向 ClearML 上传。
- 接续控制器 PID25906 实际存活，等待原 INF 控制器 PID17534 的全量 query 完成收据，再独立重读全部原始与 gzip 文件、检查 host 输入、等待 GPU/RAM 准入，最后创建 metadata-only Task 并启动单次连续全3316帧 cooperative 推理。当前未创建协同 Task、未声称协同 GPU 已运行。
- INF 在 12:11:24 已完成全部 3316 帧 query 传输，原始总量 1,987,849,187 bytes；原生评价仍在同一 CUDA PID4149966 中进行。后续指标格式化阶段的 3316 计数不能误认成重跑推理。原生评价、预测独立读回和完整 J 尚未验收。
- 完整 INF 输出读回控制器 PID26176 已启动并等待同一原运行完成；完成后复制完整预测/评价输出、核对全部文件集合及 size/SHA。另已冻结 INF/cooperative 的全帧独立预测几何与身份复算器，仅将此前 vehicle 审计器改为 side 路径和显式输入 SHA 绑定，数值算法及 1e-8 容差保持；尚未执行，不能据此宣称通过。
- 12:10 五个既有 teacher/recovery-off API 状态仍 in_progress、无完整新 artifact；teacher1337/3407 日志继续推进，off2027/off1337/off3407 仍分别显示358/452、179/179、151/179的当前可见 schedule。API 日志不替代独立验收，未重启任何原任务。

### 2026-10-07 12:19 CST：INF 原生评价 OOM 与已完成预测的同上限接续

- 原 INF CUDA PID4149966 在作者评价格式化阶段退出 137；内核日志确认 12:13:56 为 memory-cgroup OOM，usage/limit 同为 6,291,456 KiB。完整推理已经结束，不把终止误记为全部实验成功。ClearML Task `b003115a4c054cf29efe76e057d73700` 实时 API 为 failed、无 artifact，保持失败，不重开、不上传完整产物、不提高 6 GiB 上限。
- 完整预测 pickle 90,490,358 bytes 已保存；原输出四文件均完成独立 size/SHA 读回。全部 3316 份原始 query 与 gzip 重新读取，Mac 记录与 host 全3316条确认记录完全一致，总原始 query 1,987,849,187 bytes。新独立收据位于 `completed-forward-preserved-after-native-evaluation-OOM/all3316-native-INF-query-and-completed-prediction-independent-byte-acceptance.json`，明确 original Task failed、native evaluation incomplete；没有伪造原运行完成收据。
- 原协同等待 PID25906、输出等待 PID26176、检测独立审计等待 PID28389 因上游真实失败退出，失败均保留。协同 GPU 没有启动。13 张 INF 原生 GT 表已独立读回；AP/TP 审计尚未执行。
- 从字节合格的原始完整预测，准备独立 CPU-only 原生 `dataset.evaluate` 接续，沿用 2 CPU/6 GiB、host RAM/磁盘保留6/8 GiB、输出2 GiB；不加载模型、不重跑GPU、不改作者评价、不把此流程用于正式同资源比较。初版从 JSON 构造 Config 导致 tuple/list 类型丢失，在 dataset 构造处失败；现场保留。v2 从原作者 Config 文件重建，只还原冻结路径，并断言归一化配置与原记录完全一致，保持原生 tuple 类型。控制器 PID31876 已启动，评价是否完成仍待实际结果。
- 本轮为 progress：查明真实失败、完整保全合格前向产物并实现同资源上限的阶段接续。完整 INF 指标、cooperative、J 和论文目标仍未完成，A–K goal 保持 active。

### 2026-10-07 12:22 CST：接续产物的完整独立审计链

- 同一 CPU 原生评价接续 PID31876 经 ps 确认存活，作者格式化首轮接近全部3316帧；尚无新失败，尚不能宣称评价完成。输入为已独立字节合格的完整预测，6 GiB cap 保持。
- 独立检测 AP/TP 控制器 PID32927 已冻结并启动，等待 PID31876 的完整评价文件读回收据；使用已验13张 INF 原生 GT 表与实际检测 JSON SHA，接续在 Mac 复算全部12个class×distance配对、101点PR/confidence和完整TP误差数组，保留原1e-12容差。旧PID28389的上游失败记录保留，不把新接续视为旧进程仍在运行。
- 独立完整预测几何/身份控制器 PID33354 已冻结并启动，同样等待该原生评价文件资格；原预测pickle、元数据、原生评价输出分别绑定。沿用全3316帧/21序列、原始框/分数/类别/身份与全部格式化数值的独立重算，容差1e-8不变，CPU-only6GiB并保留host6/8GiB准入。此处仅为可执行依赖链，审计尚未通过。
- V2V身份seed3407原准备子进程 PID13427 仍约99% CPU；原C/G与验收控制器36187/40091/59641/76539/83647/48570/48729/48730均经实际ps确认存活。4/8卡原配方和全部授权/资产门槛保持，不替换成单卡工作。
- 本轮为progress：将全量独立审计接到实际接续输出，保存新控制器与输入SHA，并确认其他活任务。完整INF跟踪指标、cooperative原生运行、正式同资源比较和完整A–K仍未完成。

### 2026-10-07 12:26 CST：原生跟踪评价进行中，完整跟踪独立复算链已绑定

- 原 CPU 接续已越过先前 OOM 的格式化阶段并进入作者原生跟踪评价。同一容器 `945be320f64a` 实机 Running=true、OOMKilled=false，PID4153366；HostConfig.Memory=6,442,450,944 bytes、NanoCpus=2,000,000,000、DeviceRequests=null，实测约3.405 GiB/6 GiB。没有 GPU 重推理或提高内存上限；原GPU失败Task仍保留。
- 跟踪输入与全阈值审计控制器 PID34935、最终汇总控制器 PID35189 已冻结并实际存活。前者依赖几何审计 PID33354，通过后按原输入/作者源SHA核对全部 GT与预测过滤、均值分数、原生插值，再独立复算完整匹配事件和阈值指标数组；保持1e-8容差。后者从独立数组按原生规则选最高MOTA阈值并重算最终汇总，保持1e-12容差，重新核对全部评价输出字节。
- 原生反向时间插值权重、TID/LGD的0.5秒时间基准、AMOTP米且越低越好继续保留并披露；没有更改原生算法以寻求通过。独立链已可执行，但尚未产生通过收据，不能计为指标验收完成。
- 本轮为progress；实际运行与冻结源码/绑定证据已进入ledger。完整INF指标、协同原生运行及A–K全部要求继续推进。

### 2026-10-07 12:30 CST：INF 原生 CPU 评价成功，cooperative 同上限分阶段入口

- 同一 CPU 接续已退出0，原生3316帧评价完整结束，生成7个文件。输入原始预测 SHA256 `4b66c230ba55e82b8c986675a239b5e5f9f43c6b7bc38971880c0fa07221db5e`；检测 JSON 396,964,696 bytes，SHA256 `66be58a9cd8fe4468d6e9087d5ba1279fbad105dd1418f29dab204821d2a5595`；跟踪 JSON 23,589,654 bytes，SHA256 `bd2584ff3b82bc62da9ac748682feb501f4b325d6845e02df018518403caf2e9`。7文件正在Mac完整读回，尚未据此宣称独立指标通过。
- 原GPU Task仍failed，CPU接续成功仅说明从同一已完成预测执行作者原生评价成功；没有重推GPU、增加6GiB上限或把分阶段流程用于正式同资源结论。
- 完整cooperative入口已改为一次连续原生GPU前向、释放模型进程、随后从相同预测执行未改写的作者CPU评价；两个进程分别沿用6GiB，输出总上限2GiB不变。实际914个参数、全部3316 query字节及输入资格绑定保持，原上游失败控制器保存不变。新controller PID37908 已启动，先等待同机CPU评价/几何/跟踪审计控制器31876/33354/34935释放资源，再全量query重读及GPU/RAM准入；当前GPU未启动、Task未创建。
- 本轮为progress：核实同上限原生评价成功并冻结完整协同分阶段执行链。完整INF独立数值链、cooperative原生结果、正式同资源和全部A–K要求继续推进。

### 2026-10-07 12:43 CST：INF 全量独立数值链通过，修正 cooperative GT 后实际运行

- INF 原始预测及全部7评价文件完成独立 size/SHA 验收；GPU失败与同6GiB CPU评价成功分别保留。独立几何覆盖45,978个原始跟踪框/994,800个检测框，格式化42,051/917,885个框，最大translation误差4.547473508864641e-13。全部12检测class×distance的AP/101点曲线误差0；6060个TP数组标量及15汇总最大误差8.881784197001252e-16。
- INF跟踪输入86,666个框、1,386,656个数值标量通过；原生插值GT249个、预测72个均保持。完整183,558次帧级匹配、1,348,379条事件及2040阈值指标标量通过，最大误差2.6645352591003757e-15；最终85项汇总最大误差2.1094237467877974e-15。独立AMOTA=0.39215501208829867，AMOTP=1.3257205306377982米（越低越好），FP7138、FN13055、IDS144、Frag400。原生GT字段14848.666666666666为类别均值，不是总量；原生插值、0.5秒TID/LGD、权重release身份未升级等边界继续披露。
- 首个cooperative Task `726ed6a819d245519242b3cba53bb07b` 在第一帧标注读取时失败，图像前向尚未开始：错误绑定vehicle侧GT，而cooperative info含合并GT，首帧7/19数量不符。失败日志、首份query与Task保留。改回作者配置的cooperative GT根目录，保持vehicle相机根目录；全部3316帧/21序列、97519原始及有效标注逐帧get_ann_info通过，13个原生cooperative GT表重新绑定。
- 修正后的Task `9258703e4d5d4bc38652875eb37ac9c8` 已创建，控制器PID47109，35.112实机CUDA PID4157809、RTX3060 UUID `GPU-68c03b8c-acc3-421c-0e7a-8da0bb9ad3fa` 已观测，前26/3316帧完成；显存约2024MiB，GPU利用率62%，host RAM/磁盘保留量仍满足6/8GiB。后续保持同一次连续推理和同6GiB CPU评价；当前吞吐包含query运输与guard，不是正式资源结果。完整cooperative/J仍未验收。
- SparseCoop三侧原训练anchor通过作者函数生成和完整文件读回：1434/1521/1495训练帧，分别选取10061/21949/19391个GT框，原生range55/110/55，每侧900×11、79328bytes。NumPy seed0为预声明复现选择，不冒充作者release种子。首份vehicle检查因切片/高级索引归约布局产生5.551115123125783e-15差异而失败；诊断证明作者布局与输出逐位相等，修正检查并复用已生成vehicle文件，没有重跑KMeans或放宽精确相等。原始132源码保持；独立KMeans中心重算及完整训练模型仍未资格通过。
- [作者Release页](https://github.com/wang-jh18-SVM/SparseCoop/releases)及已检查问题页未提供训练后checkpoint入口；README仅提供初始ResNet50链接。该搜索边界不等于证明任何地方均无权重，其他用户问题中的指标不当作本项目结果。本轮为progress，完整A–K仍active，无提交或推送。


### 2026-10-07 12:53 CST：SparseCoop anchor 独立输入检查与三种子 native v2 输入收口

- SparseCoop 三侧原始训练输入从最终 `native-infos/` 归档成员按注册 size/SHA 读取；逐行独立筛选得到 vehicle 10061、INF 21949、cooperative 19391 个 GT，共 51401 个，与原始选中数组的顺序和数值字节完全一致。用 `math.fsum` 独立计算尺寸，最大绝对误差 3.3306690738754696e-16，满足预先冻结的 1e-12；2700×11 anchor 全部有限，cos/zero 列完全相符。
- 独立最近中心分配及簇均值仅作为诊断：vehicle 最大中心均值差 8.8818e-15，INF 0.04767525，cooperative 0.06497174。未重算 KMeans 初始化/优化轨迹，不能据此宣称其独立全流程验收；也不修改原 anchor 或放宽阈值。首次归档选择碰到 `before-author-velocity-update/` 历史成员的失败已保留，v2 精确限定最终成员。实际训练权重仍待资格。
- 原 C 输入/验收进程正常完成，seed3407 的 32 序列、773396 行已独立准入，cross positive 77573、temporal positive 323807。三种子 native v2 输入合计 2364754 行已准入；这不等于模型 checkpoint 通过。
- seed3407 原训练闭包保持 27 文件；本地输入包 533108419 bytes、SHA `abc54e5379737cf5ecb80275e2fb67881d4fbaa6dc0218c13e2f0ceddace4e2f`，全体 archive 成员已读回。原 fit/holdout 两个头均有有效监督。注册与实际存储读回由 PID52184 执行，尚未据上传状态认定完成；训练未调度，原 4/8 卡准入保持。
- CoopTrack cooperative 同一 Task `9258703e4d5d4bc38652875eb37ac9c8`、控制 PID47109、实际 GPU PID4157809 在 35.112 RTX3060 持续运行。04:51:12 UTC 原 forward 79/3316，后续本地 query ACK 前缀97条；供给数不等于完成预测数。本轮没有重新启动原回放，运行时不作正式资源指标。
- 本轮 checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-anchor-independent-check-and-three-seed-inputs-20261007T045234577800Z.json`；A–K 目标继续 active，全部历史失败与授权门槛保留，未提交或推送。


### 2026-10-07 12:56 CST：seed3407 原注册进程与顺序准入接续

- 原输入注册 PID52184、Task `3a412aeaca70409d87146f768d97a389` 仍存活，尚未返回四产物完整读回凭据；未重启或重复创建。
- 新守卫 PID54151 已实际启动并确认原发布进程命令。完整注册读回到达后，它重新读取四个注册文件的 size/SHA，核对64成员验收、27文件闭包、semantic identity、原完整行 admission 与双头覆盖，冻结 seed3407 后继控制器。后继仅在 seed2027 原任务准入之后调用原冻结4/8卡调度器，保留 seed1337→2027→3407 顺序与物理卡交集检查。当前未调度训练，未取得 checkpoint 资格；超时或前驱失败会保留失败，不自动重发。
- API 实查原五任务仍 in_progress、无注册产物。两个 teacher 的452 schedule 观察到289与298，其他分段也有进展；recovery-off 的旧进度加新心跳不算验收。35.112 的协同控制 PID47109 保持原运行。
- 本轮 checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-seed3407-publication-guard-and-original-training-successor-20261007T045623911412Z.json`（909个绑定引用）。A–K目标仍 active，未提交或推送。


### 2026-10-07 13:00 CST：SparseCoop 全量原生训练/val 相机依赖通过

- 从已存在的 `spd-official-train-source-20260930/materialized-inputs/inputs` 与已验 val 图像复用原字节，在新目录 `/Volumes/Data/test/recover-before-fuse/artifacts/rbf-CoopTrack-exact-mmdetection3d-native-source-20261006T232725831265Z/original-SparseCoop-full-native-camera-source-reuse-20261007T0459/images` 组装4392张唯一 JPEG，总1088707702 bytes。每侧训练1521张、val675张；每张源与副本均完整size/SHA核对、JPEG全部像素解码，帧/序列/路径对应原元数据。没有修改原图、native元数据或label。此前4563是三侧metadata的缺失引用数，不是唯一图像数；该目录依赖缺口现已解决。
- 重新验证完整原生67序列转换归档SHA，读取六份最终train/val PKL。vehicle train1434、INF train1521、cooperative train1495，各val675，共6475帧；全部6475相机引用及2170 cooperative辅助INF引用均找到同split图像并再次核对SHA。仅46 train与21 val，无test图像。
- 两份完成凭据：`all-native-train-val-camera-dependencies-byte-and-decode-acceptance.json` 与 `all-six-native-info-camera-reference-acceptance.json`。这是完整原生图像输入资格，不等于实际权重/完整模型/训练通过；source release identity仍未升级。原KMeans优化轨迹独立重算仍未完成。
- seed3407原注册PID52184与守卫PID54151、CoopTrack cooperative原控制PID47109保留；没有以暂未返回注册凭据为由重启。
- 本轮checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-SparseCoop-full-native-camera-inputs-accepted-20261007T050018744440Z.json`（918个引用）。A–K active；未提交或推送。


### 2026-10-07 13:05 CST：SparseCoop 作者骨干权重字节与独立张量解析

- 下载README原链接的 `resnet50-19c8e357.pth`，完整102502400 bytes，SHA `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`，匹配官方文件名hash前缀且本地完整重读。
- 旧tar格式以受限pickle读取数据描述、NumPy独立重建storage/view；267个storage与267个tensor，共25610152元素全部有限并记录逐张量shape/dtype/SHA。未执行文件中的任意全局类，旧Parameter映射为本地受限五字段数据包装。首次padding必须零、随后未识别旧Parameter包装的两次失败及源码/log均保留；四字节padding按PyTorch原legacy加载器明确跳过，未修改权重。
- 原作者训练脚本使用8卡；单端总batch48/800e、协同总batch32/48e，并需要先训练单端、导出INF查询、以vehicle checkpoint初始化协同。本轮没有改配方或启动新GPU训练。骨干初始化权重不等于SparseCoop实际训练checkpoint；完整原生模型加载仍待验。
- 本轮checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-SparseCoop-original-backbone-qualified-20261007T050526624864Z.json`。原注册/守卫/协同回放保持，A–K active，未提交推送。


### 2026-10-07 13:08 CST：三侧SparseCoop完整模型CPU初始化守卫

- 已冻结并启动PID60488，在35.112等待原资源门槛：容器4GiB、主机保留6GiB、传输最多128MiB，磁盘保留8GiB。05:07:50 UTC MemAvailable 9970352128 bytes，尚不满足合计资源条件；保留原协同GPU进程，不提高上限。
- 准入后使用原132文件源码及原环境，三份完整config逐侧调用作者`build_detector`与`init_weights`。仅修改骨干与anchor存储路径；保持原网络、batch、epoch。检查265骨干张量相对独立解码SHA逐字节相等（fc两张量不属于检测backbone）、原360×11 anchor初始化和全状态CPU有限性。
- 当前仅守卫运行，完整模型初始化/前向/训练尚未通过。seed3407原注册已上传四产物，完整bundle读回已越过268435456/533108419 bytes；未据上传状态认定完整读回成功。原PID52184、54151、47109继续。
- checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-SparseCoop-three-native-model-CPU-guard-20261007T050838175167Z.json`（940引用）。A–K active，未提交推送。


### 2026-10-07 13:12 CST：seed3407 注册输入全读回与协同几何验收接续

- seed3407 Task `3a412aeaca70409d87146f768d97a389` 的四个注册产物以及64个bundle成员完成全字节读回；receipt SHA `37d5edd3622d6685d311a3af13e1d1a0571a59d19f1d492eda5144da24475ce7`。原PID52184与守卫54151正常收口；后继PID60748实际存活，等待原seed2027首个空闲卡准入，训练尚未调度。
- CoopTrack cooperative原PID47109继续，976/3316为05:09:05 UTC的query供给数，不能当作完成预测数。已启动后继PID62205：等待同一完整forward/CPU evaluation成功，按终态manifest全量SCP并验证每个输出size/SHA；随后使用未改的INF/cooperative几何独立审计器与cooperative元数据，保持1e-8阈值。CPU审计先等待原SparseCoop CPU守卫60488终止，保持6GiB容器与6GiB主机余量，避免两项CPU检查抢占同一资源。
- 输出与几何均尚未通过；完整检测/跟踪独立评价仍须接续。checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-seed3407-registered-input-and-cooperative-audit-chain-20261007T051212112378Z.json`。A–K active，未提交推送。


### 2026-10-07 13:16 CST：协同完整检测独立评价接续

- 已启动PID63632，原INF通过的独立AP/TP脚本逐字节不变，仅绑定实际cooperative的GT表与后续输出路径。13张cooperative GT表正在读取，必须全部size/SHA通过后才接续；当前partial copy不构成验收。
- 控制器等待PID62205的完整原输出字节读回，随后在Mac以2线程计算12组class×distance的完整匹配、101点PR/confidence、AP与所有TP误差数组/摘要；预先冻结容差1e-12不变。此项尚无数值通过结论，完整跟踪assignment/summary接续仍待准备。
- 原协同PID47109继续，05:14:53 UTC query供给1526/3316，不作为完成预测数。其余CPU守卫和身份训练接续保留。checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-cooperative-independent-detection-successor-20261007T051527636859Z.json`；A–K active，未提交推送。


### 2026-10-07 13:25 CST：完整协同tracking独立链及终态资源守卫接续

- 已启动完整tracking后继PID70419，等待原PID62205的原始输出全字节读回与cooperative几何验收。三阶段分别核验全部3316帧/21序列原生过滤、均分和插值输入；40阈值下逐帧MATCH/SWITCH/MISS/FP及全部metric arrays；最终AMOTA/AMOTP等摘要与作者evaluate输出。assignment/metric源码逐字节复用；输入仅改变原生GT侧路径，最终数值聚合逐字节不变。保持原1e-8/1e-12容差、6GiB容器/6GiB主机保留。尚无协同tracking数值通过结论。
- 原SparseCoop等待守卫PID60488因只读SSH资源查询TimeoutExpired终止，失败完整保留；未发生资源准入、传输或模型运行。新PID72172接续，串行等待协同tracking终态，再在原10小时绝对期限内容忍并记录只读查询超时；原worker、准入后传输与模型执行代码逐字节不变，4GiB/6GiB/128MiB门槛不变。初始化尚未开始，不作模型或权重资格结论。
- 原cooperative GPU4157809在同轮SSH复核中仍存活、RTX3060 UUID不变；随后SSH观测有连接超时，不能据此判定原GPU停止，也没有重启。原47109/62205/63632保持。checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-full-cooperative-tracking-and-SparseCoop-guard-resume-20261007T052557439417Z.json`，全部A–K目标及授权门槛保留，未提交推送。


### 2026-10-07 13:40 CST：DMSTrack第二套完整独立指标与求和差异因果定位

- 在Mac完整重算1993帧/9序列、871176个GT×prediction 3D IoU。独立顶点/边交点面积与SciPy分配，没有导入作者corner、polygon_clip、Munkres或metric函数。保留原轨迹分数逐项求和、反复均分和跨阈值valid标记。37个阈值TP/FP/FN/IDS/Frag全部与原Py3.7 Task日志一致；AMOTA0.4351586619561412、sAMOTA0.9150411854995134差值为0，AMOTP0.5794228435944282差值2.220446049250313e-16，保持1e-12容差。18个原始GT/预测及完整原日志size/SHA在计算后再次通过。AMOTP是mean3D IoU，越高越好。
- 只改变第二实现的求和函数为新版builtin sum，其他函数AST完全相同，又逐项复现既有失败modern作者引擎的全部37行计数和三个指标，17个阈值计数改变。确认此前差异来自Python3.12浮点sum语义改变及原评价器反复均分造成的等分数边界变化；官方说明：https://docs.python.org/3.12/library/functions.html#sum 。没有把modern数值作为论文改进，没有改变阈值或原预测。
- 本地评价wrapper仅在已验原作者module命名空间恢复原CPython3.7逐项sum，作者源码/数值函数体不变。相关61项测试通过；实际完整修复wrapper评价PID78832、严格acceptance后继PID79300存活，尚无该修复wrapper整体验收结果。历史失败、原completed Task注册失败均保留，不重开原Task或重跑模型。
- 35.112原GPU4157809继续，原生推理1887/3316；原47109/62205/63632/70419及Sparse CPU守卫72172保持。独立指标组件通过不等于J、同资源I或论文全收口。checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-DMSTrack-second-independent-full-metrics-and-sum-cause-20261007T054050635534Z.json`。全部A–K仍active，未提交推送。


### 2026-10-07 13:54 CST：用户进度实查与DMSTrack修复指标收口

- 修复后的作者完整评价与原Py3.7任务三个指标差值0；37阈值计数全部相符。原7个作者文件、32个评价输出及输入SHA执行后再次通过；第二独立实现最大指标差值2.220446049250313e-16。原CPU预测、全部失败和Task注册失败保留，原模型不重跑。GPU原环境、完整cloud输出及同资源仍未通过，J不计完成。
- 35.112同一GPU PID4157809实际存在；原生forward2057/3316，RTX3060 UUID GPU-68c03b8c-acc3-421c-0e7a-8da0bb9ad3fa，2050/12288MiB。日志ETA2578秒仅对应前向，不包括CPU评价/运输/独立验收。原47109/62205/63632/70419及Sparse守卫72172保持。
- 五任务API仍in_progress且无产物；teacher日志452段当前观察318/452与316/452，属于单段而非teacher全体进度。recovery-off三项日志计数保持358/452、179/179、163/179，心跳不代表完整通过。三种子原4/8卡训练守卫仍等物理空闲，未调度/checkpoint未资格。
- 本次实查收据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-user-current-progress-20261007T055412371220Z.json`；checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-repaired-DMSTrack-metrics-and-user-progress-20261007T055412371220Z.json`。A–K仍active，未提交或推送。


### 2026-10-07 14:01 CST：DMSTrack原Python3.7/CUDA11.3公开二进制资产全量收口

- 精确绑定作者SETUP中的Python3.7、Torch1.12.0、Torchvision0.13.0与CUDA11.3。官方索引实际href为download-r2.pytorch.org，原域名断言失败已保留；r2资产返回403，旧官方download.pytorch.org同一文件名返回200，以索引原完整SHA不变取得。没有换模型、版本、SHA或扩大2GiB下载上限。
- Torch文件1837658577bytes，SHA77b89f124e0403659ee034e38b479ec1761ed72bb6d767d4339eeac2db986426；9615成员/3447431668展开bytes全部CRC与RECORD SHA通过。Torchvision文件23393586bytes，SHA4eb85dae2f07322f02f7e6bb3aaea0baacd8d28166bbfad6a82aeacac841556a；160成员/69019701展开bytes全部通过。两文件再次完整size/SHA读回后生成组件收据。
- 原CPU环境和已验CPU输出不变；没有安装到35.112或执行新GPU推理。原协同47109及输出/geometry62205、detection63632、tracking70419、Sparse守卫72172当前实际存在。SSH只读环境资源观察超时/255单列，不据此判原任务终止或重启。下一步须资源释放后核验35.112实际内存/磁盘，在独立环境导入验证，再完成作者GPU1993帧和独立指标。原CUDA detector完整环境、GPU原生方法和同资源未计完成。
- 资产收据：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-public-author-native-asset-acquisition-20261006T214405929601Z/original-DMSTrack-CUDA113-Py37-runtime-public-assets-20261007T054759430306Z-official-legacy-endpoint-v3/both-original-Py37-CUDA113-public-runtime-assets-byte-qualification.json`；checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-DMSTrack-original-CUDA113-public-runtime-assets-20261007T060155950416Z.json`。本轮progress，A–K active，历史失败与授权门槛保留，无提交或推送。


### 2026-10-07 14:09 CST：DMSTrack独立原Py37/CUDA11.3环境实际接续守卫

- 原9775公开wheel成员绑定已进入实际guard PID96594，等待47109→62205→70419→72172实际进程释放并取得各自终态证明。缺少进程且无终态会保存失败，不重开前驱；只读SSH资源观测失败在固定10小时绝对期限内记录和再查，远端变更失败不自动重试。原cooperative/detection/tracking/Sparse链保持。
- 新环境拥有的RAM上限6GiB，CPU import容器2GiB/2线程，主机余量6GiB、磁盘余量8GiB；按14GiB总MemAvailable及6GiB shm空闲准入。两wheel5.3775GB传输+展开峰值在6GiB内；全部抽取文件SHA核对后才删除新环境自有的已完成transport副本，原Mac公开wheel永久保留，既有数据/失败/环境不删。
- 作者原Python3.7.16、CPU依赖目录及源目录只读复用；GPU版Torch/vision来自新的独立目录。未传任何RBF被拒绝的CUDA branch-state/Linux priority包。CPU阶段明确无GPU设备，验证原Torch1.12.0+cu113/vision0.13.0+cu113、CUDA11.3版本、公共二进制CPU dense数值及作者入口--help；不把此项当作完整模型或GPU前向资格。所有9775文件执行后重读，远端完成收据及7输出再按原size/SHA完整取回后才收口。
- 当前guard确实存活，remote_mutation_started=false；源SHA保持、三脚本语法通过，只能说明接续与冻结已准备，实际部署/导入尚未通过。同轮35.112 SSH确认原GPU4157809存活、native2176/3316；日志ETA2722秒只对应前向。
- binding：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-public-author-native-asset-acquisition-20261006T214405929601Z/original-DMSTrack-CUDA113-Py37-runtime-public-assets-20261007T054759430306Z-official-legacy-endpoint-v3/owned-Py37-CUDA113-runtime-preparation-after-original-predecessors-20261007T060722315748Z/preexecution-binding.json`；checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-DMSTrack-owned-CUDA113-runtime-predecessor-guard-20261007T060954266004Z.json`。本轮progress；A–K active，未提交或推送。


### 2026-10-07 14:36 CST：M3完整历史差异定位及用户进度实查

- 原M3 seed2027 first/replay三对完整原流覆盖3316帧/21序列，19680483个float leaf逐位比较，六文件执行前后完整size/SHA通过。association全字节一致；预测仅两处yaw数值差异，均为bicycle 0003:000001，最大4.440892098500626e-16；诊断22处yaw差异，其余数值、协方差、分数、身份、匹配与birth/kill决策完全一致。207帧原字节差异主要由commit哈希链继续传播。实际轨迹经birth→miss→下一次CI update后数值收敛，原失败没有被覆盖。
- 第二独立检查完整car评价载荷：3316帧（含空car帧），79780个car预测，176780771 canonical bytes，first/replay SHA均为d3a7524809efb7a7ef464a9e09d9d27c681e6969e581551f2c2ebe8bd094201f。没有运行新指标或改变M3统计准入；历史CPU触发状态未捕获，完整真实错误因果反事实未通过，E/K/论文不计完成。
- 云端Task 6d1756dafd48425a8e90df2c2ad65549 completed；三个注册资产、10个归档成员独立完整读回并再次与本地源size/SHA核对。归档79217 bytes，仅诊断收据与脚本，没有新原生模型执行或受拒绝执行包运输。注册凭据：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-M3-full-historical-differential-registered-archive-20261007T063031573991Z/independent-readback-receipt.json。
- 实查35.112同一RTX3060 GPU PID4157809仍存活，原生forward 2961/3316；日志817秒ETA只对应前向，后续CPU评价及独立验收尚未完成。47109/62205/63632/70419/72172/96594实际存活；DMSTrack原CUDA环境守卫等待47109，remote_mutation_started=false。
- 五注册任务仍in_progress、artifact为空；teacher452段最新331/452与328/452，其他段同时推进，不能当作完整teacher百分比。recovery-off日志仍358/452、179/179（单序列）、163/179，未获全体产物。原4/8卡训练守卫保留，无新训练调度/checkpoint资格。
- 用户实查：/Volumes/Data/test/recover-before-fuse/receipts/rbf-user-current-progress-20261007T063656078725Z.json；checkpoint：/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-M3-complete-historical-differential-and-user-progress-20261007T063656078725Z.json（1055个绑定引用）。A–K仍active，历史失败及门槛保留，无提交或推送。


### 2026-10-07 14:49 CST：DMSTrack原1993帧GPU完整回放与字节读回实际接续

- 已冻结并启动PID16521；实际ps/source SHA核对通过，等待原环境守卫96594真正终止及成功收据。原runtime/47109/62205/63632/70419/72172仍保留；当前未创建GPU Task、未传输或运行新GPU模型。接续不存在新环境替换或前驱重启。
- 原9696作者条目、41895原生feature文件、两个原GPU epoch3 checkpoint的68张量描述与5990完整原生输出路径绑定。每项原CLI参数与已验作者command函数逐项完全一致；作者main仍seed0/num_epochs0/val全9序列1993帧，10Hz并披露train/test重叠；原作者源码与CPU输出只读。Torch.load观察仅传递原load参数与原返回字典，逐张量完整CPU字节SHA核对实际GPU载入状态；无map_location，不修改模型值。作者本身strict load和完整forward执行仍待实际运行。
- 35.112同一UUID准入要求无其他计算进程；GPU总驻留上限6GiB、GPU空闲余量2GiB，容器6GiB/2CPU，主机RAM余量6GiB、磁盘余量8GiB，新目录总输出加归档上限2GiB。实际GPU host PID/驻留量、host余量和输出字节持续观察；超限、同卡出现其他进程或自身观察器异常时仅停止自己container，不干预其他任务。原16:07:22 UTC准入绝对期限不扩展。两个未调度准备版本的资源观察缺口与源码/收据保留，最终版本独立冻结，无已运行源修改。
- 完整原入口成功后，所有原源/feature/公开runtime/权重再次完整SHA重读；5990原输出+10执行收据归档6000成员，每个成员独立SCP读回、完整size/SHA核对并安全落盘。仅小完成收据注册ClearML，原生大预测/模型不上传；完整独立GPU指标与同资源仍未通过，J/论文不计完成。
- 本轮SSH同一原CoopTrack PID4157809观测：原生forward 本次观测未取得可靠计数/3316；未据观察超时推断原进程终止或重启。接续binding：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-public-author-native-asset-acquisition-20261006T214405929601Z/original-full-native-GPU-after-owned-CUDA113-runtime-final-guard-20261007T064708594945Z/preexecution-binding.json；checkpoint：/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-DMSTrack-original-full-GPU-actual-successor-20261007T064914643552Z.json（1068引用）。本轮progress，A–K active，未提交或推送。

- 06:48与06:49 UTC两次SSH只读观察分别连接/握手超时，无法确认更新后的remote GPU计数；未据此认定原GPU终止。16521/96594/47109及原验收链仍实际存活；最新成功native帧证据仍为06:35 UTC的2961/3316，不以query供给3251替代。最终checkpoint：/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-DMSTrack-original-full-GPU-final-live-successor-20261007T065103647172Z.json。


### 2026-10-07 15:36 CST：SparseCoop 原生路端720锚点初始化补验

原失败检查器把三种配置的锚点形状统一写为360×11；源SHA绑定的作者路端配置实际为720，原InstanceBank按原锚点文件的前num_anchor行构造float32参数。保留原失败、结果、日志和原检查器，仅对路端补跑原模型构造及初始化；车端、协作端已通过产物未重跑。

路端CPU进程正常退出，实际耗时6.001秒，813个状态张量、48,096,607个参数元素全部有限；265个ResNet backbone张量逐字节等于原权重独立解码结果，原900×11训练锚点中的前720×11共7920个标量经本地独立读回逐值一致。132个作者文件与原backbone、路端锚点在运行前后全SHA核对不变。沿用4GiB、2核CPU、6GiB主机RAM预留和8GiB磁盘预留，未使用GPU；作者48批量/8GPU/800epoch训练配方保持原样。

三种原生配置的CPU初始化证据合并于 `/Volumes/Data/test/recover-before-fuse/artifacts/rbf-CoopTrack-exact-mmdetection3d-native-source-20261006T232725831265Z/original-SparseCoop-INF-only-config-exact-CPU-init-20261007T072925550050Z/independent-three-original-model-CPU-initialization-component-index.json`。新INF-only worker继承的all_three_initialized字段仅覆盖其唯一选定路端配置；三端结论来自合并索引与原车端、协作端收据。实际训练checkpoint和完整模型前向均未合格，完整J与论文资格仍为false。

资源处理先完成原缓存字节保全：SparseCoop共享内存wheel缓存164份共2,317,667,141字节；3份官方wheel已有本地完整相同SHA，另161份264,353,550字节正按源绑定只读运输到 `/Volumes/Data/test/recover-before-fuse/artifacts/rbf-CoopTrack-exact-mmdetection3d-native-source-20261006T232725831265Z/owned-SparseCoop-completed-wheel-transport-cache-preservation-20261007T073352165286Z`，512MiB运输上限及900秒deadline先行冻结。尚未删除缓存，原安装环境、模型、作者代码和历史失败保持。CoopTrack已完成路端的camera缓存另行逐文件核对原本地完整图像；缓存释放必须再证明完整字节保全与无活消费者，不能降低DMSTrack原准入门限。


缓存字节保全与释放已完成（2026-10-07T07:48:02.052098+00:00）：164份原wheel与3441张原路端图像共3,437,375,383字节完整可从本地恢复，删除仅限逐个SHA相同的两个完成任务缓存树中3605个文件，父目录保留。共享内存可用由3,094,450,176增至6,539,255,808字节，原DMSTrack的6GiB门限未改变。首版只读/proc检查PermissionError收据保留且未删除文件；不可读项经进程及父进程身份核对仅为sd-pam/systemd或sshd/sshd认证服务，修订检查单列这些系统服务例外，其余用户进程FD/cwd/maps与运行Docker挂载交集均为空。安装环境、模型、作者源、元数据、预测、日志与历史失败未被该释放动作修改。完整收据：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-CoopTrack-exact-mmdetection3d-native-source-20261006T232725831265Z/owned-SparseCoop-completed-wheel-transport-cache-preservation-20261007T073352165286Z/exact-two-completed-transport-caches-release-authentication-service-scope-20261007T074451933059Z/complete-exact-cache-release-independent-local-readback.json`。两个原teacher452-event schedule最新控制台均344/452；五原Task仍in_progress且无注册产物，不代表完整验收。下一步仍按原runtime守卫实际状态推进，后续GPU阶段10GiB磁盘准入尚须核对，不增大或降低原上限。


2026-10-07T07:53:08.887582+00:00状态修正：原runtime96594已通过资源准入，但原控制binding的SCP进程触及60秒运输观察上限后终止；其原GPU16521与指标23628均保留上游失败并正常退出，均无GPU模型执行或新Task。随后只读复查连接已恢复；原远端CPU目录中唯一binding.json实际2,383,124字节、SHA bfcaad67ddb7d730a5024c9479767e812ac92ce2d3e3d70a53cd896615b8c780，与原冻结输入完整相同；原RAM目录空，preparer/smoke/wheels尚未传入。当前不能称原守卫仍活或runtime已完成；下一步从这些完整控制字节独立绑定接续，保留所有旧失败、原源码、900秒单asset运输上限和原绝对deadline，不重启失败PIDs。终态远端字节收据：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-public-author-native-asset-acquisition-20261006T214405929601Z/original-DMSTrack-CUDA113-Py37-runtime-public-assets-20261007T054759430306Z-official-legacy-endpoint-v3/owned-Py37-CUDA113-runtime-preparation-after-original-predecessors-20261007T060722315748Z/actual-terminal-complete-control-binding-only-readback-20261007T075207338497Z.json`。35.112实际UUID仍GPU-68c03b8c-acc3-421c-0e7a-8da0bb9ad3fa，GPU74MiB、无计算进程；内存16,860,291,072、shm6,539,255,808、disk8,873,086,976字节。完整A-K仍active，完整J/I/K与论文验收仍false。


### 2026-10-07 16:32 CST：原运行时完整字节续传与当前进度核验

成功任务的源副本和重复运输文件共9695个、2,361,306,884字节已逐个核对完整本地保全后释放；磁盘可用8,874,102,784→11,255,468,032字节。原5990项完整CPU输出树、原GPU权重/features/ZIP、环境、失败counterfactual与日志保持。9项实际GT源差异完整归档、9683项与原本地作者源完整相同；原副本释放有独立索引，未宣称仍驻留。首版遇活SFTP的无删除失败保留，随后实际等SFTP终止再执行。

原Torch SCP900秒超时留下1,042,129,920字节，实际前缀SHA及独立完整Mac副本已验。新源绑定仅传缺少795,528,657字节，716.92秒完成，在原900秒上限内；实际原完整wheel1,837,658,577字节SHA77b89f124e0403659ee034e38b479ec1761ed72bb6d767d4339eeac2db986426通过，原失败前缀在续传阶段未改。Torchvision23,393,586字节全SHA也通过；当前65885运行原未修改preparer CPU导入及9775文件资格，68623 GPU和68627双指标守卫等待，尚无GPU模型执行。新RAM allowance5,400,321,024加保留前缀仍为原6GiB，总transport+展开峰值6,419,633,452小于6,442,450,944；CPU/GPU/运输/绝对期限与数值容差未放宽。69658仅在完整runtime终态、完整前缀保全及无消费者后可释放冗余RAM前缀；失败则保留。

08:30:41 UTC实查35.112同一RTX3060 UUID，GPU74MiB、无计算进程；单机GPU继续此主机。五原Task08:31仍in_progress、无注册产物，teacher452事件段353/452和354/452，不能作为整个teacher完成率；recovery-off2027/1337该段358/452和349/452，3407最新console单段171/179为旧日志状态，不能据其ETA判全任务完成。CoopTrack全3316帧21序列独立数值组件、SparseCoop三配置CPU初始化和DMSTrack原CPU全1993帧9序列双指标收据本轮SHA重新核对通过。完整训练、teacher、learned回放、两数据集主表/消融、至少3次同资源重复、Long-SCOPE与完整论文仍待收口；SPD探索性负结果和V2V重叠披露保留。

用户实查：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-user-current-progress-20261007T083255095405Z.json`；完整checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-current-progress-DMSTrack-source-exact-wheel-and-preserved-cache-20261007T083255095405Z.json`。原脏工作树保留，未提交或推送。

状态接续核验 2026-10-07T08:35:04.848758+00:00：65885已成功终止，原runtime完整9775文件资格及4份返回收据当前SHA全部通过，CPU导入已完成。原69658缓存释放因/proc417609权限观察失败，在删除前终止；只读实查该进程State Z且FDSize0，新源冻结单列已终态且无文件表的进程，继续保留所有活消费者/SFTP及未知进程门限。73576成功释放唯一冗余失败RAM前缀1,042,129,920字节，完整原失败字节独立Mac文件与历史失败保留，原新runtime无改。实际68623在MemAvailable13,271,363,584、disk11,251,073,024及同GPU空闲条件通过原准入，创建Task21416cb7d67146ffad011e811ae0a7b7，已进入original_complete_GPU_native_execution调度阶段；尚未独立观察模型forward帧数，不以Task/调度阶段当作全集完成。68627独立GPU双指标继续等待完整6000成员读回。用户最新收据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-user-current-progress-20261007T083504848758Z.json`；checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-current-progress-complete-original-runtime-and-GPU-dispatch-20261007T083504848758Z.json`。

2026-10-07T08:35:50.662048+00:00终态修正：68623原GPU主机脚本在host.py:50导入tools.event_track_v2x.run_public_baseline时ModuleNotFoundError tools，发生在原GPU容器和模型forward启动前；实际远端失败完整取回于`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-public-author-native-asset-acquisition-20261006T214405929601Z/original-full-native-GPU-after-source-exact-prefix-suffix-runtime-20261007T082324619246Z/actual-remote-original-GPU-import-failure-readback-20261007T083550662048Z.json`，本地controller也保留终态失败。原progress.json execution阶段已过期，不作为活任务/模型运行证据。原runtime与完整输入前置字节资格保留，未自动重试模型；下一步先源绑定修复远端导入依赖和路径。最新用户实查`/Volumes/Data/test/recover-before-fuse/receipts/rbf-user-current-progress-20261007T083550662048Z.json`，checkpoint`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-current-progress-runtime-complete-actual-GPU-import-failure-20261007T083550662048Z.json`。


### 2026-10-07 16:42 CST：DMSTrack完整原生GPU实际启动，完整接续路径资格修复

原Task21416因tools导入失败，全部本地/远端失败与冻结源码保留，独立核对Task执行身份后将其状态纠正为failed；没有重开旧Task或旧进程。根因是接续binding仅改remote_output，三个adapter远端路径及native_argv最后输出路径仍指原0647目录。统一新拥有目录后，原adapter三个文件源码不变，原host/entry/controller也不变；所有原CLI参数逐项相同，仅唯一输出路径变化。真实远端CPU import资格完整核对3文件size/SHA、实际导入co_filename、作者原CLI及native输出contract，未加载权重或做模型前向；qual目录单独保存，没有删旧失败。

新GPU controller75778、双独立指标guard75781实际存活；Task6c484a6eade04f2b9f920a026d78700c，实际owned Docker container27dee2cfd33601ff706f7cab8e958c05d011c9b35733168c93c6d20e08de4d5f主机PID4192011，在35.112同一RTX3060上运行。原Python3.7.16/Torch1.12.0+cu113/vision0.13.0+cu113/CUDA11.3版本与整数CUDA稠密算子资格通过；两个原GPUcheckpoint的全部68张量从实际GPU加载值独立比较原size/shape/SHA完全一致，无map_location、原seed0与num_epochs0不变。08:40:42实际147+114+144+198+52=655/1993帧连续affi原生文件已经观察，GPU实际驻留2908MiB低于原6GiB cap；这是活执行进度，不是完整输出或指标验收。原6GiB CPU/2核/6GiB主机余量/8GiB disk余量、2GiB output、2GiB GPU余量、21600秒exec、16:07:22 UTC准入deadline、6000成员全集与指标1e-12全部保持。

独立两指标仍等原5990输出+10收据共6000成员完整归档及全字节独立读回，不重算模型或强制GPU预测数/阈值等于CPU。运行时/作者/41895特征/两个checkpoint/6.324GB ZIP都须运行前后全SHA再验；完整J/I/K与论文仍未完成。用户收据`/Volumes/Data/test/recover-before-fuse/receipts/rbf-user-current-progress-20261007T084220790804Z.json`；checkpoint`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-original-full-GPU-actual-model-running-and-complete-owned-path-qualification-20261007T084220790804Z.json`。本轮progress，A-K active，原脏树/其他活任务保留，无提交或推送。

2026-10-07T08:46:24.157948+00:00原A-K计划完成度实查：6份9D缓存和对应6份完整运动数值验收再次读取，54,588source帧/3,060,073候选覆盖，因果速度/原初始人口先验/完整9×9协方差与缓存字节数值资格成立；不把它称完整tracking/论文。三种子identity native-v2输入及CoopTrack/SparseCoop组件收据当前SHA再核对。完整C/D/G/H/I、E历史因果/F通用事件及K全表证据仍缺，不用组件计数折算论文总完成率。08:45:39 UTC原DMSTrack同GPU4192011仍running、2908MiB，实际原生affinity1993/1993帧文件齐；原CPU评价/最终entry退出、6000成员全字节与独立GPU双指标仍待，不能计fullJ。Plan audit：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T084538313161Z.json`；checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-full-plan-completion-audit-and-actual-GPU1993-files-20261007T084624157948Z.json`。负M4vsM0 AMOTA95%区间[-0.09639589576961337,-0.04199523249068433]保留，SPD已见val探索性/V2V重叠披露不变。


### 2026-10-07 09:05 UTC 当前 plan 完成度核验

DMSTrack 在 35.112 的原生 GPU 执行成功退出：1993 帧、9 序列、5990 原生文件、42769 条预测、37 阈值，GPU 总驻留峰值 3049259008 bytes，无资源违规。4 份实际原生终端 JSON 已完整字节读回并独立验收组件；原始完整归档已生成（830751115 bytes），本次核验仍在运输，6000 成员完整字节和双独立指标验收尚待，不宣称完整 J 或论文通过。

C 原首空闲守卫 83647 缺失，已核对没有相应 Task 或创建意图；来源绑定的守卫 81599 继续原检查器及科学消费者，保留旧记录，沿用保守原绝对期限 09:52:09.647348 UTC，最新检查仍等待不交叠的 4/8 卡资源。其余种子与 SPD 守卫保留。

A–K 仍为 active：B 的输入与数值组件已验；C→D→G/H→I→K 的完整模型、teacher、学习式回放、同资源重复与论文证据尚未收口。SPD 已见 val 的负结果和 V2V 内容重叠继续披露。当前逐项审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T090517006396Z.json`；全范围 checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-current-plan-native-GPU-terminal-and-C-source-bound-guard-20261007T090517006396Z.json`。未提交或推送。


### 2026-10-07 09:12 UTC DMSTrack 原生 GPU 全量产物字节验收

830751115 bytes 完整原始归档已读回，6000 成员（5990 原生文件 + 10 份记录）大小与 SHA 全部通过。独立完整清单核验同时通过全部运行前预期文件、4 份实际原生终端记录、两份原始 epoch3 权重与输出 epoch0 文件的完整字节等价。原 Task `6c484a6eade04f2b9f920a026d78700c` 已 completed，小型注册验收记录也已从实际 artifact 完整下载并验证大小/SHA。

原模型控制器 75778 正常终止；原独立评价控制器 75781 和作者评价子进程 94721 已实际存活，按冻结 1e-12 容差重新计算完整 GPU 结果，第二套独立评价仍待接续。不得将 Task completed 或全量字节验收替代双指标、完整 J、同资源或论文验收。C 首空闲守卫 81599 继续等待原4/8卡准入，无最终 checkpoint。

全范围进度 checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-DMSTrack-complete6000-native-GPU-bytes-and-actual-two-metric-chain-20261007T091220121384Z.json`。A–K 原始目标、失败产物和授权门槛保持，未提交或推送。


### 2026-10-07 09:19 UTC DMSTrack 完整原生 GPU 双独立指标验收与调度状态修正

原 DMSTrack GPU 输出已完成双独立评价：1993 帧、9 序列、42769 条预测、31419 个 GT 对象、37 个实际阈值。AMOTA = 0.4351586619561412，sAMOTA = 0.9150411854995134，AMOTP = 0.579422890540757（mean 3D IoU，越高越好）。作者源码锁定重算最大误差 1.1102230246251565e-16，第二独立几何/分配评价最大误差 0；冻结 1e-12 容差未调整，全部阈值 TP/FP/FN/IDS/FRAG 计数一致。第二评价完整覆盖 871176 个 GT–预测 3D IoU 对。最终独立索引重新核对当前6000成员、21196评价环境文件、完整9序列GT、作者评价源码和报告字节，不重跑模型。

独立组件索引：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-public-author-native-asset-acquisition-20261006T214405929601Z/original-full-native-GPU-complete-owned-path-bindings-20261007T083754739520Z/independent-complete-original-GPU-two-evaluator-component-index-20261007T091726637173Z/independent-complete-original-GPU-native-execution-two-metric-and-full-byte-component-index.json`。该组件通过不替代完整 J、同资源 I 或论文验收；V2V 内容重叠与标称10Hz继续披露，实际 GPU AMOTP 与 CPU 的约4.7e-8差异按原结果保留。

SPD 原生路线1337/3407守卫48729/48730现已终止，均保留 AssertionError 失败记录及最后103次检查；最后检查仍为 C 首空闲训练优先，未创建GPU Task。失败时间与原10小时上限一致，但失败 JSON 无断言行，具体失败断言不作已证实结论。没有重启或延长等待上限。C81599及2027/3407原守卫继续原资源准入和冻结期限。

完整 A–K checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-DMSTrack-original-GPU-complete-independent-metrics-and-preserved-SPD-terminal-guards-20261007T091926072863Z.json`。未提交或推送。


### 2026-10-07 09:29 UTC CoopTrack 作者版本与权重身份补验

CoopTrack 作者源码固定提交 `29f1c52c8a0ec0e2a753f0695eb4e288bc5ed399` 的127个文件及Git blob独立重核；三份跟踪checkpoint对应作者 Hugging Face release `91b46fd08c950572131c7c0f039b01bf543a4151`，当前完整大小、LFS SHA、作者声明 MD5 全部一致，safe state NPZ 的2548个参数值、形状、dtype与原安全读回结果逐项一致。合作原生GPU模型实际严格加载914参数，基础设施模型实际严格加载817参数；两个实际绑定的127源码文件均与作者提交逐文件一致，实际模型前向记录的源配置改动限于原绑定的评估路径/worker及作者公开的INF查询输出开关。

新联合组件索引重新核对全部11份原生合作输出、372532995 bytes，并绑定原几何/检测/跟踪独立数值证据及完整作者身份链。`source_weights_release_identity_upgraded=true` 仅对这个新独立组件成立，旧记录保持原值。基础设施原CPU评价OOM及失败Task保留；没有重跑模型、指标或改变作者逆时间插值、TID/LGD时基。索引：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-CoopTrack-exact-mmdetection3d-native-source-20261006T232725831265Z/independent-author-release-to-actual-cooperative-and-INF-execution-qualification-20261007T092640708392Z/complete-cooperative-native-numeric-and-author-release-identity-component-index-v2.json`。

J 仍缺 SparseCoop 原生训练checkpoint和完整前向、Long-SCOPE 作者资产；正式同资源 I 仍未完成。当前物理4/8卡门槛没有合格候选，五个teacher/recovery-off仍 in_progress且无最终artifact。C81599/48570/60748已核实存活，继续原期限和原门槛；未延长或重启已终止SPD守卫。

全范围checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-CoopTrack-complete-native-numeric-and-author-release-identity-20261007T092944453379Z.json`。完整A–K继续active，未提交或推送。


### 2026-10-07T09:48:33.055658+00:00 — 当前 plan 与 M3 条件数值机制

A–K 尚未整体验收；组件计数不能换算为论文完成率。六份 9D 缓存及完整协方差组件已验。C 最终 checkpoint、D 完整 teacher/优先级、G/H 正式回放、I 同资源至少三次测量和 K 完整论文表仍是关键路径。

原始 23 源状态的四次有界 CPU 诊断保持输入、theta、heading 和 NumPy 扩展字节不变。默认 AVX512 匹配 FIRST，禁用 AVX512 匹配 REPLAY；五个源 yaw 各相差一个 binary64 ULP，最大差 4.440892098500626e-16。独立条件机制核验与实际注册 Task 5a5c8c9a5c5240b19df29ed0173511a7 completed 的 13 成员全文字节读回均完成。历史 CPU 触发配置未记录，完整错误因果反事实未证实；原始严格失败与 M3 统计排除保留。初始 stdin 绑定失败保留，未执行模型。

C 最新实际检查 2026-10-07T09:46:44.091777+00:00 仍等待不相交空闲 4/8 卡，训练未派发；原截止 09:52:09.647348 UTC 不延长。SPD 两原 guard 终止失败保留，不自动重启。单机 GPU 使用 10.100.35.112。DMSTrack/CoopTrack 已验原生组件复用；SparseCoop 全前向/训练权重、Long-SCOPE 作者资产及正式同资源比较仍缺。

审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T094833055658Z.json`。Checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-current-plan-and-M3-conditional-dispatch-acceptance-20261007T094833055658Z.json`。保留脏工作树，未提交或推送。


### 2026-10-07T10:00:24.836271+00:00 — SparseCoop 完整 train 锚点独立优化与 C 原截止终态

独立 NumPy 实现未调用作者 generator 或 sklearn KMeans，消费三侧完整 train 的 51,401 个过滤 GT，执行每侧10次初始化及原300迭代上限、原方差缩放停止条件。全部2700个有序中心8100坐标与封存原结果匹配，最大绝对差2.842170943040401e-14，冻结阈值1e-8；完整输入前后SHA及最优初始化选择独立核对。作者内部逐迭代轨迹未记录，不能宣称逐步bitwise匹配。独立计算在900秒与512MiB观察峰值上限内完成，原锚点未改，未重跑已验作者生成任务。

注册 Task f763c5c03965406697824233fbf11196 completed，20成员全字节读回，archive23349599 bytes/SHA 47ce3c8a41d2697c8e3e1b9bf10e90122ff4b46d8f90d9132c5d7f362813d488。注册器沿用模板的错误artifact key已经通过仅改元数据key的补充绑定纠正为 SparseCoop-independent-full-train-KMeans-path；完整URI/size/SHA和执行源码不变，SDK completed-status编辑失败单列保留。未通过改状态或重跑科学任务解决。SparseCoop车端/路端原8卡batch48/800epochs，协作原8卡batch32/48epochs，协作依赖车端iter23200和路端train-denoising/val queries；初始化、锚点资格不是训练checkpoint或完整前向资格。

C81599已在09:52:09.647348UTC原绝对截止终止，确切失败为deadline assertion；09:52:47实际同名训练Task查询为空、durable dispatch journal不存在，2027/3407两原后继guard仍存活。没有自动延长、重启或修改原门槛。09:50实际五原teacher/recovery-off任务均in_progress且无注册artifact；物理4/8卡合格候选为空。当前完整A–K目标继续active。

完整checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-SparseCoop-independent-KMeans-and-C-deadline-terminal-20261007T100024836271Z.json`；plan审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T100024836271Z.json`。脏工作树与历史失败保留，未提交推送。


### 2026-10-07T10:13:19.598445+00:00 — 第一阶段完整配对 birth-score 条件反事实

复用原三种子 M0/M3/M4 封存全文输出及已验独立完整回放，不重跑模型或 producer。每种子3316帧21序列：association 文件SHA相同、全部原源记录精确相同、全部非score node字段精确相同；M0/M4仅残余birth有效分数不同，M3/M4仅残余node更新分数不同。逐序列首次输出分歧独立核验得到57个birth-only证据：此前全部输出状态相同、当前assignments及非birth tracks相同，差异仅为新birth接受或初始score。另19个首次离散决策分歧有birth阈值证据，例如加权0.130661拒绝、原始0.994102接受。所有记录均覆盖全三种子，不以个例代替全输入检查。

每种子26648个、总79944个GT帧观测逐项配对；标签来自已验几何互唯一及全序列最优ID归属，unknown/删失保持，GT仅在封存预测后评价使用。M4观察错误区间数为201/193/188，M0为205/209/195；区间减少不代表质量提升。复用已验10000次sequence cluster bootstrap：M4−M0 AMOTA95%CI[-0.09639589576961337,-0.04199523249068433]，Frag95%CI[-111.66666666666663,-7.0]，负AMOTA/更少碎片的质量权衡完整保留。条件birth分数总效应合格，不主张物理身份因果证明或恢复独立优势。

实际注册Task 6b5a4b086ca5427c99aec71e741253c1 completed，29成员、7834311压缩bytes全部逐字节读回，archiveSHA0e28bac259241d35f5579e9fd210dedb1a301cb3e27e7be00d4e0d970aaa9327。M3 seed2027严格失败、统计排除及原CPU历史触发未知继续保留。完整M0/M2融合替代反事实和更广错误链仍待；完整E/K/A–K均未完成。C81599原截止终态保持，2027/3407原后继guard当前ps仍存活，不延长或重启。

完整checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-complete-paired-birth-score-counterfactual-20261007T101319598445Z.json`；plan审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T101319598445Z.json`。脏工作树保留，未提交推送。


### 2026-10-07T10:33:07.092960+00:00 — 完整融合替代条件反事实与已复核案例图

原三种子 M0/M2 的全部3316帧/21序列保留相同源输入、配对、权重、score及birth规则，独立核验25314个匹配组件、10692个几何变化节点与57个首次输出 geometry-only 证据。总79944个GT帧观测仅用于封存预测后的评价，配对标签边际与原六份评价记录精确一致。实际Task faf143961d434e51b7f146902e02157d completed，31成员、9110837压缩bytes全部独立读回。M2−M0 AMOTA95%CI[-0.013493470799593519,0.010003762895546624]、Frag95%CI[-35.333333333333314,52.333333333333314]均跨零；不同种子观察错误区间数有增有减。该条件更新效应不证明物理错误配对身份或恢复独立优势，原M3严格失败与统计排除保持。

新案例图严格绑定已验birth/fusion反事实和原10000次配对bootstrap数值，显示同node birth阈值、源状态与融合协方差几何、总体质量区间。首次PNG的底部注释重叠已单列视觉失败记录；新版本仅增加画布高度/底部留白，科学数值逐字节相同。重新打开PNG确认坐标、图例、注释、footnote无可见重叠或裁切。Task a282ae3be9ee435fb4ea3e2fb857b8fa completed，10成员、247469压缩bytes全部实际注册字节和成员读回通过。SVG保持可编辑文本；图仅为SPD已见val探索性组件，完整K/论文仍未验收。PNG：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-controlled-counterfactual-reviewed-case-figure-20261007T103110966416Z/paired-counterfactual-cases.png`；独立读回：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-controlled-counterfactual-reviewed-case-figure-20261007T103110966416Z/independent-readback-receipt.json`。

18:29北京时间实际查询五个原teacher/recovery-off任务均in_progress、无最终artifact；C2027/3407原控制器48570/60748在18:30实际存活，C1337控制器81599已终止，原绝对等待截止不延长。完整A–K继续active，下一步接续F合法历史/事件补验和原模型/teacher/回放依赖，单机GPU继续35.112。当前checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-reviewed-controlled-counterfactual-case-figure-20261007T103307092960Z.json`；plan审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T103307092960Z.json`。脏工作树与历史失败保留，未提交或推送。


### 2026-10-07T10:42:30.347759+00:00 — F 封存同输入基线协议边界的完整事务补验

复用已验六个交叉/缺观测重现输入和全部30个JPDA、PKF、本地scan-MHT封存数据库，在独立副本上执行原始重传、冲突event ID、非单调输出、未来到达、withheld到达、跨序列、重复raw身份/检测行键和非法parent请求。原模型与producer不重跑，原上限不增加，单CPU线程、900秒及观察峰值1GiB上限冻结；Torch2.6.0/NumPy1.26.4实际执行未初始化CUDA。原八项共240请求完成；独立核验发现重复raw请求同时撞node ID和检测行键，而实际SQLite先报告(source,frame,detection_index)冲突，首次错误断言与源码保留。补30个只重复node ID、改变frame键的请求，确认独立node_id UNIQUE拒绝路径；未重跑已完成八项请求。

独立核验器不导入生产代码，逐项核对请求语义、实际错误消息、完整SQLite schema及所有行，读回全部435个原事件预测/审计/commit链、2253行原SQL数据。共270请求：30次重传字节相同，240拒绝；事务内两类UNIQUE失败均完整rollback，全部表及历史状态不变，失败后重传仍成功。原210项源码/输入/产物SHA前后不变。实际Task c63cbbea20274378b7a66a21375f43be completed，475成员、11824969压缩bytes全部实际注册大小/SHA及成员全文读回通过。独立组件：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-sealed-baseline-protocol-boundary-20261007T103734879032Z/independent-combined-protocol-boundary-component.json`；注册读回：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-sealed-baseline-protocol-boundary-20261007T103734879032Z/independent-readback-receipt.json`。

这是有限模型本地基线协议边界组件，不是正式Linux MHT资格、通用rolling历史过期合同、学习式分配或同资源成本测量，完整F及论文仍未完成。18:41北京时间五个原teacher/recovery-off仍in_progress无最终artifact；C2027/3407原守卫当前具体ps核实存活，C1337截止终态保留，不重启或延长。单机GPU继续35.112。完整A–K active，checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-complete30-baseline-protocol-boundary-20261007T104230347759Z.json`；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T104230347759Z.json`。脏工作树和失败保留，未提交推送。


### 2026-10-07T10:51:23.557140+00:00 — F 原错误gate/预算输入的同输入概率基线补验

复用原三fixture种子错误gate和预算耗尽的六份完整原raw/rows/事件时间，新增JPDA、PKF和本地scan-MHT K1/K4/K16，共30完整运行150事件。原18个RBF运行与producer不重跑。新运行及其独立完整评价绑定既有冻结backend与原helper/oracle，实际Torch2.6.0/NumPy1.26.4 CPU单线程，900秒与观察峰值1GiB执行门槛；未使用GPU或调整任何模型/搜索/数值上限。运行前source注册次序已通过独立文件名、恢复原source和source-freeze-v2补充绑定明确，旧绑定保留。

独立重算全部scan后验、合法history、选择、全部输出9D均值/协方差及72个SQL分支状态cache，150事件commit/audit完整链核对通过，最大9D差2.498001805406602e-16，原1e-8数值与1e-11质量容差不变。六输入逐项配对48个运行的raw/rows/时间。新30基线final模型regret均0、预声明评价身份损失均1/3；错误gate原RBF也regret0但真实标签不在允许动作空间。预算受限原RBF final模型regret约0.174993，身份损失仍1/3。首比较器错误地预期所有RBF也零regret，失败记录保留，改为消费并保留原正regret，不改产物、不重跑科学执行。预算耗尽仅对原RBF路径成立，新JPDA/PKF/MHT未称同预算或同资源。fixture种子不是训练模型种子，不作真实数据/物理身份结论。

实际注册Task b7c5bc4395e94b72ab0cb2f40834d047 completed，119成员、875241压缩bytes全部实际注册大小/SHA和归档成员全文独立读回。比较：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-same-input-gate-budget-JPDA-PKF-local-MHT-20261007T104444601634Z/qualified48-same-input-gate-budget-comparison.json`；读回：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-qualified48-gate-budget-baseline-evidence-20261007T104942518702Z/independent-readback-receipt.json`。通用rolling历史expiry、其余per-scenario基线、学习式分配、正式Linux MHT和同资源测量仍未完成；完整F/A–K继续active。五原teacher/recovery-off18:43 metadata仍in_progress、无最终artifact，C2027/3407原守卫当前实际ps存活，C1337原期限终态保持。单机GPU继续35.112，脏工作树及历史失败保留，未提交或推送。checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-complete30-gate-budget-baseline-20261007T105123557140Z.json`；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T105123557140Z.json`。


### 2026-10-07T10:58:10.820671+00:00 — F 原空侧/无边输入的45个完整同输入基线

从已验九个原registered SQLite逐字节读回raw、潜在因子和初始空输入/后续new事件时间，复用both_sources_empty、source_zero_empty、allowed_parent_edges_empty三case/三fixture种子。新JPDA/PKF和本地scan-MHT K1/K4/K16共45运行90完整事件；原九运行不重跑。全部输入仅允许birth，唯一合法模型根作为after-prediction参考，不使用物理GT，不能称物理身份性能或恢复优势。原模型/搜索/数值上限不增加，CPU单线程900秒/观察峰值1GiB门槛、无CUDA初始化。

独立完整scan、合法history、后验/模型regret、commit/audit和9D读回通过；60个full9D预测状态最大差0，完整模型regret和唯一birth参照损失均0。九输入54个原/新运行逐项raw/rows/事件时间一致。auditor控制台沿用旧total=30显示值，实际45份报告与all45验收清单明确绑定，未为遥测文案重跑科学执行。注册Task 055be2f7ca194a22b01ad3994ebffa8c completed，138成员、458918压缩bytes完整actual registered大小/SHA及成员全文读回通过。比较：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-same-original-empty-edge-JPDA-PKF-local-MHT-20261007T105328274006Z/qualified54-same-original-empty-edge-input-comparison.json`；读回：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-qualified54-empty-edge-baseline-evidence-20261007T105621880701Z/independent-readback-receipt.json`。

完整F仍缺通用rolling expiry、原late输入的其他方法对照、学习式分配、正式Linux MHT和同资源成本资格。C/D/G/H/I/K关键依赖仍未完成，五个原任务18:51 console/metadata均in_progress无最终artifact；C2027/3407具体原守卫当前ps存活，C1337截止终态保留。单机GPU继续35.112，脏工作树、原失败及已验组件保留，不提交推送。完整A–K active，checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-complete45-empty-edge-baseline-20261007T105810820671Z.json`；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T105810820671Z.json`。


### 2026-10-07T11:13:00.082976+00:00 — F 原迟到事件离散验收及同输入15概率/MHT基线

先前原9个late-before-deadline RBF运行虽有输入和9D数值验收，旧acceptance明确identity_ranking_and_exact_posterior_regret_admitted=false。此次前置检查拒绝把该状态当完整离散验收，保留原false记录及缺口receipt，原producer不重跑。新独立oracle补验完整126事件：合法根身份history、Z、前沿残余区域及遗漏质量、eta/组件乘积、同动作Hammingrisk/regret、因果Bayes更新及kappa放大界；另补active类SHA/精确权重、retained Z、输出类SHA和完整audit链。全部9D输出最大差2.220446049250313e-16，regret全0。三个K1均由此前未active、非旧输出且仅前沿覆盖的a,b,b历史成为当前合法输出；predecessor已discovered且previously_unexpanded=false，明确不声称此前从未展开，也不是物理GT或真实身份风险收益。

复用原三fixture输入逐项raw/因子/到达可用集/decision和reference时间，新增JPDA、PKF、本地scan-MHT K1/K4/K16，共15完整运行210事件537个full9D预测状态，最大独立差4.440892098500626e-16。保持CPU单线程、900秒及观察峰值1GiB准入、既有模型/数值上限，无CUDA初始化或deny包运输。完整scan后验、合法history、MAP/tie、MHT保留分支和SQL缓存、JPDA混合between-mean协方差、PKF独立Joseph推导验收均通过。与原9运行组成24同输入比较；最终RBF和MHT K4/K16精确模型regret0，JPDA/PKF/MHT K1约0.4，全部保留。不声称同预算/同资源或正式作者Linux MHT。首比较器错误地把audit prediction_sha256的commit-body语义比作完整blobSHA；失败/source保留，新v2按已验证commit_sha256核验，科学输出不改且不重跑。

Task 5a2c3bb784054919b04f93ba65cf1a7e completed，90成员、1340730压缩bytes实际注册大小/SHA及全部成员全文独立读回。原126补验：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-original-late-deadline-discrete-admission-20261007T110703695263Z/independent-original126-admission.json；比较：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-same-original-late-deadline-JPDA-PKF-local-MHT-20261007T110810648845Z/qualified24-same-original-late-deadline-comparison.json；注册读回：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-qualified24-late-deadline-evidence-20261007T111103380947Z/independent-readback-receipt.json。完整F仍缺通用rolling历史expiry、其余适用scenario对照、学习式分配、正式Linux MHT与同资源成本；完整A–K active。19:11北京时间五原teacher/recovery-off metadata仍in_progress无最终artifact，C2027/3407原守卫具体ps仍live，C1337原截止终态保留，无重启/增cap。单机GPU继续35.112。脏工作树、历史失败保留，无提交或推送。checkpoint：/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-complete-original-late15-baseline-20261007T111300082976Z.json；审计：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T111300082976Z.json。


### 2026-10-07T11:25:02.377020+00:00 — F 两轮按年龄滚动身份前缀过期反例

核对当前method/theory/experiments合同：过期不可恢复必须改变覆盖声明，前沿不能重复覆盖，窗口内不同身份状态不能因矩接近而合并。当前persistent.window_us仍只是决策loss范围，一次固定deadline MAP交接不等于通用rolling。此次在单独有限诊断运行中复用原late原始输入并明确增设一个窗口内不确定观测及第二轮独立组件；严格按state_us < decision_us - 150000对先前已到达的连续前缀逐次固定到旧当前输出，仅删除不兼容的历史，保留所有兼容的窗口内分支。该规则不是整个当前history MAP塌缩；旧SQL保留审计，已删支持不从存档重建；每次过期明确撤销full-history保证。两个周期、三fixture种子、K4/K16及不启用过期的matched control，共12实际运行144事件，六过期运行各6次事务共36，prefix深度3/4/6/9/10/12，全部事务确有历史删除，全部六运行均有窗口内多个历史保留。

完整独立scan/合法history/精确后验和same-action modelrisk/regret/commit/audit及分支9D/cache重算通过，513个输出9D状态、724个SQL分支cache9D状态，最大差2.220446049250313e-16。初始预测字节一致，两个周期后到达的消歧证据均无法恢复已删c到b前缀；过期运行late-event regret约0.333333至0.5。K4保留control第二周期也约0.166667非零，K16保留control约0至Decimal舍入1e-81，混合/负结果原样保留，不把控制都写成零。实际CPU单线程900秒/观察峰值1GiB门槛和既有数值上限不变，无CUDA。精确oracle缓存只重用完全相同nodes/loss输入的完整计算，未改变oracle或穷举域。最初auditor结果沿用模板的JPDA/PKF未执行声明，另存scope失效记录；v2误用原auditor SHA触发拒绝，其stderr/源码保留；v3通过独立source-binding保留原plan和producer SHA，全部重算完成，最终范围明确MHT-only。科学producer/输出未更改或重跑。

Task 243523d18ca24627a4a0898fb6d7b593 completed，62归档成员、1115959压缩bytes实际注册及全文读回。验收：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-rolling-aged-prefix-support-expiry-20261007T111700424909Z/independent-admission-receipt-v3.json；配对比较：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-rolling-aged-prefix-support-expiry-20261007T111700424909Z/qualified12-six-paired-rolling-expiry-comparison.json；注册：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-qualified12-rolling-prefix-expiry-evidence-20261007T112209507452Z/independent-readback-receipt.json。当前只覆盖按state排序的有限prefix-age丢弃反例，不是乱序/超窗口到达的一般合同、重开/rollback准入、可恢复separator-prior有界内存或真实长序列窗口实现；不声称物理GT/恢复性能、正式Linux MHT或同资源。完整F及A–K仍active，其他原门槛保留。19:23五原任务metadata仍in_progress无最终artifact；C2027/3407原守卫具体ps live，C1337截止终态保持，无自动重启或增cap。单机GPU35.112不变，脏工作树/历史失败保留，不提交推送。checkpoint：/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-complete12-rolling-prefix-expiry-20261007T112502377020Z.json；审计：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T112502377020Z.json。


### 2026-10-07T11:41:32.711696+00:00 — F 非连续年龄过期支持、窗口内倒序信息与封存重开/rollback接口

新增可复用本地MHT适配 /Users/lbin/transvision/tools/event_track_v2x/rolling_mht_tracking.py：RollingMHTConfig单独序列化retention_us，schema/config/policySHA绑定；按state_us < decision_us-retention_us对任意已到达观测索引构造约束，固定到上一当前输出在这些索引的身份，保留全部兼容的已有history，不能从旧存档重建已删身份支持。新超窗口消息整事件拒绝，包括valid+expired混合批次；当前reference之后的信息拒绝；原eventID重传优先按原requestSHA返回旧字节，冲突ID拒绝。_load/reopen核对policy、active/frontier约束及原committed审计record，不产生full-history保证。该适配未替代原活模型，未实现SQL存档回收或可恢复separator prior。

三fixture种子/K4/K16六新完整运行42事件，包含q(1.15秒)已到达后r(1.125秒)在1.25秒deadline前合法到达，排序CI状态重放并非负时间传播；1.28秒的过期索引[0,1,2,4]保留未过期位置3，不能用连续prefix长度代替。24次过期事务，102个完整9D输出、147个SQL分支cache，最大独立差2.220446049250313e-16。完整穷举scan/history/posterior/regret和commit/audit核验通过，每个42个事件均真实close并按database/headSHA reopen成功，原数值/模型/CPU单线程900秒/观察峰值1GiB上限不增，无CUDA。新的retention配置与policy串单列，不声称原作者MHT合同或同资源。

在私有副本执行每运行十个协议探针，60请求：6原重传字节相同；54拒绝覆盖冲突ID、expired-only、mixed-expired、future-reference、isolated nodeID UNIQUE、先插valid再nodeID冲突、非连续mask更新后注入失败、tampered policy、tampered support。独立核验不导入生产代码，逐项完整SQLite schema/表行/原raw语义与DB文件SHA读回，所有失败与重传的SQL状态相同。六次注入失败证明确已产生[0,1,2,4]暂存mask然后完整rollback，随后真实下一提交与原已验event3的prediction/audit SHA完全一致。六原新科学producer未重跑，原最终库及中间封存库SHA保持。协议source首次生成缩进错误在执行前拒绝，源码/原stderr/execution保留，v2使用独立binding后完成60探针；这没有改变已验科学源码和输出。

实际Task 97732380d0b04870ad94a8542dccc676 completed，175成员、4868106压缩bytes注册大小/SHA及归档成员全文独立读回。源码、输入、全部科学库、中间库、60probe私有库及完整before/afterSQL、失败、独立验收全部绑定。验收：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-noncontiguous-age-mask-MHT-20261007T113105524630Z/independent-complete-age-mask-and-protocol-component.json；注册：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage2-qualified6-age-mask-and60protocol-evidence-20261007T113754701574Z/independent-readback-receipt.json。当前支持语义与有限晚到/age/重开/回滚接口已验，但SQL bounded memory、可恢复separator-prior、正式真实全集窗口接入未验；完整F/A–K active，C/D/G/H/I/J/K关键依赖不缩小。19:39五原teacher/recovery-off仍in_progress无最终artifact，C2027/3407原守卫ps live，C1337原截止终态保留。单机GPU35.112不变，无重启/增cap、deny源码运输、提交或推送。checkpoint：/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-complete6-age-mask-and60protocol-20261007T114132711696Z.json；审计：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T114132711696Z.json。


## SPD 与 V2V 关键依赖分开核验（2026-10-07）

只读调用原冻结 prerequisites/publication/verify_existing，重新核验 SPD seed1337、3407 完整46序列主森林回执、最终模型、注册前置资产和实际 teacher 的源码/计划/recipe。两项仍 in_progress，尚无最终注册产物；完整 teacher 目标不计通过。SPD 可以在其原 teacher 实际完成后直接收集和独立验收，不依赖尚待训练的 V2V native-v2 模型。SPD seed2027 原 main rank5 坏资产门槛保持，不第三次读取。

V2V 的三种子最终 native-v2 checkpoint/全行前向与匹配 teacher 是另一条依赖链，冻结4/8 CUDA rank不变，单机GPU仍35.112。SPD优先级仍需独立的新 Linux CPU runtime 回执；原 source transport 自动审批拒绝保持，不改运输方式绕过。首次观察器误读不存在的 split 键，失败源码/日志保留；v2仅改为核对原 plan 的准确 full original paired train scope，所有实际资格门槛不变。没有修改生产器、创建/重启任务、提交或推送。

当前只读依赖资格证据：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-separate-dataset-core-prerequisite-audit-v2-20261007T115423523403Z/independent-dataset-prerequisite-audit.json`。完整A–K active；checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-separate-dataset-core-prerequisites-20261007T115524931888Z.json`；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T115524931888Z.json`。


## 原 teacher 完成后本机采集与完整目标验收接续（2026-10-07）

已启动只读观察原 SPD teacher1337/3407 的本机接续 PID81770，并由当前 ps 和首轮实际 Task 元数据证实存活。固定观察窗口30分钟，截止 2026-10-07T12:28:57.021307+00:00，每300秒检查，不自动重启、延长窗口或重跑生产器；每原Task采用非阻塞独占 flock 防重复。只有原Task completed且实际源码/计划/recipe匹配，才调用原冻结 collect_final_refit_capacity_teacher.py，随后 accept_final_refit_capacity_teacher.py 全部7445事件/46序列数值目标验收。实际主回执、发布证明、checkpoint、完整原采集源码与引用均预先固定SHA。任何采集或验收失败保留日志和终态，不重试；没有 priority拟合/源码运输/GPU派发。当前两Task仍in_progress，无最终产物，完整D不计完成。

C seed2027 原PID48570当前缺失，最后实际检查00121于11:52:50UTC仍等待1337、无Task；原程序36000秒断言存在，但具体退出异常/返回码未获证明，不从时间猜测。seed3407 原PID60748仍当前ps存活，1337原截止终态保留，均不重启或增加cap。控制器源码：`/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-original-two-teacher-completion-pipeline-20261007T115857021307Z/controller.py`；当前实际观察：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-original-guards-and-new-teacher-local-completion-pipeline-20261007T120022324655Z.json`。A–K继续active，checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-original-teacher-local-completion-pipeline-20261007T120022324655Z.json`；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T120022324655Z.json`。


## 全13配置可观测错误片段、跨度及删失独立补验（2026-10-07）

从已验封存逐帧标签按GT身份构造最大连续同一错误prediction/全序列owner的原生帧片段，使用实际完整预测帧钟确定缺GT、未知、更换错误prediction、明确correct及序列结束的边界。未导入原Episodes实现或原tracker。全部13×3316帧、273序列范围、2654原错误片段的所有字段、实际间隔、event links与原产物完全一致；所有标签计数重算一致，输入SHA前后不变。

2654片段中2618右删失、1499单帧；各配置未知身份对应占77.41%–80.96%。所有配置观测跨度中位数为0，因为单帧片段占多数；不把0解释为立即恢复或真实错误持续时间。M0/M2/M4三种子均值及样本标准差有完整描述性汇总，M1 std为null；M3 seed2027原严格复跑失败保留，M3不报完整三种子聚合。跨度只是首末错误观察端点之差，不估计连续物理时长/生存曲线，不跨未知区间合并，也未识别恢复因果效果。

Task b104a9e55c444aff8698e8e97e684cb4 completed；32成员、45599压缩bytes实际注册大小/SHA及所有成员全文读回。独立验收：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-independent13-observed-error-span-and-censoring-20261007T120454255453Z/independent-complete13-observed-error-span-and-censoring.json`；表：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-independent13-observed-error-span-and-censoring-20261007T120454255453Z/observed-error-span-and-censoring.md`；读回：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-qualified13-observed-error-span-censoring-evidence-20261007T120718553780Z/independent-readback-receipt.json`。旧原生预测/观察大文件由已有验收SHA引用，不重新上传或重跑。E/K/full paper仍false。20:07原五Task实时metadata仍in_progress无artifact，teacher接续PID81770与C3407原PID60748当前ps存活；原期限及其他门槛不变。单机GPU35.112，脏工作树与失败保留，无提交推送。checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-complete13-observed-error-span-censoring-20261007T120828897599Z.json`；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T120828897599Z.json`。


## 原五worker遥测与实际科学日志间隔（2026-10-07）

只读读取原5worker的30分钟/60秒聚合资源统计及原Task最近100个原始带时间日志。5worker均有当前heartbeat/遥测，三recovery-off各30个GPU usage均值样本均为0；CPU统计有非零活动，但host/worker聚合不能证明各Task的计算PID或SQL提交进展，busy资源继续预留，不按GPU利用率回收。teacher1337/3407服务端记载的新科学日志距观察约32秒/74秒；recovery-off2027约12.425小时、1337约4.603小时、3407约0.810小时无新科学日志。3407最后日志为rank0自身12/12序列完成，不是所有rank/整Task完成。

teacher1337日志发送端比服务端记载时间快约93.9秒，原发送时间与出现负age的原观察均保留；新诊断明确使用服务端@timestamp，不把不同主机时钟当统一时钟。日志间隔和零GPU都不足以证明死锁或终态。只读进程检查直连34.18及经可信35.112均因SSH host key verification失败，34.26连接超时；均未执行远端命令，未修改主机信任。已向用户请求34.18可信SSH指纹/用户以继续只读核实；其他已授权工作和原teacher接续保持。

诊断：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-original-five-worker-resource-and-server-progress-gap-diagnosis-20261007T121503466918Z.json`。所有原任务/资源预留/caps和拒绝源码运输门槛保持，无停止、重启或增cap。完整A–K active；checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-original-worker-scientific-progress-gap-diagnosis-20261007T121542027909Z.json`；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T121542027909Z.json`。


## SparseCoop 当前作者训练权重发布与原始资产依赖核验（2026-10-07）

匿名只读获取作者GitHub README/releases/tags/current main及其完整recursive tree、HF方法名查询，共6个有2MiB响应上限/20秒timeout且无自动重试的metadata请求。当前main仍1f5741ccc62be8d8d5bedd296615b75b4c10fc7e，与已验源一致；README SHA仍06f9979e8da9d3d8ac8dcb825d3fa97b93c28933edbb235e3288e9c2afa4250c。完整164项tree未截断，未找到pth/pt/ckpt/safetensors/onnx候选；releases/tags第一页均为空，HF SparseCoop名称查询第一页为空。只对这些明确端点和当前时点下结论，不宣称全球没有权重。

AST只读抽取原配置：vehicle/INF num_gpus8,num_epochs800；coop8/48，精确load_from为work_dirs/V2X-Seq-SPD-2hz/vehicle-side/sparsecoop_v1-1_r50_544x960_bs6x8_800e/iter_23200.pth。协作原查询需要7个INF普通字段和4个train-denoising字段，列入缺资产合同。现有完整CPU初始化/锚点/ResNet50均不是trained SparseCoop；没有改8GPU配方、没有以单卡小配方替代，也没有启动新训练/前向。需要作者trained vehicle/INF/coop及适用query输出，或可准入的原8GPU完整训练路径。单机GPU继续35.112。

当前证据：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SparseCoop-current-author-checkpoint-publication-audit-20261007T121806098381Z/independent-current-author-checkpoint-and-exact-native-dependency-audit.json`；全部原HTTP响应size/SHA已入ledger。完整J及论文false；teacher原接续81770与C3407原60748当前ps仍live，固定期限不变。完整A–K active；checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-SparseCoop-current-author-train-asset-dependency-audit-20261007T122059119188Z.json`；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T122059119188Z.json`。


## 原五任务完整46序列rank进度下界与rank2/0002定位（2026-10-07）

原bootstrap明示每个完成序列会报告rank、completed_sequences、total_sequences及累计committed_events。只读日志观察器预先固定25pages/Task、500events/page、180秒总cap、64MiB总raw响应；首版events.get_task_events(log)后端API500 iter失败日志保留，v2只改为文档get_task_log时间重叠导航。不存在注册产物下载/源码运输/任务重启。实际扫描7/4/7/8/8页，去重2861/1379/2740/3088/3055个日志事件。所有完成报告的seq名、rank分配prefix及累计事件数都对应原SHA绑定的46序列/7445事件；时间分页边界同timestamp覆盖未独立证明，因此只给已见报告下界，不称全日志穷尽。

teacher1337至少14/46、teacher3407至少4/46；recovery-off2027至少40/46、1337至少35/46、3407至少35/46。三recovery-off所有非rank2的rank均已报完整自身prefix；rank2尚无完成报告，其首分配序列均0002。原日志明确context该序列最新事件分别358/452、349/452、337/452；3407对应rank2最后服务端日志2026-10-06T23:14:22.073Z，最近rank0完成0093不代表rank2仍进展。进度id的PID前缀分别200/624/624，仅为日志报告进程提示，非已验主机PID。接续只读检查具体三个原rank2进程/0002 live SQL commit状态，仍待可信SSH host指纹/用户；不停止/重启、不回收busyGPU或增cap。

诊断：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-original-five-whole-rank-sequence-log-observer-v2-20261007T122410355509Z/independent-specific-rank2-sequence0002-diagnosis.json`；全部生产器日志下界不等于完整注册输出/teacher目标/独立状态验收。A–K active、完整D/H/paper false。checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-original-five-whole-rank-prefix-and-rank2-diagnosis-20261007T122721868244Z.json`；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T122721868244Z.json`。脏工作树、原失败和原资源/期限保留，无提交推送。


## 两teacher本机观察器原30分钟期限终态（2026-10-07）

原固定截止12:28:57.021307UTC保持；actual terminal.json reason=fixed_local_observer_deadline_no_task_restart，completed为空，当前ps证实81770缺失。最后实际原Task检查00007仍两项in_progress、无artifact；无seed*/stage-started记录，采集/目标验收未启动。没有重启/延长本机观察器，也没有改动原teacher。以后只在原Task实时completed时直接调用已冻结完整采集/目标验收入口，不把观察器到期当原Task终态。

终态证明：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-original-two-teacher-local-observer-fixed-deadline-terminal-20261007T123009964795Z.json`；完整D/paper false，A–K active；checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-original-teacher-observer-fixed-deadline-terminal-20261007T123009964795Z.json`；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T123009964795Z.json`。


## V2V 冻结 train 概率校准的完整 official_test 诊断（2026-10-07）

原 train 已验三种子校准参数未变，未用 test 拟合或选择阈值。对完整9序列、1993配对帧/种子、3986源帧/种子、合计695319候选，按原严格XY<2m、raw-score稳定排序、每源帧一对一规则生成离线二元标签。所有源/ego相对姿态由组合与线性求解交叉核对；向量与标量标签完全一致，全部校准概率最大差1.6653345369377348e-16。首版先经过世界坐标造成的表达式差异未通过原容差，失败源码/日志/记录已保留；v2只改相对变换的计算顺序，原1e-10/1e-12容差、匹配规则及拟合参数未变。

两侧三种子均值：Brier 0.0860010554→0.0766759732，NLL 0.3049736157→0.2597893612，ECE 0.0915106701→0.0125950904。固定10个等宽bin，固定1337、10000次配对序列cluster bootstrap：校准减原始Brier95%CI[-0.0107285031,-0.0074646621]；NLL[-0.0541094245,-0.0360336091]；ECE[-0.0849125387,-0.0609924469]。三种子样本标准差、各侧及逐序列充分统计在完整receipt。独立审计重构每个候选的分组与bin统计，multinomial计数加权对照索引gather，全部9个区间最大差5.551115123125783e-17。

登记Task 2cb7c3393af049f2a0a625c58c80c308 completed，18成员完整云端逐字节回读。分析：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-V2V-frozen-train-full-test-calibration-relative-pose-v2-20261007T124241866822Z/independent-calibration-analysis.json`；统计审计：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-V2V-frozen-train-full-test-calibration-relative-pose-v2-20261007T124241866822Z/independent-bootstrap-and-partition-audit.json`；登记：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-qualified-V2V-full-test-calibration-stat-audit-evidence-20261007T124557273739Z/independent-readback-receipt.json`。

已知train/test内容重叠继续披露；这里只是native GT ROI下二元候选存在概率，不能视为独立泛化、联合身份后验TV或恢复质量证书，完整K/paper仍false。A–K范围不缩减；5个原Task20:44仍in_progress且无artifact，未重启/停止原任务，单机GPU仍限定35.112。checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-V2V-full-test-frozen-train-calibration-20261007T124721478702Z.json`；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T124721478702Z.json`。


## V2V 冻结校准可靠性图与稀疏 bin 完整边界（2026-10-07）

只读取已登记三种子完整offline充分统计，未读新GT、重跑模型或重新拟合。PNG/SVG/PDF包含两侧合计、vehicle侧、infrastructure侧三面板及原始/校准概率分布。三种子按候选描述性汇总，原10000次、1337、序列级同draw配对bootstrap保持，独立multinomial计数加权与索引聚合的全部bin频率区间完全相等。图示汇总口径不是前一表的种子均值，不能混用。

全60个侧×概率版本×bin的样本、正例、平均概率、经验频率、有效bootstrap次数均记录。vehicle/raw最高bin有89个候选，9955/10000次重采样非空；该bin频率区间条件于非空，45次空bin明确保留，未将空bin当零频率或隐藏。图脚注明确该条件及已知train/test内容重叠，不能视为独立泛化或联合身份后验校准。

图：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-V2V-frozen-calibration-reliability-figure-20261007T125131204503Z/calibration-reliability.png`（3150×2100 PNG已看图核验，无文字裁切）；SVG解析和PDF头通过，PDF未独立渲染，不标PDF视觉验收。绘图依赖在独立target安装，固定matplotlib3.9.4及10个wheel SHA，原NumPy1.26.4科学运行时未改。登记Task 6cd2e4377371408bb87f83b3607d9e80 completed、9成员/435724 archive bytes完整云端读回：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-qualified-V2V-reliability-figure-evidence-20261007T125254387702Z/independent-readback-receipt.json`。

20:52原5Task仍in_progress、无最终artifact；最新80条原日志显示两teacher有局部事件计数进展，不用于全Task完成率/ETA，recovery-off原rank2停留记录仍保留。没有停止/重启原任务或延长cap；单机GPU仍限定35.112。完整K/paper false，A–K active。checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-V2V-calibration-reliability-figure-20261007T125407579433Z.json`；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T125407579433Z.json`。


## SPD 完整逐侧 seen-val GT 的校准诊断前置验收（2026-10-07）

V2 seen-val缓存每帧900个原始query，原train校准仅覆盖raw-score>=0.05/all-class-top64候选；不能将全query概率或合并car跟踪GT当同规则校准诊断。只读35.112原validation转换manifest及INF原PKL，复用本机已验vehicle原PKL，按原SHA分别423ae092...和bbb18d07...恢复；原manifest b5b13da5...声明无删帧、无插值、未改raw标签。full-train转换目录的val PKL实际76bytes空集已核对，未拿来替代val GT。

独立restricted-pickle/原始JSON审计完整3748 vehicle帧、3441 INF帧、21序列、107331对象。每个label token、source-local ID、原中心、尺寸、yaw及作者coarse类别mapping均与原始封存JSON一致，原1e-8/1e-10容差未变，最大box差0。vehicle car/bicycle/pedestrian分别35414/9697/2869；INF分别43048/12824/3479。三种子完整V2缓存逐帧side/sequence/frame/box timestamp完全匹配，GT仅离线独立产物，不进入推理缓存，未重新拟合任何校准参数。

登记Task 592144f55f1a457ab21d721d425beeb9 completed，11成员/43833788 archive bytes原PKL、manifest、完整offline GT及源码/验收全部云端逐字节回读。独立GT验收：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-original-side-val-GT-independent-audit-20261007T130311374272Z/independent-side-GT-admission.json`；登记：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-qualified-SPD-original-side-val-GT-evidence-20261007T130458021918Z/independent-readback-receipt.json`。此项只完成完整GT前置资格，实际SPD二元概率诊断/统计尚待执行，seen-val仍探索性，不能声称独立泛化、联合身份后验或完整K/paper。

21:06原5Task仍in_progress、无最终artifact。单机GPU仍在35.112；没有停止/重启任务、运输被拒绝源码或延长cap。A–K保持active；checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-SPD-full-original-side-val-GT-calibration-prerequisite-20261007T130747698569Z.json`；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T130747698569Z.json`。


## SPD 完整冻结 train 校准诊断及当前 plan 完成度（2026-10-07）

此前逐侧GT前置资格之后，三种子完整21序列、7189源帧/种子的实际校准诊断已完成。读取19410300个原始query，严格复用raw-score>=0.05/all-class-top64、稳定原query顺序与原转换GT顺序、同类别源侧XY<2m一对一匹配；共486409个选定候选、140823个正例。参数未变，val未用于重新拟合或选择。该7189源帧校准队列与3316帧跟踪决策队列不同，不能混作G/H完整回放。

12个侧×类别分组的三种子均值、样本标准差(ddof=1)和固定1337的10000次序列级配对cluster bootstrap全部通过独立重构审计；最大标量概率差3.3306690738754696e-16，最大区间差1.0408340855860843e-16。两侧全类别均值Brier 0.134062→0.119628，NLL 0.430023→0.387660，ECE 0.093794→0.016690。INF侧bicycle三项95%区间全部跨0，不能宣称该分组稳定改善。此项仍为已见val探索性候选存在概率，不能代替联合身份后验、恢复质量证书或独立泛化。

登记Task 65f4c812fe1d4eb6a1a1fce85adc7679 completed；15成员/9199386 archive bytes完整云端回读，源码、完整候选NPZ/索引JSONL、诊断与统计审计均已封存。完整表：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-full-frozen-train-calibration-diagnostics-20261007T131148921445Z/calibration-diagnostics.md`；回读：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-qualified-SPD-full-score-calibration-evidence-20261007T131534944520Z/independent-readback-receipt.json`。

21:22北京时间实时核对，原2个teacher及3个recovery-off均in_progress、无最终artifact；元数据心跳不代表完整科学输出通过。A–K工作包均尚未证明整包完成，B输入/数值组件已完成；主要缺口仍为C最终模型、D完整teacher/优先级/learned回放、G/H两数据集主表与消融、I同资源至少3次重复，以及J剩余SparseCoop/Long-SCOPE和K完整论文表图。2027原rank5损坏、被拒绝的源码运输门槛、历史失败与原cap保留；不将不同粒度计数相加为论文完成率。单机GPU仅35.112，没有重启任务、延长cap或提交推送。

A–K保持active，full K/paper false。当前checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-SPD-full-calibration-and-user-plan-audit-20261007T132251708750Z.json`；完整度审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T132251708750Z.json`。


## SPD 四类可靠性图、240 bin 独立候选审计与稀疏边界（2026-10-07）

从已验三种子完整offline充分统计生成all-classes/car/bicycle/pedestrian四套PNG/SVG/PDF，各有两侧合计、vehicle、INF可靠性及候选分布。固定1337、10000次共享序列cluster bootstrap，每个bin独立multiplicity和index-gather区间完全一致；240个bin的样本、概率均值及正例频率额外从486409个原始offline候选、完整源侧帧索引和类别独立重算通过。未读新GT、未执行模型或重新拟合。

保留所有零候选bin的未定义区间、非空draw计数及稀疏边界。vehicle/pedestrian/calibrated bin5只有1个候选、6463/10000次非空；条件区间退化不代表估计精确或跨序列稳定。INF侧bicycle三项指标区间跨0仍保留。按候选描述性汇总与三种子均值指标表不同；seen-val继续探索性，无联合身份后验、恢复或资源结论。四张3150×2100 PNG逐图看图通过，无文字裁切；SVG解析有效；PDF仅头检查，未独立渲染。

登记Task 3661b36638174f258a8ae322b2c8da52 completed，29成员/1643569 archive bytes全部云端回读。图与稀疏边界：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-frozen-calibration-all-class-reliability-figures-20261007T132424303236Z/reliability-figures.md`；候选审计：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-frozen-calibration-all-class-reliability-figures-20261007T132424303236Z/independent-240-bin-candidate-audit.json`；登记：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-qualified-SPD-all-class-reliability-figure-evidence-20261007T132627690679Z/independent-readback-receipt.json`。原首个bin审计准备源在执行前核对帧索引字段后被v2替代，未执行旧源，二者保留封存。

21:26:59原5Task仍in_progress、无最终artifact；原C3407 PID60748 ps存活，等待前驱，未重启；冻结4/8卡物理交集准入eligible_choices=[]。未延长cap，单机GPU仍仅35.112。完整A–K保持active，K/paper false。checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-SPD-four-calibration-reliability-figures-20261007T132746327314Z.json`；完整度审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T132746327314Z.json`。


## B 原始要求边界及新增复查上限、C3407前驱核验（2026-10-07）

原plan B要求训练内运动选择、初始先验、完整9×9协方差以及train/official_test合同与9D缓存；跟踪表验收属于G/K，不增加为B输入完成门槛。六套原运动/缓存12个receipt SHA均当前一致，原数值验收保留。新增全部依赖逐文件复查在读取payload前发现218500条清单超过固定200000文件上限，失败记录保留，未增大cap或自动重试；不能将该新增完整SHA复查声称通过。失败：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-B-original-requirement-completion-audit-20261007T133558454472Z/preexecution-inventory-cap-failure-preserved.json`。

C3407 PID60748当前ps存活，原progress仍等待未生成Task ID的seed2027前驱；冻结物理4/8卡准入eligible_choices=[]。保留10小时原上限，不重启或替换原模型训练路径。核验：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-C3407-current-bound-predecessor-observation-20261007T133757697865Z.json`。完整A–K未完成，goal active；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T133757697865Z.json`。


## 五份原校准图PDF的完整独立渲染核验（2026-10-07）

SPD all-classes/car/bicycle/pedestrian及V2V all-classes共5份原PDF，每份1页，原SHA均保持。通过系统Poppler在150DPI实际渲染完整页面并逐页看图：标题、轴标签、图例、统计条件和已见val/内容重叠边界完整，未观察到缺字、裁切或重叠。此前只有PDF头检查的资格被本次完整PDF视觉核验补齐；原历史receipt不改写。渲染核验不新增统计、模型前向或研究结论。

登记Task 35c6d3017d994e65a37696c2c552e466 completed，15成员/955411 archive bytes（原PDF、渲染PNG、完整核验及源码）全云端回读。核验：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-five-calibration-PDF-independent-render-20261007T133906971393Z/independent-visual-review.json`；登记：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-qualified-five-calibration-PDF-visual-review-20261007T134006798675Z/independent-readback-receipt.json`。原5Task最新仍in_progress且无最终artifact；A–K仍active，完整K/paper false。checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-five-calibration-PDF-full-render-review-20261007T134112330254Z.json`；完整度审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T134112330254Z.json`。


## 已验两种子完整主森林的模型输出类别恢复普查（2026-10-07）

读取seed1337/3407各46原序列、7445原train事件的完整独立结构/因果/质量算术/新鲜状态验收。809/728条前驱恢复证明分别对应726/653个唯一原事件、786/704个输出分量决策；合并多前驱额外条目23/24，不能将1537条证明称1537个独立恢复事件。此前已发现9/12条，此前未展开800/716条，合并投影74/77条，仅过期输出0/0条。capacity-undecided分量决策1541/1502仍保留，不是事件数，也未证明完整后验最优性。

全部1537行与原已验oracle记录逐字段一致、行SHA有效，逐条绑定原event ID、源帧、参考/决策时钟、最终checkpoint SHA、SQLite SHA和原序列验收SHA；独立内存SQL聚合重建92个逐序列统计和两种子总计。原大型SQL归档未重新读取或重跑，未读取GT。字典序首条的各discovered/merge组实例可追溯，不按质量挑选。恢复这里只指当前模型输出类别的前驱投影此前位于唯一残差区域、且不属于此前保留/输出类别；物理身份成功率留空、实际质量提升与同资源收益未证明。seed2027原rank5缺失不补造，三种子均值/样本标准差/bootstrap不报告；train轨迹不替代G/H正式test表。

登记Task fabfd27bc2b74170817c9edc716e1b15 completed，110成员/2098693 archive bytes完整云端回读，包含全部92原序列receipt及1537条记录、原主验收与事件、分析源码和独立聚合。表：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-qualified-two-original-main-model-recovery-census-20261007T135230095464Z/model-recovery-census.md`；独立聚合：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-qualified-two-original-main-model-recovery-census-20261007T135230095464Z/independent-complete-recovery-census-audit.json`；登记：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-qualified-two-main-recovery-census-evidence-20261007T135449232973Z/independent-readback-receipt.json`。原5Task最新仍in_progress、无artifact；A–K保持active，完整K/paper false。checkpoint：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-complete-two-main-model-recovery-census-20261007T135623601083Z.json`；审计：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T135623601083Z.json`。


### 2026-10-07 两原始主森林恢复类别与当前决策范围完整核验

seed1337/3407 全部92个原SQLite逐字节SHA核对，共79,755,673,600 bytes。覆盖全部1537条恢复前驱来源记录；先前整事件扫描触及原1GiB/序列JSON上限的失败保留。新读取方式仅定位全部恢复事件及紧邻前序事件，共2386份原完整记录，最大421,997,821 bytes/序列，原600秒、80GiB、1GiB/序列和4096范围节点上限未扩大。全部所需旧组件确在紧邻前序事件，主核验406秒完成。

当前共享decision_indices上的身份根分配：1337有716/809、3407有621/728条不同于此前所有active及output类别。忽略根编号后的节点间身份分组：分别632和562条仍不同，合计1194条；其余343条不满足这一更严格条件。记录数包含合并前驱，不能当作1537次独立恢复。独立递归SQL重新构造全部原prefix身份根，并核对源摘要、成员范围、所有旧候选及分组；SQL/字典计数一致。

完整新证据108成员/1,503,509 bytes已注册并整包云端逐字节读回，Task 13594ba7c94d443492966dccd373bfcd completed。原大SQLite不重复装入新证据包；所有输入SQL SHA及92序列原证明均保留。这证明原模型所选恢复类别在当前范围上存在身份分配/分组差异，不证明恢复关闭反事实、物理身份正确性、总体质量收益或同资源优势；缺2027不补值，A–K全目标仍active。

独立核验：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-all1537-original-recovery-targeted-current-scope-v2-20261007T141008207513Z/independent-all1537-scope-audit.json

云端读回：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-all1537-original-current-decision-scope-evidence-20261007T142015368967Z/independent-readback-receipt.json


### 2026-10-07 原恢复决策风险诊断与后续资源限制

全部1537条恢复前驱去重为1490个当前组件决策（1337:786、3407:704）。直接从92个原SQL的active权重与prefix根重算保留后验下身份Hamming风险、η及已报告风险界，冻结绝对容差1e-10，104.6秒完成；不重训、不重放、不读取GT。另以完整源键和SQL聚合独立核对全部记录及阈值计数/中位数。917/1490个风险界≥1−1e−12，805个恰为1；18个动作搜索未完成，48个选择落在当前active之外。所有numeric_certificate=false，当前风险诊断不是形式质量证书、真实身份风险或恢复关闭因果收益。两种子中位风险界约1，不补缺失2027。完整14成员/1,389,252 bytes已注册并整包云端读回，Task cb8b9d8dcb6d4e23949c52696607b37f completed。

用户后续资源指令：多卡GPU只用A100；CPU科学实验使用L40；单卡GPU使用10.100.35.112。原4/8卡等科学配方不改成单卡。公共GPU准入要求队列和候选Worker均标明A100，并保留全部Worker物理交集、pending任务、90秒心跳、卡数与禁用队列检查；10项软件准入检查通过。原C3407控制器的派发源码指针绑定新冻结A100门槛，原PID/源码/10小时期限保持，不停止或移动现有实验。后续CPU科学实验必须在L40机器且禁用CUDA；本地源码准备、调度与证据封存是控制面。当前无合格空闲A100；teacher下界14/46、5/46，recovery-off40/46、35/46、35/46，均非整体独立验收。

风险独立证明：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-all-original-recovery-conditional-risk-and-eta-independent-20261007T142735894346Z/independent-recovery-risk-bound-diagnostics.json

资源策略：/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-user-A100-L40-112-resource-policy-20261007T143311211570Z/resource-policy.json


### 2026-10-07 L40完整恢复风险分解与独立10000次序列bootstrap

CPU科学分析及独立审计实际在10.100.35.121-L40S运行，CUDA禁用；Task 6192b40de1d6442ea7565b43ef8e8549、918d89a0a1274f9289d266dd389d4cb4均completed。覆盖全部1490个原当前决策、1537条前驱和92个种子/序列范围，固定1337生成10000次共享46序列cluster bootstrap；独立SQL重算全部序列充分统计、240000个指标值及24个95%区间，最大绝对差7.105427357601002e-15，原容差1e-10未放宽。

917个近单位已报告风险界中913个已有近单位η；仅11/1490个动作优化gap超过1e-10。风险界>0.1但gap≤容差的比例：1337为90.97%（95% CI 88.20–93.50%），3407为94.03%（91.76–96.13%）。当前浮点模型界的主要诊断限制是遗漏质量η。此结论没有GT、正式数值证书、恢复关闭因果收益或同资源物理质量证明；缺2027不补造，不报告三种子均值/标准差。

两个Task全部6个注册产物、11个归档成员、4723586 bytes整包云端读回，size/SHA全部一致。本地仅源码准备/调度/封存/读回控制面。原5GPU Tasks在23:05北京时间仍in_progress且无最终artifact，原C3407 PID60748当时存活9:57:41，原10小时上限未延长；无空闲合格A100。此前完整rank确认下界保持teacher14/46、5/46，recovery-off40/46、35/46、35/46，本次仅刷新Task元数据，没有新的科学序列进度结论。

云端整包独立读回：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-L40-complete1490-risk-decomposition-two-Task-whole-readback-20261007T150007127433Z/independent-two-Task-readback-receipt.json

全A–K checkpoint：/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-L40-complete1490-risk-decomposition-20261007T150645434773Z.json

完成度审计：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T150645434773Z.json


### 2026-10-07 23:08 原teacher进展与C3407原期限终止

按原冻结25页×500条/Task、180秒、64MiB上限完整读取原5Task的rank序列前缀日志。teacher1337的rank0/序列0018完成报告对应原分配与344累计事件，服务器时间2026-10-07T15:04:22.157Z；整体完成下界从14/46增至15/46。teacher3407仍5/46，recovery-off2027/1337/3407分别40/46、35/46、35/46。5Task全部in_progress且无最终注册artifact；每Task仍有1个未知rank，不将未见日志当作零进展或全部日志覆盖，也不以producer完成报告替代独立验收。

原C3407 PID60748已确认退出。原日志明确在冻结source第16行time.monotonic()-start<36000断言失败；第121次原检查于15:08:33.230471UTC仍无2027 Task，未创建3407训练Task。保留原日志/检查/source SHA，未重启、延长期限或绕过资格。多卡A100、CPU科学L40、单卡112继续适用；现有5GPU回放保持原运行。全A–K目标仍active。

期限终止证明：/Volumes/Data/test/recover-before-fuse/receipts/rbf-C3407-original10h-deadline-terminal-20261007T150947581618Z.json

完整rank观察：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-original-five-full-rank-after-L40-bootstrap-20261007T150757235220Z/independent-all5-original-rank-sequence-prefix-log-observation.json

全计划完成度审计：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T150947581618Z.json


### 2026-10-07 未来多卡A100调度源码补齐

核查正式CPU基线后，其匹配完整输入/checkpoint与原Linux运行资格仍未齐备，MHT单独授权门槛保持。源码准备补齐9个多卡调度入口的A100参数/日志，修复共享active_reservations和snapshot对Worker task=null的处理。公共available门槛现在拒绝订阅A100队列的任何非A100 Worker，即使该Worker当前忙碌，避免任务等待期间被其他型号接走。保留所有型号的物理卡别名占用、原live任务预留、pending、90秒心跳、禁用队列和4/8卡检查。

预先封存before/after源码、14项独立路由情景、null快照及活/终态预留检查均通过，scoped git diff --check通过。仅未来调度源码准备；具体Task仍需重新匹配冻结完整调用依赖，不改原不可变freeze/运行Task/已退出控制器、不改训练与模型、不新建Task或运输被拒包。23:12北京时间原5Task均in_progress且无artifact，无合格空闲A100，原完整rank已验日志下界15/46、5/46、40/46、35/46、35/46保持（本轮仅更新metadata）。不将软件检查计入论文完成率。

源码与控制面核验：/Volumes/Data/test/recover-before-fuse/receipts/rbf-future-A100-policy-closure-preparation-20261007T151404414966Z.json

全计划完成度审计：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T151404414966Z.json


### 2026-10-07 SPD当前最终模型与旧CPU资源入口合同差异

三种子当前final-refit checkpoint及训练/full-forward/独立数值receipt SHA全匹配，全部21项评分源码SHA与当前模型一致。旧resource_sweep.validate_spec要求full_official_train布尔字段，但当前已验refit checkpoint没有该字段；不得改checkpoint或因字段缺失要求重复已验训练。旧schema只允许一个cache，而当前三种子的detector/calibration upstream分别不同，不能将某一个seed缓存混用到其他模型。完整资格应按seed绑定当前独立训练与forward证据、各自完整val cache/events，方法间保持同seed相同输入；不能套用旧单缓存合同或旧MHT checkpoint。三份val cache/events/input-binding原路径和SHA已绑定新完整checkpoint，仍仅metadata/源码准备，未读取GT或执行新CPU科学实验。

完整当前CPU JPDA/PKF replay仍需匹配来源、L40 runtime数值资格、预先冻结全部原工作上限；正式I仍需train选择、隔离环境、预热及每配置至少3次资源测量。MHT和被拒Linux priority/CUDA包门槛保持。原5Task在23:17均in_progress无最终artifact；完整A–K仍active。

差异核验：/Volumes/Data/test/recover-before-fuse/receipts/rbf-current-SPD-final-model-resource-entry-contract-gap-20261007T151715212210Z.json

全计划审计：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T151812260164Z.json


### 2026-10-07 当前最终refit按种子资源入口v2源码实现

旧rbf_cpu_resource_sweep_v1单缓存/full_official_train门槛保留。新增rbf_current_final_refit_cpu_resource_sweep_v2，显式按1337/2027/3407绑定各自完整val缓存、schedule、checkpoint和原训练/forward/独立numeric/plan证明；所有方法在同seed共用同一输入，逐配置需完整三种子，每配置资源重复至少3次。当前模型无需改checkpoint字段：独立核对完整full-train配方、冻结epoch选择、46序列全行覆盖、训练模型/权重/产物SHA及原1e-4数值准入。worker再次验证证明、源和输入。新CPU路径在加载模型前要求Linux、CUDA_VISIBLE_DEVICES空串和实际ClearML last_worker为10.100.35.121-L40S。

8项纯metadata/调度软件测试通过：原v1门槛、同seed跨方法输入绑定、跨seed缓存/模型拒绝、缺失proof、完整行与失败计数、数值容差与fixture/test选择拒绝、独立priority资格及L40放置要求。软件首次tmp祖先symlink失败保留并修正fixture为resolved临时目录，生产symlink门槛不放宽；旧入口原源码从HEAD重读并核对上一轮记录SHA后封存。实际三种子原证明metadata通过，原全数值行229212/225177/223355保持。测试不执行Torch/模型/回放，新的L40运行时数值、完整科学replay、同资源结果仍未取得。原拒绝包/MHT门槛、失败和dirty tree保留，未transport/新建Task/commit/push；全A–K active。

源码与软件/原证明准入：/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-current-final-refit-resource-contract-v2-20261007T152431475784Z/source-preparation-and-software-admission.json

全计划审计：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T152850174498Z.json


### 2026-10-07 当前CPU资源路径完整本地输入与既有源码门槛核验

当前三种子final-refit checkpoint及weights按原size/SHA本地物化为标准checkpoint.json/weights.pt目录，无模型重新保存或metadata改写。各seed仍绑定各自val缓存与原完整输入证明。此前森林events.json是恢复事件envelope，不能直接传给旧schedule_rows；改为绑定既有原schedule.json完整字节，并以原冻结parser核对全部3316帧/21序列、split、源帧唯一性、时钟次序与无GT字段，保留空输入，不做schedule缩减或时间协议变更。

resource_sweep.source_snapshot所需全部819源码文件本地按原相对路径和SHA封存。源码清单与此前被拒Linux优先级消费者有8个同路径文件，其中6个字节完全相同；仅软件/metadata准备，不把整个依赖自动当作已准入的干净CPU包。原自动审批拒绝及待明确授权保持，未发送或改运输方式、未创建L40 Task，实际Linux数值/完整回放和同资源结论仍未取得。23:32北京时间原5Task均in_progress且无最终artifact；此前原C3407期限终态保持，不重启或延长。全A–K目标active。

本地完整输入/门槛证明：/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-current-final-refit-CPU-full-input-and-source-gate-20261007T153150173034Z/input-preparation-and-retained-gate-audit.json

819源码冻结与原任务观察：/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-current-final-refit-CPU-full-input-and-source-gate-20261007T153150173034Z/immutable-source-freeze-and-original-five-observation.json

3316/21原schedule parser准入：/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-current-final-refit-CPU-full-input-and-source-gate-20261007T153150173034Z/original-full-val-schedule-parser-admission.json

全计划审计：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T153432824897Z.json


### 2026-10-07 23:35 原teacher新增前缀与5个真实Worker绑定

原冻结全rank日志observer/source SHA与25页×500条/Task、180秒、64MiB上限保持。teacher1337确认完成下界从15/46升至16/46；teacher3407仍5/46，recovery-off2027/1337/3407仍40/46、35/46、35/46。完整输出尚无artifact，当前扫描未见coupled_failure_before_receipt_upload；这不证明所有日志覆盖或所有失败不存在，每Task仍1个未知rank。全部5Task实际Worker.task ID匹配且心跳年龄<90秒，确认live，不把心跳当作独立科学完成或重启依据。当前无合格空闲A100，C原期限终态不重启，CPU源码/MHT原门槛保持。

具体live绑定与进展：/Volumes/Data/test/recover-before-fuse/receipts/rbf-original-five-actual-worker-live-and-new-teacher-prefix-20261007T153649398208Z.json

全计划审计：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T153649398208Z.json


### 2026-10-07 Long-SCOPE作者资产与原生论文合同

再次读取作者wang-jh18-SVM公开owned repository API，完整单页56仓库未发现Long-SCOPE候选；原始API与arXiv v1 HTML/abstract共578778 bytes保留，90秒/每响应2MiB/总6MiB上限固定，无自动重试，不推断其他位置没有资产。原生code/checkpoint仍未取得，不填论文AP/AMOTA/FPS作为实验产物。

原论文4.1/4.2当前核对：V2X-Seq car范围扩至150米（精确ROI坐标/shape与timestamp频率未推断）；ResNet50 ImageNet，900固定queries加最多40 GQG动态query，CAA4层，48 epoch单端+24 epoch协作，batch16/AdamW/lr2e-4/weight_decay0.01/cosine，图像960×540。原论文4卡3090；后续依用户要求只能4卡A100，不将hardware变化或论文FPS当作同资源结果，也不把本论文48/24配方替换已冻结SparseCoop作者8卡原生配方。原论文来源https://arxiv.org/html/2604.09206v1；官方CVF页面本次工具读取报Internal Error保留观察，不能声称成功刷新其页面字节。仅完整原生复现要求准备，未实现/训练/运行Long-SCOPE，fullJ/论文仍false，完整A–K active。

原生合同：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-Long-SCOPE-current-primary-native-contract-20261007T154248261191Z/primary-paper-native-contract.json

作者资产刷新与范围：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-Long-SCOPE-current-primary-native-contract-20261007T154248261191Z/asset-and-native-contract-receipt.json

全计划审计：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T154530198990Z.json


### 2026-10-08 原始 B 要求收口、完整证据表与 SparseCoop 原配方

B 按目标文件的原始输入、运动与缓存范围完成：训练内选择、因果速度、首观测/未匹配人口先验、完整 9×9 残差协方差及六个 train/official_test split×seed 运动合同和 9D 缓存均有既有独立验收。当前又核对 50 个源码、模型、manifest 和验收文件 SHA；合计 54,588 个源帧、3,060,073 个候选。追加全依赖重哈希的 200,000 文件 cap 失败仍保留，未宣称重哈希通过，也未增加 cap。跟踪指标与论文资格继续属于 G/K。原始 B 收口审计：[rbf-B-original-requirements-complete-existing-independent-evidence-20261007T165207430405Z.json](/Volumes/Data/test/recover-before-fuse/receipts/rbf-B-original-requirements-complete-existing-independent-evidence-20261007T165207430405Z.json)。

K 的概率诊断依原合同分别报告 NLL/Brier/ECE 和质量界数值验证，不额外要求完整联合 TV 校准证明。已有检测存在分数诊断不是联合后验误差证明；全部方法、序列和种子的适用诊断与恢复证据仍需完成。

三张新增证据表均只转录已验原值，没有重新计算科学统计：

- [stage1-spd-seen-val-evidence-table-20261008.md](/Users/lbin/transvision/docs/recover-before-fuse/stage1-spd-seen-val-evidence-table-20261008.md)
- [native-public-method-evidence-table-20261008.md](/Users/lbin/transvision/docs/recover-before-fuse/native-public-method-evidence-table-20261008.md)
- [complete1490-recovery-risk-evidence-table-20261008.md](/Users/lbin/transvision/docs/recover-before-fuse/complete1490-recovery-risk-evidence-table-20261008.md)

恢复风险表保留两种子、全部 1,490 决策、12 项诊断与 24 个区间；72 个展示数值及 24 个实际 Markdown 单元格逐格回读。缺失 seed2027 未填补，遗漏质量较大、界较松的结果保留，不作真实恢复成功率或同资源优势主张。

SparseCoop 原作者 8 卡三阶段配方封存：车端 batch48/800epoch/23,200iter；路侧 batch48/800epoch/24,800iter；协作 batch32/48epoch/2,208iter。完整 7 个配置/启动文件绑定 SHA；协作 load_from 精确对应车端 iter23200，并要求路侧完整 train-denoising/val 查询。未来多卡仅 A100；完整输入、实际 A100 原生环境和物理空闲卡组尚须准入，未创建/提交训练或执行查询导出。作者 local_test.sh 硬编码 GPU1，不能直接用于 112 的单卡任务。封存配方：[original-three-phase-A100-native-training-recipe.json](/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-SparseCoop-original-three-phase-A100-training-recipe-20261007T170023527280Z/original-three-phase-A100-native-training-recipe.json)。

第二阶段合同 3.3 明确要求学习式优先级对照、普通 MHT 多 K 成本曲线及峰值内存/尾延迟；这些证据仍未齐备，F 保持未完成，有限理论测试不替代整包验收或第三阶段研究结论。原 5 个 teacher/recovery-off Task 在 2026-10-07T17:03:40 UTC 仍 in_progress、有具体 Worker 绑定、无产物；心跳不证明科学计算正常。保留 2027 原 rank5 坏资产、各源码运输与 MHT 授权门槛、原控制器终态及脏树。

当前完整 A–K 审计：[rbf-plan-completion-audit-20261007T170104116587Z.json](/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-audit-20261007T170104116587Z.json)。仅 B 完成；整体目标 active，未自动提交或推送。


### 2026-10-08 DMSTrack 原生成本计时软件准备

已新增 `tools/event_track_v2x/native_cost_timing.py`，对资格 SHA 为 `4a78f056d364795091e3a2b4f609a2b07a874874ff428aef1e56dde6ff904da6` 的作者入口，在内存中为完整逐帧循环及 tracking 核心调用增加显式 CUDA 同步计时。作者文件保持原字节；完整逐帧范围含特征读取、原生诊断与输出写入。序列初始化和作者评价须另报整入口耗时，不能混入 tracking 吞吐。

3 项软件检查通过：合成回放返回值与帧覆盖、异常保持失败、SHA 和歧义调用拒绝。实际作者源码编译及去除计时作用域后的 AST 回读一致；这不证明 GPU 数值或预测一致性。正式执行仍需全集预热、至少 3 次完整重复、1993 帧/9 序列覆盖、预测独立回读及 CPU/GPU 总驻留内存测量。当前尚无成本 controller、实际计时运行或同资源结果。

35.112 当前磁盘可用 9452285952 字节，小于原定 8 GiB 保留空间与 2 GiB 输出上限之和；未删除原产物或放宽门槛。

冻结准备：`/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-DMSTrack-full-native-cost-instrumentation-20261007T182637063469Z`。验收边界收据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-DMSTrack-native-cost-source-preparation-20261007T182747478320Z.json`。


### 2026-10-08 DMSTrack 原生预热与单次测量入口准备

新增 `tools/event_track_v2x/run_DMSTrack_native_cost_entry.py`：每组在同一进程中先完成作者原入口的完整预热，再重新初始化模型/跟踪状态完成测量；每阶段输出独立保留。原入口 CUDA/权重资格代码复用并保持 SHA。原入口每阶段 smoke 后会清空 allocator cache，合同明确保留该行为，不能声称 CUDA 分配缓存已预热。整入口墙钟包含原资格检查与作者评价；逐帧和 tracking 核心耗时另外记录。

5 项软件检查通过，含合成 1993 帧 warmup/measured 集成与异常后的 `__main__`、argv 状态恢复；第一次 fixture 使用保留计时变量名被拒绝的失败仍保留。作者原源码 AST 结构回读与 Python 3.7 语法解析通过，但实际原 Python 3.7/CUDA 运行、预测一致性、9 序列唯一帧覆盖、CPU/GPU 总驻留内存与独立正式统计尚未验收。外部准入 controller 与实时驻留内存监控尚未创建，也没有上传或启动新实验。

三组完整重复共保留 6 个阶段。每阶段沿用原 2 GiB 输出上限；若全部产物在 35.112 留存，预留容量为 6×2 GiB 加原 8 GiB 主机保留空间，共 20 GiB。此前单阶段 10 GiB 准入检查不能替代这个完整重复留存容量检查。

冻结入口与合同：`/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-DMSTrack-native-cost-warmup-pair-entry-20261007T183122948572Z`。


### 2026-10-08 DMSTrack 完整逐帧计时独立读回准备

新增 `verify_DMSTrack_native_cost_phase.py`，实际科学读回限 L40 CPU、禁用 CUDA。绑定原已验 native manifest SHA `fb40392484930db53f55ba8fa355facfb5da680514e42739dd62cbe608dc65bc`，要求 5990 个原生输出成员及 1993 个唯一帧键/9 序列完整集合。计时行总数不能替代唯一帧键覆盖。非 PDF 科学输出逐字节 SHA 一致；PDF 全字节读取并留存，但不以创建时间等 PDF 字节差异判断预测一致性。所有失败行、重复/漏帧、非有限或负延迟、核心耗时超过帧耗时、原生输出变化均拒绝。完整输入/运行时资格、总驻留内存及三组成本重复仍为另外必须满足的门槛。

5 项软件检查通过，实际科学读回尚未执行、L40 Task 尚未创建。本次 35.112 文件系统观察未发现可提供足够空间的其他持久卷；未使用 tmpfs 替代磁盘保留空间门槛。

冻结完整准备：`/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-DMSTrack-native-cost-full-phase-readback-20261007T183511529097Z`。收据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-DMSTrack-native-cost-independent-phase-readback-preparation-20261007T183511529097Z.json`。


### 2026-10-08 DMSTrack 原生成本内存门槛软件准备

新增 `native_cost_resource_guard.py`，分别记录 Linux cgroup memory.current/memory.peak、当前 memory.stat 分类及原进程 VmRSS/VmHWM；不能把当前分类当作峰值分类，不能把 cgroup 文件页记账当作进程 RSS。原 6 GiB CPU 硬上限必须匹配；OOM、hard-limit 事件和 RSS/cgroup 超限均拒绝。nvidia-smi 解析记录含 CUDA context 的进程总驻留内存；明确为采样观察，不声称精确瞬时 GPU 峰值。其他用户出现在同物理 GPU 时属于隔离违规，未来 controller 必须只停止本次拥有的容器。

原预热/测量入口已在每阶段前后接入 CPU cgroup 和 RSS 门槛。9 项软件检查通过；35.112 当前用户会话 cgroup v2 memory 接口实读通过。实际成本容器接口、外部 host 采样循环、只停止自有容器的 controller 及实际计时/独立验收尚未完成。没有上传或启动成本任务。

冻结准备：`/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-DMSTrack-native-cost-total-memory-guard-20261007T184151718909Z`。收据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-DMSTrack-native-cost-total-memory-guard-preparation-20261007T184151718909Z.json`。


### 2026-10-08 DMSTrack 新独立 controller 源码准备

新增 `run_owned_DMSTrack_cost_pair_host.py` 及三组独立绑定；只涉及五个明确 allowlist 的公开原生代码文件，原已终止 controller 不重启。运行前核对源、原模型、特征、Python、环境锁、9775 个 runtime 文件与 adapter 的完整字节。原 CPU/GPU 6 GiB、阶段输出 2 GiB、主机保留 8 GiB 与单阶段 21600 秒时限保留。Docker 私有 cgroup namespace 固定。逐秒采样记录实际时间戳，检查总 RSS/cgroup 内存、GPU 总驻留内存及外来 GPU 进程；采样不等同精确瞬时 GPU 峰值。异常或外来进程出现时，仅按本次 Docker --cidfile 创建的精确 CID 停止自有容器，保留失败及部分产物，不重试。

每组控制日志上限 256 MiB，且必须计入该组两阶段合计 4 GiB 原输出预算。逐组剩余留存准入分别 20/16/12 GiB，避免把首组留存占用再重复要求成额外 20 GiB。第一组 20 GiB 全合同当前仍不满足。

完整 17 项软件检查通过，包含磁盘不足时不创建进程，以及隔离违规时只停止本次 CID。实际源码上传、GPU 成本运行、实际容器 namespace/监控资格和 L40 全集独立读回仍未执行。

冻结 source/controller/绑定：`/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-DMSTrack-native-cost-new-owned-controller-20261007T185850381958Z`。收据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-DMSTrack-native-cost-new-owned-controller-preparation-20261007T185850381958Z.json`。


### 2026-10-08 DMSTrack 远端磁盘准入探针实读

已将严格五文件 allowlist 的公开原生成本源码及单个 8.55 MB 元数据绑定送至 35.112 新的独立目录，固定上传上限 32 MiB；不含之前被拒绝的 RBF runtime 包。源码包有额外文件即拒绝，host controller 禁止自有 import 生成 bytecode，以保持五文件集合。远端 Docker run help 实读确认 cgroupns/cidfile/gpus/memory-swap/read-only 参数支持。

本次执行模式固定 disk_admission_probe_only=True，即使磁盘通过也不创建 native 进程；不属于 GPU 成本运行或 CPU 科学实验。实际 controller exit=1，原因是可用 9426120704 字节低于首组 20 GiB 留存/保留空间要求。完整小字节读回证明五个源文件和绑定 SHA 一致、pair 目录为空、该精确容器名称无 Docker 对象，control 目录仅有保留的失败收据。未读/运算原生模型或科学输入，未放宽任何门槛。

成本容器真实 runtime/cgroup/监控资格、GPU 全集重复和 L40 独立读回仍未完成；不能以此磁盘拒绝路径代替正式成本验收。源码：`/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-DMSTrack-native-cost-owned-preexecution-probe-final-20261007T190648287872Z`。实际远端收据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-native-cost-112-actual-disk-only-preexecution-rejection-20261007T190842460250Z.json`。


## 2026-10-08 04:07 CST — DMSTrack 原生成本既定资源限制补齐

原生成本控制器补齐原绑定的 2 GiB 空闲显存预留；完整输入 SHA 读回后、创建容器前再次检查物理 GPU 占用；运行时继续检查预留。每阶段必须至少具有自身 GPU 进程采样，避免只有 CPU 或初始化观测仍写返回凭证。进程退出后及最终元数据写入后复核阶段/总输出、控制日志容量和 8 GiB 主机磁盘保留。均沿用原有容量限制；采样峰值仍不是精确硬件高水位。

21 项软件检查通过，Python 3.7 语法解析通过。初始测试夹具提前创建输出被入口空目录检查拒绝，已保留失败记录并修正夹具；未放宽入口限制。原生 GPU 成本实验和 L40 独立三次重复验收仍未执行。新五文件源码包仅本地封存，旧远端探测源码保留。证据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-native-cost-existing-resource-contract-closure-20261007T200601664581Z.json`；源码：`/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-native-cost-existing-resource-contract-closure-20261007T200601664581Z`。

112 可回收缓存候选已收窄为四条 2026-07-27 非共享中间构建层，合计约 20.957 GB。逐个 exact-id 只读查询确认；已向用户请求删除这四条既有缓存的具体授权，回复前不清理、不启动依赖清理的成本运行。候选不是镜像、卷、原始输入或实验输出。清单：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-112-four-specific-obsolete-build-cache-review-20261007T200734698661Z.json`。


## 2026-10-08 04:15 CST — 原生成本完整帧时延与资源独立读回准备

独立读回器基于完整且逐字节匹配原生预测的帧队列计算 p50/p95/p99、最大延迟；分开报告 core、frame 和完整作者入口吞吐。core 吞吐不能当作端到端 FPS。零时长不会编造无穷吞吐。原始资源行新增当前活动阶段和空闲显存预留数值，读回器检查两阶段 GPU 采样、固定物理卡、PID、CPU/GPU 容量、OOM、时序和排他性；cgroup 与 RSS 分列，GPU 峰值仅为采样峰值。24 项软件检查和 Python 3.7 语法解析通过，新增数值 reserve mock 的夹具失败保留后修正。

证据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-native-cost-independent-latency-resource-readback-preparation-20261007T201632824323Z.json`。新五文件 GPU 包和独立读回器本地封存，三次重复准备绑定同步新源码 SHA。未上传新包、未清理缓存、未启动原生成本运行、未在本地执行真实实验统计；未来真实 CPU 读回仍须 L40。输入/runtime/完整期限与三次重复整体验收未完成，不能称正式成本或同资源比较通过。缓存清理问题仍待明确回复。


## 2026-10-08 04:24 CST — 112 原生 Python 私有 cgroup 实际元数据资格

在原镜像 85525aef…、UID/GID 1000、私有 cgroup、只读根目录、2 CPU、6 GiB memory/memory-swap 和固定 pids/tmpfs 下，实际标准库探针读取 memory.current/peak/max/events/stat、进程 RSS 与 monotonic 时钟。镜像内 Python 3.8 探针通过；原生绑定 Python 3.7 第一探针遗漏 native-env 可执行路径挂载，OCI 启动失败已保留。加入 native-env 只读挂载并以 -S 禁用 site 初始化后的独占新元数据探针通过，原 Python 二进制 SHA 精确，memory.max=6442450944，OOM/max 事件为零，private cgroup 自路径 0::/。实际成本控制器原有 CPU 根目录挂载已覆盖此 native-env，无需改变成本输入/runtime。

未提供 GPU、CUDA_VISIBLE_DEVICES 为空、NVIDIA_VISIBLE_DEVICES=void，无 Torch、模型、特征、推理或真实统计。全部五个远端小产物哈希读回、元数据输出 SHA 与本地观测精确，并确认该探针容器已释放。这仅接受监测接口元数据资格；GPU 开启的完整成本容器、三次重复与 L40 验收仍未完成。探针：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-112-original-Py37-private-cgroup-exact-executable-mount-probe-20261007T202328785909Z.json`；独立读回：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-112-original-Py37-private-cgroup-metadata-independent-byte-readback-20261007T202445088020Z.json`。缓存清理仍待用户明确回复，未删除任何旧状态。


## 2026-10-08 04:37 CST — M3 seed2027 原精确源码合同恢复

复核原11文件源码合同发现当前 detection_cache_v2.py SHA=9578ffd6… 与原封存 227b9094… 不同，直接运行当前工作树会被原源码门槛拒绝。只读 git show f8d7f54 恢复精确原 12342-byte 模块，其他10文件当前字节与原合同一致。独立新目录封存全部11文件并逐个按原 frozen-plan SHA 检查通过；工作树和 HEAD 未更改，没有 checkout/reset。原运行时严格要求 Torch 2.9.1+cu128、1 CPU thread；原计划未记录 NumPy CPU feature 选择，新复跑还须完整 L40 输入/runtime/依赖资格，不能复用当前不匹配缓存源码或声称小探针代替完整复跑。

M3 历史极小 yaw 差、CPU dispatch 可控复现与原始具体触发未知均保留，不放宽字节容差，不纳入合格三种子统计；未重跑已验配置，未上传新源码、未创建 Task。原拒绝 Linux priority/CUDA payload 门槛仍保留。证据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-M3-original-exact-sealed-source-recovery-20261007T203737461418Z.json`。


## 2026-10-08 04:51 CST — M3 精确源码与原 CPU 运行时 L40 实际准入任务

11 个原始合同文件扩展为 55 个导入时依赖；限定按原11文件与旧 git f8d7f54 选取，不把整个819源码树打包。227564-byte source-only 归档、322661-byte bootstrap 与55成员 SHA 均冻结；与此前被拒绝 Linux priority 包路径交集为空，无数据/权重/预测/凭据。函数内部及动态 import 的全部执行仍未证实。

使用原成功 L40 CPU 任务所绑定的不可变 image e5b249… 与原 files host 35.118:8081；指定 GPU3-L40S 队列、实际空闲 Worker gpu3,4,5 的新鲜心跳与空队列在创建前及 enqueue 前各检查。新任务 `06ef5d40f4c7415c854f92295c43dba7` 已排队，尚不算通过。明确 CUDA 空、1 CPU、2 GiB memory/memory-swap、pids128、进程CPU120秒和代码段墙钟120秒。仅标准库元数据、原生产模块 import 及 Torch/NumPy/Scipy 版本验证，不读取权重/缓存或调用生产 run。成功后仍须实际注册资产全 SHA 读回；失败保留，不自动重试或放宽限额。

源码与Task脚本/镜像/参数精确检查通过：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-M3-L40-original-source-runtime-import-dispatch-20261007T205038396628Z.json`。未运行完整M3复跑，未覆盖旧严格失败，未执行 Linux priority/CUDA 被拒绝路径；原五个科学任务不变。


## 2026-10-08 04:57 CST — L40 原 M3 导入任务实际失败与运行时缺项

任务 06ef5d40f4c7415c854f92295c43dba7 在预定 L40S gpu3,4,5 Worker 实际执行后终态 failed，注册 failure-preserved；模块 cryptography 不存在，原生产 runner import 与原 Torch/NumPy 版本断言均未到达。直接未认证文件URL返回401已保留；随后用标准 SDK 下载全部失败字节，完整SHA匹配注册值。不存在模型、输入缓存或回放执行，未重启失败Task，也未扩大内存/CPU/时间限额。

另用原同SHA镜像的无GPU、-S标准库 importlib.metadata 小探针仅查分发元数据：Torch2.6.0、NumPy1.26.4、Scipy1.14.1、cryptography缺失。此项是基础镜像元数据，不代替 L40 Task 在agent环境下的完整运行时资格；但证明只补cryptography不足以使基础镜像匹配原严格 Torch2.9.1+cu128 合同。下一步须在L40准备独立固定原版本环境后重新资格验证，不能换成基础Torch2.6的较易执行路径。

失败读回：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-M3-L40-runtime-import-failure-SDK-readback-20261007T205609526546Z.json`；元数据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-M3-immutable-image-distribution-metadata-20261007T205516677819Z.json`；合并前置项：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-M3-L40-runtime-contract-prerequisite-closure-20261007T205751185052Z.json`。原11文件与新增55导入源码资格准备保留，完整M3和E/K/A–K仍未完成，旧历史失败不覆盖，原priority/CUDA授权门槛保留。


## 2026-10-08 05:17 CST — 当前 plan 完成度与资源约束核对

按原 A–K 整包合同，仅 B 完成，其他十包仍缺完整验收；不按异质工作包数量计算论文完成百分比。新多卡 GPU 仅 A100、科学 CPU 仅 L40S、单卡 GPU 仅 35.112；原五个活任务保留。SDK 本次实读五个任务均 in_progress、无注册产物。最近完整有界日志观察 teacher1337/3407 序列完成下限为24/46、9/46，recovery-off 为40/46、35/46、35/46，均不代替目标或全队列独立验收。

原 Torch2.9.1+cu128 官方 wheel 已取得900881648字节，两次完整 SHA 匹配官方索引；11612成员、1722521215解压字节的CRC与RECORD全部校验通过。仅公开资产字节准备，无本地科学计算或环境安装；L40导入任务仍failed，完整运行时和M3复跑未通过。17项固定版本依赖仅元数据选定，完整依赖环境仍未资格验证。旧失败、原授权门槛与112缓存待授权状态保留，未清理、重启、提交或推送。

证据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-user-plan-completion-and-resource-policy-20261007T211741817480Z.json`。


## 2026-10-08 05:25 CST — M3 原 Torch 固定运行时公开资产全集字节资格

取得31个 Linux CPython3.12 安装包，共4132568910字节：原严格 Torch2.9.1+cu128、17个固定依赖、13个通用依赖。所有完整SHA、CRC与主包RECORD成员校验通过；再次全文件SHA核对，并按31个实际wheel根METADATA逐项检查Python及默认PEP508依赖，均满足。生成纯公开资产离线wheelhouse硬链接和带SHA的requirements锁；未安装、import科学库或执行本地科学计算。NumPy1.26.4沿用已控数值探针，SciPy1.14.1匹配基础镜像元数据；其他未记录历史版本的依赖是本次预声明环境选择，不冒充原环境记录。

保留三项控制面失败：缺少独立packaging模块、启动路径时间戳错误、setuptools初版checker误把16个vendor RECORD计为主包RECORD。前两项发生在网络/源码执行之前；setuptools已有完整SHA精确文件未重新下载。按[官方wheel格式](https://packaging.python.org/en/latest/specifications/binary-distribution-format/)定位唯一根RECORD后，512成员包括全部嵌套vendor RECORD作为payload均完整哈希通过，未放宽任何成员哈希、字节或工作容量限制。历史失败收据不覆盖。

L40实际环境资格与完整M3复跑仍未完成；旧导入Task失败保留，未创建新的L40任务。下一步为L40独立固定环境安装及原源码import资格，再接完整科学输入；原A–K范围、授权门槛与资源政策不变，112旧缓存未清理。证据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-M3-original-Torch-fixed-runtime-public-assets-complete-20261007T212541528281Z.json`；实际31包离线锁：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-M3-full-original-Torch-fixed-runtime-offline-lock-20261007T212407830863Z/complete-actual-wheel-metadata-offline-lock.json`。


## 2026-10-08 05:33 CST — L40 固定公开运行时安装与原源码导入新任务

新独立任务9b086aa5323745c68d3025e82fe686f3已排至原GPU3-L40S CPU队列，入队前实际Worker空闲心跳与空队列再次验证；先前导入失败任务06ef5d40…保留。原55成员源码SHA、227564字节source-only归档不变，与原被拒绝Linux priority包路径交集仍为零。31公开wheel按已验URL/size/SHA下载并二次完整SHA后离线安装到本任务独立目录，不替换基础镜像或全局环境；无模型、缓存、预测、凭据。

保持1 CPU、2 GiB memory/memory-swap、pids128、120 CPU秒；原源码导入子阶段仍为120墙钟秒。公开资产准备阶段沿用先前1200秒边界，安装/已下载总输出12 GiB、安装8 GiB、初始磁盘20 GiB和8 GiB保留空间在执行前冻结。隔离子解释器-I -S手动指定独立site-packages及原source；版本、模块来源、CPU thread、CUDA关闭、cgroup事件与完整已安装文件哈希分别记录，尚无运行资格结果。环境是本任务临时产物，不冒充持久安装或完整M3科学复跑。

首次排队前控制面检查使用SDK get_script diff字段失败，任务仍created，未执行。按后端data.script.diff读取342752字节源码SHA精确后只将同一已创建任务排队；原控制面错误收据保留，未重启任何科学失败Task。源码与计划：`/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-M3-L40-fixed-public-runtime-and-original-import-20261007T212842735296Z`；入队证据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-M3-L40-fixed-public-runtime-and-original-import-20261007T212842735296Z-same-created-Task-backend-readback-enqueue.json`。


## 2026-10-08 05:40 CST — 实际共享 L40 CPU 队列合同修正与新任务

9b086aa…实际被L40 gpu0,1,2取走，与该任务冻结expected_worker gpu3,4,5不同；仅停止这个新建source/runtime-only任务，状态stopped，agent报告User aborted: stopping task (3) status_changed，具体OS进程退出尚未另证。无注册环境资格产物，既有五个科学Task未停止。两个L40 Worker实际均订阅8d0f…同一队列；此前SDK workers.get_all模型未暴露task字段，缺失字段不是空闲证据，原过强idle表述保留并纠正。随后完整分页读取七个活跃Task元数据，未找到这两个L40 Worker的active绑定；仍不冒充OS进程空闲测量。

按用户CPU使用L40的主机约束，新增独立任务3e804ea3a278460d82b5e405469cc8a3在运行前冻结此同一物理主机两个共享队列Worker为接收集合，bootstrap实际reload Task核对last_worker后才能准备资产。硬件、原镜像、55源成员SHA、1CPU、2GiB、120CPU秒、120导入墙钟秒及所有容量边界不变，原9b stopped与06ef failed均保留。新源码额外实际核对cpu.max的quota等于period，不以缺失Worker字段推断空闲；初次和入队前均检查活跃Task分页穷尽、无这两个Worker绑定、共享队列为空和两Worker新鲜心跳。

后端实读新Task in_progress，last_worker=L40 gpu0,1,2处于冻结接收集合；尚无注册资产、完整wheel准备、实际安装或原源码导入资格结果，不称M3通过。源：`/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-M3-L40-fixed-runtime-actual-shared-CPU-worker-contract-20261007T213819863292Z`；入队：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-M3-L40-fixed-runtime-actual-shared-CPU-worker-contract-20261007T213819863292Z-dispatch.json`；当前观察：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-M3-shared-CPU-worker-fixed-runtime-live-20261007T214008305132Z.json`。


## 2026-10-08 05:45 CST — L40 实际 cgroup 接口安装前失败与标准库探针

3e804ea…在冻结集合中的实际gpu0,1,2 Worker执行，但安装前资源接口检查失败：/sys/fs/cgroup/cpu.max不存在。注册failure-preserved全部字节经SDK回读，SHA精确；stage=preexecution、downloaded_bytes=0，无31包下载/安装、科学库import、模型/输入或完整M3。v2 CPU接口未暴露已证，不能仅据不存在直接宣称具体v1配置已证。没有重启失败Task或扩大原容量/CPU/墙钟限额。失败证据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-M3-L40-actual-cgroup-interface-failure-readback-20261007T214357389780Z.json`。

新独立标准库元数据探针按固定少量v1/v2候选路径读取cpu/memory/cpuacct和/proc/self/cgroup，以确认实际接口与private namespace；每文件64KiB、总1MiB、CPU120秒/墙钟120秒、1CPU/2GiB/pids128，冻结同L40两Worker接收集合及原镜像，零科学源码、数据、模型或框架计算。该探针仅解决实际资源监控接口前置项，不冒充环境、M3或A–K通过。源码：`/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-L40-owned-cgroup-interface-standard-library-probe-20261007T214446950163Z`。


## 2026-10-08 05:58 CST — L40 v1 资源接口资格与组合内存门槛保留

标准库探针cae19c…completed，全注册字节SHA精确。实际私有memory/cpu/cpuacct/pids controller路径均/，v1配额100000/100000=1 CPU，memory.limit_in_bytes=2147483648，probe failcnt/oom_kill为零；v2 cpu.max和memory.max未暴露。新v1适配器以cpuacct纳秒计数转换微秒，并分开记录memory.failcnt、under_oom状态、oom_kill累计数，不把v1状态伪称v2事件计数。六项软件检查验证原CPU/内存边界、非私有controller拒绝、host swap及精确/增大memsw边界，均通过；这些不代替科学实验。

新Task cda035…实际在L40允许Worker执行，终态failed；memsw控制接口不存在而host SwapTotal非零，因无法证明冻结2GiB内存+swap上限在preexecution拒绝。完整failure-preserved SDK SHA读回精确，downloaded_bytes=0，无安装、模型、源码框架import或科学输入。旧Task不重启，限制不扩大。核对[内核官方文档](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v1/memory.html)后，swappiness=0关于limit reclaim的说明不被替换成该硬memsw上限证明；未改全主机swap、内核、启动配置或他人进程。

新独立原五任务日志有界观察仍为in_progress，序列完成报告下限24/9/40/35/35，各46序列；原主2027 rank5注册size/SHA不变，没有第三次下载。35.112信任保持的SSH df只读观察root可用9427464192字节，未见其他满足20GiB首组门槛的持久盘；未删除旧缓存。L40当前新鲜Worker元数据仅同一35.121两Worker，没有据此泛称所有历史/离线L40不可用。环境与M3仍未准入，A–K仅B整包完成，五原科学任务保留运行。汇总证据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-M3-L40-private-v1-resource-and-combined-memory-gate-20261007T215829459642Z.json`。


## 2026-10-08 06:13 CST — V2V native-v2 A100 活跃Task物理绑定准入补齐

新增控制面gpu_worker_task_admission.py，以独立有界分页Task元数据(last_worker/status/execution.queue)补齐Worker上报间隙的物理卡占用；不以Worker task字段缺失或null推断空闲。运行Task的精确Worker/IP/card绑定保留，queued Task在queue.entries尚未上报时阻断其池及重叠池；分页未穷尽、状态缺失或物理绑定未知均不准入。原selector90秒心跳、A100限定、4/8卡配方、物理交集和旧预留保持。另一组已知无交集四卡仍可准入，没有把同池所有运行Task泛称整池不可用。

13项标准库软件测试在工作树与最终独立冻结源码均通过，全部只用软件元数据夹具，无本地科学输入或拟合。冻结dispatcher除新增fleet import/查询外与此前合格冻结版本AST一致；训练consumer SHA651f3719…与原27科学模块/模型/损失/预算不变。仓库入口Task.create捕获参数对齐已合格冻结版本，禁用自动repo/package/Task.init捕获以保护显式source-only绑定和脏工作树。

真实有界Task分页穷尽、七个活跃GPU Task均成功解析物理Worker绑定，A100两组0–3及4–7仍为原recovery-off任务保留。三种子原已资格输入/注册资产SHA逐项实读并通过完整新source gate；Task.create/enqueue被测试进程硬禁用，三者均返回waiting_for_disjoint_idle_physical_GPU4_or_GPU8。未创建训练Task、未重启旧controller或延长旧期限，无checkpoint或C/论文通过。L40硬memsw上限缺项、112磁盘/旧缓存授权门槛、坏rank5和旧被拒绝包门槛保留。证据：`/Volumes/Data/test/recover-before-fuse/receipts/rbf-native-v2-A100-active-Task-binding-closure-20261007T221314389970Z.json`；冻结源码：`/Volumes/Data/test/recover-before-fuse/source-freezes/v2v4real-native-v2-A100-complete-active-Task-binding-closure-20261007T221204809342Z`。


## 2026-10-08 06:34 CST — K 完整 SPD 输出分数可靠性诊断 L40 独立任务

新增标准库 stage1_output_score_reliability.py，针对全部13逻辑配置/21序列/3316帧的原输出score，固定car/ego严格50m ROI与BEV IoU>=0.5。匹配先最大合法一对一数量再最大IoU总和，不使用confidence；相同解按原输入顺序和冻结Hungarian算法确定。概率对象显式为“输出框匹配GT”的条件可靠性，不是联合身份后验、真实身份风险或质量界证书。分别报告NLL/Brier/10等宽bin ECE、GT/TP/FP/FN/召回率；0/1错误分数保留无穷NLL，不裁剪。M3 seed2027原严格失败保留，raw诊断不进入合格三种子统计，M1仍仅一个确定性配置。

28个原完整输入（13预测、13报告、GT及manifest）共3191235081字节本地仅全SHA核对；11个几何/帧commit函数与已独立资格评价器AST相同。10项合成测试在工作树与最终冻结评价器通过，包括375个小规模矩形匹配矩阵与穷举最优目标一致、重复预测、空输出、门槛、严格ROI和无穷NLL。无本地真实GT匹配、模型、统计或科学回放。最初支持包checker错期望19成员，实际18=3 source/plan+13 reports+2 GT；错误收据与首次源码/归档保留，未执行科学数据。纠正计数后的全支持包6817693字节完成注册与SDK完整SHA回读；旧被拒Linux priority包路径交集零。

新science Task abe9a31680e14956b39ffb251d568eec 已在L40 gpu3,4,5 in_progress，后端bootstrap SHA与最终冻结精确，无注册科学结果。输入支持注册Task 3fd887016942467798c1a67b21eba5b1 completed仅字节登记，复用原13配置输出archive Task403f1e4479d7472aaadbdd38e64d957b的既有4067488121字节注册资产。初次/入队前均分页穷尽active Task元数据、确认同L40队列无queued/运行绑定及90秒新鲜心跳；两个同主机worker均在冻结接收集合，缺失Worker task字段不作idle证明。

此新独立标准库诊断合同预先冻结1CPU、科学子进程2GiB RLIMIT_AS（明确仅地址空间上限，不冒充组合RAM+swap）、600CPU秒、900秒科学墙钟、1200秒下载准备、10GiB下载/解包上限、18GiB初始磁盘与8GiB保留、256MiB gzip完整输出。不用于替代M3原2GiB memory+swap合同或重启M3失败Task。原5科学任务保持、A100多卡/CPU L40/单卡112政策保持，未清理缓存、提交或推送；K/E/A–K未完成。

冻结源码：/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-SPD-stage1-complete-output-score-reliability-L40-execution-20261007T223205139469Z；实际入队：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-stage1-complete-output-score-reliability-L40-execution-20261007T223205139469Z/dispatch-receipt.json。


## 2026-10-08 06:36 CST — 新分数诊断在旧文件服务认证处失败

abe9a316…终态failed，实际L40 gpu3,4,5；failure-preserved完整1493字节SDK回读SHA59663e06e717c78a6208df1e4c9455f0845e95440e0a45ca13ad727b89955d72精确。stage=byte-qualified-download，原预测归档SDK get_local_copy返回None，未进入科学评价。后端console证实旧注册资产10.100.34.118:8081返回HTTP401，新6817693字节support下载成功。没有修改URL、读取/迁移凭据、重启旧Task或放宽任何限额；开始对本地既有完整SHA合格的13个first预测原文件做byte-only运输派生归档，拟登记到当前文件服务，全部原预测字节/score/GT/协议保持。失败证据：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-stage1-complete-output-score-reliability-L40-execution-20261007T223205139469Z/actual-failure-readback-receipt.json；实际认证原因：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-stage1-complete-output-score-reliability-L40-execution-20261007T223205139469Z/actual-old-file-service-401-cause.json。K仍未完成。


## 2026-10-08 06:42 CST — 原13配置预测字节派生归档登记进程实证等待

13个原first预测成员逐项完整SHA再次匹配，byte-only gzip level1派生归档1110223011字节，SHAdba47c3c466ce2b158af244179a58ec04a3bf0f9eb5190860c1950b34bfaa62c；没有重跑预测或科学统计。新字节登记Task1016134cd19441e4add775153cface0e最后状态created，尚无注册产物；本次具体上传进程PID27917/exec session35107实证存活，RSS123808KiB，与当前10.100.35.118:8081 TCP ESTABLISHED。此证据仅为活字节登记进程，不是科学Task运行或上传/读回完成。

后继L40科学dispatch已冻结，plan、评价器及bootstrap与失败Task使用版本逐字节相同；仅更换合格预测注册资产引用。须等待同一原登记handle完成及完整SDK SHA读回后才能一次创建/入队新科学Task，不能重启原failed Task、因观察超时重传或扩大工作容量。原科学5任务、脏工作树、历史失败及旧授权门槛保持。下一次具体等待收据：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-stage1-score-reliability-L40-current-file-service-source-20261007T223833599267Z/verified-current-transport-wait-checkpoint.json；冻结入口：/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-SPD-stage1-score-reliability-L40-current-file-service-source-20261007T223833599267Z/dispatch.py。A–K goal保持active，仅B原整包完成。


## 2026-10-08 06:59 CST — 输出可靠性分箱反例与独立全标签验收准备

原输入运输进程PID27917/session35107持续实证存活；预测归档Task1016134…已有manifest及1110223011字节archive注册元数据，独立全文件回读仍待原进程完成，不从注册SHA推断实际读回成功。并行新增交替环残差图匹配核验，独立于producer Hungarian；全标签审计核对每个原frame/time/track/score、GT匹配可行性及最优性、空帧、完整per-sequence/aggregate NLL/Brier/ECE/FP/FN/召回率，不调用producer accumulator。数值容差预声明1e-10，仅浮点诊断核对，不冒充形式风险/质量界证书。

合成反例发现旧producer int(score*10)在score=0.8999999999999999上错误分到bin9，而声明区间为bin8。新增邻接浮点边界检查先在旧源码实际failed，反例收据保留；修正为显式比较九个声明下边界，不改原score、匹配、ROI、bin定义、原28输入SHA或资源限额。旧未执行science的source/plan/support保留；旧入口标记superseded-before-Task-creation。新版本21项合成检查（producer11、独立匹配5、全标签/标量变异拒绝5）在最终冻结源码通过，无本地真实GT/IoU/指标统计。

新支持包6823092字节SHA508badb015930e3b98c17181c2f89f88c0a11092ce3209459e3a71113de2ec1b完成上传与SDK完整回读；Task81adc1…原默认mark_completed请求未改变created状态（ignore_errors默认不能视为成功）。实读后仅对这个自建且已验的byte-only登记Task调用force=True、ignore_errors=False，后端get_by_id证实completed；不改科学任务终态。原登记状态收据保留并新增纠正证据。现有科学五Task和A100两组物理占用本次metadata分页穷尽仍保持，无可准入A100选择；不重启旧controller或迁移活任务。

新入口：/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-SPD-stage1-score-reliability-declared-bin-boundary-and-independent-audit-20261007T224932745891Z/dispatch.py；源码/合成资格：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-stage1-score-reliability-declared-bin-boundary-and-independent-audit-20261007T224932745891Z/software-preparation-receipt.json；字节Task后端纠正：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-stage1-score-reliability-declared-bin-boundary-and-independent-audit-20261007T224932745891Z/support-registration-backend-terminal-correction.json。新science Task尚未创建，须等待同一原预测登记进程完整回读，再明确关闭自建byte-only Task并核对后端completed，方可入队。完整K/E/A–K仍未完成；资源政策多卡A100、CPU L40、单卡112保持。


## 2026-10-08 07:07 CST — 原teacher1337新增两个序列前缀完成报告

按原观察器完整source SHA7e155f…和25页/500事件/180秒/64MiB原边界，新的五Task只读日志观察完成。原teacher1337下限24→26/46，rank1第4个序列0050完成（源前缀累计594事件），rank3第4个序列0055完成（累计691事件），均核对原46序列rank分配与7445事件完整原输入数目。teacher3407仍9/46；recovery-off仍40/46、35/46、35/46，各一rank无报告表示unknown，不冒充零或完成。五Task均in_progress且无注册产物，日志timestamp分页边界未证明穷尽，新增前缀是producer报告，不能替代teacher目标或整组独立验收，不据此给整体ETA。

证据：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-original-five-after-score-protocol-fix-native-rank-progress-20261007T230337860115Z/independent-original-five-new-prefix-delta.json；完整观察：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-original-five-after-score-protocol-fix-native-rank-progress-20261007T230337860115Z/independent-all5-original-rank-sequence-prefix-log-observation.json。原worker/card预留、脏工作树和所有旧失败保持，未重启/迁移科学Task。原预测归档回读PID27917/session35107仍活，实际.partial文件已849346560/1110223011字节（07:06实读）；未当作完整SHA或K完成。


## 2026-10-08 07:12 CST — 原预测完整当前文件服务读回通过与修正后L40科学Task

原PID27917/session35107正常完成，预测派生archive全1110223011字节经SDK回读SHA精确dba47c3c466ce2b158af244179a58ec04a3bf0f9eb5190860c1950b34bfaa62c。原注册Task1016134…默认mark_completed仍created的收据保留；仅在全字节回读通过后对这个自建byte-only Task明确force=True/ignore_errors=False，独立backend get_by_id证实completed，不改任何科学Task状态。旧HTTP401失败abe9a316…不重启、不改URL或凭据；旧分箱bug源码、支持包及合成反例保留。

修正后的science Task295f547b49144ed3a77bb9029beccaf8已一次创建/入队，当前实际L40 gpu3,4,5 in_progress，无注册科学结果。初次/入队前均active Task分页穷尽、L40共享queue空、同一主机两个worker90秒新鲜且无active绑定；原五Task保持，两组A100仍预留。后端bootstrap SHA8d2c351e…与冻结相同；计划SHAa1469a43…、producer6cef16da…绑定声明bin规则修正。全部原28输入refs、13配置、GT/ROI、最大匹配数量再IoU规则、1CPU/2GiB地址空间/600CPU秒/900科学墙钟/1200下载墙钟及容量边界不变。标准库科学评价由L40执行，无本地GT匹配、拟合、NLL或统计重算。

本次仅完成实际输入运输准入和科学Task执行开始，等待完整13结果与全部标签独立审计；M3 seed2027原严格失败不进入合格三种子结论，M1单配置、SPD seen-val探索性、启发式输出score非联合身份后验均保留。K/E/A–K仍未完成，原B整包通过不扩大。证据：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-stage1-score-reliability-declared-bin-boundary-and-independent-audit-20261007T224932745891Z/original-transport-gate-closed-and-actual-L40-science-dispatched.json；当前完整plan：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-original-transport-closed-and-L40-science-running-20261007T231304337763Z.json。


## 2026-10-08 07:22 CST — 原score诊断实际ROI计数门槛失败与同原范围修正

295f547…实际L40 gpu3,4,5终态failed。failure-preserved538字节SHA2f1bfefd…及science-failure-log915字节SHA285accd8…全SDK读回精确；stage=science，首配置失败于original full car/ROI counts differ，不是超时/内存/资产失败。原Task及partial目录保留，不重启。原report.car_predictions是pre-ROI car census，不能当作ego严格50m的预测数；已验native metric完整序列人口M0 seed1337 ROI内49497，pre-ROI76010，GT26648，与IDTP+IDFP/IDTP+IDFN精确一致。

新gate逐序列精确核对已验native metric帧数、GT和ROI预测数，再核对aggregate，pre-ROI仅作上界census，不删除检查。13原native metrics完整SHA绑定report.files，纳入41冻结输入（原28数据refs不变）。13配置仍3316帧/21序列/26648GT；M3 seed2027原严格失败保持。最终冻结12生产/5匹配/5全标签共22合成检查通过，独立audit另用native人口正/错误反例通过。仅已验元数据及字节处理，无本地真实GT匹配、NLL或统计重算。

support Task6572b3…completed，7329522字节SHAd948c4a9…SDK全读回；force completion仅自建字节登记Task。新science Task8cd87a77eb4f48408f449793bac7c58c一次创建/入队，actual L40 gpu3,4,5 in_progress、无注册结果。入队前独立active Task分页穷尽/共享queue空/新鲜worker及无绑定均核对；原五Task保持。1CPU/2GiB地址空间/600CPU秒/900科学墙钟/1200下载及全部容量不变，不扩大失败cap；原M3 RAM+swap门槛保持。收据脚本数字开头keyword语法错误发生在执行前，单列保留并修正字段名，科学Task不受影响。

源：/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-SPD-stage1-score-native-qualified-ROI-population-20261007T231724884312Z；人口范围：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-stage1-score-native-qualified-ROI-population-20261007T231724884312Z/ROI-count-scope-reconciliation.json；新dispatch：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-stage1-score-native-qualified-ROI-population-20261007T231724884312Z/dispatch-receipt.json；当前plan：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-native-ROI-population-correction-and-new-L40-science-20261007T232225806014Z.json。K/E/A–K未完成，仅B原整包通过。


### 2026-10-08 07:32 CST：13 配置分数诊断完成，独立审计已在 L40 运行

生产 Task `8cd87a77eb4f48408f449793bac7c58c` completed，实际 CPU worker `10.100.35.121-L40S:gpu3,4,5`。完整结果归档 15,925,934 bytes，SHA256 `06a1b7776d00587113a17a7a345313932e5a49191f18b1b9ca098b221f2d465d`，SDK 全字节及 16 个成员完整回读通过；13 个配置各 21 序列、3316 帧。独立标签/标量/交替环审计 Task `7f4d464d97974b57810cc56b0fab41c2` 已实际入队并 in_progress，未扩大资源上限。数值独立验收尚未完成；M3 seed2027 strict 历史失败仍保留；完整 E/K、同资源比较与论文均未完成。整包仍仅 B 已验收。

后续资源执行规则：多卡 GPU 仅 A100，CPU 仅 L40，单卡 GPU 使用 35.112；保留五个原活任务，不迁移或重启。源码生成时 JSON 引号包裹的 dispatch.py 未执行任何代码、未创建 Task，已保留并冻结正确的 dispatch-executable.py；当前真实 Task ID 来自后者的实际准入回执。

当前计划回执：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-producer-score-complete-independent-L40-audit-running-20261007T233146814266Z.json


### 2026-10-08 07:36 CST：独立审计活任务与配对分数统计准备

上一 goal 回合完成生产全字节回读并实际启动审计，属于实际进展。本轮原审计 Task `7f4d464d97974b57810cc56b0fab41c2` 最新 23:35:47 UTC 实时 SDK in_progress，worker `10.100.35.121-L40S:gpu3,4,5`，尚无注册产物，不按停机或完成处理。当前完整 active-Task 分页确认 A100 两个原 recovery-off 仍分别占用 0–3、4–7，不派发 V2V 多卡训练。五个原任务按完整 rank/原始 46 序列/7445 事件合同重读日志，下界仍 26/9/40/35/35；日志前缀不是整体验收。

已冻结后续 emitted-score 统计源码 /Volumes/Data/test/recover-before-fuse/source-freezes/rbf-SPD-emitted-score-paired-sequence-statistics-20261007T233402423779Z/stage1_output_score_statistics_original_numeric_types.py。固定 seed1337、10000 次整个序列配对重采样，所有方法/种子共享 multiplicities；Python random MT19937 明确单列为本诊断新统计合同，不能宣称与先前 PCG64 tracking draws 相同。NLL/Brier/ECE/recall 逐次合并充分统计量，ECE 重新汇总分箱而非平均序列 ECE，NLL 无裁剪、无限与 undefined 不中途删除；均值/样本标准差保留确定性 M1 的 singleton SD null。M0/M2/M4 三种子、M1 确定性可以在原始资格通过后统计；M3 seed2027 raw 保留，合格三种子与配对比较禁止。15 项合成算术检查及整数 0/1 score 类型补验已通过；最初严格浮点相等与只准 float 的准备版本保留，未执行真实数据。真实统计尚未运行，必须先完成独立审计全归档 SHA/数值验收，并冻结实际 audit 输入 SHA。完整 A–K 仍 active，仅 B 整包完成，E/K/paper false。

本轮完整计划回执：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-independent-score-audit-live-and-paired-statistics-software-20261007T233630538957Z.json


### 2026-10-08 07:44 CST：13 配置分数诊断及 10000 次配对统计独立验收

独立标签审计 Task `7f4d464d97974b57810cc56b0fab41c2` 已完成，全 13×3316 帧、21 序列，各条标签的原始绑定、ROI、几何与交替环匹配检查通过，独立 NLL/Brier/ECE 重算最大差 0。完整 109433B 归档及 3 个成员 SDK 全 SHA 回读通过。统计 Task `37bb509b7bfd41ee901aeaf1f33ce1cb` 也在 L40 CPU 完成；84009B 归档及 5 个成员全 SHA 回读通过。seed1337、10000 whole-sequence shared draws、均值/样本 SD/配对 percentile 区间由不同源码和展开 draw 算术完整复算，最大差 2.220446049250313e-16。生产统计/复算器顺序运行于同一隔离 child；只声称源码/数值算术独立，不声称环境或进程独立。资源上限未扩大；无真实科学计算在本地执行。源码/计划/两原任务输入 SHA/运行时/结果完整绑定，byte-only source support Task `bffe7a9811a5488c9f44a0f36a1f2566` 经全 SDK readback 后独立 backend completed。

已将全部原始配置、合格 M0/M2/M4 三种子和确定性 M1 汇总、12 个配对区间复制到 stage1-spd-seen-val-evidence-table-20261008.md。M3 seed2027 raw 保留且三种子/配对统计排除，未填缺失值。M4 输出置信度诊断改善与先前 HOTA/AMOTA 下降、tracking FN 代价并列保留，不推广为恢复独立优势；逐帧 IoU recall 也不能替换 tracking 多阈值或身份指标。该诊断不是联合身份后验、真实身份风险、η/TV 证书。完整 A–K、E/K 和论文仍未完成，只有 B 整包完成。

当前完整计划回执：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-complete13-score-diagnostic-and-statistics-accepted-20261007T234435416193Z.json


### 2026-10-08 07:50 CST：原任务事件级进度与总 ETA 边界

上一 goal 回合完成 13 配置分数诊断和 10000 次配对统计独立验收，属于实际进展。本轮完整 active-Task metadata 分页仍确认两个原 A100 recovery-off Task 分别占用物理 0–3 与 4–7；未派发新 V2V 训练，也未移动/重启原任务。

Task 事件头部实时回读表明 teacher1337/3407 进度更新时间距 23:49:48 UTC 仅约 17/38 秒，仍有实际事件进展。三个 recovery-off 也均 SDK in_progress、无注册最终输出，但各最后已报告事件进度距观测约 8.1/16.2/6.4 小时。最新有 context 的记录都指向 rank-2/0002：seed2027 367/452，seed1337 与3407各349/452；“无序列完成报告”不等于无事件执行。其他 rank 已报告完所属序列；rank2仍在首序列且后续序列未被整体接受。历史日志中没有 coupled_failure_before_receipt_upload；Task heartbeat和事件进度应分别报告，观测空档不等于终态或重启授权。GPU最后标量利用率接近0，但缺少已绑定当前进程栈证据，不能据此宣布死锁或释放原卡。

用户询问总完成时间：最新 teacher 单个 schedule 日志 ETA 约1.5小时和14小时，明确不能作为 teacher 整个46序列 Task、后继训练/回放/成本或 A–K 总 ETA；recovery-off旧局部ETA缺乏新事件支持，不用于总排期。完整目标仍缺真实最终模型、完整 teacher/priority、两数据集完整方法/消融、训练内资源选择和≥3次同资源成本、公开基线实际权重/前向及全部论文收口。总完成日期保持未知，不制造“几天”结论。下一可执行工作为精确确认 rank2 长时段计算/失败状态，等待原产物时仍保留全部科学/输入/授权门槛。

当前实时证据：/Volumes/Data/test/recover-before-fuse/receipts/rbf-original-five-live-event-progress-and-user-ETA-20261007T234948562101Z.json


### 2026-10-08 07:57 CST：原 recovery-off 控制流与当前主机门槛

上一回合为有证据的原活任务等待；本轮从实际 Task script.diff 解析并逐字节 SHA 绑定原 REPLACEMENTS 的 paper_decision.py 与 experiment_progress.py，仅阅读，不导入/执行/上传。A100 两个原 Task bootstrap SHA 均 e8614dafd2a918a8c447f45572fababc00ed3ab3c0d1a6afe2643dac6d0e736b：torch.multiprocessing.spawn(...join=True) 完成前不会注册任何 rank archive；每个 rank 的 sequence_progress 在整序列 replay、SQL integrity及原 factor逐行比较之后才产生。进度 reporter 没有独立 background thread，事件未更新不能当进程终态。动作搜索原 queue/steps<budget 条件、4096动作/8192 frontier 保留；不将有限循环上限解释为已证明当前进程栈或死锁不存在。

当前检查默认及有效 SSH known-hosts 配置均无 34.18/34.26 的现有主机信任或别名替代；未发起该两IP SSH连接，也未新增 trust/keys。已异步请求用户提供已有可信入口的别名（不请求私钥/口令）；未回复不作授权。trusted 35.112 的只读 SSH 复核 root free=9430536192B、原 GPU UUID 一致，GPU free=11838MiB；仍未达原21474836480B磁盘门槛。未上传或启动新 native-cost 实验，未删除旧缓存，不缩小容量边界。原活任务、CPU L40/多卡 A100/单卡112规则及所有旧失败/授权门槛不变。完整 A–K仍active，仅B整包完成。

本轮证据：/Volumes/Data/test/recover-before-fuse/receipts/rbf-original-rank2-runtime-flow-and-existing-host-access-gates-20261007T235814549934Z.json
完整计划回执：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-rank2-runtime-flow-and-host-gates-20261007T235814549934Z.json


### 2026-10-08 08:01 CST：原 teacher1337 序列完成下界 26→27

同一原 Task 的新原生日志完成报告已按 46 序列 rank::world 分配和各原序列事件计数核验，不采用无 context 的 schedule 百分比。teacher1337 新增一个完整报告，下界27/46；teacher3407 为9/46，recovery-off2027/1337/3407为40/35/35/46。五个原Task仍in_progress、未注册最终归档；不把 producer prefix 当完整 teacher 目标验收。原 runtime/资源预留、旧失败和权限门槛保持。已有 SSH 别名问题仍待用户信息，不重复询问、不新增主机信任。当前为有真实进展的等待，不因长时运行标记整个目标 blocked。

独立前缀差分：/Volumes/Data/test/recover-before-fuse/artifacts/rbf-original-five-after-rank2-control-flow-diagnosis-20261007T235948772697Z/independent-original-five-new-prefix-delta.json
完整计划回执：/Volumes/Data/test/recover-before-fuse/receipts/rbf-plan-completion-original-teacher1337-new-prefix27-20261008T000052434168Z.json
