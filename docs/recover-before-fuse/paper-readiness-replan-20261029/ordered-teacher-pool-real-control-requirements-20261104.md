# 候选并行核验：真实对照前仍须完成的要求

这是源码审查后的缺口清单，不是实际数值或性能准入。候选池合成协议检查已通过；随后适配器及结构检查也已按具体批准在112执行11项控制，通过原函数AST完整还原、断言保留、单一候选调用替换及篡改拒绝。两者都不代表真实worker、原完整轨迹或性能通过。当前0072、V2V及停止的3090任务均不改动。

## 原调用点与不变量

来源谱系核对发现：3407已合格0087单遍入口使用686e72bf…/6586c45b…，当前候选固定2aecc66e…/37ddf0a7…。不能直接混用；1337/0087原20事件为15160标签/436因子，3407为17307标签/442因子。完整边界及当前合格优化/存储公平比较要求见`ordered-teacher-real-control-source-family-boundary-20261104.md`。尚未运行新的真实对照。

冻结trajectory.py SHA37ddf0a7437064d41f6df36a0879eb67b19a430e306c2fb1c8d6e48827c894a9在每个allocation selection中先验证完整causal context、eligible candidate顺序及chosen行为，再逐record执行before-state相等、原numeric.verify_candidate、after-witness读取。随后才更新选中状态、prefix/frontier、remaining、blocked、spent/proposed及最终事件输出。

允许调查的边界仅为该selection内互相独立的numeric.verify_candidate调用。不能直接把事件分给worker；也不能提前提交选中状态或删除父before/after读取。父序列、组件分区、到达时钟、累计factor hash、完整prefix/membership目录及最终全部见证引用相等检查须全部保留。

## 当前合成检查未覆盖

1. **实际worker导入**：原numeric源码2aecc66e…导入时还会按绝对ROOT加载冻结identity-forest oracle和legal-action decoder。21:05已获具体批准，在原e5镜像及只读source-freezes中导入原pool/reader/numeric/trajectory并成功构造适配函数；五源SHA前后相同，SQLite连接尝试0，未提供DB/权重、未调用数值或启动worker/pool。此项证明原模块加载/绑定接口可用，尚不证明真实spawn initializer或真实数据工作。后者仍须完整只读命名空间与来源SHA。
2. **实际见证读取**：真实compact reader.decode(db,key)、原raw_bytes、完整raw SHA、canonical JSON和同sequence检查尚未由已运行的mock Executor测试覆盖。不能以mock witness_refs替代真实载荷核验。
3. **父函数集成**：`ordered_teacher_trajectory_adapter.py`已准备，112结构检查通过“仅包裹原record循环、仅替换候选调用、额外检查worker引用覆盖，去掉这些适配后完整原AST不变”。适配器SHA为f749bb85bfca927f98a6d3eb21421cf02e761797f51a52c127341a6a9728cdf8，检查器SHA8761ab51d7c3ec973cb28e84a32cff5d50f1c526302a2e72fb3b00c6d08a8f36。真实模块绑定和完整原轨迹尚未执行。父before-state、after-state、原maximum归约次序及失败传播仍须在真实对照证明。不得用worker引用子集检查替代原最终全集相等断言。
4. **真实完整结果对照**：同一原容器/输入/模型/预算下，原完整轨迹和候选完整轨迹均须保留全部原检查；逐候选完整18维特征、signed target、charged steps、完整事件输出及全部最终目录/引用必须一致。原1e-8与原因子1e-4保留，不能只比汇总counts/max error。
5. **端到端测量**：计入spawn、原输入SHA、原独立解码、数据库读取、IPC、父串行工作和最终收集；记录CPU累计、墙钟与聚合RSS。共享112负载只能作诊断，不能当作I项隔离同资源比较。2/4/8 workers均不能直接声称线性加速。

## 准入与授权

候选池两文件暂存/合成协议批准、新适配器两文件暂存/一次结构检查批准及原模块来源导入/绑定一次批准均已执行。结构检查证据在112 `/home/lbin/rbf-approved-ordered-teacher-trajectory-structural-controls-20261104/complete-small-output-source-SHA-readback.json`；11项exit0，0.698秒，未导入真实oracle。随后明确获批的源码加载在原e5镜像成功，容器exit0、模块步骤0.107秒、外部启动至收集0.370秒，仅导入不调用数值；7份小文件全SHA绑定于 `/home/lbin/rbf-ordered-teacher-worker-loader-preparation-20261104/approved-execution-once/complete-small-source-output-SHA-readback.json`。两次均未读DB/权重、未启动真实并行、未改活任务。尚未获真实数据对照、发布或活任务切换许可。先把原完整输入与真实对照/收集入口具体准备，再遵守已有具体来源发布/计算和新失败不自动重试门槛。不运行Mac计算，不占用3090，不为测试停止或重启0072/V2V。所有准备和失败继续保留。

## 当前结论

源码给出可调查的候选级并行边界，但没有真实数值等价或速度证据。当前仍按已授权原入口收口0072与V2V；本清单不新增科学合格计数、teacher资格、priority准入或A–K完成声明。
