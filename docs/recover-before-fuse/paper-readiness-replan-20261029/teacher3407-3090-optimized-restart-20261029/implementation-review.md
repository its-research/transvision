# teacher3407 3090 优化替换审查

本候选只准备在新目录；未停止、重启、上传或运行任何实验。父任务已完成本机0087原版/候选完整两阶段对照，仍须新Linux原环境准入。

- 当前3090任务执行四路CPU独立验收、CUDA禁用，每路约60CPU秒/60墙钟秒。CPU增长不单独证明有效计算；最新记录仍无完整新序列报告。
- 最小算法变更只把每棵树的3处重复jump JSON解析改为一次映射。原数值算子、遍历、12个结构断言、1e-8语义及1e-4因子标准均保留，不缓存数值或树验收结果。
- 新增真实verify_candidate调用完成量，区分numeric和trajectory；只观察、返回原对象，并在finally恢复。15秒进度同时给出真实标签和树/前缀计数。
- 根据父任务最新资源核验，下一版由bootstrap调度8个独立完整序列进程，要求新鲜CPU>=8、可用RAM>=64GiB，保持每序列BLAS1及轨迹顺序；不是增加同一序列的线程。

原mixed生产输入为11旧+35新，共46序列、7,445事件、5,560,061标签、223,355因子行。38,287,360字节原包与11,438,080字节旧覆盖包已本轮完整重读SHA。主森林225,177因子行不适用于本teacher合同。

0087原报告按原source-family完整30项断言和传输收口重新核对，可原样复用。26份五项报告也属于同一teacher3407的原mixed输入，保留其来源/容差/终态和SHA，不重复；它们不能替代尚缺的目标+轨迹+因子结果。完整六项为raw context、structure、causal、fresh_state、action及teacher target/trajectory，另要求原模型因子与native来源链。

本目录check_source_family_reports.py作为来源分组组件；join_optimized_source_bound_reports.py处理原0087+45优化报告，所有report头保持原样。最终网络收集器由父任务接入；execution binding及终态readback都必须齐备。full46 target join仍不代表完整六项或论文验收，原六项join入口的旧protocol字面量需以新的显式source adapter扩展。

旧任务工作根为/clearml_agent_cache/teacher-full46-independent-8c007e2ab4a84fc0985ddbf0cfedf94b。先保存平台状态、全部已注册产物SHA和日志，再仅停止被授权的旧独立验收。旧工作目录不删除；新任务逐字节校验后复用输入。当前旧bootstrap未提供外部停止时完整封存所有未注册工作目录的保证，不能宣称这些数据已完整归档。部分事件或标签计数不能冒充可复用完整报告。

27项无数值控制通过，真实原0087来源验收通过，新增8种错误终态凭证拒绝通过。详见software-control-results/result.json、source-family-controls/result.json和numeric-entry-and-collector-file-lock.json。所有Linux对照、发布与替换执行由父任务统筹；本审查不新增授权。
