# teacher3407 3090验收优化独立审查

审查日期：2026-10-29。范围为源码与已存在证据的只读分析；未启动数值检查、上传、停止或重跑任务。当前0087同机完整对照由主代理执行，本审查不重复它。

## 结论

优先采用已经准备的每棵树一次解析 jumps 候选，并以当前完整0087对照实际结果决定是否进入Linux替换。该候选保持独立树重算和所有原断言，局部正确性论证充分；但它只解决重复JSON解析，不能据此宣称所有大序列瓶颈已经解决，也不能预先承诺提速倍数。

现阶段保留四个独立序列进程最稳妥。四路各约一个CPU核，与冻结Python树遍历的串行执行相符。提高每路Python线程数不能使受GIL约束的DFS、字典和JSON循环自动并行。不要为利用GPU重写浮点算子；这会引入不必要的数值和部署风险。

## 实际重复工作与候选覆盖

- 原 `verify_candidate` 对每个标签分别构造 before/after 两棵完整树。原 `oracle.verify_database` 又读取一次 before，独立查询对应观测槽位和因子。原 `trajectory.verify_database` 还会对实际选中状态、事件初始状态和最终输出状态重新构树。
- 原 `_verify` 在二进制祖先循环内反复解析祖先 jumps JSON，最后再解析当前 jumps。候选先解码 `self.p` 内每一条记录，再执行原顺序的整个检查。它降低的是这些重复解析次数，保持每标签重新计算树质量、动作、目标、误差和结构约束。
- 当前 `verification-reader.py` 已有每只读连接64MiB、最多64条的压缩旧格式payload缓存。命中该缓存仍会在冻结oracle中执行zlib解压、JSON解析、canonical字节检查。不要把现有payload缓存误称为已经缓存了数值结果或整棵树。
- 已有原始0000工作量清单绑定195事件、133,758标签、24,289次分配组；仅raw-label阶段就有至少267,516次树构造和401,274次见证读取。151–175事件区间的前缀核验下界为115,131,796行，176–195为122,955,540行。这是原封存上下文推算的工作量，不是当前3090调用栈或已执行计数。
- 原进度只每25事件发出一次。长时间计数不变既不能证明空转，也不能证明有效计算；替换入口应额外报告标签计数、树构造数、累计前缀行、CPU/墙钟和阶段，避免继续只依赖25事件粒度。

## 正确性审查

`jumps_once.checked_method` 仅接受冻结 `_verify` 的两个祖先表达式和一个当前表达式；新增每树局部解码表后反向恢复AST，并断言与原AST完全一致。原12项assert数量、DFS顺序、浮点表达式均保留。`__init__`、`ancestor`、`cut`和`upper`未替换。候选局部表在本次方法返回后释放，不跨树复用已验结构或数值。

已有控制覆盖三种完整合成轨迹，以及12类树篡改、5类轨迹篡改和3类绑定/恢复拒绝。它们支持实现局部性质，不能替代当前20事件17,307标签两阶段真实输入对照。

潜在边界：候选提前解析所有行，会比原遍历更早拒绝某些无效JSON；有效输入结果不应变化。新增解码表会增加瞬时内存，必须实测RSS峰值；保持原资源准入而不靠提高预算掩盖问题。旧冻结函数运行时也必须保持 `sys.flags.optimize == 0`，不能仅依赖候选动态编译的 `optimize=0`。

同机paired比较应保持当前策略：原数值函数逐标签实际重算、标签序号/组件/before/after/result摘要逐字节一致；两臂完整报告除elapsed外一致。历史跨平台 `max_abs_error` 仅作诊断并不意味着放宽原1e-8逐值检查。所有其他事件、标签、来源和语义字段仍严格绑定。

## 多核安全边界

1. **跨完整序列多进程**：边界清楚，进程拥有独立只读连接、全部历史和输出目录。可以在资源核验后调整队列与并发，但不能仅按空GPU推断足够CPU/RAM/磁盘。
2. **单序列raw候选内多进程**：理论上可把已绑定record/context/config和只读见证分发给独立进程，按原序合并返回；需要显式证明全标签覆盖、无遗漏重复、汇总与原序完全一致，并计入IPC及每进程缓存内存。当前候选未实现或验证，不应与本次最小解析优化混入。
3. **trajectory按事件分片**：不适合作为直接优化。`previous/catalog/memberships/owners`、累计因子哈希、预算、选择顺序及归档前缀计数跨事件连续；不能从任意事件以空历史开始而称全序列验收。
4. **数值/树检查结果缓存**：当前候选明确不缓存。未来如考虑，必须另有不可变来源键及等价证明；不能跳过原独立检查来缩短本次任务。

## 最小可执行切换建议

1. 收齐当前本机完整0087原版/候选numeric与trajectory、逐标签摘要和独立SHA读回；实测端到端墙钟/CPU和阶段占比。`root_verify`等计时为嵌套inclusive值，不可直接求和。
2. 新Linux入口先验证冻结源SHA、原Python环境、assert启用、动态派生方法与12断言；执行既有合成正反控制。完整真实0087在原Linux环境的控制可以作为同一次优化重跑入口的前置阶段，未通过即终止，不自动重试；保留结果，不进入45序列。
3. 对长序列只增加观测与来源读取，不重复完整原版长序列。记录候选前缀规模、标签速率、RSS以及逐阶段耗时，确认无明显内存放大；0087提速不能外推为0002最终ETA。
4. 保存旧任务已注册产物与可访问日志，区分完整独立结果和未收口工作目录。新入口和资格通过后才按用户授权停止旧任务；不声称未注册目录已完整封存。
5. 新任务复用已合格0087目标/轨迹/因子范围；26份五项补验属于另一验收范围，只能在同数据库和来源身份匹配后汇合，不能充当45序列目标完成。完整46序列仍须各自所有检查与收集器收口。

若当前完整对照显示收益很小，下一候选应依据实测占比优先解决重复见证解码或immutable数据库支持查询，而非立即增加并发。可考虑按已封存数据库建立只读成员/槽位/因子索引，保留原排序和逐prefix切片；这是另一项实现，必须有完整对照，当前尚未完成。

## 核对过的源码与证据

- `paper-readiness-replan-20261029/prefix-jumps-once-candidate/jumps_once.py`
- `paper-readiness-replan-20261029/prefix-jumps-once-candidate/synthetic-controls.json`
- `paper-readiness-replan-20261029/teacher3407-jumps-once-full0087-control/report-contract-repair/compare_once.py`
- `paper-readiness-replan-20261029/teacher3407-jumps-once-full0087-control/sealed0000-workload-accounting.json`
- `teacher-witness-payload-cache-candidate-20261027/relative-tolerance-contract-v2/control-inputs/source/independent-storage-v2/{oracle,trajectory,frozen_tree}.py`
- `teacher-witness-payload-cache-candidate-20261027/full46-source-preserving-candidate/a100-full46-payload-cache-preparation-20261028/entry/verification-reader.py`

以上路径相对本工作区 `docs/recover-before-fuse`。本审查不构成新增数值合格、Linux资格或完整46序列验收。
