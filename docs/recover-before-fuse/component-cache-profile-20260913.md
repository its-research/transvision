# 分量缓存淘汰开销剖析

## 结果与限制

2026-09-13，在同一真实 train 数据库副本的单次 teacher 探测中，按分量维护缓存访问顺序后，耗时由 1.038 s 降至 0.594 s，约减少 43%。探测结果与质量上界不变。新增索引容器浅层大小为 929,048 字节，约 0.89 MiB；不是零内存优化，也不是进程峰值内存测量。

这是定位和减少局部实现开销的证据，不是完整训练、推断尾延迟或同资源算法优势。该探测仍触发资源限制，模型内遗漏质量上界仍为 1；没有因此完成计算分配 teacher 或证明恢复机制有效。

## 输入与操作边界

原 teacher 已主动中止，保留 13 个已提交事件。原数据库为 `/private/tmp/spd-identity-fulltrain.jtGEnO/allocation-teacher-seed-1337-v1/sequence-0000.sqlite`；本次仅对 `/private/tmp/rbf-teacher-probe.PeVfiw/probe.sqlite` 副本操作，没有恢复或改写原 teacher 输出。

副本按数据库摘要和最后预测摘要校验后打开。预测头为 `83833a7fcd186a836a7d26fdaa6c85def092e19e2e89b646322d63a33e8f2c1b`。在相同缓存及模型绑定下，对已有状态执行一个不加入新观测的有界决策，截取最大前沿分量的首次 teacher 探测后立即退出；事务回滚，不增加已提交事件。决策时刻为 `1626155125281153` μs，参考时刻为 `1626155125181153` μs。首次准备因遗漏缓存／评分器绑定而在推断前失败，不计为成功探测。

优化前后均为 133 个分量节点、4,096 个前沿、8,192 个缓存条目和 9 个候选。探测返回值均为 `target=0`、`model_bound_after=1`、`charged_steps=0`、`resource_limited=true`。

## 修改及回归

`persistent_component_tracking.py` 中的 `_CacheView.popitem()` 原先遍历全局缓存查找本分量条目。剖析中发生 5,362 次淘汰，累计约 0.571 s，占该次探测约 55%；大量时间用于扫描其他分量的条目。

现由 `_GlobalLRU` 同时维护全局顺序和各命名空间的访问顺序索引。分量内淘汰直接取本地首尾键，不再遍历无关分量；全局淘汰和清理同步更新索引。索引仅持有已有键的引用，不复制缓存值。条目上限、分支、势函数、权重和风险公式不变；索引额外内存随全局条目上限增长。

缓存及直接推断回归 105 项通过；补充的持久化、恢复、beam、概率后端、计算分配和资源入口回归 79 项通过，耗时 25.55 s。新增测试包含 3 个随机种子各 1,000 次操作与朴素全局顺序实现逐步对照，以及存在 6,000 个无关条目时禁止全局遍历的检查。补充回归回执为 `/private/tmp/rbf-teacher-probe.PeVfiw/indexed-cache-backend-regression.xml`。

剖析后仍有候选遍历、分支重建和资源上界问题需要解决。不能把本次 43% 的单探测时间减少外推为完整序列加速；需要继续进行完整序列吞吐、内存、质量上界及跟踪指标验证。

## 审计摘要

| 文件 | SHA-256 |
|---|---|
| `baseline-probe.pstats` | `7c0ecd295bd961280d25daba19912e3305fbddf5652ed8ece0419b6247198318` |
| `indexed-probe.pstats` | `c35dc08a9b7d13dcc64cf30642120110d56aa49926aa99091582c6cbee5dfbbf` |
| `indexed-cache-backend-regression.xml` | `658adc796155cd4df7874778ab01b7600fe4dd69c33bea85fd779fa33dd437ff` |

三个文件均位于 `/private/tmp/rbf-teacher-probe.PeVfiw/`，保留用于审计。本次未发布数据库、预测流或这些剖析文件到 ClearML。
