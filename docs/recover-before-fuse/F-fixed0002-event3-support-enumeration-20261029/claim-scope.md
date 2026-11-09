# 固定0002事件3：历史类恢复与当前输出的区别

本次只分析既有17个固定恢复事件中，唯一在前一事件活跃类权重与分支状态摘要相同的候选。没有按GT或指标重新挑选案例，没有运行模型、序列回放或改动现有任务。

**结论：这是历史身份类恢复见证，不能用作当前输出恢复或跟踪质量收益案例。** 前一事件的活跃类相同，并不代表完整历史支持相同。

| 范围 | 独立核实结果 |
|---|---|
| 事件2、组件34，原全支持 | 4节点，12个合法父边配置，汇合为10个身份类 |
| recovery-off在事件2执行时的历史限制 | 允许其中7个身份类 |
| recovery-off进入事件3时的前驱限制 | 仅允许前次保留/输出类`[0,1,0,3]` |
| 主方法事件3选中类投影到该前驱 | `[0,1,0,0]`，此前仅由前沿覆盖，被上述关闭恢复的边界排除 |
| 两路事件3的选中历史类 | 不同 |
| 两路选中分支保存的状态载荷 | 2,104字节，逐字节相同；事件3审计状态摘要也相同 |
| 两路当前完整输出 | 28个预测，完整字节相同 |

独立计算分别使用根身份类递推、原始父边配置求和，以及另一脚本的父边连通分量分组。原ATOL/RTOL均为`1e-8`；独立复核最大绝对差为`2.7755575615628914e-17`。先执行了25组小图双方法控制与3项相关历史/同帧冲突检查。原主数据库与失败recovery-off归档均完整SHA核对，只读打开；后者在内存中读取，没有恢复或更改其原事务。

在**事件2的四节点冻结因子模型**内，保留类质量占比约1.6935%，后来选回类约85.2494%。这些是该有限模型下的归一化质量，不能解释成物理目标正确率、校准概率或事件3完整35节点组件的概率。没有完整枚举35节点合并组件，也没有新增9D数值重算或GT评价。

历史恢复的算法现象保留；当前28个输出完全相同，意味着本配对事件不能提供当前输出改变或质量改善证据。后续仍需独立的同状态干预、真实交叉/遮挡原因、当前合法输出改变及物理正确性证据。本结果不关闭F或整个A–K。

证据：

- [原数据有限支持报告](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/F-fixed0002-event3-support-enumeration-20261029/fixed0002-event3-exact-support-and-claim-scope.json)，SHA256 `d1fc82c3e675be1c760c22821d933f0b94b8010a0fcb01fc856501004b69859a`。
- [独立枚举读回](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/F-fixed0002-event3-support-enumeration-20261029/independent-support-enumeration-readback.json)，SHA256 `8b0b1bfd8ad308d9a8623cf16c3e73d39ee5b6e29d19c01fc0dc5f467d38c03e`。
- [状态载荷与当前输出字节对照](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/F-fixed0002-event3-support-enumeration-20261029/selected-state-and-current-emission-byte-equivalence.json)，SHA256 `7e9da4f19977065f4e143fa72046d9dfe41b0690d09376aa363394521c337b6b`。

三个脚本均保留在本目录，原来源路径及SHA在报告中。输出排他创建，避免覆盖原证据。
