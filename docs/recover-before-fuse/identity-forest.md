# 联合身份森林：已实现的有限窗口原语

实现：`transvision/models/event_track_v2x/identity_forest.py`。
该原语联合约束跨端和跨帧身份。现已通过 `forest_tracking.py` 接入有限窗口的
3D 输出与独立分支状态重放；完整序列管理和真实数据训练仍未完成。
研究范围仅为 car；它不改变既有 `DetectionCacheV2` 或封存后端。

## 表示与概率对象

观测按实际到达顺序固定为节点 \(i=0,\ldots,n-1\)。节点选择前驱
\(p_i<i\) 或新生 \(p_i=-1\)，沿前驱链得到根身份 \(r_i\)。
同一身份中，不允许出现两条来自同一 `(source_id, frame_id)` 的观测。
互斥约束检查整个连通身份，不能通过另一端的节点绕过。

给定合法森林集合 \(\mathcal H\) 和有限正势，模型为

\[
q(h)=\frac{\exp\{\sum_i\theta_i(p_i)\}}{Z},\qquad
Z=\sum_{h\in\mathcal H}\exp\{\sum_i\theta_i(p_i)\}.
\]

两个不同前驱森林可能表达同一身份划分。当前模型把它们作为不同历史，
对相同划分的概率求和；没有声称这种参数化对历史数量无偏。
势的来源与真实分布校准仍需另行验证。重新评分替换绝对模型势，不把重复历史当作
独立新证据再次相乘。旧节点及旧前驱支持不能被重新评分操作删除。

## 可恢复压缩与上界

高权重合法叶子保存在 active 集，其他历史由互不重叠的前缀 frontier 覆盖。
对前缀 \(c\)，放松后续身份互斥得到

\[
Z(c)\le U(c)=w(c)\prod_{i\ge |c|}\sum_{p\in S_i}\exp\theta_i(p).
\]

每个合法完整历史恰被一个 active 叶子或 frontier 前缀覆盖，因此

\[
Z\le Z_U=Z_K+\sum_{c\in frontier}U(c),\qquad
\eta\le 1-Z_K/Z_U.
\]

新到达节点把旧叶子重新变为前缀；重新评分可使此前未枚举的历史被优先展开。
容量不足时停止细化，保留未展开区域，不用平均状态替代身份分支。
active、frontier、节点、提交数、已发现叶子数和动作搜索分别有显式限额。
追加导致存储超限时，在修改状态之前失败，要求外部明确执行窗口交接。

`validate_snapshot` 校验哈希绑定、前缀互斥、支持覆盖、原始权重及质量重算。
哈希只是本地完整性检查，不是可信签名；浮点上界是模型内估计，不是区间算术证书。

## 身份动作与风险

动作空间是全部合法森林，不限制为 active 集。损失为非 anchor 节点的归一化
根身份错误：\(\ell(a,h)=|I|^{-1}\sum_{i\in I}1[r_i(a)\ne r_i(h)]\)。
当 \(I\) 为空时损失定义为 0。
有限预算搜索报告条件风险的剩余优化差距 \(\delta\)，则在精确数学中

\[
R_q(a)-\min_bR_q(b)\le\eta+\delta.
\]

这里左侧的 \(q\) 是完整有限窗口模型；搜索基于 active 条件分布。
分解完整风险后，active 条件部分最多增加 \((1-\eta)\delta\)，
遗漏部分最多增加 \(\eta\)。因此上式成立。
它不是 HOTA、IDF1 或未知真实后验的风险保证，也不覆盖窗口交接时丢失的历史。

## 已验证与未完成

34 项独立测试覆盖小场景 Cartesian 枚举、质量界、全动作风险、跨端／时序互斥、
迟到重新评分、未枚举分支恢复、节点追加、重复消息、未来输入拒绝、容量保护、
提交链和篡改拒绝。旧决策回执保持不可变。

冻结检测节点、候选势、分支状态重放和窗口内 birth/kill 已接入，见
[有限窗口跟踪接入](forest-tracking.md)。尚需完成三个种子训练、跨窗口稳定 ID、
同资源多假设基线和双真实数据集效果。
虚拟 carry anchor 是接口保留项；当前没有把单 MAP 窗口交接冒充完整后验延续。

在 transvision 根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/event_track_v2x/test_identity_forest.py -q -p no:cacheprovider
```
