# 联合排序 beam：阶段验证记录

本阶段在 transvision 新增不可恢复联合排序 beam，移除旧组合 Top-K 预选对当前
新证据的限制。默认入口为 `run_fixed_beam_v2.py --selection joint`，原 `batch`
和 `node` 仍可显式运行。研究代码没有写入论文项目；论文构建工具保持原位。

最终相关回归 **263 项通过，0 跳过，约 11.57 秒**，覆盖 15 个测试文件。
回执：[joint-beam-regression-20260913.xml](../../work_dirs/recover-before-fuse/joint-beam-regression-20260913.xml)。
SHA256：`94ddf1521ed74f40e0931b48cc30ad33aad81d5eaea7f17914be84bb51b718a7`。
运行入口 `--help`、中文技术文档检查和 `git diff --check` 通过。

## 1. 已验证内容

- 延迟笛卡尔积游标的完整枚举、排序、并列项和最大剩余分数，与独立穷举一致。
- 旧 Top-2 预选会排除当前最优完整扩展的反例中，联合排序找到了该配置。
  对照的原始行势相同；这不是对历史已删分支的恢复。
- 三分量合并、多个宽度与随机势、旧行重评分，保留类及绝对权重与独立完整
  父森林穷举后筛选旧保留域的结果一致；实际动作 regret 未超过记录的模型界。
- 迟到观测、原始 CI 重放、120 次输出中途合并、正常关闭重开和旧输出不可变。
- 三类容量失败回滚原始观测、旧行权重、成员映射、状态及输出。初始旧组合
  的一次评分也纳入分量及全局候选评分计数。
- 五个持久化后端在同一小型训练产物上产生一致的逐事件行势摘要；三种 beam
  都通过 V2 预测及封存数据库重开检查，正式入口拒绝测试权重冒充全 train 权重。

完整假设、候选域与停止条件见[联合排序推导](joint-beam-ranking.md)。浮点裕量
不是区间数值认证；上界只约束固定模型，不是 HOTA／IDF1 或真实后验的保证。

未重新运行全部 EventTrack 测试。原 239 项固定 beam 回执与更早的完整测试回执
继续作为各自阶段的历史证据，不能与本阶段数量相加或替代本阶段验证。

## 2. 未完成项与操作范围

尚未完成新身份模型的真实全量训练、学习式计算分配、经典 MHT／JPDA／PKF／
矩匹配对照、严格同资源恢复消融、SPD 全 validation 和 V2V4Real 官方 test 验证。
操作计数不表示延迟等价，beam 宽度不表示总 RSS 或磁盘上限。没有真实跟踪增益
或论文创新成立的结论；运行入口保持 `paper_eligible=False`。

本阶段没有提交、推送、远端源码同步、读取真实评估载荷、发布 ClearML 或上传
GT、模型、预测流。此前远端源码同步被权限审核拒绝，仍待明确授权；不能通过
本地回归记录将其标为已运行远端实验。

## 3. 回归时的文件身份

以下为回归时的 SHA256，不把现有脏工作树称为冻结 Git 提交。

```text
transvision/models/event_track_v2x/persistent_joint_beam.py
629c997e8cdb1857c6632d6395eafd6f9aa13dee21b8f47b9d62704a68df8491
transvision/models/event_track_v2x/persistent_beam_tracking.py
cb84d36715a4a158a8acf8d8cb92505865816847c10336b4418a9b589bdadf7e
tools/event_track_v2x/run_fixed_beam_v2.py
7a66097f18a4ac36d29027a73a0b0b2f79f6c161a489271571cf8c9ecb739d6d
tools/event_track_v2x/run_persistent_forest_v2.py
59c5692b28066835456796723aaab2bf778ce5c51381766a27e1aad8451cee06
tools/event_track_v2x/run_component_persistent_forest_v2.py
d10dc06b7fec35b09aa739545b066f9c5021addb7ae3f2b5ee822d334e63d39e
tools/event_track_v2x/run_trained_persistent_forest_v2.py
89040dfc4542deae7af047b8c1fab0e25125cec0cdb6552cc67f40b6f6f66ee0
tests/event_track_v2x/test_persistent_joint_beam.py
9f46d94e27dc5d8e8497e6eafd768dd9540373ef094fb55b9889839e4b7e24b7
tests/event_track_v2x/test_persistent_beam_tracking.py
cfd96d3e139a5f48477a8057d400a881263cdf5fe576030e7ea9a803838f3acc
tests/event_track_v2x/test_fixed_beam_cache.py
b0774f8f3395b0c65a8a8fb43a177547e300429432160b892b28f693049e8534
tests/event_track_v2x/test_train_forest_identity.py
e4913cd706bce5cb18a016ea67690f210dd9c8d25c0f757f801e86ba5063c887
```
