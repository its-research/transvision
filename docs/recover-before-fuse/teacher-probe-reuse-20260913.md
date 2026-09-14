# 同一决策内的教师试算复用

## 目的与正确性边界

`AllocationTeacherTracker` 在每个分量分配点为所有候选生成反事实标签。原实现会反复试算未变化的分量。本次为这些确定性试算增加有界缓存；模型势函数、候选、预算、分配策略、损失与风险公式均不变。

在一个事件的分配循环中，已到达因子、配置及模型固定，只有被选中的分量改变搜索状态。若当前分量的搜索状态、操作、实际生效的容量和风险上下文都相同，则确定性试算给出相同的更新状态、计费步骤及风险上界。因此可以复用该标签，但不能跨事件、因子修订或其他未入键的语义变化复用。这里用完整字段相等判断，不用学习置信度或近似状态摘要。

容量键使用 `min(局部上限, 当前分量占用 + 全局剩余量)`，与实际执行器的裁剪规则一致。预算影响操作选择，操作仍在每次试算前重新生成。每个分量最多缓存一个状态；缓存内 active／frontier 句柄引用总数最多 65,536，缓存于事件退出时释放。返回字典独立复制，调用方不能通过修改返回值污染后续标签。

## 验证范围

79 项相关回归通过，耗时 27.13 s，覆盖教师数据导出、训练夹具、资源比较入口、持久化分量和缓存行为。其中包括：

- 3 个种子、两种缓存上限下，缓存与禁用缓存的教师逐项比较预测、全部候选标签、分配轨迹、风险界、分量结果和请求预算。
- 有效容量、决策时刻、损失作用域、损失权重或操作变化时不复用旧结果。
- 原试算的 SQL／缓存回滚、异常后的清理与精确重试、未来输入拒绝、旧输出幂等。
- 诊断入口只修改数据库副本，拒绝摘要不符、预测头不符或仍有事务旁路文件的源数据库。

回执：`/private/tmp/rbf-teacher-probe.PeVfiw/teacher-memo-and-backends-20260913.xml`。SHA-256：`fc57f9061b5a501b16f65727e0aa5a6cbb40b2b4067f462dd39d8c5f73ca6556`。

## 真实状态诊断

输入来自已中止 teacher 的固定因子数据库：452 个观测、13 个已提交事件。它保留原 CPU 身份模型产生的势，本次不替换为新 A100 权重；这样对照只改变试算缓存。输入为 `/private/tmp/rbf-teacher-probe.PeVfiw/probe.sqlite`，SHA-256 为 `eda42976d9c16be56bd8fc4714517377e350febaad833e0564b7c4de31401c20`，原预测头为 `83833a7fcd186a836a7d26fdaa6c85def092e19e2e89b646322d63a33e8f2c1b`。

两种模式各复制数据库，在参考时刻和决策时刻分别加 1 μs 后追加同一个无新输入的诊断事件。不读取新观测、GT 或未来帧，不修改原数据库或原事件。这不是新增的官方训练帧，也不计为完整序列回放。

启用缓存耗时 54.018 s，禁用缓存耗时 215.876 s，均计入 `cProfile` 开销。这是先启用、后禁用的单次副本对照，约 4 倍的耗时比不是随机重复资源基准，也不外推为部署加速比。两种模式的正式搜索均计费 256 步。

| 计数 | 启用缓存 | 禁用缓存 |
|---|---:|---:|
| 教师请求 | 2,304 | 2,304 |
| 实际试算 | 264 | 2,304 |
| 复用结果 | 2,040 | 0 |
| 峰值缓存句柄引用 | 7,904 | 0 |

预测文件逐字节相同，SHA-256 为 `8b97153416a1a18f2b8c5820b6527fe1be6200c4068cc806d577b9f8d3d4820b`。完整审计逐字段比较后，只有四个缓存执行计数字段不同；全部候选标签、正式分配轨迹、分量状态、风险界和因子摘要一致。

结果位于私有根目录 `/private/tmp/rbf-teacher-probe.PeVfiw/`：

| 产物 | SHA-256 |
|---|---|
| `memo-event-v1/receipt.json` | `27d94828439151efacd5a961f7bacf4b56ebfbf31325e910c725f9efee378d21` |
| `memo-event-v1/event.pstats` | `5e7ae4029da4a3618b0d72f2dca5a20e987f65f27b544af772b2247d761a6767` |
| `uncached-event-v1/receipt.json` | `7cfbf047e09ae027e073ea39eeb295e39cc9d6ad27f592952b4f21aeac0e9ac6` |
| `uncached-event-v1/event.pstats` | `d3603cb16dcfdadda6f5dccee21b2870f9183b6975584302f3fa22985e1e13ce` |

模型 regret 上界仍为 `0.9950920209910156`，接近归一化损失的平凡上界 1。它不是实际身份错误率，也不能提供本事件的有用性能保证。剖析还显示重复优先级统计、前沿上界排序和配分函数求和占用较多时间。缓存复用没有解决松弛界过宽或完整序列的可扩展性问题。

全部 2,304 个目标值的范围为 `[0, 6.8774877631646864e-18]`，没有绝对值超过 `1e-12` 的目标。135 节点的分量占用全部 256 个正式搜索步骤，最终遗漏质量上界仍为 1；另一个 133 节点分量的前沿已达 4,096，但未分配正式步骤。本事件说明当前确定性策略和「一步模型风险界下降」目标存在无有效进展的局部状态，不证明整个 train 的标签均退化，也不支持学习式分配已改善真实跟踪。继续全量拟合前，需处理界的可计算紧度、饱和状态下的训练信号及跨分量预算分配问题；不得把接近零的 MSE 当作方法有效。

## 复现

从 transvision 根目录执行，输出路径必须不存在：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /private/tmp/eventtrack-v2-checks.bYZm1l/bin/python \
  tools/event_track_v2x/profile_allocation_teacher_event.py \
  --database /private/tmp/rbf-teacher-probe.PeVfiw/probe.sqlite \
  --database-sha256 eda42976d9c16be56bd8fc4714517377e350febaad833e0564b7c4de31401c20 \
  --prediction-sha256 83833a7fcd186a836a7d26fdaa6c85def092e19e2e89b646322d63a33e8f2c1b \
  --output /private/tmp/rbf-teacher-probe.PeVfiw/memo-event-NEW
```

禁用缓存时增加 `--disable-probe-cache`，并指定另一个新输出目录。工具保留完整诊断预测、审计、运行摘要及剖析文件，不上传 ClearML。输出标记 `scheduled_replay=false`、`latency_benchmark=false`、`tracking_validation=false`、`paper_eligible=false`，不得填入论文跟踪主表或在线延迟表。
