# 学习式计算分配：阶段验证记录

本阶段在 transvision 接通离线反事实教师、优先级训练分片、三种子拟合、冻结
检查点及 V2 回放。共享搜索操作保持同一实现，学习结果只控制分量选择顺序。
研究代码没有写入论文项目，论文构建工具未迁移。

## 1. 回归证据

| 阶段 | 结果 | 回执 |
| --- | --- | --- |
| 初次相关回归，17 个测试文件 | 290 通过，0 跳过，12.03 秒 | [相关回执](../../work_dirs/recover-before-fuse/learned-allocation-regression-20260913.xml) |
| 完整 EventTrack 回归，视图元数据修复前 | 1399 通过，1 跳过，334.35 秒 | [完整回执](../../work_dirs/recover-before-fuse/learned-allocation-full-regression-20260913.xml) |
| 视图元数据修复后的最终相关回归，17 个测试文件 | 291 通过，0 跳过，11.63 秒 | [最终相关回执](../../work_dirs/recover-before-fuse/learned-allocation-regression-20260913-v2.xml) |

唯一跳过项为 `test_official_golden_cases`，原因是当时的研究测试环境未安装评估器。
它不代表官方评估通过。三个计数属于不同阶段，不能相加。

后续已建立独立评估环境，并执行该测试及 7 个真实引擎合成案例，详见
[评估环境验证记录](evaluator-runtime.md)。后续记录不改写本节历史回归计数。

完整回归期间，额外检查发现只读 NumPy 数组仍可修改 `dtype`，使模型签名不变
而输出改变。随后修复为内部只存不可变字节和形状，每次返回独立数组视图；新
测试同时修改外部视图的类型与形状，确认后续输出及模型身份不变。修复后重跑
相关回归，没有重跑完整 EventTrack 套件，不能把完整回执当成最终源码的全套证明。

回执 SHA256：

```text
learned-allocation-regression-20260913.xml
9ac5d29659391eb3541a1edd704a59723522cc2e66aa713005b17607b3efa9ac
learned-allocation-full-regression-20260913.xml
21b33ad665dc2c2bfce62b1867cad45b4139e9f9220e1ed7b63a20418ec0b0c1
learned-allocation-regression-20260913-v2.xml
9878d6a568eb19d7df5ebeb124d5bc42d8e78b7ec30adf86a96809e84c938172
```

三个新增或扩展入口的 `--help`、中文技术文档检查与 `git diff --check` 通过。

## 2. 机制与数据边界检查

- 独立穷举验证学习排序下的真实模型遗漏质量和最终实际动作 regret 上界。
- 教师反事实前后数据库、共享缓存及行为输出一致；试算异常后完整回滚并可重试。
- 未到达输入在特征生成前拒绝；val 缓存不能用于教师采集。
- 三种子拟合确实更新权重，CPU 重复运行得到相同模型身份；改变留出标签不
  改变拟合权重。留出仅隔离优先级模块，不冒充上游全流程隔离。
- 两个学习模块同时启用，接通身份势、教师轨迹、优先级训练及 V2 回放；六个
  后端逐事件原始行势摘要一致，优先级不篡改候选或因子。
- 正常关闭重开要求同一优先级模型，重复事件返回原回执，模型变更被拒绝。
- 正式训练／加载入口拒绝将小型夹具冒充完整 train 轨迹；文件变更使训练失败。

目标函数、数学假设、运行入口与成本口径见[实现说明](learned-allocation.md)。
训练目标是一操作模型上界进展，不是长期 VoI，也不是 HOTA／IDF1 增益。

## 3. 尚未完成

尚未采集真实全量教师轨迹，也未进行新模块真实训练、全 train 最终重拟合、
同资源强基线比较、SPD 全 validation 或 V2V4Real 冻结后的官方 test。上游数据
仍存在 in-sample 限制，闭环状态分布偏移和真实资源成本尚待验证。

本阶段未进行远端同步、提交、推送或 ClearML 发布。此前远端源码同步仍待
明确授权；当前新增源码也不自动包含在此前六文件的同步请求中。
论文目标保持未完成，所有新入口保持 `paper_eligible=False`。

## 4. 最终相关回归时的实现身份

以下为最终源码 SHA256，不把脏工作树视为冻结 Git 提交。

```text
transvision/models/event_track_v2x/allocation_policy.py
f6289176120cdcad399194293b5b218ed78e92770f1fdd9bb86dba89ba6e5deb
transvision/models/event_track_v2x/learned_component_allocation.py
7433fa9d3538bbc9fcfafee34b015d957db6a641027783a941c7d06e1689f078
transvision/models/event_track_v2x/allocation_training.py
40670e5fe7e0178efabfcea60014243ec03a6e5d795f2e58c775415328fb50aa
transvision/models/event_track_v2x/persistent_component_tracking.py
f1ff9acd7a48116e4f522dcba147a3d691909b5f437a4f4b44d27546c627c3a1
tools/event_track_v2x/collect_allocation_training.py
dc83f5824071497a7c2e9afa0d1f3858af1da6d0cf9eaa3449429961a9a3fa9f
tools/event_track_v2x/train_allocation_policy.py
3a28a6610fc65ee7bbe4731314f6c057389d0e78f90e681d965c72c78d5e8188
tools/event_track_v2x/run_component_persistent_forest_v2.py
7e82efc5ab19d957ebc586cbec278da8c70c4af76828c0cebddc8ba2c80e5ba4
tools/event_track_v2x/run_persistent_forest_v2.py
88683cb1594026787c2dfd7c0d1968a868284df27300ade0869d328797c00a74
tests/event_track_v2x/test_learned_component_allocation.py
8d2da7ffdc691aff5466a3ca5ba314613cbd5429fd2d40fcad3a43bc259b3e8f
tests/event_track_v2x/test_allocation_training.py
f4edbce2c9cb706dbac12c5dff05e97d33294f488861a80186366c3fd99b82fb
tests/event_track_v2x/test_train_forest_identity.py
6d8efd08c97849f45ef4f0a1b98c3cb66b7c10489851d2335c172579c3961c49
```
