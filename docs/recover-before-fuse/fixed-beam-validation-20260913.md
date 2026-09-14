# 固定宽度 beam：阶段验证记录

本轮在 transvision 实现并测试两种不可恢复 beam，以及正式 SPD V2 回放入口。
没有修改论文项目代码，没有提交、推送、远端同步、真实数据训练或 ClearML 发布。

最终相关回归 **239 项通过，0 跳过，约 9.68 秒**。回执为
[fixed-beam-regression-20260913-v2.xml](../../work_dirs/recover-before-fuse/fixed-beam-regression-20260913-v2.xml)，
SHA256：`7475f4e7fbe544d3192756d35f271c4a1729e0112db640dee4acc02f00a5154e`。
两个更新后的运行入口 `--help`、中文技术文档检查和 `git diff --check` 均通过。
完整方法说明中的 `相同学习势` 被文案检查器误识别为称呼，经人工确认保留。

14 个相关测试文件覆盖 beam、可恢复持久化后端、共享缓存、训练监督和权重接入。
首轮 239 项通过，0 跳过，约 9.31 秒，回执为
[fixed-beam-regression-20260913.xml](../../work_dirs/recover-before-fuse/fixed-beam-regression-20260913.xml)，
SHA256：`97765a2b5c9334f7b8a0f8d73270e9b500a11668b7bee167561be4c064e9d6a9`。
后续显式防止关闭风险回退时被浮点求和误触发，最终回执另存，不覆盖该阶段记录。

本轮没有重新运行全部 EventTrack 测试。上一阶段 1289 项通过、1 项跳过的完整
回执保留为历史证据，不用于声称新增 beam 的全套验证已完成。

## 已经得到的机制证据

- 四个持久化后端在同一训练权重和 V2 夹具上产生相同的原始行势摘要。
- Top-1 可以因历史剪枝而失去后续纠错机会；同例 Top-2 保留候选后可以作出正确
  选择。因此该例不能证明恢复方法优于增加普通 beam 的宽度。
- 整批排序可以避免忽略本批已到达后续行的错误，但不能恢复前一事件已删的身份类。
- 旧乘积 Top-K 预选后再进行当前批次排序，不等于完整笛卡尔积的新扩展 Top-K。
  已有反例测试防止将其误标为精确全历史排序或经典 MHT 复现。
- 连续跟踪、分量合并、正常关闭重开、旧输出不可变、状态重放和容量失败回滚通过。

数学保证范围、候选域、资源成本和运行方式见[实现说明](fixed-beam-baselines.md)。
当前没有真实数据 HOTA／IDF1 增益，没有同延迟或同内存曲线。新入口继续标记
`paper_eligible=False`；经典强基线与完整论文实验仍未完成。

## 最终实现文件身份

以下为最终相关回归时的 SHA256，不将已有脏工作树视为冻结 Git 提交。

```text
transvision/models/event_track_v2x/persistent_beam_tracking.py
06388d244905271ad4f69c690cd71a661facec7f61d763a4a07added0f6b8548
transvision/models/event_track_v2x/persistent_component_tracking.py
bce6102462a2c3e661fdb9d066caba6ab70e4ff41bf44ce96988ffc3580ca8c0
tools/event_track_v2x/run_fixed_beam_v2.py
749c1ec2c6f7bc945b30fe60b564957d6fd7f9270776f1a577ca63a72c10a47e
tools/event_track_v2x/run_persistent_forest_v2.py
bdad83080bc8e05cfe71e993bad5b752c300770fa5ec26b0a3b778adb0387d29
tools/event_track_v2x/run_component_persistent_forest_v2.py
7bef323f05cf01ca30bc37e0bbe90f9a4d58218c83ec52e93b73f1e1b8e12374
tools/event_track_v2x/run_trained_persistent_forest_v2.py
33a5dd491e89080ba45ba5265c15f75949f693eb33f8ac6024f2caf5b87b20ea
```
