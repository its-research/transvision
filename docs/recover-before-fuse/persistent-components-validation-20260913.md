# 持久化分量实现：阶段验证记录

所有本轮研究代码、测试、说明和回执位于 transvision。论文构建工具及其共享依赖
留在论文项目；本轮没有新增或修改论文项目代码，没有 Git 提交、推送、远端源码
同步、真实数据训练或 ClearML 发布。

## EventTrack 完整本地回归

`tests/event_track_v2x/` 共 1290 项：**1289 项通过，1 项跳过，约 335.22 秒**，
没有失败或错误。这是该目录的完整回归，不是整个 transvision 仓库的全部测试。

回执：[persistent-components-full-regression-20260913.xml](../../work_dirs/recover-before-fuse/persistent-components-full-regression-20260913.xml)。
SHA256：`fbb2b1def461a3f19d80dfbeae568603a8d3804a57a0d95cd1e7d50abe987ff2`。

唯一跳过项位于 `test_tracking_evaluation_v2.py:104`，原因为本机未安装官方评价器
运行时。因此本轮没有完成官方评价器端到端复验。下述 180 项相关回归包含在本次
完整回归中，不另行相加；历史回执和历史实验结果均未覆盖。

## 已完成的相关回归

12 个相关测试文件共 **180 项通过，0 跳过，约 8.09 秒**。覆盖持久化联合后端、
分量存储与输出、V2 接收、冻结旧势下的新证据恢复，以及训练／推断一致性。

回执：[persistent-components-regression-20260913-v3.xml](../../work_dirs/recover-before-fuse/persistent-components-regression-20260913-v3.xml)。
SHA256：`5ab15fad21d1278f33636542c24cc914044f84bc361240f287ba60226e2cc1ce`。

此前的 176 项、177 项回执保留为阶段记录；它们是本次覆盖的子集，不累加计数。
测试只使用穷举场景、合成标注和正式 V2 格式夹具，不代表真实 train／val 性能。
分量 CLI 的 `--help` 和 `git diff --check` 通过。

中文技术文档检查没有错误。两个建议项经人工确认：`相同学习势` 中的 `同学`
属于字串误报；`截止条件` 表示 deadline，保留原义。证明条件、模型内上界、工程
验证和真实实验状态按中文技术文档规范分开说明。

## 实现文件身份

以下 SHA256 对应本次 180 项回归时的实现，未将整张脏工作树视为冻结提交。
逐次实验入口还会记录其直接依赖源码的哈希，并在运行结束时复核。

```text
transvision/models/event_track_v2x/persistent_component_store.py
c1dc23dd933664a73215e530f0a5e038b3e48389621c18b3f8365c1b4b5a316a
transvision/models/event_track_v2x/persistent_component_tracking.py
197d0cc0991d3a56d7afc96a868c3ce0c6ba42d953482474c668280e21903d39
transvision/models/event_track_v2x/persistent_forest.py
952bf1ac1f9fe440e0f053b43f1fb021e52d07f3c72a69f8505220e7eba4a2c8
transvision/models/event_track_v2x/persistent_cache_stream.py
a25e3cb5d0a760f56e27f5fde23af5d1bff89e4783798bf7ada734f1703fe0f5
tools/event_track_v2x/run_persistent_forest_v2.py
caca6ef18901634cff4e2b7fb92a7b1562fd62bbcb3b6f29d19b8b228eb137de
tools/event_track_v2x/run_component_persistent_forest_v2.py
700ef279c0e1a638fa5fa36258fd7698a4ca2215e598897c294d942738b63edc
```

## 方法证据边界

[实现与推导](persistent-components.md)给出完整身份类求和、前沿重叠计数、分量
分解及实际动作风险界的条件。测试验证这些条件下的计算，没有证明真实后验
校准或 HOTA／IDF1 增益。恢复事件也不直接等于纠错事件。

仍需真实 train 规模检查、正式三种子训练与隔离选择、学习式计算分配、同资源
强基线、SPD 全 validation 和 V2V4Real 官方 test 方法验证。新运行入口维持
`paper_eligible=False`。本轮源码的远端同步需要相应授权；旧 19 项结果的 ClearML
发布授权不自动涵盖新源码、训练权重、GT 或完整预测流。
