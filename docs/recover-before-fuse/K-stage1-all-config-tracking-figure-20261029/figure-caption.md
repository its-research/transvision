# 第一阶段全部原始跟踪结果图

![SPD 第一阶段全部13配置、10项原始跟踪指标](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/K-stage1-all-config-tracking-figure-20261029/SPD-all13-original-tracking-values.png)

**建议图注。** SPD 已见验证集的探索性机制结果。每个配置均覆盖原21序列、3,316帧，评价对象为 car；各点直接取原始完整数据集报告，经来源SHA与独立指标回执绑定。M0、M2、M3、M4分别展示1337、2027、3407三个原种子；M1为单个确定性 all-unmatched 配置，不构造重复种子或标准差。M3种子2027的历史严格预测复现未通过，以十字和星号保留原值，不作为合格复现或合格三种子推断。HOTA、AssA、DetA、IDF1及AMOTA由原0–1数值乘100显示；AMOTP为米，越低越好。FP、FN、IDS、Frag取每次运行自身的原nuScenes最大MOTA召回阈值，阈值可不同，不能解释为统一阈值下的计数。各面板使用独立纵轴范围，横向偏移仅用于区分种子；图中没有均值、置信区间或拟合线。

图保留全部原配置及负结果。M4的碎片数下降与HOTA、IDF1、AMOTA下降和FN增加同时存在；这一组观察不能支持恢复机制已有独立质量优势。已验配对统计仍以既有完整序列bootstrap回执为准，本图未重新计算指标或bootstrap。SPD已见验证集的探索性范围、V2V完整测试范围和完整A–K论文完成门槛保持分开。

可导出的矢量图：[SVG](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/K-stage1-all-config-tracking-figure-20261029/SPD-all13-original-tracking-values.svg)。

来源与校验：[全13配置原表](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/K-stage1-component-table-20261027/stage1-thirteen-config-bound-component-table.json)、[实际SVG点位与原报告直接校验](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/K-stage1-all-config-tracking-figure-20261029/figure-original-source-and-svg-readback.json)。

本次完成论文呈现准备，新增科学验收通过数为0，完整K及A–K仍未完成。
