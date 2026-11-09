# 论文文字与当前实验合同一致性核对

本轮只读核对 /Users/lbin/thesis/tracking/recover-before-fuse/experiments.md、paper-outline.md 与 cvpr2017/main.tex，没有编译、计算指标或改动实验。

原实验合同开头的2026-10-05用户修订与当前goal均要求：SPD为car，已见official val只作探索性评价；V2V4Real为官方test全部9序列合并vehicle，已知train/test内容重叠披露，不能再套用旧car或严格物理会话独立性前置条件。main.tex摘要及scope段已一致；paper-outline.md摘要仍把SPD与V2V都写成car。本轮将这三行文字对齐原合同，结果占位符与未证实边界原样保留。

原大纲在本目录paper-outline-before-class-scope-fix-20261102.md完整保留。没有改候选、阈值、标签映射、评价器、模型、输入、预算或容差；没有改写任何历史预测。main.tex明确完整方法实证尚待完成，不作准确率、实时或SOTA结论；现有checkpoint为局部partial-label surrogate，不冒充精确joint-NLL或校准已证明。

另将旧总对账JSON的当前执行入口指向唯一当前计划及authoritative_current_status，避免旧“运行中/待批准”字段触发重复提交。原JSON完整备份，旧字段仍保留为历史；没有重封科学收据或模型登记，没有新增任何完整资格。
