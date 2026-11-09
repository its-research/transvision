# 论文主文证据边界审查

2026-11-06，只读检查/Users/lbin/thesis/tracking/recover-before-fuse/cvpr2017/main.tex及本仓库paper-commit-readiness.md。未编译、统计、修改论文外部目录或运行Mac计算。

主文摘要与结论明确完整方法效果和发表就绪仍pending，未将运行、单序列资格、软件测试或文献数字填成整体验收。表格P明确是Pending而非零或失败。现有负面探索性诊断仍保留，没有因后续准备成功改写为收益。

主文已区分模型条件风险界与跟踪指标；SPD已见val为描述性结果；V2V已知train/test重叠明确披露；原DMSTrack AMOTP为3D IoU、方向较高为好，不跨数据集平均。上述文字边界不能替代真实桥接及完整指标验收。

需在最终数值填表前再审核第94行关于sequence-isolated training防止泄漏的表述：它最多说明训练内分组/选择的隔离，不能据此推断已知train/test内容重叠消失；必须与后面的公开重叠披露一致，并在模型来源合同证实后限定范围。当前未改该句或宣称发现训练违规。

论文仍缺完整priority资格、双数据集全集方法/消融、强单端与同资源MHT、恢复正反失败轨迹、最终资源曲线及逐序列统计。已有Stage1统计不重跑、不扩张为完整K项；软件提交回执的测试数量不换算实验完成率。

后续填表规则：每个结果绑定配置、seed、split/序列全集、预测产物SHA、独立指标版本/收据与资源合同；不将teacher目标改进称为GT身份恢复收益，不把单schedule耗时当A–K ETA。主文审查通过只表示没有在本次查看范围发现已填入的整体成功宣称，不能判为论文完成。

## 后续窄幅文字修正

已将main.tex第94行原“Frozen detection features and sequence-isolated training prevent a stronger detector or leakage from being credited to inference.”替换为两句明确协议要求：匹配推理比较必须固定检测和特征；train-only序列分组与checkpoint选择需单独核验，并不能证明已知重叠的V2V train/test内容互斥。原后续pending标记及实验协议中对已见SPD val、V2V已知重叠的披露均保留，现场读回一致。

这次仅编辑并读回该段文本，未编译论文、计算SHA或实验指标、训练/重跑模型，未提交或推送。此前上方“未改该句”的描述属于初次只读审查历史，以本段更新为准。修正文案不构成任何输入/模型资格或论文科学结果通过，也不改变完整A–K目标。
