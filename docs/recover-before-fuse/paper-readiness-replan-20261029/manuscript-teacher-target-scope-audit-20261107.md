# 主稿teacher目标范围核对：2026-11-07

本轮读取原实验合同及当前main.tex。主稿摘要/实验/局限仍明确完整方法验证pending、SPD seen-val负结果、V2V已知train/test重叠与公开方法未对齐，不新增成功或SOTA结论。

发现并最小修正main.tex第140行的一处目标混淆：旧文“A train-only teacher can compare identity loss…”不能准确描述当前冻结数值目标。已核对112原冻结oracle SHA256 `2aecc66e8c6667b8cf3dc1b6b70eda596049d9eab5cfc2a63ab29e9d5298eef5` 中verify_candidate：`target = weight * (first['risk_bound'] - second['risk_bound']) / max(1, done)`；其weight来自因果窗口内组件作用域，保留有符号model progress。

主稿现明确teacher目标是因果窗口权重乘以模型风险界前后下降量，再除以max(1,charged steps)；保留负增益，不等于GT身份损失或真实风险下降。优先级训练及独立收益pending、pairwise ranking仅proposed的边界保留。

实际修改 `/Users/lbin/thesis/tracking/recover-before-fuse/cvpr2017/main.tex` 并读回确认。Mac仅文本查看/编辑；未执行LaTeX、SHA、数值、模型、统计或新实验，未提交推送，未停止/重启现有任务。本审查不是完整理论证明审计或整篇发表就绪证明。

补充核对main.tex的truncation regret与supplement.tex第15–50行：共同动作集、TV定义、条件近似项、模型误差项和解码delta系数在两文一致；本轮只作手工文本/推导范围审查，未新增数值证明或完整理论资格声明。共同动作集/有限有界损失/未知真实模型误差等前提仍保留。

补充材料第133行teacher目标说明已同步主稿：signed causal-scope weighted model-risk-bound decrease per charged step，不是GT身份风险收益；要求记录正/零/负目标频率及out-of-fold来源，训练选择仍pending。仅文本编辑与读回，未在Mac计算、编译、提交或改实验配置。
