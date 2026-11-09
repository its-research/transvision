# K：指标口径与数据暴露声明的收口要求

本记录来自2026-11-03只读核对原论文合同，不新增数值结果、冻结日期或科学资格。原thesis文档不修改。

| 结果范围 | 指标口径 | 当前可引用的范围 | 禁止混用 |
|---|---|---|---|
| SPD既有21序列、3316帧输入来源消融 | 原AMOTP为米，越低越好；HOTA原oriented BEV IoU、AMOTA原XY距离合同 | thesis README中的车/路/双端工程表及其原限制 | 不能把该表当新恢复方法结果，不能改为V2V的IoU方向；已见val保留探索性声明 |
| V2V原生DMSTrack完整组件 | 原AMOTP为平均3D IoU，越高越好；37个实际recall阈值、原40分母合同 | 已核对最终GPU组件索引及七引用报告；原来源与预测范围保持 | 不与SPD AMOTP合并跨数据集均值；不把原生小2D高度未匹配预测ignored规则下的FP当计入全部预测的统一FP |
| 当前RBF方法正式表 | 依原两数据集协议；同输入/同冻结参数/完整种子序列 | 尚须完整回放、独立评价及成本证据 | 不能拿作者原生组件填入当前RBF结果格；不能用局部0080几何诊断或软件测试填指标 |

来源：/Users/lbin/thesis/tracking/recover-before-fuse/README.md第26–41行明确SPD范围；experiments.md第180行起及v2v4real-official-test-protocol.md第55行起明确跨数据集口径、test隔离和已见声明。V2V原生最终来源见J-native-pose-contract-resolution-20261103.md及112 J-existing-native-GPU-component-records-20261103/record-reference-SHA-audit.json。

## 数据暴露审计，不能由汇总记录补造盲测资格

1. 原官方train/test内容重叠必须披露；用户已经明确额外物理session映射不是本项目准入前置，不重新发明该门槛。
2. 原生作者baseline的完整test性能、当前方法的实际test性能、输入格式/来源/几何校验，以及保持原函数与预算的实现纠错，必须分别记录首次暴露及用途。已经阅读的历史作者test结果不得称为从未见过；也不能仅因读过baseline就无证据断言当前模型用test训练。
3. 对当前正式RBF方法，应以既有带时间的配置、checkpoint、输入、评价规则及源码SHA证明冻结和选择顺序；不得由本次文档日期生成新的“首次冻结”。方法/超参数若在已见test性能后调整，按原合同标为探索性，不能复称独立确认。纯确定性复跑或评价器纠错则保留原attempt、变更理由和等价性证据，不能自动等同重新调参或自动豁免。
4. train内选择、三种子和全序列配对统计仍须原完整证据；原population velocity prior不称测得零速度，候选存在概率校准不称联合身份概率校准。
5. 正式汇合时每个表格单元格绑定dataset/split/类别/ROI/频率/评价器/单位方向/配置与checkpoint来源/SHA/完整预测及独立收据/暴露与变更历史。缺项明确未完成，不用文献数字或软件准备状态填表。

本记录只明确最终核对条件，尚未完成逐表/逐尝试暴露审计、当前RBF全指标或全K验收；不改变任何活任务或原容差。
