# V2V4Real 官方划分与合并 vehicle 评价

2026-10-05，用户要求按原论文或公开复现路径推进，不再强制额外的物理采集会话独立性证明；随后明确采用原论文合并 vehicle 的评价标准。本页记录当前决定，替代旧协议中的对应限制。

决定 ID：`v2v4real-official-benchmark-vehicle-v1`。机器可读决定见
`configs/event_track_v2x/v2v4real-official-benchmark-20261005.json`。

## 当前执行规则

- 使用官方 train 和完整 official test；不自行生成新的 v2 划分。允许按官方基准保留原 train 成员，并披露已知重复内容。已有 overlap-controlled v1 单独保存，不作为官方基准的强制前置。
- 缺少原始 recording/session/bag 映射不再阻止该基准的训练、冻结、推断、评价和论文表图，也不再是本 goal 的未完成项。会话来源追查移出关键路径。
- V2V4Real 将 car、van、pickup truck、semi-truck、bus 合为 vehicle。新 GT、校准标签、评价与汇总必须采用同一合并语义；不把只含 Car 的旧标签或旧 car 指标直接改名。SPD 仍评价 car。
- 原始字符串不按论文类别英文名称臆造：冻结官方 `project_world_objects` 排除显式 `obj_type == 'Pedestrian'`，合并 `vehicles` 中其余对象。本机旧标签已见 Car、Truck、ConcreteTruck，保留原始类型用于审计；缺少类型仍报错，不默认为 Car。
- 开发选择使用已经冻结的 train 内序列分组；双方 CAV 与同一序列留在同组，不声称已证实的物理会话。公开 validation 不可获得时，沿用已记录的 train 内选择合同并注明差异；不以 official test 替代训练时的 validation。
- 继续使用 `v2v4real-nominal-10hz-formal-v1` 的标称时间和因果到达规则；全类别 raw score >= 0.05 / top64 候选竞争与三个种子不变。合并发生在标签及评价语义层，不能提前删掉非 Car 候选。新任务同时绑定本决定及新类别配置，旧配置与产物哈希不改写。
- official test 不用于训练、校准、阈值或 checkpoint 选择。预测与 GT 评价分离；按真实覆盖与指标记录结果。复用适用的已有模型和产物，不因本决定重复训练或重做诊断。
- 历史 `physical_session_provenance_verified=false`、`formal_independent_test_eligible=false` 等回执保留原样。它们不再全局禁止当前官方基准路线；也不能把本次授权写成“物理会话独立性已证实”。

## 公开依据与报告口径

[原论文 §5.1](https://arxiv.org/html/2303.07601v2#S5.SS1)列出 train / validation / test 为 14,210 / 2,000 / 3,986 帧；
[官方 README 的测试说明](https://github.com/ucla-mobility/V2V4Real#test-the-model)要求推断时将 checkpoint 配置中的 `validation_dir` 指向 `v2v4real/test`。
本次检查这些公开说明，未发现额外提交物理会话映射的前置要求。

原论文 §4.1 合并不同车辆类型，并以 ego 周围 x=[-100,100] m、y=[-40,40] m 为评价区域。V2V4Real 结果采用合并 vehicle 口径；只有输入、ROI、评价器和指标定义一致时才与公开数字直接比较。原生 AMOTA / AMOTP 与项目 HOTA / AssA / IDF1 分开注明定义，保留 RBF 标称时间及注入延迟的实际设置，不将其冒称实测通信。

已记录的 294 种跨 split 相同点云及全量 YAML 来源审计继续作为数据集限制披露。官方基准实验可以完成；证明无物理采集会话重叠不再是本实验的完成条件。

## 优先级与接续

本决定在 V2V4Real 范围内，替代 thesis 官方 test 协议中仅 car 的要求，以及旧 docs、台账和心跳中“先取得物理会话映射／解决全部重叠才可推进”的条件。旧严格独立性审计工具、旧 car 结果和全部冻结文件不改写；SPD 不受此修改影响。

复用已拉取的官方归档、已验收的官方 test 输入投影和合适的检测器／raw 输出。先核对模型实际训练类别，再生成新的 vehicle GT 与 train-only vehicle 校准，接续缺失的 test 缓存、RBF 模型、推断和独立评价。旧6序列 strict-Car 校准标签与 overlap-v1 train-only 预测不能充当新 vehicle test 的标签或预测。

原检测器的实际训练范围须明确为22个 detector-fit 序列及6个 calibration 序列；若复用，不能称32序列全 train 重训。需要最终全 train 拟合时使用既有选择结果并另记任务，不重做已完成且仍适用的 raw 导出。
