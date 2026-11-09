# 已有模型复用与论文声明边界

2026-11-03 02:43 北京时间。此为已有记录的字节与声明一致性复核，不是新增模型、276份原收据或数值前向验收；未扫描大缓存、加载模型或追加训练。

## 可复用的已验范围

- 既有SPD与V2V三种子各自保留原checkpoint、权重、缓存及输入绑定。不能仅凭 `paper_eligible=false` 推导必须重训；实际输入或训练/选择协议不符时，才按原C合同判断必要训练。
- SPD固定train Top1/K4已有三种子×两配置，共6组合、276份原序列收据；已有来源汇合把它们与当前登记的checkpoint/model绑定。原目标文件10月5日的“剩100序列”不是新增执行清单，不重跑这段已验范围。
- 138是种子—序列配对数，不是138个独立物理序列；原唯一训练序列为46，schedule为7445事件。677744/2364754是跨种子数值行实例，不能当独立样本数。
- 复用这些train验收不能关闭正式G/H预测、独立跟踪指标或I同资源比较，也不自动关闭完整B/C。

## 论文中必须保留的限制

| 证据 | 可支持的范围 | 不能替代的结论 |
|---|---|---|
| 原逐行前向/因子数值验收 | 实现与独立数值参考一致、原模型字节/输入可绑定 | 校准概率、真实联合身份后验或精确联合NLL |
| 当前模型训练元数据 | 原局部部分标签目标与各自选择/全量refit来源 | 将 `local_partial_label_objective=true` 改称精确joint NLL |
| 已有候选存在概率校准报告 | 原固定候选域内的二元存在概率诊断 | 联合身份概率校准、恢复风险证书或跟踪指标 |
| SPD原val结果 | 已见val的探索性比较 | 独立未见验证泛化 |
| V2V官方test结果 | 原官方命名范围，并明确已知train/test内容重叠 | 独立泛化；也不额外强加物理会话映射作为原官方基准前置条件 |
| 模型相对teacher/界与精确事件 | 在冻结模型及原假设下的目标/决策性质 | 真实身份风险保证或总体任务收益 |

原metadata flags（包括paper_eligible、calibration_guarantee、strict_pipeline_isolated_selection）原样保留，不改写false来关闭门槛。若实证不支持恢复的独立优势，按原目标保留负结果并调整主张。

## 当前下一步

1. 原模型/输入绑定匹配的已有A验收直接复用；保留主森林和teacher正在进行的独立收口，不因字段命名重新训练。
2. 在正式G/H或I启动前，继续核对各方法实际使用的种子权重、特征、运动/协方差、cache及schedule；旧MHT缓存不匹配的问题不能由这份复用记录豁免。
3. D仍须完整teacher→优先级holdout/checkpoint→独立接入；受限H支持必须有自己的teacher/priority，不能移植unrestricted目标。
4. K报告中将候选存在校准、联合身份模型性质、恢复机制和最终跟踪收益分开陈述；正式指标、同资源至少三次隔离重复及完整统计仍按原A–K要求完成。

## 本轮复核依据与限度

- `BC-six-model-reuse-and-claim-boundaries-20261031/current-six-model-registry-and-limits.json`，SHA256 `44ee9ccaa7afdd2703e4bc5a458b4c51903e419cdb40c3cb486b38757fc69200`。
- `BC-current-model-fixed-train-pair-source-join-20261031/complete-existing-train-current-model-pair-join.json`，本轮112读回SHA256 `bf6ec7dabfddbddd6e76b03dac4ecb2e3c6439f027ae282c63c5cc2471c725aa`。
- 原目标明确要求披露V2V内容重叠，SPD val保持已见探索性，不重复已验且适用的训练。
- 112证明 `/home/lbin/rbf-approved-V2V-unit6-control-preparation-20261101/BC-existing-records-claim-scope-audit-20261103.json`。只复核了上述记录字节和六个模型/六个基线组合的声明配对一致性；本轮没有重新读取checkpoint/权重或276原收据，不宣称新增科学资格。
