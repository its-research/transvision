# teacher2027前提证据接续合同（静态审阅，2026-11-05）

本次仅读取现有源码及112已经保存的元数据结果，没有执行准入器、数值检查、模型或创建teacher。

## 已有证据及其边界

112已有结果 `/home/lbin/rbf-approved-V2V-unit6-control-preparation-20261101/teacher2027-mainforest-metadata-source-readback-once-20261103/result.json` 明确记录：来源Task为abaf7070a8ac4a5597dfedf88f7c8806；登记归档7311360字节、341成员及全部成员SHA已独立读回；46条序列/数据库/报告来源绑定可用。该结果同时明确teacher_prerequisite_qualified=false、teacher_task_created=false、numerical_or_semantic_acceptance=false。读取此结果不增加新的来源或数值验收次数。

## 原前提入口要求

源码 `tools/event_track_v2x/rbf_final_refit_teacher_prerequisites.py` 的gate要求如下，准备兼容来源时必须逐项保留：

1. 独立driver的source-freeze及全部源码SHA与原绑定一致；种子2027最终模型、checkpoint、完整NN数值验收来源严格匹配，原旧模型不能替代最终refit。
2. 完整接受收据符合原KIND，46唯一序列、7445事件，原语义ATOL/RTOL=1e-8；全部TRUE_FLAGS和FALSE_FLAGS原语义不变。
3. byte-admission与接受收据SHA绑定；原执行ledger中必须唯一登记原接受收据及其SHA。
4. 原source_gate及binding.validate通过真实completed Task、登记产物与最终模型谱系核对，平台General/recipe_sha256、General/plan与原接受来源一致。
5. 46份逐序列报告与byte-admission的数据库SHA、事件数、原始节点数一致；structure、causal、fresh_state、conditional_action_search、runtime_raw_cache203_context五项完整范围均可回溯。容量未决不得计为最优解；上下文不可读取GT/test。
6. 状态、预测、恢复、动作及容量计数必须从各分项严格求和匹配完整接受收据，不能用元数据条目数代替。

## 当前接口缺口与下一步

已读元数据索引提供sequence_id、database_sha256、rank、report_sha256、semantic_mode，不能单独满足上述逐序列事件/节点、五项原报告、完整接受收据、byte-admission、ledger及最终模型核对。因此下一可执行准备是：定位已经合格的2027完整接受收据及其全部来源文件，列出精确路径/SHA映射，明确其与原前提入口的字段差异；如需新适配器，应只还原已验证字段并严格拒绝来源/计数/范围不符，不修改原科学判据。

只有所需文件齐备且来源/接口检查范围已获授权，才运行一次前提资格检查；前提通过也仅证明main_prerequisite_verified，不证明teacher_runtime_or_targets_admitted。不能因46条元数据来源已读回就创建teacher，也不能把当前主森林Task当teacher。

其他关键路径继续：1337/0056原活任务保持运行；成功后原7文件SHA/EOF及完整收集，随后0040、0094各自资源准入。3090仍停用。Mac仅文本审阅与记录，无实验计算，无提交或推送。

## 18:44来源路径定位补充

已接受收据现存112 `.../teacher2027-prerequisite-format-audit-20261103/accepted-supplied-train.json`，1862字节；明确记录40新序列为four_frozen_plus_independently_paired_optimized_fresh、六旧序列复用、完整原五项科学标准通过，但并非每序列都执行五份原冻结函数。新前提不能伪装成旧报告格式或抹去此执行差异。

七份新rank报告位于元数据根 `ranks/rank1..7/verification.json`，各报告中的实际逐序列记录在顶层verification内。rank0不在该归档的attempt目录，其原封存报告现存 `/home/lbin/rbf-mainforest-reference-alias-package-preparation-20261102/job/reuse/rank0-verification.json`，91457字节；原已执行读取器对该文件绑定SHA979dd0ea38489215b487297ae554b1bdf6147543c51e1c98c29447f921660e53。本轮仅读取元数据字段，没有重跑读取器或重复数值。

rank0的六个序列为0000、0018、0034、0049、0066、0081，world_size=8、rank=0；每条含events、nodes、database_sha256、semantic_checks及原因子差异。新入口需保留该独立路径及原SHA，不能假设所有八rank都位于新归档，不能为补路径重复计算。新接受收据本身缺少原kind/seed/model/byte-admission等九字段的事实与11月3日原静态审阅一致；来源映射和最终模型谱系仍是下一准备范围，不因此取得teacher生产资格。
