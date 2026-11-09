此入口直接连接完整证据、可训练 manifest 和冻结 priority trainer。它不把准备、1 个序列或目标检查当作完整 teacher；本轮只执行无模型的本地拒绝控制。

`mixed_admission.py` 接受两个真实生产来源 `75f081…` 的 11 序列和 `832a23…` 的 35 序列，逐序列保留原 source_task、计划、归档、数据库、原生提交链及验证来源。训练 manifest 的 `source_sha256` 严格对应原冻结 consumer 的 21 个文件；生产来源独立存放在 `producer_runtime_sources_by_task`，不会合并成一个实际执行源。原 48 项条件按原审计顺序逐项落实为 source-aware 合同，旧单源 gate 不改，仍然拒绝混合输入。不能将此称为旧单源 48 个表达式原封不动接纳了混合来源。

证据目录沿用 `teacher3407_portable_preserved_evidence_catalog_v1`，所有原绝对路径只作为身份，文件从 catalog 的安全相对路径读取并核对完整大小/SHA。除原 reader 所需根，还须提供：

- `consumer_request`：`mixed_teacher_priority_consumer_request_v1`，seed3407、真实 SPD/train、原 `1e-8`/`1e-4`；46 行按原 exporter 的序列字典序，原 7,445 事件、1,004,381 组、5,560,061 标签及 223,355 因子行。每行包括原 input-evidence 行字段，并补真实 `production_plan`、`original_five_reports`；不得填造报告或接受缺项。
- request 还提供 `main_prerequisite`、原 `training_binding`、两个原 witness recipe；`GT_read/test_read/strict_pipeline_isolated_selection/capacity_undecided_counted_as_optimal/exact_identity_loss_or_posterior_optimum_target_claimed` 均须 false。这些声明不取代真实 plan、cache、因子、原五项及原目标报告检查。
- catalog 必须包含本目录 source-pins 指向的原 reader/audit/roles/原训练模块与 exporter 源、46 份实际 plan、完整元数据快照、原 cache manifest。逐字节保留历史 JSON，不改路径字段或来源头。
- 新验证来源使用 `3090_jumps_once`，额外根 `jumps_manifest/jumps_driver/jumps_joiner/optimized_execution_binding`。未来真实 task_id 从已完成的收集和提交绑定读取；不预填任务。原 30 项逐报告断言、优化结构证明、原报告 SHA、completed 终态及 45 项收集覆盖都要通过。0087 始终保留原 restored112 身份。

`run_source_bound_priority.py admit --bundle … --catalog-sha256 … --request-sha256 … --output 新目录` 只在以上真实证据完整后复制原证据、原 groups 字节并创建 manifest。分片顺序不变，不重新导出标签或重算 oracle。完整缺证时在创建输出目录前拒绝。

`train --data … --manifest-sha256 … --output 新目录 --authority … --authority-sha256 … --runtime-admission … --runtime-admission-sha256 …` 还要求独立同入口 Linux 准入，以及绑定实际 manifest/entry/adapter/source pins 的一次执行授权。runtime 记录必须绑定实际 frozen_consumer_root、解释器、主机与 Python/Torch/NumPy，原 CPU-only、1 线程、Torch 2.6.0+cu124、NumPy 1.26.4；入口只检查，不改资源限制。未核验的 Linux 记录不可由本地软件控制替代。

训练时只把冻结 `fit_priority` AST 中一次 `verify_exported(data, manifest)` 调用改为显式 `mixed_admission.verify_exported(data, manifest)`。旧 import 原样保留并注明仅导入未执行；不做 sys.modules 或 import monkeypatch。反向替换必须恢复整个原函数 AST。其余算法、源相等检查、37/9 train 内部划分、10 轮、float64、Adam0.001、最早最小 holdout 选择、checkpoint/weights 导出全不变。新 adapter 身份写旁证；独立 checkpoint 和 runtime 验收仍在训练之后。

当前仍缺真实完整46六项/因子/来源 bundle 和该 consumer 的 Linux 准入及单次训练执行证据。没有运行模型，没有创建训练 manifest，没有发布或切换远端任务。运行 `python3 -B test_consumer_boundaries.py` 仅执行拒绝/来源/AST 软件检查；结果不代表完整输入准入或 D 完成。
