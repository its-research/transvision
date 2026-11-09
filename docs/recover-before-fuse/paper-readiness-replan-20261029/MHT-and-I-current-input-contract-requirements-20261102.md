# MHT与同资源扫描：当前输入合同修复顺序

2026-11-02 07:51，北京时间。属于来源与接入准备，不是正式实验或完成证明。

1. 保留原模型与严格来源检查。112运行原checkpoint加载器的七源码检查表达式，三个SPD模型均通过；21次修改来源SHA均被拒绝。没有导入模型、加载权重或执行前向。
2. 复用现有paper MHT后端。旧Linux MHT合同绑定旧checkpoint与旧car缓存；当前三个模型的all-class seen-val缓存身份均不同，不能只换checkpoint常量。既有Linux完整执行合同决定门槛保留。
3. 复用原事件桥。三种子既有input-binding与当前模型身份匹配，各3316事件、21序列；历史事件SHA分别55a64e37…、12d5e6d8…、2ea2727f…。此次只核对小收据和转换源码，未重新校验完整事件、缓存或payload。下一步先完整重验原文件SHA，再证明events包装JSON到CLI JSONL的逐事件等价，保留100ms决策、不可用源排除、空事件及全部输入字段。不把计划到达时间当真实网络测量。
4. 正式I扫描需逐种子输入绑定。原scan_paper_resources将同一cache_manifest/schedule交给所有seed；原CLI的frozen_cache_identity检查在两个数据集共18组元数据配对中，仅同种子6组通过，跨种子12组拒绝。候选合同应为每个seed显式绑定其合格cache/schedule/GT及checkpoint，并在同seed各方法间共享输入；不得修改checkpoint元数据或删除严格检查。
5. 逐种子输入修复必须贯穿计划、执行、产物及freeze校验，不只改命令行。保留train-only选择、原资源网格、三种子、原约束与原容差；三种子实例不等于每配置至少三次隔离资源重复。正式资源结论仍须实际全集、独立验收、同硬件/线程/预热及至少三次重复。
6. 先完成源码/输入/环境与独立验收入口准备，再沿用具体发布、运行合同和资源准入门槛提交。当前没有创建MHT任务、扫描任务、正式冻结配置或科学验收。

证据：MHT-current-source-cache-contract-audit-20261102.json；I-resource-scan-current-seed-input-gap-20261102.json；MHT-existing-event-bridge-locator-20261102.json。所有SHA、AST与谓词核对均在112执行；Mac仅文本与字节传输。
