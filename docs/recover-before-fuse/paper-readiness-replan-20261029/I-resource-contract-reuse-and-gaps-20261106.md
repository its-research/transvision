# I项：原资源合同复用与缺口

本轮只读审查既有合同和工具，没有执行资源扫描、SHA计算或性能统计；旧记录不是新实时资格。

原`I-resource-scan-contract-readiness-current-20261011.json`已固定train选择、三种子、K/MHT宽度1/4/16/64、展开64/256/1024和动作1024/4096；只对支持相应预算的后端建立组合，不能做无效笛卡尔积。既有36个构造候选及逐项前提在`resource-cost-independent-20261011/frozen-grid-candidates/per-candidate-execution-prerequisites.json`，候选软件检查通过不等于真实执行准入。

原10月12日成本证据汇合记载六个完整cohort、44670个事件timing已有p99及组件存储证据。它们缺相同硬件、明确预热和独立重复，不能称正式同资源；也不能再把“历史p99全部缺失”列为阻碍。原始成本可复用为历史诊断，不重复测量来伪装新隔离结果。

现有`tools/event_track_v2x/run_resource_sweep_v2.py`入口明确各后端/配置需要三种子、至少三次新进程重复、随机完整区组，且不宣称相同预算等于相同资源。旧`scan_paper_resources.py`每配置三种子不构成同一训练配置三次资源重复。

正式执行前必须绑定：

1. 数据集、原train开发分组、完整cache/schedule和评价协议；test不选配置。
2. 原检测、特征、运动及适用身份模型；learned还须匹配支持/预算的完整teacher与priority资格。
3. 实际CPU/GPU来源准入、物理UUID、线程、预热、冷/热缓存政策和隔离合同；不能抢占活任务、不能用共享112验收负载结果充正式性能。
4. 完整成本：raw/factor、前沿/prefix、分支状态、其余SQLite/索引/缓存、模型前向与重算、teacher probe、RSS/GPU内存、吞吐及p50/p95/p99/max。
5. 每个配置至少三次独立资源重复，与三种子轴分开；完整预测必须经独立验收，所有终态和失败保留。
6. 原正式Linux MHT合同与具体准入；不能以局部beam、精确fixture或相同K代替同资源MHT。

当前不启动正式扫描：完整learned及支持专用路径尚未具备，原实际缓存/模型/隔离组合也未形成可执行清单。后续可按配置分别准入，无需等所有配置同时完成；但不得放宽原候选或以空闲卡为由进行无依赖训练。
