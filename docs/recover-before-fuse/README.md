# Recover Before Fuse 文档导航

整理日期为 2026-11-09。本次整理提供文档入口；实验进度须依据带时间的原记录及平台实时证据核实。

## 阅读顺序

1. [原始 A–K 目标](/Users/lbin/.codex/attachments/fe7b3ffd-6692-4c11-8198-014dbd00c8f7/goal-objective.md)规定完整交付范围。
2. [发表证据计划入口](paper-readiness-replan-20261029/publication-readiness-plan.md)指向当前维护的完整执行计划。
3. [执行与证据导航](paper-readiness-replan-20261029/README.md)帮助定位授权交接、具体任务和论文结论边界。
4. [A–K 汇合记录](full-A-K-post-optimized-audit-reconciliation-20261021.json)保存跨任务来源与验收记录。先查其 `authoritative_current_operating_record_20261102` 指针，再核对具体记录时间；文件中同时保留较早快照。

## 按主题查阅

| 主题 | 入口 |
| --- | --- |
| 完整计划与发表门槛 | [publication-readiness-current-20261102.md](paper-readiness-replan-20261029/publication-readiness-current-20261102.md) |
| 任务与授权交接 | [next-authorized-actions-handoff.json](paper-readiness-replan-20261029/blanket-approved-remote-dispatch-20261101/next-authorized-actions-handoff.json) |
| 硬件路由与存储约定 | [resource-routing-current.json](resource-routing-current.json)、[local-storage-policy-current.json](local-storage-policy-current.json) |
| V2V 正式协议与数据重叠 | [标称 10Hz 协议](v2v4real-nominal-10hz-formal-v1.md)、[重叠与发布边界](v2v4real-overlap-and-release.md) |
| 概率跟踪与计算分配 | [probabilistic-tracking-v2.md](probabilistic-tracking-v2.md)、[learned-allocation.md](learned-allocation.md) |
| Teacher 序列组装 | [teacher-sequence-assembly-20260913.md](teacher-sequence-assembly-20260913.md) |
| 同资源实验 | [paper-resource-scan.md](paper-resource-scan.md) |
| 公开方法证据 | [native-public-method-evidence-table-20261008.md](native-public-method-evidence-table-20261008.md) |

## 目录与记录含义

| 位置或命名 | 用途 |
| --- | --- |
| 本目录的主题 `.md` | 方法、协议、设计和专项分析 |
| `paper-readiness-replan-20261029/` | 发表计划、执行交接、后续尝试及证据整理 |
| 各带日期的专项目录 | 对应任务的准备、投递、日志、收集与验收材料 |
| `heartbeat-*`、`manual-current-progress-*` | 指定时刻的观测快照 |
| `before-*`、`*.before-*`、失败尝试目录 | 变更前状态与失败留证 |

软件控制、来源 SHA、生产成功和独立数值验收各自有对应证据。引用完成结论时，须同时给出范围、原容差、来源绑定及完整收集记录。旧的待批字段与运行快照按各自时间解释，接续前核对最新人工授权及一次性执行记录。

## 当前路径说明

2026-11-09 本机只读检查确认，实际工作树及 `.git` 位于 `/Volumes/Data/test/recover-before-fuse/local-workspace/transvision`。随后按用户要求移除了 `/Users/lbin/transvision` 符号链接。

外置盘实验材料的根目录为 `/Volumes/Data/test/recover-before-fuse`。仓库中的 `test` 链接当前指向 `/Volumes/Data/test`。这份导航记录当前结构；仓库迁回本机的工作尚未执行。

历史收据和冻结脚本可能仍包含 `/Users/lbin/transvision`。使用前应核对调用方所需路径与挂载映射，原证据中的路径按原文保留。
