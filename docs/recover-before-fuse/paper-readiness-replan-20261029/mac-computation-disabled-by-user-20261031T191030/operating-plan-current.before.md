# Recover Before Fuse 当前执行与论文目标

计划记录：2026-10-31T18:57:16.296051+08:00。各任务进展保留证据的实际采样时间；文档更新时间不代表计算推进。半小时定时任务已删除，活跃Goal按既有授权推进。

旧启动说明及旧控制结果完整保存在[历史计划快照](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/current-plan-display-reconciliation-20261031T1730/operating-plan-current.md.before)；以下只显示当前状态，不将已完成0056或rank7读回继续列为活任务。

## 三层可验收目标

**L1 完整方法（A, B, C, D, E, F）**
- 每条数据集/种子/支持域链的原输入、模型、完整回放、teacher与学习式优先级独立合格
- 机制诊断、精确恢复及合法状态/动作成立；准备、来源或局部序列不替代整链验收

**L2 公平全集证据（G, H, I, J）**
- I0仅在train完成资源选择并冻结，随后执行两个数据集全部主表和消融
- 同势Top-K、强单端、同资源MHT与完整恢复方法公平比较；公开方法遵循各自原生协议
- I1冻结后至少3次隔离重复，成本包含全部相关cache、因子、frontier、状态、模型与重算

**L3 可复现论文（K）**
- 全指标、逐序列与三种子统计、配对bootstrap、正负案例、来源/执行/产物一体追溯齐备
- 明确探索性与已知重叠；不掩盖失败、不承诺实验完成必定发表；原A–K全部完成条件逐项闭合

## 最短依赖路径

A及适用B/C → D按数据集/种子/支持域完整链 → I0 train选择冻结 → G/H全集 → I1正式隔离成本及K统计/论文；E/F与J按各自依赖并行。

I0 属于资源选择阶段；必须先于正式 G/H。I1 属于资源测量阶段；冻结后至少三次隔离重复，不能用当前共享机器运行时替代。

## 当前优先顺序



1. **teacher3407_optimized_replacement**：0010/0016/0018/0022/0023共5份目标/轨迹/原因子报告已完整SHA读回并按原每报告来源条件核对。原任务8路运行，最新观察2026-10-31T18:54:58.100033+08:00；实际阶段计数：0000 trajectory 76932/133758；0001 trajectory 113655/153593；0002 numeric 96593/167073；0004 trajectory 161070/161329；0005 trajectory 85790/93970；0008 trajectory 84081/110033；0025 trajectory 146085/169588；0029 numeric 58246/130985。阶段计数不是完整序列资格。继续原完整45份终态收集，再与0087及46份五项报告同库同源汇合，当前不放行训练。

2. **teacher1337_remaining36**：0016首次完整尝试当前阶段=independent_teacher_full_search_trajectory 75/143; last_callback=2026-10-31T18:12:30.540699+08:00; no new event callback at18:37; CPU growth alone is not effective work proof，阶段观察时间2026-10-31T18:37:43.165710+08:00；当前阶段证据为/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/user-current-status-20261031T183737/112-0016.json。下载/暂存、预检、数值运行和完整六项/原因子/来源收集分别验收；只有原完整终态与同源模型汇合通过才增加资格。当前完整teacher 15/46，剩31序列；原输入/预算及1e-8/1e-4保持。

3. **teacher1337_rank1_six_lossless_storage_preparation**：rank1已有5/6存储合格；0001原压缩库构建后正在最终见证回读，18:54实测22500/28685，尚须原独立逐见证/typed-table/schema及压缩解压SHA和终态收集。保留原退出76934，不重复启动。

4. **mainforest2027**：2026-10-31T16:05:11.691801+08:00原来源/未使用一次授权核对通过；物理空闲准入=False；原因=another A100 task is active; preserve it。不抢占、不提前入队；必须原实际UUID交集、CPU/RAM/分盘及环境/输入全部准入后才可执行。

5. **priority_linux**：0056原31份映射已完整合并。实际优先级消费者已读取两个生产来源的5份3090原报告（644579标签），原30项条件及对应五项同库同模型检查通过；打包Python模块保留.py后缀，未改源码/报告。仍须原3090完整45份终态收集、46序列六项来源汇合、完整训练请求及原48条件、Linux准入和绑定入口的一次训练授权；当前不训练。

6. **V2V_unit5**：现有冷缓存272诊断快照的38活组件均为成员数+根的最小完整路径；固定原归档970394及活成员34748/根38，准备阶段下限1005180>原cap1000000，搜索分配之前已超5180。仅降本事件work或GPU batch不能修复该快照；须研究此前少产生计费记录或显式新无损表示，并独立验证完整支持/历史成本和原容差。历史94578超过单链基线不等于可删除；不自动重放失败事件或改cap。

7. **SparseCoop_native_integration**：本次八卡失败和三份产物SHA完整保留；先等待新Linux24worker对照具体授权并完成实际字节/RNG及内存诊断。该对照不替代原八卡192worker模型门槛，未授权再次模型重跑。

## 观察到的科学覆盖

计数依据为机器可读运营计划中的SHA绑定证据；各运行观察保留原采样时间，文件更新时间不代表实验进度。

| 路径 | 完整或分项覆盖 | 下一缺口 |
|---|---|---|
| teacher1337 | 15/46完整；1913事件、1437681标签、47222因子行 | 余31序列；0016首次完整验收已启动，仍须原终态、六项/因子及完整来源收集 |
| teacher3407 | 五项46/46、7445事件；完整六项1/46 | 原五项余0序列；目标/轨迹、原因子及同库同源汇合 |
| 主森林2027 | 6/46完整 | 40序列；A100物理不重叠且准入通过后已授权一次 |
| V2V | 5/81 bound回放分项；完整9序列cohort 0/9 | 原生指标、全部方法主表及I0/I1仍须独立完成 |
| 原始来源 | 5/5组完整SHA | 来源资格不替代数值验收；继续逐序列存储和独立验收 |

五项覆盖不等于完整teacher，来源组完成不等于数值验收。总ETA未知。

## 减少管理开销

- 当前执行队列只有四类；历史成功控制和已失败尝试单列保存，不反复列作活项
- 每次优先完整收集当前尝试的数值与终态，再新增优化；没有证据不标通过或运行
- 同一不变来源和已合格报告复用；合同要求的发布、传输、执行及收集SHA仍完整核验，日常状态巡检不重复读取多GB旧文件
- 既有控制通过且未改动时不重跑；优化需完整结果一致和端到端改善，观察提速不当正式隔离成本
- 原48条件实际consumer已完成49项软件控制及独立审查；只在真实完整teacher与来源齐备后验证正向消费和Linux，不再增建同义readiness层；软件准备计0个新科学验收
- 每条已合格数据集/种子/支持域链独立放行下一依赖，不等待其他无依赖链；不改变该链的全集门槛
- 每轮报告真实开始时间、墙钟、当前阶段、生产证据时间、CPU累计/增量和未知ETA；队列/创建时间不冒充计算时间

## 不可缩减的论文范围

A 回放验收；B 正式9D运动及协方差；C 最终身份模型；D 全量teacher与学习式优先级；E M0–M4诊断；F 精确恢复；G 两数据集正式方法主表；H 全部恢复/历史/分配消融；I train资源选择冻结及隔离成本；J 各公开方法原生复现；K 独立指标、三种子统计、真实正负案例和可复现实验包。

SPD已见探索性和V2V已知重叠必须披露；原生方法与机制表分开，缺失公开实现不能用文献数值替代。实验完成不自动证明恢复优势或论文接收。

[机器可读当前运营计划](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/operating-plan-current.json)

当前来源与运行证据：

- [teacher1337完整来源与模型汇合索引](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/original1337-fifteen-targets-fifteen-factors-current-source-join-20261031T1552.json)
- [teacher3407最新实际阶段计数](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/teacher3407-3090-optimized-restart-20261029/dispatch/attempt-observation-20261031T185458100033.summary.json)
- [完整A–K工作包、授权与完成门槛](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/publication-readiness-plan.md)
- [分阶段墙钟、CPU增量及平台只读观察](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/user-current-status-20261031T183737/current-progress-runtime-summary.json)

## 用户最新GPU策略

不默认禁用GPU；后续入口允许使用通过物理交集及运行环境准入的GPU，保持独立重算、原输入/模型/预算和原容差。CPU实现不能因解除屏蔽就称为GPU计算，现有封版尝试如需切换须先完成完整一致性验证并保留已有结果。

0056未设置CUDA屏蔽，其冻结算法为NumPy CPU实现；3090当前已启动任务仍是先前CPU封版。此次不更改或重启现有进程，后续GPU候选不得继承统一禁用条件。

当前1337完整覆盖以[当前SHA绑定来源与模型汇合索引](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/original1337-fifteen-targets-fifteen-factors-current-source-join-20261031T1552.json)为准；所有旧索引保留为历史。
