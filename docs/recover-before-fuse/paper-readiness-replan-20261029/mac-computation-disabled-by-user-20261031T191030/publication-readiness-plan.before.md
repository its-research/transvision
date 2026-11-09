# Recover Before Fuse：当前执行与论文收口计划

生成：2026-10-31T18:57:16.360582+08:00。运行观察更新：2026-10-31T18:42:01.276324+08:00；覆盖计数基准：2026-10-31T15:52:23.805667+08:00。每条证据保留自己的采样时间，完整A–K尚未完成。

[持久运营计划](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/operating-plan-current.md) · [机器可读运营计划](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/operating-plan-current.json)

## 三层验收目标

**L1 完整方法**

- 每条数据集/种子/支持域链的原输入、模型、完整回放、teacher与学习式优先级独立合格
- 机制诊断、精确恢复及合法状态/动作成立；准备、来源或局部序列不替代整链验收

**L2 公平全集证据**

- I0仅在train完成资源选择并冻结，随后执行两个数据集全部主表和消融
- 同势Top-K、强单端、同资源MHT与完整恢复方法公平比较；公开方法遵循各自原生协议
- I1冻结后至少3次隔离重复，成本包含全部相关cache、因子、frontier、状态、模型与重算

**L3 可复现论文**

- 全指标、逐序列与三种子统计、配对bootstrap、正负案例、来源/执行/产物一体追溯齐备
- 明确探索性与已知重叠；不掩盖失败、不承诺实验完成必定发表；原A–K全部完成条件逐项闭合

A及适用B/C → D按数据集/种子/支持域完整链 → I0 train选择冻结 → G/H全集 → I1正式隔离成本及K统计/论文；E/F与J按各自依赖并行。

## 当前科学覆盖

| 路径 | 已核实覆盖 | 剩余 |
|---|---|---|
| teacher1337 | 15/46完整，1913事件、1437681标签、47222因子行 | 31序列 |
| teacher3407 | 五项46/46、7445事件；完整六项1/46 | 原五项尚余0序列；完整目标/轨迹、因子及混合来源消费准入 |
| 主森林2027 | 6/46完整 | 40序列，已授权一次A100完整验收 |
| V2V | 5/81单元 | 76单元，容量与姿态协议阻碍 |
| 来源读取 | 5/5组完整SHA | 当前all5_complete_original_source_SHA_readback_no_numeric_admission；未用rank  |

五项覆盖沿用其记录时间；本轮新增完整六项teacher序列0。准备、来源校验、启动和CPU增长均不等于科学验收。

## 本次运行与历史中断

- 3090 teacher：eight_existing_processes_actual_label_counters_progressing_registered_subset_preserved。0010/0016/0018/0022/0023共5份目标/轨迹/原因子报告已完整SHA读回并按原每报告来源条件核对。原任务8路运行，最新观察2026-10-31T18:54:58.100033+08:00；实际阶段计数：0000 trajectory 76932/133758；0001 trajectory 113655/153593；0002 numeric 96593/167073；0004 trajectory 161070/161329；0005 trajectory 85790/93970；0008 trajectory 84081/110033；0025 trajectory 146085/169588；0029 numeric 58246/130985。阶段计数不是完整序列资格。继续原完整45份终态收集，再与0087及46份五项报告同库同源汇合，当前不放行训练。
- 本机0056：46of46_original_five_scopes_and_authorized_source_summary_qualified。原五项46序列7445事件完成，获批来源汇总也已通过。继续等待3090完整45序列目标/轨迹、原因子和原完整传输，再以版本化完整46入口同源汇合；暂不训练。
- 来源读取：all5_complete_original_source_SHA_readback_no_numeric_admission。五组23序列161原文件的整包/文件SHA已收口；保留全部来源和历史失败。继续各序列无损存储及原独立数值验收，不重复读取来源、不把来源通过计为teacher完整合格。
- 历史0056首次尝试缺终态，历史rank7远端exit=1；原失败及原因未知记录保留。它们不代表以上当前尝试状态，新尝试也不证明旧故障原因已解决。
- 当前来源观察：[原始带时间戳记录](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/teacher1337-rank2-complete-five-source-batches-20261030T1425/complete-five-batch-source-current-join.json)；完整归档和选定文件SHA未收口前不供数值使用。
- 0056中断查因：[带时间戳核验记录](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/interruption-resolution-20261029T2046/0056/interruption-resolution.json)；不计作新增完整teacher序列。
- rank7来源中断查因：[带时间戳核验记录](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/interruption-resolution-20261029T2046/rank7/rank7-cause-review.json)；不计作新增完整teacher序列。
- 3090最新真实计数：[带时间戳核验记录](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/teacher3407-3090-optimized-restart-20261029/dispatch/attempt-observation-20261031T170151405463.summary.json)；不计作新增完整teacher序列。
- 新优化版Linux准入：[带时间戳核验记录](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/teacher3407-3090-optimized-restart-20261029/dispatch/linux-qualification-readback.json)；不计作新增完整teacher序列。
- 混合来源实际训练入口：[带时间戳核验记录](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/priority-source-aware-transition/actual-consumer-implementation-20261029T211141.json)；不计作新增完整teacher序列。
- 0087本地报告修正版：qualified_local_complete0087_optimization；本次开始2026-10-29T20:15:29.288848+08:00；观察2026-10-29T20:41:38.020763+08:00；墙钟932.5288007079507秒。最新未知/17307标签，CPU累计未知秒（生产时间2026-10-29T20:31:01.861500+08:00）；独立读回完成=True。

3090当前生产计数及CPU（各记录时间可能不同，陈旧值不计新进展）：

任务18abed2123c14d63ac411ec7ea6ba611；观察2026-10-31T17:01:51.405463+08:00；平台状态in_progress；绑定10.100.34.27-3090:gpu0,1,2,3,4,5,6,7。
- 0000：trajectory，numeric完成133758标签，trajectory完成74030标签；CPU采样墙钟153243.3500655461秒、累计153203.09999999998秒，增量119.96999999997206/120.00483153574169墙钟秒。内部计数对应计算墙钟153133.4591387976秒，采样与生产记录不混作同一时刻。
- 0001：trajectory，numeric完成153593标签，trajectory完成111350标签；CPU采样墙钟153243.34985327907秒、累计153199.97秒，增量60.0/60.002578776329756墙钟秒。内部计数对应计算墙钟153145.21385196783秒，采样与生产记录不混作同一时刻。
- 0002：numeric，numeric完成95559标签，trajectory完成0标签；CPU采样墙钟153244.46942952834秒、累计153197.91秒，增量59.98000000001048/60.001236313953996墙钟秒。内部计数对应计算墙钟152830.56945974566秒，采样与生产记录不混作同一时刻。
- 0004：trajectory，numeric完成161329标签，trajectory完成159542标签；CPU采样墙钟153244.52857524157秒、累计153199.86秒，增量59.98000000001048/60.00142935104668墙钟秒。内部计数对应计算墙钟153126.2553018406秒，采样与生产记录不混作同一时刻。
- 0005：trajectory，numeric完成93970标签，trajectory完成84043标签；CPU采样墙钟153183.26168749295秒、累计153142.44999999998秒，增量59.98999999999069/60.00128614902496墙钟秒。内部计数对应计算墙钟153068.2837093938秒，采样与生产记录不混作同一时刻。
- 0008：trajectory，numeric完成110033标签，trajectory完成79573标签；CPU采样墙钟153183.31015307456秒、累计153142.79秒，增量59.98000000001048/60.00071908533573墙钟秒。内部计数对应计算墙钟153073.77961714752秒，采样与生产记录不混作同一时刻。
- 0023：trajectory，numeric完成160973标签，trajectory完成152115标签；CPU采样墙钟61861.40571835078秒、累计61843.21秒，增量59.98999999999796/60.0014219712466墙钟秒。内部计数对应计算墙钟61770.850409757346秒，采样与生产记录不混作同一时刻。
- 0025：trajectory，numeric完成169588标签，trajectory完成125068标签；CPU采样墙钟55921.27311923355秒、累计55906.16秒，增量59.9800000000032/60.001524459570646墙钟秒。内部计数对应计算墙钟55825.57893513702秒，采样与生产记录不混作同一时刻。

## 当前队列

| 类别 | 项目 | 实际状态 | 下一步 |
|---|---|---|---|
| in_progress | teacher3407_optimized_replacement | eight_existing_processes_actual_label_counters_progressing_registered_subset_preserved | 0010/0016/0018/0022/0023共5份目标/轨迹/原因子报告已完整SHA读回并按原每报告来源条件核对。原任务8路运行，最新观察2026-10-31T18:54:58.100033+08:00；实际阶段计数：0000 trajectory 76932/133758；0001 trajectory 113655/153593；0002 numeric 96593/167073；0004 trajectory 161070/161329；0005 trajectory 85790/93970；0008 trajectory 84081/110033；0025 trajectory 146085/169588；0029 numeric 58246/130985。阶段计数不是完整序列资格。继续原完整45份终态收集，再与0087及46份五项报告同库同源汇合，当前不放行训练。 |
| in_progress | keep_teacher3407 | eight_existing_processes_actual_label_counters_progressing_registered_subset_preserved | 0010/0016/0018/0022/0023共5份目标/轨迹/原因子报告已完整SHA读回并按原每报告来源条件核对。原任务8路运行，最新观察2026-10-31T18:54:58.100033+08:00；实际阶段计数：0000 trajectory 76932/133758；0001 trajectory 113655/153593；0002 numeric 96593/167073；0004 trajectory 161070/161329；0005 trajectory 85790/93970；0008 trajectory 84081/110033；0025 trajectory 146085/169588；0029 numeric 58246/130985。阶段计数不是完整序列资格。继续原完整45份终态收集，再与0087及46份五项报告同库同源汇合，当前不放行训练。 |
| resource_or_dependency_wait | readonly_interruption_reconciliation | read_only_evidence_complete_causes_unresolved | 旧0056和旧rank7中断原因未证实，失败证据保留；新0056已合格，rank7获准重读已完整通过，不再作为活任务。 |
| completed | teacher3407_five_scope_supplement | 46of46_original_five_scopes_and_authorized_source_summary_qualified | 原五项46序列7445事件完成，获批来源汇总也已通过。继续等待3090完整45序列目标/轨迹、原因子和原完整传输，再以版本化完整46入口同源汇合；暂不训练。 |
| completed | source_remaining23 | all5_complete_original_source_SHA_readback_no_numeric_admission | 五组23序列161原文件的整包/文件SHA已收口；保留全部来源和历史失败。继续各序列无损存储及原独立数值验收，不重复读取来源、不把来源通过计为teacher完整合格。 |
| resource_or_dependency_wait | mainforest2027 | existing_once_source_release_preflight_passed_A100_physically_busy_and_eight_queue_pending | 2026-10-31T16:05:11.691801+08:00原来源/未使用一次授权核对通过；物理空闲准入=False；原因=another A100 task is active; preserve it。不抢占、不提前入队；必须原实际UUID交集、CPU/RAM/分盘及环境/输入全部准入后才可执行。 |
| in_progress | teacher1337_remaining36 | 15_of46_complete_31_remaining_0016_first_full_attempt_running | 0016首次完整尝试当前阶段=independent_teacher_full_search_trajectory 75/143; last_callback=2026-10-31T18:12:30.540699+08:00; no new event callback at18:37; CPU growth alone is not effective work proof，阶段观察时间2026-10-31T18:37:43.165710+08:00；当前阶段证据为/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/user-current-status-20261031T183737/112-0016.json。下载/暂存、预检、数值运行和完整六项/原因子/来源收集分别验收；只有原完整终态与同源模型汇合通过才增加资格。当前完整teacher 15/46，剩31序列；原输入/预算及1e-8/1e-4保持。 |
| resource_or_dependency_wait | priority_linux | actual0056_overlay_and_both_producer_five_target_consumer_preflight_passed_waiting_full46_and_Linux | 0056原31份映射已完整合并。实际优先级消费者已读取两个生产来源的5份3090原报告（644579标签），原30项条件及对应五项同库同模型检查通过；打包Python模块保留.py后缀，未改源码/报告。仍须原3090完整45份终态收集、46序列六项来源汇合、完整训练请求及原48条件、Linux准入和绑定入口的一次训练授权；当前不训练。 |
| resource_or_dependency_wait | teacher3407_complete_six_scope_join | five_scopes_source_qualified_waiting_complete45_target_transport_then_full46_join | 原五项46/46来源汇总通过；目标/轨迹/原因子已有5份注册子集通过原每报告条件，仍须原任务全部45份终态收集，随后复用0087执行完整46序列同库同源六项汇合；当前完整六项1/46，子集不能放行训练。 |
| resource_or_dependency_wait | SparseCoop_native_integration | failed_full3registered_SHA_readback_original192worker_gate_not_passed | 本次八卡失败和三份产物SHA完整保留；先等待新Linux24worker对照具体授权并完成实际字节/RNG及内存诊断。该对照不替代原八卡192worker模型门槛，未授权再次模型重跑。 |
| resource_or_dependency_wait | V2V_unit5 | single272_capacity_exception_diagnostic_complete_fullSHA_rollback_preserved | 现有冷缓存272诊断快照的38活组件均为成员数+根的最小完整路径；固定原归档970394及活成员34748/根38，准备阶段下限1005180>原cap1000000，搜索分配之前已超5180。仅降本事件work或GPU batch不能修复该快照；须研究此前少产生计费记录或显式新无损表示，并独立验证完整支持/历史成本和原容差。历史94578超过单链基线不等于可删除；不自动重放失败事件或改cap。 |
| needs_authorization | M3_2027 | single_BLAS_retry_pending | 等待单线程BLAS候选专项一次批准，保持原21序列3316帧逐字节标准。 |
| resource_or_dependency_wait | 0010_source_bound_parser_retry | series_once_approved_versioned_source_consumer_prepared_real_collection_pending_serial_dependency | 一次授权尚未使用；新解析协议与合同文件名的版本化读取器已完成17项来源及29项元数据控制，原数值断言/原汇总计数保留。待原串行前驱、来源SHA与新鲜112准入满足后首次执行；实际完整收集与版本化核验仍为必需，不修改0080或0057。 |
| needs_authorization | V2V_pose | native_pose_contract_unresolved | 1993帧合成控制、独立逐对象读回和一次执行编排已准备；17+10项软件检查和8项授权拒绝检查通过。等待本次全帧转换控制专项授权，原矩阵/刚体门槛/容差、正式预测全集资格均保留；不运行指标、不动现有任务。 |
| resource_or_dependency_wait | MHT_and_DMSTrack_cost | contract_and_formal_isolation_gate | 保留原MHT合同决策和失败预热重跑门槛；I0冻结后才做I1至少3次隔离成本测量。 |
| resource_or_dependency_wait | Long_SCOPE | bounded_primary_and3_namesake_README_recheck_no_verified_author_native_asset | 本轮arXiv仍v1，CVF HTML/PDF请求403；GitHub检索的3个同名README均为其他领域，已按原文排除，不能代替作者实现/权重。原原生协议不变；继续等待可验证代码、权重及完整split/ROI/频率/评价器/环境资产，不填文献性能数字，不发送未授权联系。 |
| completed | teacher3407_lexical_cache_full0087_control | local_exact_result_qualified_trajectory_regression_not_deployed | 数值阶段改善、轨迹阶段变慢；不将双阶段版本替换远端。仅可准备分阶段候选，实际集成及Linux准入仍缺。 |
| needs_authorization | teacher3407_idle_wait_full0087_control | full_source_bound_pair_entry_and_independent_reader_prepared | Await specific once-only idle-wait full0087 pair approval; old report-fix grant consumed and unapproved lexical cache pair remains separate. Resource preflight before execution; preserve current serial audit and3090 tasks. |
| needs_authorization | teacher3407_full0087_fresh_SQL_pair_once | complete_real_input_pair_and_independent_reader_prepared | 等待完整fresh_state/所选输出配对对照专项一次批准；原版与缓存版各完整2773状态/20事件/1261输出，原1e-8、同一冻结环境、默认线程/GPU可见性，不动活任务；通过后仍不自动切换。 |
| completed | teacher1337_0055_local_lossless_storage_only | complete_lossless_storage_separate_followup0055_full_numeric_already_qualified | 本项原无损存储已完整通过；后续0055首次六项/3123原因子/来源与模型同源汇合也已通过，见当前合格索引。存储资格与数值资格分别保留，不重复准备入口或重新运行0055。 |
| completed | teacher1337_remaining17_local_lossless_storage_after0055 | 17of17_original_storage_complete_final_SHA_collected_no_numeric_acceptance | 原17项存储终态和完整收集均通过；17项+0055共18份当前字节索引及234文件证明已合格，208份/232194025036字节重新完整核对，26份不变身份复用原完整SHA。rank1首次0022已另项启动；本项无新增数值资格，保留全部源数据。 |
| completed | teacher3407_numeric_only_lexical_full0087_control | completed_exact_local_pair_shared_wall_plus3percent_CPU_lower_no_deployment | 已完成同次20事件两阶段17307标签逐值及独立SHA读回；整体墙钟+3.3%、CPU-10.6%仅诊断，不替换远端。优先补实际cgroup资源及可保留进度的扩并行方案；Linux/大序列仍未准入。 |
| completed | teacher3090_CPU_quota_readonly_probe | first_diagnostic_Task_completed_metadata_only_partial_scope_quota_not_read | 首轮8进程CPU累计增量约8核已完整SHA读回；配额文件因辅助命名空间检查权限异常未读取，不据此扩大并行。 |
| completed | teacher3090_CPU_quota_namespace_auxfix_once | completed_full_SHA_readback_quota_unknown | 实际配额仍未知；先准备挂载点及控制器识别候选，新诊断另需具体授权，不自动重试。 |
| completed | teacher1337_first0057_complete_once | original_full_six_and_factor_collection_passed_14th_qualified_sequence | 0057六项/5525因子/来源全SHA和原汇合通过；完整资格14/46，接续已授权0048首次运行。 |
| in_progress | teacher1337_first0048_and0016_complete_once | 0048_full_qualified_0016_stage19_SHA_and_frozen_preflight_passed_first_numeric_running | 0048原153事件六项、2672因子及完整来源收集和同源汇合已通过，现按同一次已授权串行槽位接续0016。0016首次完整尝试当前阶段=independent_teacher_full_search_trajectory 75/143; last_callback=2026-10-31T18:12:30.540699+08:00; no new event callback at18:37; CPU growth alone is not effective work proof，阶段观察时间2026-10-31T18:37:43.165710+08:00；当前阶段证据为/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/user-current-status-20261031T183737/112-0016.json。下载/暂存、预检、数值运行和完整六项/原因子/来源收集分别验收；只有原完整终态与同源模型汇合通过才增加资格。当前完整teacher 15/46，剩31序列；原输入/预算及1e-8/1e-4保持。 0016仍须原143事件、3469因子完整终态及同源收集，不能以暂存或预检替代数值验收。 |
| completed | SparseCoop_fused_prefix_Linux_control | complete_both64batch_fullbyte_layout_RNG_exact_all3registered_SHA_readback | 本次Linux完整流水线对照通过；原版124.80秒/候选129.32秒，共享缓存顺序对照不作正式速度结论。原192worker峰值内存和模型资格未通过，不自动重试。 |
| needs_authorization | SparseCoop_fork_GC_Linux24worker_control | exact_package_independent825file_collector_and110software_controls_held_ready | 等待112640字节11成员包的具体发布及一次Linux24worker对照授权；须ubuntu26-2整机空闲、原环境/输入/RAM/磁盘准入通过。不执行模型、优化器或192worker，不以此替代原八卡门槛。 |
| needs_authorization | V2V_unit6_held_complete_preparation | original304events38754factors_probe_dispatch_terminal_pipeline_91source_controls_held_for_specific_authority | 固定顺序单元6的探针、一次调度、完整产物读回及六项终态入口已准备；91项源码/软件检查和原NumPy/BLAS只读预检通过。等待两份Task源码发布、首次探针及完整执行的具体授权，再检查物理交集与原资源准入；未发布、未提交，失败单元5仍保留。 |
| completed | teacher3407_full46_local_source_summary_field_v2_once | one_authorized_local_source_summary_check_passed | 获批一次本地汇总检查已通过，45单尝试读取器与已合格0056跨尝试读取器明确保留；不再重复本项，继续等待3090完整45目标/轨迹传输及46同源汇合。 |
| completed | teacher1337_rank1_local_original_materialization | complete_registered_archive_and42_original_local_files_stream_and_disk_SHA_qualified | 6序列42原文件来源已完整通过；原压缩包及全部原文件保留。后续无损存储已准备21项来源控制、原存储体及终态收集函数AST一致，必须原17项完整收集及新资源准入后才执行；当前不增加teacher数值资格。 |
| completed | F_0080_extra_source_representation_check | authorized_v4_original_source_lineage_and_full_SHA_readback_passed | 原来源链、66步基线、全7组件搜索激活控制及另项获批选中9D输出对照均已完成并完整SHA读回，不重复执行。原输出全部通过1e-8复现，控制对照有1个共同track至少一项状态输出超出原容差（本次为均值；协方差和分数仍在容差内）；此项只证明局部当前输出影响，GT质量、完整在线干预和正式同资源证据仍缺，完整F/H/I/K未通过。 |
| completed | teacher1337_rank0_local_original_materialization | complete_registered_archive_and42_original_local_files_stream_and_disk_SHA_qualified | 6序列42份原文件及完整归档来源均已通过，原数据保留；后续按原输入、资源及独立数值门槛准备，无来源重读或数值重复执行。 |
| in_progress | teacher1337_rank1_six_lossless_storage_preparation | 5of6_storage_qualified_first0001_running | rank1已有5/6存储合格；0001原压缩库构建后正在最终见证回读，18:54实测22500/28685，尚须原独立逐见证/typed-table/schema及压缩解压SHA和终态收集。保留原退出76934，不重复启动。 |
| resource_or_dependency_wait | teacher1337_rank0_six_lossless_storage_preparation | current18_index_qualified_wait_rank1_all6_complete | rank1现5/6存储合格、0001整理运行；六项原收集全部通过后才按0081→0034→0018→0049→0066→0000接续rank0，不把存储资格计为数值验收。 |
| completed | F_original_four_premise_exact_arithmetic | original4literal_fixture_rational_and_independent_decimal_refutations_source_bound | 保留原4夹具的独立前提反例；真实输入的有效κ、模型/真实后验误差、共同动作集和全局耦合适用性仍须验收，不声称生产程序自动检测或真实恢复收益。 |
| completed | teacher1337_local0010_first_selected_source_materialization | complete_original_archive35stream_SHA_and_selected7_source_disk_SHA_qualified | 0010七原文件来源及全rank6归档验证通过，原36清单全部252文件已有本地原文件；0010数值重跑仍须原串行前驱、新鲜112资源和原版本化入口，不能以来源通过代替其124事件/2003因子完整验收。 |
| completed | F0080_complete66_independent_baseline_once | authorized_original_full66_baseline_and_complete_SHA_readback_passed | 原来源链、66步基线、全7组件搜索激活控制及另项获批选中9D输出对照均已完成并完整SHA读回，不重复执行。原输出全部通过1e-8复现，控制对照有1个共同track至少一项状态输出超出原容差（本次为均值；协方差和分数仍在容差内）；此项只证明局部当前输出影响，GT质量、完整在线干预和正式同资源证据仍缺，完整F/H/I/K未通过。 |
| completed | F0080_full7_generated_search_control_once | authorized_once_complete66_raw_SHA_parity_full7_control_and8outputs_SHA_qualified | 原66步生成基线逐SHA一致后完成7组件控制，8产物完整SHA读回。6组件身份根/动作相同，组件41根667→486、当前损失节点16/17改变。两侧256搜索步、6143前缀、额外动作搜索0；条件模型风险非GT，遗漏/模型风险界仍1。保留该局部控制，不重复计算；下一阶段尚缺完整在线9D/GT/隔离同资源证据，不能放行全F/H/I/K。 |
| completed | F0080_selected9D_pair_once | authorized_selected9D_pair_and6output_SHA_complete_original_baseline_passed_local_effect_only | 已按选中9D专项一次批准完成，原7组件输出1e-8复现通过，全部6产物完整SHA读回；对照有1个共同track的状态输出超出原容差。无GT/模型/搜索/完整序列执行，不重复计算；后续仍缺GT质量、完整在线因果和隔离同资源证据。 |
| completed | F0080_fixed_output_joint_GT_geometry_once | authorized_once_original_baseline_and6SHA_passed_ROI_geometry_unchanged | 已按一次批准完成原50m/2m单帧GT几何对照及6产物完整SHA收集：两侧8预测/3GT、1匹配及匹配ID/距离和相同。唯一变化轨迹约61.08m，原ROI外；保留无ROI内改善的结果，不重复计算。原生指标、速度真值、连续身份及完整在线/隔离成本仍未通过，完整F/H/I/K未完成。 |
| resource_or_dependency_wait | teacher1337_rank2_five_lossless_storage_preparation | current18_source_gate_prepared_original5_attempts_unstarted | 保留当前0001及已安排rank0本机存储串行工作；其后重新核验资源，再依0023→0036→0054→0070→0002执行原无损整理和完整收集。当前仅前置来源准入已准备，未启动、未增加数值资格。 |

当前待批仅：M3_2027, V2V_pose, teacher3407_idle_wait_full0087_control, teacher3407_full0087_fresh_SQL_pair_once, SparseCoop_fork_GC_Linux24worker_control, V2V_unit6_held_complete_preparation。3090优化后重跑及0087本地对照已有直接授权，不重复询问。

历史成功控制和失败尝试另存execution_history，不作为活任务；不能因失败记录被保留就自动重试。

## 六个交付阶段

**R0 来源、进度和授权总账**

- 61份关键证据完整SHA复核完成
- 保留所有活任务；明确可复用覆盖与原失败
- 准备下一入口，只在原授权和物理/输入准入满足时执行

验收：所有待执行单元有来源、一次授权状态和收集器。

**R1 完整训练回放与teacher收口**

- 保持已通过Linux准入并完成切换的3090八进程验收；同一任务真实计数增长时等待原终态，完整收集后与原五项、原因子同库同源汇合，不重复切换。
- 3407保留0087完整结果；原五项已完整读回46/46，尚余0序列。目标/轨迹、原五项与原因子须同库同源汇合，不把单项覆盖当完整六项。 当前执行：原五项46序列7445事件完成，获批来源汇总也已通过。继续等待3090完整45序列目标/轨迹、原因子和原完整传输，再以版本化完整46入口同源汇合；暂不训练。
- 1337保留15/46完整数值结果；五组来源23序列161文件完整SHA已收口。余31序列按已记录系列授权、存储准备、串行前驱及重新资源/输入准入完成原六项和因子；来源通过不替代数值验收。
- 0056与rank7历史失败证据保留，历史原因未知不覆盖当前新尝试；新失败不自动重试或推进下一rank。
- 逐数据集/种子/支持域完成原六项与因子；原始来源SHA不替代数值结果。
- A100物理空闲和准入通过后执行2027已批准一次完整验收，保留6份已合格结果。

验收：所有所需种子teacher完整覆盖、来源SHA、目标和因子独立通过；各序列具备原六项完整独立结果及因子；按来源分别绑定，不继承主森林或其他种子结果。

**R2 机制与公开路径阻碍并行解决**

- M3严格复现和真实恢复因果链
- V2V容量与原姿态门槛问题
- SparseCoop原八卡资源/一步资格；Long-SCOPE作者资产

验收：对应阻碍通过原标准；没有资产不勾选完成。

**R3 学习式配置和资源冻结**

- 按合格完整链进行priority train/holdout选择与独立NumPy排序。
- I0：仅在train选择原支持范围内的资源与配置，冻结正式G/H的输入、模型、预算及参数。

验收：三种子优先级与合法动作、状态、成本接入独立通过；I0冻结先于G/H正式主表与消融，不用测试结果选参数。

**R4 正式主表与消融全集**

- G/H全部方法、数据集、种子和条件完整回放
- 产物逐字节读回并独立评价；不重复已合格单元

验收：完整预测、全部指标、逐序列与三种子结果。

**R5 隔离成本、统计与投稿证据**

- I1：冻结后每配置至少3次正式隔离成本测量，使用同硬件/线程/预热合同，包含全部成本。
- 配对bootstrap、表图、真实案例和失败披露
- 结论与全部证据可追溯审计，生成可复现实验包

验收：原A–K完成门槛全部满足；无局部完成替代；论文主张由实际正负结果支持。

## 原A–K完整范围

**A 现有回放和独立验收收口**

验收：每种子/配置完整来源、因子、状态、动作、输出覆盖；所有产物完整SHA与独立数值报告，而非平台completed。

**B 正式9D运动与协方差**

验收：train-only选择与完整test缓存身份；所有方法消费相同适用输入且没有未来信息。

**C 最终身份模型资格**

验收：每数据集各1337/2027/3407模型来源和逐行前向；checkpoint选择仅train开发分组，不择优报告种子。

**D 完整teacher与学习式优先级**

验收：所有适用训练序列、教师目标和因子完整；各种子独立优先级checkpoint与因果运行时接入；teacher模型目标不宣称真实身份风险；3407完整目标/轨迹与另外五项原oracle逐数据库汇合；当前45任务或46个groups完好不能替代六项完整teacher准入。

**E M0–M4机制诊断**

验收：M0/M2/M3/M4各3种子、M1一个确定性结果；原关联/预测复现标准及所有21序列3316帧覆盖；原容差下完整独立指标与可追溯因果链。

**F 精确模型与恢复机制**

验收：有来源可追溯的有效恢复而非重排序；完整理论前提、反例和真实输入/模型适用性；独立精确参照和原专项容差。

**G 两个数据集真实机制主表**

验收：同检测/embedding/运动/源输入预算；所有学习方法3种子、完整序列、独立预测与指标；V2V train/test重叠披露；额外物理session映射不新增为门槛。

**H 恢复、历史与计算分配消融**

验收：每支持域独立输入/排序/状态/成本核验；恢复有可区分贡献，不被M3修复或更强特征解释。

**I 同资源与成本曲线**

验收：资源选择不看test；完整配置、实际资源、3重复、warmup和成本归因；无并行验收污染正式资源比较。

**J 公开方法原生复现**

验收：作者版本/实际权重/原生特征/split/类/频率/ROI/runtime/预测指标全部绑定；原生与统一输入适配表分列；不能把一步资格或CPU边界对照写成完整训练。

**K 独立指标、统计和论文交付**

验收：全部核心表无缺种子/序列、无占位指标；若保留原恢复优势主张，两数据集完整方法均须在合格协议下优于强单端和同资源多假设基线；否则原论文完成门槛仍未满足；全部恢复主张有因果见证且无未来信息/历史回写；两数据集对强单端、同势Top-K、同资源MHT完整比较；不以明显FP/FN或总体质量下降换局部身份改善。

## 运行规则

- 当前执行队列只有四类；历史成功控制和已失败尝试单列保存，不反复列作活项
- 每次优先完整收集当前尝试的数值与终态，再新增优化；没有证据不标通过或运行
- 同一不变来源和已合格报告复用；合同要求的发布、传输、执行及收集SHA仍完整核验，日常状态巡检不重复读取多GB旧文件
- 既有控制通过且未改动时不重跑；优化需完整结果一致和端到端改善，观察提速不当正式隔离成本
- 原48条件实际consumer已完成49项软件控制及独立审查；只在真实完整teacher与来源齐备后验证正向消费和Linux，不再增建同义readiness层；软件准备计0个新科学验收
- 每条已合格数据集/种子/支持域链独立放行下一依赖，不等待其他无依赖链；不改变该链的全集门槛
- 每轮报告真实开始时间、墙钟、当前阶段、生产证据时间、CPU累计/增量和未知ETA；队列/创建时间不冒充计算时间
- 不默认禁用GPU；后续入口允许使用通过物理交集及运行环境准入的GPU，保持独立重算、原输入/模型/预算和原容差。CPU实现不能因解除屏蔽就称为GPU计算，现有封版尝试如需切换须先完成完整一致性验证并保留已有结果。

总ETA未知。两数据集完整公平主表、恢复的独立价值、真实因果案例、独立全指标/统计和成本证据同时闭合才达到论文证据齐备；实验完成并不保证优势或发表。

[完整机器可读计划](/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/publication-readiness-plan.json) · [原始目标](/Users/lbin/.codex/attachments/fe7b3ffd-6692-4c11-8198-014dbd00c8f7/goal-objective.md) · [正式实验合同](/Users/lbin/thesis/tracking/recover-before-fuse/experiments.md)
