# 单元7生产终态接续复核

2026-11-06 17:36 北京时间，经112读取ClearML Task02fb34e987ff49398bc7bed33d618e44，仍in_progress；native墙钟3841.165秒、CPU3842.50秒、输出211086135字节，oom/oom_kill均0。完整生产archive/seal/result尚未注册，不启动独立阶段。

本轮只读核对112上的实际dispatch_unit7_scientific_stages_prepared.py、unit7_complete_independent_task_prepared.py、complete_unit7_pipeline_source_guarded_candidate.py及unit7_collection_storage_admission.py。

独立接续顺序：

1. 精确原producer completed、bootstrap SHA与四项必要产物匹配。
2. 前四卡worker空闲、八卡worker无任务、后四卡仍占用，保证当前共享队列只有指定前四卡可接取；心跳小于120秒。若后四卡也空闲，现有唯一调度条件不成立，不能直接沿用该调度器入队。
3. 原source preflight身份和1129字节报告完整SHA复核，发布包身份不变。单次dispatch claim不得重复。
4. 新容器完整下载11成员包并核验原SHA；内存按宿主available与cgroup剩余额度较小值核验8GiB门槛。
5. 来源门槛重新检查；完整流式扫描注册archive，核验整包SHA、gzip CRC/ISIZE、所有成员类型/路径/重复项与解压总字节；该扫描不保留正文。完整收集仍须逐文件SHA，扫描不能代替。
6. 容量包含全部注册产物、全部解压成员、26690158字节scorer输入、512MiB输出日志余量及8GiB磁盘保留；报告不把软件估计冒充未来实际峰值。
7. 容量通过后才消费pipeline execution claim；执行完整收集、原六项独立检查及终态全SHA。最后再次确认producer仍completed且注册集合未变。

输出小报告上传，所有输出另有完整SHA清单及原生保存路径；大产物仍需核对实际可持续保存，不能将清单视为跨主机完整备份。原科学授权各一次仍保持，生产已使用、独立尚未使用。没有创建独立Task或改动现有任务。

## 22:41 运行可观测性源码核对

112经ClearML读取实际02fb任务脚本确认：原生子进程stdout/stderr写入worker本地`complete-unit.log`；父进程循环仅上传wall、子进程CPU、内存和输出字节，不上传已提交事件计数；只有child终态后才注册`complete-unit-log`（脚本199–216行）。因此当前平台日志没有事件计数是入口观测边界，不能解释为零事件、卡住或已完成百分比，也不能用CPU/字节外推221事件完成度。仍不修改活入口、不另起宿主诊断；生产终态后沿原已授权产物收集路径核对完整日志、seal和归档。

22:41:38任务仍in_progress、仅9份准入产物。112保存的源码观察报告`/home/lbin/rbf-V2V-unit7-original-source-preparation-20261106/production-observability-source-review-20261106T2241.json` SHA256 `1f7177194395520fafae0acd694919c41d845b663d17ba2588f6299999bebcc2`。只读源码审查，不是数值验收或任务切换。

## 已注册资源产物完整读回补充

在112通过自有SDK缓存完整读回生产Task 02fb34e987ff49398bc7bed33d618e44 的 `complete-unit-resource`，5802字节，SHA256 `5a0922c137375b0a880fb80886ef140726c050bef83a6f41d7be0d12f5f49c29` 与注册值一致。报告记录准入时四个物理UUID：1642eeac-6fb2-2254-4570-96e99452c91c、9223f2c3-17fd-8061-278c-abafe4c8a1de、cb25f779-7739-80fd-839f-518b96d9fdce、d19154bd-5c8e-8bbb-f9e3-4692c8fbe550；原单元221事件、28288因子行与原分区/配置/模型/缓存绑定存在。

该报告是任务准入时的证据，不是当前GPU进程、利用率、内存/磁盘空闲快照。不能用其中原MemAvailable或磁盘数值替代新鲜准入，也未核实其他活任务当前物理UUID。因此已知三个worker的声明资源交集仍需保留，物理交集未证明，不新增调度、不改现有任务。此次仅读回注册小产物，无新探针/源码发布/模型运行。

## 2026-11-07 01:31 九份注册准入小产物全部读回

112 `admission-nine-small-artifacts-readback-once` 保存完整9份原JSON，共141831字节；逐份原注册大小/SHA一致。汇总 `complete-readback.json` SHA256 `465bff415590165baed7e369d713405a0db00aa0564282e8b11202bd0ee5290f`。范围包含actual-production-runtime-manifest、complete-unit-resource、nominal-original-path-readback、四device原行准入、production-runtime-overlay及runtime-admission-result。完整文件与各原注册SHA保存在112该目录，可供终态收集来源连接；不要重新下载以增加检查数。

这仅证明准入小产物完整读回，不证明当前生产完成、全部模型行数值通过、完整轨迹通过或独立六项验收。full_scorer/full_replay/full_sequence/full81等原未通过标志必须保留；不能将selected_original_rows_passed或CUDA_runtime_admitted扩大为正式科学接受。未读取大数据库、未运行模型/数值、未修改任何任务，Mac只写本文字。

## 已注册准入报告内部来源交叉核验

112对已保存小报告核对：151文件actual runtime manifest的完整SHA等于overlay登记SHA；原/候选source SHA与manifest.verification_overlay一致，sources字典中persistent_forest.py的40847字节候选SHA也直接相等；仅_predict AST变化声明及65536预算保持；4设备原行报告SHA与runtime-admission报告一致，每设备93条原选择行、原1e-4容差，两个报告的4物理UUID列表一致；原单元221事件/28288因子一致。完整连接报告 `admission-cross-binding-complete.json` SHA256 `5d24574491b2e6426e9001ac6b630482bb5ae7f4715a8539e7796c98ce2cc18b`。

以上是准入报告间静态字段与SHA连接，不重新证明报告内部的AST/模型数值，不证明当前GPU占用或生产终态。历史packaging manifest的dispatch_ready=false与unmet_launch_gates属于打包时状态，不拿来推断当前Task失败；后续实际准入与完整数值资格仍分别核验。full_scorer/full_replay/full_sequence/full81未接受标志保留。未调用模型、oracle、数据库或新任务。
# 2026-11-07 终态接续入口静态复核补记

独立Task6ad715d5…外层注册报告收集入口已在112准备：`read_unit7_registered_outputs_once_prepared.py`，6229字节，SHA256 `5753192d4a67b20c6a40d210a474f973869efd1078e71db9e0324581c5d6a27e`。只绑定该Task、原script SHA28094277…、原producer02fb34e…和实际原科学grant；未终态返回held、不创建输出或消费下载尝试。终态后要求注册集合稳定，逐份全SHA/精确字节读回，并核对native inventory的原路径、≤2MiB报告来源、编码键唯一与terminal身份。大native payload不由此外层下载；它只接受传输身份，不自行接受六项数值或full81。

112语法及6项纯合成协议控制已通过（合法、变SHA、缺报告、路径逃逸、编码键碰撞、错误Task），未调用oracle或实际报告。随后已实际核对非终态门槛，Task仍in_progress，输出held_not_terminal=true、download_attempt_consumed=false。最终目录`unit7-independent-registered-readback-once`尚未创建；不要以软件检查通过代替实际终态/报告收集。后续由同一入口首次读回真实终态，再逐项审查原六项报告及worker终态收据，不新增数值尝试。

03:35后静态核对实际ClearML生产script：第204–205行CPU来自`/proc/{child.pid}/stat`的utime+stime，只统计`run_original_sequence_replay_candidate.py`这个原生回放子PID，不包含其后代进程累计量。第201行stdout/stderr写complete-unit.log，只有child终态后第216行才注册该日志；因此平台周期日志无法提供内部事件/当前栈。若回放包装器再启动后代，低CPU不能说明所有计算均停；本轮有限两个准备目录和已验SHA的11.38MB派生源码包未含该包装器正文，尚未证明有/无后代。不能把低CPU直接称死锁、空转或主机资源问题。未附加进程、读内存、发信号或发布新探针。

03:23:10经112读取Task02fb34e…全部已有scalar系列：仅`:monitor:gpu`和`:monitor:machine`，没有业务事件完成计数。最后monitor x=38889，GPU0–3 utilization约52.635/58.327/49.769/59.038%，但task归属显存系列仅GPU0约0.209GB、其余0，且同机八卡worker另有训练。不能把设备总体利用率归因于V2V、宣称V2V在四GPU计算，或由machine CPU26.994%替代native process CPU增量。该x的采样时间也不是生产事件时间。任务当时仍in_progress；没有新增事件计数/终态或物理UUID排他证明。无需继续重复读取相同monitor系列来估算完成率；未来以业务终态/完整归档及独立验收为准。

在112只读检查当前文件，未执行collector/oracle：`collect_complete_unit7_candidate.py`仍含Mac形态B根路径，`collect_unit7_six_checks_linux.py`旧准备版也要求该命名空间。不得直接在Mac运行这些脚本，也不能把文件名含linux视为可在任意路径执行。当前科学调度入口应保持为`dispatch_unit7_scientific_stages_prepared.py --stage independent`所绑定的原28094277…入口及ebfd68fb…source-guarded pipeline；由已准备的运行命名空间承接，不临时替换成上述旧准备版。

静态核对当前dispatcher仍要求producer completed及原生产script SHA08a0ea9a…、原grant/source contract，才创建独立阶段；单次claim排他写入，未执行该阶段。旧Linux collector的收口条件包括监督器真实exit0、六项报告集合、221事件/28288因子、数据库与注册产物完整SHA、strict case join和原1e-8/1e-4；`full81_accepted`仍false。这些是代码要求，不是已通过证据；实际接续还必须使用当前合同中钉住的最终terminal-reader版本及完整容量准入，不凭旧准备报告中的pending/语法通过字段作当前资格判断。

