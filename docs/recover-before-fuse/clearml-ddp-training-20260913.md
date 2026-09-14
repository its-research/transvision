# ClearML 四卡、多机身份训练记录

搜索优先级模块的新增四卡训练器及其部署前提见[优先级 DDP 记录](priority-ddp-training-20260913.md)。
本文三个已完成种子仅指身份势模型，不代表优先级训练已完成。
本轮新增[5090 容器就绪检查](worker-files-readiness-20260913.md)确认指定 API／文件端口
均返回 `Network unreachable`，跨机器训练仍未启动；未重复提交已完成的身份种子。

## 范围与当前状态

用户要求训练在 ClearML 至少四卡 worker 上执行，排除 V100，并在可行时使用多台机器并行。研究代码仍位于 canonical transvision；论文项目仅保留原有论文构建工具。

本轮已完成资源查询、真实四进程 DDP 测试、部署包传输和内部 ClearML 发布。暂存目录为 `10.100.35.112:/home/lbin/Desktop/rbf-ddp-20260913.7ko8AP/`。2026-09-13，种子 1337、2027、3407 的 A100 任务均已完成 10 轮并通过结果回读核验；每轮均覆盖 90,651 行、1,440 个全局 batch，四个 rank 的模型摘要一致。三个运行还通过了跨种子的源码、数据、拟合配置和选择规则一致性检查。两个 5090 任务在四卡计算预检后因文件网络不可达失败，没有训练 epoch。剩余两个种子随后在现有 A100 四卡队列顺序完成，未实现多机同时训练。API 与文件服务分别为 `http://10.100.35.118:8008`、`http://10.100.35.118:8081`。

| 种子 | 第三轮 Task ID | 实际 worker |
|---|---|---|
| 1337 | `2cc70989d6fd48f7bfd6c715db9984f2` | `10.100.34.18-A100:gpu4,5,6,7` |
| 2027 | `c9366684866f489681ae924167610780` | `10.100.34.130-5090:gpu0,1,2,3` |
| 3407 | `04e19cbe1f5c4a959d1d77661c55794c` | `10.100.34.130-5090:gpu4,5,6,7` |

A100 前三轮的训练损失分别为 0.09160654021133754、0.06823111909203562、0.06785310157769969；每轮耗时分别为 117.638、107.616、107.489 s。首轮四个 rank 分别处理 22,682、22,666、22,655、22,648 行，总计 90,651 行。四个 rank 均有非空样本和实际优化器梯度更新。上述损失仅为训练目标，不是跟踪指标或方法收益。

## 已完成种子 1337 的验收

[A100 任务日志](http://10.100.35.118:8080/projects/e30491b7c54a48aea056748ec079fd95/experiments/2cc70989d6fd48f7bfd6c715db9984f2/output/log) 的在线状态为 `completed`，控制器输出 `RBF_DDP_TRAINING_COMPLETE`。10 轮训练耗时 1,087.804 s，末轮训练损失为 0.06536643790163578。运行记录包含四个不同的 A100 GPU UUID、同一容器主机、CUDA/NCCL 后端和四个实际 rank。此记录证明四卡训练实际执行，不证明四卡带来加速。

以下五份文件已从内部 ClearML 下载，逐项核对服务端 SHA-256；共 1,069,055 字节。首次下载因客户端原缓存目录不可写失败，随后仅为下载进程设置独立 `CLEARML_CACHE_DIR`，没有改变原目录权限或全局配置。

| 产物 | SHA-256 |
|---|---|
| `receipt.json` | `2c58f62e74a00269bca08bce612e3877150d84ba014e3ffce7b7a6e0f83fd983` |
| `plan.json` | `4b778956039c7f9ed447e88266b7f3cd1078bc958bb76e3deb11604d9310f20c` |
| `seed-1337/checkpoint.json` | `70609d3a7bf330d0d56b92150d0f97db511dfe4545c07cc193e993d0b77a0e96` |
| `seed-1337/epochs.jsonl` | `5807cedc1888a399b8f6b6fffbca674d2779cee0c70e41c58b25741e05c5f64b` |
| `seed-1337/weights.pt` | `7f991227a4689ba3d80a7e9e114d1b2cac790ddab24653ac568253eb7404d1e0` |

本地结果目录为 `/private/tmp/rbf-a100-result-1337.besRjh/`。新增离线验收入口 `tools/event_track_v2x/audit_forest_identity_ddp.py`，核对原训练清单、计划、全部 epoch、rank 覆盖和检查点关系，并通过正式加载器实际加载权重。加载后的模型参数摘要为 `fd291b7717bfb329c1dc76429465d851279b91cd9bf4cca413980fd94c8c7694`；参数相对初始化变化，全部子模块为 eval，全部参数禁止梯度。

验收回执 `offline-audit-uuid.json` 的 SHA-256 为 `a160d2ef94da29512ad5a6ea3cc9cd4276ddcc442f7cd9886c363dd6edfc3631`。离线核验器不声称自行检查远端状态；上文 `completed` 来自另一次在线 SDK 查询。两类证据不能相互替代。

这只是一个种子的开发训练验收。回执继续保留 `complete_three_seed_campaign=false`、`strict_pipeline_isolated_selection=false`、`tracking_validation_performed=false` 和 `paper_eligible=false`。未读取新的 val/test 指标，不能从训练损失下降推导 HOTA、IDF1 或恢复机制收益。

## 已完成种子 2027 的验收

[种子 2027 日志](http://10.100.35.118:8080/projects/e30491b7c54a48aea056748ec079fd95/experiments/fbd1abc3470544bea4fd18b271309930/output/log) 的在线状态为 `completed`，控制器给出完成标记。10 轮耗时 1,064.707 s；首轮／末轮训练损失为 0.08687675482603603／0.06517958428597766。五份产物已通过客户端独立缓存下载、服务端摘要核对和本地正式加载器验收，共 1,069,048 字节。

| 产物 | SHA-256 |
|---|---|
| `receipt.json` | `b1b5ee598c6371420b6646fb8d28ddfaca34b401939bf8a8d441db869ba6d7e0` |
| `plan.json` | `23747ef41d12704cc2931050247216162bd192f78cdc30c256fe5b5c37001b37` |
| `seed-2027/checkpoint.json` | `0d0289fc153ea66b5e022c4cc1dd02f1d4f0e552599a9d4f01e924e5dda0a33f` |
| `seed-2027/epochs.jsonl` | `2ab6eab9296b80e0326cdfda35fd889c90f85087fd5978e3800c267abbe1ebfd` |
| `seed-2027/weights.pt` | `7fd1d529d92e85ff77cf7c13e3f0654ee1a3cf6ac7179a6ab73e34c3abdcb1a8` |

本地结果目录为 `/private/tmp/rbf-a100-result-2027.yOejRx/`。离线验收回执 `offline-audit.json` 的 SHA-256 为 `3ed1ed9cd4d2196d905ce87225c5b7a4692ef3938f66d7022deca2d5d3afd74d`；实际加载的模型参数摘要为 `092507371b6945af28f18d5c01582db55618d221f46fb5a60ca9a9433b78a97b`。验收核对 4 个不同 GPU UUID、全部 10 轮覆盖、同步权重、初始化后的参数变化和冻结加载状态。训练数据摘要与种子 1337 相同，仍为固定 full train 开发训练，不是跟踪验证或完整三种子验收。

## 已完成种子 3407 与三种子汇总验收

[种子 3407 日志](http://10.100.35.118:8080/projects/e30491b7c54a48aea056748ec079fd95/experiments/6d91e8945e1449aab02582f29bc12cb8/output/log) 的在线状态已回读为 `completed`，控制器输出完成标记。10 轮耗时 1,071.837 s；首轮／末轮训练损失为 0.08820979941026222／0.06591343524820031。五份产物共 1,069,055 字节，已下载并逐项核对服务端摘要。

| 产物 | SHA-256 |
|---|---|
| `receipt.json` | `7d9fd6b0b45cc3e899bdfad7a8300f8c4abb5ada869b42d857f84dfd2915bda3` |
| `plan.json` | `e41af7b8d3b28a12fb4b4b9c7ad40e5715a1a5606f8ab627612b5faef3d8e1e1` |
| `seed-3407/checkpoint.json` | `96d1db472bef678112de2187046c7d8002df35bfe0dfeb153c746948d59f5a36` |
| `seed-3407/epochs.jsonl` | `4f961f7e4ccff71a560b142014026a5b11d8461b9c4d17fb1f50ee657ef0470f` |
| `seed-3407/weights.pt` | `ae3f1a50364353d9c24d5d973d3e6aaeacaa5beb4ceca390830a0f03bf9c6b8a` |

结果目录为 `/private/tmp/rbf-a100-result-3407.txs5K6/`。离线回执 `offline-audit.json` 的 SHA-256 为 `963facd5eafbc83cec42672b5a896b983c8fd31dce839e6dd35d985a8525423a`。正式加载的模型参数摘要为 `6ff18f64469a973a819a0fdd37b0657d6e3f92371fa68235af118ab2652ef4e9`；同样通过四个不同 GPU UUID、全部训练覆盖、同步权重和冻结加载检查。

新增汇总入口 `tools/event_track_v2x/audit_identity_campaign.py`。它重新调用三个单次验收，要求三个独立目录恰好对应种子 1337、2027、3407；除种子和实际运行设备外，全部计划字段必须相同。它还检查不同种子的模型摘要、各文件在验收期间未变，以及原始 full train 清单。14 项聚合逻辑测试通过，包含缺失／重复种子、重复目录、源码或配置不一致、模型重复和中途文件变化拒绝。实际三次 GPU 产物另经正式验收，不由测试夹具代替。

汇总回执为 `/private/tmp/spd-identity-fulltrain.jtGEnO/a100-three-seed-campaign-audit.json`，SHA-256 为 `0b0e77735c7ea57198c0f353339dbcf3b7a2044556945c799cece851519e0f64`。测试回执为同目录 `campaign-audit-tests-20260913.xml`，SHA-256 为 `349c31c4c230b2e4a8eede82c8af9c77c008cea047fb16e80f7049800ca12e86`。

只有汇总回执记录 `complete_three_seed_campaign=true`；原始三个单次回执保持不变。汇总仍明确 `remote_live_status_checked=false`、`multi_machine_concurrency_verified=false`、`strict_pipeline_isolated_selection=false`、`tracking_validation_performed=false` 和 `paper_eligible=false`。在线完成状态由另一次 ClearML 查询提供，不能由离线文件推断。三种子训练完成不等于优先级模型训练完成，也不等于全跟踪 validation 或论文收益成立。

## 并行资源分配

2026-09-13 11:25（北京时间）的在线 worker 查询显示以下三组没有当前任务，对应队列为空。随后三分钟的 GPU 遥测均返回利用率 0%；这是 worker 聚合遥测，不是逐卡独占证明。提交前仍需刷新占用状态，运行时核对四张实际分配的 GPU。

| 种子 | 物理机器 | GPU 编号 | 队列 |
|---|---|---|---|
| 1337 | `10.100.34.18-A100` | 4、5、6、7 | `GPU4-A100` |
| 2027 | `10.100.34.130-5090` | 一组四卡 | `GPU4-5090` |
| 3407 | `10.100.34.130-5090` | 另一组四卡 | `GPU4-5090` |

原并行计划使用 5090 的 0–3 与 4–7 卡，任务领取由现有 ClearML 调度器决定。计划中的三个独立任务各自使用四进程 DDP，合计两台物理机器、12 卡；不是跨机器组成一个 DDP 作业。5090 网络失败使该并行计划未能实现，不能把预检同时使用的卡数写成训练并发数。

A100 0–3 卡已有其他任务。提交程序排除与运行任务或排队八卡任务重叠的 worker，不将空闲八卡 worker 与其四卡子组重复计数，不停止其他任务、不改变队列或 worker 配置。V100 不进入提交矩阵。

2026-09-13 14:12（北京时间）再次通过现有 SDK 只读查询：A100 4–7 卡、5090 0–3 卡和 4–7 卡对应的四卡 worker 均无任务，现有四卡队列为空。A100 0–3 卡仍运行 `sam3_infer_roadside_20260901`，未占用或中止它。三个 A100 身份训练任务均回读为 `completed`，两个 5090 尝试仍为 `failed`。本次只验证调度状态，没有重新进入 5090 容器验证网络恢复，因此不能把空闲 worker 等同于可成功训练的 worker。没有重复提交已完成种子，也没有修改全局队列、路由或认证。

2026-09-13 15:02（北京时间）刷新现有 SDK 资源快照，仍有上述三组可调度的四卡 worker。它们分布在两台物理机器：一台 A100、一台 5090；5090 的两组四卡不能计为两台机器。本次没有取得 5090 到指定文件服务的连通性恢复证据，也没有重复投递已完成种子。后续独立训练可在网络条件满足后按四卡组并行；同一模型、样本和种子的重复任务仍须拒绝。

2026-09-13 17:00 前后的 SDK 只读回查仍返回 A100 4–7 卡和 5090 两组四卡可调度；三个 A100 身份种子均为 `completed`，两个历史 5090 尝试仍为 `failed`，失败日志保留到 `10.100.35.118:8081` 的网络不可达错误。从本机及 `10.100.35.112` 管理机访问 `10.100.34.130:22` 均超时，未能进入宿主机或容器核验网络恢复。SSH 超时本身不证明训练容器到文件服务器仍不可达；本次只能确认未取得恢复证据。没有新建重复种子任务、修改全局网络或换用未确认的文件服务地址。

## 训练合同与实现

新增入口为 `tools/event_track_v2x/train_forest_identity_ddp.py`。运行时要求单个 worker 上至少四个 CUDA/NCCL rank；本次部署控制器进一步固定为四卡，且只接受 A100 或 5090。CPU/Gloo 仅用于测试，不接受为真实完整训练运行。

每个种子使用同一份完整 train 分片、10 轮、全局 batch 64、原网络结构和原优化器参数。各 rank 分到全局 batch 的不同样本，不补重复行、不丢尾批次。局部损失按 `world_size × local_rows / global_rows` 缩放，抵消 DDP 对 rank 梯度的平均；空尾批次 rank 参与反向传播，但贡献为零、不增加样本计数。该缩放依据 [PyTorch DDP 的梯度平均语义](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html)，不是新的理论贡献。

每轮核对全局样本覆盖、各 rank 的实际非空 batch、非零梯度次数及同步后的模型摘要。只有三个种子的正式成功回执齐全才能报告三种子完成。单个种子的回执明确记录 `complete_three_seed_campaign=false`。

这是冻结上游 in-sample 输入上的开发训练，不是严格 OOF 模型选择；`paper_eligible=false`。A100 与 5090 的硬件差异会写入记录，不将这批混合硬件运行作为同算力基准，也不声称跨 GPU 位级复现。

## 部署包及发布范围

本地包位于 `/private/tmp/spd-identity-fulltrain.jtGEnO/ddp-package-v1/`。仅包含派生训练分片、必要研究源码和清单；不含 GT 全文、原始数据、完整检测／预测流或已有模型。派生分片含训练监督，不能称为无标签数据。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `source.tar.gz`，97 个源码文件 | 383,856 | `971319a89a2085df39738ac5f7cf6883a89d0783a187ee0621d81ee9720ec4ab` |
| `train-rows.tar.gz` | 150,530,976 | `6c00fbb160ceb31ff239605bcf5b9c1faafbc04b5e10695024bc62b84dc184ea` |

包清单 SHA-256：`63169e2f526e18fee0c5fcd05984de39e6a370a46c9804ca599e4c2d87697c52`。数据 manifest SHA-256：`6356cf9420b41efc38c20838ff6a87eccf643d0a5361475ebfbd392b1a658391`。

提交程序默认仅查询资源，必须显式 `--execute` 才会上传和入队。执行阶段将这份部署包交给内部 ClearML，运行结束后保留新训练的身份检查点及有限训练审计；这是新增多机训练的部署范围，不冒用历史 19 份结果文件的发布授权。未进行公共发布。

部署上传与三任务提交已通过独立工具审批。暂存传输共五个文件、150,948,640 字节；初次调用因本机 rsync 不支持 `--info=stats2` 退出，改用 `--stats` 后完成，没有重复创建暂存目录。内部包任务为 `79fa0cea63774a4b959e6ef4fd09deed`；清单和两个压缩包上传后均已回读摘要核对。

第三轮控制器 SHA-256 为 `88773aa1ba8814946240465f7bab01b9b5f8c2c856613cc232b3b2535ea9ca05`，提交器为 `c88b0cddd2efa9b7de399632208a0a24904fc388867993ba1ab261ae81883a8b`。三轮都使用同一训练源码和派生样本包；第二、三轮仅修正外层部署控制器，没有改动训练模型或数据。

## 启动失败及修复记录

首轮三个任务在进入 GPU 预检前因 `General/` 参数命名空间失败；新增参数规范化和冲突检查后，第二轮三组均通过四卡矩阵计算，实际为 PyTorch `2.10.0+cu128`、CUDA `12.8`。A100 型号为 `NVIDIA A100-PCIE-40GB`，5090 型号为 `NVIDIA GeForce RTX 5090`。

第二轮随后因文件服务地址不匹配失败：worker 的 `api.files_server` 指向 `http://10.100.34.118:8081`，包的 URL 指向用户指定的 `http://10.100.35.118:8081`。SDK 代码确认只对配置内的文件主机附带正常认证头。第三轮通过任务内的官方 `CLEARML_FILES_HOST` 设置使用指定地址，不关闭认证、不复制其他账户凭据，也不改变服务器或 worker 的全局配置。下载失败时直接报错，不尝试其他主机或凭据。

| 轮次 | 种子 1337 | 种子 2027 | 种子 3407 |
|---|---|---|---|
| 1，参数读取失败 | `198c1b067cbd4c0d845230b23c2500e2` | `1a0f76d2d8c44ff9a2cc44ec4e9efdf0` | `3ddd410388ce4eb9bf7d641bfe36676c` |
| 2，文件下载失败 | `85a5ea97725b48be80a438f4adc84e15` | `af59e57876194ea4b517ec6916d1debc` | `4ba80bd9ab384f72a6a42e6912346f6f` |

失败任务和各版本脚本保留，第三轮创建独立 Task ID，没有重置或覆盖原任务。失败轮次不是训练结果。

第三轮的端点配置修复已让 A100 成功下载、校验并训练，但 5090 出现独立网络问题：两个容器均报告到 `10.100.35.118:8081` 的 `Network is unreachable`，而不是第三次相同的 401。已保留日志，没有关闭鉴权、改变容器网络隔离或修改宿主机路由。

为核实 `10.100.34.118` 是否属于同一服务器，尝试只读 SSH 查询用户指定的 `10.100.35.118` 网卡地址，但现有账号认证失败，未获得物理服务器地址证明。已请用户确认两个地址的服务关系及允许的访问地址，或由管理员修复 5090 路由；未将训练数据或模型转发到未确认地址。A100 不受此等待影响。

提交器先增加了 `--seeds 2027 3407`，通过 11 项测试，回执 `/private/tmp/spd-identity-fulltrain.jtGEnO/clearml-selected-retry-regression-20260913.xml` 的 SHA-256 为 `0661d88d084d67f9b6836302f71fba33470e9a12cf9c2d1542bd973c63163345`。这一中间版本没有独立部署。

种子 1337 完成后，只读检查确认 A100 4–7 卡空闲、四卡队列为空、其他 A100 作业继续占用 0–3 卡。第四版提交器使用 `--seeds 2027 3407 --a100-fallback --attempt 4`，只将剩余两个种子交给现有四卡 A100 队列。该模式允许任务顺序执行，不要求两个空闲四卡 worker；不占用八卡队列、不重新训练 1337，也不改变其他作业。提交前按同一训练包和种子跨硬件检查，拒绝另建已有 `queued`、`in_progress` 或 `completed` 的种子任务。因此网络恢复后不能再把同两个种子重复提交到 5090。

| 种子 | 第四轮 Task ID | 提交队列 | 12:26 提交回读 |
|---|---|---|---|
| 2027 | `fbd1abc3470544bea4fd18b271309930` | `GPU4-A100` | `queued` |
| 3407 | `6d91e8945e1449aab02582f29bc12cb8` | `GPU4-A100` | `queued` |

提交后的首次在线回查显示种子 2027 已由 `10.100.34.18-A100:gpu4,5,6,7` 执行，状态为 `in_progress`。第 1 轮完成 90,651 行、1,440 个全局 batch，4 个 rank 的权重摘要一致，训练损失为 0.08687675482603603，耗时 107.044 s。当时种子 3407 为 `queued`，没有 worker 或结果文件。这是历史启动快照；两者之后均完成，验收见上文。

第四版提交器 SHA-256 为 `ca78afc351851c12b7144582f83df29f62624736df6aba25f08bd8af9a44a2c1`，远端文件名为 `submit_forest_identity_ddp_v4.py`。仍使用第三版已验证控制器、同一包任务和同一数据／训练源码摘要，没有重新上传派生训练包。15 项提交器测试通过，耗时 0.04 s；回执 `/private/tmp/spd-identity-fulltrain.jtGEnO/clearml-a100-fallback-tests-20260913.xml` 的 SHA-256 为 `02ce92cc623b9a38aad4eef779bb816da766be8b228b54e6c442572fb6f9f160`。这些任务是同机继续训练，不是已达成多机并行。

## 已完成检查与后续验收

30 项测试通过，耗时 4.19 s，包括四个独立 Gloo 进程的梯度对照、尾批次覆盖、检查点加载、原单进程回归、GPU 重叠排除及安全解包。测试回执为 `/private/tmp/spd-identity-fulltrain.jtGEnO/ddp-preflight-tests-20260913.xml`，SHA-256 为 `de641ed95bbf139956e67ef183d327f6fbc4932f856824585dba862c8a10ad8b`。首轮受沙箱回环网络限制失败；随后获准绑定本机回环接口，完成真实多进程回归，未将失败轮算作通过。

部署修复后，10 项外层控制器测试通过，耗时 0.05 s，新增覆盖 `General/` 前缀、重复参数和下载主机限制。回执 `/private/tmp/spd-identity-fulltrain.jtGEnO/clearml-files-host-regression-20260913.xml` 的 SHA-256 为 `e4512de0732c4e37248ded0828d492f264838a768a3354feb2f36a5e588af236`。这些测试不替代真实 CUDA/NCCL 训练验收。

种子 1337、2027、3407 均已通过 GPU 矩阵计算、NCCL/DDP 初始化、实际训练和完整结果核验。控制器拒绝 CPU 回退，`torchrun` 非零退出即失败；`queued` 或 `in_progress` 不能代替训练完成。

离线验收器的 10 项 DDP 测试通过，耗时 3.05 s，包括真实四进程梯度对照、空尾批次、结果加载、篡改 epoch 文件拒绝和伪造三种子完成标记拒绝。回执 `/private/tmp/spd-identity-fulltrain.jtGEnO/ddp-artifact-audit-final-tests-20260913.xml` 的 SHA-256 为 `e0b1a0764b65936a7ba8b5982997f279b0170c6ca6d246b788ad5fbe0095e36e`。

前置本地 CPU 三种子训练已经完成，详见 [本地训练记录](spd-local-training-20260913.md)。随后启动的本地计算分配 teacher 已在切换部署时主动中止，进程返回 130；已有部分审计和输出保留，没有完整 teacher 回执。部分轨迹已出现质量上界宽松和风险回退，尚不能用于宣称算法优势，计算分配训练与真实跟踪验证仍未完成。

针对 teacher 的局部缓存开销，已完成不改变推断结果的索引优化和回归，详见 [分量缓存剖析](component-cache-profile-20260913.md)。该改动没有进入本次封存的训练源码包，不影响三个身份模型训练输入及来源一致性。

后续又完成 [教师试算复用对照](teacher-probe-reuse-20260913.md)：同一诊断状态的 2,304 个标签与预测保持一致，实际试算减少到 264 次。但该状态的标签全部接近零，模型风险界接近 1，揭示当前一步目标的局部退化。它不构成可恢复机制的真实跟踪收益；完整教师采集与优先级拟合仍未完成。

针对该退化，已增加[前沿完整类候选变体](frontier-completion-20260913.md)，并使用已核验的 A100 种子 1337 开始首个 train 序列全部 195 帧的教师诊断。它是离线样本准备，不是本地参数训练，也不能替代 46 个 train 序列的完整采集。

该开发序列已于 14:22 完成全部 195 帧并导出候选组。194 帧触发风险回退，187 帧的模型风险界不低于 0.99；尚不支持收益主张，也未启动真实优先级训练。完整回执、负面结果与后续覆盖感知入口测试见[当前动作风险记录](decision-local-risk-20260913.md)。多机训练条件保持不变，但不能将空闲 GPU、教师完成或小型夹具测试当成全套实验完成。
