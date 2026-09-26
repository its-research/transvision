# SPD 嵌套训练准入进展

## 实际状态

2026-09-18 本轮通过 ClearML API 核验样本任务 `46d52f291a7d4d89ab3a281b6aa47364`：`in_progress`，最新更新时间 `00:51:54.844 UTC`，无产物。日志已到 `SEALED_INPUTS_READY`；不据此认定样本完成或内部计算持续推进。本轮没有重启该任务，没有修改其冻结代码包。

## 训练范围核验

直接读取远端已有校准 JSON 并重新计算 SHA-256：

| 资产 | 哈希 | 事实及限制 |
|---|---|---|
| fit37 校准 | `47f6c236440c4e946530faeef57a5a64cd4ce183beaf9ab7093e90dd7fc6802e` | 37 个 fit 序列；`in_sample_detector_predictions=true`，不是独立检测器预测上的校准 |
| held9 crossfit 校准 | `9e6c73af77142604bda1962d92460ba4a412262e9b5128f602d69638f7e136b0` | 最终校准 fit 为原 9 个 holdout 序列；`diagnostic_scope=exploratory_after_inspection_of_held9`；不能用于宣称同一 9 组的身份选模与上游隔离 |

两个资产均明确 `paper_eligible=false`，不得修改历史证据标签。当前运行的样本任务使用 held9 校准缓存，仍有开发诊断价值，但不能直接通过正式训练器的隔离门槛。

## 新增下一轮划分计划

已添加独立工具 `work_dirs/rbf-paper-stages-20260917/tools/event_track_v2x/plan_nested_training_partition.py`。保持原外层 37/9 不变，在外层 fit 内采用固定哈希划分 30 个检测器训练组与 7 个校准组；身份模型 fit 仍为 37 组，选模为原 9 组。不使用官方 val/test，不修改官方划分。

输入见 `spd-nested-outer-partition-20260918.json`，生成结果见 `spd-nested-training-plan-20260918.json`，结果绑定输入文件 SHA-256。这里沿用已有序列 ID 作为待核验的组标识；若多个 ID 属于同一采集组，必须先按原始元数据归组并重新生成，不能仅凭 ID 不同宣称独立。

该结果为 `planned_not_trained`，故 `upstream_sequence_isolated=false`。仍需官方 train 完整性及采集组映射证据、30 组上的检测器训练/选择记录、7 组上的校准记录、对应新缓存与训练监督绑定，才能形成真实隔离回执。原 9 组已被检查的事实必须披露，不能称为未触碰测试集。

复现命令（输出必须不存在）：

```sh
python work_dirs/rbf-paper-stages-20260917/tools/event_track_v2x/plan_nested_training_partition.py \
  --partition docs/recover-before-fuse/spd-nested-outer-partition-20260918.json \
  --output /tmp/spd-nested-training-plan-new.json
```

新增测试 `test_nested_training_partition.py`：11 passed。检查角色互斥、外层不变、输入顺序无关、重复/缺失/非法组与额外 test 字段拒绝；这不是训练完成或性能验证。下一步先核验采集组映射并准备相应 train-only 输入，再开展检测器重训、独立校准、身份训练与选模。已有任务产物到达后单独验收，不与新计划混用。

## 续轮：SPD 序列分组核验及 ClearML 发布

重新读取论文 `experiments.md` 第 4.1 节确认：SPD 要求官方序列级隔离，V2V4Real 才要求原始采集会话级隔离。不能把后者的会话门槛额外施加给 SPD；也不宣称本次核验证明了物理采集会话独立。

对固定 SHA-256 的原始 metadata 执行 `audit_spd_sequence_partition.py`，核验结果：两端均覆盖全部 46 个声明序列；车端 8,504 帧、路端 7,834 帧；7,445 个跨端配对均能反查到对应源帧且属于同一序列；两端分别不存在跨序列时间戳复用及时间区间重叠。未读取 GT 标签。官方 train 成员资格沿用既有固定来源审计，本次元数据检查不单独替代该来源证明。

本次 `sequence_to_group` 采用官方 sequence ID，`grouping_unit=official_sequence`，不再等待 SPD 物理会话映射。此前通用计划的 `recording_group` 字段只是输入分组占位语义，不能作为物理会话隔离证据；SPD 实际执行应绑定本序列回执。后续仍需新检测器训练及校准记录，`upstream_training_isolation_verified=false`。

已发布 ClearML 任务 **`2158a7186d184d8bb18e7ef74c45b0ea`**，任务实时回读为 completed；每个 artifact 均完整下载流回读并核验注册哈希、实际字节数与本地哈希：

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| sequence-audit | 10462 | `6895832c96994567d40953fd6fe53d040c47766313b49aff02362326145b64ac` |
| outer-partition | 397 | `82d12e9d58fefcace20b7ad8a0fb9f46a8f626bad0578b593c87f2c5acbca2a2` |
| nested-plan | 2574 | `387bdcc476374a76db576f5ebd5b1513fe55edc4be5d04ffb570e3bc43f67f6e` |
| audit-source | 5075 | `7e2e2e9f1f54e5af9fea2648416069e5021e431e49d71ae2c420beb1db59223f` |

原始元数据哈希：vehicle `300f1efe0d3359413cdd293840fc1214a95395ab9cb801131a44910e0523a24c`；infrastructure `691e7cc55b4221604ba13cab6049c99ce696b2d6a3329b1d79dd328a8b9e5ff9`；cooperative `787b7e4dfa3fb9d97ba0a6a4a216a2990af990dc9226ceec01053e7d2e7da984`。

新增序列核验测试 8 项，与嵌套划分测试合计 **19 passed**。发布任务 completed 只代表资料发布验收完成，不代表检测器/身份训练完成。下一步从 ClearML 拉取上述绑定，准备 30 组检测器训练输入并核验排除另 16 组，再启动新的隔离训练。

## 续轮：真实输入子集预检

新增 `prepare_spd_nested_inputs.py`，只接受固定父输入 manifest SHA `65815abf7ec0d8aadccc1bf8d1fc095cca26ae09292392ddcebaf838743df8da` 和已发布的 nested-plan SHA。通过已核验父清单选择 30 个序列的原始图像、标签与标定引用，不读 val/test；复制时逐文件校验字节哈希，生成独立新 manifest 和空 val/test 的转换器兼容 split。不复用原 37 组转换结果冒充 30 组输入。

在管理机真实父数据目录执行只读预检，结果如下：

| 项目 | 结果 |
|---|---:|
| 车端训练帧 | 5,360 |
| 路端训练帧 | 4,989 |
| 原始载荷文件 | 67,454 |
| 原始载荷字节 | 2,610,172,840 |
| 排除序列数 | 16（7 校准 + 9 选模） |
| 管理机当轮可用字节 | 3,197,083,648 |
| 最低复制空间（载荷 + 1 GiB） | 3,683,914,664 |

预检明确 `all_payload_hashes_verified=false`、`input_view_created=false`：此阶段只检查引用、元数据哈希及文件大小，所有载荷完整哈希须由后续复制完成验证。空间不足，未在管理机创建大文件副本或删除旧资产；应在 worker 新容器内从 ClearML 拉取父包，再生成新子集、重新转换及打包发布。

子集工具测试 13 项；连同序列审计和嵌套划分测试共 **32 passed**，包含真实小样本复制、排除校准文件、manifest 哈希、同大小内容篡改拒绝、空间不足不产生输出及已存在输出拒绝。不是正式数据转换或重训完成。

后续可复用父包任务 `5da9693dfce54d85b57ab0ca663dbb41`，本轮实时核验 completed。`package-manifest` SHA `9c3f3b8b0c59db69da6c4280b506bf575b6ca3b43559e3a9c28b2828ca7ec6ac`；`train-inputs` 3,235,557,989 bytes，SHA `2286b2d17866ebfba7b563fad7cce158f0c66f6c38ca40f537b8264f57393477`。现有 `run_cooptrack_a100.py` 的 cohort 验证固定为 37/9，新 30/7/9 包必须增加独立 cohort 验证及批量配置，不能只替换包名绕过检查。

## 续轮：worker 准备任务已启动

完成独立入口 `run_spd_nested_preparation_clearml.py`，打包三个固定工具并发布源码冻结任务 `becfe98115674c64ab644285382ff8b7`。代码包和入口均完整回读核验后提交任务 `b00a845c27fc49778d778c8cb6f5c170`；实际接取 worker 为 `10.100.34.18-A100:gpu0,1,2,3`，当轮观察为 in_progress，尚无产物。原样本任务继续在 GPU 4–7 上运行，未停止或重启。调度前同时核验 GPU4/GPU8 队列为空、共享八卡 worker 空闲，避免重叠占卡。详见 `spd-nested-preparation-dispatch-20260918.json`。

准备入口执行：容器 45 GiB 可用空间预检 → 拉取哈希固定的父包 → 保留容器内父目录 → 生成仅 30 序列输入并校验每个文件 → 用固定官方源码重新转换且检查帧/标签没有删除或插值 → 仅打包 inventory 允许文件 → 上传并完整流回读校验。此任务不训练模型，产物准备完成必须以最终上传回读和任务终态证明。

准备入口及直接关联测试 **43 passed**。后续训练入口在 runtime 隔离工作树新增 `eventtrack_a100_nested_package_v1` 验证，检查 30/7/9 角色互斥、固定双层哈希划分与 ClearML 原计划实字节绑定；34 项训练控制器测试通过。30 个序列允许的四卡候选 batch-per-GPU 为 2、4，不能沿用 37 序列下的 8。本地控制器修改发生在准备包冻结之后，不改变已提交任务使用的原工具字节；正式训练仍未启动。

## 续轮：真实子集复制与转换已完成，整包待发布

任务 `b00a845c27fc49778d778c8cb6f5c170` 已产生 `input-readback`：1,155 bytes，SHA-256 `0c4e363a329c8cc20e78229433d4de4eee46beaf14983e8f4cba08c668c9032a`。管理机独立下载完整回执并比对注册哈希及长度，确认 `input_view_created=true`、`all_payload_hashes_verified=true`，67,454 个原始载荷、2,610,172,840 bytes；新输入 manifest SHA `0717627ca135430fa765ffc2a4a0a7323a5b0cfb7106fb44ab8d1f98a57317c6`。

后续日志出现 `NESTED_CONVERSION_VERIFIED`。转换记录报告车端 5,360 帧、56,647 条标注（442 帧未报告位置）；路端 4,989 帧、77,285 条标注；转换 manifest 的内部 `content_sha256` 为 `775aeab5dd34075fe6ae2c0bcc9fb29ff02cf069b8e70a498971f3e8f301496d`，不能将内部内容哈希当作完整 JSON 文件哈希。

截至当轮最后核验，任务仍 in_progress（更新时间 `2026-09-18 01:14:32.646 UTC`），仅有上述回执，尚未取得完整训练包。新增独立 `verify_spd_nested_package.py`，待任务 completed 后流式下载整个 archive，逐文件核对 inventory、哈希、metadata 序列及帧数、空 val/test 划分、输入与转换哈希链，并拒绝额外/缺失/链接/重复文件。不会在管理机落盘第二份大包。

独立验证器、原流验证器及训练入口相关测试共 **49 passed**。工具已部署管理机隔离目录，实际完整训练包验收尚未运行；重训仍未开始，不能以工具测试或中间转换日志代替包验收。

## 续轮：整包独立验收通过，seed 1337 重训已提交

准备任务现已 completed，全部产物完成上传与 worker 回读。另在管理机独立流式读取完整 2,568,106,796 bytes 训练包，核验 67,489 个文件、2,858,065,740 解压字节、全部 inventory 哈希及真实 metadata 帧数与划分。流式验证退出码 0，未在管理机存储第二份大包。

训练包 manifest SHA `3fb26359309434a17e7cd93d8fed1b433a60f60d898e3bc531507515e94437e5`；archive SHA `46379199afd1a4d012b7a36f859364a0d3b93a836f8082b1256c49d9f8553c78`；独立验收回执 SHA `612444f44bdf5d163742b12bc24b511ee6ba5fc03073d23404b1a12b46e4c1d5`。

验收回执、训练控制器、验证器及直接依赖、测试源码与真实 49 passed / 0 skipped / 0 failures / 0 errors XML 已发布到独立冻结任务 `165f12fe774c4ff1b43fbe25b628a496`，每个 artifact 完整回读校验后任务 verified completed。训练控制器 SHA `be8651ce7192ef3aa1a224d8330edeb692fdfeeec8ced1ab73231e3c298b3c2c`。

再核验 GPU 0–3 worker 与八卡共享 worker 空闲、GPU4/GPU8 队列均无等待作业后，提交 `f2ced9be51ef4e49a1a05dd8f4ecc3d3`（SPD nested-fit30 R50 dual-side seed1337 attempt1），每端 24 epochs，四 A100，batch-per-GPU 候选 2/4，保持原学习率并记录选择。新任务已入队；排队不等于实际优化器启动。冻结源码不再修改，详见 `spd-nested-fit30-training-dispatch-20260918.json`。

2027/3407 新嵌套训练仍未运行；独立校准、身份训练、基线和论文统计均不能因本次输入包验收而标记完成。

## 续轮：四卡探测进入实际计算，后续权重验收适配完成

训练任务 `f2ced9be51ef4e49a1a05dd8f4ecc3d3` 已成功完成车端 batch-per-GPU=2、每 rank 32 iterations 的探测。`probe-b2-vehicle-side-profile` 已从 ClearML 完整回读，1,685 bytes，SHA `9a910b69dc947c5909521659bb375bd9ae890436c9e5c6855fb3bdb626caa75f`，`success=true`、`returncode=0`、四个不同 rank 均为 A100。最大 peak reserved 3,661,627,392 bytes，单卡 total 42,405,855,232 bytes；探测 elapsed 61.26 s。这是批量与运行正确性探测，不是正式每端 24 epochs 完成，探测权重不用于正式训练。

在独立 runtime 工作树扩展 `collect_spd_development_training.py` 与 `spd_export_training_binding.py`：保持历史 37/9 包和控制器哈希准入，同时新增固定新包/控制器/输入哈希的 30/7/9 验收路径。新路径要求真实 completed、双端最终权重及启动/配置/完成回执齐备，拒绝 probe、早期权重、混用旧包、校准组进入 fit、超过 30 序列承载能力的 batch，以及未声明 seed。收集后仍执行原完整模型/优化器 tensor 检查，不把 metadata 相等当作权重安全加载通过。

本轮这些改动与训练入口相关回归 **66 passed**。历史多种子控制器测试改用明确历史 fixture，避免把更新后的工作树代码误当成旧冻结字节；新增测试将当前真实源码哈希绑定新控制器。运行中的训练控制器 SHA 仍为 `be8651ce7192ef3aa1a224d8330edeb692fdfeeec8ced1ab73231e3c298b3c2c`，未修改。本轮尚未运行真实最终权重收集，因为训练未完成。

## 续轮：正式车端训练启动已核验

首轮 SSH 状态观察因 `Can't assign requested address / Broken pipe` 中断；重新查询同一任务后恢复，未重启或重复提交实验。ClearML 仍报告训练任务 in_progress，正式训练日志在 `2026-09-18 01:28:14.924 UTC` 到达 `50/8040`，与此前 32 步 probe 区分。

从 ClearML 完整回读并校验四项启动证据：batch-selection SHA `82368b26327c85fc0c955c9e13bc42f8eec8b25ec08d69643c92a5749b39b1cf`；车端 launch SHA `8572aa097a9eb3852403983f28db8de3e30c9901aae3f9bc01684c2cce6feaba`；optimizer-startup SHA `3f142c20b2ac6c48821ba40164581ce167ebef60ded9527e191e8a6d59d5af6a`；resolved config SHA `2ddc2fe761423c94a448f700261068a2e5e7200d61289e9cfcd1809508833e13`。

核验选定每卡 batch=4、四卡全局 batch=16，launch 与 optimizer 资源一致；`batch_probe_only=false`；optimizer 内 config 哈希匹配真实配置；启动 loss `21.805768966674805` 有限，backbone 最大参数变化 `0.00010743364691734314` 为有限正值。launch 声明 seed 1337、24 epochs、30 个 fit 序列、ImageNet-only 初始化，无 val/test 加载及原标签修改。因此可以确认正式车端训练已启动且发生参数更新；不能据此宣称路端训练启动、任一端完成或性能达标。

原行样本任务 `46d52f291a7d4d89ab3a281b6aa47364` 在同轮仍 in_progress，暂无产物，不据观察连接中断或运行耗时认定其终止。

## 续轮：行样本完成，三种子训练均已安排

行样本任务后来正常 completed。manifest、readback 和 rows 三项产物均从 ClearML 独立完整回读验证；rows 为 223,119,444 bytes，SHA `42532b6f78e2b0ca2074267dfbd36c4481ddd432e346c2305b0f5bc3e8d84087`。worker 回执报告 46 序列、238,746 节点、112,169 监督行均逐行重载。该资产使用此前 held9 校准版本，保持开发诊断资格，不改为严格隔离训练证据；详见 `spd-paper-rows-completion-20260918.json`。

行样本完成后 GPU 4–7 释放，确认八卡共享 worker 未运行、两个队列无等待任务，再提交 seed 2027，已由 GPU 4–7 接取。随后将 seed 3407 放入 GPU4-A100 队列，等待任一现有四卡 worker 完成，不停止其他任务、不启用重叠八卡 worker。三种子使用同一冻结包及控制器，均每端 24 epochs：

| Seed | Task ID | 本轮观察状态 |
|---|---|---|
| 1337 | `f2ced9be51ef4e49a1a05dd8f4ecc3d3` | in_progress，车端已观察到 390/8040 |
| 2027 | `040a01e2ab9d4ee59f55e0e52581228f` | in_progress，初始化 |
| 3407 | `d707e30734f14c489a7609d2d5e01125` | queued，等待四卡 worker |

排队/初始化不是实际优化器训练完成。三种子最终权重、独立校准、身份训练及论文实验仍待完成；不得把本表当作三种子结果表。

## 续轮：新权重到原始候选导出的衔接

在 runtime 隔离工作树修改 `spd_export_dispatch.py` 与 `run_spd_raw_cache.py`：保留旧 seed1337/37组资产身份，新增三个已登记嵌套训练 task 的独立 collection 身份，绑定新 package/controller 哈希及实际 seed。导出命令显式传递 `--training-cohort nested-detector-fit`，子进程将其交给训练回执验证器重新计算并核对 30/7/9 分组。错误 task/seed、旧包改名、混用控制器/collection 类型被拒绝。

候选导出策略保持全部原始 query、下游 `raw score >= 0.05 / all-class top64`，不改变冻结 ImageNet appearance、GT 隔离、ROI/NMS 或调度策略。推断本身沿用固定随机种子 1337；训练种子由权重和 training_binding 单独记录，不以推断种子冒充训练种子。

导出调度、训练绑定、收集器回归 **57 passed**，新增 CLI 参数 `--help` 冒烟通过。未启动任何新权重候选导出，未修改运行训练的冻结源码；仍须训练真正 completed、最终权重收集和 tensor 验收后冻结新的导出代码包再运行。本轮最后观察到 1337 为 920/8040、2027 为 250/8040，3407 queued；迭代仅表示车端当前进度。
