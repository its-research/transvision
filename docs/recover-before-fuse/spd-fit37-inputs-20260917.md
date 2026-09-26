# SPD fit-only 输入准备回执

已完成 ClearML 训练包字节核验、37 个 fit 序列独立输入视图及首次转换。转换输出发现非有限速度标签，尚未放行训练；完整输入包发布仍待完成。

## ClearML 原始包核验

- 包任务：`9a7e7a9213954b57a35403f777e74561`。
- 训练包 SHA-256：`c793bc74ec35140c894fd7f3d14fd2cce231f0b919eee3fdbeca7515b3d1e0c1`，压缩载荷 4,070,034,766 bytes。
- 流式读取并与本地现存输入和转换目录逐文件比较，106,567 个文件、4,568,746,778 bytes 全部一致。没有额外保存第二份压缩包，也未用未核验的桌面文件替代 ClearML 来源。
- 仅保证包内全部文件匹配，不宣称本地目录的额外文件已审计。后续投影只允许已登记清单内的文件。
- 最初任务 `889cd16e89674361bd0fc93d013a9963` 完成流式校验和附件上传，但列表附件被 ClearML 编码为 pickle，JSON 回读失败。该失败保留。
- 修复任务 `46eb1d1123314af3b3fd828d5bb8d9a7` 以禁止类实例化的 primitive-only 读取器核验原附件及逻辑哈希，另发布 JSON `entries` 清单并回读成功；没有重复传输整包。
- 文件清单逻辑 SHA-256：`e0de326b13d068128f21aa40483a002db8a9aa813461630957d263377e6531ac`。

## 官方 split 核对

原 manifest 引用的 split 哈希为 `4453e56e371b9787f9847845b43ed81e2fcfd18eb6a7f49492ca152c4df054d3`。固定 CoopTrack 源码包中 `data/split_datas/cooperative-split-data-spd.json` 的实际哈希为 `0a59332221f3618639f34fb70678d03c5221ab80fd69f2cf41acffca0d8859fd`，因此首次准备被正确拒绝，未创建输出。

随后直接核对该固定官方源码包：46 个 train 序列与冻结划分的全集相同；车端 8,504 个、路端 7,834 个 train frame ID 与本地 metadata 集合完全一致，无多余或缺失。新视图绑定此次实际核验的官方 split 字节，保留旧引用哈希，不猜测哈希差异仅由排版引起。只读取 split 元数据，没有读取官方 val/test 的标签或图像。

## 已生成的独立视图

- 目录：`10.100.35.112:/home/lbin/Desktop/rbf-spd-development-fit-20260917`。
- 37 个 fit、9 个 holdout，沿用 `c6622188bea749dbbabcf3b968bc59d2` 的预先冻结划分；官方 train/val 不变。
- 车端 6,611 帧，路端 6,163 帧；复制载荷 3,288,291,627 bytes。
- 只复制 fit 帧引用的图像、标注和标定，每份副本重新核验 SHA-256。原标签字节不修改；原输入、holdout 文件和旧模型均保留。
- `input-manifest.json` SHA-256：`65815abf7ec0d8aadccc1bf8d1fc095cca26ae09292392ddcebaf838743df8da`。
- 清单和准备源码已发布到 ClearML `181db9d4349b464ca24b87c205eab2ee`，清单回读核验成功。完整视图载荷尚未上传；不得称为远端训练输入已就绪。

## 软件检查及下一步

三个直接相关测试文件合计 **14 passed**，覆盖流式分块、原始字节、危险路径/链接拒绝、旧附件禁止类实例化、划分覆盖及权重来源绑定。

入口位于 `work_dirs/rbf-pointpillar-runtime-20260917/tools/event_track_v2x/`：`verify_spd_training_package_stream.py`、`confirm_spd_stream_inventory.py`、`prepare_spd_development_fit.py`。部署的准备版本为 `prepare_spd_development_fit_v2.py`；旧失败版本保留。

第一次匿名 HTTP 请求收到 401，改为标准 ClearML SDK 认证流读取后成功，未改变权限或输出凭据。复制前验证可保留 3 GiB 空间；转换和打包前必须重新检查实际余量。

下一步：以固定官方转换器生成 fit-only 转换数据，核验无 holdout 帧，再将完整输入版本发布 ClearML，接入四 A100 隔离检测器训练。现有 A100 控制器只接受 full-train/46 序列，必须显式扩展独立 cohort，不能把 37 序列伪装成 full-train。

## 首次转换与数值准入检查

固定镜像缺少 `.git`，原 Git 检查拒绝运行且未创建输出。隔离入口新增固定归档核验方式，保留原 Git 检查路径；源码归档 SHA-256 为 `abe990d81039b71afc6191ad886978c971ec6607377f1792677ba2f9893d9bb5`。解压前拒绝路径穿越、重复成员和链接，转换前后比对源码树全部文件字节。源码及输入均以只读方式挂载，运行禁用网络和 Python 字节码写入。

- 入口：`work_dirs/rbf-pointpillar-runtime-20260917/tools/event_track_v2x/convert_cooptrack_fit_inputs.py`；远端独立部署名为 `convert_cooptrack_fit_inputs_archive_v1.py`，未覆盖冻结历史入口。
- 转换与输入准备相关回归：24 passed。
- 输出：`10.100.35.112:/home/lbin/Desktop/rbf-spd-development-conversion-20260917/converted`。
- 车端 6,611 帧、75,277 条标注；路端 6,163 帧、96,848 条标注。验证未删帧、未插补标注；两个内部 val pickle 均为 0 帧。
- 转换清单内容哈希：`f2afc9688ae8e854d9375037c8b27956afab060dcfa68d09a487c43ad233cced`。转换退出码 0 后，独立重读清单内 30 个文件核验大小及 SHA-256，全部一致。
- **未通过数值准入**：官方转换器 `spd_to_uniad.py:1092` 的速度差分出现零时间间隔。独立检查发现车端 `gt_velocity` 有 12 个非有限数值、路端有 292 个；这是数值元素数量，不是目标数量。其他已检查的顶层数值 ndarray 字段未发现非有限值，不代表递归全部数据验证完成。

不得将转换退出码 0 当作训练输入合格。本次输出保留为诊断证据，尚未发布为合格 ClearML 训练载荷，也未启动 A100 训练。下一步核对重复时刻与速度监督消费路径，明确缺失速度的有效性掩码处理后生成新版本；不得直接将非有限速度改成零并宣称真实静止。

## 真实速度标签的损失保护核验

后续检查发现既有 `run_cooptrack_detector.py` 已实现 `install_unavailable_velocity_guard`，在 `criterion.loss_bboxes` 入口对非有限派生速度屏蔽两维损失权重，保留几何框监督；原始标签不改写。因此暂不重做转换，也不把非有限值直接清零成有效监督。

对本次两端所有真实速度数组执行独立梯度审计，车端 75,277 个框中 6 个速度无效，路端 96,848 个框中 146 个速度无效；全部是 infinity，没有 NaN。已检查几何框数组全部有限。实际速度目标通过既有保护后，测试损失及全部梯度有限，无效速度梯度为零，几何测试目标梯度保留，输入目标不变。

- 审计入口：`tools/event_track_v2x/audit_spd_velocity_targets.py`（隔离 runtime worktree）。
- 远端回执：`/home/lbin/Desktop/rbf-spd-development-conversion-20260917/velocity-mask-audit.json`。
- 被验证训练入口 SHA-256：`02b5af6b1562115793f8071fadaf49be570fe1af6cd304a36a3622de82dc098a`。
- 审计范围仅为真实速度数组通过保护函数及合成有限几何测试目标的梯度，不是完整模型前向、数据加载器或 A100 训练验收。正式运行必须绑定同一保护入口并验证实际损失有限。

官方数据集会把 x 速度为 NaN 的目标设零，原始 dense head 某些路径会按整行有限性丢掉框回归；不能仅凭这些上游行为放行。本批没有 NaN，不触发前者；后续需在实际检测器路径验证保护确实覆盖损失调用。ClearML 完整载荷发布、37 序列 cohort 接入和 GPU 冒烟仍待完成。

## 独立开发训练包

新增 `eventtrack_a100_development_package_v1`，与旧 `eventtrack_a100_migration_package_v1` 的 46 序列 full-train 包区分。控制器检查 37/9 互斥划分及冻结哈希排序规则，解包后绑定输入清单、转换清单、fit/holdout ID。四卡批量候选为每卡 2、4、8，不使用超过 37 条流的每卡 10；旧 full-train 批量规则保留。

- 打包目录：`10.100.35.112:/home/lbin/Desktop/rbf-spd-development-package-20260917`。
- 包清单 SHA-256：`9c3f3b8b0c59db69da6c4280b506bf575b6ca3b43559e3a9c28b2828ca7ec6ac`。
- 训练数据归档：3,235,557,989 bytes，SHA-256 `2286b2d17866ebfba7b563fad7cce158f0c66f6c38ca40f537b8264f57393477`。
- 源码归档：545,130 bytes，SHA-256 `33250c65975d47e1aaf3f86e6c8553c4384fd6b38b8fb05f572a4192870bafbc`，包含已验证的速度保护训练入口。
- runtime 和 ImageNet R50 权重通过明确的 `artifact_task_id` 引用原 ClearML 包，消费时仍核验字节及 SHA-256，不重复上传。
- 三个相关测试文件：36 passed、2 skipped（本地缺少 Torch）；真实目标梯度审计另在远端 Torch 环境通过。包生成退出码 0，ClearML 上传已启动，尚待回读确认。

入口 `package_spd_development_a100.py` 只打包清单允许的文件；不包含 holdout 图像与标签。A100 正式训练尚未启动，上传成功与 GPU 训练成功须分别验收。

### ClearML 发布与训练调度

包任务 `5da9693dfce54d85b57ab0ca663dbb41` 已完成上传。独立通过 ClearML SDK 流式回读训练归档 3,235,557,989 bytes、源码归档 545,130 bytes、包清单 1,876 bytes，三个 SHA-256 均与上述本地固定值相同。没有下载第二份大包到管理机。

只读调度检查确认 `GPU4-A100` 队列为空，`10.100.34.18-A100:gpu4,5,6,7` worker 空闲；GPU0–3 的其他任务保持不变。随后提交任务 `214693f23e844eacaa4d85360cf898b0`，名称 `D2 A100 development-fit R50 seed-1337 batch-auto attempt-1`。控制器 SHA-256 为 `10c0ca1911b770b718fb0753d0266f6ada7faceecb0182b0a0f9a3b8909b6e38`，绑定上述包清单。

控制器先执行车端及路端批量/优化器探测，通过后分别训练 24 epochs；本任务是共享上游检测器开发训练，不等于下游身份模型三种子训练。入队成功不是模型训练通过，需继续核验 worker 绑定、实际探测结果、有限损失、检查点与两端完成回执。

提交后独立回查：任务状态 `in_progress`，实际执行 worker 为 `10.100.34.18-A100:gpu4,5,6,7`，尚无探测或训练产物。只能确认 worker 已接单，尚不能确认优化器已启动。

### attempt-1 文件服务认证失败与 attempt-2

后续确认 attempt-1 已 `failed`，未产生训练产物。包清单下载 HTTP 401，`get_local_copy()` 返回 None，控制器构造 Path 时失败。实际资产（包含复用旧包）均注册在 `http://10.100.35.118:8081`，worker 默认 `api.files_server` 为 `http://10.100.34.118:8081`；不能把该错误解释为模型或数据训练失败。

修复仅为任务容器显式配置 `CLEARML_FILES_HOST=http://10.100.35.118:8081`，并在下载返回空路径时给出明确错误。不更换凭据、不绕过认证、不改变训练数据或模型参数。相关控制器回归 22 passed。

attempt-2 任务 `3505e45aa07643b781361150e66f021f` 已提交到 `GPU4-A100`，仍绑定包 `5da9693dfce54d85b57ab0ca663dbb41` 及相同包清单哈希。新控制器 SHA-256 为 `7a4734891009c0ccb3bf458e646d4d8cd8382865a252f6bc3904b5e1c93fdc3c`。attempt-1 失败记录保留，尚待核验 attempt-2 完成下载及实际优化器启动。

2026-09-17 13:19:58 UTC 的 worker 活动回查显示 attempt-2 仍由 GPU4–7 执行，状态 `in_progress`；日志已确认 `api.files_server=http://10.100.35.118:8081`、环境初始化成功及控制器开始执行。尚无 `EVENTTRACK_MATERIALIZED`、优化器启动或探测产物，不能宣称资产已在 worker 完整落地。经管理机进一步只读查询 worker 容器时遇到 SSH host key verification failed，未绕过主机身份检查；继续以现有 ClearML 任务状态和日志观测，不因此重启任务。

后续同一任务日志新增 `EVENTTRACK_MATERIALIZED runtime.tar.gz`，说明固定运行环境已按控制器流程完成下载、字节哈希核验、归档路径检查及解包。此前静默期间监控采样持续推进，不能判为停机。此时仍无训练探测产物，训练数据、源码加载及优化器启动须分别继续核验。

### attempt-2 缺失 libGL 与 attempt-3

attempt-2 随后确认 `failed`：runtime、train-inputs、source 三个归档全部报告 materialized，文件服务认证及包身份验证已通过；但四卡环境预检导入 OpenCV/MMCV 时出现 `ImportError: libGL.so.1: cannot open shared object file`。未进入优化器探测，无训练结果。

任务容器追加 `CLEARML_APT_INSTALL=libgl1`，通过现有 ClearML agent 的容器初始化安装依赖，不修改宿主环境、数据包、模型或超参数。控制器相关回归仍为 22 passed。

attempt-3：`9f4eb07e4fab4f63a9f30c470f91255b`，已提交 `GPU4-A100`；控制器与包哈希均保持 attempt-2 的固定值，差异是上述容器依赖配置。前两次失败任务保留。尚待确认 libgl1 安装、四卡库导入和优化器启动。

### attempt-3 首档四卡探测已运行

同一任务随后确认 libgl1 安装、三类归档 materialize 和四卡预检通过，实际运行库版本为 Torch 1.9.1+cu111、MMCV 1.4.0、MMDetection3D 0.17.1。四个 rank 分别完成 6,611 个车端标注与 18 个空目标帧 pipeline 预检。

车端每卡 batch=2 的 32 步探测已结束并上传 `probe-b2-vehicle-side-profile`。第 16 步启动证据：loss=23.844093322753906，骨干最大绝对参数更新 8.614826947450638e-05，优化器状态 369 项，world_size=4，有效 batch=8。四个 rank 均报告 32 步结束，最大 reserved 显存 3,661,627,392 bytes。日志中的速度掩码计数为 0，不能据此声称本探测已覆盖非有限速度目标；该类目标的独立梯度审计仍为前述单独证据。

这里只证明首档真实 GPU 探测成功，不是 24 epochs 训练完成或论文效果。后续 batch=4/8、路端探测、批量选择及两端正式训练仍须继续验收，探测权重不会复用于正式训练。

车端三档探测随后全部成功，均为四个 rank、32 步、退出码 0，ClearML profile 字节回读与 SHA-256 一致：

| 每卡 batch | 最大 reserved 显存 bytes | profile SHA-256 |
|---|---:|---|
| 2 | 3661627392 | `6d5a274fe4d84d51fe781895ef23015086bf98a90851a7aad47323eb945f1333` |
| 4 | 6884950016 | `b0236804fa8cf5a2c127730bf87c3876bc32c6a20064c8c5b01eca736733ac65` |
| 8 | 13270777856 | `8d9e81cacfb0509e8b3ba857080e388042aad2885775064e930ee4ffe0101200` |

batch=8 第 16 步 loss=26.67584800720215，骨干最大绝对更新 9.590142872184515e-05，有效 batch=32。任务继续执行路端探测；尚不能将车端通过视为两端批量选择完成。

### 两端探测完成，车端正式训练已启动

路端 batch=8 四卡 32 步探测成功，profile SHA-256 `71f7a7f1484b9d1d112919d2da77b24700ae40689362e4a8a9543df496bd5b5c`，最大 reserved 显存 13,423,869,952 bytes。批量冻结回执 SHA-256 `0fe518183e55d05c6fe6ca66b3fcfb59104869ec22f307bcc32de316fbf34b90`，每卡 8、四卡有效 batch=32、37 条序列流，均已回读核验。

车端正式训练已从初始权重开始：`batch_probe_only=false`，24 epochs，4,968 次 micro-iterations，fit ID 与预先冻结的 37 序列一致。第 16 步 loss=26.441986083984375，骨干最大绝对参数更新 0.00011351332068443298；启动回执、运行配置及第 16 步检查点已上传。

- optimizer-startup JSON SHA-256：`4d1189b84ae89f52e10d9f6ef465620e2c44b50e721b75b0a42c8ed00a349391`。
- launch-receipt JSON SHA-256：`7900809e5d32bccd637580d1b4dffa210078bb8a1eb83f76c9f3ce047964eb9c`。
- 上述两个 JSON 已实际回读核验；启动检查点尚未在管理端独立回读，不以附件存在代替其字节核验。
- 最新观察到第 20/4968 步，loss=18.0660，任务仍 `in_progress`。日志此时给出的车端剩余时间约 4 小时，只是早期估计；之后还需路端正式训练。

此阶段仅确认正式检测器训练真实启动，未完成训练、开发集评价或任何论文性能结论。

### 启动检查点字节核验及真实无效速度分支

随后通过 ClearML SDK 流式回读 `vehicle-side-startup-iter-16.pth` 全部 400,419,244 bytes，SHA-256 为 `4b634b3d43d8c2540a48350fd4958b4dc0b00637a30a8290377276e7154d58ea`，与登记值一致；没有在管理机保存第二份权重。此项核验是字节完整性，不等于对整个 checkpoint 内张量完成有限性扫描。

正式训练推进到第 40/4968 步，loss=16.4896、grad_norm=26.8855。第 30 至 40 步之间出现两条 `EVENTTRACK_VELOCITY_TARGET_MASK undefined derived velocity retained with zero loss weight`，随后训练继续输出有限损失。这证明完整模型训练路径实际触发了无效速度保护；两条日志不代表两个目标，也不能替代最终累计统计。任务仍在运行，训练与后续评价均未完成。

### 阶段检查点的实际字节及模型张量检查

同一任务最新观察推进到 670/4968 步，仍由 GPU4–7 worker 执行。回读当时 `vehicle-side-latest-checkpoint` 指向的 `iter_414.pth`：400,419,692 bytes，SHA-256 `92cadffce799815d19ca3e86f71829eee3037d70751f093228dc1217b6f161f9`，与登记值一致。

使用 `torch.load(..., weights_only=True, map_location='cpu')` 在内存读取，无 GPU、无重复磁盘副本。包含 meta/optimizer/state_dict；模型 state_dict 651 项、33,631,011 个张量元素，全部模型张量有限。meta.iter=413 与文件命名 iter_414 的计数差异如实保留。确认存在 optimizer 字段，但尚未检查全部优化器张量、随机数/采样器恢复及重启等价性，因此不能把本检查称作完整可复现续训验收。

本检查针对阶段权重，不是最终模型验收，也未做开发集性能评价。latest 附件将随训练变化，复核本次证据应使用上述文件名及哈希，而非仅凭 latest 标签。
# 三种子后续控制器准备

ClearML 发布补充：源码冻结任务 `b7426afc40c64d25a475af6c76805816` 已完成。三种子控制器、提交器、收集器及两份直接测试源码共 5 个资产逐项回读，字节数和 SHA-256 与本地源文件一致；控制器仍为下述 `86ca2d7…`。该任务是源码发布，不是训练任务，也不是自包含运行环境包；收集器的 `spd_export_training_binding.py` 等运行依赖仍须随实际部署绑定。未启动 seed-2027/3407，未改动当前 seed-1337 作业。此前“尚未发布”的表述保留为当时阶段记录，以本条发布证据更新当前状态。

后续绑定补充：新三种子控制器 SHA-256 为 `86ca2d7dd2aab807579b9c6ac5938d707fb92e9f4fa6f6710e77681674039a9a`，已加入本地收集器的显式准入列表，保留旧控制器 `7a4734891009c0ccb3bf458e646d4d8cd8382865a252f6bc3904b5e1c93fdc3c` 仅允许 seed-1337。实际内联源码哈希、任务参数哈希、训练包哈希须一致；任务种子必须属于对应版本许可值，且双端完成绑定的种子必须与任务参数相同。联合回归 47 passed，包括新源码哈希锁定及拒绝旧控制器冒称其他种子。此为本地冻结绑定，尚未发布新控制器 ClearML 资产或提交新种子训练。

隔离工作树的 `submit_cooptrack_a100.py` 与 `run_cooptrack_a100.py` 已将此前固定的 1337 改为显式 `--seed {1337,2027,3407}`；ClearML General 参数、所有批量探测与双端正式训练命令、批量选择和完成回执使用同一个种子。提交任务名、标签和本地回执文件包含种子，防止不同种子覆盖同一回执。控制器回归 26 passed。

本项仅为后续作业准备，未提交新训练。当前任务 `9f4eb07e4fab4f63a9f30c470f91255b` 使用既有内联冻结源码，仍是 seed-1337。现有最终权重收集器继续只接纳既有冻结控制器哈希；新控制器必须另行冻结、登记哈希并扩展相应收集绑定后才能用其结果进入导出，不得因参数化修改自动放宽来源准入。
