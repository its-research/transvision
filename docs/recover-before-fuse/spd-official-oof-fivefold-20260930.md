# SPD 官方 train 固定五折检测器输入与转换

本轮使用仓库 `development_split.py` 的固定五折协议：46 个官方 train 序列，salt 为 `eventtrack-v2x-spd-development-5fold-v1`。held-out 大小为 10/9/9/9/9，fit 大小为 36/37/37/37/37。官方 val、test、test_A 保持隔离。

所有完整回执、命令、任务 ID 和哈希登记在 `test/recover-before-fuse/receipts/20260928-execution-ledger.json`；`test` 指向 `/Volumes/Data/test`。本文件记录截至本轮的依赖链，不将后续未验收步骤视作完成。

资格边界：上述数据身份绑定的是现有 ClearML 资产。`/Users/lbin/thesis/tracking/experiment-runbook.md` 的 D1 要求与官方发布对象逐字节核验；本轮回执没有证明这一要求已满足。因此输入、转换和训练包验收是工程字节及隔离验收，正式 D1/D6 和论文结果资格仍按原门禁保持关闭。

## 已验收的输入与转换

- ClearML 原始包任务：`9a7e7a9213954b57a35403f777e74561`；训练归档 SHA-256 为 `c793bc74ec35140c894fd7f3d14fd2cce231f0b919eee3fdbeca7515b3d1e0c1`。
- 独立原始清单任务：`46eb1d1123314af3b3fd828d5bb8d9a7`；清单 SHA-256 为 `e04f07ad6e4eeb7a505ca33eed31bafd27d4157e40cc34f73e9e961a07136ce0`。
- 官方 split SHA-256 为 `0a59332221f3618639f34fb70678d03c5221ab80fd69f2cf41acffca0d8859fd`；固定五折 manifest SHA-256 为 `1c9d23f291be09e686a2bdcc60ef19ea58a5c2894978cc5f6a6b2a81f43c2bd6`。
- 本机与 Linux 主机 112 的原始 106,536 个输入文件均绑定 ClearML 清单。五折本机 fit 视图的 426,143 个文件已独立回读，每个官方 train 序列恰好 held-out 一次。
- 五折远端转换使用历史固定 Python 3.8 容器和 `abe990d81039b71afc6191ad886978c971ec6607377f1792677ba2f9893d9bb5` 源码归档。150 个转换文件独立回读通过，帧与原始标注 token 覆盖一致，无删帧、插值或标签改写。

转换验收回执：`receipts/spd-official-oof-fivefold-conversion-independent-readback-20260930.json`，SHA-256 为 `2d1669c075e4384793f7ea0287f84b5f06bc51ed3f61715ec07331945377f171`。

## 速度监督与训练包

五折真实转换目标含非有限派生速度，但均无 NaN。实际目标数组经历史安装的损失保护核验，几何梯度保留，无效速度梯度置零；原标签保留。该审计只覆盖损失保护，完整检测器训练仍待执行。

速度保护回执：`receipts/spd-official-oof-fivefold-velocity-guard-independent-readback-20260930.json`，SHA-256 为 `ba29b5616d6264b5c01e3952779e4ef28cda57d10e982ec56823551020925ea9`。审计 runner 字节与 ClearML `5da9693dfce54d85b57ab0ca663dbb41` 的固定源码包成员一致。

五折完整 fit-only 训练包均已独立 SHA-256 核对。包验收回执：`receipts/spd-official-oof-fivefold-local-package-acceptance-20260930.json`，SHA-256 为 `811b8e0f3a5044bc7a0d4deba712dbec7315530e0649b5e4695eea474ef1e75e`。源码固定在 `source-freezes/spd-official-oof-fivefold-package-20260930`。

## 尚未完成的依赖

第 0 折包上传任务为 `2eb8b26ba2af4dd69df27e19fa27d309`。上传中断后的恢复沿用同一任务 ID；是否完成以 ClearML 实时附件及独立字节回读为准。其他四折包尚未发布。

独立回读使用 `accept_spd_official_oof_fold_package_clearml.py`；训练派发使用 `submit_spd_official_oof_fold_gpu4_readback_v2.py`，要求每折存在绑定任务 ID、fold、归档与 manifest 字节哈希的 `clearml-independent-readback.json`。旧派发器源码保留，新源码固定在 `source-freezes/spd-official-oof-package-readback-dispatch-v2-20260930`。

包发布和独立回读之后，才按配置、fold 与 seed 去重派发四卡检测器。控制器保留固定五折成员关系和每侧 24 epoch；派发器依据物理 GPU 交集排除同机四卡/八卡 Worker 碰撞，实际设备写入训练回执。后续还需训练完成、检查点冻结与张量验收、held-out 推断及 pooled OOF 覆盖验收；这些步骤均未由输入转换或包自检证明完成。

完整 Stage2、等资源基线、正式独立评价及论文性能仍按原冻结计划推进。本轮没有提交或推送代码。

后续训练派发改用独立命名的 `submit_spd_official_oof_fold_gpu4_eta_readback_v3.py` 与 `run_cooptrack_official_oof_gpu4_eta_v2.py`。新源码冻结在 `source-freezes/spd-official-oof-training-eta-v2-20260930`；原冻结源码保留。阶段 ETA 依据连续迭代日志的时间差，启动、缺少近期进度及完整实验 ETA 标为未知。独立子进程日志冒烟通过，仅证明日志行为，不证明 GPU 训练完成。训练仍须先通过上传包独立字节回读及物理 GPU 碰撞门禁。

五折 held-out 图像/标定视图已在 `artifacts/spd-canonical-oof-heldout-image-pose-20260930` 独立回读通过：16,338 个主体帧恰好覆盖一次，73,856 个图像与原始标定文件字节匹配，标签、val/test 未读取或复制。五折均通过现有 DetectionCacheV2 输入合同。回执为 `receipts/spd-canonical-oof-heldout-image-pose-independent-readback-20260930.json`，SHA-256 为 `81eb06939e27491036bfcc2d6e684953a3828a2c55e4333d23b50a5172691504`；源码固定在 `source-freezes/spd-canonical-oof-heldout-image-pose-20260930`。这只完成推断输入隔离与字节验收，位姿变换、检测预测、OOF 校准及正式资格仍待验收。

上述 16,338 帧的原始 LiDAR→世界变换已单独生成并独立验收：vehicle 使用两级标定组合，infrastructure 使用原始虚拟 LiDAR 标定；齐次矩阵与逆变换全量核验，最大绝对差为 `4.547473508864641e-13`。回执 `receipts/spd-canonical-oof-heldout-raw-poses-independent-readback-20260930.json` SHA-256 为 `cdfe9e0ac37cdf00ef9db979830e962ec60ff6f9e0262e9c4cdae7871fdd0df9`；表保存在 `artifacts/spd-canonical-oof-heldout-raw-poses-20260930`。没有赋予独立 pose 到达时间，也没有生成检测预测或评价结果。

第 0 折包 ClearML 上传已完成，登记归档与 manifest 哈希匹配；完整独立回读仍在运行。原回读器因 JSON manifest 的 HTTP deflate 传输缺少 Content-Length 而严格拒绝，失败回执保存在 `receipts/spd-official-oof-fold0-initial-readback-range-failure-20260930.json`。后续回读使用单独命名的 `accept_spd_official_oof_fold_package_clearml_identity_v2.py`，明确请求 identity 编码并拒绝压缩响应；精确 Range、长度和内容哈希检查保留。新源码冻结在 `source-freezes/spd-oof-package-identity-readback-v2-20260930`。第 1 折包正在上传，训练均未派发。

第 0 折包完整独立字节验收已通过，回执 `artifacts/spd-official-oof-fivefold-20260930/fold-0-package/clearml-independent-readback.json` SHA-256 为 `138e6cb1eac68c8120df2f61fa154e9bbbbc0ec787334541be5f8573c0450c38`。本机回读吞吐较低，原未完成记录保留，核验迁移到已核验的 Linux 112 主机；`accept_spd_oof_package_remote112_v3.py` 从本机配置获取临时认证请求头，仅经 SSH stdin 传递且不持久化凭据。完整归档与 manifest 均使用 identity Range 独立 SHA 回读；源码冻结于 `source-freezes/spd-oof-package-remote-readback-v3-20260930`。训练派发仍需实时物理 GPU 碰撞检查，排队不代表训练验收。

第 0 折 seed1337 四卡检测器任务 `531c35277bd2487aa9b48e9f7400d229` 已通过独立包字节门禁后派发到 `GPU4-A100`，派发时唯一空闲无碰撞 Worker 为 `10.100.34.18-A100:gpu0,1,2,3`。计划仍为每侧 24 epoch，使用带阶段 ETA 的冻结控制器；排队和运行均不等于训练验收，检查点冻结及 held-out 推断待依赖完成。

上述首个训练任务 `531c35277bd2487aa9b48e9f7400d229` 已在启动阶段失败，零迭代、无训练产物：Worker 注册文件服务为 `10.100.34.118`，直接获取登记在 `10.100.35.118` 的 manifest 返回 HTTP 401。脱敏失败回执为 `receipts/spd-oof-fold0-seed1337-initial-training-failure-20260930.json`。新 v4 下载控制器按既有成功验收器的两个注册地址映射下载，完整字节计数和 SHA 校验保留；源码冻结在 `source-freezes/spd-official-oof-filehost-v4-20260930`，原任务和所有旧源码不改。关联修复通过显式 `--retry-of` 绑定零迭代无产物失败任务，不改变实验拟合配置，仍待 GPU 环境执行验证。

文件路由修复的关联尝试任务为 `c92230b20c8a4002b961369e88cc6ded`，显式绑定原失败任务 `531c35277bd2487aa9b48e9f7400d229`，已确认入队后进入 `in_progress`。此状态只证明派发和 Worker 生命周期，尚不证明优化器启动或任何 epoch 完成。

文件路由修复任务 `c92230b20c8a4002b961369e88cc6ded` 已完成所有包下载与解包，但在 MMCV/OpenCV 库导入时因缺少 `libGL.so.1` 失败，仍为零迭代无产物；脱敏失败回执在 `receipts/spd-oof-fold0-filehost-v4-runtime-failure-20260930.json`。依据历史 `spd-fit37-inputs-20260917.md` 的同类成功修复，新 v5 派发器仅添加任务容器 `CLEARML_APT_INSTALL=libgl1`，控制器字节完全不变，旧失败保留；源码冻结于 `source-freezes/spd-official-oof-libgl1-v5-20260930`。需等新任务实际导入与四卡验收，不由历史记录代替当前执行。

容器依赖修复任务 `517e10afb294404a96fc6c19f5ce88f6` 已派发 `GPU4-A100`，关联 `c92230b20c8a4002b961369e88cc6ded`。控制器哈希保持 `ab5dc3b9144b113664bd55d8b3bb37c780cf0f53791bf8f9e896dfec41ca2bb3`，变化仅为任务容器安装 `libgl1`；当前入队不视为 GPU 就绪或训练完成。

四卡 batch=2 显存探针已独立回读通过：任务 `517e10afb294404a96fc6c19f5ce88f6` 的 vehicle-side 四个 rank 均执行 32 次优化器迭代，实际设备为 A100-PCIE-40GB。回执 `artifacts/spd-oof-fold0-vehicle-b2-probe-readback-20260930/acceptance-receipt.json` SHA-256 为 `526bb89cda465bb04e8d2e5b02489156aa974a20eebd70eeb54c2a698d40892f`。这证明当前库导入及四卡 batch=2 探针执行，尚不证明批量选择、每侧 24 epoch 正式拟合或论文实验完成。

2026-09-30 16:14 UTC 有界实时核验：第 0 折 seed1337 已进入 vehicle-side 正式拟合，最新任务日志为 `Iter [60/5040]`，报告 loss 有限，阶段剩余 ETA 约 3 小时 38 分钟；基础设施侧及发布/验收耗时未计入，完整实验 ETA 仍未知。startup 产物存在但尚未独立验收，完整拟合未完成。第 1 折上传与 SPD CPU seed1337 仍运行；CPU 无任务级进度，ETA 未知。证据 `receipts/bounded-live-status-20260930T161428Z.json`。

第 0 折批量选择与正式优化器启动已独立字节验收，回执 `artifacts/spd-oof-fold0-batch-startup-readback-20260930/acceptance-receipt.json` SHA-256 为 `98a639d0b61a08cbd5addde0409ce309b63a3d9cacbd345d70bf62eb497211a0`。车端四卡 batch=2/4/8 及基础设施端 batch=8 的 32 次迭代探针符合 80% 显存门槛，最终每卡 8；正式拟合第 16 步骨干更新与优化器状态成立，fit36/held-out 隔离匹配，完整拟合仍未完成。

第 1 折完整 ClearML 包独立回读已通过，回执 SHA-256 `d3c85f98993649eb7f7cbc32f1a8a46cf11357884bf74dde30c3ab5b8aaf83a4`；seed1337 训练任务 `12b06f9674494dc381b4d6566d848b20` 已经入队确认，派发时唯一空闲无碰撞 Worker 为 A100 GPU4–7。第 2 折上传任务 `ff393d012aa046a4b96c060c76d105b9` 已开始运行，初始阶段不足以计算可靠 ETA，标未知；不将上传中或入队视为实验完成。

后续两侧检查点独立字节冻结入口已准备：`freeze_spd_official_oof_detector_bytes.py`，源码固定在 `source-freezes/spd-official-oof-detector-byte-freezer-20260930`。该入口要求 completed、绑定已独立回读包、两侧启动/配置/完成回执及最终迭代 checkpoint 名称一致，并完整重算全部产物 SHA；四卡 batch 探针与 fit/held-out 隔离仍检查。运行中第 0 折实时门禁测试被拒绝且未创建输出，回执 `receipts/spd-official-oof-detector-byte-freezer-live-guard-20260930.json`。尚未在完成任务上验收，字节冻结也不代替张量/前向或 held-out 推断验收。

已从历史已完成源码任务 `8641d7b497e14234ba2d8aeb505f0d8d` 独立回读七个真实导出文件，原字节保存在 `artifacts/spd-canonical-oof-export-source-recovery-20260930`，回执 SHA-256 `9dae02d6de1fab01faf463855b8e90589a1489c729b97de4b7989955251d9ca7`。当前导出器绑定旧 fit37/full46 输入及 ImageNet appearance，并硬编码单卡 A100；主工作树缓存构建器仍只接受旧解码标识。后续需独立命名的 canonical OOF 适配：绑定本轮 fold-specific 完成权重与 held-out 图像/位姿、补齐 label-free infos、允许实际兼容 GPU、保持覆盖前全 query 解码、128D appearance 与独立回读。不得将历史缓存或旧输入身份改名视为 OOF 完成。

新增独立模块 `spd_canonical_oof_export_binding.py`，源码及依赖固定于 `source-freezes/spd-canonical-oof-export-binding-20260930`。绑定当前固定五折 fit/held-out、已完成训练的独立字节回执、配置和最终 checkpoint；parsed launch/startup/completion 必须匹配已冻结原始字节，不接受旧 fit37 身份。五折软件正例和 12 个泄漏/篡改拒绝检查通过，回执 `receipts/spd-canonical-oof-export-binding-software-check-20260930.json`。测试完成回执及 checkpoint 是明确的软件 fixture，无真实训练完成或推断验收；实际 label-free infos、导出器接通、张量/前向与缓存完整读回仍待推进。

用户追加要求 V100 同步实验后，实时确认 V100 两组四卡 Worker 空闲且无八卡占卡交集；第 2 折 seed1337 后续训练指定 `GPU4-V100`。依赖续接进程 PID55445 / session87730 已运行，等待原上传任务 `ff393d012aa046a4b96c060c76d105b9`，完成后独立全量字节回读，再重新核验物理占卡并按配置/种子去重派发。源码 `source-freezes/spd-oof-fold2-v100-continuation-20260930`，启动回执 `receipts/spd-oof-fold2-v100-continuation-started-20260930.json`。当前尚未创建 V100 训练任务，不将本机等待进程称为 GPU 实验完成；保留 A100 两折运行。

从已登记 ClearML label-free 输入包 `9199a9d7af164056920dd0fe5d0c0247` 的本机已核验 Linux 缓存恢复两份 infos，input manifest SHA 与登记包匹配，两份 pickle SHA 为 `fade9143bf19cf268ab3b7bd9bf94578987aea33443475b9618d4f84ad3d2cc0` / `1a2b616cedbdd7865289f975b58f75a02548f227f688c26fd313284209801313`。v2 审计对五份原始位姿表分别核对验收哈希，对全部 16,338 主体帧核验字段白名单、帧/时间和位姿组合；无 GT/未来引用，未归一化原始位姿。回执 `receipts/spd-canonical-oof-historical-label-free-infos-audit-v2-20260930.json` SHA-256 `bf964fcc7fbdbf4b0c482269fda0cc130437a588b51df20f80a49b7b3d6269cb`。首版审计回执保留；v2 补充位姿表逐项字节绑定。旧输入引用的历史训练权重不继承；新各折 infos 尚未裁出，本项也不重复宣称图像 SHA、实际前向或完整缓存验收。

第 2 折完整包独立回读通过，SHA-256 `411c20e008e4e2a30b36acfe8039b746b9e6f4890aa07fed1370f69dc6f8bde3`。seed1337 任务 `3ca98c3f5cce4795bcc2bc829ecfeacb` 已由 V100 GPU0–3 接单，2026-09-30 17:11 UTC 为 in_progress、零迭代、无产物；ETA 未知。A100 第 0、1 折继续运行，无重复训练。派发与实时状态回执分别为 `receipts/spd-oof-fold2-v100-dependency-dispatch-20260930.json` 和 `receipts/spd-oof-fold2-v100-first-live-status-20260930.json`。

五折 held-out 推断 infos 已独立读回全部 16,338 帧并核对图像 SHA、字段及数组值/类型。固定 Linux Python3.8.20 / NumPy1.19.5 容器实际读取十份可移植 pickle 通过，bundle SHA `b04b70a9c08dfbfbbe709521c58493fd6f2680bccc901047252d0f0abeebed85`；回执 `receipts/spd-canonical-oof-infos-legacy-runtime-readback-20261001.json`。不代替模型 pipeline、张量/前向或完整 OOF 缓存验收。第 3 折上传任务 `1ca5052846304d46a35822c0398ce0e9` 运行中，session11283 等待同一上传完成后接续 V100。

canonical OOF 导出候选已接通逐折训练字节冻结、真实 held-out infos 和严格 payload 门禁；源码独立命名 `run_spd_canonical_oof_raw_cache.py`，历史源码保留。五折实际输入门禁覆盖 16,338 帧，错误输入 SHA、fit/held-out 混入、test 读取和历史缓存改称 OOF 均拒绝；Python3.8 语法检查通过。候选冻结在 `source-freezes/spd-canonical-oof-raw-export-candidate-20261001`，软件回执 SHA `be89ec73c097bf780e3e5975a08a0fe83633bd76c02f77139995bf64651a4e84`。保留全 query、无预选 ROI/NMS/topk、固定 128D appearance、原始时间及位姿，并记录实际 CUDA 设备和基于帧进度的阶段 ETA。尚未上传或在已完成检查点上运行真实模型；raw cache 不含已校准协方差，不视为 DetectionCacheV2 或完整论文验收。

V100 第 2 折 v5 任务 `3ca98c3f5cce4795bcc2bc829ecfeacb` 已在零迭代、无产物时失败：容器 DNS 无法解析 Ubuntu 源，libgl1 未安装，随后 OpenCV 导入缺 libGL。失败完整脱敏记录 `receipts/spd-oof-fold2-v100-libgl1-v5-startup-failure-20261001.json` 保留。第 3 折本机 v5 派发等待 PID74798 已终止，上传不停止。已准备仅含九份 GL 依赖的 1.14MB 离线候选，不包含 glibc/loader，SHA `5ac68c58a292e435a0ee55c98f0bd2720a9b088343afc813c262bdc552cc0e10`；全部库独立字节核对。相同固定目标容器与原始 runtime 的无网络导入对照探针 session29411 仍待结果，暂不宣布运行时验收或提交 GPU 重试。

离线 GL 对照验收通过：固定目标 Docker `e5b249d993f9...`、原始 runtime SHA `b6b39c66...`、network=none，基线缺 libGL，新增库后 OpenCV4.5.5/Torch1.9.1+cu111/MMCV1.4.0/mmdet3D0.17.1 导入通过，无 GPU 前向验收。依赖 ClearML 任务 `1e0730c280e846bebd92cc4de49893e3` 已完成，三个产物独立读回回执 SHA `1b44c626d3866cee41d870c605099be23d02727cd5760189b581f80204365b52`。v6 控制器 SHA `d33ccec0c08ab978a72dc5f6730491b45a0791c0a42c5fdc38e21183b22a9053`，仅任务内库路径及固定字节门禁新增，cohort/批量/训练命令/phase ETA/materialization 函数 AST 均不变，源码固定 `source-freezes/spd-official-oof-offline-gl-v6-20261001`。第 2 折修复任务 `d0d2b630acdb47adbdb297173d687d9f` 已派发 GPU4-V100，明确 supersedes 原失败 ID，未视为训练通过。第 3 折续接改用 v6，session99710/PID86479 等待原上传，旧本机续接终止记录保留。新 v6 检查点字节冻结入口 `freeze_spd_official_oof_detector_bytes_offline_gl_v6.py` 绑定该控制器；运行中的 v4/v5 A100 任务仍使用原 freezer。

最终检查点前向候选 `verify_spd_canonical_oof_checkpoint_forward.py` 已准备，要求绑定 completed-task 字节冻结、final checkpoint SHA、训练启动/完成原字节与真实 held-out 输入。旧 Torch1.9.1+cu111 对 synthetic envelope 软件正例及五类坏 envelope 拒绝检查通过，Python3.8 语法通过，源码冻结 `source-freezes/spd-canonical-oof-checkpoint-forward-candidate-20261001`。后续实际执行须检查全部 state tensor 有限、完成 iteration 与 strict load，并对两侧两 shard 的 first/middle/last 与各 held-out scene 首帧真实前向、原 query 覆盖及权重不变验收；当前无真实最终 checkpoint/GPU 前向或完整 OOF 覆盖验收。第 3 折上传 17:46:55 UTC 的进程传输采样约剩 1,107 秒；只是传输 ETA，不含字节回读、派发和训练。

v6 V100 任务 `d0d2b630acdb47adbdb297173d687d9f` 已通过离线 GL 与四卡运行库导入，实际四张 Tesla V100-PCIE-32GB；四个 rank 的标注 pipeline 预检均完成。随后 sampler 的随机种子 NCCL broadcast 报 `NCCL2.7.8 unhandled system error`，零优化器迭代，失败 profile 产物保留。完整脱敏日志 `receipts/spd-oof-fold2-v100-v6-sampler-failure-full-log-20261001.json`。第 3 折本机 v6 续接 PID86479 已停止，原上传继续。应先以独立无数据通信诊断核验 NCCL 环境，不自动重试拟合或修改成员关系。

独立四卡无数据 NCCL 诊断 `501461a3f38646c0a8c4f3203e1a1f2b` 已派发并确认 in_progress。固定相同 runtime/离线 GL/Docker，baseline、NCCL_SHM_DISABLE=1、禁用 SHM 加 loopback 三组各有 90 秒进程上限；广播和两种 payload 的 all_reduce 检查所有四个 rank，开启 NCCL INFO 日志。无数据/标签、模型或优化器，排队和运行不视为通信通过。源码固定 `source-freezes/spd-nccl-gpu4-no-data-probe-20261001`，派发回执 `receipts/spd-nccl-gpu4-no-data-probe-dispatch-20261001.json`。17:55:42 UTC 第 3 折传输 ETA 约 575 秒，不含发布后验收、派发或训练。

V100 Python 环境清理探针 `a64b8d64f8b542dd97241ebde4a82397` 完成但三组通信均失败；全产物独立回读 SHA `20fe00d9244f96d8ee2ec201b46c9650ac12beca9e67ef4fb68c3e35719e4d82`，NCCL INFO 明确缺 `libnvidia-ml.so.1`。原容器只启 compute，NVIDIA 官方文档说明 NVML 需要 utility。相同冻结控制器，仅增加容器 compute,utility 的无数据探针 `cb451dd7590c41c5a84d4a5ac9bca545` 已派发，尚未通过。原失败与无效 v1 Python 探针均保存，不将 completed 诊断状态视为通过。第 3 折上传完整独立回读通过 SHA `b43e9e0e830fe2052a89ae5f84b5b16b1d26cebe8ab0e2aad5e1a0aac513ea35`；第 4 折包发布启动。第 0 折 held-out 推断输入 910,916,711 字节归档全量 inventory 独立核验通过，SHA `ddf79108d34ef24f4f91760a1e727fa86a088257455095f6131aa48ae3ca0873`；尚无实际预测。

V100 utility-only 修复后的无数据探针 `cb451dd7590c41c5a84d4a5ac9bca545` 已完成，全产物独立回读 SHA `f3c4036a7319fa7d368f9ba8eab2ef3c4b4e8b57df735d00432bba42efb0514c`；实际 V100 GPU0–3 四 rank 的广播及 1/262144 元素 all_reduce 全部成功，三组各 exit0。新增 v7 派发器保留 v6 控制器原字节，仅添加容器 utility 能力及严格通信/历史失败 profile 门禁。第 2 折 seed1337 关联原 v6 零优化器失败后新任务 `6134c1c99e2141b3b919801f6874ff43` 已 in_progress；不能据此宣称训练完成。启动 ETA 未知，后续按既有迭代日志给阶段 ETA。原所有失败回执保留。

实时 Worker 分配已核验：V100 GPU4–7 接第 2 折 `6134c1c99e2141b3b919801f6874ff43`，GPU0–3 接第 3 折 `ca63b926f4c9412b84ab2f77484fc7fe`，重叠八卡 Worker 未占卡。两任务 in_progress 环境准备，完整训练与检查点未完成；A100 第 0/1 折并行运行。派发配置、通信证明与源码哈希写入执行台账。

五折 held-out GT-free 图像/标定/位姿/portable infos 已完整打包并逐归档所有成员独立核验：3,937,843,575 字节、73,896 文件；总回执 SHA `a38ed021adb55158c9546d33f94280efe83cbd44e88dd68cefad79f7c374b98c`，未生成预测。发布 session14872 已确认等待正在运行的第 4 折上传 `99f20b72842240b795ab94bc72b87a07`，完成后只做一次完整 remote112 字节验收，再顺序发布 held-out 包；不将上传登记哈希当远端独立读回。第 4 折 fit 续接 session88454 等待上述独立 receipt，之后重新检查 GPU4-2080ti 的物理卡交集并去重派发 v7；这只是当前空闲候选资源，批量仍由实际显存探针决定，尚无第 4 折训练任务。启动/命令/哈希回执 `receipts/spd-oof-heldout-input-publication-and-fold4-continuation-start-20261001.json`。

V100 第 2、3 折 vehicle-side batch2 四 rank 各 32 优化器迭代与 80% 显存门槛已实际执行并独立产物验收，回执 SHA `ead887d1d6110fdbf5d818dd8248ca3e6286d404807a6599baeff8768828de8c`。这仅是 probe，不是最终 batch 选择或正式训练完成。第 4 折上传进程 PID2469 两次 nettop 样本给出约 26 分钟传输剩余 ETA（2026-09-30 18:39:16 UTC）；只属传输估计，完整实验 ETA 未知。已有心跳更新为包含当前四折 GPU 任务及两个本机依赖续接，避免沿用旧状态重复派发。

V100 第 2/3 折最终 batch 选择及正式 vehicle-side 启动元数据已独立字节核验：每卡8、总batch32；两侧所选 batch 显存探针各四 rank32迭代、显存<=80%，fit37/held9 成员与配置、启动16步骨干更新/优化器状态记录匹配。回执 SHA 分别为 `e3798c4f34f9f8ca7049b5d7c11aa32ef933539df6938a07ff9dfadd6c5f9858` 与 `254b246122480fb0c3db7ac29bcfcd9962dcb066dedbbcfb15ddd1fb1d3ff4de`。这是实际 metadata 及选定 probe 的验收，不包括 startup checkpoint 张量或最终训练完成。最新任务级车端进度 130/5112、90/5328；阶段 ETA 约3小时17分/3小时38分，整体实验 ETA 未知。冻结入口 `source-freezes/spd-canonical-oof-batch-startup-metadata-auditor-20261001`，后续 completed 两侧检查点仍需原完整 freezer 与真实前向验收。

held-out 输入包发布后的独立全字节入口 `accept_spd_oof_heldout_package_remote112.py` 已冻结，绑定五折总 admission SHA 与上传任务 manifest/admission 参数；三项产物（完整输入归档、package manifest、原独立 archive receipt）使用已冻结 identity Range reader 在112完整重算，认证只经临时 SSH stdin。源码见 `source-freezes/spd-oof-heldout-package-remote-readback-20261001`；仅语法核验完成，尚未在已发布 held-out 包上执行，不宣称远端字节验收。

补齐各折校准/关联头/VoI 头的 fit-only 特征输入：从原已验收46序列 GT-free 输入中组合固定其余4fold，创建独立 `canonical-oof-fit-feature-inputs`，不改变划分、不读取 GT/val/test、不生成预测。五折共65,352主体帧引用（36/37fit）全文件及每个 portable infos 值/类型/dtype/shape 均独立回读一致；原 held-out 评分入口逐折明确拒绝这些新fit视图。回执 SHA `3ff6673dc459939051b5d16ac59f305594f0570e195940dca4d547cd95fc7a76`，源码 `source-freezes/spd-canonical-oof-fit-feature-inputs-20261001`。这仅是下游 fit 特征推断的输入依赖，后续须独立 fit 导出合同、已完成检查点和真实推断，不用于 pooled OOF 评分。第4折上传 `99f20b72842240b795ab94bc72b87a07` 已completed及登记哈希核对，等待既有续接做全量远端字节验收。

五折 fit GT-free infos 已在112固定 `sha256:85525aef...` Docker、network none、Python3.8.20/NumPy1.19.5 中实际读回全部10 pickle 和65,352 fold内主体帧：公开 NumPy array 构造、所有数值有限、next/prev/sweeps为空、各fold fit成员和held排除检查通过。回执 `receipts/spd-canonical-oof-fit-feature-infos-legacy-runtime-readback-20261001.json`；仅序列化兼容验收，仍无真实 fit 特征预测或校准/关联/VoI训练完成。第4折原包完整远端字节验收 SHA `24c52adf6f18d4eef57a54dc0088d0d330f56bcb6b28914e20799f06f823e3e5`，训练 `a4ad4a451bd94a79a96cea8f38b083c3` 已由 `ubuntu26-2:gpu4,5,6,7` 接单；环境启动0迭代，ETA未知。五折seed1337现均已派发，不视为全五折拟合完成。

第4折两侧最终选定 batch4 四 rank32步显存探针和 vehicle 正式启动元数据已独立读回，回执 SHA `2ba1b43b663f9e1a55b38f2f9dfeeffa71d942c3045191fe7e76e8b674dd4e9e`；实际设备见原回执，不代替最终训练或权重张量验收。第0折 held-out 输入包 ClearML任务 `a3d193998acc44be9c0f1853db130471` completed，三项产物在112全字节独立回读匹配，回执 SHA `01dd14ded83b2bb2f7c139972e88ecbd2c5aef40577d24dec9a9ff3bce293bbb`；尚无预测。其余四包由原协调器继续顺序上传，禁止重复进程。2026-09-30 19:26 UTC Worker快照确认两个V100四卡分别运行fold2/3，八卡Worker虽空但与四卡占用交集，不派发额外重叠任务；完整snapshot `receipts/spd-fivefold-gpu-workers-20260930T192620Z.json`。

已新增独立 fit 特征输入门禁与训练证据绑定合同，原 held-out 门禁/绑定保持原字节。五折真实 fit 输入全 payload SHA、元数据/infos SHA、帧覆盖、图像来源与 split 清点通过，fit 与 held-out 门禁互相拒绝相反角色；实际输入验收 SHA `711e0977efaf36014159198dad93a54b901b401602d3fca4f5d2cbaec1d32fed`。绑定合同只接受对应折两侧24 epoch完成权重和原始启动/完成 evidence 字节，强制fit特征不具held-out评分资格；5个合成正例/15个泄漏或篡改反例软件检查通过，SHA `95a5aff78652f78e9f2ad34931866b0f057a7af5cdd9f9b1c604d02400e2f2e0`。源冻结 `source-freezes/spd-canonical-oof-fit-feature-export-contract-20261001`；尚未接通真实fit exporter/完整缓存读回，尚无实际完成权重验收、fit预测或校准/关联/VoI头训练完成。CPU教师seed1337实时仍in_progress、无迭代或产物证据，ETA未知，保留原任务不重启。

第1折 held-out 输入包任务 `5fc360af80a94f0c9ea00e4cf630ef9e` 已completed，完整889927281字节归档及另外两项产物在112独立回读通过，SHA `0d784e7a98dc535fdb6ae762f87893d9d18c87c29cdea2e956e384322eaa398c`；这是输入发布验收，不是预测或评分。原协调器继续发布fold2–4。

独立 fit-only raw producer 与全 NPZ/JSON 字节读回器已接通为候选，使用新kind `eventtrack_fit_feature_raw_detector_cache_v1`、明确评分不合格，并绑定fit输入和对应fold完成检测器证据；两种确定序列分片、无GT/annotation/future、无优化器、全query解码、128D ImageNet appearance、冻结状态检查与原held-out实现一致。数组/几何函数AST相同，Python3.8语法检查及旧/held-out cache角色拒绝通过；源冻结 SHA `ef9c00e7981dcc0e14fa0a7a686f206c09f04a68b45076420f3f59acc31556a1`，路径 `source-freezes/spd-canonical-oof-fit-feature-raw-export-candidate-20261001`。原held-out源码未改，尚无实际GPU forward/raw cache验收或源码上传，正式D2 calibration cache与D4头训练仍缺。

2026-09-30 19:42 UTC 第0折训练实时任务仍in_progress；vehicle已到5040/5040，final checkpoint/completion已登记，路端刚进入10/4536。只是阶段切换观察，完整两侧训练及最终权重冻结尚未完成；路端初期无可靠ETA。快照 `receipts/spd-oof-fold0-live-20260930T194227Z.json`，不提前启动下游推断。

第2折 held-out包任务 `a8c84687be8848f48dcf4cc8fd6633e3` completed，三项产物在112全字节读回匹配，归档755250949字节；验收SHA `f772b8f5e168697ad246aa08ec3cf1ab3ead60ee514b10d5527ca1c1b9189b3f`。fold0/1/2均已远端独立验收，不重复；原PID10596/session14872继续发布fold3/4。训练仍运行，第0折路端80/4536的阶段ETA约3小时20分，仅当前阶段，不是整体实验完成时间。

五折fit特征发布采用已冻结metadata overlay，避免重复上传各fold共用GT-free图像/标定；原fit输入membership/metadata/infos字节不改。共40 metadata文件，压缩27490108字节，archive SHA `9869c334cb7e1a2b2bead5573ac9f8991cca5ada43391c623e0f82890726625c`；独立逐成员源字节相同验收SHA `a5707455ccd7310e0315378b388e5d9d3172a5829715d1be0005a882f228817b`。按对应fold其余4个已验收held包image/calib清单核对fit全共享payload清单；只复用原资产，不新分割。source-freezes/spd-canonical-oof-fit-feature-overlay-package-20261001；尚未实际重建fit输入、上传或远端读回，后续需独立materializer/full gate验收，仍无预测或论文指标。

第3折held-out发布 `94b44b24091a4d10825011229bd5eadd` 三产物远端全字节验收通过SHA `d97a639dfa875508fab7fbdef108f29daeb0292e7c1eb4ea3c0393e6cd5e4d3d`；原协调器继续fold4。fit overlay实际重建全五折及全部文件SHA/metadata/infos/frames/角色门禁通过，65,352主体帧引用，验收SHA `8a303104a6e1fe6c00410e48466418c29a4f962f7a36abdb19d9881b0969c44a`；重建目录保持，不替换原fit输入。已有授权下开始仅27.5MBmetadata overlay发布，包含独立成员与实际重建回执；未完成远端验收，不视为GPU推断。源码冻结在spd-canonical-oof-fit-feature-overlay-publication-20261001与spd-canonical-oof-fit-overlay-remote-readback-20261001；第1折A100日志也已进入路端训练，完整任务仍运行。

全部五折held-out输入已ClearML发布及独立远端字节验收；最后fold4任务 `e2718eea669543f9918ac5ee8682f9fb`，SHA `238b612b12698f3cbaf0f6198d8930a11ad1b94926f4a4f6518e7174fc91356c`。原发布协调器PID10596/session14872正常terminal，不重启。实际重建通过的fit overlay发布 `135fae3c1eb74afbb70a8574e759edfb`，四产物（overlay归档、manifest、原独立readback、实际重建receipt）112全字节回读通过SHA `0a2d53e84f346383852efadb9dec05a7064d8e0543c3bc4c2156f4bd1d04e889`；publication session63327与远端验收session4877均terminal。完整云端输入证据索引 `receipts/spd-canonical-oof-fivefold-cloud-inference-input-publication-20261001.json`，仅汇总既有六组完整读回证据，不是预测/校准/关联/VoI或D1正式发行绑定验收。

2026-09-30 20:13 UTC五GPU训练实时均in_progress且近期phase日志持续更新：A100路端670/4536与310/4536，V100车端2400/5112与2350/5328，第4折车端2060/10296。阶段ETA约3小时10分/3小时26分/1小时49分/1小时58分/3小时21分；整体未知，未触发completed freezer。CPU教师seed1337仍in_progress/0/无产物，心跳更新，但最后日志timestamp1790671035702距今约35小时；Worker aggregateCPU约2.9%不证明该任务计算。只读lbin@10.100.35.121 SSH公钥认证失败，本机无对应别名，不能观测进程/判定死锁；已询问可用本机账户/别名，未索要秘密。保留原任务/上限，ETA未知；回执spd-teacher-seed1337-log-timestamps-20260930T201426Z.json与spd-teacher-seed1337-worker-telemetry-20260930T201638Z.json。

新增四卡 sampled checkpoint forward controller候选，仅在原训练completed、完整byte-freeze与原controller/package SHA绑定通过后才下载runtime/source/GT-free held-out和两侧冻结证据；明确不下载train-inputs标签归档。四子进程各分配一张可见CUDA卡，保持原已冻结verifier，两侧各两sequence shard；全部日志/可用receipt先上传，任一失败不产生通过summary。透传子进程帧进度/ETA，整体未知。源码 `source-freezes/spd-canonical-oof-checkpoint-forward-controller-candidate-20261001`，目前仅Python3.8语法及源码审阅，不含实际边界运行/GPU前向验收；bootstrap/dispatcher/source publication仍待接通。

2026-09-30 20:40 UTC 再次响应 V100 并行要求：现有两组四卡 V100 分别运行 fold2 `6134c1c99e2141b3b919801f6874ff43`（3070/5112，车端日志ETA约1小时22分）与 fold3 `ca63b926f4c9412b84ab2f77484fc7fe`（3020/5328，约1小时33分）。物理0–7卡已被这两项占用，八卡Worker不额外派发；保留A100两折与2080Ti第4折同步训练。完整两侧训练尚未完成，整体ETA未知；快照 `receipts/spd-fivefold-gpu-live-20260930T204005Z.json`。

四卡前向控制器五项提前拒绝软件边界（字节篡改、未验收、训练运行中、controller篡改、package错配）全部在下载前拒绝，回执SHA `9c54ef8e20632a92f625d008fb24c78537e7f94c85efc465b447ec614d40fbb3`，不替代真实GPU运行。含bootstrap的10文件源码归档35754字节SHA `7090d401592a86e3db8ee25110c8d91ff67c2d699c9ae8398346df1c8b657075`，ClearML源码任务 `72068e936baf4f61a0c51bb2065167e7` completed；三产物独立Linux112全字节回读验收SHA `db23bda5ba8b1fcc0cf2f222cb6af35b8ab7f6dfad37604ec8b43031fe5ee696`。已授权源码上传，无数据/权重打包；bootstrap与controller尚未在真实GPU任务运行。下一步接通配置/种子去重和物理GPU碰撞门禁派发器，只接受原两侧训练completed且完整检查点冻结证据。

前向派发器 `submit_spd_oof_checkpoint_forward_gpu4.py` 已冻结在 `source-freezes/spd-canonical-oof-forward-dispatch-20261001`。绑定原训练completed、完整11产物本机/登记字节、两侧24epoch证据、held-out输入、独立云端源码；配置/种子/byte-freeze/verifier去重。对同机4/8卡Worker和队列全部空闲监听者检查物理交集，90秒心跳、未知绑定失败关闭，enqueue前重新核验占卡；无GPU型号白名单。1个可派发软件正例和7个拒绝反例通过（只读fixture，不创建真实Task）；未提前派发。20:46 UTC 五折仍in_progress，V100车端3220/5112与3160/5328，阶段ETA约1:16/1:27；完整实验ETA未知。实时回执 `receipts/spd-forward-prerequisite-live-20260930T204600Z.json`。完整检查点freezer通过后才调用派发器，不将软件边界通过当真实GPU验收。

独立前向验收入口 `accept_spd_oof_checkpoint_forward_gpu4.py` 冻结在 `source-freezes/spd-canonical-oof-forward-acceptor-20261001`。只接受真实任务completed，独立完整回读9产物（4报告、4日志、summary），绑定执行bootstrap/源码/archive/controller/verifier/云输入及原训练11产物；逐侧验证完整权重迭代/严格load/有限tensor/状态未变，四分片帧/原位姿清单与真实infos一致，summary字节和实际设备一致。全五折20分组132采样位已根据已验收输入算出；1合成正例与10篡改反例软件检查通过，不代替GPU运行。20:50 UTC全部训练仍in_progress、无全两侧冻结，尚无真实forward任务验收；实时回执spd-training-forward-gate-live-20260930T205021Z.json。完整OOF预测、D2校准缓存、D4头、正式独立评价与论文仍未完成。

完整held-out raw export四卡控制器候选 `run_spd_oof_complete_raw_export_gpu4.py` 冻结在 `source-freezes/spd-canonical-oof-complete-raw-controller-candidate-20261001`；原raw producer/query decoder/cache verifier字节不变。强制对应fold完整训练byte-freeze和独立sampled-forward验收后再下载资产；四单卡子进程，两侧各两确定scene分片，保留全query/128D appearance/原时间位姿、GT-free边界。失败保留四log；成功后再次独立CPU逐数组/metadata/full-frame回读，打包并逐成员字节核验，发布4cache+4manifest+4readback及summary；仅job内验收，云端独立验收另行执行，calibration/formal_v2始终False。五项软件拒绝在0下载前通过（篡改/错status/train/acceptor/running forward）；尚无源发布/bootstrap派发或真实cache。20:55 UTC五训练继续运行，无对应完整byte-freeze/forward验收，未新增GPU任务。

完整raw推断启动脚本与九文件源码包已发布任务 `fc3d2af1feb849eea4aa2222ba15dc5b` completed，归档27866字节SHA `306bd3bd2a5b5b5247e27754842393e779578eda96f7feb8549b6861bf039de3`，manifest SHA `d3901dc04fc3606c5441a939f40fa200bf169ede5d7889fb2f011b2d66587fd7`。三产物Linux112独立全部字节回读通过，回执SHA `bf6ea32a77ac930a3f8a7ede6ffa23bf1b1178cdcd814f1fa9cf0c8013c234f3`。旧sampled forward/source/raw producer源码保持；bootstrap只有源码字节验收，无真实GPU运行。完整raw派发器和完整云端cache独立内容验收仍待接通，对应fold完成训练/冻结/独立前向验收之前不派发，不视为D2正式calibration cache或论文结果。

完整raw派发器已冻结 `source-freezes/spd-canonical-oof-complete-raw-dispatch-20261001`：在byte-freeze、真实独立forward九产物本机/注册字节、固定bootstrap/source/controller身份及四分片验收均满足后，才配置源码任务fc3d2af1feb849eea4aa2222ba15dc5b、原producer/cache verifier哈希与原输入索引；去重及四/八卡物理交集门禁同前，1正例/7拒绝软件检查通过，无真实GPU任务。21:00 UTC五训练仍in_progress，V100车端3580/5112（阶段ETA约1:01）与3540/5328（约1:12），完整训练/实验ETA未知。heartbeat配置通过app工具更新并返回ACTIVE，新增已验收源码任务、两套冻结派发/验收入口及禁止提前执行条件；保持15分钟、有意义变化才通知。所有软件通过不代替真实实验验收。

独立完整raw云端内容验收准备中发现旧pipeline通过四元数重构后float32矩阵并不严格等于原标定（五fold共7124帧超出拟定2-ULP对照，最大旋转差4.76837158203125e-7，translation float32差0）；保留失败完整清点回执spd-raw-pose-pipeline-precision-mismatch-20261001.json，不通过放宽容差冒充原始保留。原源码任务fc3d2af1feb849eea4aa2222ba15dc5b与原冻结派发器在实际GPU启动前退休，禁止派发，receipt spd-canonical-oof-complete-raw-v1-retired-before-dispatch-20261001.json；sampled forward和已有训练不受影响。单独raw_pose_v2 producer仅修改导出metadata来源：从已验收held-out资产内原始calibration字节float64 homogeneous组合，不改变模型输入、权重、decode、appearance、split；增加明确来源/helperSHA。全五fold16338帧独立对照已有冻结raw pose table，最大误差4.547473508864641e-13，模型输入/decode函数AST不变；source-freezes/spd-canonical-oof-raw-pose-metadata-v2-candidate-20261001。仍仅metadata输入读回，不是GPU预测/完整cache/cloud验收，新的v2 controller/bootstrap/source派发待接通。

v2完整raw四卡控制器、bootstrap和10文件源码包已独立冻结及发布任务 `b771090298cd4190824e50d2155157e0` completed。archive28557字节SHA `a5321bf92b42ef22de678e7b67a688c7421ef0c54949e57036fd5287c0be3a40`，manifestSHA `04681ea607045761eea078de04a62e764e351a0d64aa2d0b07760d0107fc0dd7`；三产物Linux112独立全量回读SHA `9de0e95633ad90a348802b9ecd33878eab7a00fd2e1c434d0def4ded35732aa7`。runtime额外绑定raw pose helperSHA和metadata来源，原query/appearance/模型输入保持；对应冻结派发入口 `submit_spd_oof_complete_raw_export_raw_pose_v2_gpu4.py` 绑定新source/producer/helper，沿用完整原训练与独立forward身份、去重、4/8卡碰撞门禁，软件1正例/7拒绝通过。v1任务和源历史保留且不得派发。21:13 UTC五训练仍in_progress，V100车端3890/5112（ETA约49分）与3840/5328（约1小时），整体含路端未知。尚无真实v2推断；完整云端cache独立内容验收仍待接通。

v2完整cloud内容独立验收入口 `accept_spd_oof_complete_raw_export_raw_pose_v2_gpu4.py` 冻结在 `source-freezes/spd-canonical-oof-complete-raw-pose-v2-cloud-acceptor-20261001`。要求raw真实任务completed且原训练/forward/source/producer/helper/byte-freeze身份均匹配；独立全量读回17产物，逐归档严格成员size/hash且拒绝越界/GT额外/缺件/重复/链接，逐NPZ与JSON读取检查全frame/query/appearance/底中心-重心几何及原始float64标定位姿，四分片全折coverage、summary及实际Worker/CUDA设备核对，仍calibration/formal_v2/paper False。软件2个正例/9拒绝边界通过，只是安全解包和位姿比较fixture，无真实cloud任务验收。21:21 UTC五训练仍in_progress，V100车端4110/5112（阶段ETA40分）与4050/5328（51分）；两侧完整训练、byte-freeze和真实forward均未完成，不提前GPU raw导出。完整实验ETA未知。

已冻结并启动已有五折训练完成后的字节冻结监测进程，PID91840 / exec session93654，经ps和原工具handle确认live。源码source-freezes/spd-oof-completed-byte-freeze-watch-20261001；每60秒只读原任务，completed且双端completion/final-checkpoint齐全才调用原已冻结freezer，完整原24epoch/split/controller/11产物验证仍由原freezer执行。未完成不提前冻结，失败/缺件/partial输出不自动重试，无新增训练。当前五折均in_progress；整体ETA未知。输出artifacts/spd-oof-completed-byte-freeze-watch-20261001含process与事件/实际命令，尚无实际字节冻结或tensor/raw/校准验收。此进程只续接字节冻结，后续forward及raw须单独验收；不代表整目标完成。

fit-only特征producer新增独立raw_pose_v2派生版本，原冻结候选保持；与held-out相同，从原始标定float64组合导出metadata，不用pipeline quaternion/float32位姿。全五fold65,352帧引用与已验收原始位姿表及对应fit membership对照通过，最大误差4.547473508864641e-13，非main函数AST保持；原fit-only输入与binding门禁仍隔离held-out。输入回执artifacts/spd-fit-feature-raw-pose-v2-readback-20261001/acceptance.json，SHA a64e2e60fa5b03098654eb711b435d18bba46fa41a5ff2e442f2d107b2c6ceb7。第一次审计误用held-out字段名的KeyError保留在schema-check-failure.json与原check.py；改用实际fit schema excluded_held_out_sequence_ids的独立check-fit-schema-v2.py完成核验，不覆盖失败。源码冻结source-freezes/spd-canonical-oof-fit-feature-raw-pose-v2-candidate-20261001；尚未上传/运行GPU fit特征导出或校准。原五折训练及字节freeze watcher PID91840仍live，尚无训练完成事件，整体ETA未知。

新增单折fit overlay重建入口及云端资产下载候选：每个fit任务只取对应其余四个已验收GT-free archive，不取本折held-out、不取train labels。cloud固定证据SHA与overlay独立remote验收SHA、completed/注册字节/归档绑定均检查，原materializer完整40metadata清单与原fit-input gate保持。source-freezes/spd-fit-cloud-input-materializer-candidate-20261001，软件5正例/20拒绝通过，无实际cloud下载/GPU；真实本机单折调用全五fold验证exec session61386在运行，fold0/1已有完整输入验收，其余未完成不算通过。新目录artifacts/spd-single-fit-overlay-materializer-readback-20261001，原已验收allfold重建目录保持。整体ETA未知。

独立fit-only四卡GPU完整特征导出controller候选已建立，不改原held-out controller。沿用原两侧completed训练byte-freeze及独立sampled-forward身份，在这些依赖通过后才获取已验收runtime/源码/R50与fit overlay+其它四fold GT-free输入，经完整fit门禁重建；四单卡children保持原fit-only producer/query/appearance及新原始float64 pose metadata。输出单独fit-only summary、fit membership与held-out评分ineligible标记；不下载train标签，不校准、不声明DetectionCacheV2。5个篡改/错admission/运行中forward反例均0下载拒绝，source-freezes/spd-fit-feature-raw-pose-v2-gpu4-controller-candidate-20261001含完整本地import闭包；无bootstrap/source上传/真实GPU Task。单折materializer实际allfold验证session61386已通过fold0..3，第4折仍运行，不能宣称全完成。

单折fit materializer实际全五fold独立调用完成：exec session61386正常exit0，每fold完整输入门禁重建与所有payload字节核验，汇总65,352帧引用；主acceptance及五fold回执hash再次匹配。artifacts/spd-single-fit-overlay-materializer-readback-20261001/acceptance.json，SHA 105f28d0e07009bd4926a11444ee4197850d06f680ceeb071fa8be94d2fa1d15；实际函数调用与参数保存commands-and-arguments.json。不重复原allfold重建/当前单折验收；这仅证明新单折入口的本机输入重建，尚无云端helper实际下载/GPU特征/校准。

fit-only四卡raw_pose_v2独立bootstrap及12文件源码包已发布ClearML任务13babb70d6e34b0ab9dc5c60b166a1ce completed，archive32,744字节SHA823482458075f3da6e5ed47b83a1ba9e3ca273be5e6c186393eb6893138104fb；manifestSHAf22ca591d771b2a744e42e7470f92b7693ad9608bd886a2fad49c367bc0eec97。三产物Linux112独立全字节回读接受SHAe73288ed7eb3dcb8b058452fa4fb8b725cc0fddbbe136201a11883f9736d2d99；source-publication/reader源码冻结独立fit目录，无GPU Task。21:48 UTC五训练仍in_progress，A100路端2640/4536与2290/4536，V100车端4770/5112（阶段ETA14分）与4710/5328（24分），2080Ti车端5920/10296（1小时48分），整体ETA未知；未提前冻结或GPU推断。下一步fit-only派发器、完整cloud独立cache内容验收仍待接通，原训练/前向依赖不可省略。

fit-only raw_pose_v2独立GPU4派发入口已冻结source-freezes/spd-fit-feature-raw-pose-v2-dispatch-20261001。绑定独立source任务13babb70d6e34b0ab9dc5c60b166a1ce、remote回读、对应单折fit metadata SHA/重建回执及fit-only binding、原两侧completed训练11产物及独立forward九产物身份；固定overlay验收文字传入runtime。名称/源/检查点/fit-input/forward去重，保留失败不重试；同机4/8卡交集、全部队列空闲listeners、90秒heartbeat检查并enqueue前再核验，不限制GPU型号。仅物理Worker与训练未完成边界软件1正例/7拒绝通过，无真实任务创建。完整fit cloud独立验收仍须接通，并纳入新增fit-input-materialization产物（fit运行共18产物，与held-out17产物不同），不能套用held-out验收kind/身份。

fit-only完整cloud内容验收入口已冻结source-freezes/spd-fit-feature-raw-pose-v2-cloud-acceptor-20261001。真实Task completed后独立完整回读18产物，严格归档成员/每NPZ与metadata size/hash/full arrays，fit-input-materialization原overlay SHA/单折帧数角色和原训练本机11产物、前向本机及登记九产物、执行源码/hash/config绑定；原始位姿来自对应其余四fold已验收table及source manifest链，拒绝本折held-out/重复/跨membership，逐float64位姿及底中心/重心几何核验。四shard全fit帧/设备/summary核对，始终held-out selection scoring/calibration/formal/paper False。2正例/9拒绝仅软件安全解包与位姿比较fixture通过，无真实cloud/cache验收。之前原held-out验收入口不变；完整Stage2/校准/等资源基线/论文未完成。

真实后续依赖监测进程PID10101 / exec session45338已启动并用ps及同工具handle核验live，上游原字节freeze进程PID91840/session93654继续，不重启。源码source-freezes/spd-oof-inference-dependency-watch-20261001含21文件，六既有派发/验收源码冻结逐一SHA匹配；软件缺完整freeze零Task查询/创建与源篡改拒绝边界通过。新进程只在原对应fold完整freeze出现后推进sampled forward -> 独立forward读回 -> held-out raw v2 -> 独立完整held读回 -> fit-only raw v2 -> 独立完整fit读回；每阶段使用原冻结CLI、原fit已实际验证环境队列优先并检测全部listener及4/8物理碰撞；无训练创建/失败重试/部分回执覆盖，无formal/calibration/paper通过。命令/实际TaskID/阶段日志/状态变化追加在artifacts/spd-oof-inference-dependency-watch-20261001，后续仍需实时核验PID/handle而不是状态文件推定运行。22:06UTC V100 fold2车端completion/final-checkpoint已登记，进入路端70/4776、阶段ETA约3小时22分；尚非独立检查点读回或双端complete。fold3车端5140/5328约8分；整个实验ETA未知。

D2后续规格/代码源审计确认新缺口：旧build_detection_cache_v2.py第123行只在split=val检查fit_sequences不相交，不能用该旧构建器的通过证明canonical train held-out与calibration-fit隔离。旧工程val缓存/校准历史保留，不重标为canonical OOF结果。preregistration§2.1/§2.3及runbook D2要求相应calibration只用其它四fold、held-out GT不能拟合、pooled OOF作选择，V2应校准score与9×9状态covariance、完整byte/numeric readback和同根vehicle mask。现真实raw推断仍未完成，canonical五fold calibration artifact/其独立拟合重算及单独OOF V2 builder admission尚缺；先补这些再谈D2/选择/论文，旧generic train-fit JSON或源码/software通过均不足。源hash与下一阶段动作回执 /Volumes/Data/test/recover-before-fuse/receipts/spd-D2-canonical-calibration-transition-source-audit-20260930T221444Z.json。无现有代码改动/结果覆写，正式评价门禁不变。

### Canonical OOF calibration boundary candidate — 2026-10-01

Separate `spd_canonical_oof_calibration_boundary.py` reconstructs the fixed fivefold partition and rejects calibration fit membership differing from the exact complementary folds. All five real package/input metadata cohorts passed; 40 negative software cases rejected held-out leakage, incomplete fit, changed partition, wrong fold, missing canonical binding and val/test use. Evidence: `test/recover-before-fuse/artifacts/spd-canonical-oof-calibration-boundary-software-20261001/acceptance.json`. The candidate is not yet integrated into a canonical V2 builder; numerical calibration and full raw-cache acceptance are not established by this check. Existing generic cache builder and running immutable training/inference pipelines remain intact.

### Source-clock ETA readback — 2026-10-01

Fold 1 parent-arrival ETA reported 0.1596 seconds at 3010/4536 while timestamped source logs reported roughly 73 minutes. The frozen v4 launcher computes rate from parent receipt intervals, which can collapse under buffered log delivery. Running immutable sources remain unchanged. Separate read-only `read_spd_training_source_clock_eta.py` uses mmdet source timestamps and the slower cumulative/recent rate; duplicate, phase/reset and clock-order boundaries were checked. Live receipt `test/recover-before-fuse/receipts/spd-fold1-source-clock-eta-20261001.json` reports 3040/4536 and about 72 minutes for the current road-side phase only. This estimate does not prove completion and excludes downstream publication/acceptance.

### Canonical OOF V2 builder candidate — 2026-10-01

Separate `build_spd_canonical_oof_detection_cache_v2_candidate.py` admits the current all-query decode source and its explicit `score_filter_changed=true` export change, while requiring no ROI/NMS/top-k preselection. Before sealing it validates exact canonical calibration complement, hash-pinned independent held-out raw readback, all registered artifact bytes, four exact extracted shard roots, corresponding fold checkpoint/config and full arrays/raw-pose audits. Five real-fold metadata leakage fixtures rejected before output; generic numerical conversion and V2 readback helpers remain AST-identical. Receipt: `test/recover-before-fuse/artifacts/spd-canonical-oof-v2-candidate-boundary-check-20261001/acceptance.json`. This is a candidate implementation only: actual cache sealing, calibration source-example provenance, independent parameter recomputation and numerical calibration remain unproven. Candidate receipts explicitly keep numerical calibration/formal V2/paper flags false. Running frozen training/inference monitors are unchanged.

### Historical calibration source recovery — 2026-10-01

Origin/EventTrackV2X fetched; HEAD equals remote and dirty changes preserved. Missing historical SPD fitter source recovered from completed ClearML source task `a5f0306a60454a57b7943c6fa249ac27` (code SHA c139f29f7b8879f46200da7a2dd6fd7cf0c39a019bf874353efe1db59f909fc4) and original completed calibration task `9273a21bdc764158bb3a22efc1de75a2` (SHA 362e64b5942ecc15026f1d69539472fb995bea86050c4ffc0f718ea1f71f2db5). Full artifact readback matched historical receipts. Parameters and supervision/example logic are inspectable in `test/recover-before-fuse/artifacts/spd-historical-calibration-settings-audit-20261001/source-recovery-receipt.json`. This seven-sequence historical nested calibration is not admitted as canonical fivefold evidence; corresponding complement supervision/raw-cache/parameter recomputation remain required.

### Canonical calibration fit supervision readback — 2026-10-01

`spd_canonical_oof_calibration_supervision.py` reads only each canonical fit-fold converted train PKL, bound to its frozen training-package conversion hash, manifest content checksum and exact fit-feature frame index. All five folds read back 65,352 fit frames in total; geometry/coarse class/source-local IDs are returned, upstream velocity/future fields excluded, velocity left unknown for past-GT reconstruction. Correct class order is imported from DetectionCacheV2 (`car,bicycle,pedestrian`); the initial duplicated-class candidate was superseded and its evidence retained. Corrected full readback: `test/recover-before-fuse/artifacts/spd-canonical-calibration-fit-supervision-readback-20261001-class-order-v2/acceptance.json`, bound to source/class identity in `source-and-class-binding.json`. Whitelist unpickler rejects eval/system globals. No held-out/val/test supervision opened. This proves fit supervision ingestion only; raw fit caches and fitted numerical calibration are still absent.

### Canonical fit past-GT velocity source readback — 2026-10-01

Recovered historical pure example functions are preserved AST-identically in `spd_canonical_oof_calibration_examples.py`; only a standalone repository import bootstrap was added after an initial boundary-test import failure, retained in `bootstrap-repair-receipt.json`. Exact complementary fit supervision and independently accepted original float64 raw poses produced past-only GT velocity targets for all five fold-fit cohorts. Every returned first-seven state dimension remained unchanged; valid velocity targets are finite and invalid targets remain NaN. The four history boundary tests verify future perturbation invariance, first-seen unknown velocity, correct known past velocity and sequence/side isolation. Receipts under `test/recover-before-fuse/artifacts/spd-canonical-calibration-past-velocity-readback-20261001/`. This is offline supervision preparation only, not prediction matching or fitted calibration; no held-out/val/test GT used.

### Canonical calibration fitting entry candidate — 2026-10-01

`fit_spd_canonical_oof_calibration_candidate.py` binds the independently accepted fit raw export source/acceptor, current completed cloud task artifact registrations, all cloud bytes, four extracted manifests, corresponding fold checkpoint/config and full fit cache arrays/raw poses before reading fit supervision. It uses exactly the recovered historical numerical recipe and writes per-side/class example NPZ records for subsequent independent parameter recomputation. Exact fit complement, past-only GT velocity and downstream raw-score>=0.05/all-class-top64 policy are retained. Evidence explicitly marks detector predictions in-sample because this detector trained on the same complementary folds; no held-out GT or held-out probability claim is permitted. Ten wrong-role/changed-hash admission cases rejected before cloud query/GT access, with actual fitting not executed. Candidate source snapshot and software receipt: `test/recover-before-fuse/source-freezes/spd-canonical-calibration-fitter-candidate-20261001/` and `test/recover-before-fuse/artifacts/spd-canonical-calibration-fitter-admission-software-20261001/acceptance.json`. Actual fit caches, numerical fitting, independent recomputation and D2 remain pending.

### Independent calibration parameter verifier candidate — 2026-10-01

`verify_spd_canonical_calibration_parameters_candidate.py` independently implements score objective/gradient optimization and covariance second-moment/shrinkage/floor reconstruction from hash-pinned per-side/class example arrays. It verifies support counts, pooled fallback source, diagnostics and scope using declared relative 1e-10/absolute 1e-12 numeric tolerances. Six synthetic groups (two pooled fallback) passed; five score/covariance/fallback/hash/held-GT tampering cases rejected. It explicitly does not independently reconstruct examples from raw/GT, so formal V2/paper flags remain false. Initial missing SciPy and cryptography failures were retained. Separate local calibration environment was created at `/Users/lbin/.local/share/recover-before-fuse/calibration-venv`; exact versions/installed RECORD hashes are in `test/recover-before-fuse/artifacts/spd-canonical-calibration-parameter-verifier-software-20261001/runtime-versions-v2.json`. Existing ClearML/remote training runtimes were not modified. Actual experiment calibration is not yet present or verified.

### Independent raw/GT example reconstruction candidate — 2026-10-01

`verify_spd_canonical_calibration_examples_candidate.py` independently rebuilds raw state/query selection, same-class one-to-one 2m XY matching, binary targets and valid-velocity residual arrays, comparing every saved score/target/residual array exactly before issuing a scoped calibration readback. It shares the admitted fit-supervision and past-only velocity helpers. Five synthetic frame cases plus strict 2m/inclusive .05 boundary passed. Integration uncovered an unfitted candidate defect: `frame_examples` returns nested `groups`; the first candidate accessed a top-level class key. `add_frame_examples` now consumes the nested group and was checked against the independent oracle. Old candidate snapshot is superseded before any actual fit/dispatch; failed implementation history is retained in `candidate-repair-receipt.json`. Raw source admission AST remains unchanged. Corrected source snapshot: `test/recover-before-fuse/source-freezes/spd-canonical-calibration-fit-and-example-verifier-candidate-v2-20261001/`; no actual raw/GT example reconstruction or calibration experiment is yet accepted.

### V2 candidate calibration readback gate — 2026-10-01

The separate canonical V2 candidate builder now requires a hash-pinned `canonical_calibration_raw_GT_examples_and_parameters_readback` matching calibration bytes, raw fit receipt, fold/full fit-frame counts, all six example hashes and explicit successful raw/GT reconstruction plus parameter recomputation. It reruns parameter readback against current example bytes, then binds calibration training task/seed/byte-freeze to the held-out detector. Six missing/wrong acceptance scope/fold/coverage cases rejected before output. Snapshot: `test/recover-before-fuse/source-freezes/spd-canonical-oof-v2-calibration-readback-gated-candidate-v2-20261001/`; software evidence: `test/recover-before-fuse/artifacts/spd-canonical-v2-calibration-readback-gate-software-20261001/acceptance.json`. The earlier candidate snapshot is retained; no actual calibration/cache is sealed or accepted by these software checks. Global formal V2/paper flags stay false.

2026-09-30 23:43 UTC：fold0 双侧检查点完整字节冻结已通过。原始前向任务 ecbea00bb3744d4bb4f978107f83af23 在 source-manifest 下载返回 401，保留失败；filehost-v2 修正任务 404fa3c53f494a3ea4ec9ba2d339ffc3 已越过源码下载并运行中，未验收。停止已确认存在同一路由缺陷的原本地派发监控 PID10101；替代监控 PID57513/session51351 仅推进 sampled forward，绑定现有修正任务并保留源哈希、原 tensor 验收器及物理卡碰撞门槛。完整 held-out/fit raw 导出须先完成单独的传输路由修订与验收，不宣称已经自动接续。全部证据在实验执行台账。

2026-09-30 23:47 UTC：held-out/fit raw-pose-v2 启动器、派发器和独立回读器已建立 filehost-v2 修正版；源码任务、预测生成器、raw pose 合同、全量数组/帧审计逻辑与资源碰撞检查未变，前向门槛绑定修正后的验收器 SHA。raw 调度监控 PID59613/session23106 已启动，仅等待前向监控的独立验收回执，按依赖推进未校准的 held-out 后 fit 全量 raw。当前不代表实际 raw 导出、校准或论文评价完成。原始失败与源码均保留。

### V100 parallel training live verification — 2026-10-01 00:03 UTC

The two V100 four-card workers are running canonical OOF seed1337 training concurrently: fold2 task `6134c1c99e2141b3b919801f6874ff43` on physical GPUs4–7, fold3 task `ca63b926f4c9412b84ab2f77484fc7fe` on GPUs0–3. Current infrastructure-side progress is 2570/4776 and 2290/5016; source-clock estimates are approximately 1h43m and 2h09m for the observed phase only, excluding publication and independent acceptance. The same-host eight-card worker overlaps both active groups, so no duplicate eight-card task was dispatched. Live task/queue/worker/artifact evidence and command are under `test/recover-before-fuse/artifacts/v100-live-check-20261001/` and registered in the execution ledger.

The filehost-v2 forward monitor was stopped after the common checkpoint metadata guard defect was diagnosed; fold1 task `b668de04d6b944cbbd964cfac5395890` is now terminal failed with four failure-log artifacts, each matching the independently inspected fold0 failure-log hash. Console tail alone does not expose the guard message. Failures and original checkpoints remain preserved. A separate MMCV1.4 zero-based metadata verifier candidate is software-checked against source bytes independently read from the frozen runtime archive; it has not yet passed actual checkpoint tensor/GPU forward acceptance. The raw dependency monitor remains waiting on independent forward acceptance, and does not admit this new candidate automatically.

### Exact MMCV checkpoint iteration correction — 2026-10-01

The actual frozen runtime archive was independently streamed and hashed on trusted Linux112; its four checkpoint/runner/optimizer/version source files match the inspected MMCV1.4 implementation. The final Nth optimizer update is saved with `meta.iter=N-1` and filename `iter_N.pth`, before the runner increments its counter. A separately named verifier preserves strict finite tensor/state load/unchanged weight checks and requires the exact counter, completed N steps, bound filename and four runtime component hashes. The independent acceptor rejects one-based/earlier counters, wrong completed count or filename, changed runtime/verifier/frame, nonfinite evidence and changed weights (nine negative fixtures). These fixtures do not prove actual forward acceptance.

Separate execution source task `2652410a467f49d982115537a3ceca5a` completed publication and independent full-byte readback: manifest SHA `22540575808bc6cbad33f2c33a42b3972f112b8a1a7bb0a9d9eb3ca1f88ef1af`, readback SHA `d7ef92ecfa41915de5b92539400d6bf720e94aa568582f942947c96356c6292b`. After rechecking the completed fold0 byte freeze, terminal diagnosed failure, configuration and physical four-card collision gates, corrected sampled-forward task `e5a7474c7cd34891a5aacc8b7bd2de8d` was dispatched to GPU4-A100. It remains unaccepted until completed artifacts independently pass `accept_spd_oof_checkpoint_forward_gpu4_mmcv14_iter_v2.py`. Fold1 is deliberately held until this common corrected path passes; raw monitors still require a separately revised new verifier/acceptor source binding. Original sources, checkpoints, failed tasks and freezes remain intact; no retraining, commit or push occurred.

### Fold0 sampled forward accepted; dependent raw launch — 2026-10-01 00:12 UTC

Corrected fold0 task `e5a7474c7cd34891a5aacc8b7bd2de8d` completed. All nine artifacts independently read back and verified by frozen iteration-corrected acceptor SHA `5867c8477c2d038db39bea3903972c27a2b9c11236c6bc935d62c1ee896f4522`. Four shards on actual NVIDIA A100-PCIE-40GB cover 28 sampled frames, strict state-dict load, 651 finite state tensors/33,631,011 state elements per shard, raw forward and unchanged weights. Receipt `test/recover-before-fuse/artifacts/spd-fold0-mmcv14-forward-acceptance-watch-20261001/independent-readback/acceptance-receipt.json` SHA `21a7272473906f885759eb311ef68aafbb4eb86157927c06b0453245368b08ae`. This is sampled forward acceptance, not full prediction coverage, calibrated V2 or paper metrics.

Separate held-out/fit raw dispatchers and acceptors now bind the new independently accepted forward source/verifier/acceptor; source producers, raw pose/array validators, model configuration and physical collision gates remain unchanged. Both real CLI preflights reject fixtures carrying the old forward acceptor before task creation. Preserved partial preparation errors concerned only snapshot path/dependency ordering and were repaired before admission. Revised frozen source candidates and proof are in `receipts/spd-raw-mmcv14-iteration-forward-binding-candidates-20261001.json` and `artifacts/spd-raw-mmcv14-forward-gate-software-20261001/acceptance.json`.

After completed-task/artifact/dedup and fresh physical collision checks, fold0 held-out full raw task `f5d168697a914214a769bcfaf65702b9` and corrected fold1 forward task `f1f2e8bb4a2c4d559004a9b863b39142` were enqueued to GPU4-A100. Runtime/task-level progress is still insufficient for ETA. Full raw cloud/content readback remains required; fit raw and calibration follow only their dependencies. No training repeats, history removal, commit or push.

### Raw controller stale forward dependency diagnosed — 2026-10-01 00:14 UTC

Fold0 full raw task `f5d168697a914214a769bcfaf65702b9` is terminal failed before predictions, with no artifacts. Its frozen execution controller rejected the new independently accepted forward receipt because it still pinned the original acceptor/source/bootstrap; the earlier revised dispatcher preflight did not exercise that embedded controller. No automatic restart or training repeat. Failure receipt: `receipts/spd-fold0-raw-controller-stale-forward-pin-failure-20261001.json`.

Separate held-out and fit controllers/bootstrap archives now bind the new forward chain. Their unchanged producer/helper bytes were copied from the original admitted archives and all archive members reread. The actual completed fold0 forward proof passes both isolated controller admission blocks; changing its acceptor back to the original value is rejected. These checks do not run inference. Candidate source manifests: held-out `776f57b6785a32233be134e80b07744204071cfd8aac98f8672ad19de2a9ab19`, fit `9484132781f5d2a6f1304fa378cd9a3670b4dc0406ff1a921edf9b6935a38881`. They are not yet published or admitted by the dispatchers. Next required steps are source publication/independent readback, separately frozen dispatch/acceptor bindings and a collision-safe corrected launch explicitly tied to this terminal failure. Fold1 corrected forward task remains separate and preserved.

### Corrected complete raw runtime sources published; folds0/1 parallel — 2026-10-01

The full held-out corrected source task `f138009a87d44a4c9aa7828748f6c466` and fit-only corrected source task `5d218685995548948ccb37b92dd337de` completed publication and independent Linux112 full-byte readback. Readback SHA values are `0ee4a73a97624c3ded74c8a39c3352437aec237971bdfd5903b2c54b24596826` and `460c47a07b8d135eb283496db1863a7ef6670719d6c8ded5a49aa542ec7f95a0`. Runtime-v3 dispatchers/acceptors now pin the complete new controller/bootstrap/source chain; original archives and all previous candidate sources remain preserved. This closes the discrepancy between local admission and executed source admission. A diagnosed prior failed raw attempt must be explicitly bound; configuration/source dedup and physical collision checks remain mandatory.

Fold1 corrected forward task `f1f2e8bb4a2c4d559004a9b863b39142` completed and all nine artifacts passed independent tensor/sample-forward acceptance. Receipt SHA `438bb2e325c0106af9984d3e838b117d94453578112900f314936e180aff6a0a` in `artifacts/spd-fold1-mmcv14-forward-independent-readback-20261001/acceptance-receipt.json`.

Corrected full held-out raw tasks were launched in parallel on disjoint A100 four-card groups: fold0 `8adee215bf91448aaddabc47a55c1b76`, explicitly bound to terminal failure `f5d168697a914214a769bcfaf65702b9`; fold1 `f13e55f7f3c643af827896889c039b0d`, first raw attempt for that fold. ETA remains unknown while dependency preparation lacks task-level throughput. Neither task is accepted full prediction coverage until completed and independently read back using runtime-v3 acceptor. Fit-only raw, calibration, formal V2 and paper claims remain pending their dependencies. Source/task/worker/hash evidence is in `receipts/spd-raw-runtime-v3-source-publication-and-parallel-dispatch-20261001.json` and the execution ledger. No training repeats or Git publication.

### Runtime-v3 dependency monitoring and calibration source preparation — 2026-10-01

Monitor PID76768/session25165 attaches the existing fold0/1 full held-out raw tasks rather than creating replacements. Frozen `watch_spd_oof_runtime_v3_folds01_raw_dependencies.py` consumes the exact independent forward receipts, rehashes its immutable closure before subprocess calls, independently accepts completed held-out cloud bytes/arrays/poses, then dispatches and independently accepts fit-only raw exports using the admitted single-fit materialization. Failed tasks or partial acceptance outputs stop that fold for inspection; no automatic failure retries, budget changes or paper eligibility. The original byte-freeze monitor remains unchanged for folds2–4. Obsolete raw monitor PID59613 was explicitly stopped because its forward receipt path and source pins refer to the replaced chain; its source/output/history remain intact, with no remote tasks stopped.

Separate calibration fitter/example verifier/V2 builder revisions bind runtime-v3 raw source tasks and exact acceptor hashes. The only changed functions are fit source admission and canonical V2 source admission; all numerical fit, independent example matching and cache decode functions are AST-identical to prior candidates. V2 admission now also explicitly requires the pinned raw acceptor. Closure snapshot `source-freezes/spd-canonical-calibration-runtime-v3-source-bound-closure-20261001` passed isolated imports/CLI help using only frozen paths; an initial direct `python -I script.py` excluded the script directory and was preserved as an invocation failure, then verified with explicit frozen paths through runpy. No actual calibration fit, V2 sealing or Linux experiment acceptance is implied. Commands, hashes and scoped evidence are in `receipts/spd-runtime-v3-raw-monitor-and-calibration-preparation-20261001.json` and the execution ledger.

### Producer-clock raw ETA and remaining fold forward dependencies — 2026-10-01 00:28 UTC

Both full raw tasks have emitted per-shard frame counters. The read-only ETA helper uses each producer's elapsed clock, retains shard identity, resets on counter/clock restart, rejects invalid counters/nonfinite elapsed time, and returns unknown until all four shards have enough observations. Parallel production ETA is the slowest shard and excludes archive publication/independent readback. Snapshots at 00:26 UTC were approximately 73s for fold0 and 92s for fold1. Fold0 later reached 914/914, 893/893, 938/938 and 870/870 across its four shards and entered cache-archive-publication; overall ETA becomes unknown for that phase. These counters do not prove cloud byte/content acceptance. Frozen phase-v3 reader recognizes the actual held-out and fit phase tags and does not report stale production ETA during publication. Source/counter fixture evidence and live snapshots remain preserved; existing training controllers were not modified.

PID80071/session67746 now monitors only folds2–4 for the original completed byte freezes, then uses the corrected verifier/source chain to dispatch a sampled forward on a compatible collision-free four-card worker and independently accept results. It never creates training tasks or retries failures. Initial prestart candidate incorrectly retained a fold0 attachment requirement; its failed source/output were retained, and a separately named v2 candidate removed that irrelevant precondition before successful start. Scope is still sampled forward only; full raw/calibration/paper dependencies are not declared complete. At 00:26 source-clock V100 estimates were approximately 80min for fold2 and 104min for fold3 infrastructure-side training, excluding publication/readback. Evidence is recorded in `receipts/spd-raw-producer-eta-and-remaining-fold-forward-monitor-20261001.json` and the execution ledger.

### Raw publication complete; independent readback underway; calibration runtime candidate — 2026-10-01 00:33 UTC

Both held-out raw tasks `8adee215bf91448aaddabc47a55c1b76` and `f13e55f7f3c643af827896889c039b0d` are completed with all 17 registered artifacts each. This is not independent acceptance. The original monitor PID76768 started fold0 independent cloud/archive/frame/array/raw-pose readback via PID81903; the process is live and downloads are in progress. Zero-length current download files reflect the original buffered helper while transfer is active; no restart is justified. Fold1 acceptance follows the same admitted monitor. Full raw acceptance and calibrated V2 remain pending.

An idle L40S CPU listener and empty queue were verified before dispatching data-free canonical calibration Linux runtime candidate probe `baed36a619b0433b8655bdde0e7971af`. Packages pin the existing local candidate environment versions: ClearML2.1.5, NumPy2.5.3, SciPy1.18.1; single-thread BLAS/OMP and empty CUDA visibility. The standalone source embeds exact AST-preserved fit_score/fit_covariance/wrap_angle and independent score/covariance oracles with original-file/function hashes, checks three deterministic score/covariance cases, and records Linux/Python/package RECORD identities. Local synthetic checks passed; remote queued/running status is not runtime acceptance, actual experiment-data fitting or a change to the pending MHT Linux full runtime contract. Source SHA `a730b8c40bd7dd89883a83e98fe5f4b40fd132333ea4527ea6961520bbbc2a19`; command/config/task/receipt evidence in `receipts/spd-canonical-calibration-linux-cpu-runtime-probe-dispatch-20261001.json` and `receipts/spd-raw-readback-and-calibration-runtime-preparation-20261001.json`. ETA unknown without task-level progress. No existing teachers, failed tasks or frozen contracts restarted or changed.

### Calibration CPU runtime fixed-version mismatch — 2026-10-01

Candidate probe `baed36a619b0433b8655bdde0e7971af` failed at `candidate package versions differ` with zero artifacts, before synthetic numerical cases or any experiment data. Declared requirements are the intended three pins; actual observed versions were not recorded before the failed assertion, so the mismatch cannot be attributed to a specific package from the current evidence. The container's CUDA initialization warning is not the declared failure and does not certify CPU numerical validity. Failure receipt and original source are preserved.

A separately frozen observation-v2 diagnostic source uploads `runtime-observation` (expected/observed versions, interpreter executable/version and platform) before retaining the exact version assertion. Its dispatcher binds the original terminal failed source/task and requires an idle L40S CPU listener. This is instrumentation of a diagnosed missing-observation condition, not a relaxation of the package contract or a restart of teachers/training. Actual Linux calibration runtime and experiment fitting remain unaccepted until independent observation/parameter evidence is available. Full raw readback continues on the existing live acceptor/monitor handles without restarting buffered transfers.

### Fold0 full raw acceptance and parallel complementary-fit export — 2026-10-01 00:41 UTC

Fold0 held-out task `8adee215bf91448aaddabc47a55c1b76` passed independent readback of all 17 artifacts, full frame/query arrays and original float64 poses: vehicle1808 plus infrastructure1807 side-frames, 3,253,500 raw queries. Acceptance receipt SHA `b8e8ea7448aa9632f44238b621d08816e1ebe0773554f6355777f6cdba64a0a7`; fold1 independent readback continues on the existing monitor. These uncalibrated predictions do not satisfy calibrated V2 or paper evaluation.

After fresh idle-worker/physical-collision and source/configuration dedup checks, fold0 fit raw task `722bbfaa3b914af783d12821df7056ee` and fold1 `a449296169324a25998e3c77c59316c8` were dispatched to GPU4-A100. This GT-free complementary-fit feature export depends on independently accepted forward/checkpoint/input evidence and can run alongside held-out readback. Dispatch is not acceptance; ETA remains unknown until task-level producer counters support an estimate. Existing monitor attaches these exact task IDs from its canonical dispatch receipts.

Separate Linux126 calibration CPU runtime task `969e56c034714dfab10e16471d68d7e1` completed; independent artifact bytes/Linux identity and three deterministic numerical cases passed, including cross-version local oracle repetition. Actual versions NumPy1.26.4/SciPy1.14.1/ClearML2.1.5 were recorded, not misreported as the earlier requested versions. Earlier mismatch failures and observation bytes remain preserved. This is data-free runtime acceptance, not experiment calibration fitting or authorization of the pending full MHT Linux contract. Receipt: `test/recover-before-fuse/artifacts/spd-calibration-linux126-runtime-independent-readback-20261001/acceptance-receipt.json`. Commands, task IDs and hashes are indexed in `receipts/spd-fold0-full-raw-and-parallel-fit-progress-20261001.json` and the execution ledger.

00:46 UTC live verification: V100 fold2 infrastructure3480/4776, current-phase ETA3695s (~62min); fold3 infrastructure3210/5016, ETA5067s (~85min). Actual workers remain physical GPUs4–7 and0–3 respectively, with no overlap. Same-host eight-card listener is not physically idle. A100 complementary-fit exports are both in_progress on disjoint four-card groups; insufficient producer samples => ETA unknown. These are phase estimates excluding publication and independent acceptance. Live snapshots and SHA256 indexed in execution ledger.

### Fit-only calibration supervision transport admission — 2026-10-01

Fold0/1 exact complementary supervision passed frozen conversion inventory/hash, canonical partition/frame/timestamp/annotation geometry checks via `load_fit_supervision`, covering12723/12615 side-frames. Separate create-once archives include only corresponding train PKLs, conversion/package/input manifests and fit frame indexes; no held-out GT, official val/test PKLs. Every archive member was independently streamed and SHA256 checked against source inventory. Receipt `test/recover-before-fuse/artifacts/spd-canonical-fit-supervision-transport-20261001/acceptance.json` and immutable preparation source are registered in execution ledger. This is local input admission/transport preparation; no cloud publication or calibration fitting has occurred. Fold0 fit producer progress at00:49 UTC supports current generation ETA803s, excluding publication/readback; overall ETAunknown.

### Fold1 full held-out raw and calibration cloud inputs accepted — 2026-10-01 00:54 UTC

Fold1 held-out raw task `f13e55f7f3c643af827896889c039b0d` passed all17 independent cloud artifacts and full frame/array/raw-pose checks, vehicle1937 plus infrastructure1786 side-frames. Receipt SHA `188cc9581d13f5a414a0507bc8883b67184c36c4b9db3c9d9e3030ae10a22581`. Raw predictions remain uncalibrated and ineligible for formal V2/paper metrics.

Canonical complementary-fit-only supervision publication tasks fold0 `9459a23705a24c1c89a2fa7613268815` and fold1 `e554bf277e58439382e30900889af509` completed and independently passed full archive/manifest bytes and exact archive-member checks. No held-out GT or val/test PKLs were included. Calibration runtime-v3 source task `641448cbbb83421ab6c63b8537c3bf99` completed and independently verified all63 frozen source files, manifest SHA `2d01f5d0bfc94c4837eedd574ef6b2176f8211dc428352b49d27573a8533c517`, archive SHA `3963c3709362d6ade9e3c7be732ff1f52555ec9c63977bbf0d13d4c0de3a0e5f`. Original frozen code and numerical recipe remain unchanged. Actual calibration still requires full fit raw acceptance and a separately bound Linux126 CPU execution controller; these cloud dependencies alone do not establish fitting or parameter acceptance. Evidence, commands and hashes indexed in `receipts/spd-fold1-full-raw-and-calibration-cloud-dependencies-20261001.json`.

### Full calibration runtime metadata and Linux126 CPU runner candidate — 2026-10-01 01:02 UTC

Metadata task `c5ca5855aed84f6e83ceac022578f5ed` completed independent cloud bytes/exact archive-member readback:17 files, five original float64 pose tables and five GT-free held-out manifests, fold0/1 byte-freeze receipts, exact cloud/overlay admissions and three unchanged download/materialization helper sources read from independently accepted fit source bundle. Metadata manifest SHA `907e5a0986daa4c46a747d0f7344d5a5f3edf0f7dd1ed718fb37c4710ae28492`, archive SHA `1ec941e8a21e504d0e0df51370e8920b10d0380dfad992bdfadf33b355efe39c`. Original fitter audits require full GT-free image/pose inputs and the original absolute manifest mount; a reduced manifest-only fit input does not satisfy them.

Separately frozen CPU runner candidate verifies Linux x86_64/Python3.12.3, exact NumPy1.26.4/SciPy1.14.1/ClearML2.1.5 versions and admitted package RECORD hashes, empty CUDA visibility/single-thread policy, all63 frozen source/17 metadata files, exact complete fit raw receipt and corresponding fold. It executes unchanged frozen fitter then full independent raw/GT example reconstruction and parameter oracle, retaining original absolute audit root. Matching inventory passed; changed hash/size, traversal, duplicate, extra and symlink fixtures rejected. This is software admission only: bootstrap/input staging/cloud dispatcher still pending and no actual calibration run has started. Candidate scope is folds0/1; remaining folds need their own accepted freezes. Commands/IDs/hashes in `receipts/spd-calibration-linux126-runtime-assets-and-runner-candidate-20261001.json`. Fit generation snapshots at00:58/01:02 UTC estimated220s for fold0 and73s for fold1 at producer sample; upload/independent acceptance ETAunknown.

### CPU calibration cloud execution chain and verified fit readback wait — 2026-10-01 01:11 UTC

Separately frozen Linux126 bootstrap stages the admitted63-file calibration closure,17 metadata files, corresponding supervision archive and full four-fold GT-free fit assets; fetches all accepted fit raw cloud artifacts and uses the AST-identical original cache extractor. It preserves exact bytes of the embedded CPU runner, original absolute audit manifest mount and original fitter/oracle. Actual job outputs would include calibration, six example arrays, runtime identity and full job-local independent example/parameter reconstruction; completed cloud outputs still require separate independent readback.

Dispatcher validates complete matching fit raw admission/current registered and local artifact bytes, source/metadata/runtime task identities, fresh idle L40S CPU listener and empty queue. It persists handles before enqueue and deduplicates by exact raw fit plus bootstrap/fold/seed; broad same-fit search blocks alternate-source silent repeats. Failed/stopped attempts are preserved. Archive fixtures reject traversal/wrong hash/extra files; separate mocked complete-registration fixture independently rejects kind/status/fold/seed/source/acceptor/role changes. These are software checks, not actual calibration acceptance.

Dependency monitor PID7213/session62961 waits for existing fold0/1 fit raw acceptance and an idle CPU listener, then dispatches once and records terminal state. Its scope deliberately excludes cloud output acceptance. fold0 fit raw task `722bbfaa3b914af783d12821df7056ee` reached completed at01:08 UTC; original acceptor PID6690 is live in full independent readback. No duplicate acceptor or actual calibration has been launched yet. Source-clock V100 snapshot01:08: fold2 infrastructure3950/4776 ETA2321s (~39min), fold3 3680/5016 ETA3736s (~62min), both unchanged disjoint physical four-card groups. Snapshot/command/source hashes indexed in `receipts/spd-calibration-linux126-cloud-execution-and-dependency-watch-20261001.json`.

### Full independent CPU calibration output acceptor prepared — 2026-10-01 01:17 UTC

A separately frozen acceptor requires the completed exact CPU bootstrap/image/worker/configuration, all11 cloud output artifacts, immutable fit raw admission, actual Linux/Python/package RECORD identity and summary-to-cloud byte inventory. It independently downloads every output then reruns the unchanged frozen full fit raw/GT example and numerical parameter oracle against existing admitted local fit inputs/supervision. The complete reconstructed report must equal the job-local independent report and cover every fit side-frame. It preserves relative1e-10/absolute1e-12 tolerance across the separately admitted local and Linux candidate runtimes; actual agreement remains unproven until execution.

Matching runtime fixture passed; fold/raw-source/runner/package/version/record/CUDA/interpreter/platform/source/eligibility changes rejected in10 isolated cases. Initial importlib test omitted the helper path and failed; preserved as an invocation failure, repaired by importing only the frozen acceptor/helper directory. CLI help/compile do not count as experiment acceptance. Independent acceptance monitor PID20150/session42863 watches existing CPU dispatch handles only, runs once after completed, and retains remote failures or partial readback failures without retries. It is currently live and waiting; no CPU calibration/output acceptance has occurred. Evidence and hashes in `receipts/spd-calibration-linux126-independent-cloud-acceptor-and-watch-20261001.json`.

### ClearML container identity interface correction; both fit exports published — 2026-10-01 01:18 UTC

Read-only live inspection found ClearML `task.data.container` is a dictionary: `.image` raises AttributeError. The actual admitted probe image matches the immutable digest through dictionary access. This was a diagnostic/preparation error, not a remote calibration experiment failure. A separately frozen identity-v2 acceptor changes only this dictionary image read in main; all other runtime/source/numerical/oracle functions remain AST-identical. Matching dictionary image accepted; wrong/missing/nondictionary identity rejected. The old acceptor and monitor are preserved; waiting PID20150 was explicitly stopped before any CPU output acceptance, replaced by live PID21131/session92082 bound to the new frozen source. No remote training/inference/teacher jobs stopped.

Both fit exports `722bbfaa3b914af783d12821df7056ee` and `a449296169324a25998e3c77c59316c8` are completed with18 artifacts each. Publication is not full acceptance. fold0 acceptor PID6690 remains live in the original full-byte/content readback; fold1 follows the same monitor after it finishes. Existing CPU dependency monitor PID7213 remains waiting on their actual full acceptance receipts; no actual calibration started. Overall ETAunknown during transfer/audit. Commands, live state and correction source hashes in `receipts/spd-calibration-independent-acceptor-container-identity-v2-20261001.json`.

### Remaining folds full raw export dependency continuation — 2026-10-01T01:26:56.265203+00:00

The separate frozen folds 2–4 monitor (PID 25469; session 56988) waits for the existing full byte freeze and independently accepted sampled forward per fold, then uses unchanged runtime-v3 held-out and fit dispatchers/acceptors. All 38 dependency files and queue/collision/execution helpers remain identical to the admitted folds 0–1 closure. It does not retry failures or rerun training. No raw task has been dispatched yet. Full raw coverage, calibration, DetectionCacheV2 and paper metrics remain unaccepted; overall ETA is unknown. Receipt: `test/recover-before-fuse/receipts/spd-folds234-runtime-v3-full-raw-dependency-watch-20261001.json`.

### Remaining folds fit-only supervision transport — 2026-10-01T01:29:59.059416+00:00

Using the unchanged frozen calibration supervision loader, folds 2/3/4 exact train complements were validated and archived with 13,172/13,762/13,080 side-frames. Each seven-file archive excludes held-out, validation and test PKLs. Archive members were independently rehashed locally. Publication and independent cloud readback started in PID 27480/session 16506; its entire validation loop is AST-identical to the accepted folds 0/1 publisher. Cloud acceptance is pending, ETA unknown. No calibration has been fitted or formal V2/paper eligibility granted. Receipt: `test/recover-before-fuse/receipts/spd-folds234-fit-only-supervision-transport-started-20261001.json`.

### Fold 2 supervision cloud readback accepted — 2026-10-01T01:33:15.930557+00:00

Cloud data task `8308b0a5e46b44e9b2c26489912a7ef0` independently read back the 13,172-side-frame fit-only supervision archive and manifest. All seven archive members, exact sizes and hashes were rechecked; no held-out/validation/test supervision was read. Acceptance SHA256: `04a8cb9c84186986dc4701d4208eff348981ebdae001cd096a6a3f8c48e1cd77`. Existing publication PID 27480 continues folds 3/4; no calibration or formal V2 acceptance is claimed.

### Fold 3 supervision cloud readback accepted — 2026-10-01T01:35:19.362872+00:00

Cloud data task `c4a36b754c804c0fb1e988ab079d9b09` independently read back all seven fit-only archive members and exact registered bytes for 13,762 side-frames. Receipt SHA256 `1d7a231035fecef542f4800b8c976173935a288653155e19df70b77baf356e8a`. Publication PID 27480 continues fold 4. No calibration, formal V2 or paper acceptance is implied.

### All five fit-only supervision cloud transports accepted — 2026-10-01T01:38:20.587446+00:00

Fold 4 cloud task `de4d764a3bf24d08bc40a1ad73b61b87` completed independent cloud/archive-member readback, 13,080 side-frames; receipt SHA `faf0dacb4abd3386b4e4fdd85a5ff442b43e24bc76f06b4c1a1bed9ec2449c89`. Publication session 16506 exited 0. All five seven-member archives and manifests were rechecked, totaling 65,352 fit side-frames across five complementary folds; held-out/validation/test GT excluded. This establishes supervision transport dependencies only. Full raw fit acceptance, actual calibration, V2 sealing and paper results remain unaccepted. Consolidated inventory: `test/recover-before-fuse/receipts/spd-canonical-oof-fivefold-fit-only-supervision-cloud-inventory-20261001.json`, SHA `0c1c6bbeb19ccd001093ddf3b02d27a26938139552a9e87fd3acb3fd57a14d01`.

### Remaining per-fold CPU metadata dependency watch — 2026-10-01T01:41:05.864747+00:00

Separately frozen PID 35058/session 95609 waits for each existing fold 2–4 training task’s accepted seed1337 complete byte-freeze receipt. It then prepares and independently reads back the 16-file per-fold metadata bundle (five unchanged pose tables, five GT-free manifests, corresponding fold freeze, original cloud/overlay admissions and three original download helpers). Four wrong identity/seed/admission guard cases were rejected. No training or calibration tasks are dispatched by this monitor; partial or failed preparation stops the fold without retry. Existing folds0/1 runtime metadata and CPU execution closure remain unchanged. Receipt: `test/recover-before-fuse/receipts/spd-folds234-linux126-per-fold-metadata-dependency-started-20261001.json`. Overall ETA unknown.

### V100 fold 2 completed; existing byte freezer active — 2026-10-01T01:49:22.081861+00:00

Task `6134c1c99e2141b3b919801f6874ff43` is completed and both final checkpoint/completion artifacts are registered. Original monitor PID 91840 started the separately frozen offline-gl-v6 byte freezer as live PID 41241 at 01:48:30 UTC. Complete 24-epoch evidence, all byte hashes and forward admission are still pending; task completion alone is not acceptance. Fold0 full raw independent readback now has all four cloud archives downloaded; full frame/array/pose audit and final receipt remain pending on original PID 6690. Readback/acceptance ETA unknown.

### Fold 0 full fit raw accepted; actual CPU calibration running — 2026-10-01T01:55:00.144210+00:00

Full independent raw readback task `722bbfaa3b914af783d12821df7056ee` passed all 18 cloud artifacts and four shard frame/array/original float64 pose audits: 12,723 fit side-frames and 11,450,700 raw queries. Acceptance SHA `2442000485dfb163dc247578a3c8db29bacae9ed9b0fcbf75737c8938e113cb3`. Existing dependency monitor dispatched CPU calibration task `9ce85a95b33a457f9d4e832083520305`; fresh live state is in_progress on `10.100.35.121-L40S:gpu3,4,5`, with separate enqueue acceptance and exact fit-readback/bootstrap binding. This is actual calibration execution, not parameter/oracle/cloud output acceptance. The independent acceptor watch remains active. Fold1 full raw readback continues PID42852; fold2 byte freezer PID41241. Calibration ETA unknown without task-level progress; V2/paper gates remain closed.

### CPU pre-data serialization failure preserved; separately frozen correction — 2026-10-01T02:01:20.170861+00:00

Task `9ce85a95b33a457f9d4e832083520305` failed with zero artifacts before any experiment-data fetch/fitting: ClearML returns General/fold_id as string `0`, while old main required a Python int. The failure receipt/source/task remain preserved. Strict parameter-v2 bootstrap admits only canonical integer or exact decimal string fold0/1; bool, float, padded/whitespace/decimal strings, unknown values are rejected. Matching parameter-v3 acceptor also normalizes the exact serialized seed1337. Embedded CPU runner, numerical/runtime/oracle functions and dispatcher admission functions are unchanged. Four fold positives, one seed positive and seven negatives passed as software checks only. New dispatcher permits only this explicitly named, live source/input/error-bound pre-data failed task; other repeats remain prohibited. Old waiting local dispatch/acceptance monitors stopped (7213/21131), new PIDs47256/47480 active; training/raw/freezer processes unchanged. New remote identity and experiment result acceptance remain pending. Receipt: `test/recover-before-fuse/receipts/spd-calibration-pre-data-parameter-serialization-failure-and-correction-20261001.json`.

### Missing transitive fit-input helper failure preserved — 2026-10-01T02:06:46.658776+00:00

Parameter-v2 task `06fafe7e02cf4fad9e9a292ec2ad4c80` passed integer decoding and staged source/metadata/fit-only supervision, then failed before fit input materialization/numerical fitting with ModuleNotFoundError for `spd_canonical_oof_fit_feature_input_gate`. Its source/task and zero-artifact failure are preserved. Exact source import closure traced to independently admitted original fit bundle task `5d218685995548948ccb37b92dd337de`; only missing local module is the input gate, SHA `373364a69facb46d9d38d6dffd652c5187d8f3d619b5d328633de127f67e515a`. Corrected 18-file metadata candidate preserves the original 17 payloads and adds that exact original helper; no numerical recipe changes. Upload/readback session9813 started; cloud admission pending. Waiting dispatch/acceptance monitors47256/47480 stopped to avoid launching the incomplete closure for fold1. Raw and training/freezer pipelines remain active. No third calibration task launched.

### Corrected helper assets cloud/import admission; fold 2 byte freeze accepted — 2026-10-01T02:10:03.429166+00:00

Metadata task `35fe5803bcfb4ac4a2e1358f98dfc286` independently verified all 18 exact member payloads, manifest SHA `61215feeba37ea9d8397e66aef0500691121b723d04040da4b6e5fd73c0c79ca`; old17 preserved. Five exact original helper modules imported successfully in the isolated local calibration environment with no data materialization, GT load or fitting; this is not remote Linux fit acceptance. Waiting metadata monitor35058 was replaced before it dispatched anything by separately frozen PID50945/session38175, which adds the same original gate helper to each remaining per-fold bundle (17 files). Fold2 full24-epoch byte freezer accepted all11 artifacts, receipt SHA `75ff7d01af2d7a7a663b8cc348e01c3fe083a56db20b9aeb8c6643aeec08877e`; original forward watcher started collision-safe GPU4-V100 dispatch. Both CPU calibration failures preserved; no third calibration task running yet. Corrected execution/acceptance bindings still need preparation against the new metadata identity. Receipt: `test/recover-before-fuse/receipts/spd-corrected-import-closure-cloud-accepted-and-fold2-byte-freeze-20261001.json`.

### 2026-10-01 CPU 标定原始导入闭包 v3

保留两次 pre-fit 失败任务 `9ce85a95b33a457f9d4e832083520305`、`06fafe7e02cf4fad9e9a292ec2ad4c80`。新执行冻结版本绑定已独立验收的 18 文件元数据任务 `35fe5803bcfb4ac4a2e1358f98dfc286`，补齐原始 input gate；拟合算法、超参数和独立数值验收规则未改变。实时预检确认完整 fold0 fit raw 输入、历史失败源码/输入/错误及空闲 L40S CPU Worker。新派发与独立验收监视启动；没有任务级进度时 ETA 未知。派发或运行不代表标定验收，正式 V2 与论文指标仍未就绪。证据：`test/recover-before-fuse/receipts/spd-calibration-exact-import-closure-v3-preflight-and-monitor-20261001.json`。

### 2026-10-01 CPU RECORD 原始字节校验修正

保留任务 `ba1cbf2cd2244a11a5f156a7a61bace1` 的 pre-fit 失败，暂停同一故障版本的 CPU 派发。Linux 无数据探针 `40360c08d1a6451593f5b2e84b6535a8` 经独立字节回读确认三个包原始 RECORD 与原验收哈希一致，差异只来自 `read_text` 换行归一化；三个跨版本数值案例通过。执行器改为原始字节读取，保留安装记录原哈希、Linux 环境和数值门槛。新 v4 派发/v5 独立验收监视覆盖 folds0/1 与单独 fold2，需完整 fit raw 验收和空闲 CPU Worker；原任务不重启、GPU 任务不改变。正式 V2 和论文指标仍未完成。证据：`test/recover-before-fuse/receipts/spd-calibration-RECORD-newline-diagnosis-independent-proof-and-v4-correction-20261001.json`。

### 2026-10-01 fold2 fit raw 并行派发

在 fold2 原始 held-out 独立回读继续运行时，按实际依赖和物理 GPU 碰撞检查，使用原冻结导出器向空闲 A100 四卡派发 fit-only raw 任务 `007087273b6c4812a2301e46bee1fa81`。该导出只依赖已验收检查点前向和 fit 输入，不依赖 held-out 回读；既有监视沿原任务回执去重接管完整验收，未改变生产器、配置、种子或数值验收。导出未验收前不得用于标定。证据：`test/recover-before-fuse/receipts/spd-fold2-fit-raw-parallel-A100-dispatched-20261001.json`。

2026-10-01：V100 fold 3 全量 held-out raw 输出任务 dd6c6d4659aa47d0970ea3fd8f6e20b4 completed，17 产物独立回读及 2576 双侧帧内容验收通过。Linux CPU 私有目录固定 wheel 完整导入探针 868783fa53cd4340bc17e0d4ac4b6614 completed，4 产物独立回读、63 文件原始导入链、原始 RECORD 字节及 3 数值案例通过；不代表完整标定验收。缺依赖失败探针保留。证据：test/recover-before-fuse/receipts/spd-linux-isolated-wheel-runtime-and-V100-fold3-accepted-20261001.json。

2026-10-01：完整 Linux CPU 标定使用单独冻结的 isolated-wheels-v5 执行版本，原始 63 文件 fitter/oracle 数值配方及容差不变。固定 wheel 私有模块目录保留原始 RECORD 字节；已验收运行时探针作为派发前置门槛。fold 0 check-only 通过，新的派发/独立验收监听器已启动，排队和运行均不代表完整标定通过；四次历史失败保留。证据：test/recover-before-fuse/receipts/spd-calibration-isolated-wheels-full-execution-prepared-20261001.json。

2026-10-01：fold 2/3 独立命名 isolated-wheels-v5 完整标定执行及 v6 独立验收器已冻结，分别绑定其已独立验收的 17 文件元数据资产及 fit-only supervision。嵌入 runner 字节和单 fold 范围已验证，原始 fitter/oracle 不变；尚未派发，必须先等待各 fold 全量 fit raw 独立验收及实时空闲 CPU 队列。证据：test/recover-before-fuse/receipts/spd-fold23-isolated-wheels-calibration-prepared-20261001.json。

2026-10-01：完整 CPU 标定 274f364207d44160a1e5a67e0e25bba2 failed（0 产物）。固定 wheel 导入已通过，原始 fitter 输入门槛误把 inference resolved cache config 哈希与 training detector config 哈希作等值比较；四个分片的 checkpoint 及 training_binding 内训练配置均匹配冻结，cache 配置字节也匹配其独立验收 manifest。后续标定监听器及未形成 task handle 的 fold 1 派发 CLI 已停止；不重启或绕过门槛。须另行冻结区分两种配置身份的校验源码并验收完整输入门槛，数值配方/容差不变。证据：test/recover-before-fuse/receipts/spd-calibration-isolated-wheels-v5-config-role-mismatch-failure-20261001.json。

2026-10-01：新配置身份修正版源码保留旧冻结包；仅 admit_fit_sources 区分 training_binding 配置 SHA 与 cache resolved 配置字节 SHA，其余计算函数 AST 一致。第一次完整输入预检暴露缺少原始 input gate（失败保留），完整 v5 源码包现补入独立验收过的同哈希原始门槛文件，共 64 文件。全量 fold 0 输入审计进程 89589 运行中，ETA 未知；只有审计完成且回执匹配，源码发布监听器才上传并独立读回，未派发标定。证据：test/recover-before-fuse/receipts/spd-config-role-original-input-gate-source-correction-prepared-20261001.json。

2026-10-01：配置身份修正版完整输入门槛审计通过：fold 0 全量 12723 双侧帧、4 分片，原始训练绑定和实际 cache 配置字节分别核验；未读 GT 或拟合。64 文件完整源码 ClearML 38cf5585e64344ed9634da9646d18e16 completed 且独立字节/归档成员回读通过。新的 config-role-v6 执行和 v7 独立验收器已冻结，派发 CLI 实时去重、复核五次历史失败及 CPU 空闲状态，未将预检当作完整标定验收。证据：test/recover-before-fuse/receipts/spd-config-role-full-input-and-cloud-source-accepted-20261001.json。

2026-10-01：fold 1 新配置身份源码的完整输入门槛审计已启动，无 GT 拟合。单独命名的 fold 1 派发监听器必须等待 fold 0 完整云端/GT 样例/参数独立验收通过、fold 1 输入门槛通过及实时空闲 L40S CPU 队列；fold 0 failed/stopped 时退出而非重复派发。独立验收监听器已准备，仅验收 completed 任务，不将排队/运行或软件检查认作实验完成。证据：test/recover-before-fuse/receipts/spd-fold1-config-role-calibration-dependencies-started-20261001.json。

2026-10-01：fold 1 修正版完整输入门槛已实际通过：12615 双侧帧及四个分片，GT 未读且未拟合。fold 2/3 config-role-isolated-wheels-v6 执行和 v7 独立验收器已冻结，绑定 64 文件源码任务 38cf5585e64344ed9634da9646d18e16、各自独立元数据与 supervision；嵌入 runner、单 fold 范围及源码任务绑定已核验，未派发，须等待各折完整 fit raw 独立验收与空闲 CPU。证据：test/recover-before-fuse/receipts/spd-fold1-full-input-accepted-fold23-config-role-execution-prepared-20261001.json。

### 2026-10-01 V100 continuation and full calibration independent failure

V100 fold3 fit raw task `2ec998882c734a8aba4141ef17da879e` completed with 18 artifacts; independent full readback remains pending, with no duplicate dispatch. V100 fold3 heldout task `dd6c6d4659aa47d0970ea3fd8f6e20b4` completed and its existing acceptance is preserved. GPU models remain unrestricted subject to runtime compatibility and physical collision checks.

Fold0 calibration task `1fe71b24b1d441e5af1c661f46eba87b` completed with 11 artifacts, but full local independent raw/GT reconstruction failed on `vehicle-side/car/residuals`. Retained failure receipt: `test/recover-before-fuse/receipts/spd-fold0-calibration-full-independent-reconstruction-failure-20261001.json`. No tolerance relaxation or acceptance claim; fold1 dispatch and V2 cache gates remain closed. Separate diagnostic reconstruction is running; overall ETA unknown.

Separately prepared V2 builder configuration-role source correction: `test/recover-before-fuse/source-freezes/spd-canonical-v2-builder-config-role-closure-20261001/source-freeze-receipt.json`. Only admission identity comparison changes; numerical cache generation functions unchanged. Not uploaded or fully admitted yet; independent calibration acceptance remains required.

Fold2 fit raw task `007087273b6c4812a2301e46bee1fa81` now independently accepted: 18 cloud artifacts, 13172 sideframes (vehicle 6805 / infrastructure 6367), actual A100 worker. Receipt SHA256 `cb396f3edde5e7bc55f4d6b21be22bb7e0871e528513499371f5dd1ef442f879`. Full actual input guard started separately; V100 fold3 fit raw independent readback remains running. Fold0 diagnostic v2 launcher fixes a diagnostic-only script generation error, retaining the original failed diagnostic log; it does not alter frozen oracle, producer, tolerance, or acceptance gate. Overall ETA unknown.

Full fold0 diagnostic reconstruction completed all 12723 frames / six groups. Scores, match labels, XYZ/dimensions/yaw residual components are exact-equal; only VXY differs, maximum absolute difference `1.4210854715202004e-14`. Cause remains a cross-runtime floating arithmetic hypothesis. No tolerance relaxation or acceptance. Separate Linux full reconstruction runner and bootstrap prepared with unchanged oracle, original precise comparison and numerical tolerances, full prior 11-artifact byte bindings and original CPU runtime validator. Dispatcher/cloud byte acceptor remain pending; no task dispatched and follow-up gates remain closed.

Separate Linux full reconstruction dispatch started after full prior 11-artifact cloud/local bytes and actual raw-input guard matched, original record/crypto probes and corrected 64-source cloud inventory were rechecked, and a fresh idle L40S CPU listener was observed. Dispatch handle will be persisted before enqueue. Independent cloud readback acceptor requires a separate completed CPU task, original full 12723-frame oracle report, exact equality with original job-local six-group report, unchanged original numeric tolerances, actual runtime RECORD identities and all three new artifact bytes. Original macOS full-exact failure remains false/preserved; no production refit or automatic downstream gate change. Acceptance watcher PID 11852 is running; overall ETA unknown.

V2 builder corrected 64-source cloud asset task `5e1f647dee6e496e825044ca45958afa` completed and all manifest/archive member bytes independently read back. Manifest SHA256 `2a99150e4f224b61d444b61fcbb52a1371bc07e319dbcf77a8f26786e477575e`; acceptance SHA256 `d42149bc550e7789b9cea3380fdcb7e6ca89fd12a6e40718ad1b185e00e6435a`. AST review confirms only `admit_canonical_sources` changed in the two builders; cache construction/numerical verification unchanged. This is source acceptance only. A full fold0 builder-admission watcher (PID 14686) now waits for completed separate Linux full reconstruction plus independently accepted cloud proof before auditing all four accepted heldout raw shards and full input/cohort/calibration bindings. It cannot produce a cache and does not claim formal eligibility.

Separate Linux full calibration reconstruction task `59421fbcfcfb4fda9beb5be9bef280e0` completed and independently accepted all three artifact bytes, exact 12723-frame/six-group raw-GT reconstruction, original parameter tolerance and original calibration artifact identity. Independent acceptance SHA256 `898c183d25f69736ce8b52d85be20b8aaffd31e32c32cf9481fdfd478ae90db2`. This proves the separate admitted Linux full reconstruction; the previous macOS exact failure remains false/preserved.

The first full V2 admission test then failed before payload audit because `audit_spd_oof_complete_raw_cache_raw_pose_v2` was missing from the 64-source bundle. Failure log retained. New separately named 67-source closure adds the unchanged audit, heldout cache primitives and input gate from the independently accepted heldout raw closure, including original cloud-tar byte checks for the two producer helpers. Original 64 files unchanged. Full v2 admission guard PID 17207 is running; new 67-source cloud publication and actual cache production remain pending.

Fold1 unchanged Linux126 CPU calibration dispatcher started after full fold0 independent Linux acceptance and fold1 full actual input guard, without claiming macOS acceptance. The prior frozen dispatcher still checks full raw cloud/local bytes, runtime/source identities, deduplication and idle L40S CPU queue and persists its task handle before enqueue. Dispatch observation pending; original fold1 independent macOS acceptor remains preserved.

Full fold0 V2 builder admission with the separately frozen original 67-file closure passed: all four actual heldout raw shards, corresponding complete input payloads, original raw pose precision, numerical calibration and fit/heldout isolation were audited. New cache production and 67-source cloud publication still pending; this is a full input admission receipt, not cache completion or paper eligibility. Fold1 CPU calibration task `4b5f528c812049e3858e90d4dafa0f6d` was created once, handle persisted before enqueue, enqueue confirmed and observed in progress. Original source/runtime/fit recipe retained.

67-source builder import closure task `dbb9de4e43b34f9f89964818e40a0a98` independently accepted all source bytes/archive members. Manifest SHA256 `db9a467cf7488d0c5b1ec3376fb6258f32c0c1d1f41522404b7a99680d69ba27`, acceptance SHA256 `e49d23c5595be2b5aefe94cbed76b95787706d4df73ea133f5ec0f1359410192`. Original 64 files unchanged; three original heldout audit imports restored.

Local monitor processes were no longer live when execution resumed. Remote experiments were preserved. Fold1 calibration is completed (11 artifacts); unchanged full independent acceptor recovered in a fresh output (PID 34016). V100 fold3 raw independent readback recovered (PID 33731) using a separately frozen transport wrapper: only complete prior files matching current artifact registration size and full SHA256 are linked into a fresh output; missing files are downloaded, original numerical/source/content acceptor and all full audits remain unchanged. Old partial output retained; no remote task resubmission.

Full fold0 CPU V2 runner and cloud bootstrap prepared under `test/recover-before-fuse/source-freezes/spd-canonical-v2-cache-linux126-fold0-cpu-execution-preparation-v1-20261001`. They use original Linux core/crypto RECORD identities and unchanged helper ASTs (verified), bind original 67-source assets and independent full Linux calibration, download only GT-free heldout inputs plus accepted raw/calibration assets, retain every query (3615 sideframes / 3253500 queries), and invoke original full builder/sealed-payload verification. Fit GT assets are not downloaded. Cache experiment dispatcher and independent full cloud-cache acceptor remain pending; no actual cache is built or accepted yet. Phase logs state overall ETA unknown; archive packing has phase-only ETA.

2026-10-01：用户明确要求 V100 同步实验。实时观察 GPU4-V100 两组四卡和重叠八卡 Worker 均空闲；fold2/3 V100 训练及 fold3 导出已 completed，fit raw 全量独立回读继续运行，不重复。fold4 训练 a4ad4a451bd94a79a96cea8f38b083c3 completed；原冻结 byte freezer PID52788 启动，单独冻结 V100 依赖监听 PID53142 等待完整字节冻结后再做实时物理碰撞检查和原始前向派发/独立验收。原生产器及验收器不变，尚未把监听/排队作为实验完成。fold1 本机完整精确重建在 vehicle-side/car/residuals 失败，保留日志和原容差，不宣称验收。ETA 未知。证据：test/recover-before-fuse/receipts/spd-V100-parallel-fold4-forward-continuation-20261001.json。

2026-10-01：fold0 完整 CPU V2 cache 执行器、派发器和独立云端内容验收器均单独冻结。实际范围 3615 双侧帧 / 3253500 queries，禁止裁剪；独立验收复核五项云端产物全部字节、归档全部成员、原 sealed V2 verifier 的全量 payload/schema/cohort 检查及逐帧原 raw/config/checkpoint/calibration 绑定。未宣称独立重新计算数值变换或正式论文准入。派发 CLI session41670 正在进行实时去重及 CPU 依赖核验，handle 尚未观察；独立验收监听 PID61823 等待 completed 后回读，排队/运行不算验收。证据：test/recover-before-fuse/receipts/spd-full-fold0-v2-cache-cpu-execution-and-independent-acceptor-prepared-20261001.json。

2026-10-01：用户要求尽量利用全部空闲 GPU。实时 snapshot 确认 A100 两组四卡、V100 两组四卡、Ubuntu26 后四卡空闲；3090 双四卡 busy，5090 八卡运行任务 a5d295f01a26429bb235a876dfeb3150 覆盖看似空闲的后四卡。已冻结并启动 fold4 heldout / fit full raw 并行依赖监听 PID64939，原生产器/验收器不变，前向独立验收完成后分别实时选队列；GPU 型号不限制，已实际运行同环境的队列优先，物理碰撞及依赖仍是必要门槛。尚未发布 raw 任务。完整 CPU V2 cache 23ec31cee62c4838b42923a0930a4238 从 queued/in_progress 转 failed（0 产物），保留任务/冻结源码，不自动重启，原因仍在核验。证据：test/recover-before-fuse/receipts/spd-maximal-ready-GPU-parallelism-and-CPU-cache-failure-20261001.json。

2026-10-01：CPU cache failed 23ec31cee62c4838b42923a0930a4238 的任务日志独立核对为 NameError: urlunparse 未导入，首次 source-manifest fetch 在 downloader 前失败，0 产物。保留原任务和 v1 源码。单独 import-v2 仅补齐 urllib.parse import，bootstrap 全部函数 AST 一致、runner 字节一致、全量 3615/3253500 范围与数值/容差不变；派发器只接受该任务明确错误/源码/输入绑定的历史失败例外，其余去重仍严格。匹配独立验收器和监听 PID67244 已冻结，新的派发 CLI session80726 进行实时核验；尚未将预检或监听作为实验完成。证据：test/recover-before-fuse/receipts/spd-CPU-cache-urlunparse-first-fetch-failure-and-import-v2-correction-20261001.json。

2026-10-01：fold4 完整两侧 detector byte freeze 验收通过，11 产物重新完整哈希核验，回执 c10172a9146bf67fb6a70d12b4d3dd0e355f0143c1dadc107767372fd9ec75cb。V100 前向监听和双 raw export 并行监听继续接续。fold1 单独 Linux 全量独立 raw/GT 重建 1bbd2940d1584bec9a71f9e1a44707b6 confirmed queued：12615 帧六组、原精确 oracle/参数容差，不重新生产拟合；本机先前失败保留，云端独立验收监听 PID82443。CPU cache import-v2 00f28a406e14429db84c315e21ffda5a failed，0 产物，构建器旧 frame field gate 不接受已冻结生产器 mandatory raw_query_count。新 schema-v3 要求严格四字段及 raw_query_count 为 int 且等于 detections，完整四分片 3615/3253500 manifest census 通过；仅该条件变化，其余所有数值变换语句 AST 一致，完整输入门槛 PID82191 运行。新源码未上传、未再次派发 cache。证据：test/recover-before-fuse/receipts/spd-fold4-byte-freeze-accepted-fold1-Linux-reconstruction-enqueued-20261001.json。

2026-10-01：按用户最大化空闲 GPU 要求，fold4 原前向验收在 V100 完成并独立回读；held-out 全量导出 41098428fe92488e9910e96021187564 在 V100 gpu0–3 completed（17 产物），完整独立回读运行；fit-only 全量导出 6babe58d2a9046a088eccb5c4f0c167a 在 V100 gpu4–7 in_progress。fold3 fit-only 原验收器全量独立回读 13762 帧及四分片输入门槛通过。fold1 Linux 独立全量重建 1bbd2940d1584bec9a71f9e1a44707b6 的 12615 帧六组例子及参数验收通过，未重新生产拟合，保留本机失败。新 full-query schema-v3 的 67 文件云端源码及完整 3615/3253500 输入门槛验收通过；完整 CPU cache af30b1e769854e22b0cb023fe803c525 completed 五产物，正在独立读回，不宣称独立数值变换或正式准入。所有 GPU 型号允许，物理碰撞/依赖/去重仍保持；ETA 无完整任务阶段进度时未知。证据：test/recover-before-fuse/receipts/spd-maximal-ready-GPU-parallelism-full-admissions-20261001.json。

2026-10-01：fold3 的四分片完整 fit raw 输入门槛通过后，原 v6 Linux CPU 校准执行器已派发 8df25947f8374b579b2c2ab45e58afd5（实时 in_progress），原精确完整 GT/参数验收监听 PID91475；不把运行视为验收。fold0 完整 V2 独立数值 oracle 已单独冻结，监听 PID98450 等待五项云端完整内容验收后，逐帧重算全部 3615 帧 / 3253500 queries 的九维状态、分数、9×9 协方差并核对原始字段/128D 外观/dtype；不导入生产器或助手，复用 NumPy 基本算术/log/exp/remainder，使用严格 array_equal、无新容差、不读取 GT。数值验收尚未运行或通过，不宣称正式准入。证据：test/recover-before-fuse/receipts/spd-fold0-full-V2-independent-exact-numeric-oracle-prepared-20261001.json。

2026-10-01：fold1 已通过 Linux 全量独立校准后，启动完整 schema-v3 V2 builder admission PID5563，复核全部四个 held-out raw 分片、67 文件已独立回读源码、fit/held-out 隔离及完整校准绑定；不读取 held-out GT，不把启动作为通过。fold2 原校准执行器的 CPU 空闲派发监听 PID99109 等待 fold3 释放 L40S。full cache 云端回读、fold4 held-out 云端回读、fold3 精确校准验收及完整数值监听均实时进程存活；fold0 sealed archive 下载阶段最近 ETA 274 秒，整体验收 ETA 未知。Goal active，不重启、不重复。证据：test/recover-before-fuse/receipts/spd-full-scope-dependencies-verified-live-fold1-admission-20261001.json。

2026-10-01：fold1 全部四个 held-out raw 分片及完整校准/V2 输入门槛通过，回执 17e4a134ef385c66b7dcc4e818ca40f3d99bc32273b040835554ddfc575b5b3c。单独冻结其完整 CPU cache 执行器和五云端产物/全归档/全查询内容验收器，覆盖 3723 帧 / 3350700 queries；原 builder b9971e6b... 保持不变，所有非 main 函数 AST 不变，更新 fold1 资产绑定，未派发或宣称完成。缓存监听 PID13513，CPU 派发监听 PID13514 等待已入队 fold2 校准 a7cd27d66a9f4adb8b88f7e4147f7a2b 释放槽位后再实时检查空闲，避免多个空闲派发器争用。fold2 原完整校准验收监听 PID13512；fold3 completed 正在独立回读。证据：test/recover-before-fuse/receipts/spd-fold1-full-V2-execution-acceptor-byte-bindings-verified-20261001.json。

2026-10-01：fold0 完整 V2 五项云端产物、全归档和 3615 帧 / 3253500 queries 内容独立验收通过，回执 cbcd00aa2a5346d4cb54c5f19fc00b12d24871faf427c0e3651369986b6c1995。随后本机独立严格数值 oracle 在校准分数失败，原失败保留。全量七字段差异 census 完成：只有 scores 1149691 个值不等，最大绝对差 2.220446049250313e-16，其他状态/协方差/原始分数/class/外观/validity 全一致。全量三角核验显示本机独立公式与原生产器 helper 在全部 3253500 分数上精确一致，二者对 Linux 保存结果同样不等；这与运行环境差异一致，但不能当成已证明具体数学库因果，更不能调整容差或宣称数值准入。后续必须在已验收原 Linux126 环境独立全量精确重算。fold3 本机原完整校准验收 returncode1 保留，启动原 oracle 全量诊断以取得具体差异，未重启生产任务。证据：test/recover-before-fuse/receipts/spd-full-fold0-V2-local-vs-Linux-numeric-triangulation-completed-20261001.json。

2026-10-01：fold0 完整独立 Linux126 数值重建 8779cec07a9741eda69061fcbcfef245 已 confirmed queued，完整 3615 帧 / 3253500 queries，原 core RECORD/三私有 crypto wheel/151 文件身份必须核验；不导入生产器数学助手，不重建 cache、不重新拟合、不放宽 array_equal。原完整 transport 函数 AST 保持一致，独立云端三产物验收器及监听已冻结；queued 不算通过。fold1 cache 82569ca6a6da450b9bf675ff83f07152 completed，旧独立验收器错误保留 fold0 calibration digest，明确定位为单一 binding literal；单独 v4 仅更正到实际 fold1 calibration SHA568914...，完整五云端产物回读运行，不重跑 cache。fold3 原本机完整 GT oracle 诊断明确 vehicle-side/car/residuals 精确差异，失败保留；未宣称仅为运行库差异。证据：test/recover-before-fuse/receipts/spd-fold0-full-independent-Linux-numeric-enqueued-and-acceptor-watch-20261001.json。

### 2026-10-01 全量 Linux 数值重建与 GPU 依赖核验

fold0 独立 Linux CPU 数值任务 `34c6f4f3cdd44df7a35f593871b210ec` completed；三项云端产物独立回读后，全部 3615 帧、3253500 查询的状态、分数、协方差和保留字段精确重建验收通过。回执 `test/recover-before-fuse/artifacts/spd-fold0-full-V2-independent-Linux126-numeric-producer-path-v3-readback-20261001/acceptance-receipt.json`，SHA256 `feaa492297214fbdc717b8669f59235af52b1800d56ec2916729ba9725534fc6`。原 macOS 精确比较失败、缺少 PurePosixPath 的任务失败及旧验收器生产/读回路径比较失败均保留；只单独冻结补导入及显式生产端路径核验版本，没有放宽数值比较。

fold3 完整 Linux 原始预测/GT 校准独立重建任务 `cd0c785bc9fc48fd8cfc345df2891e2f` 已运行，独立验收监视 PID 40924；尚未完成验收，ETA 未知。V100 两个 fold4 全量 GPU 导出任务均 completed，held-out 独立全量验收已通过（SHA256 `ede311f46acf46e91ce3d745ca4a5dedb07e37201fe91d93d470d46a83a38448`），fit 独立回读继续。五折校准/V2 和 OOF 选择冻结尚不完整，不重复已完成 GPU 实验，不提前派发全 46 序列多种子训练。GPU 型号不再限制，实际卡数、环境和物理 Worker 碰撞检查继续生效；5090 八卡任务覆盖表面空闲四卡。完整 V2、正式独立评价和论文性能均未宣称通过。

### 2026-10-01 fold1 缓存内容与 fold3 完整校准独立验收

fold1 全量 V2 云端内容回读通过：任务 `82569ca6a6da450b9bf675ff83f07152`，回执 `test/recover-before-fuse/artifacts/spd-canonical-v2-cache-linux126-fold1-independent-calibration-binding-v4-readback-20261001/acceptance-receipt.json`，SHA256 `7f3b0541ea5c22038d89523319df2e3397333e0f1cdec213db6f1b47246318fd`。其独立数值重建尚缺，内容通过不能替代数值验收。

fold3 完整 Linux 校准独立重建任务 `cd0c785bc9fc48fd8cfc345df2891e2f` completed，三项云端产物独立读回后，13762 fit 帧、六组例子及原参数合同验收通过；没有重新拟合或放宽容差。回执 `test/recover-before-fuse/artifacts/spd-canonical-calibration-linux126-fold3-separate-full-reconstruction-fold-binding-v2-readback-20261001/acceptance-receipt.json`，SHA256 `f001573ea732d2e6bd809991cd2a0b47055cca2baef012fd690002f0fddc16ba`。旧验收器残留 fold1 编号失败保留，单独冻结 fold-binding-v2 修正三个编号检查；原 macOS 失败仍保留。

fold2 原始完整校准 oracle 诊断 PID 45798 正在本机运行，命令及输入哈希保存在 `test/recover-before-fuse/artifacts/spd-fold2-original-full-calibration-oracle-local-failure-diagnosis-20261001/diagnostic-command.json`。独立 Linux 重建 runner/bootstrap 已单独冻结到 `source-freezes/spd-canonical-calibration-linux126-fold2-separate-full-reconstruction-prior-source-v2-20261001`；完整诊断、派发器及云端验收器绑定未结束，未提交新任务。所有正式评价和论文性能门槛仍未通过。

### 2026-10-01 fold2 完整 Linux 独立重建派发

原始本机完整 oracle 已覆盖 13172 帧，在 `vehicle-side/car/residuals` 精确比较处失败；原日志 SHA256 `c195bb9aeee71979411f84636b4b6fee46933687873d554f38dbdcf55a8f2dc1`，保留且未放宽容差。独立 Linux CPU 重建任务 `1a1ab2680d724bed96cbc6693474d20a` 已由冻结派发器核验原任务、11 产物、全量 raw/GT 输入及空闲 L40S CPU 后提交；enqueue 回执位于 `test/recover-before-fuse/artifacts/spd-canonical-calibration-linux126-fold2-separate-full-reconstruction-execution-v3-20261001/dispatch-enqueue-acceptance.json`。独立验收监视 PID 55820；仅 completed 且三项产物独立读回、完整原 oracle 与参数合同通过后才能验收。源码冻结 `source-freezes/spd-canonical-calibration-linux126-fold2-separate-full-reconstruction-execution-v3-20261001`；原拟合没有重启或重做。

fold3 全量 V2 输入准入已启动 PID 56459，使用现有 calibration-venv NumPy/SciPy 运行时；ClearML-only venv 缺 SciPy 的失败留存，新的独立输出位于 `test/recover-before-fuse/artifacts/spd-v2-builder-fold3-full-query-schema-v3-full-admission-calibration-runtime-v2-20261001`。这是完整输入核验，尚未生产或验收 V2 缓存。fold4 fit 原始导出最后一 shard 独立读回仍运行，任务级总体 ETA 未知。

### 2026-10-01 fold3 全量 V2 缓存依赖继续

fold3 完整 builder 输入准入通过（SHA256 `1d8514ede370d53473c593bcf756337068b3ecabf56ffd0a4d176ba1f587e165`），覆盖四个实际 raw 分片及已独立验收原校准。2576 帧、2318400 查询的执行源码单独冻结：runner SHA256 `ee7b6f2a19dde0877c78014a7a12af157c2e138e3d089e62b84baff3d6c2a6b2`，bootstrap SHA256 `737a98ba9a4583d22fcad892ac7b1cbe0b48141a65ffb076d3e1c8c750e019f9`。原 immutable builder `b9971e6b5ea196317fc1c74db575b5c5ba10ded0b375f7b1131cc93f0fccc578` 不变；所有非 main 数值/传输 helper AST 不变。

CPU 依赖协调 PID 63350 已启动，输出 `test/recover-before-fuse/artifacts/spd-fold3-full-V2-cache-idle-CPU-continuation-v1-20261001`：先核验实时空闲 L40S CPU 队列和 Worker，再用冻结派发器去重提交；完成后做五项云端产物、完整 tar 成员及全帧字段/元数据独立读回，不重试失败、不提前宣称完成。精确数值重建另行验收；正式评价和论文性能仍未完成。

### 2026-10-01 fold2 完整校准独立验收通过、fold3 V2 已提交

fold2 Linux 独立重建任务 `1a1ab2680d724bed96cbc6693474d20a` completed，三项云端产物独立读回，13172 帧、六组原始预测/GT 例子及原参数合同通过；回执 `test/recover-before-fuse/artifacts/spd-canonical-calibration-linux126-fold2-separate-full-reconstruction-acceptance-watch-v1-20261001/fold-2-independent-readback/acceptance-receipt.json`，SHA256 `01c776d98324ae2743a73fd2320037a19865534ce0da3dbd9aeaec20baccdd21`。原 macOS 精确残差失败保留，没有重新拟合或放宽容差。

fold3 全量 V2 缓存任务 `a61b5ee37c654ad38e74225e7e1199d1` 已在实时空闲 CPU 门槛后提交，独立回读监视 PID 63350 继续；completed 不代替五产物及全帧验收。fold1 独立 Linux 精确数值 oracle 的 3723 帧、3350700 查询执行源码已冻结到 `source-freezes/spd-fold1-full-V2-independent-Linux126-numeric-reconstruction-v1-20261001`，数值公式字节不变、helper AST 不变；未提交，完整派发器/独立验收器绑定仍需完成。

fold4 V100 fit 全量原始导出 `6babe58d2a9046a088eccb5c4f0c167a` 的全部云端字节、四个完整分片、帧/数组/raw poses 独立验收通过；回执 `test/recover-before-fuse/artifacts/spd-fold4-full-raw-parallel-idle-gpu-continuation-20261001/fit-readback/acceptance-receipt.json`，SHA256 `a3d43b2f5cad3f201fe2904389d493960bb044da49a84944a0d7af880bc90565`。下一依赖为 fold4 完整校准输入准入及拟合；尚未完成完整 V2 或论文验收。

### 2026-10-01 fold4 完整校准输入与执行源码准备

fold4 独立验收 fit raw 后，17 文件校准元数据已按原冻结 prepare/publish 命令发布，ClearML 任务 `0f52c8a17eb741a4a85a00172400861d` completed；manifest SHA256 `413164b84ee360ef44dc82c67e017b64cb7eecaa1908c14fee747eed1c65bee1`，独立字节/归档成员验收回执位于 `test/recover-before-fuse/artifacts/spd-canonical-calibration-linux126-fold4-metadata-full-byte-freeze-20261001/independent-readback/acceptance-receipt.json`，SHA256 `01cbefa3514c937d3d6f887b7a2abcbd88ebd83ba7ef13391cfbd70a2637ff48`。元数据不是校准结果。

完整 fit 输入准入 PID 73535 使用原冻结 fitter，覆盖实际四个分片；总体 ETA 未知。fold4 拟合与完整 job-local raw/GT 重建执行源码已冻结到 `source-freezes/spd-canonical-calibration-linux126-fold4-config-role-isolated-wheels-v1-prepared-20261001`，runner SHA256 `315f42cfa2cd0afd59b4264ffceff80ad13c5ffbf54bfbf45f81c6f853da0ca4`，bootstrap SHA256 `2e259f4c543de2a1a60eff28e191d662a9ad671470230c01685b68d70c4c8d19`，helper AST 不变；完整准入及派发器/独立验收器尚需完成，未派发校准任务。

### 2026-10-01 fold4 全量准入通过与完整校准派发

fold4 完整 fit 准入覆盖 13080 帧（vehicle-side 6853、infrastructure-side 6227）和实际四分片，回执 `test/recover-before-fuse/artifacts/spd-calibration-config-role-fold4-full-input-guard-check-20261001/acceptance-receipt.json`，SHA256 `0b7a13a35a7107f5b6470cb3452dcc942796a1261b992cb41d12330ef653b0bb`。原 frozen fitter 及所有实际 checkpoint/config/raw payload 绑定已核验。

完整 fold4 校准执行源码冻结 `source-freezes/spd-canonical-calibration-linux126-fold4-config-role-execution-v2-20261001`，派发会核验原 64 文件源码、接受的 Linux/私有 wheel 环境、fold4 元数据、完整 fit raw 和实时空闲 L40S CPU，并按配置及种子去重。派发命令输出 `test/recover-before-fuse/artifacts/spd-canonical-calibration-linux126-fold4-config-role-execution-v2-20261001-dispatch.log`；执行 handle 在创建后、enqueue 前写入同名 artifacts 目录的 `dispatch.json`。独立验收监视 PID 83651，读取全部 11 个云端产物并运行原完整 raw/GT oracle；源码 `source-freezes/spd-canonical-calibration-linux126-fold4-independent-acceptor-supervision-binding-v2-20261001`，首次准备时残留的 fold3 supervision 字典键已在单独冻结 v2 中修正，数值合同不变。排队、running 或 completed 均不替代此验收。

fold3 缓存任务 `a61b5ee37c654ad38e74225e7e1199d1` completed，五产物独立全量读回仍运行，未宣称验收通过。

fold4 完整校准实际任务 `75b4ea3549484aac9402228550555d7a` 已 enqueue 确认并运行；独立验收未完成，ETA 按任务阶段日志更新，总体未知。未重复现有任务，未提交或推送代码。

### 2026-10-01 fold1 完整独立数值重建合同与依赖派发

fold1 全量精确数值重建的 frozen dispatcher/acceptor 已补齐：`source-freezes/spd-fold1-full-V2-independent-Linux126-numeric-dispatch-v1-20261001`、`source-freezes/spd-fold1-full-V2-independent-Linux126-numeric-acceptor-v1-20261001`。只接受原缓存任务 `82569ca6a6da450b9bf675ff83f07152` 的完整 cloud 内容验收、原 bootstrap 和五项输入文本绑定；数值 oracle 覆盖 3723 帧、3350700 查询，原公式字节不变、`array_equal` 不放宽、无 producer math 导入、不重新生产缓存或拟合。fold0 的本机失败不被虚构为 fold1 失败，原记录全部保留。

依赖派发监视 PID 90408，源码 `source-freezes/spd-fold1-full-V2-independent-numeric-idle-CPU-continuation-v1-20261001`，输出 `test/recover-before-fuse/artifacts/spd-fold1-full-V2-independent-numeric-idle-CPU-continuation-v1-20261001`。实时 L40S CPU 空闲门槛后去重提交，任务 handle 在 enqueue 前保存；完成后独立读回三项证明产物，不自动重试失败。仅监视启动不算实验完成。

### 2026-10-01 全量缓存验收与空闲 GPU 依赖核验

fold3 全量缓存任务 `a61b5ee37c654ad38e74225e7e1199d1` 五产物独立读回已通过，回执 SHA256 `781bd0f899a95e3f6a1b0847f14ecf8a67a03473d7d1a8d0a1e1dc7e26d7a97d`；尚不等于独立数值重建通过。fold4 全量校准任务 `75b4ea3549484aac9402228550555d7a` 已 completed、11 产物，原始独立验收继续运行。fold2 全量 V2 输入准入 PID 3064，命令与冻结 SHA 保存于 `test/recover-before-fuse/receipts/fold2-full-v2-input-admission-launch-20261001.json`。fold1 独立数值重建监视器 PID 90408 正派发 CPU 任务。

实时 Worker 快照保存在 `test/recover-before-fuse/receipts/idle-gpu-dependency-audit-20261001T0647.json`。A100、V100 有空闲物理卡；5090 八卡 Worker 运行中，四卡空闲显示不能避开物理碰撞。后续 GPU 型号不限制，但已完成的五折检测器和原始导出不重跑；下一轮 GPU 训练仍需全量 OOF 数值验收及选择冻结。当前总 ETA 未知。

### 2026-10-01 fold2 全量缓存派发及 fold1 数值重建绑定修复

fold2 缓存任务 `413f3ca30de44817b7dac2f67273e126` 已开始运行，3,166 帧、2,849,400 个查询，生产完成后仍须五产物独立字节读回及另行数值重建。执行 source freeze、命令、PID 和哈希见 `test/recover-before-fuse/receipts/fold2-full-cache-execution-launch-20261001.json`；原始 builder、运行时合同和数学不变。

fold1 独立数值任务 `0f8ff160522b4916b22b3fe6e07cb0f8` failed、零产物，错误为 `full original cloud content acceptance differs`，在数值比较开始前被引导脚本中残留的 fold0 内容验收哈希拒绝。保留失败任务和全部冻结源码；单独 content-binding-v2 只修正一个 bootstrap SHA 绑定，独立数值 runner 字节完全不变。修正后的空闲 CPU 派发与验收监视器 PID 10821，命令和新 bootstrap 哈希见 `test/recover-before-fuse/receipts/fold1-independent-numeric-content-binding-v2-launch-20261001.json`；尚未验收。总 ETA 未知。

fold4 原始 macOS 独立校准验收失败：`vehicle-side/car/residuals` 不满足严格一致性。失败日志和哈希保存在 `test/recover-before-fuse/receipts/spd-fold4-full-calibration-original-MAC-oracle-failure-20261001.json`，保留原生产任务、11 产物和验收失败；原因尚未证明。下一步另行在冻结 Linux126 运行时执行完整 13,080 帧原始 GT 重建并独立回读，不修改拟合、阈值或原始产物。

### 2026-10-01 fold4 完整 Linux 校准重建准备

原始任务 `75b4ea3549484aac9402228550555d7a` 全部 11 产物已逐项核对 ClearML 注册大小、哈希与本地字节，清单在 `test/recover-before-fuse/receipts/fold4-original-calibration-11-artifact-byte-inventory-20261001.json`。独立执行与验收器分别冻结于 `source-freezes/spd-canonical-calibration-linux126-fold4-separate-full-reconstruction-execution-v1-20261001` 和对应 acceptor-v1；完整 13,080 帧 GT 重建调用原始冻结 oracle，不重新拟合，不放宽数值阈值。原始 macOS 失败保留。

空闲 CPU 派发与完成后独立读回监视器 PID 17956 已启动，命令和源 SHA 在 `test/recover-before-fuse/receipts/fold4-full-Linux-calibration-reconstruction-launch-20261001.json`。尚未验收，ETA 未知。

### 2026-10-01 fold1 全量严格数值验收通过及 fold3 重建准备

fold1 独立任务 `4dfcb46cc8e0489faa055aba7d0c5d76` completed，三个证明产物独立云读回、注册及本地字节复核通过。完整 3,723 帧、3,350,700 个查询的状态、分数、协方差与保留原始字段均严格 `array_equal`，未导入生产数学模块、未放宽阈值。验收回执 `test/recover-before-fuse/artifacts/spd-fold1-full-V2-independent-numeric-idle-CPU-content-binding-v2-20261001/independent-readback/acceptance-receipt.json`，SHA256 `cd8eb66a97528296ca39687e9026130bab14c23f12504bd3371aa128b6684f11`。初次绑定失败仍保留。

fold4 完整 Linux 校准重建任务 `df1228533d6f4d7fb270ed370409ea1e` 已运行，尚待独立验收。fold3 完整独立数值重建（2,576 帧、2,318,400 个查询）单独冻结执行、派发及验收器，逐查询数值代码与 fold1 完全相同；空闲 CPU 监视器 PID 24594 已启动，命令及哈希见 `test/recover-before-fuse/receipts/fold3-full-independent-numeric-execution-launch-20261001.json`。尚未验收，ETA 未知。

### 2026-10-01 fold2 全量数值重建依赖准备

完整云缓存读回进程 PID 17554 存活，正在接收 630,027,212 字节封存归档，不因等待重启。后续数值重建准备器仅在完整独立读回通过、注册与本地全部产物大小/哈希匹配后生成并冻结 fold2 绑定、派发和验收源码，覆盖 3,166 帧、2,849,400 查询；逐查询原数学代码保持字节一致，ETA 未知。

准备器 v1 的日志/生成 JSON 换行转义发现错误，在只等待、尚未生成数值执行器或创建云任务阶段停止 PID 31790，保留源码和输出。单独 newline-v2 只修正输出换行并启动 PID 32139；命令/哈希见 `test/recover-before-fuse/receipts/fold2-full-independent-numeric-preparer-newline-v2-launch-20261001.json`，无远端任务重启、未放宽验收门槛。

fold4 独立 Linux 重建任务 `df1228533d6f4d7fb270ed370409ea1e` 实时核验为 failed、零产物，`prior calibration bytes differ`，原因是 bootstrap 残留 fold2 校准 SHA，尚未进入 GT 数值重建。保留失败任务与原 source freeze。另行 calibration-binding-v2 修正单一校准哈希为 `9ffd329d2b3ef3f149a8edec835260b08b46046e3d2690ec8c529c9a632d91ed`，runner 完全不变；派发器新增原失败身份/错误核验，监视日志输出换行修正，无阈值放宽。监视器 PID 32879，命令及哈希见 `test/recover-before-fuse/receipts/fold4-full-Linux-reconstruction-calibration-binding-v2-launch-20261001.json`。尚未验收，总 ETA 未知。

### 2026-10-01 fold2 缓存独立读回和 fold3 精确数值验收通过

fold2 缓存任务 `413f3ca30de44817b7dac2f67273e126` 全部五项云产物已完整读回，注册大小、哈希及本地字节再次复核一致；3,166 帧、2,849,400 查询，回执 SHA256 `e8fc359a15f29140cfd71522d85fc302acd853cc4247c35bc02b8c89da26db3a`。这仍不等于独立数值重建通过。依赖准备器随后冻结原始数学不变的完整 fold2 数值执行/派发/验收器，并启动空闲 CPU 监视器 PID 32599，源码及命令在 `test/recover-before-fuse/artifacts/spd-fold2-full-independent-numeric-after-content-acceptance-newline-v2-20261001/launch-receipt.json`。

fold3 独立数值任务 `3136639422fe47fdb7821c6def914234` completed，三项云证明与本地字节再次复核；完整 2,576 帧、2,318,400 查询严格状态、分数、协方差和原始保留字段重建验收通过，无生产数学模块导入、无阈值放宽。回执 `test/recover-before-fuse/artifacts/spd-fold3-full-V2-independent-numeric-idle-CPU-continuation-v1-20261001/independent-readback/acceptance-receipt.json`，SHA256 `a08693e53404df7b3b3d46057b062291c08c37a46cdfb562375d0f0c81f8528f`。正式五折 V2 与论文评价门槛仍关闭，ETA 未知。

### 2026-10-01 fold4 全量 V2 输入准入准备

修正后的完整 Linux 校准重建任务 `9c89e47443ff41bc9301380d2b61fe7a` 已运行。全量 V2 builder 输入准入单独冻结于 `source-freezes/spd-v2-builder-fold4-full-query-schema-v3-full-admission-calibration-runtime-v1-20261001`，PID 36106；仅在该任务完整独立回读通过、三项证明字节核验及 13,080 帧原 GT 重建通过后，调用不变的 67 源文件 builder 审计实际四个 heldout raw 分片。fold4 heldout 原验收回执四分片合计 3,258 帧、2,932,200 查询。命令、源 SHA、日志在 `test/recover-before-fuse/receipts/fold4-full-v2-builder-admission-launch-20261001.json`。准入和缓存生产均未完成，总 ETA 未知。

### 2026-10-01 fold4 完整执行绑定修复

fold4 独立任务 `9c89e47443ff41bc9301380d2b61fe7a` failed、零产物，runner 中仍有 fold2 校准哈希，GT 数值重建未开始；保留该失败、前次 bootstrap 失败及原 macOS oracle 失败。单独 complete-binding-v3 修正 runner 校准 SHA，重新绑定 bootstrap 内嵌 runner 字节及调度/验收 SHA；核查整个源码无旧折号/帧数/校准哈希，并确认嵌入 runner 与冻结文件完全一致。原始 oracle/fitter 和阈值不变。监视 PID 45768，回执 `test/recover-before-fuse/receipts/fold4-full-Linux-reconstruction-complete-binding-v3-launch-20261001.json`。

原 fold4 输入准入因上游任务失败而关闭，保留其输出。新的 dispatch-binding-v2 准入 PID 46095 只绑定 complete-v3 bootstrap 的实际任务句柄与完整独立验收，仍不宣称缓存完成；回执 `test/recover-before-fuse/receipts/fold4-full-v2-builder-admission-dispatch-binding-v2-launch-20261001.json`。fold2 独立数值任务 `965b076cd14c4807b3958639a969dc0c` 已运行，尚待三项证明独立读回。总 ETA 未知。

fold2 完整独立数值任务 `965b076cd14c4807b3958639a969dc0c` completed，三个证明产物独立回读并再次核对注册及本地大小/哈希通过；3,166 帧、2,849,400 查询严格数值重建通过，未放宽阈值。回执 `test/recover-before-fuse/artifacts/spd-fold2-full-V2-independent-numeric-idle-CPU-continuation-v1-20261001/independent-readback/acceptance-receipt.json`，SHA256 `e0c6037624b6613e28f94a534b2897e5894bbfed2e4b62618e3796e65a1369d5`。现 fold0–3 的完整独立数值重建已验收，fold4 仍缺；complete-v3 校准重建任务 `71915c3a5d144cae85972bf4e46bed5a` 已派发，尚不代表完成。

### 2026-10-01 最后一折全量缓存依赖准备

fold4 完整校准重建任务 `71915c3a5d144cae85972bf4e46bed5a` 实时仍 in_progress，尚无产物。校准完整独立读回与 builder 四分片实际准入均通过后，单独准备器将基于已冻结 fold2 模板绑定实际 fold4 原始预测、校准、元数据、全部证明和哈希，冻结执行/派发/验收器并在空闲 L40S CPU 上派发 3,258 帧、2,932,200 查询全量缓存。原始 builder 和阈值不变，不重复 fold0–3。准备器 PID 52772，命令、日志及源 SHA 见 `test/recover-before-fuse/receipts/fold4-full-cache-after-admission-preparer-launch-20261001.json`；缓存未生产、未验收，ETA 未知。

### 2026-10-01 fold4 缓存后的完整严格数值重建准备

校准重建任务 `71915c3a5d144cae85972bf4e46bed5a` 实时日志已进入原始完整样例和参数独立重建阶段，尚未验收。最后一折数值依赖准备器 PID 59191 仅在实际 fold4 缓存完整五项云产物独立读回通过、全部云注册/本地字节大小与哈希一致后，冻结基于原始数学的独立执行/派发/验收源码。覆盖 3,258 帧、2,932,200 查询，严格 `array_equal`，无生产数学模块导入、无阈值放宽。命令与源 SHA 在 `test/recover-before-fuse/receipts/fold4-full-independent-numeric-preparer-launch-20261001.json`；缓存与数值重建尚未完成，ETA 未知。

### 2026-10-01 五折完整独立校准重建完成

fold4 任务 `71915c3a5d144cae85972bf4e46bed5a` completed，三项云证明独立回读及注册/本地字节再次核验通过，完整 13,080 帧原 GT 样例与原参数严格重建通过；回执 SHA256 `ea653eff74c4dae110cc0e0539e8504e18a81e21a115d6f4cb49b97f9bbfc58f`。未重新拟合、未放宽阈值，所有历史失败保留。至此五折该校准重建项已验收，仍非完整正式 V2 或论文评价。

fold4 输入准入因本地目录日期被广泛折号替换误改为 `fivefold-40260930` 失败，真实资产仍在 `fivefold-20260930`。保留失败源/日志；单独 date-path-v3 修正路径。缓存生成适配器新增日期修复和只匹配完整折号的安全检查；数值生成适配器修正仍指向 fold2 的整数常量，尚未生成任何数值执行器或创建云任务。只停止并保留等待中的本地准备器，原云任务均未重启。新 PID：准入65745、缓存准备66012、数值准备65747；命令/哈希见 `test/recover-before-fuse/receipts/fold4-local-date-path-and-numeric-fold-constant-corrections-20261001.json` 及 `fold4-full-cache-date-safe-census-preparer-v3-launch-20261001.json`。数学、校准及阈值不变，ETA 未知。

fold4 全量 builder 输入准入已通过，四个实际 raw 分片完整审计，回执 SHA256 `08b0f3584ad4883eef68c85891db2d01b292169bce2ccaa973283fd45b8c6742`。实际执行/派发/验收器已冻结，目录日期及折号再次核查通过，bootstrap 内嵌 runner 与文件字节一致。缓存任务 `802ed26d42f341bfa49183854ad3f5d6` 已派发并运行，覆盖 3,258 帧、2,932,200 原始 900-query 查询；完成后仍需五项产物独立完整回读及另行严格数值重建，不能视为正式 V2 完成。命令和源 SHA 在 `test/recover-before-fuse/artifacts/spd-fold4-full-V2-cache-after-admission-date-safe-census-v3-preparer-20261001/launch-receipt.json`，派发句柄在 `test/recover-before-fuse/artifacts/spd-fold4-full-V2-cache-idle-CPU-continuation-v1-20261001/dispatch.json`。ETA 未知。

### 2026-10-01 fold4 缓存完成及回执折号绑定修正

缓存任务 `802ed26d42f341bfa49183854ad3f5d6` completed，五项产物已注册，完整独立读回仍进行中。源码检查发现原验收器仅最终结果回执保留 `fold_id=3`，此前输入、运行合同与范围均检查 fold4；保留原源码及读回记录，停止尚在等待的数值准备器 PID65747，未重启云任务。单独 receipt-binding-v2 验收器绑定 fold4，重新核验云注册/保留下载字节，并完整展开、审计所有缓存内容；原数学和阈值不变。依赖监视器 PID75192 等待原读回结束后执行修正验收，成功后才启动增加 `fold_id==4` 门槛的数值准备器。命令、源 SHA 和 PID 在 `test/recover-before-fuse/artifacts/spd-fold4-cache-receipt-binding-v2-dependent-continuation-20261001/launch-receipt.json`。当前未宣称缓存独立验收或数值重建完成，ETA 未知。

实时 Worker 显示 A100/V100 空闲，3090 与 5090 存在实际重叠占用；不能依据八卡或四卡 Worker 的空闲表象跳过物理碰撞检查。五折检测器及 raw 导出已完成，不重复占卡；完整46序列训练仍等待完整 pooled OOF 选择冻结。

### 2026-10-01 canonical D4 GPU 输入路径审计

检查 thesis D4 与现有训练入口：`train_association.py` 默认每序列2帧 GT 开发 canary，其输入由单端 GT 标签几何和 GT 身份历史构造，并明确 formal/ranking 不合格；`learned_association.py` 已有 pairwise BCE 与双向 assignment 原语，但并不证明真实 canonical predicted-cache 数据或训练完成。`predicted_association_v2.py` 固定旧 full-train checkpoint/calibration/cohort，不能重标为新五折结果；`train_paper_identity.py` 的 row-surrogate 也不代替配对身份风险 VoI。下一批 GPU 关联训练须先构造各折 complementary-fit-only 预测特征/隔离 GT 标签/gate 内 hard negatives，后续完整配对 VoI 和144-grid raw-cell OOF 选择才允许 full46 重训；不能在缓存完成后直接跳到 full46。原代码/失败未修改。文件哈希与具体观察在 `test/recover-before-fuse/receipts/spd-canonical-oof-D4-next-GPU-source-transition-audit-20261001.json`；该审计不是训练或论文验收。

### 2026-10-01 canonical 预测特征准入组件

新增 `transvision/models/event_track_v2x/canonical_oof_predicted_features.py`，候选仅复用冻结203维 encoder 和 V2 gravity-state/class covariance 数学；校准字节/fold/fit-held隔离、GT-free raw数组与公共元数据白名单、明确fit/held角色、决策时刻可用时间均检查。四个已验收fold0 fit raw分片各一个真实900-query帧，核对原注册数组/元数据字节后，选中预测特征与现有 encoder 严格逐值一致；16个角色/未来时间/GT字段拒绝检查通过。冻结副本在 `test/recover-before-fuse/source-freezes/spd-canonical-oof-prediction-only-feature-adapter-candidate-20261001`，验收范围及源哈希在 `receipts/spd-canonical-oof-predicted-feature-adapter-real-input-checks-20261001.json`。该组件不接收GT标签，不创建监督映射、全量训练集或权重，不宣称canonical D4完成。下一项仍需真实跨主体身份标签映射与gate内hard negatives的独立证据。

### 2026-10-01 五折全量缓存内容读回完成

fold4 实际缓存任务 `802ed26d42f341bfa49183854ad3f5d6` 的完整五产物/所有封存文件/数组内容独立读回已通过，修正fold_id=4回执 SHA `799d3548038f668cce4dbf9f02783b26cc5986d1aa1072ea92302a1e2055caef`，覆盖3,258帧/2,932,200查询；原fold_id=3回执及源码保留，不作为合格fold4证明。至此五折缓存内容读回齐全，fold4严格数值重建仍未验收。

自动生成数值派发器因仍指向旧回执目录而在创建Task前失败。单独content-path-v2仅修正本地派发输入路径并增加fold_id==4门槛，原远端runner/bootstrap和验收器不变，失败日志保留。新监视PID94325，严格数值任务 `8841ca17073243f1adf6153de7dff94f` 已queued；排队不代表完成。命令/哈希在 `test/recover-before-fuse/receipts/fold4-cache-full-content-accepted-numeric-content-path-v2-launch-20261001.json`，ETA未知。

后续实时核验：五个缓存产物注册及本地大小/SHA再次一致；数值任务8841ca17073243f1adf6153de7dff94f已in_progress、零产物，仍未独立验收。

### 2026-10-01 五折完整缓存数值验收齐全

fold4数值任务 `8841ca17073243f1adf6153de7dff94f` completed，三项云注册和本地证明字节再次核验通过，完整3,258帧/2,932,200查询精确重建，无阈值放宽。独立回执SHA `103d8937ddf0ba2362e3cbcefb5d7390fde8eb84a7761db6c630cbd15f1403d4`。五折内容与严格数值回执现已汇总，16,338主体帧/14,704,200查询，manifest帧身份全46序列恰好held-out一次；分别生成同根 vehicle-only mask 引用原frame记录，未再跑检测器或复制数组。汇总索引 `test/recover-before-fuse/artifacts/spd-canonical-oof-fivefold-content-numeric-index-and-same-root-mask-20261001/acceptance-index.json`，SHA `37b2ba0369972223e54e214d0b9232fc25397db63a91e36da352d09f827b6f44`；mask实际consumer replay未验收，D1官方绑定/D3实测trace/D4完整选择/正式论文资格仍未完成。

单端校准PKL只给source-local ID，不能靠两侧ID相同或几何临近生成跨主体正标签。在ClearML Dataset691397743f934284b9419582adcce0f6已有独立保留的V2X-Seq-SPD.zip中找到cooperative元数据及标签，本机全ZIP再次SHA匹配10bbf7130114e693cda8c99c6f3e0f88949b2224be9725a268cd5ee3825c97c1。新的fit-only身份映射提取在读取888个完整fit pair后严格失败：sequence0004、vehicle001143、infra001185的veh_track_id090341/token7026d28f-face-3b47-9c59-50a7d1dca4a7缺单端引用。失败输出/源码保留；只读fold0 fit标签且逐字节绑定已有fit输入。后续单独quality census PID10946记录完整fit内引用缺项，不修改标签、不接纳训练监督，回执在 `receipts/spd-canonical-oof-fit-identity-first-reference-inconsistency-20261001.json` 及census-launch。该资产绑定仍不是官方发布身份的证明，不能提高D1资格。

完整五折fit身份引用quality census完成，清点配对数分别5721/5769/6082/6338/5870；引用问题15/158/158/157/160，native重复局部ID帧4/136/134/135/135。问题按sequence/vehicle/infra与原SHA跨fold去重保存到 `receipts/spd-native-fit-reference-quality-fivefold-deduplicated-issues-20261001.json`，重复不能当独立样本计数。该census不输出可训练正/负标签；重复局部ID须进一步用原始annotation-token tuple逐项核验，缺失/冲突不靠几何推定。保留首次严格映射失败及首轮census duplicate-ID失败。

### 2026-10-01 原始 annotation tuple 映射恢复及五折独立核验

此前162处“引用问题”已逐项对原始native文件核验：137处是重复局部track ID下的唯一annotation-token tuple；25处token也唯一存在，所有局部ID经ASCII十进制整数表示验证相等，差异仅前导零。不是原始标签缺失，不按几何或同名推定对应。新v2提取器要求数值局部ID相等且annotation token唯一、同一pair的cooperative物理ID唯一，保留raw native/cooperative ID字符串及所有原始ZIP/失败，不改写任何标签。分类回执SHA `d93b8d9f58dd20141844410c39c39b28c7bfbdc099726e0453bcaf4e428f298e`。

五折全fit annotation tuple 映射已完成，配对5721/5769/6082/6338/5870、双端正链接26930/26009/28972/29983/29582。第二进程逐pair重开原始cooperative JSON，并由冻结loader核对独立转换的fit-only PKL注释token/数值ID；原始绑定与转换覆盖全部一致，五折回执SHA `8b262a694a806c19326fa3b2c11e9e224ae07db1e62d4bc83731053bcfe18606`，路径 `test/recover-before-fuse/artifacts/spd-canonical-oof-fit-annotation-tuple-independent-readback-20261001/acceptance-receipt.json`。这是监督来源链接验收，不代表预测匹配、完整训练数据、GT-free在线特征或论文资格全部完成。

新增离线 `canonical_oof_association_targets.py`，与预测特征接口隔离：未知cooperative身份不作为负样本，有未知gate内备选时不强制dustbin assignment；输出pairwise/左右assignment的独立监督mask。四个真实fit配对覆盖前导零和重复局部ID案例，绑定已冻结203维预测特征及原2m类内GT匹配；双向one-to-one、unknown掩码、empty-side和重复GT匹配拒绝检查通过。该组件测试使用all-true gate fixture，不当作真实训练gate验收；全量数据和GPU权重尚未生成。回执 `receipts/spd-canonical-oof-association-targets-real-fit-input-checks-20261001.json`，SHA `e3a12d02d6691148ef7e1c343bb1c64cd8202e2ce52bfe8adc20c3e536395fbe`。下一项为真实预测/位姿/时间驱动gate与全量配对数据准入，之后才派发五折GPU关联头。

## 2026-10-01 真实预测 gate 与关联 fit 数据构建

预测-only 世界坐标/共同时间传播、保守 2(P+R) 三维 gate 的真实四对输入检查通过；0.90/0.95/0.99 嵌套与标量独立重算通过。五折 v1 因车端图像晚于点云 100 ms 截止时间停止，失败及部分输出保留。单独命名 arrival-v2 保留未按时到达观测的空样本，不延长截止时间；五折并行构建，已完成的构建立即启动独立全量读回。所有命令、源哈希、进程及路径存于 receipts 对应 launch 回执。构建和运行中不等于验收，hard-negative 场景完整覆盖、GPU 关联头训练及论文资格仍未宣称。

五折 arrival-v2 全量独立验收通过：源字节、203 特征、真实保守 gate、离线标签、未知标签掩码、双向 assignment 与 100 ms 可用性全部重算。独立验收索引见 receipts/spd-canonical-oof-fivefold-real-gate-fit-examples-independent-acceptance-index-20261001.json。五个 ClearML 数据/源码上传任务已创建，尚在上传及独立云读回阶段；源码发布不等于 GPU 实验。冻结四卡 DDP 训练结构与原 203 特征模型 AST 一致，固定候选 recipe（seed1337/24 epoch/全局 batch16/AdamW3e-4），masked BCE + 双向 assignment CE，绑定 fold/cohort/配置/数据验收与 checkpoint。需云字节验收后才派发；独立 watcher 每15秒检查，最长1800秒，派发前实时检查物理 GPU 交集。运行环境、GPU 损失 preflight、实际训练及结果验收仍待执行；hard-negative 场景完整覆盖及论文资格仍未宣称。

上传与派发的独立折依赖更新：v1 的全五折云读回屏障只是调度实现，不是实验协议要求。该 watcher 在派发任何 GPU 任务前已退出，源与日志保留；单独命名的 v2 每折完成自身云读回即派发，无需等待其他折。v2 派发日志追加保留所有 waiting/created/queued 记录，通过配置与任务 ID 去重，失败任务不自动重启，物理 GPU 交集每次实时核验。fold3 云字节及全部归档成员已验收；其余四折上传/读回中。

## 2026-10-01 GPU 真实训练与 car-only 范围修正

Torch 2.6.0+cu124 / NumPy 1.26.4 / Python 3.12.3、固定镜像与 NCCL_P2P_DISABLE=1 已通过真实四卡 forward/backward/全局损失 probe，并完成独立产物回读。旧运行时/文件地址/NCCL 失败保留，不重启。通用执行源码 v2 下的五个 all-class 关联诊断训练已 completed，24 epoch 日志、五产物字节和实际设备均独立回读；因 thesis experiment-preregistration.md 第377行要求主表、补充与后续改进均 car-only，全部封存为历史诊断，不进入 car-only 实验验收，尚未宣称 tensor/forward 验收。见 receipts/spd-canonical-oof-fivefold-all-class-training-scope-exclusion-20261001.json。

单独命名 car-only-v3 对原冻结 raw-score>=0.05/all-class-top64 候选做 class0 训练视图，不更改缓存或原始标签文件；双向 assignment/dustbin 索引重新映射，全五折逐样本独立核对通过。执行源码 ClearML 任务 a634e645001446b1b758a870d5430f29 完成且五产物全部字节读回。固定 seed1337/24 epoch recipe 无 held-out GT 或 checkpoint selection。

实际新任务：fold0 313f544fffba4097a4eedfd5c15e28fc、fold1 c2654a4beead469b913f2a1a6c969dc8（A100）；fold2 fd7b30a1975341cc9cfc362597425246、fold3 1048bcde7452403b81af67d58c980dbf（V100）；fold4 831ff985d9754745a14c30b70667eb96（GPU4-2080ti）。fold4 原创建阶段 V100 容量变化被二次碰撞检查拒绝，原 Task 保留并经重新实时核验转队列，无重复创建。每任务四卡，共20卡申请，设备最终以 runtime 产物为准。任务级 ETA 写入训练 JSON 日志；排队或输入下载无进度时未知。完成后仍须 checkpoint tensor/forward 独立验收，再推进 held-out GT-free 关联、VoI 与同资源基线。hard-negative 场景认证、144 pooled OOF选择、实测网络trace、正式独立评价和论文性能仍未宣称完成。详见 receipts/20260928-execution-ledger.json。

car-only fold0–3 已 completed/24epoch 且完成五产物全字节冻结；fold4仍运行、无任务级进度时 ETA 未知。独立 tensor/真实fit特征 forward 验收已分别入队：fold0 f4505a4811774fd0bd1d296726466689、fold1 df46584c0b2a4e738f178652251dc803（A100），fold2 5128d902d14b459fb2b11d77a7b2ea07、fold3 d74fb0902cfe4a2ab9c839b9ca7f237e（V100）。新验收器绑定已读回checkpoint哈希/训练Task/fit特征来源，检查所有tensor键/shape/dtype/finite、严格state loading、独立functional forward等价、空端与四卡数值一致性；尚待真实执行完成及独立receipt字节读回，入队不代表通过。journal: receipts/spd-canonical-oof-car-association-fourfold-independent-GPU-forward-dispatch-20261001.json。

最新验收：car-only五折均completed/24epoch/五产物全字节冻结；fold0–3 tensor及真实fit manual-functional forward/空端/四卡一致性已真实完成、独立回执读回通过，索引 receipts/spd-canonical-oof-car-association-fourfold-GPU-forward-acceptance-index-20261001.json。fold4验收Task 0da6a822d2a342069548d6301b975e1f 已入GPU4-A100队列，尚不宣称通过；source与checkpoint字节绑定，实际设备以最终回执为准。下一项为完整held-out GT-free关联/VoI因果标签与训练及等资源基线接通；当前训练/前向验收不等于正式评价、hard-negative场景认证或论文性能完成。

fold4独立验收0da6a822d2a342069548d6301b975e1f现已completed；注册source和完整receipt字节独立接受，真实A100四卡tensor/fit-functional forward/空端/四卡一致性均通过。receipt SHA02c94929339f898abdc5d844f07612b37770b620bcd4e8359e79dc615727c16c。至此五折car-only固定24epoch关联头训练、checkpoint字节冻结与GPU tensor/forward准入均完成；后续完整关联推断/VoI/同资源基线与论文指标仍待接通。

## 2026-10-01 完整held-out关联输入及带gate的Top-H接通

公开cooperative/data_info四个帧/序列字段与冻结raw metadata全量清点：五折公开配对1724/1676/1363/1107/1575，共7445；完整车端参考帧1808/1937/1699/1409/1651，共8504，存在1059个未配对车端和389个未配对路端。无配对缺失raw identity。未读annotation文件。公开配对子集不能冒充完整输入。

单独命名full-heldout-car-association-input-v1覆盖全部8504车端参考事件：无公开配对则保留右端空/unmatched上下文；车端在box+100ms不完整时不使用任一端特征，未延长期限。另保留全部16338源帧的source-events清单，389个未配对路端事件明确留给完整tracker/channel consumer，没有伪造配对或丢弃它们。五折构建及第二进程全量重算均exit0：每NPZ字节、203编码、原始query索引、car范围、100ms到达、共同世界时间状态与保守2(P+R)三维gate距离完全一致；原producer特征/gate助手不被独立verifier调用。汇总 receipts/spd-canonical-oof-full-heldout-car-association-input-independent-acceptance-index-20261001.json；这是推断输入准入，未宣称完整channel tracking replay或论文指标。

新增canonical_oof_association_hypotheses.py带明确bool几何gate，保留Top-H(1/3/5)联合one-to-one和全unmatched备选；72个小模型由独立穷举assignment能量验证best5/forbidden gate/显式null/归一化与empty-side。冻结source-freezes/spd-canonical-oof-gate-aware-TopH-small-model-checked-v1-20261001，检查回执 receipts/spd-canonical-oof-gate-aware-TopH-small-model-exhaustive-check-20261001.json。权重只表示保留能量候选的截断分布，不冒充校准全后验或Stage2漏质量证书。下一项是已验收输入的云字节发布与实际GPU checkpoint绑定的全量logits/多gate Top-H推断，再独立全量读回；尚未派发新推断，不宣称GPU推断完成。

已授权云资产发布进入真实执行：五折full held-out car input每折单独manifest与完整archive，5个独立publisher进程PID72029/72030/72031/72032/72033，exec session41577已poll确认live；构建archive→注册两artifact→完整云bytes和每archive member独立重算。命令/输出log/Task ID保存于receipts/spd-canonical-oof-heldout-car-association-cloud-input-publication-launch-20261001.json及各fold cloud-input-v1 output。云发布/独立cloud准入未完成时不派发GPU，进程或Task存在不算完成；保留partial和失败，不重新启动已存在publisher。下一项为实际checkpoint绑定的GPU logits和gate-aware Top-H runner/boot/dispatch闭包。

## 2026-10-01 全量关联GPU推断与独立精度诊断

全五折云输入两产物及全部archive成员已独立接受，索引 receipts/spd-canonical-oof-heldout-car-association-cloud-input-fivefold-acceptance-index-20261001.json；不重跑发布。实际GPU推断seed1337：fold0 a5493d4545114f1fb412af74768c9536、fold1 83892d574e894bdeb2d87252910874df（A100），fold2 6731eaf22bc2449f9e04308f80b19f94、fold3 d6a824a28c8241bc89819a7ae5e20421（V100），fold4 7147a5c69fc742cdbaca4173e6ba0577（2080Ti）。每折4卡，checkpoint/独立前向/source/held-input云bytes绑定；输出完整pair/unmatched logits、多gate(.90/.95/.99)和TopH(1/3/5)联合备选以及所有源事件清单，未使用GT/优化器。fold0–3实时completed五产物登记，fold4运行；仍需完整独立读回，不能计论文评价完成。

新NumPy独立reference不依赖Torch：限定CPU float32 storage/Tensor ZIP重建白名单，float64线性/LayerNorm/GELU与pair/dustbin公式。此前五fold真实fit fixture用于入门检查，容差预设atol/rtol1e-4。首轮harness未剥batch轴触发shape失败，剥离后V100 fold2/3误差~1e-7且通过，A100 fold0/1/4最大误差~1e-3并超过既定容差。失败完整保存，未放宽容差或修改旧产物，不能宣称跨设备精确前向/推断已接受。候选A100四卡full-float32 precision probe已真实派发，显式NVIDIA_TF32_OVERRIDE=0、TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0和Torch matmul/cudnn allow_tf32=False，只复用旧权重/GT-free fit fixture，不重训、不用held-out标签。probe source-freezes/spd-canonical-oof-association-A100-float32-precision-probe-v1-20261001；Task ID/worker交集在同名dispatch receipt。必须等probe completed并独立receipt读回，才能确定precision原因及单独命名后续推断合同，不能把排队当通过。

全五折v1 GPU推断都completed，全部五产物字节独立读回；完整8504帧CPU NumPy公式及多gate TopH分支重算全部实际执行，不依赖Torch或production NN/assignment helper。fold2/3/4（V100/V100/2080Ti）零numeric failures，最大绝对误差约3.9e-6/4.6e-6/4.0e-6；该三折完整车端参考输入和源事件、每query/几何距离/每TopH联合最优能量/权重/unmatched全部通过，禁止重跑。fold0/1（A100）分别842/878帧超过预设atol/rtol1e-4，最大差约0.0047/0.0054；byte/覆盖/TopH数学关系通过但NN精度不接受，失败回执原样保留。索引 receipts/spd-canonical-oof-heldout-car-association-inference-v1-full-independent-result-index-20261001.json。

关闭TF32的真实A100四卡probe af97a56b7bfb40e4996821c1faf4c1a6 已completed并完整artifact/source独立接受，四rank NumPy参考独立重算最大差2.27e-7，原weights不变、atol/rtol不变。单独命名highest-float32-TF32-disabled-v2推断仅针对明确失败fold0/1，沿用旧训练checkpoint/已接受held输入/TopH math；不重训、不改变gate或deadline，不重跑已接受的fold2/3/4。新Task source/环境显式NVIDIA_TF32_OVERRIDE=0、TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0、torch.set_float32_matmul_precision(highest)、CUDA matmul/cudnn allow_tf32=False，运行时产物记录实际flags和设备。Task绑定旧numeric failure receipt、已接受precision probe，派发前物理四/八卡碰撞核验。journal receipts/spd-canonical-oof-full-heldout-car-association-GPU-inference-full-float32-v2-dispatch-20261001.json；尚未称两折v2实际完成或全量数值通过，更不是完整channel tracking replay/VoI/论文指标。


### 2026-10-01 五折完整车辆参考关联推断独立验收

五折 8,504 个车辆参考事件、76,536 个 gate × Top-H 单元全部通过独立字节、身份、NumPy 全量网络数值与假设最优性核验。fold0/1 使用显式 TF32 关闭的 full-float32-v2；fold2/3/4 保留已通过的 v1。原 A100 v1 两折数值失败回执保留，未放宽 1e-4 容差，未重训。16,338 个源帧清单保留；本验收不等于完整源事件/信道/跟踪回放，不构成论文指标。下一依赖为 canonical OOF 合同下的完整源事件消费与跟踪回放，随后 fit-only 配对因果 VoI 标签。回执：`receipts/spd-canonical-oof-fivefold-heldout-car-association-final-independent-acceptance-index-20261001.json`，SHA256 `0fc67c84f7f56e7634fb875b6cf6e801cbbcc0977a5c44dab866654928698908`。关联推断 ETA：已完成；完整回放 ETA：未知。


完整源事件回放入口准备：对 16,338 源帧按原始 source image/box 完整可用时间定位最早车辆 100ms deadline，16,184 帧可分配，154 帧在本序列最后车辆 deadline 之后。154 帧逐项保留，不能静默丢弃、延长已提交 deadline 或捏造车辆评价事件；后续源事件消费者必须处理这部分生命周期。此次仅元数据准入审计，未执行 tracker、网络映射或 agent mask 消费。回执 `receipts/spd-canonical-oof-source-event-replay-admission-audit-20261001.json`。


20261001 canonical V2 源帧入口集成：新 CanonicalOOFCacheArrivals 按角色/折绑定冻结双主体缓存，16,338 源帧全消费（含154末尾帧），same-root vehicle-only mask 仅消费8,504车辆载荷，逐帧拒绝提前可用与重复前缀幂等检查通过；完整独立 ledger/实际数组字节回读已启动，待终态及回执才接受该阶段。新 canonical_oof_v2_features 解除旧固定 calibration SHA 的历史全train绑定，五折完整8,504参考输入的实际 V2 car query IDs/203 features 与先前独立接受 raw 输入完全相同，最大绝对误差0。旧 frozen source 未改动。仍未运行完整 tracking/lineage/source-time mixture/channel replay，不是论文指标，也未为填卡重复GPU验收。


上述全量到达 ledger 独立回读已exit0并接受，回执 `artifacts/spd-canonical-oof-full-source-cache-arrival-consumer-v1-20261001/independent-ingestion-acceptance.json` SHA256 `94454539b5241ad7db9ae80053af018707beada6bb03a8205789d20127d521d0`。V1冻结执行结果完整保留；生产入口另外修复返回receipt嵌套列表暴露问题，v2使用deepcopy，真实冻结帧的调用者修改返回值后重复消费仍得到未改写first receipt/相同prefix回归通过；这是v2局部回归，不冒充完整v2远端执行验收。

2026-10-01 RBF authoritative scope correction: thesis tracking/recover-before-fuse/{experiments,method}.md excludes old EventTrack VoI/C1–C9/144 from RBF completion. All-class score>=.05/top64 candidate competition remains; car-only is evaluation/aggregation. Existing car-only fivefold training/inference and V2 ingestion are preserved diagnostics, not joint temporal RBF core. Original nested joint model source task 7a7586375f1c467b91956abf3a681d21 full code bytes independently recovered, 474 safe members, existing checkpoint source files match. Required three-seed all-train-row GPU deployment candidates dispatched: 1337 16ab085f9e874513a395f1390937d9c5 A100; 2027 27e498ab50a94726af71154062471d5c V100; 3407 1d84721b1eb549b4b4adb9f0ed5ca587 2080Ti. Each four physical GPUs; original frozen model/parser, all rows without GT query selection, no optimizer, CPU original unbatched deployment probes and history ablation, TF32 disabled, dynamic row ETA. Queued is not accepted or paper performance complete. receipts/rbf-joint-identity-all-row-GPU-admission-v1-dispatch-20261001.json.

2026-10-01: Actual RBF nested joint cross-source + temporal model GPU candidates all completed: A100/V100/2080Ti respectively. All 15 cloud output artifacts independently byte read back, all46 train sequence rows per seed counted. Full separate NumPy float64 reference (no Torch or production NN/attention/motion imports) recomputed 677744 rows: seed1337 229212 max error6.5586e-6, seed2027 225177 max6.2194e-6, seed3407 223355 max8.3769e-6; zero rows outside fixed atol/rtol1e-4. Original CPU float32 tensor ZIP read restricted to explicit storage/tensor globals, full original weights/input archive/shard hashes verified, all-class queries chosen without GT. Source freezes rbf-joint-identity-full-independent-numpy-v1-20261001; receipt /Volumes/Data/test/recover-before-fuse/receipts/rbf-joint-identity-three-seed-all-row-full-independent-numeric-acceptance-20261001.json SHA256 4c4733bb1362b5237c8d4a28f2b027b4767d59383f68e5a6a0654b50fcb21d19. This admits original nested model row potentials for the declared paired train schedule; it does not admit whole forest replay, final all-train paper checkpoints, or independent evaluation. Historical fulltrain joint weights have legacy car-only/in-sample row protocol and remain excluded from RBF main all-class training. Next dependency: retrieve original hash-bound cache and cooperative arrival schedule; do not synthesize empty events or infer full schedule from nonempty query rows.


## 2026-10-01: RBF joint-model coupled forest transition (separate from historical car-only OOF)

Original three-seed cache metadata recovery tasks ff523cf6b0cf4730b562334cb9a7f6e3, 4a2a40bf788049d5925dd23ff362c90d, 017d45d49b4b4faebc761c9f874a8d1b completed on L40S CPU without GPU frameworks. All metadata cloud bytes and membership in original cache manifests were independently checked. Each seed preserves 16,338 source frames, 7,445 original paired train events, 11,197 available source deliveries and 3,693 deadline-unavailable frames; 1,448 unpaired source frames remain outside this original paired schedule. Prediction-only schedule/events contain no GT or system-error offsets.

`paper_runtime.py` formerly used minimum delivered information time as feature origin, unlike original training minimum sealed-sequence box-reference time. The targeted repair preserves arrival gates and all seven checkpoint scoring sources. All 677,744 real row features at index 141 exactly match the corrected training origin and differ from the old origin across all 46 sequences. Receipt: `test/recover-before-fuse/receipts/rbf-paper-runtime-training-origin-all-real-rows-parity-20261001.json`. Initial report-serialization error and HTTP401 upload-host/auth routing failure are preserved; neither is a model numerical failure.

Source publication cd523627a8684f00b616ea789655f2e1 and prediction-only events were independently read back. Coupled train development replay uses original admitted joint nested checkpoints, current frozen recoverable/beam backends, per-sequence four-GPU work, immutable SQLite commits and original neural-factor parity at unchanged atol/rtol 1e-4. Initial RBF startup constructor checks rejected a missing serialized inherited `coverage_admission_version=1`; zero rank replay outputs were recorded. Failed tasks remain preserved. Separately corrected config-v2 source publication 6106d146e32048cfbb6ad0a5f631bf3c adds the existing field without changing model, algorithms or caps; only diagnosed terminal zero-work RBF failures qualify for the corrected named candidates. Existing Top-K tasks continue and are not repeated. Dispatch journals: `receipts/rbf-original-joint-coupled-forest-GPU4-v1-dispatch-20261001.json` and v2 equivalent. Configuration/startup, whole output byte readback, fresh branch-state numerical replay and equal-resource comparisons are separate gates; this is not full Stage2 or paper performance.
