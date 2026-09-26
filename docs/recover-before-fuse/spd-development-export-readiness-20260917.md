# SPD 开发权重导出衔接检查

## GPU 导出调度（2026-09-18）

### 三片已发布、缺失分片补跑

第二次任务 `7c9e1ea83f9e42b19410a968aa2c3d37` 最终为 failed。车端分片 0 在首次导入 MMCV/YAPF 时发生语法缓存 EOFError；其余三片完成真实 GPU 前向，日志中的独立回读全部通过、权重未改变。三片清单已实际回读，绑定 fit37 最终权重与 `raw-head-all-queries-no-roi-no-nms-v1`。归档另在管理机完整流式回读，大小和 SHA-256 均与 ClearML 一致，不额外落盘。

| 分片 | 帧数 | 原始候选数 | 归档 bytes | 归档 SHA-256 |
|---|---:|---:|---:|---|
| infrastructure/0 | 4161 | 3744900 | 973485234 | `87e998c1f649a852f3c80dc1e478a641a0c688c652d850bfb1da1ece26ceb9bb` |
| infrastructure/1 | 3673 | 3305700 | 858522920 | `fdc190933ab7407d0beacf6819f575cbf71ae7e3b34c7203862640a3f6948613` |
| vehicle/1 | 4248 | 3823200 | 746058658 | `d2cf57f6f5ce054d29380748596b17a464d057cc46da2ab414dfe18136dda3ed` |

在隔离源码中加入并发前串行依赖初始化，以及显式单分片选择；相关测试 28 passed，独立固定 CPU 容器串行导入返回 SERIAL_IMPORT_OK。尚不把 CPU 导入测试称为并发问题已在 GPU 验证。

修正版七文件源码冻结任务 `8641d7b497e14234ba2d8aeb505f0d8d` 已逐文件回读，inventory SHA-256 `7788e58c9124832bd155c788929099750c730f584c608536ad9cae781f242c59`。第三次任务 `bd986e3719624c81835802e4c850feb8` 只请求 vehicle-side/shard0；已核验参数，当前查询为 in_progress、GPU 4–7，环境准备完成但尚无分片结果。部分补跑 summary 显式 whole_input_coverage_verified=false，后续须跨任务验证四片覆盖，不重复计算三个成功分片。

以下保留调度历史快照，不代表最新任务状态。

- 新入口位于隔离 runtime worktree：`run_spd_export_clearml.py`、`spd_export_dispatch.py`、`bootstrap_spd_export.py`。绑定上文最终训练收集回执，拉取无 GT 输入，四个独立子进程分别运行两端各两个分片；发布前独立回读，并核验权重、分片、计数与清单身份。相关绑定和分片验证测试 22 passed，不代表真实导出验收。
- 第一版代码冻结任务 `fbc3dbb6f97f456ea0547248b3d120a0`，七份源码实际回读通过。首次运行 `f4b1bae94ecc4085befbadfb47f170a7` 已失败：共享队列分配到 GPU 0–3，而启动器仅允许 GPU 4–7；在下载实验资产与推断前报 `wrong worker`，保留失败记录。
- 修正版允许同一 A100 的 GPU 0–3 或 GPU 4–7 四卡工作器，仍拒绝其他工作器。新版代码任务 `569df796f6c342c38635836c3d8ff149` 七份源码实际回读通过，inventory SHA-256 为 `2452a83d34cb4474e30c09ea0d8d49ba24bf625cef78ec66b77b007d521d9a72`。
- 第二次任务 `7c9e1ea83f9e42b19410a968aa2c3d37` 已提交 GPU4-A100，本次查询为 in_progress、工作器 GPU 4–7，日志仍为容器依赖准备。尚无真实前向或缓存结果。提交过程曾因容器字段为 dict 发生客户端异常；重新查询并复用同一 created 任务完成配置后排队，没有重复创建运行任务。

后续以该任务实时状态、分片日志、全量回读和 ClearML 产物为准；不得将 in_progress 或代码冻结标记为正式缓存完成。

## 两端训练完成与最终权重验证（2026-09-18）

ClearML 任务 `9f4eb07e4fab4f63a9f30c470f91255b` 已实际查询为 `completed`。两端 completion、final-checkpoint、launch、optimizer-startup 和实际 detector config 共 10 份产物已从 ClearML 流式收集并逐份核验大小、SHA-256 与训练来源绑定。收集器正常退出，成功回执位于管理机 `/home/lbin/Desktop/rbf-spd-final-collection-9f4eb07e-20260918/collection-receipt.json`。

- seed 1337，fit37/holdout9，24 epochs；车端最终迭代 4968，路端 4632。
- 收集回执已发布至独立 ClearML 审计任务 `dcb8b31d3d194f6aa6695d95228e01c8` 并实际流式回读：5049 bytes，SHA-256 `281977e4ed1184931a55de3b5f726b9625b6cada4971f7558b2e4fc97563fd91`。此回执不包含后续架构加载验证，不代表 GPU 推断完成。
- 车端最终权重：400419820 bytes，SHA-256 `15a2187c4890d95cf10a3e38c93d69a986a7814b5610723036376d4e4fd63ae9`。
- 路端最终权重：400420268 bytes，SHA-256 `c76bf1a389b97a23ed96128803fea503f63c338948d0ecb9c9eef0cc3354186b`。
- 两端安全张量加载均通过：各 651 个模型张量、33631011 个模型元素、738 个优化器张量均有限；保存的 meta.iter 分别为 4967/4631，符合 MMCV 计数规则。

随后在固定镜像 `sha256:85525aefed5d9a5d5f6d72d7853f1a9ba3744206c889c767a8dfac8202c1d2ab` 中，以无网络、无 GPU、只读挂载方式，使用官方固定源码和实际配置构建两端 detector，再调用 `model.load_state_dict(..., strict=True)`。两端均返回 missing=[]、unexpected=[]，命令退出码 0。配置仅应用导出入口相同的 pretrained=None、batch_size=1、train_cfg=None，并检查 train_det=True、history/future reasoning=False。

收集回执保留收集当时的 architecture_load_verified=false，不覆盖历史证据；后续 CPU 严格加载是独立验证，目前证据在本轮命令输出。该验证尚未形成独立 ClearML 审计回执。GPU 前向、外观特征、真实缓存导出、跟踪评价及断点恢复等价性均未由此证明，paper_ranking_eligible 仍为 false。

## 历史训练状态核验（2026-09-18，路端 850 步快照）

本节保留当时的检查过程，不代表当前状态。ClearML 任务 `9f4eb07e4fab4f63a9f30c470f91255b` 当时实际查询为 `in_progress`。

- 已读取 `vehicle-side-completion`：seed 1337、fit37、24 epochs、`iter_4968.pth`，每卡 batch 8、有效 batch 32；回执中的权重 SHA-256 为 `15a2187c4890d95cf10a3e38c93d69a986a7814b5610723036376d4e4fd63ae9`。`official_val_result_available=false`，不能当作验证集结果。
- 已确认 `vehicle-side-final-checkpoint` 附件存在。本轮仅读取完成回执，没有重新流式核验该大文件；附件 metadata 为空，不能将 metadata 当作独立哈希证据。
- 路端最新日志时间为 2026-09-17 18:01:34 UTC（北京时间 9 月 18 日 02:01:34），实际进度 850/4632，total_loss=4.9836、grad_norm=19.4262；该日志值有限，不代表后续训练质量或最终收敛。
- 两端最终收集门槛尚未满足：当前只有车端 completion/final-checkpoint，路端仍在训练。继续使用原任务，不重复启动、不修改冻结源码；双端完成后执行严格收集、架构加载及真实缓存导出。

正在训练的 fit37 检测器任务为 `9f4eb07e4fab4f63a9f30c470f91255b`。本检查只定位下一阶段真实入口，不替换运行中的源码，不升级旧缓存为新权重证据。

## 已核对的来源

旧全 train 原始预测任务 `a23f17b1cc7d4b6b8bd521587a0d65d4` 的入口是 ClearML 内联 `run_clearml_cache.py`，并引用：

- 输入包 `9199a9d7af164056920dd0fe5d0c0247`，清单 SHA-256 `9fd08a71200410cea3f20b4d5c284068b010e69ca27d6810ee57207ad732f18e`。
- 代码包 `c39e2280d3894737b82c393dd4117dea`，清单 SHA-256 `b824df05b27475f32d464eb577a8a3c7c105a1625dded2d1d1ba2c2f25e3cf49`。
- 代码归档 9,560 bytes，SHA-256 `8bd84ec90e856e3112c56d30164c35170dd17f9b451f00d52c7d071893c0e86e`，本轮实际回读验证。
- `run_cache.py` SHA-256 `a0c1a17fb50d7429a4247edd1b31fcc3c51446f13eb5e0becc44ae76ff056be6`；`cache_primitives.py` SHA-256 `a041cb18c2ee24f4a2b4c731088e11181fb7dac1b62f6c3ee7925ef8316bab8e`，来自已核验包清单。

`install_raw_detector_decode` 在 tracking summary 覆盖低分 query 前保存原始头输出，每帧强制只消费一次，再调用上游解码器。它不是直接输出全部未解码 query；会保留上游 decoder 的 top-k、阈值及几何范围规则。

固定官方车端配置显示 decoder `max_num=300`、`score_threshold=0.0`、`with_nms=False`，同时存在 `post_center_range=[-61.2,-61.2,-10,61.2,61.2,10]`。runtime tracker 的 0.3/0.4 阈值与 decoder 不是同一个筛选层，不能混为一谈。

## 下一步验收要求

1. 回收并版本化上述真实源码，不以通用预测占位接口代替。新控制器绑定最终 fit37 权重及其完成回执，不能继续绑定旧 full-train 权重。
2. 核对运行时 resolved config、实际 `_det_instances2results` 和 coder，实现主协议全类别 raw score>=0.05/top64；明确原生解码 ROI 与论文评价 ROI 的边界，不在未核实前称为全部原始 query。
3. 输入只包含图像、时间和坐标元数据，GT 不进入导出。新开发划分必须保留 fit/holdout 身份；旧控制器硬编码的 46 序列覆盖检查不能直接充当 37 序列导出验收。
4. 正式训练完成后重新生成预测、特征及校准输入；历史缓存和旧校准参数不能仅靠更换 manifest 获得新权重身份。

当前未生成新缓存，未完成导出适配或开发集评价。训练继续运行，最新观察为第 60/4968 步，损失有限；该迭代数是本轮快照，不是固定当前状态。

## 解码调用链的进一步核验

实际固定官方代码 `cooptrack.py::_det_instances2results` 从最后一层 logits 计算 sigmoid 后逐 query 最大类别分数，传入 `DETRTrack3DCoder`。后者先按该分数取 `max_num=300` 个 query，再 denormalize 几何框，默认 `with_mask=True` 按 `post_center_range` 裁剪；`with_nms=False` 时不执行 NMS。类别是每个 query 的 argmax，不是仅取 car。

因此旧导出器虽然绕过了 tracking summary 的覆盖，仍不是未裁剪原始 query 导出。即使查询数量不超过 300，空间裁剪也可能在下游全类别 top64 前移除高分候选。不能直接将该缓存标记为主协议合格，也不能静默把该 ROI 当作论文评价 ROI。

后续主协议适配应显式保留全部 query 的解码候选，关闭前置 ROI/NMS/额外 top-k，之后使用统一 raw score>=0.05、全类别 top64 规则。上游 coder 已有 `with_mask=False` 接口，但仍保留其 `max_num` 和可能的 NMS，故仅传该参数不足以证明完整性。需要同时验证导出数量、候选身份、配置及边界测试，并以新协议 manifest 隔离旧输出。运行中的检测器训练不受此推断导出修正影响。

## 未裁剪解码模块的软件验证

隔离 runtime worktree 新增 `tools/event_track_v2x/cooptrack_raw_query_decode.py`。直接保留每个 query，使用三类别 sigmoid 最大分数和 argmax 类别，调用传入的官方 denormalize 函数，不执行 ROI、NMS、分数阈值或 top-k，并输出原始 query index。非有限头输出、非有限解码框和非正尺寸会拒绝。

安装函数仅允许 eval-only、单帧 detector，且必须先于旧 raw-head snapshot adapter 安装，使后者捕获的新解码函数消费覆盖前的头输出。模块本身不替代原始头快照保护，也不执行下游 top64。

远端独立 Torch CPU 验证 7 passed，目录 `/home/lbin/Desktop/rbf-uncropped-decoder-check-20260917`。覆盖 400 个 query 不被 top300 截断、ROI 外保留、全类别 argmax、低分和重叠框保留、三类非有限值拒绝、结果接口及安装顺序。尚未接入真实 `run_cache.py`，尚无新权重真实帧导出结果，不能据此标记完整缓存路径完成。

## 真实导出入口源码接通

已从上述固定 ClearML 包恢复源码到隔离 runtime worktree，命名为 `run_spd_raw_cache.py` 和 `spd_cache_primitives.py`，历史包与运行中的训练不变。新入口先安装未裁剪解码，再安装覆盖前原始头快照；每帧验证 query index 为完整顺序、解码与快照帧计数一致、导出数等于原始 query 数。每帧清单记录 `raw_query_count`，独立 verifier 拒绝不完整数量。

新 `detector_decode_source` 为 `raw-head-all-queries-no-roi-no-nms-v1`，launch 明确记录前置 ROI/NMS/top-k 为 false、解码模块哈希，以及下游才执行 raw score>=0.05/all-class-top64 的边界。旧 cache builder 尚不接受该新标识，需要单独适配验收，不能混排历史缓存。

Torch 组合测试最终 9 passed；新增场景用真实快照适配器配合模拟单帧模型，将 summary 中全部 logits/boxes 清零，验证原始 400 个 query、类别及 ROI 外坐标在连续两帧仍完整保留。第一次组合测试有一项因测试局部变量作用域错误失败，修正后全量重跑通过。CLI `--help` 与 `git diff --check` 通过。

尚未使用真实图像运行本入口；需完成 label-free 开发输入、最终权重来源绑定、新缓存标识接入，并在可用 GPU 上验证实际模型 forward 及特征提取。软件组合测试不等于真实帧推断或正式缓存完成。

## 新解码标识的缓存封装边界

`build_detection_cache_v2.py` 已增加独立未裁剪解码分支，要求前置 ROI/NMS/top-k 均为 false、完整候选策略字符串、合法解码模块哈希及逐帧 `raw_query_count == detections`；禁止同一封装包混入旧解码分片。旧解码分支继续保持原校验，不把旧缓存重新标记为新协议。

DetectionCacheV2 的数组和元数据 schema 不变，新来源通过已有原始 manifest/config 哈希绑定。封装本身保留全部候选，不进行 raw top64 选择；下游协议选择仍须单独验证。清单中的声明与计数是软件边界证据，不能取代真实模型运行核验。

相关测试文件 35 passed，覆盖新标识封装及完整回读、旧协议兼容、混排拒绝、ROI 声明、缺失代码身份和数量不一致拒绝；`git diff --check` 通过。真实 label-free 输入准备、最终权重绑定与真实帧导出仍未完成。

## 现有图像位姿输入包的流式核验

新增 `audit_spd_inference_input_stream.py`，实际通过 ClearML 流读取 `9199a9d7af164056920dd0fe5d0c0247` 的 cache-inputs 全部 3,927,967,368 bytes，SHA-256 `e12b335c8eaa6d9a534609ddddef371e9e10e168c01918da9559ef5eca3606cb` 一致，没有保存第二份归档。

- 归档 16,353 个成员，其中 16,338 张图像，等于车端 8,504 帧加路端 7,834 帧。
- 路径检查未发现 label/labels/annotations 目录，拒绝绝对路径、穿越、重复成员和链接。
- input manifest 与两端 frame-index 哈希逐项核验，覆盖官方 train 46 序列；manifest 声明无 val/test/GT payload。
- 读取器回归 7 passed。第一次启动因部署目录中的旧 helper 缺少 ChunkReader 而在导入阶段失败，未开始传输；随后在独立目录部署匹配版本并完成上述核验。

核验范围不包括两个 image-pose-infos.pkl 内部字段，尚不能单凭无标注目录证明 GT 完全隔离。后续需按已固定 infos 哈希检查字段白名单、单帧信息及 37/9 开发身份；旧 training-config 也需与新权重架构及来源分开绑定。该包的训练任务引用是历史 full-train 来源，不可当作新 fit37 检测器的训练来源。

## 逐帧字段与图像审计已完成

找到历史输入目录 `10.100.35.112:/home/lbin/Desktop/eventtrack-cooptrack-runtime-20260911/cache-20260912/inputs`，两端 pickle 字节哈希与已核验 ClearML 包清单相同。使用限制 numpy 必要类型的 Unpickler 读取，拒绝其他全局类型；逐帧严格检查 row/camera 字段白名单、有限数组、frame ID/sequence/timestamp 与 frame-index 相等、next/prev/sweeps 为空、lidar_path 为 unused、图像路径与 token 一致，并回读全部图像验证 SHA-256。

| 端 | fit37 帧 | holdout9 帧 | 已核验图像 |
|---|---:|---:|---:|
| vehicle-side | 6611 | 1893 | 8504 |
| infrastructure-side | 6163 | 1671 | 7834 |

划分来自固定 fit 输入清单 `65815abf7ec0d8aadccc1bf8d1fc095cca26ae09292392ddcebaf838743df8da`，37/9 互斥且全集等于官方 train 46 序列。防 GT 字段、未来引用、未知 camera 字段、路径穿越、时间变化、非有限值及不允许的 pickle 全局类型测试 8 passed。

ClearML 审计任务 `03ee442514624e7e95c4277924f79696` 已完成，audit 与源码均上传并实际回读核验。audit SHA-256 `f69e364f9c77c6c7aad58104e95585820a70640089f634a4fbe5a45c7f2f87bd`，源码 SHA-256 `620f541767918e2ce27962cde8338c0812257e19186cb0f5c527233d7f5c188e`。

本项只证明指定推断输入资产的字段和字节边界，不表示旧训练配置与新权重已匹配，也不表示真实新缓存已生成。下一步仍为最终权重与配置绑定、真实模型导出验证。

## 最终训练证据绑定入口

新增 `spd_export_training_binding.py` 并接入 `run_spd_raw_cache.py`。导出不再以旧输入包中的 training-config 作为模型配置；必须显式传入本次实际配置、launch/startup/completion 回执。检查 startup 的 resolved-config 哈希、正式非 probe 身份、有限损失及非零骨干更新；最终检查点 SHA 和文件迭代号必须与 24 epochs 的完成回执、launch 预算一致。两端身份、种子、四卡 world size、batch 与冻结 fit37/holdout9 规则必须匹配。

导出 launch 记录三个训练证据文件哈希和 binding，仍标记 `paper_ranking_eligible=false`；这不是自动授予论文准入。外层 ClearML 调度器仍需核验任务状态、附件原始字节和来源，不能把用户可编辑 JSON 本身当作不可伪造证明。

训练绑定、无 GT 输入和缓存封装三个测试文件合计 54 passed，覆盖 probe、旧配置、早期 checkpoint、错误权重、全 train 混用、val/test 读取、无更新、NaN、种子与批量不一致拒绝。CLI 和 diff 检查通过。尚未使用最终权重执行真实导出。

## ClearML 最终产物收集入口

新增 `collect_spd_development_training.py`：只接受 completed 任务，校验固定数据包、清单和实际 controller script 字节哈希，要求两端 completion/final-checkpoint/launch/startup/config 附件齐全。收集时从 ClearML 流读取并核验每份文件的大小和 SHA-256，再调用训练绑定校验；所有检查完成后才写 collection-receipt。失败的部分文件保留但不生成成功回执，不自动授予论文准入。

收集器与训练绑定测试合计 17 passed，覆盖未完成状态拒绝、控制器字节变化及缺失最终证据。对真实任务 `9f4eb07e4fab4f63a9f30c470f91255b` 的只读检查返回 `ready_to_collect=false`，原因 `training not completed: in_progress`，未下载最终权重。完整成功收集路径还需真实训练完成后验证，不能用这次拒绝检查替代。

本轮最后观察车端第 580/4968 步，loss=10.3772，仍无两端完成回执。

## 实际固定环境的数据管线预检

真实训练任务的 controller script 字节哈希与收集器固定值一致。车端实际运行配置回读哈希为 `996378ab3839a6c7010f07331cdc78f0cd60a474fe236afa5b088a07a6d09b94`，与 optimizer-startup 中 resolved-config 哈希一致；不是旧输入包的 training-config。

在管理机固定镜像 `sha256:85525aefed5d9a5d5f6d72d7853f1a9ba3744206c889c767a8dfac8202c1d2ab` 中，使用只读源码和已验证图像位姿输入、network=none、不分配 GPU，对四个分片分别创建新进程运行真实 `build_inference_dataset`，核验全分片 token/sequence 覆盖，并实际加载每片首、中、末三张图像通过 `check_model_inputs`。

| 端/分片 | 索引覆盖帧数 | 序列数 | 实际加载样本索引 |
|---|---:|---:|---|
| vehicle/0 | 4256 | 23 | 0, 2128, 4255 |
| vehicle/1 | 4248 | 23 | 0, 2124, 4247 |
| infrastructure/0 | 4161 | 23 | 0, 2080, 4160 |
| infrastructure/1 | 3673 | 23 | 0, 1836, 3672 |

四个命令退出码均为 0。首次导入遇到镜像系统 libstdc++ 缺少 CXXABI_1.3.15；指定 `LD_LIBRARY_PATH=/opt/cooptrack/lib:/opt/cooptrack/lib/python3.8/site-packages/torch/lib:/usr/local/cuda/lib64` 后通过。未修改宿主库、冻结镜像或运行中的 A100 训练。

本次是 12 张真实图像的数据处理管线预检，不是全部图像的模型前向，也没有加载最终检测器权重、执行外观网络或生成预测。全量图像字节核验来自前述独立审计，两个范围不可混淆。真实推断仍待训练完成及导出调度。
# 最终检查点张量放行补充

导出完成信号补充：`run_spd_raw_cache.py` 已接通单独 CPU 子进程调用 `spd_cache_primitives.py --inputs`，重新读取每个 NPZ/JSON，校验载荷哈希、逐帧覆盖和总数。只有独立回执与生产者的帧数、检测数和清单哈希一致，才打印 `EVENTTRACK_CACHE_COMPLETED`。失败保留诊断目录并返回失败；不在严格缓存目录中增添额外文件。新增 7 项完成门槛测试通过（含真实校验进程失败传播），DetectionCacheV2 35 项回归通过、入口 help 通过。真实最终权重的 GPU 导出尚未运行，不能将这些软件检查作为正式缓存验收。

2026-09-17：隔离工作树 `collect_spd_development_training.py` 在下载哈希、最终迭代和训练绑定通过后，新增 `weights_only=True` CPU 加载核验：模型状态非空且全部有限；优化器状态非空、张量及浮点标量有限；MMCV 保存的 `meta.iter` 必须等于最终微迭代数减一。任一失败不写收集成功回执。此检查不等于架构严格加载或断点恢复等价性验证，回执显式保留两者为 false。

- 远端已有 Torch 环境：新增测试 7 passed；本地收集与绑定回归 17 passed。
- 正在运行的任务 `9f4eb07e4fab4f63a9f30c470f91255b` 在本轮观测中仍为 in_progress，车辆端日志推进至 810/4968；后续阶段快照已到 `iter_828.pth`。
- 该阶段快照流式回读 400419820 bytes，SHA-256 `d1b7393a7ed6ef84297114607211a86ed1ca73c809a07a0a970930e94d88c9e7`；安全加载成功，meta.iter=827，651 个模型张量、33631011 个模型元素、738 个优化器张量通过有限性检查。
- 未下载替代最终权重、未启动正式导出、未改动运行中的训练源码；上述阶段快照不得充当训练完成或论文结果。
