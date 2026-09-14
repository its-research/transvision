# V2V4Real 四卡 A100 运行环境检查

当前训练只允许 A100，单个任务至少使用 4 卡。2026-09-14 的独立检查任务已在四张 A100 40GB 上完成小矩阵运算，但现有镜像缺少检测器依赖，尚不能运行 V2V4Real 的 PointPillar。检查完成不表示检测器、分布式训练或论文实验完成。

## 实际执行结果

[ClearML 检查任务](http://10.100.35.118:8080/projects/e30491b7c54a48aea056748ec079fd95/experiments/cb8522b6682049ceaeb3c9202bb1d64f/output/log) 为 `cb8522b6682049ceaeb3c9202bb1d64f`，状态为 `completed`。任务记录的 worker 是 `10.100.34.18-A100:gpu4,5,6,7`；北京时间 01:19:09 启动，01:19:44 完成，记录的活动时长为 35 秒。

每张卡完成一次 `64×64` 全一矩阵相乘，输出元素总和均为 `262144`。每卡报告总显存 `42,405,855,232 B`，检查时可用 `41,849,257,984 B`。这只验证分配设备和基本 CUDA 运算，不验证 NCCL、DDP、体素化、NMS 或检测器前向。

运行环境为 Python `3.12.12`、PyTorch `2.10.0+cu128`、CUDA `12.8`，容器可找到 `nvcc`。检查使用现有镜像，未修改共享镜像、worker、队列或其他运行任务。

15 项预声明依赖中，5 项导入成功，10 项以 `ModuleNotFoundError` 失败：

| 状态 | 模块 |
| --- | --- |
| 可导入 | NumPy `2.4.2`、SciPy `1.17.0`、PyYAML `6.0.3`、OpenCV、torchvision `0.25.0` |
| 导入失败 | `numba`、`open3d`、`spconv.pytorch`、`cumm`、`shapely`、`einops`、`timm`、`tensorboardX`、`skimage`、`opencood` |

这是预声明模块的检查，不是官方全部依赖的完整兼容性证明。探针没有安装这些模块，没有读取数据或加载权重。ClearML agent 按任务配置准备 SDK，不等于探针修改了其他任务的环境。

01:09 的只读快照仅发现一台在线 A100 主机。GPU 0–3 当时运行其他任务；GPU 4–7 空闲。八卡 worker 与忙碌卡重叠，且 `GPU8-A100` 队列带有 `force_workers:off`，不能把它计作另一个空闲资源组。本次不具备多机并行条件；快照不是后续任务的资源预留。

## 提交和验收边界

新增代码为：

- `tools/event_track_v2x/probe_v2v4real_a100_runtime.py`：先检查恰好 4 张 A100，再做小矩阵运算；各模块在独立子进程导入，单项最多 20 秒。
- `tools/event_track_v2x/submit_v2v4real_a100_runtime.py`：默认只读；仅在显式 `--execute`、探针 SHA-256 匹配且四卡 A100 容量检查通过后创建检查任务。

提交入口使用现有的 worker 重叠检测，排除 V100、5090 和八卡队列。相同探针已有 `created`、`queued`、`in_progress` 或 `completed` 任务时不重复创建。若创建后容量变化，保留新建任务并停止入队；若入队响应不确定，先按已打印的任务 ID 查状态，不盲目重试。

58 项相关测试通过，覆盖新探针、提交限制及原有 A100-only 优先级训练入口。远端三个源文件逐一通过哈希核对，完成后的 ClearML 脚本再次与本地探针哈希一致。已有身份模型的三个种子没有重训。

审计及测试报告保存在 `work_dirs/recover-before-fuse/v2v4real-runtime-20260914/`：

| 文件 | SHA-256 |
| --- | --- |
| `a100-runtime-task-audit-v1.json` | `81e285bd4b501be2576f9fe26d6cb3a690b13da573044ef21cd90d6f2eb6dea3` |
| `runtime-submission-tests-v1.xml` | `402d683753cff8c6448f47fdcf2ff1d272769cf8889c5096b42d7898f98dc2bb` |
| 探针源码 | `082fe4c6313fba1ff6b5f2f7d59e0be54dc0b7f6bc3cd9cfac868cd2a73896ab` |
| 提交入口源码 | `768472fb3d4df81a395e756aa7982b9cb769919c147e192f5696efbfaa87c855` |
| 复用的 `submit_forest_identity_ddp.py` | `ca78afc351851c12b7144582f83df29f62624736df6aba25f08bd8af9a44a2c1` |

本次内部部署只传输三个小型源文件及诊断日志，不含 GT、模型、点云或预测流。当前论文项目未新增研究代码。

## 官方检测器来源：仍待补齐

官方 README 的 No Fusion 和 Late Fusion 使用同一个 [PointPillar 模型包](https://ucla.app.box.com/v/UCLA-MobilityLab-V2V4REAL/file/1619899319493)。公开页面元数据为 `late_fusion.zip`，大小 `27,749,283 B`，文件版本 `1780971131493`，SHA-1 为 `105c1f93498f3f8e31a866f7ccc5c4157769a86d`。这是网页元数据，不是本地完整包校验结果。[官方入口](https://github.com/ucla-mobility/V2V4Real#benchmark)

本次 Mac 锁屏使正常浏览器下载不可用；页面提供的内容地址直接访问返回 HTTP 401，已停止这条下载路径，没有绕过认证。模型包尚未取得，不能宣称检查过包内配置、权重或训练来源。恢复下载需要先解锁 Mac，再通过官方公开页面操作。

已单独阅读固定提交 `5a821e13753bafc611f95c47bc1a306acdcb0f7c` 的公开配置和训练代码。仓库配置将训练目录设为 `train`、验证目录设为 `validate`，每个 epoch 保存权重并计算 validation loss。这不能证明发布权重采用何种选模规则，也不能证明包内配置与仓库示例完全一致。[配置](https://github.com/ucla-mobility/V2V4Real/blob/5a821e13753bafc611f95c47bc1a306acdcb0f7c/opencood/hypes_yaml/point_pillar_late_fusion.yaml)、[训练入口](https://github.com/ucla-mobility/V2V4Real/blob/5a821e13753bafc611f95c47bc1a306acdcb0f7c/opencood/tools/train.py)

公开配置采用 `SpVoxelPreprocessor`，体素大小 `[0.4,0.4,8]`，源车点云范围 `[-70.4,-40,-5,70.4,40,3]`，分数阈值 `0.20`、NMS 阈值 `0.15`。这个检测输入范围不能替代已固定的 GT 评价范围，也不能在包内配置未核实前用作正式缓存合同。

下载到的四份官方文本保留在本地独立参考目录 `/private/tmp/rbf-v2v4real-detector-20260914.NNxp0Z`，不随本项目源码包再分发：

| 文本 | SHA-256 |
| --- | --- |
| `point_pillar_late_fusion.yaml` | `138c4ad3508fdd7061f5b290c83ad8f0772fea33f0af0e2be56330423a3c92d1` |
| `point_pillar.py` | `21f36cce8ed105c8a125d147534c3d85cb7d95f92a46f1e8f1041a40fb821af5` |
| `requirements.txt` | `2cb097223bdc7cc36445a49d035cd63c3f24da8e24b68eb26e12151b5c5c99fd` |
| `train.py` | `64a02008bd8043c1cf64f14d4bd43e6316c32cf7db5752eeecc15c5dd0d6644f` |

## 后续执行条件

1. 为检测器准备隔离环境，固定依赖并在四卡 A100 上验证体素化、NMS 和前向。保留现有可运行的身份模型训练环境。
2. 从官方入口取得模型包，核对版本、内容哈希、包内配置和可获得的训练/选模信息。缺失的来源信息继续标为待确认。
3. 只用已准备的 train 点云进行检测缓存接入；不向推理输入暴露 GT，不依据 GT 删除 Truck 预测，不把 Truck 或 ConcreteTruck 合入 Car 标签。
4. 取得真实检测输出后才能验证 `DetectionCacheV2` 接入。正式比较保持冻结检测器、相同输入预算及严格 Car 范围；V2V4Real 官方 test 仍仅用于最终评估。

本次未产生新的参数训练、V2V4Real 检测缓存、跟踪指标或论文主表结果。SPD 的两个现有 CPU 教师进程继续运行，未重启或改动它们绑定的源码。
