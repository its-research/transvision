# 2026-09-17 实验缺口与启动回执

## 最终验收：已完成

任务 `b95a706d83604ebcbebd94f476d83f69` 已于 **2026-09-17 04:58:37 UTC / 12:58:37 北京时间**进入 `completed`。完整 `RBF_POINTPILLAR_SMOKE` 报告已独立复核并保存为 [验收 JSON](pointpillar-a100-acceptance-20260917.json)。这完成当前四 A100 原生 PointPillar 冒烟实验，不表示论文训练或真实数据效果已完成。

| 验收项 | 实际证据 |
| --- | --- |
| 分配资源 | `10.100.34.18-A100:gpu4,5,6,7`，四张 NVIDIA A100-PCIE-40GB |
| 官方源码 | 固定提交 `5a821e13753bafc611f95c47bc1a306acdcb0f7c`，14 文件逐项 SHA-256 校验，包含 LICENSE，无上游源码改动 |
| 私有运行环境 | `isolated=true`，NumPy 1.26.4 从任务私有 venv 加载，其余直接依赖均匹配固定版本 |
| 原生体素化 | 37 点生成 3 个体素，点数 `[32, 1, 1]`，坐标与完整特征精确匹配 |
| 官方旋转 NMS | CPU 执行，保留索引 `[0, 2]` |
| 四卡模型前向 | 每卡 8,057,360 参数；`psm=[1,2,50,88]`，`rm=[1,14,50,88]`；所有输出通过有限值检查 |
| 卡间一致性 | 两个输出头四卡 SHA-256 完全相同，最大绝对差异均为 0 |
| 基础环境 | PyYAML、NumPy、SciPy、Torch 基础环境版本前后不变；libgl1 仅安装在任务容器 |
| 回归 | 43 passed，1 skipped（可选本地 CPU 参考集成）；真实 GPU 结果单独验收 |
| 边界 | 合成输入、随机权重；未读取数据集、未加载检测器权重、未训练、未验证 DDP；`paper_eligible=false` |

最终任务脚本 SHA-256：`87e74c247e7b34708b9f5e61927e2b48a81fcc01ad884841d5f2349b61541b36`；已与 ClearML 实际执行的 script diff 摘要独立比对。

修复归因：首轮 NumPy 版本不符经 `python -I` 隔离后消失；随后实际暴露 Open3D 缺少 `libGL.so.1`，通过任务容器安装 `libgl1` 解决。GitHub 不稳定改为已验证官方源码包；SDK 文件服务 NIC 不匹配复用现有训练入口的进程级 `CLEARML_FILES_HOST` 修复。PyPI 外网低速改用既有内部缓存，并额外校验 Open3D wheel 与官方 PyPI 摘要一致。基础算法、种子、输入和数值检查未变。

修复源码已从临时目录移至本仓库 `work_dirs/rbf-pointpillar-runtime-20260917` 隔离工作树（分支 `codex/pointpillar-runtime-isolation`），三个文件未提交。原工作树的冻结源码未改动；[补丁备份](pointpillar-runtime-fix-20260917.patch) 可应用于基准 `f8d7f54`，完整测试与部署源码在隔离工作树。未提交或推送 Git。

最终复现入口（同名已完成任务会拒绝重复创建）：

```sh
/home/lbin/miniconda3/bin/python /home/lbin/Desktop/rbf-pointpillar-cached-20260917/tools/event_track_v2x/submit_v2v4real_a100_runtime.py \
  --probe-kind pointpillar --execute \
  --probe-sha256 87e74c247e7b34708b9f5e61927e2b48a81fcc01ad884841d5f2349b61541b36 \
  --source-bundle /home/lbin/Desktop/rbf-pointpillar-cached-20260917/official-source.json
```

以下为历史过程记录，状态以本节最终验收为准。

## 持续验收：网络故障与第三轮

后续运行记录（成功必须同时有最终报告和 `completed`）：

| 任务 | 观测结果 |
| --- | --- |
| `0cb7a581266b475a9801cc4af9809124` | `failed`：确认私有 venv 的 numpy 1.26.4 与其他全部直接依赖版本正确，随后官方 Open3D 导入缺少 `libGL.so.1`。 |
| `5474bd20a71347ed8acea2982e910eb0` | 已增加任务容器 `libgl1`，但 GPU 节点 GitHub 持续超时；模型执行前主动停止，终态 `stopped`，保留日志后改用相同哈希官方源码包。 |
| `383ffc12669f4eaba78a92f0e52c78fe` | `failed`：内部源码 artifact 下载 401；worker 默认文件服务 NIC 与 artifact 的 35.118 地址不一致。 |
| `bb2548a772924674b80490eb65a43f08` | 源码认证下载通过；因 PyPI 外网约 0.1–0.2 MB/s 低速，模型计算前主动停止，终态 `stopped`，转用已有内部缓存；未更改服务器权限。 |
| `b95a706d83604ebcbebd94f476d83f69` | `completed`，完整报告及独立检查通过，见最终验收。 |

最新源码 SHA-256：`178974011d85409839159ec2425e28e47a8fa0a9a3480c29eb34d21bd60c4c16`。官方 14 文件包 SHA-256：`eecf545b896deabc6cd2cd7aa31096402daf889153a01cfcc9bd5e37d0c43f79`；严格验证 commit、文件集合和每个文件 SHA 后才写入任务私有目录。模型/权重种子/输入/数值门槛不变。定向测试 `42 passed, 1 skipped`。

第二轮 `6709d15b788e4f239d6669002c039b2c` 最终 `failed`（2026-09-17 04:22:37 UTC）：下载官方源码时 `urllib.error.URLError: [Errno 101] Network is unreachable`，尚未进入子进程，未验证 NumPy 修复。

第三轮在同一隔离工作树增加最多 4 次传输重试（退避 1/2/4 秒），哈希错误仍立即失败；源码下载前置，避免外网不可用时先安装依赖。定向测试 `36 passed, 1 skipped`。第三轮任务 `0cb7a581266b475a9801cc4af9809124`、脚本 SHA-256 `5c57cfe2e64b9f896431900b5a9994abeac7c7aab9b12bd080609d1b1e043f96`，部署目录 `/home/lbin/Desktop/rbf-pointpillar-retry-20260917`，实际 worker `10.100.34.18-A100:gpu4,5,6,7`。已观察一个官方文件首次失败后重试成功、全部源码校验结束，正在安装依赖；最终验收待回报。

## 后续核验：首轮失败与隔离修复重跑

首轮任务 `eaf3e357a6a749fc8e541d5f1de1ec2c` 终态为 `failed`，最后更新 2026-09-17 03:17:16 UTC。依赖安装和官方源码下载后，子进程在版本检查处报 `ValueError: runtime version mismatch: numpy`，未执行模型前向，无论文指标。日志显示安装 numpy 1.26.4 成功，但没有输出实际加载版本，故调度器 Python 路径污染仍是待 GPU 重跑验证的原因假设，不能称已彻底解决。

修复在独立工作树 `/private/tmp/rbf-pointpillar-runtime-20260917`（分支 `codex/pointpillar-runtime-isolation`）进行，原冻结源码未改动。pip 与模型子进程统一采用 `python -I`，忽略继承的 PYTHONPATH/PYTHONHOME；增加实际版本、模块路径日志和私有 NumPy 来源检查。模型、官方源码指纹、输入与依赖版本不变。

定向回归：`34 passed, 1 skipped`；跳过项为缺少官方参考源码的可选 CPU 集成，不计 GPU 成功。新增两项测试覆盖外部包元数据路径污染和虚拟环境/PYTHONHOME 隔离。

修复版已提交：

- ClearML ID：`6709d15b788e4f239d6669002c039b2c`
- 名称：`RBF V2V4Real pointpillar-smoke 5531defe211ff four-A100`
- 队列：`GPU4-A100`
- 提交回执状态：`queued`；随后核验为 `in_progress`，实际 worker 为 `10.100.34.18-A100:gpu4,5,6,7`，2026-09-17 04:11:30 UTC 已进入 `RBF_POINTPILLAR_BOOTSTRAP` 依赖准备阶段。完成未验证。
- 脚本 SHA-256：`5531defe211ff0d11b871dd959741d91be45fe395a86a09f8e537d049e539634`
- 管理机独立部署目录：`/home/lbin/Desktop/rbf-pointpillar-isolated-20260917.TrjsvK`

以下保留首轮启动时的历史快照，不代表当前终态。

## 核验范围

依据相邻论文目录 `tracking/recover-before-fuse/experiments.md`、本仓库训练及基线回执、已发布提交 `807096490746da7c10b1b79e5b11ca0ad889abcd` 的代码交付边界，以及当日 ClearML 实时任务记录核验。软件测试通过不代表论文效果成立。

## 尚缺实验与前置条件

| 项目 | 当前证据与缺口 |
| --- | --- |
| 主协议三种子训练 | 历史 1337/2027/3407 身份训练已完成，但属于旧开发协议；不能替代原始全类别 score≥0.05/top64 主协议的训练、冻结与评价。 |
| 学习优先级与校准 | 完整连续序列教师、优先级训练、独立存在分数校准及冻结回执仍缺；不可使用局部、近零标签教师冒充完整训练。 |
| V2V4Real 原生链路 | 已完成四 A100 环境盘点；尚缺原生 PointPillar GPU 验证、真实权重/点云短序列闭环及独立评价。 |
| 主方法与消融 | 主协议下固定顺序、上界优先、学习优先及恢复/历史关闭的正式比较未完成；既有恢复负结果必须保留。 |
| 基线与资源对照 | MHT K=4 全 val 对照暂停，未取得完整新增原生指标；部分公开算法实现及原协议复现仍缺。不得以通用跟踪器替代。 |
| 正式双数据集与表图 | 尚缺主协议真实逐序列结果、三种子统计、配对 bootstrap、校准/恢复/总资源曲线及论文表图的正式数据。SPD val 为探索性，V2V4Real official_test 仅用于最终评价。 |

缺口判断不是逐算法完整验收；详细代码未完成项以 `8070964` 的 `paper-commit-receipt.json` 为准。本次没有重跑已完成的旧协议身份训练。

## 已启动的下一项

- 任务：`RBF V2V4Real pointpillar-smoke bb83b2cdbaff four-A100`
- ClearML ID：`eaf3e357a6a749fc8e541d5f1de1ec2c`
- 项目：`Thesis/Recover-Before-Fuse/Training`
- 队列：`GPU4-A100`
- 实际 worker：`10.100.34.18-A100:gpu4,5,6,7`
- 查询状态：`in_progress`；2026-09-17 03:12:23 UTC 的记录已包含 `Starting Task Execution` 和 `RBF_POINTPILLAR_BOOTSTRAP`，正在安装任务私有环境依赖。
- 另一组 0–3 号卡正在执行其他任务；未修改、停止或占用该组任务。未使用与之重叠的八卡 worker。
- 脚本：`tools/event_track_v2x/run_v2v4real_pointpillar_smoke.py`
- 脚本 SHA-256：`bb83b2cdbaff804108f58f921c85e3ff085d009881cc4ffe107327a05af24756`
- 官方源码固定提交：`5a821e13753bafc611f95c47bc1a306acdcb0f7c`，14 个文件逐一校验哈希。

本项验证真实体素化实现、PointPillar 四卡逐卡前向和 CPU 旋转 NMS，使用 37 个合成点及随机权重。不是 DDP 训练、完整检测器复现或真实数据性能实验；`paper_eligible=false`、`dataset_read=false`、`checkpoint_loaded=false`。提交与启动不表示通过，需最终 `RBF_POINTPILLAR_SMOKE` 报告和任务终态共同确认。

## 部署与验证边界

仅向内部管理机既有隔离目录传输四个已有脚本，上传后校验全部 SHA-256；提交器校验空闲且不重叠的四 A100 资源、脚本指纹和重复任务。没有修改源码、更新原工作树或提交推送代码。

本地定向 pytest 重跑未执行成功：系统 Python 缺少 pytest，原 `.venv/bin/python` 已不存在；不计为测试通过。本次使用未修改的已冻结脚本，GPU 运行结果待任务回报。

提交命令（管理机既有认证环境；重复调用会检查同名任务）：

```sh
/home/lbin/miniconda3/bin/python /home/lbin/Desktop/rbf-v2v4real-pointpillar-20260914.uhCEJS/tools/event_track_v2x/submit_v2v4real_a100_runtime.py \
  --probe-kind pointpillar --execute \
  --probe-sha256 bb83b2cdbaff804108f58f921c85e3ff085d009881cc4ffe107327a05af24756
```
