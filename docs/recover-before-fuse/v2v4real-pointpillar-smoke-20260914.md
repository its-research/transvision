# V2V4Real PointPillar：CPU 验证完成，四卡检查待部署授权

已在 `transvision` 实现任务独立的 PointPillar 运行检查入口，并完成固定官方网络的 CPU 前向验证。72 项相关测试通过。新的 A100 检查尚未部署、未入队，也没有产生真实检测缓存、参数训练或跟踪指标。

## 已验证与未验证内容

CPU 集成测试校验官方提交 `5a821e13753bafc611f95c47bc1a306acdcb0f7c` 的 14 份源文件哈希，再加载未修改的 PointPillar 及配置解析辅助函数。配置用 `yaml.safe_load` 读取，显式调用 `load_point_pillar_params`，不执行 YAML 指定的任意函数。测试不加载官方数据集 loader。[固定网络](https://github.com/ucla-mobility/V2V4Real/blob/5a821e13753bafc611f95c47bc1a306acdcb0f7c/opencood/models/point_pillar.py)

在固定随机种子 1337、手工构造的 3 个体素输入上，分类头 `psm` 的形状为 `[1,2,50,88]`，回归头 `rm` 为 `[1,14,50,88]`。两次前向输出逐元素相同，全部有限，输入张量没有被修改。

这里的体素张量由测试直接构造。CPU 前向成功不证明 spconv 体素化、旋转 NMS、GPU 前向、DDP 或预训练权重可用。原论文运行环境也未被复现。上一轮已完成的四卡小矩阵检查见 [A100 运行环境记录](v2v4real-a100-runtime-20260914.md)，不能将两项检查拼成一次完整检测器验收。

## 新运行入口及检查条件

`tools/event_track_v2x/run_v2v4real_pointpillar_smoke.py` 是待执行入口。它先要求 worker 恰好分配 4 张 A100，再在任务目录创建独立 venv。venv 复用基础镜像中的 PyTorch，以下直接依赖安装到 venv，不修改共享镜像配置：

| 依赖 | 固定版本 |
| --- | --- |
| NumPy | 1.26.4 |
| SciPy | 1.14.1 |
| PyYAML | 6.0.2 |
| spconv-cu126 | 2.3.8 |
| cumm-cu126 | 0.7.11 |
| Open3D | 0.19.0 |
| Shapely | 2.0.7 |

spconv 的官方元数据要求 `cumm-cu126>=0.7.11,<0.8.0`，不能混用最新的 cumm 0.8.x。本轮选择的版本有 Python 3.12 的 Linux wheel；这不等于已经通过实际安装验证。[spconv](https://pypi.org/project/spconv-cu126/2.3.8/)、[cumm](https://pypi.org/project/cumm-cu126/0.7.11/)

安装仅使用官方 PyPI，并生成包含间接依赖和下载哈希的 pip report。执行前后比较基础环境中 NumPy、PyTorch、PyYAML、SciPy 的版本；这项比较只覆盖四个列明的包，不是整个文件系统的不可变证明。任务仍继承基础镜像的其他依赖，因此不能把 venv 称为脱离基础镜像的完整环境包。

运行任务从固定 GitHub 提交获取并校验 14 份源文件，不在 ClearML 脚本里嵌入上游源码。检查使用原版 `SpVoxelPreprocessor`、`PointPillar` 和 `nms_rotated`，预期验证：

1. 37 个合成点被分为 3 个体素，单体素最多保留 32 点；坐标、计数和特征必须与明确的参考数组完全一致。
2. 两个重合框保留高分者，远处第三个框保留，旋转 NMS 返回 `[0,2]`。这是 CPU NMS 检查，不是 CUDA NMS 性能结果。
3. 四张 A100 分别运行相同随机权重的官方网络，检查头部尺寸、有限值和跨设备输出差异。容差预先固定为 `atol=1e-6、rtol=1e-5`。

上述三步在本轮均尚未于 A100 执行。即使全部通过，也只是冻结检测器接入前的运行条件检查；仍需取得实际权重、验证来源、接入真实 train 点云、完整后处理和 DetectionCacheV2。

## 远端部署阻塞及解除条件

计划经内部控制机 `10.100.35.112` 向 ClearML `10.100.35.118` 的 `GPU4-A100` 队列提交。平台审批拒绝了向控制机 SCP 发送私有源码，理由是尚缺对这个传输方式和目的地的明确授权。拒绝后没有改用其他手段传输同一源码，也没有创建新的 ClearML 任务。

本地没有已配置的 ClearML SDK/配置文件，当前工具中没有 ClearML 连接器，不能直接完成替代提交。已创建的专属远端目录为 `/home/lbin/Desktop/rbf-v2v4real-pointpillar-20260914.uhCEJS`；本次源文件部署未完成。

需授权的文件只有以下 4 份，共 `33,990 B`，不含数据集、GT、权重、点云、预测或上游源码正文：

| 文件 | SHA-256 |
| --- | --- |
| `run_v2v4real_pointpillar_smoke.py` | `bb83b2cdbaff804108f58f921c85e3ff085d009881cc4ffe107327a05af24756` |
| `submit_v2v4real_a100_runtime.py` | `3cb085093bdaa409d4b3f2dcccab99c93a3532ecaedf2435de85a7cc06d9f79a` |
| `submit_forest_identity_ddp.py` | `ca78afc351851c12b7144582f83df29f62624736df6aba25f08bd8af9a44a2c1` |
| `probe_v2v4real_a100_runtime.py` | `082fe4c6313fba1ff6b5f2f7d59e0be54dc0b7f6bc3cd9cfac868cd2a73896ab` |

四个文件都位于 `tools/event_track_v2x/`。复用的历史身份训练提交模块只提供已核实的镜像、项目和容量检测函数；新入口只选 A100，不执行历史的混合硬件训练矩阵。

取得明确部署授权后，先核对四个远端文件的哈希，再通过新入口指定 `--probe-kind pointpillar` 和上述探针 SHA-256 提交。若入队返回不确定，按打印的任务 ID 查询，不盲目重试。不能复用旧库存检查的完成状态作为这次前向检查的验收结果。

另外，Mac 仍处于锁屏状态，官方模型包下载尚需解锁后通过官方公开页面继续。本次未绕过锁屏或 Box 认证，也未依据不完整来源信息冻结模型为正式论文检测器。

## 证据位置

测试报告与待部署清单保存在 `work_dirs/recover-before-fuse/v2v4real-pointpillar-20260914/`。`smoke-tests-v3.xml` 的 SHA-256 为 `0f3513bec2a190eb2111b130cd4e20f7f54425b5539466da26c3fb0903e752c5`，覆盖 72 个测试、0 个跳过，包括真实官方网络的 CPU 前向。清单中的 `a100_pointpillar_forward_verified=false` 与 `clearml_task_created=false` 保留实际状态。

固定参考源码位于本地独立目录 `/private/tmp/rbf-v2v4real-runtime-build-20260914.V3C7R6/official-source`，不放入项目交付快照。本轮未更改两个仍在运行的 SPD 教师进程所绑定的源码，未重复训练三个已完成的身份模型种子。
