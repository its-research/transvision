# SPD 指标评估环境与合成案例验证

独立的 Python 3.10 评估环境已通过封存版本、源码指纹和 7 个可手算案例的检查。
相关回归为 131 通过、0 跳过。这是 SPD 指标适配器的工程验证，不是新方法的真实
数据集结果，也不是官方排行榜一致性证明。研究代码位于 transvision；论文
Makefile、latexmkrc 和构建脚本保持原位。

## 1. 环境与来源

研究测试继续使用原有 Python 3.12 环境；评估使用独立的 CPython 3.10.19 环境。
[TrackEval 1.0.0 的包元数据](https://pypi.org/project/trackeval/1.0.0/)
限制 Python 版本小于 3.12。新增入口不会通过修改评估公式来适配研究环境。

固定的 8 个评估依赖版本为 NumPy 1.26.4、SciPy 1.15.3、pandas 2.3.3、Shapely
2.0.7、pyquaternion 0.9.9、motmetrics 1.4.0、nuscenes-devkit 1.2.0 和 TrackEval
1.0.0。验证器还校验 motmetrics、nuscenes、trackeval 三个完整 Python 源码树，
以及现有 SPD 适配器的 SHA256。仅检查包版本或成功 import 不足以通过验证。

本机运行路径：`/private/tmp/eventtrack-evaluator-v1.zT0Eh9/venv/bin/python`。
运行时报告的平台为 `macOS-27.0-arm64-arm-64bit`。临时环境可能被系统清理；
环境重建依据为仓库中的锁文件和审计记录，不依赖临时目录长期存在。

- [版本输入](../../environments/event_track_v2x/requirements-evaluation-v1.in)：说明直接依赖，不锁定全部 wheel。
- [macOS arm64 / Python 3.10 锁文件](../../environments/event_track_v2x/requirements-evaluation-macos-arm64-py310.lock)：50 个实际安装 wheel 的公共 URL 和 SHA256。
- [安装审计](../../environments/event_track_v2x/evaluation-wheel-install-audit-20260913.json)：安装来源和同版本 wheel 替换记录。

50 个安装版本与审计记录逐项一致，`pip check` 通过。锁文件不包含 Python
解释器、操作系统、pip 或 setuptools，也不适用于 Linux。尚未在第二个全新
环境中按锁文件重新安装；不能据此宣称跨机器可复现已获验证。

### TrackEval 与官方源码的关系

PyPI 的 `trackeval==1.0.0` 链接到 `kovalp/TrackEval`，不能将整个包称为与
`JonathonLuiten/TrackEval` 字节完全一致。此次逐字节核对采用
[官方仓库的固定提交](https://github.com/JonathonLuiten/TrackEval/tree/12c8791b303e0a0b50f753af204249e622d0281a)。

| 核对文件 | 安装版本相对固定官方提交的差异 |
| --- | --- |
| `metrics/hota.py` | 5 处 `np.float` 改为 `float` |
| `metrics/identity.py` | 3 处 `np.int` 改为 `int` |
| `metrics/_base_metric.py` | 字节完全一致 |
| `_timing.py` | 字节完全一致 |

[NumPy 官方说明](https://numpy.org/doc/1.20/release/1.20.0-notes.html)确认上述
别名对应 Python 内置类型。验证器检查原文件哈希、安装文件哈希、替换次数和
替换后的字节一致性，拒绝额外差异。这是四个核心文件的核对，未声称全包所有
模块都与官方仓库一致。旧封存适配器中的「unmodified official metric engines」
表述必须按本节限定理解，不能据此作全包来源声明；本次未修改封存适配器。

### 本机兼容性处理

初次安装的 SciPy 1.15.3 macOS 14 wheel 出现 `dlopen` 错误：
`section '__DATA/__thread_bss' has a zero-fill section type, but offset field is not zero`。
替换为同版本的官方 macOS 12 arm64 wheel 后，实际指标运算通过。两个 wheel
的哈希均有记录；没有降级版本或修改 SciPy、TrackEval、nuScenes 指标源码。

TrackEval 和 nuscenes-devkit 的依赖分别引入 `opencv-python` 与
`opencv-python-headless`，两者在本环境均为 4.11.0.86。它们共享 `cv2` 命名空间；
`pip check` 不检测这类文件重叠。本次 HOTA、Identity 和 nuScenes 跟踪计算不使用
OpenCV 算法，因此此环境只作为受限的评估运行环境，不作为通用视觉开发环境。

## 2. 七个合成案例

所有目标均为 car，使用原生时间戳、不插帧。新增 4 个案例使用 4 × 2 × 1.5 米
直立框；每帧间隔为 150100 微秒。数值期望独立写出，再交给实际引擎验证，
不是把一次运行的输出复制为期望。

| 案例 | 可手算约束 |
| --- | --- |
| 完美轨迹 | 原封存 golden case |
| 全空预测 | 原封存 golden case |
| 身份切换 | 原封存 golden case |
| 2 帧完美序列与 4 帧身份切换序列共同汇总 | HOTA = √(2/3)，AssA = 2/3，IDF1 = 2/3，IDTP/IDFP/IDFN = 4/2/2 |
| 单 GT 对应两条重复预测轨迹 | HOTA = √(1/2)，AssA = 1，DetA = 1/2，IDF1 = 2/3 |
| 预测框沿 x 方向偏移 1 米 | BEV IoU = 3/5，HOTA = 12/19，IDF1 = 1；nuScenes AMOTA = 1、AMOTP = 1 米 |
| 预测框旋转 90°、中心不变 | BEV IoU = 1/3，HOTA = 6/19，IDF1 = 0；nuScenes AMOTA = 1、AMOTP = 0 米 |

后两个案例说明指标口径差异：本适配器的 HOTA 使用 19 个 BEV IoU 阈值，
IDF1 使用 0.5 阈值；nuScenes 使用严格小于 2 米的平面中心距离匹配。
nuScenes AMOTP 在这里是米，越小越好。此结论不能套用到其他数据集同名指标。

跨序列案例检查 `combine_sequences` 的计数汇总，避免把序列指标算术平均
冒充总体指标。检查使用硬异常和数值容差 `1e-10`；Python `-O` 会使封存
golden case 中的断言失效，因此验证器在创建输出前直接拒绝优化模式。

## 3. 复现方式

以下命令在 transvision 根目录运行。仅适用于 macOS arm64，且已安装 Python
3.10；不要复用训练环境。安装会从公共 Python 包站点下载锁定的 wheel。

```sh
RBF_EVAL_ENV="$(mktemp -d /private/tmp/rbf-evaluation.XXXXXX)"
python3.10 -m venv "$RBF_EVAL_ENV"
"$RBF_EVAL_ENV/bin/python" -m pip install --require-hashes \
  -r environments/event_track_v2x/requirements-evaluation-macos-arm64-py310.lock
"$RBF_EVAL_ENV/bin/python" -m pip check
```

随后运行不读取真实 GT 或预测流的严格自检。输出目录必须是新目录；如已存在，
选用新的输出名，保留原收据。

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLBACKEND=Agg MPLCONFIGDIR="$RBF_EVAL_ENV/matplotlib" \
XDG_CACHE_HOME="$RBF_EVAL_ENV/cache" \
"$RBF_EVAL_ENV/bin/python" tools/event_track_v2x/verify_evaluation_runtime_v2.py \
  --output work_dirs/recover-before-fuse/evaluator-selftest-local
```

默认检查安装文件和已封存的参考哈希。若还要现场重新核对上游原始字节，
从上文固定官方提交取得表中 4 个文件，保留相对目录结构，再增加
`--reference-root <保存这些文件的目录>`。不要使用随时间变化的 master 文件。
此次正式自检提供了该参数，`reference_bytes_rechecked=True`。

成功时生成 `receipt.json` 和四份绑定哈希的计划、环境、案例文件。任一校验
失败时退出非零，不产生完成收据；已经创建输出目录的运行会保留
`failure.json`。不要将失败目录或 `paper_eligible=False` 的收据登记为论文结果。
输出含计时字段，重复运行的文件哈希不要求一致；收据应绑定各自运行的文件。

## 4. 本次验证记录

最终代码的自检收据：
[evaluator-selftest-20260913-v2](../../work_dirs/recover-before-fuse/evaluator-selftest-20260913-v2/receipt.json)。
它确认 8 个版本、3 个源码树、封存适配器和 7 个真实引擎合成案例均通过。
初次自检目录也保留；最终版本额外拒绝布尔值冒充数值。

相关回归覆盖 7 个测试文件：严格验证器、跟踪评估、环境合同、适配器、源消融、
机制诊断及出生分数诊断。结果为 **131 通过、0 失败、0 跳过，4.57 秒**，见
[JUnit 记录](../../work_dirs/recover-before-fuse/evaluator-regression-20260913.xml)。
其中验证器的失败路径使用夹具；不能把这类单元测试冒充真实指标运行。
真实指标证据来自独立自检和已执行的 `test_official_golden_cases`。

新增验证器在原 Python 3.12 研究环境另跑了 29 项控制流单元测试，全部通过；
这些测试不要求安装评估包，不是第二个环境的真实指标验证。两份文档的中文
技术文案检查和 `git diff --check` 通过。

```text
tools/event_track_v2x/verify_evaluation_runtime_v2.py
6feeaded0cae48a5e34223fb82dc2f15f9687a244e8afbc42ab077bfb7f0f79c
transvision/models/event_track_v2x/tracking_evaluation_v2.py
659fbf0cafad0e942eb2bb72023f3a4b9cc4c4cfca6883503b48f3c5c3c9b7ec
requirements-evaluation-macos-arm64-py310.lock
86ee85c44479a5fce226469ba71fe0a015730c756d6423089348b90d3c0c6e1e
evaluator-selftest-20260913-v2/receipt.json
eac09c2c1c8a88e4236b4f12d190a5233f588c287d61bac17710e36824c7362b
evaluator-regression-20260913.xml
df372759e14cb546d1d0cf4c70149aafa2ddba443c210cbb8cd322d4890059d4
```

本次没有重跑完整研究套件，也没有更改历史回归计数。原 Python 3.12 研究环境
仍不安装这些评估包，完整研究回归里的历史跳过记录保持有效。

## 5. 尚未完成的论文验证

本次未读取真实 GT、预测缓存或任何 test 数据，未进行训练、远端同步、提交、
推送或 ClearML 发布。真实全量身份模块训练、计算分配训练、同资源强基线、
SPD 全 validation 及 V2V4Real 冻结后的官方 test 均未因本次自检而完成。
SPD 的测试边界不变；V2V4Real 按用户后续授权使用官方 test 进行最终评估。

当前验证仅覆盖封存的 SPD 适配器，不能替代 V2V4Real 官方协议核对。
论文结论仍需真实数据和资源匹配实验支持；项目目标保持未完成。
