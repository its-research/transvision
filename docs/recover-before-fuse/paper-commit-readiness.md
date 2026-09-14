# Recover Before Fuse 提交前说明

本次交付是可独立审阅的论文软件实现增量，不是整篇论文代码完成或实验收益证明。完整缺项见 [验收矩阵](paper-implementation-acceptance.md)。没有执行 Git 暂存、提交、推送或正式训练。

## 提交范围

- 主协议：原始全类别 `score >= 0.05 / top64`，car 仅在评价层筛选；旧 car-first 协议和证据独立。
- 在线方法：合法动作解码、原始状态重建、持久化组件、三种展开策略、历史消融、资源与恢复审计。
- 软件流水线：连续序列训练、三种子配置、训练内检查点选择、CPU/Gloo 验证、原生缓存适配、独立评价及表图生成器。
- 基线入口：内部跟踪后端、复用原算法的 clean-link learned+CI / M0–M4、源码锁定的公开基线调用器，以及 CPU 资源扫描。
- 回归测试、配置、环境与验收文档。没有真实数据、权重、生成的预测、性能数字或大型测试产物。

本次格式化仅覆盖修改或新增的 Python 文件。`setup.cfg` 关闭短语句合行，与 flake8 的 E701 规则一致；`.pre-commit-config.yaml` 为 flake8 6 固定 Python 3.10，避免 Python 3.12 f-string 分词误报。CLI 初始化仓库路径以及评价测试检查可选依赖之后的导入，采用逐条 `E402` 标注，不全局关闭检查。

四个历史配对模块 `tracking_v2.py`、`tracking_mechanisms_v2.py`、`tracking_birth_score_v2.py`、`predicted_association_v2.py` 保持基线字节不变，不修改历史哈希针脚。

## 验证与复现

最终证据见 [提交回执](paper-commit-receipt.json)：研究回归 3222 passed、6 skipped；独立评价 36 passed；补跑官方指标 golden case 1 passed；11 个 CLI 冒烟及适用的静态、格式和文本检查通过。该回执记录源码哈希、实际测试数量和边界，不充当真实实验回执。

在仓库根目录执行。研究环境使用 Python 3.12 和 `environments/event_track_v2x/requirements-research-tests.txt`。评价环境独立使用 Python 3.10；macOS arm64 使用 `environments/event_track_v2x/requirements-evaluation-macos-arm64-py310.lock`，其他平台必须重新验证兼容依赖，不能直接套用该平台锁。

以下环境变量需设为本机已准备环境的绝对路径；输出目录必须新建，保留历次失败回执：

```sh
export RBF_RESEARCH_PYTHON=/path/to/research/bin/python
export RBF_EVALUATOR_PYTHON=/path/to/evaluator/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RBF_CHECK_OUTPUT=$(mktemp -d /tmp/rbf-code-check.XXXXXX)

"$RBF_RESEARCH_PYTHON" -m pytest -q -p no:cacheprovider --tb=short \
  --ignore=tests/event_track_v2x/test_train_inference_evaluator.py \
  --junitxml="$RBF_CHECK_OUTPUT/research.xml" tests/event_track_v2x

"$RBF_EVALUATOR_PYTHON" -m pytest -q -p no:cacheprovider --tb=short \
  --junitxml="$RBF_CHECK_OUTPUT/evaluator.xml" \
  tests/event_track_v2x/test_train_inference_evaluator.py \
  tests/event_track_v2x/test_paper_evaluation.py

git diff --check
```

提交检查按 `.pre-commit-config.yaml` 中的版本运行。本次直接执行了对应钩子的命令，并未安装 Git hook。研究回归和评价回归有重复测试，不相加声称独立覆盖数量。跳过原因见最终回执。

主方法入口为 `tools/event_track_v2x/run_paper.py`；配对基线和资源扫描分别见 [配对基线说明](paper-pair-baselines.md) 和 [资源扫描说明](paper-resource-scan.md)。两个数据格式的准备、训练冒烟、冻结、推断及独立评价集成由 `test_paper_end_to_end.py` 和 `test_paper_pair_baselines.py` 验证，均显式标记为 fixture。

## 不应随本次提交宣称完成的事项

- 独立恢复关闭消融、官方 V2V4Real 原始身份标注自动转换、完整校准冻结编排与正式教师全集回执。
- Graph Lap-CoMOT、Long-SCOPE 完整算法；CoTrack 的可核验完整算法资料或官方代码。
- 真实 PointPillar 权重、锚框/原生编译依赖及真实短序列验证；GPU/NCCL、多卡全量训练和正式双数据集实验。
- 公开基线真实复现、配对基线任意异步流及资源扫描自动调度、官方 pooled 非线性指标的重采样置信区间。

这些是整体计划的未完成项，不以通用跟踪器、模拟成功回执或描述性统计替代。建议提交标题：`feat(event-track): add audited paper protocol and software pipeline`。提交时应使用隔离工作树并包含新增文件，不只提交 `git diff` 显示的已跟踪文件。
