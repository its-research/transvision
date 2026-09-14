# JPDA／PKF 持久化接入验证记录

2026-09-13，本轮最终 EventTrack 全套回归为 **1538 通过、0 失败、1 跳过**，
命令总耗时 340.90 秒。新增代码、测试、文档和回执均在 transvision。论文项目
的 Makefile、latexmkrc 及构建工具未迁移；没有新增论文项目研究代码。

该记录只验证本地实现和合成夹具，不是新方法真实训练或论文性能结果。
接口、数学定义及设计差异见[实现说明](probabilistic-tracking-v2.md)。

## 1. 最终回归

在 transvision 根目录执行：

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/private/tmp/eventtrack-v2-checks.bYZm1l/bin/python -m pytest -q -p no:cacheprovider \
  tests/event_track_v2x \
  --junitxml=work_dirs/recover-before-fuse/probabilistic-full-regression-20260913.xml
```

研究环境为 Python 3.12.14、NumPy 1.26.4、SciPy 1.14.1、Torch 2.9.1 和
pytest 8.4.2。临时环境路径可能失效；依赖记录见
[研究测试版本文件](../../environments/event_track_v2x/requirements-research-tests.txt)。

[最终 JUnit 回执](../../work_dirs/recover-before-fuse/probabilistic-full-regression-20260913.xml)
包含 1539 项，套件计时为 340.864 秒。唯一跳过项是
`PredictionValidation.test_official_golden_cases`，原因是研究环境未安装官方
评价器运行时。此前独立 Python 3.10 环境已运行该测试及真实指标引擎合成案例，
见[评价器验证](evaluator-runtime.md)。本轮未重跑独立评价器环境，不将其计入
上述通过数，也不将研究环境跳过误报为评价器整体未验证。

## 2. 覆盖范围

- exact／LBP 与三种 Gaussian 更新器的组合；双端到达、后续帧 ID、源帧互斥。
- 原始父路径按身份根求和，与独立条件森林转换的因子摘要一致。
- 检测分数不乘未匹配质量；联合 MAP 与逐节点 Bayes 的对称新生反例。
- 合法迟到及源时刻超前的原始测量投影，不反向传播已压缩先验；拒绝未来输入。
- 状态写入后故障的事务回滚、推断容量拒绝、重复接收幂等及源时刻保留。
- 正常关闭重开与锚点／Gaussian 状态篡改检测；不回写旧预测。
- 60 帧、120 个观测，在第 30 帧后关闭重开，三种更新器的预测与连续运行一致。
- 实际训练过的同一夹具检查点接入九个后端，逐事件原始势函数摘要完全一致。
- 正式 CLI 拒绝部分 SPD val 日程、夹具检查点及不成对的检查点路径／摘要参数。

本轮修改了共享持久化基类的审计扩展点，以及共同 V2 回放入口。因此最终采用
完整 EventTrack 回归，不只验证新增的 JPDA／PKF 文件。源码在最终回归后未改。

## 3. 分阶段记录与初次失败

| 回执文件，均位于 `work_dirs/recover-before-fuse/` | 结果 | 范围 |
| --- | --- | --- |
| `probabilistic-tracking-regression-20260913.xml` | 62 通过、1 失败 | 初版持久化基线及原接收测试 |
| `probabilistic-tracking-regression-20260913-v2.xml` | 64 通过 | 显式区分两种硬锚点解码器后 |
| `probabilistic-cache-regression-20260913.xml` | 17 通过 | 新 V2 接收及共享回放 |
| `probabilistic-learned-replay-20260913.xml` | 26 通过 | 实际夹具训练、九后端回放及接收 |
| `probabilistic-temporal-regression-20260913.xml` | 18 通过 | 最终跟踪源码，含长序列重开 |
| `probabilistic-full-regression-20260913.xml` | 1538 通过、1 跳过 | 本轮最终完整 EventTrack 套件 |

初次失败来自测试误认为对称场景的逐节点 Bayes 解码必然保留一个旧根匹配。
实际边缘满足新生概率略高于匹配概率，因此该损失允许两个新生。实现现已显式
提供联合 MAP 与逐节点 Bayes 两个选项，默认联合 MAP；测试同时保留两者的
不同结果。该选择基于可穷举反例，未使用真实 val 指标调参。初次失败回执未删除。

以上阶段范围重叠，计数不能相加。60 帧测试证明正常重开的预测一致性，不是
长时间服务稳定性、崩溃恢复、恒定内存或真实场景鲁棒性证明。

## 4. 源码及回执 SHA256

```text
transvision/models/event_track_v2x/persistent_probabilistic_tracking.py
4d57a1b1020ca858bf207b567b63cae75bdfda366508522fb343acd4165f0664
transvision/models/event_track_v2x/persistent_forest.py
c0ad339d02d700cf98ea449a3f23d485b3995f26a59c41cc704acffe5563850c
tools/event_track_v2x/run_persistent_forest_v2.py
b1f61ab3bf58822669d3ec69bb7ec151fa9f1bb811d3b66d02704670cb99161a
tools/event_track_v2x/run_probabilistic_tracking_v2.py
96eb8d68baec6f5e34c5d266e68d4455a9313e4322fd7aa62f0201f4419fcbd8
tests/event_track_v2x/test_persistent_probabilistic_tracking.py
54b77a6c215b1b38d05758679cde18a5ce8a0a3d092593f99a1a73acf036ac24
tests/event_track_v2x/test_probabilistic_tracking_cache.py
a25e2e44d80c8ec63af39954be44fc015ca11d65007424a4136cb7395dfc420f
tests/event_track_v2x/test_train_forest_identity.py
10b4e33b732719afb554c68aa02389c98135e0c099215d83a9eb32399778ab4e
probabilistic-full-regression-20260913.xml
c67043edfbfb80f140c99f30657eb6bbf55cbeaf74a370523cef97df02257c86
probabilistic-temporal-regression-20260913.xml
f8b6f2981d6fa31e3f623b74c9fdf76de0f8f90c1a3a748bb4b3eba8fb103f23
probabilistic-learned-replay-20260913.xml
565acbc940ee4baf31302baf5d55235bc358750f305b2aa3972d680ba4ccddbb
```

## 5. 论文完成边界

本轮没有读取真实 GT／train／val／test 载荷，没有远端同步、Git 提交、推送或
ClearML 发布。此前远端源码同步的权限限制未通过本轮本地接入解除。

尚需真实全量身份模块和预算分配模块训练、经典 MHT／公开方法复现、同状态
时间协议的恢复消融、同资源曲线、完整 SPD val，以及冻结后 V2V4Real 官方
test。只报告 car；SPD test/test_A 继续排除。当前代码与夹具证据不能替代这些
实验，也不能支持「已优于强单端或同算力多假设基线」的结论。
