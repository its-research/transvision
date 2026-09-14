# 研究代码迁移与本地回归结果

2026-09-12 已完成研究代码迁移。论文构建仍留在
`/Users/lbin/Desktop/Codes/thesis`；研究开发在本 transvision 仓库进行。
没有提交或推送 Git，没有创建或发布 ClearML 任务。

## 迁移验收

58 个研究源码及测试文件，共 1,194,848 字节，已移到
`legacy/thesis-research-20260912/`。全部文件的原始 SHA256、权限和原路径移除状态
复核通过；8 个共享构建依赖／测试文件仍在论文仓库。
范围内剩余研究源码为 0，不代表论文仓库完全没有构建代码或历史证据脚本。

原始文件可按[迁移清单](../../legacy/thesis-research-20260912/migration-manifest.json)
逐项恢复。[归档说明](../../legacy/thesis-research-20260912/README.md)列出恢复与运行边界。

## 验证结果

下表是迁移完成时的历史回归记录。后续新增的缓存流和有限窗口跟踪测试不计入
此处 1007 项；其定向回归另见[有限窗口接入记录](forest-tracking.md)。

| 检查范围 | 结果 | 边界 |
|---|---|---|
| 论文 `make check` | 通过 | 合同与正文一致性，不是送审门禁放行或 PDF 重编译 |
| 论文 `make test-build` | 225 项通过 | 159 个根测试、10 个计划测试、56 个共享构建依赖测试 |
| 迁入研究代码的六组兼容回归 | 361 项通过，0 跳过 | 原始源码字节不变，显式复制依赖后在 transvision 中执行 |
| `tests/event_track_v2x/` 最终完整回归 | 1007 项通过，1 跳过 | 共 1008 项；约 357.79 秒，不是全仓测试 |
| 迁移后的 TFD 单独入口 | 16 项通过 | 是上述 361 项的子集，不另计独立覆盖 |

最终完整回归包含新增身份森林的 34 项测试，以及迁移范围／恢复／拒绝越界的 9 项
测试。唯一跳过项是 `PredictionValidation.test_official_golden_cases`：本机未安装
官方评价器运行时。不能据此声称本轮完成官方评价器端到端复验。

回执保存在本机 `work_dirs/legacy-thesis-tests/`，未外发：

- [迁移回归](../../work_dirs/legacy-thesis-tests/run-_fk5jsl9/results.json)，SHA256：
  `f37c3b13ce3a4b2da1e071151cb0d3c4e4d9e1f784766368923c181a735d5348`。
- [最终完整回归](../../work_dirs/legacy-thesis-tests/active-research-regression-final-20260912.xml)，SHA256：
  `4c625bdc6679365766cf622169a5699d63cb2d2dec257104829d8f2b3cb9ce0b`。
- [TFD 单独入口](../../work_dirs/legacy-thesis-tests/run-2ghhp5lx/results.json)。

这些回执属于本地测试记录，不是可信时间戳、外部签名或论文性能证据。

## 环境诊断与修正

最终环境使用 Python 3.12、NumPy 1.26.4、SciPy 1.14.1、Torch 2.9.1、
pytest 8.4.2、Shapely 2.0.7、Pillow 12.3.0、cryptography 46.0.5。
直接依赖固定在 `environments/event_track_v2x/requirements-research-tests.txt`。
它不是检测器训练环境的替代品。

补齐原临时环境缺少的 Pillow 后，首轮完整回归有 13 项失败，均来自旧 AUC 模块
调用 NumPy 1.26 不提供的 `trapezoid`。隔离的 NumPy 2.2.6 诊断回归有 4 项失败、
11 项错误；几何事件审计要求原 NumPy 1.26.4／Shapely 2.0.7 组合。
两轮失败回执均保留，未覆盖：

- `active-research-regression-20260912.xml`
- `active-research-regression-numpy2-20260912.xml`

最终没有升级原 NumPy 环境，也没有放宽几何审计的版本约束。
仅在 `publication_gate.py` 和 `validation_registry.py` 为梯形积分选择兼容名称：
存在 `np.trapezoid` 时使用它，否则使用 `np.trapz`。
没有修改 AUC 公式、公共字节范围、评价阈值或历史结果；最终完整回归在原锁定环境通过。

## 方法完成边界

新增[联合身份森林](identity-forest.md)支持原始观测上的联合互斥、可恢复前沿、
模型内质量估计及有限预算动作搜索。后续已接入窗口内 3D 状态重放与输出，
但完整序列和跨窗口稳定 ID 生命周期仍未完成。
新增模型的三个种子训练、同资源强基线及 SPD／V2V4Real 方法效果实验仍未完成。
本次没有读取真实 test 数据；V2V4Real 官方 test 的后续授权与 train 内选择约束不变。
