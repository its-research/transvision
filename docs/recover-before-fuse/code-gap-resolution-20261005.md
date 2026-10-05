# Recover Before Fuse 代码缺口处理

依据 thesis 的 `experiments.md`、`method.md` 与 2026-10-05 用户决定：V2V4Real
使用官方划分、合并 vehicle 评价；SPD 保留 car 评价。此记录只确认实现与软件检查。

## 已补代码

| 缺口 | 实现 | 软件证据与范围 |
|---|---|---|
| recovery-off 仅能 bound 分配 | `recovery_off_allocation.py`：受限历史支持下的 teacher、learned；独立源码/配置 checkpoint 绑定 | teacher→export→fit→load→learned 合成路径；不复用 unrestricted 模型 |
| recovery-off learned 无独立排序检查 | `rbf_independent_recovery_off_learned_trajectory.py` 与 CLI | 从前一事件提交历史重建限制、候选、18 特征、NumPy MLP 排序和成本；连同执行路径 69 项通过 |
| bound 消融尚无完整独立 CPU 入口 | `prepare_recovery_off_full_independent_CPU.py` 与结构/动作检查器 | 原冻结生产器下的全链夹具及负例 92 项通过；未启动真实全队列验收 |
| V2V4Real 严格 Car 与新 vehicle 口径混用 | `paper_evaluation_policy.py`、GT 生成与独立原码 oracle、`fit_v2v4real_vehicle_calibration.py` | 保留 raw_class，按官方规则合并；校准只使用 train，旧 Car 回执不自动升级 |
| 新校准不能接到已有原始候选 | `build_native_paper_cache.py --raw-cache` | 仅重新计算 calibrated scores；其他数组逐字节不变；拒绝先 NMS 的旧缓存 |
| 原始导出器只接受 train，无法产生 official_test 原始候选 | `run_v2v4real_official_test_raw_features.py` | 私有加载固定原始推断核心，绑定 checkpoint/test projection；保留 3,986 个源帧及空候选帧，18 项合成检查通过 |
| 原始检测输出缺少合法 9D 缓存接线 | `build_v2v4real_raw_native_cache.py` | 支持独立 train/test envelope；显式因果速度和完整协方差合同，空帧保留，34 项检查通过；执行源码在开始、转换后和读回后核验 |
| 原生指标与 RBF 世界坐标输出未接通 | `evaluate_v2v4real_native_vehicle.py --convert --evaluate-converted` | 消费显式姿态/官方帧映射，保留全 9 序列 1,993 帧；调用固定官方 3D IoU 评价器，AMOTP 方向为高优 |
| 公开基线仅检查进程退出 | `public_native_outputs.py`、`run_public_baseline.py` | CoopTrack 显式保存原始预测、收集原生评价目录；DMSTrack 九序列格式与摘要读取；SparseCoop 旧收集保留 |

恢复关闭 teacher/learned 的只读源码、命令、原始失败日志与 69 项检查回执：
`test/recover-before-fuse/source-freezes/rbf-recovery-off-teacher-learned-code-v1-20261005/software-preparation.json`。
原 bound 模式在 8 组预算/宽度配置的 24 个事件中预测及 audit 字节保持一致。

bound 完整 CPU 检查器准备回执：
`test/recover-before-fuse/source-freezes/rbf-recovery-off-bound-full-independent-CPU-v1-20261005/qualification.json`。

Vehicle GT/校准/缓存/协议相关检查 75 项通过，另有 17 项独立 metric 检查通过；
两个最初因环境失败的既有 pipeline 回归已在现成 torch260 环境中重跑，2 项通过。
训练计划也已纳入新类别规则模块的源码哈希；对应检测/训练回归 2 项再次通过。
原始失败日志保留。公开基线检查 55 项通过，包含直接执行固定官方 `save_results`
函数生成文本再由新收集器读取的对照。以上套件可能覆盖相关功能，不相加为实验数量。

原生坐标转换/评价的 60 项检查通过，包含真正调用固定官方评价器的完整空帧夹具、
完美/平移框、0.25 IoU 阈值边界和输入篡改拒绝。原参考公式、匹配规则、阈值未改；
原 numba 函数在 `NUMBA_DISABLE_JIT=1` 下执行相同 NumPy 函数体，运行记录明确保留此设置。
首轮组合测试受 macOS 沙箱共享内存限制，最终同一组测试在许可环境下通过；失败日志保留。

## 已有实现，缺运行或资产

M0–M4、主路径 teacher/learned、资源扫描器、JPDA/PKF/MHT 后端和论文表图入口均已存在。
`scan_paper_resources.py` 可冻结候选×三种子并逐项启动完整 schedule 回放和独立评价；
尚缺资源清单或真实产物不等于缺少扫描算法。

仍需真实教师数据、训练后的 checkpoint、完整回放、独立验收及同资源比较。
V2V4Real 新 vehicle GT、校准、姿态/官方帧映射、运动及协方差合同须绑定真实输入；
通用接口或合成夹具不会生成这些证据。已知 train/test 重叠继续披露；物理会话映射
不再作为用户授权的官方基准路径前置条件。

公开基线还缺可复现运行环境、实际权重/特征和输入映射。Long-SCOPE 的完整作者实现
或足以复现的算法细节与权重仍未获得，未用其他跟踪器冒名补齐。

所有已有远端任务、失败记录和冻结产物保留，代码准备未重启实验。
2026-10-05 用户随后明确授权“阶段性 C+P ALL，然后继续完成后续实验”；
该次阶段性提交推送的实际结果另记执行台账，后续仍不自动提交推送。

## 本轮代码与证据索引

汇总回执：`test/recover-before-fuse/receipts/rbf-remaining-code-gap-resolution-20261005.json`。
各模块源码快照、命令、测试日志与子回执均在汇总回执中逐项绑定 SHA256；
实验族清单只检查入口存在与语法，不把它的 14 个类别当作已完成实验数。

下一步依赖顺序为：读取已有 GPU 任务的真实终态和产物 → 完整独立 CPU 验收；
V2V4Real 则新建 vehicle GT 与 train-only 校准，生成 GT-free official_test 原始候选，
接入经验证的因果速度/完整协方差和帧映射，再运行模型与官方评价。
这些任务未因本轮软件检查通过而被标为完成。
