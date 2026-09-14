# 训练内资源扫描与冻结

入口：`tools/event_track_v2x/scan_paper_resources.py`。只允许主协议 `rbf-all-class-top64-v1` 和 train。仅规划不会启动作业；执行必须显式调用 run。全部输出目录必须不存在。

## 输入契约

plan 的 spec JSON 需要以下字段：

| 字段                                                       | 内容                                                                                                                                              |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| protocol                                                   | 完整 PaperProtocol 字段；dataset 为 spd 或 v2v4real，split 必须 train                                                                             |
| fixture                                                    | 显式布尔值                                                                                                                                        |
| seeds                                                      | `[1337, 2027, 3407]`                                                                                                                              |
| objective                                                  | HOTA、IDF1、AMOTA 最大化，或 AMOTP 最小化                                                                                                         |
| constraints                                                | 正值上限；可用 peak_rss_bytes、state_file_bytes、p95_seconds、model_forward_calls、posterior_search_steps、action_search_steps、assignment_solves |
| cache_manifest / schedule / gt_manifest / environment_lock | 各为绝对文件 path 与 sha256；GT 只交给独立评价进程                                                                                                |
| candidates                                                 | 每项 name 唯一且只含小写字母、数字、连字符、下划线；configuration 是冻结配置文件绑定                                                              |
| candidates\[\].checkpoints                                 | 非 geometry 必需；种子字符串到 checkpoint.json 文件 path/sha256 的映射，种子必须相符                                                              |
| candidates\[\].priorities                                  | 学习优先级必需；同样按种子绑定 checkpoint.json，与该候选资源设置分别匹配                                                                          |

从 `configs/event_track_v2x/paper/resource-scan.json` 选择要物化的候选配置；不能静默修改已训练优先级的限额绑定。文件绑定指向 checkpoint.json，运行器将父目录交给现有加载器。完整、可执行的模拟输入见 `tests/event_track_v2x/test_paper_resource_scan.py`。

## 命令

以下参数是待替换的说明符，不能直接作为正式实验运行：

```sh
python tools/event_track_v2x/scan_paper_resources.py plan --spec SPEC.json --output PLAN_DIR
python tools/event_track_v2x/scan_paper_resources.py run \
  --plan PLAN_DIR/plan.json --plan-sha256 PLAN_SHA256 --output RUN_DIR \
  --python ABSOLUTE_RESEARCH_PYTHON --evaluator-python ABSOLUTE_EVALUATOR_PYTHON
python tools/event_track_v2x/scan_paper_resources.py freeze \
  --plan PLAN_DIR/plan.json --plan-sha256 PLAN_SHA256 \
  --run RUN_DIR --receipt-sha256 RECEIPT_SHA256 --output FREEZE_DIR
```

当前执行为 CPU 推断；不代表 GPU 延迟或显存测试。每项候选的每个种子启动新进程，随后用独立评价环境读取 GT。失败保留日志和 failure.json，不生成完成回执。源码、配置、种子、调度、数据及预测/评价文件绑定在运行和冻结阶段重新核对。

冻结要求三种子齐备且每个种子均满足全部资源约束。按平均目标指标选择；完全同分按候选名打破平局；无可行候选不冻结。plan 必须先声明指标和上限，不能用 val 或 official_test 选择。

## 资源和证据边界

峰值 RSS 是每个新进程的生命周期峰值，包含 Python 和依赖开销；macOS 以字节记录，Linux 从 KiB 换算。数据库磁盘字节另列，不等于常驻内存。异构计算计数可能重叠，不相加声称 FLOPs。geometry 的三次执行是重复测量，不是三个独立训练模型。

冻结结果标记 `software_selection_frozen`、`full_dataset_verified=false`、`paper_results_verified=false`、`equal_resources_claimed=false`。环境锁存在与哈希匹配，不自动证明其描述覆盖实际运行时所有依赖。官方全集、真实 GPU 环境与论文收益须另行验证。

## 优先级检查点选择

论文入口 `train_paper_priority.py fit` 显式启用 `select_best_train_holdout=True`，以训练内序列 holdout 的最小 MSE 选择 epoch，同分保留最早 epoch，并实际恢复对应权重。检查点记录选择轮次和损失；不使用官方 val/test。

旧 `fit_priority` 默认仍为固定最终 epoch，避免改变既有调用的行为。head holdout 不等于上游检测器和教师训练全过程均已隔离；`strict_pipeline_isolated_selection` 和 `paper_eligible` 仍保持 false。正式全集教师回执仍是未完成项。
