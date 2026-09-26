# 全未匹配偏好的离线监督诊断

ClearML Task `5888e0be40e24a2abfe22648f6dfbf8c` 已 completed，报告与 runner 已上传并回读。CPU 运行，没有占用 A100、模型前向、权重更新或阈值调整。本项使用 train 内标签作独立离线诊断，不读取 official_test，不将 GT 注入在线推断。

## 固定输入与检查范围

读取训练任务 `4abd3528a43e41bc82e19087d622925a` 的 `training-smoke-report`，从 label_audit 复用 40 个源帧的 331 个匹配候选标签；该报告读取 SHA-256 为 `e67c2bfc4e7d4dbe1ea5ab7a5ac953f27508b85b5f240716c9bcca23b25e5318`。冻结势函数来自 `6726803abe7147438d37c3191375fa92` 的三种子及 1337 重复输出，四份输入均按前项固定 SHA-256 核验。

仅分析 delay=0 的 20 个同帧跨端决策，避免引入跨帧物理身份正确性的额外假设。按 native ID 相等生成正边；已标注 token 没有对端同 ID 时对应未匹配目标。未被 GT 匹配的 2229 个候选保持未标注，不当作负样本。标签来自原离线 IoU 匹配规则，不是已验证的完整关联真值。

仅在完整候选前向产生的冻结 logits 上切出已标注子矩阵，再计算双向分类和匹配增益；这不是按训练候选子集重新前向，不能用于断言候选分布变化的因果效应。

## 实测结果

原训练有历史输入的 19 帧共有 311 个双向监督行：231 个 null 目标（74.2765%），80 个具有对端相同身份的正目标（40 条正边，各计两个方向）。20 帧合计 44 条正边。以下指标均来自冻结完整候选 logits 的已标注切片：

| 种子 | 19 帧 null 目标正确数 | 19 帧正目标正确数 | 20 帧正边最大增益 | 正边平均增益 |
|---|---:|---:|---:|---:|
| 1337 | 231/231 | 0/80 | -4.291645884513855 | -5.339697115334936 |
| 2027 | 231/231 | 0/80 | -2.4479050636291504 | -4.0013472085649315 |
| 3407 | 231/231 | 0/80 | -4.493245363235474 | -5.381396055221558 |

增益定义为 `2*pair_logit-left_unmatched_logit-right_unmatched_logit`。所有正边增益均为负。移除未标注候选的竞争、但保持已有 logits 不变，仍全部选择未匹配。种子 1337 的第四份重复结果逐项一致。

代码检查确认训练跨端损失为双向分类交叉熵平均，不是全局关联精确 NLL。GT-matched token 不必具有对端对应，因此不能把 matched-token-only 训练误解为全正关联训练。null 目标占多数与本次输出一致，但不足以单独证明类别比例是根因；总训练损失还包含时序损失，不能将其下降解释为跨端正关联已经学会。

## 验收与复现

- 新增测试 4 passed，覆盖身份而非候选序号匹配、无共同身份的 null 目标、未标注候选不当负样本和重复身份拒绝。Python 编译检查及两工作树 `git diff --check` 通过。
- 回读任务状态 completed；四份结果各 20 帧，重复一致。
- `null-diagnostic-report` 文件 SHA-256：`e49da91ae83e429a1fabbe239b31b081c02eee1a06d7bf89f0451ccb0a37c687`。
- 隔离工作树入口：`tools/event_track_v2x/run_v2v4real_null_diagnostic.py`；测试：`tests/event_track_v2x/test_null_diagnostic.py`。不改冻结主模型源码，不提交 Git。

远端使用已有 CPU 环境（ClearML、PyTorch、NumPy），命令：

```bash
cd /home/lbin/Desktop/rbf-proxy-decode-20260917
PYTHONPATH=. CLEARML_CACHE_DIR=/home/lbin/Desktop/rbf-next-experiment-cache \
  /home/lbin/miniconda3/bin/python tools/event_track_v2x/run_v2v4real_null_diagnostic.py
```

重复执行会按源码名称返回已有任务。

本项 `paper_eligible=false`，不输出跟踪指标或泛化收益。下一项建议固定相同权重、时间与样本，比较训练候选子集及完整候选的重新前向，分别检查 pair/unmatched 分数，以区分训练本身和上下文变化的影响；当前尚未执行该对照。
