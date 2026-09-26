# 阶段 14：固定预算小样本过拟合诊断

任务 `165fb1e684a344788ff945e6ab8ca8b7` 已 completed。实际在四 A100 执行，三种子 × 两目标以及种子 1337 两目标重复均通过预定过拟合判据。八份诊断 checkpoint 和报告已上传 ClearML 并回读核验 SHA-256、seed、模式、有限参数及重复一致性。

## 预先固定设计

复用训练任务 `4abd3528a43e41bc82e19087d622925a` 的三个 3-epoch checkpoint；不是从随机初始化开始，也不是续用旧 Adam 动量。每个条件从相同 checkpoint 和全新 Adam 开始，学习率 0.001、固定 400 步、每 50 步检查一次。预算固定，不根据结果提前停止或选择最佳 checkpoint。

按时间顺序选第一个非空且有共同身份的 train 时序 batch，实际 frame index=1；当前左/右/历史 token 数 9/12/7。包含五条正关联边（十个双向正目标）及十一个未匹配目标。GT 用于训练监督与候选子集选择，不进入特征数值；这是 oracle train-only 诊断。

两条件仅改变优化目标：joint 为原跨端＋时序损失，cross_only 为仅跨端损失。种子 1337/2027/3407 分配三张卡，第四卡重复 1337。总计 3,200 个小 batch 优化步骤，不是全量训练或 DDP。

## 最终实测

| 种子 | 目标 | 跨端 CE | 时序 CE | 正目标正确 | 未匹配目标正确 | 正增益真实边 |
|---|---|---:|---:|---:|---:|---:|
| 1337 | joint | 0.000253516 | 0.000254685 | 10/10 | 11/11 | 5/5 |
| 1337 | cross_only | 0.000146594 | 1.397991538 | 10/10 | 11/11 | 5/5 |
| 2027 | joint | 0.000244957 | 0.000217550 | 10/10 | 11/11 | 5/5 |
| 2027 | cross_only | 0.000188642 | 1.316823721 | 10/10 | 11/11 | 5/5 |
| 3407 | joint | 0.000351765 | 0.000211923 | 10/10 | 11/11 | 5/5 |
| 3407 | cross_only | 0.000186258 | 1.446338058 | 10/10 | 11/11 | 5/5 |

初始三个种子均为正目标 0/10、未匹配目标 11/11、正增益边 0/5。joint 的最终最小正边增益分别为 15.897627831、16.249610901、14.814989090。第四卡两种目标均逐参数与首卡一致，最大差为 0。

结论限定：现有模型与联合目标可以记住这个训练样本，不存在该样本上不可摆脱的全未匹配状态。联合损失也能成功，因此本结果不支持直接关闭时序损失作为必要修复。仍不能证明全序列训练已收敛、特征具有可泛化身份信息，或把 400 步单样本预算直接用于正式训练选模。

## 实现与证据

- 隔离源码：`work_dirs/rbf-pointpillar-runtime-20260917/tools/event_track_v2x/run_v2v4real_overfit_diagnostic.py`。
- runner SHA-256：`b0c653dbafc670b20bc1b0e9e817561dcd8af98913a903d741d2e52676281af1`。
- 报告 `overfit-diagnostic-report` SHA-256：`d7bc23261b5f4aab906322156cadf793f6e0573613871b6c453a534d1f524455`。
- 三项相关测试文件合计 11 passed；本阶段新增 4 项。
- 诊断权重明确 `diagnostic_only=true`、`paper_eligible=false`，不升级为正式冻结模型。

复现命令（已有同源码任务时返回 ID，不重复创建）：

```bash
cd /home/lbin/Desktop/rbf-frozen-inference-20260917
CLEARML_CACHE_DIR=/home/lbin/Desktop/rbf-next-experiment-cache \
  /home/lbin/miniconda3/bin/python tools/event_track_v2x/run_v2v4real_overfit_diagnostic.py \
  --submit tools/event_track_v2x/run_v2v4real_proxy_identity_inference.py
```
