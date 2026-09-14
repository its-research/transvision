# learned+CI 与 M0–M4 主协议入口

入口为 `tools/event_track_v2x/run_paper_pair_baselines.py`，配置为 `configs/event_track_v2x/paper/pair-*.json`。算法复用原 `tracking_v2.py`、`tracking_mechanisms_v2.py`、`tracking_birth_score_v2.py`；不是森林 Top-K 的别名，也没有重新解释 M3/M4。

四个历史模块保持原字节与原审计哈希。适配器通过实例独立的依赖字典复用原方法 code object，不使用用户代码或源文本求值，不修改模块全局变量；代价计数也在实例范围内。测试核验原 code object 身份和全局对象不变，状态快照对嵌套列表和绑定做独立复制。

本轮代码核实：旧 learned+CI 的 `frame_features` 本来就是原始全类别 score >= 0.05 / top64。原先验收矩阵中的缺口是统一协议入口、数据格式与资产绑定，不是旧配对代码存在 car-first 筛选。森林历史版本的 car-first 权重仍须隔离。

## 方法边界

| 方法       | 实际计算                                                                    |
| ---------- | --------------------------------------------------------------------------- |
| learned-ci | 原 Top-H 全局配对、CI 分量、矩匹配、残余分数与时序生命周期                  |
| M0         | 同一算法加诊断；模拟序列预测、关联与原 learned+CI 字节级一致                |
| M1         | 两端都保留，强制唯一 all-unmatched；不加载或调用配对模型，无训练种子        |
| M2         | 同一配对及权重，仅 matched 分量用路端传播后的 mean/cov，随后仍做原矩匹配    |
| M3         | 同一配对，residual 通用 score 改为原校准分数；影响关联 cost、新生和更新     |
| M4         | residual 通用 score 仍乘 unmatched mass；仅新生门限和初始分数使用原校准分数 |

M3/M4 使用 `infrastructure.scores`，不是 `raw_scores`；不创建 unmatched mass 为零的 residual。M4 给定同一入帧状态时保持非新生表达式，新增出生仍会影响未来自由运行状态。M3−M4 不是可加性的独立时序效应。

当前入口只接受 100 ms clean-link 配对：每个事件恰有两端源帧，均已到达；event frame_id 和 reference_us 对应车端源帧。超过截止、缺端、重传和任意异步流明确拒绝，不能与完整异步 RBF 入口混称。两种缓存格式分别为 SPD DetectionCacheV2 和 V2V4Real NativePaperCache；只在独立评价时取 car。

## 资产契约

配对模型是 203 维 `PredictedAssociation`，与身份森林权重不兼容。必须提供新的、显式绑定的 `rbf_pair_checkpoint_v1` JSON 文件，包含：dataset、fit_split=train、candidate_protocol、feature_recipe=`rbf-all-class-pair-203-v1`、seed、fixture、architecture、calibration_sha256、frozen_cache_identity，以及 weights 和 training_receipt 的相对文件路径/哈希。种子只接受 1337、2027、3407。

训练回执必须独立绑定 dataset/train/protocol/seed/fixture 和 weights_sha256。加载器使用 tensor-only 权重读取及严格参数校验。不把手工写一个 manifest 视为真实训练完成，也不自动将旧权重升级为正式主协议证据；真实资产的训练与完整性还需审计。

校准文件支持 `rbf_pair_calibration_v1` 或兼容缓存构建器的 `eventtrack_train_calibration_v1`，但必须显式记录 dataset、fit_split=train、candidate_protocol、fixture 和双方三类 score 参数。其哈希必须与缓存生产者和检查点一致。V2V4Real 的原生特征/检测器身份必须匹配该数据集的配对权重，不能直接挪用 SPD 生产者绑定。

M1 不接受 checkpoint 参数，不构造虚假的三种子样本标准差；它仍需 train-only 校准和冻结缓存。

## 运行

以下大写参数是待替换说明符，不是正式资产示例：

```sh
python tools/event_track_v2x/run_paper_pair_baselines.py \
  --cache CACHE_DIR --cache-sha256 CACHE_MANIFEST_SHA \
  --schedule SCHEDULE.jsonl --schedule-sha256 SCHEDULE_SHA \
  --configuration CONFIG.json --configuration-sha256 CONFIG_SHA \
  --dataset spd --split val \
  --calibration CALIBRATION.json --calibration-sha256 CALIBRATION_SHA \
  --checkpoint CHECKPOINT.json --checkpoint-sha256 CHECKPOINT_MANIFEST_SHA \
  --output NEW_OUTPUT_DIR
```

V2V4Real 使用 train 或 official_test；SPD 使用 train 或探索性 val。M1 去掉 checkpoint 两项。模拟资产必须加 `--fixture`，不能改标签冒充真实证据。所有输出目录要求不存在。

输出包括预测、配对、逐帧诊断/成本、配置、源代码哈希、时间统计、各序列最终状态快照及回执。`paper_pair_baselines.restore` 可将有哈希绑定的快照载入同配置新实例，继续输出保持提交链；不是对已有历史结果重写。

结果直接兼容 `evaluate_paper.py` 的独立 GT 评价。训练内资源扫描的旧 `run_paper.py` 调用器尚未自动调度此入口，应显式分开编排；不能据此称所有基线已完成等资源比较。

## 成本和验收

首轮局部 89 项通过；完整回归随后发现冻结源码哈希不兼容。撤回历史文件修改并隔离适配后，210 项直接依赖和审计通过，1 项跳过。原失败回执保留，修复后没有重跑整套全量回归。这些计数不能相加当作独立测试总数。

记录实际模型前向、配对 logit 元素、特征候选数、跨端 Murty 内部 Hungarian 求解次数和时序 Hungarian 求解次数。记录缓存、输出流、状态快照的磁盘字节及端到端步延迟；磁盘不是常驻内存，Top-H 不是总计算成本。此算法没有可恢复森林，不填伪造的前沿成本。

`test_paper_pair_baselines.py` 覆盖原计算/重开字节对照、非 car 占满 top64、两缓存格式、模拟优化器更新后冻结加载、六种方法、CLI 与库输出对照、迟到拒绝、协议拒绝和独立评价。旧 M4 回归另外验证严格 birth-only 干预。模拟成功不证明真实全量训练、正式双数据集实验、旧封存结果复现或论文收益。
