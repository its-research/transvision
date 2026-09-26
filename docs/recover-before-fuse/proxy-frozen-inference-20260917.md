# 完整候选的无 GT 冻结关联推断

Task `6726803abe7147438d37c3191375fa92`，提交 GPU4-A100，初始 queued。独立代理协议沿用 nominal-10hz-proxy-replay-v1：100 ms 标称间隔、100 ms inclusive deadline、远端 0/100/200 ms 模拟延迟、最多 100 ms 观测年龄，未来帧不进入当前参考决策。

## 本项范围

加载上项真实匹配候选训练冒烟的三个种子 checkpoint，eval + requires_grad_(False)，在每个因果可用源帧的全部 64 个候选上运行既有 RecoverableIdentityModel(feature_dim=256)。不读取 GT、不按匹配结果筛候选、不执行优化器。history 固定为上一帧 CAV0 的全部 64 个候选；同时进行空 history 前向，对比 pair logits，验证历史路径实际参与计算。

前三张卡运行种子 1337/2027/3407，每种子三档延迟、每档 20 个决策，共 180 个主决策。第四卡重复 1337 的 60 个决策用于数值一致性检查，不视为额外随机种子。检查 finite logits、完整形状、时间可用性、checkpoint 前后参数不变以及历史关闭响应。

输出为 pair、temporal、unmatched logits 及 context，并保留源帧路径、anchor 索引和调度。没有合法匹配解码、身份恢复、轨迹 ID 或跟踪评价；logits 不是校准后验。200 ms 首帧因不可用而为空，不能把这种因果过滤称为 GT 筛选。

训练与本次推断使用同一 train 短序列，因此不是 out-of-sample 测试。原训练只用 331 个 GT-matched token，而此项前向接收所有候选，分布变化只作接入诊断，不据此证明泛化。paper_eligible=false，主协议不变。

## 固定来源

- 输入任务 `e59d0f4d522b4ad8940d41cf3d462449`，候选 SHA-256 `f75d569f593c87640621f0d91b9daa84fd16764267a27719a1e00815ca28914f`。
- checkpoint/source 任务 `4abd3528a43e41bc82e19087d622925a`。三个文件哈希为：1337 `b02836a0b74fbc0b95a213890e2da46da75ae9739f7a5fdd9e86b1a1e406154b`；2027 `839ee350da4da375db6e3e90cb0b15061adaddccd610409e4a962a19236724b7`；3407 `9e2689b34a57e2265e71474a5312b215d71932c8db2d5ff363597aec693cf41a`。
- 模型源码 SHA-256 `14674c5db70908e607f1bb75ea5df1d3661f0ba08b854ebf37364e350a2df170`。
- 调度 helper SHA-256 `5d8c19a9733bb95a953ea3558b7e23420896ae4e1c8715082a8fef3adb0c6e08`。
- 新 runner SHA-256 `dd921a37eed307381bc08e155733987ee37fd8064fa74bd648ef8e5141eb311d`，位于隔离工作树 `tools/event_track_v2x/run_v2v4real_proxy_identity_inference.py`。

资产直接从 ClearML 读取并核验哈希。远端部署目录 `/home/lbin/Desktop/rbf-frozen-inference-20260917`，执行 `tools/event_track_v2x/run_v2v4real_proxy_identity_inference.py --submit tools/event_track_v2x/run_v2v4real_proxy_replay.py`，使用 SDK 环境和专用 ClearML cache；提交前检查非重叠空闲四 A100。

## 验收

本地直接回归 7 passed，覆盖完整候选约束、子集拒绝及调度边界/历史不变。运行成功需 completed、frozen-inference-report、四份 potentials 文件、三个种子完整覆盖、第四卡一致、权重不变及资产回读 SHA-256 匹配。没有提交 Git。

## 完成回执

实时核验任务 completed，实际 worker `10.100.34.18-A100:gpu4,5,6,7`。三个种子各 60 决策，第四卡重复 60 决策，四份输出都已从 ClearML 回读、weights_only 严格加载并核对 SHA-256。所有当前 CAV0 输入均为完整 64 token，远端按固定因果调度可用性输入，GT 未读取，模型参数前后逐项相等。

| 种子 | 有历史时 pair 输出发生变化的决策数 | 历史关闭最大绝对 logit 差 |
|---|---:|---:|
| 1337 | 57 | 0.678032636642456 |
| 2027 | 57 | 0.678249716758728 |
| 3407 | 57 | 0.3510138988494873 |

每个延迟组首帧没有历史，因此每种子可检查历史作用的决策数为 57。该数值只说明历史输入实际影响前向，不说明影响方向正确或准确率提高。第四卡复跑 1337，所有输出张量的最大绝对差为 0。

输出 SHA-256：1337/device0 `a517db46f4cc20bf4992aed6089676694c07ce437be3905574c6b3e0d5102bb2`；2027/device1 `c7871ec964c58e8b9196013308c7df2ca6e5d7b6a8aa7ef5314ca9c45d157023`；3407/device2 `cfe691671f2808080f1585b74b68d6dad1dca1f79855b4f65ffd2379464f6512`；1337/device3 `0013b0b613c505c9f1beed7942ce2e1bbeaf7a33d7bf90b3d26dca8279974aa5`。

日志包含 ClearML 重复 input model 名称警告；入口对原始 checkpoint 文件哈希、seed、维度做了显式检查，最终权重不变检查通过。仍不能将此轮 potentials 输出当作合法身份匹配或跟踪结果。
