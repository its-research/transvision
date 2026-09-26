# 独立标称 10 Hz 代理时间诊断

用户于本轮明确允许独立代理时间协议。协议 ID `nominal-10hz-proxy-replay-v1`，不改变真实时间主协议或历史资产，不作为真实网络延迟、deadline 可靠性或论文跟踪收益证据。

## 预先固定规则

- 每序列从零开始，以排序后的帧 ordinal × 100,000 us 为代理源时刻；不是测量时间。
- CAV 0 即时可用；CAV 1 分别模拟延迟 0、100,000、200,000 us。
- 决策 deadline 为参考源时刻 + 100,000 us，边界包含等号。
- 仅允许 source_time <= reference_time、arrival_time <= deadline，观测年龄 <= 100,000 us；选最新可用源帧，没有则缺失，不补造观测。
- 即使未来帧在 deadline 前可用，也不纳入当前参考帧。
- 迟到旧帧可在后续决策中合法选入，不回写历史决策；分帧提交 hash 链及前缀重放互验。
- 仅将选中观测角点从其源帧坐标转换到当前 ego 坐标，不外推目标运动，不将陈旧观测称为当前预测状态。
- 0 与 100 ms 组在此 inclusive deadline 规则下应选取相同当前帧；200 ms 组预期首帧缺失、后续选前一帧。该预期是预设调度逻辑，不是论文收益。

协议 SHA-256：`ee4c5365359b9f75455a6264236fc66014b374fc06430d1dc4c5e82363f0da4c`。

## 任务与资产

- ClearML task `7a3227d1a9e640039af8114f0dece7cc`，提交 GPU4-A100；初始回执 queued，不代表完成。
- 原始候选 Task `e59d0f4d522b4ad8940d41cf3d462449`，SHA-256 `f75d569f593c87640621f0d91b9daa84fd16764267a27719a1e00815ca28914f`。
- 无标签位姿 Task `01c9989cc5f24931a16a0605e6f164f9`，SHA-256 `aabdb5fe0811546ddca5c009206a4e57e44909b143a6e16b53fe04180294f0e0`。
- 两者从 ClearML 拉取并检查哈希、40 源帧配对及字段；不读取 GT。公开单类权重选模来源仍未核实，paper_eligible=false。
- 冻结辅助代码 SHA-256 `c2074314c8263811c426f3ad1e02b3fe9f1966b910df29d26e5233e47e67d0e3`。
- 新入口 `work_dirs/rbf-pointpillar-runtime-20260917/tools/event_track_v2x/run_v2v4real_proxy_replay.py`，SHA-256 `5d8c19a9733bb95a953ea3558b7e23420896ae4e1c8715082a8fef3adb0c6e08`。
- 部署目录 `10.100.35.112:/home/lbin/Desktop/rbf-proxy-replay-20260917`；该目录中运行 `tools/event_track_v2x/run_v2v4real_proxy_replay.py --submit tools/event_track_v2x/run_v2v4real_alignment_smoke.py`，使用管理机 SDK 环境及专用 ClearML cache。

## 验证与限制

本地 8 passed，覆盖独立调度预期、全部 20 前缀历史一致性、future/late/expired 拒绝及坐标辅助回归。四 A100 运行将对三个延迟组的 60 个决策执行因果选择与 FP64 跨时刻变换，和独立 NumPy 实现及卡间结果比较。

预期保存 proxy-protocol、proxy-replay-report、proxy-replay。只有 completed、报告通过和回读哈希/覆盖一致才能标记本项完成。未运行学习跟踪器、恢复森林、运动模型或训练；不是正式 DetectionCacheV2。授权仅为此类独立代理协议，不自动证明训练会话隔离或标签物理身份正确性。

## 完成验收

任务已实时核验 completed，worker 为 `10.100.34.18-A100:gpu4,5,6,7`。三组共 60 个决策输出回读成功，SHA-256 `b9340281679f918239f566335d946b8e7ac29484b764c2e65bc462e54a565256` 与报告一致。

各卡结果均符合预设规则：0/100 ms 组各 20 个决策无缺失、无陈旧观测；200 ms 组 20 个当前源帧均越过 deadline，首个决策缺少远端观测，后续 19 个决策合法选用前一帧。历史前缀提交不变，四卡一致性通过，跨时刻坐标变换与独立 NumPy 实现最大绝对差 `4.263256414560601e-14`。

上述结论仅验收因果调度和几何重放实现。当前源帧迟到不等于远端观测永久丢弃；读取陈旧观测不等于恢复身份。未证明学习模型、身份恢复或跟踪收益。
