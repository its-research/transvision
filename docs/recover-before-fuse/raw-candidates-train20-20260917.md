# 原始候选与冻结特征验证

2026-09-17，前置任务 `2910b8ddacfc4cdfae29bdc4d8300027` completed，40 个源帧后处理、四卡参考输出与预测文件哈希复核通过。下一项固定相同数据与公开权重，验证 NMS 前 raw score >= 0.05 / top64 候选及其原生冻结特征。

## 提交与入口

- ClearML task：`e59d0f4d522b4ad8940d41cf3d462449`，队列 GPU4-A100。
- 提交回执 queued；不以提交状态代替运行验收。
- 随后实时核验 in_progress，worker 为 `10.100.34.18-A100:gpu4,5,6,7`，正在初始化容器；尚无候选验证结果。
- 入口 `work_dirs/rbf-pointpillar-runtime-20260917/tools/event_track_v2x/run_v2v4real_postprocess_smoke.py`。
- 提交源码 SHA-256：`1a599b75dbf6069e22e7647dac4cf35b070e59aeee9f1242533c82313b0f4e9a`。
- 新部署目录 `10.100.35.112:/home/lbin/Desktop/rbf-raw-candidates-train20-20260917`，未覆盖前次修复部署目录。
- 回归：41 passed、1 skipped，包括独立 top64 排序、阈值边界、相同分数稳定排序、空候选、框/特征单元映射和非法输入拒绝。

## 数据与算法边界

ClearML 注册表 `14675561902041d7ab77fa2adbf2aeb0`，catalog SHA-256 `73bfa37d664ae0842368b5327be077627f06e1934530390b8a6571d77676f01d`，资产 `v2v4real-train20-gt-free`，输入 SHA-256 `3a0c61991d6a30f0fb4f8d73365b5f4275e34ba1d96ebc79b02a727a8439e317`。Worker 直接从 ClearML 拉取，20 对帧、40 个源 PCD，无标签。

候选由全部发射 anchor 的 sigmoid 原始分数 >= 0.05 筛选，按分数降序、扁平 anchor 索引升序确定稳定 top64，在 NMS/ROI 前选取。不使用 car-first 或前项过滤后的框来选取候选。

通过分类头 forward-pre-hook 读取未修改的冻结 head-input BEV 特征，按 anchor 的空间单元取原生通道向量。保留原始维度，不补造 128 维特征、速度、协方差、世界位姿或时间戳。每卡共享参考帧的候选索引、分数、框与特征须通过一致性比较。

公开权重为单类 vehicle 头，不能声称完成多类主协议验证。权重选模来源仍未核实，paper_eligible=false。不是 DetectionCacheV2 正式导出、训练、DDP、跨车融合或论文性能实验。继续运行原后处理作为旁路回归，但后处理输出不参与本项候选选择。

## 验收

必须核对 completed、candidate-report 与 raw-candidates-frozen-features。要求 40 个唯一源帧、每帧至多 64 候选、对应框与特征数量一致、分数范围合法、四卡参考输出一致，以及回读文件 SHA-256 与报告一致。未取得以上证据前保持运行未验收。

复现提交：在部署目录使用原环境运行 `tools/event_track_v2x/run_v2v4real_postprocess_smoke.py --submit --postprocess-source postprocess-source.json --pretrained-helper tools/event_track_v2x/run_v2v4real_pretrained_smoke.py`；入口按源码哈希去重并检查非重叠四 A100 空闲。代码尚未提交 Git。

## 后续完成核验

2026-09-17 再次查询状态 completed，candidate-report 确认 40 源帧、2,560 候选、256 通道冻结特征、四卡参考输出一致。下一项准备程序从 ClearML 回读 raw-candidates-frozen-features，SHA-256 `f75d569f593c87640621f0d91b9daa84fd16764267a27719a1e00815ca28914f` 校验通过，并验证帧配对、数组尺寸、数值有限性、分数范围/排序。该结果仍不构成正式性能证据。
