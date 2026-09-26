# 双车坐标对齐验证

前置原始候选实验 e59d0f4d522b4ad8940d41cf3d462449 已 completed。下一项复用其 40 源帧、2,560 候选，不重复检测器前向、不重选候选。

## 提交

- Task：`01c9989cc5f24931a16a0605e6f164f9`，GPU4-A100，提交回执 queued。
- 入口：隔离工作树 `work_dirs/rbf-pointpillar-runtime-20260917/tools/event_track_v2x/run_v2v4real_alignment_smoke.py`。
- 提交源码 SHA-256：`c2074314c8263811c426f3ad1e02b3fe9f1966b910df29d26e5233e47e67d0e3`。
- 部署目录：`10.100.35.112:/home/lbin/Desktop/rbf-alignment-train20-20260917`。
- 本地直接回归 97 passed，包括原生 YAML 安全解析、位姿刚体合法性、候选白名单、框尺寸/中心/旋转及不修改输入。
- 初次资源预检未通过，尚未创建任务；重新读取 worker 确认 GPU4–7 空闲后通过原有预检提交，未中断或修改其他作业。

## 固定资产与隔离

原始 train_04 由 ClearML 注册表 `14675561902041d7ab77fa2adbf2aeb0` 读取；固定 SHA-256 `aab4d7753b5bb62221f1a81602c52df168c0c263c578ec0ddd35c8fbd1be008c`。候选来自前置任务，SHA-256 `f75d569f593c87640621f0d91b9daa84fd16764267a27719a1e00815ca28914f`。

独立准备程序安全解析对应 40 个 annotation-bearing YAML，仅输出 source_to_world 矩阵及原 YAML 哈希；准备步骤读取了含标注的元数据，不能称整个准备过程未接触 GT。推断任务仅取得 pose-only JSON，无原 YAML 或标签字段。准备解析器源码及哈希保存于新任务的 preparation-* 资产和参数。

pose-only SHA-256：`aabdb5fe0811546ddca5c009206a4e57e44909b143a6e16b53fe04180294f0e0`。所有输入从 ClearML 拉取；不把帧号解释成时间戳。

## 实验与验收范围

使用 source-local 中心框 xyzhwl_yaw 构建八角点；独立 NumPy 矩阵变换与四张 A100 上 FP64 Torch 齐次坐标求解互验。逐帧核验 source-to-world、source-to-other-CAV、world-to-source 往返误差以及四卡一致性，保存 world/other-CAV corners 与原候选索引和分数。

预期资产 alignment-report、aligned-candidates。验收需 completed、40 源帧覆盖、2,560 候选、误差阈值通过、结果回读哈希一致。启动/排队不表示通过。

这是坐标适配验证，不是匹配准确率、跟踪指标或训练；未确认源时钟，未导出正式 DetectionCacheV2。单类公开权重及选模来源限制仍保留，paper_eligible=false。

复现：在部署目录设置 `CLEARML_CACHE_DIR=/home/lbin/Desktop/rbf-next-experiment-cache`，用 `/home/lbin/miniconda3/bin/python tools/event_track_v2x/run_v2v4real_alignment_smoke.py --prepare NEW_POSE_FILE.json` 创建新位姿文件，再以 `--submit NEW_POSE_FILE.json` 提交。准备文件 create-once，任务按源码哈希去重；未提交 Git。

## 完成验收

2026-09-17 实时确认任务 completed，实际 worker `10.100.34.18-A100:gpu4,5,6,7`，日志 Process completed successfully。alignment-report 和 aligned-candidates 回读确认 40 源帧、2,560 候选，输出 SHA-256 `84210ca345f5f48a6ab78a925188908dad409cc3005ca600c1ee47f8d928b118` 与报告一致。

四张 NVIDIA A100-PCIE-40GB 均通过一致性检查；各卡跨车变换最大绝对差 `4.263256414560601e-14`，往返误差 `3.108624468950438e-14`。这是数值实现一致性，不是外参实测精度，也不是检测或跟踪性能。真实时钟、正式缓存、训练和论文指标仍未完成。
