# V2V4Real train20 解码与 NMS 验证

## 前置结果与当前状态

2026-09-17 实时复核：前一项 `cd7b2edc7c414588a4ad4b641bb8f909` 已 completed，forward-report 确认 40 个源 PCD、严格公开权重加载、四张 A100 参考输出完全一致，未读取 GT、未训练，尚未验证后处理。

本项任务 `463162b3f0984550948eed6d1d6946fa` 已提交 GPU4-A100，实时观察为 in_progress，worker 为 `10.100.34.18-A100:gpu4,5,6,7`。初次核验处于容器初始化，尚无后处理结果；启动不等于验收通过。

## 固定输入与验证边界

- 从 ClearML 注册表 `14675561902041d7ab77fa2adbf2aeb0` 直接读取 `v2v4real-train20-gt-free`；不读取桌面数据或 test 标签。
- catalog SHA-256：`73bfa37d664ae0842368b5327be077627f06e1934530390b8a6571d77676f01d`。
- 输入 SHA-256：`3a0c61991d6a30f0fb4f8d73365b5f4275e34ba1d96ebc79b02a727a8439e317`，20 个配对帧、40 个源 PCD。
- 官方 V2V4Real 源码固定为 `5a821e13753bafc611f95c47bc1a306acdcb0f7c`。执行哈希固定的 BasePostprocessor，以及 VoxelPostprocessor 的原样 AST 推断方法；不加载训练/Cython 和可视化入口。不宣称完整官方 loader 复现。
- 验证 8,800 anchors 解码与独立 NumPy 实现一致、零残差和空输出、真实帧 score 过滤、旋转 NMS、ROI、四卡共享参考输出一致。
- 每个源使用自身局部坐标和单位变换，没有跨车融合、DetectionCacheV2 导出、训练、DDP 或正式检测/跟踪指标。
- 使用公开权重原协议 score > 0.2、NMS 0.15；不是论文主协议 raw score >= 0.05/top64。公开模型选模来源未核实，`paper_eligible=false`。

## 实现与复现

隔离工作树：`work_dirs/rbf-pointpillar-runtime-20260917`，保留冻结主仓源码。

- 入口：`tools/event_track_v2x/run_v2v4real_postprocess_smoke.py`
- 入口 SHA-256：`73152a3cfb41b3044b10cc438f641a53db1b3f4351d95f4c57ba570097294fb9`
- 测试：`tests/event_track_v2x/test_postprocess_real_smoke.py`
- 本次直接依赖回归：32 passed、1 skipped；本地无 Torch，GPU 集成由远端任务执行。
- 远端提交目录：`10.100.35.112:/home/lbin/Desktop/rbf-postprocess-train20-20260917`

```sh
cd /home/lbin/Desktop/rbf-postprocess-train20-20260917
CLEARML_CACHE_DIR=/home/lbin/Desktop/rbf-next-experiment-cache \
CLEARML_FILES_HOST=http://10.100.35.118:8081 \
/home/lbin/miniconda3/bin/python tools/event_track_v2x/run_v2v4real_postprocess_smoke.py \
  --submit --postprocess-source postprocess-source.json \
  --pretrained-helper tools/event_track_v2x/run_v2v4real_pretrained_smoke.py
```

提交入口按源码哈希检查重复任务，并检查非重叠四 A100 空闲条件。成功验收必须同时核对 completed、postprocess-report 和 local-source-predictions 的 40 个唯一源帧及校验值。源码和本地测试未提交 Git。

## ROI 依赖修复

原任务最终 failed：框解码与 NMS 后，`box_utils.get_mask_for_boxes_within_range_torch` 导入 `opencood.data_utils.datasets.GT_RANGE` 时遇到 ModuleNotFoundError。精简官方源码包遗漏了这个运行时依赖。

从同一官方提交取得 `datasets/__init__.py`，SHA-256 为 `56e02cf9227ac6f4f4b6d4a08501dbb947e696722997a9ea723110e8d375c85e`。只提取其原始字面量赋值 `GT_RANGE = [-100, -40, -5, 100, 40, 3]` 为依赖模块；不执行数据集导入或标签读取，不改变 ROI 数值或后处理算法。输入包、权重、过滤与 NMS 阈值保持原样。

直接依赖回归 38 passed、1 skipped，包括 ROI 字面量提取、无数据集导入和非法 ROI 拒绝。修复入口 SHA-256：`8d5462ed6363a42dda495b86613b78249b9e8766094b836b82d9a1ce4d11cf35`。

重试任务 `2910b8ddacfc4cdfae29bdc4d8300027` 已 in_progress，实际 worker `10.100.34.18-A100:gpu4,5,6,7`。原失败任务保留；此处启动状态不是运行验收结论。

### 修复运行验收

2026-09-17 随后实时核验重试任务为 **completed**，日志 `Process completed successfully`。从 ClearML 回读 postprocess-report 和 local-source-predictions：

- 40 个唯一源帧完整覆盖，共 688 个后处理输出框；此数量不是准确率或性能指标。
- postprocessing_verified、numpy_decoder_verified、empty_output_verified 均为 true。
- 四张 NVIDIA A100-PCIE-40GB 的共同参考帧均输出 19 框，通过规定数值容差的一致性比较。
- GT 未读取，未训练，基础环境版本未改变；完整协同流水线、DetectionCacheV2 与 DDP 仍未验证，paper_eligible=false。
- 预测文件回读 SHA-256 与报告一致：`335d613c5b24d8cc60e1e3bfbc4d69d6a89b3d2ce4fd39776cb0fa23363c6d7f`，回读帧数 40、框数 688 均一致。
- ClearML 保存的 script.diff 哈希与本地提交源码一致：`8d5462ed6363a42dda495b86613b78249b9e8766094b836b82d9a1ce4d11cf35`。运行报告记录的执行文件哈希为 `c515c5478e9047386fa33a264034178a06108e74ef4a510baca12552eb12bf61`；两者分别保留，不声称执行文件与提交文件字节相同。

修复范围限于缺失的官方 ROI 常量依赖，原始失败任务和本次成功任务均保留，未提交或推送 Git。
