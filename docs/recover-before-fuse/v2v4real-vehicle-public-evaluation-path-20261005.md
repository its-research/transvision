# V2V4Real 合并 vehicle 的公开评价接入依据

日期：2026-10-05。本文记录来源核对及本机软件接入；合成 fixture 不构成检测、跟踪性能结果。
执行决定以 [官方划分与合并 vehicle 决定](v2v4real-official-benchmark-decision-20261005.md) 为准：
V2V4Real 使用公开官方划分和合并 vehicle 口径；SPD 仍为 car；RBF 全候选竞争不变。
不增加物理采集会话映射门槛，不修改既有失败、重叠审计或冻结产物。

## 1. 原论文与公开跟踪路径

[V2V4Real 原论文 v2](https://arxiv.org/html/2303.07601v2) 的 §3.1、§4.1、§5.1 给出：

- 标称 10 Hz；train / validation / test 为 14,210 / 2,000 / 3,986 双车帧。
- car、van、pickup truck、semi-truck、bus 合为一个 vehicle 评价类别。
- 测试固定 Tesla 为 ego；GT 转入 ego 坐标，重复对象优先采用 ego 标注。
- 检测评价区域为 ego 周围 x=[-100,100] m、y=[-40,40] m；检测指标为 AP@IoU 0.5、0.7。
- Sync 保留传感器固有不同步；Async 用非 ego 前一时刻模拟 100 ms 传输延迟。

本轮阅读上述原论文及官方入口，没有发现要求额外提供“绑定归档的物理采集会话映射”才能评价的条款。
采用公开划分不等于已证明归档无重叠；现有重叠证据按当前决定继续披露，不阻止该公开基准路线。

[DMSTrack 论文 v2（ICRA 2024）](https://arxiv.org/html/2309.14655v2) §IV-A—C 明确使用
32 个 train 序列 / 7,105 时刻及 9 个 test 序列 / 1,993 时刻，10 Hz；采用 AB3DMOT
3D IoU 阈值 0.25，AMOTA 为主指标。论文说明原 V2V4Real tracking 代码当时未发布，
因此作者用 CoBEVT 检测输出和 AB3DMOT 自建基线。接入该代码应称“DMSTrack 公开评价实现”，
不能仅因指标名称相同就声称逐值复现原论文基线。

## 2. 已核对的固定来源与调查边界

项目 [public-baselines.json](../../configs/event_track_v2x/paper/public-baselines.json) 固定 DMSTrack 为
`d3b9949499c8e68ea33060873bd1cb95b6d4d323`。
本轮独立读回既有本机源码回执及其 5 个小文件，逐项核对 SHA-256、字节数和 Git blob SHA-1，全部匹配：

`test/recover-before-fuse/artifacts/v2v4real-official-public-session-provenance-investigation-20261001T231605Z/DMSTrack-selected-source-download-receipt.json`

回执 SHA-256：`eeff0ae8f8b70de2b59004e45ac935d0525d8d214d6d81b5da2732157f182d87`。

| 固定提交内路径 | 本机核验的 SHA-256 |
| --- | --- |
| `docs/DATA.md` | `76cfb90bc3225b62f984b7f89a45da1ad48c44f8721e3debd15b11e4f31db302` |
| `docs/INFERENCE.md` | `e706aab5be9e0773235d5b7cf086e7bfdf87291dab34639d04a16fbabd676cb6` |
| `V2V4Real/opencood/tools/inference.py` | `1a3c3b6c9afda4d66edcda9a9df790114376e39933fd960abb0551703cf623e5` |
| `V2V4Real/opencood/tools/debug_utils.py` | `e392e1f366eedeee6b70c6b37793f774206388ce3fbb0b0055f788e369668004` |
| `V2V4Real/opencood/data_utils/datasets/basedataset.py` | `d788e865ee0b2952ac917eb8a4f77f68181b68f38c87de0b43bbea8e4896f291` |

另读取 `/private/tmp/rbf-public-dms_main-20261005.txt`；其 SHA-256 为
`4a78f056d364795091e3a2b4f609a2b07a874874ff428aef1e56dde6ff904da6`，
与已有固定源码测试 `test_public_native_checkpoints.py` 的预期一致。
其中 `get_config` 明列 train 的 32 个序列、累计 7,105 时刻，以及 `val` 的 9 个序列、累计 1,993 时刻。
其 `track_and_evaluate` 已用显式 V2V4Real 布尔参数调用原生评价函数。

初次网页调查读取的是 DMSTrack `master` 的 evaluator；该调查没有被当作固定版字节证据。
后续已从固定提交下载并核验 [evaluate.py](https://github.com/eddyhkchiu/DMSTrack/blob/d3b9949499c8e68ea33060873bd1cb95b6d4d323/AB3DMOT/scripts/KITTI/evaluate.py)
及实际使用的 `munkres.py`、`mailpy.py`、`dist_metrics.py`、`box.py`、`kitti_oxts.py` 和 official-test seqmap。
七文件 SHA-256 均硬编码在下述 wrapper 的 `SOURCE_HASHES` 中；evaluator 为
`a009b926cffbc396c9a131392b47623e32aacc268acc3d4c3c99b3b9e64d39de`，seqmap 为
`840783eed9ab01cdd359015b03ef9b9ba0f867a1ab78882e7dd2d8323f8d1c5f`。
固定版也确认相同路由坑、3D IoU 0.25、val=test 及 AMOTP↑。

UCLA 既有源码引用版本为 `5a821e13753bafc611f95c47bc1a306acdcb0f7c`，见
[旧 GT 几何来源说明](v2v4real-ground-truth-20260914.md)。本轮官网 README、配置和 loader 的网页调查使用
`main`，没有把它们冒称为该旧提交的新字节验收。旧 GT 文档中的 strict-Car 预过滤已由当前 vehicle 决定替代；
其中原几何验收不会自动变成合并 vehicle 的验收。

## 3. 检测入口：实际键是 validate_dir

[官方 README](https://github.com/ucla-mobility/V2V4Real/blob/main/README.md) 的测试说明写的是
`validation_dir`，但[实际 loader](https://github.com/ucla-mobility/V2V4Real/blob/main/opencood/data_utils/datasets/basedataset.py)
及[示例配置](https://github.com/ucla-mobility/V2V4Real/blob/main/opencood/hypes_yaml/point_pillar_fax.yaml)
使用 `validate_dir`。应核对实际被加载的键，而不是新增一个程序不读取的拼写。
公开训练目录为 `root_dir`，非训练读取目录为 `validate_dir`；测试时后者指向 `v2v4real/test`。

```bash
python opencood/tools/train.py --hypes_yaml CONFIG_FILE
python opencood/tools/inference.py --model_dir CHECKPOINT_FOLDER --fusion_method FUSION_STRATEGY
```

以上仅列公开命令模板；本轮没有运行。检测 AP 与下述 tracking 3D IoU 阈值是不同评价路径，不能混用。
输入点云裁剪范围也不能替代最终评价 ROI。

## 4. 跟踪入口、val 别名与类别合并

固定版本的 [INFERENCE.md](https://github.com/eddyhkchiu/DMSTrack/blob/d3b9949499c8e68ea33060873bd1cb95b6d4d323/docs/INFERENCE.md)
提供 DMSTrack `main_dkf.py --dataset v2v4real ... --evaluation_split val --seq_eval_mode all` 路径，
以及以下 CoBEVT + AB3DMOT 示例：

```bash
cd AB3DMOT
python3 main.py --dataset v2v4real --det_name cobevt
python3 scripts/KITTI/evaluate.py cobevt_Car_val_H1 1 3D 0.25
```

**这里的 `val` 是后续公开实现对 9 序列 / 1,993 时刻 test 的别名，不是原论文 2,000 帧 validation。**
这是论文公开 test 描述与固定 `main_dkf.py` 中相同规模配置的对应关系；当前本机归档到该 seqmap 的逐帧绑定仍须核验。
不以这个名称为由把 official test 用作训练、校准、阈值或 checkpoint 选择集。

固定 [转换器](https://github.com/eddyhkchiu/DMSTrack/blob/d3b9949499c8e68ea33060873bd1cb95b6d4d323/V2V4Real/opencood/tools/inference.py)
的 `transform_and_save_tracking_label_to_ab3dmot_format` 将导出 GT 类型写为 `Car`；
检测导出对应类型索引为 2。因此新接入中的 `Car` 应明确记为 **合并 vehicle 的后端编码别名**，
并同时转换 prediction 与 GT。它不表示只取原始 `obj_type == 'Car'`。

类别集合和排除规则遵循当前决定及固定官方转换函数：保留真实原始类型审计，不凭论文中的英文名称臆造
ZIP 的字符串枚举；底层 `project_world_objects` 显式排除 `obj_type == 'Pedestrian'`，
不能理解成对 `vehicles` 内所有记录不加区别地统一纳入。新 vehicle 标签不得由旧 strict-Car 标签简单改名得到。

在本轮阅读的 [master evaluator](https://github.com/eddyhkchiu/DMSTrack/blob/master/AB3DMOT/scripts/KITTI/evaluate.py) 中，
CLI 根据结果目录名包含 `cobevt`、`late_fusion`、`multi_sensor_kalman_filter` 或
`multi_sensor_differentiable_kalman_filter` 来判断 `evaluate_v2v4real`。RBF 自己的结果名可能落入 KITTI 分支。
该版本 Car loader 还会将未合并的 Van 作为相邻类别忽略。
因此未来 wrapper 必须显式选择 V2V4Real、完整序列、公开 test 别名和 3D IoU 0.25，
且在入口前完成已验证的 vehicle→Car 映射；不能用伪装 CoBEVT 结果名规避路由问题。
完整函数签名和依赖在固定 evaluator 字节补齐后核对，本页不提供未经该步骤验收的可执行 wrapper。

## 5. 指标方向与现有 RBF 评价器的区别

[原论文 tracking 表](https://arxiv.org/html/2303.07601v2#S4.SS2) 和
[DMSTrack 论文表 I/II](https://arxiv.org/html/2309.14655v2#S4) 给出的方向一致：

| 指标 | 公开 V2V4Real / DMSTrack 方向与含义 |
| --- | --- |
| AMOTA | ↑；跨 recall 阈值平均的跟踪准确率，DMSTrack 主指标 |
| AMOTP | ↑；IoU 精度式平均值，不是距离误差 |
| sAMOTA | ↑ |
| MOTA | ↑ |
| MT | ↑ |
| ML | ↓ |
| 通信 MB | ↓；须记录实际载荷计量口径 |
| 检测 AP@0.5 / AP@0.7 | ↑；独立的检测评价路径 |

现有 [evaluate_paper.py](../../tools/event_track_v2x/evaluate_paper.py) 调用
[tracking_evaluation_v2.py](../../transvision/models/event_track_v2x/tracking_evaluation_v2.py)，
AMOTA/AMOTP 来自 nuScenes 适配器：全局 XY 中心距离严格小于 2 m 匹配，AMOTP 为米制平均中心距离，**↓**。
同一入口另报 TrackEval HOTA/AssA/DetA/IDF1，基于世界 XY 的定向 BEV IoU；这些不是原生 3D IoU 0.25 的替代品。
`evaluate_paper.py` 本身已写明 `native_protocol_reproduction=False` 与
`nuScenes_mean_center_distance_m_lower_is_better_not_native_AB3DMOT_AMOTP`。

新公开基准表必须给指标引擎、定义、类别、ROI、阈值和单位独立命名。
不得仅把现有输出的 `AMOTP` 列方向翻转，或把原有 car 指标改成 vehicle，来宣称公开协议完成。
RBF 机制分析可继续保留原指标，但应与原生公开基准表区分。

## 6. 已实现的软件入口与真实输入缺口

[evaluate_v2v4real_native_vehicle.py](../../tools/event_track_v2x/evaluate_v2v4real_native_vehicle.py)
提供原生 KITTI 输入评价，以及 `--convert --evaluate-converted` 的 world mean9 / ego-corners 转换后评价入口。
它显式调用固定 `evaluate(..., evaluate_v2v4real=True, seq_eval_mode='all', v2v4real_split='val')`，
不依赖输出目录名称；GT 和预测都统一写 backend `Car`。全 9 序列 / 1,993 时刻必须有输入行，空帧亦保留。
原 `roty` 及数值函数体保持不变；现代 Numba 对原始混合数值列表 JIT 编译不兼容，因此记录
`NUMBA_DISABLE_JIT=1`，执行原 NumPy 函数体。此运行时选择写入每份评价回执。

转换 manifest 的 `kind` 为 `v2v4real_world_to_native_vehicle_conversion_input_v1`；它必须逐项绑定文件路径和 SHA-256：

- `predictions`：原 RBF replay 的 world `mean9`、`frame_id`、`box_reference_timestamp_us` 和 `predictions` JSONL。
- `ground_truth_manifest`：合并 vehicle 的原生 ego-corners GT manifest；其 `frames_sha256` 绑定同目录 `frames.jsonl`。
- `frame_mapping`：每行显式给出 source `sequence_id/frame_key/frame_ordinal/ego_cav`、预测 `frame_id` 与参考时间，以及 native `sequence_id/frame_index`。
- `world_to_ego`：每个 source frame 的显式、有限、正旋转 4×4 列向量刚体变换，绑定相同 ego 和参考时间。

还必须给 `fixture`、固定 source commit 及明确的 `prediction_class_mapping`，例如已绑定单类 detector 的
`{"car":"vehicle"}`；不据名称推断车种或模型语义。类映射只用于评价，既有 RBF 全候选竞争不受改动。
`bicycle`、`pedestrian` 只能显式排除，不能映射到 `vehicle`；`car` 通道的 native 单类含义仍须来源 sidecar 证明。
代码对全部帧作四方身份/覆盖核验；源码、输入、输出、序列内预测 ID 双射均记录哈希。
缺失、重复或时间/ego 不匹配直接失败，不能凭非空预测补空帧，不能按场景名排序猜官方序号。
固定 exporter 的 native 帧号是场景内 ordinal，因此逐行要求 `native_frame_index == frame_ordinal`；
只有集合齐全但重排时序的映射也会失败。

几何转换提取固定 `box_utils.py` 的 `boxes_to_corners_3d`、`project_box3d`、`corner_to_center`，
以及固定 `inference.py` 原始三个数组变换语句。原导出的 2D 框四个零值被保留；原评价器对未匹配、
2D 高度不超过 25 的预测所用忽略规则也保持不变。该细节必须披露，不能擅自填造 2D 框改变 FP 计算。
速度项不写入原生 3D 框；字符串 ID 按序列生成稳定双射，保留逐帧身份连续性。

软件 fixture 覆盖已知几何、原生 IoU 边界、AMOTP 方向、错误路由与输入篡改，完整合成 cohort 可进入原固定评价函数。
真实运行仍须准备并核验当前归档的官方序列映射、逐帧 world-to-ego、vehicle GT、真实预测及训练/选择绑定。
回执分别保留 `poses_and_frame_mapping_independently_accepted=False`、`full_rbf_native_path_accepted=False`
和 `paper_performance_complete=False`；转换或软件测试通过不会自动改写真实实验验收。
这些是具体输入和实现的绑定检查，不是物理会话独立性门槛。本次没有运行公开模型、训练或提交远端任务。
