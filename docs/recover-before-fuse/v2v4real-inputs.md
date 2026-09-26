# V2V4Real 原始输入准备

本入口将已解压的 V2V4Real 单一 split 转为点云与位姿输入，并提供原始标签读取接口。它不执行检测、训练、跟踪或指标评估，也不生成 `DetectionCacheV2`。已用官方 `train_04.zip` 验证 4 个序列、438 个双车时刻、876 份源观测的读取、投影和哈希回读；不是完整 train/test 接入。证据见 [真实分卷接入记录](v2v4real-native-inputs-20260913.md)。

代码位于 `transvision/models/event_track_v2x/v2v4real_inputs.py` 和 `tools/event_track_v2x/prepare_v2v4real_inputs.py`。只在 transvision 维护研究代码；论文构建工具保持原位。

## 与官方实现的关系

核对来源固定为 V2V4Real 提交 `5a821e13753bafc611f95c47bc1a306acdcb0f7c`。

- 官方 loader 的非训练输入来自 `validate_dir`，文件名 stem 在源码中称为 mocked timestamp。因此不能按配置键名判断实际 split，也不能把 stem 当成微秒时间。[数据读取源码](https://github.com/ucla-mobility/V2V4Real/blob/5a821e13753bafc611f95c47bc1a306acdcb0f7c/opencood/data_utils/datasets/basedataset.py)
- 官方变换函数同时支持六维位姿和矩阵。实际 `train_04.zip` 的 876 个 `lidar_pose` 均为 4×4 NumPy 矩阵；新投影 v2 保留矩阵原值。六维输入分支仍按 `[x, y, z, roll, yaw, pitch]` 和 `Rz(yaw) Ry(-pitch) Rx(-roll)` 处理，角度单位为度，不是通用 ROS 欧拉角约定。[变换源码](https://github.com/ucla-mobility/V2V4Real/blob/5a821e13753bafc611f95c47bc1a306acdcb0f7c/opencood/utils/transformation_utils.py)
- 官方投影在 `obj_type` 缺失时默认 Car，且只显式排除 Pedestrian。这里的标签读取器保留原始 `obj_type`；缺失时失败，不补成 Car。[目标投影源码](https://github.com/ucla-mobility/V2V4Real/blob/5a821e13753bafc611f95c47bc1a306acdcb0f7c/opencood/utils/box_utils.py)
- DMSTrack 的转换脚本把导出的跟踪 GT 类型统一写为 Car，不能直接据此证明其包含严格的原始 Car 类。[转换源码](https://github.com/eddyhkchiu/DMSTrack/blob/d3b9949499c8e68ea33060873bd1cb95b6d4d323/V2V4Real/opencood/tools/inference.py)

按用户限定，论文结果采用严格 `Car`，不把 Truck、ConcreteTruck 或 Pedestrian 合入 Car。当前读取入口保留原始类别供离线核验，不实施类别映射，不计算分类指标。公开基线若使用合并车辆类，不能直接拼入严格 Car 主表，须按相同类别范围重跑。

## 使用前提

需要 Python 3.10 或更高版本、NumPy 和 PyYAML；本轮验证环境为 Python 3.12，依赖固定在 `environments/event_track_v2x/requirements-v2v4real-preparation.txt`。不要把依赖安装到已封存的 SPD 评估环境。

输入目录必须只包含计划使用的一个 split，格式为 `sequence/CAV/frame.yaml` 与同名 `.pcd`。准备器要求每个序列恰有两个非负数字 CAV 目录，两端 YAML/PCD 的 frame key 完全一致；不取交集，不静默丢帧。

还需要：

- 每个序列的显式 ego CAV ID，以 JSON 字符串保存。测试集的 Tesla 身份必须由原始来源核实，不能只按目录排序猜测。
- 本地来源证据文件，记录官方卷、split、下载来源及已核查的哈希。准备器保存该文件字节及哈希，但**不验证这些陈述的真实性**。
- 可容纳全部点云副本的空间。准备器复制而不建立硬链接；原始点云后续变化不会改变已生成副本。

官方 ZIP 可通过公开页面正常下载。新增 `tools/event_track_v2x/extract_v2v4real_archive.py` 校验指定分卷的长度、发布端 SHA-1 和固定元数据摘要，逐文件核对 CRC、生成 SHA-256，并拒绝路径越界、重复条目、链接和超限解压。它不包含下载凭据或自动下载流程；单卷成功不等于官方 split 完整性证明。命令见 [真实分卷接入记录](v2v4real-native-inputs-20260913.md)。

## 操作

先运行清单检查。此模式只访问目录和文件元数据，不读取 YAML、点云或标签内容，也不写出文件：

```bash
python tools/event_track_v2x/prepare_v2v4real_inputs.py \
  --source-root /data/v2v4real/test \
  --inventory-only
```

检查 `sequences`、`cav_ids`、`frame_keys`、`ignored_entries` 和两种帧数：`paired_frame_count` 是双车配对时刻数，`source_frame_count` 是单车观测数。不要与论文未注明计数单位的数字直接相等比较。

核实来源和 ego 映射后，使用一个不存在的输出目录：

```bash
python tools/event_track_v2x/prepare_v2v4real_inputs.py \
  --source-root /data/v2v4real/test \
  --split test \
  --ego-agents /data/protocol/v2v4real-test-ego.json \
  --source-evidence /data/protocol/v2v4real-test-source-evidence.md \
  --output /data/prepared/v2v4real-test-inputs-v1
```

输出结构：

```text
v2v4real-test-inputs-v1/
  inputs/
    manifest.json
    frames.jsonl
    pcd/<sequence>/<CAV>/<frame>.pcd
  audit.json
  source-evidence.bin
```

只向预测进程提供 `inputs/`。`audit.json` 保存来源路径、原始 YAML 哈希和准备行为；不要把父目录或原始数据目录作为预测输入。该目录布局本身不是操作系统权限隔离。

准备进程会解析可能含 GT 的原始 YAML，回执明确记录这一点。它只将 `lidar_pose` 和源坐标系到世界坐标系的变换写入 `frames.jsonl`，不会导出 annotation、关联 ID、类别或原始 YAML 路径。

真实分卷使用 NumPy 数值标签和 dtype/ndarray alias。读取器按固定语法将二进制浮点载荷解码为数值，不执行 YAML 指定的 Python 构造函数、`eval` 或 pickle。仅接受 float32/float64、明确字节序和最多 16 个数组数值；继续拒绝其他对象标签、通用 alias、递归对象、重复键、merge key、过深结构和非有限值。注册仅作用于本地 SafeLoader 子类，不修改 PyYAML 全局读取器。[PyYAML 安全读取说明](https://pyyaml.org/wiki/PyYAMLDocumentation)

矩阵旋转正交性与行列式采用 `1e-5` 导入容差，齐次末行采用 `1e-8`；不做正交化或数值修复。该容差用于容纳发布数据的序列化残差，不是跟踪阈值。v2 显式记录新的位姿格式；读取器仍接受旧 v1 六维输入，但拒绝以 v1 声明承载矩阵。

消费前调用 `load_prepared_frames(inputs_root, expected_manifest_sha256=...)`，传入已独立保存的准备回执中的 `input_manifest_sha256`。该函数校验 manifest、完整 frame index、每份点云的尺寸与 SHA-256、位姿矩阵、双车覆盖和顺序，并拒绝额外文件与字段。省略固定摘要时只检查内部一致性，无法发现整套清单被重写。函数不验证 PCD 格式或官方数据来源。

点记录另由 `read_native_pcd(path, expected_sha256=...)` 读取。首个真实分卷的 876 份 ASCII XYZRGB 点云已与 Open3D 官方输入公式精确对照，输出不改变点序的 float32 XYZI；不读取原始 YAML，不执行 ROI 或检测。格式范围、强度公式和复现入口见 [点云内容校验](v2v4real-pcd-inputs-20260914.md)。

## 时间、标签与失败行为

`frame_ordinal` 从零开始；`frame_key` 保留原始字符串。该原始准备清单仍标记 `time_basis=ordinal-only-no-clock`，本身不产生 event time 或 arrival time。正式实验已另行冻结 [`v2v4real-nominal-10hz-formal-v1`](v2v4real-nominal-10hz-formal-v1.md)：只有通过该协议的公共帧映射、split admission 和哈希绑定后，才可使用 `frame_ordinal * 100000 us` 的标称时间；不得称为真实采集时间或实测通信时延。

`read_annotations()` 是离线标签接口，不由预测输入准备流程调用。它保留 `object_id`、`ass_id` 和原始类别，不进行类别合并、ROI 筛选、跨车去重或指标计算。实际分卷有 7,859 条源帧目标记录的 `ass_id=-1`；这个值不能直接用作所有目标共享的身份 ID。`corners_in()` 是通用几何计算，不是原生 GT 转换器：真实框数值应先按源车局部坐标解释，再投影到 ego。

另已实现 [严格 Car 的原生 GT 入口](v2v4real-ground-truth-20260914.md)，在首卷 438 个双车时刻上与固定 DMSTrack 晚融合数值路径对照。它使用官方 ID 换算、两阶段 ROI 及先去重后筛选顺序，输出与推断投影完全分开。编号规则对齐不等于物理身份连续性已经验证；原生本地轨迹存在跨帧映射 ID 变化，不能自动修补或忽略。

发现帧缺失、ego 映射缺失、链接文件、非法 YAML 或源文件在准备过程中发生可检测变化时，准备失败。输出目录已存在时不覆盖。失败会删除本次唯一暂存目录，保留原始输入及现有输出；修正原因后使用新的输出目录重跑。

## 仍未完成的实验接入

- 官方卷获取、完整性和真实 train/test 会话重叠审计。分卷元数据已更新，审计入口见 [分卷核查与重叠审计](v2v4real-overlap-and-release.md)；当前官方目录另有 `val.zip`，不改变本研究采用 test 的协议。
- 严格 Car 的基线适配，以及 ego、ROI、时间及原生身份评估口径冻结。
- 冻结检测器和 LiDAR 特征定义；不能把 BEV 特征标成 SPD 的 ImageNet R50 特征。
- V2V 专用检测缓存契约与现有跟踪器的正式适配。
- 三种子全训练、同资源基线、冻结后的 official test 及 HOTA/原生指标验证。

现有 SPD `DetectionCacheV2` 未修改，仍只允许 SPD train/val 和原特征约定。所有准备输出固定标记 `official_split_membership_verified=false`、`detection_cache_created=false`、`paper_eligible=false`。这些标记不会因格式检查通过而升级。
