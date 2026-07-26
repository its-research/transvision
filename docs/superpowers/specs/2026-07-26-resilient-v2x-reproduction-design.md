# Resilient V2X 完整实现与实验复现设计

**日期：** 2026-07-26

**目标分支：** `resilient_v2x`

**目标仓库：** `/Users/libin/transvision`

**论文规范仓库：** `/Users/libin/ResilientV2X`

## 1. 目标

在现有 Transvision 与 MMDetection3D 框架内，实现论文《Resilient V2X:
Reliability-Aware V2X Collaborative 3D Detection Under Missing Modalities and
Communication Latency》定义的完整方法、训练流程、实验矩阵、基线适配、指标聚合、
性能分析、可视化和证据导出。

交付物必须满足以下要求：

1. 论文中的因果边界、PTF、可靠性计算、DER descriptor、蒸馏目标和 unsupported
   语义都有明确且可测试的代码实现。
2. 训练、推理和实验均由版本化配置驱动，禁止依赖模型文件中的隐式默认行为。
3. 论文未提供的实现细节采用本文档固定的工程复现选择，并在协议和运行证据中明确标记，
   不把这些选择表述为原论文已经验证的事实。
4. 所有统计量从逐 seed 原始评估输出计算，禁止手工录入均值、标准差、PDR 或方法差值。
5. 外部基线通过可追溯适配器接入，不在本仓库中编造不兼容的替代实现。
6. 当前机器没有 PyTorch、CUDA 和 DAIR-V2X-C 数据，因此不能声称已完成 GPU 训练或论文
   数值复现。代码验证和 GPU/数据验收状态必须分别报告。

## 2. 当前状态与改造边界

### 2.1 当前代码状态

目标分支当前仅包含以下 Resilient V2X 相关改动：

- `transvision/models/detectors/resilient_v2x.py`
- `transvision/models/detectors/__init__.py` 中的模型注册

现有 365 行原型不能直接构建或训练，主要问题包括：

- `ptf_cfg` 与 `der_cfg` 缺少必需的 `in_channels`；
- PTF 默认输出仍含时间维度，不能传入二维 warp；
- PTF 没有 horizon、置信度 `Q`、mask、`alpha`、`k` 或 unsupported 分支；
- 不存在因果到达集合、源选择和 ego 当前坐标系对齐；
- 没有 ego/RSU × LiDAR/Camera 四分支历史；
- DER descriptor 与论文不一致；
- 缺少 teacher、蒸馏、故障注入、配置、测试、实验和证据代码；
- 当前 DAIR evaluator 忽略传入预测并重新运行另一模型，不能作为论文证据。

### 2.2 仓库责任边界

`/Users/libin/transvision` 负责：

- 数据预处理和时序 manifest；
- 因果历史、故障与时延协议；
- 模型、训练、推理和评估；
- 实验矩阵、统计、profiling、可视化和证据包。

`/Users/libin/ResilientV2X` 负责：

- 已审查协议与证据台账；
- 论文结果注册表和 LaTeX 生成；
- 投稿门禁。

本任务不覆盖或删除论文仓库中未跟踪的
`experiments/experiment-data-intake.md`。Transvision 的证据导出器默认只生成候选证据包，
不会自动修改论文结果注册表。

## 3. 总体架构

现有 `transvision/models/detectors/resilient_v2x.py` 改为薄编排层，核心算法拆分为：

```text
transvision/models/resilient_v2x/
├── __init__.py
├── contracts.py
├── geometry.py
├── causal_repair.py
├── ptf.py
├── routing.py
└── distillation.py
```

各文件职责如下：

- `contracts.py`
  - 定义 agent、modality、时间戳、到达状态、分支支持状态、源年龄、可靠性、
    observed/propagated 标记和 unsupported reason。
  - 提供运行时 shape、范围和协议断言。
- `geometry.py`
  - 将每个 agent/modality/source-time 特征变换到当前 ego BEV。
  - 仅补偿坐标系变化，不承担目标运动预测。
- `causal_repair.py`
  - 构造 arrived-source set。
  - 选择最新因果有效源。
  - 应用 `h <= k`、时延、位姿和标定有效性边界。
  - 计算 `gamma`、分支特征、可靠性、availability 和 age metadata。
- `ptf.py`
  - 实现 horizon-conditioned nonlinear PTF 和 linear-PTF 行为。
  - 输出累计 backward-sampling displacement `D` 和 confidence `Q`。
- `routing.py`
  - 实现双端模态聚合、LiDAR/Camera/Synergy 三专家、完整 DER descriptor、
    masked softmax 和诊断输出。
- `distillation.py`
  - 实现 frozen teacher、feature MSE 和 Bernoulli logit distillation。

完整推理数据流为：

```text
四路 history metadata
  -> 解析 causal arrival、fault、support 和 source selection
  -> 只加载/编码因果可用的 raw slices
  -> 当前 ego 坐标系对齐
  -> PTF 修复与 age decay
  -> 双端模态聚合
  -> 三专家 DER
  -> TransFusion 检测头
```

推理诊断至少包含：

- 每个分支选择的 source tick index 和 protocol time；
- source horizon；
- supported/unsupported 状态与原因；
- `D`、`Q` 和 `gamma`；
- branch reliability；
- observed/propagated flags；
- DER descriptor；
- expert support mask 和 routing weights。

unsupported branch 的 source、`D`、`Q`、`gamma` 序列化为 `null`，同时写
`ptf_queried=false` 和明确 reason；branch reliability 仍按协议写数值 0。内部 neutral
zero tensor 不能冒充预测出的 `D/Q` diagnostics。

部署配置只构建 student，不包含 teacher 参数、teacher forward 或蒸馏损失。

## 4. 数据与因果协议

### 4.1 数据集和 split

默认工程协议使用 DAIR-V2X-C cooperative vehicle-infrastructure 数据：

| 字段 | 固定值 |
|---|---|
| 数据集 | DAIR-V2X-C |
| 类别 | `Car` |
| 训练样本 | 4,813 |
| 验证样本 | 1,783 |
| 测试样本 | 2,688 |
| checkpoint selection / engineering table | `val` |
| final controlled claim | reserved one-shot `test` |
| split 文件 | `data/split_datas/cooperative-split-data.json` |
| split SHA-256 | `d048aeeca548fb194c548b798e6fc08488c4dd350ad223a154c028ed0a58de6c` |
| 检测范围 | `[0, -46.08, -3, 92.16, 46.08, 1]` 米 |

数据准备工具生成一个版本化时序 manifest。每个 target sample 必须记录：

- sequence/batch ID；
- target logical tick index `n_t` 和 protocol time `tau_t_ms`；
- ego 与 RSU 的 LiDAR/Camera 路径；
- 每个 `(agent, modality, history position)` 的 logical source tick index `n_s`、
  protocol time `tau_s_ms`、raw capture timestamp 和原始 frame ID；
- 每个 packet 的稳定 packet ID；不在 base manifest 中保存动态 delay/arrival/fault；
- ego、RSU 与传感器位姿/外参；
- 数据与标定有效性；
- split 和原始样本 ID。

历史不能跨 sequence。缺失文件、重复时间戳、非单调 source time、无法解析的位姿或标定在
数据准备阶段作为协议错误报告；运行时发现的单样本无效 metadata 进入明确的 unsupported
分支。

manifest 顶层 `content_sha256` 对去掉该字段后的 canonical JSON payload 计算；单条 sample
不保存包含自身的 manifest hash，避免自引用。

`dataset_release_sha256` 对所有被 split 引用的 raw image、point cloud、label、pose、
calibration 和官方 metadata 文件生成按相对路径排序的
`(relative_path,size,sha256)` canonical inventory，再对 inventory 取 SHA-256。prepared temporal
manifest 的 `content_sha256` 与该 release fingerprint 都进入所有 plan/materialized IDs。

每个 training/evaluation plan 在启动前另生成 immutable `transport_overlay.jsonl.zst` 和
`fault_overlay.jsonl.zst`。transport overlay 对每个 `(epoch,sample_id,packet_id)`（评估无
epoch）记录 realized `delay_ms` 与 `arrival_tau_ms`；fault overlay 记录每个
`(epoch,sample_id,agent,modality,n_s)` 的 mask。两者按固定 key 排序、使用固定 zstd 参数，
其 uncompressed canonical-record SHA-256 进入 protocol、plan 和 materialized run IDs。
模型只读取 base manifest 与当前 job overlays 的组合，禁止把动态 schedule 回写 base manifest。

DAIR cooperative record 先定义离散 logical tick index `n`；论文中的 `t/s` 映射为该 tick 的
有量纲 protocol time `tau`，而不是将 LiDAR 的 `pointcloud_timestamp` 误当作 Camera
timestamp。一个官方 paired record 定义同一 logical tick；不跨 record 做最近邻重配对或插值。
四路 raw capture timestamps 必须分别保存、
各自严格单调，且同一 logical tick 内任意两路 capture skew 不得超过 50 ms；超限 sample
在 manifest validation 时 fail closed。该 50 ms 是本文档的工程同步容差，不声称来自论文。
`n` 是每个官方 sequence 内按 paired-record 顺序生成的 contiguous ordinal。任一 modality
相邻 raw capture interval 不在 `[50,150]` ms 时在该处切分新的 protocol sequence，history
不得跨越该 gap；切分清单和因此失去完整 history 的 samples 进入 manifest。

代码中离散 index 记为 `n`，有量纲的 protocol time 定义为
`tau_n_ms=n*Delta_t_ms`。论文记号 `t/s` 在实现文档中分别对应 `tau_t/tau_s`；禁止把 tick
index 与毫秒直接相加或相除。horizon 为无量纲整数
`h=n_t-n_s=(tau_t_ms-tau_s_ms)/Delta_t_ms`。

### 4.2 时间、历史和到达语义

| 字段 | 固定值 |
|---|---|
| `Delta_t_ms` | 100 ms |
| 历史长度 `k` | 3 个 sampling intervals |
| 每个 history tensor 的长度 | 当前 cutoff 加前三帧，共 4 个位置 |
| 训练时延 | 从 0、1、2、3 帧均匀采样 |
| 固定评估时延 | 0、100、200、300 ms |
| 非整数时延 | 不做特征插值；按 `arrival_tau_ms` 选择最新已到达 source |
| `alpha` | 0.9 |

RSU arrived set 为：

```text
C_R(n_t) = {n_s | tau_s_ms + delay_ms(n_s) <= tau_t_ms}
```

固定评估时延 `d_ms` 对该样本所有 RSU source packets 使用
`delay_ms(n_s)=d_ms`，LiDAR 和 Camera 共享同一 packet delay。训练时每个 RSU source
tick 的 delay 独立、均匀采样于 `{0,100,200,300}` ms，但同一 source 的两种 modality 共享该值。
采样由 `(protocol seed, epoch, sample_id, n_s)` 的稳定 hash 决定，不依赖
dataloader worker 数量或读取顺序。

当 arrived set 为空时，不访问 `n_s_R`、RSU source mask 或 RSU delay；RSU 全部分支输出
neutral feature、零 availability、零 reliability 和 `empty_arrival_set` reason。

若 source 存在但 `h > k`，不得 clip、复制当前帧或外推；分支输出 neutral feature 和
`unsupported_horizon` reason。超出训练时延区间、时间戳无效、位姿缺失和标定缺失采用同样的
fail-closed 处理。

### 4.3 故障协议

训练时，LiDAR 和 Camera 在每个 `(agent, modality, n_t)` 上独立采样故障，
且只遮蔽 target logical tick `n_t`；更早的历史帧不随该次单帧故障一起删除。默认：

```text
p_L = 0.3
p_C = 0.3
```

训练故障由 `(protocol seed, epoch, sample_id, agent, modality, n_t)` 的
稳定 hash 决定；固定评估 condition 不采样故障，且不随 seed 改变。

数据准备阶段从训练 split 中排除无法形成任何 supported branch 的 target sample。运行时若四个
分支仍因无效 metadata 或空历史而全部 unsupported，则保留诊断、跳过该样本的普通检测监督，
并记录 `all_branches_unsupported`，不把它表述为已修复检测。

论文主表的固定故障是 **global-target-tick transient fault**：

- `L-Fail`：ego 与 RSU 的 LiDAR 在 logical tick `n_t` 同时缺失；
- `C-Fail`：ego 与 RSU 的 Camera 在 logical tick `n_t` 同时缺失。

更早、仍处于 causal history 内的 source 保持可用，因此 PTF 可以在 `h <= k` 时修复。
这里的 E+R 描述故障生成 scope，不保证在非零时延下两个 selected inputs 都被改变：
RSU `n_t` packet 尚未到达时，mask 它不会改变 RSU 当前 selected source。假设历史完整，failed
modality 的 expected horizons 为：

| fixed delay | Full `(h_E,h_R)` | L/C-Fail `(h_E,h_R)` |
|---:|---:|---:|
| 0 ms | `(0,0)` | `(1,1)` |
| 100 ms | `(0,1)` | `(1,1)` |
| 200 ms | `(0,2)` | `(1,2)` |
| 300 ms | `(0,3)` | `(1,3)` |

因此主表非零时延故障不是“两个当前输入同时丢失”的证据。第 12.5 节另设
arrival-relative source-fail 诊断，专门遮蔽实际 selected inputs。

student 处理顺序固定为：

```text
只读取 history metadata
  -> 计算 packet arrival set
  -> 应用 source-time fault mask
  -> 对每个 candidate 校验 metadata 和 h <= k
  -> 在全部 valid candidates 中选择最新 source
  -> 只加载并编码 mask 后仍可用的 history slices
  -> 几何对齐和 PTF 修复
```

unarrived、failed 或 invalid slice 的 raw image/points 不得进入 student encoder。训练期 oracle
teacher raw inputs 放在独立 `teacher_inputs` contract 中，student forward 无法读取该字段。
若最新 arrived/unmasked candidate 的 pose、calibration 或文件无效，但更早 candidate 完全有效，
选择更早 candidate，并在 diagnostics 中保留被拒 candidate 及原因；只有 valid candidate
集合为空时 branch 才 unsupported。

连续故障实验使用 LiDAR、E+R、0 ms，分别遮蔽连续 1、2、3、4 帧。长度 4 会耗尽
`k+1` 个可用 history positions，必须触发 `empty_modality_history` no-extrapolation 路径。
另用显式提供 `h=k+1` source 的协议 fixture 验证 `unsupported_horizon` 路径。

### 4.4 确定性

受控结果使用 seeds `0`、`1`、`2`。每个 run 固定 Python、NumPy、PyTorch、CUDA、
dataloader worker 和 sampler seed。协议、split、config 或 dependency lock 任一变化都会产生新的
protocol/plan ID，并进一步改变相应的 materialized training/evaluation ID。

所有 protocol 随机键先按 UTF-8 canonical JSON 编码并加 domain tag，再取 SHA-256 前 8 bytes
作为 big-endian uint64。Bernoulli 使用 `u=(x+0.5)/2^64` 并判断 `u<p`；四档 delay 使用独立
domain tag 的 `x mod 4`。augmentation 以独立 hash 产生 PyTorch Generator seed。禁止使用
Python 内置 `hash()` 或依赖 worker 调用顺序的全局 RNG。

训练设置 `cudnn.benchmark=False` 并启用框架可用的 deterministic algorithms；若 custom op
只能使用 nondeterministic kernel，必须在 environment/run manifest 明确列出，不能声称 bitwise
reproducible。受控三 seed 聚合要求相同 hardware fingerprint 和 world size。

## 5. 编码器与 BEV 表示

### 5.1 LiDAR

ego 和 RSU 共享一套 PointPillars 编码器：

- voxel size：`[0.16, 0.16, 4]` 米；
- point features 固定为 `[x,y,z,intensity]`，不附加 distance channel；
- 每个 pillar 最多 100 points；
- `max_voxels=(16000, 40000)`，分别用于 train 和 val/test；
- PillarFeatureNet 使用 max pooling、输出 64 channels，`BN1d(eps=0.001,momentum=0.01)`；
- PointPillarsScatter `output_shape=[576,576]`；
- SECOND 的 `layer_nums=[3, 5, 5]`、`layer_strides=[2, 2, 2]`、
  `out_channels=[64, 128, 256]`；
- SECONDFPN 的 `upsample_strides=[1, 2, 4]`、`out_channels=[128, 128, 128]`；
- SECOND/SECONDFPN 所有 standard conv/deconv `bias=False`，使用
  `BN2d(eps=0.001,momentum=0.01)+ReLU`，SECONDFPN `upsample_cfg=deconv`；
- 明确取 `pts_neck_out[0]`，并断言为 `[N,384,288,288]`；
- 三层 FPN feature 拼接为 384 channels，再用
  `1x1 Conv(384,256) + GroupNorm(32) + SiLU` 投影；
- 最终 BEV shape：`256 x 288 x 288`；
- 最终 BEV 分辨率：0.32 m。

### 5.2 Camera

ego 和 RSU 共享一套 camera encoder：

- 每个 agent 使用一个 DAIR 前向 camera；
- 输入尺寸：`256 x 704`；
- 使用固定、保纵横比的 bilinear resize，令宽度为 704；随后 bottom-aligned crop 到
  `256 x 704`。DAIR 标准 `1080 x 1920` 图像因此 resize 为 `396 x 704`，裁掉顶部 140 rows；
  resize 后高度不足 256 或原始尺寸不符合 manifest 时 fail closed；
- intrinsics 和 image augmentation matrix 显式包含 resize scale 与 top crop offset；
- 输入先由 BGR 转 RGB，再使用 mean `[123.675, 116.28, 103.53]`、std
  `[58.395, 57.12, 57.375]` normalization；
- ImageNet-pretrained ResNet-50，使用 stages 2、3、4 的
  `[512, 1024, 2048]` features；
- ResNet `frozen_stages=1`、`norm_eval=True`，BatchNorm affine parameters 不训练；
  pretrained weight 必须是本地 artifact 并在 training manifest 记录 SHA-256，不允许只记录 URL；
- GeneralizedLSSFPN 使用 `GroupNorm(32)+SiLU`，输出 256 channels，供
  `LSSTransform` 使用；
- 明确取 `img_neck_out[0]`，并断言为 `[N,256,32,88]`；其他 tuple levels 不传给 LSS；
- `LSSTransform` 的 image/feature size 为 `[256,704]`/`[32,88]`，
  intermediate BEV channels 为 64；
- LSS depth bins：`[1, 100)` 米，步长 1 米；
- 因此 depth bins 数为 99，LSS depth/feature head 固定为
  `Conv1x1(256,99+64)`，对 99 depth logits 做 softmax；使用 `LSSTransform` 而非
  LiDAR-assisted `DepthLSSTransform`，`downsample=1`；
- LSS bounds 为 `x=[0,92.16,0.32]`、`y=[-46.08,46.08,0.32]`、
  `z=[-3,1,4]`；
- LSS 输出再经 `1x1 Conv(64,256) + GroupNorm(32) + SiLU`；
- 输出：`256 x 288 x 288`。

所有 agent/time slice 先展平到 batch 维度进行共享编码，再恢复
`[B, agent, time, C, H, W]`，避免创建重复权重。camera slice 使用固定顺序、
每次最多两个 views 的 chunked encoding；chunk size 写入 config 和 run manifest，
训练与评估不得自动改变。

### 5.3 几何对齐

每个 source feature 先用 source agent pose、target ego pose 和 sensor extrinsics 组成
`T_(a,n_s->E,n_t)`，再通过二维 BEV backward grid sampling 对齐到 target ego frame。

BEV tensor 顺序固定为 `[B,C,Y,X]`。cell `(i,j)` 的中心为
`x=0+(j+0.5)*0.32`、`y=-46.08+(i+0.5)*0.32` 米。若 source feature 已位于 source-agent
LiDAR frame，则
`T_(a,n_s->E,n_t)=inverse(T_world_E,n_t) @ T_world_a,n_s`；backward sampler 把每个 target
cell 经 `inverse(T_(a,n_s->E,n_t))` 映射回 source grid。Camera LSS 输出先落在对应 source-agent
LiDAR frame，因此使用相同路径，不另设隐式 camera frame 约定。

要求：

- identity transform 保持 feature 不变；
- 平移与 yaw rotation 使用 detection range 与 BEV resolution 转换为规范化 grid；
- `grid_sample(mode="bilinear",padding_mode="zeros",align_corners=False)`；
- source index `(j,i)` 的 normalized coordinate 使用
  `u=2*(j+0.5)/288-1`、`v=2*(i+0.5)/288-1`，避免半 cell 偏移；
- 变换矩阵不可逆、含非有限值或 metadata 缺失时，不执行近似对齐，直接 unsupported；
- 数据增强必须同步更新所有 source pose、extrinsics 和 target boxes。

PTF displacement 定义在已经完成上述静态对齐的 target-ego grid 上，channel 顺序为
`[D_x,D_y]`，分别加到 X/column 和 Y/row sampling coordinate。

## 6. PTF

每个 modality 使用独立参数的 `HorizonConditionedPTF`，但 ego 与 RSU 共享该 modality 的 PTF。

### 6.1 输入

每个 modality 先构造一次上下文，再分别为 ego 和 RSU 的 supported branch 发起 query。
上下文包含：

- ego 四个 history positions；
- RSU 四个 history positions；
- 八个 availability masks；
- learned 64-D agent embedding，按固定 `{ego,RSU}` index 加到对应 projected feature；
- learned 64-D relative-time embedding，按固定 `{0,1,2,3}` history index 加到对应
  projected feature。

同一 modality 的八个 positions 共享 projection：
`Conv3x3(256,128,stride=2,bias=False)+GroupNorm(32)+SiLU`，再接
`Conv3x3(128,64,stride=2,bias=False)+GroupNorm(16)+SiLU`，从
`256 x 288 x 288` 降到 `64 x 72 x 72`。无效位置先乘零，再将八个 mask planes 与八个
projected features 一起拼接，因此 context 输入固定为 `8*64+8=520` channels。每个 branch
query 复用对应 learned 64-D agent embedding，并附带 requested horizon `h`；unsupported
branch 不发起 query。这里的 history features 均已由第 5.3 节变换到同一个 target ego frame。

encoder 对 unavailable positions 使用 gather-valid/encode/scatter-zero；不能先编码任意占位内容
再乘 mask。相同有效输入在 unavailable raw payload 任意变化时必须产生完全相同的 eval 输出。

### 6.2 网络

PTF context network 为：

```text
concat context
  -> 1x1 Conv(520,128,bias=False) + GroupNorm(32) + SiLU
  -> 3 x [
       3x3 Conv(128,128) + GroupNorm(32) + SiLU
       + 3x3 Conv(128,128) + GroupNorm(32)
       + identity residual + SiLU
     ]
```

`h` 归一化为 `h/k`，与 64-D query-agent embedding 拼成 65 dimensions，经
`Linear(65,128)+SiLU+Linear(128,768)` 生成三个 block 各自独立的
128-channel FiLM scale/bias。最后一层 zero-initialize，FiLM 使用
`normalized * (1+scale) + bias`，应用于每个 residual block 的第一层 GroupNorm output。

两个输出 head 为：

- displacement head：
  `Conv3x3(128,64)+GroupNorm(16)+SiLU+Conv1x1(64,2)`，定义
  `D_low=8*tanh(raw_D)` low-resolution cells；
- confidence head：
  `Conv3x3(128,64)+GroupNorm(16)+SiLU+Conv1x1(64,1)`，再接 `sigmoid`。

displacement final Conv zero-initialize；confidence final Conv weight/bias zero-initialize，使初始
`D=0`、`Q=0.5`。LiDAR 与 Camera 各自拥有完整独立的一套上述参数。

head 在 `72 x 72` 输出后 bilinear 上采样到 `288 x 288`，`align_corners=False`。
displacement 上采样后再乘 4，因此最终位移限制为每个轴正负 32 个 full-resolution BEV cells；
confidence 只做 bilinear 上采样，不做尺度乘法。

### 6.3 语义

- backward sampling 使用 `x + D(x)`；
- `h=0` 时通过显式分支强制 `D=0`；
- nonlinear PTF 直接按 requested horizon 预测累计 `D(h)`；
- linear-PTF 将整个 PTF query 的 FiLM horizon 固定为 `1/k`，预测
  `V_low=(8/k)*tanh(raw_V)` 和 one-step `Q_1`，再令 `D_low(h)=hV_low` 并使用 `Q_1`；
  因此 displacement 严格随 h 线性，且在 `h<=k` 时仍满足正负 32 个 full-resolution cells
  的总位移边界；
- `Q` 始终限制在 `[0,1]`；
- 默认不添加 motion、flow 或 confidence 标签损失；
- `D` 和 `Q` 通过 detection、feature distillation 和 logit distillation 间接学习。

## 7. 因果修复与可靠性

对每个 `(agent, modality)` 分支：

1. 枚举不越过 ego cutoff 或 RSU arrived cutoff 的 candidates；
2. 对每个 candidate 应用 fault mask，并校验文件、timestamp、pose、calibration 和 `h<=k`；
3. 从 valid candidate set 选择最大 `n_s*`；
4. 计算 `h=n_t-n_s*=(tau_t_ms-tau_s*_ms)/Delta_t_ms`；
5. 计算 `gamma=alpha^h`；
6. 应用 `gamma * warp(aligned_source, D(h))`。

可靠性定义为：

- 当前 ego observed source：`eta=1`；
- 其他 supported source：`eta=gamma * GAP(Q(h))`；
- unsupported source：`eta=0`。

每个分支同时输出：

- `b_obs`；
- `b_prop`；
- `rho=h`；
- source tick index 和 protocol time；
- unsupported reason。

模态聚合器为：

```text
concat(E, R)
  -> 1x1 Conv(512 -> 256, bias=False)
  -> GroupNorm(32)
  -> SiLU
  -> depthwise-separable residual block
```

本文所有 256-channel depthwise-separable residual block 固定为
`3x3 depthwise Conv + GroupNorm(32) + SiLU + 1x1 pointwise Conv + GroupNorm(32)
+ identity + SiLU`。输入输出 channel 都是 256。

模态可靠性为两 agent branch reliability 的平均值。

传给 DER 的四项 age 均归一化为 `rho/k`；unsupported branch 的 age 和两个 flags 均为 0，
其不可用性只由 support mask 表达。八项 flags 的固定顺序为
`[L_E_obs,L_E_prop,L_R_obs,L_R_prop,C_E_obs,C_E_prop,C_R_obs,C_R_prop]`，
四项 age 的顺序为 `[L_E,L_R,C_E,C_R]`。normalized RSU delay 不得使用 target packet 的
未来时延；该字段只读取两个 selected、arrived RSU modality packets 的 realized
delay `arrival_tau_ms-source_tau_ms`，对 supported RSU branches 取最大值并除以
`k*Delta_t_ms`，最后 clamp 到 `[0,1]`；没有 supported RSU branch 时为 0。未到达 target
packet 的 delay、固定评估 condition label 和未来 arrival time 均不得进入 descriptor。

## 8. DER

### 8.1 专家

- LiDAR expert：两个 256-channel depthwise-separable residual blocks；
- Camera expert：相同结构；
- Synergy expert：
  `1x1 Conv(512 -> 256,bias=False)+GroupNorm(32)+SiLU` 后接两个相同 residual blocks。

### 8.2 Descriptor

DER descriptor 固定为 783 dimensions：

| 组成 | 维度 |
|---|---:|
| GAP(LiDAR expert) | 256 |
| GAP(Camera expert) | 256 |
| GAP(Synergy expert) | 256 |
| `r_L, r_C` | 2 |
| LiDAR observed/propagated flags | 4 |
| Camera observed/propagated flags | 4 |
| LiDAR ego/RSU ages | 2 |
| Camera ego/RSU ages | 2 |
| normalized RSU delay | 1 |
| 总计 | 783 |

Gate 为：

```text
LayerNorm(783)
  -> Linear(783, 256)
  -> SiLU
  -> Linear(256, 3)
```

expert validity 定义为：

- LiDAR expert：至少一个 LiDAR branch supported；
- Camera expert：至少一个 Camera branch supported；
- Synergy expert：LiDAR 与 Camera aggregate 均 supported。

invalid expert 的 feature 在 GAP 和 routing 前显式乘零，防止 convolution bias 通过 descriptor
影响其他 valid expert 的权重。

至少一个 expert valid 时，在 valid experts 上执行 masked softmax，权重和为 1。
全部 invalid 时返回 `[0,0,0]`、零 fused feature 和整体 unsupported 状态。
所有 neutral feature 都是正确 shape/device/dtype 的常数零张量，不使用 learned missing token。
predict 时 all-invalid sample 直接返回空 prediction；train 时该 sample 在进入 detection head 前
从有效 batch subset 中剔除。

## 9. Detection head 与蒸馏

### 9.1 Detection head

使用 TransFusionHead：

- input channels：256；
- hidden channels：128；
- classes：1 (`Car`)；
- proposals：300；
- auxiliary prediction：enabled；
- decoder layers：1，8-head self/cross attention，FFN channels 256；
- self/cross attention dropout、FFN dropout 均为 0.1，decoder norm 为 LayerNorm，
  position encoding input/output 为 2/128；
- `num_heatmap_convs=2`、`num_heads=8`、`bn_momentum=0.1`、
  `conv_cfg=Conv1d`、`norm_cfg=BN1d`、`bias=auto`；
- `nms_kernel_size=3`，test NMS disabled；
- base voxel/grid 为 `[0.16,0.16,4]`/`[576,576,1]`，head
  `out_size_factor=2`；
- regression heads 为 center 2、height 1、dimension 3、rotation 2，code size 8；
  decode 后统一输出 `[x,y,z,length,width,height,yaw]`；
- post center range 与 detection range 完全一致，score threshold 为 0，避免在 AP 前
  丢弃低分 proposals；
- class activation：independent sigmoid；
- Gaussian heatmap 使用 overlap 0.1、minimum radius 2；
- Hungarian costs 为 focal/regression/3D IoU，weights 分别为 0.15/0.25/0.25；
- classification、heatmap、bbox loss weights 分别为 1.0/1.0/0.25，八项 bbox code
  weights 均为 1；classification focal loss 固定 `gamma=2.0,alpha=0.25`。

resolved model config 必须显式写出本节及第 5 节所有 constructor fields；schema 禁止依赖
PillarFeatureNet、SECOND、SECONDFPN、GeneralizedLSSFPN、LSSTransform 或 TransFusionHead
的省略默认值。config test 对完整 resolved dict 做 golden snapshot。

### 9.2 Teacher

Teacher 与 student 使用相同 architecture。每个 seed 单独训练一个 teacher；teacher 在同步、
全模态、无故障、0 ms 输入上训练并保存最佳 checkpoint，同 seed student 只能使用对应 seed
teacher。所有 student 以对应 same-seed teacher checkpoint strict-load 初始化；包括
no-distillation ablation，以隔离蒸馏 loss 的影响。Student 训练时：

- teacher 处于 `eval()`；
- 所有 teacher parameters 的 `requires_grad=False`；
- teacher forward 位于 `torch.no_grad()`；
- 每次构建训练任务时验证 teacher checkpoint SHA-256；
- 自动测试 teacher parameters 无梯度。

optimizer parameter groups 只包含 student；student training/deployment checkpoint 也只序列化
student state，不复制 frozen teacher weights。resume 时按 manifest 中的路径与 SHA-256 重新加载
teacher，并再次验证 hash。

### 9.3 Distillation loss

总损失为：

```text
L = L_det + lambda_feat * L_feat + lambda_logit * T_d^2 * L_bernoulli
```

固定参数：

| 字段 | 值 |
|---|---:|
| `lambda_feat` | 1.0 |
| `lambda_logit` | 1.0 |
| `T_d` | 4.0 |

`L_feat` 为 student fused feature 与 stop-gradient teacher fused feature 的 element-wise MSE，
按 `C*H*W` 归一化。

由于 TransFusion 使用 sigmoid classification，logit distillation 不使用 one-class softmax。
对同一 dense heatmap grid 和 class element：

```text
p_T = sigmoid(z_T / T_d)
p_S = sigmoid(z_S / T_d)
L_bernoulli = mean(KL(Bernoulli(p_T) || Bernoulli(p_S)))
```

概率在数值计算前 clamp 到 `[1e-6, 1-1e-6]`。检测范围内的全部 dense heatmap elements
参与蒸馏，背景不被静默排除。对 `all_branches_unsupported` 样本，检测和蒸馏损失均跳过；
该样本只产生 unsupported 诊断，不能借 teacher target 绕过 fail-closed 语义。

## 10. 训练协议

Teacher 和 student 各训练 50 epochs。

| 字段 | 固定值 |
|---|---|
| optimizer | AdamW |
| betas | `(0.9, 0.999)` |
| learning rate | `2e-4` |
| weight decay | `0.01` |
| warm-up | 前 500 optimizer-update steps 线性 warm-up |
| warm-up start ratio | `0.001` |
| schedule | cosine decay |
| minimum learning rate | `2e-6` |
| gradient clip | global norm 35 |
| precision | AMP |
| per-device batch size | 1 |
| global batch size | 4 |
| seeds | 0、1、2 |
| validation interval | 每个 epoch |

在 1、2、4 张 GPU 上分别使用 4、2、1 个 gradient accumulation steps，保持 global batch
size 为 4。训练 sampler 每个 epoch 确定性 shuffle 后 `drop_last=True`，4,813 个 train samples
中的 history-eligible 数量记为 `N_train`，由 prepared manifest 固定并进入 protocol hash。
每个 epoch 形成 `U=floor(N_train/4)` 个完整 global batches，轮换丢弃 shuffle 后不足四个的
尾部 samples；禁止 partial accumulation。因此每个成功 job 固定 `50*U` 个 optimizer
updates，并要求 `50*U>500`。LR scheduler 只在 optimizer update 后 step：updates 1--500
从 `0.001*lr` 线性升到 base LR，updates 501--`50*U` cosine decay 到 `2e-6`。其他 world
size 由协议校验器拒绝。

AMP update 顺序固定为 scaled backward、累积完成、unscale、global-norm clip 35、
optimizer step、GradScaler update、LR scheduler step；non-finite gradient 时该 run 立即失败，
不静默跳过 optimizer update。

controlled training 在启动前枚举 corruption/metadata support，并要求每个 scheduled sample
至少一个 branch supported；运行中任何 `all_branches_unsupported` sample 使 controlled job
fail，而不是改变有效 batch size。通用非受控 forward 仍可跳过该样本监督。DDP 在每个 microbatch
all-reduce supported count；local invalid rank 返回与 student parameters 相连的 differentiable
zero loss 以保持 collective 顺序。若 global supported count 为 0，不执行 optimizer/scaler/scheduler
step，并立即以 `zero_valid_global_batch` 终止 controlled run。

数据增强在所有 agent、modality 和 history positions 上保持一致：

- BEV yaw rotation：`[-pi/4, pi/4]`；
- scale：`[0.95, 1.05]`；
- translation standard deviation：`[0.2, 0.2, 0.2]` 米；
- BEV y-axis flip：概率 0.5；
- image 使用第 5.2 节的固定 resize 和 bottom-aligned crop，不执行独立 random
  crop、rotation 或 flip；
- camera intrinsics 和 image augmentation matrix 同步记录 resize/crop；
- RGB ImageNet normalization 使用第 5.2 节的固定 mean/std；
- 不使用会破坏跨 agent/history 一致性的独立 object sampling。

每个 `(seed,epoch,sample_id)` 只生成一份 augmentation record。clean teacher view 和
corrupted/delayed student view 必须复用完全相同的 BEV transform、image resize/crop matrix
和 target boxes；两者只在因果 arrival/fault mask 与 source selection 上不同。集成测试逐元素
比较 teacher/student augmentation matrices，防止 feature/logit distillation 空间错位。

teacher 和所有 multimodal student checkpoint 均按固定 Full @ 0 ms val BEV AP@0.7 最大值
选择，不查看其他最终实验 condition；LiDAR-only baseline 使用第 13.3 节的 LiDAR-Full 规则。
指标并列时先选 detection loss 更低的 checkpoint；仍并列时选更早 epoch。

## 11. 消融与容量匹配

受控变体为：

1. full nonlinear PTF + DER；
2. no PTF；
3. linear PTF；
4. static three-expert；
5. uniform gate；
6. no reliability；
7. no delay metadata；
8. no distillation。

所有 student ablation 均实例化相同 module shapes。变体仅切换 forward behavior：

- no PTF：强制 `D=0`、`Q=1`；
- linear PTF：`D(h)=hV`；
- static experts：忽略 gate hidden output，直接复用 gate 最后 `Linear(256,3)` 的三项 bias
  作为与样本无关的 logits，不新增参数；
- uniform gate：valid experts 等权；
- no reliability：descriptor 中 reliability 固定为 1/0 support indicator；
- no delay metadata：descriptor 中 age 和 delay fields 固定为 0；
- no distillation：student architecture 不变，只移除训练期蒸馏 loss。

这样所有 student 变体的参数 shape 和部署参数量一致。

容量匹配 concat baseline 使用相同 encoders、causal source selector、geometry alignment 和
detection head，以 concat fusion 替代 PTF、modality aggregation 和 DER。它保留相同的
`h<=k` support boundary，但对最新 supported feature 固定 `D=0`、`Q=1`，再 concat 四个
branch features为 1024 channels。adapter 固定拓扑为
`Conv1x1(1024,w,bias=False)+GroupNorm(32)+SiLU`，接两个 w-channel
depthwise-separable residual blocks，再接
`Conv1x1(w,256,bias=False)+GroupNorm(32)+SiLU`；w-channel block 与第 7 节结构相同，只把
channels 改为 w。容量匹配工具按 `w in {32,64,...,2048}` 穷举 trainable parameter count，
选择与 full student 绝对相对误差最小的 w，并列时取较小 w；最优误差超过 1% 时 fail。
resolved w 和两边参数明细写入 config/evidence，训练开始后不得调整。

## 12. 实验矩阵

### 12.1 主结果

执行：

```text
condition in {Full, L-Fail, C-Fail}
latency in {0, 100, 200, 300 ms}
seed in {0, 1, 2}
```

每个 seed 只训练一个 full student，共 3 个 student training jobs；对应的同 seed teacher
另有 3 个 training jobs。每个 student checkpoint 执行 12 个固定评估，因此共 36 个主结果
evaluation runs。0 ms 的三个 condition 输出 BEV AP@0.5、BEV AP@0.7、
3D AP@0.5 和 3D AP@0.7；其他时延至少输出 BEV AP@0.7。

上述首轮 36 runs 使用 val，并明确标记为 model-selection/engineering evidence，不宣称是无偏
最终结果。所有 architecture、ablation 和 config 冻结后，才允许同一批 selected checkpoints
对 reserved test 执行一次同构 36-run matrix；test 结果才可进入 final controlled claim。
读取 test metrics 后再修改模型或超参数会产生新 protocol version，旧 test 结果不得用于新版本。

### 12.2 消融

每个变体执行：

- Full @ 0 ms；
- L-Fail @ 0 ms；
- Full @ 300 ms；
- seeds 0、1、2。

每个非 full 变体、每个 seed 训练一个 student checkpoint，再用该 checkpoint 执行三个评估；
full variant 与主结果复用相同 training/evaluation ID，不重复训练或评估。

### 12.3 故障概率敏感性

训练概率 `p_L=p_C` 分别取 0、0.1、0.3、0.5；每个概率和 seed 产生独立 student
training job，再分别评估 Full @ 0 ms、L-Fail @ 0 ms 和 Full @ 300 ms。
`p=0.3` 的 training/evaluation jobs 复用 full runs。

### 12.4 连续故障

固定 LiDAR、E+R、0 ms，连续故障长度为 1、2、3、4。长度 4 必须使两个 LiDAR branch
输出 unsupported 且 LiDAR expert invalid，不能生成伪造的 LiDAR repaired feature。此时整体模型
仍可使用 Camera expert，并可报告明确标为 camera-only no-extrapolation fallback 的端到端 AP。
使用三个 full student checkpoints，`4 durations x 3 seeds = 12` 个 evaluation runs，不重新训练。

### 12.5 附加故障范围

E-only 和 R-only 对 LiDAR、Camera 分别在 0 ms 与 300 ms 执行
`arrival-relative-source-fail` 诊断：先在无 fault mask 的 arrived set 上确定每个 scoped
branch 的 `n_s*`，只遮蔽该 source 一次，再重新选择更早 valid source；不递归遮蔽第二次选择。
因此 R-only @ 300 ms 会真实改变或使 RSU branch unsupported，不与 Full @ 300 ms 构造相同
输入。结果进入证据包，但不自动填入论文主表。使用三个 full student checkpoints，共
`2 scopes x 2 modalities x 2 latencies x 3 seeds = 24` 个 evaluation runs。

### 12.6 定性结果

在读取任何模型预测前，对 val sample IDs 计算 SHA-256，以固定 protocol seed 排序并选择前
12 个样本。固定使用 seed 0 full student checkpoint 和已存在的
Full @ 0 ms、L-Fail @ 0 ms、C-Fail @ 0 ms、Full @ 300 ms predictions；相同样本用于四组
对照，禁止按预测好坏重新挑选。

## 13. 基线策略

### 13.1 Transvision 内置基线

以下方法复用现有模型代码并增加统一数据、协议和证据适配：

- FFNet；
- CoFormerNet；
- BEVFusion。

BEVFusion 增加 DAIR-V2X-C cooperative config，不使用当前 nuScenes config 生成受控结果。

适配器必须输出：

- declared capabilities：agents、modalities、history、fault 和 latency；
- model/repository commit；
- resolved config；
- checkpoint hash；
- shared split/protocol ID；
- raw predictions；
- evaluator output；
- environment；
- compatibility determination。

当前 FFNet 和 CoFormerNet 的现有路径是 LiDAR-only；适配后可进入统一的 LiDAR-only latency
reference protocol，但不能填充 multimodal Full/L-Fail/C-Fail controlled rows。只有实际接受
ego/RSU × LiDAR/Camera 四路 raw inputs 的方法才可进入 multimodal controlled comparison；
不得因为 `C-Fail` 对 LiDAR-only 模型无影响而将其标记为兼容。

### 13.2 外部基线

以下方法使用外部适配器，不在本仓库重写模型：

- V2X-ViT；
- CoBEVT；
- LRCP。

外部适配器由 declarative manifest 定义 repository path、commit、environment、prepare/train/predict
commands、output parser 和 checkpoint。缺少任一必需字段时 fail closed。只有 split、class、range、
input capabilities、fault、latency 和 evaluator 均与 shared protocol 匹配的结果才能进入
controlled rows；否则只标记为 cross-protocol reference。

### 13.3 基线 job matrix

capacity-matched concat 固定使用第 10 节完整训练协议。其他方法保留其公开、版本化 optimizer
recipe，但统一使用本设计的 split、50 epochs、global batch size 4、seeds、fault/delay sampling、
checkpoint selection、evaluator 和 evidence contract；任何偏差进入 compatibility report。

- 每个 controlled-eligible method：3 个 training jobs；
- 四路 multimodal method：每个 checkpoint、每个 split 执行第 12.1 节 12 个 conditions，
  三 seeds 共 36 个 evaluation runs；
- LiDAR-only reference method：每个 checkpoint执行
  `{LiDAR-Full, LiDAR-Fail} x {0,100,200,300 ms}`，每个 split 三 seeds 共 24 个
  evaluation runs，结果进入单独 reference table；
- 外部 manifest 缺少源码、数据、checkpoint 或可执行命令时，matrix 生成 `blocked` job
  manifest 和 reason，不生成 metrics；
- 只有在 shared training/evaluation protocol 下实际重训的方法才标记 controlled；
  仅解析 public checkpoint 的结果一律标记 cross-protocol。

LiDAR-only protocol 中，`LiDAR-Full` 表示 ego/RSU LiDAR 无 fault mask；
`LiDAR-Fail` 表示两端 LiDAR 的 global logical tick `n_t` 按第 4.3 节同时遮蔽，Camera branches
不存在。其非零时延 RSU 行为同第 4.3 节 expected-horizon table，不把它改写成
arrival-relative failure。LiDAR-only checkpoint 固定按 LiDAR-Full @ 0 ms val BEV AP@0.7
选择，并采用第 10 节相同 tie-break。

baseline evaluation 明确分两阶段：

- engineering/model-selection val：每个 multimodal method 36 runs，每个 LiDAR-only method
  24 runs；
- architecture/config 全部冻结后的 reserved test：只对 locally trained、controlled-eligible
  checkpoints 执行同构矩阵，因此每个 multimodal method 再增加 36 runs，每个 LiDAR-only
  method 再增加 24 runs，不重新训练。

所以完成 val+test 后，每个 eligible multimodal method 共 72 evaluation runs，每个 eligible
LiDAR-only method 共 48。upstream public checkpoint 只允许按 manifest 声明生成
`split=val` 的 cross-protocol reference plan，不计入上述 controlled counts；
`split=test` 无条件在 checkpoint、ground truth、dataloader、predictor 和 evaluator 访问前
阻断，也不属于 reserved-test scope。

## 14. 评估、统计与证据

### 14.1 Evaluator

新增 evaluator 直接消费 MMEngine runner 传入的 predictions。禁止从 `last_checkpoint` 重新构建并
运行另一模型。

Evaluator 固定：

- class：Car；
- range：本协议 detection range；
- box 统一为当前 ego LiDAR 坐标系
  `[x,y,z,length,width,height,yaw]`，yaw 单位为 radians；
- 坐标系为 right-handed、X forward、Y left、Z up；`z` 是 box bottom center，
  yaw=0 时 length 轴沿 +X，正 yaw 绕 +Z counter-clockwise；
- GT 与 prediction 都按 reference point `(x,y,z_bottom)` 落在 detection range 的半开区间
  `[min,max)` 进行过滤，不使用 gravity center；不应用 KITTI difficulty 或 2D image
  truncation 过滤；
- BEV IoU 使用 oriented rectangle polygon intersection；3D IoU 使用该 BEV intersection
  area 乘 z-overlap，再除以 3D union volume；
- BEV/3D AP thresholds：0.5 和 0.7；
- split：resolved protocol split；
- raw per-sample prediction export；
- 指标内部使用 float64，最终以百分数输出；
- prediction 先按 score 降序排序，同分时按 `sample_id` 和 sample 内原始 prediction index
  稳定排序；
- 每个 sample、class 和 IoU threshold 内，prediction 按上述顺序与尚未匹配的最高 IoU GT
  做一对一 greedy matching；IoU 恰好等于 threshold 时计为 TP，最高 IoU 并列时选择原始
  GT index 更小者；
- 全 split 累积 TP/FP 后计算 precision/recall；
- AP 使用 continuous interpolated PR area：在 recall 两端添加 0/1 sentinel、在 precision
  两端添加 0 sentinel，反向生成单调 precision envelope，再对每个 recall change 的矩形面积求和；
- split 中 GT 总数大于零但 prediction 为空时 AP 为 0；GT 总数为零时视为协议错误并 fail closed；
- evaluator version 固定写为 `resilient-v2x-ap-v1`，不得将其数值标记为其他 evaluator 的
  protocol-compatible 结果；
- golden fixtures 验证 matching、稳定排序、precision/recall interpolation、空预测、空 GT、
  首个 recall increment、range filtering 和百分数转换。

### 14.2 统计

聚合器只读取逐 seed `metrics.json`，计算：

- mean；
- sample standard deviation，固定 `ddof=1`；
- PDR；
- method differences；
- table best-value candidates。

PDR 使用：

```text
100 * (normal - degraded) / normal
```

fault PDR 的 `normal` 为同 method、seed、latency 下的 Full condition，`degraded` 为
L-Fail/C-Fail；latency PDR 的 `normal` 为同 method、seed、condition 下的 0 ms，`degraded`
为同 condition 的 100/200/300 ms。PDR 与方法差值先按相同 seed 配对计算，再报告三个 paired
values 的 mean 和 sample standard deviation；normal 为零、seed 无法一一配对或 operands
协议不一致时拒绝计算。

method difference 固定为 `candidate AP - reference AP`，单位为 AP percentage points。
表格 best candidate 按 metric schema 的方向在未四舍五入 mean 上选择：AP 越大越好，
PDR/latency/memory 越小越好；差值绝对值不超过 `1e-12` 时并列标记，最后才格式化显示。

### 14.3 Plan ID 与 materialized run ID

matrix 首先生成不依赖数据或 checkpoint 的 `template_id`，覆盖
method/variant/condition/latency/seed/split 与论文 result-ID slot。dataset/base manifest 可用后，
dry-run 再枚举不依赖未来 checkpoint 的 plan IDs。training plan ID 由以下 canonical fields
生成：

- method、variant、seed；
- resolved training config、training protocol、transport/fault overlay hashes；
- split、dataset release 和 prepared temporal manifest hashes；
- code commit、environment lock 和 requested hardware profile/world size；
- `teacher_checkpoint_role`，取
  `none`、`initialization_only` 或 `initialization_and_distillation`；
- upstream teacher training plan ID；role 为 `none` 时为 null。

teacher jobs 和普通 baseline jobs 使用 `none`；Resilient full/普通 ablation students 使用
`initialization_and_distillation`；no-distillation student 使用 `initialization_only`。

上游依赖完成后，materialized training ID 在 training plan fields 上增加 actual hardware
fingerprint、model initialization hash，以及 role 非 `none` 时的 resolved teacher checkpoint
hash。任何 baseline 都不被强制要求 Resilient teacher。

evaluation plan ID 由 model provenance plan ID、condition、latency/fault duration、split、
resolved evaluation config/protocol/overlay hashes、dataset/manifest hashes、code/environment、
requested hardware profile 和 evaluator version 生成。model provenance plan ID 是 local
training plan ID 或 upstream checkpoint plan ID。checkpoint 可用后，materialized evaluation
run ID 再增加 selected/upstream checkpoint hash 和 actual evaluation hardware fingerprint。

canonical JSON 使用 UTF-8、sorted keys、无额外 whitespace、`allow_nan=false` 的确定性编码。
缺少 dataset/base manifest 时 dry-run 输出完整 template coverage 和明确 blocked dependency，
但不伪造 production plan ID。数据可用时 dry-run 输出完整 plan DAG、expected counts、blocked
dependencies 和 plan IDs；执行器维护
append-only、hash-chained `plan_realizations.jsonl` 把每个 plan 映射到 materialized ID，并验证
一个 plan 只有一个被证据
聚合器接受的 successful realization。任何 training checkpoint 可被多个 evaluation plans 引用，
不允许为每个 condition 静默重训。

受控 job 启动前要求所有 tracked code/config changes 已提交；dirty tracked worktree，或
`transvision/`、`configs/`、`tools/`、`scripts/`、`tests/`、`environments/` 下存在 untracked
文件时直接失败。

### 14.4 证据包

每个 training job 生成：

```text
training_runs/<materialized-training-id>/
├── successful_attempt.json
└── attempts/<attempt-id>/
    ├── resolved_config.py
    ├── protocol.json
    ├── transport_overlay.jsonl.zst
    ├── fault_overlay.jsonl.zst
    ├── training_manifest.json
    ├── training_manifest.sha256
    ├── environment.json
    ├── train.log
    ├── checkpoints/best.pth
    ├── checkpoint.sha256
    ├── selected_checkpoint.json
    └── validation_metrics.json
```

每个 evaluation run 生成：

```text
runs/<materialized-evaluation-id>/
├── successful_attempt.json
└── attempts/<attempt-id>/
    ├── resolved_config.py
    ├── protocol.json
    ├── transport_overlay.jsonl.zst
    ├── fault_overlay.jsonl.zst
    ├── run_manifest.json
    ├── run_manifest.sha256
    ├── environment.json
    ├── model_provenance.json
    ├── predictions/
    ├── evaluator_output.json
    ├── metrics.json
    └── diagnostics/
```

两个 manifest 均使用 schema version 1，并记录除 manifest 本身及其 detached `.sha256`
文件之外各 artifact 的 SHA-256、大小和相对路径。manifest 完成 canonical serialization 后由
detached `.sha256` 文件校验，禁止自引用 hash。

`model_provenance.json` 使用 discriminated union：

- `kind=local_training`：解析 immutable training manifest、successful attempt 和 selected
  checkpoint hash；
- `kind=upstream_checkpoint`：记录 repository URL/path、commit、license、checkpoint
  URI/hash、upstream config/environment 和当前 evaluation environment；该类只能产生
  cross-protocol result，除非另有本地 shared-protocol training evidence。

attempt directory 创建后不可覆盖。非 test 的 `--force` 只创建下一个 attempt ID；
`successful_attempt.json` 以 create-exclusive 方式写一次且不可修改；一旦写入，聚合器只接受
该 attempt，后续 attempts 只能作为 diagnostics。environment 记录 GPU 型号/数量、driver、
CUDA runtime、cuDNN、CPU、RAM 和
deterministic-kernel 状态。证据聚合器拒绝缺失文件、空文件、hash mismatch、seed 不足、
protocol mismatch、hardware mismatch 和 evaluator version mismatch。

reserved test 为每个 locally trained model 建立完整 `test_matrix_plan_id` claim family：
multimodal 为 36 runs，LiDAR-only 为 24 runs。在生成第一条 test prediction 前，执行器向
append-only、hash-chained `test_access_ledger.jsonl` 写入 dataset release、matrix plan、全部
checkpoint hashes、code/config/protocol hashes 和 attempt IDs。test split 禁止 `--force`；
失败只能在相同 attempt 中 deterministic resume。一旦 ledger 或任一 prediction/metrics artifact
已存在，不允许为同 method/variant 和 dataset release 创建不同 config/protocol 的第二个 test
claim family。

证据导出器把受控结果映射到论文 `results/registry.json` 的稳定 result IDs，输出候选 JSON patch
和 evidence bundle。默认不直接修改论文仓库；显式 `--apply` 时仍必须通过论文仓库的 submission
gate。

## 15. Profiling

对 Full @ 0 ms 的 full student 和 capacity-matched concat baseline 报告：

- trainable parameter count；
- FLOPs；
- peak allocated GPU memory；
- end-to-end inference latency。

固定 profiling 协议：

| 字段 | 固定值 |
|---|---|
| batch size | 1 |
| precision | FP32 主报告；AMP 附加报告 |
| agents | 1 ego + 1 RSU |
| history | 4 positions |
| warm-up | 50 iterations |
| timed repetitions | 200 |
| synchronization | 每次计时前后 `torch.cuda.synchronize()` |
| latency statistics | median、mean、p95 |
| memory | `reset_peak_memory_stats` 后读取 `max_memory_allocated` |
| FLOP convention | fvcore，MAC 计为一个 operation |

profiling corpus 在读取模型输出前，对 Full @ 0 ms val 的 eligible sample IDs 以固定
`profile-v1` seed 做 SHA-256 排序并取前 32 个。`profile_corpus.json` 固定记录 sample IDs、
temporal manifest hash、每个 raw input file hash、四路每帧 point counts 和 image shapes。
50 次 warm-up 与 200 次 timed repetitions 都按 round-robin 遍历该 corpus；两个方法必须使用
同一 corpus。FLOPs 按 32 个 samples 分别计算并报告 min/mean/p95/max；memory 是 200 次计时
区间的全局 peak。

profiling process 固定 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、
`torch.set_num_threads(1)`、`torch.set_num_interop_threads(1)`，绑定到 manifest 记录的一颗独占
physical CPU core；`pin_memory=True`、host-to-device `non_blocking=False`、TF32 disabled、
`cudnn.benchmark=False`。mean 使用 float64；median 与 p95 使用
`numpy.quantile(method="linear")`。

latency 起点位于原始、已解码 CPU image arrays/point tensors 进入 deterministic
resize/crop、history packing 和 data preprocessor 之前，终点位于 detection post-processing
完成并同步之后。包含协议 masking、resize/crop、CPU-to-GPU transfer、voxelization、BEV
encoding、alignment、PTF、DER、detection head 和 post-processing；不包含磁盘读取、图片 JPEG
解码或 evaluator。teacher 完全排除。

FLOP 报告同时列出 unsupported operators，不把未知 operator 默认为零成本。

## 16. 实验工具布局

```text
configs/resilient_v2x/
├── _base_/
├── teacher.py
├── student.py
├── concat_capacity_matched.py
└── ablations/

transvision/dataset/
├── resilient_v2x_dataset.py
└── transforms/resilient_v2x.py

transvision/evaluation/metrics/
└── resilient_v2x_metric.py

transvision/experiments/resilient_v2x/
├── protocol.py
├── matrix.py
├── evidence.py
├── aggregate.py
├── schemas/
└── adapters/

tools/resilient_v2x/
├── prepare_data.py
├── validate_protocol.py
├── run_matrix.py
├── aggregate_results.py
├── profile.py
├── render_qualitative.py
└── export_evidence.py

scripts/
├── train_resilient_v2x_teacher.sh
├── train_resilient_v2x_student.sh
└── evaluate_resilient_v2x.sh

tests/resilient_v2x/
├── test_protocol.py
├── test_geometry.py
├── test_ptf.py
├── test_routing.py
├── test_distillation.py
├── test_metric.py
└── test_integration.py

environments/resilient_v2x/
├── environment-linux-64.lock.yml
├── constraints.txt
└── Dockerfile

docs/resilient_v2x/
└── README.md
```

`run_matrix.py` 支持：

- `--dry-run`；
- 无数据时枚举全部 matrix template IDs/result slots；有 manifest 时进一步枚举
  training/evaluation plan DAG、plan IDs、blocked dependencies 和 expected counts；
- 上游 checkpoint 产生后解析 materialized IDs；
- 按 method/variant/condition/latency/seed 过滤；
- 单 GPU；
- MMEngine distributed launcher；
- 断点续跑；
- 完成状态检查；
- deterministic command manifest。

## 17. 环境锁定

主环境固定为：

- Python 3.10；
- PyTorch 2.0.1；
- torchvision 0.15.2；
- CUDA 11.8；
- MMEngine 0.10.7；
- MMCV 2.1.0；
- MMDetection 3.2.0；
- MMDetection3D 1.3.0；
- fvcore 0.1.5.post20221221；
- zstandard 0.22.0。

可复现实验的主平台固定为 Linux x86_64；macOS 仅作为静态与纯协议开发平台，不与主实验
environment ID 混用。

实现提供：

- 精确 conda environment lock；
- pip constraints；
- NVIDIA CUDA 11.8 Dockerfile；
- environment capture script；
- custom CUDA extension build check。

最终 lock 中记录上述精确版本和下载 artifact hashes；所有版本必须先通过
MMDetection3D 1.3.0 的 compatibility check 和最小构建测试。

## 18. 错误处理

错误分为三类：

1. **协议错误**
   - split/hash 不符；
   - config 缺失；
   - 外部基线 commit 不符；
   - evaluator 或 environment 不兼容。
   - 行为：启动前失败，不产生 metrics。
2. **样本 unsupported**
   - empty arrival set；
   - empty modality history；
   - `h > k`；
   - 无效 timestamp、pose 或 calibration；
   - 全 expert unavailable。
   - 行为：neutral feature、零 reliability、明确 reason；不得静默修复。
3. **运行错误**
   - non-finite loss；
   - non-finite `D/Q`；
   - routing weights 非法；
   - checkpoint/artifact hash mismatch。
   - 行为：终止 run，写入 failed manifest，不进入聚合。

## 19. 测试与验收

### 19.1 纯协议测试

- fixed-delay 与 jitter/out-of-order arrived set；
- tick index/protocol milliseconds unit conversion；
- per-modality raw timestamp monotonicity 和 50 ms skew boundary；
- ego 与 RSU cutoff；
- source selection，包括 newest-invalid/previous-valid fallback；
- sequence boundary；
- empty history；
- `h=k` supported 与 `h=k+1` unsupported；
- E-only、R-only、E+R masks；
- global-target-tick expected-horizon table 和 arrival-relative one-shot mask；
- deterministic corruption；
- canonical protocol/training/evaluation ID；
- evidence hash 和 completeness。

### 19.2 PyTorch 核心测试

- PTF 输入输出 shape；
- `h=0` displacement 恒为零；
- 任意 supported horizon 的 displacement 不超过正负 32 个 full-resolution cells；
- 在 fixture 显式写入非零 final-head weights 后，nonlinear 和 linear horizon 行为不同；
- `Q` 位于 `[0,1]`；
- 相同样本的结果不依赖 batch 中其他样本的 latency；
- unavailable raw payload 任意改变不影响相同 valid inputs 的 eval output；
- backward warp identity、translation 和 rotation；
- decay 使用 total source age；
- DER descriptor 恰为 783 dimensions；
- valid expert weights 非负且和为 1；
- all-neutral weights 和 fused feature 为零；
- teacher 无梯度；
- teacher/student augmentation matrices 逐元素相同；
- student feature/logit distillation 有限且可反向；
- AMP forward 无 non-finite tensor。

### 19.3 配置与集成测试

- 所有 Resilient configs 可解析；
- resolved model config golden snapshot 不含隐式 constructor defaults；
- `Runner.from_cfg` 可构建 teacher、student 和每个 ablation；
- synthetic micro dataset 完成一批 loss 和 predict；
- evaluator golden fixtures；
- experiment dry-run 覆盖全部 required template IDs/result slots；synthetic manifest fixture
  进一步覆盖 plan ID realization；
- duplicate run 去重；
- DDP local/global invalid-count collective 和 controlled failure 行为；
- three-seed aggregation；
- capacity difference 不超过 1%；
- evidence export 通过 schema 与 hash 验证。

### 19.4 CUDA/数据验收

在具备 CUDA 和 DAIR-V2X-C 的环境中依次执行：

1. custom ops build；
2. one-batch finite-loss smoke test；
3. one-batch prediction 与 evaluator；
4. two-epoch teacher/student smoke training；
5. 主实验 matrix dry-run；
6. 一个 Full@0 ms、一个 L-Fail@0 ms 和一个 Full@300 ms 实际 run；
7. profiling smoke；
8. evidence bundle export。

完整三 seed 训练和表格结果仅在上述验收通过后执行。

### 19.5 本地完成声明

当前 Mac 环境的完成声明只包括实际执行通过的静态、纯协议和可用的 CPU 测试。
需要 CUDA、DAIR 数据或外部 checkpoint 的验收项必须显示为 `not executed`，不能显示为通过。

## 20. 明确不在本任务内的内容

- 生成、下载或提交 DAIR-V2X-C 数据；
- 伪造 checkpoint、日志、预测或论文结果；
- 在无 GPU/data 的本机完成完整训练；
- 将外部基线源码无许可证地复制进仓库；
- 把 cross-protocol public numbers 填入 controlled rows；
- 多 ego、多 RSU 或任意网络拓扑；
- 长于 `k` 的故障恢复；
- 标定误差鲁棒性；
- V2XSet 跨数据集实验；
- 带宽优化声明；
- 自动提交论文或推送远端分支。

## 21. 完成标准

实现阶段结束时必须满足：

1. 现有不可运行原型已被模块化、因果一致的实现替代。
2. 论文定义的四个 agent/modality 分支、PTF、reliability、DER 和 distillation 均有测试。
3. teacher、八个 student variants（1 个 full + 7 个 non-full ablations）和
   capacity-matched concat baseline 均有可解析配置。
4. 主结果、消融、敏感性、连续故障、复杂度和定性实验均可由统一工具枚举。
5. 三个内置基线和三个外部基线均有统一适配接口。
6. evaluator 直接评估传入 predictions，并通过 golden fixtures。
7. 运行证据、统计和论文 result-ID 映射均可自动生成并 fail closed。
8. 所有在当前环境可执行的检查通过；无法执行的 GPU/data 检查被准确列出。
9. 所有预期新文件先由 `git ls-files` 确认已纳入版本控制；提交前
   `git diff --cached --check`、提交后 `git show --check --oneline HEAD` 通过，且没有覆盖
   论文仓库中的未跟踪用户文件。

## 22. 实施分解

该系统按四个独立可验收阶段实施，每个阶段使用单独的 implementation plan：

1. **核心算法与契约**
   - contracts、geometry、causal repair、PTF、DER 和核心单元测试；
   - 交付可用 synthetic tensors 独立运行的模型核心。
2. **数据、检测器与蒸馏**
   - 时序 manifest、transforms、四路 encoder、TransFusion、teacher/student 和训练配置；
   - 交付可由 `Runner.from_cfg` 构建并完成 synthetic one-batch loss/predict 的系统。
3. **评估、实验与证据**
   - evaluator、矩阵编排、统计、profiling、定性渲染和 evidence bundle；
   - 交付覆盖论文 controlled result IDs 对应 templates、并能在 manifest fixture 上解析
     plan IDs 的 deterministic dry-run。
4. **基线适配与可复现环境**
   - 三个内置基线、三个外部适配器、环境 lock、Docker、运行文档和最终回归；
   - 交付统一协议下的基线入口；外部资产缺失时保持明确 fail-closed。

阶段之间只通过本设计定义的协议、tensor 和 evidence interfaces 连接。后续阶段不得通过修改前一
阶段的语义来绕过测试或实验门禁。
