# Recover Before Fuse 代码缺口处理

依据 thesis 的 `experiments.md`、`method.md` 与 2026-10-05 用户决定：V2V4Real
使用官方划分、合并 vehicle 评价；SPD 保留 car 评价。此记录只确认实现与软件检查。

## 已补代码

| 缺口 | 实现 | 软件证据与范围 |
|---|---|---|
| recovery-off 仅能 bound 分配 | `recovery_off_allocation.py`：受限历史支持下的 teacher、learned；独立源码/配置 checkpoint 绑定 | teacher→export→fit→load→learned 合成路径；不复用 unrestricted 模型 |
| recovery-off learned 无独立排序检查 | `rbf_independent_recovery_off_learned_trajectory.py` 与 CLI | 从前一事件提交历史重建限制、候选、18 特征、NumPy MLP 排序和成本；连同执行路径 69 项通过 |
| bound 消融尚无完整独立 CPU 入口 | `prepare_recovery_off_full_independent_CPU.py` 与结构/动作检查器 | 原冻结生产器下的全链夹具及负例 92 项通过；未启动真实全队列验收 |
| V2V4Real 严格 Car 与新 vehicle 口径混用 | `paper_evaluation_policy.py`、GT 生成与独立原码 oracle、`fit_v2v4real_vehicle_calibration.py` | 保留 raw_class，按官方规则合并；校准只使用 train，旧 Car 回执不自动升级 |
| vehicle 校准 GT 横跨多个原始 volume | `merge_v2v4real_vehicle_ground_truth.py` | 冻结 partition 选取 calibration-fit；拒绝重复序列、不完整帧与跨口径合并，17 项检查通过；实际 1,017 帧合并已独立验收 |
| 新 vehicle 校准缺少独立全量验收 | `verify_v2v4real_vehicle_calibration.py` | 独立匹配、梯度/受约束驻点、概率与诊断；16 项软件检查通过，三种子实际 386,238 个候选全量验收通过 |
| 新校准不能接到已有原始候选 | `build_native_paper_cache.py --raw-cache` | 仅重新计算 calibrated scores；其他数组逐字节不变；拒绝先 NMS 的旧缓存 |
| 原始导出器只接受 train，无法产生 official_test 原始候选 | `run_v2v4real_official_test_raw_features.py` | 私有加载固定原始推断核心，绑定 checkpoint/test projection；保留 3,986 个源帧及空候选帧。补齐导入前的原冻结扩展构建，27 项检查通过（原 18 项及新增 9 项）；真实 Linux/CUDA 运行待验 |
| 原始检测输出缺少合法 9D 缓存接线 | `build_v2v4real_raw_native_cache.py` | 支持独立 train/test envelope；显式因果速度和完整协方差合同，空帧保留，34 项检查通过；执行源码在开始、转换后和读回后核验 |
| 原生指标与 RBF 世界坐标输出未接通 | `evaluate_v2v4real_native_vehicle.py --convert --evaluate-converted` | 消费显式姿态/官方帧映射，保留全 9 序列 1,993 帧；调用固定官方 3D IoU 评价器，AMOTP 方向为高优 |
| 公开基线仅检查进程退出 | `public_native_outputs.py`、`run_public_baseline.py` | CoopTrack 显式保存原始预测、收集原生评价目录；DMSTrack 九序列格式与摘要读取；SparseCoop 旧收集保留 |

恢复关闭 teacher/learned 的只读源码、命令、原始失败日志与 69 项检查回执：
`test/recover-before-fuse/source-freezes/rbf-recovery-off-teacher-learned-code-v1-20261005/software-preparation.json`。
原 bound 模式在 8 组预算/宽度配置的 24 个事件中预测及 audit 字节保持一致。

bound 完整 CPU 检查器准备回执：
`test/recover-before-fuse/source-freezes/rbf-recovery-off-bound-full-independent-CPU-v1-20261005/qualification.json`。

Vehicle GT/校准/缓存/协议相关检查 75 项通过，另有 17 项独立 metric 检查通过；
两个最初因环境失败的既有 pipeline 回归已在现成 torch260 环境中重跑，2 项通过。
训练计划也已纳入新类别规则模块的源码哈希；对应检测/训练回归 2 项再次通过。
原始失败日志保留。公开基线检查 55 项通过，包含直接执行固定官方 `save_results`
函数生成文本再由新收集器读取的对照。以上套件可能覆盖相关功能，不相加为实验数量。

原生坐标转换/评价的 60 项检查通过，包含真正调用固定官方评价器的完整空帧夹具、
完美/平移框、0.25 IoU 阈值边界和输入篡改拒绝。原参考公式、匹配规则、阈值未改；
原 numba 函数在 `NUMBA_DISABLE_JIT=1` 下执行相同 NumPy 函数体，运行记录明确保留此设置。
首轮组合测试受 macOS 沙箱共享内存限制，最终同一组测试在许可环境下通过；失败日志保留。

## 后续代码、运行与资产

M0–M4、主路径 teacher/learned、资源扫描器、JPDA/PKF/MHT 后端和论文表图入口均已存在。
`scan_paper_resources.py` 可冻结候选×三种子并逐项启动完整 schedule 回放和独立评价；
尚缺资源清单或真实产物不等于缺少扫描算法。

仍需真实教师数据、训练后的 checkpoint、完整回放、独立验收及同资源比较。
V2V4Real 新 vehicle GT、校准、姿态/官方帧映射、运动及协方差合同须绑定真实输入；
通用接口或合成夹具不会生成这些证据。已知 train/test 重叠继续披露；物理会话映射
不再作为用户授权的官方基准路径前置条件。

公开基线还缺可复现运行环境、实际权重/特征和输入映射。Long-SCOPE 的完整作者实现
或足以复现的算法细节与权重仍未获得，未用其他跟踪器冒名补齐。

official_test 原始候选入口已新增 `--rank/--world-size`，按排序后的原始完整双端
帧对序号取模分配，保留空检测帧。分片使用独立 kind，不能直接交给完整缓存消费者。
`merge_v2v4real_official_test_raw_shards.py` 核验全部 rank、原始 3,986 行覆盖、
帧身份/姿态/PCD 哈希、源码与 checkpoint、逐产物字节、候选 schema 和计数；
合并后才生成原完整 envelope。软件夹具不代表真实 GPU 输出验收。
`run_v2v4real_official_test_gpu_shards.py` 已接通固定云输入下载、投影逐成员核验、
四/八卡子进程与完整合并；`prepare_v2v4real_official_test_gpu_source.py` 将三份
最小运行源码按字节嵌入 ClearML script。`submit_v2v4real_official_test_raw_gpu.py`
按 checkpoint/seed/原始 frames 去重，读取上传后的 script 并重新检查物理 GPU
交集后才排队。`readback_v2v4real_official_test_raw_gpu.py` 只接受 completed 的
候选，独立下载九项产物并核对字节、schema 和全部源帧覆盖；不宣称检测器数值、
运动/协方差、跟踪或论文指标通过。真实任务执行与回读仍待验收。
原生扩展在目标 Linux/CUDA 环境的真实构建和导入也仍待验收。

所有已有远端任务、失败记录和冻结产物保留，代码准备未重启实验。
2026-10-05 用户随后明确授权“阶段性 C+P ALL，然后继续完成后续实验”；
该次阶段性提交推送的实际结果另记执行台账，后续仍不自动提交推送。

## 本轮代码与证据索引

汇总回执：`test/recover-before-fuse/receipts/rbf-remaining-code-gap-resolution-20261005.json`。
各模块源码快照、命令、测试日志与子回执均在汇总回执中逐项绑定 SHA256；
实验族清单只检查入口存在与语法，不把它的 14 个类别当作已完成实验数。

下一步依赖顺序为：读取已有 GPU 任务的真实终态和产物 → 完整独立 CPU 验收；
V2V4Real 则新建 vehicle GT 与 train-only 校准，生成 GT-free official_test 原始候选，
接入经验证的因果速度/完整协方差和帧映射，再运行模型与官方评价。
这些任务未因本轮软件检查通过而被标为完成。

`prepare_v2v4real_official_test_vehicle_gt.py` 已补齐官方 test 三 volume 的
评价专用 GT 准备与完整合并：按冻结 test projection 的 ego 与全部原始帧生成，
逐 volume 调用既有固定 native GT oracle；合并必须覆盖全部九序列/1,993 paired
帧，空 GT 帧不能删除。24 项覆盖边界与原训练 GT 合并回归通过。
真实 CPU 执行已启动，输入为既有三个已核验 ZIP；启动回执在
`test/recover-before-fuse/artifacts/v2v4real-native-official-test-vehicle-GT-v1-20261005/launch-receipt.json`。
终态与完整独立原码对照仍需核验。该产物仅供最终独立评价，未参与训练、
校准、预测、恢复决策或任何模型选择，也不代表已有跟踪指标。

本轮真实 official_test GT 已完成并独立读回：三 volume 共九序列、1,993 paired
帧、3,986 源帧、31,419 vehicle 对象，与固定原码最大角点误差 0。
回执为 `test/recover-before-fuse/artifacts/v2v4real-native-official-test-vehicle-GT-native-identity-fix-v3-20261005/independent-readback-receipt.json`。
原 test_02 两次失败保留：先是被官方排除的 Pedestrian 含负尺寸，再是同端重复
身份。本机实现现按官方顺序先排除 Pedestrian，并在 vehicle 路径保留第一个
有效重复身份框；严格 Car 历史路径不改变。原码重复身份/ROI 夹具及 GT 回归通过。
已通过的 test_01/test_03 复用，未重复解压和生成。

official_test raw GPU v1 的 seed1337/2027 在下载阶段失败，未生成预测产物。
两失败产物已使用本机会话认证独立读回并与云注册哈希一致。原 StorageHelper
未可靠携带认证，下载入口现显式携带会话 token 且禁止重定向；token 不写入源码、
产物或日志。新 `authenticated-download-v2` 派发路径保留失败前驱，拒绝运行中、
已成功、有预测产物或不匹配的前驱，并按配置/种子去重。52 项链路检查及 17 项
包装/重试边界检查通过（有重叠，不相加为实验数）。冻结源码为
`test/recover-before-fuse/source-freezes/v2v4real-official-test-authenticated-raw-GPU-chain-20261005T035733360785Z`。
新版本仍需真实 Linux/CUDA 执行、完整输出独立回读和后续数值验收。

认证版八卡3090 seed1337 又在访问注册文件服务 `.35` NIC 时 ConnectTimeout；
新失败产物已独立读回，历史任务保留。运输层现固定使用同服务 `.34` NIC，保留
原路径并继续完整大小/SHA 核验、记录注册与实际服务 host。原 source 的
3,979,124 字节已通过新运输层读取，SHA 与原注册值一致；这仅证明本机读取。
新冻结版本为
`test/recover-before-fuse/source-freezes/v2v4real-official-test-authenticated-LAN34-raw-GPU-chain-20261005T040454236249Z`。
派发 `--attempt authenticated-LAN34-v3` 仅接续失败前驱，运行中或成功的前驱保留。

LAN34 seed1337 已完成大投影包下载/原 SHA 检查，但旧解包器错误要求 archive
根目录含 publication 回执，任务失败。原真实回执是独立 ClearML artifact
`publication-receipt`，675,616 字节，SHA
`530f97f1d74a72dd7979a3999190b7e64972a3455d39d88795bc4061a8809231`；
本轮已读回。原 tar 的成员有 `projection/` 前缀，与该外部回执的相对 inventory
逐一对应。新解包器绑定外部回执、校验全部原成员，保留原前缀，拒绝重复、缺失、
多余和大小/哈希错误的成员；不伪造或重新发布输入。18 项包装/重试边界检查通过。
新冻结为
`test/recover-before-fuse/source-freezes/v2v4real-official-test-original-publication-raw-GPU-chain-20261005T041435955980Z`。
`--attempt original-publication-envelope-v4` 查最新已有前驱，活前驱不重跑并保留
它的物理 GPU 绑定。以上仍需目标环境与完整输出独立验收。

随后 v4 seed1337 在额外自回执成员检查处失败。原发布源码已从哈希匹配的原
source archive 读取：先写 `projection/publication-receipt.json`，再把它加入 tar
清单，因此 JSON 内列3,993项，archive有3,994项；差额恰为675,616字节自回执。
新实现只把这个原路径、原 SHA 固定的自回执加入预期集合，其他额外文件仍拒绝。
原发布源码、独立回执及完整原归档读回总量一致的证据在
`test/recover-before-fuse/artifacts/v2v4real-original-publication-self-member-contract-20261005/acceptance-receipt.json`。
缺失/额外/篡改自回执负例及重试边界检查通过；最新冻结为
`test/recover-before-fuse/source-freezes/v2v4real-official-test-original-self-member-raw-GPU-chain-20261005T042627381807Z`，
派发参数 `--attempt original-publication-self-member-v5`。旧v4及其他失败继续保留。

阶段提交后已继续完成真实 vehicle 校准 GT：三个 train volume 共 1,775 paired 帧
全部与固定原生参考逐帧比对，最大角点误差 0；按原 partition 保留 6 序列、
1,017 paired 帧、16,787 vehicle 对象。独立回执为
`test/recover-before-fuse/artifacts/v2v4real-native-vehicle-calibration-GT-v1-20261005/independent-GT-union-receipt.json`。
这项 GT 验收不包含校准参数、跟踪或论文指标。新增合并代码属于阶段提交后的工作。

后续三种子 train-only vehicle 存在概率校准已拟合并完成全量独立数值验收，
输入、GT、原几何回执及 ego 参考系逐项绑定；共 386,238 个候选，概率最大差
2.220446049250313e-16。回执为
`test/recover-before-fuse/artifacts/v2v4real-native-vehicle-calibration-independent-v1-20261005/acceptance-receipt.json`。
生产候选的验收标志保持原值，独立结论另存；该结果未拟合速度/协方差，也未验收
9D 原生缓存、跟踪性能或论文指标。初次缺 cryptography 的环境失败完整保留，
成功运行使用已有 equivalence-py312 环境。

2026-10-05 runtime-attested-v6：v5 两个任务均 failed；实际 Agent 日志均声明
`CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL`，因此云端固定 requirements 没有执行。
seed1337 八份 rank 日志确认原 Cython 构建成功后缺 shapely；失败产物已独立读回。
新增 source-bound 运行时入口，在任何 source/projection/checkpoint 下载前，以
rank 子进程同一 sys.executable 检查全部原声明包的版本和实际 import；缺项时安装
原固定版本，然后启动全新解释器重新核验并上传 runtime-dependencies 回执。
包管理原始输出不写入日志或云产物，避免私有 index 凭据泄露。未修改模型、推断
核心、检查点、数据或数值容差。24 项软件检查通过不等于实际运行时或推断验收。
重试版本 runtime-attested-v6 只消费失败、同 checkpoint/frame 且无预测输出的
前驱；仍运行或成功的实验保持不动。

v6 seed1337 Task d9b8c32212c24a59b3245c9cb9faee2d 实际安装 Python 包后，
全新解释器 import gate 仍拒绝 open3d；尚不能确认错误是缺 OS 库还是其他导入依赖。
seed2027 Task8b8160e461824722879ab88dee9ed0a6 继续实时核验，未因观察等待重启。
新增 runtime-diagnostics-v7 使失败后也先上传 before/after 精确包版本、import
错误类型及白名单 shared-library basename，不输出完整错误文本或包管理器输出。
25 项软件检查通过；该源码冻结于
`test/recover-before-fuse/source-freezes/v2v4real-official-test-runtime-diagnostics-raw-GPU-chain-20261005T045158602363Z`。
此版本不安装任何猜测的 OS 依赖，实际失败产物将决定下一步环境修复。

v7 seed1337 Task5175ee72403b4b6593210c841b4302f8 已 failed，其4622字节运行时
回执已独立读回并验SHA bf738504c80662f28a89f64e9eae26b8d5bb3104d532542474eaee7054ead91d。
十二项固定 Python 包版本正确，只有 open3d import 失败：OSError 缺 libX11.so.6。
详见 `test/recover-before-fuse/receipts/v2v4real-runtime-v7-live-20261005T045444116073Z.json`。
v8 在实际 import 报错的已知共享库范围内，安装对应系统包并记录 dpkg 版本、架构，
每次安装后重新启动解释器核验；未知库/非root/非Linux仍拒绝，不猜测依赖、不放宽
版本或数值门槛。27软件检查通过，实际环境与推断仍需独立验收。
v8 冻结目录为
`test/recover-before-fuse/source-freezes/v2v4real-official-test-runtime-system-libraries-raw-GPU-chain-20261005T045635384924Z`。

后续 raw 云读回增加 runtime-dependencies 为第十项必需产物；验收器核对实际执行
runtime 源SHA、固定十二项Python包版本及import结果、同一rank解释器、OS包版本/
架构和原始输出不持久化的声明。预测仍逐行绑定原3986源帧、九序列和源点云/位姿。
21项验收器检查通过（含版本/import/source/interpreter/scope/OS包篡改拒绝及完整
3986空帧schema/覆盖反例）。这不替代独立检测数值、原生9D运动、跟踪或论文指标。
GPU运行中的源码未因验收器更新而修改、取消或重跑。

v8 seed1337 Task476db1732ca844d69e8e4ad34751295d failed，OS安装返回100；
另一seed保持实时核验，不据此推断终态。发现已登记ClearML依赖任务
1e0730c280e846bebd92cc4de49893e3的离线GL/X11包含当前缺失库，archive1141794B
SHA5ac68c58a292e435a0ee55c98f0bd2720a9b088343afc813c262bdc552cc0e10，
manifest3005B SHA2905061e79777c2858471fa2ee3ccbfaf3cf064fa62429dfa4e1424d34530f82。
v9只在实际缺库且当前Linux x86_64时下载这个已登记包，严格验十个普通archive
member（九库与manifest）的大小/SHA，拒绝symlink/未知/重复/截断成员，不包括
libc/loader；仅设置当前容器子进程LD_LIBRARY_PATH，重新启动解释器import核验。
不继承旧SPD运行验收，不修改宿主机、不调用apt，不修改模型/原core/容差。
28软件检查通过，8独立读回边界检查通过，包含真实既有库包全量验证和篡改拒绝。
冻结目录：
`test/recover-before-fuse/source-freezes/v2v4real-official-test-runtime-offline-libraries-raw-GPU-chain-20261005T050952798157Z`。
其independent-checker绑定同runtime源码，并另读ClearML原manifest比对全部库字节
声明，仍不替代检测数值、9D、跟踪或论文验收。

### 原冻结点云读取器的延迟包导入（v10）

v9 seed1337 的十二个固定 Python 包及 Open3D 实际导入已完成源绑定独立核验，
但八个推断 rank 失败；原 `v2v4real_pcd.read_native_pcd` 延迟导入
`tools.event_track_v2x.prepare_v2v4real_inputs._read` 时缺少原 project 路径。
失败任务、十项原字节回读和限定范围的运行时验收均保留。

v10 在新 rank 解释器中先拒绝已加载的 ambient tools/transvision 命名空间，再将
经过原 archive SHA 和全部 403 文件检查的 project 路径绑定到原 preparer 和 inputs
模块，核验实际模块来源与原 SHA。原 PCD 函数、bounded `_read`、检测 core、权重、
扩展定义、候选协议和数值容差不改。输出另记录两个原模块的路径和 SHA。
隔离解释器实际调用原 PCD 函数，并验证 RGB 强度/不可变数组、错误 SHA、符号链接、
污染命名空间和篡改源码拒绝；五项新检查与九项扩展检查通过，另外 36 项包装、
运行时和读回边界检查通过。软件检查不代表 GPU 推断或论文指标完成。

### 原生检测数值检查的后续准备

`v2v4real_native_numeric_reference.py` 提供独立 NumPy float64 检测头解码、
sigmoid、固定阈值及 stable top64、双线性零填充、128 通道池化/L2 和原七维布局
参考。原设备 float32 全 anchor 分数必须先与数学参考按固定 1e-4 容差核验，再
按实际分数精确检查稳定排序和阈值；不以数学参考的不同舍入改变原 tie 规则。
只给 selected 分数不足以验收竞争空间。完整逐行检查还需真实 head/BEV 证据。

`capture_v2v4real_native_heads.py` 用只读 hooks 获取这些真实张量，核对原 core
SHA、禁止 TF32、不改变 predict 返回对象，并在成功/失败时清理全部新增 hooks。
原模型、权重、输入处理和 core 源码不改。六项 stub 网络 hook 边界检查及七项
原 SHA 绑定 Torch 数学对照检查通过。当前只完成代码准备；尚未形成完整原帧/
checkpoint 绑定的 GPU 采集、独立远端 CPU 数值验收和 cohort 回执，不宣称网络
forward、9D、跟踪或论文指标已经通过。
### Independent trace transport and coverage checker (2026-10-05)

`tools/event_track_v2x/verify_v2v4real_native_numeric_traces.py` now checks a pinned
trace index against a pinned accepted raw index, including empty prediction
rows. It requires exact frame/PCD, checkpoint, original source/core and raw-index
bindings, verifies each NPZ SHA before loading, rejects missing/duplicate frames
and escaping paths, and invokes the independent NumPy post-head reference.
Progress reports actual checked rows and an ETA limited to this check.

Six synthetic transport/binding tests passed. No real GPU trace has been
collected or independently accepted with this checker. The original-frame GPU
driver, trace storage sizing/packaging, source freeze and cloud dispatch still
need preparation. Network forward, causal 9D motion/covariance, tracking and
paper metrics remain outside this check's acceptance scope.

The row capture driver is now implemented in
`tools/event_track_v2x/capture_v2v4real_native_numeric_rows.py`. It requires the
original frozen CUDA detector and registered seed checkpoint, exact accepted
raw-row coverage and PCD hashes, runs the original point reader and masks, and
captures read-only heads/BEV while comparing returned outputs with accepted raw
payloads. It stores lossless traces with per-frame/source/checkpoint bindings
and limits storage through an explicit byte budget; exceeding that budget
cannot yield a completion index. Five synthetic boundary tests passed. No
actual GPU execution has passed. The standalone runtime wrapper, practical
trace sizing/packaging and independent cloud readback remain necessary before
dispatch and acceptance.

`tools/event_track_v2x/run_v2v4real_native_numeric_capture.py` now supplies a
standalone CLI for the original archive, frozen extensions, point-reader binding,
official frame admission, registered checkpoint, deterministic CUDA setup and
read-only trace loop. It requires the complete 3986-frame original schedule and
accepted raw rows. CLI import/help and three fail-before-runtime tests passed;
this does not qualify an actual CUDA runtime. Practical full-BEV trace capacity,
packaging/transport and independent cloud readback remain unresolved, so no new
GPU capture experiment has been dispatched.

The default numerical trace layout is now `compact-cells-v1`: complete original
classification/regression heads, anchors and all-anchor float32 scores remain;
the BEV trace retains the exact float32 channels of every spatial cell needed
by independent bilinear sampling of all selected candidates. Cell coordinates
are canonical, complete and unquantized. Unsampled BEV values are outside this
evidence scope, so this still does not independently verify the full network.
Eight new synthetic checks cover exact agreement with the full BEV reference,
boundary/outside sampling, empty predictions, missing/duplicate cells and
quantization rejection. Together with affected reference/transport/driver/runtime
tests, 29 checks passed. The capture driver records a conservative array-byte
capacity bound (at most 256 cells per frame), retains failed preflights, and
cannot produce a completion index when the explicit budget is exceeded.
Actual CUDA traces, measured storage sizes, cloud packaging and independent
readback are still pending; no new GPU task was submitted for this preparation.

`tools/event_track_v2x/v2v4real_numeric_trace_archive.py` now provides create-once
transport packaging and byte readback for the full 3986-row compact trace
cohort. It binds the accepted raw index, checks every payload SHA, rejects
extra/duplicate files and unsafe members, budgets TAR overhead before writing,
and records a separate manifest without recursive self-hashing. Readback checks
the registered archive/manifest identity and every member before numerical
checking. Five synthetic transport tests passed. This prepares cloud transport;
it does not upload, dispatch, collect real traces or accept numerical results.

The numeric capture CLI now partitions the original complete frame pairs across
four or eight ranks while retaining the full accepted raw index binding. A
create-once shard merger checks exact disjoint original row coverage, capacity
proofs and payload hashes before copying unchanged traces into a full cohort.
Ten affected sharding/driver/runtime software checks passed. These are synthetic
transport checks, not actual GPU collection or independent numerical acceptance.
ClearML input admission, the GPU task wrapper and independent cloud readback
remain to be connected before dispatch.

`admit_v2v4real_numeric_raw_inputs.py` binds fresh runtime extraction to the
previously accepted seed/task proof and pinned raw archive, manifest and index.
It requires all 3986 original rows, the exact ordinary-member inventory, per
member hashes and a positive unpacked-byte budget. Four synthetic transport and
wrong-proof/seed/capacity checks passed. It does not reclassify the existing raw
results as new acceptance. The cloud wrapper still needs to supply these pinned
assets and run actual GPU capture followed by independent numerical readback.

`run_v2v4real_numeric_capture_gpu_shards.py` now connects pinned original
source/projection/checkpoints and the already accepted raw inputs to four/eight
rank capture subprocesses, exact full-cohort trace merging, budgeted packaging
and ClearML candidate uploads. It preserves failed rank logs and stops only its
own surviving child processes on failure. Import and five admission boundary
checks passed. No CUDA trace has been collected with this wrapper; the source
bundle/dispatcher, measured storage admission and independent cloud numerical
readback must still be qualified before an actual task is dispatched.

A fresh-interpreter review found that the numeric CLI itself imported the
`tools` namespace before the unchanged original point-reader binder could claim
it. Direct rank execution now loads numeric helpers in a private package with
relative helper imports; ambient `tools`/`transvision` remain rejected. Original
source binding and original point-reader code are unchanged. Two fresh-process
namespace checks and affected runtime/driver/compact checks passed (18 total).
This removes a pre-CUDA import conflict; actual CUDA/runtime acceptance is still
pending. Earlier preparation freezes are preserved as history.

The numeric GPU bootstrap now embeds the full 14-file execution dependency
closure and three existing accepted raw proofs. Bundle and individual file
hashes bind materialization; damaged bytes fail before writing the execution
root. A cold import test loads the wrapper from only the generated code tree
and confirms the direct rank keeps original namespaces available. Three new
bundle checks passed. The actual three seed proofs were bound by their existing
registered SHA values in the frozen candidate bundle. No upload or GPU dispatch
occurred; storage admission, parent progress reporting, independent cloud
numerical readback and a source-bound collision-safe dispatcher remain pending.

The numeric GPU wrapper now checks current container filesystem free bytes
before downloading inputs against declared trace, merge/archive, raw extraction
and conservative runtime/metadata budgets. This measures free space and does
not claim a host reservation. It also relays bounded rank JSON progress without
printing unrelated log text, counts original rows once, and leaves ETA unknown
until every rank has real progress. The ETA covers only trace capture. Eight
storage/progress and affected wrapper checks passed; the updated source bundle
was frozen with the existing proof pins. Actual worker capacity and GPU trace
execution are still unaccepted; independent cloud readback and collision-safe
source-bound dispatch remain to be qualified.

The first cloud numerical readback CLI now admits completed exact-script/bundle
candidates, binds the existing accepted seed proof, reads trace/runtime/rank
artifacts, checks archive members and runs full-row NumPy post-head mathematics.
Its lidar range comes from the pinned original source archive. Accepted local raw
bytes are transported into a fresh comparison directory without new raw export.
Numeric rank receipts now attest the actual rank interpreter, fixed runtime
package imports and TF32 state. Ten source/status/runtime admission and affected
isolation checks passed. This is preparation, not a completed cloud acceptance;
original point-reader/extension evidence, exact execution input attestations and
full readback boundary qualification remain before dispatch.

### Original input and rank attestation preparation (2026-10-05 10:25 UTC)

The numerical readback now checks the exact seven registered execution inputs
against their original owning Tasks, sizes, hashes and allowed transport hosts.
Each rank must bind the two point-reader modules to their original archive bytes,
the original extension builder AST, exact build/import commands, interpreter and
unchanged source inventory. The wrapper uploads each extension receipt and built
binary separately; the reader checks their SHA and size against the rank binding.
Thirteen new substitution and boundary checks passed. The source, three accepted
raw proofs and independent checker are frozen in
`v2v4real-numeric-original-attestation-preparation-20261005T102509846579Z`.
This remains software preparation. Cloud integration review, collision-safe
source-bound dispatch and actual complete GPU traces with independent numerical
readback remain; no network forward, 9D or tracking acceptance is claimed.

### Source-bound numerical dispatcher preparation (2026-10-05 10:35 UTC)

The new dispatcher preserves existing candidates, checks the original registered
assets, serializes local dispatches, selects idle physical GPU intersections with
the existing fleet policy, and rechecks that fleet after uploaded source and all
parameters are read back. Positive storage budgets and an exact-source dispatch
qualification are mandatory. Nine new deduplication and pre-authentication
boundary checks passed. This dispatcher was frozen separately in
`v2v4real-numeric-dispatch-preparation-20261005T103512769626Z`; it has not been
executed against the cloud or certified for dispatch. Complete integration and
readback boundary review still precede creating its dispatch qualification.

### Completed Top1 seed2027 cohort (2026-10-05 10:44 UTC)

The existing CPU reader completed all 46 original supplied train sequences:
7,445 events, 225,177 observations, 1,150,265 stored branch states and 205,435
chosen predictions. Its final acceptance SHA is
`190ec3bef2184a1d09bf9c4b41ab89f3b78c4cb92a59ed365981ba7c060b9ee3`.
Readback of all 46 sequence receipts matched the final event, observation, state
and prediction totals and maximum state error `3.228706191293895e-11`.
This certifies its declared causal raw commits, 203 feature contexts and fresh
state/output numeric scope. Exclusive recovery, equal-resource performance,
complete online method and paper results remain unaccepted. Reader PID3256
finished naturally and must not be restarted. Other CPU readers remain active.

The numerical cloud reader additionally requires the complete 3,986-row execution
scope, exact registered storage arithmetic and actual free-byte admission,
per-rank original frame-pair counts and actual CUDA device fields. Eight new
execution boundary checks passed; the independent checker was frozen in
`v2v4real-numeric-execution-boundary-preparation-20261005T105008751715Z`.
The GPU bootstrap and bundle hashes are unchanged. No numerical capture was
uploaded or dispatched in this heartbeat.

### First numeric dispatch stopped before enqueue (2026-10-05 11:05 UTC)

The uploaded parameter readback gate rejected the first candidate before queueing.
ClearML returned integers and booleans as their exact string representations.
A local artifact loop also shadowed the intended task name. The retained Task is
`10ad70300e824f4f907863084f274ced`, currently `created` with name `manifest`;
no GPU execution began. The dispatcher now checks exact SDK serialization and
keeps the candidate name separate from artifact names. Seven new regression
checks passed. The corrected dispatcher invalidates its previous qualification
SHA. The next step must qualify that correction and resume this same candidate
with exact script/config readback and fresh physical GPU collision admission;
creating a duplicate is not authorized by this correction.

### Same-candidate resume and numerical capture dispatch (2026-10-05 11:20 UTC)

Seven new resume checks passed. The exact source, checker and corrected dispatcher
were qualified in `v2v4real-numeric-qualified-same-task-resume-20261005T111833110772Z`.
The bootstrap and execution bundle remain unchanged. The original seed1337 Task
`10ad70300e824f4f907863084f274ced` was resumed with its uploaded script and all
parameters unchanged, after exact independent readback and fresh physical GPU
collision checks. It was observed queued on GPU8-3090. Seed2027 Task
`494d5870c60745149a595c21dc5cda99` was observed queued on GPU4-2080ti after the same
admission gates. Seed3407 had no idle disjoint executable worker; no Task was
created and its ETA remains unknown.

The three original dispatch receipts and their hashes are registered through
`v2v4real-numeric-dispatch-three-seed-index-20261005T112600546500Z.json`, SHA
`dc921a22a8fc9dfe2d3d57dccda3cb6c40a0eae3f3ddb2edf5a7b12c1dbdaf4c`.
Queue admission does not establish actual GPU execution, complete traces or
independent post-head numerical acceptance. Readback must use this exact frozen
checker only after each producer completes. Existing live candidates must never
be restarted or duplicated.

The single bounded live audit recorded six new K4 seed1337 sequence acceptances:
Top1/K4 totals are 167/276 (46/24/12/30/31/24). Nine CPU/control processes remain
alive and three recovery-off tasks remain in progress. These new sequence
receipts do not establish another complete cohort or paper acceptance.

### First numerical producer completed (2026-10-05 11:32 UTC)

Seed1337 Task `10ad70300e824f4f907863084f274ced` completed and registered a
1,384,888,320-byte numerical trace archive, SHA
`9a8f3833b9ba1dda4d8df21a06170cef6f7a583f079bf294c0222e124032d0f5`.
The frozen independent checker inventory was verified before starting its cloud
readback (PID17088, session9253). Readback is currently transferring bytes;
independent numerical acceptance is still pending. Preserve this active reader.
Seed2027 remained in progress. Seed3407 passed fresh fleet admission after the
first producer completed and queued Task `e77b04958fe64f6a8c708fc7dc627e4e` on
GPU8-3090; its dispatcher finished normally. No accepted raw production was
repeated. The bounded audit also found one new K4 seed1337 sequence acceptance,
bringing Top1/K4 to 168/276. No additional complete cohort was accepted.

### All three numerical producers completed (2026-10-05 11:48 UTC)

The single bounded audit observed seed2027 Task
`494d5870c60745149a595c21dc5cda99` and seed3407 Task
`e77b04958fe64f6a8c708fc7dc627e4e` completed with full 3,986-row capture progress
and registered trace/runtime/extension artifacts. Both now have first independent
readers using the unchanged qualified frozen checker, after its complete source
inventory SHA verification: PID24901/session71803 and PID24944/session12207.
Seed1337 reader PID17088 remains active and is still transferring its original
trace archive. Preserve all three readers. Producer completion does not establish
independent numerical acceptance, network forward correctness, 9D or tracking
metrics. Top1/K4 remains 168/276, nine forest CPU/control processes remain alive,
and recovery-off tasks remain in progress. No new GPU tasks were dispatched.

### Seed1337 independent post-head numerical acceptance (2026-10-05 12:18 UTC)

The original seed1337 reader completed normally (session9253, exit0). Its frozen
independent checker accepted all 3,986 rows and 235,801 candidates at unchanged
atol/rtol 1e-4. Receipt `independent-post-head-numeric-receipt.json` in
`v2v4real-numeric-seed1337-independent-readback-20261005T1134` has SHA
`52b8f0334342c8aea78191bd716de51f3dae995f17dad6af898e427e43a0c92c`.
All 29 downloaded cloud inputs were independently rechecked for size/SHA, along
with the trace index, accepted raw proof/index, bootstrap and frozen checker
inventory. The registered recheck receipt is
`v2v4real-numeric-seed1337-independent-acceptance-20261005T121818661180Z.json`,
SHA `26c8d514b7d725da50651dec43e5ee1aedab6e223a7f67b05be20293392952c8`.

This acceptance covers independent post-head mathematics and bound runtime,
source, frame, checkpoint and transport evidence. It does not verify the entire
network forward, admit a 9D state cache, accept tracking or complete paper metrics.
Do not restart the completed reader or repeat GPU capture. Seed2027 and seed3407
readers PID24901 and PID24944 remain active. The single bounded audit also found
three new K4 seed2027 sequence acceptances, bringing Top1/K4 to 172/276
(46/24/12/31/35/24), without another complete cohort. Nine original CPU/control
processes remain alive and three recovery-off tasks remain in progress.

### Three-seed independent post-head acceptance (2026-10-05 12:48 UTC)

Seed2027 and seed3407 original readers completed normally (sessions71803 and12207,
exit0). Each independently checked all 3,986 rows at unchanged atol/rtol 1e-4,
with 229,686 and 229,832 candidates. Their 17 and 29 cloud inputs respectively,
trace indexes, accepted raw proofs/indexes, bootstrap and frozen checker inventory
were rechecked for size/SHA and registered. No mathematical or GPU run was repeated.
The three-seed index is
`v2v4real-three-seed-independent-post-head-numeric-acceptance-index-20261005T124849854392Z.json`,
SHA `56705fa1a2891eba34d37d459f4bff18586f4f5556c547f62043ea59c61e6690`.
This completes the declared post-head numerical scope only. Entire network forward,
9D state cache, tracking and paper metrics remain unaccepted. Preserve all three
completed readers and producers. Top1/K4 remains172/276; nine original CPU/control
processes remain alive and three recovery-off tasks remain in progress.
