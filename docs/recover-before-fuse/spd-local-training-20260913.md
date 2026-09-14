# SPD 全 train 本地身份训练执行记录

2026-09-13 开始为真实身份训练准备本地输入。研究代码在 transvision 执行；本地阶段的训练输入、GT 和模型保存在本机私有临时目录，不写入论文项目或 OneDrive。用户随后要求改用 ClearML 至少四卡、跨机器并行训练，并排除 V100。新增部署范围与状态见 [ClearML 多机训练记录](clearml-ddp-training-20260913.md)，不与历史 19 份结果文件的发布混同。

## 路径与来源

本地工作根目录：`/private/tmp/spd-identity-fulltrain.jtGEnO`。该目录由 `mktemp -d` 创建，不是公开共享目录，也不是永久归档；后续保留策略需要在运行完成后确定。

远端只读来源为 `lbin@10.100.35.112`：

- 缓存：`/home/lbin/Desktop/transvision/work_dirs/recover-before-fuse/spd-train-v2-20260912/cache/`，约 2.2 GB。
- 转换监督：`/home/lbin/Desktop/eventtrack-cooptrack-runtime-20260911/conversion-runs/full-train-converted-attempt-2/` 下两端的 `spd_infos_temporal_train.pkl`，分别为 99,469,794 和 118,396,025 字节。
- 精确配对监督：`/home/lbin/Desktop/V2X-Seq-SPD-train-only/V2X-Seq-SPD/cooperative/`，约 128 MB。
- 原监督审计：`/home/lbin/Desktop/eventtrack-cooptrack-runtime-20260911/association-20260912/prepared-v2/supervision/target-audit.json`。

远端已核对的 SHA-256：

| 对象 | SHA-256 |
|---|---|
| V2 train manifest | `1137740ecdf2aca7372536998ac485585ca89f4792bf68e287fa75d2e07a6fa0` |
| 车端转换监督 | `d4b328e4e83d73b09d6961aec05747c13647ead92b576d6605da3a8f2d2577bb` |
| 路端转换监督 | `e0e78a0bd2ce8584353780fb5fae2f33ba1822e7a40b22ad9884b960d226213e` |
| target audit | `96443d50d74a49d98df6828eb550c3d2086ec0b6bbd625c2f1b8707aeadc2880` |

本地 manifest 与上述摘要相同，内容为 train、46 个序列、16,338 个源帧、`gt_in_cache=false`、`test_payloads_read=false`。2026-09-13 07:40（北京时间）缓存传输进程正常退出，传输了 32,677 个文件，共 2,212,385,164 字节；这是传输完成状态，不是模型训练完成。反序列化 PKL 前必须核对代码中固定的两个摘要。

## 执行顺序与状态

三种子本地开发训练已全部完成，总训练耗时 1,118.715 s，进程正常退出。三个检查点均已通过正式加载器复核：全部 30 轮各覆盖 90,651 行、1,440 个 batch，权重摘要一致，模型参数相对初始化确实变化，加载后全部子模块为 eval、全部参数禁止梯度。成功回执 SHA-256 为 `a8d4d9cfdf68a602f0e5dfca0f4c9a8030e1d87f9c894a01998e703702559b9d`。这些是 CPU 开发训练结果，不是四卡训练或跟踪验证结果。

两个 PKL 在反序列化前均通过固定 SHA-256 检查；cooperative 元数据以及全部 7,445 个标签文件通过审计清单的摘要、大小和路径覆盖检查。准备过程经过 `VerifiedForestCache` 的全载荷和覆盖检查，完整训练清单 SHA-256 为 `6356cf9420b41efc38c20838ff6a87eccf643d0a5361475ebfbd392b1a658391`。

完整监督预检已通过：

| 范围 | 源帧 | 序列 | car 标注 | 同帧 car track ID 重复的帧 |
|---|---:|---:|---:|---:|
| 车端 | 8,504 | 46 | 74,097 | 1 |
| 路端 | 7,834 | 46 | 98,542 | 144 |

基于全部 16,338 个源帧及 7,445 对 cooperative 连接，`AnnotationIdentityIndex` 成功构造。172,639 条 car 标注中，171,351 条状态为 `matched`，1,288 条为 `ambiguous`；未被标为歧义的身份连通分量共 3,169 个。同帧重复 ID 或类别冲突会使受影响的整个身份连接被标为歧义，不能挑选其中一个标注冒充确定监督。这些数量描述标注及索引，不是检测匹配准确率、最终训练行数或跟踪指标。

执行次序如下，四步均已完成。后续 ClearML 多机训练使用独立计划和输出目录，不覆盖本轮产物。

1. 用 `tools/event_track_v2x/prepare_forest_training.py` 读取完整 V2 缓存、两端转换监督、cooperative 标签及固定 target audit，在临时工作根目录下生成新的 `identity-rows-v1/`。
2. 核对全部 46 个序列、7,445 个 cooperative 输出时刻，以及实际 ingested/source-unavailable 和各类监督数量。缓存封装的 16,338 个源帧数不能冒充实际进入方法的观测数。
3. 使用新 manifest 的实际 SHA-256 运行 `tools/event_track_v2x/train_forest_identity.py`，输出到新的 `identity-fit-v1/`。采用现有默认配置：三个种子 1337、2027、3407，各 10 轮，batch 64、hidden 128、4 个 attention heads、dropout 0.1、AdamW learning rate 0.0003、weight decay 0.0001、gradient clip 5、geometry weight 1。
4. 每轮必须遍历全部有效监督行；保留排除原因、训练损失、失败记录和全部种子产物。不能用部分种子或中间权重填写完成回执。

本轮样本准备进程从 canonical transvision 目录启动，使用上述私有工作根目录下的 `cache/`、`converted/`、`projection/` 和 `target-audit.json`；输出到 `identity-rows-v1/`，标准输出及错误保存在 `preparation-v1.log`。环境设置 `PYTHONDONTWRITEBYTECODE=1`，OpenMP、OpenBLAS、Accelerate 线程上限均为 4。日志采用禁止覆盖模式；已有活跃进程时不能重新启动同一输出路径。

## 完整样本与时间可用性

最终样本含 166,079 个 car 观测，90,651 行有效监督，其中 90,303 行具有可区分的已知候选。全部监督类型如下；不从这些数量推导检测或跟踪精度。

| 类型 | 行数 |
|---|---:|
| `parent` | 90,233 |
| `birth` | 418 |
| `out_of_support` | 442 |
| `uncertain_birth` | 1,010 |
| `ambiguous` | 943 |
| `unmatched_prediction` | 73,033 |

「全 train」指完整 46 个序列和 7,445 个配对输出时刻；不表示 16,338 个封存源帧均进入模型。按既定 `reference + 100 ms` 配对快照，时间与日程覆盖分别为：

| 来源 | 封存源帧 | 配对日程内 | 实际接收 | 超过决策时刻 | 日程外 |
|---|---:|---:|---:|---:|---:|
| 车端 | 8,504 | 7,445 | 3,752 | 3,693 | 1,059 |
| 路端 | 7,834 | 7,445 | 7,445 | 0 | 389 |
| 合计 | 16,338 | 14,890 | 11,197 | 3,693 | 1,448 |

3,693 个被排除的配对源帧均因车端图像时间超过决策时刻，并非框参考时间超过决策时刻。全部 8,504 个封存车端帧的「图像时间减框参考时间」最小值／中位数／95 分位数／最大值为 58.186／98.396／128.0084／155.793 ms。这些是缓存记录的时间差，不是实测网络时延，也不证明时钟错误。

当前快照入口不将超时帧排队留到下一输出时刻，因此时间可用性是本开发运行的重要输入限制。不能用它单独证明「身份压缩导致协同下降」。若采用迟到帧队列或其他决策延迟，必须另立输入合同、重新准备匹配的训练数据，并在各后端使用相同输入；不能混用本轮权重和数字并声称条件未变。本轮不会在训练中改动冻结时间条件。

## 三种子训练执行

真实训练输出位于私有工作根目录的 `identity-fit-v1/`，日志为 `identity-fit-v1.log`。固定计划 SHA-256 为 `c797d183188df8644ccc085538bf8cf4e2f0ddbabc9c5a6e4d657c74fdc5689f`；计划记录 `full_official_train=true`、`local_fixture_only=false`、CPU、三个种子及上述超参数。数据、计划和源码检查通过后才发生优化器更新。

种子 1337 的 10 轮均覆盖 90,651 行、1,440 个 batch；第 1 轮局部训练损失为 0.09241237579857786，最后一轮为 0.06549035082108057。每轮约 36–38 s，是当前主机本轮训练日志值，不是部署推断尾延迟、同资源比较或真实身份指标。所有轮次记录 `validation_metrics_read=false`。

11:03 已通过正式检查点加载器复核种子 1337：权重与记录的摘要一致、模型参数相对初始化确实变化、全部子模块为 eval、全部参数禁止梯度，且固定计划摘要匹配。仅记录摘要，权重文件继续留在私有临时目录。

| 种子 1337 产物 | SHA-256 |
|---|---|
| `checkpoint.json` | `f0172c2f87887b4ec2b39c41d50e6a9610c7dd5aca88b263c3d6475109c5ced0` |
| `weights.pt` | `f121b2ff7a71901755294cbc2cd5e648e9c02089e91facd54ea092cd488ec4e5` |
| `epochs.jsonl` | `21abb3ffabbab3b3e7e5e8e6f526ebc03a45236775670f3614516b887684eec1` |
| 实际加载的模型参数 | `97eb4b867be7b1d87f05cda1673d9506262c36ec17a6375a0073e5b971db10be` |

其余两个种子的最终复核如下。所有轮次均记录 `validation_metrics_read=false`；当前没有新方法的跟踪增益结论。

| 种子 | 首轮损失 | 第 10 轮损失 | `checkpoint.json` SHA-256 |
|---|---:|---:|---|
| 2027 | 0.08704603881567632 | 0.06544865728531471 | `678c6227ed108ea180d0190d1b1262896c9adb9d46ebaf4ee92a78415c883d03` |
| 3407 | 0.08798817184015545 | 0.06560619926397504 | `a255b1e2a606c822da3c96b6ebe69d94318d799198f58632853df39d326e4045` |

本轮 CPU 训练时使用的关键代码 SHA-256如下。后续 DDP 接入提取了单进程入口中的公共预检函数，所以下表是历史训练来源，不代表当前工作树摘要。

| 文件 | SHA-256 |
|---|---|
| `transvision/models/event_track_v2x/forest_training_data.py` | `215a3413689819888cbeb49e3cbe1ed736c9c9a79a9742cfc9f94cff13024473` |
| `transvision/models/event_track_v2x/forest_supervision.py` | `8129b0b2dee374e726e7f7a00263e1b4e55cff8329d22081a46bae59d22590da` |
| `tools/event_track_v2x/prepare_forest_training.py` | `4725fb46d3354e7f71d5073f44a362949a38782e912c3ed3dc11687177b85983` |
| `tools/event_track_v2x/train_forest_identity.py` | `225ad273f571d802ad42d591ef1457fc167bc66ce1408ff00ecc703a21233ee0` |

样本准备和训练程序还会分别绑定完整的代码依赖摘要；以上四项不替代运行时的完整来源检查。

本轮使用本地 CPU，已检查 Torch 2.9.1 可用。启动环境将 OpenMP、OpenBLAS 和 Accelerate 的线程上限设为 4；不占用远端 GPU。线程设置不构成实时性或性能优势结论。

## 样本读取优化与回归

训练开始前，剖析发现 `TrainingShard.example()` 会为重叠上下文反复重建相同观测并校验协方差。现改为每个分片内按节点索引延迟构造、复用不可变的 `RawIdentityDetection`。分片数组和有效行索引采用不能重新打开写权限的字节缓冲区；数组映射只读，避免缓存与原数组不一致。

复用对象不含 GT 标签、神经嵌入、logits 或梯度，不改变样本顺序、候选、损失或优化器更新。缓存最多包含当前分片的全部节点，训练切换序列后释放；额外内存随单序列节点数增长，并不是零内存成本。没有增加跨序列、跨 epoch 的全数据常驻缓存。

在同一个四节点夹具上读取 8,000 行，`cProfile` 总计时间由 0.841 s 变为 0.088 s，观测构造次数由 16,000 次降为 4 次。这只是定位重复构造的本地剖析，不是完整训练吞吐、尾延迟、峰值内存或论文资源优势。

最终相关回归为 **57 项通过，耗时 2.42 s**，涵盖：

- 复用与每次重新构造的三个种子 CPU 训练，最终模型哈希逐项一致。
- 原始观测逐字段一致、重叠上下文共享对象、分片间不共享对象、数组写入及重新打开写权限均被拒绝。
- 批量／逐行 logits 和梯度一致、监督掩码、来源合同、冻结检查点及跟踪入口回归。

回执为 `work_dirs/recover-before-fuse/immutable-training-shard-20260913-v2.xml`，SHA-256 为 `9c34055dc1818e28206b34c29eb6d56a26cf901b0c37bb9b43ff0f645ff0ab6d`。首轮测试在写 XML 时被目录权限阻止，进程返回失败；没有将首轮报告为成功回执。随后在获批的目标目录写入权限下完成上述回归。

优化前 `forest_training_data.py` 的 SHA-256 为 `691a4e4591fa03b52208c73003731018e8faf07816b208a6d913c7299c803d5f`。优化完成后才启动真实样本准备；本轮准备计划已绑定优化后的源码。准备与训练期间不再修改绑定源码，也不复用旧准备产物。

## 研究边界

只构造 car 身份训练样本，不训练或报告 pedestrian。SPD official val/test 不用于这次训练或选择；V2V4Real 是独立的后续数据集任务。

当前冻结检测器与校准使用完整 train，因此本轮是 in-sample 上游输入上的固定末轮开发训练，**不是 OOF，也不是严格序列隔离的模型选择结果**。全 train 训练完成后仍需学习式计算分配、同资源强基线、严格来源隔离及真实验证。所有新回执继续保留 `paper_eligible=false`。

原先只读下载不扩大此前 19 份文件的 ClearML 发布范围。用户新增的多机训练要求已进入独立部署流程；获批的暂存传输仅包含最小研究源码、派生 car 训练分片和两个启动脚本，不包含 GT 全文、原始数据或完整预测流，也不修改现有远端仓库。
