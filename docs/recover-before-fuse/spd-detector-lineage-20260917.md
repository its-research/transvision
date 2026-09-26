# SPD 检测器来源与训练内划分核验

本轮从固定 ClearML 索引追踪缓存生产者，确认双端完整训练和真实权重字节；同时冻结独立开发所需的 train 内划分。不是正式缓存准入通过或论文实验完成。

## 已核实来源

- 索引任务 `14675561902041d7ab77fa2adbf2aeb0`，catalog SHA-256 `73bfa37d664ae0842368b5327be077627f06e1934530390b8a6571d77676f01d`。
- 双端检测器任务 `f3bae495b36646d0819396ba2ef64ad8` 实时状态 completed；两端均有 24 epoch 完成回执和四 A100 rank 运行 profile。
- 车端 5,112 iterations，路端 4,704 iterations；每卡 batch 10，总 batch 40。启动回执列出相同的 46 个官方 train 序列，初始化声明为 ImageNet R50，而非 SPD 训练权重；声明未加载官方 val/test。
- 车端最终权重 400,419,820 bytes，SHA-256 `7c329b7c0e14cee89c91a276453bef96c58ba90d05a242d9cb069cb8a643a36c`。
- 路端最终权重 400,420,268 bytes，SHA-256 `6ae503fc5f087e7192a4746cb2bcab13328cba99e96144058d66a2a599404700`。
- 两份权重均从 ClearML 回读完整字节核验，与旧 raw cache 任务 `a23f17b1cc7d4b6b8bd521587a0d65d4` 的生产者哈希一致。未反序列化执行权重内容。

先前四次失败和一次 stopped 的启动训练任务保留。不能只检查这些早期任务就断言没有完整训练；也不能因为后续任务 completed 就把所有下游旧缓存自动转正。

## 可复用的 ClearML 输入包

包任务 `9a7e7a9213954b57a35403f777e74561`，manifest SHA-256 `8d4c45019abae11fae1258c2d448ac744f5e316bc5e9348c62425e272484c39f`。

| Artifact | 字节数 | 用途 |
|---|---:|---|
| train-inputs | 4,070,034,766 | 官方 train 的原始输入投影和转换结果；需继续核验并生成隔离视图 |
| runtime | 2,504,425,555 | 冻结 CoopTrack 运行环境 |
| source | 551,492 | 官方源码及训练入口 |
| pretrained | 102,530,333 | ImageNet R50 初始化 |

本轮仅回读该包 manifest，没有下载和解压全部上述载荷。不能把包元数据核验等同于全部输入字节和来源已独立核验。旧启动回执中的历史 source waiver 未作为本轮新的授权。

## 新冻结的开发划分

采用现有 paper protocol 的 `rbf-train-holdout-v1:` SHA-256 排序规则，20% train 内 holdout，得到 37 个 fit、9 个 holdout。两侧相同，不使用标签或性能，不改变官方 train/val。此划分仅在已声明的 46 个 train ID 内生成；完整原始 split 还需在准备输入时核验。

holdout：`0002, 0004, 0018, 0041, 0048, 0056, 0078, 0080, 0081`。

fit：`0000, 0001, 0005, 0008, 0010, 0016, 0022, 0023, 0025, 0029, 0030, 0032, 0033, 0034, 0035, 0036, 0037, 0040, 0047, 0049, 0050, 0054, 0055, 0057, 0060, 0062, 0066, 0068, 0070, 0072, 0077, 0079, 0082, 0084, 0087, 0093, 0094`。

现有全 train 检测器已见这 9 个 holdout 序列，不能用于声称严格隔离的上游开发。下一项应准备 fit-only 数据视图，在 fit 上训练上游检测器及校准；holdout 用于开发选择。官方 val 继续标为探索性，不碰 test/test_A。

## 回执与软件检查

独立 ClearML 审计任务 `c6622188bea749dbbabcf3b968bc59d2`，来源哈希、划分和审计源码已发布；回执回读内容一致。脚本为 `work_dirs/rbf-pointpillar-runtime-20260917/tools/event_track_v2x/audit_spd_detector_provenance.py`。

5 项本地相关测试通过，验证划分稳定、完整、互斥及训练完成/权重/rank 绑定。首次下载因默认共享 cache 权限失败，改用任务专用 cache 后成功，未修改共享权限。发布执行曾遇自动权限审查超时，按工具指示重试一次后成功，没有重复任务。

旧缓存的非 OOF、legacy 协议、校准及来源边界保持不变。本次 `formal_cache_admission=false`、`paper_performance_verified=false`。未运行新的检测器训练，未提交 Git。
