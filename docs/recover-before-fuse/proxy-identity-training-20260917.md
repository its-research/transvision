# 真实候选的代理时间关联训练冒烟

Task `6f2073bc97414845a0c0a86e7aa04ef1`，GPU4-A100，初始提交 queued。独立协议 `nominal-10hz-proxy-matched-token-training-v1`；不是完整 Recover Before Fuse、正式时序训练、DDP 或性能评价。

## 固定配置

- 固定 train20 的 40 个源帧、2,560 原始候选及原生 256 维冻结特征。使用 nominal 100 ms 源间隔、模拟延迟 0、100 ms deadline，主协议不改。
- 现有 RecoverableIdentityModel 显式指定 feature_dim=256；默认仍为 203，原默认权重形状不变。没有把 256 维任意截断/填充为旧特征。
- hidden=32、heads=4、dropout=0，Adam lr=0.001，固定 3 epoch。
- 种子 1337、2027、3407 分别在前三张 A100 训练；第四张复跑 1337，检查同种子参数一致性。全部检查点保留，不择优。
- 当前帧两源特征分别为 left/right，上一帧 CAV0 特征为 history；GT 只用于离线选择及 loss，不作为 forward 特征。
- GT 为已有独立严格 Car 投影，ego CAV0；候选 world corners 变换回当前 ego。按 XY 凸包 BEV IoU 最大总和一对一分配，再过滤 IoU < 0.5。这不是 3D IoU，不称官方检测评价。
- 仅保留匹配 GT 的 token 进行训练；没有匹配的候选从本次 loss 中省略，逐源帧报告 omitted_candidates。不能称完整 raw-top64 训练，也不能用于证明误检下的关联性能。
- 使用既有 cross_assignment 与 temporal_assignment 两类损失；检查两头有效梯度、训练损失/梯度有限、检查点严格加载一致。没有 held-out 验证、阈值搜索或 checkpoint 选择。

## 固定资产与代码

候选 Task `e59d0f4d522b4ad8940d41cf3d462449`，SHA-256 `f75d569f593c87640621f0d91b9daa84fd16764267a27719a1e00815ca28914f`。

对齐 Task `01c9989cc5f24931a16a0605e6f164f9`，world corners SHA-256 `84210ca345f5f48a6ab78a925188908dad409cc3005ca600c1ee47f8d928b118`，pose-only SHA-256 `aabdb5fe0811546ddca5c009206a4e57e44909b143a6e16b53fe04180294f0e0`。

独立 train GT Task `a1e99f9f4eaf4208b32b7210fb58c57a`，SHA-256 `8de81690ad2d3b3b6a971a31a9ab0ae3e884abcfbf5f105a3eb3caf9acb1edb6`。全部由 ClearML 拉取并核验，不读取 test。物理身份正确性和公开 detector 选模来源仍未确认，paper_eligible=false。

隔离工作树入口 `tools/event_track_v2x/train_v2v4real_proxy_identity_smoke.py`，SHA-256 `9520d1d0e161a8325e8cc36ee48ea07d4472067d0c3e9274b2daaf59f922ebd9`。模型源码 SHA-256 `14674c5db70908e607f1bb75ea5df1d3661f0ba08b854ebf37364e350a2df170`。

远端目录 `/home/lbin/Desktop/rbf-identity-smoke-20260917`，执行 SDK Python 的 `tools/event_track_v2x/train_v2v4real_proxy_identity_smoke.py --submit transvision/models/event_track_v2x/learned_identity.py`；默认按源码及模型哈希去重，提交前检查四 A100 空闲。

## 验证与验收

管理机执行现有 learned_identity 回归及新增 feature_dim/匹配测试，18 passed。本地主仓源码未改；改动仅在隔离工作树，未提交 Git。

成功需任务 completed、三种子及第四卡重复训练记录、两类有效梯度、严格检查点回读，以及 ClearML training-smoke-report 和四份检查点资产。训练损失即使下降也不是泛化或论文跟踪收益证据。

## 依赖修复重试

首轮任务最终 failed：容器未安装 set_packages 中声明的 Shapely，标签匹配时报 ModuleNotFoundError；尚未训练。保留该任务。增加容器启动阶段固定安装 shapely==2.1.2（no-deps、binary only），并在 runner 开始检查版本，不修改宿主依赖。

修复后 18 项回归再次通过。新任务 `4abd3528a43e41bc82e19087d622925a` 已 queued；修复 runner SHA-256 `93a4c90ae2a024086151ec4bac419c0751925b9446b68f8886b1576b6aaa57fa`，模型源码、数据和训练配置均未改变。

## 完成验收

随后实时核验新任务 **completed**，实际 worker `10.100.34.18-A100:gpu4,5,6,7`。从 ClearML 回读 training-smoke-report 与四份检查点，各文件 SHA-256 均与报告一致。

40 源帧中共 331 个候选匹配监督标签、2,229 个未纳入本次训练，形成 19 个连续时序 batch。每个种子固定 3 epoch / 57 步，两类头的梯度范数均有限且大于零，检查点严格加载通过。

| 种子 | epoch 1 平均训练损失 | epoch 2 | epoch 3 |
|---|---:|---:|---:|
| 1337 | 2.986373 | 2.674158 | 2.522042 |
| 2027 | 2.981250 | 2.588442 | 2.416746 |
| 3407 | 2.932619 | 2.630897 | 2.413225 |

第四卡复跑 1337 的模型参数最大绝对差为 0，训练损失序列一致；第四卡不是独立统计种子。不同 checkpoint 文件序列化哈希可不同，参数比较与文件完整性分别核验。

以上仅为同一小段 train 上的训练链路验收，不是泛化、检测/跟踪指标或恢复收益。下一步才可考虑三份冻结权重在完整候选上的无 GT 推断接入；不能把 GT-matched token 子集当成完整候选协议结果。
