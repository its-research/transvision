# 固定权重的当前候选与历史上下文对照

ClearML Task `c4a1be329650484ba8f52fbe6257fa2e` 已 completed，实际 worker 为 `10.100.34.18-A100:gpu4,5,6,7`，报告和四份张量产物均已回读核验。

## 固定设计

复用训练任务 `4abd3528a43e41bc82e19087d622925a` 的种子 1337/2027/3407 checkpoint、原模型源码，以及训练监督报告；候选取自 `e59d0f4d522b4ad8940d41cf3d462449`。所有资产从 ClearML 获取并核对固定 SHA-256。使用原 train 短序列第 1–19 帧、delay=0、标称 100 ms 间隔与 deadline；不使用 official_test，不改变主协议。

四组条件为当前两端候选 subset/full × 上一帧 CAV0 历史 subset/full。subset 的选择及顺序严格复用原训练 label_audit，full 保留全部 64 个候选。每个种子 19×4=76 次重新前向，三个主种子共 228 次；第四张卡重复 1337，不作为第四个种子。

模型 eval、冻结梯度，不优化、不选 checkpoint、不调阈值。对共同身份正边记录 pair 与两端 unmatched 的差值、边增益、双向正目标 top-1 正确数；保存每组所有输出张量。验收包含模型参数不变、输出有限及第四卡重复最大差不超过 1e-5。

GT 用于诊断子集的选择，不能标为 GT-free 推断；数值特征仍来自冻结检测器。此项是 oracle 输入上下文诊断，`online_eligible=false`、`paper_eligible=false`，不能把子集表现用作可部署性能或跟踪结果。比较固定模型对候选/历史变化的响应，也不能单独证明训练失败的完整原因。

## 实现与验证

入口位于隔离工作树 `work_dirs/rbf-pointpillar-runtime-20260917/tools/event_track_v2x/run_v2v4real_context_ablation.py`。runner SHA-256：`8ffddb86f09c7223bfb872a26003cf5584c22d43659b86eec2f914ad5b9d2481`。

测试 `test_context_ablation.py` 和前项 `test_null_diagnostic.py` 合计 7 passed（0.08 s），覆盖原训练顺序、完整索引、非法子集及标签语义。工作树 diff 空白检查通过。未修改冻结模型源码，未提交 Git。

远端复现/幂等提交命令：

```bash
cd /home/lbin/Desktop/rbf-frozen-inference-20260917
CLEARML_CACHE_DIR=/home/lbin/Desktop/rbf-next-experiment-cache \
  /home/lbin/miniconda3/bin/python tools/event_track_v2x/run_v2v4real_context_ablation.py \
  --submit tools/event_track_v2x/run_v2v4real_proxy_identity_inference.py
```

提交前检查互不重叠的空闲四 A100，并在上传后再次核验。重复命令返回同源码已有任务，不重复排队。不修改其他 worker 或作业。

## 完成门槛

任务 completed、`context-ablation-report` 与四份 `context-potentials-*` 发布；回读报告与张量核验形状覆盖、哈希、权重检查和重复结果，才能报告实测结论。本次全部通过。

## 完成结果

三个主种子各 76 次前向、第四卡重复 76 次，合计 304 次；不把重复计为独立种子。每种子每组包含 40 条真实对应边、80 个双向正目标。十二个主条件均为正增益边 0/40、正目标 top-1 正确数 0/80。

| 种子 | 当前子集/历史子集：正边平均增益 | 当前完整/历史子集 | 当前子集/历史完整 | 当前完整/历史完整 |
|---|---:|---:|---:|---:|
| 1337 | -5.333313668 | -5.333313560 | -5.338551927 | -5.338551915 |
| 2027 | -3.847570360 | -3.847570324 | -3.868961155 | -3.868961048 |
| 3407 | -5.300488675 | -5.300488710 | -5.328166664 | -5.328166616 |

增益为 `2*pair-left_unmatched-right_unmatched`，为负表示该对应边不如两端各自未匹配。训练输入条件（当前子集/历史子集）也没有正对应边达到正增益。因此在本短序列与冻结权重下，失败不能只归因于从训练子集切换到完整候选。改变历史确实影响平均分数，但不足以使正边跨过零点。

源码核对：当前 token 分别编码，再以历史作为 attention 的 key/value；没有当前 token 间 self-attention。固定历史时，改变其他当前候选不应改变同一 token 的数学输出；表中近似相等的平均增益与此结构一致，微小差异不能解释关联失败。此处未据平均数推断每个张量严格相等。

参数前后逐项不变；种子 1337 第四卡重复所有输出最大绝对差为 0，回读后也逐张量相等。四份张量均严格加载、76 个记录完整，pair/temporal 形状与三路输入尺寸一致，数值全部有限。

报告 SHA-256：`f12042454818747b6f36ef815b1518352163593bf7c77f4d8fd458f0cbacb261`。

四份张量文件 SHA-256：

- 1337/device0：`731ded0ad4147f0b0b7c6dd08caa6f3622a574006e1ab31e0aa9eb77edb90f2d`
- 2027/device1：`be0439e3c03595e5b46c4c56fc7e2e7c92f89313877c4bfbe23ac3ae7cb7eebe`
- 3407/device2：`330995490a0ce7afb28db3543ac2a04722b6f6ce21f54aabbecda5863aa3197f`
- 1337/device3：`2d95df2fdf1598ca9f26c09c693c31f2d69bf927bf4977709f7270606a7a5534`

下一项应回到训练侧，用预先固定预算的训练内小样本过拟合对照检查正关联是否可学习，并分别记录正目标、null 目标与时序损失，区分训练不足和损失竞争。尚未执行该项，不据本结果认定某一根因，也不直接修改正式训练协议。
