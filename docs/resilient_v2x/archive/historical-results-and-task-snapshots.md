# 历史结果与任务快照

> 本文件保存从根 README 移出的历史跨协议复现、旧链与诊断快照；不作为当前论文证据。

<details>
<summary>历史跨协议复现、旧链与诊断快照（归档；不作为当前论文证据）</summary>

## 本仓历史复现结果

| Car | Latency | 3D AP@0.50 | 3D AP@0.70 | BEV AP@0.50 | BEV AP@0.70 | 结果标签 |
| :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| FFNet-B-V | 0 ms | 51.60 | 29.99 | 56.62 | 49.15 | 官方代码库结果（TransVision benchmark） |
| FFNet-B-F | 0 ms | 55.48 | 31.54 | 63.15 | 54.27 | 官方代码库结果（TransVision benchmark） |
| FFNet-B-F（本次复现） | 0 ms | 65.16 | 39.60 | 71.27 | **62.44** | 本仓 ClearML 复现 |
| FFNet | 0 ms | 55.81 | 30.23 | **63.54** | 54.16 | 论文原始结果（FFNet Table 2） |
| FFNet | 200 ms | 55.37 | 31.66 | 63.20 | **54.69** | 论文原始结果（FFNet Table 2） |
| FFNet (w/o pred) | 200 ms | 50.27 | 27.57 | 57.93 | 48.16 | 论文原始结果（FFNet Table 2） |
| TF-L-V | 0 ms | 56.40 | 34.69 | 62.08 | 52.48 | 官方代码库结果（TransVision benchmark） |
| TF-L-F | 0 ms | **58.46** | **37.28** | 62.73 | 54.21 | 官方代码库结果（TransVision benchmark） |
| CoFormerNet | sync | 55.34 | 35.95 | 60.65 | 51.26 | 本仓 ClearML 复现 |

`conf=0.2`

FFNet-B-F 本次复现：ClearML [`9859bc7fbb694ca1b26f2a641711b4d2`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/9859bc7fbb694ca1b26f2a641711b4d2/output/log)，official 3-class complemented，40 epoch，final val（1789 samples）。

CoFormerNet 复现结果：ClearML [`2ea50800c8ff4bb3b0f87b4058d16ccc`](http://10.100.34.118:8080/projects/8fb6dbc7a09a4163961d4992f218ee26/experiments/2ea50800c8ff4bb3b0f87b4058d16ccc/output/log)（fusion formal eval），DAIR-V2X-C `vic-sync`，LiDAR-only，1789 个验证样本，评测范围 `[0,-46.08,-3,92.16,46.08,1]`。veh-only formal：3D@0.5/0.7 = 55.55/36.27，BEV@0.5/0.7 = 60.89/51.39。

- FF-B-V：FFNet Basemodel veh-only（re-implementation）
- FF-B-F：FFNet Basemodel fusion（re-implementation）
- TF-L-V：TransFusion-L veh-only

## 历史 ResilientV2X Results

受控证据：ClearML [`2992081bc95949f198e062c136810736`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/2992081bc95949f198e062c136810736/output/log)，seed `20250218`，student [`77afadda645f44748e1236eb91b5e664`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/77afadda645f44748e1236eb91b5e664/output/log)，dataset `fc242933c3ac43c2b47aaa3bd7f4a920`，1337 个验证样本，11330 个 Car ground truth。

受控协议：DAIR-V2X-C、v2 manifest、官方 validation split、Car-only、LiDAR + Camera、评测范围 `[0,-40,-3,80,40,1]`。L-Fail 和 C-Fail 使用 `E+R`。AP 与 PDR 单位均为 `%`。

历史受控基线证据：V2X-ViT-style [`e4724cc06cfb4ebba12b62e24c096b85`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/e4724cc06cfb4ebba12b62e24c096b85/output/log)、CoBEVT-style [`84504e62d4014067bf22f9c0b75844a9`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/84504e62d4014067bf22f9c0b75844a9/output/log)、CoFormerNet-style [`bc0f0901f0c3413fac7452b888c7e528`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/bc0f0901f0c3413fac7452b888c7e528/output/log)、MIT-HAN BEVFusion-style [`4311cf87922148a69541931ac64955c9`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/4311cf87922148a69541931ac64955c9/output/log)、FFNet-style [`49e32c1bac4b4bc4a0c2f014338bdb88`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/49e32c1bac4b4bc4a0c2f014338bdb88/output/log)；均完成历史 12 条件并保存 metrics、predictions、plan 与 run contract，不作为本轮最终同条件结果。

### DAIR-CAUSAL-1337-v1 统一评测

固定证据：1337 个有因果历史的 validation 样本、11330 个 Car ground truth、`0/100/200/300 ms × Full/L-Fail/C-Fail` 共 12 条件、每条件 `unsupported_sample_count=0`。Manifest content SHA-256 为 `715ac6f7a14225e20327eed0650c55abdc0cb98431830164e84545238099645d`，overlay-index content SHA-256 为 `77bd4585dbb02901f862b8da6aa208a504674b824a3d55cf15005aacbeeeaaff`，ordered sample-ID SHA-256 为 `a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a`。

统一训练条件：4 GPU、每卡 batch size `2`、global batch size `8`、FP32、50 epoch、每 10 epoch 验证、seed `20250218`。

本轮正式训练同时均匀覆盖上述 12 个条件；所有方法只从同一 clean teacher 加载共享的 LiDAR encoder、camera encoder、detection projection 和 detection head，方法特有 fusion 均重新初始化。

本轮正式任务（上海时间 `2026-08-12 07:14`）：

| 项目 | ClearML 身份 | 状态 | 结果 |
| :--- | :--- | :--- | :---: |
| 12 条件训练 dataset | [`7c59fabb`](http://10.100.34.118:8080/datasets/simple/481fbefdd5ef4be79c9a461ee92ca12b/experiments/7c59fabb9da949e6b3c94c732f000975) | Final；25702 files；`train_condition_matrix`；12 条件；epoch 0–49；seed `20250218`；远端文件 SHA 全匹配 | index content `60fe994270af82d9cb1c1e6a709c3766f407205ba3cffb5cc8e21ffad2957d89`；training overlay file `16d7aeb7ec05fa3560594f910d35c8faceb06487c13f5bdc3a4f451c547ca194`；evaluation overlay 为 1337 样本、3 状态 × 4 延迟 |
| 旧 Sealed source（21 个已完成方法） | [`4f7fac00`](http://10.100.34.118:8080/datasets/simple/dbcf73898c9247faacabb921d26259a1/experiments/4f7fac0078a4419a907fec6ff9e306c8) | Final；source role `21` | tree `5c984ad49b5232d7f6d053fb641895283477efcbf2de40b36d9b3f3c6f8e28b6`；archive `655f33d9b684c26ff4ff577d5c5f3175d4fc4eac40426eb55820e7a8aa8bc40d` |
| 新 Sealed source（后续 5 个方法） | `351feedbbe81481fa31f1e9ae11a3f4e` | Final；`disconet/late_fusion/how2comm` 已完成；`where2comm/resilient_v2x` 运行中 | tree `ad511d88b731cb45ef2defb873712bdb2a325c648634b66c349fe1c1459510e4`；archive `b94a01c2acf2cc456fe9729f7c40e990e6d44b65e789c6fed11989a673f4f6da` |
| SOTA 候选 Sealed source | [`c9ca3075`](http://10.100.34.118:8080/datasets/simple/ad352e2b33fe412588356917953f5781/experiments/c9ca3075434e44e29951e4aca36e9046) | Final；仅新增 E1/E2/E3 三个配置；修改 0；删除 0 | tree `8a22d6d600a52117d01cfde5fa5f20fa1de269ec9c9ba040228f5e8e3a5e8767`；archive `a249546f16f236d42c8346a33b137dea27c037bf0df73c643da4132c8e589557` |
| SOTA Round-2 P0 Sealed source | [`ca2ef9dd`](http://10.100.34.118:8080/datasets/simple/487ffc19c67b4f5aa1b9cdfe634c0fa0/experiments/ca2ef9dd8a984df6b05693fb02e89f34) | Final；相对 E1/E2/E3 source 仅新增三因素 P0 配置；修改 0；删除 0 | tree `bff6f84c0e4989d07c4061b8b0e302a5705accd6688d3a175ed9e3e24b5d259d`；archive `8a33214c68b956730da1ab6664a008a50f8a6163cd20f299ac21f2b9158536c3` |
| SOTA Round-2 P2 Sealed source | [`85823804`](http://10.100.34.118:8080/datasets/simple/b93050bdafc348709e0b252bf855b874/experiments/858238049cad4d13918384bb8faec630) | Final；以 P0 source 为父；仅新增 student bbox loss 2.5 配置；修改 0；删除 0 | tree `25d1a9a1b67f525a02220c34739855a83d39706cc78e9f2bbaf882b47c438331`；archive `54e0cf371a6c8b68b041b84f8568efa72c432f0d38c9dfce89f005b2605ee1b3` |
| Shared-only clean teacher | [`487dab26`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/487dab2664a8485fa0cc7c4e2a0c3df8/output/log) | `GPU4-A100:gpu0-3` 完成；选择 epoch 30；model `d962f6ba`；checkpoint SHA `7516eb82…` | BEV AP@0.70 `65.9669`；3D AP@0.70 `38.0034` |
| Clean-teacher quality gate | [`f041d43e`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/f041d43e48c14ba4a4562281860d13f6/output/log) | 完成并通过；门槛为 BEV AP@0.70 ≥ `59.6257`、3D AP@0.70 ≥ `30.0` | `passed=true` |
| 26 方法训练旧 controller | [`6525107e`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/6525107e60ae4104a2800731d74ecd4e/output/log) | `failed`；进度 artifact revision `14`；后续任务转入不可变恢复链 | [`support_residual` `95e72da2`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/95e72da24d464ab08d117dedabd6652e/output/log) 已完成；epoch-50 clean-1789 BEV/3D AP@0.70 `62.1724/39.1890`；2 个 OutputModel |
| 26 方法训练旧恢复 controller | [`f8c36e50`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/f8c36e508c7d453dadc766207a5b25b2/output/log) | `failed`；progress revision `79` | 已由 schema-v4 恢复链接管 |
| 26 方法训练 schema-v4 controller | [`1011e98e`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/1011e98e10f64c428880af1d4b1d542b/output/log) | `in_progress`；progress artifact revision `37`；26 个确定任务实时状态为 25 项完成、1 项运行 | recovery schema `4`；source roles `21`；target adoptions `4`；rerun `[]` |
| 训练来源等价证明（旧任务） | [`5d379051`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/5d3790517a554d7196b6fd73ae99f17d/output/log) | `failed` | 已归档；不作为本轮正式证据 |
| 正式训练来源等价证明 P | [`7274a1a5`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/7274a1a5344a44c18aa7bcc6d1cb2e95/output/log) | `in_progress` | schema-v4 readback；无错误日志；正式 1337 依赖链首节点 |
| 全部 26 个正式方法 1337×12 planner/watcher W | [`7734387d`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/7734387ddfb74b11ba6d84f3fea0bb97/output/log) | `in_progress` | 依赖 P；无错误日志；26 个评测任务待创建 |
| 26 方法 × 12 条件正式 leaderboard L | [`f502bdd3`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/f502bdd329ad4ef4b4b6cf5c5f52aba0/output/log) | `in_progress` | 依赖 W；无错误日志；正式结果待汇总 |
| 26 方法 × 12 条件独立可比性审计 A | [`e19bab92`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/e19bab92884248f4ac07167e7eb66170/output/log) | `in_progress` | 依赖 L；无错误日志；正式审计结果待生成 |
| 单种子正式候选选择器 S | [`b8876e88`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/b8876e88bb494985a900e3d627439776/output/log) | `in_progress` | 依赖 A；无错误日志；正式候选与三种子结论待生成 |
| 被替代的正式 P/W/L/A/S 链 | `f1411180/45940984/a386557e/8a5e29fe/01cdc028` | `stopped` | 已由上述 schema-v4 不可变链替代；不作为正式证据 |
| 正式训练已完成（25/26） | schema-v4 恢复链 | 50 epoch；val/10 epoch；clean-1789 final | 见下表 |
| 正式训练运行中（1/26） | [`resilient_v2x` `54d28bc5`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/54d28bc513794051810fd383140ae96e/output/log) | `in_progress`；5090 gpu0–3；E45 | E40 `59.8596/38.0468`；clean-best E30 `62.0951/38.3299`；ETA `07:42`（clean-1789 BEV/3D AP@0.70；上海时间） |
| SOTA 候选 fastlane | E1 [`f0c3082f`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/f0c3082f3aa34a81805903e0ffdc8610/output/log)、E2 [`969c8fce`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/969c8fce6d24446299561772b3955274/output/log)、E3 [`dc037315`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/dc037315c0684c3d854a2fd7c19a2a2f/output/log) | E1 `5090 gpu4–7` 运行中；E2 `V100 gpu4–7` 运行中；E3 等待 E1 完成 | E1 E33、E30 `62.1005/37.7739`、ETA `08:37`；E2 E14、E10 `61.2868/33.7779`、ETA `14:30`；E3 预计 `08:37` 后接续并约 `12:50` 完成（上海时间） |
| SOTA Round-2 P0（三因素全集） | source [`ca2ef9dd`](http://10.100.34.118:8080/datasets/simple/487ffc19c67b4f5aa1b9cdfe634c0fa0/experiments/ca2ef9dd8a984df6b05693fb02e89f34)、template [`13f87c0f`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/13f87c0fd45d4621b3c93c0ff88702d8/output/log)、task [`8883c51c`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/8883c51ced4f4951a45edbaefe6342d4/output/log) | `in_progress`；A100 gpu4–7；4 GPU；global batch 8；FP32；50 epoch；val/10；seed `20250218` | E11；E10 `61.2109/33.2758`；ETA `13:14`（上海时间） |
| SOTA Round-2 P2（P0 + student bbox loss 2.5） | source [`85823804`](http://10.100.34.118:8080/datasets/simple/b93050bdafc348709e0b252bf855b874/experiments/858238049cad4d13918384bb8faec630)、template [`ac35b8f7`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/ac35b8f7403c41e1a8c50137abded09b/output/log)、task [`f5d3820b`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/f5d3820b4cdf416183c8f1fee566abe3/output/log) | `in_progress`；A100 gpu0–3；4 GPU；global batch 8；FP32；50 epoch；val/10；seed `20250218` | E4；首个验证待生成；ETA `14:15`（上海时间） |
| SOTA 候选正式 controller | [`ab1add67`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/ab1add678c1a4e518bebb9726ccb7040/output/log) | `in_progress`；等待正式训练 controller `1011e98e` 完成 | E1/E2/E3 正式门控结果待生成 |
| 已作废训练诊断 | `ptf_none` [`8cfe11af`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/8cfe11afa70c488f984522381117bbf3/output/log)、`ptf_linear` [`9664ab21`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/9664ab21140b4adcb71fa0eff2b75418/output/log)、`router_static` [`016c4dbd`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/016c4dbde2c642f7b6a65fc527a09f27/output/log) | `stopped/stopped/failed`；均无可用 OutputModel | `router_static` epoch-50 clean-1789 BEV/3D AP@0.70 `60.3004/38.8487`（仅诊断，不进入正式比较） |
| 全部 26 个正式方法 1337×12 planner/watcher（旧任务） | [`0d38b314`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/0d38b3148b5e48e489ecd68c5fb647cc/output/log) | `failed` | 已归档；不作为本轮正式证据 |
| 26 方法 × 12 条件正式 leaderboard（旧任务） | [`54bd98ce`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/54bd98ce684448038ac78d07af625bcb/output/log) | `failed` | 已归档；不作为本轮正式证据 |
| 26 方法 × 12 条件独立可比性审计（旧任务） | [`7d6086e4`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/7d6086e473f0402fab7e7cccab5c5cf0/output/log) | `failed` | 已归档；不作为本轮正式证据 |
| 单种子正式候选选择器（旧任务） | [`3fc44d59`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/3fc44d59dfd0422797f2e6ced2359700/output/log) | `failed` | 已归档；不作为本轮正式证据 |
| Portable Source-D v2 证据 | [`39387848`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/3938784889d74c1f861db2b20f10ec5b/output/log) | 完成；4 个 sealed artifacts；standalone `d5b759f3…`；Source-D `e7a9ab0f…`；equivalence `1156fe53…`；独立终审 P0/P1/P2=`0/0/0` | 完成；旧 [`9b1739d8`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/9b1739d8c9ba4ddb8ce9f5f50294e0be/output/log) 已停止且 0 artifacts |
| 模型本地归档 | `artifacts/trained_models/completed-live/` | 25 个已完成方法的 final 与 clean-val best 均已归档并通过 manifest 校验；clean teacher 单独归档；Where2comm manifest SHA256 `b7b5c58d…` | 50 个方法 checkpoint 文件；clean teacher 不计入 50 个 |

本轮已完成方法 clean-1789 结果（BEV/3D AP@0.70，epoch-50 final）：

| 方法 | 训练任务 | BEV AP@0.70 | 3D AP@0.70 | 备注 |
| :--- | :---: | ---: | ---: | :--- |
| `support_residual` | [`95e72da2`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/95e72da24d464ab08d117dedabd6652e/output/log) | 62.1724 | 39.1890 | completed |
| `ptf_none` | [`05519cde`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/05519cde286c4abbad7d6dd258242b0f/output/log) | 59.6622 | 37.5916 | completed |
| `ptf_linear` | [`0ca63093`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/0ca6309321a84dfbadd217a45d6dfdfd/output/log) | 62.1307 | 38.7716 | completed |
| `router_static` | [`18f3e5aa`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/18f3e5aa083a4f4a9576bc7d252ef62b/output/log) | 60.1427 | 38.7059 | completed |
| `no_distillation` | [`efe6522d`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/efe6522d87a44c55b1de7f9c144e5393/output/log) | 56.9729 | 34.2720 | completed |
| `coformernet` | [`8b77a367`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/8b77a3674dfe405388aae39ef82d06ef/output/log) | 58.2811 | 34.2021 | completed |
| `router_uniform` | [`ec13af4b`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/ec13af4bb48740b8aa836ea412c2930f/output/log) | 62.0288 | 39.1491 | completed |
| `no_reliability` | [`6313638e`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/6313638e35964ea98f54394a43d74b31/output/log) | 62.3015 | 38.9609 | completed |
| `no_delay_metadata` | [`135055f2`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/135055f2786944c0922494877f32efb3/output/log) | 60.3097 | 38.9779 | completed |
| `concat_capacity_matched` | [`175e6087`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/175e6087f66646e282a93439fecf292a/output/log) | 59.8929 | 37.7758 | completed |
| `ffnet` | [`5d75349b`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/5d75349b10fc4ee49b8c8375ff77cba5/output/log) | 59.0364 | 36.5589 | completed |
| `bevfusion` | [`48cf33da`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/48cf33da96114669b56a7646582c2835/output/log) | 59.0766 | 36.3993 | completed |
| `v2x_vit` | [`1c44caea`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/1c44caeac9064ac18c08cab83e00ae81/output/log) | 58.3858 | 33.7307 | completed |
| `cobevt` | [`39ba9fa0`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/39ba9fa05ed74391bab65470c65029fc/output/log) | 58.8134 | 33.9320 | completed |
| `linear_no_distillation` | [`63fbc743`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/63fbc743ec934ebda56bb75d2c05f96d/output/log) | 57.3437 | 34.5039 | completed |
| `no_distillation_peak_lr_3e4` | [`10b04170`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/10b041700f1f4b56873011784605152a/output/log) | 57.6066 | 35.8030 | completed |
| `ego_only` | [`60bb6bd3`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/60bb6bd310874f1f9b2737ecd4e07611/output/log) | 57.1068 | 33.7356 | completed |
| `fcooper` | [`21368e82`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/21368e8260cc4e5392fe2dbdf116e36f/output/log) | 50.8394 | 24.6312 | completed |
| `attfuse` | [`d549a22b`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/d549a22b211d46dbbe8f7ed7785b27e0/output/log) | 57.1470 | 31.5802 | completed |
| `when2com` | [`8e2c9b24`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/8e2c9b24f8d94249ab428086f98b9393/output/log) | 56.6432 | 30.7121 | completed |
| `where2comm` | [`65e3e641`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/65e3e6418e7748f38ac7d391af0dbf78/output/log) | 57.2396 | 33.4195 | completed；final epoch 50 |
| `v2vnet` | [`fc565520`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/fc565520239544119ef3dff83530ef98/output/log) | 60.0321 | 39.0477 | clean-best epoch 30 `61.1763/37.3791` |
| `disconet` | [`322c0855`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/322c0855a92145fcae2d87244ede50aa/output/log) | 57.9811 | 32.3151 | completed；final epoch 50 |
| `late_fusion` | [`84d46946`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/84d46946945d42b3ad79f3dd6df166a1/output/log) | 57.0239 | 31.2524 | completed；final epoch 50 |
| `how2comm` | [`c494c3b2`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/c494c3b27c344cecae273ebfdf875eed/output/log) | 56.5380 | 30.3284 | completed；final epoch 50 |

本轮 14 个论文基线正式结果：全部使用上述 1337 样本、12 条件、global batch size `8`、FP32、50 epoch、val/10 epoch 和同一 clean teacher 合同；正式 1337×12 评测链已部署，结果待完成。

| 方法 | 实现标记 | 训练任务 / 状态 | clean-1789 BEV/3D AP@0.70 | 1337×12 评测任务 | Full 0 ms BEV AP@0.7 | 12 条件均值 | 12 条件最差值 |
| :--- | :---: | :--- | :---: | :---: | ---: | ---: | ---: |
| CoFormerNet | controlled baseline | [`8b77a367`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/8b77a3674dfe405388aae39ef82d06ef/output/log)；completed | `58.2811/34.2021` | —（待创建） | —（待测） | —（待测） | —（待测） |
| FFNet | controlled baseline | [`5d75349b`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/5d75349b10fc4ee49b8c8375ff77cba5/output/log)；completed | `59.0364/36.5589` | —（待创建） | —（待测） | —（待测） | —（待测） |
| BEVFusion | controlled baseline | [`48cf33da`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/48cf33da96114669b56a7646582c2835/output/log)；completed | `59.0766/36.3993` | —（待创建） | —（待测） | —（待测） | —（待测） |
| V2X-ViT | controlled baseline | [`1c44caea`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/1c44caeac9064ac18c08cab83e00ae81/output/log)；completed | `58.3858/33.7307` | —（待创建） | —（待测） | —（待测） | —（待测） |
| CoBEVT | controlled baseline | [`39ba9fa0`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/39ba9fa05ed74391bab65470c65029fc/output/log)；completed | `58.8134/33.9320` | —（待创建） | —（待测） | —（待测） | —（待测） |
| Ego-only | controlled baseline | [`60bb6bd3`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/60bb6bd310874f1f9b2737ecd4e07611/output/log)；completed | `57.1068/33.7356` | —（待创建） | —（待测） | —（待测） | —（待测） |
| F-Cooper | controlled baseline | [`21368e82`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/21368e8260cc4e5392fe2dbdf116e36f/output/log)；completed | `50.8394/24.6312` | —（待创建） | —（待测） | —（待测） | —（待测） |
| AttFuse | controlled baseline | [`d549a22b`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/d549a22b211d46dbbe8f7ed7785b27e0/output/log)；completed | `57.1470/31.5802` | —（待创建） | —（待测） | —（待测） | —（待测） |
| V2VNet | controlled baseline | [`fc565520`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/fc565520239544119ef3dff83530ef98/output/log)；completed | final `60.0321/39.0477`；best E30 `61.1763/37.3791` | —（待创建） | —（待测） | —（待测） | —（待测） |
| When2com | controlled baseline | [`8e2c9b24`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/8e2c9b24f8d94249ab428086f98b9393/output/log)；completed | `56.6432/30.7121` | —（待创建） | —（待测） | —（待测） | —（待测） |
| Where2comm | controlled baseline | [`65e3e641`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/65e3e6418e7748f38ac7d391af0dbf78/output/log)；completed | final epoch 50 `57.2396/33.4195` | —（待创建） | —（待测） | —（待测） | —（待测） |
| Late Fusion | controlled baseline | [`84d46946`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/84d46946945d42b3ad79f3dd6df166a1/output/log)；completed | final epoch 50 `57.0239/31.2524` | —（待创建） | —（待测） | —（待测） | —（待测） |
| DiscoNet | controlled baseline | [`322c0855`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/322c0855a92145fcae2d87244ede50aa/output/log)；completed | final epoch 50 `57.9811/32.3151` | —（待创建） | —（待测） | —（待测） | —（待测） |
| How2comm | controlled baseline | [`c494c3b2`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/c494c3b27c344cecae273ebfdf875eed/output/log)；completed | final epoch 50 `56.5380/30.3284` | —（待创建） | —（待测） | —（待测） | —（待测） |

被替代的 source `bf92ca5d`、`29d8c0f3`，失败链 `05dc7aba/2a7ed8ab/8c230db5`，以及未入队链 `90522bc4/cff2f725/c86285ea` 均已归档，不作为正式证据。质量门控缺失的旧链 `7d7fc20d/1f844e01/5f034d33` 已停止并归档，不作为正式证据。上一版 source `b5bf05bdca8a` 及更早 source 的 4 个 watcher、22 个未启动任务，以及旧诊断训练/评测 `0c0152ba/efd5eead`、`cdea3b68/b813fd31`、`d14539b0/01d6ba90` 均已停止并归档；不进入本轮主表。旧 global-batch-4 任务已停止，不用于最终比较；清单见 [`docs/resilient_v2x/archive/global-batch-4-superseded.md`](docs/resilient_v2x/archive/global-batch-4-superseded.md)，旧稿无证据数值见 [`docs/resilient_v2x/archive/old-draft-results.md`](docs/resilient_v2x/archive/old-draft-results.md)。

上一版主表任务归档快照（上海时间 `2026-08-10 04:45`；非本轮正式结果）：

| 方法 | 训练任务 | 新版 1337×12 评测任务 | 状态 | 新版结果 |
| :--- | :--- | :--- | :--- | :---: |
| Resilient V2X | [d7fd0d84](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/d7fd0d8488b64a3aa0f5dea4c9ea6f06/output/log) | [80066fc1](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/80066fc18bde47509c9dd936017d1f5c/output/log) | global-batch-8 重训排队；评测依赖已创建 | —（待测） |
| Ego-only L+C（controlled baseline） | [32745fbc](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/32745fbc19cc446e8b31e50d6d1fbc2d/output/log) | [7d134889](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/7d13488908b0475ea064348151fc0b21/output/log) | 50 epoch 训练完成；统一评测排队 | —（待测） |
| Late Fusion-style L+C（controlled adaptation） | [3c1afb87](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/3c1afb871e4448899640c466b77002ad/output/log) | [90a4b952](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/90a4b952fee64496990b98ca0edd7b10/output/log) | 训练排队；评测依赖已创建 | —（待测） |
| F-Cooper-style L+C（controlled adaptation） | [ae71123f](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/ae71123f82b745e389dc6e81d95bf613/output/log) | [45b91062](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/45b9106240a34439a77e1b4916274184/output/log) | 50 epoch 与 final-checkpoint contract 完成；统一评测排队 | —（待测） |
| AttFuse-style L+C（controlled adaptation） | [dc59e174](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/dc59e174649e44fe84c2208003949707/output/log) | [024e5041](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/024e5041f76048c485067d948ef2058c/output/log) | 训练中；评测依赖已创建 | —（待测） |
| V2VNet-style L+C（controlled adaptation） | [d7628d45](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/d7628d45c91e479aaa10059be5c43946/output/log) | [72a4ba7c](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/72a4ba7c78934e778cd4451af0b7e0d2/output/log) | 训练排队；评测依赖已创建 | —（待测） |
| DiscoNet-style L+C（controlled adaptation） | [446bcece](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/446bcece88a443fdbedf2b93adead235/output/log) | [532c7c97](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/532c7c977ae0473db10b34c0a5e14d31/output/log) | 训练排队；评测依赖已创建 | —（待测） |
| When2com-style L+C（controlled adaptation） | [cdea3b68](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/cdea3b6830384aafbc98b88ae8a68744/output/log) | [b813fd31](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/b813fd31b83649c18b4503bcf4a2dc1d/output/log) | 训练中；评测依赖已创建 | —（待测） |
| Where2comm-style L+C（controlled adaptation） | [056a4a45](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/056a4a45f4284131a1f89418b8887d35/output/log) | [c63d06b0](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/c63d06b07f7e4e00a7f84e5897cc8eff/output/log) | 训练排队；评测依赖已创建 | —（待测） |
| How2comm-style L+C（controlled adaptation） | [ca05e4ab](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/ca05e4ab93f94799a9321a09d71ffe71/output/log) | [99dde8df](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/99dde8df64f04acd848a53c098fb57e5/output/log) | 训练排队；评测依赖已创建 | —（待测） |
| V2X-ViT-style L+C（controlled adaptation） | [a0d376ed](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/a0d376ed58ac4ea6a696c2f43f38ba93/output/log) | [f532345f](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/f532345fcf0c436b87373b3e8a0aaf88/output/log) | global-batch-8 重训排队；评测依赖已创建 | —（待测） |
| CoBEVT-style L+C（controlled adaptation） | [ce80972f](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/ce80972f63cc47f39992558b1db8c10f/output/log) | [b9837e05](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/b9837e05c59244bd988fba33037cd57e/output/log) | global-batch-8 重训排队；评测依赖已创建 | —（待测） |
| CoFormerNet-style L+C（controlled adaptation） | [4861fb56](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/4861fb5617ca4a7787644929d6fb116f/output/log) | [070d2536](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/070d2536f10448369193a9a1d9c5d2ce/output/log) | global-batch-8 重训排队；评测依赖已创建 | —（待测） |
| MIT-HAN BEVFusion-style L+C（controlled adaptation） | [05072944](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/0507294403a04f609ed191b5fa02f68e/output/log) | [96c3b510](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/96c3b510a7c14444941d8cd0ef7bbbe0/output/log) | global-batch-8 重训排队；评测依赖已创建 | —（待测） |
| FFNet-style L+C（controlled adaptation） | [d14539b0](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/d14539b0ae47416d80f5ac5a260baa01/output/log) | [01d6ba90](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/01d6ba9015244da98b66eacef578d5ff/output/log) | global-batch-8 重训排队；评测依赖已创建 | —（待测） |

上一版消融与改进任务归档快照（上海时间 `2026-08-10 04:45`；非本轮正式结果）：

| 变体 | 训练任务 | 新版 1337×12 评测任务 | 状态 | 新版结果 |
| :--- | :--- | :--- | :--- | :---: |
| No PTF | [494f4c7f](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/494f4c7f26c244558ec95fec722add7f/output/log) | [9aaae560](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/9aaae56030b845dda35345e0bbb77fc9/output/log) | global-batch-8 重训排队；评测依赖已创建 | —（待测） |
| Linear PTF | [83fd1c2a](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/83fd1c2a99b741cda32255dd563d5f65/output/log) | [5222fb74](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/5222fb74f7ed4553a78b4aa5723d32ea/output/log) | global-batch-8 重训排队；评测依赖已创建 | —（待测） |
| Static three-expert | [32500ba2](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/32500ba25a1a4333947f96a17261bc16/output/log) | [72976ba9](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/72976ba977134d1a82340ea508476c27/output/log) | global-batch-8 重训排队；评测依赖已创建 | —（待测） |
| Uniform gate | [1ce0396e](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/1ce0396eb1094c2c828f1943490b4df4/output/log) | [f5d43a75](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/f5d43a75df154e3187285ba6943a9fd3/output/log) | global-batch-8 重训排队；评测依赖已创建 | —（待测） |
| No reliability | [cab5235c](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/cab5235c38cb41d4ad37880d71e5d14a/output/log) | [9ade738e](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/9ade738e28554c5c80c13ccf55cea86e/output/log) | global-batch-8 重训排队；评测依赖已创建 | —（待测） |
| No delay metadata | [34ca7786](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/34ca778640ec44ca9e6f53e1d1be34f6/output/log) | [67351634](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/67351634c0284917ab665de74f0baa71/output/log) | global-batch-8 重训排队；评测依赖已创建 | —（待测） |
| No distillation | [10604830](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/1060483084cd428aa457cf619cbedec6/output/log) | [3f89473f](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/3f89473f2cb6434f99d8fd7f52d5d507/output/log) | global-batch-8 重训排队；评测依赖已创建 | —（待测） |
| Capacity-matched concat | [67918c82](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/67918c82bba84776bd5d6f7c26f54906/output/log) | [a29de3e9](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/a29de3e9665c4d30b68470086ed59cae/output/log) | global-batch-8 重训排队；评测依赖已创建 | —（待测） |
| Linear PTF + no distillation | [8ab23f8b](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/8ab23f8b71d046359b2e61ae126cea7d/output/log) | [1f439ca7](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/1f439ca7d4694ff0ad53362adda97f4b/output/log) | 训练中；评测依赖已创建 | —（待测） |
| Weak feature distillation | [0c0152ba](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/0c0152ba17284f5fb5c43b05d6244074/output/log) | [efd5eead](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/efd5eeadf6ab4c628a58efa004e1757c/output/log) | 训练中；评测依赖已创建 | —（待测） |
| No distillation, peak LR 3e-4 | [9e601b50](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/9e601b50bfdd4aa19cd84282ae91b4e3/output/log) | [2618a45d](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/2618a45d18fe4e1797f9ffecfdc4d9bf/output/log) | 训练排队；评测依赖已创建 | —（待测） |

### 历史消融与受控基线（本仓 ClearML 复现，1789 个验证样本）

| 变体 | 3D AP@0.50 | 3D AP@0.70 | BEV AP@0.50 | BEV AP@0.70 | 结果标签 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| No PTF | 62.82 | 33.80 | 68.40 | 57.48 | 本仓 ClearML 复现（[task](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/327c0a46afad439289402a4c155d9650/output/log)） |
| Linear PTF | 63.36 | 35.21 | 69.03 | 58.07 | 本仓 ClearML 复现（[task](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/0d84d9269d9345adb126bf5dca43b2e6/output/log)） |
| Static three-expert | 62.91 | 35.02 | 68.60 | 57.74 | 本仓 ClearML 复现（[task](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/cbee478f2d084dc19d5baf15a48355a4/output/log)） |
| No distillation | 36.62 | 16.71 | 43.58 | 31.97 | 本仓 ClearML 复现（[task](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/8a8fc0f9e26640e582fecf202f71613e/output/log)） |
| CoFormerNet | 63.93 | 33.32 | 69.49 | 58.17 | 本仓 ClearML 复现（[task](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/cc18359bbc314429bd90c45fad0b2cd0/output/log)） |
| Router-uniform | 63.38 | 35.45 | 68.90 | 58.06 | 本仓 ClearML 复现（[task](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/dfd25f7cab354cb78532199120906a0a/output/log)） |
| FFNet-style | 66.48 | 33.47 | 74.00 | 60.58 | 本仓 ClearML 复现（[task](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/ddbeeec499fb4b55bcd13bc14823b9df/output/log)） |
| No reliability | 65.17 | 34.65 | 70.98 | 59.68 | 本仓 ClearML 复现（[task](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/aadbc8aba0f9484d89274d6967c5e0a7/output/log)） |
| No delay metadata | 62.90 | 34.84 | 68.45 | 59.31 | 本仓 ClearML 复现（[task](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/3d8b301acff844368f8a0a4cfb50efe4/output/log)） |
| Capacity-matched concat | 63.51 | 35.69 | 68.99 | 58.19 | 本仓 ClearML 复现（[task](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/6a94c21ecd1a4948b5f751d4e63a1e38/output/log)） |
| MIT-HAN BEVFusion-style | 65.36 | 35.73 | 70.97 | 59.51 | 本仓 ClearML 复现（[task](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/5e0060f09c58431fb39bbdea6c5448e8/output/log)） |
| V2X-ViT-style | 62.50 | 33.42 | 68.10 | 56.98 | 本仓 ClearML 复现（[task](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/09bafcdff19f4da992f53d2ccbfdfc12/output/log)） |
| CoBEVT-style | 64.97 | 34.81 | 72.45 | 59.21 | 本仓 ClearML 复现（[task](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/9ff8802857c843ffba8de0eea92bab6a/output/log)） |

| 标记 | 状态 |
| :--- | :--- |
| `—（待测）` | 尚未完成 |
| `数值‡` | 公开论文参考值，协议不同 |
| `数值§` | 根据公开 AP 数值计算 |
| `数值¶` | 受控协议单随机种子实测值 |
| `数值 ± 标准差` | 至少 3 个随机种子的聚合结果 |
| `N/A` | 该条件不适用 |

### 历史受控 1337 正常输入比较

主指标：BEV AP@0.7。

| 方法 | 模态 | 骨干网络 | BEV AP@0.7 |
| :--- | :---: | :--- | ---: |
| V2X-ViT-style | LiDAR + Camera | PointPillars + ResNet-50/LSS | 58.998¶ |
| CoBEVT-style | LiDAR + Camera | PointPillars + ResNet-50/LSS | 59.804¶ |
| CoFormerNet-style | LiDAR + Camera | PointPillars + ResNet-50/LSS | 58.560¶ |
| MIT-HAN BEVFusion-style | LiDAR + Camera | PointPillars + ResNet-50/LSS | 59.942¶ |
| FFNet-style | LiDAR + Camera | PointPillars + ResNet-50/LSS | 60.803¶ |
| Resilient V2X（seed `20250218`） | LiDAR + Camera | PointPillars + ResNet-50/LSS | 59.975¶ |

### 历史受控 1337 RSU 时延比较

主指标：BEV AP@0.7。

| 方法 | 模态 | 0 ms | 100 ms | 200 ms | 300 ms | PDR |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| V2X-ViT-style | L + C | 58.998¶ | 57.389¶ | 57.277¶ | 57.256¶ | 2.95¶ |
| CoBEVT-style | L + C | 59.804¶ | 59.823¶ | 59.738¶ | 59.605¶ | 0.33¶ |
| CoFormerNet-style | L + C | 58.560¶ | 58.568¶ | 58.600¶ | 58.585¶ | −0.04¶ |
| MIT-HAN BEVFusion-style | L + C | 59.942¶ | 59.785¶ | 59.788¶ | 59.688¶ | 0.42¶ |
| FFNet-style | L + C | 60.803¶ | 60.297¶ | 58.219¶ | 57.622¶ | 5.23¶ |
| [FFNet](https://proceedings.neurips.cc/paper_files/paper/2023/file/6ca5d2665de83394f437dad0c3746907-Paper-Conference.pdf) | LiDAR | 54.16‡ | N/A | 54.69‡ | 52.44‡ | 3.2§ |
| [CoFormerNet](https://doi.org/10.3390/s24134101) | LiDAR | 54.59‡ | N/A | 54.65‡ | 53.29‡ | 2.4§ |
| Resilient V2X（seed `20250218`） | L + C | 59.975¶ | 57.936¶ | 57.868¶ | 57.829¶ | 3.6¶ |

### 历史受控 1337 单模态故障

结果格式：`BEV AP@0.5 / BEV AP@0.7（基于 AP@0.7 的 PDR）`。

| 方法 | Normal | L-Fail | C-Fail |
| :--- | :---: | :---: | :---: |
| V2X-ViT-style | 68.361¶ / 58.998¶ | 63.261¶ / 30.747¶（↓47.9¶） | 68.366¶ / 59.063¶（↑0.1¶） |
| CoBEVT-style | 72.962¶ / 59.804¶ | 66.714¶ / 35.946¶（↓39.9¶） | 72.955¶ / 59.699¶（↓0.2¶） |
| CoFormerNet-style | 69.806¶ / 58.560¶ | 65.649¶ / 34.100¶（↓41.8¶） | 69.787¶ / 58.528¶（↓0.1¶） |
| MIT-HAN BEVFusion-style | 73.251¶ / 59.942¶ | 68.843¶ / 35.035¶（↓41.6¶） | 71.332¶ / 59.978¶（↑0.1¶） |
| FFNet-style | 74.272¶ / 60.803¶ | 62.204¶ / 28.512¶（↓53.1¶） | 74.293¶ / 60.759¶（↓0.1¶） |
| Resilient V2X（seed `20250218`） | 68.955¶ / 59.975¶ | 69.319¶ / 48.915¶（↓18.4¶） | 68.940¶ / 60.005¶（↑0.1¶） |

### 历史受控 1337 模态故障与时延联合退化

主指标：BEV AP@0.7。

| 条件 | 0 ms | 100 ms | 200 ms | 300 ms |
| :--- | ---: | ---: | ---: | ---: |
| Full | 59.975¶ | 57.936¶ | 57.868¶ | 57.829¶ |
| L-Fail（E+R） | 48.915¶ | 49.018¶ | 49.106¶ | 44.636¶ |
| C-Fail（E+R） | 60.005¶ | 57.919¶ | 57.897¶ | 57.865¶ |

### 历史受控 1337 容量匹配消融

主指标：BEV AP@0.7。

| 变体 | Full | L-Fail | 300 ms | 评测任务 |
| :--- | ---: | ---: | ---: | :--- |
| Full nonlinear PTF + DER | 59.975¶ | 48.915¶ | 57.829¶ | [2992081b](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/2992081bc95949f198e062c136810736/output/log) |
| No PTF | 59.635¶ | 28.952¶ | 57.487¶ | [10c08838](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/10c08838df2443afa561ef157cabe0e2/output/log) |
| Linear PTF | 60.356¶ | 51.032¶ | 58.118¶ | [d408940f](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/d408940f1e824b23aeb9285384e62568/output/log) |
| Static three-expert | 60.046¶ | 50.943¶ | 57.976¶ | [66c0ee28](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/66c0ee28f11745449c53db170e542e7b/output/log) |
| Uniform gate | 60.383¶ | 49.827¶ | 58.291¶ | [a9fd7482](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/a9fd7482194f411f8906bc661195aee3/output/log) |
| No reliability | 60.177¶ | 49.269¶ | 58.115¶ | [91568893](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/9156889384b849dfb14f04e668befa70/output/log) |
| No delay metadata | 59.764¶ | 50.787¶ | 57.544¶ | [b0a5c0f0](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/b0a5c0f08a8b4b1fba05cd63bc5d2ee9/output/log) |
| No distillation | 62.161¶ | 50.621¶ | 60.128¶ | [e74e15ee](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/e74e15eeee7844a9a8fbdfaf056a7ea0/output/log) |
| Capacity-matched concat | 60.344¶ | 49.463¶ | 58.258¶ | [62cd060b](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/62cd060bdc9143b0983e0752a0d7efd0/output/log) |

### 连续故障持续时间

主指标：0 ms、`E+R` 下的 BEV AP@0.7。

| 持续时间 | BEV AP@0.7 | 支持状态 |
| ---: | ---: | :--- |
| 1 帧 | —（待测） | 历史窗口内 |
| 2 帧 | —（待测） | 历史窗口内 |
| 3 帧 | —（待测） | 历史窗口内 |
| 4 帧及以上 | —（待测） | neutral/unsupported |

### 部署复杂度

| 模型 | 参数量（M） | FLOPs（G） | 峰值 GPU 显存（GB） | 端到端时延（ms） |
| :--- | ---: | ---: | ---: | ---: |
| Capacity-matched concat | —（待测） | —（待测） | —（待测） | —（待测） |
| Resilient V2X student | —（待测） | —（待测） | —（待测） | —（待测） |

### 历史三随机种子汇总（未完成）

| 数据集 / 条件 | BEV AP@0.5 mean | std | BEV AP@0.7 mean | std | 3D AP@0.5 mean | std | 3D AP@0.7 mean | std | 完成种子数 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DAIR-V2X-C / Resilient V2X / Full / 0 ms | 68.955¶ | —（待测） | 59.975¶ | —（待测） | 65.372¶ | —（待测） | 35.142¶ | —（待测） | 1 / ≥3 |

### 历史补充结果

| 记录 | 指标 | 0 ms | 100 ms | 200 ms | 300 ms |
| :--- | :--- | ---: | ---: | ---: | ---: |
| CoFormerNet public | BEV AP@0.5 | 69.33‡ | N/A | 69.13‡ | 68.60‡ |
| FFNet controlled baseline | BEV AP@0.7 | N/A | 60.297¶ | N/A | N/A |
| CoFormerNet controlled baseline | BEV AP@0.7 | N/A | 58.568¶ | N/A | N/A |

| 条件 | Agent scope | Resilient V2X − CoFormerNet-style BEV AP@0.7 |
| :--- | :---: | ---: |
| L-Fail | E+R | +14.815¶ |
| C-Fail | E+R | +1.477¶ |
### E-only / R-only 诊断


| 故障 | Agent scope | 0 ms | 100 ms | 200 ms | 300 ms |
| :--- | :---: | ---: | ---: | ---: | ---: |
| L-Fail | E-only | —（待测） | —（待测） | —（待测） | —（待测） |
| L-Fail | R-only | —（待测） | —（待测） | —（待测） | —（待测） |
| C-Fail | E-only | —（待测） | —（待测） | —（待测） | —（待测） |
| C-Fail | R-only | —（待测） | —（待测） | —（待测） | —（待测） |

### V2XSet

| 轨道 | Full 0 ms | Full 100 ms | Full 200 ms | Full 300 ms | L-Fail | C-Fail |
| :--- | ---: | ---: | ---: | ---: | :---: | :---: |
| V2XSet-Standard / LiDAR-only | —（待测） | —（待测） | —（待测） | —（待测） | N/A | N/A |
| V2XSet-Pair / LiDAR + Camera | —（待测） | —（待测） | —（待测） | —（待测） | —（待测） | —（待测） |

</details>
