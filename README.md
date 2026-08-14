# Transvision

## 结果来源标签

| 标签 | 含义 |
| :--- | :--- |
| `论文原始结果` | 原论文表格或图中直接报告的数值；保持原指标、单位与协议 |
| `官方代码库结果` | 论文作者官方仓库报告的基准数值，不等同于论文原表 |
| `本仓 ClearML 复现` | 本仓任务日志可追溯的复现实测数值 |
| `本仓受控结果` | 本仓统一受控协议下的实测数值 |

## 当前受控实验状态

状态快照：上海时间 `2026-08-15 01:03`。本节是当前论文结果的唯一活动入口；未完成项只保留 `—（待测）`，不以诊断值或跨协议值代填。

### DAIR-CAUSAL-1337-v1 统一评测

| 项目 | 固定协议 |
| :--- | :--- |
| 数据与样本 | DAIR-CAUSAL-1337-v1；1337 samples；11330 Car ground truth；dataset `7c59fabb9da949e6b3c94c732f000975` |
| 条件 | `0/100/200/300 ms × Full/L-Fail/C-Fail` 共 12 条件；L-Fail 和 C-Fail 使用 `E+R`；每条件 `unsupported_sample_count=0` |
| 指标 | BEV AP@0.5、BEV AP@0.7、3D AP@0.5、3D AP@0.7；四项全部报告 |
| 训练合同 | seed `20250218`；4 GPU；每卡 batch size `2`；global batch size `8`；FP32；50 epoch；val/10 epoch |
| 数据指纹 | manifest `715ac6f7a14225e20327eed0650c55abdc0cb98431830164e84545238099645d`；overlay index `77bd4585dbb02901f862b8da6aa208a504674b824a3d55cf15005aacbeeeaaff`；ordered sample IDs `a8d8184f7fd9d1212ae29cddb427f48a0cad39e7843d95d5ac609a8a4286cf3a` |
| 结论范围 | 单种子、同协议 DAIR 受控领先；不与跨协议公开结果混称全局 SOTA |

本轮固定使用一个随机种子；不再执行“至少 3 个随机种子”合同，也不报告未生成的 mean±std。

### 正式依赖链与评测队列

| 节点 | ClearML task ID | 状态 | 结果 |
| :--- | :--- | :--- | :--- |
| P：训练来源等价证明 | [`7e244a71`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/7e244a711751469b8cfdb25d77b05269/output/log) | `completed` | formal provenance 已提交 |
| W：26 方法正式评测 watcher | [`74bf35de`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/74bf35decc2340b496167c11ec50f54b/output/log) | `in_progress`；`services` | A100 并行候选合同已通过；26 项中 6 项完成、2 项运行、18 项已创建 |
| L：正式 leaderboard | [`cc7b54e4`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/cc7b54e4dcca4b92a4247f92987add60/output/log) | `in_progress`；`services` | 等待 26 项正式评测完成 |
| A：独立可比性审计 | [`dd287fe5`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/dd287fe58e264bcca48fe222b6981048/output/log) | `in_progress`；`services` | 等待 leaderboard 与完整证据 |
| S：单种子候选选择器 | [`d7ea54ce`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/d7ea54ce540d4b0486904c5910100885/output/log) | `in_progress`；`services` | 等待审计完成后生成最终 winner |
| CoFormerNet 正式评测 | [`d8fac863`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/d8fac86325e047e5aa25f5ce899902b6/output/log) | `completed` | `DAIR-CAUSAL-1337-v1`；Full 0 ms BEV AP@0.7 `58.6980`；12 条件均值 `49.9536`；最差值 `31.7329`；`12×1337 / 11330 / 0`；metrics SHA-256 `423323cacd2faf115e1f8a5a2526b58a993c9a23fa154eea829939a8ffbc11b0` |
| FFNet 正式评测 | [`144397bf`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/144397bfa9c242bc9a92a1279922558b/output/log) | `completed` | Full `59.2396`；均值 `49.7097`；最差 `29.9796`；`12×1337 / 11330 / 0`；metrics SHA-256 `e4f7c3578e65e11358a232ba0f38d02d83a5fc45dc76cf8c188039a596e86e24` |
| V2X-ViT 正式评测 | [`cb2675d7`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/cb2675d7e0e845268420f5c0d5248ece/output/log) | `completed` | Full `58.6679`；均值 `49.9924`；最差 `30.7127`；`12×1337 / 11330 / 0`；metrics SHA-256 `49556d0b1c4b0ddb5a3a6ac835a3ae353c8553509e2cf32903bcd55b9a69e515` |
| CoBEVT 正式评测 | [`23f4d7f0`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/23f4d7f082284aa2be098a90f0a596e8/output/log) | `completed` | Full `59.1071`；均值 `50.7776`；最差 `33.3430`；`12×1337 / 11330 / 0`；metrics SHA-256 `07dbc919b5f9ac7b90a9fe5f5f8896aee0315d12070ef7d05e41bd1e70848462` |
| BEVFusion 正式评测 | [`c3b87760`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/c3b87760b78742cb8e7de3a506a08999/output/log) | `completed` | Full `59.7471`；均值 `51.0931`；最差 `33.6540`；`12×1337 / 11330 / 0`；metrics SHA-256 `43977ebdae38e74882bb659c53d4c10e2cb20eba1a7a5c77b2514bccabd21c1f` |
| ResilientV2X 正式评测 | [`7deb18e5`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/7deb18e532324850bee1fb4279a838b7/output/log) | `completed` | Full `62.3620`；均值 `58.4076`；最差 `50.8587`；`12×1337 / 11330 / 0`；metrics SHA-256 `5a5f111876544de2403ef18ca3b7cb616ba6047277119a4b3461ad5cc0a56a65` |
| `support_residual` 正式评测 | [`718ba3d3`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/718ba3d3421249eb9f31d5216a8f160a/output/log) | `in_progress`；`GPU4-A100` 0–3 | —（待测） |
| `ptf_none` 正式评测 | [`e6ad9de8`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/e6ad9de8555e49acb6b7a7f869642c2c/output/log) | `in_progress`；`GPU4-A100` 4–7 | —（待测） |
| 其余 18 方法正式评测 | 由 W 按固定优先级绑定 | `created`；按四卡资源门控依次释放 | —（待测） |

26 个同名旧空壳已归档，不进入论文结果。归档回执：[`execute-receipt-20260812T051245.388941Z.json`](artifacts/resilient_v2x/formal-evaluation-orphan-reconciliation/execute-receipt-20260812T051245.388941Z.json)；seal `882773639e93a3f60590ba044b17da499790f2269697bac6ead5a650862b59a0`。评测恢复回执：[`recovery-receipt-20260812T1450CST.json`](artifacts/resilient_v2x/formal-evaluation-recovery/recovery-receipt-20260812T1450CST.json)，seal `d95e2bc2bb93ea6c8c69ea683d31ba09ec2bf219c7a0c09591c0ff12de60fcdf`。A100 并行合同 W/L/A/S exact-ID 恢复回执：[`a100-parallel-recovery-20260813T1443CST.json`](artifacts/resilient_v2x/formal-successor-runtime-recovery/a100-parallel-recovery-20260813T1443CST.json)，seal `b025db64c5836fd0a1687638a8358aa7bad6c54db62e81246cdb8e9da53a9bfd`；未创建替代任务。旧失败链仅保留在 archive。

### 候选训练、评测与模型归档

| 候选 | 训练 task ID | 训练状态 | 正式评测 | 论文 final checkpoint |
| :--- | :--- | :--- | :--- | :--- |
| E1 `support_residual_linear` | [`f0c3082f`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/f0c3082f3aa34a81805903e0ffdc8610/output/log) | `completed` | [`c6f26cc7`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/c6f26cc7902142c090ec238856409ac6/output/log) `completed`；Full `62.3024`；均值 `58.1732`；最差 `50.1283`；`12×1337 / 11330 / 0`；metrics SHA-256 `c175a1db5f8d31ba6f18119a38df834ebd1dda918ed46ef75a69a7aaaf596148` | 已归档；epoch 50；SHA-256 `a0841635846eafbd1053a86847f402a75efdf376ead600beffaf6281afa23668` |
| E2 `no_reliability_linear` | [`969c8fce`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/969c8fce6d24446299561772b3955274/output/log) | `completed` | [`c337353b`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/c337353b188c41cba2ad92fcaabcaf35/output/log) `completed`；Full `62.4457`；均值 `58.6384`；最差 `50.3353`；metrics SHA-256 `643d56561db7ae3e0c1565efc8004ec7fd3c36fc6b6c9aee57cfd9d6b36233d7` | 已归档；epoch 50；SHA-256 `4813503a7022bdfdd3f404bf8918f42c100abb807b61ac3d9c8d2c4e508ec557` |
| E3 `support_residual_no_reliability` | [`dc037315`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/dc037315c0684c3d854a2fd7c19a2a2f/output/log) | `completed` | [`27a82d39`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/27a82d39e6354697ad1532b6399bbde2/output/log) `completed`；Full `62.2258`；均值 `58.1152`；最差 `50.8195`；metrics SHA-256 `20e189151a56e0b957e4d404689a518222b115c51daf96bd6c1aeb485d2c9cae` | 已归档；epoch 50；SHA-256 `0259775f532c5b876ab960a6b62d4c99a8c82007217229f840bdbb342fad8e0f` |
| P0 `support_residual_no_reliability_linear` | [`8883c51c`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/8883c51ced4f4951a45edbaefe6342d4/output/log) | `completed` | [`8d7d39dc`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/8d7d39dcc540475a8827d3cac4b9068e/output/log) `completed`；Full `62.5318`；均值 `58.4021`；最差 `50.6362`；metrics SHA-256 `930f648985a2e794a49e08a126d06210d3642fa193198015376b267ef146f1f8` | 已归档；epoch 50；SHA-256 `210208a21ae944e0b73fc92e6a15e570f20f452889a3895282c483f8ad9f2a2c` |
| P2 `P0 + bbox loss 2.5` | [`f5d3820b`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/f5d3820b4cdf416183c8f1fee566abe3/output/log) | `completed` | [`908fe861`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/908fe86179a24a64811d92b3af4d903f/output/log) `completed`；Full `62.1244`；均值 `57.9189`；最差 `50.3869`；metrics SHA-256 `1300289425faea3f82ec12b329ffa700bf7d017612a03c368e4fb7482970a4da` | 已归档；epoch 50；SHA-256 `c12ff4d8a5c25932b7806e02b72927805dbf7d93e90a67e4c9d55364fa2e5a3e` |
| 候选评测 controller | [`b6f0fbab`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/b6f0fbab32a5478183a45b3ca833fc01/output/log) | `completed`；`services`；源码 SHA-256 `1fe760c9461c1a4f98d157597ba47f581f527e2790c7e368c00dd65c3a9599f8` | E1/E2/E3/P0/P2 全部完成；manifest SHA-256 `d864b721da3f14f2c618a6b55a6bf58557301a92893a285d15c69fe5202341e6` | N/A |

候选控制器恢复回执：[`candidate-recovery-20260813T0915CST.json`](artifacts/resilient_v2x/formal-original-queue-recovery/candidate-recovery-20260813T0915CST.json)；seal `84be42946fc108fc9469190fb7474bbeae95a515e9c5a6827e0f2fd349cc1f`。E1 运行态参数兼容修复回执：[`candidate-e1-defaults-recovery-20260813T0943CST.json`](artifacts/resilient_v2x/formal-original-queue-recovery/candidate-e1-defaults-recovery-20260813T0943CST.json)；seal `32291605a241b030fadf62b72842efa06b1d277cbfc9f3711cb68a8933f41c7d`。E1 正式证据读取修复回执：[`candidate-e1-evidence-recovery-20260813T1028CST.json`](artifacts/resilient_v2x/formal-original-queue-recovery/candidate-e1-evidence-recovery-20260813T1028CST.json)；seal `fa1e131bf70940a55dda63ec0a29aa788ef3aaa1f77fa19c6722986e451028ed`。A100 并行升级回执：[`candidate-a100-parallel-upgrade-20260813T1117CST.json`](artifacts/resilient_v2x/formal-original-queue-recovery/candidate-a100-parallel-upgrade-20260813T1117CST.json)；seal `2f02476a640171cb64e088fbd3b47982ba0f64dc81ba15bb37026ecf4952a2fb`。权威队列读回恢复回执：[`candidate-a100-readback-recovery-20260813T1128CST.json`](artifacts/resilient_v2x/formal-original-queue-recovery/candidate-a100-readback-recovery-20260813T1128CST.json)；seal `25c2b4ac485d8c0d7f91fd2a46af9e083562dca72759888fb0e72f70965263ed`。所有恢复均保持 task ID `b6f0fbab32a5478183a45b3ca833fc01`，未创建替代任务。

本地归档目录：`artifacts/trained_models/completed-live/`。只有 epoch-50 final 可进入论文；clean-best 仅作诊断。

### 单次受控领先结果

五个固定受控基线为 FFNet、CoFormerNet、V2X-ViT、CoBEVT、BEVFusion。门槛为：Full 0 ms BEV AP@0.7 ≥ 五基线最佳值 − `0.5 AP`，且 12 条件均值、最差值分别严格高于五基线最佳均值、最佳最差值。

| 对象 | Full 0 ms BEV AP@0.7 | 12 条件均值 | 12 条件最差值 | 门控 |
| :--- | ---: | ---: | ---: | :--- |
| CoFormerNet（本仓受控结果） | 58.6980 | 49.9536 | 31.7329 | `completed`；五基线之一 |
| FFNet（本仓受控结果） | 59.2396 | 49.7097 | 29.9796 | `completed`；五基线之一 |
| V2X-ViT（本仓受控结果） | 58.6679 | 49.9924 | 30.7127 | `completed`；五基线之一 |
| CoBEVT（本仓受控结果） | 59.1071 | 50.7776 | 33.3430 | `completed`；五基线之一 |
| BEVFusion（本仓受控结果） | 59.7471 | 51.0931 | 33.6540 | `completed`；五基线之一 |
| 五基线最佳 | 59.7471 | 51.0931 | 33.6540 | `completed` |
| ResilientV2X（当前方法） | **62.3620** | **58.4076** | **50.8587** | `pass`；Full `+2.6149`，均值 `+7.3145`，最差 `+17.2047` |
| E1 `support_residual_linear`（候选） | **62.3024** | **58.1732** | **50.1283** | `pass`；Full `+2.5553`，均值 `+7.0801`，最差 `+16.4743` |
| E2 `no_reliability_linear`（候选） | **62.4457** | **58.6384** | **50.3353** | `pass` |
| E3 `support_residual_no_reliability`（候选） | **62.2258** | **58.1152** | **50.8195** | `pass`；当前候选排序第一 |
| P0 `support_residual_no_reliability_linear`（候选） | **62.5318** | **58.4021** | **50.6362** | `pass` |
| P2 `P0 + bbox loss 2.5`（候选） | **62.1244** | **57.9189** | **50.3869** | `pass` |
| 最终 winner | —（待 S 正式封存） | —（待 S 正式封存） | —（待 S 正式封存） | —（待选） |

### 论文后续实验占位

| 论文接口 | 状态 | 结果 |
| :--- | :--- | :--- |
| Table I–IV：winner 与五个受控基线 | 五个受控基线与当前 ResilientV2X 已完成；等待候选最终选择 | —（待最终 winner） |
| Table V：winner 单因素消融 | 等待 winner identity | —（待测） |
| Table VI：`p=0.0` | 训练 [`4c4ec658`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/4c4ec65836ad463da0041a1caf732aba/output/log) `completed`；正式评测 [`63ab06eb`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/63ab06ebdd1649f7ba8b77262a211cc5/output/log) `in_progress`；`GPU4-5090` | —（待测） |
| Table VI：`p=0.1` | 训练 [`bdeee071`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/bdeee07140b245409e89db2e935dd02f/output/log) `completed`；正式评测 [`087bc9f0`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/087bc9f0f679425da29c2329aa239197/output/log) `in_progress`；`GPU4-5090` | —（待测） |
| Table VI：`p=0.2` | 训练 [`5b426f33`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/5b426f335aea422f840d1d3c7879744a/output/log) `completed`；正式评测 [`8a8ccd4f`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/8a8ccd4fb1a9420d8d49372c541deaab/output/log) `in_progress`；`GPU4-A100` 0–3 | —（待测） |
| Table VI：`p=0.3` | 训练 [`e431d93a`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/e431d93a909f4888a0d79561b5ca749c/output/log) `completed`；正式评测 [`a698dc19`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/a698dc196a144104996b13528ddbc95e/output/log) `queued`；`GPU4-5090` | —（待测） |
| Table VI：`p=0.5` | 训练 [`59d1c2bb`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/59d1c2bbb91741e98f46564accc58477/output/log) `in_progress`；epoch 19/50；正式评测 [`f0b0faeb`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/f0b0faeb2c2a46eab8038ce562453ae6/output/log) 已绑定完成依赖 | —（待测） |
| Table VII：`q=1` LiDAR | [`cb5eb3f9`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/cb5eb3f9cb624018b504049bf9492d30/output/log) `completed`；`1337 / 11330 / 0` | BEV AP@0.5/0.7 `69.9425 / 53.1879`；3D AP@0.5/0.7 `61.8763 / 29.8698` |
| Table VII：`q=1` Camera；`q=2,3` LiDAR/Camera | [`d9d17062`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/d9d1706284614fef8775a80d17c0cf91/output/log)、[`4417e476`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/4417e476be21492d8e3dc817b3f1f932/output/log)、[`fa3423b9`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/fa3423b90d5944db99552a2cbf9f66a6/output/log)、[`66121b53`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/66121b5326e2460880f0f91cb844197b/output/log)、[`759463bc`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/759463bca4424072ba3d5e9679f94fe3/output/log) 已排队 | —（待测）；`q≥4` 为 unsupported/no-extrapolation |
| Table VII：winner / 容量匹配 concat 参数、FLOPs、显存、端到端时延 | [`86b92bac`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/86b92bacc8c74a80b9583130dc89dba4/output/log)、[`22ed132a`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/22ed132a00ef4dd0b2294f8dd3ed4f4f/output/log) 已排队；pair validator [`c2593441`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/c2593441377747558c5c1ce56302757a/output/log) 已绑定 | —（待测） |
| E-only / R-only 诊断 | [`30d330b4`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/30d330b42c6840aeb0e7163787fe0df0/output/log)、[`2af4e6d4`](http://10.100.34.118:8080/projects/6e43f972e5ea4cee901a7c8855fce8cd/experiments/2af4e6d4982c472586ec22bd3a05e2c5/output/log) `in_progress` | —（待测） |
| V2XSet-Standard | 本轮非阻塞，不训练 | —（待测） |
| V2XSet-Pair | 本轮非阻塞，不训练 | —（待测） |

### 排除项与归档

旧稿 `†` 数值、旧 global-batch-4、clean-1789、跨协议公开值、无 manifest/run contract/sealed source/evaluator evidence 的旧权重，均不进入当前受控表。旧 global-batch-4 任务已停止，不用于最终比较；清单见 [`docs/resilient_v2x/archive/global-batch-4-superseded.md`](docs/resilient_v2x/archive/global-batch-4-superseded.md)。旧稿数值见 [`docs/resilient_v2x/archive/old-draft-results.md`](docs/resilient_v2x/archive/old-draft-results.md)。

[历史结果与任务快照归档](docs/resilient_v2x/archive/historical-results-and-task-snapshots.md)

## 论文与官方代码库原始结果

以下数值保持来源中的原始指标和单位，不跨数据集换算。`—` 表示来源未报告。

### 【论文原始结果】CoFormerNet（Sensors 2024）

来源：[CoFormerNet 原论文](https://doi.org/10.3390/s24134101)。

#### DAIR-V2X，原论文 Table 1

| 方法 | 融合类型 | 时延（ms） | 3D AP@0.5 | 3D AP@0.7 | BEV AP@0.5 | BEV AP@0.7 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| EMIFF（Camera） | Intermediate | 0 | 15.61 | — | 21.44 | — |
| PointPillars | Non-fusion | — | 48.06 | — | 52.24 | — |
| VoxelNet | Non-fusion | — | 52.40 | 34.69 | 58.08 | 49.18 |
| FFNet | Intermediate | 0 | 55.81 | 30.23 | 63.54 | 54.16 |
| TransIFF | Intermediate | 0 | 59.62 | 46.03 | — | — |
| CoFormerNet | Intermediate | 0 | 61.03 | 39.14 | 69.33 | 54.59 |
| Early Fusion | Early | 200 | 54.63 | 38.23 | 61.08 | 50.06 |
| Late Fusion | Late | 200 | 52.43 | 36.54 | 58.10 | 49.25 |
| FFNet | Intermediate | 200 | 55.37 | 31.66 | 63.20 | 54.69 |
| TransIFF | Intermediate | 200 | 53.47 | 37.21 | — | — |
| CoFormerNet | Intermediate | 200 | 60.97 | 38.97 | 69.13 | 54.65 |
| Early Fusion | Early | 300 | 51.37 | 37.25 | 58.28 | 49.81 |
| Late Fusion | Late | 300 | 51.35 | 36.24 | 56.89 | 48.79 |
| FFNet | Intermediate | 300 | 53.46 | 30.42 | 61.20 | 52.44 |
| TransIFF | Intermediate | 300 | 51.02 | 31.74 | — | — |
| CoFormerNet | Intermediate | 300 | 60.63 | 37.28 | 68.60 | 53.29 |

#### V2XSet，原论文 Table 2

| 方法 | 时延（ms） | 3D AP@0.5 | 3D AP@0.7 |
| :--- | ---: | ---: | ---: |
| V2X-ViT | 0 | 88.23 | 71.27 |
| FFNet | 0 | 89.47 | 72.66 |
| CoFormerNet | 0 | 90.23 | 72.95 |
| V2X-ViT | 200 | 83.61 | 61.49 |
| FFNet | 200 | 85.45 | 70.23 |
| CoFormerNet | 200 | 89.28 | 71.02 |
| V2X-ViT | 300 | 80.71 | 54.66 |
| FFNet | 300 | 83.31 | 57.93 |
| CoFormerNet | 300 | 84.22 | 59.01 |

#### 消融，原论文 Table 3

| TAM | SMCA | End2End | 时延（ms） | 3D AP@0.5 | 3D AP@0.7 | BEV AP@0.5 | BEV AP@0.7 |
| :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| × | × | × | 0 | 55.67 | 35.12 | 63.78 | 54.26 |
| ✓ | × | × | 0 | 58.25 | 36.27 | 67.84 | 54.31 |
| ✓ | ✓ | ✓ | 0 | 60.67 | 39.12 | 70.78 | 55.26 |
| × | × | × | 200 | 52.40 | 34.69 | 58.08 | 49.48 |
| × | ✓ | × | 200 | 55.40 | 34.44 | 63.14 | 52.32 |
| ✓ | × | × | 200 | 57.11 | 35.09 | 66.93 | 52.87 |
| ✓ | ✓ | × | 200 | 59.24 | 36.06 | 67.53 | 54.02 |
| ✓ | ✓ | ✓ | 200 | 60.97 | 38.97 | 69.13 | 54.65 |

#### 速度，原论文 Table 4

| 方法 | 推理时间 | 3D AP@0.5 |
| :--- | ---: | ---: |
| PointPillars | 31 ms | 48.06 |
| VoxelNet | 95 ms | 52.40 |
| TransIFF | 110 ms | 59.62 |
| FFNet | 101 ms | 55.81 |
| CoFormerNet | 122 ms | 61.03 |

### 【论文原始结果】FFNet（NeurIPS 2023）

来源：[FFNet 原论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/6ca5d2665de83394f437dad0c3746907-Paper-Conference.pdf)。DAIR-V2X、Car、范围 `[0,-39.12,100,39.12]`。

#### 融合方法比较，原论文 Table 1

| 方法 | 融合类型 | 时延（ms） | 3D AP@0.5 | 3D AP@0.7 | BEV AP@0.5 | BEV AP@0.7 | AB（Byte） |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| PointPillars | Non-fusion | — | 48.06 | — | 52.24 | — | 0 |
| AutoAlignV2 | Non-fusion | — | 50.32 | — | 53.88 | — | 0 |
| Early Fusion | Early | 200 | 54.63 | 38.23 | 61.08 | 50.06 | 1.4×10^6 |
| Late Fusion | Late | 200 | 52.43 | 36.54 | 58.10 | 49.25 | 5.1×10^2 |
| DiscoNet | Middle | 200 | 50.76 | 28.57 | 58.20 | 48.90 | 1.2×10^5 |
| V2VNet | Middle | 200 | 49.67 | 26.96 | 56.02 | 46.32 | 1.2×10^5 |
| FFNet | Middle | 200 | 55.37 | 31.66 | 63.20 | 54.69 | 1.2×10^5 |
| FFNet-C1 | Middle | 200 | 55.17 | 31.20 | 62.87 | 54.28 | 1.7×10^4 |
| Early Fusion | Early | 300 | 51.37 | 37.25 | 58.28 | 49.81 | 1.4×10^6 |
| Late Fusion | Late | 300 | 51.35 | 36.24 | 56.89 | 48.79 | 5.1×10^2 |
| DiscoNet | Middle | 300 | 49.03 | 27.39 | 55.81 | 47.28 | 1.2×10^5 |
| V2VNet | Middle | 300 | 48.51 | 27.00 | 55.81 | 46.32 | 1.2×10^5 |
| FFNet | Middle | 300 | 53.46 | 30.42 | 61.20 | 52.44 | 1.2×10^5 |
| FFNet-C1 | Middle | 300 | 54.10 | 29..87（原文排版） | 60.76 | 53.28 | 1.7×10^4 |

#### 特征预测消融，原论文 Table 2

| 方法 | 时延（ms） | 3D AP@0.5 | 3D AP@0.7 | BEV AP@0.5 | BEV AP@0.7 | AB（Byte） |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| FFNet | 0 | 55.81 | 30.23 | 63.54 | 54.16 | 1.2×10^5 |
| FFNet（without prediction） | 0 | 55.81 | 30.23 | 63.54 | 54.16 | 6.2×10^4 |
| FFNet-V2（without prediction） | 0 | 55.78 | 30.22 | 64.23 | 55.00 | 1.2×10^5 |
| FFNet | 200 | 55.37 | 31.66 | 63.20 | 54.69 | 1.2×10^5 |
| FFNet（without prediction） | 200 | 50.27 | 27.57 | 57.93 | 48.16 | 6.2×10^4 |
| FFNet-V2（without prediction） | 200 | 49.90 | 27.33 | 58.00 | 48.22 | 1.2×10^5 |

### 【论文原始结果】V2X-ViT（ECCV 2022）

来源：[V2X-ViT 原论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990106.pdf)。原文 AP 单位为 `0–1`；noisy 设置为 `0.2 m / 0.2° / 100 ms`。

#### V2XSet，原论文 Table 1

| 方法 | Perfect AP@0.5 | Perfect AP@0.7 | Noisy AP@0.5 | Noisy AP@0.7 |
| :--- | ---: | ---: | ---: | ---: |
| No Fusion | 0.606 | 0.402 | 0.606 | 0.402 |
| Late Fusion | 0.727 | 0.620 | 0.549 | 0.307 |
| Early Fusion | 0.819 | 0.710 | 0.720 | 0.384 |
| F-Cooper | 0.840 | 0.680 | 0.715 | 0.469 |
| OPV2V | 0.807 | 0.664 | 0.709 | 0.487 |
| V2VNet | 0.845 | 0.677 | 0.791 | 0.493 |
| DiscoNet | 0.844 | 0.695 | 0.798 | 0.541 |
| V2X-ViT | 0.882 | 0.712 | 0.836 | 0.614 |

#### 组件消融，原论文 Table 2

| Base | MSwin | SpAttn | HMSA | DPE | AP@0.5 | AP@0.7 |
| :---: | :---: | :---: | :---: | :---: | ---: | ---: |
| ✓ | × | × | × | × | 0.719 | 0.478 |
| ✓ | ✓ | × | × | × | 0.748 | 0.519 |
| ✓ | ✓ | ✓ | × | × | 0.786 | 0.548 |
| ✓ | ✓ | ✓ | ✓ | × | 0.823 | 0.601 |
| ✓ | ✓ | ✓ | ✓ | ✓ | 0.836 | 0.614 |

#### DPE 时延消融，原论文 Table 3

| 时延 | AP@0.7 without DPE | AP@0.7 with DPE |
| ---: | ---: | ---: |
| 100 ms | 0.639 | 0.650 |
| 200 ms | 0.558 | 0.572 |
| 300 ms | 0.496 | 0.514 |
| 400 ms | 0.458 | 0.478 |

#### 推理速度，原论文 Table 4

| 方法 | V100 推理时间 | Perfect AP@0.7 | Noisy AP@0.7 |
| :--- | ---: | ---: | ---: |
| V2X-ViT-S | 28 ms | 0.696 | 0.591 |
| V2X-ViT | 57 ms | 0.712 | 0.614 |

### 【论文原始结果】CoBEVT（CoRL 2022）

来源：[CoBEVT 原论文](https://proceedings.mlr.press/v205/xu23a/xu23a.pdf)。

#### OPV2V Camera Track，原论文 Table 1

| 方法 | Vehicle IoU | Drivable Area IoU | Lane IoU |
| :--- | ---: | ---: | ---: |
| No Fusion | 37.7 | 57.8 | 43.7 |
| Map Fusion | 45.1 | 60.0 | 44.1 |
| F-Cooper | 52.5 | 60.4 | 46.5 |
| AttFuse | 51.9 | 60.5 | 46.2 |
| V2VNet | 53.5 | 60.2 | 47.5 |
| DiscoNet | 52.9 | 60.7 | 45.8 |
| FuseBEVT | 59.0 | 62.1 | 49.2 |
| CoBEVT | 60.4 | 63.0 | 53.0 |

#### OPV2V LiDAR Track，原论文 Table 2

| 方法 | AP@0.7 | AP@0.7（64× 压缩） |
| :--- | ---: | ---: |
| No Fusion | 60.2 | 60.2 |
| Late Fusion | 78.1 | 78.1 |
| Early Fusion | 80.0 | — |
| F-Cooper | 79.0 | 78.8 |
| AttFuse | 81.5 | 81.0 |
| V2VNet | 82.2 | 81.4 |
| DiscoNet | 83.6 | 83.1 |
| FuseBEVT | 85.2 | 84.9 |

#### nuScenes 单车地图分割，原论文 Table 3

| 方法 | Vehicle IoU | 参数量（M） | FPS |
| :--- | ---: | ---: | ---: |
| VPN* | 29.3 | 4.0 | 31 |
| OFT | 30.1 | — | — |
| Lift-Splat | 32.1 | 14 | 25 |
| FIERY | 35.8 | 7 | 8 |
| CVT | 36.0 | 1.2 | 35 |
| SinBEVT | 37.1 | 1.6 | 35 |

#### 压缩，原论文 Table 4

| 压缩率 | 大小（KB） | Vehicle IoU |
| ---: | ---: | ---: |
| 0× | 524 | 60.4 |
| 8× | 66 | 60.1 |
| 16× | 33 | 58.9 |
| 32× | 16 | 56.2 |
| 64× | 8 | 54.8 |

#### FAX 组件消融，原论文 Table 5

| Local | Global | Vehicle IoU | Drivable Area IoU | Lane IoU |
| :---: | :---: | ---: | ---: | ---: |
| × | × | 52.6 | 57.9 | 42.0 |
| ✓ | × | 57.8 | 61.5 | 49.2 |
| × | ✓ | 57.9 | 60.8 | 48.6 |
| ✓ | ✓ | 60.4 | 63.0 | 53.0 |

### 【论文原始结果】MIT-HAN BEVFusion（ICRA 2023）

来源：[BEVFusion 原论文](https://arxiv.org/pdf/2205.13542)。

#### 3D 检测，原论文 Table I–II

| 数据集 / Split | 方法 | 模态 | mAP | NDS / mAPH | 其他 |
| :--- | :--- | :---: | ---: | ---: | :--- |
| nuScenes test | BEVFusion | C+L | 70.2 | 72.9 | 253.2 G MACs；119.2 ms |
| nuScenes val | BEVFusion | C+L | 68.5 | 71.4 | 253.2 G MACs；119.2 ms |
| Waymo test L1 | BEVFusion† | C+L | 85.7 | 84.4 | 3 frames |
| Waymo test L2 | BEVFusion† | C+L | 80.8 | 79.5 | 3 frames |

`†`：原论文使用 test-time augmentation。

#### nuScenes 地图分割，原论文 Table III

| 方法 | 模态 | Drivable | Ped. Cross. | Walkway | Stop Line | Carpark | Divider | Mean IoU |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BEVFusion | C | 81.7 | 54.8 | 58.4 | 47.4 | 50.7 | 46.4 | 56.6 |
| BEVFusion | C+L | 85.5 | 60.5 | 67.6 | 52.0 | 57.0 | 53.7 | 62.7 |

#### 天气与光照，原论文 Table IV

| 方法 | 模态 | Sunny mAP/mIoU | Rainy mAP/mIoU | Day mAP/mIoU | Night mAP/mIoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| BEVFusion | C | — / 59.0 | — / 50.5 | — / 57.4 | — / 30.8 |
| BEVFusion | C+L | 68.2 / 65.6 | 69.9 / 55.9 | 68.5 / 63.1 | 42.8 / 43.6 |

### 【官方代码库结果】MIT-HAN BEVFusion

来源：[MIT-HAN BEVFusion 官方仓库](https://github.com/mit-han-lab/bevfusion)。

| 数据集 / Split | 输入 | mAP | NDS / mIoU |
| :--- | :---: | ---: | ---: |
| nuScenes val | Camera | 35.56 | 41.21 NDS |
| nuScenes val | LiDAR | 64.68 | 69.28 NDS |
| nuScenes val | Camera + LiDAR | 68.52 | 71.38 NDS |
| nuScenes val map segmentation | Camera + LiDAR | — | 62.95 mIoU |
| nuScenes test | Camera + LiDAR | 70.23 | 72.88 NDS |
| nuScenes test，BEVFusion-e | Camera + LiDAR | 74.99 | 76.09 NDS |

### 【论文原始结果】ADLab BEVFusion（NeurIPS 2022）

来源：[ADLab BEVFusion 原论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/43d2b7fbee8431f7cef0d0afed51c691-Paper-Conference.pdf)。

#### 泛化能力，原论文 Table 1

每格为 `mAP / NDS`。

| 输入 | PointPillars | CenterPoint | TransFusion-L |
| :--- | :---: | :---: | :---: |
| Camera | 22.9 / 31.1 | 27.1 / 32.1 | 22.7 / 26.1 |
| LiDAR | 35.1 / 49.8 | 57.1 / 65.4 | 64.9 / 69.9 |
| Camera + LiDAR | 53.5 / 60.4 | 64.2 / 68.0 | 67.9 / 71.0 |

#### nuScenes 主结果，原论文 Table 2

| Split | 方法 | mAP | NDS |
| :--- | :--- | ---: | ---: |
| val | BEVFusion | 67.9 | 71.0 |
| val | BEVFusion* | 69.6 | 72.1 |
| test | BEVFusion | 69.2 | 71.8 |
| test | BEVFusion* | 71.3 | 73.3 |

`*`：原论文使用 BEV-space data augmentation 训练。

#### LiDAR 有限视场，原论文 Table 3

每格为 `LiDAR-only mAP/NDS → BEVFusion mAP/NDS`。

| LiDAR FOV | PointPillars | CenterPoint | TransFusion-L | TransFusion LC |
| :---: | :---: | :---: | :---: | :---: |
| ±π/2 | 12.4/37.1 → 36.8/45.8 | 23.6/48.0 → 45.5/54.9 | 27.8/50.5 → 46.4/55.8 | 31.1/49.2 |
| ±π/3 | 8.4/34.3 → 33.5/42.1 | 15.9/43.5 → 40.9/49.9 | 19.0/45.3 → 41.5/50.8 | 21.0/41.2 |

#### LiDAR 目标点丢失，原论文 Table 4

每格为 `LiDAR-only mAP/NDS → BEVFusion mAP/NDS`；增强训练行的 LiDAR-only 值未报告。

| Robust Aug. | PointPillars | CenterPoint | TransFusion-L | TransFusion LC |
| :---: | :---: | :---: | :---: | :---: |
| × | 12.7/36.6 → 34.3/49.1 | 31.3/50.7 → 40.2/54.3 | 34.6/53.6 → 40.8/56.0 | 38.1/55.4 |
| ✓ | — → 41.6/51.9 | — → 54.0/61.6 | — → 50.3/57.6 | 37.2/51.1 |

#### Camera 故障，原论文 Table 5

每格为 `mAP / NDS`。

| 方法 | Clean | Missing Front | Preserve Front Only | 50% Frames Stuck |
| :--- | :---: | :---: | :---: | :---: |
| DETR3D | 34.9 / 43.4 | 25.8 / 39.2 | 3.3 / 20.5 | 17.3 / 32.3 |
| PointAugmenting | 46.9 / 55.6 | 42.4 / 53.0 | 31.6 / 46.5 | 42.1 / 52.8 |
| MVX-Net | 61.0 / 66.1 | 47.8 / 59.4 | 17.5 / 41.7 | 48.3 / 58.8 |
| TransFusion | 66.9 / 70.9 | 65.3 / 70.1 | 64.4 / 69.3 | 65.9 / 70.2 |
| BEVFusion | 67.9 / 71.0 | 65.9 / 70.7 | 65.1 / 69.9 | 66.2 / 70.3 |

#### Camera Stream 消融，原论文 Table 6

| BE | ADP | Large Backbone | mAP | NDS |
| :---: | :---: | :---: | ---: | ---: |
| × | × | × | 13.9 | 24.5 |
| ✓ | × | × | 17.9 | 27.0 |
| ✓ | ✓ | × | 18.0 | 27.1 |
| ✓ | ✓ | ✓ | 22.9 | 31.1 |

#### Dynamic Fusion 消融，原论文 Table 7

| CSF | AFS | PointPillars mAP/NDS | CenterPoint mAP/NDS | TransFusion mAP/NDS |
| :---: | :---: | :---: | :---: | :---: |
| × | × | 35.1 / 49.8 | 57.1 / 65.4 | 64.9 / 69.9 |
| ✓ | × | 51.6 / 57.4 | 63.0 / 67.4 | 67.3 / 70.5 |
| ✓ | ✓ | 53.5 / 60.4 | 64.2 / 68.0 | 67.9 / 71.0 |

### 【论文原始结果】How2comm（NeurIPS 2023）

来源：[How2comm 原论文 Table 1](https://papers.neurips.cc/paper_files/paper/2023/file/4f31327e046913c7238d5b671f5d820e-Paper-Conference.pdf)。DAIR-V2X LiDAR-only；100 ms 传输时延；`0.2 m / 0.2°` 定位与航向噪声；通信量不超过 1 MB。

| 方法 | 3D AP@0.5 | 3D AP@0.7 |
| :--- | ---: | ---: |
| No Fusion | 50.03 | 43.57 |
| Late Fusion | 48.93 | 34.06 |
| When2com | 46.64 | 32.49 |
| F-Cooper | 49.77 | 35.21 |
| AttFuse | 50.86 | 38.30 |
| V2VNet | 52.18 | 38.62 |
| DiscoNet | 51.44 | 40.01 |
| V2X-ViT | 51.68 | 39.97 |
| CoBEVT | 56.08 | 41.45 |
| Where2comm | 59.34 | 43.53 |
| How2comm | 62.36 | 47.18 |

### 【官方代码库结果】CoBEVT / OpenCOOD

来源：[OpenCOOD 官方结果](https://github.com/DerrickXuNu/OpenCOOD/tree/31ba16025da27ffe4e336f011290dfbc66f9a1f1#results-of-3d-detection-on-v2xset-lidar-track)。该表不是 CoBEVT 论文原表。

| 数据集 / 输入 | 设置 | AP@0.5 | AP@0.7 |
| :--- | :--- | ---: | ---: |
| V2XSet / LiDAR | Perfect | 84.9 | 66.0 |
| V2XSet / LiDAR | Noisy | 81.1 | 54.3 |

## Reproduction

- [ResilientV2X 配置](configs/resilient_v2x/README.md)
- [复现实验指南](docs/resilient_v2x/reproduction.md)
- [论文—代码覆盖矩阵](docs/resilient_v2x/paper-coverage.md)
- [旧稿结果归档](docs/resilient_v2x/archive/old-draft-results.md)

## Reference

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.3.0)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X), legacy Transvision reference commit: `c65a55617f7d0a9b78dc9d107370c95bcac55dca`
- [FFNet-VIC3D](https://github.com/haibao-yu/FFNet-VIC3D), audited commit: `52164dfe00764c9a9925539e99689cf25b88eace`
