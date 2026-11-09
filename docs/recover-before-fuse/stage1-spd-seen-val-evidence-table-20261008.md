# SPD 已见 val：M0–M4 机制证据表

范围：13 个原始逻辑配置，每配置 21 序列、3316 帧。以下数值直接取自已独立核对的原回执；本次仅排表，未运行模型、评价器或 bootstrap。

这是已见 val 的探索性机制结果。M3 seed2027 保留严格复放失败；M3 三种子的均值与标准差仅作原始描述，不进入合格区间或配对推断。M1 只有一个确定性配置，标准差为 NA。

HOTA、AssA、DetA、IDF1、AMOTA 均保留 0–1 原始量纲；AMOTP 为中心距离（米），越小越好。FP、FN、IDS、Frag 为计数。均值 ± 样本标准差描述原种子差异；置信区间来自固定 1337、10000 次共享序列级 cluster bootstrap，不表示未知数据泛化或种子重采样不确定性。

## 原配置均值与样本标准差

| 配置 | 严格复放 | HOTA | AssA | DetA | IDF1 |
| --- | --- | --- | --- | --- | --- |
| M0 | 通过 | 0.201841 ± 0.000559433 | 0.279589 ± 0.00162141 | 0.149549 ± 6.04745e-05 | 0.17621 ± 0.000411135 |
| M1 | 通过 | 0.197858 ± NA | 0.276902 ± NA | 0.14506 ± NA | 0.171828 ± NA |
| M2 | 通过 | 0.201686 ± 0.000821265 | 0.280895 ± 0.00192516 | 0.148568 ± 0.00015243 | 0.175557 ± 0.00112629 |
| M3 | 仅原始描述：2027 失败 | 0.196463 ± 0.000439517 | 0.277307 ± 0.000670947 | 0.142631 ± 0.000369598 | 0.171061 ± 0.000862304 |
| M4 | 通过 | 0.166462 ± 0.00134933 | 0.237258 ± 0.0021834 | 0.12063 ± 0.00226598 | 0.132087 ± 0.00317224 |

| 配置 | 严格复放 | amota | amotp | fp | fn | ids | frag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 通过 | 0.33138 ± 0.00796337 | 1.31165 ± 0.000151102 | 5784 ± 206.371 | 13860.7 ± 47.4377 | 83.3333 ± 14.0475 | 341 ± 8.18535 |
| M1 | 通过 | 0.326844 ± NA | 1.30889 ± NA | 4736 ± NA | 14740 ± NA | 71 ± NA | 325 ± NA |
| M2 | 通过 | 0.328325 ± 0.00299262 | 1.31099 ± 0.00594928 | 5439.33 ± 80.8538 | 14156 ± 314.204 | 88 ± 6 | 334 ± 9.53939 |
| M3 | 仅原始描述：2027 失败 | 0.301966 ± 0.00479286 | 1.31436 ± 0.00172839 | 4417.67 ± 1213.57 | 15802.7 ± 1064.33 | 66.6667 ± 16.2891 | 288.667 ± 38.371 |
| M4 | 通过 | 0.260395 ± 0.00330959 | 1.30972 ± 0.000337312 | 4576.67 ± 248.888 | 15740.7 ± 321.872 | 74.6667 ± 2.08167 | 281.333 ± 7.37111 |

## 原始逐配置结果

| 原配置 | 严格复放 | HOTA | AssA | DetA | IDF1 |
| --- | --- | --- | --- | --- | --- |
| M0-seed-1337 | 通过 | 0.20131 | 0.278249 | 0.149544 | 0.17619 |
| M0-seed-2027 | 通过 | 0.202425 | 0.281391 | 0.149491 | 0.175809 |
| M0-seed-3407 | 通过 | 0.201786 | 0.279125 | 0.149611 | 0.17663 |
| M1-all-unmatched | 通过 | 0.197858 | 0.276902 | 0.14506 | 0.171828 |
| M2-seed-1337 | 通过 | 0.201546 | 0.280513 | 0.148613 | 0.176831 |
| M2-seed-2027 | 通过 | 0.202568 | 0.282982 | 0.148693 | 0.175149 |
| M2-seed-3407 | 通过 | 0.200944 | 0.279189 | 0.148398 | 0.174692 |
| M3-seed-1337 | 通过 | 0.195959 | 0.276742 | 0.142275 | 0.170075 |
| M3-seed-2027 | 失败，保留原值 | 0.19677 | 0.278049 | 0.142606 | 0.171433 |
| M3-seed-3407 | 通过 | 0.196659 | 0.277131 | 0.143013 | 0.171674 |
| M4-seed-1337 | 通过 | 0.165109 | 0.235862 | 0.119304 | 0.128917 |
| M4-seed-2027 | 通过 | 0.166471 | 0.239774 | 0.119339 | 0.132084 |
| M4-seed-3407 | 通过 | 0.167807 | 0.236138 | 0.123246 | 0.135262 |

| 原配置 | 严格复放 | amota | amotp | fp | fn | ids | frag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0-seed-1337 | 通过 | 0.335537 | 1.3117 | 5924 | 13806 | 98 | 334 |
| M0-seed-2027 | 通过 | 0.336405 | 1.31148 | 5547 | 13885 | 70 | 350 |
| M0-seed-3407 | 通过 | 0.322199 | 1.31176 | 5881 | 13891 | 82 | 339 |
| M1-all-unmatched | 通过 | 0.326844 | 1.30889 | 4736 | 14740 | 71 | 325 |
| M2-seed-1337 | 通过 | 0.331521 | 1.30891 | 5346 | 14386 | 82 | 324 |
| M2-seed-2027 | 通过 | 0.327865 | 1.30637 | 5488 | 13798 | 94 | 343 |
| M2-seed-3407 | 通过 | 0.325588 | 1.3177 | 5484 | 14284 | 88 | 335 |
| M3-seed-1337 | 通过 | 0.306551 | 1.31308 | 3023 | 17028 | 48 | 245 |
| M3-seed-2027 | 失败，保留原值 | 0.29699 | 1.31366 | 5233 | 15108 | 74 | 317 |
| M3-seed-3407 | 通过 | 0.302358 | 1.31632 | 4997 | 15272 | 78 | 304 |
| M4-seed-1337 | 通过 | 0.256583 | 1.30953 | 4312 | 16110 | 73 | 273 |
| M4-seed-2027 | 通过 | 0.262538 | 1.30951 | 4612 | 15520 | 74 | 287 |
| M4-seed-3407 | 通过 | 0.262063 | 1.31011 | 4806 | 15592 | 77 | 284 |

## 合格配对差异：候选减 M0 的 95% 区间

只展示原回执准入的 M1、M2、M4 对比；没有为失败的 M3 补造区间。原始全集区间与其余指标保存在数值 JSON。

| 配对 | 指标 | 95% 区间 |
| --- | --- | --- |
| M1-minus-M0 | HOTA | [-0.00670126, -0.00145227] |
| M1-minus-M0 | AssA | [-0.00905249, 0.00395597] |
| M1-minus-M0 | DetA | [-0.00659915, -0.00279092] |
| M1-minus-M0 | IDF1 | [-0.00908369, -0.000533065] |
| M1-minus-M0 | amota | [-0.0218455, 0.0106929] |
| M1-minus-M0 | amotp | [-0.0284759, 0.00325803] |
| M1-minus-M0 | fp | [-2390.09, 772.342] |
| M1-minus-M0 | fn | [-831.675, 2176.75] |
| M1-minus-M0 | ids | [-50.6667, 33.6667] |
| M1-minus-M0 | frag | [-75.6667, 51.0083] |
| M2-minus-M0 | HOTA | [-0.00265338, 0.00236767] |
| M2-minus-M0 | AssA | [-0.00539725, 0.00806297] |
| M2-minus-M0 | DetA | [-0.00158603, -0.000449457] |
| M2-minus-M0 | IDF1 | [-0.0034985, 0.00257374] |
| M2-minus-M0 | amota | [-0.0134935, 0.0100038] |
| M2-minus-M0 | amotp | [-0.0075282, 0.0191885] |
| M2-minus-M0 | fp | [-892.683, 1029.69] |
| M2-minus-M0 | fn | [-1105.69, 822.342] |
| M2-minus-M0 | ids | [-17.675, 41.3333] |
| M2-minus-M0 | frag | [-35.3333, 52.3333] |
| M4-minus-M0 | HOTA | [-0.0415282, -0.0297142] |
| M4-minus-M0 | AssA | [-0.0523485, -0.0328923] |
| M4-minus-M0 | DetA | [-0.0344708, -0.0243578] |
| M4-minus-M0 | IDF1 | [-0.0532043, -0.0360867] |
| M4-minus-M0 | amota | [-0.0963959, -0.0419952] |
| M4-minus-M0 | amotp | [-0.0202976, 0.00965846] |
| M4-minus-M0 | fp | [-2188.02, 307.042] |
| M4-minus-M0 | fn | [361.325, 2904.35] |
| M4-minus-M0 | ids | [-30.3333, 30.6667] |
| M4-minus-M0 | frag | [-111.667, -7] |

M4 的 AMOTA 区间完全为负，FN 区间完全为正，同时 Frag 区间完全为负：碎片减少伴随整体质量和漏检代价。该结果不支持据此声称恢复机制已有独立优势。M2 的 AMOTA 区间跨零，仍须保留混合结果。

尚未由本表完成的要求：物理身份误配与传播／恢复的因果证据、同资源比较、两数据集完整方法主表和最终论文结论。

## 追溯

- [track](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-all-thirteen-full-stream-provenance-20261006T170240089706Z/independent-sequence-statistics/statistics-receipt.json)；SHA256 `c4115d1eacc5020e085956a305b774ef709af115f8f7a7cd7311af35a316a0a7`。
- [native](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-nuscenes-complete-shared-bootstrap-after-admission-20261006T181425503427Z/actual-full-bootstrap/bootstrap-receipt.json)；SHA256 `3def79e22a108c8dbe275f433d332e616f66532e7736a162b65ba7b91592b176`。
- [native_readback](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-nuscenes-complete-shared-bootstrap-after-admission-20261006T181425503427Z/independent-statistics-output-readback.json)；SHA256 `13d112cc76e5cc307d21c9e83f99f7b628091b8ef9aaddaa3694efe954ff13d8`。
- [完整原数值及来源](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-SPD-seen-val-paper-table-20261007T160448079067Z/all-original-accepted-values.json)。


## 输出分数与逐帧 TP 的可靠性诊断（2026-10-08 07:41 CST）

以下逐配置原值已完成生产归档和独立审计的完整 SHA 回读。每配置为 21 序列、3316 帧；原始 car 输出分数 p 对应同帧 native ego 严格 50m ROI 内、BEV IoU ≥ 0.5 的一对一匹配标签 y。匹配最大化有效匹配数量，再最大化 IoU 总和，不使用分数排序。NLL/Brier/ECE 条件于已经输出的预测；另列 TP/FP/FN/recall，避免将低输出量解释为整体质量提升。它不是联合身份后验、物理身份风险、质量界证书或 TV 证明；未用 val 拟合校准或选择参数。SPD 仍为已见 val 探索性评价。

生产 Task `8cd87a77eb4f48408f449793bac7c58c` 与独立审计 Task `7f4d464d97974b57810cc56b0fab41c2` 均 completed，实际 CPU worker `10.100.35.121-L40S:gpu3,4,5`。独立审计逐条回读标签/原帧，并以交替环残差图检查匹配最优性；全部统计重算与生产原值最大绝对差为 0，冻结容差仍为 1e-10。这是浮点诊断验收，不是形式证明。

| 配置 | NLL | Brier | ECE | TP | FP | FN | recall | 严格复跑资格 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| M0-seed-1337 | 1.59606674 | 0.393342952 | 0.455149327 | 8378 | 41119 | 18270 | 0.314395077 | 通过 |
| M0-seed-2027 | 1.59147822 | 0.393401308 | 0.455375608 | 8419 | 41357 | 18229 | 0.315933654 | 通过 |
| M0-seed-3407 | 1.59182669 | 0.392592645 | 0.454982113 | 8394 | 41604 | 18254 | 0.314995497 | 通过 |
| M1-all-unmatched | 1.59403764 | 0.397248968 | 0.460521915 | 8753 | 43771 | 17895 | 0.328467427 | 通过 |
| M2-seed-1337 | 1.59491509 | 0.392409072 | 0.453071954 | 8291 | 41213 | 18357 | 0.311130291 | 通过 |
| M2-seed-2027 | 1.59051033 | 0.392314821 | 0.453054448 | 8350 | 41394 | 18298 | 0.313344341 | 通过 |
| M2-seed-3407 | 1.59408126 | 0.392875623 | 0.453592933 | 8336 | 41631 | 18312 | 0.312818973 | 通过 |
| M3-seed-1337 | 1.65906839 | 0.406287326 | 0.467183732 | 8600 | 44317 | 18048 | 0.322725908 | 通过 |
| M3-seed-2027 | 1.65638667 | 0.405772664 | 0.466613031 | 8620 | 44215 | 18028 | 0.323476434 | 历史失败，仅保留 raw |
| M3-seed-3407 | 1.65107821 | 0.404866928 | 0.46610543 | 8646 | 44217 | 18002 | 0.324452116 | 通过 |
| M4-seed-1337 | 1.392296 | 0.347744838 | 0.412925591 | 9080 | 58410 | 17568 | 0.340738517 | 通过 |
| M4-seed-2027 | 1.38521859 | 0.346696624 | 0.412690455 | 9094 | 58683 | 17554 | 0.341263885 | 通过 |
| M4-seed-3407 | 1.41867915 | 0.354369632 | 0.419648918 | 9023 | 56018 | 17625 | 0.33859952 | 通过 |

全部原配置 NLL 状态均为 finite，未裁剪分数。M3 seed2027 的完整标签可进行 raw 诊断，但其历史严格复跑失败继续禁止合格三种子汇总和配对比较。10000 次整个序列配对统计及独立复算正在新的 L40 Task `37bb509b7bfd41ee901aeaf1f33ce1cb` 运行；本节尚不填该任务的均值、样本标准差或区间。

- [原始完整分数诊断回执](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-stage1-score-native-qualified-ROI-population-20261007T231724884312Z/complete-results-full-SDK-readback/score-reliability-receipt.json)；SHA256 `4802fee909008b6e389b526ea5290cca56cad7df7abfdcaebff8875331807d40`。
- [独立全标签与统计验收](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-score-independent-complete-label-audit-bootstrap-20261007T232516395126Z/complete-independent-audit-acceptance.json)；SHA256 `03e4738737873be211c51e78846d84dfb6ace168b7dd3eba2a31a4d40b7d4344`。

该组件不完成 E 的物理误配传播链、K 的全部方法/数据集统计、I 的同资源成本或最终论文。


## 输出分数诊断的已验配对统计（2026-10-08 07:44 CST）

本节替代上一节“统计任务运行中”的历史状态。Task `37bb509b7bfd41ee901aeaf1f33ce1cb` 已 completed，实际 CPU worker `10.100.35.121-L40S:gpu3,4,5`。结果归档和全部 5 个成员 SDK 全字节 SHA 回读通过。不同源码的独立复算器从已经审计的逐序列标量/分箱统计出发，展开全部共享重采样序列并重算原值、均值/样本标准差和区间，最大绝对差 `2.220446049250313e-16`，低于冻结 `1e-10`。独立性范围是源码与数值算术；生产统计和复算器在同一个隔离 child 内顺序执行，不声称独立进程或环境复现。

按整个原生序列有放回抽样，固定 seed1337，10000 次，每次 21 序列；所有方法/种子共享 multiplicities。该分数诊断使用明确冻结的 Python random MT19937；先前 tracking 表 PCG64 的 draw 文件仍保留，不将两种抽样声明为同一 draw。每次重新合并 NLL/Brier 充分统计量和 ECE 分箱计数，ECE 不平均逐序列 ECE。先在每个相同 draw 合并序列，再在原种子匹配的完整方法内求种子均值；区间是 2.5%/97.5% percentile 线性插值。M1 为单个确定性对照，不构造假的三种子标准差；M3 不填合格三种子汇总。

| 方法 | n | NLL 均值 ± 样本 SD | Brier 均值 ± 样本 SD | ECE 均值 ± 样本 SD | 逐帧 recall 均值 ± 样本 SD |
|---|---:|---:|---:|---:|---:|
| M0 | 3 | 1.59312388 ± 0.00255453499 | 0.393112301 ± 0.000450980724 | 0.455169016 ± 0.000197485066 | 0.315108076 ± 0.000775441996 |
| M1 | 1 | 1.59403764（SD 不适用） | 0.397248968（SD 不适用） | 0.460521915（SD 不适用） | 0.328467427（SD 不适用） |
| M2 | 3 | 1.59316889 ± 0.00233982977 | 0.392533172 ± 0.000300292363 | 0.453239778 ± 0.000305966003 | 0.312431202 ± 0.00115684017 |
| M4 | 3 | 1.39873125 ± 0.0176341032 | 0.349603698 ± 0.00416056259 | 0.415088321 ± 0.0039513417 | 0.34020064 ± 0.00141127377 |
| M3 | — | 不合格三种子；raw 值保留 | — | — | — |

| 相对 M0 的方法差值 | 指标 | 95% 序列级配对区间 |
|---|---|---:|
| M1-minus-M0 | NLL | [-0.0160544535, 0.0143033453] |
| M1-minus-M0 | Brier | [0.000839153152, 0.00697374625] |
| M1-minus-M0 | ECE | [0.00106182147, 0.00899393611] |
| M1-minus-M0 | recall | [0.00753367884, 0.0194351437] |
| M2-minus-M0 | NLL | [-0.00677360274, 0.00725156016] |
| M2-minus-M0 | Brier | [-0.00226533054, 0.00126388535] |
| M2-minus-M0 | ECE | [-0.0038666315, -3.5979059e-05] |
| M2-minus-M0 | recall | [-0.00462783341, -0.000759259943] |
| M4-minus-M0 | NLL | [-0.25734942, -0.136465691] |
| M4-minus-M0 | Brier | [-0.0564418423, -0.0308335543] |
| M4-minus-M0 | ECE | [-0.0530983702, -0.0268617664] |
| M4-minus-M0 | recall | [0.0187793562, 0.0310478071] |

M4 的输出分数诊断 NLL/Brier/ECE 差值区间均低于零，逐帧 IoU recall 差值为正；此前原生 tracking 的 HOTA/AMOTA 和 FN 负结果仍成立，不能将分数诊断的改善当作整体跟踪质量或恢复独立收益。此处的逐帧 IoU TP/FN/recall 与原生 tracking 多阈值 AMOTA、轨迹身份和 FN 统计属于不同评价口径，不能互相替换。M2 的 ECE 差值区间低于零，但 NLL/Brier 区间跨零、逐帧 recall 差值区间低于零，保留混合结果。NLL/Brier/ECE 只评价已经输出预测的原始 heuristic score；没有校准拟合或真实身份风险证明。

- [完整统计原值与区间](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-emitted-score-paired-sequence-statistics-20261007T233402423779Z/complete-statistics-full-SDK-readback/score-statistics-receipt.json)；SHA256 `403562d9a4537cdfeaf31f3dd30cc641bd15506d1271eb31c0583b1882b308e4`。
- [独立统计复算验收](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-emitted-score-paired-sequence-statistics-20261007T233402423779Z/complete-statistics-independent-acceptance.json)；SHA256 `0d083dbba3a3d47feb0203fa32b7d917d4df1a8b3be5e059732a4dde7340c3c2`。
- [全部共享序列 multiplicities](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-emitted-score-paired-sequence-statistics-20261007T233402423779Z/complete-statistics-full-SDK-readback/shared-sequence-multiplicities.json)；SHA256 `7609a39002bfbf00ed93301d856ae1e3d15539e92d958c08ff1c8f927da14a56`。

这完成当前 13 配置分数诊断统计组件，仍不完成完整 E/K、其余方法/数据集主表、同资源测量或论文最终收口。


## 已验输出分数诊断图（2026-10-08）

原始 13 个运行的 NLL/Brier/ECE/逐帧 IoU recall 散点图，以及 12 个已验配对 bootstrap 95% 区间，均直接复制统计 receipt 原值。未重算科学指标或 bootstrap；PNG 已目视检查，SVG 已导出但未独立渲染检查。M3 seed2027 历史严格失败保留，M3 不进入合格三种子配对比较。两份已验 receipt 有 3 个末位浮点差异，最大 5.551115123125783e-17；保留 v1 过严逐字节一致检查失败，v2 沿用原独立统计 1e-10 容差核对呈现一致性，不改变科学验收规则。

- [emitted-score-13-original-runs.png](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-accepted-score-diagnostic-paper-figures-20261008-v2/emitted-score-13-original-runs.png)
- [paired-score-difference-95-percentile-intervals.png](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-accepted-score-diagnostic-paper-figures-20261008-v2/paired-score-difference-95-percentile-intervals.png)
- [emitted-score-13-original-runs.svg](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-accepted-score-diagnostic-paper-figures-20261008-v2/emitted-score-13-original-runs.svg)
- [paired-score-difference-95-percentile-intervals.svg](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-accepted-score-diagnostic-paper-figures-20261008-v2/paired-score-difference-95-percentile-intervals.svg)
- [captions.txt](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-accepted-score-diagnostic-paper-figures-20261008-v2/captions.txt)
- [render-source-binding-receipt.json](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-accepted-score-diagnostic-paper-figures-20261008-v2/render-source-binding-receipt.json)
- [visual-review-receipt.json](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-accepted-score-diagnostic-paper-figures-20261008-v2/visual-review-receipt.json)

这些是已验分数诊断的论文呈现组件；原生 tracking 负结果继续保留，不证明整体跟踪收益、真实身份风险或恢复独立价值，不完成完整 K。

图与来源证据已经单独 byte-only 注册，Task `78cc30b1a46e4651bdef8dd579cade47` 后端 completed；完整 SDK 读回 archive 及 12 个成员 size/SHA 均匹配。[注册读回记录](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-accepted-score-diagnostic-paper-figures-20261008-v2/figure-registered-full-SDK-byte-readback.json)。这不改变原科学统计 Task，也不将整个 E/K 升级为完成。


2026-10-08 补充：两份原 SVG 已使用独立于 Matplotlib 的 sharp/libvips/librsvg 渲染，并目视检查四面板、坐标、图例、区间端点、零线与范围说明；所检查显示分辨率下无可见裁切。原 SVG SHA 与已注册文件一致，原注册包及此前仅 PNG 检查记录保留不变。[独立 SVG 检查记录](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-SPD-score-figures-independent-SVG-review-20261008-v1/visual-review-receipt.json)。未重算任何科学指标，不改变完整 E/K 未完成状态。


## 错误持续时间的观测边界（2026-10-19 证据整理）

已有封存片段回执的算术核对如下；本次没有重新运行追踪器、匹配器或读取全部原始片段。M3 seed2027 的失败结果单列，不纳入合格配置汇总。

| 配置 | 可观测错误片段 | 右删失片段 | 观测到正确状态结束 | 严格复放 |
| --- | ---: | ---: | ---: | --- |
| M0-seed-1337 | 205 | 204 | 1 | 通过 |
| M0-seed-2027 | 209 | 206 | 3 | 通过 |
| M0-seed-3407 | 195 | 193 | 2 | 通过 |
| M1-all-unmatched | 219 | 218 | 1 | 通过 |
| M2-seed-1337 | 196 | 195 | 1 | 通过 |
| M2-seed-2027 | 208 | 206 | 2 | 通过 |
| M2-seed-3407 | 212 | 210 | 2 | 通过 |
| M3-seed-1337 | 212 | 209 | 3 | 通过 |
| M3-seed-2027 | 221 | 213 | 8 | 失败，仅保留记录 |
| M3-seed-3407 | 195 | 193 | 2 | 通过 |
| M4-seed-1337 | 201 | 198 | 3 | 通过 |
| M4-seed-2027 | 193 | 189 | 4 | 通过 |
| M4-seed-3407 | 188 | 184 | 4 | 通过 |

12 个严格复放通过配置共记录 2433 个片段，其中 2405 个右删失（98.85%）。该汇总仅描述这些配置的片段记录，同一场景会跨配置重复出现，不能当成独立样本比例或恢复成功率。

失去几何对应、对应歧义、GT 不在 ROI、未分配、身份归属改变或序列结束均不等于错误得到修复。右删失片段的观测跨度不能当成完整错误持续时间；不得据此直接比较平均修复时间或声称恢复缩短了错误传播。已有片段也没有证明物理源身份及干预的因果作用。

因此，本表支持已见 SPD val 的探索性指标与可观测片段描述，仍不足以完成第一阶段的物理误配—融合—birth/kill—碎片因果链要求。M3 完整受控复跑通过后也不能自动消除这项缺口。

来源：[封存片段合同绑定回读](/Users/lbin/transvision/docs/recover-before-fuse/stage1-observed-error-censoring-contract-bound-readback-v2-20261012.json)；SHA256 `b2b658becc2850669494d2c8964a1e6e2a09909d95adb855ae2b6ea6841c105c`。


## M3 seed2027 完整受控重跑（2026-10-21）

本次按原计划执行全部 21 序列、3316 帧，回放耗时 136.559743 秒，生产进程正常结束。严格逐字节比较仍失败：关联输出与原结果相同，预测和诊断输出不同。完整字段比较仅发现数值及其派生哈希差异；这不能确定历史失败根因，也不能替代严格复现通过。

对本次新预测使用冻结评价器独立评价后，全部 21 序列的 HOTA、AssA、DetA、IDF1 与原结果逐值相同；总体 AMOTA、MOTA、FP、FN、IDS、Frag 相同。AMOTP 差值（新减原）为 -2.0851542714694915e-11 米。此处是 SPD 已见 val car-only 探索性评价的中心距离 AMOTP，不能套用 V2V 原生 AMOTP 的单位和方向。

本次结果单列为受控重跑的指标敏感性证据，不替换原始 13 配置数据，不加入严格通过的三种子汇总或配对统计。此前错误片段、删失与因果识别边界继续有效；没有由指标一致推导恢复独立收益、完整 E/K 验收或历史根因。没有放宽容差或追加重跑。

来源：[本次独立评价与外置盘 SHA 读回](/Users/lbin/transvision/docs/recover-before-fuse/M3-independent-metric-readback-20261021.json)、[预测全字段差异](/Users/lbin/transvision/docs/recover-before-fuse/M3-full-prediction-difference-20261021.json)、[诊断全字段差异](/Users/lbin/transvision/docs/recover-before-fuse/M3-diagnostic-field-comparison-20261021.json)。
