# 公开方法原生评价：已验数值与协议边界

本表直接引用已独立验收的原生输出与指标组件，未重跑预测或评价。三个数值行分别使用各自数据集、类别和作者评价协议；不能据此跨协议排序，也不能作为同资源比较。

## DMSTrack：V2V 官方 test merged vehicle

| 帧／序列 | AMOTA ↑ | sAMOTA ↑ | AMOTP：平均 3D IoU ↑ |
| --- | --- | --- | --- |
| 1993／9 | 0.435158662 | 0.915041185 | 0.579422891 |

## CoopTrack：SPD 原生评价

| 类别范围 | 帧／序列 | AMOTA ↑ | AMOTP（米）↓ | IDS | Frag |
| --- | --- | --- | --- | --- | --- |
| car | 3316／21 | 0.349177253 | 1.32444318 | 261 | 288 |
| 三类别汇总 | 3316／21 | 0.125340967 | 1.69858076 | 341 | 387 |

DMSTrack 的 AMOTP 是平均 3D IoU；CoopTrack 的 AMOTP 是距离。不能将两者当作同一单位、同一方向的误差。CoopTrack 三类别汇总不能替代 car 单类行。

## 完成边界

- DMSTrack：原 GPU Task `6c484a6eade04f2b9f920a026d78700c`；1993 帧、9 序列，42769 条预测、31419 个 GT 对象。37 个实际召回阈值行按作者除数 40 汇总；两套独立指标实现一致，原冻结容差 1e-12。V2V 已知 train/test 内容重叠继续披露；时间为标称 10Hz。
- CoopTrack：原 Task `9258703e4d5d4bc38652875eb37ac9c8`；3316 帧、21 序列。完整输出的几何、检测与跟踪组件及作者 127 个源码文件、实际 coop914／INF817 个加载状态张量身份已有验收。原生逆时间插值保留；TID/LGD 使用作者每帧 0.5 秒设置，该设置不推导为输入采样频率。
- 两方法的正式隔离资源测量尚缺证据。各原生执行合同还须逐项绑定 split、类别、ROI、频率、特征及环境。本表汇总所列组件，未新增这些协议要素的独立验收。完整原生资格与最终论文表仍未全部完成。

## 尚无合格完整数值行的方法

- SparseCoop：已有三套原生初始化和完整 train 的 2700 个有序 anchor 中心核对；尚无已验训练 checkpoint 与完整前向，不能用初始化结果充当复现指标。原生 label-cleaned 2Hz 范围应单列；后续多卡执行限 A100，保留原 8 卡配方。
- Long-SCOPE：最近有界作者资产搜索尚未取得合格实现和权重。已绑定作者论文的原生配置，但不填写文献指标作为本次实验结果；后续多卡执行限 A100，保留 4 卡数量。

## 追溯

- [DMSTrack 原验收组件](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-public-author-native-asset-acquisition-20261006T214405929601Z/original-full-native-GPU-complete-owned-path-bindings-20261007T083754739520Z/independent-complete-original-GPU-two-evaluator-component-index-20261007T091726637173Z/independent-complete-original-GPU-native-execution-two-metric-and-full-byte-component-index.json)；SHA256 `62f653b48d732dacd9dee02903d31e6403d81300dcb9c3c417dad52984b3ed8f`。
- [CoopTrack 原验收组件](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-CoopTrack-exact-mmdetection3d-native-source-20261006T232725831265Z/independent-author-release-to-actual-cooperative-and-INF-execution-qualification-20261007T092640708392Z/complete-cooperative-native-numeric-and-author-release-identity-component-index-v2.json)；SHA256 `9cd8d9933c600d9e60e75ea846cc66e12632f584b152e41f5e9d70b9e404287f`。
- [完整原生数值与来源](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-native-public-method-protocol-table-20261007T160941218739Z/original-native-table-values.json)。


2026-10-08 08:50 北京时间重新读取已绑定作者 wang-jh18-SVM 的公开 owned repository API：单页 56 仓库，未命中 Long-SCOPE 名称/描述/homepage 候选。范围仅此公开作者账户，不能据此推断其他位置不存在代码。arXiv 当前 submission history 仍列 v1（2026-04-10），CVF 官方页本次直接读取返回 403；搜索索引页不作为成功实时获取官方页面字节。[此次资产检查记录](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-Long-SCOPE-current-public-owned-assets-check-20261008T005040215690Z/public-asset-check-receipt.json)。没有新增合格作者实现/权重，不填文献指标为实验值；J 仍未完成。
