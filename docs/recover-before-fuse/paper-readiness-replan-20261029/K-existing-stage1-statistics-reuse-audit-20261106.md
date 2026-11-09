# K项：已有Stage1统计复用边界

本次仅文本审查现有源码与原收据，没有在Mac计算SHA、重算统计或运行指标。因此下列为已有证据内容核对，不是新全字节资格。

原收据：`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-stage1-nuscenes-complete-shared-bootstrap-after-admission-20261006T181425503427Z/independent-statistics-output-readback.json`。

该独立收据记录2026-10-06T18:52:03Z完成13×10000×15完整draw数组读回、345个标量及105个区间检查，原固定绝对容差1e-12，最大差2.842170943040401e-14，非有限draw为0。原producer收据SHA为3def79e22a108c8dbe275f433d332e616f66532e7736a162b65ba7b91592b176，draw数组SHA为02cd4dd172176b74f94694f76bf062c90d3c20423a194d54e41d2d9eda799474。

原TrackEval序列统计收据位于`artifacts/rbf-stage1-all-thirteen-full-stream-provenance-20261006T170240089706Z/independent-sequence-statistics/statistics-receipt.json`，范围为原21序列3316帧、13个运行；包含HOTA/AssA/DetA/IDF1序列统计。

现有入口可复用：`tools/event_track_v2x/bootstrap_stage1_nuscenes.py`、`verify_stage1_nuscenes_bootstrap_summary.py`、`plot_stage1_qualified_paired_effects.py`。原协议保留PCG64 seed1337、10000次整序列共享抽样；原run轴由campaign结果确定，不能用JSON排序键冒充数组轴。M1仅一次运行，样本标准差为null；M3-seed2027严格复现失败，M3没有合格推断区间。nuScenes距离AMOTP是米且越低越好，不是V2V原生IoU AMOTP。

无需因K项整体未完成就从头重跑这组已完成统计。正式复用前按既有来源授权在Linux核对原登记与保存SHA；本次不自动跨机传输或重新执行。

仍缺完整K：最终learned/恢复消融/主表配置与两个数据集的合格预测、对应逐序列指标与三种子统计、正式同资源成本、完整真实错误持续/因果链和最终论文表图。该原收据明确`full_K_or_paper_acceptance=false`，不能将Stage1统计通过扩大为全部论文通过。保留负结果与探索性SPD已见val边界。
