# train20 标签与源时钟准入审计

2026-09-17，前置双车坐标对齐任务 `01c9989cc5f24931a16a0605e6f164f9` completed 后，执行独立训练标签/元数据检查。该步骤是训练准入审计，不是性能实验或参数训练；纯 CPU 执行，未占用 A100。

## 任务与复现

- ClearML：`a1e99f9f4eaf4208b32b7210fb58c57a`，实时复核 **completed**。
- 入口：`work_dirs/rbf-pointpillar-runtime-20260917/tools/event_track_v2x/audit_v2v4real_train20_readiness.py`。
- 入口 SHA-256：`fcc1e3db319e36bbeb1f3cff45506f52b7f3570e3f01ea1943b30b0f9f358e16`。
- 部署目录：`10.100.35.112:/home/lbin/Desktop/rbf-readiness-train20-20260917`。
- 执行：部署目录内设置 `CLEARML_CACHE_DIR=/home/lbin/Desktop/rbf-next-experiment-cache`，运行 `/home/lbin/miniconda3/bin/python tools/event_track_v2x/audit_v2v4real_train20_readiness.py`。任务按源码哈希去重；源码及依赖哈希保存在任务资产/参数中。
- 所有数据均来自 ClearML：固定注册表 `14675561902041d7ab77fa2adbf2aeb0` 的 train_04，及前置候选任务 `e59d0f4d522b4ad8940d41cf3d462449`。实际字节均经固定 SHA-256 检查。未读取 official_test。

## 回读结果

admission-report、metadata-clock-inventory、train20-strict-car-gt 均已从 ClearML 回读，数量相互一致：

| 项目 | 观测结果 |
|---|---:|
| 配对帧 / 源元数据 | 20 / 40 |
| 严格 Car GT 实例 | 295 |
| 相邻帧共享原生 ID 次数之和 | 271 |
| 原始两源 Car / ConcreteTruck 标注条目 | 404 / 19 |
| 同帧重复原生 ID 次数 | 54 |
| 重复 ID 对应框中心最大距离 | 2.569344997406006 m |
| 含 time/stamp/clock 候选字段的源元数据 | 0 / 40 |

标签使用现有 `v2v4real-real-matrix-late-gt-strict-car-first-id-two-stage-roi-v1`，明确 ego CAV 0、严格 Car、两阶段 ROI 与先出现 ID 去重。上述数量不是检测召回率、跟踪指标或身份正确率。重复 ID 几何差异只作诊断，不据此修复标注或选择模型。

首帧顶层元数据键为 ego_speed、gps、lidar_pose、true_ego_pos、vehicles。审计扫描非对象注释字典的时间相关字段名，没有发现候选源时钟；这不证明其他分卷、其他文件或上游采集记录也没有时钟。帧号保持 ordinal-only-no-clock，不按假定频率造时间戳。

## 准入结论

标签可以通过现有原生适配器准备，物理身份正确性未验证。正式时序训练尚未获准：源时间单位/传感器与到达时间语义未确认，公开 detector 选模来源未核实，正式原生缓存和训练会话划分未冻结。formal_temporal_training_ready=false，paper_eligible=false。

GT 只进入此独立 train 准备/审计任务，未修改已有预测、候选或特征。下一步应先核实官方时钟/同步协议与 train 会话分组，不能将此次标签导出当作训练完成。

## 验证范围

本地入口、位姿与原生输入相关回归 84 passed；完整 ground_truth 测试在本地因缺少 Torch 于收集阶段失败，未报告为通过。管理机具备 Torch，实际 20 配对帧原生 GT 准备成功并经 ClearML 回读验收。保留本地失败事实；代码未提交或推送 Git。
