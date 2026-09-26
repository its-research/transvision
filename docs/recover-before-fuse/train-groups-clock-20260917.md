# 完整 train 名称分组与时钟依据核验

ClearML task `45fbaf44ff534eeca41f52d49a880676`，CPU 元数据审计，不是训练或性能实验。读取固定注册表 `14675561902041d7ab77fa2adbf2aeb0` 的 catalog，SHA-256 `73bfa37d664ae0842368b5327be077627f06e1934530390b8a6571d77676f01d`。7 个分卷复用此前载荷入库验证的序列清单；train_04 缺少该字段，另从 ClearML 取得归档、核验哈希并读取中央目录。没有打开 GT 或 official_test 载荷，不声称此次重新完整读取全部训练分卷。

结果：8 个 train 分卷，32 个不同原始序列名；按去除末尾数字分段后缀分成 15 个名称组。无重复序列名跨分卷出现，但有同一名称组的不同片段分散在多个分卷，不能用 zip 分卷直接划分训练/开发集。该名称规则只是可审阅的保守分组代理，不是采集会话的独立来源证明；没有冻结划分、没有证明跨 split 无重叠。

首轮任务 `ab6dd2e38d2e41c89b2b21ec155a5ced` 因 `testoutput_CAV_data_2022-03-17-10-43-13` 没有分段后缀而失败。保留失败记录；后续支持完整日期时间名称作为自身名称组。回归 7 passed。

官方固定提交的 [BaseDataset](https://github.com/ucla-mobility/V2V4Real/blob/5a821e13753bafc611f95c47bc1a306acdcb0f7c/opencood/data_utils/datasets/basedataset.py) 将文件名键称为 mocked timestamps，延迟计算以 100 ms 量化为帧数。这支持标称 10 Hz 的索引时间协议，但不能证明逐帧传感器采集或网络到达时刻。未把 issue 提问者的推测当作作者确认，也未从日期时间格式序列名推导真实逐帧时钟。

正式真实时间协议依然缺少来源。可选的下一步是另立 `nominal-10hz-proxy` 诊断协议、保持原主协议不变，在所有回执中明确代理时间和模拟延迟；这需要用户确认，不能把代理时间结果当作真实延迟或 deadline 实验证据。否则继续寻找可核验的源时钟记录。无论哪种路径，训练/开发划分和公开权重选模来源仍需单独处理。

入口在隔离工作树 `work_dirs/rbf-pointpillar-runtime-20260917/tools/event_track_v2x/audit_v2v4real_train_groups.py`；远端 `/home/lbin/Desktop/rbf-train-groups-20260917`。源码与资产读取辅助脚本已保存为 ClearML artifacts，`train-group-report` 保留完整名称组、序列到分卷映射及资格标记。代码未提交 Git。
