# V2V4Real 缺失分卷接收

用户负责浏览器下载到桌面；Agent 不再操作浏览器，负责核验与上传。初次交接时下载进行中、已验证上传数为 0；后续完成回执见下方，保留初始执行记录。

官方目录当日复核共有 train_01–08、test_01–03、val.zip。已有 train_04 复用 ClearML 注册表 `98d72fb5afc94b418aa0836f927fa7d4`，不重复传。其余 11 包约 14.56 GB。val 仅归档，不改变 train 内开发、official_test 最终评价的论文协议。

批处理入口 `work_dirs/rbf-data-registry-20260917/process_downloads.py`，使用 `/private/tmp/rbf-pointpillar-tests-20260917/bin/python`。本次等待上限 7200 秒；只读取桌面上原文件名、大小匹配的 ZIP，忽略 `.crdownload`。不移动或删除用户正本。

每包执行：官方元数据固定 SHA256 → 载荷大小和发布端 SHA1 → SHA256 → ZIP 路径/类型/体积限制与全部 CRC → SCP → 管理机再次哈希 → ClearML 上传 → 强制下载回读。任何错误保留记录并停止，未通过的文件不计成功。

管理机暂存目录为 `10.100.35.112:/home/lbin/Desktop/rbf-v2v4real-volumes-20260917`，开工可用约 38 GB，每包回读前保留至少 5 GiB 余量。新 ClearML 任务在第一个合格分卷到达后创建；所有缺失卷上传完毕才生成继承旧索引的新版本，并核实 completed 终态。原索引不覆盖。

当前进度：`work_dirs/rbf-data-registry-20260917/download-ingestion-progress.json`；逐卷回执：同目录 `*.zip.verified.json`；远端回执：`ingestion-state.json`。后续状态应读取这些回执，不将本页交接时计数当作最终状态。

实现与测试位于隔离工作树 `work_dirs/rbf-pointpillar-runtime-20260917`。校验/注册表/解压直接回归 32 passed；新增实际下载尚未完成，真实缺失卷上传回读尚待执行。为使 ZIP 检查不加载训练依赖，仅将已有解压器的 native inventory 导入改为延迟加载，既有调用与测试接口保持兼容。

## 最终提交回执

- ClearML Task：`14675561902041d7ab77fa2adbf2aeb0`，已独立重读确认为 `completed`。
- 新增 11 包，14,561,208,099 bytes，全部通过发布端 SHA1、SHA256、ZIP 安全结构/CRC、传输后哈希及 ClearML 强制下载回读。
- 复用 train_04 后，官方原始压缩包共 12 个：train 8、official_test 3、val 1，总计 15,393,811,548 bytes。
- 新统一索引包含 30 个命名资产，并继承旧索引与历史引用。Catalog 245,225 bytes，SHA256 `73bfa37d664ae0842368b5327be077627f06e1934530390b8a6571d77676f01d`。原索引未覆盖。
- `val.zip` 多一层 `val/` 目录；只对这个明确分卷规范化检查视图，原 ZIP 字节不变，路径安全检查不放宽。新增正反例后直接回归 35 passed。
- train_01 首次 SCP 断连后使用 rsync 恢复，已完成。两次失败原因保存在 `download-ingestion-progress.json` 的 `failure_history` 中。
- 本地完成回执：`work_dirs/rbf-data-registry-20260917/v2v4real-ingestion-state.json`；新索引副本：同目录 `catalog-v2.json`。
- 新消费者缓存从新索引成功拉取 test_01 原始包并验哈希。随后仅清理哈希匹配的临时回读 ZIP 副本，共 15,194,790,739 bytes；桌面、远端暂存原件与 ClearML 正本未删除，可用空间约 24 GB。
- 数据任务完成后服务端拒绝追加附件，因此未重开数据任务；补充测试、失败重试、使用说明及清理回执归档至关联审计任务 `26b424b45e6f452d910627c8a5aa2b7c`，同样已确认 completed。最终提交记录为本地 `final-submission-receipt.json`。

后续在隔离工作树运行：

```bash
python tools/event_track_v2x/paper_data_registry.py fetch \
  --registry 14675561902041d7ab77fa2adbf2aeb0 \
  --catalog-sha256 73bfa37d664ae0842368b5327be077627f06e1934530390b8a6571d77676f01d \
  --asset v2v4real-test-01-raw --purpose archive
```

原始包含 GT，仅允许归档/准备，不直接交给推断；official_test 不用于训练或选模，val 仅归档。原始数据齐备不表示检测缓存、正式训练或论文性能实验完成。
