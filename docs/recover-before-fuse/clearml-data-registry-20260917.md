# 论文数据 ClearML 归档回执

后续完整原始数据版本为 `14675561902041d7ab77fa2adbf2aeb0`，已补齐 V2V4Real 分卷；当前索引和拉取命令见 [缺失分卷提交回执](v2v4real-download-ingestion-20260917.md)。下文保留首版记录及当时缺项。

已盘点的现有论文数据统一登记至 `Thesis/Recover-Before-Fuse/DataRegistry`。后续新实验先从固定索引拉取，不以桌面/临时目录为数据来源。原有冻结任务不改写。

- Registry Task：`98d72fb5afc94b418aa0836f927fa7d4`
- Catalog Artifact：`catalog`，232,968 bytes
- Catalog SHA256：`f6d8fe076a0224b7a738f0aa8b5f86034304c86d20543a5bdcdc77dd4f30aaea`
- 19 个命名资产，283 个历史 Artifact 引用，覆盖 39 个任务。
- 新上传并强制下载回读：7,080,579,259 bytes，含 V2V4Real train_04、公开权重、SPD train V2 缓存、train-only 数据、本地结果及审计记录。
- 复用 SPD Dataset：`691397743f934284b9419582adcce0f6`，24,185,541,105 bytes；本轮核对文件元数据，未重复下载整个 Dataset。消费者拉取时逐文件验哈希。保留 unverified-source-bytes 标记。
- 软件回归：36 passed、1 skipped；真实服务新缓存拉取成功，带 GT 原始包的推断请求被拒绝。

## 后续入口

代码位于 `work_dirs/rbf-pointpillar-runtime-20260917` 隔离工作树，未提交 Git。进入该工作树后运行：

```bash
python tools/event_track_v2x/paper_data_registry.py fetch \
  --registry 98d72fb5afc94b418aa0836f927fa7d4 \
  --catalog-sha256 f6d8fe076a0224b7a738f0aa8b5f86034304c86d20543a5bdcdc77dd4f30aaea \
  --asset v2v4real-train20-gt-free --purpose inference
```

依赖已配置的 ClearML SDK 与私有网络。管理机部署副本位于 `10.100.35.112:/home/lbin/Desktop/rbf-data-registry-20260917/paper_data_registry.py`，Python 为 `/home/lbin/miniconda3/bin/python`。每次实验记录索引 ID/SHA256、资产 ID/SHA256；新增数据创建新索引版本。无桌面回退，不自动解压原始 Artifact，不将 GT 包直接传给推断。

完整索引、上传回执和测试 XML 保存在本仓库 `work_dirs/rbf-data-registry-20260917/`；索引正本及资产在 ClearML。首次索引生成的 SDK 导入兼容错误已修复，失败计划/回执和旧源码仍归档可查。

最终入口源码为该任务的补充 Artifact `registry-tool-source-v3`；另有 `registry-tool-tests`、`registry-software-tests`、`registry-usage`。补充附件不覆盖固定 catalog；v3 修正 SDK 对 created → completed 转换默认忽略错误的问题，要求服务端重读确认。

## 尚缺资产与科学边界

V2V4Real 除 train_04 外的 train 分卷、official_test 尚未取得；正式双数据集主协议结果尚未生成。因此本次是现有资产归档完成，不是论文全量数据齐备或实验完成。历史 Artifact 未逐一提升为合格训练输入；旧 car-first、detector-in-sample、模拟数据及公开权重选模来源未核定等限制保持不变。

桌面原始文件和既有数据目录未删除。既有大包复用引用，不重复上传。
