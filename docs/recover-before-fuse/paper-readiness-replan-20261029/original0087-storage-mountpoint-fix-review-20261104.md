# 0087独立存储准入：启动失败与挂载点修正

后续终态（11月6日复核）：下文“未获批准/未执行”为历史。用户随后批准的挂载点修正版已经执行，原容器rbf-approved-original0087-storage-mountfix-once-20261104实测exited/exit0，开始2026-11-05 00:38:35、结束00:39:24（北京时间），报告墙钟49.021777秒。storage_only_qualified=true，未调用数值/模型或启动pool；本次已消耗一次许可，不能重复运行。

112结果位于 /home/lbin/rbf-original0087-independent-storage-control-mountfix-once-20261104。terminal SHA1a5c924915acde239d6635c154815f652c752ac29df66b04dced2a286e4bdf33，compact 99500032字节、SHAc987ee53925324b4d124a26b6d2e7a64776e631bf4972a8615c26d83f95b87db，原库SHA64a24631d844df9cdc192328f37a0625894be405455d053a1c5bec6c2b9c8eeb。独立读取及保存证明在同root，存储通过不能替代完整真实并行数值对照或任务切换。11月6日仅读回已有终态和6份保存报告，未重跑存储转换、控制或数值。

本次启动已失败并保全，未自动重试。失败是准备遗漏：只读namespace中未预建 `storage-control/output`，OCI初始化无法创建输出子挂载点。容器 `1a09c233854a566e68827397c0da5b92f412a585444a16284d34a3fb274f3749` 为created、PID0、StartedAt零值；入口、合成控制和真实转换均未启动。0072保持原容器running。

六文件51175字节原SHA、语法、镜像e5b249…及资源预检此前已通过：23:37空闲10326851584字节，原需9247285248字节，MemAvailable27469348864字节。它们不代表存储资格通过。

## 失败保全

112原目录 `/home/lbin/rbf-original0087-independent-storage-control-once-20261104` 保留，失败保全清单SHA `87624fdf882997a075c6ed0d7186c15fff6e564263afb30603c68c7ee453d71a`。修正准备及全记录清单SHA `6af8811b3525e628c21227c6f7b8f6da4c3134cd50ddb8702160f211a6fe0ac2`。本目录 `original0087-storage-failure-readback-20261104T2338` 保存小文件副本；22份保存来源、记录和清单由112反向完整SHA及EOF核验通过，证明SHA `83910d6a72dce6f4a825c625f369c244ff8a1eacad34aad7c7c2adf8a0973bdb`。Mac仅传输，未计算SHA。

## 已准备但未执行的修正

- 新root：`/home/lbin/rbf-original0087-independent-storage-control-mountfix-once-20261104`；新容器：`rbf-approved-original0087-storage-mountfix-once-20261104`。原失败root及容器不修改、不删除。
- 复用同一六文件的112已验副本；入口SHA `c0305dec9c98efd7e86bbab61a703b8b741822483ab199d7977cbaa08e193931`，独立检查器SHA `2eabc4f4dd6194d71a2f073d4f214f0f82f892a33117cb8d92d50cfac349b22e`，不改代码或算法。
- 只增加一项文件系统准备：在新的readonly namespace挂载之前，预建空目录 `namespace/storage-control/output`。raw目录及reader文件挂载点保留原准备，输出写入独立的新host output；绑定新实际批准记录并更新其stage grant SHA。
- 再核对原镜像、全部来源SHA和目录存在性、新鲜资源、无既有同次容器/输出后，才执行一次相同正例＋三项拒绝控制；全部通过才处理原311898112字节数据库、最大623796224字节compact输出、独立全字节/表/schema/顺序检查。
- 原8GiB保留、32MiB控制余量、原数据与现有任务不动；不加CPU/RAM/线程/affinity限制、不禁GPU、不运行数值oracle/模型/pool、不传输大库、不删源。

此修正尚未获得直接重跑批准，未创建新实际授权或运行新容器。此前一次启动批准已用于失败启动；依据“新失败不自动重试”约定，不能自行再启动。通过存储检查也不等于完整0087数值/性能对照或切换资格。
