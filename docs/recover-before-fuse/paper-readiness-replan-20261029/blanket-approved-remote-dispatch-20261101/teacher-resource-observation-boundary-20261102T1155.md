# 3090 teacher资源只读核验边界

11:53–11:55通过既有ClearML只读接口核对Task18abed2123c14d63ac411ec7ea6ba611，未提交探针或修改任务。

- 平台最后上报memory_free_gb约149、disk_free_percent=17.4；get_last_scalar_metrics及原last_metrics不含本次采样的明确时间戳/文件系统路径。因此不把查询时间当采样时间，不据百分比推导当前工作卷剩余字节，也不归因宿主GPU利用率为本任务CUDA计算。
- 原resource-admission注册产物3970字节，SHA31085d389be9b92d60134c5ff6826c96d96b211fc2739af6d8357149d7c5b19f，已在112重新完整读回且注册SHA/size未变。
- 原启动准入记录work_free_bytes=3,402,591,543,296，required_free_bytes=751,619,276,800，reserved_shared_disk_bytes=454,501,766,510，8并发峰值输入39,847,164,324字节。它只证明原启动准入，不是当前绝对剩余空间。
- 原CPU affinity为0–111、parallel_sequences=8；当前CPU吞吐、cgroup配额、共享负载和各计算阶段并不能由affinity或宿主cpu_usage百分比推出。按真实内部完成计数及线程CPU证据判断进展，保持原进程。

当前证据未证明新增内存/磁盘失败，也不足以重新批准资源合同。没有据监控百分比停止任务、改变限制或删除数据。新入口仍须独立新鲜物理/CPU/RAM/工作卷准入，正式同资源I测量仍需隔离重复。

证据：teacher-resource-monitoring-readonly-20261102T1153.json、teacher-resource-metric-raw-times-readonly-20261102T1154.json、teacher-resource-artifact-registry-readonly-20261102T1154.json、teacher-resource-admission-source-readback-on112-20261102T1155.json。
