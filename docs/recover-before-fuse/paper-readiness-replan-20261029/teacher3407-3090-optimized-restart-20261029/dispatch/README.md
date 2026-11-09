# Teacher3407 优化切换入口（准备，尚未发布或提交）

此目录只有显式执行命令才会连接 ClearML。`release.json` 默认 `sealed_for_execution=false`，任何任务操作都会拒绝；草稿不能证明准入。

顺序固定：

1. 固化最终 bootstrap、entry、manifest 与覆盖包；填写原输入 URL/SHA、完整文件读回与发布回执。新覆盖包的具体发布授权单列。
2. 绑定完整本地 0087 独立读回（两阶段各20事件、17,307标签，逐标签/原报告一致且实际总耗时改善）。
3. 从真实已核实资源记录分别填写 ubuntu26-2 的整机 nvidia-smi index→UUID 和3090宿主 DeviceMinor→UUID。资格模式仅使用已验整机index对应的实际可见UUID集合，独立读取/proc的minor仅作UUID集合边界，不把index冒充minor；整机替换仍严格核对宿主minor。冻结镜像不变。所有准备SHA完整核对后封存 release；此后改变release必须重新审查，不能沿用已消耗尝试。
4. `dispatch_once.py validate --release release.json` 只做离线门槛核验。
5. `dispatch_once.py qualify --release release.json --execute`：ubuntu26-2所有4/8worker与队列全闲后，通过原四卡队列提交一次 CPU-only 完整0087 Linux资格。每次只提交一个任务，未终态不循环等待。
6. 资格 completed 后运行 `collect_linux_qualification.py --release release.json`，完整读回全部注册产物及全日志，核验实际run_mixed入口、20事件/17,307标签、442因子、原容差、两阶段精确一致与Linux实测改善。平台completed本身不放行。Mac直接比较四份逐标签摘要的553,824字节与长度，核对paired evidence的每个SHA都存在于下载集合，并把实际run_mixed的numeric/trajectory原报告逐字段与两臂frozen_result比较（仅排除elapsed_seconds）；不重算数值。停止前dispatcher再执行同一字节/报告核验。
7. `dispatch_once.py replace --release release.json --execute`：先验证Linux完整收集→旧任务注册产物和全日志停止前快照→按已有直接授权请求停止旧任务→终态后第二完整快照→3090三个worker及两队列、全部物理交集均空闲→同镜像八个独立整序列进程执行剩45，复用原0087。不改逐序列数值顺序，不使用GPU。

`replace` 遇异步停止未结束会返回。下一次只能恢复同一已记录停止/创建/入队过程；clone结果未知时按唯一名称查回一个原task，0个或多个都拒绝再次创建。入队结果未知时不再次入队，失败/停止任务永不自动重试。若旧task在本次stop请求前自行失败/完成/停止，先保留核对，不直接重跑。

`preserve_registered_and_logs.py` 仅保存平台注册产物与 `events.get_task_events` 滚动游标全日志（不把服务端上限10,000当总数）；每份下载全SHA读回。停止前后的两快照独立保存。未注册旧工作目录仅由新bootstrap只读盘点、保留，不声称已完整归档。

全量资源合同：8可用CPU、64GiB可用RAM、700GiB磁盘门槛和原精确454,501,766,510字节teacher保留量；每序列BLAS1，CUDA禁用。Linux资格为1CPU/2GiB可用RAM/10GiB磁盘准入，只核验可用量，不额外限制112或修改其他任务。

`release.json` 必备发布结构：`original_input` / `optimized_overlay` 各有 `url, bytes, sha256, complete_file_readback{path,bytes,sha256}, publication_receipt{path,bytes,sha256}`；对应发布回执须包含相同url/sha256/bytes和`all_bytes_SHA_read_back=true`或`stage=full_SHA_readback_passed`。新包授权文件同时须明确`approved_publish_and_full_SHA_readback=true,approved=true,direct_human_reply_received=true,scope=teacher3407_3090_optimized_overlay_publication_once,publication_count=1,destination,package_sha256,package_bytes`，不得用优化重跑授权代替。

`qualify-state.json` / `replace-state.json` 为唯一提交记录，含 `task_id, source_sha256, overlay_sha256, manifest_sha256, original_input_sha256, source_family`。最终45网络收集器据此绑定，不重写报告为旧来源家族。原1e-8/1e-4不变，不自动删除数据、重试失败、commit或push。

软件检查：`python -B test_dispatch_software.py`。仅假API、合成临时文件及已通过的本地0087证据SHA；不导入ClearML、不联网、不执行数值。
