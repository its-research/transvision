# 单元7只读内部日志观察：待具体授权

北京时间2026-11-07 09:47:02，经112/ClearML刷新：独立任务6ad715d5b78a4a9ba7c2d1d064752b8f仍in_progress，平台开始09:11:58.446，无注册产物。平台墙钟约35分钟包含初始化；数值开始、六项计数与ETA未知，不证明停滞。不得再次投递原独立任务。

112准备目录：`/home/lbin/rbf-V2V-unit7-original-source-preparation-20261106`。

本轮只读恢复此前截断的准备结果，三入口语法通过：

- 探针unit7_native_log_progress_probe_prepared.py，5064字节，SHA256 6a14cceb8eef9e5041bf0d3fcfbd45d403b4171be65f1b1c716a9aabab12294f。
- 投递器dispatch_unit7_log_probe_once_prepared.py，4287字节，SHA256 28d797e5c3e3e66c7d1a2c2a22ecd581325cf8eec7e5e76582e5560ef8e8dd50。
- 收集器collect_unit7_log_probe_once_prepared.py，2228字节，SHA256 110f6b73363cddc0a10861da411eda21ba7f6659b77c0a2395b2e9646ac3908b。

7项合成控制通过：根目录缺失、限定尾部、敏感行过滤、文件/父目录/根目录符号链接拒绝、FIFO不阻塞拒绝。未读真实任务输出。未创建探针claim、submission或Task。

拟单次发布至原API 10.100.35.118:8008，在ubuntu26-2八卡worker自身空闲、前四卡仍绑定目标任务且心跳新鲜时执行CPU只读观察。须明确批准与前后四卡活任务的物理组重叠；不调用GPU、不读数据库/模型/PID/环境、不发信号、不改目标或资源限制。固定19项日志/终态文件白名单，只读挂载已有父目录，采样两次间隔5秒，报告上传原10.100.34.118:8081并完整SHA读回。

新Task源码发布及CPU物理重叠例外尚未获具体授权，不执行。原单元7科学执行批准不替代此项。目标若先终态则跳过探针，沿原已准备5753192d…收集器完整读回注册产物并核对六项原标准，不因等待重跑或修改现有任务。
