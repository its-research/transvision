# 部署前审阅项（非测试报告）

## 新候选准备状态

duplicate-output-binding-v2/已准备新版verify_saved_collection.py（7999字节）及check_saved_collection.py（8543字节），原六文件未改动。独立读取器要求六个重复出口全部存在且大小/SHA对应，成员正文仍逐一完整读回校验；合成fixture补齐四日志成员，并新增六项重新登记SHA但归档不一致、六项归档缺重复出口的拒绝用例。此为未执行源码准备，不是通过报告；未在Mac执行或计算SHA，未暂存到112。

已向用户提出替代此前未执行六文件请求的软件验证范围：保留原四文件、换入上述两文件，共53767字节；只在112做SHA/语法/合成控制，不使用真实模型、数据库或SDK/HTTP，不发布或启动Task。等待具体答复，不能将自动goal续行或旧未答请求视为批准。

本记录只来自源码文本复核。六文件待批版本保持不变；未运行、暂存或发布。以下项不改原oracle、数值阈值或科学范围，也不构成额外科学要求。

## 独立保存证据的重复出口一致性

verify_saved_collection.py已逐项检查全部注册文件SHA、归档每个声明成员SHA、完整成员集，以及单独上传的sixscope join与归档内join字节一致。但完整保存还有重复出口：terminal-remaining45-collection.json和collector.stdout/stderr、sixscope.stdout/stderr。

正式部署前应明确核对这些单独注册文件与归档中targets/terminal-remaining45-collection.json、runtime/collector.stdout、runtime/collector.stderr、runtime/sixscope.stdout、runtime/sixscope.stderr的大小及SHA均相同，并要求这些成员全部存在。不能因为两套文件各自SHA有效，就自动视为同一组执行证据。

相应合成fixture应包含完整重复出口，并补“单独文件已重新注册SHA，但与归档对应成员不同”的拒绝例，以及“缺少一个runtime成员”的拒绝例。这比仅改保存文件导致注册SHA失败更直接覆盖交叉绑定缺口。

这不是已观察到真实产物不一致；原task_entry.preserve_outputs会从同一owned文件归档并再检查源文件SHA，且入口身份有固定SHA约束。这里补的是独立读回链自身的覆盖。当前候选不得据源码存在宣称完整Linux软件资格通过。

## 注册快照不是独立可信根

capture_registration输出的binding必须从真实ClearML只读查询取得，入口SHA、publication Task等预期值来自具体已批投递收据；不能由保存目录自生成expected值。下载完成后重新capture并与before全身份相等，之后Linux文件读回再使用最终binding。

verify_saved_collection纯函数的合成成功只说明协议检查可工作，不能证明真实SDK版本schema、HTTP传输/保存反向SHA或真实A100环境。正式路线仍需要原资源/物理交集/终态门槛和完整传输后核验。
