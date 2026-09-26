# 标称 10 Hz 正式变体：运行与评价软件验收

本轮完成时钟协议从缓存到主方法、内部基线、独立评价和比较表的绑定验证。结果属于软件 fixture 验收，不是正式真实数据实验。持续目标「完成所有实验」仍未完成。

## 已实现

- `paper_nominal_clock.nominal_run_binding` 要求显式冻结 `remote_delay_us`、`deadline_us` 和 `max_age_us`。远端注入延迟限定为 0/100/200 ms，deadline 和最大年龄均为 100 ms。
- 参考时刻和 frame key 必须对应公共名义帧映射。按每端最新合法观测核验完整 delivery 身份、到达时刻和 payload hash；拒绝未来帧、过期帧、遗漏合法端、非最新帧及不一致 deadline。
- 主方法、geometry、单端、Top-K、不可恢复、MHT、JPDA/PKF 的重放 plan 写入 `protocol.time_protocol`。学习模型还须满足缓存生产者与时钟身份一致。
- learned+CI 和 M0–M4 接入相同时间身份，但明确限制为零注入延迟 clean-link 条件；不能宣称支持任意迟到或重传流。
- 独立评价器原有完整 protocol 相等检查能够阻止删除或替换时钟身份后的评价。报告保留该身份，比较表原有完整 protocol 比较拒绝标称/旧协议、不同注入延迟混排。
- 不修改原 DetectionCacheV2 数值字段、不重标旧结果、不修改数据划分。

## 验证证据

最终 **74 passed，0 skipped，0 failures，0 errors**。覆盖九个文件：时钟、paper contract、原生流水线、端到端、优先级选择、资源扫描、pair 基线、独立评价和森林训练数据。

端到端新增第三种情况：V2V4Real 标称时钟 fixture，从准备、训练、冻结到内部方法、优先级策略、独立评价。pair suite 同样新增标称缓存下的真实基线代码和 CLI 调用；不是通用占位基线。

- ClearML 独立验收任务：`3873549ccf72451393e7e3a440d7a329`。
- XML SHA-256：`da9f201e0bc7d25d27c267eeedf09bbe75a7d3cf0586930c7c85ee7c41ffb53c`。
- 相关源码、测试源码、XML 和验收回执均已上传并回读核验。
- 远端最终隔离目录：`/home/lbin/Desktop/rbf-nominal-final.ToKGEE`。
- 本地代码目录：`work_dirs/rbf-paper-stages-20260917`。未修改冻结主工作树代码、未提交 Git。

保留运行过程：第一轮 SSH 中断后，重新核验原进程已结束且 XML 为 73 项全部通过，没有因观测中断重复启动。随后新增 pair 基线支持和主入口模型身份检查，才运行最终 74 项回归。第一轮目录 `/home/lbin/Desktop/rbf-nominal-runtime.if8Y01` 保留。

## 尚未完成

本验收未生成真实正式缓存，未进行正式训练、跨序列泛化评价、公开外部基线复现或真实论文表图。V2V4Real 重叠训练采集组处理仍待明确决定，检测器来源仍待准入；SPD 资产也需单独检查。下一步应回到数据与权重准入，不重复已通过的软件 fixture 来代替真实实验。
