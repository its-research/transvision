# 0008隔离失败后的限定接续

## 最新结果：2026-11-09 14:57北京时间

用户批准 `call_7dcd86d0a97f4a3583b52ef092782226/0` 后，修正版软件复验在112通过，含授权12项拒绝、报告15项拒绝及正例。探针原3,996字节SHA未变。一次性Task `b48cd725f534460a86663720855a787c` 实际绑定A100 GPU4–7并completed；2,277字节注册报告完整HTTP及保存后SHA均为 `1011e0435395b6f2dabbf2275c4c8c4600a5cf50e30a5a9b77b387293ea1abe2`。

平台本次开始14:55:50.737、终态14:56:31.436；策略读取阶段14:56:18.247156至14:56:18.741040。平台时长包括初始化与收尾，不是数值计算时间。未执行模型、数据库读取、命名空间操作或GPU计算。

观测：Seccomp=2、Seccomp_filters=1；当前AppArmor标签 `docker-default (enforce)`；unprivileged_userns_clone=1；apparmor_restrict_unprivileged_userns=1；user/mnt/net命名空间上限均907540；有效能力位 `00000000a80425fb`。本诊断不能证明具体哪一层拒绝了原unshare调用，不能把内核总开关开启当作实际容器可创建命名空间。

只读诊断合格不等于执行隔离合格。下一步需准备并审查能够保留原只读挂载及断网要求的运行方案，明确是否涉及worker配置或单容器权限；任何权限变化和新失败重跑仍须具体授权。没有重试原隔离操作，没有启动0008。记录：`teacher1337-A100-0008-report-preparation-20261109/optimized-deployment/policy-terminal-readback-once/`。

以下保留此前准备与软件失败过程；“等待授权”已被本次批准及单次执行替代，不重复使用该次数。

原Task `91f8db1232eb4989acf6d95585af1b7d` 已failed，三份注册产物完整SHA读回；没有执行数值。再次只读平台启动日志及登记容器参数，未取得能区分seccomp、AppArmor或内核用户命名空间开关的直接证据。不能凭Operation not permitted确定根因，也不能自动增加权限。

下一项候选仅观察自身容器：`/proc/self/status`中Uid/Gid、能力位、NoNewPrivs及Seccomp字段，当前AppArmor标签、uid/gid映射和用户/挂载/网络命名空间开关。未知或权限拒绝逐项记录，不冒充关闭或无限。不读取环境、其他进程、数据库或模型；不调用unshare，不调整安全策略，不运行GPU计算。

源码 `teacher1337-A100-0008-report-preparation-20261109/probe_namespace_policy_readonly.py` 为3,996字节，112测得SHA256 `493f447ee2e5ea9ec844971642a8a7c0eff8c4e11e8dd40527c98ff3554f8d20`。初次语法及字段筛选1正3拒绝通过；尚未发布Task、尚无实际策略观测。

生成收集器时发生嵌套shell引号丢失，后续合成报告控制在首个正例因`report[task_id]`错误失败。原失败源码保留为 `collect_policy_failed_generation.py`。没有完成整组软件资格，也没有真实报告。根因是入口准备错误，不是平台失败。

修正版 `collect_policy_repaired_once.py` 直接保留带引号字段，并纠正单一`policy`注册产物名称和原stdout比较遗留。`check_policy_repaired_once.py` 已准备持久化单次结果与失败留证；尚未执行，不能声称修复已通过。原探针源码不变。

等待具体授权：两份修正文件暂存112、单次软件复验；全部通过后才发布上述3,996字节Task至原API，在重新空闲与物理交集准入通过的A100四卡执行一次只读策略观察，报告上传原文件服务器并完整SHA读回。该范围不包括再次unshare、权限提升、0008数值或修改现有任务；失败不自动重试。

该诊断成功也不代表隔离、真实输入或完整保存容量通过。后续应以实际策略证据选择保留原只读/断网标准的运行方案；不得用报告字段代替执行约束。
