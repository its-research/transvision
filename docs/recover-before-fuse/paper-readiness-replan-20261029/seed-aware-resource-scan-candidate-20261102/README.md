# 逐种子输入与显式资源重复候选

状态：源码及合成软件控制通过；尚未发布、安装、准入或执行真实实验。

最终候选：scan_paper_resources_seed_inputs_repetitions_candidate.py，29,522字节，SHA256 4b9a2b8355d4e443ac6ff2539ad48d20c386b59c5bd239d9e6b661518f63a1f6。原tools/event_track_v2x/scan_paper_resources.py保持原字节SHA 9298db0d…；未修改模型、独立验收器、输入、预算、损失或容差。

新计划v3要求seed_inputs精确包含1337、2027、3407，各自完整绑定cache_manifest、schedule、gt_manifest的路径和SHA；不允许同时给共同输入以免歧义。每个种子内部各方法消费同一输入元组。新增measurement_repetitions必须显式为至少3的整数，布尔值拒绝。每候选生成3种子×R独立trial ID；原执行器每trial仍使用新子进程和原GPU预热/资源合同。

计划、回放命令、评价命令、完成收据、freeze输入/事件/GT检查全链使用同seed绑定。freeze拒绝少算、重复或篡改repeat，即使外层plan/receipt SHA均重新提供；保持原指标方向、各trial资源约束与全体可行性要求。原旧版单一输入路径仍通过合成回归；其旧种子实例不冒充正式重复测量。

31项控制在112执行；使用明确的合成协议对象及拦截子进程的fixture launcher测试真实候选函数，没有真实模型、GT评价、数据回放或资源测量。最初测试脚本漏注入sha_file的NameError发生在fixture计划前，原执行日志保留；v2只补测试命名空间后通过。不是科学失败重跑。

集成目标位置为冻结包中的tools/event_track_v2x/，保持原ROOT目录计算和项目依赖。此处docs下文件为待集成源码，不作为当前工作目录直接运行的生产入口。下一步：绑定完整合格的每seed train缓存/事件/GT与环境、封版源码及完整源清单、真实模块/编译依赖准入、隔离资源与具体发布/执行授权。保留MHT Linux完整合同决定门槛。尚不能宣称运行环境、资源重复或论文同资源比较通过。

证据：source-change-report-v2.json、routing-repeat-controls.json、saved-source-readback-on112.json。SHA与AST及软件控制全部112执行，Mac仅字节传输与文本记录。早期无重复版本和失败测试证据保留，不用于实际投递。

## 真实模块加载发现的阻碍

2026-11-02 07:57:35，原e5镜像真实项目模块加载在__init__→preregistration阶段因缺少cryptography失败，尚未进入扫描器计算。152份源码（2,184,793字节）已暂存并逐份SHA记录；失败执行记录和完整源码清单已读回外置盘seed-resource-scan-module-import-old-readback-20261102，两份保存后完整SHA在112核验通过。不能据31项合成控制放行真实运行时。

修正版保持同一镜像和候选，仅将原已缓存cryptography46.0.3、cffi2.1.1、pycparser3.0的三个公共wheel离线安装至新容器独立/control/deps。全部wheel已对既有PyPI快照核SHA，不更换NumPy/Torch/SciPy，不修改现有环境，不屏蔽GPU或新增CPU/RAM/线程限制。具体一次性预检授权已请求，尚未执行；入口SHA fbaec2b4…，21项授权/隔离安装控制通过。即使导入成功，仍不等于数值算子、模型、完整回放或资源测量资格。

## 2026-11-03 03:08：持续日志下的原deadline检查候选

审阅发现run_stage_with_eta在每次reader.read有数据时直接continue，原deadline只在无日志时检查。新增独立scan_paper_resources_deadline_candidate.py保留所有原参数、上限、测量和数值路径，仅在读取前对仍活的子进程检查同一个deadline；已退出子进程继续完整排空日志，不增加超时后的宽限或更改预算。

112合成控制 /home/lbin/rbf-resource-deadline-software-controls-20261103/deadline-controls.json 通过。原候选SHA4b9a2b8355d4e443ac6ff2539ad48d20c386b59c5bd239d9e6b661518f63a1f6；新候选SHAad2890ed1e288058bc25d693fd519e5d9fa34841ca1d2c520345e998d9179580。原函数连续可读日志fixture在模拟第6秒才检查2.5秒上限，新函数第3秒首次越过后立即处理；零deadline仍活子进程在读取前拒绝，已退出子进程无误杀，其余顶层AST逐项一致。只使用合成文件/时钟/Popen，未向真实进程发送信号，未运行模型、GT或资源测量。

原候选和21项依赖修复准备均未替换；新候选未进入原封版源清单、未发布、未集成或获正式测量准入。后续必须重新绑定其完整源码SHA、执行原软件回归和真实依赖加载，保持已有新失败不自动重试及具体发布授权门槛。本项证明时间上限控制修正，不证明同资源比较已完成。

03:10补充：新deadline候选在112通过原31项planner/run-command/freeze回归。check_deadline_routing_repeats_on112.py仅将原测试的目标文件和输出报告名改为独立候选，未改变控制条件：三种子×三重复共9个独立trial、各seed cache/schedule/GT路由、少于3/布尔/缺少重复拒绝、错误seed、重新计算外层SHA后的cache/events/GT篡改和重复repeat仍拒绝、旧common-input路径保留。原实际候选函数被调用，子进程launcher、输入及协议为显式fixture；真实子进程/模型/GT执行数为0。远端报告/home/lbin/rbf-resource-deadline-software-controls-20261103/deadline-routing-repeat-controls.json。Mac保存副本仅不透明传输，不宣称新完整SHA读回或正式资源资格。下一步仍是待具体授权的真实模块加载修复及新候选源清单集成；没有启动测量。
