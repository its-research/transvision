# 新完整入口0087资格：既有输入复用定位

2026-11-02在112只读核对：原19文件入口副本中的candidate-input-manifest.json完整SHA仍为235690ce3e619606add1c7ebd8a09f8d2e107e321f53e9d0e0829943891d0120。0087行身份及全部存储成员见native0087-current-manifest-source-discovery-on112.json。

- 来源832a23d19dec4ddab3aa07963e5f6b27，20事件、17,307标签。
- 已有归档路径：/home/lbin/rbf-teacher-numeric0087-reference-complete-20261028/archives/832a23d19dec4ddab3aa07963e5f6b27/replay-rank3-sequence0087.tar.gz，现场大小38,252,385字节；登记SHA为74e32853cf9510d958461b2793e19df98881b9daf261d961843b0d78f4799c36。此次未重扫归档SHA，启动前必须重验。
- 原数据库104,509,440字节，登记SHA6fa8f386718155f26b7f5e1605633bd9f62cd7cd851285bbdc90830921ec04f9，与已封存helper完整0087有序输入/结果对照的数据库一致。此为原清单身份相合，不表示新完整入口已实际通过。
- 既有参考目录：/home/lbin/rbf-teacher-numeric0087-reference-complete-20261028/references，manifest.json及reference-00.bin至09.bin。新原冻结加载器必须完整核验它们；目录存在不能代替字节资格。
- 运行时包候选：/home/lbin/rbf-teacher-source-load-restored-20261027T1246/base/runtime-and-references.tar.gz，现场大小32,545,107字节；应与原运行时c2f672f4ffcbf1ea5705a1049695e50162e0e4501e3d7dc4f7db415d8e08e619核对，不盲目复用旧base目录内旧驱动。

实际资格入口应使用新run_mixed_teacher_single_pass.py及原19文件已验依赖、完整原生来源/模型/cache/schedule/因子检查；与封存原0087逐标签输入和结果摘要严格比较。当前原helper控制已经使用，实际新完整入口尝试仍需封版入口/收集器、资源核验和相应具体授权。不能把历史共享负载耗时用作正式同资源I结论。

read_native0087_qualification.py已准备，112语法检查SHA为834631a764d801e6f26a1e5342aaaf0419d62d94a4428208184bdfe9101b2cc1，尚未执行其main。它固定使用原independent_readback.py（959f95c9…）的validate_results，核对原三个完整报告SHA、三组各553,824字节请求/结果流及新驱动同源协议、因子/原生字段、监督器记录。只排除原冻结读取器已允许的elapsed_seconds差异，不新加误差门槛。原基线计算不重复；本次新完整入口执行仍消耗其自身资格授权，不能借用已用helper批准。读回器结果继续要求真实启动身份及保存后字节传输证明，单独通过也不放行46序列、切换或优先级训练。下一步需完成实际启动/收集编排和软件控制，再提出可审核的具体资格请求。

等待0050终态及完整保存后才安排112实际资格；不更改其进程、当前3090/A100任务或已有一次性批准的范围。后续应优先复用上述字节，不为准备入口再传一份大包。所有旧目录、失败结果和原来源保留。

用户已直接批准“批准准入后资格核验一次”，记录approved-native0087-full-entry-once.json；此前specific-native0087-qualification-request.json保留为请求时历史，不再作为待授权状态。具体入口467d7bd1…、控制器4430837b…、输入锁d8807e48…已准备，14项授权拒绝和启动/收集先等待后记尝试的软件检查通过。批准未消耗：11:42:32实测0050仍running，最新轨迹日志仍50/169。必须先完成0050终态及完整留证，再核验空闲112资源；不提前调用launch路径。

条件满足后的命令（Mac仅编排与字节传输）：

```
/Users/lbin/.local/share/clearml/venv/bin/python guard_native0087_after_0050.py launch
```

上式文件路径相对于本准备目录；实际执行应使用完整绝对路径。前驱守卫guard_native0087_after_0050.py先独立Docker inspect原CID/原启动时间/零重启，仍running则不调用启动器。终态后还要求完整收集文件存在，并在112重新验证保存的小产物字节、原清单/gzip收集完成标志及未变的授权/控制器SHA；失败前驱则等待完整失败留证审查，不重试它。守卫check模式只有只读核验；launch模式仅在守卫通过后调用原4430837b…控制器，后者继续检查空闲容器、GPU进程、RAM、磁盘及全部输入SHA，不通过则等待且不消耗数值次数。终态后仅一次collect，完整保存后SHA在112复核。不得将该批准扩展到源码发布、完整46序列新执行或活任务切换。
