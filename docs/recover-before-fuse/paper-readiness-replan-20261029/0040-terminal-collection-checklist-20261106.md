# 0040终态收集条件（静态核对，不表示完成）

15:22准备进展：在112找到已成功0056原收集与Linux适配。仅从112既有源码重建0040原收集器8287字节，其完整SHA fca7073a9836d74a8bfb3f00cfbf15e3d34898d5f97a77cc2c6c00b6f6feac63与原登记完全一致；原postcheck SHA61a6ea94…也匹配，不需新Mac源码传输。新Linux适配器9015字节、SHA61e745ab76e0f318f85eacc40ef49e5e5c41973b76126387e1efc81baccae07b，原数值/metadata postcondition整段AST一致；原7来源改接已获条件批准的精确流式SHA/EOF证明，固定CID/源清单、七文件路径/大小/SHA、EOF、无Mac计算及无远程载荷副本。实际协议断言1正12篡改拒绝通过，合成证明未写入真实receiver路径。修正原0056适配中遗留的“actual_SSH_exit_code=0”表述：本0040版明确记录本地collector子进程退出码及legacy_SSH_transport_used=false，不冒称本地读回执行了SSH。

准备目录/home/lbin/rbf-approved0040-terminal-collection-preparation-20261106，冻结e5镜像、原postcheck依赖只读映射及收集输出计划已准备。须先原数值容器成功终态、已批准原7流完成且实际客户端exit0/源metadata稳定、新鲜资源通过，才可启动一次收集；当前未启动收集或来源流，未重算数值，不改变既有进程。

2026-11-06 14:08 +08 实测：固定容器仍running、无OOM、无重启；teacher trajectory最新日志仍为75/193（13:24:47），不能把此前legal-action的175计数当作teacher trajectory进度。原7文件流式核验已经取得条件授权、协议控制通过，但必须等本容器成功终态及新鲜资源准入；下文“尚未收到批准”是历史状态，已被本段替代。Linux-only完整收集适配仍须准备并核对，不能在Mac执行原collector。

本次唯一容器：e10debded922b03538693e17d849cbe595905282dcb097f06e56ca863d030908，真实开始2026-11-06 10:02:52.856674431+08。保留活进程、失败结果、原输入和模型；不新增资源限制、不在Mac计算。

原run_first0040_complete_cache_candidate.py的执行顺序已直接核对：

1. 完整读回35882个原见证，保存independent-storage.json。
2. 原driver.verify_sequence在已绑定compact传输上执行六项；transport.open_count至少6，193事件、原最终模型、0040行数、完整序列及capacity-undecided不算最优等断言保持。
3. sequence-numeric-result.json保存时status仍为numeric_checks_passed_pending_final_seal，不能据此宣布终态成功。
4. independent_original_factors完成2633行原容差检查，生成original-factors.json。
5. 离开原存储上下文后，重新完整核对全部冻结源码、reader、合同、checkpoint及压缩样本SHA。
6. 只有最后terminal.json为completed、sequence_numeric_passed=true、verified_scratch_released=true，且实际容器exited0，才构成数值运行成功。失败保存failure.json；不自动重试或清理失败scratch。

原collect_first0040_once.py还要求：193事件/2633因子、原六项及COMMON/postconditions、35882见证/469表、全部小报告及原压缩输入完整SHA、原容器与镜像身份、注册rank4原归档大小/SHA刷新、七份原来源完整SHA，最后再封存和加入正式索引。不能仅凭日志193/193、容器exit0或原7来源流通过就加为23/46。

原collector的Mac main含SHA/解码和原7文件SHA循环，禁止在Mac运行。后续需沿既有0056成功Linux-only收集方式保留全部原检查；涉及新源码暂存则先取得具体批准，数值本身不重复。原7反向核验两文件5644字节及13572152775字节数据流的请求已提出，尚未收到具体批准；成功退出、新鲜资源和合成协议通过都是前提，不能提前发送。

0094原首次数值许可继续保留，但须0040完整数值+来源+小产物保全+正式汇合全部通过后才重新准入，不让未完成收口的0040与0094重叠。

## 最终小产物保全入口准备

112文件 `/home/lbin/rbf-approved0040-terminal-collection-preparation-20261106/preserve0040_small_outputs_prepared.py`，6334字节，SHA05db218c2c7e4c6c19d4e12a92ad4b4240d9fd3f0d8716933154df4bf9bd5525。21:38:06，12项合成完整性/拒绝检查通过，报告完整SHA91648f5e44cbbd8c11039a4fb37bd6ff2db670165225bbe3b20b364eb4828197。未处理实际0040结果或开始传输。

实际阶段pack须在已授权来源/客户端终态、新鲜资源及完整Linux收集成功后执行，CLI `--stage pack --expected-SHA256 <实际收集证明SHA>`。只归档完整collection小文件（单文件不超过原收集器2MiB界限）与原7来源证明；不复制大数据库或原压缩输入。OUT为C/external-preservation-once，排他创建，保留失败，不自动重试；源/副本身份与完整SHA核对。

随后Mac只以cat/SSH保存collected-small-results.tar及小元数据到新外置目录，不本地计算SHA。保存归档再反向经SSH交给同一112入口 `--stage verify-saved --expected-SHA256 <实际archive-member-manifest.json SHA>`：完整整包SHA/大小、每成员SHA/大小、路径/类型/重名与精确EOF全部通过，才生成saved-external-full-SHA-proof.json。该次反向检查有排他attempt记录；任何失败保留。真实保存文件前后stat稳定及客户端正常退出仍须外层确认，不将直接喂112原归档冒称外置读回。

实际pack/verify-saved、外置结果传输与正式index23汇合均尚未执行；脚本/合成通过不增加22/46计数。正式源输入、六oracle和1e-8/1e-4容差不变。

## 一次性实际收集启动器

112已准备launch0040_collection_once_prepared.py，5748字节，SHA433052ad758420995517d0322fb0ce5a5a89498bd231b68c22c877acdfffaf7e。21:49:10，7项Docker参数构造/拒绝合成控制通过，报告SHA927ae4f16e827f4569d4822affe5c4a6dda98f20d39dd05b314d908528604991；没有执行Docker或消耗实际收集次数。

实际入口要求C/original7-client-evidence-once/manifest.json的原SHA参数，并完整读取原客户端实际留下的七份文本：client-scope.txt、client-start-UTC.txt、client-finished-UTC.txt、actual-client-exit-code.txt、client-success.txt、source-before-metadata.txt、source-after-metadata.txt。manifest.files逐文件绑定bytes/sha256，须由112实际核对，Mac只传输这些已产生原字节，不计算SHA或制造成功状态。当前原客户端未退出，不提前创建成功材料。

启动器严格要求真实exit文本0、原success/scope精确协议、七份原源文件metadata前后相等且路径/大小绑定原source_preparation、UTC开始结束顺序；再调用固定SHA的来源/资源preflight，核对原计划SHA和全部原入口。排他collection-launch-once.json之后才可能创建单个保留容器，保留失败，不自动重试。TMPDIR只准备自身空目录；没有新增CPU/RAM/线程限制、GPU屏蔽或privileged/PID权限。真实容器启动后只记录实际状态，不从launch成功推断collection完成。

完整客户端材料、整份来源SHA/EOF、实际资源及输出容器终态都尚未完成；启动器准备/7控制通过不构成这些实际资格。

## 当前启动器版本更新：真实Zsh单行协议

22:19静态复核及Mac原client-scope.txt只读观察确认print -r多参数实际写为单行空格分隔。原未执行433052ad版本的splitlines匹配不适用；保留旧文件和7项参数构造检查，禁止用旧文件启动。没有修改正在运行的客户端或消费实际收集次数。

当前入口为C/launch0040_collection_once_clienttext_prepared.py，5716字节，SHAeb0b0c62f6c7effc7cb2e63ac047b5d6a197879dfcdb160fa9dbebe7626dc916。仅两处scope/success改为严格单行字节匹配，其他源码可逆还原到旧433052ad，所有来源、退出、元数据、UTC、资源、排他claim和Docker参数不变。

22:21:22在112通过16项客户端闭合协议合成控制，报告SHA3146e2daf64f6560ea687344f5e0e2ed7ab308676bb330107491e951008f5c4a。正常单行协议通过；非零退出、多行scope/success、缺token、元数据变化、错误路径/大小/数量、倒退或非UTC/坏时钟、来源清单变化、错误manifest SHA、材料后改与符号链接均拒绝。原小来源清单字节只在112测试目录复制，无数据库/模型载荷读取、无Docker执行。真实客户端仍未资格化，本次不构成原来源或实际收集合格。

### 22:49 小产物容量边界复查

112对实际0040终态工作目录只做文件元数据扫描，唯一超过2MiB的文件是原压缩输入`sample/sequence-0.compact.sqlite.gz`，584,774,531字节。它仍由收集器完整SHA重读并保留源文件，不纳入小产物外置归档。现存其他实际文件均未触发2MiB小文件门槛；Docker日志及未来收集元数据仍须在实际收集/打包时核验，不提前宣称整个归档合格。22:49:39原来源接收进程仍存在、完整文件5/7、最终来源证明未产生。没有执行最终收集或打包。
