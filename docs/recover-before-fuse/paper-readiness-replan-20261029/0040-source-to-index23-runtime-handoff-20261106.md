# 0040来源完成后的实际执行交接

> **23:35实际状态：原67509已exit0，来源7/7、最终收集和36成员外置反向SHA全部通过。首次正式索引在读取原核验器源码前因缺只读挂载失败，尚无新索引。不要再执行原传输/小证据/最终收集/外置读回，它们已用完且通过。** 挂载修正待单独回复：J/`index23-readonly-source-tree-mountfix-prepared.json`，SHA256 `5893d930a1e48286719626cbfaea61f4c0a99fa08f5bd4824b40b337a3706aa9`；仅换原来源完整只读树和新output-readonly-source-mountfix-once目录，执行源码0b2d…不变。准备及失败保全报告SHA256 `c8cd7083a0b8f385b5a3905ac6f5699a983f35048084e02d092c7770272bad1a`。新失败不自动重试。

> 当前入口选择：最终索引须使用 J/`join0040_source_bound_index23_local_transport_prepared.py`（SHA256 `0b2d53094c700b8a3facaaba84ef71ff3e4d15be405b9d33589f4bbec12d3cd4`）及 `index23-local-transport-mount-plan-prepared.json`（SHA256 `12a90ff1f1368ad7e4109cae1b4df811649eb4522bcfad31c321349c61d2d4d0`）。旧索引入口会要求本次不存在的SSH退出字段，不可执行；不得补造该字段。以下历史段落及末尾修正说明均须一并阅读。真实来源、收集、外置读回和正式索引尚未完成，不得提前增加计数。

本文件是已准备入口的索引，不是完成证明。当前正式teacher1337仍22/46；原来源客户端exec session67509不可重跑。C=`/home/lbin/rbf-approved0040-terminal-collection-preparation-20261106`，J=`/home/lbin/rbf0040-index23-preparation-20261106`。

## 1 原在途流终态

原接收器PID1161087；源证明目录 `/home/lbin/rbf-approved-teacher2027-cohort-and0040-source-checks-20261106-once/0040-original7-stream-once`。必须有完整complete-source-seven.json，全部7路径/size/原SHA与source_preparation SHA7d1f3e7720b9ade8ffed70350649a7064af57f60033c7d1e75999148fbb91625一致，精确EOF通过。原成功数值容器e10debded922b03538693e17d849cbe595905282dcb097f06e56ca863d030908不得停止或重启。

同时确认原session实际exit0；Mac原`0040-original7-opaque-client-observed-once`的client-success、实际exit0和前后source metadata都已真实产生。7个客户端小文本的精确文件名与manifest协议见0040-terminal-collection-checklist-20261106.md。只传这些真实字节到C/original7-client-evidence-once，由112核对完整SHA；不在Mac计算、不提前制造成功记录。

## 2 单次完整Linux收集

C/launch0040_collection_once_clienttext_prepared.py，SHAeb0b0c62f6c7effc7cb2e63ac047b5d6a197879dfcdb160fa9dbebe7626dc916。语法、16项客户端协议控制通过，尚未执行。CLI `--expected-client-manifest-SHA256 <实际112核对值>`。

旧433052ad版本保留但不再作为启动入口：实际Zsh print输出单行scope/success，旧准备版按多行解析。新版本仅将两处改为严格单行字节匹配，其余源码可逆还原，原Docker参数构造未变。实际客户端未改动；16项测试只有合成小文本，不证明原客户端已成功或来源流已结束。

它调用固定preflight SHA6e3f8e74271e2cb48edab320ec1a81b69ed48ba2fe140da45e11383d4f6ea49a及计划SHA25f5eefb51c732418d83d9a16e63abf8c5b8ca99c39ffad3461641227d1e4c10，fresh磁盘/MemAvailable门槛须实际通过，排他claim后仅创建一个保留容器。核心收集器SHA61e745ab76e0f318f85eacc40ef49e5e5c41973b76126387e1efc81baccae07b，原postcheck SHA61a6ea943a1206fdb77364ee12ac2a8ba7f1291db4f0b4df47effeeb4e8467bc。

实际容器完成后检查C/final-collection-on112-after-source7-once/first0040-complete-source-target-factor-readback.json及全部小结果SHA。193事件、183852标签、2633因子行及六项原检查/1e-8、原因子1e-4必须完整通过；不能以容器启动或exit0单独替代收集证明。

## 3 小结果外置保全与独立反向读回

C/preserve0040_small_outputs_prepared.py，6334字节，SHA05db218c2c7e4c6c19d4e12a92ad4b4240d9fd3f0d8716933154df4bf9bd5525；12项合成完整性/拒绝控制通过，实际pack/transfer/verify-saved均未执行。

先以实际collection证明SHA执行pack，再将所得小tar与元数据以不透明cat/SSH保存到Mac外置新目录；Mac不计算SHA、不压缩或解包。保存文件反向送回112的verify-saved，严格整包size/SHA、所有成员size/SHA及精确EOF通过后，才产生C/external-preservation-once/saved-external-full-SHA-proof.json。外层还需真实客户端exit0与保存文件前后stat稳定。不得直接把112原tar喂验证器冒充Mac保存副本读回，不复制大数据库或原压缩输入。

## 4 原审计与完整索引汇合

使用J/index23-local-transport-mount-plan-prepared.json及J/join0040_source_bound_index23_local_transport_prepared.py，完整SHA见文首。旧index23-execution-plan-with-source-prepared.json和旧1f9c6b1f入口仅保留历史，不执行。新入口CLI仍要求实际三份source7、collection、external证明SHA。全部旧源挂载只读，新输出和readback空挂载点已预建；未来collection/readback实际存在后才挂载它。

旧22索引SHA4e11854a880c67f597ad26626d8004542cb7c325b79b20242f24292e800ff7a5。原checked_prior/audit/join库完整SHA及旧行不变断言保持。旧22在新mirror只读命名空间实测exit0，报告SHAf43672cfafe9d2df6185ca21ae9510d96fdb2efb993e5fbfbad9cab5f277d3ab；零SQLite连接、无网络、无数值模型调用。这不替代新0040审计。

正式运行仍须匹配原冻结e5镜像、1000:1000、只读原来源和正确挂载，保持现有容器。成功后预期一致性计数是23序列/3227事件/2467556标签/78314因子；当前不得填写成已达成。full46、priority training、A–K标志均须false。容器内P开头逻辑别名并不证明Mac物理文件已解包；Mac归档完整保全与112实际来源位置须分开记录。

正式索引在第一份小结果归档之后才产生，因此还须单列保全：`formal-index-join-once.json`、`source-bound-path-mapped-collection-receipt.json`、`original-collection-metadata-audit.json`及新23序列索引。由112计算四文件实际size/SHA，原字节保存到Mac外置新目录，再将实际保存副本流回112逐文件验证完整SHA与精确EOF，核对客户端真实成功及前后元数据稳定。不得把首份收集归档的读回证明当作这四份后生成文件也已保存；不在Mac解包、计算SHA或伪造逻辑别名文件。该步骤未执行，正式计数仍22。

## 5 后继

上述完整收口后再新鲜112资源准入，按原19文件/676780809字节范围暂存0094；0094为198事件、11314因子行，不套用历史188/4964文字。0094原7文件反向流读的具体条件请求仍待回复，不能由数值系列授权推定已批准。

任一实际新失败保留结果、不自动重试。当前V2V仍独立生产；其后继还须完整归档及整机资源交集准入，新增八卡占用不能忽略。teacher2027发布和其他待批控制不由本交接获得授权。全程不在Mac计算、不删除源数据、不提交或推送。

### 2026-11-06 22:36 北京时间：客户端证据接收协议检查

112 上接收入口 `receive0040_actual_client_metadata_once_prepared.py` 已核对 SHA256 `039d5d3e1c28c2753383f28829fec72cbde1d3fe6737d5309500829ef6fd4ba5`。独立合成目录 `client-metadata-receive-synthetic-once` 的12项检查通过，涵盖有效七文件、截断、额外EOF、顺序、重复、零长度、非标准头、超长、路径及已存在目录拒绝。报告SHA256 `ed707410972de7b071e4db2c90e468a1dcd7e3f26be153c7bf2175ea57f9511c`。未接收实际客户端材料，未执行Docker或最终收集。

22:35:57 实测原接收PID1161087仍存在，五份文件完成，完整七文件证明不存在，failure.json不存在。仍须等待原传输句柄完成；不得重跑、合成实际成功材料或提前执行最终收集。当前传输从20:05:32起约2小时30分24秒，不能把数值阶段10小时01分47秒与传输混算。

### 22:48 正式索引入口替代：明确本地收集退出证据

**不得执行旧 `join0040_source_bound_index23_prepared.py`**：旧原审核器semantic_gates要求actual_SSH_exit_code，而0040本地收集收据正确地只有actual_local_collection_exit_code及legacy_SSH_transport_used=false。未执行实际汇总前已发现，不得伪造SSH成功值。

新的112准备入口为 `/home/lbin/rbf0040-index23-preparation-20261106/join0040_source_bound_index23_local_transport_prepared.py`，SHA256 `0b2d53094c700b8a3facaaba84ef71ff3e4d15be405b9d33589f4bbec12d3cd4`；兼容模块 `local_collection_transport_adapter_prepared.py` SHA256 `e97cfc74614485485371f2da4e3e966785b52ac6a4f39f68ff77e26b03e6fdbc`。原审核源码SHA严格绑定且文件不改，仅在内存替换semantic_gates的运输退出assert：0040要求严格int零退出、禁止SSH字段、legacy_SSH_transport_used严格false、来源SHA格式；其他序列保持原SSH检查。其余数值断言和audit函数正文不变，输出明确声明semantic运输检查已适配，不再笼统声称原audit全部未修改。

16项合成协议检查通过，报告 `/home/lbin/rbf0040-index23-preparation-20261106/local-transport-adapter-preparation-once.json` SHA256 `72ab7b2684cadca077efa522883016f525914e13632a19a7087942ffcb9660fb`。初次准备的镜像路径查找错误已在报告留存；没有调用实际audit或join。新的挂载方案 `index23-local-transport-mount-plan-prepared.json` SHA256 `12a90ff1f1368ad7e4109cae1b4df811649eb4522bcfad31c321349c61d2d4d0` 增加整个准备目录同路径只读挂载，以使容器能加载此兼容源码；原来源挂载保持只读。正式收集、外置完整读回、实际源SHA及新鲜资源门槛仍全部必须通过。尚未生成23序列正式索引。

### 22:51 运输兼容的完整合成报告判定对照

112原semantic_gates与本地运输适配版通过17组成对检查：规范报告通过；16类事件数、因子行、两种容差、后置来源、task身份、六oracle完成、标签数、capacity-undecided、fresh-state容差、完整序列标记、原因子行、存储SHA及终态篡改，两侧均拒绝。报告 `/home/lbin/rbf0040-index23-preparation-20261106/local-transport-full-report-synthetic-parity-once.json` SHA256 `a702fcef5115b811abb6f51e6a6e1fb40fad25a1644a251aff8dbf550e9c01c8`。合成common.original_postconditions明确模拟，不证明真实来源或数值合格，不替代实际终态收集与正式审核。未执行真实汇总。

### 实际客户端七文件传输命令准备

本目录 `0040-actual-client-metadata-opaque-send-prepared.zsh` 已作为纯文本准备并读回审查，尚未运行或作Mac语法计算。它只stat并cat实际客户端七份小记录，输出明确长度的FILE帧及END；只接受由112计算的实际七来源证明SHA，112执行前先核对接收器039d…完整源码SHA。接收器本身先要求完整来源证明，再用独占目录接收，最终由eb0…启动器的纯client_closure检查真实退出码、原记录及来源前后元数据。不得补造缺失文件；不得在原67509句柄未成功终态、七来源证明未通过时调用。一次执行失败不自动重试。该脚本不执行Docker、最终收集、SHA或模型于Mac。

### 23:03 冻结镜像内运输适配加载通过

112新只读控制容器`rbf0040-local-transport-frozen-load-once`使用原e5镜像、原user1000及只读准备树，通过原join/audit导入和新运输兼容绑定，实际退出0（23:03:19.177755–23:03:19.337435北京时间）。原audit函数对象保持，SQLite连接0，无torch导入；未调用真实audit/join、数值oracle或模型。小报告 `/home/lbin/rbf0040-index23-preparation-20261106/local-transport-frozen-load-once/result.json` SHA256 `ace65f96b87944758d4b232ce0f7f3f04a3fee34f7017c18b7f678e9be31e4b6`，加载入口SHA256 `c6daab1f13967c912445fd0f4beb62129924aed8f21bd47b09761a8c2e83dc97`。此为软件连接验证，不是0040正式验收；前驱各项门槛不变。

### 外置保存两段纯字节编排准备（尚未执行）

同目录已准备 `0040-small-result-external-opaque-save-prepared.zsh` 和 `0040-small-result-external-reverse-verify-prepared.zsh`。前者只在112已有实际合格pack后，以新独占外置目录保存prepared-archive、member-manifest及tar原字节，不计算SHA或解包；后者以实际112 manifest SHA为参数，把Mac已保存tar流送回112，先锁定05db…保全核验器源码，再执行原verify-saved整包及成员SHA/EOF检查。保留开始/结束与文件元数据；部分失败文件不删、不自动重试。反向成功后仍须由112或人工检查保存文件前后元数据一致及实际远端proof，不能只凭客户端exit0正式汇总。两脚本仅文本准备，未在Mac作语法测试或任何数值计算，当前未开始保存或反向读回。

### 23:39 首次正式汇总失败证据外置保全通过

失败请求、stdout/stderr、实际退出、容器身份（不含环境变量）、挂载修正计划及保全收据共7成员已保存外置 `0040-index23-failed-attempt-preservation-20261106/failed-small-results.tar`，30,720字节，SHA256 `0cda126b3101a10b74c1d8f22b2133ff289e24accc6888d7208e721e74f5c525`。实际Mac保存副本反向由112原verify_stream完整核验整包、全部成员及EOF通过；证明SHA256 `18d9d442e5ec7099c1809519ccc9b5f4049c368a2c7b2966f6412c86b9a659f9`。Mac仅传输；保存文件前后stat已留存。未重跑正式汇总或数值，挂载修正版仍待批准。

23:40在112对实际0040收据及冻结audit/join源码做静态字段连接检查，除已明确适配的SSH→本地退出字段外，全部18个直接读取字段均由实际已验收据提供；未调用真实audit/join。报告SHA256 `41c2e7381cb8a40dfa4053d5d85ff0ed912bfc626951877b6a465f4680b6ff1b`。挂载修正版仍未执行，不能以静态字段覆盖代替正式汇合。
