# 0025收集入口的远端迁移缺口

> 2026-11-01更新：未执行的最终收集入口已补齐原索引审计器要求的`readback`及`all-small-files-SHA-readback.json`布局。当前版本11643字节、112计算SHA256为`2778230e1cd05ada98f1360ec039796c3b2641b807be3eff40e09ef1e85f5892`；12项控制确认原REMOTE/SEAL、来源身份及原数值条件不变。原08edee6c版本已保留，未增加尝试。详情见`0025-final-legacy-view-preparation-20261101.json`及`0025-final-legacy-view-remote-controls-20261101.json`。以下初版记录保留为历史；0025仍未启动，真实来源许可与原全部门槛均未因软件准备而通过。

2026-10-31，本次只读源码检查；未运行收集器，未启动0025，未读取7份大来源内容，未执行本机SHA或报告检查。

当前原入口为 `/Volumes/Data/test/recover-before-fuse/local-workspace/transvision/docs/recover-before-fuse/paper-readiness-replan-20261029/teacher1337-unused-collector-identity-repair-20261031/sequence0025/collect_first0025_once.py`。该入口不能在Mac原样执行：

- `main`在本机验证返回小文件SHA、读取后再次SHA，并检查manifest、contract、原数值报告及封版源码SHA。
- `common.original_postconditions(n)`在本机加载冻结检查器并执行报告条件；迁移后这些条件应在112按原函数执行。
- 最后循环对Mac上的7份原来源做完整SHA。仅有历史传输证明、大小一致或小产物通过，不能替代当前原来源检查。

原入口已经在112计算任务目录及gz的SHA；该部分继续保持远端。Mac只保留文件写入、stat稳定性检查、SSH传输、ClearML注册元数据查询和运营记录。

迁移必须保留的0025原条件：172事件、3936观测及因子行；原6项oracle，语义ATOL/RTOL=1e-8、因子1e-4；26821见证、379 typed tables；原始库SHA `749ea0db5e61af16f8f8e2b0c1402f18d1ab7498f2a69a60cd4aa20cd096cc87`、整理库SHA `b894ca4f316a96af3d3cac26b51e6b9295b3dd6f9b0212af6cb5c3c9df8e8a10`；replay-rank3注册归档size/SHA；原镜像、实际入口及退出0；原manifest所有成员与gz、保存后小产物逐字节SHA；7原来源当前完整SHA和原报告postconditions。不能因为迁移而把任何一项降为stat检查。

0016的分离模式可供复用：先收集远端终态与小产物，原来源按具体授权流式送112计算SHA且不落大副本，最后在112完成全部来源/目标/因子汇合。0025不能直接套用0016的固定容器、成员清单、rank7、来源SHA、事件数或一次性标识。

执行门槛：0016完整来源汇合通过、当前0025输入/模型与一次性授权绑定、112资源重新准入、原0025运行完成、0025适用来源传输许可及远端迁移入口验证。当前均不能从“17份控制器已准备”推断这些收集门槛已经通过。本文件不是收集器、控制测试结果、传输授权或数值合格证据，不增加完整合格计数。

## 已实现的报告检查部分

`check0025_terminal_reports_remote.py`已准备，5254字节；112计算的SHA256为`935f9e7fefb0bacdfaf317896cd4eb297daefb4927433f1a4d901343d51d2fff`。33项合成门槛、篡改拒绝及语法检查在112通过，记录为`0025-terminal-reader-remote-controls-20261031.json`。原收集器完整SHA在112核对为`64e70383da9f7bcb0222d082f5c4a1f99e8411e3ae3ed7d28b1dcc40bea9e5d4`。

新检查器仅允许Linux执行，保留原数值报告条件；实际使用时还必须验证冻结postcheck与accept源码完整SHA，再调用原`original_postconditions`，只迁移其数据根位置。合成正例使用替身回调，不声称已经验证真实0025报告或原postconditions正向数据路径。原始7来源、注册归档、全部输入/输出及保存后SHA汇合明确仍必需。

终态传输控制器、0025专属来源流式SHA入口、收据独立复核及最终汇合均已实现为未激活候选；真实正向执行和来源收口尚未完成。不能用软件控制替代完整来源收口。未更改原入口或当前一次性标识，未激活0025。

## 已实现的远端终态清单部分

`collect0025_terminal_inventory_remote.py`已准备，6394字节，112计算SHA256为`bcfecfe07dddb1cdc0360d343bfa7de604f109980a2fad2c0c00740f828b0a54`。25项合成容器身份、路径、SHA拒绝及语法检查在112通过，记录为`0025-terminal-inventory-remote-controls-20261031.json`；当前contract、manifest和报告读取器的完整SHA均由112核对。

执行时先核对调用方从原一次性启动收据取得的CID与StartedAt，以及冻结镜像、固定容器名称/入口和退出0；仍运行或失败的任务不读取目录计算SHA。之后在112核验全部manifest成员、原gz、所有小产物和冻结报告条件；结束再次检查同一容器终态。当前0025未启动，实际CID/StartedAt尚不存在，不使用合成控制中的CID或日期。未来传输控制器必须从权威原尝试收据绑定它们，不能接受任意候选身份。

这些是软件控制，不是实际容器/终态/报告收集验证。该清单仍明确要求保存到Mac后把实际字节送112校验、刷新注册归档，并完成原7来源当前SHA和最终来源/目标/因子汇合；完整收集入口尚未就绪。

## 已实现的仅传输控制部分

`collect0025_outputs_via112_once.py`已准备，9645字节，112计算SHA256为`71fe9c59af5f3b93f38ac01672aaea60699b8b0f2c2ef622d341ea098cb7c428`；21项合成收据、路径、禁止本机hashlib及嵌入源码语法检查在112通过，记录为`0025-output-transport-remote-controls-20261031.json`。尚未实际收集或访问0025容器，不称端到端正向验证通过。

候选要求原数值启动收据（mode=numeric、SSH/launch退出0、一次性标识、固定名称/入口及原contract SHA）与独立保存的原启动身份观察共同绑定CID和StartedAt，再与112的原numeric-launch.json及实际docker inspect比较。拒绝已重启容器。活任务仅报告待退出，不占终态收集输出目录；终态在新目录执行一次，失败保留且不自动重试。Mac只保存及转发小文件，实际保存字节返回112计算SHA；原归档replay-rank3注册size/SHA经ClearML刷新。

0025尚未启动，因此没有实际启动收据、启动观察或容器身份；不得用合成测试值代替。原7文件当前SHA和最终汇合、完整真实正向路径仍缺，候选不签发完整序列资格，不增加15/46计数或放行后续训练。

## 未激活的7来源流式SHA入口

`stream0025_source_SHA_via112_once.py`已准备，7402字节，112计算SHA256为`b6f0cabe624dde52e2e65c043aa0d90a125e022efd395a898abd98bbe5ce2ce7`。17项合成授权拒绝、来源清单/helper绑定及语法控制在112通过，记录为`0025-source7-stream-remote-controls-20261031.json`。原清单SHA为`c41d5c2453ed7f4250484b9a51e91d36da9644d959e524800b3bd980e7e0ccee`，7文件总计10246055314字节；冻结常量缓冲helper不变，SHA为`93c62b65d1910d70f238580207ffb6419a4fa9341197ea5571cb67847526b547`。

入口明确要求`0025_original7_Mac_to112_stream_SHA_once`适用许可，逐字节一致的7文件列表、目的地、严格整数单次执行、禁自动重试、不落远端大副本等字段；0016许可不能冒充0025许可。还要求0025终态收集已通过、同一实际容器成功退出、原清单在112与Mac传输字节一致及源文件stat稳定。Mac只打开文件描述符转发字节；原始SHA、长度和exact EOF在112核验，不删除源数据。

本轮没有取得该未来范围的真实许可，没有打开任何7来源内容，没有建立来源流式尝试目录，没有启动0025。实际保存元数据复核及最终来源/目标/因子汇合仍待真实执行；软件控制不能替代这些真实门槛。

## 已实现的来源收据独立复核

`check0025_source7_stream_receipts_remote.py`已准备，4297字节，112计算SHA256为`59aca55e4ef4b11de3518304cc003b958c0060d9fe72d68374b9dace395c3d4a`。29项合成一致性与篡改拒绝检查在112通过，记录为`0025-source7-receipt-checker-remote-controls-20261031.json`；真实收据尚不存在。

复核要求原清单完整SHA绑定、7来源精确匹配、逐项传输长度/SHA/exact EOF、不落payload、不运行模型、原始文件stat前后一致，以及同一成功数值容器终态、原contract/manifest和数值报告条件。拒绝替代scope、重复计数、改动终态或来源、漏收据及把分项冒充full46。它只证明输入收据之间的一致性；最终收集器仍须直接复核真实任务输入/产物和冻结原postconditions、刷新注册归档、核验实际保存字节，并签发单序列来源/目标/因子汇合。

## 未激活的最终汇合入口

`finish0025_source_target_factor_remote_once.py`已准备，10776字节，112计算SHA256为`08edee6ca45d25fbec709aec22fb0ae73813150657af53f882c8c5b06f1305fd`。10项合成组件SHA、边界、篡改拒绝和已保存证明SHA控制在112通过，记录为`0025-final-source-join-remote-controls-20261031.json`。这不是实际数据正向验收，0025真实数值与来源传输均未执行。

入口将全部7份组件源码严格按SHA绑定，在112刷新原终态输入、gz与小产物、原冻结postconditions、source7收据及当前源文件stat，核对原启动收据和同一未重启容器，再与ClearML注册replay-rank3归档size/SHA汇合。Mac保存的实际小文件和最终证明字节仍送112计算SHA；源文件stat与归档注册身份在前后复核。输出为单序列证明和`complete-join-receipt.json`，任何失败保留且不自动重试。

将来只能在0016完整来源汇合、0025原一次性执行与输入/模型/环境准入、适用具体来源传输授权和全部原条件实际通过后使用。须有真实完整证明、其远端SHA、complete-join-receipt及无failure记录后，另行执行原完整来源/模型索引汇合，不能由候选或单序列证明直接放行46序列资格或优先级训练。当前入口未替换权威原准备记录，当前完整计数仍15/46。
