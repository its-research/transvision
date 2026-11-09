# 0033终态后索引汇合：来源绑定要求

本文件是准备要求，不是0033资格收据。所有SHA、报告校验及索引计算只在112执行，Mac不计算；不改现有核验器或活任务。

## 必须先满足

- 唯一传输会话27022成功退出；112原核验器生成 `original7-stream-once/complete-source-seven.json`，7份文件的原登记SHA和精确EOF全部通过。不得根据进程IO累计推断已验字节。
- 原0033数值容器61c04adb17a4d6e60eb560a7bb9f729ecebdc15bbd757d505cd25ae786de631c保持exit0、无重启，原六项报告及3679行因子均通过原标准。
- 已准备112-only收集器实际成功，完整小产物、原压缩输入SHA、注册原rank7 archive身份和原数值后置检查全部通过。

## 汇合时必须额外显式绑定

现有最终收据检查了原7文件证明的内容，但未在输出中保存该证明的路径/SHA。因此索引消费者必须同时读取、完整核验并记录原 `complete-source-seven.json` 的路径和SHA，而不能只相信最终收据内的 `all7_original_source_members_current_SHA_reread` 布尔值。

该证明须绑定0033、1337、原7路径/大小/SHA、source_preparation完整SHA、原成功数值容器ID、Mac_computation=false、remote_payload_saved=false；冻结核验器源码必须保留精确EOF断言。保存证明的实际SHA应在112计算后进入新索引来源链，不在本准备文件预填未知值。

## 旧索引与不重复计数

原19序列索引：`0050-index-join-on112-20261102/original1337-nineteen-targets-nineteen-factors-current-source-join-20261102.json`，SHA `58473f65491fa6a1af71439b5a70c31073a2467277da923a2f8db33be20d6819`。

使用原 `join_collected_teacher1337_receipt.py` 的 `checked_prior` 验证完整旧来源链，保留其来源模型绑定、46行原数据库/成员索引身份、所有已加入报告及因子证明检查；原算术和防重复条件不变。若新收据协议需适配，只能显式适配新收据的审计入口，不能绕过旧索引检查或修改旧报告头。

0033应由原冻结行核对189事件、3679因子行、原数据库SHA a5f9761d15d356956e397462000b5e8b749e996de48a4a53f2b3cd8f23e07986；标签数须从完整校验后的原targets报告读取，不由准备文件代替。

成功后写入新目录、新索引，保留19索引及全部旧失败。完整46序列、优先级训练和A–K完成标志仍为false；不因单序列汇合放行训练。不提前把准备状态称作20/46合格。
