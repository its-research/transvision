# MHT原事件JSON包到JSONL表示准备

2026-11-07 00:07北京时间。MHT及正式同资源比较仍未启动，本记录不是实际输入、模型或回放资格。

## 原路径与缺口

复核既有Nov2合同：复用`run_paper.py`的paper MHT后端，不绕过旧car-only CLI对当前all-class模型的拒绝。三种子原事件桥已有3316事件/21序列来源；112相关MHT/seen-val目录中没有events.json正文，现有文件仅为源码及小收据。Mac外置原文件每份1,996,180字节，合计5,988,540字节，Mac仅stat读取大小，没有计算SHA。

原路径为 `/Volumes/Data/test/recover-before-fuse/artifacts/rbf-seen-val-forest-input-bridge-v1-20261005/seed{seed}/events.json`，原登记SHA：

- 1337：55a64e37595cc3b40f2b548ee2f0a020866b1225368df09372d62af35146ecea
- 2027：12d5e6d8a7eba8ba10be44683649f6bfdff2b861c020c9f8688a8bcd3df59773
- 3407：2ea2727f1bc44c81af68b48e7090224ec39affd7bc5007ca30ff978cd3d85f73

原桥接源码7d5f9f48e0e68a8a50b72a0472221bc5a9778f00572faff152fc4ec4aadb2093保持；原包的100ms scheduled arrival不是实测网络历史。旧source binding中的未到达源排除及原缓存资格仍须独立保持，格式转换不能代替原缓存/来源全量准入。

## 112候选及控制

准备目录 `/home/lbin/rbf-MHT-existing-events-jsonl-bridge-preparation-20261107`。

- `existing_events_jsonl_representation_prepared.py` SHA256 `640042d53541b1ca3609c415d153f812738d24c199e3d01e4d144f19e69c6ba5`。逐事件输出JSONL；sidecar保留全部非events包元数据。独立检查逐字段类型与完整值、嵌套内容、顺序、空事件、clock以及浮点符号零，不调用producer的序列化函数来替代值比较。
- 18项合成检查通过；报告SHA256 `9bfeaee2fd3c06b7903d2c8e48fc21c8a317595aa142a8b76d6a6a4ab83db405`。覆盖遗漏/额外/重排/空白事件、EOF、源SHA、跨seed、clock类型、嵌套来源、符号零及元数据/资格标记篡改。没有真实事件读取或跟踪器调用。
- `receive_and_check_original_events_once_prepared.py` SHA256 `fcee83c91a6b8dd35b1a6613b58f343f429ab48fbef75104a7298ec52e7318e8`。两个一次阶段：receive只收原三文件并全SHA/EOF；represent在新目录绑定原source receipts/当前checkpoint元数据及全部3316事件/21序列，检查输入前后SHA、输出逐值等价。
- 授权函数1正14拒绝通过，合成许可没有保存成人类授权；报告SHA256 `0cd1da7859323100c2b0bfa2234b0d26dc3e0842814ab57762f829cac45541cc`。

真实接收目录receive-actual-once和表示检查目录represent-actual-once均尚未创建。真实入口必须有单独人类批准、原source_files完整size/SHA、两个阶段各1次以及禁止MHT/model/metric/retry的严格授权。新失败保留，不自动重试；不改变预算、容差、线程、CPU/RAM限制或GPU可见性。不存在真实软件对照成功或正式MHT资格声明。

## 待批及后续

已请求原三包Mac→112暂存及单次完整表示检查的具体授权；teacher1337系列授权不覆盖这批MHT输入，未提前传输。

即使表示检查通过，仍缺缓存帧/数组完整字节、源到事件的完整关系、完整模型及运行环境准入、原Linux MHT执行合同决定、train-only资源冻结、全部回放及独立验收、每配置至少3次隔离同资源重复。SPD继续标记seen-val探索性，不推成两数据集正式比较，不以旧checkpoint/cache换常量冒充当前合同。
