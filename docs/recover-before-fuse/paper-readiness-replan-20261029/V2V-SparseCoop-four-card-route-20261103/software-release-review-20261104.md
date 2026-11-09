# 四卡原环境软件复验：发布与单次执行审阅稿

**本审阅版本已暂停发布，不可执行。** 11月4日00:44复用112现有原Python3.8/Torch1.13只读环境的软件检查中，计数和toy状态两项通过，但loader报告因相对路径失败；旧组件包SHA862f14c5…包含此缺陷。先前10795字节包/13698字节入口的具体发布执行请求已撤回，即使随后收到旧问题的批准也不运行该已知失败版本。详见RELEASE-HELD-20261104.json；失败23成员已完整SHA保全。下文保留为当时审阅稿，不是当前就绪声明。报告路径修正版已在另一目录通过独立标准库跨目录单元验证；失败的原完整矩阵未重跑，旧包未重打或发布。原硬件四卡指令及其他用户授权不变。

本项只为SparseCoop后续四卡资格建立原Python3.8.20/Torch1.13.0+cu116软件组件证据，不包含真实样本、作者模型、FP16模型一步、完整训练或正式资源比较。V2V仍沿自己的四卡入口和原准入门槛；本项不消耗V2V生产/数值次数。不禁用CUDA，不停止、抢占或修改现有任务，不删除来源或失败证据。

## 已准备的具体资产

组件包four-card-runtime-components.tar.gz：10795字节、9成员，SHA256 `862f14c5dcb8e40de0bb63030c0d5e9dbb78e5bf3fdb798e976a2045f6a40c94`；112逐成员完整读回和Mac保存后反向SHA通过。

ClearML新软件Task脚本original_runtime_components_bootstrap.py：13698字节，SHA256 `1359d4586e454045f1ba74f7d6404941f2ad86eea95a6076db222bc6e3b5f525`。组件包需向既有FILES `http://10.100.34.118:8081`发布并完整HTTP SHA读回；该脚本需向既有API `http://10.100.35.118:8008`创建新Task提交。两项均尚未授权或执行。最终版将原运行时子进程PYTHONPATH固定为已验组件目录，避免继承SDK工具或agent库路径。

支持源码均已在112准备并验证相应软件协议，尚未执行真实SDK流程：

| 文件 | 字节 | SHA256 |
|---|---:|---|
| publish_runtime_components_once.py | 5627 | d5e3d2e4da7eb063bcb79ce4f42a6563f159352dad6b06cf0432844bd80f436e |
| dispatch_runtime_components_once.py | 7639 | 4dc8e0fc2d690be57feb8b4f5e0ec22bbd2beda331cc3548eeb39afb9a8af0d1 |
| four_card_queue_guard.py | 2236 | 353f3f2167d498e0b9feacd047d3d03118e3d8efcba82c5317f0aee5049c9b27 |
| collect_runtime_components_once.py | 9089 | 2f760e07edb41f442cae2d4b396fe079f167e5950cbda336905fefebbc1fa78b |
| verify_runtime_component_outputs.py | 8249 | 22e6dd8e3c20dda70722200a2759f2330b26bb14f1dc09884cb3431f9267a43c |
| candidate-worker-bindings.json | 607 | 5efa477751f1b5ddd7581e3c0efca8ea509f40335990a07e4e04be93f1c7521c |

最终同目录版本在112：`/home/lbin/rbf-fourcard-original-runtime-software-release-final-20261104`。最终八文件SHA交叉绑定及源码AST复核通过，收据final-release-readiness-on112.json保存反向SHA20b6e92c1874728ff4018f38bde57314b5c1108ae051e2d96b2fc4591b8a9ae6。没有真实grant，只有本审阅稿；自动续行不等于批准。旧目录和各软件控制保留。

## 执行与收集顺序

1. 具体授权后，发布器新建空script的发布Task，不探测/上传周边仓库；只上传固定10795字节组件包，完整HTTP读回成功后才完成发布。
2. 一次性投递器核对全部源码/包/候选绑定及实际grant；两次刷新四/八卡worker和队列交集，原e5容器合同完整继承。只在存在不重叠空闲四卡候选时创建/入队一次；已有同名或once claim则拒绝，不自动重试。
3. 共享队列由任一组实际接取后，按自身worker映射重新核对真实四UUID、计算进程、原生缓存挂载和保存空间。历史八UUID/前四探针只提供待复核身份，不提供新鲜容量资格。
4. 从原FILES读取三份已发布原运行时资产，约2.45GB压缩字节；逐包完整SHA、成员/链接/跨包覆盖核验，实测容量通过才恢复。保留原版本与CUDA11.6库路径，不安装新模型依赖。
5. 统一入口核对真实解释器和六源码SHA，执行计数、CPU toy状态隔离与loader接线三项软件检查；原数据流逐batch、真实四GPU模型一步、完整训练均保持未资格。
6. 新Task终态后，收集器只下载注册小报告/日志至112，前后完整注册绑定相同、每文件完整SHA通过后运行独立读回器。成功路径要求13份输出；失败/缺失留证，不重启或重跑计算。此新软件读回的容量合同为实际小产物总量加512MiB，不能用于降低teacher最终大收集的原门槛。注册小文件总量上限256MiB，另保留来源绑定和收据；获批后以不透明字节传输保存到Mac外置/Volumes/Data/test/recover-before-fuse/SparseCoop-fourcard-runtime-software-collection-20261104，并由112逐文件验证保存反向SHA，Mac不计算。

## SDK工具环境与未完成证据

112系统Python未安装ClearML；普通venv初始化因缺ensurepip未完成，残留保留。已有e5镜像的pip已将官方PyPI ClearML2.1.5与公开依赖安装到独立/home/lbin/rbf-clearml-sdk215-site-20261104，不改系统、镜像或冻结模型环境，安装/审计容器及日志保留。初次附加依赖审计脚本因未安装packaging未开始，随后用原镜像pip内置解析器离线检查工具目录32项依赖全部通过，没有另装packaging。现有默认ClearML配置由SDK正常使用，未导出凭证；只读连接原API/FILES并查询原父Taskcompleted通过，未创建/上传Task、未导入torch。SDK准入收据保存反向SHA207dded52e7bb36566300f421b027d9385b7bced979ba91f836efbf59ea646e7，依赖审计a5305ec65c796a27b0d9ba456a15667e0fd59483408213880e8ff0e1dd8e120d。

目前仅完成语法、授权拒绝、合成协议和mock注册控制；真实SDK创建/上传/HTTP/原环境恢复/软件正向结果均未证明。需在完整准备后取得本项新包发布、脚本提交、一次软件执行及完整小产物读回的具体授权，才能进行实际流程。此前八卡模型失败或其他包授权不覆盖本项。
