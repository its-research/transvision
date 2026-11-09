# V2V / SparseCoop：用户四卡路由指令

用户2026-11-03明确要求“V2V/SparseCoop 使用四卡”。后续两项只按四张物理GPU准备；旧八卡入口、失败结果和已验组件保留，不修改活任务，不让空八卡worker跨占现有四卡任务。

## V2V：已有四卡入口，纠正过宽整机等待

原V2V-unit6-held-original-complete-preparation-20261030/dispatch_unit6_once.py已使用GPU4-2080ti队列fba9d89ec29e485a9f35df12dd7639ad，候选为ubuntu26-2前/后四卡，按物理mask交集排除忙组。它没有要求另一组四卡无Task；只要求目标组和重叠八卡worker空闲，以及两队列无在途排队。故此前将V2V也一概视为“八卡必须等整机释放”的描述过宽，已纠正；不修改这个原冻结dispatcher或它的SHA锁。

22:52真实平台注册：front0–3有e9614d78…，rear4–7无Task，两队列为空，三worker有last_activity_time。rear是当前候选，不是已经通过真实GPU UUID/计算进程、共享CPU/RAM、原生磁盘及完整产物容量准入。旧probe会过期，源依赖/完整收集与具体发布许可仍保持。原unit6仍304事件、38754因子行、原模型/输入/预算、语义1e-8及因子1e-4；不消耗尚未使用的生产/独立次数。

## SparseCoop：四物理卡的新资格版本，不能伪装八rank通过

原qualify_eight_rank.py明确world_size==8，NCCL每rank唯一GPU；author两端配置global batch48、每GPU6、24 workers/GPU，训练BN norm_eval=False且requires_grad=True。vehicle/infrastructure grad clip max_norm分别25/1，FP16 loss_scale32；这些原差异不可抹平。

四卡候选优先评估4个物理rank、每卡microbatch6、两次累积、有效48。它目前没有等价资格，不能只改world_size断言或把原配置num_gpus改为4（那会令每卡batch12并改变BN）。也不把8个NCCL rank直接塞进4卡，官方通信合同要求进程独占其GPU。

必须在原模型与冻结输入上验证：完整48样本及原logical sampler/RNG顺序；可训练BN更新和每rank时序/instance-bank状态；跨rank损失归一化；两微步之间不提前step/clip/scale-update；原FP16 unscale、梯度裁剪、优化器状态；以原optimizer step计数的LR warmup、800epoch预算、checkpoint/evaluation间隔。普通梯度累积会改变runner.iter与BN更新次数，不能称天然等价。必要的logical-stream状态隔离和计数适配须先资格对照，不能缓存数值答案或使用GT修补训练。

若不能在原标准下验证这些不变量，不启动完整训练，不降低容差或默默改有效batch/损失/预算。四卡硬件路由本身已有用户直接指令；它不自动批准新源码包发布、此前失败重跑或已经用掉的单次数值授权。旧八卡失败/GC/heap输入组件继续保留，四卡实测不冒称原八卡完整资格。

## 当前操作状态

### 2026-11-04晚间：真实集成还存在调用顺序约束

只读复核现有组件：LogicalStateBank.finish要求当前逻辑流backward_complete=True后才能切换下一流；LogicalCountTable则须收齐全部八逻辑流同序的实际forward计数后seal，才能verify_replay取得完整batch分母。因此不能把两个已通过软件控制的组件简单套在普通“两微步”训练循环上，就称真实训练入口已经接通。它们验证的是各自协议，不是跨组件的真实模型前向/collective/反向时序。

正式资格入口必须明确解决该依赖：如果采取计数预遍历/重放，须证明预遍历不提交BN、instance-bank、去噪、sampler/RNG或优化器状态变化，真正重放使用同一原参数/输入和逐计数身份；额外forward及其内存/耗时须如实计入成本并核对是否符合原预算，不能隐去。若采用保留多流计算图的方案，须先解决原状态切换与autograd版本/共享buffer、峰值显存和原独立backward合同，不能静默移除finish门槛。当前没有选择、实现或执行任一路径，也没有新模型资格、训练授权消耗或Task提交。

下一道实质门槛仍是原完整48样本逻辑batch的真实loss、BN/时序状态、梯度、FP16 unscale/clip、optimizer/LR时钟独立对照；不得退化为再次跑通用计数/状态合成例来替代它。ubuntu26-2当前V2V在途，四卡软件成功报告不赋予抢占或新失败重跑许可。

已更新resource-routing-current.json和新fresh-worker-queue-observation.json。尚未改原冻结训练源码、发布新包、创建/排队Task、启动模型或进行Mac计算。后续准备四卡SparseCoop资格候选及相应独立读回；V2V沿已有四卡route继续补齐已列来源/保存门槛，不再等待无关前四卡清空。

参考的官方实现约束：
- PyTorch distributed NCCL每进程独占GPU：https://docs.pytorch.org/docs/stable/distributed.html
- MMCV原梯度累积hook及BatchNorm警告：https://mmcv.readthedocs.io/en/v1.4.3/_modules/mmcv/runner/hooks/optimizer.html

这些网页说明风险与实现机制，不替代锁定运行时上的实际软件/数值资格。

## 22:57 首次112合成数值控制完成

已在112执行独立小型有理数控制，未导入torch/MMCV、模型、权重、数据库或GT，不调用GPU、不增资源caps或禁用CUDA。报告synthetic-training-state-contract-on112.json共1365字节，source及Mac保存反向SHA均3c67880e8b2e8b74b2caef28c2d4a11c67edaaf1557e46092c43e08b5cfe432e。

该反例的原单次BN running mean=7/20、variance=5/4、num_batches=1；同一物理rank共享BN两微步变为933/200、59/40、num_batches=2。八逻辑stream各自保存BN状态则在四物理rank的合成调度中完全复现原值。跨rank归一化的同一例，原完整逻辑batch loss=17/3、分波归一化后平均=63/13，改为完整逻辑batch归一化恢复17/3。三个optimizer step不能用六个micro runner迭代的LR时钟替代。

这些是确切反例/协议算术，不是实测SparseCoop loss、梯度或模型收益。真实框架BN/FP16、sampler/RNG/temporal cache、四卡完整数值资格均明确false。后续资格入口需在原锁定环境/原输入上实现并验证这些状态和global-reduction约束，不能只换累积hook就启动完整训练；原1e-8/1e-4等专项标准不变。

## 23:10 源码状态映射与完整逻辑计数软件组件

对九份既有作者公开源码的AST/完整SHA核对已在112完成，未执行源码、未把源码持久保存在112；报告author-state-and-reduction-static-audit-on112.json共15405字节，源及保存反向SHA ffc14e655097d6df123843918d03a065c567174177b228c83703471f2874010b。该控制没有重验原234MB归档，不冒称新全来源资格。

真实head两个归一化源位置为sparsecoop_head.py:886的检测positive count、:1106的denoising valid count。instance_bank有45项构造外赋值记录，cross_agent_interaction8项、target2项、group sampler5项；这是写入记录而非唯一状态总数，且不覆盖框架/扩展的间接修改。缓存包括原cached_feature/anchor/metas/confidence、跨端ID映射及去噪状态，不是只保存BN buffer即可；普通state_dict也不保证包含这些Python状态。原GroupInBatchSampler用world_size决定全局batch/stride、每槽group generator与numpy随机skip/flip，四rank重新切分必须保留原逻辑八流来源。

新增logical_count_table.py（4181字节，SHA59da52963591799845a27b6c99c91248a5c28279dd1e8ea38195ca5ce4323317）与check_logical_count_table.py（4332字节，SHA7ff9252070fcc93e30d695affa8e2ff18294bfe22712bfcfd9bb7ed4e450225b）。仅是通用标量计数协议，CPU Fraction无torch/模型/数据/GPU/SDK。112新目录/home/lbin/rbf-four-card-logical-count-software-20261103真实软件控制exit0，18拒绝检查及完整8计数独立均值/到达顺序/零clamp对照通过；报告809字节SHA963a944d793b598b6f41817dbdb56a427f18e8b3663b606d97f90d2f40ee7de5，保存反向SHA通过。

该组件固定4物理rank×2逻辑流、每完整逻辑batch调用序列和8个输入身份、优化器step；拒绝缺rank/调用不齐、错source/step/site/count、重复登记/封存/越界/无法证明的FP32整数累加，不截断数据或降低预算。它没有运行或替代原loss、FP16、梯度同步/优化器、实际sampler/RNG或BN/时序状态隔离。真实四卡SparseCoop资格仍false；将计数表接入真实训练前，还需完整逻辑流forward/replay和全局collective协议、状态隔离与真实框架独立一步验证。不可仅凭此软件组件启动完整训练。

## 23:21 逻辑状态调度器的真实框架组件测试完成

实现logical_state_bank.py（3344字节，SHA8d21522ab34cda4887d81eba8484f089c1dc847fa2c4f9840f8c5f54c0809e04）与check_torch_logical_state_bank.py（8631字节，SHA51d09af11b8b512198dea902696c310c56053de2266f8a30bbf10eacd424aa8e）。前者是框架无关的opaque state/RNG回调调度器，固定每物理执行器拥有两个逻辑stream，先完成各自backward再切换，二stream完成后才接受一次optimizer update。它不自动发现真实SparseCoop全部状态，也不把调用者布尔标志当独立backward证明；未来实际入口必须给出完整源绑定回调和轨迹。

112新目录/home/lbin/rbf-four-card-state-bank-framework-control-20261103，使用原85525aef…镜像，未修改镜像/安装依赖；保留容器rbf-fourcard-state-bank-software-20261103及全部源/日志/报告。实际容器StartedAt2026-11-03T15:21:16.31120426Z、FinishedAt15:21:20.139474264Z，exit0/OOMKilledfalse。没有CPU/RAM/线程caps，--gpus all保持设备暴露；玩具网络全部CPU FP64，没有真实GPU计算或SparseCoop数据/权重/训练。

真实Torch1.9.1+cu111小型网络对照：8个独立持久逻辑模型对比4个执行器各承载2逻辑状态，分别在无/有activation checkpoint的两组中各执行3次optimizer update。逐流检查trainable BN/num_batches_tracked、dropout、Python/NumPy/Torch RNG、temporal cache及identity-map，完整逻辑batch梯度、clip后参数、SGD momentum一致（合成FP64 ATOL/RTOL1e-12，不改变原训练/事件阈值）；6项错顺序/提前切换/未backward/提前或重复step等拒绝通过。

报告state-bank-framework-controls-on112.json共1149字节，SHA b6b3c17887977d5fde467c54fb7bced1ade9c7a3c499da07a08db7b88198331a，Mac保存后反向SHA在112通过；执行记录state-bank-framework-execution-on112.json。来源源文件前后SHA相同。

范围仍仅软件组件：实际8GPU或4GPU并行、原Py38/Torch113、原SparseCoop head/sampler/FP16/global collectives/真实optimizer一步均未通过。opaque回调在toy上全覆盖不证明原作者45等写入/框架间接状态全捕获。不可据此启动完整训练或宣称原八卡等价；下一步需要在锁定原环境和真实四卡入口接入源绑定状态/计数适配，并独立验证真实一步。

## 原运行时声明门槛的软件控制完成

重新读取原作者environment.yaml，原Python=3.8.20、torch=1.13.0+cu116；旧恢复入口使用frozen-runtime.tar.gz（2392552132字节，e27c561e…）及python-base-libraries.tar.gz（53072125字节，0b55237d…），实际执行器为/work/native-py38-venv/bin/python。没有运行旧八卡bootstrap或以112已有Torch1.9.1替代原版本。

新增original_runtime_contract.py与check_original_runtime_contract.py，在112新目录/home/lbin/rbf-fourcard-original-runtime-metadata-controls-20261103执行exit0：一个正例、24个缺失/错误版本/类型/卡数/逻辑流/batch拒绝控制通过。报告runtime-contract-controls-on112.json完整读回及保存字节反向SHA均为c6178285bacae3b84f99f4e5eb9c1f1cf385361e4376ef2b6516ca6b8746958e。Mac仅文本编辑及传输，所有控制与SHA在112完成；未调用框架、GPU、真实模型、原数据或更改活任务。

此门槛仅核对声明，不能独立证明恢复包当前字节、运行解释器/框架身份、四卡UUID或实际训练语义。真实原环境组件复验、作者模型状态回调接入、完整逻辑batch归一化/FP16优化步骤及四卡独立数值资格仍待完成；发布与新原生计算授权门槛不因这些软件控制通过而消失。V2V继续原四卡路由，完整来源补充及产物保存门槛仍保留。

## 原DDP与随机流接入约束补核

再次逐行读取原qualify_eight_rank.py：MMDistributedDataParallel明确broadcast_buffers=False，find_unused_parameters沿原cfg；不是默认广播BN buffer的DDP。每rank原Python/NumPy/Torch/CUDA均seed1337，不能把先前toy中1337+rank的压力测试初始化直接移植到真实入口。真实四卡适配须保留八逻辑流的原seed及原sampler/worker派生规则；模型构造、init_weights和DDP初始参数同步也需保持原调用次序。

原Fp16OptimizerHook(distributed=True)由IterBasedRunner注册，runner.run驱动原train_step及loss parse；因此四卡候选不能仅在模型外部做两次loss.backward，再沿用旧hook而重复allreduce/unscale/clip/step。实现前必须分别绑定原loss标量归一化collective、DDP梯度reducer、FP16 master参数/optimizer hook与runner时钟；每原48样本只允许一次原optimizer更新。原一步资格max_iters=1不能冒称800epoch完整训练预算已执行。

上述是原入口的直接文本审查，不是新SHA或框架执行。此前组件报告仍只证明toy范围；没有新增实际模型资格，也没有修改旧入口或活任务。

## 八逻辑数据流构造组件已准备并在112检查

原GroupInBatchSampler构造器本身接受world_size/rank显式参数；原builder也按rank派生worker种子（24*rank+worker_id+1337）。新增logical_loader_pair.py注入原来源已准入的采样器、DataLoader、collate和worker_init_fn回调，固定物理p拥有p及p+4逻辑流，各保留batch6、world8、24worker、原collate与种子公式；拒绝逻辑rank缺失、错卡及共享同一可变dataset实例。该组件不创建iterator或启动worker，不修改原作者源码。

112目录/home/lbin/rbf-fourcard-logical-loader-wiring-20261103真实exit0；四组共八loader的回调接线与六项拒绝检查通过，报告logical-loader-wiring-controls-on112.json源及Mac保存反向SHA均5ff5dc2d12f2249249eb9c2c85ed02efbea87f3f7184726b4ed0b60e5f81ab52。未导入原框架或运行原采样器，未证明实际样本字节/RNG或四卡模型资格。Mac仅编辑和传输。

保真路线仍总计192worker，不能宣称四卡天然解决既有CPU内存问题；调用者还必须在迭代器创建/前进时隔离主进程NumPy/Torch随机流，并验证两条真实DataLoader的prefetch/worker RNG和逐batch完整指纹。若后续减少worker、改prefetch或共享dataset，须另做原完整输入/RNG对照，不将其混入这次原配置组件。该组件尚未发布到ClearML，不消耗任何模型尝试。

## 23:35运行时复用与磁盘实测

2026-11-03T23:35:45+08:00经SSH只读检查：112根盘Available=6427922432字节，Use%=99%，无正在运行的Docker容器；宿主/work/native-py38-venv/bin/python及/opt/cooptrack/bin/python未发现。保留85525aef…与e5b249d9…镜像，未删除、启动或安装依赖。这不证明每个停止容器中都不存在原环境，也不把已有1.9.1镜像当1.13环境。

原恢复还使用已发布public-native-build-prerequisites.tar.gz（1357245字节，SHA f575ef3795b06cc670009879348a3563ee8bdc789e97ae12ffdb8d7c6d5ea727，Task535500e0eece4125919040355b3ebd43）；仅仅读回原发布清单，未重新下载或验证服务器当前字节。其135头文件和10库属于原Python/Ubuntu公共依赖，不需要以新依赖版本替代。

当前不能在112的6.43GB余量下假定2.39GB运行时压缩包连同全部解压、组件输出和保留余量可容纳；未下载或解压该包，也未清理其他镜像腾空间。原Py38/Torch113组件复验应准备到空闲且不重叠的四卡ClearML worker，恢复时完整核对上述原资产及原目录/库路径，并先量出压缩和解压保存需求。该准备不包含模型执行、编译原生算子或旧八卡bootstrap运行；实际新Task源码发布与执行仍须具体许可。

## 运行时容量检查器准备与软件控制

原runtime-package.json仅登记压缩2392552132字节，没有解压payload总量，不能由压缩体积直接推算准入。新增inspect_runtime_archive.py（3355字节，SHA9c7af068f3ac6db5d1fc5de114fc1a2d3433726c4ed6f783d5df18d2d373be7d）：Linux下先完整核对压缩原SHA/字节，再流式扫描tar成员统计普通文件payload及链接/目录数量，前后核对源身份；从不解压、下载、安装、删除或运行模型。结果明确不证明实际磁盘块占用、链接目标安全、完整资源或四卡训练资格。

112新目录/home/lbin/rbf-fourcard-runtime-capacity-inspector-20261103的一项合成成功、五项大小/SHA/重复/越界/原前缀拒绝检查通过，exit0；控制报告capacity-inspector-controls-on112.json完整读回并反向SHA093977648539bf2abf3f1932bb3a5ebba1c1a16dd63607bc6b2723489d69a885。只使用自造小包，没有扫描真实运行时；Mac仅编辑、传输。真实容量扫描及四卡原环境恢复仍需进入具体获准的Linux来源/资源流程，不能把合成报告填作真实包容量。

## 原运行时组件统一执行入口

run_original_runtime_components.py已准备，6161字节，SHA0b3f3409cdf0686cdb37c109c6f5f4a5a1a85e8b99d5b81586b50102903c0afc。固定六个组件源码SHA；要求Linux、未优化解释器、原/work/native-py38-venv/bin/python、Python3.8.20及Torch1.13.0+cu116/CUDA11.6。通过后仅串行执行原逻辑计数、toy状态隔离、loader接线软件控制，每项独立日志/退出码、输出SHA及最终源前后绑定均留证。失败无自动重试，输出新目录排他创建；不访问网络/SDK、不恢复环境、不创建worker、模型、GPU或资源caps，不授予完整训练资格。

112新目录/home/lbin/rbf-fourcard-runtime-component-runner-controls-20261103核对七份源码AST通过，并使用当前真实错误解释器执行预期拒绝控制：源六项SHA正确，入口在原解释器身份门槛退出，stages为空，没有进入任何组件或模型。该次是拒绝控制而非原模型失败重跑。报告runner-rejection-control-on112.json完整读回并由112核验保存反向SHA0586398608364ceee99574ea517cdaaf676523d531b259546d4e75e9d8cfcde7；Mac仅传字节。

尚缺原环境下正向执行、实际四卡UUID/显存/磁盘准入、原sampler逐batch字节与RNG、作者模型状态回调、原FP16/DDP一次更新对照；统一入口通过也仅是软件组件证据，不能消除这些后续门槛。原Task/发布许可未扩大，未运行旧八卡入口。

## 原运行时软件组件候选包已封存

112将六个组件文件、统一执行器及容量检查器连同完整SHA manifest封成four-card-runtime-components.tar.gz：10795字节、9成员，SHA862f14c5dcb8e40de0bb63030c0d5e9dbb78e5bf3fdb798e976a2045f6a40c94。九成员完整逐字节读回通过；包与component-package-readback-on112.json均保存到Mac外置工作树后再反向流到112核对，分别为上述包SHA与收据SHA4adb0265b01a26b784e9760d664e9282364cadce581948f8b6f55cd8b13e4208。

包仅含本轮自编通用软件和manifest，没有作者模型、运行时大包、数据或GT。它不是四卡完整训练入口；尚须另备ClearML四卡物理调度/原资产恢复/容量与报告上传bootstrap及独立收集器，完成具体发布授权后才能执行。未发布FILES/API、未创建或入队Task，不把候选封包当原环境正向资格。

## 四卡原环境软件bootstrap候选已准备

original_runtime_components_bootstrap.py最终12644字节，SHA1562a161f9724388cacbb48fc5f5182fdb6dd4871c921ec8d4cc9588b3aa7509。需具体once授权参数、指定ubuntu26四卡worker、独立准入的四个UUID及原生ext4/xfs缓存父目录。它拒绝目标GPU已有计算、已有原环境覆盖、下载字节超限/SHA不符、原前缀外成员、链接逃逸和跨归档文件覆盖；先检查压缩总量加8GiB保留，再完整扫描三份原资产并检查系统解压payload加保留空间，之后才恢复原环境。CUDA11.6库路径保留，无依赖版本替换或原生编译。只执行封存软件组件，最终报告与各阶段小日志上传，原目录保留、失败不自动重试，不包含真实模型/输入或训练。

112最终新目录/home/lbin/rbf-fourcard-runtime-bootstrap-final-controls-20261103语法、合成路径一正四拒绝及组件包九成员控制通过；报告bootstrap-final-software-controls-on112.json源与保存反向SHA678f82ab528e167265e0f6b78fc7693dfb98db5d67d68a50bbe2c1cca73c84b3。首轮软件源码及合成文件保留；最终版仅补回原CUDA11.6库路径后重新检查。没有执行ClearML main、真实资产下载/恢复或模型，也没有新失败实验重跑。

此候选还未通过真实SDK/HTTP/schema、原运行时成员兼容、实际恢复或组件正向执行；四卡调度器、独立产物读回器及具体新发布/执行授权仍需完成。不得直接入队或复用旧八卡调度。容量payload不包含文件系统额外开销；实测恢复后仍须复核保留余量，不能只凭压缩/成员算术视为完整保存容量通过。

## 独立保存报告读取器已准备并通过合成协议控制

verify_runtime_component_outputs.py（SHA2ea115525a5a91dac13db46d3807736585226ee104caf64895ca068eea71ff11）要求来自真实SDK/投递收据的外部binding，检查入口SHA、终态、发布包SHA、四UUID、原三资产身份、全部13个注册小产物与源/子报告交叉SHA、原解释器/框架、六源码固定SHA及三个组件exit0；仍返回真实四卡模型/数据流/完整训练未资格。读取器不下载或运行模型，可信binding不能由产物目录自行生成。

check_runtime_component_outputs.py（SHA41f1a0bdc24f68a6f3cbee6a18d1ba6b8fdf8d9c3d5a717c1481a3a2c811636a）在112新目录/home/lbin/rbf-fourcard-runtime-reader-controls-20261103真实exit0：一项合成正例及十项错误Torch/Python、源码缺失、阶段失败、越界模型声明、UUID/入口/终态/注册稳定性/发布身份拒绝通过。没有读取真实产物或执行原环境；本轮报告runtime-reader-controls-on112.json源与保存反向SHA均dd82506f4d2b27eef0b0c7df1b75f1918a1422c1428d4dfaaddbd3a7a66929b1，Mac仅传输。

仍缺实际SDK/HTTP收集与前后注册快照绑定、只读调度准入/一次投递器、具体发布执行授权及真实原环境正向结果。不得将合成的13文件覆盖证明当作真实平台完整上传或四卡模型验收。

## 23:52队列交集准入与投递前阻碍

four_card_queue_guard.py（SHA353f3f2167d498e0b9feacd047d3d03118e3d8efcba82c5317f0aee5049c9b27）及检查器在112完成两项正例、八项拒绝控制：另一组忙不挡空闲组；八卡worker占用、两组忙、陈旧/未来观测、重复worker、非预期队列consumer、四/八卡队列在途均拒绝。控制收据保存反向SHA7e4de84e174f178086d641be25cd86411b3c7cd7850659161e484a2bff7acc50。

23:51:54真实ClearML只读worker/队列快照four-card-live-scheduling-observation-20261103.json经112在23:52:07检查：前四卡仍为e9614d78…，后四卡/八卡worker空，两队列空；唯一候选ubuntu26-2:gpu4,5,6,7，实际GPU UUID和运行时/磁盘准入仍false。live-scheduling-candidate-on112.json保存反向SHA86b1d4ffa5abf5a2cf76957b7fc0b97dc86793bc69fec9950410e51b2bbe61a5。未创建或排队Task，快照仅证明该时点候选，不能留作后续过期调度依据。

两组四卡同为GPU4-2080ti队列consumer。若投递时两组都空闲，平台可由任一组接取，当前bootstrap却要求一个expected_worker；正式投递前须将入口改为与所有已准入候选的worker→UUID/缓存绑定映射一致，或取得平台实际支持的定向调度证据。不能仅在客户端选一个worker，再假定共享队列一定投给它，也不以修改别人的worker/任务解决。当前未把这项未解决风险藏在“投递器已准备”中。

同轮23:52:19真实刷新teacher3407仍in_progress、26组封存，0002 trajectory59095/167073，child CPU436463.850823102秒、wall436655.28555978835秒；0055/0056/0057等内部计数也继续增长。没有新增封存，不重复收旧26组，不据CPU增长宣称剩余数值验收完成。

## 共享队列接取绑定的候选修正

bootstrap不再要求预猜一个expected_worker，改为General/candidate_bindings_json精确列出前/后两个原四卡consumer的GPU_UUIDs/cache_parent映射。实际接取后只用自身worker对应行，继续执行真实UUID、计算进程、磁盘与恢复门槛；两组必须各四UUID且彼此不交，八卡worker、缺组、重复/重叠UUID、相对磁盘路径或额外未绑定字段都拒绝。不更改worker、队列或现有Task。映射必须来自具体准入记录，不能凭空填写或放宽物理验证。

新版original_runtime_components_bootstrap.py为13673字节，SHA cf4fe85e956e348a1981fd0d003d30fb28588a8b103e5e2d212f9cbd367e9c4b，替代此前1562a161…候选，旧版在112原目录和历史记录保留。112新目录/home/lbin/rbf-fourcard-shared-queue-binding-controls-20261104完成两组正例及七项拒绝控制、语法检查；shared-queue-binding-controls-on112.json读回反向SHA6bd7e5b8d8acf55d432170b0cc88335394406dda8f8c504e2960a1db446e5075。

独立读取器ENTRY_SHA同步严格改绑新版，当前读取器SHA560974b676d2538ce2faed80f969d977804d5b786562555ee2e6fc3f51cf86b2；在112另一新目录重复原一正十拒绝协议检查通过，控制报告仍为dd82506f…，未读取真实产物。该修正只消除入口“猜接取组”的协议缺口，尚没有完整投递器/SDK收集器或两组真实UUID/缓存准入；不构成已经可执行的四卡资格。未发布包/脚本或启动Task。

## 一次性投递器准备与候选身份来源

dispatch_runtime_components_once.py已准备（7639字节，SHAa440bd9cf3b865d0306fc68eed77f9085c244aa390e6eef78154c58b2348b605），只允许Linux执行。来源/授权/包/读回器身份检查先于SDK导入，保留单次claim和失败Task，克隆原完成Task49b7ee412456475d9514b7f78ccbada6的完整容器合同，创建及入队前分别刷新四/八卡队列交集，不覆盖worker参数或活任务。只授权新软件环境复验，不沿用原八卡模型资格次数。

真实SDK只读确认：原父Task仍completed，原e5b249…容器及参数与磁盘探针一致；磁盘探针0d16e98cd819465bb6c2e76e8e5fdcf3确由前四卡接取并completed，原报告SHA bbcf91cb…记录前四UUID及/clearml_agent_cache的ext4挂载。candidate-worker-bindings.json（607字节，SHA5efa477751f1b5ddd7581e3c0efca8ea509f40335990a07e4e04be93f1c7521c）前组取该实测四UUID，后组取原八UUID清单剩余集合；这是明确来源的候选分组，不是后组新鲜物理核验。每次实际接取仍须核对该组UUID及真实磁盘，不沿用旧5.62TB容量作当前合格证据。

112新目录/home/lbin/rbf-fourcard-dispatch-preparation-controls-20261104语法与纯内存授权一正九拒绝控制通过；没有写出真实grant，也未导入SDK或创建Task。dispatcher-software-controls-on112.json完整读回反向SHA0224c192e7958f71c5c93e100b12fe4a514d96bc28acaafb9542e7b1e36e8deb。此为软件门槛，不是实际SDK投递成功证据；仍待SDK下载/前后注册绑定收集器与具体发布执行授权，再统一提出可审核请求。

## 11月4日最终发布候选与SDK工具准备收口

发布器、一次性投递器、bootstrap、完整注册小产物SDK收集器及独立读取器已具体准备；最终审阅范围/源锁见software-release-review-20261104.md和final-release-readiness-on112.json。最终SDK路径隔离版bootstrap13698字节，SHA1359d4586e454045f1ba74f7d6404941f2ad86eea95a6076db222bc6e3b5f525；reader22e6dd8e…、dispatcher4dc8e0fc…、collector2f760e07…均互相严格绑定。组件包10795字节/9成员SHA862f14c5…不变。112证明最终bootstrap只多原子进程PYTHONPATH隔离，其他已验协议函数未变；三工具只改对应源码常量，独立读取器原一正十拒绝控制再次通过。

收集器mock源/注册/路径一正十六拒绝通过（先前源码6dab…），发布器空script/禁仓库发现和授权一正七拒绝通过（最终d5e3d2e…）；软件报告完整保存反向SHA分别3360bcab…、44c91b86…。真实SDK创建/上传/HTTP收集尚未执行，不把mock当接口成功。

112隔离SDK工具ClearML2.1.5已安装，32项工具依赖离线审计通过；原父Task只读查询成功，API/FILES严格匹配，无torch导入或凭证导出。首个普通venv因无ensurepip未完成、附加审计脚本缺packaging的准备错误均保留；使用已有e5的pip安装到新工具目录及内置解析器完成检查，不改系统或原模型环境。最终同目录/home/lbin/rbf-fourcard-original-runtime-software-release-final-20261104八文件SHA/交叉锁已通过，准备收据保存反向SHA20b6e92c…。

目前尚无本项实际grant、FILES新包发布或API新软件Task。下一道门槛是具体批准固定新包发布/完整读回、固定入口脚本提交、空闲不重叠四卡软件复验一次、全注册小报告/日志上传与112/外置Mac保存逐文件SHA读回。该批准不覆盖真实四卡模型资格或完整训练，不改变其他任务/原输入/容差；失败保留、不自动重试。

## 当前覆盖上文准备状态：既有原环境发现、真实软件失败和旧请求撤回

11月4日空间审计发现/home/lbin/rbf-sparsecoop-runtime-restoration-20261024仍保有原native-py38-venv。此前宿主标准路径未发现的检查不应被解释为该环境不存在。使用保护的855镜像及只读venv/toolkit挂载，00:42真实加载Python3.8.20、Torch1.13.0+cu116、NumPy1.23.5，未挂载作者模型或样本、未调用模型或禁用CUDA，也未重新核对整个运行时归档身份。

在该既有环境中执行自编无真实数据的软件组件控制（不是尚待批准的四卡ClearML Task）：00:44逻辑计数exit0、toy状态隔离exit0；loader报告步骤因Path(name)从/output而非/job查找源码发生FileNotFoundError，完整矩阵exit1。原两项通过报告和全部失败输出均保留；没有把部分成功重标完整通过、GPU资格或模型资格。

23成员失败保全归档13328字节，完整逐成员读回及Mac保存反向SHA c0aea744913fc1a89e59bd8da35e911cd72b3fb6544de92bab7f40b59f9b0385；收据SHA15c50d89…，本地original-runtime-failure-20261104/。旧组件包862f14c5…及入口1359d458…仍有此报告缺陷：原发布/执行请求撤回，RELEASE-HELD-20261104.json已写入本地及112最终准备目录；即使随后收到旧问题批准，也不发布或执行旧版本。

另目录loader-report-path-fix-candidate-20261104/仅将报告读源码改为Path(__file__).resolve().with_name(name)，新检查器SHA7af9ae6bca8e00b27172ea0bf3211fa229b7bf82155477c1332a5eaa9b19d741，原loader helper未改。独立Python3.8标准库跨工作目录单元验证exit0、八逻辑loader接线/六拒绝报告完成；没有Torch/真实DataLoader、数据或模型调用，也未重跑失败的完整矩阵。修正证明保存反向SHA40005fc77fad80e8106953777b0383ef4567fbe85281169bd5cc1f282cc9a520。旧发布包未重打或发布，实际四卡/真实原模型一步仍未资格。

## 2026-11-07：真实head归一化接线位置重新定位

112现存作者源码 `/home/lbin/rbf-sparsecoop-runtime-restoration-20261024/author-source/projects/mmdet3d_plugin/models/sparsecoop_head.py` 可直接只读使用，完整SHA `504cc5cefafd292febce6575b997b045cea1aa58cab8397d1fb686ebd66e2991` 与本目录既有author-state-and-reduction-static-audit-on112.json登记相同。未导入/执行该源码或模型。不能再把作者head路径不可用列为当前准备阻碍。

实际原代码第886行是 `max(reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0)`，第1105–1106行是对dn_valid_mask的同样全局均值后clamp。四卡八逻辑流必须先复现全部八流同一次完整batch的未clamp计数及原归约顺序，再在原位置clamp；分别clamp两个wave或改变归一化分母均不合格。

当前实质阻碍仍是上文count/state时序与真实模型资格，不是这些调用点或软件矩阵。未选择或实现未经验证的预遍历重放/保留双图方案，未改变作者函数、配置或真实batch；无模型步/训练授权次数消耗。下一道模型资格须明确验证状态事务、真实计数、完整48样本、梯度/FP16/clip/optimizer/LR及成本。不能以本次源码定位或软件成功替代它。

### 原loss dtype路径的静态核对

同一SHA 504cc5ce…的真实方法为 `SparseCoopHead.loss(self, model_outs, data, feature_maps=None)`，原装饰器写作 `force_fp32(apply_to='model_outs')`；DN计数在未单独装饰的 `prepare_for_dn_loss(self, model_outs, prefix='')`。检测计数在原代码转为reg.dtype，DN计数转为reg_weights.dtype；后者由dn_reg_target.new_tensor构造。

因此不能仅凭装饰器名称就假设所有计数归约总为float32，更不能直接用Python精确整数/有理数均值宣称与原NCCL浮点归约逐值一致。真实四卡资格应留存实际FP16启用状态、两处计数dtype/原局部值/全局结果，并在原数值标准核对原8逻辑流归约、loss与梯度；不能改动作者装饰器、类型转换或偷偷采用更高精度来获得通过。此次只有AST读取，没有导入torch、执行模型或改源码。计数/状态时序的真实集成仍未完成。

### 计数/状态依赖的真实作者源码候选：归约点协作暂停

112新目录 `/home/lbin/rbf-sparsecoop-four-card-loss-coroutine-preparation-20261107` 对真实原head（SHA504cc5ce…）的loss和prepare_for_dn_loss制作源码级候选：仅将两个reduce_mean调用点改为计数请求yield，将loss中两处DN准备调用转为yield from。候选源码 `author_loss_coroutines_source_only_prepared.py` SHA256 `928df93434aa62406c442a8bbeed4f177cda7893d49249d373ba135ab7d3c4df`；反向恢复变换后两个原函数完整AST（含断言、装饰器及全部数值语句）逐一相等，语法编译通过。证明SHA `947c8ade234779bc8e6e043f0f12354fae30390b162b9a65dbf5e8a4c0b846fb`。

该方案意图在同一原forward图上暂停loss，收齐8逻辑流原计数后继续，避免计数预遍历增加模型前向；但还没有执行资格，不能称额外forward已经消除或数值等价。当前文件明确source-only、不可直接import投产：原force_fp32包装器通常只包生成器创建，不会自动覆盖后续resume，必须单独保持原上下文；两逻辑流图与独立BN/缓存状态所有权、原dtype归约、DDP梯度平均与FP16/clip/optimizer、实际显存均未证明。原LogicalStateBank.finish门槛未移除；不能靠静默放宽它切换未backward状态。不同逻辑流的实际collective顺序不符应拒绝，而非伪造空计数。

未导入作者模块或torch、未执行loss/model、未加载原样本、未发布/提交Task，原作者源码未改。真实4卡模型资格仍未通过，本候选只推进真实源码接线准备，不能替代下一道完整48样本loss/状态/梯度/step对照。

### 原装饰器输入转换与生成器恢复上下文适配准备

112 `rbf-sparsecoop-four-card-loss-coroutine-preparation-20261107/scoped_author_loss_coroutine_adapter_prepared.py` SHA256 `02576da7e79011f6759519bbb40b4dc25c6bb965ac0238fb9450c0a83846a182` 已准备。未来绑定前严格检查原head SHA、原loss签名及原force_fp32参数；保持原装饰器进行启用判断与输入转换，在其内部捕获grad/CUDA autocast上下文，并对每次next/send/throw/close重建作用域。DN生成器在同一作用域yield from，方法只临时绑定指定原head实例，退出检查并恢复，不改作者源文件。

仅使用contextvars合成生成器的4项恢复/返回/异常/关闭/所有权协议检查通过（报告 `scoped-loss-adapter-software-only.json` SHA256 `09c3d11ff74461e8a3017042d61283935c953708bda7c5dcb7211a4fd7584e2c`），未导入torch、作者head或真实模型。真实原Torch1.13/MMCV force_fp32分支、AMP、输入转换与autograd行为尚未资格，不能从合成contextvars推断数值等价。逻辑状态/RNG/图所有权、实际归约、DDP/FP16/clip/optimizer、完整输入与容量仍缺，未发布或执行模型资格。

### 原冻结MMCV装饰器来源绑定

112已只读检查native-py38-venv/lib/python3.8/site-packages/mmcv/runner/fp16_utils.py，SHA256 `a636d3a1dd9f227fb986ded7676ec039d3d8346e1d02a16efbdc1919f0f0e9c1`。原force_fp32仅在模块fp16_enabled为真时转换指定输入，并在autocast(enabled=False)中调用原函数；若原函数改为generator，该作用域确实不覆盖后续恢复。本核对没有导入MMCV/Torch。

新版 `scoped_author_loss_framework_pinned_prepared.py` SHA256 `4102bd31c63ffc315afdb98f37123afb299d427227cf4c9d2b6f89e25bc508a5` 进一步要求原Torch1.13.0+cu116及原force_fp32源码全SHA，语法通过。证明SHA `dce729094a2742f8c386167189d5d64a59d63bc162997332f14d4a9b5e83e261`。真实AMP/模型数值未执行或资格。

宿主native-py38-venv/bin/python是指向/opt/cooptrack/bin/python的链接，宿主当前不能直接解析；这是既有冻结容器内路径合同，不能单凭宿主exists=false认定冻结环境坏了，不改链接、不重装依赖、不复跑此前单次loader。未来框架资格应使用原镜像和已核对挂载，保持原CPU/RAM/线程/GPU路由合同。

### 原冻结框架8组合软件核验已准备，待具体批准

112探针 `check_original_framework_scoped_coroutine_prepared.py` 4267字节，SHA256 `560299298376bb54559f2afe2b5cea71c169cb76ec97d13fa74d8cfe0572598b`；原框架绑定适配器4102bd31…不变。探针仅合成CPU张量，原force_fp32启用/未启用×grad开/关×ambient AMP开/关8组合，比较普通调用与恢复调用的dtype/grad/AMP、结果与外部作用域恢复，不读取训练数据或加载作者模型。未执行。

独立只读两文件树framework-control-two-file-tree-prepared，避免挂载整个准备目录。新窄挂载计划 `original-framework-context-control-narrow-plan-prepared.json` SHA256 `cd414b550f95c3919d1a457ad8a841630243a4e9c9b8af8543c9e251f91288ab`，原镜像完整ID `sha256:85525aefed5d9a5d5f6d72d7853f1a9ba3744206c889c767a8dfac8202c1d2ab` 已只读核对；仅原venv/toolkit及两文件代码RO、独立output/tmp RW，无新CPU/RAM/线程限制或CUDA屏蔽。

当前单次启动/完整SHA收集器 `execute_collect_framework_context_terminal_safe_prepared.py` SHA256 `054c034b0a310f7d01a739ef3ee1ee35b1da57a21bfcc4dc44e7cfa154c62254`，语法通过，实际grant/资源/源SHA/镜像通过后才建立once-claim并执行；报告解析/断言失败仍留存全部退出/日志/小产物SHA，不重试、不将失败称通过。旧初版4cb19b80…保留但非当前候选。

已请求具体一次软件核验批准，尚无回复，不运行容器、不伪造grant。此项不包含真实作者head调用、四卡模型资格、全训练或新ClearML源码发布；不消耗那些尚未执行的次数。软件通过也不能替代完整48样本loss/BN/时序/梯度/FP16/clip/optimizer及内存资格。

### 真实归约暂停请求加入源位置与DN前缀

112新版 `scoped_author_loss_tagged_sites_prepared.py` SHA256 `9bfb9a08748c045255de7b34ca0d7e6f6d09198bec12480119b53082d341e1a2` 将暂停请求从无区别二元组改为携带原函数、源码行号及DN prefix的五元组：loss:886/空前缀，prepare_for_dn_loss:1106/原prefix，最后仍为原计数tensor表达式。供后续调度器在tensor归约前核对全部8逻辑流的同一调用点/前缀/顺序，不能只配对第N个标量；调度器与真实collective尚未实现或资格。

只执行自身AST变换器（非作者函数）：去除暂停标签并逆变换后，两原函数完整AST一致；ScopedGenerator与torch_scope_factory和4102bd31版AST完全不变。报告 `tagged-sites-source-review.json` SHA256 `8729a1d2ecd3ab2e00091babed92df95b6e22228e9778725c3a5b7a46003da5e`。旧框架探针、两文件树及其待批请求未修改，不能用新候选替换已请求范围。未导入torch/head、未执行loss/model，不宣称实际collective或四卡等价。

### 两自有逻辑流到八流归约元数据的协调层准备

112 `owned_two_stream_loss_scheduler_prepared.py` SHA256 `faf43cb36d439efee3b7b97399691b583213191adb9db660c0c423e100daafb8` 接受固定自有逻辑对(p,p+4)，在activate_state回调内恢复loss协程；交换元数据必须完整包含0–7八流且轮次、请求/结束状态、原函数/行号/DN前缀、dtype和标量形状一致，然后才调用原数值归约回调。未伪造其他流计数或归约结果，也不自行调用backward/optimizer。

8项纯软件协议控制通过，包含正常完成与site/prefix/dtype/shape/identity/round/done七类差异均在归约前拒绝；报告SHA `0dfd834b20b07f96f03aaf03f0713b0bf3a637e4b860aa02ab5975612e2608ab`。真实activate_state、跨rank元数据交换、原dtype/NCCL归约与所有权回调均尚未资格，测试用合成生成器不触及真实state/model/tensor。不能把此协调层视为完整4卡训练入口或移除旧LogicalStateBank.finish约束。后续仍须真实完整48样本及全状态/梯度/FP16/optimizer/内存对照。
