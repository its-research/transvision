# 模型复用与选择隔离范围同步

## 本轮V2V来源连接收口

112已生成七份证据文件的完整SHA汇总：`/home/lbin/rbf-C-original-training-source-provenance-20261107/V2V-three-model-original-protocol-evidence-crosswalk-20261107.json`，SHA256 `5c6c9bff7fd56bc4e4241421f9cc5a00ac4819dd55bedb33b24b08d4285b690a`。三种子的原源码闭包、原输入元数据、分组监督、训练/最终checkpoint登记、十轮选择历史及既有2364754行独立前向收据已完成本轮身份连接。不要再次重复这些来源核验来替代缺失实验。

本轮未发现需要因源码身份不符而重训的事实；建议继续复用原模型在其已证明的原合同范围内的证据。该汇总是crosswalk而非新独立接受收据，不能接受新runtime/新test/受限H支持；V2V重叠和隔离限制、未校准surrogate、未完成teacher/priority及G–K门槛仍保留。SPD原代码包读取仍独立待批，C整体未完成，A–K未完成。后续优先推进实际teacher/回放/评价依赖，不继续堆叠相同元数据检查。

## 当前C项决策表：证据不足不等于需要重训

| 必须证明的合同 | 当前可用证据 | 尚缺的决定性连接 | 不得据此宣称 |
|---|---|---|---|
| 原训练源码身份 | SPD1337 checkpoint给出9份原SHA；3份源码精确匹配 | 原选择Task登记代码包完整SHA及另外6份原文件匹配 | 脏工作树或同名runtime就是原训练器 |
| 连续时序及因果截止 | 原row_context有同序列、到达时间和前驱索引检查 | 原数据构造器/调用入口及实际manifest，证明连续历史确实供给到训练 | 函数具备guard即证明全部训练无泄漏 |
| 跨端与时序监督 | 原epoch元数据分别记录两类loss/监督行；原supervision保留unknown | 原训练调用、目标构造与实际输入行的来源连接，分别核对两项损失 | loss非零即证明完整监督合同成立 |
| 选择与最终模型 | SPD1337原选择Task、epoch3及full-train refit指针；六模型既有registry | 每模型实际选择/训练收据与最终checkpoint逐一连接 | 1337的隔离资格可外推其他种子或V2V |
| 运行时适用性 | 原train-forward独立逐行数值证据及train回放连接 | 与当前数据、特征、支持域、预算、runtime的精确对应 | train-forward通过即接受新test或受限H支持 |
| 概率及科学声明 | surrogate已明确命名，V2V内容重叠/选择限制保留 | 正式评价、校准和适用协议下的独立证据 | 前向一致就是概率校准或独立泛化 |

执行分支：先关闭原来源与适用性缺口；完整匹配则复用已验原模型，不重复训练。只有查实输入、监督或模型合同不符，才提出保持原科学标准的新训练/冻结/独立前向计划。来源暂不可读列为“未证明”，不伪造失败原因、不直接判定必须重训。teacher/priority与两数据集G/H、I同资源重复仍各有完整独立门槛。

源码资产诊断：原Task 7a7586375f1c467b91956abf3a681d21的code登记1207564字节、SHA f3649c4232ac29379b3a0eb39b4965bd61d284a7a0e365e40292001da8c1547f。首次读取401失败已保留；112同一资产在既有文件服务器附加SDK现有认证后HEAD200且大小匹配。尚未下载正文，不能关闭表中源码身份缺口；认证修正版完整读取单次请求待直接批准。此表为文本证据整理，不是新模型接受收据。

本轮只读核对method.md第61–67行、既有BC-method-contract-model-reuse-crosswalk-20261101T2238.json、BC-six-model-reuse-and-claim-boundaries-20261031/current-six-model-registry-and-limits.json及source-only论文包改动记录。原方法允许明确命名的surrogate；exact_joint_nll=false或paper_eligible=false本身不能触发无条件重复训练。现有数值行一致、跨端/时序监督、完整协议适用、校准、teacher及priority资格仍是不同门槛，不声称C全部关闭。

当前 `/Users/lbin/thesis/tracking/recover-before-fuse/cvpr2017/main.tex` 第211行旧“Train separate…”已改为条件复用：仅当输入、特征、跨端/时序监督、选择及运行时来源匹配时复用，适用性失败才提出train/refit；既有train-forward不能自动接受新的val/test或受限支持配置。第96行明确V2V选择元数据不具严格pipeline隔离认证，前向一致不消除该限制、已知train/test内容重叠或证明独立泛化。

两处与10月31日source-only澄清方向一致，但未整包覆盖当前主稿，保留当前teacher目标/Long-SCOPE ROI及其他用户修改；未复制旧包中的指标或把旧审阅包当当前正式产物。实际文本已读回确认，无数值/模型/训练/LaTeX/统计/SHA在Mac执行，无提交推送，无模型资格标志改写，无活任务修改。

后续模型资格仍按原crosswalk完成源输入、连续序列时序、目标构造与checkpoint选择来源连接；只有真实合同缺项才进入新训练提案。完整D/G/H与三次隔离资源重复仍必须独立收口。

## C项后续真正需要的训练来源连接

2027/3407最终checkpoint文本及实时平台登记补连：外置原登记checkpoint分别给出model026e4604…/e72a2ac4…，full-train refit epoch9/1，selection Task f7da8e72146742f191ded68d8b57f481/dabafc1bbeb544b3bca2cbdf4d207b4c。112实时查询两Task均completed；它们与1337共同指向原code Task7a758637…及同一f3649c42…SHA，但训练行来源分别为2f289360fd5d4bbd8d355ea66e2af50e（1fbe9ca1…）和f25a0dab1c1d44debf9aa2452e872465（1012829b…）。登记manifest分别7c5ae25f…和b719faf1…，与各最终checkpoint文本字段吻合。完整参数小记录在112 `SPD2027-3407-selection-source-registration-pointers.json`。Mac只读已有JSON文本，本轮未新验其SHA；112未下载产物正文。由此可以共享原代码包的身份核验工作，但三份训练输入与实际监督/历史适用性仍需分别证明，不能把共同代码指针当共同数据资格。

112有限目录定位补记：`rbf-teacher-final-model-reference-20261028`实际是四份predictions JSONL及reference-manifest，不是另外两种子的训练checkpoint目录。`rbf-teacher2027-full-binding-extra-metadata-20261106-once/old-checkpoint-metadata.json`明确为旧2027模型1a904b31…、dataset7c5ae25f…、internal-holdout epoch9；不得作为当前full-train-refit最终checkpoint。其source_sha256确实包含与1337指针一致的原训练文件SHA，可作原源码身份交叉指针，但不替代最终模型/选择收据连接。此处只读小元数据，不读取predictions正文或权重；未证明其他种子最终checkpoint在112不存在。

复核现有complete-existing-train-current-model-pair-join：它把当前checkpoint与train split上Top1/K4六组合回放收据相连，不单独证明优化器训练的连续时序来源。不能因文件名含train/source join就将C全部接受。

112原SPD1337 checkpoint完整SHA ac5d41d9…再次通过，并直接提取9个训练/监督/数据构造文件的原SHA、全部source_sha256、原dataset/plan、选择checkpoint、selection Task b20f15594b6a48b68731868cc1d095f0和runner身份。原元数据为train-only nested selection/full-train refit、epoch3、strict_pipeline_isolated_selection=true、validation_or_test_selection=false。证据 `/home/lbin/rbf-C-original-training-source-provenance-20261107/SPD1337-frozen-training-source-pointers.json` SHA256 `acec9244df31c33622fd2cb0485868d758ebd91cee32a77589149084493b94d6`。

此为原小元数据全SHA及精确指针，不是源码正文/实际训练样本新验收。下一步应按这些原SHA定位冻结prepare_forest_training、forest_row_context、forest_supervision、forest_training_data及原训练runner，连接实际原输入manifest中的连续序列/因果截止/监督构造与checkpoint选择；不以脏工作树同名文件代替，不重训或修改teacher绑定。其余种子/数据集仍需各自原来源边界，不能由1337外推。

## 原SHA源码定位及部分静态追踪

SPD现有文件复用新增：112的V2V原输入副本内`train_forest_identity.py`14905字节，完整SHA3888b1e479282d50cd2ed245ff2c5a93c5dae4057355aeb649f25a48da678fa1精确匹配SPD checkpoint源指针。证据`SPD-original-train-forest-existing-source-match.json`；未重试服务器下载，未执行源码。与此前3份模型源码合计已定位4/9指定文件，剩prepare_forest_training、spd_nested_provenance、DDP入口、train_paper_identity及SPD原training_data仍待精确版本。

仅静态读该通用helper：fit_dataset会随机排列序列及valid_indices，但每个样本仍由TrainingShard.example取完整已存context；随机训练样本顺序不等于删除时序上下文。该helper本身用set_valued_parent_loss、fixed-final-epoch/no-validation-search，不能拿它证明最终SPD nested selection或分开的cross/temporal两head路径；最终调用仍须原train_paper/DDP与选择来源连接。不要因为helper注释/默认选择和最终checkpoint不同就先判定实际训练协议矛盾，必须检查真实调用入口。

### V2V匹配版本数据构造器的静态结论

完整闭包和role连接补齐：112三种子各29份原source（共87文件）完整SHA均与原bundle-plan semantic.source_sha256一致，三plan SHA依次4924b1ce…/daaf3c8e…/29146e79…，与原recipe登记一致。证明`V2V-three-complete-training-source-closure-byte-join.json` SHA256 423c7d6714329d830e0dc7f32c6f24ae41348579a5607ad26b9afb1302e0ca8f。不是新import/模型资格。

另读取已验原row-admission及partition，三种子manifest/proof序列顺序一致、原group map一致、fit/selection组不交叉；各28fit/4selection序列，两role均有跨端和时序正边。原证明计数（cross/temporal）：1337 fit70361/267938、selection7668/62104；2027 fit73599/285026、selection7232/63983；3407 fit69124/262366、selection8449/61441。记录`V2V-three-original-role-target-evidence-join.json`。这是对既有独立证明的分组汇合，不重新计算GT/目标；name-group代理不等于物理会话独立，原限制保留。

三种子原独立forward/选择连接新增：112完整读取原acceptance-receipt，SHA分别5f7597b4…/45757fb0…/3b97e447…，与原qualification登记一致；其checkpoint、weights元数据身份、manifest、seed及原ATOL/RTOL=1e-4绑定全部一致。每种子32序列、801957/789401/773396行，逐序列失败行0，分项行数汇总与总行数一致。原十轮selection history完整1–10，按已存holdout score及最早epoch规则选出的9/8/9均与原selection、checkpoint及独立接受收据一致；official_test_read=false。本轮只是已有证据连接，不重新计算loss或forward，不读取weights。原独立验收耗时212.097/210.646/206.614秒是历史单CPU验收，不是当前生产耗时或正式同资源成本。

新完整连接记录在112 `V2V-three-original-numeric-acceptance-selection-join.json`。它关闭了本轮列出的模型/输入/原forward接受/选择元数据对应缺口，不取消upstream isolation未认证、train/test重叠及surrogate/未校准边界，不使teacher、全在线RBF、G/H正式评价或A–K自动合格。C整体仍须综合实际训练输入/历史/监督来源及SPD分支，而非因局部链闭合直接标完成。

三种子最终训练收据连接新增：112已有checkpoint-{原Task}/training内final/checkpoint.json、receipt.json、plan-before-updates.json共9文件，完整SHA/大小均与原independent-byte-readback及当前ClearML注册一致。最终模型分别6e3524ab…/117e3331…/199d75b3…，三manifest SHA均匹配对应checkpoint dataset；receipt钉住checkpoint/plan，selected_epoch9/8/9与refit_history长度9/8/9一致。完整新连接记录在112 `V2V-three-final-checkpoint-receipt-live-registration-join.json`。未读权重或预测行。原生产receipt的checkpoint_independently_qualified=false保持（生产时状态），不能覆盖后续独立接受收据；下一步连接已有独立32序列前向接受结果及原完整selection history，不能把平台completed/生产receipt当独立资格。

三种子原输入元数据本轮完整SHA汇合：112对各自manifest、row-admission、partition、training-source-closure四文件重新读全SHA，12项均与各原bundle-plan semantic绑定一致。每种子32序列，节点数1337=801957、2027=789401、3407=773396；原独立v2目标证明中cross-source positive edges分别78029/80831/77573，temporal分别330042/349009/323807，prediction-context逐字节不变及target/byte精确检查均true，official_test_labels_read=false。证明在112 `V2V-three-input-metadata-original-plan-SHA-join.json`。这些是原证明内容及其完整字节连接，不是本轮重算边或重读npz。

三份partition原SHA同为02373d0f…，明确filename-prefix-proxy-only、physical_session_provenance_verified=false；原plan的actual_training_started=false是打包时历史字段，不推翻平台后续completed，也不能静默改写。下一步仍需与原训练receipt、选择历史、最终checkpoint/独立forward收据完整连接；保留上游隔离和内容重叠披露，不用当前12项SHA替代模型整体或论文科学接受。

V2V1337进一步原链核对：112现有`train_v2v4real_joint_identity_native_v2.py`完整SHA b5eeb3664fcee94c4b33a5cc685067a0988b7aab02855f62729d094adedeed36与原checkpoint指针一致。真实`phase`按manifest records逐序列建立TrainingShard，从valid_indices分batch取`shard.example`（包含既有历史context），JointHeadLoss调用batched_row_logits及separate_identity_losses，分别all_reduce两head计数并按原local_global_head_loss归一化，监督全局计数非零才backward/clip/optimizer.step；每序列seen必须等于supervised_rows。不能把8-rank行切片误写成训练前破坏context的随机帧对打散。

原专用contract静态要求32序列、7105scheduled frames、14210来源帧、15个预测context字段逐字节复用、两类正parent edge在fit/holdout两role均非零；holdout score为两head逐序列macro均值之和，按最小score及最早epoch选择。native_v2保留原合同校验，仅在独立v2目标证明上派生原kind调用旧校验，不篡改存储原证明。是否实际满足仍由manifest/proof/receipt证明，不由代码assert存在证明。

本轮112完整SHA核对1337原rows/manifest.json为8a09d50928a5a676908b790cda7559c2be578164c089867db0245512505549fb，与原checkpoint dataset字段一致；实际train、32序列/32shards、801957节点、7105scheduled frames、14210 ingested frames、native v2目标合同及official_test_labels_read=false。只读小manifest与源码，未读取npz训练行、未重跑oracle或训练。尚需三种子完整proof/partition/receipt与checkpoint来源汇合，不以1337小manifest字段代替全量来源或C全部资格。

已从原准入记录找到正确112副本根 `/home/lbin/rbf-independent-native-20261011/seed{1337,2027,3407}-input-{1fbc7d3cef6f,282a208a5563,1dee2f35c2bf}/inputs/source/tools/event_track_v2x/`。六份原专用入口现存，112完整SHA全部匹配：contract.py 6539字节/3fdf9c4b…、native_v2.py 2207字节/f4ebc416…。证明 `V2V-three-seed-original-contract-source-match.json` 已保存在112 C来源目录。此前四个runtime目录查无同名文件不再构成定位阻碍；不需新下载这些源码。仅关闭源码身份定位，不宣称实际训练历史、数值或选择隔离新通过；未执行源码或读训练行。

V2V原训练平台连接新增：112实时读取166327b7…/ab40ed42…/2a656949…三Task，均completed，script SHA均e0e33ec0149b2a0dc0c15ee07ee95f15a7bcc1a12b42090c01a6762cc6898d75，入口native_identity_v2_transport_candidate.py；recipe SHA依次f050cb54…/c3944911…/3c2d858b…。它们均注册checkpoint、identity-training、plan、receipt、recipe、runtime-before-launch；未下载正文。完整登记指针在112 `V2V-three-native-training-registered-source-pointers.json`。已有20261011 submission文本对应这三个recipe身份；1337原input-bundle Task0bb8e724879749aeb438f7f9d0e7d91e，576267181字节、SHA1fbc7d3c…、semantic_training_identity cd3fe7b7…、native_target_contract author_native_mapped_id_one_explicit_association_v2。此为原输入包定位，不授权重新下载、不证明包正文当前可用或实际调用已审完；应优先沿既有ssh-admitted-inputs/独立checkpoint记录定位曾核验副本，避免同名源码搜索。

112逐段读取cf1bf208…原源码（未导入/执行）：`prepare_training_rows`第232行起要求VerifiedForestCache且split=train、非空sequence-contiguous schedule；逐cohort序列重置历史，reference严格递增，decision=reference+100000us，信息时间晚于decision的来源被排除，cache.load_arrived在该decision下读取。每序列observations/times累计供给older候选，当前batch传入原build_row_contexts；标签单独送make_row_supervision。它具备连续历史构造能力，不是仅孤立随机帧对构造器。

`labels_for_frame`先在完整原检测帧上按原同类严格2m匹配，再按既定score/top-count选择取监督，避免移除高分重复后把未匹配低分框提升为匹配。`write_shard`写203维features、9Dmean/cov、source、三类时间、context索引及独立positive/known/reason监督数组；不能把“标签不作模型输入”误写成“shard完全没有监督标签”。TrainingShard校验文件SHA、数组集合/shape/count及只读后备，example验证索引有序唯一/当前行末位/填充，不自行证明真实数据来源合格。

manifest仍显式paper_eligible=false、local partial-label surrogate且无校准声明；失败保存completed_shards和partial_outputs_not_training_ready。**尚缺**V2V原专用contract/native_v2入口的精确原SHA正文、它对本builder的实际调用方式、三种子原manifest/历史覆盖及训练损失调用连接。同一有限runtime目录未找到两个专用tools入口；不能从通用builder存在，推断V2V实际训练必然使用了全部这些分支，亦不能据此关闭C或选择隔离限制。下一项审查应定位原专用入口和真实manifest，而非继续重复验证该源码身份。

V2V分支新定位：外置原checkpoint文本三种子selected_epoch为9/8/9，selection均original_train_name_group_holdout_then_full_train_refit，strict_pipeline_isolated_selection=false、validation_or_test_selection=false；未在Mac重验SHA，也不因此取消既有重叠披露。1337原source指针中的forest_training_data.py为cf1bf2082cedf729da7b1b2f6eaf109a4759c2154dbaa26d2a5b170e0548f904，恰与112现有runtime文件完整SHA一致。112已只做SHA/AST身份核验并保存`V2V-training-data-existing-source-identity.json`；未执行源码或读取训练行。该文件对SPD原24ac5018…不匹配、对V2V1337指针匹配，两结论并不矛盾；后续V2V构造/连续历史静态审查可沿此精确版本推进，仍不能代替实际输入、训练调用及全三种子资格连接。

112在既有rbf-recovery2027-threaded-runtime-20261019/runtime找到三份与原SPD1337 checkpoint全SHA完全一致的源码：forest_row_context.py（8090dbe4…）、forest_supervision.py（de120242…）、forest_training.py（e136f877…）。training_data.py同名文件实际cf1bf208…不等于原24ac5018…，已拒绝作为原训练证据；其余五个tools训练/准备/选择入口在该有限目录未找到，不能据此声称112全局不存在。

仅静态读源码：row context检查同序列/决策前到达/前驱索引/时间及几何gate；supervision区分matched/unknown/out_of_support/uncertain_birth，local set-valued surrogate不把无监督行称零loss；logit构造含203维原特征、时间/源输入、temporal attention/scorer、跨端pair/unmatched与几何项。未执行函数或模型，caller是否完整供给历史及实际输入/标签隔离仍须原数据构造器与准备入口证明，不能从函数guard推出全pipeline无泄漏。

有限查找报告SHA `7116615425d1ebb411f69429de6c5631a2cbcaf8ec3695430832e1794ad7a04e`；三原源码静态追踪 `/home/lbin/rbf-C-original-training-source-provenance-20261107/SPD1337-original-three-source-static-trace.json` SHA256 `a00568b900e6c118e80ec115e08cbc6eae9796fade0b8bd2954a160b3cf13381`。接续定位其余六个精确原版本，不以脏代码或不匹配runtime替换；不重训、不扩大为C全部通过。
