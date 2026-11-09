# J：V2V原生评价表示合同的关闭要求

本文件是源码审阅后的修复与验证要求，不授予执行许可、不改变现有评价器、不新增数值资格。当前源码由112于2026-11-03 02:52北京时间核对完整SHA：评价器062a896f1597c3257eafb321bcb625df2144690b52cdd92febf08252269ea7da，原输入fe1f93d037d40706936711a2be8c8f4d6446ebe2527ee1e2f11fb4835e7c9133。

## 已定位的阻碍

原输入接受有限精度原标定的1e-5刚体表示检查；当前通用转换入口在读取每帧world_to_ego时调用rigid_transform，要求正交性及行列式1e-6、齐次行1e-10。历史1,993帧原作者求逆已在原锁定环境精确复现，但354帧仍被通用门槛拒绝。这不是求逆来源尚未证明，也不是显存、batch或GPU内核问题。不能再安排同一求逆复现来冒充解决此阻碍。

当前convert的GT路径直接使用冻结生产器的corners_ego，恢复torch float32后调用作者native转换；仅预测路径从world mean9经原world_to_ego进入native转换。因此，为全部GT增加ego→world→ego往返不是当前源码的必要步骤。该额外往返已经产生负结果：最大角点差0.00115203857421875米，超过原1e-4角点标准，必须保留，不能把它称为通过或用于替换原GT。此源码判断不证明预测转换或原生指标已经合格。

## 修复候选必须满足的门槛

1. 先形成明确协议决定：当前通用严格刚体路径保持原样；如需另设原始标定来源证明路径，必须作为明确命名、独立绑定来源的候选审阅，不能悄悄提高通用atol、投影为正交矩阵、改原标定值或绕过失败。原1e-8状态、1e-4因子/角点及专项门槛均保留。
2. 候选绑定原1,993帧矩阵逐字节SHA、原作者转换源码及精确环境、九序列/空帧/时间/ego/frame覆盖，严格拒绝任意替换来源、缺帧、重复映射、矩阵变更和非有限值。已通过的来源证明复用，不重复生成不同数值的侧车。
3. 将GT与预测转换验证分开：GT严格沿用原corners_ego和作者native函数；预测按作者原变换路线验证中心、尺寸、yaw、角点顺序、坐标轴及identity映射。不可用额外GT往返负结果推导预测一定错误或正确；不可只检中心来声称完整角点一致。框几何仅消费状态前7维，速度和9×9协方差必须由B项及实际回放的完整独立状态证据另行证明，不能由框转换通过推导9D资格。
4. 先做源代码与合成正反例检查，再按具体单次授权做真实完整转换对照。所有计算在112或合格Linux，Mac只传输。旧失败、负结果、候选/原版输出、完整逐项差异和SHA均保留。未通过前不替换正式入口，不运行指标补数。
5. 完整原生转换通过仍只关闭表示桥接：DMSTrack还需作者版本、实际权重/特征、split、merged vehicle、标称10Hz、原ROI与原生预测资格；正式指标须全九序列独立回读，核对AMOTP单位/方向。SPD已见val探索性和V2V内容重叠披露保留。

## 当前结论

J仍未完成。已确认来源求逆精确复现与通用表示门槛冲突；未确认任何新来源专用路线或正式指标资格。下一项可审阅产物是严格来源路径的协议与候选差异，随后才是对应的具体真实对照授权。此路径与主森林保存后SHA、teacher验收及V2V unit6生产相互独立，不停止它们等待此修复。

## 03:12审阅补充：已有来源证明模块应复用

既有original_pose_attestation.py已于10月27日按1,993帧执行来源证明，历史candidate-source-attestation-full-readback-v2.json登记原源码SHAe448da579d51af8800358f2c065ad6ca17f5b1607951236c575eafd98367136c，inverse identity最大误差1.8189894035458565e-12，原1e-10门槛。九项负例含单ULP变更、未绑定identity矩阵、转置、clock/ego/sequence变更、非有限值、非显式contract和额外字段，均拒绝；原354帧通用刚体拒绝保留。本轮只读审阅，未重新执行历史数值或重新核验所有原始资产SHA。

因此下一项不是重写相同的来源attestor或重复1,993帧求逆。应复用其严格原来源证明，先明确审阅预测world→native的作者几何等价性和独立命名的表示协议；协议及真实对照未获具体授权前，不集成评价器、不以attestor返回矩阵绕过当前rigid_transform、不改任何容差。

## 03:16 原生2D字段与FP边界的作者源码核对

112完整SHA/AST审计 /home/lbin/rbf-approved-V2V-unit6-control-preparation-20261101/J-native-bbox-source-audit-20261103/source-audit.json 确认：冻结inference.py（SHA1a3c3b6c…）的预测导出line465原样写box_2d='0,0,0,0'，GT导出line540写'0 0 0 0'。当前转换器四零字段与此作者约定一致，不能为“修复”而自行制造2D投影框。冻结evaluate.py（SHAa009b926…）line657保留条件：邻近类别或tt_height<=self.min_height，且not tt.valid时计为ignored；源码默认min_height=25。

该规则会影响未匹配预测的计数。原生复现必须逐项绑定并保留该规则；统一输入/统一指标表须使用明确相同的另行冻结评价合同，不能拿原生FP数量冒充“计入全部预测”的FP，不能只替某方法更换2D字段或忽略规则。所有方法各自的原生行与统一比较行分开，负结果和协议限制保留。后续应核对跟踪输出导出器如何传递这些字段，再做真实完整指标资格；本轮没有执行任何几何函数或指标。

当前模块说明中“actual 2D fields”与原convert元数据“four zeros as fixed inference export”需在未来协议封版时明确区分原生作者约定和外部显式原生输入。当前评价器062a…源码完全未改，不以文档审阅改变冻结源清单或已有任务。

## 03:18 作者跟踪结果导出链核对

112读取原完整作者源码清单complete-pinned-source-readback-v2.json，按原commit d3b9949499c8e68ea33060873bd1cb95b6d4d323逐SHA核对io.py（d65826c3…）、main_dkf.py（4a78f056…）和model.py（f6430d12…），AST审计不执行其函数。结果 /home/lbin/rbf-approved-V2V-unit6-control-preparation-20261101/J-native-result-export-source-audit-20261103/audit.json。

main_dkf把results[hypo]每行传入save_results；model输出拼接state、track id与trk.info，update/birth保留检测info。save_results明确从res[10:14]取bbox2d，原样写入tracking evaluation文件。因此2D字段不只停留在检测导出端，确有源码通路到最终指标输入。本轮没有读回所有真实跟踪输出，不能据源码宣称每行实际全零或推算FP影响大小；这些仍须实际输出和独立计数证明。

另发现原作者native CPU完整组件索引original-DMSTrack-complete-native-CPU-forward-two-metric-implementations-component-index-v2.json记录历史9序列/1,993帧/42,769预测行及两套指标实现。该索引明确native_GPU_runtime_reproduction_qualified=false、same_resource_cost_qualified=false、registered_full_output_cloud_artifacts_qualified=false、full_J_qualified=false。应逐引用核对并复用原已通过组件，不把CPU完整组件与当前RBF world→native表示桥接混为一条路径，也不据此重复全部原生CPU前向或宣称J完成。

## 03:20 历史CPU组件报告逐引用SHA复用核对

112 J-existing-native-CPU-component-records-20261103/record-reference-SHA-audit.json确认CPU组件索引SHA21bf15b4f21cb1775e258e917af9e2b88d4ff48e8dbe398e7faf99a10d514f8f，全部10份引用报告完整SHA匹配。包括原输出归档成员收据、作者37行阈值计数核验、第二独立几何/指派指标实现及现代sum语义修复证据。本轮只核对报告，未重新扫描原预测/权重/大归档或运行指标。

注意CPU索引的GPU未完成字段只是该历史时点；已发现更晚的GPU Task6c484a6eade04f2b9f920a026d78700c完整组件索引，正在112另行核对。不得把CPU早期false作为后续GPU终态，也不把GPU组件误当当前RBF姿态转换完成。

## 03:22 修正：较新原生GPU组件已存在且引用SHA核对通过

112 J-existing-native-GPU-component-records-20261103/record-reference-SHA-audit.json确认最终组件索引SHA62f653b48d732dacd9dee02903d31e6403d81300dcb9c3c417dad52984b3ed8f及全部七份引用报告完整SHA一致。作者Task6c484a6eade04f2b9f920a026d78700c的历史范围为9序列、1,993帧、42,769预测行，完整GPU执行/输出及两套独立指标实现组件均已接受；记录还覆盖6,000产物成员与21,196运行时文件、完整GT及作者源码。AMOTP明确为平均3D IoU，越大越好。

这份较晚证据替代CPU历史索引对“GPU尚未完成/云注册未完成”的早期状态。该CPU索引保留为历史，不再据此安排原生GPU前向或两套指标的重复运行。当前来源索引复用检查只重新读了七份报告和索引，未重新读取所有预测/权重/大归档或执行指标；后续引用仍须绑定原报告的具体来源与适用范围。

边界仍为full_J_qualified=false、same_resource_qualified=false、full_paper_qualified=false。该原生DMSTrack组件不等于当前RBF world→native转换通过，不等于当前V2V方法81单元合格，不补齐其他公开方法，也不能替代同资源比较。J剩余工作应围绕缺失方法、统一比较协议和同资源/论文汇合展开，不重复该已验作者GPU组件。
