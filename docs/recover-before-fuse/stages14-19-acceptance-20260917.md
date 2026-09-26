# 阶段 14–19 执行与准入回执

> **历史回执（已被后续协议决定部分取代）**：本文件记录 2026-09-17 当时的验收边界。自 2026-09-20 起，V2V4Real 正式实验采用 [`v2v4real-nominal-10hz-formal-v1`](v2v4real-nominal-10hz-formal-v1.md)；实测时钟不再是该变体的准入条件。下文关于“保留原主协议”和“必须补齐实测时钟”的表述仅保留为历史证据，不得用于阻止或标记新变体。数据重叠、检测器来源、GT 隔离和正式三种子运行条件仍然有效。

本回执记录本轮实际完成内容，不将软件测试、过拟合或资产审计升级为正式论文实验。阶段 14 完成；阶段 15–19 的正式完成条件尚未满足。当前没有三种子正式双数据集结果或可填入论文的性能数字。

## 阶段验收矩阵

| 阶段 | 本轮实际工作 | 状态 | 仍需满足 |
|---|---|---|---|
| 14 小样本过拟合 | 三种子 × joint/cross_only，固定 400 步，另重复 1337；八个 checkpoint 回读 | 完成 | 只证明一个训练样本可学习，不证明泛化 |
| 15 训练修复与冻结 | 联合目标也能过拟合，未据诊断擅自关闭时序损失或调 null 阈值；检查正式训练/选择接口 | 正式训练方案未冻结 | 合格数据及 train 内分组；完整候选监督、连续序列训练、校准与 checkpoint 选择 |
| 16 数据与协议准入 | 全 train/test PCD 内容重叠审计；回读时钟准入与名称分组报告；修复原生缓存主协议前置 NMS 缺陷 | 部分完成，准入未通过 | 真实时钟依据或明确另立协议；重叠训练处理；detector/embedding 来源；合格原生缓存 |
| 17 正式三种子与完整方法 | 两数据格式的 fixture 准备、训练、冻结、主方法与独立评价回归通过 | 软件路径验证，正式实验未运行 | 阶段 15/16 准入、正式身份/优先级权重及完整真实序列 |
| 18 基线与消融 | pair 基线、资源扫描接口回归通过；复核六个公开基线的来源/实现清单 | 软件局部验证，正式比较未完成 | 同输入/同资源真实运行，公开算法运行时与权重，缺失算法实现/资料 |
| 19 统计与表图 | 三种子统计、配对 bootstrap、校准/删失/恢复检查及 Table 1–4/Figure 1–5 fixture 生成测试通过 | 生成器验证完成，论文产物未完成 | 真实逐序列评价与资源记录；不填 fixture 数字或文献分数 |

## 14：已经消除的疑点

ClearML `165fb1e684a344788ff945e6ab8ca8b7` completed。详细记录见 [阶段 14](stage14-overfit-20260917.md)。同一 batch 的左/右/历史 token 数 9/12/7，六个主条件全部达到正目标 10/10、未匹配目标 11/11、正增益真实边 5/5；第四卡两目标参数逐项一致。

因此不能再把旧结果解释为这个样本上模型无法学习关联，也不能据此认定必须删掉时序目标。原先 3 epoch/57 步冒烟未证实收敛。正式训练需要在合格 train 内重新建立训练/开发分组与选择回执；不得把单 batch 诊断权重作为最终模型。

## 16：跨 split 重复已经得到实证

ClearML `00d61a24eae543679d9f2ef85882a420` completed。从注册表 `14675561902041d7ab77fa2adbf2aeb0` 拉取全部 train 8 卷与 official_test 3 卷，每卷 SHA-256 与 catalog 核对；不解压到数据集目录、不读 YAML/GT。

- train PCD 成员 14,210，official_test PCD 成员 3,986。计数单位为源点云文件，不擅自解释为双车配对帧数。
- 先以文件大小/CRC 筛选跨 split 候选，再对 588 个 PCD 载荷算 SHA-256。
- 确认 294 种完全相同的 PCD 内容，train 和 official_test 各出现 294 份；均涉及 `testoutput_CAV_data_2022-03-15-09-54-40_0`。
- 报告 SHA-256：`6eb4ae8cc2d86b01d542a656eb0c380cc126c42d20e5688eef50c41d449cc260`，已回读核验。
- 未修改 train/test，未计算 test 性能，未将字节不相同解释为物理会话独立。

该结果要求明确标注训练去重变体，保持 official_test 不变；仅删除完全相同文件还不自动证明相邻片段独立。按采集组排除需与名称代理组和真实会话证据区分。当前原始文件和 ClearML 正本均保留，后续可重用下载缓存；审计后管理机剩余空间快照约 10.88 GB，正式全量缓存/训练前须重新核算。

本轮从 ClearML 回读 `a1e99f9f4eaf4208b32b7210fb58c57a/admission-report` 和 `45fbaf44ff534eeca41f52d49a880676/train-group-report`：真实源时钟仍未核实，40 个样本源元数据没有时间候选字段；15 个名称组尚不是经核实的物理会话划分。原用户授权仅支持独立标称时钟诊断，不能自动改称实测延迟主实验。

公开检测器权重配置的 `validate_dir` 指向 test，不能凭此断言权重已泄漏，但也不能证明选模独立。应取得来源证明，或用合格 train 独立重训并保留选模回执。现有 SPD registry 的 train cache/materialization 也标为 legacy/in-sample、`paper_eligible=false`；不能把这些旧权重/缓存改标签充当新主协议证据。

## 16：修复了原生候选合同

在新隔离工作树 `work_dirs/rbf-paper-stages-20260917`，基于已发布提交 `807096490746da7c10b1b79e5b11ca0ad889abcd` 检查发现：`native_arrays` 原先先执行旋转 NMS，再交给 raw top64 选择，会提前删除候选。

修复为统一 `PaperProtocol.select` 的 raw score≥0.05、稳定 top64，NMS 不参与主协议候选集合；原 `rotated_nms` 函数保留作原生协议工具。缓存 producer 字段明确改为 `raw_score_0.05_stable_top64_pre_nms_no_evaluation_roi`，不覆盖旧缓存。

新测试验证重叠框仍保留、分数相同时按原 anchor 顺序、低 raw score 不因校准进入候选、64 上限、输出特征对应。完整原生缓存仍需独立生成和准入；现有 256 维原生 head-input 诊断特征不冒充此入口的 128 维中心采样/池化/L2 特征。

## 17–19：软件验证与环境

ClearML 软件回执 `0363d50565814e26bb391198b24abc26`，发布最终 XML、三份修复源码与环境版本。

最终统一回归 **47 passed，0 skipped**，覆盖七个文件：`test_paper_contract.py`、`test_paper_pipeline.py`、`test_paper_end_to_end.py`、`test_paper_priority_selection.py`、`test_paper_resource_scan.py`、`test_paper_pair_baselines.py`、`test_paper_evaluation.py`。新增诊断与重叠检查另有 16 项直接相关本地测试通过；两组测试职责不同，不将它们计作真实数据结果。

保留失败过程：本地最初运行 contract 测试因缺 Torch 得到 7 passed/2 failed；切换管理机正确环境后解决。最初 CPU 回归 32 passed/2 skipped，因未配置独立评价器；建立任务私有 evaluator venv 后端到端 2 项补跑通过，最后统一 47 项无跳过。未修改管理机基础 Python 包。

环境：Python 3.10.19、NumPy 1.26.4、SciPy 1.15.3、Torch 2.9.1、pytest 9.0.2、nuscenes-devkit 1.2.0、TrackEval 1.0.0、matplotlib 3.10.8、motmetrics 1.4.0。评价器环境复用基础科学库，仅在独立 venv 安装评价器包；这是本次软件环境快照，不冒充正式冻结的完整依赖锁。

远端目录 `/home/lbin/Desktop/rbf-paper-stages-20260917`；以 `evaluator-venv/bin/python -m pytest` 运行上述文件，设置 `PYTHONPATH=.`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`RBF_EVALUATOR_PYTHON=/home/lbin/Desktop/rbf-paper-stages-20260917/evaluator-venv/bin/python`。最终 XML SHA-256 为 `1742571aaab1b4a827054ed175a4ba0dccd58c6ed886635b01834e6e572d52bd`。

## 18：不能以占位实现补齐的部分

发布版本 `configs/event_track_v2x/paper/public-baselines.json` 明确区分：CoopTrack、SparseCoop、DMSTrack 有官方源码锁定与命令适配，但本论文正式运行/权重/协议还未验收；Graph Lap-CoMOT、Long-SCOPE 的完整实现未齐；CoTrack 的完整算法与配置来源未齐。

本轮再次检索一手资料：[Long-SCOPE 全文](https://arxiv.org/html/2604.09206v1)、[Graph Lap-CoMOT 全文](https://arxiv.org/html/2506.09469v1)可以读取；[CoTrack 出版方](https://www.sciencedirect.com/science/article/pii/S0893608025005064)检索摘要可见，直接全文读取失败。没有因此断言未开源，也没有用通用跟踪器冒名代替。

当时的继续条件为补齐实测时钟、detector 选择独立性和基线必要资料。此处关于实测时钟和不升级标称 10 Hz 的决定已由 2026-09-20 的正式变体决定取代；detector 选择独立性、基线资料以及不擅自修改 train/test 的限制继续有效。现阶段未提交 Git，未改论文目录、冻结主工作树或既有完成任务的报告。

## 用户确认后的全 train 时钟核验

ClearML `873f8992b7b9440b9aba811b0d063132` completed，报告已回读，SHA-256 `df230e6d3875682f7fb0a0bd6cff83bf03b4614b0e5a6ad177f0b71110812870`。

核验覆盖全部八个 train 分卷的 14,210 份逐帧 YAML 与对应 14,210 个 PCD 文件头。YAML 仅 compose 节点，不运行 Python/YAML tag 构造器，不遍历对象注释；该准备任务读取了含 GT 的 train YAML，不能称为 GT-free 数据读取，但未触及 official_test、未生成预测。

时间字段关键词、PCD 头字段和时间旁文件名候选均为 0。所有 PCD 的 FIELDS 都是 `x y z rgb`；13,390 份 YAML 顶层有 ego_speed/gps/lidar_pose/true_ego_pos/vehicles，其余 820 份没有 gps。这个检查不把 gps 向量或文件创建时间解释为传感器时钟，也不证明未公开的原始采集文件没有时间。

一手来源补核：

- [固定官方 BaseDataset 源码](https://raw.githubusercontent.com/ucla-mobility/V2V4Real/5a821e13753bafc611f95c47bc1a306acdcb0f7c/opencood/data_utils/datasets/basedataset.py)：extract_timestamps 直接读取 YAML 文件名，其说明为 mocked timestamps。
- [时间戳问题 #18](https://github.com/ucla-mobility/V2V4Real/issues/18)：本轮官方只读 API 返回 comments=[]，未取得作者对实测时间来源的回复。
- [数据重叠作者回复](https://github.com/ucla-mobility/V2V4Real/issues/21#issuecomment-1823711998)：维护者确认同一片段放入两侧是失误，并给出去除 train 片段或保留以复现原 benchmark 的选择。该回复不证明两种选择都具有独立泛化资格，本轮没有自动采纳任何划分变化。
- [官方 README](https://github.com/ucla-mobility/V2V4Real/blob/main/README.md)：明确要求推断前将 validation_dir 指向 test。因此已发布配置指向 test 本身不能证明训练时以 test 选模；仍需实际训练运行记录证明其来源。此前的来源未核实判断保留，但不把配置字段当作泄漏的直接证据。

全 train 时钟检查新增 3 项测试通过。一次从主工作树错误启动新隔离脚本测试导致 2 个收集错误，切换到正确隔离目录后，诊断/对照/重复审计四文件共 16 项通过；没有修改导入路径以掩盖错误。

所需外部材料清单见 [主协议证据请求清单](measured-clock-source-evidence-20260917.md)。没有代用户向作者发邮件或发布 issue。
