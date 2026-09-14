# JPDA／PKF 的持久化 DetectionCacheV2 接入

新增三种单历史时序基线：JPDA-CI、JPDA-Kalman 和解耦 PKF。它们接入现有
`DetectionCacheV2` 接收、冻结逐行势函数、输出格式及审计链。实现全部位于
transvision；论文项目保留构建工具，不新增研究代码。

这是本地基线适配，不是公开程序的端到端复现。后续已使用完成四卡 A100 训练的
身份模型进行 train 诊断，并开始[完整 SPD val 验证](spd-probabilistic-full-val-20260914.md)。
完整 val 指标及同资源比较尚未完成；所有结果仍标记 `paper_eligible=False`。
以下方法规格与历史测试记录保留，最新执行命令以完整 val 记录为准。

## 1. 推断对象和硬身份锚点

每次到达的连续 source/frame 观测组成一个 scan。固定过去已提交的身份根
$r(p)$，由原始父节点势构造条件匹配势：

\[
\log\psi_{rj}=\operatorname{logsumexp}_{p:r(p)=r}\theta_{jp},\qquad
\log\psi_{r0}=0,\qquad \log\psi_{0j}=\theta_{j,-1}.
\]

同根父路径求和；已经占用当前 source/frame 的身份根被排除。双端观测分别
处理，允许同一目标在一次输出前接收两端信息。交错重复的 source/frame scan
被拒绝，不通过重新排序悄悄改变到达语义。

关联可显式选择 exact 或 LBP，数学定义与数值限制见
[单帧核心](probabilistic-baseline-cores.md)。边缘概率用于状态更新，硬身份锚点
用于 ID、原始观测成员及生命周期。右侧未匹配概率不是检测存在概率，不能乘到
原检测分数上。

硬身份解码器有两个选项，必须在真实评估前冻结：

| 选项 | 目标 | 相对全新生配置的匹配代价 |
| --- | --- | --- |
| `joint-map`，默认 | 最大化条件联合配置权重 | $\log\psi_{0j}-\log\psi_{rj}$ |
| `marginal-bayes` | 最小化新增观测的逐节点身份根 0/1 损失之和 | $\beta_{0j}-\beta_{rj}$ |

两者均通过带新生虚拟项的 Hungarian 求解，满足一对一约束。选定根内使用最小
父节点序号作为可校验的代表路径，不用该路径的权重代替整个根类别的权重。
LBP 模式下，`marginal-bayes` 只针对近似边缘最优，不是真实后验 Bayes 保证。

一个旧根、两个对称候选、匹配势均为 1、新生势均为 $\varepsilon=e^{-8}$ 时：

\[
Z=2\varepsilon+\varepsilon^2,\quad
\beta_{rj}=\frac1{2+\varepsilon},\quad
\beta_{0j}=\frac{1+\varepsilon}{2+\varepsilon}.
\]

逐节点 Bayes 解码会选两个新生；联合 MAP 则选一个匹配和一个新生。这是不同
损失的决策差异，不是数值错误，也不说明哪个选项的 HOTA 更高。该反例保留为
测试，禁止在 val 上选择有利的解码器后再称其预先冻结。

## 2. 状态更新和时间处理

每个硬身份根只保存一份 Gaussian 状态。所有允许关联的软边缘参与旧根更新，
新生硬锚点初始化独立状态。三种模式的差异为：

| 模式 | 条件更新及压缩 | 关键限制 |
| --- | --- | --- |
| `jpda-ci` | 对每个测量做 CI，再连同未匹配先验进行矩匹配 | 不是完整历史混合分布 |
| `jpda-kalman` | Joseph 形式的 Kalman 条件更新，再矩匹配 | 假设先验与测量误差独立 |
| `pkf` | 使用未重新归一化的关联边缘做解耦信息更新 | 不保留跨轨迹协方差 |

矩匹配包含分支间方差。PKF 不裁剪可表示的正关联权重，也不把匹配质量归一化
到 1。核心公式及独立数值对照见[单帧核心](probabilistic-baseline-cores.md)。

原始测量投影到当前输出参考时刻，随后按 scan 顺序更新。已经压缩的先验只向
前传播；已合法到达、但源状态时刻晚于输出参考时刻的原始测量允许负时间投影，
其数量写入审计。尚未到达的输入仍被拒绝。

该近似不是精确 OOSM，也不是可恢复后端按源状态时刻重放原始分支的协议。
即使选择 `jpda-ci`，也不能声称只改变了身份压缩一项。论文仍需独立完成同状态
更新器、同时间协议的恢复消融。

轨迹出生阈值、存活分数衰减和最大年龄复用硬锚点原始观测。分数取原始检测
分数的相应最大值，不乘右侧未匹配质量。软更新不会自动改变硬观测成员或延长
其生命周期；这是一项明确的基线设计，不能称为完整存在概率滤波。

## 3. 持久化、重开和审计

入口为 `PersistentProbabilisticTracker`。数据库新增 `identity_anchors` 和
`gaussian_tracks`，与原始观测、势函数、缓存接收记录和输出在同一事务提交。
推断或状态更新失败时全部回滚，重试不留下半次接收。

只维护一个不可回改的历史身份图；磁盘上的原始因子保留，不代表该基线具有
身份恢复能力。旧输出不回写。正常关闭后，使用数据库摘要和最后预测摘要重开，
并检查锚点、Gaussian 状态、因子、配置及输出审计摘要。当前验证不包含断电或
崩溃中断后的恢复保证。

重复事件返回原提交；重复缓存消息沿用最初到达记录，不重复读取载荷或更新
状态。审计记录每个 scan 的势函数、边缘、硬锚点、求解器工作量及状态更新量。
同时删除继承接口中不适用于本基线的完整历史质量上界字段，明确记录：

```text
recovery_enabled = false
same_state_time_protocol_as_recoverable = false
full_history_posterior_bound = null
unmatched_mass_scales_detection_score = false
paper_eligible = false
```

状态工作量包含测量投影、先验／输出投影、条件 Gaussian 更新或 PKF 信息项、
状态写入。它不是与其他后端等价的 FLOP 单位，不能据此声称计算预算相同。

每个 scan 默认最多 2048 个候选旧根；因子矩阵分配前检查容量。`JPDALimits`
约束单 scan 的 DP／LBP 工作量，不是整个事件的累计推断预算。历史观测、前缀
和输出另受持久化上限约束；继承的保底新生前缀也实际占用存储。历史生命周期
查询仍会扫描磁盘标量记录，不声称恒定总内存、磁盘、尾延迟或稠密场景可扩展。

## 4. 冻结后的全 SPD val 入口

`tools/event_track_v2x/run_probabilistic_tracking_v2.py` 只接受完整官方 SPD val
日程和缓存：21 个序列、3316 次输出、7189 个源帧。缓存、日程、检查点及源码
摘要写入计划；运行后检查源码未变，再生成完整回放回执。

提供检查点时，必须来自完整官方 train，且上游检测／特征生产者与缓存一致。
夹具检查点被拒绝。不提供检查点时是显式几何开发基线，不会冒称学习式结果。
本程序不训练、不读取 GT、不做 val 参数或检查点选择、不发布到 ClearML。

在仓库根目录运行以下模板前，先将变量设置为已冻结的实际路径和 SHA256；
`RBF_OUTPUT` 必须是新的结果目录。此模板不是已完成真实回放的记录。

```sh
python tools/event_track_v2x/run_probabilistic_tracking_v2.py \
  --cache "$SPD_VAL_CACHE" --cache-sha256 "$SPD_VAL_CACHE_SHA256" \
  --schedule "$SPD_VAL_SCHEDULE" --schedule-sha256 "$SPD_VAL_SCHEDULE_SHA256" \
  --checkpoint "$RBF_CHECKPOINT" --checkpoint-sha256 "$RBF_CHECKPOINT_SHA256" \
  --association lbp --update-rule jpda-ci --anchor-decoder joint-map \
  --output "$RBF_OUTPUT"
```

范围只含 car。SPD val 已参与研究，不称为未见确认集；SPD test/test_A 不进入
该入口。V2V4Real 按用户授权保留官方 test 作为冻结后的最终评估，其数据和
适配工作不由本入口替代。

## 5. 已验证范围及后续工作

三个更新模式均完成正式 V2 格式的夹具回放、输出校验、事务回滚、重复消息和
正常重开测试。另有 60 帧、120 个观测的时序夹具，在第 30 帧后关闭重开，三种
模式的预测均与连续运行逐字节一致；不声称缓存相关审计元数据也逐字节一致。

用实际训练过的同一夹具检查点，对九个后端逐事件核对原始
`factor_rows_sha256`：单体可恢复、分量可恢复、逐节点 beam、预选笛卡尔积
batch beam、联合笛卡尔积 beam、学习式预算分配、JPDA-CI、JPDA-Kalman、PKF。
摘要全部一致。该证据证明夹具上的原始行输入一致，不证明真实全量输入、状态
模型、资源成本或精度相同。

具体测试回执见[接入验证记录](probabilistic-validation-20260913.md)。经典 MHT、
公开方法复现、新方法真实全量训练、严格同资源实验、SPD 全 val 和 V2V4Real
冻结后的官方 test 仍未完成；当前不能据此支持论文的性能或创新结论。

后续已新增[新进程 CPU 资源测量入口](resource-sweep.md)，支持共同原始输入的
重复回放和逐帧测量，但尚未完成真实数据的同资源对照。

## 6. 2026-09-14 完整训练序列实测

三种适配后端现已完成固定 train 序列 `0001` 的 183 帧回放和原生指标计算。
[实测与硬身份不变性记录](spd-probabilistic-seq0001-20260914.md)保存了全部三组结果，
以及与关闭恢复 beam 的共同原始因子流核验。不是全量 val 或同资源实验。
独立数据库审计还确认：三组硬身份历史、输出 ID、成员及分数相同，但连续状态和
身份指标不同。因此不能把指标差值直接解释为身份决策差值。
