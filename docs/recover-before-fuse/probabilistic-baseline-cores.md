# JPDA／PKF 基线核心：数学定义与接入边界

本阶段新增单帧 JPDA 边缘推断、Gaussian 矩匹配、解耦 PKF 更新，以及原始森林
势函数到条件单帧匹配的转换。最终相关回归为 214 通过、0 跳过。代码位于
transvision，没有向论文项目写入研究代码。

这些模块是强基线的推断和状态更新核心。本记录保留核心阶段的验证范围；后续
已新增 [DetectionCacheV2 持久化时序适配](probabilistic-tracking-v2.md)。两阶段
都不是公开方法的端到端复现，不能将本记录登记为真实数据集结果。

## 1. 单帧后验的对象

令左侧为已存在轨迹，右侧为本次观测。复用 `LogAssociationFactors` 中原有的
正 matched/unmatched 势及固定候选支持。对合法的一对一部分匹配 $A$：

\[
w(A)=\prod_{(i,j)\in A}\psi_{ij}
\prod_{i\text{ 未匹配}}\psi_{i0}
\prod_{j\text{ 未匹配}}\psi_{0j},\qquad
q(A)=w(A)/Z.
\]

边缘概率定义为 $\beta_{ij}=\sum_{A\ni(i,j)}q(A)$，两侧未匹配概率分别为
$\beta_{i0}$、$\beta_{0j}$。因此每行和每列连同对应未匹配项的质量均为 1。
这不是对每一行单独做 softmax：不同轨迹不能独立占用同一观测。

JPDA 的全局合法配置求和和 Gaussian 混合更新可参照
[Stone Soup 官方教程](https://stonesoup.readthedocs.io/en/stable/auto_tutorials/08_JPDATutorial.html)。
这里的 $q$ 是输入势函数定义的模型分布，不自动等于校准的真实后验。

## 2. 精确模式：遗忘已封闭列的动态规划

提取全未匹配常数，并定义相对匹配权重：

\[
C=\prod_i\psi_{i0}\prod_j\psi_{0j},\qquad
g_{ij}=\log\psi_{ij}-\log\psi_{i0}-\log\psi_{0j}.
\]

于是 $Z=C\sum_A\exp(\sum_{(i,j)\in A}g_{ij})$。分量按照完整允许边图拆分，
不使用置信度阈值删边。每个分量必要时转置，使需记录占用状态的列数不增加。

处理第 $i$ 行之前，以位集合 $S$ 记录仍可能被后续行引用、且已被占用的列。
第 $i$ 行可以未匹配，或选择任一未占用的允许列 $j$。若 $E_i$ 为最后一条
入边恰好在第 $i$ 行的列集合，转移为：

\[
T_i(S,0)=S\setminus E_i,\qquad
T_i(S,j)=(S\cup\{j\})\setminus E_i.
\]

令 $F_i(S)$ 为到达该前沿的总权重，$v_i(0)=1$、$v_i(j)=e^{g_{ij}}$。
前向递推对所有合法转移求和：

\[
F_{i+1}(S')=\sum_{S,a:T_i(S,a)=S'}F_i(S)v_i(a).
\]

列一旦没有未来入边，其占用状态不再影响任何合法后续配置，因此可以遗忘。
多个过去路径到达同一前沿时相加，不取最大值。由归纳法，前向层包含且只包含
所有合法前缀的权重；最后 $F_n(\varnothing)=Z/C$。

反向权重 $B_i(S)=\sum_a v_i(a)B_{i+1}(T_i(S,a))$ 给出边缘项：

\[
\beta_{ij}=\frac{\sum_{S:j\text{ 合法}}F_i(S)e^{g_{ij}}
B_{i+1}(T_i(S,j))}{Z/C}.
\]

右侧未匹配质量在对应列封闭时直接累加，不使用 $1-\sum_i\beta_{ij}$，避免
丢失可表示的极小剩余质量。测试中的 $e^{-500}$ 未匹配质量仍大于零。

实现使用 log-domain 浮点运算，不是区间算术证书。最坏复杂度仍随前沿宽度
指数增长；400×400 稀疏链案例的成功不证明稠密真实场景可承受。

## 3. LBP 模式：近似不是遗漏质量证书

大分量可显式选择 LBP；不会在精确模式超限时自动切换。采用
[Williams 与 Lau 的关联图 BP](https://arxiv.org/abs/1209.6299) 消息形式：

\[
\mu_{i\to j}=\frac{e^{g_{ij}}}{1+\sum_{k\ne j}e^{g_{ik}}\nu_{k\to i}},\qquad
\nu_{j\to i}=\frac{1}{1+\sum_{\ell\ne i}\mu_{\ell\to j}}.
\]

代码在对数域计算排除当前项的和，使用前缀／后缀累积，避免「总和减最大项」
的相消。由消息形成轨迹行边缘和观测列边缘，并同时检查消息变化及行列一致性。

停止条件只证明数值上接近该迭代的不动点，**不证明接近完整后验**。2×2、全边
权重均为 1 的反例中，精确单边概率为 $2/7$，LBP 收敛值约为 $0.276393$。
因此 LBP 返回 `log_partition=None`、`posterior_error_bound=None`，不生成遗漏
质量证书。树形支持上的结果另与精确推断对照。

## 4. JPDA 与 PKF 的状态更新必须分开

本节假设已存在轨迹具有 Gaussian 先验，采用线性观测模型；先验误差与测量
误差独立。它不是未知相关性下的 CI。对先验 $(m,P)$、观测 $(z_j,R_j,H)$，
普通 Kalman 更新给出条件后验 $(m_j,P_j)$，未匹配分支为 $(m_0,P_0)=(m,P)$。

JPDA 更新为：

\[
\bar m=\sum_{j\ge0}\beta_{ij}m_j,\qquad
\bar P=\sum_{j\ge0}\beta_{ij}\big[P_j+(m_j-\bar m)(m_j-\bar m)^T\big].
\]

条件协方差采用 Joseph 形式；矩匹配保留分支间方差，不只平均 $P_j$。
状态角度使用以先验为锚的局部坐标，不宣称得到圆周上的全局 Gaussian 分布。

解耦 PKF 核心依据[原论文第 IV-B 节](https://arxiv.org/html/2411.06378v2#S4.SS2)
的扩展观测更新。用信息形式计算同一 M-step：

\[
J=P^{-1}+\sum_j\beta_{ij}H^TR_j^{-1}H,\qquad
m^+=m+J^{-1}\sum_j\beta_{ij}H^TR_j^{-1}(z_j-Hm),\qquad P^+=J^{-1}.
\]

因为扩展噪声块为 $R_j/\beta_{ij}$，其信息块就是 $\beta_{ij}R_j^{-1}$。
由此直接得到上述形式，并避免对极小正权重做除法。零权重不提供信息；没有
正权重裁剪。未匹配质量不能通过对匹配权重重新归一化而抹掉。

独立测试构造扩展观测矩阵及块对角噪声，逐项核对信息形式的结果。JPDA 与
PKF 共用同一边缘概率，差异仅来自本节状态更新。此处是一次 E-step 加一次
解耦 M-step，不声称 EM 已迭代收敛、保留轨迹间协方差或复现原论文完整系统。

一个先验 $N(0,1)$、两个观测 $-2,+2$、测量方差均为 1、三分支概率均为
$1/3$ 的例子中，两者均值均为 0；JPDA 方差为 $4/3$，PKF 方差为 $3/5$。
更小的方差本身不等于更准确、更一致或身份保持更好。

## 5. 原始森林势函数到条件单帧匹配

`condition_forest_scan` 要求所有新增节点来自同一真实 source/frame，且均已
到达。过去的身份根配置必须显式给出。若 $r(p)$ 为过去父节点 $p$ 的身份根，
新节点 $j$ 到既有根 $r$ 的势为：

\[
\psi_{rj}=\sum_{p:r(p)=r}\exp(\theta_{jp}),\qquad
\psi_{r0}=1,\qquad \psi_{0j}=\exp(\theta_{j,-1}).
\]

同一根的等价父路径必须求和，不能用最大父路径代替。已在过去配置中占用本次
source/frame 的根不能再次匹配。新节点之间的同帧父边本就在原森林中违反身份
互斥，也不进入匹配支持。

固定过去身份根后，合法森林扩展与部分匹配具有相同的根配置集合；每个匹配
配置内父路径的权重乘积按分配律求和，恰得到上述势的乘积。测试独立枚举合法
森林扩展，核对每个匹配的权重和配分函数。

这里右侧未匹配表示原模型的**新身份分支**，不是已确认的出生或检测存在概率。
该转换只覆盖给定过去身份配置的条件分布，不等于可恢复森林的完整历史后验。
混合来源批次被拒绝，不能为了套用一对一矩阵而禁止一个目标同时接收双端观测。

## 6. 接口与资源限制

| 入口 | 返回或行为 |
| --- | --- |
| `jpda_marginals.exact_jpda` | 精确消元的单帧边缘、绝对 log-Z、前沿工作量 |
| `jpda_marginals.lbp_jpda` | 近似边缘、消息残差及消息更新量；没有 log-Z 或后验误差界 |
| `jpda_filter.jpda_scan` | 边缘及逐轨迹矩匹配结果，不保留跨帧假设 |
| `pkf_filter.pkf_scan` | 相同关联求解器下的解耦 PKF 信息更新 |
| `jpda_forest_bridge.condition_forest_scan` | 固定过去身份根下的单 source/frame 势转换与来源绑定 |

`JPDALimits` 默认限制 1000000 个因子矩阵／向量单元、累计 250000 个前向状态、
5000000 次前向及反向转移、每分量 10000 轮 LBP、全局 10000000 次方向消息
更新。达到上限就抛错，不返回截断分布、不静默降级，也不修改输入对象。
这些是算法工作量限制，不是操作系统峰值内存或尾延迟承诺。

## 7. 验证记录

在 transvision 根目录、原研究 Python 3.12 环境运行 9 个相关测试文件：

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/private/tmp/eventtrack-v2-checks.bYZm1l/bin/python -m pytest -q -p no:cacheprovider \
  tests/event_track_v2x/test_jpda_marginals.py \
  tests/event_track_v2x/test_jpda_filter.py \
  tests/event_track_v2x/test_pkf_filter.py \
  tests/event_track_v2x/test_jpda_forest_bridge.py \
  tests/event_track_v2x/test_identity_forest.py \
  tests/event_track_v2x/test_hypothesis_bank.py \
  tests/event_track_v2x/test_recoverable_identity.py \
  tests/event_track_v2x/test_tracking_v2.py \
  tests/event_track_v2x/test_core_estimation.py
```

结果为 **214 通过、0 失败、0 跳过，1.78 秒**。证据见
[最终 JUnit](../../work_dirs/recover-before-fuse/probabilistic-baseline-core-regression-20260913-v3.xml)。
它包括 125 组小型随机图的独立全配置求和、400×400 稀疏链、资源超限、极小
未匹配质量、LBP 有环反例、Kalman／PKF 独立形式对照、角度及条件森林转换。
没有使用真实训练、validation 或 test 载荷，也没有调用官方完整 JPDA／PKF
跟踪程序作端到端对照。

首个组合回归命令误写了不存在的 `test_fusion.py`，没有执行测试；失败回执保留，
不计入通过数量。修正后 174 项通过，再加入森林转换后得到上述最终 214 项。
这些阶段计数不能相加，本次也未重跑完整 EventTrack 套件。

最终源码和回执 SHA256：

```text
jpda_marginals.py
64e8c3dde52fbc3e9ae12699334b0a8a9ac9f2b83bcd18056778cf52fc7bddce
jpda_filter.py
5cccadcb2d066cadd1be43881eb771ae71eafee7e2c314f78bab6999f2b701bc
pkf_filter.py
66eca20f5a2e5cf9f136ceebda0808f28f6c223481b223da3833eb6452b840eb
jpda_forest_bridge.py
aed7809dbd5b0b450b1fc14cdb1121878c29a5f7169e00c03e848a303c6c6753
probabilistic-baseline-core-regression-20260913-v3.xml
95bf8c03a116425c8f515566b8587747e74bbf2a51a9c5b3c167a6e25340f439
```

## 8. 核心阶段边界及后续接入

核心阶段尚未处理持久化 V2 接收与时序生命周期。后续适配已补上多源到达、
同帧互斥、出生／死亡、硬身份锚点及输出幂等，并验证同一实际训练的夹具
检查点下九个后端的原始逐行势摘要相同。实现差异和完整测试范围见
[时序接入说明](probabilistic-tracking-v2.md)，不以夹具对照代替真实全量验证。

单帧核心默认 Kalman 独立误差假设，而当前可恢复后端使用 CI；论文主表还需要
控制状态更新器的对照，不能把换用 Kalman 的收益归因于身份压缩。对跨端相关
观测的处理、运行预算和所有阈值必须在正式评估前冻结。

经典 MHT／公开方法复现、真实全量训练、同资源比较、SPD 全 val
以及 V2V4Real 冻结后的官方 test 均未完成。本阶段没有远端同步、Git 提交、
推送或 ClearML 发布。论文目标保持未完成，不能据此声称已超过强基线。
