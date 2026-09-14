# 经典多假设基线：K-best 部分匹配组件

`tools/event_track_v2x/ranked_partial_assignment.py` 已实现带计算上限的 K-best 部分匹配。
它是后续经典 MHT 基线的单扫描组件，尚未实现跨帧分支状态、全局历史排名、生命周期
或 DetectionCacheV2 回放。当前实验没有导入它；运行中的 114 文件推断绑定保持不变。

排序依据来自经典 [Murty 排名思想](https://pubsonline.informs.org/doi/10.1287/opre.16.3.682)，
这里适配允许两侧 unmatched 的矩形问题，不将该算法包装为论文创新。
单个区域的最优解使用 SciPy 1.14.1 的 `linear_sum_assignment`；其实现是修改后的
[Jonker–Volgenant 算法](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.optimize.linear_sum_assignment.html)，
不是本项目自行实现的 Hungarian。下面给出本实现的映射与覆盖论证。

## 1. 部分匹配与矩形分配的对应

设左侧有 \(n\) 个对象，右侧有 \(m\) 个观测。合法配置用
\(a_i\in\{-1,0,\ldots,m-1\}\) 表示，\(-1\) 为左侧 unmatched；非负列必须互异，
且满足 `allowed` 门控。matched 势与两侧 unmatched 势均为正，输入为有限对数势。
定义

\[
b=\sum_j\log\psi_{\varnothing j},\quad
v_{i,-1}=\log\psi_{i\varnothing},\quad
v_{ij}=\log\psi_{ij}-\log\psi_{\varnothing j}.
\]

则完整绝对权重满足

\[
\log w(a)=b+\sum_i v_{i,a_i}.
\]

减去右侧 unmatched 项是必要的：配对后该右侧观测不再同时贡献 unmatched 权重。
每个左侧行只允许使用自己的一个私有虚拟列，其他行的虚拟列禁用；不添加虚拟右侧行。
于是每个合法部分匹配恰好对应一个有限矩形分配，不因虚拟行排列产生重复假设。
求解器逐行减去该行最大允许势，转成非负最小化代价；该平移对每个合法分配增加相同
常数，不改变最优解。输出权重仍从原始势重新计算，不使用平移后的代价充当概率。

## 2. 首个分歧位置给出互斥区域

一个区域保存固定前缀 \(p\) 和排除集合 \(E\)。设该区域最优配置为 \(a^*\)，
前缀长度为 \(r\)。对每个 \(i=r,\ldots,n-1\) 创建子区域：固定
\(a_{<i}=a^*_{<i}\)，并排除 \((i,a_i^*)\)，同时保留原排除条件。

任意其他合法配置 \(a\ne a^*\) 有唯一的首个分歧行 \(i\ge r\)，因此恰好属于一个
子区域；反之，每个子区域都不含 \(a^*\)。子区域互斥，其并集等于父区域去掉最优配置。
矩形部分匹配必须包含最后一行的分支：最后一行仍可能在 matched 与 unmatched 间变化，
不能照搬某些方阵实现省略最后一行的写法。

未求解区域以区域总质量上界作为队列键；总质量也不小于其中任意单个配置的权重。
求解后用该区域的最优绝对权重作为队列键。只有已求解项成为队首时才输出，因此在精确
算术与精确最优求解条件下，每次输出均不劣于任何尚未输出配置。
同权排序依赖固定 SciPy 实现和队列插入次序，不承诺全局字典序。

## 3. 剩余质量与预算停止

令 \(C_R\) 为固定前缀占用的右侧列，\(J_i(R)\) 为门控、排除条件以及
\(j\notin C_R\) 共同允许的剩余列。保留固定前缀，放松剩余行之间的列唯一约束，得

\[
U_R=\exp\!\left(b+\sum_{i<r}v_{i,p_i}\right)
\prod_{i\ge r}\left[
\mathbf 1_{(i,-1)\notin E}\exp(v_{i,-1})+
\sum_{j\in J_i(R)}\exp(v_{ij})\right].
\]

每个合法配置都是乘积展开中的一项，额外项均非负，所以 \(Z_R\le U_R\)。
由于未输出区域互斥且覆盖完整剩余支持集，令 \(U=\sum_R U_R\)、
\(Z_K=\sum_{a\in K}w(a)\)、真实剩余质量为 \(T\)，有

\[
T\le U,\qquad
\eta=\frac{T}{Z_K+T}\le\frac{U}{Z_K+U}.
\]

默认限制为最多 10,000 次分配求解、10,000 个队列区域、1,000,000 个矩阵容量单位
和 4,096 个输出。矩阵预检使用 \(n(m+n)+n+m\)，不把该限制称为实际内存字节上限。
耗尽求解预算时保留当前区域；子区域无法全部容纳时，恢复父区域且不输出其最优项，
从而不会产生未记录的遗漏。`requested_k_reached` 与 `support_exhausted` 分别表示
取得所需 K 个配置和穷尽支持集，不能混用。

实现使用 log-domain float64 与上侧留量；上述数学不等式针对精确算术。
`interval_arithmetic_certified=False` 明确说明这不是严格区间算术证书，极端动态范围
仍需单独数值审查。此质量界只约束固定扫描的模型分布，不是完整历史、未来证据、
真实身份损失或 HOTA 的界。结果保留区域约束与因子摘要，但尚无持久化／恢复调用接口。

## 4. 已执行验证与未完成范围

组件的 60 项测试使用独立笛卡尔积穷举参照，覆盖 0–4 行列、门控、两侧 unmatched、
同权去重、权重平移、Top-K 前缀、预算耗尽后的互斥完整覆盖和非预期求解器异常。
另有 32×32 稠密问题检查，生产实现未调用穷举器。与已有 JPDA 桥接／边际测试合并，
最终 JUnit 为 111 项通过，0 失败、0 错误、0 跳过；其中 60 项属于本组件。

| 文件 | SHA-256 |
| --- | --- |
| `tools/event_track_v2x/ranked_partial_assignment.py` | `8a97903cba29560bcdd7bb8d50035360f00766c6336be1a3b2b854381882452a` |
| `tests/event_track_v2x/test_ranked_partial_assignment.py` | `a956f67e9fe1ba2fe99e32960a9bbf2571e9a18ab78e09d7e654b637a6ae60e9` |
| `ranked-partial-assignment-tests-v2.xml` | `0636307ef05d6c0044fb5b1c8647084cbaedc7956a91effa8c6c2943c053115e` |

另读取已完成 seed-1337 JPDA-CI 的真实 SPD val 记录，固定选择记录顺序的前 8 个条件
扫描，不依据指标筛选。从 `log_pair`、`log_birth`、`allowed` 重建输入，左侧 unmatched
对数势为 0，并逐一扫描核对原因子摘要。按缓存中的 `anchors` 重建原 joint-MAP 配置，
比较双方的一最佳绝对对数权重。8 个扫描的差值均为 0；前两项无历史对象，仅有一个合法
配置，其余六项均取得 K=4，分别调用 59、62、7、51、40、10 次分配求解器。

该抽样只验证输入兼容性与一最佳权重，不验证真实数据的所有 Top-K 排名或完整 MHT。
没有重新运行跟踪器、读取 GT、训练参数或重新计算指标。源完整回执摘要为
`173385770346deb8564ddde75ea6c0f225b118b5364f52b32eacbcb2b829942e`，源 tracking 摘要为
`fd0f75f998176abb9cc07921e0ad537d40633e0cd3b80d0203caaa9466f7657c`，结束后再次核对。
测试 XML 与 `ranked-assignment-real-prefix-v1.json` 均保存在
`/private/tmp/spd-identity-fulltrain.jtGEnO/`，后者仅含扫描形状、计数与摘要。
该抽样记录的 SHA-256 为
`8208dfff5ffc138e1ecc8c95283d2801b9ee63893e06cdfa79389d99942be651`。

完整经典 MHT 仍需分别维护各历史的连续状态，以各历史自身的势展开，再进行全局排名、
生命周期管理和不可回写的因果输出。不能复用单一路径生成的整条因子流，假称验证了
所有备选历史。完成该接入、原生评价和同资源比较前，不在论文主表填入 MHT 结果。
