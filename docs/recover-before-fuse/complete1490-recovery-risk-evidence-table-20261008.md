# 已验两种子完整 train 回放：恢复决策的模型风险诊断

该表覆盖 SPD seed1337、seed3407 各 46 个完整 train 序列中的全部恢复决策：分别为 786、704 个，共 1,490 个，关联 1,537 条前驱证明。seed2027 的原 rank5 归档仍损坏，未填补该种子。这里的统计对象是发生恢复的当前组件决策，不是所有跟踪帧，也不是官方 test 指标。

每格为点估计 [95% 序列 cluster bootstrap 区间]。均值和比例按决策数加权；按 46 个序列有放回抽样，同一轮对两个种子使用相同序列抽样次数。原统计固定随机种子 1337、10,000 次，全部 24 个区间均有 10,000 个有效样本。没有将决策或种子×序列当作独立抽样单位；这里分别报告两个种子，不构造三种子均值或样本标准差。

| 诊断项 | seed1337 | seed3407 |
|---|---:|---:|
| 遗漏质量上界 η 的决策均值 | 0.87485352 [0.83336144, 0.91357832] | 0.91679934 [0.89342024, 0.93952847] |
| 保留模型内的动作优化差距均值 | 7.8336118e-05 [1.2996587e-05, 0.00017172667] | 8.5321559e-05 [1.3930882e-05, 0.00017937451] |
| 保留模型条件 Hamming 风险均值 | 0.050911402 [0.041684933, 0.05998113] | 0.045579101 [0.035547467, 0.056064318] |
| 报告的模型风险界均值 | 0.87486514 [0.83337099, 0.91360785] | 0.91682647 [0.89345805, 0.93953697] |
| η ≥ 1−10⁻¹² 的决策比例 | 0.57379135 [0.49867012, 0.65007386] | 0.65625 [0.60245902, 0.71118231] |
| 风险界 ≥ 1−10⁻¹² 的决策比例 | 0.57633588 [0.50083096, 0.65255527] | 0.65909091 [0.60662089, 0.71333513] |
| 风险界 > 0.1 且优化差距 ≤ 10⁻¹⁰ 的比例 | 0.90966921 [0.88203674, 0.93496019] | 0.94034091 [0.91764706, 0.96133361] |
| 优化差距 > 10⁻¹⁰ 的比例 | 0.0076335878 [0.0025061872, 0.014045108] | 0.0071022727 [0.0015922934, 0.013431835] |
| 选中动作位于此前 current-active 集之外的比例 | 0.025445293 [0.01242236, 0.04] | 0.039772727 [0.019547797, 0.06386613] |
| 当前分区相对至少一个前驱分区改变的比例 | 0.78371501 [0.74211025, 0.82352941] | 0.765625 [0.7310902, 0.80288549] |
| 当前分区相对所有前驱分区改变的比例 | 0.77480916 [0.72954545, 0.81766716] | 0.76420455 [0.72922985, 0.80160049] |
| 保留条件风险 ≤ 0.05 且 η > 0.9 的比例 | 0.64631043 [0.59046403, 0.70216005] | 0.67329545 [0.62670694, 0.72144029] |

所有数值为无量纲均值或 0–1 比例，未将比例换算为百分数。η 和风险界均是原浮点模型诊断；风险界按原实现 `min(1, η + optimization_gap)` 报告。优化差距属于保留条件模型，不能替代对完整真实后验或跟踪质量的评价。

两种子中，风险界大于 0.1 而优化差距不超过 10⁻¹⁰ 的比例分别为 0.90966921、0.94034091。该结果支持“报告的界主要受遗漏质量限制”的模型内诊断。界接近 1 的决策比例分别为 0.57633588、0.65909091，应保留界较松这一限制。

前驱分区改变与选中此前未激活动作是回放内可追溯的机制证据，不能解释为真实身份恢复成功率。原分析未读 GT；没有重跑模型、回放或训练。全部 `formal_numeric_certificate=false`；当前表不证明恢复的真实因果收益，也不证明同资源优势。

科学计算与独立审计均在 `10.100.35.121-L40S:gpu3,4,5` 上执行 CPU 工作，CUDA 禁用。对应 Task 为 `6192b40de1d6442ea7565b43ef8e8549` 和 `918d89a0a1274f9289d266dd389d4cb4`。独立审计重算全部 240,000 个 bootstrap 指标值，最大绝对误差 7.105427357601002e−15，原容差 1e−10。任务耗时不是正式同资源成本数据。

来源及逐格追踪：

- [science-receipt.json](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-L40-complete1490-risk-decomposition-two-Task-whole-readback-20261007T150007127433Z/science/all-original-members/science-receipt.json)，SHA256 `05758c3cb42e87b2b501bcd0735ec0fa32b277af6a5da7defe4ea5af2fdc1424`。

- [complete-two-seed-risk-decomposition-intervals.json](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-L40-complete1490-risk-decomposition-two-Task-whole-readback-20261007T150007127433Z/science/all-original-members/complete-two-seed-risk-decomposition-intervals.json)，SHA256 `c8a706000236d63c17ffb2ec23fa7ed71110f06033348b70a978d6666e45cfdf`。

- [independent-audit-receipt.json](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-L40-complete1490-risk-decomposition-two-Task-whole-readback-20261007T150007127433Z/independent-audit/all-original-members/independent-audit-receipt.json)，SHA256 `c0bc28d01ccf27fdda6b5a01319c0a3d00bbb4cf6af4f192ccecf4f69c72aa65`。

- [independent-two-Task-readback-receipt.json](/Volumes/Data/test/recover-before-fuse/artifacts/rbf-L40-complete1490-risk-decomposition-two-Task-whole-readback-20261007T150007127433Z/independent-two-Task-readback-receipt.json)，SHA256 `db759c37ec5d02cd09ae2bc7218d111b499eb0bab400fc878504fcc069b641e5`。
