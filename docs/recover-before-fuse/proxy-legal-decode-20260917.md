# 冻结势函数的合法跨端关联解码

ClearML Task `449faf625bde41dab614a1bf53823ccc` 已 completed。本项在管理节点 CPU 上完成，不占用 A100。复用任务 `6726803abe7147438d37c3191375fa92` 的四份冻结 potentials，下载后逐项核对 SHA-256；不读取 GT、不更新权重、不改变候选或代理时钟协议。

## 方法与边界

复用 `predicted_association_v2.py` 中既有 `k_best_assignments` 和 `association_hypotheses`，通过 AST 提取原函数体，避免引入无关 SPD 数据依赖。源码 SHA-256 为 `6667b5d5cac6079a578c97437d7043dbeee331f65ef0c25605d06e83851f3cb4`。运行 Top-1、Top-3，保留全部未匹配动作，核对一对一约束、token 完整覆盖、有限能量、动作去重及保留集合内权重归一化。保留集合内的权重不是完整后验或校准概率。

三个种子各三档模拟延迟、每档 20 帧，共 180 个主决策。第四份输入为种子 1337 重复，共 60 个决策，仅检查一致性。完整候选保持每个可用源 64 个；200 ms 延迟组首帧远端为空。每个决策另取至多 3×3 子问题作独立穷举，核对 Top-3 能量；子问题不替代完整候选解码。

沿用独立 nominal-10hz-proxy-replay-v1 协议。本项不输出时序轨迹、稳定身份或恢复事件，不做跟踪评价。数据是训练内短序列，检测器为既有单类别头，`paper_eligible=false`，不作为主协议实验结果。

## 结果

| 种子 | 主决策数 | Top-1 全未匹配 | Top-3 MAP 全未匹配 | 非空决策中的最大边增益 |
|---|---:|---:|---:|---:|
| 1337 | 60 | 60 | 60 | -3.980753416195512 |
| 2027 | 60 | 60 | 60 | -2.292308896780014 |
| 3407 | 60 | 60 | 60 | -4.412734866142273 |

所有主决策的 MAP 配对数为 0。Top-3 每种子保留 178 个动作（59 个非空决策各 3 个，1 个远端空决策 1 个），并非所有备选动作都没有配对。第四份 1337 重复的规范化解码结果完全一致。

独立全边检查使用 `gain(i,j) = 2*pair(i,j) - left_unmatched(i) - right_unmatched(j)`。对既有对称行列 log-softmax 能量，匹配一条边与两端都未匹配相比，归一化项抵消；四份输入中没有任何决策包含正增益边。因此，在当前能量定义下，全部未匹配是严格优于任意非空匹配的 MAP 选择。该检查支持这是冻结模型分数与当前能量共同产生的偏好，不能据此认定具体训练根因；未临时改阈值或强制配对。

## 验收与资产

- 相关测试 8 passed（0.21 s）：包含 6 类形状、每类 8 个随机独立穷举用例，以及类别掩码/空动作和非法列重复检查。
- 四份输出各 60 个决策，合计 240 次运行时小问题穷举核对通过；完整候选合法性和 Top-1/Top-3 MAP 能量一致性通过。
- ClearML 状态 completed，`decode-report`、`decoder-source`、`runner-source` 及四份 `decoded-potentials-seed-…` 资产均已发布；四份解码输出回读后的规范化 JSON SHA-256 与报告一致。
- 规范化输出 SHA-256：1337/device0 与 device3 均为 `807bc5a1d5124a8fc35f6bd92ae0e014f6652201ddf4451de4f0b64918d93727`；2027/device1 为 `baac0c9b3b5dd1f1884d912c52586f2ba7dc0c2e7a52e0b82d0d4a4c788fef22`；3407/device2 为 `cf44ce5cda8d071f39e6ef3a870ba657f856f86fcc035cd9ce9dd174523743ee`。这些是规范化内容哈希，不冒充上传文件字节哈希。
- 全边增益是 completed 后的独立只读核验，记录于本回执，未修改已完成任务的报告。

代码位于隔离工作树 `work_dirs/rbf-pointpillar-runtime-20260917`：`tools/event_track_v2x/run_v2v4real_proxy_decode.py` 与 `tests/event_track_v2x/test_proxy_decode.py`。未修改冻结主源码或提交 Git。

远端部署目录 `/home/lbin/Desktop/rbf-proxy-decode-20260917`，复现入口：

```bash
cd /home/lbin/Desktop/rbf-proxy-decode-20260917
CLEARML_FILES_HOST=http://10.100.35.118:8081 \
CLEARML_CACHE_DIR=/home/lbin/Desktop/rbf-next-experiment-cache \
/home/lbin/miniconda3/bin/python tools/event_track_v2x/run_v2v4real_proxy_decode.py \
  --decoder-source transvision/models/event_track_v2x/predicted_association_v2.py
```

入口发现同源码已完成任务时返回已有 ID，不重复创建。环境需要 ClearML、PyTorch、NumPy、SciPy；本项没有 GPU 依赖。CPU 耗时包含初次调用开销，不作为等资源性能比较。

下一步应诊断训练目标、未匹配分数及从 331 个 GT-matched token 训练到完整候选推断的分布变化。当前证据只表明合法解码接入成功，同时暴露全未匹配问题；不表明跨端关联或论文性能目标已实现。
