# V2V4Real 分卷核查与训练数据重叠审计

本页记录 2026-09-13 的来源核查及重叠审计入口。官方分卷元数据和重叠检查已实现；当天后续已完成 `train_04.zip` 载荷核验和真实输入投影，见 [分卷接入记录](v2v4real-native-inputs-20260913.md)。完整下载、真实跨 split 重叠审计、训练资格和 test 性能验证仍未完成。研究代码及记录均保存在 transvision，论文构建工具未改。

## 官方目录的新证据

本次通过[官方公开 Box 目录](https://ucla.app.box.com/v/UCLA-MobilityLab-V2V4REAL/folder/279924274808)核实到 12 个文件：8 个 train 分卷、3 个 test 分卷和 `val.zip`。目录显示发布者为 XU HAN、企业所有者为 UCLA。这更新了此前「未见独立 val 下载项」的历史记录，但不改变本论文以官方 test 做最终评价、仅用 train 做训练和选择的协议。

以下是本次新增确认的文件身份；版本号、发布端 SHA-1 和完整训练分卷列表已保存到 `configs/event_track_v2x/v2v4real-release-20260913.json`。

| 文件 | Box file ID | 压缩字节数 |
|---|---|---:|
| [test_01.zip](https://ucla.app.box.com/v/UCLA-MobilityLab-V2V4REAL/file/1619923463037) | 1619923463037 | 633,582,640 |
| [test_02.zip](https://ucla.app.box.com/v/UCLA-MobilityLab-V2V4REAL/file/1619910080470) | 1619910080470 | 1,228,581,180 |
| [test_03.zip](https://ucla.app.box.com/v/UCLA-MobilityLab-V2V4REAL/file/1619915891150) | 1619915891150 | 1,551,328,836 |
| [val.zip](https://ucla.app.box.com/v/UCLA-MobilityLab-V2V4REAL/file/1619910191481) | 1619910191481 | 1,045,857,068 |

三个 test 分卷合计 3,413,492,656 字节；8 个 train 分卷合计 10,934,461,824 字节。以上不包含解压、GT 隔离、检测缓存、模型和临时副本所需空间。`val.zip` 只核查了公开元数据，未纳入本实验输入。

SHA-1 是发布端提供的完整性元数据，不是现代抗碰撞安全保证，也不能替代本地 SHA-256 和归档内容清单。原元数据快照保留 `payload_verified=false`，不改写历史状态；后续 `train_04.zip` 的载荷核验由独立分卷回执证明。

早期直接下载地址检查返回 HTTP 401，且未确认文件落盘。后续通过 Chrome 官方公开页面及正常 Save 对话框成功下载 `train_04.zip`，未提取会话凭据或绕过认证。历史 401 不能用于判定数据必须付费，也不能据此重试已完成的分卷。

## 重叠审计范围

作者曾[确认 train/test 中存在重复片段](https://github.com/ucla-mobility/V2V4Real/issues/21#issuecomment-1823711998)。旧问题不能证明本次发布版本仍然重叠或已修复；必须检查实际输入。

实现位于 `transvision/models/event_track_v2x/v2v4real_overlap.py`。审计器先用外部固定的 manifest SHA-256 调用 `load_prepared_frames()`，完整检查双方的位姿/点云投影，再比较：

- 相同原始序列 ID。
- 显式采集会话映射中相同的 session ID。
- 完全相同的点云文件字节 SHA-256，即使序列、CAV 或 frame key 已改名。

审计不读取原始 YAML、GT 标签或模型输出，不计算 test 性能，不修改任何输入。以 `testoutput_` 开头的 train 目录不会仅因名称前缀而被排除。

训练内部的分组先按同一会话及相同点云字节建立连通分量。这一步只使用 train。若某个 train 节点与 test 重叠，回执另列出整个训练分量的受影响范围；不自动删除样本，不改变 test 清单，也不把筛除后的 train 称为完整官方 train。

完全相同字节是保守的重叠信号，不等于已经识别了同一物理事件。反过来，没有字节匹配也不能排除点顺序变化、重编码或近重复帧。会话映射证据的真实性和公开权重的训练/选择来源仍需独立核查。

## 运行前提

先按 [原始输入准备说明](v2v4real-inputs.md)生成 train/test 两套投影，并独立保存各自的 `input_manifest_sha256`。

准备显式的 session map JSON：

- `kind` 为 `v2v4real_session_map_v1`，`dataset` 为 `V2V4Real`。
- `train_manifest_sha256`、`test_manifest_sha256` 绑定完整输入。
- `session_evidence_sha256` 绑定已核查的采集会话来源说明文件。
- `sequences` 包含且只包含 `train`、`test` 两个对象。每个对象将全部原始序列名映射到非空 session ID，不得漏项或增加其他 split。

不能仅删除序列名末尾的 `_0`、`_1` 来生成已经核实的会话身份。准备器保存证据文件摘要，但不自动判断证据陈述为真。

在 shell 中先设置来自独立回执的 `TRAIN_MANIFEST_SHA`、`TEST_MANIFEST_SHA`，再运行：

```bash
python tools/event_track_v2x/audit_v2v4real_overlap.py \
  --train-inputs /data/prepared/v2v4real-train-inputs-v1/inputs \
  --test-inputs /data/prepared/v2v4real-test-inputs-v1/inputs \
  --train-manifest-sha256 "$TRAIN_MANIFEST_SHA" \
  --test-manifest-sha256 "$TEST_MANIFEST_SHA" \
  --session-map /data/protocol/v2v4real-sessions.json \
  --session-evidence /data/protocol/v2v4real-session-evidence.md \
  --output /data/audits/v2v4real-overlap-v1.json
```

输出文件必须不存在，且必须放在两个不可变输入目录之外。错误时不覆盖已有审计。

| 退出码 | 含义 |
|---:|---|
| 0 | 已实施的检查未发现重叠；不等于训练资格通过 |
| 3 | 发现重叠，审计文件已保存，应停止接受开发 fold 或训练资格 |
| 2 | 配置、输入完整性或输出路径检查失败 |

`report_sha256` 是去掉该字段后、按排序键和紧凑分隔符编码的 JSON 内容摘要，不是带缩进输出文件的字节摘要。后续执行应独立保存该值；自带摘要不是真实性证明。

`validate_development_folds(report, assignments)` 检查拟定的 train 序列级 fold 映射：必须覆盖全部 train、至少两个 fold、无 test 序列，且会话/重复内容连通分量不得跨 fold。发现未解决的跨 split 重叠或审计摘要不符时拒绝接受。该函数不自动选择 fold，也不授权训练。

## 本轮证据与边界

合成输入上的重叠审计与原始准备回归共 108 项通过，包含重命名副本、会话连接的传递污染、fold 泄漏、清单绑定、输入篡改和不写回输入的检查。测试报告：`work_dirs/recover-before-fuse/v2v4real-overlap-20260913.xml`。

加入原 SPD `DetectionCacheV2` 与森林缓存消费测试后，最终相关回归为 151 项通过（2.33 秒）。最终报告为 `work_dirs/recover-before-fuse/v2v4real-overlap-final-20260913.xml`，SHA-256 为 `08e041d151dc538eb9a700dabe53fae839db5d7cb6a9fda356a14675ba5dd60b`。这不是全仓测试，也不是正式数据性能验证。

真实数据重叠审计尚未运行。所有回执固定保留 `official_split_membership_verified=false`、`session_provenance_verified=false`、`training_eligibility_verified=false` 和 `paper_eligible=false`。只有核实真实归档及独立来源后，后续资格流程才能作出进一步结论；不得手动翻转这些标记来启动正式实验。
