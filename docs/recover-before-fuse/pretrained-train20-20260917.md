# 公开权重与真实 train 短序列前向验证

## 资产与协议边界

用户提供桌面的 `late_fusion.zip` 和 `train_04.zip`。已核对两个包的 SHA-1 与先前官方页面记录一致，并核对 train SHA-256。未移动或修改桌面正本。

| 资产 | SHA-256 |
| --- | --- |
| train_04.zip | `aab4d7753b5bb62221f1a81602c52df168c0c263c578ec0ddd35c8fbd1be008c` |
| late_fusion.zip | `393e64854a5b9fe7b86d490f9d6a0180178348e5f69fa174261308a7b36e06a8` |
| config.yaml | `6beea95da0830f763cce8b4580269e589ee4f555d225c995648038f6dda6e8fc` |
| net_epoch60.pth | `9221830d15badc39196164575d49f08dea05e02cea59ef387ed506de919e3fc9` |
| GT-free real-input.zip | `3a0c61991d6a30f0fb4f8d73365b5f4275e34ba1d96ebc79b02a727a8439e317` |

模型包实际配置 `validate_dir=/media/runshengxu/easystore/kitti2opv2v/test`。这不能证明发布权重的实际选模过程，但无法支持训练/选模未接触 test 的结论。因此本次只做公开权重接入验证，`paper_eligible=false`；不能作为主协议无泄漏正式性能证据。

## 固定输入与验收条件

- 从已哈希固定的官方 train 分卷按序列名排序选第一段 `testoutput_CAV_data_2022-03-17-12-04-22_0`，取最早 20 个双车 frame key，共 40 个 PCD。名字中的 testoutput 不改变其官方 train 分卷归属。
- 约 63 MiB 输入包只包含 40 个 PCD、模型配置、权重与哈希清单，不含原始标签 YAML；打包测试验证标签排除、配对与路径越界拒绝。
- 采用已通过四 A100 验证的原生官方 PointPillar/体素化代码，固定源码提交不变；使用官方 PCD 读取、ego mask、range mask，保持点顺序，无随机增广。不是逐字复现原论文完整 loader。
- `torch.load(weights_only=True)` 读取权重，`load_state_dict(strict=True)` 校验完整模型匹配。配置只允许固定哈希内容，特定 NumPy 网格字段转换为与几何独立核对的 `[352,200,1]`，禁止其他 Python YAML 构造，不执行 yaml_parser。
- 四卡分担 40 个源帧；另在每卡处理同一参考源帧，要求输出形状正确、数值有限、卡间 `atol=1e-6, rtol=1e-5` 一致。
- 这是原生体素化和预训练模型头部前向验证；不做框解码/NMS/DetectionCacheV2、不计算检测或跟踪指标、不训练、不是 DDP。不得将运行通过等同完整检测器或论文性能通过。

## 提交回执

- ClearML：`cd7b2edc7c414588a4ad4b641bb8f909`
- 队列：`GPU4-A100`，提交前要求空闲且不与现有任务重叠的四 A100。
- 提交回执：`queued`；随后于 2026-09-17 06:30:07 UTC 核验为 `in_progress`，实际 worker 为 `10.100.34.18-A100:gpu4,5,6,7`，已输出 `RBF_PRETRAINED_STAGE dependencies`。完成未验证。
- 入口 SHA-256：`db8a2b2ed55fd4d41bba9022068cae9ced5b2d5a2bdcd8d97271f4eed6a4c348`
- 运行辅助脚本 SHA-256：`87e74c247e7b34708b9f5e61927e2b48a81fcc01ad884841d5f2349b61541b36`
- 回归：46 passed、1 skipped；跳过的是可选本地 CPU 官方源码集成。实际发布配置的受限解析另行通过。
- 源码/测试保留在 `work_dirs/rbf-pointpillar-runtime-20260917` 隔离工作树，新入口为 `tools/event_track_v2x/run_v2v4real_pretrained_smoke.py`，未提交或推送 Git。
- 本地输入包：`work_dirs/recover-before-fuse/pretrained-train20-20260917/real-input.zip`。
- 部署目录：`10.100.35.112:/home/lbin/Desktop/rbf-pretrained-train20-20260917`。

成功需核对 `RBF_PRETRAINED_REPORT` 中 40 个唯一源帧、严格权重加载、四卡参考结果，以及 ClearML 的 completed 终态。排队与启动不算成功。
