# V2V4Real 首个真实分卷接入

已完成官方 `train_04.zip` 的下载、载荷校验、全卷 YAML 兼容性修正和不含标签的输入投影。范围为 4 个 train 序列、438 个双车时刻、876 份源观测；没有运行检测器、参数训练、跟踪或 test 指标。A100-only、每个训练任务至少 4 卡的约束不变。

## 真实输入纠正了两项假设

最初的准备器只在合成文件上测试，禁止全部 YAML alias，并假定位姿为六维数值。该分卷 876 份 YAML 全部使用 NumPy 标签及 alias，`lidar_pose` 均为 4×4 矩阵，因此旧读取器全部拒绝。

新增 `v2v4real_numpy_yaml.py` 只解码固定浮点语法：dtype 参数、字节序、二进制长度、数组形状和版本均有白名单。它不调用标签所命名的 Python 函数，不使用 pickle 或任意对象构造。未放开通用 alias、递归对象、merge key 或配置中的 `yaml_parser`。

矩阵按原值保存，不转换为欧拉角，也不做正交化。全卷最大旋转正交残差为 `1.2450415081133315e-6`，最大行列式残差为 `1.387115193041133e-6`。导入校验使用 `1e-5` 容差，避免把微小序列化残差误判为无效位姿。这个改动不涉及跟踪性能调参。

官方代码也区分矩阵与六维输入分支；源间矩阵变换采用 `inv(T_target) @ T_source`。[官方变换源码](https://github.com/ucla-mobility/V2V4Real/blob/5a821e13753bafc611f95c47bc1a306acdcb0f7c/opencood/utils/transformation_utils.py)

## 分卷及字段证据

来源：[官方 train_04.zip](https://ucla.app.box.com/v/UCLA-MobilityLab-V2V4REAL/file/1619912061459)，Box 文件 ID `1619912061459`，元数据快照版本 `1780984576659`。压缩长度为 `832603449` 字节，解压文件总长度为 `2345591197` 字节。

| 原始序列后缀 | CAV | 双车时刻数 |
|---|---|---:|
| 2022-03-17-12-04-22_0 | 0、1 | 124 |
| 2022-03-17-12-14-33_0 | 0、1 | 89 |
| 2022-03-17-12-14-33_1 | 0、1 | 80 |
| 2022-03-17-14-44-47_1 | 0、1 | 145 |

所有序列名保留 `testoutput_CAV_data_` 前缀；不能仅凭名字将这些 train 序列判为 test。两端 YAML/PCD 文件键逐一一致，不取交集或丢帧。

全卷按「源帧中的目标记录」统计得到 Car 10,476 条、ConcreteTruck 280 条、Truck 225 条。这不是独立物体数。论文继续采用严格 Car；另外两类不得静默合入 Car。

7,859 条目标记录的 `ass_id=-1`，其余 3,122 条为其他值。审计保留原值，不将负值映射为一个公共身份，也不把此计数当作 GT 身份协议已经核定。当前未完成原生标签参考系、跨车身份去重和跟踪评估器对齐。

## 复现与结果位置

本次数据暂存根目录为 `/private/tmp/rbf-v2v4real-official-20260913.0KfLfb`。原始 ZIP 与含 GT 的 `train04-v1/payload/` 不提供给推断进程；只向未来推断进程提供 `train04-projection-v2/inputs/`。数据尚未上传到 ClearML。

小型审计回执、测试报告和对应代码快照另存于 `work_dirs/recover-before-fuse/v2v4real-native-20260913-2339/`；这个备份不含原始 YAML、点云、模型或预测流。

校验和解压入口仅接受元数据快照中的 train/test 分卷。输出路径必须不存在，父目录必须已存在且可写；解压前检查至少能容纳全部未压缩数据和 1 GiB 余量。示例命令在 transvision 根目录执行：

```bash
python tools/event_track_v2x/extract_v2v4real_archive.py \
  --archive /data/v2v4real/train_04.zip \
  --release configs/event_track_v2x/v2v4real-release-20260913.json \
  --release-sha256 de8a40e6815f7cf43bead33df3a774e972f42823222389cc5314ea623aa62da7 \
  --output /data/v2v4real/train04-v1
```

工具拒绝路径越界、重复文件、链接、非 YAML/PCD 载荷及超限条目。单卷载荷核验不证明整个 split 齐全。失败前不覆盖原文件；若发布阶段失败，保留不完整输出供检查，不能把没有最终 `receipt.json` 的目录当作成功。

随后用 `audit_v2v4real_native_volume.py --volume ... --receipt-sha256 ... --output ...` 全量重读 YAML 并检查全部文件摘要。该审计不会生成跟踪 GT，也不验证 PCD 点记录格式。输入投影命令和显式 ego 参数见 [准备说明](v2v4real-inputs.md)。本次兼容性回放显式选择 CAV 0；不宣称已凭目录名识别 Tesla。

| 证据 | SHA-256 |
|---|---|
| 原始 ZIP | `aab4d7753b5bb62221f1a81602c52df168c0c263c578ec0ddd35c8fbd1be008c` |
| `train04-v1/receipt.json` | `fce0371cd22bdfebb22810aa4b9e20c155fe0222221ba6a253af433111448641` |
| `train04-projection-v2/inputs/manifest.json` | `b795768c3f24039a6b819d90e2c0f732e4a7ffaf76cf12a974bfea4ec5364349` |
| `native-volume-audit-v2.json` | `114fb24e5f2fd969d849b413de0aaeca29cb30e33df1c4006a38d6a95ac74b80` |
| `native-input-tests-v2.xml` | `bb5334c67862058ec895ac4b6d947fbab543c976368209b1271390a3c8eee6ea` |

发布端 SHA-1 `f661ee6acb50eaa81e9d8b611f2fd3bb06bca3f7` 已与下载文件一致；现代内容身份另由上述 SHA-256 记录。元数据快照不是发布者数字签名。

## 验证与剩余工作

相关回归 186 项通过，覆盖数值标签白名单、非法类型、递归 alias、矩阵格式、数值精度、ZIP 安全与完整性、原始数据准备、重叠审计和 SPD DetectionCacheV2。实际全卷另通过 876 帧解析、全部源文件摘要检查、输入投影及固定 manifest 摘要的完整回读。这些测试不等于点云格式验证或跟踪指标验证。

仍需获取其余官方 train/test 分卷，核验点云格式、严格 Car GT 和原生身份协议，冻结检测器及特征，再完成训练、同资源基线和官方 test。任何单卷、合成测试或输入投影都不能升级为双数据集论文性能证据。

2026-09-14 补充：首卷点云格式和全部点记录已通过独立校验及 Open3D 数值对照，见 [点云内容校验](v2v4real-pcd-inputs-20260914.md)。上述 YAML 审计及历史回执保持不变；原回执中未做 PCD 验证的声明仍描述当时执行范围。
