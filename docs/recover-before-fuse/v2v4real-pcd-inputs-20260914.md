# V2V4Real 点云内容校验

官方 `train_04.zip` 的 876 份源点云已完成逐点读取，并与 Open3D 0.19.0 执行的官方输入公式逐数组精确对照。总计 43,961,879 个点；所有输出有限，没有丢点、重排、ROI 筛选或位姿变换。范围仍是 4 个 train 序列，不是完整数据集、检测器复现或跟踪性能验证。

## 与官方输入公式对齐

真实文件采用 PCD 0.7、`DATA ascii`、`FIELDS x y z rgb`，四列均声明为 float32。第四列是打包颜色的浮点位模式，不是可以直接作为强度的数值。

官方 `pcd_to_np()` 使用 Open3D 的点坐标和颜色红通道，拼接后转为 float32。[固定版本源码](https://github.com/ucla-mobility/V2V4Real/blob/5a821e13753bafc611f95c47bc1a306acdcb0f7c/opencood/utils/pcd_utils.py)

新读取器 `transvision/models/event_track_v2x/v2v4real_pcd.py` 将第四列恢复为 float32 位模式，再计算

\[
I=\frac{(\operatorname{reinterpret}_{u32}(rgb_{f32})\gg16)\mathbin{\&}255}{255}.
\]

输出为按原顺序排列的只读 `N×4` float32 数组。读取器只接受本次观测到的 ASCII XYZRGB 格式；遇到二进制、压缩、额外字段或其他类型时失败，不推测转换方式。它检查完整行数、列数、每个数值、尺寸关系、非有限值和 float32 溢出，并限制单文件为 128 MiB、点数为 200 万。

格式约定依据 [PCD 格式说明](https://pointclouds.org/documentation/tutorials/pcd_file_format.html)。当前实现不是通用 PCD 读取器，不能将一个分卷的通过结果外推到未下载分卷。

## 重跑内容校验

使用独立环境安装 `environments/event_track_v2x/requirements-v2v4real-pcd-oracle.txt`；不要修改已封存的 SPD 推断或 A100 训练环境。固定 Python、NumPy、Open3D 及其他直接依赖，审计另记录全部已解析的传递依赖版本。

在 transvision 根目录执行，输出文件必须不存在且位于不可变输入目录之外：

```bash
python tools/event_track_v2x/audit_v2v4real_pcd.py \
  --inputs /private/tmp/rbf-v2v4real-official-20260913.0KfLfb/train04-projection-v2/inputs \
  --manifest-sha256 b795768c3f24039a6b819d90e2c0f732e4a7ffaf76cf12a974bfea4ec5364349 \
  --open3d-oracle \
  --output /data/audits/v2v4real-train04-pcd.json
```

工具只读取不含 GT 的点云与位姿投影，不读取原始 YAML 或标签。它先核验完整投影，再逐帧比较 `np.hstack((points, colors[:, :1])).astype(np.float32)`，要求形状与数组数值完全一致。失败不会写入成功回执；修正原因后使用新输出路径。

这次执行使用 Python 3.12.14、NumPy 1.26.4、Open3D 0.19.0，验证的是固定源码中的输入公式。它没有执行原论文完整环境或检测模型，因此 `original_paper_runtime_reproduced=false`。

## 回执和验证范围

点云审计为 `/private/tmp/rbf-v2v4real-pcd-20260914.XNCc72/pcd-audit-v1.json`，SHA-256 为 `ef642f83dc891dc575742dbb2f6264c79e4bb47c26271273223dd0db392535cd`。回执包含每个文件的源摘要、点数、输出数组摘要和逐维范围；不包含完整点云或标签。

六组 V2V4Real 回归共 187 项通过，无跳过，耗时 2.59 s。覆盖 ZIP 载荷、原始数值 YAML、输入投影、重叠检查和点云读取；其中点云测试还覆盖全部 256 个红通道值、非法格式、顺序保持以及故意注入的 Open3D 不一致。报告为同目录的 `native-pcd-regression-v3.xml`。这些是相关回归，不是全仓测试。

审计、测试报告和代码快照保留在 `work_dirs/recover-before-fuse/v2v4real-pcd-20260914/`。不上传原始 YAML、完整点云、GT、模型或预测流。

## 后续实验边界

下一步仍需核实原生标签参考系、负 `ass_id` 的身份语义、跨车身份去重和严格 Car GT，再冻结检测器及特征定义。首卷点云对齐不替代这些工作。其余官方 train/test 分卷、全量会话重叠审计、检测缓存适配、训练及冻结后的官方 test 均未完成。

GPU 参数训练继续只使用 A100，每任务至少四卡。点云解析和教师离线试算不是 GPU 参数训练，也不构成论文协同增益证据。
