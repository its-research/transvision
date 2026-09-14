# V2V4Real 严格 Car 的原生 GT 接入

已实现独立的离线 GT 准备入口，采用 DMSTrack 的真实数据晚融合 GT 数值路径，并在进入该路径前按原始 `obj_type == Car` 过滤。首个官方 train 分卷的 438 个双车时刻已与固定参考函数逐帧对照：ID 和 ROI 保留集合一致，7,088 条输出框记录的最大角点差为 0。不是独立目标数，也不是检测或跟踪指标。

## 坐标系：框先在源车，再转到 ego

真实数据 loader 向目标投影传入的是源车到 ego 的变换矩阵；晚融合先在单位矩阵下生成源车局部框，再用当前帧位姿转换。不能把原始 `location + center` 当成世界坐标，再对它直接施加 ego 位姿逆矩阵。[官方真实数据入口](https://github.com/ucla-mobility/V2V4Real/blob/5a821e13753bafc611f95c47bc1a306acdcb0f7c/opencood/data_utils/datasets/intermediate_fusion_dataset.py)、[DMSTrack 晚融合入口](https://github.com/eddyhkchiu/DMSTrack/blob/d3b9949499c8e68ea33060873bd1cb95b6d4d323/V2V4Real/opencood/data_utils/datasets/late_fusion_dataset.py)

记 `lidar_pose` 矩阵为源车到世界的 \(T_s,T_e\)，局部目标姿态为 \(T_{o|s}\)，则原始目标角点满足

\[
p_e=T_e^{-1}T_sT_{o|s}p_o.
\]

实际晚融合协议还有一次局部直立框化：先按官方角点边长均值与四条边的角度算术均值取得 `xyzlwhyaw`，重建直立框，再投影到 ego。保留这个步骤，不以最小面积拟合或角度环形均值替换它。`NativeAnnotation.center_world` 是既有通用接口的字段名，新 GT 模块明确按源车局部数值使用它；既有通用 `corners_in()` 不是原生 GT 转换器。

## 类别、身份和范围的固定顺序

1. 只保留原始 Car；不把 Truck、ConcreteTruck 或 Pedestrian 合入。
2. 在源车坐标系中进行直立框化，要求至少两个角点落入 `[-100,-40,-5,100,40,3]` 的三维范围。
3. 使用当前源车与 ego 位姿投影角点；按显式 ego 优先的 CAV 次序，对相同 ID 保留首个框。
4. 对去重后的框要求全部八个角点落入 ego 的 `x∈[-100,100]、y∈[-40,40]` 范围。此处不再次筛选 z。

上述两种范围判据不同，且去重发生在最终 ego 筛选之前。如果首个框在最终范围外，不能改选另一端的重复框。这个反例已加入测试。[固定 GT 函数](https://github.com/eddyhkchiu/DMSTrack/blob/d3b9949499c8e68ea33060873bd1cb95b6d4d323/V2V4Real/opencood/data_utils/post_processor/base_postprocessor.py)、[固定几何和范围函数](https://github.com/eddyhkchiu/DMSTrack/blob/d3b9949499c8e68ea33060873bd1cb95b6d4d323/V2V4Real/opencood/utils/box_utils.py)

官方 ID 换算为：有非负 `ass_id` 时使用它；否则使用 `object_id + 100 × cav_id`。CAV 0 的本地编号与共享编号可以有意重合；不能给二者分别加前缀，破坏跨车关联。新入口拒绝同一源车同一帧内两个 Car 共享一个映射 ID，不通过平均框消除冲突。

输出按数值 ID 排序，不沿用 Python `set` 的遍历顺序；框与 ID 的绑定不变。ID 只在各序列内有意义，跨序列评价或训练标签必须保留序列命名空间。

## 首卷证据及身份局限

原始源帧目标记录为 Car 10,476 条、ConcreteTruck 280 条、Truck 225 条；经过严格 Car 和上述 GT 路径后保留 7,088 条逐帧框记录。全量比较覆盖 438/438 个双车时刻，而不是抽样。

角点比较事先固定绝对容差为 `1e-4 m`、相对容差为零，用于容纳 100 m 范围内复合 float32 运算；实际最大差为 `0 m`。ID 和 ROI 保留集合要求精确一致。对照执行固定提交 `d3b9949499c8e68ea33060873bd1cb95b6d4d323` 中未修改的数值函数，依赖初始化被隔离，Car 预过滤由本项目显式执行。这不是完整 OpenCOOD loader、原论文软件环境或检测器复现。

按 `(sequence, cav_id, object_id)` 统计，183 个严格 Car 本地标注轨迹中，有 33 个在不同帧对应多个官方映射 ID。这个数字不是物理身份错误数，也不是跟踪器 ID switch；不能依据它自动合并身份或回写 GT。当前导出忠实保留原规则，尚未证明原生标注的物理身份连续性。训练标签和身份失败分析必须继续区分这种标注变化与算法错误。

原公开方法通常不区分上述车辆类别。严格 Car 结果必须通过同类别重跑得到，不直接与合并车辆类别的公开数字拼表。当前入口只接受 train；这不改变最终评估采用官方 test 的约定，test 接入须在协议冻结后单独启用。

## 执行和消费

代码为 `transvision/models/event_track_v2x/v2v4real_ground_truth.py`、`tools/event_track_v2x/prepare_v2v4real_ground_truth.py`。依赖 NumPy、PyTorch 和 PyYAML；本次使用已存在的独立研究环境，没有修改 A100 训练环境。

参考文件单独存放于 `/private/tmp/rbf-v2v4real-labels-20260914.ipsYwC`。其中 `box_utils.py`、`transformation_utils.py`、`common_utils.py`、`base_postprocessor.py` 和 `datasets_init.py` 来自上述固定 DMSTrack 提交；入口逐一检查内置 SHA-256 后才执行列明的数值定义。参考源码保留在本地研究目录，不放入项目代码快照或上传。[UCLA 上游许可](https://github.com/ucla-mobility/V2V4Real/blob/5a821e13753bafc611f95c47bc1a306acdcb0f7c/LICENSE)

输出父目录须已存在；输出目录必须不存在，并位于原始分卷和推断输入之外。在 transvision 根目录执行：

```bash
python tools/event_track_v2x/prepare_v2v4real_ground_truth.py \
  --volume /private/tmp/rbf-v2v4real-official-20260913.0KfLfb/train04-v1 \
  --receipt-sha256 fce0371cd22bdfebb22810aa4b9e20c155fe0222221ba6a253af433111448641 \
  --ego-agents /private/tmp/rbf-v2v4real-official-20260913.0KfLfb/ego-agents.json \
  --oracle-root /private/tmp/rbf-v2v4real-labels-20260914.ipsYwC \
  --output /data/evaluator/v2v4real-train04-gt
```

目录包含 `frames.jsonl`、`audit.jsonl` 和最后写入的 `manifest.json`。前两份分别是完整 GT 和逐帧筛选审计，不能交给推断进程。缺少最终 manifest 的部分输出保留用于诊断，不能当作完成。未提供 `--oracle-root` 时仍可生成 GT，但回执不会标为通过参考对照。

离线消费者调用 `load_train_ground_truth(root, expected_manifest_sha256=...)`，必须传入独立记录的 manifest 摘要。它核验文件清单、两个流的摘要、帧覆盖、ordinal、ego、严格 Car、有限角点和每帧 ID 唯一性。此函数不读取原始 YAML 或点云。输出继续使用 `ordinal-only-no-clock`，不伪造微秒时间。

本次最终产物为 `/private/tmp/rbf-v2v4real-labels-20260914.ipsYwC/train04-strict-car-gt-v2/`；小型 manifest、测试报告和本项目源码快照备份到 `work_dirs/recover-before-fuse/v2v4real-gt-20260914/`。备份不含 GT 全文、原始 YAML、点云、模型或第三方源码。

最终 manifest SHA-256 为 `fb1bd1d4a7b9b6d714a5764d3dee47c209dab8991a0a492d6e21c6fbd9a1897a`。完整 GT 流为 `efb3baecf9543f70c23d8779b7b443f649136b6950fedcadffd45d84e5831919`，逐帧审计流为 `9233ff6c3669728efcde81c143761317cc2fb263e887899cef92758fdc1d40f3`。新增消费者已按固定 manifest 摘要完整回读，确认 438 帧、7,088 条框记录和全部绑定源码摘要一致。

GT 与原始输入相关回归 187 项通过，耗时 1.97 s，包括独立参考函数对照和篡改后重新封装的拒绝测试。未运行全仓测试、检测器、参数训练或跟踪指标。其余分卷、冻结检测器与特征、正式缓存适配、三种子训练和官方 test 实验仍待完成；GPU 参数训练继续限定 A100、每任务至少四卡。

测试报告 `native-gt-regression-v2.xml` 的 SHA-256 为 `fc755ae63d730c8478b65a9cfaaa82d32d2a743aee4409f50598d6daeb337fb8`。这里的 187 项是本次 GT 与输入测试集合，不能与此前同样数量的 PCD 回归重复相加。
